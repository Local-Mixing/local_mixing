#!/usr/bin/env python3
"""Run the isolated C2 zero-slice CNF attack without printing target/model data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from ._paths import run_paths
else:
    from _paths import run_paths

RUN_DIR: Path | None = None


EXPECTED_VARIABLES = 7_470_401
EXPECTED_CLAUSES = 58_335_816
EXPECTED_HEADER = f"p cnf {EXPECTED_VARIABLES} {EXPECTED_CLAUSES}"


def replace_private(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def status(run_dir: Path, value: str) -> None:
    replace_private(run_dir / "status.txt", value)


def record_exit_code(run_dir: Path, phase: str, code: int) -> None:
    replace_private(run_dir / f"{phase}_exit_code.txt", str(code))


def main() -> int:
    global RUN_DIR
    args = run_paths(c2=True)
    os.umask(0o077)
    run_dir = RUN_DIR = args.run_dir
    tools_dir = args.tools_dir
    challenge = json.loads((run_dir / "challenge.json").read_text(encoding="utf-8"))
    if (
        challenge.get("n") != 128
        or challenge.get("total_wires") != 512
        or challenge.get("final_gate_count") != 7_469_889
        or challenge.get("logical_input_wires") != "0..127"
        or challenge.get("zero_input_wires") != "128..511"
        or challenge.get("logical_output_wires") != "128..255"
    ):
        status(run_dir, "FAILED_CHALLENGE_LAYOUT")
        return 2
    target = challenge.get("target_logical_output_hex")
    if not isinstance(target, str) or len(target) != 34 or not target.startswith("0x"):
        status(run_dir, "FAILED_CHALLENGE_TARGET")
        return 2

    cnf_path = run_dir / "c2_zero_slice.cnf"
    encoder_log = run_dir / "encoder.log"
    status(run_dir, "CNF_RUNNING")
    with encoder_log.open("wb") as log:
        encoder_rc = subprocess.run(
            [
                str(tools_dir / "mpmct1_zero_slice_to_cnf"),
                str(run_dir / "c2_final.txt"),
                str(cnf_path),
                "128",
                target,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    record_exit_code(run_dir, "encoder", encoder_rc)
    if encoder_rc != 0:
        status(run_dir, "FAILED_ENCODER")
        return encoder_rc or 2
    cnf_path.chmod(0o600)
    with cnf_path.open("r", encoding="ascii") as source:
        actual_header = source.readline().rstrip("\r\n")
    if actual_header != EXPECTED_HEADER:
        status(run_dir, "FAILED_CNF_HEADER")
        return 2
    replace_private(run_dir / "cnf_header_check.txt", "VALID " + EXPECTED_HEADER)

    solver_output = run_dir / "kissat_output.txt"
    status(run_dir, "SAT_RUNNING")
    with solver_output.open("wb") as log:
        solver_rc = subprocess.run(
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=60s",
                "24h",
                str(tools_dir / "kissat"),
                "--seed=2",
                "--time=86400",
                str(cnf_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    solver_output.chmod(0o600)
    record_exit_code(run_dir, "kissat", solver_rc)

    if solver_rc == 10:
        status(run_dir, "DECODING")
        decode_log = run_dir / "decoder.log"
        with decode_log.open("wb") as log:
            decode_rc = subprocess.run(
                [
                    sys.executable,
                    str(args.decoder),
                    "--final",
                    str(run_dir / "c2_final.txt"),
                    "--solver-output",
                    str(solver_output),
                    "--n",
                    "128",
                    "--target-hex",
                    target,
                    "--out",
                    str(run_dir / "verified_witness.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        record_exit_code(run_dir, "decoder", decode_rc)
        if decode_rc == 0:
            status(run_dir, "SAT_VERIFIED")
            return 0
        status(run_dir, "SAT_VERIFICATION_FAILED")
        return decode_rc or 2
    if solver_rc == 20:
        status(run_dir, "UNSAT")
        return 0
    if solver_rc in (0, 124, 137, 143):
        status(run_dir, "TIMEOUT_OR_UNKNOWN")
        return 0
    status(run_dir, "FAILED_KISSAT")
    return solver_rc or 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.umask(0o077)
        if RUN_DIR is None or not RUN_DIR.is_dir():
            raise
        run_dir = RUN_DIR
        replace_private(run_dir / "status.txt", "FAILED_SUPERVISOR")
        replace_private(run_dir / "supervisor_error.txt", type(error).__name__)
        raise
