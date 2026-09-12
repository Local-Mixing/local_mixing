#!/usr/bin/env python3
"""Restart C1 Kissat verbosely while preserving its original absolute deadline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

if __package__:
    from ._paths import run_paths
else:
    from _paths import run_paths

RUN_DIR: Path | None = None
import subprocess
import sys
import time


EXPECTED_HEADER = "p cnf 7469493 58331926"


def replace_private(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def write_private_json(path: Path, record: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def utc(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def record_exit_code(root: Path, phase: str, code: int) -> None:
    replace_private(root / f"{phase}_exit_code.txt", str(code))


def status(root: Path, value: str) -> None:
    replace_private(root / "status.txt", value)


def main() -> int:
    global RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-start-epoch", required=True, type=int)
    parser.add_argument("--absolute-deadline-epoch", required=True, type=int)
    args = run_paths(parser)

    os.umask(0o077)
    root = RUN_DIR = args.run_dir
    cnf_path = root / "c1_zero_slice.cnf"
    with cnf_path.open("r", encoding="ascii") as source:
        if source.readline().rstrip("\r\n") != EXPECTED_HEADER:
            status(root, "FAILED_CNF_HEADER_ON_VERBOSE_RESTART")
            return 2

    challenge = json.loads((root / "challenge.json").read_text(encoding="utf-8"))
    target = challenge.get("target_logical_output_hex")
    if (
        challenge.get("n") != 128
        or challenge.get("total_wires") != 512
        or challenge.get("logical_output_wires") != "128..255"
        or not isinstance(target, str)
        or len(target) != 34
        or not target.startswith("0x")
    ):
        status(root, "FAILED_CHALLENGE_ON_VERBOSE_RESTART")
        return 2

    restart_epoch = int(time.time())
    remaining = args.absolute_deadline_epoch - restart_epoch
    if remaining <= 0:
        status(root, "TIMEOUT_OR_UNKNOWN")
        return 0

    old_output = root / "kissat_output.txt"
    archived_output = root / "kissat_output.pre_verbose.txt"
    if not old_output.is_file() or archived_output.exists():
        status(root, "FAILED_VERBOSE_LOG_HANDOFF")
        return 2
    old_output.rename(archived_output)
    archived_output.chmod(0o600)

    options = [
        "--seed=1",
        f"--time={remaining}",
        "--verbose=1",
        "--statistics=1",
    ]
    write_private_json(
        root / "kissat_verbose_restart_status.json",
        {
            "absolute_deadline_epoch": args.absolute_deadline_epoch,
            "absolute_deadline_utc": utc(args.absolute_deadline_epoch),
            "cnf_header": EXPECTED_HEADER,
            "external_timeout_seconds": remaining,
            "kissat_options": options,
            "original_start_epoch": args.original_start_epoch,
            "original_start_utc": utc(args.original_start_epoch),
            "remaining_wall_seconds_at_restart": remaining,
            "restart_epoch": restart_epoch,
            "restart_utc": utc(restart_epoch),
        },
    )

    solver_output = root / "kissat_output.txt"
    status(root, "SAT_RUNNING_VERBOSE")
    with solver_output.open("wb") as log:
        solver_rc = subprocess.run(
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=60s",
                f"{remaining}s",
                str(args.tools_dir / "kissat"),
                *options,
                str(cnf_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    solver_output.chmod(0o600)
    record_exit_code(root, "kissat_verbose", solver_rc)

    if solver_rc == 10:
        status(root, "DECODING")
        decode_log = root / "decoder.log"
        with decode_log.open("wb") as log:
            decode_rc = subprocess.run(
                [
                    sys.executable,
                    str(args.decoder),
                    "--final",
                    str(root / "c1_final.txt"),
                    "--solver-output",
                    str(solver_output),
                    "--n",
                    "128",
                    "--target-hex",
                    target,
                    "--out",
                    str(root / "verified_witness.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        record_exit_code(root, "decoder", decode_rc)
        if decode_rc == 0:
            status(root, "SAT_VERIFIED")
            return 0
        status(root, "SAT_VERIFICATION_FAILED")
        return decode_rc or 2
    if solver_rc == 20:
        status(root, "UNSAT")
        return 0
    if solver_rc in (0, 124, 137, 143):
        status(root, "TIMEOUT_OR_UNKNOWN")
        return 0
    status(root, "FAILED_KISSAT_VERBOSE")
    return solver_rc or 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.umask(0o077)
        if RUN_DIR is None or not RUN_DIR.is_dir():
            raise
        root = RUN_DIR
        replace_private(root / "status.txt", "FAILED_VERBOSE_SUPERVISOR")
        replace_private(root / "verbose_supervisor_error.txt", type(error).__name__)
        raise
