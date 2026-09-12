#!/usr/bin/env python3
"""Restart C2 Kissat verbosely without extending its original absolute deadline."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from ._paths import run_paths
else:
    from _paths import run_paths

RUN_DIR: Path | None = None


EXPECTED_HEADER = "p cnf 7470401 58335816"
ORIGINAL_START_EPOCH = 1_786_319_872
ORIGINAL_DEADLINE_EPOCH = ORIGINAL_START_EPOCH + 86_400


def replace_private(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def utc(epoch: int) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    global RUN_DIR
    args = run_paths(c2=True)
    os.umask(0o077)
    run_dir = RUN_DIR = args.run_dir
    tools_dir = args.tools_dir
    cnf_path = run_dir / "c2_zero_slice.cnf"

    if (run_dir / "cnf_header_check.txt").read_text(encoding="ascii").strip() != "VALID " + EXPECTED_HEADER:
        replace_private(run_dir / "status.txt", "FAILED_VERBOSE_CNF_VALIDATION")
        return 2
    with cnf_path.open("r", encoding="ascii") as source:
        if source.readline().rstrip("\r\n") != EXPECTED_HEADER:
            replace_private(run_dir / "status.txt", "FAILED_VERBOSE_CNF_VALIDATION")
            return 2

    old_output = run_dir / "kissat_output.txt"
    archived_output = run_dir / "kissat_output_pre_verbose.txt"
    if old_output.exists() and not archived_output.exists():
        old_output.replace(archived_output)
        archived_output.chmod(0o600)
    old_exit = run_dir / "kissat_exit_code.txt"
    archived_exit = run_dir / "kissat_pre_verbose_exit_code.txt"
    if old_exit.exists() and not archived_exit.exists():
        old_exit.replace(archived_exit)
        archived_exit.chmod(0o600)

    restart_epoch = int(time.time())
    remaining_seconds = ORIGINAL_DEADLINE_EPOCH - restart_epoch
    if remaining_seconds <= 0:
        replace_private(run_dir / "status.txt", "ORIGINAL_DEADLINE_EXPIRED")
        return 0
    options = [
        "-v",
        "--statistics",
        "--seed=2",
        f"--time={remaining_seconds}",
    ]
    restart_record = {
        "original_start_epoch": ORIGINAL_START_EPOCH,
        "original_start_utc": utc(ORIGINAL_START_EPOCH),
        "original_deadline_epoch": ORIGINAL_DEADLINE_EPOCH,
        "original_deadline_utc": utc(ORIGINAL_DEADLINE_EPOCH),
        "restart_epoch": restart_epoch,
        "restart_utc": utc(restart_epoch),
        "remaining_seconds_at_restart": remaining_seconds,
        "external_timeout_seconds": remaining_seconds,
        "kissat_options": options,
        "cnf_header": EXPECTED_HEADER,
    }
    replace_private(run_dir / "verbose_restart_status.json", json.dumps(restart_record, sort_keys=True))

    solver_output = run_dir / "kissat_output_verbose.txt"
    replace_private(run_dir / "status.txt", "SAT_RUNNING_VERBOSE")
    with solver_output.open("wb") as log:
        solver_rc = subprocess.run(
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=60s",
                f"{remaining_seconds}s",
                str(tools_dir / "kissat"),
                *options,
                str(cnf_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    solver_output.chmod(0o600)
    replace_private(run_dir / "kissat_verbose_exit_code.txt", str(solver_rc))

    if solver_rc == 10:
        challenge = json.loads((run_dir / "challenge.json").read_text(encoding="utf-8"))
        target = challenge.get("target_logical_output_hex")
        if not isinstance(target, str) or len(target) != 34 or not target.startswith("0x"):
            replace_private(run_dir / "status.txt", "FAILED_VERBOSE_CHALLENGE_TARGET")
            return 2
        replace_private(run_dir / "status.txt", "DECODING_VERBOSE")
        decode_log = run_dir / "decoder_verbose.log"
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
        decode_log.chmod(0o600)
        replace_private(run_dir / "decoder_verbose_exit_code.txt", str(decode_rc))
        if decode_rc == 0:
            replace_private(run_dir / "status.txt", "SAT_VERIFIED_VERBOSE")
            return 0
        replace_private(run_dir / "status.txt", "SAT_VERIFICATION_FAILED_VERBOSE")
        return decode_rc or 2
    if solver_rc == 20:
        replace_private(run_dir / "status.txt", "UNSAT_VERBOSE")
        return 0
    if solver_rc in (0, 124, 137, 143):
        replace_private(run_dir / "status.txt", "TIMEOUT_OR_UNKNOWN_VERBOSE")
        return 0
    replace_private(run_dir / "status.txt", "FAILED_KISSAT_VERBOSE")
    return solver_rc or 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.umask(0o077)
        if RUN_DIR is None or not RUN_DIR.is_dir():
            raise
        run_dir = RUN_DIR
        replace_private(run_dir / "status.txt", "FAILED_VERBOSE_SUPERVISOR")
        replace_private(run_dir / "verbose_supervisor_error.txt", type(error).__name__)
        raise
