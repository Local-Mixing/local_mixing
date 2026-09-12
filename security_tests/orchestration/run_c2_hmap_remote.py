#!/usr/bin/env python3
"""Run the isolated C2 degree-1 affine test without printing private data."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

if __package__:
    from ._paths import run_paths
else:
    from _paths import run_paths

RUN_DIR: Path | None = None


def replace_private(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> int:
    global RUN_DIR
    args = run_paths(c2=True)
    os.umask(0o077)
    run_dir = RUN_DIR = args.run_dir
    tools_dir = args.tools_dir
    replace_private(run_dir / "status.txt", "HMAP_RUNNING")
    with (run_dir / "hmap.log").open("wb") as log:
        return_code = subprocess.run(
            [
                "/usr/bin/time",
                "-v",
                "-o",
                str(run_dir / "resources.txt"),
                str(tools_dir / "hmap_affine"),
                "--c",
                str(tools_dir / "source_c.g57"),
                "--c-format",
                "g57",
                "--g",
                str(run_dir / "c2_final.txt"),
                "--g-format",
                "mpmct1",
                "--n",
                "128",
                "--degree",
                "1",
                "--c-step",
                "210",
                "--g-step",
                "74699",
                "--batches",
                "96",
                "--train-batches",
                "72",
                "--seed",
                "12345",
                "--out",
                str(run_dir / "c2_final_d1"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    (run_dir / "hmap.log").chmod(0o600)
    (run_dir / "resources.txt").chmod(0o600)
    replace_private(run_dir / "hmap_exit_code.txt", str(return_code))
    if return_code == 0:
        replace_private(run_dir / "status.txt", "HMAP_COMPLETE")
        return 0
    replace_private(run_dir / "status.txt", "FAILED_HMAP")
    return return_code or 2


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
