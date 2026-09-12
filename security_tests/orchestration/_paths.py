"""Keep historical run artifacts independent of the installed source location."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(os.environ.get("SECURITY_REPO_ROOT", Path(__file__).resolve().parents[2]))
ARTIFACT_ROOT = Path(os.environ.get("SECURITY_ARTIFACT_ROOT", REPO_ROOT / "red_team_tests"))
SOURCE_DIR = Path(__file__).resolve().parent


def run_paths(
    parser: argparse.ArgumentParser | None = None,
    *,
    c2: bool = False,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse explicit paths, retaining the old checkout's defaults.

    Deployed C1/C2 runs should pass their existing --run-dir and --tools-dir.
    The defaults reproduce the old source checkout's parent calculations;
    moving this package must never move status files or solver output.
    """

    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--run-dir", type=Path, help="existing run directory containing the challenge and logs")
    parser.add_argument("--tools-dir", type=Path, help="existing compiled solver/heatmap tools directory")
    parser.add_argument("--decoder", type=Path, default=SOURCE_DIR / "decode_mpmct1_zero_slice_model.py")
    args = parser.parse_args(argv)
    args.artifact_root = args.artifact_root.resolve()
    args.run_dir = (args.run_dir or (args.artifact_root / "_orchestration" if c2 else args.artifact_root)).resolve()
    args.tools_dir = (args.tools_dir or (args.run_dir.parent / "tools" if c2 else args.run_dir / "tools")).resolve()
    args.decoder = args.decoder.resolve()
    return args
