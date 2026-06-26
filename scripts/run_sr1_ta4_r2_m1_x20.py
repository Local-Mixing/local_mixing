#!/usr/bin/env python3
"""Generate and collect the sr1/ta4/r2/m1/x20 circuit and heatmap."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOCAL_OUT = REPO / "sgc" / "parameter_test" / "r2"
TARGET = "cc@129.114.109.6"
JOB = "sss_sr1_ta4_r2_m1_x20"
REMOTE_OUT = "./sgc/parameter_test/r2"
REMOTE_GADGET = f"./gadgetized/parameter_test_n64m1000_{JOB}.txt"
REMOTE_FINAL = f"{REMOTE_OUT}/{JOB}.txt"
REMOTE_LOG = f"{REMOTE_OUT}/{JOB}.log"
REMOTE_HEATMAP_LOG = f"{REMOTE_OUT}/{JOB}_heatmap.log"
REMOTE_PNG = f"{REMOTE_OUT}/{JOB}_mixing_heatmap_very_enhanced.png"
GATE_CAP = 650_000


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


def main() -> int:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    sss = [
        "cargo",
        "run",
        "--release",
        "sss",
        "-n",
        "64",
        "-m",
        "1",
        "-x",
        "20",
        "-s",
        "./circuits/parameter_test_n64m1000.txt",
        "-d",
        REMOTE_FINAL,
        "-r",
        "2",
        "-i",
        f"{REMOTE_OUT}/{JOB}_int.txt",
        "--gadgetize",
        "--shuffled",
        "--gates_ahead",
        "3",
        "--rg-frequency",
        "2",
        "--type_attempts",
        "4",
        "--shooting_times",
        "1",
        "--gadget_path",
        REMOTE_GADGET,
    ]
    heatmap = [
        "python3",
        "./heatmap/heatmap.py",
        "--n",
        "128",
        "--i",
        "100",
        "--x",
        "Gadgetized source",
        "--y",
        "Final r2 circuit",
        "--c1",
        REMOTE_GADGET,
        "--c2",
        REMOTE_FINAL,
        "--path",
        REMOTE_PNG,
        "--enhance",
        "0.005",
    ]

    sss_command = " ".join(shlex.quote(part) for part in sss)
    heatmap_command = " ".join(shlex.quote(part) for part in heatmap)
    remote = f"""
set -e
cd ~/local_mixing
mkdir -p {shlex.quote(REMOTE_OUT)} gadgetized
if [ ! -s {shlex.quote(REMOTE_FINAL)} ]; then
  rm -f compression_histogram.csv
  {sss_command} > {shlex.quote(REMOTE_LOG)} 2>&1
  if [ -f compression_histogram.csv ]; then
    cp compression_histogram.csv {shlex.quote(f"{REMOTE_OUT}/{JOB}_compression_histogram.csv")}
  fi
fi
gates=$(python3 -c "s=open('{REMOTE_FINAL}').read().strip(); print(s.count(';') + (0 if not s or s.endswith(';') else 1))")
printf '%s\\n' "$gates" > {shlex.quote(f"{REMOTE_OUT}/{JOB}_gate_count.txt")}
if [ "$gates" -le {GATE_CAP} ] && [ ! -s {shlex.quote(REMOTE_PNG)} ]; then
  source ./.venv/bin/activate
  maturin develop >> {shlex.quote(REMOTE_HEATMAP_LOG)} 2>&1
  {heatmap_command} >> {shlex.quote(REMOTE_HEATMAP_LOG)} 2>&1
fi
""".strip()

    result = run(
        ["ssh", "-n", TARGET, f"bash -lc {shlex.quote(remote)}"],
        check=False,
    )
    run(
        [
            "rsync",
            "-az",
            f"{TARGET}:~/local_mixing/sgc/parameter_test/r2/",
            str(LOCAL_OUT) + "/",
        ],
        check=False,
    )
    run(
        [
            "rsync",
            "-az",
            f"{TARGET}:~/local_mixing/gadgetized/parameter_test_n64m1000_{JOB}.txt",
            str(LOCAL_OUT) + "/",
        ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
