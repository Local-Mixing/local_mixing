#!/usr/bin/env python3
"""Generate the requested point function and run one SSS experiment on nho."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOCAL_OUT = REPO / "sgc" / "pf_test"
TARGET = "cc@129.114.108.170"
KEY = 7_124_193_401
INPUT_WIRES = 62
TOTAL_WIRES = INPUT_WIRES + 2
SOURCE_NAME = f"pf_key{KEY}_n{INPUT_WIRES}.txt"
JOB = f"sss_pf_key{KEY}_n{INPUT_WIRES}_sr2_ta4_r1_m2_x20"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


def main() -> int:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    source = f"./sgc/pf_test/{SOURCE_NAME}"
    destination = f"./sgc/pf_test/{JOB}.txt"
    intermediate = f"./sgc/pf_test/{JOB}_int.txt"
    gadget = f"./sgc/pf_test/{JOB}_gadgetized.txt"
    sss_log = f"./sgc/pf_test/{JOB}.log"
    metadata = f"./sgc/pf_test/{JOB}_metadata.json"

    sss = [
        "bash",
        "security_tests/campaigns/legacy_runner.sh",
        "sss",
        "-n",
        str(TOTAL_WIRES),
        "-m",
        "2",
        "-x",
        "20",
        "-s",
        source,
        "-d",
        destination,
        "-r",
        "1",
        "-i",
        intermediate,
        "--gadgetize",
        "--shuffled",
        "--gates_ahead",
        "3",
        "--rg-frequency",
        "2",
        "--type_attempts",
        "4",
        "--shooting_times",
        "2",
        "--gadget_path",
        gadget,
    ]
    sss_command = " ".join(shlex.quote(part) for part in sss)

    remote = f"""
set -e
cd ~/local_mixing
mkdir -p ./sgc/pf_test

if [ ! -s {shlex.quote(source)} ]; then
  cargo build --release --features challenge-tools --bin point_function \
    > ./sgc/pf_test/point_function_build.log 2>&1
  rm -f ./sgc/pf_test/point_function_output.log
  target/release/point_function -n {INPUT_WIRES} -k {KEY} \
    > ./sgc/pf_test/point_function_output.log \
    2> ./sgc/pf_test/point_function_stderr.log &
  generator_pid=$!
  found=0
  for _ in $(seq 1 600); do
    if grep -q '^len = ' ./sgc/pf_test/point_function_output.log; then
      found=1
      break
    fi
    sleep 1
  done
  if [ "$found" -ne 1 ]; then
    kill "$generator_pid" 2>/dev/null || true
    wait "$generator_pid" 2>/dev/null || true
    echo "Point-function circuit was not emitted within 600 seconds" >&2
    exit 1
  fi
  sed -n '2p' ./sgc/pf_test/point_function_output.log > {shlex.quote(source)}
  kill "$generator_pid" 2>/dev/null || true
  wait "$generator_pid" 2>/dev/null || true
fi

source_gates=$(python3 -c "s=open('{source}').read().strip(); print(s.count(';') + (0 if not s or s.endswith(';') else 1))")
source_sha=$(sha256sum {shlex.quote(source)} | cut -d' ' -f1)
printf '%s  %s\\n' "$source_sha" {shlex.quote(SOURCE_NAME)} \
  > ./sgc/pf_test/{SOURCE_NAME}.sha256

if [ ! -s {shlex.quote(destination)} ]; then
  rm -f compression_histogram.csv compression_log.txt
  start=$(date +%s)
  set +e
  {sss_command} > {shlex.quote(sss_log)} 2>&1
  code=$?
  set -e
  end=$(date +%s)
  if [ -f compression_histogram.csv ]; then
    cp compression_histogram.csv ./sgc/pf_test/{JOB}_compression_histogram.csv
  fi
  if [ -f compression_log.txt ]; then
    cp compression_log.txt ./sgc/pf_test/{JOB}_compression.log
  fi
  printf '{{"job":"%s","key":%s,"input_wires":%s,"total_wires":%s,"source_gates":%s,"source_sha256":"%s","shooting_times":2,"type_attempts":4,"rounds":1,"m":2,"x":20,"server":"nho","exit_code":%s,"runtime_seconds":%s}}\\n' \
    {shlex.quote(JOB)} {KEY} {INPUT_WIRES} {TOTAL_WIRES} \
    "$source_gates" "$source_sha" "$code" "$((end-start))" \
    > {shlex.quote(metadata)}
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
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
            f"{TARGET}:~/local_mixing/sgc/pf_test/",
            str(LOCAL_OUT) + "/",
        ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
