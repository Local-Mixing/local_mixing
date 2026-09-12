#!/usr/bin/env python3
"""Run adaptive point-function SSS attempts until c2 fits the heatmap cap."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOCAL_OUT = REPO / "sgc" / "pf_test"
TARGET = "cc@129.114.109.32"
REMOTE_ROOT = "./sgc/pf_test"
SOURCE = f"{REMOTE_ROOT}/pf_key7124193401_n62.txt"
KEY = 7_124_193_401
CAP = 650_000
X_VALUES = [30, 40, 60, 80, 120, 160, 240, 320, 480, 640]


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ssh(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        ["ssh", "-n", TARGET, f"bash -lc {shlex.quote(command)}"],
        check=check,
    )


def collect() -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-az",
            f"{TARGET}:~/local_mixing/sgc/pf_test/",
            str(LOCAL_OUT) + "/",
        ],
        check=False,
    )


def gate_count(remote_path: str) -> int | None:
    result = ssh(
        "cd ~/local_mixing && "
        f"python3 -c {shlex.quote(f'''s=open({remote_path!r}).read().strip(); print(s.count(';') + (0 if not s or s.endswith(';') else 1))''')}",
        check=False,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip())


def sss_attempt(x: int) -> tuple[str, int | None, int, int]:
    job = f"sss_pf_key{KEY}_n62_sr2_ta4_r1_m1_x{x}"
    final = f"{REMOTE_ROOT}/{job}.txt"
    gadget = f"{REMOTE_ROOT}/{job}_gadgetized.txt"
    intermediate = f"{REMOTE_ROOT}/{job}_int.txt"
    log = f"{REMOTE_ROOT}/{job}.log"
    metadata = f"{REMOTE_ROOT}/{job}_metadata.json"

    existing = gate_count(final)
    if existing is not None:
        return job, existing, 0, 0

    command = [
        "bash",
        "security_tests/campaigns/legacy_runner.sh",
        "sss",
        "-n",
        "64",
        "-m",
        "1",
        "-x",
        str(x),
        "-s",
        SOURCE,
        "-d",
        final,
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
    joined = " ".join(shlex.quote(part) for part in command)
    remote = f"""
cd ~/local_mixing
mkdir -p {REMOTE_ROOT}
rm -f compression_histogram.csv compression_log.txt
start=$(date +%s)
set +e
{joined} > {shlex.quote(log)} 2>&1
code=$?
set -e
end=$(date +%s)
if [ -f compression_histogram.csv ]; then
  cp compression_histogram.csv {REMOTE_ROOT}/{job}_compression_histogram.csv
fi
if [ -f compression_log.txt ]; then
  cp compression_log.txt {REMOTE_ROOT}/{job}_compression.log
fi
printf '{{"job":"%s","key":{KEY},"input_wires":62,"total_wires":64,"shooting_times":2,"type_attempts":4,"rounds":1,"m":1,"x":{x},"server":"localtest","exit_code":%s,"runtime_seconds":%s}}\\n' \
  {shlex.quote(job)} "$code" "$((end-start))" > {shlex.quote(metadata)}
exit "$code"
""".strip()
    started = time.time()
    result = ssh(remote, check=False)
    runtime = round(time.time() - started)
    count = gate_count(final) if result.returncode == 0 else None
    collect()
    return job, count, result.returncode, runtime


def heatmap(job: str, *, gadgetized: bool) -> int:
    mode = "gadgetized_c1" if gadgetized else "original_c1"
    c1 = (
        f"{REMOTE_ROOT}/{job}_gadgetized.txt"
        if gadgetized
        else SOURCE
    )
    c2 = f"{REMOTE_ROOT}/{job}.txt"
    png = f"{REMOTE_ROOT}/{job}_{mode}_heatmap_enhance_0.01.png"
    log = f"{REMOTE_ROOT}/{job}_{mode}_heatmap_enhance_0.01.log"
    xlabel = "Gadgetized source" if gadgetized else "Original point function"
    command = [
        "python3",
        "./security_tests/heatmaps/heatmap.py",
        "--n",
        "128",
        "--i",
        "100",
        "--x",
        xlabel,
        "--y",
        "Final circuit",
        "--c1",
        c1,
        "--c2",
        c2,
        "--path",
        png,
        "--enhance",
        "0.01",
    ]
    joined = " ".join(shlex.quote(part) for part in command)
    result = ssh(
        f"cd ~/local_mixing && source ./.venv/bin/activate && "
        f"maturin develop >> {shlex.quote(log)} 2>&1 && "
        f"{joined} >> {shlex.quote(log)} 2>&1",
        check=False,
    )
    collect()
    return result.returncode


def main() -> int:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    summary_path = LOCAL_OUT / "adaptive_m1_summary.json"
    attempts: list[dict[str, object]] = []

    source_sha = ssh(
        f"cd ~/local_mixing && sha256sum {SOURCE} | cut -d' ' -f1"
    ).stdout.strip()
    if source_sha != "eddef493cb1849dc6305c05149ef4d6d93157188a7058170b11573e9d1592b07":
        raise RuntimeError(f"Unexpected source checksum: {source_sha}")

    successful_job = ""
    for x in X_VALUES:
        job, count, code, runtime = sss_attempt(x)
        row = {
            "job": job,
            "x": x,
            "gate_count": count,
            "exit_code": code,
            "runtime_seconds_observed": runtime,
            "qualifies_for_heatmap": count is not None and count <= CAP,
        }
        attempts.append(row)
        summary_path.write_text(
            json.dumps(
                {
                    "gate_cap": CAP,
                    "source_sha256": source_sha,
                    "attempts": attempts,
                    "successful_job": successful_job,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if code != 0:
            continue
        if count is not None and count <= CAP:
            successful_job = job
            break

    if not successful_job:
        raise RuntimeError("No X value produced a circuit within the heatmap cap")

    summary = {
        "gate_cap": CAP,
        "source_sha256": source_sha,
        "attempts": attempts,
        "successful_job": successful_job,
        "heatmaps": {},
    }
    for gadgetized in (False, True):
        mode = "gadgetized_c1" if gadgetized else "original_c1"
        summary["heatmaps"][mode] = {
            "exit_code": heatmap(successful_job, gadgetized=gadgetized),
            "enhance": 0.01,
            "incremental": False,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
