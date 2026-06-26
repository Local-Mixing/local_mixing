#!/usr/bin/env python3
"""Recompute successful parameter-test heatmaps with a narrow color range."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sgc" / "parameter_test"
PNG_OUT = OUT / "png_very_enhanced"
ENHANCE = 0.005

SERVERS = {
    "n64test": "cc@129.114.109.63",
    "testpieces": "cc@129.114.109.6",
    "nho": "cc@129.114.108.170",
    "localtest": "cc@129.114.109.32",
    "llmtest": "cc@129.114.108.159",
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ssh(target: str, remote: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            target,
            f"bash -lc {shlex.quote(remote)}",
        ],
        check=check,
    )


def computed_jobs(alias: str) -> list[dict[str, str]]:
    path = OUT / f"{alias}_worker_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as summary:
        return [
            row
            for row in csv.DictReader(summary)
            if row.get("heatmap_status") == "computed"
        ]


def remote_script(job_id: str) -> str:
    c1 = f"./gadgetized/parameter_test_n64m1000_{job_id}.txt"
    c2 = f"./sgc/parameter_test/{job_id}.txt"
    png = f"./sgc/parameter_test/{job_id}_mixing_heatmap_very_enhanced.png"
    log = f"./sgc/parameter_test/{job_id}_very_enhanced_heatmap.log"
    command = [
        "python3",
        "./heatmap/heatmap.py",
        "--n",
        "128",
        "--i",
        "100",
        "--x",
        "Gadgetized source",
        "--y",
        "Final circuit",
        "--c1",
        c1,
        "--c2",
        c2,
        "--path",
        png,
        "--enhance",
        str(ENHANCE),
    ]
    joined = " ".join(shlex.quote(part) for part in command)
    return f"""
cd ~/local_mixing
source ./.venv/bin/activate
maturin develop >> {shlex.quote(log)} 2>&1
{joined} >> {shlex.quote(log)} 2>&1
""".strip()


def worker(alias: str) -> int:
    PNG_OUT.mkdir(parents=True, exist_ok=True)
    target = SERVERS[alias]
    for row in computed_jobs(alias):
        job_id = row["job_id"]
        filename = f"{job_id}_mixing_heatmap_very_enhanced.png"
        local_png = PNG_OUT / filename
        if local_png.exists():
            continue

        result = ssh(target, remote_script(job_id), check=False)
        remote_png = f"{target}:~/local_mixing/sgc/parameter_test/{filename}"
        run(["rsync", "-az", remote_png, str(PNG_OUT) + "/"], check=False)

        log_name = f"{job_id}_very_enhanced_heatmap.log"
        remote_log = f"{target}:~/local_mixing/sgc/parameter_test/{log_name}"
        run(["rsync", "-az", remote_log, str(OUT) + "/"], check=False)

        if result.returncode != 0:
            print(f"{job_id}: failed ({result.returncode})", flush=True)
        elif local_png.exists():
            print(f"{job_id}: complete", flush=True)
        else:
            print(
                f"{job_id}: completed remotely but PNG collection failed",
                flush=True,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=sorted(SERVERS), required=True)
    args = parser.parse_args()
    return worker(args.server)


if __name__ == "__main__":
    raise SystemExit(main())
