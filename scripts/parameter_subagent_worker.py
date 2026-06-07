#!/usr/bin/env python3
"""Per-server worker for the 64x1000 parameter sweep.

Each worker owns exactly one server and runs the round-robin slice assigned to
that server. This intentionally avoids a central long-lived SSH controller.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shlex
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sgc" / "parameter_test"
REMOTE_CIRCUIT = "./circuits/parameter_test_n64m1000.txt"
HEATMAP_WIRES = 128
HEATMAP_INPUTS = 100
HEATMAP_C2_GATE_CAP = 650_000
HEATMAP_ENHANCE = 0.01

SERVERS = [
    ("n64test", "cc@129.114.109.63"),
    ("testpieces", "cc@129.114.109.6"),
    ("nho", "cc@129.114.108.170"),
    ("localtest", "cc@129.114.109.32"),
    ("llmtest", "cc@129.114.108.159"),
]

SR_VALUES = [1, 2, 3]
TA_VALUES = [1, 2, 3, 4]
R_VALUES = [1]
M_VALUES = [1, 2, 3]
X_VALUES = [10, 20]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gadgetized").mkdir(parents=True, exist_ok=True)
    (OUT / "worker_logs").mkdir(parents=True, exist_ok=True)


def run(command: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def ssh(target: str, remote: str, *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
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
        timeout=timeout,
    )


def rsync(source: str, dest: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return run(["rsync", "-az", source, dest], check=check)


def job_id(sr: int, ta: int, rounds: int, m: int, x: int) -> str:
    return f"sss_sr{sr}_ta{ta}_r{rounds}_m{m}_x{x}"


def all_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    number = 1
    for sr in SR_VALUES:
        for ta in TA_VALUES:
            for rounds in R_VALUES:
                for m in M_VALUES:
                    for x in X_VALUES:
                        jobs.append(
                            {
                                "number": number,
                                "id": job_id(sr, ta, rounds, m, x),
                                "sr": sr,
                                "ta": ta,
                                "rounds": rounds,
                                "m": m,
                                "x": x,
                            }
                        )
                        number += 1
    return jobs


def assigned_jobs(worker_index: int) -> list[dict[str, object]]:
    return [job for job in all_jobs() if (int(job["number"]) - 1) % len(SERVERS) == worker_index]


def remote_paths(job: dict[str, object]) -> dict[str, str]:
    jid = str(job["id"])
    return {
        "final": f"./sgc/parameter_test/{jid}.txt",
        "intermediate": f"./sgc/parameter_test/{jid}_int.txt",
        "gadget": f"./gadgetized/parameter_test_n64m1000_{jid}.txt",
        "histogram": f"./sgc/parameter_test/{jid}_compression_histogram.csv",
        "log": f"./sgc/parameter_test/{jid}_attempt1.log",
        "meta": f"./sgc/parameter_test/{jid}_remote.json",
        "heatmap_log": f"./sgc/parameter_test/{jid}_heatmap.log",
        "metric": f"./sgc/parameter_test/{jid}_mixing_metric.json",
        "png": f"./sgc/parameter_test/{jid}_mixing_heatmap.png",
    }


def sss_command(job: dict[str, object], paths: dict[str, str]) -> list[str]:
    return [
        "cargo",
        "run",
        "--release",
        "sss",
        "-n",
        "64",
        "-m",
        str(job["m"]),
        "-x",
        str(job["x"]),
        "-s",
        REMOTE_CIRCUIT,
        "-d",
        paths["final"],
        "-r",
        str(job["rounds"]),
        "-i",
        paths["intermediate"],
        "--gadgetize",
        "--shuffled",
        "--gates_ahead",
        "3",
        "--rg-frequency",
        "2",
        "--type_attempts",
        str(job["ta"]),
        "--shooting_times",
        str(job["sr"]),
        "--gadget_path",
        paths["gadget"],
    ]


def heatmap_python(paths: dict[str, str]) -> str:
    return f"""
import json
import sys
import numpy as np
sys.path.insert(0, './heatmap')
import heatmap as plotter
import local_mixing as heatmap_rust

c1 = {paths['gadget']!r}
c2 = {paths['final']!r}
png = {paths['png']!r}
metric = {paths['metric']!r}
with open(c2, 'r', encoding='utf-8', errors='replace') as f:
    c2_text = f.read().strip()
c2_gate_count = 0 if not c2_text else c2_text.count(';') + (0 if c2_text.endswith(';') else 1)
if c2_gate_count > {HEATMAP_C2_GATE_CAP}:
    payload = {{
        'heatmap_status': 'too large to compute',
        'heatmap_skipped': True,
        'heatmap_enhance': {HEATMAP_ENHANCE},
        'heatmap_skip_reason': 'c2 gate count exceeds cap',
        'heatmap_c2_gate_cap': {HEATMAP_C2_GATE_CAP},
        'c2_gate_count': c2_gate_count,
        'heatmap_inputs': {HEATMAP_INPUTS},
        'heatmap_wires': {HEATMAP_WIRES},
        'heatmap_png': '',
    }}
    with open(metric, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0)
results = heatmap_rust.heatmap({HEATMAP_WIRES}, {HEATMAP_INPUTS}, False, c1, c2, True, 0, False, False, False)
points = np.asarray(results, dtype=float)
values = points[:, 2]
plotter.plot_heatmap_raw(results, png, 'Gadgetized source', 'Final circuit', vmin=0.5 - {HEATMAP_ENHANCE}, vmax=0.5 + {HEATMAP_ENHANCE})
payload = {{
    'heatmap_status': 'computed',
    'heatmap_skipped': False,
    'heatmap_enhance': {HEATMAP_ENHANCE},
    'c2_gate_count': c2_gate_count,
    'mixing_mean': float(np.nanmean(values)),
    'mixing_std': float(np.nanstd(values)),
    'mixing_min': float(np.nanmin(values)),
    'mixing_max': float(np.nanmax(values)),
    'heatmap_inputs': {HEATMAP_INPUTS},
    'heatmap_wires': {HEATMAP_WIRES},
    'heatmap_png': png,
}}
with open(metric, 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print(json.dumps(payload, sort_keys=True))
""".strip()


def remote_script(job: dict[str, object]) -> str:
    jid = str(job["id"])
    paths = remote_paths(job)
    sss = " ".join(shlex.quote(part) for part in sss_command(job, paths))
    heatmap_code = shlex.quote("exec(" + repr(heatmap_python(paths)) + ")")
    return f"""
cd ~/local_mixing
mkdir -p circuits sgc/parameter_test gadgetized
rm -f compression_histogram.csv
start=$(date +%s)
set +e
{sss} > {shlex.quote(paths['log'])} 2>&1
code=$?
end=$(date +%s)
if [ -f compression_histogram.csv ]; then cp compression_histogram.csv {shlex.quote(paths['histogram'])}; fi
printf '{{"job_id":"%s","exit_code":%s,"start_epoch":%s,"end_epoch":%s,"runtime_seconds":%s}}\\n' {shlex.quote(jid)} "$code" "$start" "$end" "$((end-start))" > {shlex.quote(paths['meta'])}
if [ "$code" -eq 0 ]; then
  heat_start=$(date +%s)
  source ./.venv/bin/activate && maturin develop >> {shlex.quote(paths['heatmap_log'])} 2>&1 && python3 -c {heatmap_code} >> {shlex.quote(paths['heatmap_log'])} 2>&1
  heat_code=$?
  heat_end=$(date +%s)
  printf '{{"heatmap_exit_code":%s,"heatmap_runtime_seconds":%s}}\\n' "$heat_code" "$((heat_end-heat_start))" >> {shlex.quote(paths['meta'])}
fi
exit "$code"
""".strip()


def collect(target: str, job: dict[str, object]) -> None:
    jid = str(job["id"])
    for pattern in [f"{jid}*.txt", f"{jid}*.csv", f"{jid}*.json", f"{jid}*.log", f"{jid}*.png"]:
        rsync(f"{target}:~/local_mixing/sgc/parameter_test/{pattern}", str(OUT) + "/")
    rsync(
        f"{target}:~/local_mixing/gadgetized/parameter_test_n64m1000_{jid}.txt",
        str(OUT / "gadgetized") + "/",
    )


def read_json_lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def summarize_job(alias: str, job: dict[str, object], exit_code: int) -> dict[str, object]:
    jid = str(job["id"])
    metric_path = OUT / f"{jid}_mixing_metric.json"
    final_path = OUT / f"{jid}.txt"
    meta_path = OUT / f"{jid}_remote.json"
    row = {
        "time": utc_now(),
        "server": alias,
        "job_number": job["number"],
        "job_id": jid,
        "sr": job["sr"],
        "ta": job["ta"],
        "rounds": job["rounds"],
        "m": job["m"],
        "x": job["x"],
        "exit_code": exit_code,
        "status": "success" if exit_code == 0 else "failed",
        "final_gate_count": final_path.read_text(encoding="utf-8", errors="replace").count(";") if final_path.exists() else "",
        "heatmap_status": "",
        "c2_gate_count": "",
        "mixing_mean": "",
        "heatmap_png": "",
    }
    records = read_json_lines(meta_path)
    for record in records:
        row.update({k: v for k, v in record.items() if k not in {"job_id"}})
    if metric_path.exists():
        try:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            row.update(metric)
        except json.JSONDecodeError:
            row["heatmap_status"] = "metric parse error"
    return row


def append_worker_row(alias: str, row: dict[str, object]) -> None:
    path = OUT / f"{alias}_worker_summary.csv"
    exists = path.exists()
    fields = [
        "time",
        "server",
        "job_number",
        "job_id",
        "sr",
        "ta",
        "rounds",
        "m",
        "x",
        "status",
        "exit_code",
        "runtime_seconds",
        "heatmap_exit_code",
        "heatmap_runtime_seconds",
        "final_gate_count",
        "heatmap_status",
        "c2_gate_count",
        "mixing_mean",
        "heatmap_png",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def successful_job_ids(alias: str) -> set[str]:
    path = OUT / f"{alias}_worker_summary.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row.get("job_id", "")
            for row in csv.DictReader(f)
            if row.get("status") == "success" and row.get("job_id")
        }


def worker(alias: str) -> int:
    ensure_dirs()
    worker_index = [server[0] for server in SERVERS].index(alias)
    target = dict(SERVERS)[alias]
    jobs = assigned_jobs(worker_index)
    completed = successful_job_ids(alias)
    log_path = OUT / "worker_logs" / f"{alias}.jsonl"
    for job in jobs:
        jid = str(job["id"])
        if jid in completed:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"time": utc_now(), "event": "resume_skip_success", "job_id": jid}, sort_keys=True) + "\n")
            continue
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": utc_now(), "event": "start", "job": job}, sort_keys=True) + "\n")
        result = ssh(target, remote_script(job), check=False)
        collect(target, job)
        row = summarize_job(alias, job, result.returncode)
        append_worker_row(alias, row)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "time": utc_now(),
                        "event": "finish",
                        "job_id": jid,
                        "exit_code": result.returncode,
                        "stdout_tail": result.stdout[-1000:],
                        "stderr_tail": result.stderr[-1000:],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=[server[0] for server in SERVERS], required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        worker_index = [server[0] for server in SERVERS].index(args.server)
        for job in assigned_jobs(worker_index):
            print(json.dumps(job, sort_keys=True))
        return 0
    return worker(args.server)


if __name__ == "__main__":
    raise SystemExit(main())
