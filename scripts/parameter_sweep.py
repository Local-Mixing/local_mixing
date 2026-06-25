#!/usr/bin/env python3
"""Run the 5-server SSS parameter sweep and collect artifacts.

This script is intentionally resumable. It records state after every material
step, runs at most one active job per server, and retries each failed SSS job
once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sgc" / "parameter_test"
STATE_PATH = OUT / "sweep_state.json"
COMMAND_LOG = OUT / "command_log.jsonl"
SUMMARY_CSV = OUT / "summary.csv"
SUMMARY_MD = OUT / "summary.md"

REMOTE_PROJECT = "~/local_mixing"
REMOTE_CIRCUIT = "./circuits/parameter_test_n64m1000.txt"
LOCAL_CIRCUIT = OUT / "parameter_test_n64m1000.txt"

INPUT_WIRES = 64
GATES = 1000
HEATMAP_WIRES = 128
HEATMAP_INPUTS = 100
HEATMAP_C2_GATE_CAP = 650_000
REMOTE_POLL_SECONDS = 60
REMOTE_PROBE_TIMEOUT_SECONDS = 30

SERVERS = [
    {"alias": "n64test", "target": "cc@129.114.109.63"},
    {"alias": "testpieces", "target": "cc@129.114.109.6"},
    {"alias": "nho", "target": "cc@129.114.108.170"},
    {"alias": "localtest", "target": "cc@129.114.109.32"},
    {"alias": "llmtest", "target": "cc@129.114.108.159"},
]

GENERATOR_ALIAS = "nho"

SR_VALUES = [1, 2, 3]
TA_VALUES = [1, 2, 3, 4]
R_VALUES = [1]
M_VALUES = [1, 2, 3]
X_VALUES = [10, 20]

STATE_LOCK = threading.Lock()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gadgetized").mkdir(parents=True, exist_ok=True)
    (OUT / "heatmaps").mkdir(parents=True, exist_ok=True)


def log_command(kind: str, command: list[str] | str, **extra: object) -> None:
    ensure_dirs()
    record = {"time": utc_now(), "kind": kind, "command": command}
    record.update(extra)
    with COMMAND_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def run_local(
    command: list[str],
    *,
    kind: str,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    log_command(kind, command)
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def ssh_command(server: dict[str, str], remote: str) -> list[str]:
    return [
        "ssh",
        "-n",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
        server["target"],
        f"bash -lc {shlex.quote(remote)}",
    ]


def ssh(server: dict[str, str], remote: str, *, kind: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ssh_command(server, remote)
    return run_local(command, kind=f"{kind}:{server['alias']}", check=check)


def rsync(source: str, dest: str, *, kind: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["rsync", "-az", source, dest]
    return run_local(command, kind=kind, check=check)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "created_at": utc_now(),
            "setup": {},
            "jobs": {},
            "servers": SERVERS,
            "generator_alias": GENERATOR_ALIAS,
        }
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(STATE_PATH)


def update_job_state(job_id: str, **updates: object) -> None:
    with STATE_LOCK:
        state = load_state()
        job = state.setdefault("jobs", {}).setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = utc_now()
        save_state(state)


def update_setup(**updates: object) -> None:
    with STATE_LOCK:
        state = load_state()
        state.setdefault("setup", {}).update(updates)
        state["updated_at"] = utc_now()
        save_state(state)


def server_by_alias(alias: str) -> dict[str, str]:
    for server in SERVERS:
        if server["alias"] == alias:
            return server
    raise KeyError(alias)


def job_id(sr: int, ta: int, rounds: int, m: int, x: int) -> str:
    return f"sss_sr{sr}_ta{ta}_r{rounds}_m{m}_x{x}"


def build_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    idx = 0
    for sr in SR_VALUES:
        for ta in TA_VALUES:
            for rounds in R_VALUES:
                for m in M_VALUES:
                    for x in X_VALUES:
                        server = SERVERS[idx % len(SERVERS)]
                        jid = job_id(sr, ta, rounds, m, x)
                        jobs.append(
                            {
                                "id": jid,
                                "server": server["alias"],
                                "target": server["target"],
                                "sr": sr,
                                "ta": ta,
                                "rounds": rounds,
                                "m": m,
                                "x": x,
                            }
                        )
                        idx += 1
    # Cheap jobs first, then m3 x20, then the very large m3 x10 heatmaps last.
    # Server assignment is still round-robin from the original grid.
    def order_key(job: dict[str, object]) -> tuple[int, int, int, int, int]:
        m3 = int(job["m"]) == 3
        x = int(job["x"])
        if not m3:
            bucket = 0
        elif x == 20:
            bucket = 1
        else:
            bucket = 2
        return (bucket, int(job["sr"]), int(job["ta"]), int(job["m"]), x)

    jobs.sort(key=order_key)
    return jobs


def remote_job_paths(job: dict[str, object]) -> dict[str, str]:
    jid = str(job["id"])
    return {
        "dest": f"./sgc/parameter_test/{jid}.txt",
        "intermediate": f"./sgc/parameter_test/{jid}_int.txt",
        "gadget": f"./gadgetized/parameter_test_n64m1000_{jid}.txt",
        "histogram": f"./sgc/parameter_test/{jid}_compression_histogram.csv",
        "heatmap_png": f"./sgc/parameter_test/{jid}_mixing_heatmap.png",
        "heatmap_metric": f"./sgc/parameter_test/{jid}_mixing_metric.json",
    }


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def setup_servers() -> None:
    ensure_dirs()
    for server in SERVERS:
        ssh(
            server,
            "cd ~/local_mixing && mkdir -p circuits sgc/parameter_test gadgetized",
            kind="mkdir",
        )


def generate_and_distribute_circuit() -> None:
    state = load_state()
    setup = state.get("setup", {})
    nho = server_by_alias(GENERATOR_ALIAS)

    if not setup.get("circuit_generated"):
        cmd = (
            "cd ~/local_mixing && mkdir -p circuits && "
            f"cargo run --release genran -n {INPUT_WIRES} -m {GATES} -d {REMOTE_CIRCUIT}"
        )
        ssh(nho, cmd, kind="generate_circuit")
        update_setup(circuit_generated=True, circuit_generated_at=utc_now())

    rsync(f"{nho['target']}:~/local_mixing/circuits/parameter_test_n64m1000.txt", str(LOCAL_CIRCUIT), kind="collect_circuit")

    for server in SERVERS:
        if server["alias"] == GENERATOR_ALIAS:
            continue
        rsync(str(LOCAL_CIRCUIT), f"{server['target']}:~/local_mixing/circuits/parameter_test_n64m1000.txt", kind=f"distribute_circuit:{server['alias']}")

    checksums: dict[str, str] = {}
    for server in SERVERS:
        result = ssh(server, "cd ~/local_mixing && sha256sum circuits/parameter_test_n64m1000.txt", kind="checksum")
        checksums[server["alias"]] = result.stdout.strip().split()[0]

    unique = sorted(set(checksums.values()))
    update_setup(circuit_checksums=checksums, circuit_checksum_verified=len(unique) == 1)
    if len(unique) != 1:
        raise RuntimeError(f"Circuit checksum mismatch: {checksums}")


def sss_command(job: dict[str, object], paths: dict[str, str]) -> list[str]:
    return [
        "cargo",
        "run",
        "--release",
        "sss",
        "-n",
        str(INPUT_WIRES),
        "-m",
        str(job["m"]),
        "-x",
        str(job["x"]),
        "-s",
        REMOTE_CIRCUIT,
        "-d",
        paths["dest"],
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
    code = f"""
import json
import sys
import numpy as np
sys.path.insert(0, './heatmap')
import heatmap as plotter
import local_mixing as heatmap_rust

c1 = {paths['gadget']!r}
c2 = {paths['dest']!r}
png = {paths['heatmap_png']!r}
metric = {paths['heatmap_metric']!r}
with open(c2, 'r', encoding='utf-8', errors='replace') as f:
    c2_text = f.read().strip()
c2_gate_count = 0 if not c2_text else c2_text.count(';') + (0 if c2_text.endswith(';') else 1)
if c2_gate_count > {HEATMAP_C2_GATE_CAP}:
    payload = {{
        'heatmap_status': 'too large to compute',
        'heatmap_skipped': True,
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
plotter.plot_heatmap_raw(results, png, 'Gadgetized source', 'Final circuit', vmin=0.45, vmax=0.55)
payload = {{
    'heatmap_status': 'computed',
    'heatmap_skipped': False,
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
"""
    return code.strip()

def remote_heatmap_script(job: dict[str, object], attempt: int) -> str:
    jid = str(job["id"])
    paths = remote_job_paths(job)
    meta_path = f"./sgc/parameter_test/{jid}_{job['server']}_attempt{attempt}.remote.json"
    heatmap_log = f"./sgc/parameter_test/{jid}_heatmap.log"
    heatmap_code = shlex.quote("exec(" + repr(heatmap_python(paths)) + ")")
    return f"""
cd ~/local_mixing
mkdir -p circuits sgc/parameter_test gadgetized
heat_start=$(date +%s)
source ./.venv/bin/activate && maturin develop >> {shlex.quote(heatmap_log)} 2>&1 && python3 -c {heatmap_code} >> {shlex.quote(heatmap_log)} 2>&1
heat_code=$?
heat_end=$(date +%s)
printf '{{"heatmap_exit_code":%s,"heatmap_runtime_seconds":%s}}\n' "$heat_code" "$((heat_end-heat_start))" >> {shlex.quote(meta_path)}
exit "$heat_code"
""".strip()


def remote_run_script(job: dict[str, object], attempt: int) -> str:
    jid = str(job["id"])
    paths = remote_job_paths(job)
    log_path = f"./sgc/parameter_test/{jid}_{job['server']}_attempt{attempt}.log"
    meta_path = f"./sgc/parameter_test/{jid}_{job['server']}_attempt{attempt}.remote.json"
    heatmap_log = f"./sgc/parameter_test/{jid}_heatmap.log"
    sss = shell_join(sss_command(job, paths))
    heatmap_code = shlex.quote("exec(" + repr(heatmap_python(paths)) + ")")
    heatmap_step = ""
    if int(job["m"]) != 3:
        heatmap_step = f"""
if [ "$code" -eq 0 ]; then
  heat_start=$(date +%s)
  source ./.venv/bin/activate && maturin develop >> {shlex.quote(heatmap_log)} 2>&1 && python3 -c {heatmap_code} >> {shlex.quote(heatmap_log)} 2>&1
  heat_code=$?
  heat_end=$(date +%s)
  printf '{{"heatmap_exit_code":%s,"heatmap_runtime_seconds":%s}}\n' "$heat_code" "$((heat_end-heat_start))" >> {shlex.quote(meta_path)}
fi
"""
    return f"""
cd ~/local_mixing
while pgrep -f "[t]arget/release/local_mixing_bin" >/dev/null; do sleep 60; done
mkdir -p circuits sgc/parameter_test gadgetized
rm -f compression_histogram.csv
start=$(date +%s)
set +e
{sss} > {shlex.quote(log_path)} 2>&1
code=$?
end=$(date +%s)
if [ -f compression_histogram.csv ]; then cp compression_histogram.csv {shlex.quote(paths['histogram'])}; fi
printf '{{"job_id":"%s","server":"%s","attempt":%s,"exit_code":%s,"start_epoch":%s,"end_epoch":%s,"runtime_seconds":%s}}\n' {shlex.quote(jid)} {shlex.quote(str(job['server']))} {attempt} "$code" "$start" "$end" "$((end-start))" > {shlex.quote(meta_path)}
{heatmap_step}
exit "$code"
""".strip()

def parse_json_lines(text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def remote_meta_complete(server: dict[str, str], job: dict[str, object], attempt: int, *, heatmap_required: bool) -> bool:
    jid = str(job["id"])
    meta_path = f"~/local_mixing/sgc/parameter_test/{jid}_{job['server']}_attempt{attempt}.remote.json"
    metric_path = f"~/local_mixing/sgc/parameter_test/{jid}_mixing_metric.json"
    probe = f"cat {shlex.quote(meta_path)} 2>/dev/null; printf '\n--METRIC--\n'; cat {shlex.quote(metric_path)} 2>/dev/null"
    command = [
        "ssh",
        "-n",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        server["target"],
        f"bash -lc {shlex.quote(probe)}",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=REMOTE_PROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode not in (0, 1):
        return False
    meta_text, _, metric_text = result.stdout.partition("\n--METRIC--\n")
    records = parse_json_lines(meta_text)
    if not any(record.get("exit_code") == 0 for record in records):
        return False
    if not heatmap_required:
        return True
    if any(record.get("heatmap_exit_code") == 0 for record in records):
        return True
    try:
        metric = json.loads(metric_text.strip()) if metric_text.strip() else {}
    except json.JSONDecodeError:
        metric = {}
    return str(metric.get("heatmap_status", "")) in {"computed", "too large to compute"}


def remote_process_active(server: dict[str, str], jid: str) -> bool:
    probe = f"pgrep -f {shlex.quote(jid)} >/dev/null"
    command = [
        "ssh",
        "-n",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        server["target"],
        f"bash -lc {shlex.quote(probe)}",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=REMOTE_PROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def reconcile_existing_remote_job(job: dict[str, object], attempt: int, *, heatmap_required: bool) -> bool:
    server = server_by_alias(str(job["server"]))
    jid = str(job["id"])
    while True:
        if remote_meta_complete(server, job, attempt, heatmap_required=heatmap_required):
            collect_job(job, attempt)
            parsed = parse_job_outputs(job)
            status = "circuit_success_deferred_heatmap" if not heatmap_required else "success"
            update_job_state(
                jid,
                status=status,
                attempt=attempt,
                server=server["alias"],
                parameters=job,
                reconciled_at=utc_now(),
                **parsed,
            )
            return True
        if remote_process_active(server, jid):
            update_job_state(jid, status="running", remote_process_active=True, last_remote_probe_at=utc_now())
            time.sleep(REMOTE_POLL_SECONDS)
            continue
        return False


def run_remote_with_reconcile(
    server: dict[str, str],
    remote: str,
    *,
    kind: str,
    job: dict[str, object],
    attempt: int,
    heatmap_required: bool,
) -> subprocess.CompletedProcess[str]:
    command = ssh_command(server, remote)
    log_command(f"{kind}:{server['alias']}", command)
    proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    last_probe = 0.0
    while True:
        code = proc.poll()
        if code is not None:
            stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(command, code, stdout, stderr)
        now = time.time()
        if now - last_probe >= REMOTE_POLL_SECONDS:
            last_probe = now
            if remote_meta_complete(server, job, attempt, heatmap_required=heatmap_required):
                proc.terminate()
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                note = "remote metadata indicated completion; stale ssh wrapper reconciled"
                stderr = (stderr or "") + ("\n" if stderr else "") + note
                return subprocess.CompletedProcess(command, 0, stdout or "", stderr)
        time.sleep(5)


def collect_job(job: dict[str, object], attempt: int) -> None:
    server = server_by_alias(str(job["server"]))
    jid = str(job["id"])
    patterns = [
        f"{jid}*.txt",
        f"{jid}*.csv",
        f"{jid}*.json",
        f"{jid}*.log",
        f"{jid}*.png",
    ]
    for pattern in patterns:
        rsync(
            f"{server['target']}:~/local_mixing/sgc/parameter_test/{pattern}",
            str(OUT) + "/",
            kind=f"collect_job:{jid}:attempt{attempt}",
            check=False,
        )
    rsync(
        f"{server['target']}:~/local_mixing/gadgetized/parameter_test_n64m1000_{jid}.txt",
        str(OUT / "gadgetized") + "/",
        kind=f"collect_gadget:{jid}:attempt{attempt}",
        check=False,
    )


def parse_job_outputs(job: dict[str, object]) -> dict[str, object]:
    jid = str(job["id"])
    log_files = sorted(OUT.glob(f"{jid}_*_attempt*.log"))
    metric_path = OUT / f"{jid}_mixing_metric.json"
    final_path = OUT / f"{jid}.txt"
    histogram_path = OUT / f"{jid}_compression_histogram.csv"
    data: dict[str, object] = {}

    if final_path.exists():
        data["final_gate_count"] = final_path.read_text(encoding="utf-8", errors="replace").count(";")
    if histogram_path.exists():
        data["histogram_path"] = str(histogram_path)
    if metric_path.exists():
        try:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            data.update(metric)
            if "mixing_mean" in metric:
                data["mixing_distance_from_half"] = abs(float(metric["mixing_mean"]) - 0.5)
        except json.JSONDecodeError:
            data["metric_parse_error"] = str(metric_path)
    for log in log_files[-1:]:
        text = log.read_text(encoding="utf-8", errors="replace")
        match = re.findall(r"After compression: (\\d+) gates", text)
        if match:
            data["last_after_compression"] = int(match[-1])
        final = re.findall(r"Final len: (\\d+)", text)
        if final:
            data["final_len_logged"] = int(final[-1])
    return data


def run_one_job(job: dict[str, object]) -> None:
    jid = str(job["id"])
    existing = load_state().get("jobs", {}).get(jid, {})
    if existing.get("status") == "success":
        return

    server = server_by_alias(str(job["server"]))
    if existing.get("status") == "running" and existing.get("server") == server["alias"]:
        attempt = int(existing.get("attempt", 1) or 1)
        if reconcile_existing_remote_job(job, attempt, heatmap_required=int(job["m"]) != 3):
            return

    for attempt in (1, 2):
        update_job_state(jid, status="running", attempt=attempt, server=server["alias"], parameters=job, started_at=utc_now())
        remote = remote_run_script(job, attempt)
        start = time.time()
        result = run_remote_with_reconcile(
            server,
            remote,
            kind=f"run_job:{jid}:attempt{attempt}",
            job=job,
            attempt=attempt,
            heatmap_required=int(job["m"]) != 3,
        )
        runtime = time.time() - start
        collect_job(job, attempt)
        parsed = parse_job_outputs(job)
        update_job_state(
            jid,
            attempt=attempt,
            exit_code=result.returncode,
            runtime_seconds=runtime,
            stdout_tail=result.stdout[-2000:] if result.stdout else "",
            stderr_tail=result.stderr[-2000:] if result.stderr else "",
            collected_at=utc_now(),
            **parsed,
        )
        if result.returncode == 0:
            status = "circuit_success_deferred_heatmap" if int(job["m"]) == 3 else "success"
            update_job_state(jid, status=status, completed_at=utc_now())
            return
        update_job_state(jid, status="failed_once" if attempt == 1 else "failed", failed_at=utc_now())


def run_one_heatmap(job: dict[str, object]) -> None:
    jid = str(job["id"])
    existing = load_state().get("jobs", {}).get(jid, {})
    if existing.get("status") == "success" and existing.get("mixing_mean") != "":
        return
    if existing.get("status") == "heatmap_running":
        attempt = int(existing.get("heatmap_attempt", 1) or 1)
        if reconcile_existing_remote_job(job, attempt, heatmap_required=True):
            return
    final_path = OUT / f"{jid}.txt"
    gadget_path = OUT / "gadgetized" / f"parameter_test_n64m1000_{jid}.txt"
    if not final_path.exists() or not gadget_path.exists():
        update_job_state(jid, status="missing_circuit_for_heatmap", heatmap_waiting_at=utc_now())
        return

    server = server_by_alias(str(job["server"]))
    for attempt in (1, 2):
        update_job_state(jid, status="heatmap_running", heatmap_attempt=attempt, server=server["alias"], parameters=job, heatmap_started_at=utc_now())
        remote = remote_heatmap_script(job, attempt)
        start = time.time()
        result = run_remote_with_reconcile(
            server,
            remote,
            kind=f"run_heatmap:{jid}:attempt{attempt}",
            job=job,
            attempt=attempt,
            heatmap_required=True,
        )
        runtime = time.time() - start
        collect_job(job, attempt)
        parsed = parse_job_outputs(job)
        update_job_state(
            jid,
            heatmap_attempt=attempt,
            heatmap_exit_code=result.returncode,
            heatmap_runtime_seconds=runtime,
            heatmap_stdout_tail=result.stdout[-2000:] if result.stdout else "",
            heatmap_stderr_tail=result.stderr[-2000:] if result.stderr else "",
            collected_at=utc_now(),
            **parsed,
        )
        if result.returncode == 0:
            update_job_state(jid, status="success", heatmap_completed_at=utc_now())
            return
        update_job_state(jid, status="heatmap_failed_once" if attempt == 1 else "heatmap_failed", heatmap_failed_at=utc_now())


def write_summary() -> None:
    state = load_state()
    jobs = build_jobs()
    rows: list[dict[str, object]] = []
    for job in jobs:
        jid = str(job["id"])
        entry = state.get("jobs", {}).get(jid, {})
        row = {
            "job_id": jid,
            "server": job["server"],
            "shuffled_shooting_rounds": job["sr"],
            "type_attempts": job["ta"],
            "rounds": job["rounds"],
            "samfs_per_insert": job["m"],
            "insert_every_gates": job["x"],
            "status": entry.get("status", "pending"),
            "attempt": entry.get("attempt", ""),
            "exit_code": entry.get("exit_code", ""),
            "runtime_seconds": entry.get("runtime_seconds", ""),
            "final_gate_count": entry.get("final_gate_count", entry.get("final_len_logged", "")),
            "mixing_mean": entry.get("mixing_mean", ""),
            "mixing_distance_from_half": entry.get("mixing_distance_from_half", ""),
            "mixing_std": entry.get("mixing_std", ""),
            "heatmap_status": entry.get("heatmap_status", ""),
            "c2_gate_count": entry.get("c2_gate_count", ""),
            "heatmap_png": entry.get("heatmap_png", ""),
        }
        rows.append(row)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    successes = [r for r in rows if r["status"] == "success"]
    ranked = sorted(
        [r for r in successes if r["mixing_distance_from_half"] != ""],
        key=lambda r: (float(r["mixing_distance_from_half"]), float(r["runtime_seconds"] or 1e18)),
    )
    with SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Parameter Sweep Summary\n\n")
        f.write(f"- Generated: {utc_now()}\n")
        f.write(f"- Jobs: {len(rows)} total, {len(successes)} success\n")
        f.write(f"- Results directory: `{OUT}`\n\n")
        if ranked:
            best = ranked[0]
            f.write("## Best Setting\n\n")
            f.write(
                f"`{best['job_id']}` with mean={best['mixing_mean']}, "
                f"distance_from_0.5={best['mixing_distance_from_half']}, "
                f"runtime_seconds={best['runtime_seconds']}.\n\n"
            )
        f.write("## Rows\n\n")
        f.write("| job | status | server | mean | heatmap | runtime | final gates |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['job_id']} | {row['status']} | {row['server']} | "
                f"{row['mixing_mean']} | {row['heatmap_status']} | "
                f"{row['runtime_seconds']} | {row['final_gate_count']} |\n"
            )


def server_worker(server: dict[str, str], jobs: list[dict[str, object]]) -> None:
    for job in jobs:
        run_one_job(job)
        write_summary()


def server_heatmap_worker(server: dict[str, str], jobs: list[dict[str, object]]) -> None:
    for job in jobs:
        run_one_heatmap(job)
        write_summary()


def run_sweep() -> None:
    ensure_dirs()
    setup_servers()
    generate_and_distribute_circuit()
    jobs = build_jobs()
    with STATE_LOCK:
        state = load_state()
        state["planned_jobs"] = len(jobs)
        state["updated_at"] = utc_now()
        save_state(state)

    by_server = {server["alias"]: [] for server in SERVERS}
    for job in jobs:
        by_server[str(job["server"])].append(job)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SERVERS)) as pool:
        futures = [
            pool.submit(server_worker, server, by_server[server["alias"]])
            for server in SERVERS
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    write_summary()

    # Only after every circuit build phase is done do we spend slots on m3 heatmaps.
    m3_jobs = [job for job in jobs if int(job["m"]) == 3]
    by_server = {server["alias"]: [] for server in SERVERS}
    for job in m3_jobs:
        by_server[str(job["server"])].append(job)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SERVERS)) as pool:
        futures = [
            pool.submit(server_heatmap_worker, server, by_server[server["alias"]])
            for server in SERVERS
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    write_summary()


def status() -> None:
    state = load_state()
    jobs = state.get("jobs", {})
    counts: dict[str, int] = {}
    for entry in jobs.values():
        counts[str(entry.get("status", "pending"))] = counts.get(str(entry.get("status", "pending")), 0) + 1
    planned = state.get("planned_jobs", len(build_jobs()))
    print(json.dumps({"planned": planned, "recorded": len(jobs), "counts": counts, "setup": state.get("setup", {})}, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "status", "summary", "jobs"])
    args = parser.parse_args()
    ensure_dirs()
    if args.command == "run":
        run_sweep()
    elif args.command == "status":
        status()
    elif args.command == "summary":
        write_summary()
    elif args.command == "jobs":
        for job in build_jobs():
            print(json.dumps(job, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

