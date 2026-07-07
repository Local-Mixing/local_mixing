#!/usr/bin/env python3
"""Run random-circuit low-target SAT timing studies.

This is a deliberately boring harness: generate random circuits, take prefixes,
build low-target CNFs, run a small set of SAT algorithms with seeds, and keep
raw/aggregate/HTML outputs up to date after every completed job.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"


@dataclass(frozen=True)
class Solver:
    name: str
    kind: str
    path: str


def parse_circuit(text: str) -> list[tuple[int, int, int]]:
    gates: list[tuple[int, int, int]] = []
    wires: list[int] = []
    overflow = 0
    for ch in text:
        if ch == ";":
            if wires:
                if len(wires) != 3:
                    raise ValueError("bad gate")
                gates.append((wires[0], wires[1], wires[2]))
                wires = []
            overflow = 0
        elif ch == "~":
            overflow += 1
        elif ch.isspace():
            continue
        else:
            base = CHARS.find(ch)
            if base < 0:
                raise ValueError(f"bad circuit character {ch!r}")
            wires.append(base + 83 * overflow)
            overflow = 0
    if wires:
        raise ValueError("unterminated gate")
    return gates


def gate_text(gate: tuple[int, int, int]) -> str:
    out: list[str] = []
    for wire in gate:
        overflow, base = divmod(wire, 83)
        out.append("~" * overflow + CHARS[base])
    return "".join(out) + ";"


def write_prefix(gates: list[tuple[int, int, int]], count: int, path: Path) -> None:
    if not path.exists():
        path.write_text("".join(gate_text(g) for g in gates[:count]))


def read_cnf_header(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open() as f:
            for line in f:
                if line.startswith("p cnf "):
                    _, _, var_count, clause_count = line.split()
                    return int(var_count), int(clause_count)
    except FileNotFoundError:
        return None, None
    return None, None


def parse_solver_output(text: str, meta: str = "") -> dict[str, object]:
    status = "UNKNOWN"
    if re.search(r"^s SATISFIABLE", text, re.MULTILINE):
        status = "SAT"
    elif re.search(r"^s UNSATISFIABLE", text, re.MULTILINE):
        status = "UNSAT"

    conflicts = None
    process_time = None
    variables_original = None
    remaining = None
    remaining_pct = None

    for line in text.splitlines():
        m = re.search(r"^c\s+conflicts:\s+([0-9]+)", line)
        if m:
            conflicts = int(m.group(1))
        m = re.search(r"^c\s+process-time:\s+(?:.*\s)?([0-9]+(?:\.[0-9]+)?) seconds", line)
        if m:
            process_time = float(m.group(1))
        m = re.search(r"^c\s+total process time.*:\s+([0-9]+(?:\.[0-9]+)?)\s+seconds", line)
        if m:
            process_time = float(m.group(1))
        m = re.search(r"^c\s+variables_original:\s+([0-9]+)", line)
        if m:
            variables_original = int(m.group(1))
        if line.startswith("c -"):
            parts = line.split()
            if len(parts) >= 10 and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", parts[2]):
                try:
                    conflicts = int(parts[9])
                except ValueError:
                    pass
                try:
                    process_time = float(parts[2])
                except ValueError:
                    pass
                try:
                    remaining = int(parts[-2])
                    remaining_pct = int(parts[-1].rstrip("%"))
                except ValueError:
                    pass

    # CryptoMiniSat restart lines do not use the same final statistics style.
    cms_lines = [line for line in text.splitlines() if line.startswith("c rst")]
    if cms_lines:
        parts = cms_lines[-1].split()
        if len(parts) >= 6:
            conflicts = parse_count(parts[5])
        if len(parts) >= 2:
            try:
                process_time = float(parts[-1])
            except ValueError:
                pass

    m = re.search(r"returncode\s+(-?\d+)", meta)
    returncode = int(m.group(1)) if m else None
    return {
        "status": status,
        "conflicts": conflicts,
        "process_time": process_time,
        "variables_original": variables_original,
        "remaining": remaining,
        "remaining_pct": remaining_pct,
        "returncode": returncode,
    }


def parse_count(value: str) -> int | None:
    m = re.fullmatch(r"([0-9]+)([KMG]?)", value)
    if not m:
        return None
    n = int(m.group(1))
    suffix = m.group(2)
    if suffix == "K":
        n *= 1_000
    elif suffix == "M":
        n *= 1_000_000
    elif suffix == "G":
        n *= 1_000_000_000
    return n


def load_completed(raw_path: Path) -> tuple[list[dict[str, str]], set[tuple[str, ...]]]:
    rows: list[dict[str, str]] = []
    keys: set[tuple[str, ...]] = set()
    if raw_path.exists():
        with raw_path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows.append(row)
                keys.add(row_key(row))
    return rows, keys


def row_key(row: dict[str, object]) -> tuple[str, ...]:
    return (
        str(row["n"]),
        str(row["circuit_id"]),
        str(row["y_id"]),
        str(row["m"]),
        str(row["k"]),
        str(row["solver"]),
        str(row["seed"]),
    )


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})
    tmp.replace(path)


def aggregate(rows: list[dict[str, object]], time_limit: float) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["solver"]), str(row["n"]), str(row["m"]), str(row["k"])), []).append(row)

    out: list[dict[str, object]] = []
    for (solver, n, m, k), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]), int(kv[0][3]))):
        times = [float(v["wall_time"]) for v in vals if v.get("wall_time") not in ("", None)]
        solved = [float(v["wall_time"]) for v in vals if v.get("status") == "SAT" and v.get("wall_time") not in ("", None)]
        unknown = sum(v.get("status") == "UNKNOWN" for v in vals)
        unsat = sum(v.get("status") == "UNSAT" for v in vals)
        sat = sum(v.get("status") == "SAT" for v in vals)
        mean = statistics.fmean(times) if times else None
        median = statistics.median(times) if times else None
        variance = statistics.pvariance(times) if len(times) > 1 else (0.0 if times else None)
        stddev = variance**0.5 if variance is not None else None
        solved_variance = statistics.pvariance(solved) if len(solved) > 1 else (0.0 if solved else None)
        out.append(
            {
                "solver": solver,
                "n": int(n),
                "m": int(m),
                "k": int(k),
                "runs": len(vals),
                "sat": sat,
                "unsat": unsat,
                "unknown": unknown,
                "unknown_rate": round(unknown / len(vals), 3),
                "mean_wall": None if mean is None else round(mean, 3),
                "median_wall": None if median is None else round(median, 3),
                "variance_wall": None if variance is None else round(variance, 3),
                "stddev_wall": None if stddev is None else round(stddev, 3),
                "solved_mean_wall": "" if not solved else round(statistics.fmean(solved), 3),
                "solved_variance_wall": "" if solved_variance is None else round(solved_variance, 3),
                "timeouts_at_limit": sum(float(v.get("wall_time") or 0) >= time_limit for v in vals),
            }
        )
    return out


def write_html(path: Path, raw_rows: list[dict[str, object]], agg_rows: list[dict[str, object]], progress: dict[str, object]) -> None:
    def table(rows: list[dict[str, object]], fields: list[str]) -> str:
        if not rows:
            return "<p>No rows yet.</p>"
        head = "".join(f"<th>{html.escape(f)}</th>" for f in fields)
        body = []
        for row in rows:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(f, '')))}</td>" for f in fields) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    agg_fields = [
        "n",
        "m",
        "k",
        "runs",
        "sat",
        "unsat",
        "unknown",
        "unknown_rate",
        "mean_wall",
        "median_wall",
        "variance_wall",
        "stddev_wall",
        "solved_mean_wall",
        "solved_variance_wall",
        "timeouts_at_limit",
    ]
    grouped_agg = []
    for solver in sorted({str(row.get("solver", "")) for row in agg_rows}):
        rows = sorted(
            [row for row in agg_rows if str(row.get("solver", "")) == solver],
            key=lambda row: (int(row.get("n", 0)), int(row.get("m", 0)), int(row.get("k", 0))),
        )
        grouped_agg.append(f"<h3>{html.escape(solver)}</h3>" + table(rows, agg_fields))
    recent_fields = [
        "finished_at",
        "solver",
        "n",
        "m",
        "circuit_id",
        "y_id",
        "seed",
        "status",
        "wall_time",
        "conflicts",
        "remaining",
        "remaining_pct",
    ]
    recent = sorted(raw_rows, key=lambda r: str(r.get("finished_at", "")), reverse=True)[:40]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Random Circuit ET Study</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#1f2328}"
        "table{border-collapse:collapse;font-size:13px;margin:12px 0 28px 0}"
        "th,td{border:1px solid #d0d7de;padding:4px 7px;text-align:right}"
        "th{background:#f6f8fa;position:sticky;top:0}"
        "td:first-child,th:first-child{text-align:left}"
        ".meta{color:#57606a}"
        "</style>"
        "<h1>Random Circuit ET Study</h1>"
        f"<p class='meta'>Updated {html.escape(str(progress.get('updated_at', '')))}. "
        f"Completed {progress.get('completed_runs', 0)} / {progress.get('planned_runs', '?')} runs. "
        f"Active workers: {progress.get('active_workers', '?')}.</p>"
        "<h2>Aggregate by solver</h2>"
        + ("".join(grouped_agg) if grouped_agg else "<p>No rows yet.</p>")
        + "<h2>Recent completed runs</h2>"
        + table(recent, recent_fields),
    )
    tmp.replace(path)


def random_hex(rng: random.Random, bits: int) -> str:
    value = rng.getrandbits(bits)
    width = (bits + 3) // 4
    return "0x" + f"{value:0{width}x}"


def solver_command(solver: Solver, seed: int, time_limit: int, cnf: Path, out_path: Path) -> list[str]:
    if solver.kind == "kissat":
        return [solver.path, f"--time={time_limit}", f"--seed={seed}", "--sat", "-s", "-v", str(cnf)]
    if solver.kind == "cadical":
        sol = str(out_path.with_suffix(".sol"))
        return [
            solver.path,
            "-t",
            str(time_limit),
            f"--seed={seed}",
            "--report=true",
            "--stats=true",
            "-w",
            sol,
            str(cnf),
        ]
    if solver.kind == "cms":
        return [
            solver.path,
            "--maxtime",
            str(time_limit),
            "--random",
            str(seed),
            "--threads",
            "1",
            "--xor",
            "1",
            "--gates",
            "1",
            "--verb",
            "1",
            str(cnf),
        ]
    raise ValueError(f"unknown solver kind {solver.kind}")


def run_command(cmd: list[str], timeout: int) -> tuple[int, str, float, bool]:
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout + 15,
        )
        text = proc.stdout
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        text = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
    wall = time.monotonic() - start
    return rc, text, wall, timed_out


def make_plan(args: argparse.Namespace) -> dict[int, list[int]]:
    if args.m_plan:
        plan: dict[int, list[int]] = {}
        for chunk in args.m_plan.split(","):
            n_s, ms_s = chunk.split(":", 1)
            plan[int(n_s)] = [int(x) for x in ms_s.split("|") if x]
        return plan
    if args.preset == "pilot":
        return {
            64: [450, 500, 550, 600, 650, 700, 750],
            128: [750, 800, 850, 900, 950, 1000, 1050],
            256: [1800, 1900, 2000, 2100, 2200, 2300],
        }
    if args.preset == "mini":
        return {
            64: [500, 600, 700],
            128: [800, 900, 1000],
            256: [2000, 2100, 2200],
        }
    raise ValueError(f"unknown preset {args.preset}")


def make_k_plan(args: argparse.Namespace, m_plan: dict[int, list[int]]) -> dict[int, int]:
    plan = {n: args.k for n in m_plan}
    if not args.k_plan:
        return plan
    for chunk in args.k_plan.split(","):
        n_s, k_s = chunk.split(":", 1)
        plan[int(n_s)] = int(k_s)
    return plan


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in re.split(r"[|,]", text) if x]


def make_k_values_plan(args: argparse.Namespace, m_plan: dict[int, list[int]]) -> dict[int, list[int]]:
    if args.k_values_plan:
        plan: dict[int, list[int]] = {}
        for chunk in args.k_values_plan.split(","):
            n_s, values_s = chunk.split(":", 1)
            plan[int(n_s)] = parse_int_list(values_s)
        return {n: sorted(set(plan.get(n, [make_k_plan(args, m_plan)[n]]))) for n in m_plan}
    if args.k_values:
        values = sorted(set(parse_int_list(args.k_values)))
        return {n: values for n in m_plan}
    scalar_plan = make_k_plan(args, m_plan)
    return {n: [scalar_plan[n]] for n in m_plan}


def ensure_tools(args: argparse.Namespace) -> None:
    for path in [args.genran, args.converter]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    for solver in load_solvers(args):
        if not Path(solver.path).exists():
            raise FileNotFoundError(solver.path)


def load_solvers(args: argparse.Namespace) -> list[Solver]:
    configured = [
        Solver("kissat_sat", "kissat", args.kissat),
        Solver("cadical_sc2025", "cadical", args.cadical),
        Solver("cryptominisat_xor_gate", "cms", args.cryptominisat),
    ]
    wanted = set(args.solvers.split(","))
    return [s for s in configured if s.name in wanted]


def prepare_inputs(args: argparse.Namespace, m_plan: dict[int, list[int]], k_values_plan: dict[int, list[int]]) -> list[dict[str, object]]:
    out_dir = Path(args.out_dir)
    circuits_dir = out_dir / "circuits"
    prefixes_dir = out_dir / "prefixes"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    prefixes_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.study_seed)
    selected: list[dict[str, object]] = []
    for n, m_values in m_plan.items():
        max_m = max(m_values)
        target_bits = n // 2
        k_values = k_values_plan[n]
        for k in k_values:
            leading_zero_bits = k if args.k_is_leading_zero else n // 2 - k
            if leading_zero_bits < 0:
                raise ValueError(f"k={k} gives negative leading-zero bits for n={n}")
            if leading_zero_bits >= n:
                raise ValueError(f"k={k} gives too many leading-zero bits for n={n}")
        for cid in range(args.circuits_per_n):
            circuit_path = circuits_dir / f"n{n}_c{cid:03d}_m{max_m}.txt"
            if not circuit_path.exists():
                subprocess.run(
                    [
                        args.genran,
                        "genran",
                        "-n",
                        str(n),
                        "-m",
                        str(max_m),
                        "-d",
                        str(circuit_path),
                    ],
                    check=True,
                )
            gates = parse_circuit(circuit_path.read_text())
            if len(gates) < max_m:
                raise ValueError(f"{circuit_path} has only {len(gates)} gates")
            for m in m_values:
                prefix_path = prefixes_dir / f"n{n}_c{cid:03d}_m{m}.txt"
                write_prefix(gates, m, prefix_path)

            candidates = [random_hex(rng, target_bits) for _ in range(args.challenge_pool)]
            y_ids = list(range(args.challenge_pool))
            rng.shuffle(y_ids)
            for y_id in sorted(y_ids[: args.challenges_per_circuit]):
                selected.append(
                    {
                        "n": n,
                        "k_values": list(k_values),
                        "circuit_id": cid,
                        "circuit_path": str(circuit_path),
                        "m_values": list(m_values),
                        "target_bits": target_bits,
                        "y_id": y_id,
                        "target": candidates[y_id],
                    }
                )
    (out_dir / "input_manifest.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/random_circuit_et_study")
    parser.add_argument("--preset", choices=["mini", "pilot"], default="pilot")
    parser.add_argument("--m-plan", default="", help="Example: 64:500|600,128:800|900")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-plan", default="", help="Example: 128:14,256:18; defaults to --k for omitted n")
    parser.add_argument("--k-values", default="", help="One or more k values, e.g. 0|5|10")
    parser.add_argument("--k-values-plan", default="", help="Example: 64:0|5|10,128:0|10|20")
    parser.add_argument("--k-is-leading-zero", action="store_true", help="Interpret k directly as the number of leading input zero bits")
    parser.add_argument("--circuits-per-n", type=int, default=2)
    parser.add_argument("--challenge-pool", type=int, default=8)
    parser.add_argument("--challenges-per-circuit", type=int, default=4)
    parser.add_argument("--seeds", default="1001,1002,1003")
    parser.add_argument("--solvers", default="kissat_sat,cadical_sc2025,cryptominisat_xor_gate")
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=18)
    parser.add_argument("--study-seed", type=int, default=20260627)
    parser.add_argument("--genran", default="/home/cc/local_mixing/target/release/local_mixing_bin")
    parser.add_argument("--converter", default="/home/cc/local_mixing/work/et_study_tools/circuit_to_cnf_lowtarget_leading0_wide")
    parser.add_argument("--kissat", default="/home/cc/local_mixing/work/kissat/build/kissat")
    parser.add_argument("--cadical", default="/home/cc/local_mixing/work/solver_portfolio_20260627/src/cadical-sc2025/build/cadical")
    parser.add_argument("--cryptominisat", default="/home/cc/local_mixing/work/solver_portfolio_20260627/src/cryptominisat5-v5.14.7/cryptominisat5")
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Regenerate aggregate.tsv/progress.json/summary.html from raw_results.tsv without running jobs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cnf_dir = out_dir / "cnf"
    solver_out_dir = out_dir / "solver_out"
    cnf_dir.mkdir(parents=True, exist_ok=True)
    solver_out_dir.mkdir(parents=True, exist_ok=True)

    ensure_tools(args)
    solvers = load_solvers(args)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    m_plan = make_plan(args)
    k_values_plan = make_k_values_plan(args, m_plan)
    selected_inputs = prepare_inputs(args, m_plan, k_values_plan)

    raw_path = out_dir / "raw_results.tsv"
    raw_fields = [
        "finished_at",
        "solver",
        "seed",
        "n",
        "m",
        "k",
        "leading_zero_bits",
        "target_bits",
        "circuit_id",
        "y_id",
        "target",
        "status",
        "wall_time",
        "process_time",
        "conflicts",
        "remaining",
        "remaining_pct",
        "variables_original",
        "returncode",
        "timed_out",
        "cnf_vars",
        "cnf_clauses",
        "cnf",
        "solver_out",
    ]
    rows, completed = load_completed(raw_path) if args.append_existing else ([], set())
    rows_obj: list[dict[str, object]] = [dict(r) for r in rows]
    lock = threading.Lock()
    sequences: list[tuple[dict[str, object], Solver, int]] = []
    if args.render_only:
        planned_runs = len(rows_obj)
    else:
        missing_runs = 0
        for item in selected_inputs:
            for seed in seeds:
                for solver in solvers:
                    sequence_has_work = False
                    for m in item["m_values"]:  # ascending by construction
                        for k in item["k_values"]:
                            row_stub = {
                                "n": item["n"],
                                "circuit_id": item["circuit_id"],
                                "y_id": item["y_id"],
                                "m": m,
                                "k": k,
                                "solver": solver.name,
                                "seed": seed,
                            }
                            if row_key(row_stub) not in completed:
                                missing_runs += 1
                                sequence_has_work = True
                    if sequence_has_work:
                        sequences.append((item, solver, seed))

        planned_runs = len(completed) + missing_runs

    def update_outputs(active_workers: int = 0) -> None:
        agg = aggregate(rows_obj, args.time_limit)
        write_tsv(raw_path, rows_obj, raw_fields)
        agg_fields = list(agg[0].keys()) if agg else [
            "solver",
            "n",
            "m",
            "k",
            "runs",
            "sat",
            "unsat",
            "unknown",
            "unknown_rate",
            "mean_wall",
            "median_wall",
            "variance_wall",
            "stddev_wall",
            "solved_mean_wall",
            "solved_variance_wall",
            "timeouts_at_limit",
        ]
        write_tsv(out_dir / "aggregate.tsv", agg, agg_fields)
        progress = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "planned_runs": planned_runs,
            "completed_runs": len(rows_obj),
            "active_workers": active_workers,
            "time_limit": args.time_limit,
            "preset": args.preset,
            "k": args.k,
            "k_semantics": "leading_zero_bits" if args.k_is_leading_zero else "legacy_n_over_2_minus_k",
            "k_values_plan": k_values_plan,
            "solvers": [s.name for s in solvers],
            "m_plan": m_plan,
        }
        (out_dir / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        write_html(out_dir / "summary.html", rows_obj, agg, progress)

    def run_one(item: dict[str, object], m: int, k: int, solver: Solver, seed: int) -> dict[str, object] | None:
        n = int(item["n"])
        cid = int(item["circuit_id"])
        y_id = int(item["y_id"])
        prefix_path = Path(args.out_dir) / "prefixes" / f"n{n}_c{cid:03d}_m{m}.txt"
        safe_target = str(item["target"]).replace("0x", "")
        leading_zero_bits = k if args.k_is_leading_zero else n // 2 - k
        stem = f"n{n}_c{cid:03d}_y{y_id:02d}_m{m}_k{k}_{solver.name}_s{seed}"
        cnf = cnf_dir / f"{stem}.cnf"
        out = solver_out_dir / f"{stem}.out"
        meta_path = solver_out_dir / f"{stem}.meta"
        if not cnf.exists():
            subprocess.run(
                [
                    args.converter,
                    str(prefix_path),
                    str(cnf),
                    str(n),
                    str(item["target"]),
                    str(leading_zero_bits),
                    str(item["target_bits"]),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        cnf_vars, cnf_clauses = read_cnf_header(cnf)
        cmd = solver_command(solver, seed, args.time_limit, cnf, out)
        rc, text, wall, timed_out = run_command(cmd, args.time_limit)
        out.write_text(text)
        meta = f"real {wall:.3f}\nreturncode {rc}\ntimed_out {int(timed_out)}\ncmd {json.dumps(cmd)}\n"
        meta_path.write_text(meta)
        sol_path = out.with_suffix(".sol")
        sol_text = sol_path.read_text(errors="replace") if sol_path.exists() else ""
        parsed = parse_solver_output(text + "\n" + sol_text, meta)
        status = str(parsed["status"])
        row: dict[str, object] = {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "solver": solver.name,
            "seed": seed,
            "n": n,
            "m": m,
            "k": k,
            "leading_zero_bits": leading_zero_bits,
            "target_bits": item["target_bits"],
            "circuit_id": cid,
            "y_id": y_id,
            "target": item["target"],
            "status": status,
            "wall_time": round(wall, 3),
            "process_time": parsed["process_time"],
            "conflicts": parsed["conflicts"],
            "remaining": parsed["remaining"],
            "remaining_pct": parsed["remaining_pct"],
            "variables_original": parsed["variables_original"],
            "returncode": rc,
            "timed_out": int(timed_out),
            "cnf_vars": cnf_vars,
            "cnf_clauses": cnf_clauses,
            "cnf": str(cnf),
            "solver_out": str(out),
        }
        with lock:
            rows_obj.append(row)
            completed.add(row_key(row))
            update_outputs(active_workers=args.concurrency)
        print(json.dumps(row, sort_keys=True), flush=True)
        return row

    def run_sequence(item: dict[str, object], solver: Solver, seed: int) -> None:
        for m in item["m_values"]:
            for k in item["k_values"]:
                row_stub = {
                    "n": item["n"],
                    "circuit_id": item["circuit_id"],
                    "y_id": item["y_id"],
                    "m": m,
                    "k": k,
                    "solver": solver.name,
                    "seed": seed,
                }
                if row_key(row_stub) in completed:
                    continue
                row = run_one(item, int(m), int(k), solver, seed)
                if row and row.get("status") == "UNKNOWN" and float(row.get("wall_time") or 0) >= args.time_limit:
                    break

    update_outputs()
    if args.render_only:
        return 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_sequence, item, solver, seed) for item, solver, seed in sequences]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                err = {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": repr(exc),
                }
                (out_dir / "last_error.json").write_text(json.dumps(err, indent=2) + "\n")
                print(json.dumps(err), file=sys.stderr, flush=True)
                raise
    update_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
