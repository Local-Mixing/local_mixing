#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path


def read_summary(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(value):
    return None if value == "" else float(value)


def aggregate(replicate_rows):
    by_gate = {}
    for rep, rows in replicate_rows:
        for row in rows:
            gate = int(row["gates"])
            wall = as_float(row["wall_time"])
            timeout = row["kissat_status"] == "UNKNOWN"
            by_gate.setdefault(gate, []).append(
                {
                    "replicate": rep,
                    "wall_time": wall,
                    "status": row["kissat_status"],
                    "timeout": timeout,
                    "conflicts": None if row["conflicts"] == "" else int(row["conflicts"]),
                    "remaining": None if row["remaining"] == "" else int(row["remaining"]),
                    "remaining_pct": None if row["remaining_pct"] == "" else int(row["remaining_pct"]),
                }
            )

    rows = []
    for gate in sorted(by_gate):
        vals = by_gate[gate]
        times = [v["wall_time"] for v in vals if v["wall_time"] is not None]
        solved_times = [v["wall_time"] for v in vals if v["wall_time"] is not None and not v["timeout"]]
        timeout_count = sum(v["timeout"] for v in vals)
        mean = statistics.fmean(times) if times else None
        median = statistics.median(times) if times else None
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        solved_mean = statistics.fmean(solved_times) if solved_times else None
        log_solved_mean = (
            math.exp(statistics.fmean(math.log(t) for t in solved_times if t > 0))
            if any(t > 0 for t in solved_times)
            else None
        )
        rows.append(
            {
                "gates": gate,
                "n": len(vals),
                "timeouts": timeout_count,
                "timeout_rate": timeout_count / len(vals),
                "mean_wall_time": mean,
                "median_wall_time": median,
                "stdev_wall_time": stdev,
                "min_wall_time": min(times) if times else None,
                "max_wall_time": max(times) if times else None,
                "solved_mean_wall_time": solved_mean,
                "solved_geomean_wall_time": log_solved_mean,
            }
        )
    return rows


def write_tsv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_raw(path, replicate_rows):
    fields = [
        "replicate",
        "gates",
        "kissat_status",
        "wall_time",
        "conflicts",
        "remaining",
        "remaining_pct",
        "cnf_vars",
        "cnf_clauses",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rep, rows in replicate_rows:
            for row in rows:
                writer.writerow({k: rep if k == "replicate" else row.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--end", type=int, default=700)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--leading-zero-bits", type=int, default=25)
    parser.add_argument("--target-bits", type=int, default=32)
    parser.add_argument("--target", default="0x91c16f14e5c78e00")
    parser.add_argument("--out-dir", default="work/sss_challenge/random64_gate_search_scaled32_25_replicates")
    parser.add_argument("--runner", default="tools/run_lowtarget_knee.py")
    parser.add_argument("--append-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    replicate_rows = []
    for rep in range(args.replicates):
        rep_dir = out_dir / f"rep_{rep:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        summary = rep_dir / "summary.tsv"
        if args.append_existing and summary.exists():
            rows = read_summary(summary)
            replicate_rows.append((rep, rows))
            continue

        cmd = [
            "python3",
            args.runner,
            "--n",
            str(args.n),
            "--start",
            str(args.start),
            "--step",
            str(args.step),
            "--max-gates",
            str(args.end),
            "--time-limit",
            str(args.time_limit),
            "--leading-zero-bits",
            str(args.leading_zero_bits),
            "--target-bits",
            str(args.target_bits),
            "--target",
            args.target,
            "--out-dir",
            str(rep_dir),
            "--bruteforce",
            "none",
            "--continue-after-unknown",
        ]
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        rows = read_summary(summary)
        replicate_rows.append((rep, rows))

        write_raw(out_dir / "raw_results.tsv", replicate_rows)
        agg = aggregate(replicate_rows)
        write_tsv(out_dir / "aggregate.tsv", agg)
        (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")

    write_raw(out_dir / "raw_results.tsv", replicate_rows)
    agg = aggregate(replicate_rows)
    write_tsv(out_dir / "aggregate.tsv", agg)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
