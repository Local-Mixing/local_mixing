#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from run_lowtarget_knee import parse_circuit, parse_kissat_output, read_cnf_header, write_prefix


def write_tsv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["leading_zero_bits"], []).append(row)

    out = []
    for k in sorted(grouped):
        vals = grouped[k]
        times = [float(v["wall_time"]) for v in vals]
        solved = [float(v["wall_time"]) for v in vals if v["kissat_status"] == "SAT"]
        unsat = sum(v["kissat_status"] == "UNSAT" for v in vals)
        unknown = sum(v["kissat_status"] == "UNKNOWN" for v in vals)
        sat = sum(v["kissat_status"] == "SAT" for v in vals)
        out.append(
            {
                "leading_zero_bits": k,
                "expected_solutions": vals[0]["expected_solutions"],
                "n": len(vals),
                "sat": sat,
                "unsat": unsat,
                "unknown": unknown,
                "sat_rate": sat / len(vals),
                "unknown_rate": unknown / len(vals),
                "mean_wall_time": statistics.fmean(times),
                "median_wall_time": statistics.median(times),
                "stdev_wall_time": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min_wall_time": min(times),
                "max_wall_time": max(times),
                "solved_mean_wall_time": statistics.fmean(solved) if solved else "",
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--gates", type=int, default=600)
    parser.add_argument("--k-start", type=int, default=20)
    parser.add_argument("--k-end", type=int, default=40)
    parser.add_argument("--k-step", type=int, default=1)
    parser.add_argument("--target-bits", type=int, default=32)
    parser.add_argument("--target", default="0x91c16f14e5c78e00")
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--out-dir", default="work/sss_challenge/random64_m600_leading_zero_sweep")
    parser.add_argument("--genran", default="target/release/local_mixing_bin")
    parser.add_argument("--converter", default="work/sss_challenge/circuit_to_cnf_lowtarget_leading0_generic")
    parser.add_argument("--kissat", default="work/sss_challenge/kissat_src_verbose/build/kissat")
    parser.add_argument("--append-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.tsv"
    rows = []
    if args.append_existing and raw_path.exists():
        with raw_path.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                row["replicate"] = int(row["replicate"])
                row["leading_zero_bits"] = int(row["leading_zero_bits"])
                rows.append(row)

    completed = {(int(r["replicate"]), int(r["leading_zero_bits"])) for r in rows}

    for rep in range(args.replicates):
        rep_dir = out_dir / f"rep_{rep:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        source = rep_dir / f"randomn{args.n}m{args.gates}.txt"
        if not source.exists():
            subprocess.run(
                [
                    args.genran,
                    "genran",
                    "-n",
                    str(args.n),
                    "-m",
                    str(args.gates),
                    "-d",
                    str(source),
                ],
                check=True,
            )
        gates = parse_circuit(source.read_text())
        prefix = rep_dir / f"prefix_m{args.gates}.txt"
        write_prefix(gates, args.gates, prefix)

        for k in range(args.k_start, args.k_end + 1, args.k_step):
            if (rep, k) in completed:
                continue
            cnf = rep_dir / f"k{k:02d}_low{args.target_bits}_top{k}_zero.cnf"
            out = rep_dir / f"k{k:02d}_kissat_{args.time_limit}s.out"
            meta = rep_dir / f"k{k:02d}_kissat_{args.time_limit}s.meta"
            subprocess.run(
                [
                    args.converter,
                    str(prefix),
                    str(cnf),
                    str(args.n),
                    args.target,
                    str(k),
                    str(args.target_bits),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            cnf_vars, cnf_clauses = read_cnf_header(cnf)
            cmd = [args.kissat, f"--time={args.time_limit}", "--sat", "-v", str(cnf)]
            start = time.monotonic()
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            wall = time.monotonic() - start
            out.write_text(proc.stdout)
            meta.write_text(f"real {wall:.2f}\nreturncode {proc.returncode}\n")
            parsed = parse_kissat_output(proc.stdout)
            expected_log2 = args.n - k - args.target_bits
            expected = f"2^{expected_log2}" if expected_log2 >= 0 else f"2^({expected_log2})"
            row = {
                "replicate": rep,
                "leading_zero_bits": k,
                "expected_solutions": expected,
                "expected_log2": expected_log2,
                "kissat_status": parsed["status"],
                "wall_time": round(wall, 2),
                "process_time": parsed["process_time"],
                "conflicts": parsed["conflicts"],
                "remaining": parsed["remaining"],
                "remaining_pct": parsed["remaining_pct"],
                "variables_original": parsed["variables_original"],
                "cnf_vars": cnf_vars,
                "cnf_clauses": cnf_clauses,
                "returncode": proc.returncode,
                "cnf": str(cnf),
                "kissat_out": str(out),
            }
            rows.append(row)
            write_tsv(raw_path, rows)
            agg = aggregate(rows)
            write_tsv(out_dir / "aggregate.tsv", agg)
            (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
