#!/usr/bin/env python3
import argparse
import random
import re
import subprocess
import sys
from pathlib import Path


def read_dimacs(path: Path):
    text = path.read_text()
    match = re.search(r"^p cnf (\d+) (\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing DIMACS header in {path}")
    return text, int(match.group(1)), int(match.group(2)), match


def write_cube_cnf(base_text: str, vars_count: int, clauses_count: int, units, out: Path):
    header = f"p cnf {vars_count} {clauses_count + len(units)}"
    text = re.sub(r"^p cnf \d+ \d+\s*$", header, base_text, count=1, flags=re.MULTILINE)
    with out.open("w") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        for lit in units:
            f.write(f"{lit} 0\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cnf", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--kissat", required=True, type=Path)
    parser.add_argument("--cubes", type=int, default=8)
    parser.add_argument("--fixed-bits", type=int, default=14)
    parser.add_argument("--var-start", type=int, default=1)
    parser.add_argument("--var-count", type=int, default=78)
    parser.add_argument("--center-hex", default="")
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_text, vars_count, clauses_count, _ = read_dimacs(args.base_cnf)
    rng = random.Random(args.seed)
    free_vars = list(range(args.var_start, args.var_start + args.var_count))
    center = int(args.center_hex, 16) if args.center_hex else None

    for idx in range(args.cubes):
        chosen = sorted(rng.sample(free_vars, args.fixed_bits))
        units = []
        for var in chosen:
            if center is None:
                bit = rng.getrandbits(1)
            else:
                bit = (center >> (var - args.var_start)) & 1
            units.append(var if bit else -var)
        cnf = args.out_dir / f"input_cube_{idx:03d}.cnf"
        out = args.out_dir / f"input_cube_{idx:03d}.out"
        err = args.out_dir / f"input_cube_{idx:03d}.err"
        write_cube_cnf(base_text, vars_count, clauses_count, units, cnf)
        cmd = [
            str(args.kissat),
            "--sat",
            f"--time={args.seconds}",
            f"--seed={args.seed + idx}",
            str(cnf),
        ]
        with out.open("w") as so, err.open("w") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se)
        output = out.read_text(errors="replace")
        status = "UNKNOWN"
        if "s SATISFIABLE" in output:
            status = "SAT"
        elif "s UNSATISFIABLE" in output:
            status = "UNSAT"
        print(
            f"cube {idx} status={status} return={proc.returncode} fixed={args.fixed_bits} out={out}",
            flush=True,
        )
        if status == "SAT":
            print(str(out))
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
