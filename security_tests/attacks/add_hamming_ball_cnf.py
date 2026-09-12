#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def parse_hex_bits(text: str, count: int):
    if text.startswith(("0x", "0X")):
        text = text[2:]
    value = int(text or "0", 16)
    return [(value >> i) & 1 for i in range(count)]


def read_cnf(path: Path):
    text = path.read_text()
    match = re.search(r"^p cnf (\d+) (\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing DIMACS header in {path}")
    return text, int(match.group(1)), int(match.group(2)), match


def add_at_most_k(clauses, next_var, lits, k):
    if k < 0:
        clauses.append([])
        return next_var
    if k >= len(lits):
        return next_var

    n = len(lits)
    s = [[0] * (k + 1) for _ in range(n)]
    for i in range(n):
        for j in range(1, k + 1):
            s[i][j] = next_var
            next_var += 1

    for i, lit in enumerate(lits):
        clauses.append([-lit, s[i][1]])
        if i == 0:
            for j in range(2, k + 1):
                clauses.append([-s[i][j]])
            continue
        clauses.append([-lit, -s[i - 1][k]])
        for j in range(1, k + 1):
            clauses.append([-s[i - 1][j], s[i][j]])
        for j in range(2, k + 1):
            clauses.append([-lit, -s[i - 1][j - 1], s[i][j]])
    return next_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center-hex", required=True)
    parser.add_argument("--radius", required=True, type=int)
    parser.add_argument("--var-start", default=129, type=int)
    parser.add_argument("--var-count", default=128, type=int)
    args = parser.parse_args()

    text, vars_count, clauses_count, _ = read_cnf(args.base)
    center_bits = parse_hex_bits(args.center_hex, args.var_count)
    diff_lits = []
    for i, bit in enumerate(center_bits):
        var = args.var_start + i
        diff_lits.append(-var if bit else var)

    extra = []
    next_var = vars_count + 1
    next_var = add_at_most_k(extra, next_var, diff_lits, args.radius)
    new_vars = next_var - 1
    new_clauses = clauses_count + len(extra)
    header = f"p cnf {new_vars} {new_clauses}"
    text = re.sub(r"^p cnf \d+ \d+\s*$", header, text, count=1, flags=re.MULTILINE)

    with args.out.open("w") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        for clause in extra:
            if clause:
                f.write(" ".join(str(lit) for lit in clause))
                f.write(" 0\n")
            else:
                f.write("0\n")
    print(f"vars {new_vars}")
    print(f"clauses {new_clauses}")
    print(f"extra_clauses {len(extra)}")


if __name__ == "__main__":
    main()
