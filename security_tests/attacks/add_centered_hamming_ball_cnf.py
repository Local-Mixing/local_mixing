#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def parse_hex_bits(text: str, count: int):
    if text.startswith(("0x", "0X")):
        text = text[2:]
    value = int(text or "0", 16)
    return [(value >> i) & 1 for i in range(count)]


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

    center_bits = parse_hex_bits(args.center_hex, args.var_count)
    header_re = re.compile(r"^p cnf (\d+) (\d+)\s*$")
    vars_count = None
    clauses_count = None
    body_lines = []

    with args.base.open() as f:
        for line in f:
          match = header_re.match(line)
          if match:
              vars_count = int(match.group(1))
              clauses_count = int(match.group(2))
              body_lines.append(None)
              continue
          if line.startswith("c") or not line.strip():
              body_lines.append(line)
              continue
          out_lits = []
          for raw in line.split():
              lit = int(raw)
              if lit == 0:
                  break
              var = abs(lit)
              idx = var - args.var_start
              if 0 <= idx < args.var_count and center_bits[idx]:
                  lit = -lit
              out_lits.append(str(lit))
          body_lines.append(" ".join(out_lits) + " 0\n")

    if vars_count is None or clauses_count is None:
        raise SystemExit(f"missing DIMACS header in {args.base}")

    extra = []
    next_var = vars_count + 1
    next_var = add_at_most_k(
        extra,
        next_var,
        list(range(args.var_start, args.var_start + args.var_count)),
        args.radius,
    )
    new_vars = next_var - 1
    new_clauses = clauses_count + len(extra)
    header = f"p cnf {new_vars} {new_clauses}\n"

    with args.out.open("w") as f:
        header_written = False
        for line in body_lines:
            if line is None:
                f.write(header)
                header_written = True
            else:
                f.write(line)
        if not header_written:
            raise SystemExit("internal error: header not written")
        for clause in extra:
            if clause:
                f.write(" ".join(str(lit) for lit in clause) + " 0\n")
            else:
                f.write("0\n")
    print(f"vars {new_vars}")
    print(f"clauses {new_clauses}")
    print(f"extra_clauses {len(extra)}")


if __name__ == "__main__":
    main()
