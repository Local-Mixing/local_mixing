#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import time
from pathlib import Path


CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"


def parse_circuit(text):
    gates = []
    wires = []
    overflow = 0
    for ch in text:
        if ch == ";":
            if wires:
                if len(wires) != 3:
                    raise ValueError("bad gate")
                gates.append(tuple(wires))
                wires = []
            overflow = 0
        elif ch == "~":
            overflow += 1
        elif ch.isspace():
            continue
        else:
            base = CHARS.find(ch)
            if base < 0:
                raise ValueError(f"bad char {ch!r}")
            wires.append(base + 83 * overflow)
            overflow = 0
    if wires:
        raise ValueError("unterminated gate")
    return gates


def gate_text(gate):
    out = []
    for wire in gate:
        overflow, base = divmod(wire, 83)
        out.append("~" * overflow + CHARS[base])
    return "".join(out) + ";"


def write_prefix(gates, count, path):
    path.write_text("".join(gate_text(g) for g in gates[:count]))


def evaluate(state, gates):
    for a, b, c in gates:
        c1 = (state >> b) & 1
        c2 = (state >> c) & 1
        state ^= (c1 | (1 ^ c2)) << a
    return state


def brute_force_lowtarget(gates, n, target, leading_zero_bits, target_bits):
    free = n - leading_zero_bits
    if free > 24:
        return {"checked": None, "sat": None, "witness": None}
    mask = (1 << target_bits) - 1
    want = target & mask
    for x in range(1 << free):
        if (evaluate(x, gates) & mask) == want:
            return {"checked": 1 << free, "sat": True, "witness": x}
    return {"checked": 1 << free, "sat": False, "witness": None}


def parse_kissat_output(text):
    if "s SATISFIABLE" in text:
        status = "SAT"
    elif "s UNSATISFIABLE" in text:
        status = "UNSAT"
    else:
        status = "UNKNOWN"

    conflicts = None
    process_time = None
    variables_original = None
    remaining = None
    remaining_pct = None

    for line in text.splitlines():
        m = re.search(r"^c conflicts:\s+([0-9]+)", line)
        if m:
            conflicts = int(m.group(1))
        m = re.search(r"^c process-time:\s+(?:.*\s)?([0-9]+(?:\.[0-9]+)?) seconds", line)
        if m:
            process_time = float(m.group(1))
        m = re.search(r"^c variables_original:\s+([0-9]+)", line)
        if m:
            variables_original = int(m.group(1))
        m = re.search(r"\s([0-9]+)\s+([0-9]+)%\s*$", line)
        if line.startswith("c") and m:
            remaining = int(m.group(1))
            remaining_pct = int(m.group(2))

    return {
        "status": status,
        "conflicts": conflicts,
        "process_time": process_time,
        "variables_original": variables_original,
        "remaining": remaining,
        "remaining_pct": remaining_pct,
    }


def read_cnf_header(path):
    with path.open() as f:
        for line in f:
            if line.startswith("p cnf "):
                _, _, var_count, clause_count = line.split()
                return int(var_count), int(clause_count)
    raise ValueError(f"missing cnf header in {path}")


def write_summaries(rows, out_dir):
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    fields = [
        "gates",
        "kissat_status",
        "wall_time",
        "conflicts",
        "remaining",
        "remaining_pct",
        "cnf_vars",
        "cnf_clauses",
        "bruteforce_sat",
        "bruteforce_witness_hex",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join("" if row.get(k) is None else str(row.get(k)) for k in fields))
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--max-gates", type=int, default=8000)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--leading-zero-bits", type=int, default=50)
    parser.add_argument("--target-bits", type=int, default=0)
    parser.add_argument("--target", default="0x91c16f14e5c78e00")
    parser.add_argument("--out-dir", default="work/sss_challenge/random64_gate_search")
    parser.add_argument("--source", default="")
    parser.add_argument("--genran", default="target/release/local_mixing_bin")
    parser.add_argument("--converter", default="work/sss_challenge/circuit_to_cnf_lowtarget_leading0_generic")
    parser.add_argument("--kissat", default="work/sss_challenge/kissat_src_verbose/build/kissat")
    parser.add_argument("--reuse-source", action="store_true")
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--bruteforce", choices=["all", "none"], default="all")
    parser.add_argument("--continue-after-unknown", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source) if args.source else out_dir / f"randomn{args.n}m{args.max_gates}.txt"
    if not source.exists() or not args.reuse_source:
        subprocess.run(
            [
                args.genran,
                "genran",
                "-n",
                str(args.n),
                "-m",
                str(args.max_gates),
                "-d",
                str(source),
            ],
            check=True,
        )

    gates = parse_circuit(source.read_text())
    if len(gates) < args.max_gates:
        raise ValueError(f"{source} only has {len(gates)} gates")

    target = int(args.target, 0)
    target_bits = args.target_bits or min(64, args.n)
    summary_path = out_dir / "summary.json"
    rows = json.loads(summary_path.read_text()) if args.append_existing and summary_path.exists() else []
    for count in range(args.start, args.max_gates + 1, args.step):
        prefix = out_dir / f"prefix_m{count}.txt"
        cnf = out_dir / f"prefix_m{count}_low{target_bits}_top{args.leading_zero_bits}_zero.cnf"
        out = out_dir / f"prefix_m{count}_kissat_{args.time_limit}s.out"
        meta = out_dir / f"prefix_m{count}_kissat_{args.time_limit}s.meta"

        write_prefix(gates, count, prefix)
        subprocess.run(
            [
                args.converter,
                str(prefix),
                str(cnf),
                str(args.n),
                args.target,
                str(args.leading_zero_bits),
                str(target_bits),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cnf_vars, cnf_clauses = read_cnf_header(cnf)
        if args.bruteforce == "all":
            brute = brute_force_lowtarget(
                gates[:count], args.n, target, args.leading_zero_bits, target_bits
            )
        else:
            brute = {"checked": None, "sat": None, "witness": None}

        cmd = [args.kissat, f"--time={args.time_limit}", "--sat", "-v", str(cnf)]
        start = time.monotonic()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wall = time.monotonic() - start
        out.write_text(proc.stdout)
        meta.write_text(f"real {wall:.2f}\nreturncode {proc.returncode}\n")
        parsed = parse_kissat_output(proc.stdout)

        row = {
            "gates": count,
            "cnf_vars": cnf_vars,
            "cnf_clauses": cnf_clauses,
            "kissat_status": parsed["status"],
            "returncode": proc.returncode,
            "wall_time": round(wall, 2),
            "process_time": parsed["process_time"],
            "conflicts": parsed["conflicts"],
            "remaining": parsed["remaining"],
            "remaining_pct": parsed["remaining_pct"],
            "variables_original": parsed["variables_original"],
            "target_bits": target_bits,
            "bruteforce_checked": brute["checked"],
            "bruteforce_sat": brute["sat"],
            "bruteforce_witness_hex": None if brute["witness"] is None else hex(brute["witness"]),
            "cnf": str(cnf),
            "kissat_out": str(out),
        }
        rows.append(row)
        write_summaries(rows, out_dir)
        print(json.dumps(row, sort_keys=True), flush=True)

        if parsed["status"] == "UNKNOWN" and not args.continue_after_unknown:
            break


if __name__ == "__main__":
    main()
