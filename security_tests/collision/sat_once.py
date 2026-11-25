#!/usr/bin/env python3
"""Solve one collision CNF (parent enforces wall-clock via subprocess timeout).

Usage: sat_once.py CIRCUIT.mpmct1 ENCODER λ [--progress-conflicts N] [--progress-log FILE]

Prints one JSON object on stdout. Exit 0 on solve, 1 on error.
On success, includes a decoded witness (x1_hex, x2_hex, digest_hex) and the
full circuit outputs (output1_hex, output2_hex), garbage wires included.

Progress is written to stderr (and --progress-log) after each Glucose slice.
Glucose has no live statistics during a single solve() call, so the search
runs in conflict-budget slices. Learned clauses are kept between slices.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pysat.solvers import Glucose3


def mpmct_header(path: Path) -> tuple[int, int]:
    header = path.read_text().splitlines()[0].split()
    if len(header) != 3 or header[0] != "mpmct1":
        raise ValueError(f"invalid mpmct1 header: {path}")
    return int(header[1]), int(header[2])


def bits_to_int(bits: list[bool]) -> int:
    value = 0
    for index, bit in enumerate(bits):
        if bit:
            value |= 1 << index
    return value


def padded_hex(value: int, bits: int) -> str:
    return "0x" + format(value, f"0{(bits + 3) // 4}x")


def emit_progress(line: str, log_path: Path | None) -> None:
    print(line, file=sys.stderr, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("lam", type=int)
    parser.add_argument(
        "--progress-conflicts",
        type=int,
        default=20_000,
        help="Conflict budget of each Glucose slice (learned clauses are kept)",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="Also append [sat] progress lines to this file",
    )
    args = parser.parse_args()
    if args.progress_conflicts <= 0:
        raise SystemExit("--progress-conflicts must be positive")

    circuit = args.circuit
    encoder = args.encoder
    lam = args.lam
    if args.progress_log is not None:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        args.progress_log.write_text("")

    in_bits = 2 * lam
    pad = lam
    out_bits = lam
    wires, gates = mpmct_header(circuit)
    vars_per_copy = wires + gates

    with tempfile.TemporaryDirectory() as tmp:
        cnf = Path(tmp) / "c.cnf"
        t0 = time.perf_counter()
        encode_cpu0 = child_cpu_seconds()
        enc = subprocess.run(
            [
                str(encoder),
                str(circuit),
                str(cnf),
                "--in-bits",
                str(in_bits),
                "--pad",
                str(pad),
                "--out-bits",
                str(out_bits),
            ],
            capture_output=True,
            text=True,
        )
        encode_secs = time.perf_counter() - t0
        encode_cpu = child_cpu_seconds() - encode_cpu0
        if enc.returncode != 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "phase": "encode",
                        "stderr": (enc.stderr or "")[-400:],
                        "cpu_hours": encode_secs / 3600.0,
                        "threads": 1,
                    }
                )
            )
            return 1

        clauses: list[list[int]] = []
        for line in cnf.read_text().splitlines():
            if line.startswith(("c", "p")) or not line.strip():
                continue
            lits = [int(x) for x in line.split() if x != "0"]
            if lits:
                clauses.append(lits)

        emit_progress(
            f"[sat] encoded clauses={len(clauses)} encode_wall={encode_secs:.3f}s "
            f"progress_conflicts={args.progress_conflicts}",
            args.progress_log,
        )
        t1 = time.perf_counter()
        # Model must be read before the solver context exits.
        model = None
        cpu0 = time.process_time()
        wall0 = time.perf_counter()
        slices = 0
        stats: dict = {}
        with Glucose3(bootstrap_with=clauses, use_timer=True) as solver:
            status = None
            while status is None:
                solver.conf_budget(args.progress_conflicts)
                status = solver.solve_limited()
                slices += 1
                stats = solver.accum_stats()
                cpu = time.process_time() - cpu0
                wall_now = time.perf_counter() - wall0
                state = "running" if status is None else ("sat" if status else "unsat")
                emit_progress(
                    "[sat] {state} slice={slices} cpu={cpu:.3f}s wall={wall:.3f}s "
                    "conflicts={conflicts} decisions={decisions} "
                    "propagations={propagations} restarts={restarts}".format(
                        state=state,
                        slices=slices,
                        cpu=cpu,
                        wall=wall_now,
                        **stats,
                    ),
                    args.progress_log,
                )
            sat = bool(status)
            model = solver.get_model() if sat else None
            solve_cpu = time.process_time() - cpu0
        wall = time.perf_counter() - t1

        record: dict = {
            "ok": bool(sat),
            "wall_secs": wall,
            "cpu_hours": (encode_cpu + solve_cpu) / 3600.0,
            "cpu_seconds": encode_cpu + solve_cpu,
            "encode_cpu_seconds": encode_cpu,
            "solve_cpu_seconds": solve_cpu,
            "threads": 1,
            "encode_secs": encode_secs,
            "clauses": len(clauses),
            "sat": bool(sat),
            "slices": slices,
            "stats": stats,
        }
        if not sat or model is None:
            print(json.dumps(record))
            return 0

        assignment = {abs(lit): lit > 0 for lit in model}

        def read_bits(start: int, count: int) -> list[bool]:
            bits: list[bool] = []
            for i in range(count):
                var = start + i
                if var not in assignment:
                    raise RuntimeError(f"model omits variable {var}")
                bits.append(assignment[var])
            return bits

        x1 = bits_to_int(read_bits(1, in_bits))
        x2 = bits_to_int(read_bits(vars_per_copy + 1, in_bits))
        if x1 == x2:
            record["ok"] = False
            record["phase"] = "equal_inputs"
            print(json.dumps(record))
            return 1

        # Independent digest check on the mpmct1 tape (pad bits stay zero).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from decode_collision_model import evaluate_mpmct1  # noqa: E402

        out1, _, _ = evaluate_mpmct1(circuit, x1)
        out2, _, _ = evaluate_mpmct1(circuit, x2)
        mask = (1 << out_bits) - 1
        d1 = out1 & mask
        d2 = out2 & mask
        if d1 != d2:
            record["ok"] = False
            record["phase"] = "digest_mismatch"
            record["report"] = {
                "x1_hex": padded_hex(x1, in_bits),
                "x2_hex": padded_hex(x2, in_bits),
                "digest1_hex": padded_hex(d1, out_bits),
                "digest2_hex": padded_hex(d2, out_bits),
            }
            print(json.dumps(record))
            return 1

        record["report"] = {
            "circuit": str(circuit),
            "in_bits": in_bits,
            "pad": pad,
            "out_bits": out_bits,
            "wires": wires,
            "gates": gates,
            "x1_hex": padded_hex(x1, in_bits),
            "x2_hex": padded_hex(x2, in_bits),
            "digest_hex": padded_hex(d1, out_bits),
            "output1_hex": padded_hex(out1, wires),
            "output2_hex": padded_hex(out2, wires),
            "search": "sat_glucose3",
            "verified": True,
        }
        print(json.dumps(record))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
