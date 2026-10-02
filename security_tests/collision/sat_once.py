#!/usr/bin/env python3
"""Solve one collision CNF (parent enforces wall-clock via subprocess timeout).

Usage: sat_once.py CIRCUIT.mpmct1 ENCODER λ
Prints one JSON object on stdout. Exit 0 on solve, 1 on error.
On success, includes a decoded witness (x1_hex, x2_hex, digest_hex).
"""

from __future__ import annotations

import json
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


def main() -> int:
    circuit = Path(sys.argv[1])
    encoder = Path(sys.argv[2])
    lam = int(sys.argv[3])
    in_bits = 2 * lam
    pad = lam
    out_bits = lam
    wires, gates = mpmct_header(circuit)
    vars_per_copy = wires + gates

    with tempfile.TemporaryDirectory() as tmp:
        cnf = Path(tmp) / "c.cnf"
        t0 = time.perf_counter()
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

        t1 = time.perf_counter()
        with Glucose3(bootstrap_with=clauses) as solver:
            sat = bool(solver.solve())
            model = solver.get_model() if sat else None
        wall = time.perf_counter() - t1

        record: dict = {
            "ok": bool(sat),
            "wall_secs": wall,
            "cpu_hours": (encode_secs + wall) / 3600.0,
            "cpu_seconds": encode_secs + wall,
            "threads": 1,
            "encode_secs": encode_secs,
            "clauses": len(clauses),
            "sat": bool(sat),
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
            "search": "sat_glucose3",
            "verified": True,
        }
        print(json.dumps(record))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
