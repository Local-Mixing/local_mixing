#!/usr/bin/env python3
"""Solve one collision CNF (parent enforces wall-clock via subprocess timeout).

Usage: sat_once.py CIRCUIT.mpmct1 ENCODER λ
Prints one JSON object on stdout. Exit 0 on solve, 1 on error.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pysat.solvers import Glucose3


def main() -> int:
    circuit = Path(sys.argv[1])
    encoder = Path(sys.argv[2])
    lam = int(sys.argv[3])

    with tempfile.TemporaryDirectory() as tmp:
        cnf = Path(tmp) / "c.cnf"
        t0 = time.perf_counter()
        enc = subprocess.run(
            [
                str(encoder),
                str(circuit),
                str(cnf),
                "--in-bits",
                str(2 * lam),
                "--pad",
                str(lam),
                "--out-bits",
                str(lam),
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
        wall = time.perf_counter() - t1
        print(
            json.dumps(
                {
                    "ok": sat,
                    "wall_secs": wall,
                    "cpu_hours": (encode_secs + wall) / 3600.0,
                    "threads": 1,
                    "encode_secs": encode_secs,
                    "clauses": len(clauses),
                    "sat": sat,
                }
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
