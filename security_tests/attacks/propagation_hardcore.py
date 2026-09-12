#!/usr/bin/env python3
from __future__ import annotations

import itertools
import sys
from array import array
from pathlib import Path

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
VAL = {c: i for i, c in enumerate(CHARS)}


def parse_circuit(path: Path) -> list[tuple[int, int, int]]:
    gates: list[tuple[int, int, int]] = []
    w: list[int] = []
    overflow = 0
    for ch in path.read_text():
        if ch == ";":
            if w:
                if len(w) != 3:
                    raise RuntimeError(f"bad gate arity {len(w)} near {len(gates)}: {w}")
                gates.append((w[0], w[1], w[2]))
                w.clear()
            overflow = 0
        elif ch == "~":
            overflow += 1
        elif ch in "\n\r\t ":
            continue
        else:
            if ch not in VAL:
                raise RuntimeError(f"bad char {ch!r} ord={ord(ch)}")
            w.append(VAL[ch] + 83 * overflow)
            overflow = 0
    if w:
        raise RuntimeError(f"unterminated gate: {w}")
    return gates


def forced_table() -> dict[tuple[int, int, int, int], list[tuple[int, int]]]:
    valid = []
    for x in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                n = x ^ (b | (1 - c))
                valid.append((x, b, c, n))

    forced: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}
    for vals in itertools.product((-1, 0, 1), repeat=4):
        rows = [
            row
            for row in valid
            if all(vals[i] < 0 or vals[i] == row[i] for i in range(4))
        ]
        outs: list[tuple[int, int]] = []
        if rows:
            for i in range(4):
                if vals[i] < 0:
                    possible = {row[i] for row in rows}
                    if len(possible) == 1:
                        outs.append((i, next(iter(possible))))
        forced[vals] = outs
    return forced


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: propagation_hardcore.py circuit.txt target_e_hex", file=sys.stderr)
        return 2

    circuit = Path(sys.argv[1])
    target = int(sys.argv[2], 16)
    gates = parse_circuit(circuit)
    n_gates = len(gates)

    state = list(range(384))
    x_vars = array("i")
    b_vars = array("i")
    c_vars = array("i")
    n_vars = array("i")
    target_wires = array("H")

    for j, (a, b, c) in enumerate(gates):
        new_var = 384 + j
        x_vars.append(state[a])
        b_vars.append(state[b])
        c_vars.append(state[c])
        n_vars.append(new_var)
        target_wires.append(a)
        state[a] = new_var

    n_vars_total = 384 + n_gates
    assignments = array("b", [-1]) * n_vars_total
    first_round = array("i", [-1]) * n_vars_total
    reason_gate = array("i", [-1]) * n_vars_total

    def set_var(var: int, bit: int, round_id: int, reason: int) -> bool:
        old = assignments[var]
        if old >= 0:
            if old != bit:
                raise RuntimeError(
                    f"conflict var={var} old={old} new={bit} round={round_id} reason={reason}"
                )
            return False
        assignments[var] = bit
        first_round[var] = round_id
        reason_gate[var] = reason
        return True

    # Input B=0 and final E=target. A and C are left free.
    for i in range(128):
        set_var(128 + i, 0, 0, -2)
        set_var(state[128 + i], (target >> i) & 1, 0, -3)

    table = forced_table()
    print(
        "gates",
        n_gates,
        "vars",
        n_vars_total,
        "initial_known",
        sum(1 for value in assignments if value >= 0),
        flush=True,
    )

    for round_id in range(1, 201):
        changed = 0
        gate_range = range(n_gates) if round_id % 2 else range(n_gates - 1, -1, -1)
        for gate_id in gate_range:
            vars4 = (
                x_vars[gate_id],
                b_vars[gate_id],
                c_vars[gate_id],
                n_vars[gate_id],
            )
            vals = tuple(assignments[var] for var in vars4)
            for pos, bit in table[vals]:
                if set_var(vars4[pos], bit, round_id, gate_id):
                    changed += 1

        known = sum(1 for value in assignments if value >= 0)
        print(
            "round",
            round_id,
            "changed",
            changed,
            "known",
            known,
            "unknown",
            n_vars_total - known,
            flush=True,
        )
        if changed == 0:
            break

    bins = 40
    assigned_bins = [0] * bins
    total_bins = [0] * bins
    round_sum = [0] * bins
    round_count = [0] * bins
    for gate_id in range(n_gates):
        var = 384 + gate_id
        bin_id = min(bins - 1, gate_id * bins // n_gates)
        total_bins[bin_id] += 1
        if assignments[var] >= 0:
            assigned_bins[bin_id] += 1
            round_sum[bin_id] += first_round[var]
            round_count[bin_id] += 1

    print("BIN index start end assigned total pct_assigned avg_round unknown")
    for bin_id in range(bins):
        start = bin_id * n_gates // bins
        end = ((bin_id + 1) * n_gates // bins) - 1
        avg = round_sum[bin_id] / round_count[bin_id] if round_count[bin_id] else -1
        unknown = total_bins[bin_id] - assigned_bins[bin_id]
        print(
            bin_id,
            start,
            end,
            assigned_bins[bin_id],
            total_bins[bin_id],
            f"{assigned_bins[bin_id] / total_bins[bin_id]:.4f}",
            f"{avg:.2f}",
            unknown,
        )

    wire_total = [0] * 384
    wire_unknown = [0] * 384
    for gate_id in range(n_gates):
        wire = target_wires[gate_id]
        wire_total[wire] += 1
        if assignments[384 + gate_id] < 0:
            wire_unknown[wire] += 1

    print("WIRE_BLOCK_UNKNOWN block unknown total pct_unknown")
    for name, lo in (("A", 0), ("B", 128), ("C", 256)):
        total = sum(wire_total[lo : lo + 128])
        unknown = sum(wire_unknown[lo : lo + 128])
        print(name, unknown, total, f"{unknown / total:.4f}")

    print("TOP_UNKNOWN_WIRES wire unknown total pct_unknown")
    for wire in sorted(range(384), key=lambda item: wire_unknown[item], reverse=True)[:20]:
        total = wire_total[wire]
        pct = wire_unknown[wire] / total if total else 0
        print(wire, wire_unknown[wire], total, f"{pct:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
