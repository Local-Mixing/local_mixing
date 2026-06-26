#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
CHAR_TO_VAL = {c: i for i, c in enumerate(CHARS)}
RADIX = len(CHARS)


def decode_circuit(path):
    gates = []
    wires = []
    overflow = 0

    lines = Path(path).read_text().splitlines()
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        for ch in line:
            if ch == ";":
                if wires:
                    if len(wires) != 3:
                        raise ValueError(f"bad gate ending near gate {len(gates)}")
                    gates.append(tuple(wires))
                    wires = []
                overflow = 0
                continue
            if ch == "~":
                overflow += 1
                continue
            if ch in "\t ":
                continue
            if ch not in CHAR_TO_VAL:
                raise ValueError(f"bad character {ch!r} in {path}")
            wires.append(CHAR_TO_VAL[ch] + RADIX * overflow)
            overflow = 0

    if wires:
        raise ValueError(f"unterminated gate in {path}")
    return gates


def encode_wire(wire):
    if wire < 0:
        raise ValueError("negative wire")
    return "~" * (wire // RADIX) + CHARS[wire % RADIX]


def encode_gates(gates):
    return "".join("".join(encode_wire(w) for w in gate) + ";" for gate in gates)


def parse_header(path):
    header = []
    with Path(path).open() as f:
        for line in f:
            if not line.lstrip().startswith("#"):
                break
            header.append(line.rstrip("\n"))
    joined = "\n".join(header)
    y = re.search(r"\b(?:Y|y_hex)=(0x[0-9a-fA-F]+)", joined)
    z = re.search(r"\b(?:Z|z_hex)=(0x[0-9a-fA-F]+)", joined)
    return {
        "header": header,
        "Y": y.group(1) if y else None,
        "Z": z.group(1) if z else None,
    }


def gate_fanouts(gates, wire_count):
    counts = [0] * wire_count
    out = [0] * len(gates)
    for i in range(len(gates) - 1, -1, -1):
        a, b, c = gates[i]
        out[i] = counts[a]
        counts[a] = 0
        if b != a:
            counts[b] += 1
        if c != a and c != b:
            counts[c] += 1
    return out


def gate_leeways(gates, wire_count):
    n = len(gates)
    prev_target = [-1] * wire_count
    prev_control = [-1] * wire_count
    left = [0] * n
    for i, (a, b, c) in enumerate(gates):
        nearest = max(prev_target[b], prev_target[c], prev_control[a])
        left[i] = i if nearest < 0 else i - nearest - 1
        prev_target[a] = i
        prev_control[b] = i
        prev_control[c] = i

    next_target = [n] * wire_count
    next_control = [n] * wire_count
    out = [0] * n
    for i in range(n - 1, -1, -1):
        a, b, c = gates[i]
        nearest = min(next_target[b], next_target[c], next_control[a])
        right = (n - i - 1) if nearest == n else nearest - i - 1
        out[i] = left[i] + right
        next_target[a] = i
        next_control[b] = i
        next_control[c] = i
    return out


def exact_hist(values):
    return dict(sorted(Counter(values).items()))


def binned_hist(values, bins):
    counts = Counter()
    for value in values:
        for label, lo, hi in bins:
            if lo <= value <= hi:
                counts[label] += 1
                break
    return {label: counts[label] for label, _, _ in bins if counts[label]}


def stats(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return {}

    def q(percent):
        return values[min(n - 1, int((n - 1) * percent))]

    return {
        "count": n,
        "min": values[0],
        "median": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": values[-1],
        "mean": sum(values) / n,
    }


def eval_state(gates, x, y, z):
    words = [x & ((1 << 64) - 1), x >> 64, y & ((1 << 64) - 1), y >> 64, z & ((1 << 64) - 1), z >> 64]
    for a, b, c in gates:
        bbit = (words[b // 64] >> (b % 64)) & 1
        cbit = (words[c // 64] >> (c % 64)) & 1
        if bbit or not cbit:
            words[a // 64] ^= 1 << (a % 64)
    return words


def verify_inverse(gates, inverse, trials, seed):
    rng = random.Random(seed)
    for _ in range(trials):
        x = rng.getrandbits(128)
        y = rng.getrandbits(128)
        z = rng.getrandbits(128)
        forward = eval_state(gates, x, y, z)
        fx = forward[0] | (forward[1] << 64)
        fy = forward[2] | (forward[3] << 64)
        fz = forward[4] | (forward[5] << 64)
        back = eval_state(inverse, fx, fy, fz)
        bx = back[0] | (back[1] << 64)
        by = back[2] | (back[3] << 64)
        bz = back[4] | (back[5] << 64)
        if (bx, by, bz) != (x, y, z):
            return False
    return True


def parse_hex128(value):
    if not value:
        return None
    value = value[2:] if value.lower().startswith("0x") else value
    return int(value, 16)


def y_xor_prefix(y_value):
    gates = []
    if y_value is None:
        return gates
    for bit in range(128):
        if (y_value >> bit) & 1:
            target = 128 + bit
            gates.extend(
                [
                    (0, 1, 2),
                    (target, 0, 1),
                    (target, 0, 2),
                    (0, 1, 2),
                    (target, 1, 0),
                    (target, 1, 2),
                    (target, 2, 0),
                ]
            )
    return gates


def verify_sideblock_wrapper(gates, wrapper, y_value, z_value, trials, seed):
    if y_value is None or z_value is None:
        return None
    rng = random.Random(seed)
    mask = (1 << 128) - 1
    for _ in range(trials):
        x = rng.getrandbits(128)
        forward = eval_state(gates, x, y_value, z_value)
        left = forward[0] | (forward[1] << 64)
        middle = forward[2] | (forward[3] << 64)
        right = forward[4] | (forward[5] << 64)
        u = middle ^ y_value
        back = eval_state(wrapper, left, u, right)
        bx = (back[0] | (back[1] << 64)) & mask
        by = (back[2] | (back[3] << 64)) & mask
        bz = (back[4] | (back[5] << 64)) & mask
        if (bx, by, bz) != (x, y_value, z_value):
            return False
    return True


def write_hist_csv(path, hist):
    with Path(path).open("w") as f:
        f.write("value,count\n")
        for value, count in hist.items():
            f.write(f"{value},{count}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--wire-count", type=int, default=384)
    parser.add_argument("--verify-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    gates = decode_circuit(args.circuit)
    max_wire = max(max(g) for g in gates) if gates else -1
    wire_count = max(args.wire_count, max_wire + 1)
    fanouts = gate_fanouts(gates, wire_count)
    leeways = gate_leeways(gates, wire_count)
    inverse = list(reversed(gates))
    header = parse_header(args.circuit)
    y_value = parse_hex128(header.get("Y"))
    z_value = parse_hex128(header.get("Z"))
    sideblock_wrapper = y_xor_prefix(y_value) + inverse

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    inverse_path = prefix.with_name(prefix.name + "_full_inverse.txt")
    inverse_path.write_text(encode_gates(inverse))
    wrapper_path = prefix.with_name(prefix.name + "_Cinv_with_side_blocks.txt")
    wrapper_path.write_text(encode_gates(sideblock_wrapper))

    fanout_hist = exact_hist(fanouts)
    leeway_hist = exact_hist(leeways)
    write_hist_csv(prefix.with_name(prefix.name + "_fanout_hist.csv"), fanout_hist)
    write_hist_csv(prefix.with_name(prefix.name + "_leeway_hist.csv"), leeway_hist)

    leeway_bins = [
        ("0", 0, 0),
        ("1", 1, 1),
        ("2", 2, 2),
        ("3-4", 3, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33-64", 33, 64),
        ("65-128", 65, 128),
        ("129-256", 129, 256),
        ("257-512", 257, 512),
        ("513-1024", 513, 1024),
        ("1025+", 1025, 10**18),
    ]

    report = {
        "source": str(Path(args.circuit)),
        "header": header,
        "gates": len(gates),
        "max_wire": max_wire,
        "fanout": {
            "stats": stats(fanouts),
            "hist": fanout_hist,
        },
        "leeway": {
            "stats": stats(leeways),
            "binned_hist": binned_hist(leeways, leeway_bins),
            "exact_hist_csv": str(prefix.with_name(prefix.name + "_leeway_hist.csv")),
        },
        "inverse": {
            "path": str(inverse_path),
            "verified_random_trials": args.verify_trials,
            "verified": verify_inverse(gates, inverse, args.verify_trials, args.seed),
            "sideblock_wrapper_path": str(wrapper_path),
            "sideblock_wrapper_gates": len(sideblock_wrapper),
            "sideblock_wrapper_verified": verify_sideblock_wrapper(
                gates,
                sideblock_wrapper,
                y_value,
                z_value,
                args.verify_trials,
                args.seed,
            ),
        },
    }
    report_path = prefix.with_name(prefix.name + "_metrics.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
