#!/usr/bin/env python3
"""Build nonlinear193/nonlinear291 circuits for the gauntlet tracer.

The builder exports a pre-gadgetized ``mpmct1`` circuit, bit-packed initial
columns, and a small metadata file.  The Rust ``gauntlet_gen`` binary consumes
those three files through its ``--gadget file`` mode.

This file intentionally supports only the two gadget constructions installed
in :mod:`gadgetization`.  Native control arms (none, secret sharing, and band
product sharing) are constructed directly by ``gauntlet_gen``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gadgetization import nonlinear193 as G  # noqa: E402
from gadgetization import nonlinear291 as W2  # noqa: E402


W2_EXTRA = 27  # 2 fallback + 24 persistent-prefix + 1 temporary ancilla
EXPECTED_GATES = {"nonlinear193": 193, "nonlinear291": 291}


def parse_chain(path: Path) -> tuple[int, list[tuple[int, int, int]]]:
    """Read the r57 source-chain subset used by the gauntlet.

    Each accepted gate has exactly one negative and one positive control.  The
    returned tuple is ``(target, negative_control, positive_control)``.
    """

    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 3 or tokens[0] != "mpmct1":
        raise ValueError(f"{path}: expected an mpmct1 header")
    n_wires, n_gates = int(tokens[1]), int(tokens[2])
    offset = 3
    gates: list[tuple[int, int, int]] = []
    for gate_index in range(n_gates):
        if offset + 3 > len(tokens):
            raise ValueError(f"{path}: gate {gate_index} is truncated")
        target, complement, fanin = map(int, tokens[offset : offset + 3])
        offset += 3
        if complement != 1 or fanin != 2:
            raise ValueError(
                f"{path}: gate {gate_index} is not r57 (comp=1, fanin=2)"
            )
        if offset + 2 * fanin > len(tokens):
            raise ValueError(f"{path}: gate {gate_index} controls are truncated")
        controls = [
            (int(tokens[offset + 2 * i]), int(tokens[offset + 2 * i + 1]))
            for i in range(fanin)
        ]
        offset += 2 * fanin
        by_polarity = {polarity: wire for wire, polarity in controls}
        if set(by_polarity) != {0, 1}:
            raise ValueError(
                f"{path}: gate {gate_index} needs one negative and one positive pin"
            )
        control_wires = {wire for wire, _polarity in controls}
        if len(control_wires) != 2 or target in control_wires:
            raise ValueError(
                f"{path}: gate {gate_index} needs three distinct target/control wires"
            )
        if not (0 <= target < n_wires) or any(
            not 0 <= wire < n_wires for wire, _ in controls
        ):
            raise ValueError(f"{path}: gate {gate_index} references an invalid wire")
        gates.append((target, by_polarity[0], by_polarity[1]))
    if offset != len(tokens):
        raise ValueError(f"{path}: trailing tokens after {n_gates} gates")
    return n_wires, gates


def majority(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a & b) ^ (b & c) ^ (a & c)


def decode_block(columns: Sequence[np.ndarray], block: Sequence[int]) -> np.ndarray:
    return G.E([columns[wire] for wire in block])


def solve_blocks(
    columns: list[np.ndarray], blocks: Sequence[Sequence[int]], target: np.ndarray
) -> None:
    """Adjust one pad so the XOR of encoded blocks equals ``target``."""

    current = np.zeros_like(target)
    for block in blocks:
        current ^= decode_block(columns, block)
    columns[blocks[0][0]] = (columns[blocks[0][0]] ^ current ^ target).astype(
        np.uint8
    )


def dump_columns(path: Path, columns: Sequence[np.ndarray]) -> None:
    matrix = np.asarray(columns, dtype=np.uint8)
    np.packbits(matrix, axis=1, bitorder="little").tofile(path)


# The verified 35-gate U0 used by the production nonlinear-carrier blind pool.
# Entries are (target, [(wire, polarity), ...], number_of_live_literals).
NC_U0 = [
    (2, [(3, 1), (0, 0)], 1), (4, [(2, 1), (3, 0)], 2),
    (0, [(2, 1), (3, 0)], 2), (0, [(2, 0), (3, 0)], 2),
    (2, [(3, 1), (0, 0)], 1), (1, [(0, 0), (2, 1)], 2),
    (3, [(0, 0), (4, 0)], 2), (3, [(1, 1), (2, 0)], 2),
    (0, [(2, 1), (3, 1)], 2), (3, [(1, 1), (2, 1)], 2),
    (2, [(3, 0), (4, 1)], 2), (2, [(1, 0), (3, 1)], 2),
    (1, [(2, 0), (0, 0)], 1), (4, [(2, 1), (0, 0)], 1),
    (1, [(4, 0), (0, 0)], 1), (2, [(3, 0), (4, 0)], 2),
    (2, [(0, 0), (4, 0)], 2), (2, [(3, 0), (4, 0)], 2),
    (3, [(2, 0), (0, 0)], 1), (4, [(3, 0), (0, 0)], 1),
    (3, [(2, 1), (0, 0)], 1), (2, [(1, 1), (4, 0)], 2),
    (2, [(0, 1), (3, 1)], 2), (2, [(0, 0), (1, 0)], 2),
    (0, [(1, 1), (0, 0)], 1), (4, [(0, 0), (0, 0)], 1),
    (4, [(1, 1), (3, 0)], 2), (2, [(0, 1), (0, 0)], 1),
    (3, [(0, 0), (2, 0)], 2), (2, [(1, 1), (4, 0)], 2),
    (1, [(0, 0), (2, 0)], 2), (4, [(0, 1), (2, 1)], 2),
    (2, [(0, 1), (4, 0)], 2), (1, [(0, 0), (2, 1)], 2),
    (3, [(1, 1), (2, 1)], 2),
]


def apply_u0(columns: list[np.ndarray], indices: Sequence[int]) -> None:
    state = [columns[index] for index in indices]
    for target, controls, live in NC_U0:
        product: np.ndarray | None = None
        for wire, polarity in controls[:live]:
            literal = state[wire] if polarity else (1 - state[wire])
            product = literal if product is None else product & literal
        if product is None:
            raise AssertionError("U0 entry has no controls")
        state[target] = (state[target] ^ product).astype(np.uint8)
    for local, index in enumerate(indices):
        columns[index] = state[local]


def band_pool(
    inputs: Sequence[np.ndarray],
    count: int,
    blind_layers: int,
    rng: np.random.Generator,
    *,
    extra_keys: int,
) -> list[np.ndarray]:
    """Build pool columns with the production-style nonlinear band fill.

    Extra neighbor input columns prevent the tiny 2**n source-chain domain from
    collapsing the pool into a table over only the eight visible inputs.
    """

    samples = len(inputs[0])
    source = [column.copy() for column in inputs]
    source.extend(
        rng.integers(0, 2, samples, dtype=np.uint8) for _ in range(extra_keys)
    )
    n_source = len(source)
    if n_source < 5 and blind_layers:
        raise ValueError("blind pool needs at least five source columns")

    for layer in range(blind_layers):
        offset = (2 * layer) % n_source
        for start in range(0, n_source - 4, 5):
            apply_u0(source, [(offset + start + i) % n_source for i in range(5)])

    band: list[np.ndarray] = []
    supports: list[set[int]] = []
    for _ in range(count):
        pivot = int(rng.integers(0, n_source))
        support = {pivot}
        column = source[pivot].copy()

        linear_max = min(7, n_source - 1)
        linear_width = min(1 + int(rng.integers(0, linear_max)), n_source - 1)
        choices = [wire for wire in range(n_source) if wire != pivot]
        for _ in range(linear_width):
            choice_index = int(rng.integers(0, len(choices)))
            wire = choices.pop(choice_index)
            support.add(wire)
            column ^= source[wire]

        eligible_band = [i for i, deps in enumerate(supports) if pivot not in deps]
        drawn: list[tuple[str, int]] = []
        for _ in range(2):
            factors: list[tuple[str, int]] = []
            for _factor in range(2):
                selected: tuple[str, int] | None = None
                for _attempt in range(64):
                    if eligible_band and int(rng.integers(0, 2)):
                        candidate = (
                            "band",
                            eligible_band[int(rng.integers(0, len(eligible_band)))],
                        )
                    else:
                        wire = int(rng.integers(0, n_source))
                        if wire == pivot:
                            continue
                        candidate = ("source", wire)
                    if candidate not in drawn and candidate not in factors:
                        selected = candidate
                        break
                if selected is None:
                    selected = ("source", (pivot + 1) % n_source)
                factors.append(selected)
            drawn.extend(factors)

            literals: list[np.ndarray] = []
            for kind, index in factors:
                base = band[index] if kind == "band" else source[index]
                polarity = bool(rng.integers(0, 2))
                literals.append(base if polarity else 1 - base)
                if kind == "band":
                    support |= supports[index]
                else:
                    support.add(index)
            column ^= (literals[0] & literals[1]).astype(np.uint8)
        band.append(column.astype(np.uint8))
        supports.append(support)
    return band


def simulate_chain(
    n_wires: int,
    gates: Sequence[tuple[int, int, int]],
    inputs: Sequence[np.ndarray],
) -> list[np.ndarray]:
    state = [column.copy() for column in inputs]
    if len(state) != n_wires:
        raise ValueError("input column count does not match source wire count")
    for target, negative, positive in gates:
        a, b = state[negative], state[positive]
        flip = (1 ^ b ^ (a & b)).astype(np.uint8)
        state[target] = (state[target] ^ flip).astype(np.uint8)
    return state


def check_borrow_isolation(
    circuit: object,
    borrows: dict[int, tuple[list[int], list[int]]],
    period: int,
) -> None:
    """Ensure strict borrow wires are read only in their gadget context."""

    owners = {
        wire: gate_index
        for gate_index, (strict, _transitory) in borrows.items()
        for wire in strict
    }
    for emitted_index, (_target, _complement, controls) in enumerate(circuit.gate_log):
        for wire, _polarity in controls:
            if wire not in owners:
                continue
            owner = owners[wire]
            in_owner = emitted_index // period == owner
            if not in_owner:
                raise AssertionError(
                    f"strict borrow {wire} for source gate {owner} read by emitted "
                    f"gate {emitted_index}"
                )


def build_nonlinear(
    n_wires: int,
    gates: Sequence[tuple[int, int, int]],
    samples: int,
    rng: np.random.Generator,
    *,
    pool: str,
    blind_layers: int,
    pool_keys: int,
    weight2: bool,
    pool_seed: int,
) -> tuple[object, int, dict[int, list[int]], list[int], list[np.ndarray], dict[int, tuple[list[int], list[int]]]]:
    """Build a chain of nonlinear193 or nonlinear291 gadgets."""

    gate_count = len(gates)
    share1 = {wire: list(range(10 * wire, 10 * wire + 5)) for wire in range(n_wires)}
    share2 = {
        wire: list(range(10 * wire + 5, 10 * wire + 10))
        for wire in range(n_wires)
    }
    cursor = 10 * n_wires
    extras: list[tuple[list[int], list[int], list[int]]] = []
    for _ in gates:
        resharing = list(range(cursor, cursor + 5))
        output = list(range(cursor + 5, cursor + 8))
        chaff = list(range(cursor + 8, cursor + 12))
        cursor += 12
        extras.append((resharing, output, chaff))
    scratch, scratch2 = cursor, cursor + 1
    cursor += 2

    decomp: tuple[int, ...] = ()
    persistent: tuple[int, ...] = ()
    temporary: tuple[int, ...] = ()
    if weight2:
        decomp = (cursor, cursor + 1)
        persistent = tuple(range(cursor + 2, cursor + 26))
        temporary = (cursor + 26,)
        cursor += W2_EXTRA
    circuit_wires = cursor

    initial = [
        rng.integers(0, 2, samples, dtype=np.uint8) for _ in range(circuit_wires)
    ]
    for _resharing, output, _chaff in extras:
        for wire in output:
            initial[wire] = np.zeros(samples, np.uint8)
    initial[scratch] = np.zeros(samples, np.uint8)
    initial[scratch2] = np.zeros(samples, np.uint8)

    inputs = [rng.integers(0, 2, samples, dtype=np.uint8) for _ in range(n_wires)]
    borrows = {
        gate_index: (list(chaff), list(resharing))
        for gate_index, (resharing, _output, chaff) in enumerate(extras)
    }

    null_pool_column: np.ndarray | None = None
    if pool == "band":
        needed = 10 * n_wires + sum(
            len(strict) + len(transitory)
            for strict, transitory in borrows.values()
        )
        pool_columns = band_pool(
            inputs,
            needed + 1,
            blind_layers,
            np.random.default_rng(pool_seed),
            extra_keys=pool_keys,
        )
        null_pool_column = pool_columns.pop()
        pool_iter: Iterable[np.ndarray] = iter(pool_columns)
        for wire in range(n_wires):
            for physical in share1[wire] + share2[wire]:
                initial[physical] = next(pool_iter)
        for gate_index in range(gate_count):
            strict, transitory = borrows[gate_index]
            for physical in strict + transitory:
                initial[physical] = next(pool_iter)

    for wire in range(n_wires):
        solve_blocks(initial, [share1[wire], share2[wire]], inputs[wire])

    # Plaintext inputs are appended to init.bin for the tracer but are not
    # circuit wires and therefore never appear in the adversarial trace.
    x_holders = list(range(circuit_wires, circuit_wires + n_wires))
    holder_columns = [column.copy() for column in inputs]

    if weight2:
        for wire in decomp + persistent + temporary:
            initial[wire] = np.zeros(samples, np.uint8)
        core_masks = {scratch, scratch2}
        auxiliary_masks = {
            wire
            for resharing, _output, chaff in extras
            for wire in resharing + chaff
        }
        circuit = W2.Weight2Circuit(
            initial,
            decomp,
            core_masks,
            auxiliary_masks,
            persist=persistent,
            temp=temporary,
        )
    else:
        circuit = G.Circuit(initial)

    def logical_value(wire: int) -> np.ndarray:
        return (
            decode_block(circuit.s, share1[wire])
            ^ decode_block(circuit.s, share2[wire])
        ).astype(np.uint8)

    for gate_index, (target, negative, positive) in enumerate(gates):
        a, b, old_target = (
            logical_value(negative),
            logical_value(positive),
            logical_value(target),
        )
        expected = (old_target ^ (1 ^ b ^ (a & b))).astype(np.uint8)
        resharing, output, chaff = extras[gate_index]
        if weight2:
            # A prior output share has become ordinary operand data, so only
            # this gadget's current borrow set is classified as mask material.
            circuit.set_masks({scratch, scratch2}, set(resharing) | set(chaff))
        G.gadget_gate(
            circuit,
            (tuple(share1[negative]), tuple(share2[negative])),
            (tuple(share1[positive]), tuple(share2[positive])),
            tuple(share1[target]),
            tuple(share2[target]),
            resharing,
            output,
            scratch,
            scratch2,
            chaff,
            vtype="r57",
        )
        if weight2:
            circuit.flush()
        share1[target] = list(resharing)
        share2[target] = [share2[target][0], share2[target][1], *output]
        if not np.array_equal(logical_value(target), expected):
            raise AssertionError(f"gadget {gate_index} failed its local decode check")

    source_final = simulate_chain(n_wires, gates, inputs)
    for wire in range(n_wires):
        if not np.array_equal(logical_value(wire), source_final[wire]):
            raise AssertionError(f"logical wire {wire} failed end-to-end decode")

    gadget_name = "nonlinear291" if weight2 else "nonlinear193"
    if weight2:
        if max(len(controls) for _target, _comp, controls in circuit.gate_log) > 2:
            raise AssertionError("nonlinear291 emitted a gate with fan-in above two")
        if any(circuit.s[wire].any() for wire in decomp + persistent + temporary):
            raise AssertionError("nonlinear291 did not restore its decomposition ancillas")
    check_borrow_isolation(circuit, borrows, EXPECTED_GATES[gadget_name])

    decode = {
        wire: list(share1[wire]) + list(share2[wire]) for wire in range(n_wires)
    }
    if null_pool_column is not None:
        holder_columns.append(null_pool_column)
    return circuit, circuit_wires, decode, x_holders, holder_columns, borrows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gadget", required=True, choices=sorted(EXPECTED_GATES)
    )
    parser.add_argument("--c-in", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="total sample count; rounded up to a multiple of 64",
    )
    parser.add_argument(
        "--pool",
        default="ideal",
        choices=("ideal", "band"),
        help="fresh independent borrows or a production-style band feed",
    )
    parser.add_argument(
        "--blind-layers",
        type=int,
        default=0,
        help="U0 butterfly layers applied before the band fill",
    )
    parser.add_argument(
        "--pool-keys",
        type=int,
        default=120,
        help="extra input columns keying the band pool",
    )
    args = parser.parse_args(argv)

    if args.n < 3 or args.samples <= 0 or args.pool_keys < 0:
        parser.error("--n must be at least 3, --samples positive, and --pool-keys nonnegative")
    if args.blind_layers < 0:
        parser.error("--blind-layers cannot be negative")

    chain_path = args.c_in if args.c_in.is_absolute() else REPO_ROOT / args.c_in
    output_prefix = (
        args.out_prefix
        if args.out_prefix.is_absolute()
        else REPO_ROOT / args.out_prefix
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    source_wires, gates = parse_chain(chain_path.resolve())
    if source_wires != args.n:
        parser.error(f"--n={args.n} but {chain_path} declares {source_wires} wires")
    samples = ((args.samples + 63) // 64) * 64
    rng = np.random.default_rng(args.seed)
    weight2 = args.gadget == "nonlinear291"
    circuit, n_wires, decode, x_holders, holders, borrows = build_nonlinear(
        args.n,
        gates,
        samples,
        rng,
        pool=args.pool,
        blind_layers=args.blind_layers,
        pool_keys=args.pool_keys,
        weight2=weight2,
        pool_seed=args.seed + 999_999,
    )

    n_gates = len(circuit.flips)
    expected = EXPECTED_GATES[args.gadget] * len(gates)
    if n_gates != expected:
        raise AssertionError(f"emitted {n_gates} gates; expected {expected}")

    circuit_path = Path(f"{output_prefix}.mpmct1")
    with circuit_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mpmct1 {n_wires} {n_gates}\n")
        for target, complement, controls in circuit.gate_log:
            encoded = " ".join(f"{wire} {polarity}" for wire, polarity in controls)
            suffix = f" {encoded}" if encoded else ""
            handle.write(f"{target} {complement} {len(controls)}{suffix}\n")

    dump_columns(Path(f"{output_prefix}.init.bin"), circuit.init + holders)

    metadata_path = Path(f"{output_prefix}.buildmeta")
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write("builder_schema\t2\n")
        handle.write(f"gadget\t{args.gadget}\n")
        handle.write(f"k\t{len(gates)}\n")
        handle.write(f"n\t{args.n}\n")
        handle.write(f"n_wires\t{n_wires}\n")
        handle.write(f"n_gates\t{n_gates}\n")
        handle.write(f"samples\t{samples}\n")
        handle.write(f"seed\t{args.seed}\n")
        handle.write("builder_checked\ttrue\n")
        handle.write(f"pool\t{args.pool}\n")
        handle.write(f"blind_layers\t{args.blind_layers}\n")
        handle.write(f"pool_keys\t{args.pool_keys}\n")
        handle.write(f"null_holder\t{str(args.pool == 'band').lower()}\n")
        handle.write("x_holders\t" + ",".join(map(str, x_holders)) + "\n")
        handle.write(f"init_cols\t{n_wires + len(holders)}\n")
        for wire in range(args.n):
            handle.write(
                f"decode[{wire}]\t" + ",".join(map(str, decode[wire])) + "\n"
            )
        for gate_index, (strict, transitory) in borrows.items():
            handle.write(
                f"borrow_strict[g{gate_index}]\t"
                + ",".join(map(str, strict))
                + "\n"
            )
            handle.write(
                f"borrow_trans[g{gate_index}]\t"
                + ",".join(map(str, transitory))
                + "\n"
            )

    print(
        f"[build:{args.gadget}] k={len(gates)} nw={n_wires} gates={n_gates} "
        f"samples={samples} features={n_wires + 2 * n_gates} pool={args.pool} "
        f"blind={args.blind_layers} builder_checked=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
