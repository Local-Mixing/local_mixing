"""Canonical 193-gate nonlinear gadgetization reference.

The five-bit encoding is ``E(x) = x0 ^ x1 ^ maj(x2, x3, x4)``. A logical target
uses two encoded shares. ``gadget_gate`` updates it with a reversible r57 gate while
re-sharing it onto fresh wires. The default operation is::

    c_out = c_in ^ (a | ~b) = c_in ^ 1 ^ b ^ (a & b)

For two-shared controls, clean scratch, and ``vtype="r57"``, this canonical construction
emits exactly 193 generalized-Toffoli gates with at most four controls. The circuit
simulator records every emitted flip and new target value for trace auditing.
"""

from __future__ import annotations

from collections import Counter
from numbers import Integral
import random as _random
from typing import Iterable, Sequence

import numpy as np


WireBlock = Sequence[int]
Control = tuple[int, int]
CANONICAL_R57_GATE_COUNT = 193
CANONICAL_MAX_PHYSICAL_FANIN = 4


def maj(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the bitwise majority of three uint8 bit-vectors."""

    return (a & b) ^ (b & c) ^ (a & c)


def E(wires: Sequence[np.ndarray]) -> np.ndarray:
    """Decode one five-wire nonlinear encoding block."""

    if len(wires) != 5:
        raise ValueError("E requires exactly five wires")
    return (wires[0] ^ wires[1] ^ maj(wires[2], wires[3], wires[4])).astype(
        np.uint8
    )


def _bit(value: int, index: int) -> int:
    return (value >> index) & 1


def _anf3(truth: Sequence[int]) -> list[int]:
    coefficients = list(truth)
    for bit in range(3):
        for value in range(8):
            if (value >> bit) & 1:
                coefficients[value] ^= coefficients[value ^ (1 << bit)]
    return [value for value in range(8) if coefficients[value]]


def _decompose_perm(
    unflipped: Sequence[int], flipped: Sequence[int]
) -> tuple[list[list[int]], list[list[int]]]:
    a_monomials: list[list[int]] = []
    b_monomials: list[list[int]] = []
    for index in range(3):
        a_monomials.append(
            _anf3([_bit(unflipped[value], index) for value in range(8)])
        )
        b_monomials.append(
            _anf3(
                [
                    _bit(flipped[value], index)
                    ^ _bit(unflipped[value], index)
                    for value in range(8)
                ]
            )
        )
    return a_monomials, b_monomials


# Majority-preserving and majority-flipping permutations selected by the canonical
# constrained search. B0 == B1, every B_i is balanced, and no non-empty XOR subset of
# the distinct B_i functions is constant.
SB_U0 = (0, 1, 2, 3, 4, 5, 6, 7)
SB_U1 = (7, 5, 6, 0, 3, 1, 2, 4)


class Circuit:
    """Reversible circuit simulator with a full emitted-gate trace."""

    def __init__(self, wire_values: Sequence[np.ndarray]):
        if not wire_values:
            raise ValueError("a circuit requires at least one wire")
        self.s = [wire.copy() for wire in wire_values]
        self.init = [wire.copy() for wire in wire_values]
        self.n = len(wire_values[0])
        if any(len(wire) != self.n for wire in wire_values):
            raise ValueError("all circuit wires must have equal length")
        self.flips: list[np.ndarray] = []
        self.newvals: list[np.ndarray] = []
        self.gate_log: list[tuple[int, int, list[Control]]] = []
        self.marks: list[tuple[int, str]] = []

    def gate(
        self, target: int, controls: Iterable[Control], comp: int = 0
    ) -> None:
        """Apply ``target ^= comp ^ AND(controls)`` and record the trace."""

        control_list = list(controls)
        product = np.ones(self.n, np.uint8)
        for wire, polarity in control_list:
            product &= self.s[wire] if polarity == 1 else (1 ^ self.s[wire])
        flip = (comp ^ product).astype(np.uint8)
        self.s[target] = (self.s[target] ^ flip).astype(np.uint8)
        self.flips.append(flip.copy())
        self.newvals.append(self.s[target].copy())
        self.gate_log.append((target, comp, control_list))

    def mark(self, label: str) -> None:
        self.marks.append((len(self.gate_log), label))


def _E_monomials(offset: int) -> list[int]:
    return [
        1 << offset,
        1 << (offset + 1),
        (1 << (offset + 2)) | (1 << (offset + 3)),
        (1 << (offset + 3)) | (1 << (offset + 4)),
        (1 << (offset + 2)) | (1 << (offset + 4)),
    ]


def _xor_monomials(*groups: Sequence[int]) -> list[int]:
    counts: Counter[int] = Counter()
    for group in groups:
        counts.update(group)
    return sorted(monomial for monomial, count in counts.items() if count % 2)


def _literals(mask: int, wires: Sequence[int]) -> list[Control]:
    return [(wire, 1) for index, wire in enumerate(wires) if (mask >> index) & 1]


def _normalize_scratch2(
    scratch2: int | Sequence[int], group_count: int, *, dirty: bool
) -> list[int]:
    """Normalize and validate secondary-scratch wires for the B groups."""

    if isinstance(scratch2, Integral):
        if dirty:
            raise ValueError(
                f"dirty scratch2 requires {group_count} distinct wires, one per B group"
            )
        return [int(scratch2)] * group_count
    try:
        wires = list(scratch2)
    except TypeError as exc:
        raise TypeError(
            "scratch2 must be an integer wire or a sequence of integer wires"
        ) from exc
    if not all(isinstance(wire, Integral) for wire in wires):
        raise TypeError("scratch2 sequence entries must be integer wire indices")
    normalized = [int(wire) for wire in wires]
    if len(normalized) != group_count:
        raise ValueError(
            f"scratch2 requires {group_count} wires for {group_count} B groups; "
            f"got {len(normalized)}"
        )
    if dirty and len(set(normalized)) != group_count:
        raise ValueError("dirty scratch2 wires must be distinct")
    return normalized


def gadget_gate(
    circ: Circuit,
    a_blocks: Sequence[WireBlock],
    b_blocks: Sequence[WireBlock],
    target_share1: WireBlock,
    target_share2: WireBlock,
    fresh_share: WireBlock,
    output_majority: WireBlock,
    scratch: int,
    scratch2: int | Sequence[int],
    chaff: WireBlock,
    *,
    U0: Sequence[int] = SB_U0,
    U1: Sequence[int] = SB_U1,
    sharedB: bool = True,
    dirty: bool = False,
    vtype: str = "r57",
) -> None:
    """Emit one folded nonlinear gadget onto ``circ``.

    ``a_blocks`` and ``b_blocks`` contain one read-only encoding or two share blocks.
    ``fresh_share`` contains five fresh random borrow wires, ``output_majority`` three
    clean wires, and ``chaff`` four fresh random borrow wires. In clean mode ``scratch``
    and ``scratch2`` start and end at zero; ``scratch2`` may be one shared integer wire
    or a sequence with one wire per B group. In dirty mode, pass a sequence containing
    one distinct ``scratch2`` wire per B group; those wires retain masked garbage by
    design. Any sequence type is accepted, including ``range``.

    Relabel the updated target to ``fresh_share`` and
    ``(target_share2[0], target_share2[1], *output_majority)`` after this call.
    """

    if vtype not in {"r57", "nab", "and", "copy"}:
        raise ValueError(f"unsupported vtype: {vtype}")
    if len(target_share1) != 5 or len(target_share2) != 5:
        raise ValueError("target shares must each contain five wires")
    if len(fresh_share) != 5 or len(output_majority) != 3 or len(chaff) != 4:
        raise ValueError("fresh_share/output_majority/chaff must have lengths 5/3/4")

    a_perm, b_perm = _decompose_perm(U0, U1)
    if sharedB and b_perm[0] != b_perm[1]:
        raise ValueError("sharedB requires B0 == B1 as functions")

    majority_wires = (target_share2[2], target_share2[3], target_share2[4])
    a_wires = [wire for block in a_blocks for wire in block]
    b_wires = [wire for block in b_blocks for wire in block]
    operand_wires = list(target_share1) + list(fresh_share) + a_wires + b_wires
    a_offset = 10
    b_offset = 10 + len(a_wires)
    a_monomials = _xor_monomials(
        *[_E_monomials(a_offset + 5 * index) for index in range(len(a_blocks))]
    )
    b_monomials = _xor_monomials(
        *[_E_monomials(b_offset + 5 * index) for index in range(len(b_blocks))]
    )
    u_all = _xor_monomials(_E_monomials(0), _E_monomials(5))
    u_run = [monomial for monomial in u_all if monomial not in (1 << 5, 1 << 6)]

    def pads_and_rest(
        monomials: Sequence[int], rng: _random.Random
    ) -> tuple[list[int], list[int]]:
        pads = [monomial for monomial in monomials if monomial.bit_count() == 1]
        rest = [monomial for monomial in monomials if monomial.bit_count() > 1]
        rng.shuffle(pads)
        rng.shuffle(rest)
        return pads, rest

    def spread(monomials: Sequence[int], rng: _random.Random) -> list[int]:
        pads, rest = pads_and_rest(monomials, rng)
        positions = (0, 3, 6, 9) if len(monomials) == 10 else (0, 4)
        order: list[int | None] = [None] * len(monomials)
        for index, position in enumerate(positions):
            order[position] = pads[index]
        remaining = iter(rest)
        for index, monomial in enumerate(order):
            if monomial is None:
                order[index] = next(remaining)
        return [monomial for monomial in order if monomial is not None]

    groups = (
        [((0, 1), b_perm[0]), ((2,), b_perm[2])]
        if sharedB
        else [((index,), b_perm[index]) for index in range(3)]
    )
    scratch2_wires = _normalize_scratch2(
        scratch2, len(groups), dirty=dirty
    )

    for group_index, (output_indices, b_function) in enumerate(groups):
        rng = _random.Random(4000 + group_index)
        secondary_scratch = scratch2_wires[group_index]

        circ.mark(f"build B (group {group_index})")
        for monomial in b_function:
            circ.gate(
                scratch,
                [(majority_wires[i], 1) for i in range(3) if (monomial >> i) & 1],
            )

        if vtype == "copy":
            operand_gates = [
                _literals(monomial, operand_wires) + [(scratch, 1)]
                for monomial in spread(b_monomials, rng)
            ]
            split = 3 if len(b_monomials) == 5 else 5
            u_gates = [
                _literals(monomial, operand_wires) + [(scratch, 1)]
                for monomial in spread(u_all, rng)
            ]
            chunks = [operand_gates[:split], operand_gates[split:], u_gates[:5], u_gates[5:]]
            rng.shuffle(chunks)
            chaff_a = [(chaff[0], 1), (scratch, 1)]
            chaff_b = [(chaff[1], 1), (scratch, 1)]
            run: list[list[Control]] = []
            for chunk_index, chunk in enumerate(chunks):
                run.extend(chunk)
                if chunk_index < len(chunks) - 1:
                    run.append([chaff_a, chaff_b, chaff_a][chunk_index])
            run.append(chaff_b)
            for output_index in output_indices:
                circ.mark(f"A{output_index} + run -> o{output_index} (copy)")
                for monomial in a_perm[output_index]:
                    circ.gate(
                        output_majority[output_index],
                        [(majority_wires[i], 1) for i in range(3) if (monomial >> i) & 1],
                    )
                for controls in run:
                    circ.gate(output_majority[output_index], controls)
            circ.mark(f"unbuild B (group {group_index})")
            for monomial in b_function:
                circ.gate(
                    scratch,
                    [(majority_wires[i], 1) for i in range(3) if (monomial >> i) & 1],
                )
            continue

        circ.mark(f"build scr2 = (b^m0^m1)*B (group {group_index})")
        masks = (
            (fresh_share[0], fresh_share[1])
            if vtype in ("r57", "nab")
            else (chaff[2], chaff[3])
        )
        b_pads, b_nonpads = pads_and_rest(b_monomials, rng)
        if len(b_monomials) == 10:
            sequence: list[int | str] = (
                [b_pads[0]] + b_nonpads[:2] + ["M0"]
                + [b_nonpads[2], b_pads[1], b_nonpads[3]] + ["M1"]
                + b_nonpads[4:6] + [b_pads[2], b_pads[3]]
            )
        else:
            sequence = [b_pads[0], "M0", *b_nonpads, "M1", b_pads[1]]
        build: list[list[Control]] = []
        for monomial in sequence:
            if monomial == "M0":
                build.append([(masks[0], 1), (scratch, 1)])
            elif monomial == "M1":
                build.append([(masks[1], 1), (scratch, 1)])
            else:
                build.append(_literals(int(monomial), operand_wires) + [(scratch, 1)])
        for controls in build:
            circ.gate(secondary_scratch, controls)

        cascade = [
            _literals(monomial, operand_wires) + [(secondary_scratch, 1)]
            for monomial in spread(a_monomials, rng)
        ]
        corrections = [
            [
                _literals(monomial, operand_wires) + [(mask, 1), (scratch, 1)]
                for monomial in spread(a_monomials, rng)
            ]
            for mask in masks
        ]
        split = (len(a_monomials) + 1) // 2 if len(a_monomials) == 5 else 5
        chunks = [cascade[:split], cascade[split:]]
        for correction in corrections:
            chunks.extend([correction[:split], correction[split:]])
        rng.shuffle(chunks)
        if vtype == "r57":
            chunks[0] = [[(scratch, 1)]] + chunks[0]
        if vtype in ("r57", "nab"):
            chunks[1] = [[(secondary_scratch, 1)]] + chunks[1]
            u_pool = u_run
        else:
            u_pool = u_all

        linear_masks = (1, 2) if vtype in ("r57", "nab") else (1, 2, 1 << 5, 1 << 6)
        u_linear = [
            _literals(monomial, operand_wires) + [(scratch, 1)]
            for monomial in u_pool if monomial in linear_masks
        ]
        u_payload = [
            _literals(monomial, operand_wires) + [(scratch, 1)]
            for monomial in u_pool if monomial not in linear_masks
        ]
        chaff_a = [(chaff[0], 1), (scratch, 1)]
        chaff_b = [(chaff[1], 1), (scratch, 1)]
        boundary_count = len(chunks) - 1
        if boundary_count == 5:
            boundaries = [chaff_a, chaff_b, u_linear[0], chaff_a, chaff_b]
        elif boundary_count == 3:
            boundaries = [chaff_a, u_linear[0], chaff_a]
        else:
            boundaries = (
                [chaff_a, u_linear[0], chaff_a]
                + [chaff_b, chaff_b] * ((boundary_count - 3) // 2)
            )[:boundary_count]
        payload = u_payload + u_linear[1:]
        rng.shuffle(payload)
        for index, controls in enumerate(payload):
            chunk = chunks[index % len(chunks)]
            chunk.insert(1 + rng.randrange(max(1, len(chunk) - 1)), controls)
        run = []
        for chunk_index, chunk in enumerate(chunks):
            run.extend(chunk)
            if chunk_index < len(chunks) - 1:
                run.append(boundaries[chunk_index])

        for output_index in output_indices:
            circ.mark(f"A{output_index} + run -> o{output_index} (group {group_index})")
            for monomial in a_perm[output_index]:
                circ.gate(
                    output_majority[output_index],
                    [(majority_wires[i], 1) for i in range(3) if (monomial >> i) & 1],
                )
            for controls in run:
                circ.gate(output_majority[output_index], controls)

        if not dirty:
            circ.mark(f"unbuild scr2, reverse (group {group_index})")
            for controls in reversed(build):
                circ.gate(secondary_scratch, controls)
        circ.mark(f"unbuild B (group {group_index})")
        for monomial in b_function:
            circ.gate(
                scratch,
                [(majority_wires[i], 1) for i in range(3) if (monomial >> i) & 1],
            )


def _decode_blocks(circ: Circuit, blocks: Sequence[WireBlock]) -> np.ndarray:
    value = np.zeros(circ.n, np.uint8)
    for block in blocks:
        value ^= E([circ.init[wire] for wire in block])
    return value.astype(np.uint8)


def run_gate(
    samples: int = 20_000,
    seed: int = 0,
    vtype: str = "r57",
    a_single: bool = False,
    b_single: bool = False,
    dirty: bool = False,
    U0: Sequence[int] = SB_U0,
    U1: Sequence[int] = SB_U1,
    sharedB: bool = True,
) -> tuple[Circuit, dict[str, object]]:
    """Build and simulate one randomly encoded gadget."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if vtype not in {"r57", "nab", "and", "copy"}:
        raise ValueError(f"unsupported vtype: {vtype}")
    a_count = 0 if vtype == "copy" else (1 if a_single else 2)
    b_count = 1 if b_single or vtype == "copy" else 2
    group_count = 2 if sharedB else 3

    target_share1 = tuple(range(0, 5))
    target_share2 = tuple(range(5, 10))
    fresh_share = tuple(range(10, 15))
    next_wire = 15
    a_blocks = tuple(
        tuple(range(next_wire + 5 * i, next_wire + 5 * i + 5))
        for i in range(a_count)
    )
    next_wire += 5 * a_count
    b_blocks = tuple(
        tuple(range(next_wire + 5 * i, next_wire + 5 * i + 5))
        for i in range(b_count)
    )
    next_wire += 5 * b_count
    output_majority = tuple(range(next_wire, next_wire + 3))
    scratch = next_wire + 3
    next_wire += 4
    if dirty:
        scratch2: int | tuple[int, ...] = tuple(range(next_wire, next_wire + group_count))
        next_wire += group_count
    else:
        scratch2 = next_wire
        next_wire += 1
    chaff = tuple(range(next_wire, next_wire + 4))
    next_wire += 4

    rng = np.random.default_rng(seed)
    wires = [
        rng.integers(0, 2, samples).astype(np.uint8)
        for _ in range(next_wire)
    ]
    scratch2_tuple = tuple(scratch2) if dirty else (scratch2,)
    for wire in output_majority + (scratch,) + scratch2_tuple:
        wires[wire] = np.zeros(samples, np.uint8)
    circ = Circuit(wires)

    a = _decode_blocks(circ, a_blocks) if a_blocks else np.zeros(samples, np.uint8)
    b = _decode_blocks(circ, b_blocks)
    c_in = (
        E([circ.init[w] for w in target_share1])
        ^ E([circ.init[w] for w in target_share2])
    ).astype(np.uint8)
    rho = E([circ.init[w] for w in fresh_share])
    u = (E([circ.init[w] for w in target_share1]) ^ rho).astype(np.uint8)
    gate_value = {
        "r57": 1 ^ b ^ (a & b),
        "nab": (1 ^ a) & b,
        "and": a & b,
        "copy": b,
    }[vtype].astype(np.uint8)
    c_out = (c_in ^ gate_value).astype(np.uint8)

    gadget_gate(
        circ, a_blocks, b_blocks, target_share1, target_share2, fresh_share,
        output_majority, scratch, scratch2, chaff, U0=U0, U1=U1,
        sharedB=sharedB, dirty=dirty, vtype=vtype,
    )

    output_share2 = (
        target_share2[0], target_share2[1], output_majority[0],
        output_majority[1], output_majority[2],
    )
    actual = (
        E([circ.s[w] for w in fresh_share])
        ^ E([circ.s[w] for w in output_share2])
    ).astype(np.uint8)
    zero = np.zeros(samples, np.uint8)
    primary_scratch_restored = np.array_equal(circ.s[scratch], zero)
    scratch2_restored = all(
        np.array_equal(circ.s[wire], zero) for wire in scratch2_tuple
    )
    scratch_restored = primary_scratch_restored and scratch2_restored
    required_ancillas_restored = primary_scratch_restored and (
        dirty or scratch2_restored
    )
    written_expected = set(output_majority) | {scratch}
    if vtype != "copy":
        written_expected |= set(scratch2_tuple)
    max_fanin = max((len(controls) for _, _, controls in circ.gate_log), default=0)
    layout = {
        "a_blocks": a_blocks,
        "b_blocks": b_blocks,
        "target_share1": target_share1,
        "target_share2": target_share2,
        "fresh_share": fresh_share,
        "output_share2": output_share2,
        "output_majority": output_majority,
        "scratch": (scratch,) + scratch2_tuple,
        "chaff": chaff,
    }
    info: dict[str, object] = {
        "a": a, "b": b, "c_in": c_in, "gate_ab": gate_value,
        "c_out": c_out, "c_out_actual": actual, "rho": rho, "u": u,
        "correct": bool(np.array_equal(actual, c_out)),
        "scratch_restored": bool(scratch_restored),
        "required_ancillas_restored": bool(required_ancillas_restored),
        "n_gates": len(circ.gate_log), "max_fanin": max_fanin,
        "NW": next_wire, "written_expected": written_expected, "layout": layout,
    }
    return circ, info


def build_chain(samples: int = 20_000, seed: int = 0) -> dict[str, object]:
    """Build two r57 gadgets with the first output feeding the second input."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    next_wire = 0
    share1: dict[int, list[int]] = {}
    share2: dict[int, list[int]] = {}
    for logical_wire in range(4):
        share1[logical_wire] = list(range(next_wire, next_wire + 5))
        share2[logical_wire] = list(range(next_wire + 5, next_wire + 10))
        next_wire += 10

    extras = []
    for _ in range(2):
        fresh = tuple(range(next_wire, next_wire + 5))
        output = tuple(range(next_wire + 5, next_wire + 8))
        scratch = next_wire + 8
        scratch2 = next_wire + 9
        next_wire += 10
        chaff = tuple(range(next_wire, next_wire + 4))
        next_wire += 4
        extras.append((fresh, output, scratch, scratch2, chaff))

    wires = [
        rng.integers(0, 2, samples).astype(np.uint8)
        for _ in range(next_wire)
    ]
    for _, output, scratch, scratch2, _ in extras:
        for wire in output + (scratch, scratch2):
            wires[wire] = np.zeros(samples, np.uint8)
    circ = Circuit(wires)

    def value(logical_wire: int) -> np.ndarray:
        return (
            E([circ.s[w] for w in share1[logical_wire]])
            ^ E([circ.s[w] for w in share2[logical_wire]])
        ).astype(np.uint8)

    targets: dict[str, np.ndarray] = {}
    expected_writes: set[int] = set()
    correct = True
    scratch_restored = True
    for gadget_index, (target, a_wire, b_wire) in enumerate(((2, 0, 1), (3, 2, 0))):
        a, b, c_in = value(a_wire), value(b_wire), value(target)
        gate_value = (1 ^ b ^ (a & b)).astype(np.uint8)
        c_out = (c_in ^ gate_value).astype(np.uint8)
        label = gadget_index + 1
        targets.update({
            f"g{label}:a": a, f"g{label}:b": b, f"g{label}:gate": gate_value,
            f"g{label}:c_in": c_in, f"g{label}:c_out": c_out,
        })
        fresh, output, scratch, scratch2, chaff = extras[gadget_index]
        expected_writes |= set(output) | {scratch, scratch2}
        gadget_gate(
            circ,
            (tuple(share1[a_wire]), tuple(share2[a_wire])),
            (tuple(share1[b_wire]), tuple(share2[b_wire])),
            tuple(share1[target]), tuple(share2[target]), fresh, output,
            scratch, scratch2, chaff, vtype="r57",
        )
        share1[target] = list(fresh)
        share2[target] = [share2[target][0], share2[target][1], *output]
        correct &= np.array_equal(value(target), c_out)
        scratch_restored &= (
            np.array_equal(circ.s[scratch], np.zeros(samples, np.uint8))
            and np.array_equal(circ.s[scratch2], np.zeros(samples, np.uint8))
        )

    return {
        "name": "nonlinear193 — two r57 gadgets chained", "circ": circ,
        "targets": targets, "written": expected_writes, "correct": bool(correct),
        "scratch_restored": bool(scratch_restored),
        "required_ancillas_restored": bool(scratch_restored),
        "max_fanin": max((len(c) for _, _, c in circ.gate_log), default=0),
    }


__all__ = [
    "CANONICAL_MAX_PHYSICAL_FANIN", "CANONICAL_R57_GATE_COUNT", "Circuit", "E",
    "SB_U0", "SB_U1", "build_chain", "gadget_gate", "maj", "run_gate",
]
