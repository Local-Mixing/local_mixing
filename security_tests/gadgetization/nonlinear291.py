"""Fan-in-two decomposition of :mod:`gadgetization.nonlinear193`.

The canonical nonlinear193 gadget contains gates with up to four controls. This module
decomposes each gate with three or more controls into fan-in-two Toffolis using a
mask-first accumulation order. A persistent common-subexpression cache reuses safe mask
prefixes and reduces the clean, two-shared r57 construction from 505 naive decomposition
gates to exactly 291 emitted gates.

Mask controls are folded into each accumulator before operand-share controls. This avoids
placing a bare operand-share product in the emitted trace. All fallback, cached, and
temporary ancillas are restored after ``gadget_gate`` returns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from . import nonlinear193 as base


Control = tuple[int, int]
WireBlock = Sequence[int]
CANONICAL_R57_GATE_COUNT = 291
CANONICAL_MAX_PHYSICAL_FANIN = 2
CANONICAL_MAX_REQUESTED_FANIN = 4


class Weight2Circuit(base.Circuit):
    """Circuit that emits only gates with at most two controls.

    ``core_mask`` identifies the primary B/scratch masks, while ``aux_mask`` identifies
    fresh random/chaff masks. ``persist`` supplies clean wires for cached mask-prefix
    products and ``temp`` supplies a transient operand accumulator. ``decomp_ancillas``
    supplies the fallback decomposition chain.

    ``maxfanin`` is the maximum fan-in actually emitted to ``gate_log``.
    ``max_requested_fanin`` records the maximum fan-in requested by nonlinear193 before
    decomposition.
    """

    def __init__(
        self,
        wire_values: Sequence[np.ndarray],
        decomp_ancillas: Sequence[int],
        core_mask: Iterable[int],
        aux_mask: Iterable[int],
        persist: Sequence[int] = (),
        temp: Sequence[int] = (),
    ):
        super().__init__(wire_values)
        self.decomp = list(decomp_ancillas)
        self.core_mask = set(core_mask)
        self.aux_mask = set(aux_mask)
        self.maxfanin = 0
        self.max_requested_fanin = 0
        self._persist_pool = list(persist)
        self._temp = list(temp)
        self._cache: dict[
            object, tuple[int, list[Control], set[int]]
        ] = {}

    def _is_mask(self, wire: int) -> bool:
        return wire in self.core_mask or wire in self.aux_mask

    @property
    def max_physical_fanin(self) -> int:
        """Maximum number of controls on any gate in the physical gate log."""

        return self.maxfanin

    def _emit(self, target: int, controls: Sequence[Control], comp: int = 0) -> None:
        """Emit a physical gate and update the actual-fan-in metric."""

        self.maxfanin = max(self.maxfanin, len(controls))
        super().gate(target, controls, comp)

    def set_masks(
        self, core_mask: Iterable[int], aux_mask: Iterable[int]
    ) -> None:
        """Reclassify masks between composed gadgets after the cache is flushed."""

        if self._cache:
            raise RuntimeError("set_masks requires an empty cache; call flush first")
        self.core_mask = set(core_mask)
        self.aux_mask = set(aux_mask)

    def _get_product(
        self,
        key: object,
        factors: Sequence[Control],
        maskset: Iterable[int],
    ) -> int | None:
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]
        if not self._persist_pool:
            return None
        wire = self._persist_pool.pop()
        self._emit(wire, factors)
        self._cache[key] = (wire, list(factors), set(maskset))
        return wire

    def _invalidate(self, wire: int) -> None:
        keys = [
            key
            for key, (_, _, masks) in reversed(list(self._cache.items()))
            if wire in masks
        ]
        for key in keys:
            cached_wire, factors, _ = self._cache.pop(key)
            self._emit(cached_wire, factors)
            self._persist_pool.append(cached_wire)

    def gate(
        self, target: int, controls: Iterable[Control], comp: int = 0
    ) -> None:
        control_list = list(controls)
        self.max_requested_fanin = max(
            self.max_requested_fanin, len(control_list)
        )
        if (target in self.core_mask or target in self.aux_mask) and self._cache:
            self._invalidate(target)
        if len(control_list) <= 2:
            self._emit(target, control_list, comp)
            return

        core = [control for control in control_list if control[0] in self.core_mask]
        auxiliary = [
            control for control in control_list if control[0] in self.aux_mask
        ]
        operands = [
            control for control in control_list if not self._is_mask(control[0])
        ]
        masks = core + auxiliary
        maskset = frozenset(wire for wire, _ in masks)

        if len(masks) == 1:
            first_prefix: Control | None = masks[0]
        elif len(masks) == 2 and self._persist_pool:
            prefix_wire = self._get_product(
                ("m", frozenset(masks)), masks, maskset
            )
            first_prefix = (prefix_wire, 1) if prefix_wire is not None else None
        else:
            first_prefix = None

        if first_prefix is not None:
            if not operands:
                self._emit(target, [first_prefix], comp)
                return
            if len(operands) == 1:
                self._emit(target, [first_prefix, operands[0]], comp)
                return
            if len(operands) == 2:
                prefix_wire = self._get_product(
                    ("mo", maskset, operands[0]),
                    [first_prefix, operands[0]],
                    maskset,
                )
                if prefix_wire is not None:
                    self._emit(target, [(prefix_wire, 1), operands[1]], comp)
                    return
                if self._temp:
                    temporary = self._temp[0]
                    self._emit(temporary, [first_prefix, operands[0]])
                    self._emit(target, [(temporary, 1), operands[1]], comp)
                    self._emit(temporary, [first_prefix, operands[0]])
                    return

        order = masks + operands
        if not order or not self._is_mask(order[0][0]):
            raise ValueError("high-fan-in decomposition requires at least one mask")
        needed = len(order) - 2
        if needed > len(self.decomp):
            raise ValueError(
                f"need {needed} decomposition ancillas, have {len(self.decomp)}"
            )
        ancillas = self.decomp
        self._emit(ancillas[0], [order[0], order[1]])
        for index in range(1, needed):
            self._emit(
                ancillas[index],
                [(ancillas[index - 1], 1), order[index + 1]],
            )
        self._emit(
            target, [(ancillas[needed - 1], 1), order[-1]], comp
        )
        for index in range(needed - 1, 0, -1):
            self._emit(
                ancillas[index],
                [(ancillas[index - 1], 1), order[index + 1]],
            )
        self._emit(ancillas[0], [order[0], order[1]])

    def flush(self) -> None:
        """Uncompute every cached product in reverse construction order."""

        for key in reversed(list(self._cache)):
            wire, factors, _ = self._cache.pop(key)
            self._emit(wire, factors)
            self._persist_pool.append(wire)


def gadget_gate(
    circ: Weight2Circuit,
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
    U0: Sequence[int] = base.SB_U0,
    U1: Sequence[int] = base.SB_U1,
    sharedB: bool = True,
    dirty: bool = False,
    vtype: str = "r57",
) -> None:
    """Emit and flush one fan-in-two nonlinear gadget.

    Mask classification is updated for this gadget before emission, which makes this
    wrapper safe for chains where an earlier gadget's fresh wires become operand shares.
    Secondary scratch follows :func:`nonlinear193.gadget_gate`: dirty mode requires one
    distinct wire per B group, while clean mode may share one integer wire.
    """

    group_count = 2 if sharedB else 3
    scratch2_wires = base._normalize_scratch2(
        scratch2, group_count, dirty=dirty
    )
    circ.set_masks(
        {scratch} | set(scratch2_wires), set(fresh_share) | set(chaff)
    )
    base.gadget_gate(
        circ,
        a_blocks,
        b_blocks,
        target_share1,
        target_share2,
        fresh_share,
        output_majority,
        scratch,
        scratch2,
        chaff,
        U0=U0,
        U1=U1,
        sharedB=sharedB,
        dirty=dirty,
        vtype=vtype,
    )
    circ.flush()


def run_gate(
    samples: int = 20_000,
    seed: int = 0,
    vtype: str = "r57",
    a_single: bool = False,
    b_single: bool = False,
    dirty: bool = False,
) -> tuple[Weight2Circuit, dict[str, object]]:
    """Build and simulate one randomly encoded fan-in-two gadget."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if vtype not in {"r57", "nab", "and", "copy"}:
        raise ValueError(f"unsupported vtype: {vtype}")
    a_count = 0 if vtype == "copy" else (1 if a_single else 2)
    b_count = 1 if b_single or vtype == "copy" else 2
    group_count = 2

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
        scratch2: int | tuple[int, ...] = tuple(
            range(next_wire, next_wire + group_count)
        )
        next_wire += group_count
    else:
        scratch2 = next_wire
        next_wire += 1
    chaff = tuple(range(next_wire, next_wire + 4))
    next_wire += 4
    decomposition = (next_wire, next_wire + 1)
    next_wire += 2
    persistent = tuple(range(next_wire, next_wire + 24))
    next_wire += 24
    temporary = (next_wire,)
    next_wire += 1

    rng = np.random.default_rng(seed)
    wires = [
        rng.integers(0, 2, samples).astype(np.uint8)
        for _ in range(next_wire)
    ]
    scratch2_tuple = tuple(scratch2) if dirty else (scratch2,)
    clean_wires = (
        output_majority
        + (scratch,)
        + scratch2_tuple
        + decomposition
        + persistent
        + temporary
    )
    for wire in clean_wires:
        wires[wire] = np.zeros(samples, np.uint8)

    circ = Weight2Circuit(
        wires,
        decomposition,
        {scratch} | set(scratch2_tuple),
        set(fresh_share) | set(chaff),
        persist=persistent,
        temp=temporary,
    )

    def decode(blocks: Sequence[WireBlock]) -> np.ndarray:
        value = np.zeros(samples, np.uint8)
        for block in blocks:
            value ^= base.E([circ.init[wire] for wire in block])
        return value.astype(np.uint8)

    a = decode(a_blocks) if a_blocks else np.zeros(samples, np.uint8)
    b = decode(b_blocks)
    c_in = (
        base.E([circ.init[wire] for wire in target_share1])
        ^ base.E([circ.init[wire] for wire in target_share2])
    ).astype(np.uint8)
    rho = base.E([circ.init[wire] for wire in fresh_share])
    u = (
        base.E([circ.init[wire] for wire in target_share1]) ^ rho
    ).astype(np.uint8)
    gate_value = {
        "r57": 1 ^ b ^ (a & b),
        "nab": (1 ^ a) & b,
        "and": a & b,
        "copy": b,
    }[vtype].astype(np.uint8)
    c_out = (c_in ^ gate_value).astype(np.uint8)

    gadget_gate(
        circ,
        a_blocks,
        b_blocks,
        target_share1,
        target_share2,
        fresh_share,
        output_majority,
        scratch,
        scratch2,
        chaff,
        vtype=vtype,
        dirty=dirty,
    )

    output_share2 = (
        target_share2[0],
        target_share2[1],
        output_majority[0],
        output_majority[1],
        output_majority[2],
    )
    actual = (
        base.E([circ.s[wire] for wire in fresh_share])
        ^ base.E([circ.s[wire] for wire in output_share2])
    ).astype(np.uint8)
    zero = np.zeros(samples, np.uint8)
    primary_scratch_restored = np.array_equal(circ.s[scratch], zero)
    scratch2_restored = all(
        np.array_equal(circ.s[wire], zero) for wire in scratch2_tuple
    )
    scratch_restored = primary_scratch_restored and scratch2_restored
    decomposition_restored = all(
        np.array_equal(circ.s[wire], zero)
        for wire in decomposition + persistent + temporary
    )
    required_ancillas_restored = (
        primary_scratch_restored
        and (dirty or scratch2_restored)
        and decomposition_restored
    )
    written_expected = (
        set(output_majority)
        | {scratch}
        | set(decomposition)
        | set(persistent)
        | set(temporary)
    )
    if vtype != "copy":
        written_expected |= set(scratch2_tuple)
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
        "decomposition": decomposition,
        "persistent": persistent,
        "temporary": temporary,
    }
    info: dict[str, object] = {
        "a": a,
        "b": b,
        "c_in": c_in,
        "gate_ab": gate_value,
        "c_out": c_out,
        "c_out_actual": actual,
        "rho": rho,
        "u": u,
        "correct": bool(np.array_equal(actual, c_out)),
        "scratch_restored": bool(scratch_restored),
        "decomposition_ancillas_restored": bool(decomposition_restored),
        "required_ancillas_restored": bool(required_ancillas_restored),
        "n_gates": len(circ.gate_log),
        "max_fanin": circ.max_physical_fanin,
        "max_physical_fanin": circ.max_physical_fanin,
        "max_requested_fanin": circ.max_requested_fanin,
        "NW": next_wire,
        "written_expected": written_expected,
        "layout": layout,
    }
    return circ, info


def build_chain(samples: int = 20_000, seed: int = 0) -> dict[str, object]:
    """Build two fan-in-two r57 gadgets with gate one feeding gate two."""

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
    decomposition = (next_wire, next_wire + 1)
    next_wire += 2
    persistent = tuple(range(next_wire, next_wire + 24))
    next_wire += 24
    temporary = (next_wire,)
    next_wire += 1

    wires = [
        rng.integers(0, 2, samples).astype(np.uint8)
        for _ in range(next_wire)
    ]
    for _, output, scratch, scratch2, _ in extras:
        for wire in output + (scratch, scratch2):
            wires[wire] = np.zeros(samples, np.uint8)
    for wire in decomposition + persistent + temporary:
        wires[wire] = np.zeros(samples, np.uint8)
    circ = Weight2Circuit(
        wires, decomposition, set(), set(), persist=persistent, temp=temporary
    )

    def value(logical_wire: int) -> np.ndarray:
        return (
            base.E([circ.s[wire] for wire in share1[logical_wire]])
            ^ base.E([circ.s[wire] for wire in share2[logical_wire]])
        ).astype(np.uint8)

    targets: dict[str, np.ndarray] = {}
    expected_writes: set[int] = set(decomposition) | set(persistent) | set(temporary)
    correct = True
    scratch_restored = True
    decomposition_ancillas_restored = True
    for gadget_index, (target, a_wire, b_wire) in enumerate(((2, 0, 1), (3, 2, 0))):
        a, b, c_in = value(a_wire), value(b_wire), value(target)
        gate_value = (1 ^ b ^ (a & b)).astype(np.uint8)
        c_out = (c_in ^ gate_value).astype(np.uint8)
        label = gadget_index + 1
        targets.update(
            {
                f"g{label}:a": a,
                f"g{label}:b": b,
                f"g{label}:gate": gate_value,
                f"g{label}:c_in": c_in,
                f"g{label}:c_out": c_out,
            }
        )
        fresh, output, scratch, scratch2, chaff = extras[gadget_index]
        expected_writes |= set(output) | {scratch, scratch2}
        gadget_gate(
            circ,
            (tuple(share1[a_wire]), tuple(share2[a_wire])),
            (tuple(share1[b_wire]), tuple(share2[b_wire])),
            tuple(share1[target]),
            tuple(share2[target]),
            fresh,
            output,
            scratch,
            scratch2,
            chaff,
            vtype="r57",
        )
        share1[target] = list(fresh)
        share2[target] = [share2[target][0], share2[target][1], *output]
        correct &= np.array_equal(value(target), c_out)
        scratch_restored &= (
            np.array_equal(circ.s[scratch], np.zeros(samples, np.uint8))
            and np.array_equal(circ.s[scratch2], np.zeros(samples, np.uint8))
        )
        decomposition_ancillas_restored &= all(
            np.array_equal(circ.s[wire], np.zeros(samples, np.uint8))
            for wire in decomposition + persistent + temporary
        )

    return {
        "name": "nonlinear291 — two fan-in-two r57 gadgets chained",
        "circ": circ,
        "targets": targets,
        "written": expected_writes,
        "correct": bool(correct),
        "scratch_restored": bool(scratch_restored),
        "decomposition_ancillas_restored": bool(
            decomposition_ancillas_restored
        ),
        "required_ancillas_restored": bool(
            scratch_restored and decomposition_ancillas_restored
        ),
        "max_fanin": circ.max_physical_fanin,
        "max_physical_fanin": circ.max_physical_fanin,
        "max_requested_fanin": circ.max_requested_fanin,
    }


__all__ = [
    "CANONICAL_MAX_PHYSICAL_FANIN",
    "CANONICAL_MAX_REQUESTED_FANIN",
    "CANONICAL_R57_GATE_COUNT",
    "Weight2Circuit",
    "build_chain",
    "gadget_gate",
    "run_gate",
]
