#!/usr/bin/env python3
"""Export deterministic local-wire templates for the nonlinear gadgets.

The templates are intentionally topology-only.  They use clean scratch,
``sharedB=True``, and two five-wire shares for every logical operand.  Native
integrators can relabel the local wire numbers while preserving the emitted
gate order.

Run from any working directory with::

    python -m gadgetization.export_templates
    python -m gadgetization.export_templates --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


# Permit ``python gadgetization/export_templates.py`` as well as ``python -m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gadgetization import nonlinear193, nonlinear291  # noqa: E402


Variant = str
Operation = str
Control = tuple[int, int]
Gate = tuple[int, int, tuple[Control, ...]]
Block = tuple[int, ...]
Blocks = tuple[Block, ...]
LayoutValue = int | Block | Blocks

VARIANTS = ("nonlinear193", "nonlinear291")
OPERATIONS = ("r57", "nab", "and", "copy")
EXPECTED_GATE_COUNTS: Mapping[tuple[Variant, Operation], int] = MappingProxyType(
    {
        ("nonlinear193", "r57"): 193,
        ("nonlinear193", "nab"): 190,
        ("nonlinear193", "and"): 193,
        ("nonlinear193", "copy"): 85,
        ("nonlinear291", "r57"): 291,
        ("nonlinear291", "nab"): 288,
        ("nonlinear291", "and"): 291,
        ("nonlinear291", "copy"): 127,
    }
)
MAX_PHYSICAL_FANIN: Mapping[Variant, int] = MappingProxyType(
    {"nonlinear193": 4, "nonlinear291": 2}
)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class TopologyTemplate:
    """One validated gadget topology in canonical local wire numbering."""

    variant: Variant
    operation: Operation
    n_wires: int
    gates: tuple[Gate, ...]
    layout: Mapping[str, LayoutValue]

    @property
    def filename(self) -> str:
        return f"{self.variant}_{self.operation}.mpmct1"

    @property
    def max_fanin(self) -> int:
        return max((len(controls) for _, _, controls in self.gates), default=0)

    def to_mpmct1(self) -> str:
        """Render the topology in the repository's standard ``mpmct1`` format."""

        lines = [f"mpmct1 {self.n_wires} {len(self.gates)}"]
        for target, complement, controls in self.gates:
            fields = [str(target), str(complement), str(len(controls))]
            for wire, polarity in controls:
                fields.extend((str(wire), str(polarity)))
            lines.append(" ".join(fields))
        return "\n".join(lines) + "\n"


def _block(start: int) -> Block:
    return tuple(range(start, start + 5))


def canonical_layout(variant: Variant, operation: Operation) -> Mapping[str, LayoutValue]:
    """Return the fixed local layout for one variant/operation pair."""

    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported operation: {operation}")

    if operation == "copy":
        layout: dict[str, LayoutValue] = {
            "target_share1": _block(0),
            "target_share2": _block(5),
            "fresh_share": _block(10),
            "a_blocks": (),
            # Copy deliberately consumes both shares of its logical control.
            "b_blocks": (_block(15), _block(20)),
            "output_majority": tuple(range(25, 28)),
            "scratch": 28,
            "scratch2": 29,
            "chaff": tuple(range(30, 34)),
        }
        if variant == "nonlinear291":
            layout.update(
                {
                    "decomposition": tuple(range(34, 36)),
                    "persistent": tuple(range(36, 60)),
                    "temporary": (60,),
                    "n_wires": 61,
                }
            )
        else:
            layout["n_wires"] = 34
    else:
        layout = {
            "target_share1": _block(0),
            "target_share2": _block(5),
            "fresh_share": _block(10),
            "a_blocks": (_block(15), _block(20)),
            "b_blocks": (_block(25), _block(30)),
            "output_majority": tuple(range(35, 38)),
            "scratch": 38,
            "scratch2": 39,
            "chaff": tuple(range(40, 44)),
        }
        if variant == "nonlinear291":
            layout.update(
                {
                    "decomposition": tuple(range(44, 46)),
                    "persistent": tuple(range(46, 70)),
                    "temporary": (70,),
                    "n_wires": 71,
                }
            )
        else:
            layout["n_wires"] = 44

    return MappingProxyType(layout)


def _as_blocks(value: LayoutValue) -> Blocks:
    if not isinstance(value, tuple) or any(not isinstance(block, tuple) for block in value):
        raise TypeError("layout entry is not a sequence of wire blocks")
    return value


def _as_block(value: LayoutValue) -> Block:
    if not isinstance(value, tuple) or any(not isinstance(wire, int) for wire in value):
        raise TypeError("layout entry is not a wire block")
    return value


def _as_wire(value: LayoutValue) -> int:
    if not isinstance(value, int):
        raise TypeError("layout entry is not a wire")
    return value


def _canonicalize_gates(
    gate_log: Sequence[tuple[int, int, Sequence[Control]]], n_wires: int
) -> tuple[Gate, ...]:
    gates: list[Gate] = []
    for gate_index, (target, complement, controls) in enumerate(gate_log):
        canonical_controls = tuple(sorted(controls))
        control_wires = [wire for wire, _ in canonical_controls]
        if complement not in (0, 1):
            raise AssertionError(f"gate {gate_index}: invalid complement {complement}")
        if not 0 <= target < n_wires:
            raise AssertionError(f"gate {gate_index}: target {target} is out of range")
        if target in control_wires:
            raise AssertionError(f"gate {gate_index}: target appears as a control")
        if len(control_wires) != len(set(control_wires)):
            raise AssertionError(f"gate {gate_index}: duplicate/contradictory control")
        for wire, polarity in canonical_controls:
            if not 0 <= wire < n_wires:
                raise AssertionError(f"gate {gate_index}: control {wire} is out of range")
            if polarity not in (0, 1):
                raise AssertionError(
                    f"gate {gate_index}: invalid control polarity {polarity}"
                )
        gates.append((target, complement, canonical_controls))
    return tuple(gates)


def build_template(variant: Variant, operation: Operation) -> TopologyTemplate:
    """Generate and validate one canonical topology entirely in memory."""

    layout = canonical_layout(variant, operation)
    n_wires = _as_wire(layout["n_wires"])
    wires = [np.zeros(1, dtype=np.uint8) for _ in range(n_wires)]

    if variant == "nonlinear193":
        module = nonlinear193
        circuit = nonlinear193.Circuit(wires)
    else:
        module = nonlinear291
        circuit = nonlinear291.Weight2Circuit(
            wires,
            _as_block(layout["decomposition"]),
            {_as_wire(layout["scratch"]), _as_wire(layout["scratch2"])},
            set(_as_block(layout["fresh_share"])) | set(_as_block(layout["chaff"])),
            persist=_as_block(layout["persistent"]),
            temp=_as_block(layout["temporary"]),
        )

    module.gadget_gate(
        circuit,
        _as_blocks(layout["a_blocks"]),
        _as_blocks(layout["b_blocks"]),
        _as_block(layout["target_share1"]),
        _as_block(layout["target_share2"]),
        _as_block(layout["fresh_share"]),
        _as_block(layout["output_majority"]),
        _as_wire(layout["scratch"]),
        _as_wire(layout["scratch2"]),
        _as_block(layout["chaff"]),
        sharedB=True,
        dirty=False,
        vtype=operation,
    )

    gates = _canonicalize_gates(circuit.gate_log, n_wires)
    template = TopologyTemplate(variant, operation, n_wires, gates, layout)
    expected_count = EXPECTED_GATE_COUNTS[(variant, operation)]
    if len(gates) != expected_count:
        raise AssertionError(
            f"{variant}/{operation}: expected {expected_count} gates, got {len(gates)}"
        )
    fanin_cap = MAX_PHYSICAL_FANIN[variant]
    if template.max_fanin > fanin_cap:
        raise AssertionError(
            f"{variant}/{operation}: fan-in {template.max_fanin} exceeds {fanin_cap}"
        )
    return template


def build_all_templates() -> tuple[TopologyTemplate, ...]:
    """Return all eight templates in stable variant/operation order."""

    return tuple(
        build_template(variant, operation)
        for variant in VARIANTS
        for operation in OPERATIONS
    )


def render_all_templates() -> dict[str, str]:
    """Return ``filename -> mpmct1 text`` without touching the filesystem."""

    return {template.filename: template.to_mpmct1() for template in build_all_templates()}


def _check_templates(output_dir: Path, rendered: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    expected_names = set(rendered)
    actual_names = {path.name for path in output_dir.glob("*.mpmct1")}
    for missing in sorted(expected_names - actual_names):
        problems.append(f"missing: {output_dir / missing}")
    for extra in sorted(actual_names - expected_names):
        problems.append(f"unexpected: {output_dir / extra}")
    for filename, expected in rendered.items():
        path = output_dir / filename
        if path.is_file() and path.read_text(encoding="ascii") != expected:
            problems.append(f"stale: {path}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEMPLATE_DIR,
        help=f"template directory (default: {TEMPLATE_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in templates without writing files",
    )
    args = parser.parse_args(argv)

    rendered = render_all_templates()
    if args.check:
        problems = _check_templates(args.output_dir, rendered)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(f"verified {len(rendered)} templates in {args.output_dir}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, body in rendered.items():
        (args.output_dir / filename).write_text(body, encoding="ascii", newline="\n")
    print(f"wrote {len(rendered)} templates to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
