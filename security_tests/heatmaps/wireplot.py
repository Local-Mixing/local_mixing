#!/usr/bin/env python3
"""Plot G57 wire participation and target counts across the full u16 wire range."""

import argparse
from pathlib import Path

import numpy as np


# Match CircuitSeq::repr in src/circuit/formats/g57.rs. Each preceding ~ adds 83.
WIRE_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
WIRE_DIGITS = {char: wire for wire, char in enumerate(WIRE_ALPHABET)}
BASE = len(WIRE_ALPHABET)
# CircuitSeq and XGate store wire IDs as u16; no circuit evaluation is needed here.
MAX_WIRES = 1 << 16
ASCII_WHITESPACE = " \t\n\r\f"


def char_to_wire(char: str) -> int:
    try:
        return WIRE_DIGITS[char]
    except KeyError:
        raise ValueError(f"invalid G57 wire character: {char!r}") from None


def count_wire_usage(circuit_str: str, num_wires: int) -> tuple[np.ndarray, np.ndarray]:
    """Count gate-pin occurrences and targets without storing a decoded circuit.

    G57 has no header. Like the canonical reader, accept surrounding ASCII
    whitespace, empty semicolon segments, and an unterminated final gate;
    reject whitespace inside the body. Reject overflowing IDs before counting.
    """
    if not 1 <= num_wires <= MAX_WIRES:
        raise ValueError(f"wire count must be between 1 and {MAX_WIRES}")
    total_counts = np.zeros(num_wires, dtype=np.int64)
    target_counts = np.zeros(num_wires, dtype=np.int64)
    wire_count = 0
    overflow = 0
    gate_number = 1

    def check_gate():
        if overflow:
            raise ValueError(f"gate {gate_number}: expected a wire character after ~")
        if wire_count not in (0, 3):
            raise ValueError(f"gate {gate_number}: expected exactly 3 wires, found {wire_count}")

    for char in circuit_str.strip(ASCII_WHITESPACE):
        if char == ";":
            check_gate()
            wire_count = 0
            gate_number += 1
        elif char == "~":
            overflow += BASE
            if overflow >= MAX_WIRES:
                raise ValueError(f"gate {gate_number}: wire ID exceeds {MAX_WIRES - 1}")
        else:
            wire = overflow + char_to_wire(char)
            overflow = 0
            if wire >= num_wires:
                raise ValueError(
                    f"gate {gate_number}: wire {wire} is outside the declared range "
                    f"0..{num_wires - 1}"
                )
            wire_count += 1
            if wire_count > 3:
                raise ValueError(f"gate {gate_number}: expected exactly 3 wires, found more")
            total_counts[wire] += 1
            if wire_count == 1:
                target_counts[wire] += 1
    check_gate()
    return total_counts, target_counts


def plot_wire_scatter(circuit_str, num_wires, x_label_str, save_path):
    total_counts, target_counts = count_wire_usage(circuit_str, num_wires)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stable ties make ranks deterministic, including the unused wires.
    sorted_indices = np.argsort(-total_counts, kind="stable")
    ranks = np.arange(num_wires)
    fig, ax = plt.subplots(figsize=(16, 6))
    try:
        ax.scatter(ranks, total_counts[sorted_indices], color="blue", s=10,
                   label="Total participation", alpha=0.7)
        ax.scatter(ranks, target_counts[sorted_indices], color="red", s=10,
                   label="Target uses", alpha=0.7)
        # Preserve all data points while bounding tick labels for wide circuits.
        ticks = np.linspace(0, num_wires - 1, min(num_wires, 24), dtype=int)
        ax.set_xticks(ticks, labels=sorted_indices[ticks])
        ax.set_xlabel(f"Wire ID, ranked by total participation ({x_label_str})")
        ax.set_ylabel("Gate count")
        ax.set_title("Gate counts per wire")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    finally:
        plt.close(fig)
    print(f"Saved wire scatter plot to {save_path}")


def wire_count_argument(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("wire count must be an integer") from None
    if not 1 <= count <= MAX_WIRES:
        raise argparse.ArgumentTypeError(f"wire count must be between 1 and {MAX_WIRES}")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=f"Plot headerless base-83 G57 wire counts (up to {MAX_WIRES:,} wires)."
    )
    parser.add_argument("--c", type=Path, required=True, help="G57 circuit file")
    parser.add_argument("--n", type=wire_count_argument, required=True,
                        help=f"physical wire count, 1..{MAX_WIRES} (IDs 0..n-1)")
    parser.add_argument("--x", required=True, help="label for the circuit")
    parser.add_argument("--out", type=Path, default=Path("wire_scatter.png"),
                        help="output plot path (default: wire_scatter.png)")
    args = parser.parse_args(argv)
    try:
        circuit_str = args.c.read_text(encoding="ascii")
        plot_wire_scatter(circuit_str, args.n, args.x, args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
