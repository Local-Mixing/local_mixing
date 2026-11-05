#!/usr/bin/env python3

"""Plot summary statistics from one or more data files.

Each input file should contain lines shaped like:

    128: a, b, c, d, e, ...

The number before the colon is the x-value. The comma-separated values are
parsed as floats and summarized per x-value:

* median: thick colored line
* min/max: thin dark line in the same color
* middle 50%: shaded region from Q1 to Q3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def summarize(values: list[float]) -> tuple[float, float, float, float, float]:
    sorted_values = sorted(values)
    count = len(sorted_values)
    q1_index = min(count - 1, count // 4)
    q3_index = min(count - 1, (3 * count) // 4)
    return (
        min(sorted_values),
        sorted_values[q1_index],
        median(sorted_values),
        sorted_values[q3_index],
        max(sorted_values),
        count,
    )


def parse_file(path: Path) -> tuple[list[float], list[tuple[float, float, float, float, float]]]:
    xs: list[float] = []
    summaries: list[tuple[float, float, float, float, float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if ":" not in line:
                raise ValueError(f"{path}:{line_number}: expected 'x: value, value, ...'")

            x_text, values_text = line.split(":", 1)
            x_value = int(x_text.strip())
            values = [int(item.strip()) for item in values_text.split(",") if item.strip()]
            if not values:
                raise ValueError(f"{path}:{line_number}: no data values found")

            xs.append(x_value)
            summaries.append(summarize(values))

    return xs, summaries


colors = {
    'ctr.balanced': 'blue',
    'ctr.random': 'green',
    'ctr.fanout-two': 'magenta',
    'ofb.balanced': 'red',
    'ofb.random': 'orange'
}

def plot_file(path: Path, ax: plt.Axes, gate_length: int | None = None) -> int | None:
    xs, summaries = parse_file(path)
    if not xs:
        return

    if gate_length is None:
        plot_xs = xs
        plot_summaries = summaries

        mins = [item[0] for item in plot_summaries]
        q1s = [item[1] for item in plot_summaries]
        meds = [item[2] for item in plot_summaries]
        q3s = [item[3] for item in plot_summaries]
        maxs = [item[4] for item in plot_summaries]

        color = colors.get(path.stem.split('.n', 1)[0], None)

        L = ax.plot(plot_xs, meds, color=color, linewidth=2.5, label=path.stem.replace('.', ' '))
        color = L[0].get_color()
        ax.fill_between(plot_xs, q1s, q3s, color=color, alpha=0.2, linewidth=0)

        ax.plot(plot_xs, mins, color=color, linewidth=1.0, linestyle=':')
        ax.plot(plot_xs, maxs, color=color, linewidth=1.0, linestyle=':')

    else:
        try:
            summary_index = xs.index(gate_length)
        except ValueError:
            raise ValueError(f"{path}: gate length {gate_length} not found") from None

        wire_match = re.search(r"\.n(\d+)(?:\D|$)", path.name)
        if wire_match is None:
            raise ValueError(f"{path}: could not extract number of wires from filename")

        plot_xs = [int(wire_match.group(1))]
        plot_summaries = [summaries[summary_index]]

    return plot_xs[0], plot_summaries[0]

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot median/min/max and middle 50% from data files.")
    parser.add_argument("files", nargs="+", type=Path, help="Input file(s) to plot")
    parser.add_argument("--logy", action='store_true', help="Log y axis")
    parser.add_argument(
        "--wires",
        type=int,
        metavar="GATE_LENGTH",
        help="Plot this gate length from each file against the wire count in its filename",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots()

    if args.wires:
        (x, d) = list(zip(*(plot_file(f, ax, args.wires) for f in args.files)))
        (mins, q1s, meds, q3s, maxs, _) = zip(*d)

        L = ax.plot(x, meds, color='red', linewidth=2.5)
        color = L[0].get_color()
        ax.fill_between(x, q1s, q3s, color=color, alpha=0.2, linewidth=0)

        ax.plot(x, mins, color=color, linewidth=1.0, linestyle=':')
        ax.plot(x, maxs, color=color, linewidth=1.0, linestyle=':')

        count = d[0][-1]
    else:
        for file_path in args.files:
            count = plot_file(file_path, ax, args.wires)


    ax.set_xlabel("number of wires" if args.wires is not None else "circuit length")
    ax.set_ylabel("# failed tests")
    ax.semilogx(base=2)

    if args.logy:
        ax.semilogy()
    else:
        plt.ylim(bottom=0)

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.title(f"Diehard(er) results ({count} circuits each)")

    fig.tight_layout()
    plt.savefig('out.png', dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())