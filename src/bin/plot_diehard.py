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
    'ofb.balanced': 'red',
    'ofb.random': 'orange'
}

def plot_file(path: Path, ax: plt.Axes) -> None:
    xs, summaries = parse_file(path)
    if not xs:
        return

    mins = [item[0] for item in summaries]
    q1s = [item[1] for item in summaries]
    meds = [item[2] for item in summaries]
    q3s = [item[3] for item in summaries]
    maxs = [item[4] for item in summaries]

    color = colors[path.stem]

    L = ax.plot(xs, meds, color=color, linewidth=2.5, label=path.stem.replace('.', ' '))
    ax.fill_between(xs, q1s, q3s, color=color, alpha=0.2, linewidth=0)

    ax.plot(xs, mins, color=color, linewidth=1.0, linestyle=':')
    ax.plot(xs, maxs, color=color, linewidth=1.0, linestyle=':')

    # return count
    return summaries[0][-1]

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot median/min/max and middle 50% from data files.")
    parser.add_argument("files", nargs="+", type=Path, help="Input file(s) to plot")
    parser.add_argument("--logy", action='store_true', help="Log y axis")
    args = parser.parse_args()

    fig, ax = plt.subplots()
    for file_path in args.files:
        count = plot_file(file_path, ax)


    ax.set_xlabel("circuit length")
    ax.set_ylabel("# failed tests")
    ax.semilogx(base=2)

    if args.logy:
        ax.semilogy()
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.ylim(bottom=0)
    plt.title(f"Diehard(er) results ({count} circuits each)")

    fig.tight_layout()
    plt.savefig('out.png', dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())