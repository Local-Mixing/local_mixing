#!/usr/bin/env python3
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
CANON4_CSV = ROOT / "bench_results_canon4.csv"
POLYCANON_CSV = ROOT / "bench_results_polycanon.csv"
MERGED_CSV = ROOT / "bench_results_merged.csv"
TABLE_MD = ROOT / "bench_canon_vs_polycanon_table.md"
PLOT_PNG = ROOT / "bench_canon_vs_polycanon.png"
FINDINGS_MD = ROOT / "BENCH_FINDINGS.md"

VARIANTS = ("rule_l_on", "rule_l_off", "wl")
COLORS = {
    "rule_l_on": "#2563eb",
    "rule_l_off": "#16a34a",
    "wl": "#dc2626",
}
LABELS = {
    "rule_l_on": "canon4 Rule L on",
    "rule_l_off": "canon4 Rule L off",
    "wl": "polycanon WL",
}


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"algo", "n", "m", "seed", "variant", "nanos", "valid"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} does not have the expected benchmark schema")
    return rows


def median_ns(rows, n, variant):
    values = [
        int(row["nanos"])
        for row in rows
        if int(row["n"]) == n and row["variant"] == variant
    ]
    return statistics.median(values)


def format_ms(nanos):
    return f"{nanos / 1_000_000:.3f}"


def validity(rows, variant):
    values = [
        row["valid"] == "1"
        for row in rows
        if row["variant"] == variant and row["valid"] in {"0", "1"}
    ]
    return sum(values), len(values)


def write_plot(ns, aggregates):
    width, height = 1200, 760
    left, right, top, bottom = 115, 55, 70, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    title = ImageFont.load_default(size=24)

    all_values = [aggregates[(n, v)] for n in ns for v in VARIANTS]
    log_min = math.floor(math.log10(min(all_values)))
    log_max = math.ceil(math.log10(max(all_values)))
    if log_min == log_max:
        log_max += 1

    def x_pos(n):
        return left + (n - ns[0]) / (ns[-1] - ns[0]) * plot_w

    def y_pos(value):
        fraction = (math.log10(value) - log_min) / (log_max - log_min)
        return top + (1.0 - fraction) * plot_h

    draw.text((left, 22), "Polynomial canonicalization benchmark", fill="black", font=title)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    for exponent in range(log_min, log_max + 1):
        value = 10**exponent
        y = y_pos(value)
        draw.line((left, y, left + plot_w, y), fill="#d1d5db", width=1)
        label = f"10^{exponent} ns"
        draw.text((12, y - 9), label, fill="#374151", font=small)

    for n in ns:
        x = x_pos(n)
        draw.line((x, top + plot_h, x, top + plot_h + 7), fill="black", width=2)
        draw.text((x - 10, top + plot_h + 12), str(n), fill="black", font=small)

    for variant in VARIANTS:
        points = [(x_pos(n), y_pos(aggregates[(n, variant)])) for n in ns]
        draw.line(points, fill=COLORS[variant], width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS[variant])

    legend_x = left + 20
    legend_y = top + 18
    for idx, variant in enumerate(VARIANTS):
        y = legend_y + idx * 30
        draw.line((legend_x, y + 8, legend_x + 42, y + 8), fill=COLORS[variant], width=4)
        draw.text((legend_x + 54, y), LABELS[variant], fill="black", font=small)

    draw.text((left + plot_w / 2 - 55, height - 50), "wire count (n)", fill="black", font=font)
    image.save(PLOT_PNG)


def main():
    rows = read_rows(CANON4_CSV) + read_rows(POLYCANON_CSV)
    fieldnames = ["algo", "n", "m", "seed", "variant", "nanos", "valid"]
    with MERGED_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ns = sorted({int(row["n"]) for row in rows})
    aggregates = {
        (n, variant): median_ns(rows, n, variant)
        for n in ns
        for variant in VARIANTS
    }

    table_lines = [
        "| n | canon4 L on (ms) | canon4 L off (ms) | polycanon WL (ms) | WL / L on | WL / L off |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    ratios_on = []
    ratios_off = []
    for n in ns:
        on = aggregates[(n, "rule_l_on")]
        off = aggregates[(n, "rule_l_off")]
        wl = aggregates[(n, "wl")]
        ratios_on.append(wl / on)
        ratios_off.append(wl / off)
        table_lines.append(
            f"| {n} | {format_ms(on)} | {format_ms(off)} | {format_ms(wl)} | "
            f"{wl / on:.2f}x | {wl / off:.2f}x |"
        )
    TABLE_MD.write_text("\n".join(table_lines) + "\n")

    validity_lines = []
    for variant in VARIANTS:
        passed, total = validity(rows, variant)
        rate = passed / total if total else float("nan")
        validity_lines.append(
            f"- {LABELS[variant]}: {passed}/{total} valid ({rate:.1%})"
        )

    findings = [
        "# Canonicalization benchmark findings",
        "",
        f"Across all n values, polycanon WL / canon4 Rule L on has a median ratio of "
        f"{statistics.median(ratios_on):.2f}x (range {min(ratios_on):.2f}x to "
        f"{max(ratios_on):.2f}x).",
        "",
        f"Against canon4 Rule L off, the median ratio is "
        f"{statistics.median(ratios_off):.2f}x (range {min(ratios_off):.2f}x to "
        f"{max(ratios_off):.2f}x).",
        "",
        "Validity:",
        *validity_lines,
        "",
        "The validity check is outside the timed region. Canon4 compares canonical "
        "polynomial forms after rewiring; polycanon uses the challenge's inferred "
        "relative-permutation round trip. Polycanon hash ties can therefore appear "
        "as genuine validity failures.",
        "",
        f"See `{TABLE_MD.name}`, `{MERGED_CSV.name}`, and `{PLOT_PNG.name}`.",
    ]
    FINDINGS_MD.write_text("\n".join(findings) + "\n")
    write_plot(ns, aggregates)
    print("\n".join(table_lines))
    print()
    print("\n".join(validity_lines))


if __name__ == "__main__":
    main()
