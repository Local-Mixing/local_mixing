#!/usr/bin/env python3
"""Render small heatmaps from a random-circuit ET aggregate.tsv file."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def color_for_runtime(seconds: float | None, limit: float) -> str:
    if seconds is None:
        return "#f6f8fa"
    t = max(0.0, min(1.0, seconds / max(limit, 1.0)))
    # A quiet green -> yellow -> red scale. It keeps all-timeout cells vivid.
    stops = [
        (0.00, (47, 179, 68)),
        (0.18, (187, 220, 76)),
        (0.45, (255, 212, 96)),
        (0.75, (251, 133, 73)),
        (1.00, (214, 64, 69)),
    ]
    for (left_t, left), (right_t, right) in zip(stops, stops[1:]):
        if t <= right_t:
            u = 0.0 if right_t == left_t else (t - left_t) / (right_t - left_t)
            return "#{:02x}{:02x}{:02x}".format(*(lerp(left[i], right[i], u) for i in range(3)))
    return "#d64045"


def text_color(hex_color: str) -> str:
    if not hex_color.startswith("#") or len(hex_color) != 7:
        return "#1f2328"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return "#ffffff" if lum < 0.48 else "#1f2328"


def low_variance(row: dict[str, str], min_runs: int) -> bool:
    runs = to_int(row.get("runs"))
    mean = to_float(row.get("mean_wall"))
    stddev = to_float(row.get("stddev_wall"))
    unknown_rate = to_float(row.get("unknown_rate")) or 0.0
    if runs < min_runs or mean in (None, 0) or stddev is None:
        return False
    return unknown_rate <= 0.25 and stddev / mean <= 0.50


def format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    if value < 10:
        return f"{value:.1f}"
    return f"{value:.0f}"


def load_progress(out_dir: Path) -> dict[str, object]:
    path = out_dir / "progress.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_rows(out_dir: Path) -> list[dict[str, str]]:
    path = out_dir / "aggregate.tsv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def heatmap_html(
    solver: str,
    n: int,
    rows: list[dict[str, str]],
    *,
    limit: float,
    min_runs: int,
) -> str:
    ms = sorted({to_int(r.get("m")) for r in rows})
    ks = sorted({to_int(r.get("k")) for r in rows}, reverse=True)
    by_cell = {(to_int(r.get("m")), to_int(r.get("k"))): r for r in rows}
    head = "<tr><th class='corner'>k \\ m</th>" + "".join(f"<th>{m}</th>" for m in ms) + "</tr>"
    body = []
    for k in ks:
        cells = [f"<th>{k}</th>"]
        for m in ms:
            row = by_cell.get((m, k))
            if row is None:
                cells.append("<td class='empty'></td>")
                continue
            mean = to_float(row.get("mean_wall"))
            median = to_float(row.get("median_wall"))
            stddev = to_float(row.get("stddev_wall"))
            runs = to_int(row.get("runs"))
            unknown_rate = to_float(row.get("unknown_rate")) or 0.0
            color = color_for_runtime(mean, limit)
            classes = ["cell"]
            if runs > 1:
                classes.append("multi")
            if low_variance(row, min_runs):
                classes.append("lowvar")
            if unknown_rate >= 0.5:
                classes.append("timeout")
            title = (
                f"{solver} n={n} m={m} k={k}\\n"
                f"mean={mean if mean is not None else ''}s median={median if median is not None else ''}s\\n"
                f"stddev={stddev if stddev is not None else ''}s runs={runs} unknown_rate={unknown_rate}"
            )
            cells.append(
                "<td "
                f"class='{' '.join(classes)}' "
                f"style='background:{color};color:{text_color(color)}' "
                f"title='{html.escape(title, quote=True)}'>"
                f"<span>{html.escape(format_seconds(mean))}</span>"
                f"<small>{runs}x</small>"
                "</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<section><h2>{html.escape(solver)} n={n}</h2>"
        f"<table class='heatmap'>{head}<tbody>{''.join(body)}</tbody></table></section>"
    )


def render(out_dir: Path, html_path: Path, min_runs: int, limit_override: float | None) -> None:
    rows = load_rows(out_dir)
    progress = load_progress(out_dir)
    limit = limit_override or float(progress.get("time_limit") or 60)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        solver = row.get("solver", "")
        n = to_int(row.get("n"))
        grouped.setdefault((solver, n), []).append(row)
    sections = [
        heatmap_html(solver, n, group_rows, limit=limit, min_runs=min_runs)
        for (solver, n), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    updated = html.escape(str(progress.get("updated_at", "")))
    completed = html.escape(str(progress.get("completed_runs", len(rows))))
    planned = html.escape(str(progress.get("planned_runs", "")))
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>ET Heatmaps</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#1f2328}"
        ".meta,.legend{color:#57606a;font-size:14px}"
        "section{margin:26px 0 38px}"
        "h1{margin-bottom:6px}h2{margin:0 0 10px;font-size:18px}"
        ".legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:16px 0 22px}"
        ".sample{display:inline-block;width:24px;height:18px;vertical-align:middle;border-radius:3px;margin-right:6px}"
        ".sample.multi{border:3px solid #24292f}.sample.lowvar{box-shadow:inset 0 0 0 4px #00bcd4}.sample.timeout{background:repeating-linear-gradient(135deg,#d64045,#d64045 6px,#b82d33 6px,#b82d33 12px)}"
        "table.heatmap{border-collapse:separate;border-spacing:3px;font-size:12px}"
        ".heatmap th{font-weight:600;color:#57606a;padding:3px 5px;text-align:center}"
        ".heatmap .corner{text-align:right}"
        ".heatmap td{width:58px;height:42px;text-align:center;border-radius:6px;position:relative;border:1px solid rgba(31,35,40,.16)}"
        ".heatmap td.multi{border:3px solid #24292f}"
        ".heatmap td.lowvar{box-shadow:inset 0 0 0 4px #00bcd4}"
        ".heatmap td.timeout{background-image:repeating-linear-gradient(135deg,rgba(0,0,0,0),rgba(0,0,0,0) 7px,rgba(0,0,0,.13) 7px,rgba(0,0,0,.13) 14px)}"
        ".heatmap td.empty{background:#f6f8fa;border:1px dashed #d0d7de}"
        ".heatmap td span{font-weight:700;font-size:13px;display:block}"
        ".heatmap td small{font-size:10px;opacity:.8}"
        "</style>"
        "<h1>ET Runtime Heatmaps</h1>"
        f"<p class='meta'>Updated {updated}. Completed {completed}"
        + (f" / {planned}" if planned else "")
        + f" rows. Fill color is mean wall time, capped visually at {limit:g}s.</p>"
        "<div class='legend'>"
        "<span><i class='sample' style='background:#2fb344'></i>fast</span>"
        "<span><i class='sample' style='background:#ffd460'></i>moderate</span>"
        "<span><i class='sample' style='background:#d64045'></i>near timeout</span>"
        "<span><i class='sample multi'></i>multiple runs</span>"
        "<span><i class='sample lowvar'></i>low variance</span>"
        "<span><i class='sample timeout'></i>many timeouts</span>"
        "</div>"
        + ("".join(sections) if sections else "<p>No aggregate rows yet.</p>"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()
    html_path = args.html or (args.out_dir / "heatmaps.html")
    render(args.out_dir, html_path, args.min_runs, args.time_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
