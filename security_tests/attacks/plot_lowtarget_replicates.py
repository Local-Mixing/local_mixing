#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def read_raw(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(
                {
                    "replicate": int(row["replicate"]),
                    "gates": int(row["gates"]),
                    "status": row["kissat_status"],
                    "wall_time": float(row["wall_time"]),
                    "timeout": row["kissat_status"] == "UNKNOWN",
                }
            )
    return rows


def read_aggregate(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(
                {
                    "gates": int(row["gates"]),
                    "n": int(row["n"]),
                    "timeouts": int(row["timeouts"]),
                    "timeout_rate": float(row["timeout_rate"]),
                    "mean_wall_time": float(row["mean_wall_time"]),
                    "median_wall_time": float(row["median_wall_time"]),
                    "stdev_wall_time": float(row["stdev_wall_time"]),
                    "min_wall_time": float(row["min_wall_time"]),
                    "max_wall_time": float(row["max_wall_time"]),
                }
            )
    return rows


def linear_fit(xs, ys):
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return slope, intercept


def fit_exponential(agg, cap):
    points = [(r["gates"], r["mean_wall_time"]) for r in agg if r["mean_wall_time"] > 0]
    xs = [g for g, _ in points]
    ys = [math.log(min(t, cap)) for _, t in points]
    slope, intercept = linear_fit(xs, ys)
    pred = [math.exp(intercept + slope * x) for x in xs]
    mean_y = sum(min(t, cap) for _, t in points) / len(points)
    ss_res = sum((min(t, cap) - p) ** 2 for (_, t), p in zip(points, pred))
    ss_tot = sum((min(t, cap) - mean_y) ** 2 for _, t in points)
    return {
        "model": "capped_mean_time ~= A * exp(B * gates)",
        "A": math.exp(intercept),
        "B": slope,
        "doubling_gates": math.log(2) / slope if slope > 0 else None,
        "r2_on_linear_time": None if ss_tot == 0 else 1 - ss_res / ss_tot,
    }


def fit_logistic_timeout(agg):
    # Binomial logistic fit on grouped timeout counts, constrained to a rising
    # timeout probability. A small sample can be nonmonotone, so an unconstrained
    # Newton step is too willing to chase local wiggles.
    xs = [r["gates"] for r in agg]
    ns = [r["n"] for r in agg]
    ks = [r["timeouts"] for r in agg]
    if not any(ks) or all(k == n for k, n in zip(ks, ns)):
        return None

    def nll(p50, beta):
        total = 0.0
        for x, n, k in zip(xs, ns, ks):
            eta = beta * (x - p50)
            p = 1 / (1 + math.exp(-max(min(eta, 40), -40)))
            p = min(max(p, 1e-9), 1 - 1e-9)
            total -= k * math.log(p) + (n - k) * math.log(1 - p)
        return total

    best = (float("inf"), None, None)
    for p50 in [480 + i * 2 for i in range(141)]:
        for beta in [0.002 + i * 0.002 for i in range(100)]:
            score = nll(p50, beta)
            if score < best[0]:
                best = (score, p50, beta)
    _, p50, beta = best

    # Local refinement around the coarse grid winner.
    step_p = 2.0
    step_b = 0.002
    for _ in range(30):
        improved = False
        for dp in (-step_p, 0, step_p):
            for db in (-step_b, 0, step_b):
                cand_p50 = p50 + dp
                cand_beta = max(beta + db, 1e-6)
                score = nll(cand_p50, cand_beta)
                if score + 1e-12 < best[0]:
                    best = (score, cand_p50, cand_beta)
                    p50, beta = cand_p50, cand_beta
                    improved = True
        if not improved:
            step_p *= 0.5
            step_b *= 0.5
            if step_p < 1e-4 and step_b < 1e-7:
                break

    return {
        "model": "P(timeout by 120s) ~= sigmoid(alpha + beta * gates)",
        "alpha": -beta * p50,
        "beta": beta,
        "p50_gates": p50,
        "negative_log_likelihood": best[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="work/sss_challenge/random64_gate_search_scaled32_25_replicates")
    parser.add_argument("--cap", type=float, default=120.0)
    args = parser.parse_args()

    out_dir = Path(args.dir)
    raw = read_raw(out_dir / "raw_results.tsv")
    agg = read_aggregate(out_dir / "aggregate.tsv")
    fit = {
        "exponential_capped_mean": fit_exponential(agg, args.cap),
        "logistic_timeout": fit_logistic_timeout(agg),
        "note": "UNKNOWN rows are right-censored at the cap; capped mean treats them as exactly the cap.",
    }
    (out_dir / "fits.json").write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")

    width, height = 1000, 760
    left, right = 80, 970
    top1, bottom1 = 70, 500
    top2, bottom2 = 560, 710
    min_gate = min(r["gates"] for r in agg)
    max_gate = max(r["gates"] for r in agg)
    y_min, y_max = 0.25, args.cap * 1.35

    def sx(gate):
        return left + (gate - min_gate) * (right - left) / (max_gate - min_gate)

    def sy_time(t):
        lt = math.log(max(t, y_min))
        return bottom1 - (lt - math.log(y_min)) * (bottom1 - top1) / (math.log(y_max) - math.log(y_min))

    def sy_rate(p):
        return bottom2 - p * (bottom2 - top2)

    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    def line(points, color, width=1, opacity=1.0, dash=None):
        if not points:
            return ""
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" opacity="{opacity}"{dash_attr}/>\n'
        )

    def circle(x, y, color, radius=3, opacity=1.0):
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}"/>\n'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="white"/>\n',
        '<text x="500" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">64-wire scaled challenge: low32 target, top25 input bits zero</text>\n',
        f'<line x1="{left}" y1="{bottom1}" x2="{right}" y2="{bottom1}" stroke="#333"/>\n',
        f'<line x1="{left}" y1="{top1}" x2="{left}" y2="{bottom1}" stroke="#333"/>\n',
        f'<line x1="{left}" y1="{bottom2}" x2="{right}" y2="{bottom2}" stroke="#333"/>\n',
        f'<line x1="{left}" y1="{top2}" x2="{left}" y2="{bottom2}" stroke="#333"/>\n',
    ]

    for t in [0.5, 1, 2, 5, 10, 20, 50, 120]:
        y = sy_time(t)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>\n')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{t:g}</text>\n')
    cap_y = sy_time(args.cap)
    svg.append(line([(left, cap_y), (right, cap_y)], "#d62728", width=1.5, dash="6 4"))
    svg.append(f'<text x="{right - 5}" y="{cap_y - 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#d62728">120s cap</text>\n')

    for p in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy_rate(p)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e6e6e6"/>\n')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{p:g}</text>\n')

    for gate in range(min_gate, max_gate + 1, 20):
        x = sx(gate)
        svg.append(f'<line x1="{x:.1f}" y1="{bottom1}" x2="{x:.1f}" y2="{bottom1 + 5}" stroke="#333"/>\n')
        svg.append(f'<line x1="{x:.1f}" y1="{bottom2}" x2="{x:.1f}" y2="{bottom2 + 5}" stroke="#333"/>\n')
        svg.append(f'<text x="{x:.1f}" y="{bottom2 + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{gate}</text>\n')

    reps = sorted({r["replicate"] for r in raw})
    for rep in reps:
        rr = sorted([r for r in raw if r["replicate"] == rep], key=lambda r: r["gates"])
        color = palette[rep % len(palette)]
        pts = [(sx(r["gates"]), sy_time(r["wall_time"])) for r in rr]
        svg.append(line(pts, color, width=1.2, opacity=0.35))
        for x, y in pts:
            svg.append(circle(x, y, color, radius=2.3, opacity=0.45))

    gates = [r["gates"] for r in agg]
    means = [r["mean_wall_time"] for r in agg]
    medians = [r["median_wall_time"] for r in agg]
    stdev = [r["stdev_wall_time"] for r in agg]
    mean_pts = [(sx(g), sy_time(t)) for g, t in zip(gates, means)]
    median_pts = [(sx(g), sy_time(t)) for g, t in zip(gates, medians)]
    svg.append(line(mean_pts, "#111111", width=2.5))
    svg.append(line(median_pts, "#d95f02", width=2.2))
    for g, mean, sd in zip(gates, means, stdev):
        x = sx(g)
        y1 = sy_time(min(mean + sd, y_max))
        y2 = sy_time(max(mean - sd, y_min))
        ym = sy_time(mean)
        svg.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#111" stroke-width="1"/>\n')
        svg.append(circle(x, ym, "#111111", radius=3.2))
    for x, y in median_pts:
        svg.append(f'<rect x="{x - 3:.1f}" y="{y - 3:.1f}" width="6" height="6" fill="#d95f02"/>\n')

    rate_pts = [(sx(r["gates"]), sy_rate(r["timeout_rate"])) for r in agg]
    svg.append(line(rate_pts, "#1b9e77", width=2.5))
    for x, y in rate_pts:
        svg.append(circle(x, y, "#1b9e77", radius=3.2))

    svg.extend(
        [
            '<text x="25" y="285" text-anchor="middle" transform="rotate(-90 25 285)" font-family="Arial, sans-serif" font-size="14">wall time (s, log scale)</text>\n',
            '<text x="25" y="640" text-anchor="middle" transform="rotate(-90 25 640)" font-family="Arial, sans-serif" font-size="14">timeout rate</text>\n',
            '<text x="525" y="750" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">gates</text>\n',
            '<rect x="740" y="52" width="210" height="82" fill="white" stroke="#ddd"/>\n',
            '<line x1="755" y1="72" x2="795" y2="72" stroke="#777" opacity="0.5"/><text x="805" y="76" font-family="Arial, sans-serif" font-size="12">individual reps</text>\n',
            '<line x1="755" y1="94" x2="795" y2="94" stroke="#111" stroke-width="2.5"/><text x="805" y="98" font-family="Arial, sans-serif" font-size="12">mean +/- sd</text>\n',
            '<line x1="755" y1="116" x2="795" y2="116" stroke="#d95f02" stroke-width="2.2"/><text x="805" y="120" font-family="Arial, sans-serif" font-size="12">median</text>\n',
        ]
    )
    svg.append("</svg>\n")
    (out_dir / "walltime_timeout_curve.svg").write_text("".join(svg))


if __name__ == "__main__":
    main()
