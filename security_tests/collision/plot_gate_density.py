#!/usr/bin/env python3
"""Plot gate-density sweep: CPU-seconds vs m/(n log₂ n).

One curve per (attack, λ). Attacks share color; curves labeled by λ.

Usage (repo root):
  python3 security_tests/collision/plot_gate_density.py \\
    --results security_tests/collision/fixtures/gate_density/gate_density_results.json \\
    --out security_tests/collision/fixtures/gate_density/gate_density.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ATTACK_STYLE = {
    "birthday": {"color": "#1f77b4", "label": "birthday"},
    "rho": {"color": "#ff7f0e", "label": "rho/DP"},
    "sat": {"color": "#d62728", "label": "SAT"},
}
LAMBDA_MARKERS = {16: "o", 32: "s", 64: "^"}


def density(cell: dict) -> float:
    if "density_n_log_n" in cell:
        return float(cell["density_n_log_n"])
    n = int(cell["n"])
    return float(cell["gates"]) / (n * math.log2(n))


def collect(results: dict) -> dict[str, dict[int, list[tuple[float, float, str]]]]:
    """attack -> λ -> list of (density, cpu_seconds, status)."""
    out: dict[str, dict[int, list[tuple[float, float, str]]]] = {
        a: {} for a in ATTACK_STYLE
    }
    for cell in results.get("cells", {}).values():
        lam = int(cell["lambda"])
        dens = density(cell)
        for attack in ATTACK_STYLE:
            m = cell.get(attack)
            if not m:
                continue
            phase = m.get("phase")
            if phase == "skipped":
                continue
            cpu = m.get("cpu_seconds")
            if cpu is None:
                continue
            status = "ok" if m.get("ok") else (phase or "fail")
            out[attack].setdefault(lam, []).append((dens, float(cpu), status))
    for attack in out:
        for lam in out[attack]:
            out[attack][lam].sort(key=lambda t: t[0])
    return out


def plot(results: dict, out_png: Path, out_pdf: Path | None = None) -> None:
    series = collect(results)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    for attack, style in ATTACK_STYLE.items():
        color = style["color"]
        for lam, pts in sorted(series.get(attack, {}).items()):
            if not pts:
                continue
            marker = LAMBDA_MARKERS.get(lam, "o")
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ok_flags = [p[2] == "ok" for p in pts]
            # Slight per-attack vertical nudge so nearby labels don't collide.
            dy = {"birthday": 8, "rho": -10, "sat": 8}.get(attack, 0)

            # One continuous polyline per (attack, λ). Dashed iff any timeout
            # is on the curve; same bitwidth marker on every point.
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                ms=7,
                lw=1.8,
                linestyle=("-" if all(ok_flags) else "--"),
                alpha=0.95,
                zorder=3,
            )

            ax.annotate(
                f"λ={lam}",
                xy=(xs[-1], ys[-1]),
                xytext=(7, dy),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
                va="center",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m\,/\,(n\log_2 n)$  (gate density)")
    ax.set_ylabel("CPU-seconds (wall × threads)")
    ax.set_title(
        "Collision cost vs gate density "
        r"($n=3\lambda$; geometric $m$ from $\frac{1}{2}n\log_2 n$ to $n^2$)"
    )
    ax.grid(True, which="both", ls=":", alpha=0.45)

    attack_handles = [
        Line2D([0], [0], color=s["color"], lw=2.2, label=s["label"])
        for s in ATTACK_STYLE.values()
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            marker=LAMBDA_MARKERS[lam],
            lw=0,
            ms=7,
            label=f"λ={lam}",
        )
        for lam in (16, 32, 64)
    ]
    style_handles = [
        Line2D([0], [0], color="#444444", ls="-", lw=1.8, label="solved"),
        Line2D([0], [0], color="#444444", ls="--", lw=1.8, label="timeout"),
    ]
    leg1 = ax.legend(handles=attack_handles, loc="upper left", title="Attack")
    ax.add_artist(leg1)
    ax.legend(
        handles=marker_handles + style_handles,
        loc="lower right",
        title="Bitwidth / status",
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        type=Path,
        default=Path(
            "security_tests/collision/fixtures/gate_density/gate_density_results.json"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "security_tests/collision/fixtures/gate_density/gate_density.png"
        ),
    )
    args = ap.parse_args()
    results = json.loads(args.results.read_text())
    pdf = args.out.with_suffix(".pdf")
    plot(results, args.out, pdf)
    print(f"wrote {args.out} and {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
