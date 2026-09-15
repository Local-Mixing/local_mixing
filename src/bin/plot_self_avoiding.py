#!/usr/bin/env python3

"""Plot walk-length, last-gate, collision-lag, and permutation histograms.

Reads the CSV written by `self_avoiding`:

    # wires=3 domain=8 factorial=40320
    length,rank,last_gate,lag
    ...

The permutation axis is the Lehmer/factoradic rank of the permutation at
the first self-intersection, scaled into `[0, (2^n)!]`. Identity sits at 0;
the reversal permutation `(N-1, …, 0)` sits at the right edge.

Collision lag is the suffix length of the identity in the generator word
(ℓ such that the last ℓ gates multiply to id).

The combined figure puts the length distribution on the left and stacks
the last-gate, collision-lag, and intersecting-permutation histograms on
the right. Walk length overlays a fitted Weibull
$N^{k-1}e^{-\\lambda N^{k}}$ (Rayleigh at $k=2$, exponential at $k=1$).
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path


def format_gate(a: int, b: int, c: int, n: int) -> str:
    if n < 10:
        return f"{a}{b}{c}"
    return f"{a}_{b}_{c}"


def base_gate_labels(n: int) -> list[str]:
    labels = []
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                labels.append(format_gate(a, b, c, n))
    return labels


def parse_csv(path: Path) -> tuple[int | None, list[int], list[int], list[str], list[int]]:
    wires = None
    lengths: list[int] = []
    ranks: list[int] = []
    last_gates: list[str] = []
    lags: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                for part in line.lstrip("# ").split():
                    if part.startswith("wires="):
                        wires = int(part.split("=", 1)[1])
                continue
            if line.startswith("length"):
                continue
            cols = line.split(",")
            lengths.append(int(cols[0]))
            ranks.append(int(cols[1]))
            last_gates.append(cols[2])
            lags.append(int(cols[3]) if len(cols) > 3 else 0)
    return wires, lengths, ranks, last_gates, lags


def percentile(sorted_vals: list[int], p: float) -> int:
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(round(p * (n - 1)))))
    return sorted_vals[idx]


def bulk_xmax(values: list[int]) -> int:
    """Upper x-limit for the bulk of `values`; wild maxima are clipped."""
    s = sorted(values)
    q1 = percentile(s, 0.25)
    q3 = percentile(s, 0.75)
    iqr = max(1, q3 - q1)
    fence = q3 + 3 * iqr
    p95 = percentile(s, 0.95)
    p99 = percentile(s, 0.99)
    return min(s[-1], max(p95, min(p99, fence)))


def _line_xs(lo: float, hi: float, n: int = 400) -> list[float]:
    if n <= 1 or hi <= lo:
        return [float(lo)]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def fit_weibull(lengths: list[int]) -> tuple[float, float] | None:
    """MLE for P(N) ∝ N^{k-1} exp(-λ N^k). Rate form: k λ N^{k-1} e^{-λ N^k}.

    k=1 is exponential; k=2 is Rayleigh.
    """
    xs = [float(t) for t in lengths if t > 0]
    if len(xs) < 2:
        return None
    n = len(xs)
    logs = [math.log(t) for t in xs]
    mean_log = sum(logs) / n
    var_log = sum((u - mean_log) ** 2 for u in logs) / n
    if var_log <= 1e-18:
        return None
    k = math.pi / math.sqrt(6.0 * var_log)
    k = min(8.0, max(0.2, k))

    def powers(shape: float) -> tuple[float, float, float]:
        den = num = quad = 0.0
        for t_log in logs:
            w = math.exp(shape * t_log)
            den += w
            num += w * t_log
            quad += w * t_log * t_log
        return den, num, quad

    for _ in range(30):
        den, num, quad = powers(k)
        if den <= 0:
            return None
        g = 1.0 / k + mean_log - num / den
        gp = -1.0 / (k * k) - (den * quad - num * num) / (den * den)
        if abs(gp) < 1e-18:
            break
        nxt = k - g / gp
        nxt = min(8.0, max(0.2, nxt))
        if abs(nxt - k) < 1e-10:
            k = nxt
            break
        k = nxt
    den, _, _ = powers(k)
    if den <= 0:
        return None
    lam = n / den
    if lam <= 0 or k <= 0:
        return None
    return lam, k


def latex_sci(x: float, digits: int = 3) -> str:
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    coeff = x / (10**exp)
    coeff_s = f"{coeff:.{digits}g}"
    if abs(float(coeff_s)) >= 10:
        coeff /= 10.0
        exp += 1
        coeff_s = f"{coeff:.{digits}g}"
    if exp == 0:
        return coeff_s
    return rf"{coeff_s}\times 10^{{{exp}}}"


def overlay_length_fit(ax, lengths: list[int], edges, wires: int) -> None:
    """Histogram-scaled Weibull MLE: P(N) ∝ N^{k-1} exp(-λ N^k)."""
    n_walks = len(lengths)
    delta = float(edges[1] - edges[0])
    ts = _line_xs(float(edges[0]), float(edges[-1]))
    fitted = fit_weibull(lengths)
    if fitted is None:
        return
    lam, k = fitted
    ys = []
    for t in ts:
        if t <= 0:
            ys.append(0.0)
            continue
        log_t = math.log(t)
        log_f = math.log(k * lam) + (k - 1.0) * log_t - lam * math.exp(k * log_t)
        ys.append(n_walks * delta * math.exp(log_f))
    special = ""
    if abs(k - 2.0) < 0.25:
        special = "  (Rayleigh)"
    elif abs(k - 1.0) < 0.25:
        special = "  (exponential)"
    lam_tex = latex_sci(lam)
    eq = (
        rf"$P(N)\propto N^{{k-1}}\,e^{{-\lambda N^{{k}}}}$"
        + "\n"
        + rf"$k={k:.2f}$, $\lambda={lam_tex}$"
        + special
    )
    ax.plot(ts, ys, color="#c44e52", linewidth=2.0, label=eq)
    extra = ""
    if wires <= 3:
        extra = f"  (k=2 is Rayleigh; birthday 8!={math.factorial(8)})"
    print(
        f"length fit  Weibull  P(N)∝ N^(k-1) exp(-λ N^k)  "
        f"k={k:.4f}  λ={lam:.6e}{extra}"
    )
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        loc="upper right",
        fontsize=10,
        borderpad=0.6,
    )


def plot_length(ax, lengths: list[int], title: str, wires: int) -> None:
    lo, hi = min(lengths), max(lengths)
    n_bins = min(256, hi - lo + 1)
    right = hi + 1
    _counts, edges, _patches = ax.hist(
        lengths, bins=n_bins, range=(lo, right), color="#3b6ea5", label="walks"
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel("walk length (gates until first intersection)")
    ax.set_ylabel("count")
    overlay_length_fit(ax, lengths, edges, wires)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def plot_gates(ax, last_gates: list[str], wires: int) -> None:
    labels = base_gate_labels(wires)
    counts = Counter(last_gates)
    ys = [counts.get(label, 0) for label in labels]
    ax.bar(range(len(labels)), ys, color="#6a9955")
    ax.set_xticks(range(len(labels)))
    rotation = 0 if len(labels) <= 8 else 90
    fontsize = 10 if len(labels) <= 24 else 6
    ax.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)
    ax.set_xlabel("last non-intersecting gate")
    ax.set_ylabel("count")
    ax.set_title(f"Final gates (n={wires})")
    ax.grid(True, axis="y", alpha=0.3)


def integer_xticks(ax) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{int(round(x))}"))


def plot_lags(ax, lags: list[int]) -> None:
    lo = min(lags)
    hi = bulk_xmax(lags)
    clipped = sum(1 for x in lags if x > hi)
    bulk = [x for x in lags if x <= hi]
    span = hi - lo + 1
    if span <= 40:
        xs = list(range(lo, hi + 1))
        counts = Counter(bulk)
        ax.bar(xs, [counts.get(x, 0) for x in xs], color="#d4a017")
        ax.set_xlim(lo - 0.5, hi + 0.5)
    else:
        ax.hist(bulk, bins=min(80, span), range=(lo, hi + 1), color="#d4a017")
        ax.set_xlim(lo, hi)
    integer_xticks(ax)
    ax.set_xlabel("collision lag (relation suffix length)")
    ax.set_ylabel("count")
    title = "Collision lags"
    if clipped:
        title = f"{title}  ({clipped} walks with ℓ > {hi} hidden)"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def plot_perms(ax, fracs: list[float], bins: int, nn: int, tick_nl: bool) -> None:
    ax.hist(fracs, bins=bins, range=(0.0, 1.0), color="#c44e52")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if tick_nl:
        ax.set_xticklabels(["0\n(id)", "", f"{nn}! / 2", "", f"{nn}!\n(rev)"])
    else:
        ax.set_xticklabels(["0 (id)", "", f"{nn}! / 2", "", f"{nn}! (rev)"])
    ax.set_xlabel("Rank of final permutation")
    ax.set_ylabel("count")
    ax.set_title(f"Intersecting permutations in $S_{{{nn}}}$")
    ax.grid(True, axis="y", alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("saw_walks.csv"))
    parser.add_argument("--wires", type=int, default=None)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("saw_walks.png"))
    parser.add_argument("--out-lengths", type=Path, default=None)
    parser.add_argument("--out-gates", type=Path, default=None)
    parser.add_argument("--out-lags", type=Path, default=None)
    parser.add_argument("--out-perms", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wires, lengths, ranks, last_gates, lags = parse_csv(args.csv)
    if args.wires is not None:
        wires = args.wires
    if wires is None:
        raise SystemExit("could not infer wire count; pass --wires")
    if not lengths:
        raise SystemExit(f"{args.csv}: no walk rows")

    nn = 1 << wires
    fact = math.factorial(nn)
    fracs = [rank / fact for rank in ranks]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], hspace=0.5, wspace=0.28)
    ax_len = fig.add_subplot(gs[:, 0])
    ax_gate = fig.add_subplot(gs[0, 1])
    ax_lag = fig.add_subplot(gs[1, 1])
    ax_perm = fig.add_subplot(gs[2, 1])
    plot_length(
        ax_len,
        lengths,
        f"Random walk length (n={wires}, {len(lengths)} walks)",
        wires,
    )
    plot_gates(ax_gate, last_gates, wires)
    plot_lags(ax_lag, lags)
    plot_perms(ax_perm, fracs, args.bins, nn, tick_nl=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    if args.out_lengths is not None:
        fig_l, ax = plt.subplots(figsize=(7, 4.5))
        plot_length(
            ax,
            lengths,
            f"Random walk length (n={wires})",
            wires,
        )
        fig_l.tight_layout()
        fig_l.savefig(args.out_lengths, dpi=150)
        print(f"Saved {args.out_lengths}")

    if args.out_gates is not None:
        fig_g, ax = plt.subplots(figsize=(7, 4.5))
        plot_gates(ax, last_gates, wires)
        fig_g.tight_layout()
        fig_g.savefig(args.out_gates, dpi=150)
        print(f"Saved {args.out_gates}")

    if args.out_lags is not None:
        fig_lag, ax = plt.subplots(figsize=(7, 4.5))
        plot_lags(ax, lags)
        fig_lag.tight_layout()
        fig_lag.savefig(args.out_lags, dpi=150)
        print(f"Saved {args.out_lags}")

    if args.out_perms is not None:
        fig_p, ax = plt.subplots(figsize=(7, 4.5))
        plot_perms(ax, fracs, args.bins, nn, tick_nl=False)
        fig_p.tight_layout()
        fig_p.savefig(args.out_perms, dpi=150)
        print(f"Saved {args.out_perms}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
