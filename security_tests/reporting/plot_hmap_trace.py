#!/usr/bin/env python3
"""Port-aware analysis and rendering for affine heatmap plates.

The input contract is the standard ``<stem>.bin`` row-major little-endian f32
matrix plus ``<stem>.meta.json`` with ``rows``, ``cols``, ``i_idx`` and
``j_idx``.  Legacy plates default to ``window`` mode.  New trace producers can
set ``"mode": "cumulative"`` (``trace_mode`` is accepted as an alias).

Window plates contain an independently fitted G snapshot in each column, so a
localized low-H ridge is meaningful.  Cumulative plates contain nested feature
sets: once a relation is available it should remain available.  Their signal is
therefore an earliest-recovery *frontier*, not an argmin ridge.  Treating the
right-hand recovery plateau as a conventional ridge produces false chronology.

Examples:
  plot_hmap_trace.py run/plate --out run/plate.png
  plot_hmap_trace.py a b --titles 'window;cumulative' --out compare.png
  plot_hmap_trace.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


MODE_WINDOW = "window"
MODE_CUMULATIVE = "cumulative"


@dataclass(frozen=True)
class Plate:
    stem: Path
    h: np.ndarray
    i_idx: np.ndarray
    j_idx: np.ndarray
    meta: dict[str, Any]
    mode: str


def _normalise_axis(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("trajectory indices must be non-empty vectors")
    if not np.isfinite(values).all():
        raise ValueError("trajectory indices contain non-finite values")
    if np.any(np.diff(values) < 0):
        raise ValueError("trajectory indices must be nondecreasing")
    span = float(values[-1] - values[0])
    if span <= 0.0:
        return np.zeros(values.size, dtype=np.float64)
    return (values - values[0]) / span


def _metadata_mode(meta: dict[str, Any], override: str = "auto") -> str:
    if override != "auto":
        return override
    if bool(meta.get("cumulative", False)):
        return MODE_CUMULATIVE
    candidates: list[Any] = [
        meta.get("trace_mode"),
        meta.get("mode"),
        meta.get("attack_mode"),
        meta.get("feature_mode"),
    ]
    analysis = meta.get("analysis")
    if isinstance(analysis, dict):
        candidates.extend((analysis.get("trace_mode"), analysis.get("mode")))
    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip().lower().replace("_", "-")
        if value in {
            "cumulative",
            "cumulative-trace",
            "whole-trace",
            "gate-delta",
            "checkpoint-state",
        }:
            return MODE_CUMULATIVE
        if value in {"window", "snapshot", "isolated", "per-snapshot"}:
            return MODE_WINDOW
    return MODE_WINDOW


def load_plate(stem: str | Path, mode_override: str = "auto") -> Plate:
    stem = Path(stem)
    meta_path = Path(f"{stem}.meta.json")
    bin_path = Path(f"{stem}.bin")
    with meta_path.open("r", encoding="utf-8") as source:
        meta = json.load(source)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{meta_path}: rows and cols must be positive")
    values = np.fromfile(bin_path, dtype="<f4")
    if values.size != rows * cols:
        raise ValueError(
            f"{bin_path}: expected {rows * cols} float32 cells, got {values.size}"
        )
    h = values.reshape(rows, cols).astype(np.float64)
    if not np.isfinite(h).all():
        raise ValueError(f"{bin_path}: matrix contains non-finite values")
    i_idx = np.asarray(meta["i_idx"], dtype=np.float64)
    j_idx = np.asarray(meta["j_idx"], dtype=np.float64)
    if i_idx.shape != (rows,) or j_idx.shape != (cols,):
        raise ValueError(f"{meta_path}: index vector lengths do not match matrix")
    _normalise_axis(i_idx)
    _normalise_axis(j_idx)
    return Plate(stem, h, i_idx, j_idx, meta, _metadata_mode(meta, mode_override))


def _target_count(meta: dict[str, Any]) -> int:
    for key in ("target_count", "n_targets", "target_bits_count"):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
    target_bits = meta.get("target_bits")
    if isinstance(target_bits, list) and target_bits:
        return len(target_bits)
    if isinstance(target_bits, int) and target_bits > 0:
        return target_bits
    value = int(meta.get("n", 0))
    if value <= 0:
        raise ValueError("metadata needs positive n or target_count")
    return value


def _tied_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def tied_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or ya.shape != xa.shape:
        return None
    rx, ry = _tied_ranks(xa), _tied_ranks(ya)
    if np.std(rx) <= 0.0 or np.std(ry) <= 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def chronology_null(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float | None, float | None, float | None, int]:
    """Tie-aware rho and a serial-structure-preserving circular-shift null."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    rho = tied_spearman(xa, ya)
    if rho is None or xa.size < 4:
        return rho, None, None, 0
    null = [tied_spearman(xa, np.roll(ya, shift)) for shift in range(1, xa.size)]
    vals = np.asarray([value for value in null if value is not None], dtype=np.float64)
    if vals.size == 0:
        return rho, None, None, 0
    sigma = float(np.std(vals))
    z = None if sigma <= 1e-12 else float((rho - float(np.mean(vals))) / sigma)
    p = float((1 + np.count_nonzero(vals >= rho)) / (vals.size + 1))
    return rho, z, p, int(vals.size)


def _core_indices(frac: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((frac >= lo) & (frac <= hi))


def _json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def analyse_window(
    plate: Plate, core_lo: float = 0.2, core_hi: float = 0.8, band_frac: float = 0.06
) -> dict[str, Any]:
    i_frac, j_frac = _normalise_axis(plate.i_idx), _normalise_axis(plate.j_idx)
    rows = _core_indices(i_frac, core_lo, core_hi)
    cols = _core_indices(j_frac, core_lo, core_hi)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("port-excluded core is empty")
    core = plate.h[np.ix_(rows, cols)]
    row_median = np.median(core, axis=1)
    row_minimum = np.min(core, axis=1)
    prominence = row_median - row_minimum
    target_count = _target_count(plate.meta)
    one_bit, two_bits = 0.5 / target_count, 1.0 / target_count
    informative = prominence >= one_bit - 1e-12

    # Average tied minima rather than choosing the first cell in a flat row.
    ridge_local = np.empty(rows.size, dtype=np.float64)
    for ri, row in enumerate(core):
        tied = np.flatnonzero(np.isclose(row, row_minimum[ri], rtol=0.0, atol=1e-7))
        ridge_local[ri] = float(np.mean(tied))
    ridge_global = np.interp(ridge_local, np.arange(cols.size), cols.astype(float))
    rho, z, p, null_count = chronology_null(
        plate.i_idx[rows][informative],
        np.interp(ridge_global[informative], np.arange(plate.j_idx.size), plate.j_idx),
    )

    contrast: float | None = None
    if np.count_nonzero(informative) > 0 and cols.size >= 3:
        selected = core[informative]
        centers = ridge_local[informative]
        width = max(1.0, band_frac * cols.size)
        band = np.abs(np.arange(cols.size)[None, :] - centers[:, None]) <= width
        on, off = selected[band], selected[~band]
        if on.size and off.size:
            pooled = math.sqrt(0.5 * (float(np.var(on)) + float(np.var(off))))
            if pooled > 1e-12:
                contrast = (float(np.mean(off)) - float(np.mean(on))) / pooled

    return {
        "mode": MODE_WINDOW,
        "core_fraction": [core_lo, core_hi],
        "core_rows": int(rows.size),
        "core_columns": int(cols.size),
        "core_mean_h": float(np.mean(core)),
        "core_minimum_h": float(np.min(core)),
        "depth_mean": float(np.mean(prominence)),
        "depth_median": float(np.median(prominence)),
        "depth_maximum": float(np.max(prominence)),
        "one_bit_threshold": one_bit,
        "two_bit_threshold": two_bits,
        "rows_at_least_one_bit": int(np.count_nonzero(prominence >= one_bit - 1e-12)),
        "rows_at_least_two_bits": int(np.count_nonzero(prominence >= two_bits - 1e-12)),
        "informative_rho": _json_number(rho),
        "chronology_z": _json_number(z),
        "chronology_p": _json_number(p),
        "chronology_null_count": null_count,
        "contrast": _json_number(contrast),
        "row_indices": [int(value) for value in rows],
        "ridge_columns": [float(value) for value in ridge_global],
        "informative_rows": [bool(value) for value in informative],
    }


def _first_crossing(gain: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(gain >= threshold - 1e-12)
    return None if hits.size == 0 else int(hits[0])


def _frontier_summary(
    plate: Plate,
    rows: np.ndarray,
    j_frac: np.ndarray,
    gain: np.ndarray,
    threshold: float,
    core_lo: float,
    core_hi: float,
) -> dict[str, Any]:
    counts = {"early": 0, "interior": 0, "late": 0, "censored": 0}
    columns: list[int | None] = []
    informative_rows: list[int] = []
    informative_cols: list[int] = []
    for local_row, global_row in enumerate(rows):
        crossing = _first_crossing(gain[local_row], threshold)
        columns.append(crossing)
        if crossing is None:
            counts["censored"] += 1
        elif j_frac[crossing] < core_lo:
            counts["early"] += 1
        elif j_frac[crossing] <= core_hi:
            counts["interior"] += 1
            informative_rows.append(int(global_row))
            informative_cols.append(crossing)
        else:
            counts["late"] += 1
    rho, z, p, null_count = chronology_null(
        plate.i_idx[informative_rows] if informative_rows else [],
        plate.j_idx[informative_cols] if informative_cols else [],
    )
    return {
        "threshold": threshold,
        "counts": counts,
        "coverage": counts["interior"] / max(1, len(rows)),
        "first_crossing_columns": columns,
        "informative_row_indices": informative_rows,
        "informative_column_indices": informative_cols,
        "rho": _json_number(rho),
        "chronology_z": _json_number(z),
        "chronology_p": _json_number(p),
        "chronology_null_count": null_count,
    }


def analyse_cumulative(
    plate: Plate, core_lo: float = 0.2, core_hi: float = 0.8, monotone_tol: float = 1e-6
) -> dict[str, Any]:
    i_frac, j_frac = _normalise_axis(plate.i_idx), _normalise_axis(plate.j_idx)
    rows = _core_indices(i_frac, core_lo, core_hi)
    cols = _core_indices(j_frac, core_lo, core_hi)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("port-excluded core is empty")
    selected = plate.h[rows]
    # Relative to G_0: later-column cropping cannot remove an input relation
    # already present in every cumulative feature set.
    gain = selected[:, :1] - selected
    target_count = _target_count(plate.meta)
    one_bit, two_bits = 0.5 / target_count, 1.0 / target_count
    diffs = np.diff(selected, axis=1)
    violations = diffs > monotone_tol
    core_end = int(cols[-1])
    core_gain = gain[:, core_end]
    one = _frontier_summary(
        plate, rows, j_frac, gain, one_bit, core_lo, core_hi
    )
    two = _frontier_summary(
        plate, rows, j_frac, gain, two_bits, core_lo, core_hi
    )
    return {
        "mode": MODE_CUMULATIVE,
        "core_fraction": [core_lo, core_hi],
        "core_rows": int(rows.size),
        "core_columns": int(cols.size),
        "baseline": "first G column",
        "core_mean_h": float(np.mean(plate.h[np.ix_(rows, cols)])),
        "core_end_gain_mean": float(np.mean(core_gain)),
        "core_end_gain_median": float(np.median(core_gain)),
        "core_end_gain_maximum": float(np.max(core_gain)),
        "final_gain_mean": float(np.mean(gain[:, -1])),
        "monotonicity_tolerance": monotone_tol,
        "monotonicity_violation_count": int(np.count_nonzero(violations)),
        "monotonicity_rows_violating": int(np.count_nonzero(np.any(violations, axis=1))),
        "monotonicity_max_increase": float(np.max(diffs)) if diffs.size else 0.0,
        "one_bit_frontier": one,
        "two_bit_frontier": two,
        "row_indices": [int(value) for value in rows],
    }


def analyse_plate(
    plate: Plate, core_lo: float = 0.2, core_hi: float = 0.8, band_frac: float = 0.06
) -> dict[str, Any]:
    if not 0.0 <= core_lo < core_hi <= 1.0:
        raise ValueError("core window must satisfy 0 <= lo < hi <= 1")
    if plate.mode == MODE_CUMULATIVE:
        return analyse_cumulative(plate, core_lo, core_hi)
    return analyse_window(plate, core_lo, core_hi, band_frac)


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def concise_metrics(stem: Path, metrics: dict[str, Any]) -> str:
    if metrics["mode"] == MODE_WINDOW:
        return (
            f"{stem}: mode=window core={metrics['core_rows']}x{metrics['core_columns']} "
            f"H={metrics['core_mean_h']:.6f} "
            f"depth(mean/med/max)={metrics['depth_mean']:.6f}/"
            f"{metrics['depth_median']:.6f}/{metrics['depth_maximum']:.6f} "
            f"rows>=1bit={metrics['rows_at_least_one_bit']} "
            f"rho={_fmt(metrics['informative_rho'])} p={_fmt(metrics['chronology_p'])}"
        )
    one = metrics["one_bit_frontier"]
    counts = one["counts"]
    return (
        f"{stem}: mode=cumulative core={metrics['core_rows']}x{metrics['core_columns']} "
        f"gain@80%={metrics['core_end_gain_mean']:.6f} "
        f"1bit(e/i/l/c)={counts['early']}/{counts['interior']}/"
        f"{counts['late']}/{counts['censored']} "
        f"rho={_fmt(one['rho'])} p={_fmt(one['chronology_p'])} "
        f"monotone_violations={metrics['monotonicity_violation_count']}"
    )


def _frontier_points(
    plate: Plate, metrics: dict[str, Any], key: str
) -> tuple[np.ndarray, np.ndarray]:
    frontier = metrics[key]
    row_indices = frontier["informative_row_indices"]
    col_indices = frontier["informative_column_indices"]
    if not row_indices:
        return np.empty(0), np.empty(0)
    i_frac, j_frac = _normalise_axis(plate.i_idx), _normalise_axis(plate.j_idx)
    return j_frac[col_indices], i_frac[row_indices]


def render(
    plates: Sequence[Plate],
    metrics: Sequence[dict[str, Any]],
    titles: Sequence[str],
    out: Path,
    field: str = "auto",
    core_lo: float = 0.2,
    core_hi: float = 0.8,
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 170,
) -> None:
    shown: list[np.ndarray] = []
    fields: list[str] = []
    for plate in plates:
        selected = field
        if selected == "auto":
            selected = "gain" if plate.mode == MODE_CUMULATIVE else "h"
        fields.append(selected)
        shown.append(plate.h[:, :1] - plate.h if selected == "gain" else plate.h)

    # A shared scale is meaningful only when panels show the same quantity.
    common_field = len(set(fields)) == 1
    if common_field:
        all_values = np.concatenate([array.ravel() for array in shown])
        auto_vmin = (
            min(0.0, float(np.percentile(all_values, 1)))
            if fields[0] == "gain"
            else float(np.percentile(all_values, 1))
        )
        auto_vmax = (
            max(float(np.percentile(all_values, 99)), 1e-12)
            if fields[0] == "gain"
            else 0.5
        )
    else:
        auto_vmin, auto_vmax = 0.0, 0.5

    fig, axes = plt.subplots(
        1,
        len(plates),
        figsize=(6.2 * len(plates), 4.6),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )
    images = []
    for panel, (axis, plate, result, title, values, selected) in enumerate(
        zip(axes[0], plates, metrics, titles, shown, fields)
    ):
        if common_field:
            panel_vmin = auto_vmin if vmin is None else vmin
            panel_vmax = auto_vmax if vmax is None else vmax
        else:
            panel_vmin = (
                min(0.0, float(np.min(values)))
                if selected == "gain"
                else float(np.min(values))
            ) if vmin is None else vmin
            panel_vmax = (
                max(float(np.max(values)), 1e-12) if selected == "gain" else 0.5
            ) if vmax is None else vmax
        cmap = "YlOrRd" if selected == "gain" else "RdYlBu"
        image = axis.imshow(
            values,
            origin="upper",
            extent=[0, 1, 1, 0],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
        )
        images.append(image)

        # Shade all four port zones and outline the analysis core.
        shade = dict(color="0.45", alpha=0.12, linewidth=0)
        axis.axvspan(0.0, core_lo, **shade)
        axis.axvspan(core_hi, 1.0, **shade)
        axis.axhspan(0.0, core_lo, **shade)
        axis.axhspan(core_hi, 1.0, **shade)
        axis.add_patch(
            Rectangle(
                (core_lo, core_lo),
                core_hi - core_lo,
                core_hi - core_lo,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle="--",
            )
        )

        i_frac, j_frac = _normalise_axis(plate.i_idx), _normalise_axis(plate.j_idx)
        if plate.mode == MODE_WINDOW:
            row_indices = np.asarray(result["row_indices"], dtype=int)
            ridge_columns = np.asarray(result["ridge_columns"], dtype=float)
            informative = np.asarray(result["informative_rows"], dtype=bool)
            if np.any(informative):
                xf = np.interp(ridge_columns[informative], np.arange(j_frac.size), j_frac)
                axis.scatter(
                    xf,
                    i_frac[row_indices[informative]],
                    s=18,
                    color="black",
                    marker="o",
                    linewidths=0,
                    label="informative tied minimum",
                )
        else:
            x1, y1 = _frontier_points(plate, result, "one_bit_frontier")
            x2, y2 = _frontier_points(plate, result, "two_bit_frontier")
            if x1.size:
                axis.scatter(x1, y1, s=22, color="black", marker="o", linewidths=0, label="1-bit frontier")
            if x2.size:
                axis.scatter(x2, y2, s=25, color="white", edgecolors="black", marker="x", linewidths=0.9, label="2-bit frontier")

        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper left", fontsize=7.5, framealpha=0.8)
        if result["mode"] == MODE_WINDOW:
            subtitle = (
                f"core depth={result['depth_median']:.4f} · "
                f"rho={_fmt(result['informative_rho'], 3)}"
            )
        else:
            subtitle = (
                f"core gain={result['core_end_gain_mean']:.4f} · "
                f"rho={_fmt(result['one_bit_frontier']['rho'], 3)} · "
                f"viol={result['monotonicity_violation_count']}"
            )
        axis.set_title(f"{title}\n{plate.mode}: {subtitle}", fontsize=10)
        axis.set_xlabel("fraction of G trajectory")
        if panel == 0:
            axis.set_ylabel("fraction of C trajectory")

    if common_field:
        label = "cumulative gain  H(G₀) − H(prefix)" if fields[0] == "gain" else "H  (0 = leak, 0.5 = hidden)"
        fig.colorbar(images[0], ax=axes, label=label, shrink=0.85)
    else:
        for axis, image, selected in zip(axes[0], images, fields):
            label = "cumulative gain" if selected == "gain" else "H"
            fig.colorbar(image, ax=axis, label=label, shrink=0.75)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _self_test() -> None:
    i_idx = np.arange(9, dtype=np.float64)
    j_idx = np.arange(11, dtype=np.float64)
    meta = {"rows": 9, "cols": 11, "n": 8, "i_idx": i_idx.tolist(), "j_idx": j_idx.tolist()}

    flat = Plate(Path("flat"), np.full((9, 11), 0.5), i_idx, j_idx, meta, MODE_WINDOW)
    flat_result = analyse_window(flat)
    assert flat_result["rows_at_least_one_bit"] == 0
    assert flat_result["informative_rho"] is None

    diagonal = np.full((9, 11), 0.5)
    for row in range(9):
        col = min(10, row + 1)
        diagonal[row, col] = 0.25
    diag_result = analyse_window(Plate(Path("diag"), diagonal, i_idx, j_idx, meta, MODE_WINDOW))
    assert diag_result["rows_at_least_one_bit"] > 0
    assert diag_result["informative_rho"] is not None
    assert diag_result["informative_rho"] > 0.9

    cumulative = np.full((9, 11), 0.5)
    for row in range(9):
        crossing = min(10, row + 1)
        cumulative[row, crossing:] = 0.25
    cum_result = analyse_cumulative(
        Plate(Path("cumulative"), cumulative, i_idx, j_idx, {**meta, "mode": "cumulative"}, MODE_CUMULATIVE)
    )
    assert cum_result["monotonicity_violation_count"] == 0
    assert cum_result["one_bit_frontier"]["counts"]["interior"] > 0
    assert cum_result["one_bit_frontier"]["rho"] is not None
    assert cum_result["one_bit_frontier"]["rho"] > 0.9

    broken = cumulative.copy()
    broken[4, 7] = 0.5
    broken_result = analyse_cumulative(
        Plate(Path("broken"), broken, i_idx, j_idx, {**meta, "mode": "cumulative"}, MODE_CUMULATIVE)
    )
    assert broken_result["monotonicity_violation_count"] > 0
    print("plot_hmap_trace self-test: PASS")


def _titles(stems: Sequence[Path], specification: str) -> list[str]:
    if not specification:
        return [stem.name for stem in stems]
    values = specification.split(";")
    if len(values) != len(stems):
        raise ValueError("--titles count must match number of stems")
    return values


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metrics", type=Path, help="JSON path (default: <out>.metrics.json)")
    parser.add_argument("--no-metrics", action="store_true", help="do not write JSON metrics")
    parser.add_argument("--titles", default="", help="semicolon-separated panel titles")
    parser.add_argument("--mode", choices=("auto", MODE_WINDOW, MODE_CUMULATIVE), default="auto")
    parser.add_argument("--field", choices=("auto", "h", "gain"), default="auto")
    parser.add_argument("--core-lo", type=float, default=0.2)
    parser.add_argument("--core-hi", type=float, default=0.8)
    parser.add_argument("--band-frac", type=float, default=0.06)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        _self_test()
        return 0
    if not args.stems or args.out is None:
        parser.error("provide at least one stem and --out, or use --self-test")
    if args.no_metrics and args.metrics is not None:
        parser.error("--metrics and --no-metrics are mutually exclusive")
    if not 0.0 <= args.core_lo < args.core_hi <= 1.0:
        parser.error("core window must satisfy 0 <= lo < hi <= 1")
    if args.band_frac <= 0.0 or args.dpi <= 0:
        parser.error("--band-frac and --dpi must be positive")

    try:
        titles = _titles(args.stems, args.titles)
        plates = [load_plate(stem, args.mode) for stem in args.stems]
        results = [
            analyse_plate(plate, args.core_lo, args.core_hi, args.band_frac)
            for plate in plates
        ]
        render(
            plates,
            results,
            titles,
            args.out,
            args.field,
            args.core_lo,
            args.core_hi,
            args.vmin,
            args.vmax,
            args.dpi,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    for plate, result in zip(plates, results):
        print(concise_metrics(plate.stem, result))
    print(f"wrote {args.out}")

    if not args.no_metrics:
        metrics_path = args.metrics or args.out.with_suffix(".metrics.json")
        report = {
            "schema": "hmap-trace-metrics/v1",
            "core_fraction": [args.core_lo, args.core_hi],
            "panels": [
                {
                    "stem": str(plate.stem),
                    "title": title,
                    "metadata_mode": plate.mode,
                    "metrics": result,
                }
                for plate, title, result in zip(plates, titles, results)
            ],
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
            destination.write("\n")
        print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
