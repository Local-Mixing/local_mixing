#!/usr/bin/env python3
"""Measure affine-ridge prominence in an hmap_affine plate.

The primary verdict uses the 20--80% interior of both circuit trajectories so
that forced input/output ports do not dominate the result.  A prominence of
0.5/n is the error change corresponding to one perfectly reconstructed bit;
1/n corresponds to two bits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_plate(stem: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    meta_path = Path(f"{stem}.meta.json")
    bin_path = Path(f"{stem}.bin")
    with meta_path.open("r", encoding="utf-8") as source:
        meta = json.load(source)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    values = np.fromfile(bin_path, dtype="<f4")
    if values.size != rows * cols:
        raise ValueError(
            f"{bin_path}: expected {rows * cols} float32 cells, got {values.size}"
        )
    matrix = values.reshape(rows, cols).astype(np.float64)
    i_idx = np.asarray(meta["i_idx"], dtype=np.float64)
    j_idx = np.asarray(meta["j_idx"], dtype=np.float64)
    if i_idx.shape != (rows,) or j_idx.shape != (cols,):
        raise ValueError("metadata trajectory indices do not match plate dimensions")
    if not np.isfinite(matrix).all() or not np.isfinite(i_idx).all() or not np.isfinite(j_idx).all():
        raise ValueError("plate contains a non-finite value")
    return matrix, i_idx, j_idx, meta


def tied_ranks(values: np.ndarray) -> np.ndarray:
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


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3:
        return None
    rx, ry = tied_ranks(x), tied_ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(
    matrix: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    n: int,
    lo: float,
    hi: float,
) -> dict[str, object]:
    i_frac = i_idx / max(float(i_idx[-1]), 1.0)
    j_frac = j_idx / max(float(j_idx[-1]), 1.0)
    rows = np.flatnonzero((i_frac >= lo) & (i_frac <= hi))
    cols = np.flatnonzero((j_frac >= lo) & (j_frac <= hi))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("interior fraction produced an empty plate")

    core = matrix[np.ix_(rows, cols)]
    prominence = np.median(core, axis=1) - np.min(core, axis=1)
    ridge_local = np.argmin(core, axis=1)
    ridge_global = cols[ridge_local]
    one_bit = 0.5 / n
    two_bits = 1.0 / n
    informative = prominence >= one_bit
    rho = spearman(i_idx[rows][informative], j_idx[ridge_global][informative])

    return {
        "fraction_window": [lo, hi],
        "row_count": int(rows.size),
        "column_count": int(cols.size),
        "mean_h": float(np.mean(core)),
        "median_h": float(np.median(core)),
        "minimum_h": float(np.min(core)),
        "maximum_h": float(np.max(core)),
        "prominence_mean": float(np.mean(prominence)),
        "prominence_median": float(np.median(prominence)),
        "prominence_maximum": float(np.max(prominence)),
        "one_bit_threshold": one_bit,
        "two_bit_threshold": two_bits,
        "rows_at_least_one_bit": int(np.count_nonzero(prominence >= one_bit)),
        "rows_at_least_two_bits": int(np.count_nonzero(prominence >= two_bits)),
        "informative_ridge_spearman": rho,
        "ridge_columns": [int(value) for value in ridge_global],
        "row_prominence": [float(value) for value in prominence],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stems", nargs="+", type=Path)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--interior-lo", type=float, default=0.2)
    parser.add_argument("--interior-hi", type=float, default=0.8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n <= 0 or not 0 <= args.interior_lo < args.interior_hi <= 1:
        parser.error("invalid n or interior window")

    records: list[dict[str, object]] = []
    for stem in args.stems:
        matrix, i_idx, j_idx, meta = load_plate(stem)
        records.append(
            {
                "stem": str(stem),
                "rows": int(matrix.shape[0]),
                "columns": int(matrix.shape[1]),
                "full_plate_mean_h": float(np.mean(matrix)),
                "full_plate_minimum_h": float(np.min(matrix)),
                "metadata": meta,
                "interior": summarize(
                    matrix,
                    i_idx,
                    j_idx,
                    args.n,
                    args.interior_lo,
                    args.interior_hi,
                ),
            }
        )

    report = {
        "method": "degree-1 affine reconstruction error; 20--80% port-excluded core",
        "n": args.n,
        "plates": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")

    for record in records:
        core = record["interior"]
        print(
            f"{record['stem']}: core={core['row_count']}x{core['column_count']} "
            f"meanH={core['mean_h']:.6f} "
            f"prominence(mean/median/max)="
            f"{core['prominence_mean']:.6f}/"
            f"{core['prominence_median']:.6f}/"
            f"{core['prominence_maximum']:.6f} "
            f"rows>=1bit={core['rows_at_least_one_bit']} "
            f"rows>=2bits={core['rows_at_least_two_bits']} "
            f"rho={core['informative_ridge_spearman']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
