#!/usr/bin/env python3
"""Roll per-seed score reports up into one table per (arm, stage).

Reports the numbers an attacker actually cares about: how many of C's level-1
pairs land in the top-k of the blind ranking, and where the worst one lands.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean


def collect(pattern: str, statistic: str):
    rows = []
    for path in sorted(glob.glob(pattern)):
        report = json.loads(Path(path).read_text())
        block = report["by_statistic"][statistic]
        ranking = block["ranking"]
        positions = [index for index, row in enumerate(ranking, start=1)
                     if row["is_level1_pair"]]
        positives = len(positions)
        rows.append({
            "seed": Path(path).name,
            "gates": report["blind_artifact_gates"],
            "saturates_at": report["trace_span_full_rank_gate"],
            "candidates": report["candidate_pairs"],
            "positives": positives,
            "precision_at_k": block["precision_at_k"],
            "hits_in_top_k": sum(1 for p in positions if p <= positives),
            "worst_rank": max(positions) if positions else None,
            "median_rank": sorted(positions)[len(positions) // 2] if positions else None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", action="append", required=True,
                        help="label=glob (repeatable)")
    parser.add_argument("--statistic", default="entry_gate")
    args = parser.parse_args()

    print(f"statistic: {args.statistic}\n")
    header = (f"{'arm':<26}{'seeds':>6}{'cands':>7}{'pos':>5}"
              f"{'prec@k':>9}{'med rank':>10}{'worst rank':>12}")
    print(header)
    print("-" * len(header))
    for spec in args.pattern:
        label, pattern = spec.split("=", 1)
        rows = collect(pattern, args.statistic)
        if not rows:
            print(f"{label:<26}{'(no reports)':>6}")
            continue
        print(f"{label:<26}{len(rows):>6}{rows[0]['candidates']:>7}"
              f"{mean(r['positives'] for r in rows):>5.1f}"
              f"{mean(r['precision_at_k'] for r in rows):>9.3f}"
              f"{mean(r['median_rank'] for r in rows):>10.1f}"
              f"{mean(r['worst_rank'] for r in rows):>12.1f}")


if __name__ == "__main__":
    main()
