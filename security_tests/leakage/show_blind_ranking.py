#!/usr/bin/env python3
"""Print a compact view of one or more blind-recovery score reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def show(path: Path, statistic: str, top: int) -> None:
    report = json.loads(path.read_text())
    block = report["by_statistic"][statistic]
    print("=" * 72)
    print(f"{report.get('label') or path.name}")
    print(f"  artifact  {Path(report['blind_artifact']).name}  "
          f"gates={report['blind_artifact_gates']}  "
          f"saturates_at={report['trace_span_full_rank_gate']}")
    print(f"  source    {Path(report['source_circuit']).name}  "
          f"gates={report['source_gates']}  n={report['n']}")
    print(f"  level-1 gates {len(report['level1_gates'])} -> "
          f"{report['level1_pair_count']} distinct pairs "
          f"of {report['candidate_pairs']} candidates")
    print(f"  [{statistic}] AUC={block['auc_positive_vs_null']}  "
          f"precision@k={block['precision_at_k']}")
    print(f"    positives {block['positives']}")
    print(f"    nulls     {block['nulls']}")
    print(f"  top {top} by {statistic}:")
    for rank, row in enumerate(block["ranking"][:top], start=1):
        flag = "  <-- LEVEL-1 GATE OF C" if row["is_level1_pair"] else ""
        print(f"    {rank:3d}. pair {row['pair']}  {statistic}={row['value']}{flag}")
    missed = [row for row in block["ranking"][top:] if row["is_level1_pair"]]
    if missed:
        print(f"  level-1 pairs ranked below {top}:")
        for row in missed:
            position = block["ranking"].index(row) + 1
            print(f"    {position:3d}. pair {row['pair']}  {statistic}={row['value']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--statistic", default="entry_gate")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    for report in args.reports:
        show(Path(report), args.statistic, args.top)


if __name__ == "__main__":
    main()
