#!/usr/bin/env python3
"""Mac-side stats sweep for circuits/fmix_bench_20260715/.

Runs fmix_stats over every raw benchmark circuit (with its .origins sidecar)
and over every fcompress output (.fc.txt) present, parses the fcompress logs,
and writes:
  reports/fmix_bench_20260715/bench_stats.tsv    - one row per (A, config, dose)
  reports/fmix_bench_20260715/bench_summary.md   - config x dose tables, mean +/- range over A1..A3
  reports/fmix_bench_20260715/stats_raw/         - full fmix_stats output per circuit

Idempotent: re-run any time; fc columns fill in as the fcompress batch lands.
Cached fmix_stats outputs are reused unless the circuit file is newer.
"""

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/rancanetti/Documents/local_mixing")
BENCH = REPO / "circuits/fmix_bench_20260715"
FMIX_STATS = REPO / "target/release/fmix_stats"
FC_LOGDIR = Path(
    "/private/tmp/claude-501/-Users-rancanetti-Documents-local-mixing/"
    "683a0809-86f7-475a-809f-2d529990c4ef/scratchpad/fc_logs"
)
OUTDIR = REPO / "reports/fmix_bench_20260715"
STATS_DIR = OUTDIR / "stats_raw"

CONFIGS = ["defaults", "temp2500", "temp10000", "swap10", "trans10", "no_swaptrans", "no_insert"]
DOSES = ["mv2000000", "mv3000000", "final", "mv6000000", "mv8000000", "final10M"]
AS_ = ["A1", "A2", "A3"]


def circuit_paths(a: str, config: str, dose: str):
    """(circuit, origins, fc_output, fc_log) paths for one cell."""
    if dose in ("final", "final10M"):
        raw = BENCH / f"{a}_{config}.{dose}.txt"
        origins = BENCH / f"{a}_{config}.{dose}.origins.txt"
        fc = BENCH / f"{a}_{config}.{dose}.fc.txt"
        fclog = FC_LOGDIR / f"{a}_{config}.{dose}.log"
    else:
        raw = BENCH / f"{a}_{config}.snapshot.txt.{dose}"
        origins = BENCH / f"{a}_{config}.snapshot.txt.{dose}.origins"
        fc = BENCH / f"{a}_{config}.snapshot.{dose}.fc.txt"
        fclog = FC_LOGDIR / f"{a}_{config}.snapshot.{dose}.log"
    return raw, origins, fc, fclog


def run_fmix_stats(circuit: Path, origins: Path | None, cache_name: str) -> str | None:
    """Run fmix_stats (cached) and return its stdout, or None on failure."""
    cache = STATS_DIR / f"{cache_name}.stats.txt"
    if cache.exists() and cache.stat().st_mtime >= circuit.stat().st_mtime and cache.stat().st_size > 0:
        return cache.read_text()
    cmd = [str(FMIX_STATS), "--input", str(circuit)]
    if origins is not None and origins.exists():
        cmd += ["--origins", str(origins)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: fmix_stats {circuit.name}", file=sys.stderr)
        return None
    if out.returncode != 0:
        print(f"  FAIL: fmix_stats {circuit.name}: {out.stderr.strip()[:200]}", file=sys.stderr)
        return None
    text = out.stdout
    cache.write_text(text)
    return text


LINE_TAG = re.compile(r"^\[fstats\] (\w+)")
KV = re.compile(r"(\w+)=([-\d.eE]+)")


def parse_fstats(text: str) -> dict:
    """Flatten fmix_stats output to {linetag_key: float}. Histograms are skipped
    (they live in the cached raw output); span lines are keyed by window size."""
    vals = {}
    for line in text.splitlines():
        m = LINE_TAG.match(line)
        if not m:
            continue
        tag = m.group(1)
        body = line[m.end():]
        if tag == "span":
            wm = re.search(r"w=(\d+)", body)
            tag = f"span{wm.group(1)}" if wm else "span"
            body = body[wm.end():] if wm else body
        elif tag == "file":
            body = line  # gates/wires/comp sit on the header line
        for k, v in KV.findall(body.split("hist[")[0]):
            try:
                vals[f"{tag}_{k}"] = float(v)
            except ValueError:
                pass
        # the header line has no tag prefix in its k=v pairs
        if line.startswith("[fstats] file="):
            for k, v in KV.findall(line.split("hist[")[0]):
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
    return vals


FC_DONE = re.compile(
    r"done in ([\d.]+)s .*: gates (\d+) -> (\d+) \(([\d.]+)%\), lits (\d+) -> (\d+) \(([\d.]+)%\)"
)


def parse_fc_log(fclog: Path) -> dict:
    if not fclog.exists():
        return {}
    m = FC_DONE.search(fclog.read_text())
    if not m:
        return {}
    return {
        "fc_time_s": float(m.group(1)),
        "fc_gates_pct": float(m.group(4)),
        "fc_lits_pct": float(m.group(7)),
    }


# Scalar columns pulled into the TSV (raw circuit), in output order.
RAW_COLS = [
    ("gates", "gates"),
    ("comp", "fossils"),
    ("width_mean", "width_mean"),
    ("width_neg_frac", "neg_frac"),
    ("fanout_mean", "fanout_mean"),
    ("fanout_zero_frac", "fanout_zero_frac"),
    ("leeway_mean", "leeway_mean"),
    ("leeway_median", "leeway_median"),
    ("leeway_wedged_lt25", "wedged_lt25"),
    ("origins_real_frac", "origins_real_frac"),
    ("origins_disp", "disp"),
    ("origins_diffusion", "diffusion"),
    ("origins_uniform", "uniform"),
    ("origins_adj_autocorr", "adj_autocorr"),
    ("origins_owin32", "owin32"),
    ("spread_gates_single_frac", "spread_single_frac"),
    ("spread_gates_p50", "spread_p50"),
    ("spread_gates_p95", "spread_p95"),
    ("spread_gates_frac_lt_ref", "spread_frac_lt_ref"),
    ("wires_target_H", "target_H"),
    ("wires_pair_H", "pair_H"),
    ("span32_mean", "span32_mean"),
    ("span256_mean", "span256_mean"),
]
# Columns from the fc circuit's own fmix_stats run.
FC_COLS = [
    ("gates", "fc_gates"),
    # NOTE: in fcompress output comp=1 is an absorbed-parity gate (t ^= NOT(...)),
    # NOT a fossil — fossil counts are only meaningful on the raw fmix files.
    ("comp", "fc_comp_parity"),
    ("width_mean", "fc_width_mean"),
    ("leeway_median", "fc_leeway_median"),
    ("span256_mean", "fc_span256_mean"),
]
FC_LOG_COLS = ["fc_gates_pct", "fc_lits_pct", "fc_time_s"]

# Headline subset for the summary tables.
SUMMARY_COLS = [
    "gates", "fc_gates", "fc_gates_pct", "fossils",
    "disp", "diffusion", "adj_autocorr", "owin32",
    "leeway_median", "spread_p50", "span256_mean", "width_mean",
]


def fmt(v, col):
    if v is None:
        return ""
    if col in ("gates", "fc_gates", "fossils", "fc_comp_parity", "leeway_median",
               "fc_leeway_median", "spread_p50", "spread_p95"):
        return str(int(v))
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.4g}"


def main():
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        for a in AS_:
            for dose in DOSES:
                raw, origins, fc, fclog = circuit_paths(a, config, dose)
                if not raw.exists():
                    print(f"  MISSING raw: {raw.name}", file=sys.stderr)
                    continue
                row = {"A": a, "config": config, "dose": dose}
                text = run_fmix_stats(raw, origins, raw.name)
                if text:
                    vals = parse_fstats(text)
                    for src, dst in RAW_COLS:
                        row[dst] = vals.get(src)
                if fc.exists() and fc.stat().st_size > 0:
                    fctext = run_fmix_stats(fc, None, fc.name)
                    if fctext:
                        fcvals = parse_fstats(fctext)
                        for src, dst in FC_COLS:
                            row[dst] = fcvals.get(src)
                    row.update(parse_fc_log(fclog))
                    if row.get("fc_gates_pct") is None and row.get("fc_gates") and row.get("gates"):
                        row["fc_gates_pct"] = 100.0 * row["fc_gates"] / row["gates"]
                rows.append(row)
                done = "fc+raw" if "fc_gates" in row else "raw"
                print(f"{a}_{config}.{dose}: {done}")

    all_cols = (["A", "config", "dose"] + [d for _, d in RAW_COLS]
                + [d for _, d in FC_COLS] + FC_LOG_COLS)
    tsv = OUTDIR / "bench_stats.tsv"
    with tsv.open("w") as f:
        f.write("\t".join(all_cols) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(c), c) if not isinstance(row.get(c), str)
                              else row[c] for c in all_cols) + "\n")

    # Summary: per (config, dose), mean and min..max range over the A replicates.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["dose"])].append(row)
    md = ["# fmix_bench_20260715 — Mac-side stats summary", "",
          "Mean over A1/A2/A3 with (min..max) range. fc_* columns appear as the",
          "fcompress batch delivers; blank = not yet compressed.", ""]
    for dose in DOSES:
        md.append(f"## dose = {dose}")
        md.append("")
        md.append("| config | " + " | ".join(SUMMARY_COLS) + " |")
        md.append("|" + "---|" * (len(SUMMARY_COLS) + 1))
        for config in CONFIGS:
            cells = [config]
            for col in SUMMARY_COLS:
                vs = [r[col] for r in groups.get((config, dose), []) if r.get(col) is not None]
                if not vs:
                    cells.append("")
                elif len(vs) == 1:
                    cells.append(fmt(vs[0], col))
                else:
                    mean = sum(vs) / len(vs)
                    cells.append(f"{fmt(mean, col)} ({fmt(min(vs), col)}..{fmt(max(vs), col)})")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    (OUTDIR / "bench_summary.md").write_text("\n".join(md) + "\n")
    n_fc = sum(1 for r in rows if r.get("fc_gates") is not None)
    print(f"\nwrote {tsv} ({len(rows)} rows, {n_fc} with fc) and bench_summary.md")


if __name__ == "__main__":
    main()
