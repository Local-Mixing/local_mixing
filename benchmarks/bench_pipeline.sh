#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LOCAL_MIXING_BIN:-"$ROOT/target/release/local_mixing_bin"}"
REPORT="${BENCH_REPORT:-"$ROOT/reports/bench_baseline.json"}"
RUNS="${BENCH_RUNS:-10}"
WARMUP="${BENCH_WARMUP:-3}"
CANON_N="${BENCH_CANON_N:-8}"
WORK="$ROOT/reports/bench_work"
RAW="$WORK/raw.tsv"

if [[ ! -x "$BIN" || ! -x "$ROOT/target/release/bench_canon4" || ! -x "$ROOT/target/release/bench_polycanon" ]]; then
  cargo build --manifest-path "$ROOT/Cargo.toml" --locked --release --bin local_mixing_bin --features benchmark-tools --bin bench_canon4 --bin bench_polycanon
fi

mkdir -p "$WORK" "$(dirname "$REPORT")"
SMALL="$WORK/small_n6.txt"
MEDIUM="$WORK/medium_n12.txt"
printf '012;345;024;135;250;431;\n' > "$SMALL"
printf '012;345;678;9ab;036;147;258;39a;4ab;50a;61b;72a;84b;\n' > "$MEDIUM"

CASES=(evaluate_small evaluate_medium canon4 polycanon)
if [[ -f "$ROOT/rantestn128m800/n128m800_source.txt" ]]; then
  CASES+=(evaluate_headline_n128m800)
fi

printf 'name\trun\tseconds\tmax_rss_kb\tstatus\n' > "$RAW"

run_case() {
  case "$1" in
    evaluate_small)
      "$BIN" circuit evaluate -n 6 -s "$SMALL" -x 0x2a
      ;;
    evaluate_medium)
      "$BIN" circuit evaluate -n 12 -s "$MEDIUM" -x 0xace
      ;;
    evaluate_headline_n128m800)
      "$BIN" circuit evaluate \
        -n 128 \
        -s "$ROOT/rantestn128m800/n128m800_source.txt" \
        -x 0x123456789abcdef00123456789abcdef
      ;;
    canon4)
      env BENCH_N="$CANON_N" "$ROOT/target/release/bench_canon4"
      ;;
    polycanon)
      env BENCH_N="$CANON_N" "$ROOT/target/release/bench_polycanon"
      ;;
    *)
      echo "unknown benchmark case: $1" >&2
      return 2
      ;;
  esac
}

export ROOT BIN SMALL MEDIUM CANON_N
export -f run_case

for name in "${CASES[@]}"; do
  echo "benchmark: $name"
  for ((i = 1; i <= WARMUP; i++)); do
    run_case "$name" > "$WORK/${name}.warmup.${i}.stdout" 2> "$WORK/${name}.warmup.${i}.stderr"
  done

  for ((i = 1; i <= RUNS; i++)); do
    time_file="$WORK/${name}.${i}.time"
    status=0
    /usr/bin/time -f '%e	%M' -o "$time_file" \
      bash -c 'run_case "$1"' bash "$name" \
      > "$WORK/${name}.${i}.stdout" 2> "$WORK/${name}.${i}.stderr" || status=$?
    read -r seconds rss < "$time_file"
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$i" "$seconds" "$rss" "$status" >> "$RAW"
    if [[ "$status" != "0" ]]; then
      echo "benchmark case failed: $name run $i" >&2
      exit "$status"
    fi
  done
done

python3 - "$RAW" "$REPORT" "$ROOT" "$RUNS" "$WARMUP" <<'PY'
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict

raw_path, report_path, root, runs, warmup = sys.argv[1:]
rows = []
with open(raw_path, encoding="utf-8") as f:
    header = f.readline().strip().split("\t")
    for line in f:
        item = dict(zip(header, line.rstrip("\n").split("\t")))
        item["run"] = int(item["run"])
        item["seconds"] = float(item["seconds"])
        item["max_rss_kb"] = int(item["max_rss_kb"])
        item["status"] = int(item["status"])
        rows.append(item)

by_name = defaultdict(list)
for row in rows:
    by_name[row["name"]].append(row)

def cmd(args):
    try:
        return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

benchmarks = []
for name in sorted(by_name):
    values = by_name[name]
    seconds = [row["seconds"] for row in values]
    rss = [row["max_rss_kb"] for row in values]
    benchmarks.append(
        {
            "name": name,
            "runs": len(values),
            "median_seconds": statistics.median(seconds),
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
            "median_max_rss_kb": int(statistics.median(rss)),
            "max_rss_kb": max(rss),
        }
    )

report = {
    "metadata": {
        "git_head": cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": cmd(["git", "status", "--short"]),
        "rustc": cmd(["rustc", "-vV"]),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore")
                if line.startswith("model name")
            ),
            platform.processor() or "unknown",
        ),
        "runs": int(runs),
        "warmup": int(warmup),
    },
    "benchmarks": benchmarks,
}

os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")

print(f"Wrote {report_path}")
PY
