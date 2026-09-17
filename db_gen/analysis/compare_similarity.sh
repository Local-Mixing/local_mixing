#!/usr/bin/env bash
# Compare explicit curated_key_structure --sample exports at every gate count.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash db_gen/analysis/compare_similarity.sh SAMPLE.tsv [SAMPLE.tsv ...]

Each sample contains lines of gates<TAB>hexblob from curated_key_structure.
Print all gate-count buckets and their pairwise/nearest-neighbor similarity
tables for each file. DB_GEN_PYTHON optionally selects the Python executable.
USAGE
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (($# == 0)); then usage >&2; exit 2; fi
for sample in "$@"; do
  if [[ ! -f "$sample" ]]; then
    printf 'Missing sample file: %s\n' "$sample" >&2
    exit 1
  fi
done

analysis_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
for sample in "$@"; do
  printf '\nSample: %s\n' "$sample"
  "${DB_GEN_PYTHON:-python3}" "$analysis_dir/sample_similarity.py" "$sample"
done
