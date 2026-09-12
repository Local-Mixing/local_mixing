#!/bin/bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../maintenance/paths.sh" || exit 1
cd -- "$DB_GEN_REPORT_ROOT" || exit 1
for f in raw orbit L8 L6 L5 L4; do
  echo "##### filter_toffoli_$f"
  python3 "$DB_GEN_SCRIPT_DIR/../analysis/sample_similarity.py" "filter_toffoli_$f.txt" 2>/dev/null \
    | grep -A 8 'gates=11' | head -9
done
