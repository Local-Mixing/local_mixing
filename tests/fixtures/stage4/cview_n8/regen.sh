#!/bin/bash
# Regenerate expected/ for the C-view fixture with the PINNED baseline
# hmap_affine (built from 492564f0 with --features security-tools; see
# docs/STAGE4_RUST_DESIGN.md §10.1). Refuses any binary whose sha256 is not
# listed in baseline_sha256.txt, so the expected files can never come from a
# binary that links the library under test.
#   HMAP_BASELINE=/path/to/baseline/hmap_affine bash regen.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${HMAP_BASELINE:?set HMAP_BASELINE to the baseline hmap_affine}"
got=$(shasum -a 256 "$HMAP_BASELINE" | cut -d' ' -f1)
if ! grep -q "^$got " "$HERE/baseline_sha256.txt"; then
  echo "refusing: $HMAP_BASELINE has sha256 $got, not a pinned baseline" >&2
  exit 1
fi
rm -rf "$HERE/expected"
mkdir -p "$HERE/expected"
cd "$HERE/expected"
# cases.tsv: name, ALLCUTS (0/1), WITNESS (0/1), arguments; "@/" = this dir.
while IFS=$'\t' read -r name allc wit args; do
  envs=(PATH=/usr/bin:/bin)
  [ "$allc" = 1 ] && envs+=("EXPOSURE_ALLCUTS=$name.allcuts.tsv")
  [ "$wit" = 1 ] && envs+=("EXPOSURE_WITNESS=$name.wit.tsv")
  # shellcheck disable=SC2086
  env -i "${envs[@]}" "$HMAP_BASELINE" ${args//@\//$HERE/} --out "$name" > "$name.stdout"
done < "$HERE/cases.tsv"
ls -la
