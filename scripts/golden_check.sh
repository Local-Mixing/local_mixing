#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LOCAL_MIXING_BIN:-"$ROOT/target/release/local_mixing_bin"}"
MODE="${GOLDEN_MODE:-check}"
OUT_DIR="${GOLDEN_OUT:-"$ROOT/reports/golden/current"}"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
COMMIT_MANIFEST="$ROOT/reports/golden/${COMMIT}.sha256"
BASELINE_MANIFEST="${GOLDEN_BASELINE:-"$ROOT/reports/golden/baseline.sha256"}"

if [[ ! -x "$BIN" ]]; then
  cargo build --manifest-path "$ROOT/Cargo.toml" --locked --release --bin local_mixing_bin
fi

case "$MODE" in
  record|check) ;;
  *)
    echo "GOLDEN_MODE must be 'record' or 'check' (got '$MODE')" >&2
    exit 2
    ;;
esac

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/circuits" "$OUT_DIR/stdout" "$ROOT/reports/golden"

SMALL="$OUT_DIR/circuits/small_n6.txt"
MEDIUM="$OUT_DIR/circuits/medium_n12.txt"
SHOT="$OUT_DIR/circuits/small_n6_shot.txt"
SHUFFLED="$OUT_DIR/circuits/small_n6_shuffled.txt"
COMPRESSED="$OUT_DIR/circuits/small_n6_compressed.txt"

printf '012;345;024;135;250;431;\n' > "$SMALL"
printf '012;345;678;9ab;036;147;258;39a;4ab;50a;61b;72a;84b;\n' > "$MEDIUM"

"$BIN" circuit evaluate -n 6 -s "$SMALL" -x 0x2a > "$OUT_DIR/stdout/evaluate_small.stdout"
"$BIN" circuit evaluate -n 12 -s "$MEDIUM" -x 0xace > "$OUT_DIR/stdout/evaluate_medium.stdout"
"$BIN" circuit compare -n 6 -i 128 -a "$SMALL" -b "$SMALL" > "$OUT_DIR/stdout/equal_self.stdout"

{
  echo "Golden determinism notes"
  echo "commit=$COMMIT"
  echo "primary_hashed_commands=evaluate,equal_self,fixture_files"
  echo "nondeterministic_cli_sources=genran,shoot,shuffle,sss,compress use rand::rng or unseeded global fastrand"
  echo "db_oracles_enabled=${GOLDEN_INCLUDE_DB:-0}"
  echo "slow_oracles_enabled=${GOLDEN_INCLUDE_SLOW:-0}"
} > "$OUT_DIR/stdout/notes.txt"

if [[ "${GOLDEN_INCLUDE_DB:-0}" == "1" && -f "$ROOT/db/data.mdb" ]]; then
  "$ROOT/security_tests/campaigns/legacy_runner.sh" shoot -i 1 -s "$SMALL" -d "$SHOT" > "$OUT_DIR/stdout/shoot.stdout"
  "$BIN" circuit compare -n 6 -i 256 -a "$SMALL" -b "$SHOT" > "$OUT_DIR/stdout/shoot_equal.stdout"
  grep -q "No mismatch" "$OUT_DIR/stdout/shoot_equal.stdout"

  "$ROOT/security_tests/campaigns/legacy_runner.sh" shuffle -n 6 -i 1 -s "$SMALL" -d "$SHUFFLED" > "$OUT_DIR/stdout/shuffle.stdout"
  "$BIN" circuit compare -n 6 -i 256 -a "$SMALL" -b "$SHUFFLED" > "$OUT_DIR/stdout/shuffle_equal.stdout"
  grep -q "No mismatch" "$OUT_DIR/stdout/shuffle_equal.stdout"

  "$ROOT/security_tests/campaigns/legacy_runner.sh" compress -n 6 -s "$SMALL" -d "$COMPRESSED" --stable_compressions 1 > "$OUT_DIR/stdout/compress.stdout"
  "$BIN" circuit compare -n 6 -i 256 -a "$SMALL" -b "$COMPRESSED" > "$OUT_DIR/stdout/compress_equal.stdout"
  grep -q "No mismatch" "$OUT_DIR/stdout/compress_equal.stdout"
fi

if [[ "${GOLDEN_INCLUDE_SLOW:-0}" == "1" && -f "$ROOT/rantestn128m800/n128m800_source.txt" ]]; then
  "$BIN" circuit evaluate \
    -n 128 \
    -s "$ROOT/rantestn128m800/n128m800_source.txt" \
    -x 0x123456789abcdef00123456789abcdef \
    > "$OUT_DIR/stdout/evaluate_headline_n128m800.stdout"
fi

MANIFEST="$OUT_DIR.sha256"
(
  cd "$OUT_DIR"
  find . -type f | sort | xargs sha256sum
) > "$MANIFEST"

if [[ "$MODE" == "record" ]]; then
  cp "$MANIFEST" "$COMMIT_MANIFEST"
  cp "$MANIFEST" "$BASELINE_MANIFEST"
  echo "Recorded golden manifest:"
  echo "  $COMMIT_MANIFEST"
  echo "  $BASELINE_MANIFEST"
else
  if [[ ! -f "$BASELINE_MANIFEST" ]]; then
    echo "Missing baseline manifest: $BASELINE_MANIFEST" >&2
    echo "Run with GOLDEN_MODE=record first." >&2
    exit 1
  fi
  diff -u "$BASELINE_MANIFEST" "$MANIFEST"
  echo "Golden check passed against $BASELINE_MANIFEST"
fi
