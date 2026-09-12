#!/usr/bin/env bash
# Preserve the historical sss/genran CLI. Never fall back to the production GSS CLI.
set -euo pipefail

if [[ -n "${SECURITY_LEGACY_BIN:-}" ]]; then
  exec "$SECURITY_LEGACY_BIN" "$@"
fi

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SECURITY_REPO_ROOT:-$(cd -- "$SOURCE_DIR/../.." && pwd)}"
exec cargo run --manifest-path "$REPO_ROOT/Cargo.toml" --release \
  --features legacy-tools --bin legacy_mixing -- "$@"
