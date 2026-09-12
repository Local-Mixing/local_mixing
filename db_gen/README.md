# db_gen — regular, curated, and wide replacement stores

Everything needed to (re)build the two immutable stores the mixers read at
runtime:

* **regular frozen store** → `FROZEN_DB_DIR`
* **curated frozen store** → `FROZEN_CURATED_DIR`

A frozen store directory contains `shard_NN.frz` (256 shards), `tables.bin`,
and optionally `filters.bin` (the in-RAM BinaryFuse8 miss filter, enabled at
runtime with `FROZEN_FILTER=1`). The runtime reader is
`src/database/frozen.rs`; nothing in this folder is needed at mixing time.

Build the tools with the RocksDB-linked feature:

```bash
cargo build --release --features db-tools
```

`db-tools` enables the existing `legacy-db-tools` feature; old build commands
and Rust feature checks remain compatible. The library module is still
`local_mixing::db_generation`, compiled for database tools (and relevant test/benchmark builds).
Shared codecs and validation live in `local_mixing::database`; the ordinary
GSS library excludes database builders. Integration tests include the
dependency-light fixture encoder directly from its authoritative source.

All database executables are explicitly gated behind this feature.
The grouped CLI is `local_mixing_bin db --help`.

## Curated store, end to end

1. **Build the composite RocksDB** (`build_curated_full`, library
   `curated_full.rs`). Output paths must be fresh — the tools refuse existing
   paths and never clear a live db. Run generation with the canonicalization
   caps lifted:

   ```bash
   env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
     target/release/build_curated_full from-identities IDENTITY_LMDB FRESH_COMPOSITE_ROCKS
   ```

   Source routes:
   * `from-identities` — from the accepted `id_g0..id_g33` identity DBs
     (the minimal-identity lineage; requires all 34). Every emitted candidate
     is re-canonicalized and must rehash to its key; a canonicalization skip
     is fatal.
   * `import-legacy-rocks SOURCE_ROCKS OUT` — import the uncapped historical
     minimal-identity RocksDB (cannot recover records a capped shortcut store
     already discarded).
   * `from-regular-shortcut REGULAR_LMDB OUT` — uncapped rebuild of the
     shuffletests shortcut from the regular DB. NOT the minimal-identity
     lineage; label accordingly.

2. **Audit**: `build_curated_full audit FRESH_COMPOSITE_ROCKS` — writes the
   completion manifest (exact key/candidate counts, content digest). Later
   stages reject an interrupted store.

3. **Freeze** (`frozen_from_lmdb`, library `frozen_build.rs`) — run the three
   staged subcommands in order. `FROZEN_OUT` must not exist when `tables`
   starts; that stage creates it. `write` refuses existing shards or a stale
   `filters.bin`, and `validate` requires all 256 shards. Move any interrupted
   output aside rather than resuming it in place:

   ```bash
   target/release/frozen_from_lmdb tables   FRESH_COMPOSITE_ROCKS FROZEN_OUT --curated --composite
   target/release/frozen_from_lmdb write    FRESH_COMPOSITE_ROCKS FROZEN_OUT --curated --composite
   target/release/frozen_from_lmdb validate FRESH_COMPOSITE_ROCKS FROZEN_OUT --curated --composite
   ```

   `--composite` (read the composite RocksDB directly) is the ONLY route for
   the uncapped curated database: LMDB caps a non-DUPSORT value at 4 GiB − 1
   and the largest curated key needs ~16 GiB, so the older documented
   `to-lmdb` → `validate-lmdb` → `frozen_from_lmdb --curated` route cannot
   represent it (that route still works for bounded stores; the frozen
   encoding is identical either way). If you pass `--min-gates N`, pass the
   same value to all three stages. `validate` checks every value
   byte-for-byte against the frozen output.

4. **Miss filter** (optional but production-standard):

   ```bash
   target/release/frozen_filters_build from-frozen FROZEN_OUT
   ```

   writes `FROZEN_OUT/filters.bin` from the served frozen artifact itself. It
   refuses overwrite, verifies every frozen key against both the in-memory and
   serialized filters, and publishes atomically only after validation. The
   same RocksDB-free command works for regular and curated frozen stores.

**Value convention:** new builds use the NATIVE convention. Set
`FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls` only when serving
the historical pre-`2ed0222a` store (see docs/history/research/CURATED_DB_COMPARISON.md).

## Regular store

The regular store can be rebuilt completely from this folder. The generator
retains the historical path contract: `db build-regular -m N` reads
`rocks_db_m{N-1}` and writes `test_rocks_db_mN`; `db combine-regular` reads its two
`rocks_db_m*` inputs and writes `test_rocks_db_m{m1+m2}`. Always run in a fresh
working directory and promote a completed `test_*` output to `rocks_db_*`
before using it as the next input. Regular intermediates predate completion
manifests: every writer refuses an existing output path, and an interrupted
build exits nonzero. Treat that output as partial and choose a fresh path (or
move the partial artifact aside for investigation); never resume or promote
it.

The deployed production corpus contains complete m1 through m6, m7 restricted
to at least 15 used wires, and m8 restricted to at least 18 used wires. The
following reproduces that lineage by extending one gate at a time. It is
extremely large and long-running; these are executable commands, not a claim
that a laptop can finish the data set:

```bash
# Run from the repository root after the release build above.
export REPO_ROOT="$(pwd)"
export LOCAL_MIXING_BIN="$REPO_ROOT/target/release/local_mixing_bin"
export MERGE_ROCKS_BIN="$REPO_ROOT/target/release/merge_rocks_parallel"
export REGULAR_FROZEN_OUT="$REPO_ROOT/build/regular-frozen"
export REGULAR_DB_DIR="$REPO_ROOT/build/regular-db"
mkdir -p "$REGULAR_DB_DIR"
cd "$REGULAR_DB_DIR"

env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  "$LOCAL_MIXING_BIN" db build-regular -m 1
mv test_rocks_db_m1 rocks_db_m1

for m in 2 3 4 5 6; do
  env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
    "$LOCAL_MIXING_BIN" db build-regular -m "$m"
  mv "test_rocks_db_m$m" "rocks_db_m$m"
done

env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  "$LOCAL_MIXING_BIN" db build-regular -m 7 --min_n 15
mv test_rocks_db_m7 rocks_db_m7

env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  "$LOCAL_MIXING_BIN" db build-regular -m 8 --min_n 18
mv test_rocks_db_m8 rocks_db_m8

"$MERGE_ROCKS_BIN" regular_combined_rocks \
  rocks_db_m1 rocks_db_m2 rocks_db_m3 rocks_db_m4 rocks_db_m5 \
  rocks_db_m6 rocks_db_m7 rocks_db_m8
"$LOCAL_MIXING_BIN" db export-lmdb \
  --source regular_combined_rocks --path regular_lmdb
```

`db combine-regular --m1 A --m2 B [--min_n N]` is the faster overlap-combination route
used for selected higher-gate builds. `db build-regular` also accepts `--min_n`,
`--max_n`, and `--no_L`. Complete m7/m8, m9, and other bands are valid
alternative artifacts, but they are not the deployed production corpus above.
Those knobs change the database contents; record them with the artifact rather
than assuming a machine-local recipe.

The regular LMDB contains named shards `00` through `ff`. Freeze and validate
it with the current frozen codec (no `--curated` flag). As above,
`REGULAR_FROZEN_OUT` must not exist before `tables`; move a partial directory
aside rather than reusing it:

```bash
"$REPO_ROOT/target/release/frozen_from_lmdb" tables   regular_lmdb "$REGULAR_FROZEN_OUT"
"$REPO_ROOT/target/release/frozen_from_lmdb" write    regular_lmdb "$REGULAR_FROZEN_OUT"
"$REPO_ROOT/target/release/frozen_from_lmdb" validate regular_lmdb "$REGULAR_FROZEN_OUT"
"$REPO_ROOT/target/release/frozen_filters_build" from-frozen "$REGULAR_FROZEN_OUT"
```

`validate` compares every frozen value byte-for-byte with its LMDB source.
The trust boundary is intentionally narrower than a semantic database audit:
the current from-scratch generator establishes each key from canonicalization
by construction, while merge, RocksDB-to-LMDB conversion, and frozen validation
check key/value framing and preserve/compare bytes. They do not re-canonicalize
every stored candidate and rehash it to its enclosing key. A future hardening
step should add a complete regular semantic audit plus a signed or digested
completion manifest before promotion.

For generator-level checks against real intermediate RocksDBs, the ignored
tests in [`../tests/db_gen/regular_validation_tests.rs`](../tests/db_gen/regular_validation_tests.rs)
cover fresh-wire dedup, capped mapping
enumeration, and parallel-merge equivalence. They remain private unit tests
of `db_generation::regular`, included with `#[path]`; no standalone Cargo test
target or public generator helpers are needed. Run each explicitly so its input
contract is visible. `REGULAR_DB_DIR` makes the numbered input paths explicit;
Cargo runs tests from the package root regardless of the invoking shell's cwd:

```bash
REGULAR_DB_DIR="$REPO_ROOT/build/regular-db" \
  cargo test --manifest-path "$REPO_ROOT/Cargo.toml" --release \
  --features db-tools --lib fresh_wire_dedup_kv_sets_match \
  -- --ignored --nocapture
REGULAR_DB_DIR="$REPO_ROOT/build/regular-db" \
  cargo test --manifest-path "$REPO_ROOT/Cargo.toml" --release \
  --features db-tools --lib capped_2rocks_enumeration_matches_full \
  -- --ignored --nocapture

# Build a small prefix-only merge, then compare it with a direct source scan.
MERGE_TEST_PREFIX=b0b1 "$MERGE_ROCKS_BIN" "$REGULAR_DB_DIR/merge_check_b0b1" \
  "$REGULAR_DB_DIR/rocks_db_m1" "$REGULAR_DB_DIR/rocks_db_m2" \
  "$REGULAR_DB_DIR/rocks_db_m3" "$REGULAR_DB_DIR/rocks_db_m4" \
  "$REGULAR_DB_DIR/rocks_db_m5" "$REGULAR_DB_DIR/rocks_db_m6" \
  "$REGULAR_DB_DIR/rocks_db_m7" "$REGULAR_DB_DIR/rocks_db_m8"
MERGE_PAR_DB="$REGULAR_DB_DIR/merge_check_b0b1" \
MERGE_SOURCES="$REGULAR_DB_DIR/rocks_db_m1:$REGULAR_DB_DIR/rocks_db_m2:$REGULAR_DB_DIR/rocks_db_m3:$REGULAR_DB_DIR/rocks_db_m4:$REGULAR_DB_DIR/rocks_db_m5:$REGULAR_DB_DIR/rocks_db_m6:$REGULAR_DB_DIR/rocks_db_m7:$REGULAR_DB_DIR/rocks_db_m8" \
MERGE_TEST_PREFIX=b0b1 \
  cargo test --manifest-path "$REPO_ROOT/Cargo.toml" --release \
    --features db-tools --lib merge_prefix_rebuild_matches \
    -- --ignored --nocapture
```

Regular keys are an ABI with the runtime canonicalizer. The compatibility
adapter in `regular.rs` uses the current `canonicalize_polys_4` implementation
for both circuit directions; do not regenerate stores after changing its
golden canonical-form hashes without an explicit migration decision. See
`docs/history/research/Mixing_Pieces_Documentation.md` §4 for the enumeration principle and
`docs/design/FULL_CURATED_DB.md` / `docs/history/research/CURATED_DB_COMPARISON.md` for curated
lineage.

## Wide sidecar work

`wide_gates.rs` enumerates wide candidates; `regular.rs` retains both the
one-wide and two-wide generation passes and the MPX1 merge operator.
`bin/wide_verify.rs` decodes every candidate and verifies its canonical key.
The support implementations now live in `support/mpx1.rs`, `support/xcanon.rs`,
and `support/wide_db.rs`. Their compatibility module paths remain
`local_mixing::engine::mpx1`, `local_mixing::circuit::xcanon`, and
`local_mixing::db_mixing::wide_db`.

Build `local_mixing_bin` and `wide_verify` with `--features db-tools`. From an
existing regular intermediate directory, the command contracts are:

```text
local_mixing_bin rocksdb_wide -m N [--min_n MIN] [--max_n MAX]
  reads rocks_db_mN (or REGULAR_DB_DIR/rocks_db_mN)
  writes test_wide_db_mN in the working directory
wide_verify test_wide_db_mN

local_mixing_bin rocksdb_wide2 -m N [--min_n MIN] [--max_n MAX]
  reads wide_db_mN in the working directory
  writes test_wide2_db_mN in the working directory
wide_verify test_wide2_db_mN
```

The second pass expects a verified first-pass store promoted to `wide_db_mN`.
All output paths must be fresh. These are separate wide sidecars; regular and
curated frozen stores and the production frozen reader retain their existing
formats. `wide_db` is an offline RocksDB probe reader, not an enabled GSS
runtime lookup path. Existing wide codec, gate-order, and roundtrip tests stay
with their owning private modules.

## Experimental census / analysis tools

The retained probes and experimental builders live in `db_gen/analysis/`.
Their Cargo target names are unchanged. These are optional analysis or
alternative construction workflows, separate from the regular/curated recipe
above: `curated_size_census`,
`curated_recanon_probe`, `curated_key_histogram`, `curated_key_structure`,
`curated_key_filter`, `curated_coverage_census`, `cross_gluing_probe`,
`frozen_class_census`, `identity_length_census`, `identity_shingle_sieve`,
`minimal_identity_filter`, `minimal_halves_probe`. Three of these were the
prototypes later productionized inside `build_curated_full` (key filter →
Sieve, shingle sieve → Sieve sizing, cross-gluing → FromFrozenIdentitiesV2).

## Deployed stores (as of 2026-08)

* `frozen_curated_v2` (3.4 GB, 32.35M keys / 250M candidates) on n64tests.
* Uncapped 16 GB curated store (25.88M keys / 1.583B candidates, validated
  0 mismatches) built composite→frozen on sattesting-nh.
* Fleet-wide native re-encode of the historical store:
  `frozen_curated_m1_m11_native` (legacy dirs kept for rollback only).

## Analysis and maintenance scripts

The three Python analysis tools in `analysis/` take explicit input/output
paths. `analysis/compare_similarity.sh` finds `sample_similarity.py` beside
it and reads the existing `reports/curated_diversity_20260815/` data by default.
It can be invoked from any working directory.

The repository-based maintenance scripts source `maintenance/paths.sh`:

| Setting | Default and purpose |
| --- | --- |
| `DB_GEN_REPO_ROOT` | Checkout containing this `db_gen/`; selects source and Cargo.toml |
| `DB_GEN_RUN_ROOT` | `$HOME/curated_diversity_20260815`; existing composite/frozen data and coverage/filter logs |
| `DB_GEN_REPORT_ROOT` | `$DB_GEN_REPO_ROOT/reports/curated_diversity_20260815`; existing similarity sample files |
| `DB_GEN_TARGET_DIR` | `CARGO_TARGET_DIR`, or `$DB_GEN_REPO_ROOT/target`; shared build and executable lookup |
| `DB_GEN_CARGO` | `$HOME/.cargo/bin/cargo`; override to select the Cargo executable |

Relative run/report overrides are resolved against the invoking directory;
relative target paths are resolved against the checkout. The build wrappers
pass the resolved target directory to Cargo explicitly. `launch_filters.sh`
launches its moved sibling `build_filters.sh`; `launch_coverage.sh` keeps both
logs in the existing run root. No script looks for source in the historical
`src_tree/` snapshot or writes logs beside its newly moved source.

`build_filters.sh` uses the current `frozen_filters_build from-frozen` interface.
It retains the existing frozen source and deployed hardlink destination; an
already present `filters.bin` is refused by the builder. Fleet scripts retain
their existing store names, remote hosts, transfer methods, and home-based
artifact roots. `deploy_v2.sh`, `distribute_filters.sh`, and `n64_keygen.sh`
also accept `DB_GEN_RUN_ROOT`. `fleet_check.sh` remains usable through remote
`bash -s`, with `DB_GEN_REPO_ROOT` defaulting to `$HOME/local_mixing` there.

`n64_keygen.sh` is DB deployment support: it prepares the fleet transfer SSH
key and frozen-store checksum manifest. It is retained under maintenance,
not classified as circuit/challenge key generation. These operational scripts
were relocated and syntax-checked only; no builds, DB operations, deployment,
cleanup, SSH, or key generation were run as part of the move.
