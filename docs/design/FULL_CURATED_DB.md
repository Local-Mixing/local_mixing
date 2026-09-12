# Full minimal-identity curated database

The installed minimal-identity frozen store is intentionally normalized: that
separate post-processing step retained at most 20 distinct friends and 512
decoded bytes per key. The live LMDB has a different shortcut-derived lineage;
a read-only sample contains up to 32 friends but still shows the 512-byte cap.
The shortcut itself limits each *source* equivalence class to its first 20
friends; that is not a final friends-per-output-key limit. The runtime also
formerly decoded at most 256 curated friends. None of these bounds is part of
the minimal-identity acceptance test.

There are two historical generators:

- `lmdb:src/bin/minimal_id.rs` writes accepted identities to `id_g0` through
  `id_g33`; `lmdb:src/bin/curated_db.rs` splits those identities. This is the
  minimal-identity lineage used by the production curated database.
- `shuffletests:src/rainbow_table/bin/build_curated.rs` is a later shortcut
  that derives identities from multi-friend regular keys. It does not run the
  minimal-identity test. It also considers only the first 20 source friends,
  caps each output value at 512 bytes in two places, and silently skips
  circuits that do not fit the one-byte circuit-length field.

`build_curated_full` preserves the first lineage and removes representation
truncation after the tests. It can consume the accepted `id_g*` identities
directly, or import the known uncapped historical minimal-identity append-value
RocksDB. Importing a 20/512-bounded shortcut output cannot recover records that
were already discarded; rebuild that corpus with `from-regular-shortcut`.
It writes each exact `(16-byte function key, circuit blob)` pair as a separate
composite RocksDB key. RocksDB therefore performs global byte-exact
deduplication, with no hot-key merge value, count cap, byte cap, or
probabilistic dedup hash.

## Rebuild from the historical minimal-identity RocksDB

All output paths below must be new. The tools refuse existing paths and never
clear the live `db/`.

Composite output receives a completion manifest (exact key count, candidate
count, and content digest) only after generation and its full audit finish.
Import, materialization, and audit reject an interrupted store that has only
the initial format marker, and materialization rechecks the manifest before it
creates its output LMDB.

The identity route requires all 34 historical `id_g*` databases and at least
one accepted identity; the legacy-import and shortcut routes likewise refuse
empty inputs. This prevents a wrong source path from being blessed as a valid
empty “complete” store.

LMDB materialization writes its own completion manifest only after all shard
data is committed and synced. `validate-lmdb` matches it to the composite
source, and `frozen_from_lmdb --curated` refuses an LMDB without it, so an
interrupted materialization cannot enter the frozen stage.

```sh
cargo run --release --features legacy-db-tools --bin build_curated_full -- \
  import-legacy-rocks SOURCE_ROCKS FRESH_COMPOSITE_ROCKS

cargo run --release --features legacy-db-tools --bin build_curated_full -- \
  audit FRESH_COMPOSITE_ROCKS

cargo run --release --features legacy-db-tools --bin build_curated_full -- \
  to-lmdb FRESH_COMPOSITE_ROCKS FRESH_LMDB

cargo run --release --features legacy-db-tools --bin build_curated_full -- \
  validate-lmdb FRESH_COMPOSITE_ROCKS FRESH_LMDB

cargo run --release --bin frozen_from_lmdb -- \
  tables FRESH_LMDB FRESH_FRZ --curated
cargo run --release --bin frozen_from_lmdb -- \
  write FRESH_LMDB FRESH_FRZ --curated
cargo run --release --bin frozen_from_lmdb -- \
  validate FRESH_LMDB FRESH_FRZ --curated
```

`validate-lmdb` checks every materialized candidate against the composite
source and requires exact key/candidate counts. `frozen_from_lmdb validate`
then checks every LMDB value byte-for-byte against the frozen output.
The frozen writer also rejects two full 128-bit keys that collide in its
76-bit address field, a bucket above 65,535 keys, or a shard above its 40-bit
offset range; these formerly could create a byte-valid file with unreachable
entries.

## Rebuild from accepted identities

```sh
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  cargo run --release --features legacy-db-tools --bin build_curated_full -- \
  from-identities IDENTITY_LMDB FRESH_COMPOSITE_ROCKS
```

Every emitted prefix and reversed-suffix candidate is canonicalized again and
must rehash to its destination key. A canonicalization skip is fatal. The
builder enumerates both directions, every rotation, and every split; exact
composite keys remove only byte-identical duplicates.

## Reproduce the `shuffletests` shortcut without truncation

If the intended source is the current regular database rather than the
historical `id_g*` test output, use:

```sh
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  cargo run --release --features legacy-db-tools --bin build_curated_full -- \
  from-regular-shortcut REGULAR_LMDB FRESH_COMPOSITE_ROCKS
```

This processes every distinct source friend (not the first 20), has no output
byte cap, exact-deduplicates globally, and re-canonicalizes every emitted
candidate to prove it belongs under its key. As in the branch code, this is a
shortcut derivation from regular equivalence classes; it is not the
`minimal_id` acceptance test and should not be mislabeled as that lineage.

## Format boundary

The legacy and current frozen value format stores a circuit byte length and
each wire in one byte. It can represent at most 85 three-byte gates and wire
indices through 255. The minimal identities used by the historical database
are well inside that boundary. The full builder checks both limits and fails
the build if either is exceeded; it never truncates, skips, or wraps an
unrepresentable circuit. Supporting larger circuits requires a versioned
value/frozen format with wider length and wire fields.

## Runtime cost

Every stored candidate is now returned by `FrozenDb::get_curated`; the former
256-candidate decode cap and the 20/512 bounded-store warning are gone. Full
positive curated values are not retained in the process-wide lookup cache.

The current frozen format is sequential, so selecting uniformly or finding a
global shortest candidate still requires decoding the complete value. The old
uncapped store was retired after pathological keys made some lookups very
large. A future indexed frozen format (candidate count plus restart offsets,
ideally grouped by gate length) is needed for fast random access without
sacrificing completeness.

New builds from `id_g*` use the native value convention. Set
`FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls` only when importing
and using the historical pre-convention-fix store whose validation requires
that compatibility transform.
