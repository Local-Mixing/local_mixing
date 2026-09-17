# The Frozen Database

The main use of our database is simple. Given a small subcircuit, we want to
find other circuits that compute the same function. We can then replace the
subcircuit with one of these circuits, either to compress it or to change its
internal structure. We refer to circuits with the same functionality as
*friends*.

There are two separate ideas here. The *rainbow table* is the mapping from a
canonical functionality to its friends. The *frozen table* is the file format
we use to store that mapping. Changing the storage does not change what it
means for two circuits to be equivalent.

## Why we moved away from LMDB and RocksDB at runtime

Earlier, we stored tables by wire count and gate count, using SQL and then
LMDB. Polynomial canonicalization allowed us to stop separating equivalent
functions by the number of unused wires around them. We could generate
circuits by gate count, canonicalize their functionality, and merge their
friends into one entry.

We also noticed that mixing never needs to insert, update, or delete a database
entry. We construct the table ahead of time and only ask it for exact keys.
Thus, the machinery needed for a mutable database is unnecessary during
mixing. LMDB maintains a transactional B-tree; RocksDB supports mutable,
compressed stores through its log and sorted tables. The frozen reader instead
goes directly to the compressed bucket containing the requested key.

We still use LMDB and RocksDB when constructing databases. Their role has
changed:

| Storage | Current use |
| --- | --- |
| RocksDB | Generate and merge regular circuits; deduplicate curated candidates during construction. |
| LMDB | Hold sharded regular input for freezing, and read existing identity sources. Bounded curated imports are also supported. |
| Frozen files | Serve regular and curated replacements during TDP database mixing. |

The ordinary TDP build therefore does not need RocksDB or LMDB. The offline
builders are enabled with `--features db-tools`. There is no runtime fallback
to an old LMDB directory if the frozen files are missing.

## What we store

Let $P(C)$ be the canonical output polynomials of a circuit $C$. The function
key is computed as

$$
k = \operatorname{XXH3\text{-}128}(\operatorname{serialize}(P(C))).
$$

The serialization and wire mapping are described in
[Polynomial Canonicalization](polynomial_canonicalization.md). The regular
builder compares the forward and inverse canonical forms and stores the
smaller one, so reversing a circuit does not require a separate entry. The
curated store uses the forward canonical form of the function being replaced.

Both stores contain lists of g57 circuits. We use the native convention

$$
[a,b,c]:\qquad a\mathrel{\oplus}=b\lor\neg c
                      =1\oplus c\oplus bc.
$$

A decoded value is a chain of records, each containing a one-byte length
followed by its gate triples:

```text
[byte length][target, positive control, negative control] ...
[byte length][target, positive control, negative control] ...
```

The length is measured in bytes, so a circuit with $m$ gates has length $3m$.
Each wire label occupies one byte. This bounds one circuit record at 85 gates;
it does not bound the number of friends under a key. The frozen decoder returns
this same representation, which lets the replacement code use the decoded
values without knowing how they were compressed.

The *regular* table comes from enumerating circuits and collecting equivalent
forms. The *curated* table is meant to supply more structurally different
replacements. In the identity-based construction, $I=AB$ implies $A=B^{-1}$,
so splitting identities supplies friends for their prefixes. We use cyclic
rotations and both directions to obtain more such pairs.

The contents depend on which source tables and construction bounds we used.
In particular, a directory name such as `frozen_m1_m11` does not by itself
establish exhaustive coverage at every length and wire count. Likewise, the
older curated limit of 20 candidates and 512 bytes per key was a restriction
of a particular build. It is not a limit of the current reader or the full
curated builder. The historical corpus sizes and measurements are discussed in
[Local Mixing History](local_mixing_history.md).

## How a frozen lookup works

Each store has the following files:

```text
tables.bin
shard_00.frz
...
shard_ff.frz
filters.bin       # optional
```

We retain 76 bits of the 128-bit key:

| Part | Bits | Purpose |
| --- | ---: | --- |
| Shard | 8 | Choose one of 256 files. |
| Bucket | 20 | Choose one of $2^{20}$ buckets in that file. |
| Tail | 48 | Identify an entry within the bucket. |
| Omitted | 52 | Not stored. |

These are consecutive bits of the serialized key bytes. The hash is converted
with `to_le_bytes()`, and `split_key` reads those bytes in order; this detail
matters when writing a compatible builder.

Each shard begins with `FRZTBL01`, an entry count, a data length, and
$2^{20}+1$ bucket offsets. The offsets are five-byte integers and stay in
memory. Across all shards, they occupy about 1.25 GiB per opened store. The
compressed circuit data stays in the files.

A lookup uses two neighboring offsets to find its bucket. Equal offsets mean
the bucket is empty, so we can return a miss immediately. Otherwise, we read
that byte range and look for the requested tail. Tails are sorted and stored
with Elias–Fano coding, which compresses an increasing sequence of integers.
Values follow in the same order. To reach a later value, the reader skips the
earlier compressed values in that bucket.

We compress values with canonical Huffman codes. The circuit header has its
own code, and each gate is coded as a whole triple. The gate code depends on
the circuit width, capped at 32, and the gate position, capped at 11. This
takes advantage of recurring gate patterns while retaining escape encodings
for less common values.

The optional `filters.bin` contains one BinaryFuse8 membership filter per
shard. If the filter says a key is absent, we avoid reading the bucket. A
possible hit still goes through the bucket lookup. Thus, a filter false
positive only causes extra work. We build filters from the frozen files
themselves and verify every retained key before publishing the filter.

Filters and decoded-value caches consume additional memory. Ordinary lookups
decode the full stored friend list; the replacement selector then walks record
offsets and constructs only the selected candidate. A separate cache retains
decoded large lists for repeated keys. This improves repeated lookups, but an
uncapped curated store can still require substantial memory for one key.

## From a key to a replacement

During [step 3 of TDP](tdp_pipeline.md), we first canonicalize the sampled
window. When curated lookup is enabled for the move, we check its forward key
in the curated store. The regular lookup is used when the curated stage has
no candidates, or when the move goes directly to regular. Some mixing policies
defer regular fallback until they have tried several curated window sizes.

For the regular lookup, we normally compute both directions but probe only
the smaller canonical form. This matches the regular builder's choice.
Historical and diagnostic lookup modes can also probe both keys. If we use
the inverse direction, we reverse the chosen friend before applying it.

We then apply the move's size rule, exclude disallowed spellings, and map the
friend back onto the original wires. A friend may need additional wires that
the window did not use. These must be placed on available wires in the
surrounding circuit, and the replacement must preserve their values as well.
A database hit therefore does not always produce an accepted replacement.

Keeping only 76 key bits means that different functions can collide. We do
not use a hash match alone as a proof of equivalence. With the default
verification enabled, the mixer compares the window and replacement on every
input when their combined support has at most 24 wires. Above that, it
compares their exact output polynomials. If the polynomial check exceeds its
budget or the 64-variable representation, the replacement is declined.
Disabling database verification removes this final check.

Control order also matters because g57's two controls are asymmetric. Current
builds use `native`, the default. `legacy-swapped-controls` swaps the two
decoded control bytes for historical stores that used the other convention.
It is a compatibility setting for those artifacts, not a setting to try when
a current store has a low match rate.

## Using an existing store

Set the directories in the `[database]` section of your TDP recipe:

```toml
[database]
regular_dir = "/path/to/regular-frozen"
curated_dir = "/path/to/curated-frozen"
lookup_miss_filter = "auto"
```

The regular store is required for normal database mixing; curated is optional.
`auto` enables filters when at least 60 GiB of available RAM is reported.
`"on"` explicitly requires a nonempty `filters.bin` in each configured store,
and `"off"` avoids loading them. Keep each filter with the exact frozen store
from which it was built. The reader checks the shard count and total entry
count, but these checks are not a content digest.

The equivalent directory settings for standalone tools are `FROZEN_DB_DIR`
and `FROZEN_CURATED_DIR`, with `FROZEN_FILTER=1` to enable filtering. See the
[README](../README.md) for running TDP with a recipe.

The reusable maintenance helpers take an explicit frozen-store directory:

```bash
bash db_gen/maintenance/build_filters.sh FROZEN_DIR
bash db_gen/maintenance/check_filters.sh FROZEN_DIR
```

[`build_filters.sh`](../db_gen/maintenance/build_filters.sh) refuses to overwrite
existing filters and validates that the new filters contain every retained key.
[`check_filters.sh`](../db_gen/maintenance/check_filters.sh) checks the file layout;
an optional second argument, `EXPECTED_SHA256`, also checks the filter digest.
That layout/digest check does not validate key membership.

For samples exported by `curated_key_structure`,
[`compare_similarity.sh`](../db_gen/analysis/compare_similarity.sh) compares each
explicit TSV input's circuits within gate-count buckets:

```bash
bash db_gen/analysis/compare_similarity.sh first.tsv second.tsv
```

## Constructing and freezing a store

Build the offline tools from the repository root:

```bash
cargo build --release --locked --features db-tools
target/release/local_mixing_bin db --help
```

Regular generation uses `db build-regular -m N` to extend the preceding gate
count, or `db combine-regular --m1 A --m2 B` to combine two existing tables.
Inputs are named `rocks_db_mN`; new outputs are named `test_rocks_db_mN`.
Run in a separate build directory and rename a successfully completed output
before using it as the next input. `--min_n` and `--max_n` restrict the used
wires, so record these choices with the generated data. For complete
generation, unset `CANON_MONOMIAL_CAP` and `CANON_RULE_L_BRANCH_CAP` rather
than carrying mixing-time search caps into the build.

After merging regular sources with `merge_rocks_parallel`, export the merged
store to sharded LMDB and freeze it:

```bash
target/release/local_mixing_bin db export-lmdb \
  --source REGULAR_ROCKS --path REGULAR_LMDB
target/release/frozen_from_lmdb tables REGULAR_LMDB FRESH_FROZEN_OUT
target/release/frozen_from_lmdb write REGULAR_LMDB FRESH_FROZEN_OUT
target/release/frozen_from_lmdb validate REGULAR_LMDB FRESH_FROZEN_OUT
target/release/frozen_filters_build from-frozen FRESH_FROZEN_OUT
```

We can also freeze regular RocksDB bands directly. Put one band directory per
line in `bands.txt`, then use `--bands` for all three conversion stages:

```bash
target/release/frozen_from_lmdb tables bands.txt FRESH_FROZEN_OUT --bands
target/release/frozen_from_lmdb write bands.txt FRESH_FROZEN_OUT --bands
target/release/frozen_from_lmdb validate bands.txt FRESH_FROZEN_OUT --bands
target/release/frozen_filters_build from-frozen FRESH_FROZEN_OUT
```

`MultiBandShards` merges keys in order and removes repeated candidate blobs
within and across bands. The bands must already be compacted so their pending
merge operands have been resolved. This route skips the intermediate LMDB
export and writes the same frozen format.

### Generate a curated database

The maintained builder is
[`build_curated_full`](../db_gen/bin/build_curated_full.rs). It produces a
**composite RocksDB** whose keys are `[function key][circuit bytes]` and whose
values are empty. Repeated function/circuit pairs collapse to one record;
the builder never needs to append all friends into one enormous mutable
value. Build it and the conversion tools with:

```bash
cargo build --release --locked --features db-tools \
  --bin build_curated_full --bin frozen_from_lmdb --bin frozen_filters_build
target/release/build_curated_full --help
```

Choose one of the following source routes. `REGULAR_FROZEN`, `FRESH_COMPOSITE`
and the other uppercase paths are placeholders to replace. Every output
must be a fresh path. These examples read the input stores and create new
outputs; they do not rebuild or modify the supplied input database.

**From an existing regular frozen store.** This is the direct route when the
regular runtime database is what you already have:

```bash
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-frozen-identities \
  REGULAR_FROZEN FRESH_COMPOSITE
target/release/build_curated_full audit FRESH_COMPOSITE
```

The builder scans all 256 shards. Within each equivalence class, pairs of
different spellings supply identities such as $AB^{-1}=I$. After local
cancellation, the builder considers rotations, both directions and accepted
split points; an identity $PQ=I$ yields $P=Q^{-1}$. Candidate functions are
canonicalized and verified against their keys before insertion. The result
depends on the supplied regular corpus: this does not enumerate every
possible reversible identity.

Optional variants use different candidate-selection policies:

```bash
# Keep splits whose halves have no detected compressible interior window.
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-frozen-identities \
  REGULAR_FROZEN FRESH_GOOD_SPLITS_COMPOSITE --good-splits --wires 32

# Deduplicate identity rotations/reversals, and reduce sibling candidates.
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-frozen-identities-v2 \
  REGULAR_FROZEN FRESH_V2_COMPOSITE --shards 256

# Add the v2 gluing phase using an already completed composite partner store.
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-frozen-identities-v2 \
  REGULAR_FROZEN FRESH_GLUED_COMPOSITE \
  --glue-partner-composite EXISTING_COMPOSITE \
  --glue-source-keys 200000 --glue-pairs 16 --glue-min-identity-gates 13
```

`--good-splits` is false by default. When enabled, its `--wires` option
(default 32) supplies the wire budget for local database probes. The scan
uses bounded local searches; “no detected compressible window” is the
appropriate interpretation. It filters individual splits rather than
discarding a whole identity because one of its arcs is compressible.

The v2 route additionally retains one candidate per identity, function key
and gate count, reducing equivalent rotation siblings. It has no
`--good-splits` option. `--shards` defaults to 256; smaller values are for
partial smoke builds. A successful completion manifest describes the
records actually built, and does not establish full-source coverage for a
partial shard run. Optional gluing samples connections through the supplied
composite store to construct longer identities; its three numeric defaults
are shown above. Record these choices alongside your build results because
v1, v2, filtering and gluing do not promise identical candidate sets.

**From accepted identity LMDB data.** Use this when you have the historical
accepted-identity corpus itself. The input must expose every named LMDB
database from `id_g0` through `id_g33`; missing databases are an error:

```bash
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-identities \
  IDENTITY_LMDB FRESH_COMPOSITE --batch-identities 4096
target/release/build_curated_full audit FRESH_COMPOSITE
```

`--batch-identities` defaults to 4096 and must be positive; it controls the
processing batch size, not candidate truncation. This route derives split
candidates from the accepted identities; it does not generate the source
`id_g*` corpus. Inputs are the historical native G57 identity blobs, not
arbitrary text circuit files.

**From regular LMDB or an existing legacy curated RocksDB.** These are two
different operations:

```bash
# Derive identities from equivalent regular friends in LMDB shards 00..ff.
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-regular-shortcut \
  REGULAR_LMDB FRESH_COMPOSITE

# Re-encode existing curated key/value lists as composite records.
target/release/build_curated_full import-legacy-rocks \
  LEGACY_CURATED_ROCKS FRESH_IMPORTED_COMPOSITE
target/release/build_curated_full audit FRESH_IMPORTED_COMPOSITE
```

The regular shortcut uses every source friend rather than the old lossy
20-candidate/512-byte restrictions, but its coverage still depends on the
regular LMDB supplied. The import route preserves keys and candidate bytes
from the known append-value curated RocksDB format and removes exact
duplicates. It does not recover candidates previously truncated from a
bounded source, re-prove imported key/function equivalence, or swap control
bytes. Preserve the source's provenance and decoded control convention.

**Optional structural sieve.** After a composite build you can generate a
smaller store which favors structural variety:

```bash
target/release/build_curated_full sieve \
  EXISTING_COMPOSITE FRESH_SIEVED_COMPOSITE \
  --shingle 6 --keep-all-below 1000 --cell-floor 1
target/release/build_curated_full audit FRESH_SIEVED_COMPOSITE
```

These are the defaults. Keys with at most 1000 candidates pass through;
larger pools are traversed shortest-first, rejecting shared relabeled
six-gate subwords in either direction, while retaining a floor per
`(gate count, wire count)` cell. The floor may keep candidates which share
a subword. This is deliberate pruning; use the unsieved source if retaining
every built candidate is the objective.

### Freeze and validate the curated result

Whichever construction you choose, audit its completed composite output,
then run all three conversion stages against **that same source**:

```bash
target/release/build_curated_full audit FRESH_COMPOSITE
target/release/frozen_from_lmdb tables FRESH_COMPOSITE FRESH_CURATED_OUT --curated --composite
target/release/frozen_from_lmdb write FRESH_COMPOSITE FRESH_CURATED_OUT --curated --composite
target/release/frozen_from_lmdb validate FRESH_COMPOSITE FRESH_CURATED_OUT --curated --composite
target/release/frozen_filters_build from-frozen FRESH_CURATED_OUT
```

Despite its name, `frozen_from_lmdb --composite` reads RocksDB directly. This
is needed for uncapped curated construction: some friend lists exceed
LMDB's limit of $4\text{ GiB}-1$ for one ordinary value. A successful builder
compacts its output and writes a completion manifest with key count,
candidate count and a content digest. `audit` recomputes those values and
checks the existing manifest; it does not bless an interrupted partial
build. The frozen reader for composite sources requires that completion
metadata. Keep the chosen construction's command, input provenance and audit
output with the resulting store.

For smaller sources, optional `to-lmdb INPUT_COMPOSITE FRESH_CURATED_LMDB`
and `validate-lmdb INPUT_COMPOSITE CURATED_LMDB` materialize and compare
`curated_00..curated_ff` databases before freezing with `--curated` alone.
Direct `--composite` freezing avoids that intermediate per-value size limit.

The first `tables` command creates its output directory, which must not
already exist. `write` refuses existing shards, and filter construction
refuses an existing `filters.bin`. An interrupted output is partial; move it
aside and use a fresh output path. `validate` compares every decoded frozen
value with the source bytes. This checks the conversion, but does not by
itself prove that every source circuit was filed under the correct function.
The current identity-derived curated generator verifies candidate keys as it
creates them; legacy import, regular merge and export preserve and check
record framing without performing a complete semantic audit of imported data.

Finally, set the finished directory in your local TDP recipe:

```toml
[database]
regular_dir = "/path/to/regular-frozen"
curated_dir = "/path/to/new-curated-frozen"
curated_control_order = "native"
lookup_miss_filter = "auto"
```

Use `legacy-swapped-controls` only when the imported artifact actually needs
it. Normal identity-derived current builds use native ordering. Check the
paths with `cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run`
before beginning a new run. Construction and validation can be substantial
offline work; compiling the tools alone does not produce the database.

## Where this lives in the code

| File | What to read it for |
| --- | --- |
| [`src/database/frozen.rs`](../src/database/frozen.rs) | File layout, key splitting, filters, decoding, and native/legacy control order. |
| [`src/database/codec.rs`](../src/database/codec.rs) | Circuit record format and composite build keys. |
| [`src/database/lookup_cache.rs`](../src/database/lookup_cache.rs) | Repeated lookup caching and minimum-direction policy. |
| [`src/stages/db_mixing/replacement.rs`](../src/stages/db_mixing/replacement.rs) | Canonical keys, curated/regular routing, friend selection, and wire placement. |
| [`src/engine/mixer/replacement.rs`](../src/engine/mixer/replacement.rs) | Verification and insertion into the live circuit. |
| [`db_gen/regular.rs`](../db_gen/regular.rs) | Regular generation, merging, and LMDB export. |
| [`db_gen/curated_full.rs`](../db_gen/curated_full.rs) | Curated construction and completion audits. |
| [`db_gen/frozen_build.rs`](../db_gen/frozen_build.rs) | Shared frozen writer and source comparison. |
| [`db_gen/bin/frozen_filters_build.rs`](../db_gen/bin/frozen_filters_build.rs) | Building and validating filters from frozen keys. |
