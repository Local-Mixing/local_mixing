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
| Frozen files | Serve regular and curated replacements during GSS database mixing. |

The ordinary GSS build therefore does not need RocksDB or LMDB. The offline
builders are enabled with `--features db-tools`. There is no runtime fallback
to an old LMDB directory if the frozen files are missing.

## What we store

Let $P(C)$ be the canonical output polynomials of a circuit $C$. The function
key is computed as

$$
k = \operatorname{XXH3\text{-}128}(\operatorname{serialize}(P(C))).
$$

The serialization and wire mapping are described in
[Polynomial Canonicalization](POLYNOMIAL_CANONICALIZATION.md). The regular
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
[Local Mixing Documentation](Local_Mixing_Documentation.pdf).

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

During [step 3 of GSS](GSS_PIPELINE.md), we first canonicalize the sampled
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

Set the directories in the `[database]` section of your GSS recipe:

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
[README](../README.md) for running GSS with a recipe.

## Constructing and freezing a store

Build the offline tools from the repository root:

```bash
cargo build --release --features db-tools
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

For the full curated construction, the composite RocksDB uses
`[function key][circuit bytes]` as its key. This deduplicates complete
function/circuit pairs without building one enormous mutable value. The
identity source below must contain all accepted `id_g0` through `id_g33`
databases:

```bash
env -u CANON_MONOMIAL_CAP -u CANON_RULE_L_BRANCH_CAP \
  target/release/build_curated_full from-identities IDENTITY_LMDB FRESH_COMPOSITE
target/release/build_curated_full audit FRESH_COMPOSITE
target/release/frozen_from_lmdb tables FRESH_COMPOSITE FRESH_CURATED_OUT --curated --composite
target/release/frozen_from_lmdb write FRESH_COMPOSITE FRESH_CURATED_OUT --curated --composite
target/release/frozen_from_lmdb validate FRESH_COMPOSITE FRESH_CURATED_OUT --curated --composite
target/release/frozen_filters_build from-frozen FRESH_CURATED_OUT
```

Despite its name, `frozen_from_lmdb --composite` reads RocksDB directly. This
is needed for the uncapped curated construction: some friend lists exceed
LMDB's limit of $4\text{ GiB}-1$ for one ordinary value. The audit records
completion and a content digest before the freeze stages accept the source.

The first `tables` command creates its output directory, which must not
already exist. `write` refuses existing shards, and filter construction
refuses an existing `filters.bin`. An interrupted output is partial; move it
aside and use a fresh output path. `validate` compares every decoded frozen
value with the source bytes. This checks the conversion, but does not by
itself prove that every source circuit was filed under the correct function.
The current curated generator verifies candidate keys as it creates them;
regular merge and export preserve and check record framing without performing
a complete semantic audit of imported data.

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
