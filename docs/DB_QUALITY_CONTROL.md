# Gate-wise quality control during DB mixing

`fmix --leakage-repair` runs an optional audit-and-repair pass after the ordinary mixing
walk and final float, before the db_mixing circuit and checkpoint are saved.
It implements the proposed hot-wire surgery without modifying the general
window sampler, its candidate selection, or its random stream. It is off by
default. The first version runs once at this stage boundary; it is not a
periodic hook inside the random walk.

## Running it

Use the frozen store and original reference appropriate to the experiment:

```bash
export FROZEN_DB_DIR=/path/to/regular
export FROZEN_CURATED_DIR=/path/to/curated
# Also bounds the existing canonicalizer's branch search.
export CANON_RULE_L_BRANCH_CAP=512
export CANON_MONOMIAL_CAP=200000

target/release/fmix --input gadget.mpmct1 --gss --db-mixing \
  --moves 100000 --output mixed.mpmct1 \
  --leakage-repair --leakage-repair-reference original.mpmct1 --leakage-repair-report mixed.qc.txt
```

The reference must share the mixed circuit's input/wire coordinates. A
narrower reference uses the same numbered input wires; extra mixed wires
receive independent random bits, including dirty borrowed wires. Do not pass
an unencoded original against encoded gadget inputs, or use this uniform-input
audit as a fixed-slice audit. Those experiments need an explicit input adapter,
which this version does not implement. `--leakage-repair-reference-format g57` also works.
Without `--leakage-repair-reference`, the immutable input to this mixing run is the reference.
For a gadgetized input that is a different audit target from the original C.

To repair a previously mixed circuit without additional random moves:

```bash
target/release/fmix --input mixed.mpmct1 --moves 0 --skip-final-float \
  --p-db 0 --p-comp 0 --p-any 0 --output repaired.mpmct1 \
  --leakage-repair --leakage-repair-reference original.mpmct1
```

QC opens the store even when the ordinary DB move probabilities are zero.
`--leakage-repair` currently refuses split and resume modes: run it on the db_mixing circuit
file before those stages. It can follow piecewise db_mixing mixing; auditing and
repair then run serially on the assembled whole circuit.

In `scripts/gss_mix.sh`, set `DB_QC=1` to enable this pass in stage 3. Set
`DB_QC_REFERENCE` to a compatible original mpmct1 reference and optionally
`DB_QC_SEED`. The report is `RUN/db_mixing.leakage_repair.txt`. Existing artifacts are still
skipped; use a fresh run directory or `--run-rerun-from-stage 3` to test QC on an existing
run. Changing QC does not change stages 1-2 or the stage-3 sampler.

## What is tested and repaired

A value segment runs from a gate writing wire `w` to the next gate writing
`w`; reads of `w` do not change its value. The detector samples these internal
segments across the entire circuit. Input/output fringe segments have no pair
of bounding gates and are reported separately.

Two empirical tests mark a segment hot:

* An exact affine fit on training inputs expresses its value as an XOR of
  selected original internal wire values plus a constant. It must also meet
  the validation accuracy on separate held-out inputs (default: exact).
* Its producer gate's firing predicate correlates with a selected original
  firing predicate. The signed Pearson/phi correlation must have consistent
  sign and pass the threshold in both partitions. Both outcomes must appear
  enough times; near-universal non-firing alone is not a correlation.

Firing evidence concerns a gate's transition, not its target value. Assigning
it to the outgoing target segment makes the repair block include that gate.
Final target-writer firing predicates are also audited and reported, although
they have no outgoing internal segment to repair. Candidate screening includes
them so moving a hot firing into the last writer does not silently count as
success.

For each hot segment, the pass computes the smallest order-convex block
containing its two endpoints in the `XGate::collides` dependency DAG. Thus
commuting gates can be omitted, while alternate dependency paths force their
intermediate gates into the block. Span, gate and support limits are checked
before DB lookup. Gathering is planned on a copy and justified by commuting
swaps; failed attempts do not rearrange the live circuit.

The separate QC lookup examines regular and curated entries in both canonical
directions, allowing equal-size, smaller or larger replacements. It limits
candidate records, gate/support sizes, polynomial work, and decoded store
data. Exact ANF equivalence is mandatory before a candidate can be applied.
The ordinary DB lookup modes and cache policy are unchanged.

Each candidate is screened in the full affected span, including both seams,
segments crossing that span and borrowed wires. A candidate is refused if
another tested hot signal remains there or coverage is incomplete. This is
conservative: an unrelated existing hot signal in that span may also prevent
acceptance. A successful splice preserves directions, generations, ancestry,
litters, indexes and checkpoint state. Segment indices are refreshed after
every accepted repair.

## Budgets and interpretation

Defaults are deliberately small for a first experiment:

| Option | Default |
| --- | ---: |
| `--leakage-repair-seed` | 20803 |
| `--leakage-repair-sample-batches` | 4 training + 4 held-out batches of 64 inputs |
| `--leakage-repair-reference-segments` | 32 affine reference features |
| `--leakage-repair-reference-firings` | 128 reference predicates |
| `--leakage-repair-scan-segments` | 2048 sampled internal segments per scan |
| `--leakage-repair-correlation` | 0.98 absolute phi |
| `--leakage-repair-max-attempts` / `--leakage-repair-max-repairs` | 32 / 8 |
| `--leakage-repair-max-span` / `--leakage-repair-max-block-gates` | 256 / 12 |
| `--leakage-repair-max-replacement-gates` / `--leakage-repair-max-support` | 24 / 24 |
| `--leakage-repair-max-candidates` | 64 examined DB records per block |

Set either reference-feature cap to zero to disable that test. Weaker firing
signals require lowering `--leakage-repair-correlation` and increasing sample batches;
for example, `--leakage-repair-correlation 0.3 --leakage-repair-sample-batches 16` is an exploratory
setting, not a calibrated significance threshold. Low-variance predicates
with fewer than eight minority outcomes per partition are skipped. The Rust
`DetectorConfig` exposes this minimum and the trace-storage budget.

The report records configuration, reference path, detailed evidence, scan and
reference coverage, before/after counts, an independent final `fresh_audit`,
and one outcome per attempted segment:

| Outcome | Interpretation |
| --- | --- |
| `repaired` | Exact equivalent candidate accepted; affected region passed configured probes |
| `block_limit` | Span, convex-block gate count or support exceeds its cap (structural failure) |
| `examined_replacements_hot` | Tested usable alternatives still expose a detected signal |
| `no_db_entry` | No key found for the block in the checked stores/directions |
| `candidate_budget` | Candidate examination stopped at its cap |
| `lookup_or_verification_limit` | Store/canonicalization/equivalence work could not finish within limits |
| `screening_limit` | A candidate's affected region could not be fully screened |
| `no_usable_replacement` | Entries exist but no distinct, fitting, verified candidate was available |
| `application_refused` | Live-mixer invariant prevented a screened replacement |

These outcomes concern the minimal convex block and bounded choices examined.
They do **not** establish that every larger containing block or every possible
scratch-wire placement fails. An overlong span is a search-budget failure,
not proof that the mathematical minimal block has too many gates. This version
does not yet search enlarged containing blocks or address either proposed
failure mode automatically.

Coverage flags matter: a global scan samples segments and reference features;
before/after counts on edited circuits are not counts of all leakage. The
held-out bank is separate from fitting but reused while selecting repairs.
The final `fresh_audit` uses another input bank, never used to select repairs,
while keeping reference-feature selection fixed. Residual hot boundary firings
are reported even if the internal repair queue is empty. Assess effectiveness
with larger independent experiments and broader reference coverage as well.
Passing these probes is not a security proof.

## Validation

The focused Rust tests exercise affine held-out rejection, sparse-predicate
correlation, boundary/seam screening, convex gathering, bounded DB enumeration,
exact equivalence and preservation of mixer metadata/checkpoints. Small
synthetic in-memory DB fixtures test the full repair controller without loading
a production frozen store. For Linux/WSL:

```bash
cargo test --lib quality
cargo test --lib db_mixing::db_replace::tests::qc_
cargo test --lib db_mixing::frozen::tests::qc_
cargo check --bin fmix
bash -n scripts/gss_mix.sh
```
