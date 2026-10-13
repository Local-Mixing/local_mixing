# Windowed ANF Delivery: Packing the DB-Mixed Circuit into ANF Blocks

*2026-10-07. Status: prototype-validated pipeline stage (python tools
snapshotted in `scripts/windowed_delivery/`, originally
`runs/ncdb_ab_20261001/`; now the packer's oracle), production artifacts at
n=64 and n=128 (`waist/final64/`, `waist/final128/` in that untracked run
tree). Update 2026-10-12: packing is the first sub-step of stage 6 (delivery)
of the six-stage pipeline and runs in Rust (`tdp_packing run`,
`docs/STAGE4_RUST_DESIGN.md`, written when it was stage 4), bit-exact against
these tools, which remain its oracle; it is followed by unpacking with
lattice refinement (`tdp_unpacking run --refine`, `docs/DELIVERY_STAGE.md`),
whose resynthesized circuit `unpacking/unpacked.mpmct1` is the published
object (its `unpacking/deliverable.tdp4`, the refined deliverable, is an
internal record; the packer's deliverable itself ships only under
`delivery.anf_only`). Current pipeline:
`docs/tdp_pipeline.md`, `docs/TDP_PIPELINE_REFERENCE.md`. All measurements
cited here are recorded in `docs/notes/ncdb_ab_20261001_RESULTS.md` (the
run's RESULTS.md) and in `runs/exposure_report_20261003/REPORT.md`
(untracked run tree).*

---

## 1. The current process

The packing sub-step of stage 6 (delivery) consumes the **last circuit
produced** (`crossing.mpmct1` by default; the stage-3 DB-mixed circuit
`db_mixing.mpmct1` under `run.skip_stages = [4, 5]`, the validated
four-stage chain this document measured) and produces the packer's
deliverable, the windowed ANF form from which the unpacking sub-step
resynthesizes the published circuit `unpacking/unpacked.mpmct1` (under
`delivery.anf_only` the packer's deliverable is itself what ships; how the
stage list came to be this one: `docs/local_mixing_history.md`).
Three steps:

### 1.1 Commuting reorder + window scheduling (`window_pack3.py`)

The gate order of the mixed circuit is re-scheduled within its exact
commutation freedom, and simultaneously partitioned into contiguous
**windows**. Scheduling and partitioning are one optimization because
the reorder is where all the leverage is (§2.3).

**Commutation model (MPMCT).** Same-target writers commute; a reader
pins against *all* later writers of the wire it reads. Pairwise
RAW/WAR edges are unsound for this gate set; the scheduler therefore
uses epoch conditions: per wire, a write counter; a reader of wire *c*
is ready iff `writes_done[c]` equals its required epoch, a writer iff
all earlier-epoch readers of its target have been emitted. Function
preservation is by construction and is additionally verified (256
bit-sliced input samples, all wires) on the emitted order.

**Value-aware planning.** Before scheduling, the stage-2 reference
circuit A and the mixed circuit B are simulated on 4,096 shared random
inputs; every intermediate value is fingerprinted. This yields the
**carried values** of B (values of A that re-appear on B's wires) and
their **spans** (birth write → next write of the carrier wire). A value
is *hidden* in the delivered artifact iff **none** of its spans
contains a window boundary; hiding is therefore all-or-nothing per
value, over all of its spans (typically ~3, up to several hundred).

**How the blocks are determined.** The partition is *not* a greedy
accumulation of gates; greedy filling is only the background activity.
The window structure is driven by **interval reservations and atomic
bursts**, in five steps:

1. **Interval extraction (per span).** For each span (birth write *b* →
   killing write *e*) of each carried value, compute the span's
   *two-sided tainted closure*: a forward taint sweep from *b* (a gate
   joins if it reads a tainted-written wire or writes a tainted-read
   wire — the sound any-tainted rule, since pairwise RAW/WAR edges are
   unsound for MPMCT), intersected with a backward taint sweep from
   *e*. The result is the minimal gate set that must share a window
   for that span to be hidden — its *members*. A span whose member set
   exceeds `W_cap` gates disqualifies (one rescue is attempted:
   extend to the next-next writer of the carrier wire). A value is
   **winnable** iff *all* of its spans have member sets ≤ `W_cap`;
   values with any raw span longer than `span_cap` are disqualified
   outright.

2. **Readiness bookkeeping (epochs).** Every gate carries an unmet
   count: a reader of wire *c* becomes ready when *c* reaches the
   write-epoch the original order gave it; a writer of wire *t* when
   all earlier readers of *t* are emitted. This is the exact
   commutation freedom — the scheduler can emit any ready gate at any
   time without changing the function.

3. **Drain phase (the greedy part).** Intervals are processed by
   ascending end position. Between bursts the scheduler emits
   **filler**: ready gates that belong to *no* living interval
   (protection count zero). Filler fills windows with material whose
   placement is exposure-irrelevant.

4. **Atomic bursts + value-level accounting.** When an interval's turn
   comes, the window is first checked: if the remaining gate capacity
   is below the interval's size (+ a pad of 4) or the composed ANF is
   above ¾ of the monomial budget, the window is closed *pre-emptively*
   so the burst is never split by a scheduled close. Then **all member
   gates of the interval are scheduled consecutively** (dependencies
   pulled from the ready heap as needed) inside the one window.
   Hiding is all-or-nothing per value: the first touched span commits
   the value; if any span dies — a window closed mid-burst, or a
   member was consumed earlier as filler/another burst
   (`resumed_later`) — **all sibling intervals of that value are
   killed** (`kill_value`) and their reserved gates are released to
   the filler pool immediately, so no capacity is spent on
   half-hidden values. After each burst, **contagion** completes any
   committed interval with ≤ 24 remaining members in place while the
   window is still open.

5. **Budget-adaptive close.** The close condition measures the *real*
   deliverable cost: the window's sparse ANF is composed incrementally
   gate-by-gate during scheduling, and the window closes when the
   monomial count exceeds the soft budget `B_mono` (hard `2·B_mono`
   inside a burst) or the gate count exceeds the soft target `G`
   (hard `2G` inside a burst). Every interval death is attributed
   (`drain_close` / `burst_close` / `resumed_later` / `sibling`), which
   is how scheduler losses are diagnosed (§2.4).

After the main pass, remaining gates are emitted by readiness, the
final window closes, the emitted order is re-verified functionally,
and the ANF emission stage (§1.2) re-composes each window
independently against the larger emission budget with per-window
verification.

### 1.2 ANF emission (`anf_compose.py`)

Each window is delivered as the canonical algebraic normal form of its
touched wires: for every wire written inside the window, the exact
GF(2) polynomial of its value at the window's end as a function of the
window's *input* wire values. Composition is sparse and exact, with
true GF(2) cancellation (per-monomial toggle — symmetric-difference
dedup is a measured bug, caught by re-verification). Monomials and
wires are canonically ordered; the within-window gate text, gate
count, gate order, and all intermediate spellings are discarded.

Budget: 16,384 monomials per window. Measured usage is far below it:
median 184 (n=64) / 207 (n=128) monomials per window, p90 ≈ 290–430,
maximum 7,673, **zero overflows** at both scales. [Corrected 2026-10-09:
`anf_compose.py` spot-checks every 500th window on 256 random states; only
`anf_dump_eval.py`'s 1,024-lane evaluation covers every window. The Rust
port (`tdp_packing`) checks every window: decoded ANF = composed ANF, and
ANF = gates, exhaustively for windows with at most 16 support wires and on
1,024 sampled states otherwise.]

### 1.2a Commutation-aware repair (`scripts/windowed_delivery/waist/order_repair.py`, `scripts/windowed_delivery/waist/repair.sh`; Rust: `stages::packing::repair`)

*Added 2026-10-07; see `docs/STAGE2_LINKABILITY.md` §3.6.* Runs on the
packed order before emission in every packing run (`repair.sh` in the
oracle chain; `stages::packing::repair` in every `tdp_packing run`, not
optional). Window boundaries are markers that stay between the same
surviving gates, and every step is a legal rewrite of the gate sequence
(two gates commute iff neither's target is a control of the other).

1. **Pair cancellation.** Delete pairs of identical gates that commute
   adjacent: do/undo pairs the mixer leaves in place. Iterated to a
   fixed point; removes about 7% of delivered gates at both scales.
2. **Leak slides.** For each interior C value still exposed, the
   predictor's support is a fixed set of wire segments; the relation
   holds from the write that completes it to the write that breaks it.
   Slide the completer forward and the breaker backward by commutation
   until no boundary separates them.
3. **Fallback.** Merge away any remaining exposing boundary, and split
   any window over 65,536 monomials at an exposure-free cut
   (`merge_bounds2.py`, `anf_split2.py`), iterated to a fixed point.

The repaired circuit is checked functionally equivalent to its input
(`verify_equiv.py`, 1,024 random full states), then emitted as in §1.2.

### 1.3 Serialization

The body (`deliverable.bin`) is the sequence of window ANFs in a binary
format (wire ids, monomial lists over window inputs), wrapped in the
container described next. Under `delivery.anf_only` that container is what
ships, and nothing else: no gate list, no boundary metadata beyond the
windows themselves, no stage provenance, and none of the circuits (not C,
not the stage-1/2 structure, not the stage-3/4/5 circuits, not the packer's
gate order). In a default six-stage run the container is an internal
record: the unpacking sub-step resynthesizes every window from it, and what
ships is the resynthesized circuit `unpacking/unpacked.mpmct1` alone, a
gate list in canonical order with no window lattice, no ANFs and no stage
provenance; the window ANFs, the bounds and the per-round files stay in the
secret-holding environment.

**Container v1 (2026-10-09).** The raw body carries no n, no version and no
integrity check, so the managed pipeline wraps it in a container
`deliverable.tdp4`: the packer writes `packing/deliverable.tdp4` (the shipped
file under `delivery.anf_only`, otherwise the refinement's starting point) and
the unpacking sub-step writes `unpacking/deliverable.tdp4` on the refined
lattice, the refined deliverable (an internal record: the published object of
a default run is the circuit `unpacking/unpacked.mpmct1` resynthesized from
it), in the same format: a 48-byte
little-endian header (`LMTDPANF`, format version 1, header size 48, flags 0,
source wires n, state wires 4n, I/O layout 1 = the classic zero slice, window
count, body length, CRC-32 of the body, CRC-32 of the header) followed by the
**unchanged** body, which is also kept as `deliverable.bin` so existing
readers keep working. `tdp_packing decode --format container` validates the
header and the body (ascending supports, changed wires and masks, mask bits
below the support size, every support index used); the format is specified
in `docs/STAGE4_RUST_DESIGN.md` §6.

| | n=64 (\|C\|=1,000) | n=128 (\|C\|=3,000) |
|---|---|---|
| stage-2 circuit | 771,029 g / 256 w | 2,240,757 g / 512 w |
| stage-3 (mixed) input | 1,648,679 g | 4,803,546 g |
| windows | 10,702 | 28,986 |
| ANF payload | 24.6 MB (×3.88 vs gate list) | 93.7 MB (×5.06) |
| deliverable.bin | 27.5 MB | 103.1 MB |
| xz-compressed | 4.52 MB | 15.3 MB |

Evaluation cost (python reference evaluator, 1,024-lane bit-sliced):
0.68 / 1.70 ms per input in gate form, 0.85 / 2.54 ms in ANF form,
bit-exact agreement between the two forms at both scales.
Compressibility is 6.1–6.7× under xz — and the compressed ANF is
*smaller* than the compressed gate list, i.e. the ×4–5 raw inflation
is redundancy of the format, not information added.

### 1.4 Parameters of the 2026-10-07 reference runs (final64/final128, G=128)

*Superseded for production by the validated configuration of
`docs/STAGE2_LINKABILITY.md` §6: stage-2 preset L8 (chains, premask,
seed-avoid-C1), packing G=512 / W_cap 512 / budget 24,576 / span 3,000 /
split budget 65,536 (`configs/tdp.toml` `[packing]`).*

Stage-2 knobs are identical at both scales (the managed embedded-
masking preset): K=2 band wires per LGI, `max_open` 3 / `min_open` 2,
quad-fire, balanced masks and balanced seed, rich band seed
(`seed_fill_nl` 4), feistel slice guards (all gates ≤ 2 controls),
carrier shuffling dose 8 with **persistent migration**
(`EMBEDDED_MASKING_SHUFFLING=8`, `..._RETURN_HOME=0`). Stage-3 flags
are identical: `--tdp --db-mixing --profile 2,2,2,2,2 --p-db 1.0
--p-comp 1.0 --p-any 0.1 --db-max-degree 9 --db-max-span 30
--db-wire-terms 1024 --db-total-terms 2048`, regular frozen store
`frozen_m1_m11` (`--p-comp`/`--p-any` are inert at `--p-db 1.0` and the
validated stage-3 schedule omits them; stages 4 and 5 pass
`--p-comp 0 --p-any 0`),
moves = ⌈2·2·1.3·(stage-2 gates)⌉. Packing and
emission are identical: `W_cap` 128, `G` 128 (hard 256), `B_mono`
6,144 (hard 12,288 in a burst; ¾ pre-burst guard), `span_cap` 3,000,
contagion threshold 24, pad 4; emission budget 16,384
monomials/window; 4,096 fingerprint lanes. Fresh CSPRNG seeds per run,
recorded in each run directory.

| parameter | n=64 run | n=128 run |
|---|---|---|
| source C | 64 wires, 1,000 gates | 128 wires, 3,000 gates |
| sandwich (|D|, s, slice gates) | 1,000 / 384 / 1,280 | 3,000 / 896 / 2,560 |
| sandwich circuit | 2,832 g / 128 w | 7,920 g / 256 w |
| stage-2 output | 771,029 g / 256 w | 2,240,757 g / 512 w |
| mixer moves | 4,009,351 | 11,651,936 |
| stage-3 output (wall time) | 1,648,679 g (~26 min) | 4,803,546 g (~76 min) |
| carried / winnable | 145,481 / 131,226 (90.4%) | 426,891 / 382,071 (89.5%) |
| hidden by construction | 125,592 (95.7% of winnable) | 337,934 (88.4%) |
| windows | 10,702 | 28,986 |
| window monomials med/p90/max | 194 / 301 / 7,682 | 207 / 332 / 2,846 |
| exposed distinct at boundaries | 23,347 (16.0% of carried) | 89,072 (20.8%) |

---

## 2. Rationale, tradeoffs, and what the process optimizes

### 2.1 The threat model that motivates windowing

**What the adversary sees.** In the default six-stage pipeline, a circuit
and nothing else: `unpacking/unpacked.mpmct1`, the canonical-order
concatenation of the resynthesized windows. Nothing else ships: not C, not
the sandwich or the masked circuit of stages 1-2, not the mixed, split or
crossed circuits of stages 3-5, not the packer's gate order, not the window
lattice, not the window ANFs. The adversary can cut the circuit anywhere,
compute the function of any segment, compose segments and resynthesize them
into circuits of its own; it is never handed the lattice or the ANFs, and it
does not see how a window was spelled before resynthesis (the packer's
spelling is gone; what it holds is the synthesizer's spelling). Under
`delivery.anf_only`, the four-stage chain this document measured, the
adversary holds the packer's deliverable instead: the lattice of windows
and, for each window, the ANF of the permutation it applies. From the ANFs
it can derive the wire state at every window boundary (on inputs of its
choice), compose adjacent windows into wider ones, and resynthesize any
window or merged window into a circuit of its own; it cannot see how the
function was spelled inside a window.

Every exposure metric in the 2026-10 program — per-value visibility,
the t\* exposure census (exact affine predictors), the equality-cluster
attack — was measured on that ANF artifact and therefore quantifies over
**cuts of the delivered spelling**: the wire values at the window
boundaries. Two measured laws made gate-granular delivery look
fundamentally leaky:

- **Reader law**: any value with ≥1 bare reader is ~100% visible under
  every-gate cuts (measured 3.2 / 55.4 / 63.8 / 100.0% visibility at
  fanout 0/1/2/≥3 on the TDP, stage 3).
- **Closure law**: function preservation forces interior basis changes
  to close (the second bracket undoes the first), so **no self-closing
  interior move can raise the min-over-cuts predictor weight at any
  dose** — a CNOT/transvection twist sweep that bloated the circuit
  ×37 left t\*=1 mass at 84%. The interior move system contains no
  weight-raising operator; the t\* spectrum belongs to the circuit
  ends.

Windowing attacks the quantifier instead of the values. In the ANF
artifact (`delivery.anf_only`) **a window has no interior cuts**: a value
is exposed only if it is alive across a window boundary. Deleting cuts
(a) makes read values hideable for the first time — the reader law
presumed a cut between every pair of gates; (b) sidesteps the closure
law — nothing is rewritten between cuts, the cuts are deleted; and (c)
annihilates all within-window text: templates, k-grams, gate provenance,
stage-2 vocabulary signatures, and the gate count itself vanish into a
canonical polynomial. What remains observable at the lattice is exactly:
values alive at boundaries, the boundary equality-linkage graph, the
outputs, and the lattice itself. In the default pipeline the published
circuit's windows *do* have interior cuts again — the synthesizer's
gates — so the lattice bounds nothing there; what windowing buys is that
the packer's and the mixer's spelling never ship and every window is
re-spelled from its function alone (the synthesizer sees no gate: the
separation guarantee, `docs/DELIVERY_STAGE.md`). The measurements that
model this adversary are the all-cuts C-view census of the resynthesized
circuit (at n=64 the unpacked circuit, resynthesized at the delivered
lattice, exposes 5 of 936 interior C values at its own cuts; the raw packed
circuit 0) and the uniform re-cut-and-resynthesize experiments, which
re-cut the circuit uniformly and never use the delivered lattice: merging
and resynthesizing uniform W=2000 windows of the n=64 crossing output
recovers 244 of 936 interior C values (241 with a 10× time budget;
canonical reorder changes it by at most 2%), while the raw circuit exposes
none at any cut; stages 4 and 5 do not close this leak, and the
refinement's exposure-aware cuts are each re-scanned so that no interior C
value is affine at a new cut (`docs/notes/stubborn_windows_20261012.md`,
`docs/notes/stage45_revival_20261011.md`; `docs/TDP_PIPELINE_REFERENCE.md`
§6b).

### 2.2 The objective function

The packer maximizes the number of **distinct carried stage-2 values
hidden** — ground truth measured by fingerprint simulation on the
emitted order, not a proxy. Secondary structure it respects:

- value-level all-or-nothing (a value with one exposed span is
  exposed; partial hiding is worthless);
- the C-view invariant: the source computation's interior must stay at
  **zero** exposure (measured: the output-only floor — exactly the 64 /
  128 outputs visible, all t\*=1 — holds through the entire delivery
  stage at both scales);
- ANF budget and window caps (digestibility and size).

### 2.3 Where the leverage is: reorder, not boundary placement

Measured on the TDP: delivered-order carrier spans are long and waists
are flat — optimal boundary placement in the *given* order is
approximately uniform placement, worth almost nothing. But a value's
minimal bare residency under commuting reorder is its **DAG interval**,
median 6 gates on stage-3 material — **×21 shorter** than its spelled
span. The entire mechanism is: reorder to compress each value's bare
residency to its DAG interval, then drop windows around the intervals.

### 2.4 The window-size tradeoff

The one real tradeoff surface is window size G:

- **Hiding rises ~+14 points per doubling of G** (measured sequence,
  migrated material: 46% hidden @G=64, 58% @G=128, 72% @G=256 — 5.2%
  of the full stage-2 vocabulary still exposed at G=256).
- The **ANF budget never binds**: median monomial usage is ~200–400 of
  multi-thousand budgets even at 512-gate windows, zero overflows
  anywhere in the program. The limit on G is architectural, not
  combinatorial: boundary count vs the circuit ends, the
  re-synthesis-hardness floor (very small windows are canonically
  re-synthesizable; very large windows reduce the boundary count
  toward the ends) [2026-10-12: no longer a floor — the unpacking
  sub-step of stage 6 resynthesizes every window by design,
  `docs/DELIVERY_STAGE.md`], and delivery size (≈ ×12 at G=512).
- Scheduler quality, not material, is the current binding constraint:
  on migration-refreshed material, 93% of carried values are winnable
  while v3 hides 72–73% of winnable; the gap is attributed by the
  death counters to burst-time closes and sibling cascades (scheduler
  v4, value-unit planning with conflict-graph pre-grouping, is the
  specified next step; estimated ceiling 94%).

Packer evolution, same ground truth (% of carried values hidden,
stage-3 input, W=20 era → G-era): naive uniform windows 5% → v1
interval-burst scheduler 21% → v2 value-level all-or-nothing 32% →
v2.1 35% → v3 budget-adaptive 46/58/72% at G=64/128/256.

### 2.5 Costs accepted

- **Size** ×4–5 raw (×≈0.9 after compression, see §1.3): the format
  buys erasure, not compression.
- **Boundaries are new visible structure** in the ANF artifact: the
  window lattice itself is observable there (the default pipeline's
  published circuit is not handed the lattice). Measured consequence: a value recurring ≥5 times
  among the carried values is *always* boundary-exposed
  (deterministic at both scales; non-exposed maximum multiplicity
  is 4). This "recurrence law" shifted the program's focus upstream to
  stage-2 carrier hygiene (the celebrity program: migration,
  hot-targeting, threshold retirement), since no boundary placement
  can hide a value whose spans tile the stream (celebrity liveness
  was measured at ~110 of 256 wires at *every* cut position).
- **Affine-metric caveat**: all exposure numbers quantify exact affine
  predictors at boundaries; the approximate/nonlinear-predictor flank
  is an open thread of the exposure program (known empty for the
  QC-style detector: agreement ≥0.98 coincides with exact equality).

---

## 3. The packer's input: stage-3 output vs. the crossing output

*Measured 2026-10-03/04 (the stage-wise exposure study, nine arms, random
family and TDP, and the direct input-choice measurement of the windowing
program). The 2026-10-07 decision to pack a mixed circuit directly, and the
retirements that went with it, are recorded in `docs/local_mixing_history.md`
(October 8, 2026). Since 2026-10-11 stages 4 (splitting) and 5 (crossing)
precede packing in the default chain and the packer consumes the last
circuit produced; the stage-3 input survives as `run.skip_stages = [4, 5]`
(`docs/notes/stage45_revival_20261011.md`: stages 4 and 5 do not close the
merged-window resynthesis leak). Current pipeline: `docs/tdp_pipeline.md`,
`docs/TDP_PIPELINE_REFERENCE.md`.*

### 3.1 What splitting and crossing buy at the cuts (measured)

Per-stage visibility % [t\*≥2 share %], random-family arms, every-cut
grids (excerpt; full table in `runs/exposure_report_20261003/REPORT.md`):

| arm | s3 mix | s4 split | s5 crossing |
|---|---|---|---|
| random f=2 | 85.1 [1.5] | 84.0 [8.3] | 76.1 [8.9] |
| f=10 | 79.5 [1.7] | 79.0 [5.1] | 76.0 [6.1] |
| eq. arity-2 | 82.6 [1.3] | 81.6 [4.5] | 78.7 [6.0] |
| twists p=.05 | 87.0 [2.2] | 87.0 [5.8] | 82.7 [7.8] |

Stage signatures, replicated at TDP scale:

- **Stage 4 (split)** is a visibility-neutral t-raiser: it moves
  4–8 points of t\*=1 mass into small parities and erases nothing. In
  carrier terms it *adds* exposure surface: 105.7k → 107.9k carried
  values on the TDP.
- **Stage 5 (crossing)** is the only post-mixing eraser at the cuts:
  −3…−5 points of visibility in every arm (105.7k → 89.8k carried on
  the TDP).
- At TDP scale the chain is flat: stage-2-view visibility
  26.5 → 26.5 → 25.6% across s3 → s5. The C-view output-only floor is
  owned by stage 2 and never moves.

### 3.2 The direct comparison: windowing s3 vs windowing s5

The windowing program's first decision was its input. Measured:

- **Crossing welds the dependency structure.** Its ×3.45 inflation
  stretches the median DAG interval from 6 gates (s3) to 26 (s5) —
  ×4 — so the reorder lever that windowing lives on is largely
  destroyed. Per-value hideable ceilings at matched window size W=20:
  69.6% of carried values (s3) vs 46.9% (s5).
- **Crossing's erasure does not pay for the welding.** In absolute
  terms at W=20: ~32k exposed values on the s3 ceiling vs ~48k on the
  s5 ceiling — on the cut metrics, windowing the raw mixed circuit
  strictly dominates windowing the crossed one, despite crossing's 15%
  carrier erasure.
- **Split only adds carriers** (§3.1), and the walk family generally
  is exposure-hostile: the order-preserving greedy walk measured
  97.4% exposure against its own 79.4% floor — verbatim-dressing
  passes are exposure-*maximal*.

Against these, a single windowing pass hides 58–72% of carried values
(G=128–256) — an order of magnitude beyond crossing's −3…−5 points —
while *deleting* within-window text rather than respelling it. On the
crossing output the packer needs about 4× the windows and far longer
(`docs/TDP_PIPELINE_REFERENCE.md`, "What each stage contributes").

### 3.3 Division of labor

- **Stage 2** — semantic hiding: which values ever exist bare (data
  path at zero bare reads; band carrier hygiene: lifetime, migration,
  dedup/retirement, seed density).
- **Stage 3** — un-invertible respelling, dilution, DB-form digestion.
- **Stage 6, delivery (windowed ANF packing, then unpacking with lattice
  refinement)** — cut deletion: *where*
  anything is observable at all, plus canonical within-window
  respelling (the ANF artifact); for the published circuit of the default
  pipeline, the re-spelling of every window from its function alone.
- **Circuit ends (sandwich/slicing)** — the t\*-spectrum of the
  survivors (interior moves provably cannot move it).

---

## 4. Production reference results (pipeline v2.1, final64/final128)

*Status: results of the 2026-10-07 G=128 runs (final64/final128); superseded
by the G=512 results and their replicates in `docs/STAGE2_LINKABILITY.md`
§4–5 and the managed defaults (`docs/tdp_pipeline.md`).*

End-to-end: classic sandwich → embedded masking (rich band seed,
persistent carrier migration dose 8, read-pressure partners, feistel
slice guards) → db_mixing → window-pack G=128 → per-window sparse ANF.
Function checks green at every stage, both scales; all windows
re-verified; zero budget overflows; fresh CSPRNG seeds recorded in the
run directories.

| leakage parameter | n=64 | n=128 |
|---|---|---|
| C-view (the secret) | 6.4% = exactly the 64 outputs, t\*=1 | 4.3% = the 128 outputs (+2 interior t\*=4) |
| stage-2-view visible (boundary lattice, exact) | 27.3% (t\*=1 22.1, t\*≥2 5.2) | 28.4% (23.6 / 4.7) |
| distinct values exposed at boundaries | 23,347 = 16.0% of carried = 3.8% of vocabulary | 89,072 = 20.8% / 5.0% |
| class uniformity (data/mid/band) | 26.5 / 27.3 / 27.7% | 28.3 / 27.2 / 28.7% |
| non-exposed max multiplicity | 4 | 4 |
| max boundary equality cluster (proxy) | 342 | 397 |

The residual exposed mass is dominated by high-recurrence band
vocabulary (the celebrity class, 330/966 values of multiplicity ≥257),
which is a **stage-2 birth phenomenon**, not a packing phenomenon:
boundary redrawing was measured infeasible (celebrity spans
median 494 / p95 2,505 packed gates, liveness ~110 wires at every cut
position), and any fix operates upstream of this stage (the lever then
in progress, threshold retirement of band carriers at the emitter, was
tested and dropped because it only redistributes rebirths; the levers
kept are chained brackets, premask and seed avoidance, preset L8:
`docs/STAGE2_LINKABILITY.md` §3.3, §4, §6). The delivery stage's own
contract — C-view at the output-only floor, within-window annihilation,
recurrence-law behavior at boundaries — is the same either way.

---

## 5. Artifacts and tools

- Design doc: `docs/windowed_delivery_design.md` (architecture + the
  stage-2 band-subsystem spec).
- Tools: `scripts/windowed_delivery/{window_pack3.py,anf_compose.py,boundary_cluster.py,wire_balance.py}`
  and `scripts/windowed_delivery/waist/{recurrence2.py,anf_dump_eval.py}`
  (snapshot of `runs/ncdb_ab_20261001/`, the packer's oracle; Rust:
  `src/stages/packing/`, `tdp_packing`); exposure scans via `hmap_affine
  --exposure --g-cuts` (boundary-lattice mode).
- Production artifacts: `runs/ncdb_ab_20261001/waist/final64/`,
  `.../final128/` (untracked run tree in the main checkout: stage-2/3
  circuits, packed order + bounds, deliverable.bin + compressed forms, full
  census logs, seeds).
- Measurement record: `docs/notes/ncdb_ab_20261001_RESULTS.md` (the run's
  RESULTS.md); stage-wise study: `runs/exposure_report_20261003/REPORT.md`
  (untracked run tree).
