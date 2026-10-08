# Windowed ANF delivery: design, measurements, and the stage-2 band redesign

*2026-10-04. Working notes from the ANF-window program (RC's design
direction; experiments in `runs/ncdb_ab_20261001/`, summarized in that
run's RESULTS.md). Status: measurement-backed design; nothing here is
committed pipeline code yet.*

## 1. The proposal

Replace stages 4–6 (split, crossing, fcompress) with a single
delivery-form stage applied to the stage-3 output:

1. reorder commuting gates (function-preserving by construction);
2. partition the circuit into contiguous windows;
3. deliver each window as the canonical vector ANF of its touched wires
   as polynomials over its input wires.

The delivered artifact is a sequence of window ANFs. Nothing else ships.

## 2. Why this erases more than the move system can

Every exposure metric in the 2026-10 program (per-value visibility, t\*,
reader law, closure law) quantifies over *cuts of the delivered
spelling*. A window has no interior cuts, so a reference value is
exposed only if it is **alive across a window boundary** — its carrier
span must straddle the cut. Consequences:

- **Read values become hideable.** The reader law (≥1 reader ⇒ ~100%
  visible) presumed a cut between every pair of gates. If a value's
  birth and all its bare reads sit inside one window, no observable cut
  ever carries it.
- **The closure law is sidestepped, not violated.** Interior rewrites
  close because function preservation forces them to; windowing does
  not rewrite between cuts, it deletes the cuts.
- **Within-window text is annihilated.** Canonical ANF is the strongest
  form of the "un-invertible respelling" that the stage-3 ledger showed
  is mixing's real contribution: templates, k-grams, gate provenance,
  stage-2 vocabulary signatures all vanish inside windows.
- **What remains exposed** is exactly: (a) values alive at boundaries,
  (b) the boundary-value equality-linkage graph, (c) the outputs.

## 3. Established facts (all measured on the TDP run, reference =
stage-2 circuit, full-resolution cuts, function checks 256/256)

Tools in `runs/ncdb_ab_20261001/`: `waist_census.py`,
`interval_census.py`, `ideal_refresh_census.py`, `window_pack{,2,3}.py`,
`anf_compose.py`, `boundary_cluster.py`, `carrier_refresh.py`.

1. **Input choice: stage-3 output, not stage-5.** Crossing's ×3.45
   inflation welds the conflict DAG (median forced interval 6 → 26
   gates); its 15% carrier erasure does not pay for it. Split adds
   carriers. The whole post-3 pipeline is dominated by windowing on raw
   stage-3 output.
2. **Boundary *placement* in the delivered order is worthless** (spans
   long, waists flat: optimal ≈ uniform). All the leverage is in the
   *reorder*: a value's minimal bare residency is the DAG interval
   I(birth → next same-target writer), median 6 gates on stage-3
   material (×21 under the spelled span).
3. **Commutation structure (MPMCT):** same-target writers commute;
   readers pin against *all* later writers of the read wire. Pairwise
   RAW/WAR edges are unsound; schedulers must use epoch conditions and
   taints must use the any-tainted-reader/writer rule.
4. **Joint packing progression at stage-3, ground truth** (% of the
   105.7k carried stage-2 values hidden): naive windows 5%; v1
   interval-burst scheduler 21%; v2 value-level all-or-nothing 32%;
   v2.1 35%; v3 (ANF-budget-adaptive windows) 46% at G=64, 58% at
   G=128, **72% at G=256 on migrated material — 5.2% of the full 571k
   stage-2 vocabulary still exposed** (band 23.1%, data 59.2%, mid
   77.0% of their carried values). Empirical window-size law: ~+14 pts
   hidden per doubling of the gate cap, with the ANF budget never
   binding (median ~750 monomials in 512-gate windows); the limit on
   window size is architectural (boundary count vs the ends,
   resynthesis-hardness floor, delivery size ×12), not combinatorial.
5. **ANF budgets are not the binding constraint.** Real composed
   windows use median ~200–400 monomials of multi-thousand budgets,
   zero overflows, zero functional mismatches (every window
   re-verified). Deliverable size ×4–9 vs the gate list: this format
   buys erasure, not compression.
6. **Residual structure after windowing:** (a) ~8k big-interval +
   ~6.5k long-span values are *reader-spread* — the ideal-refresh
   census shows refresh-after-last-read barely helps them; only
   re-masking *between* reads does; (b) the boundary equality-cluster
   graph keeps its band mega-clusters (~1.2k runs vs froth ceiling
   ~50) — windowing shrinks the proxy population, not the linkage;
   (c) data/mid stragglers (91–96% of their carried values) are welded
   by bare reads — they need consumer-fused masking, not scheduling.

## 4. Value/wire separation (carrier migration)

Stage 2 currently binds each value lineage to a fixed wire for the
whole computation: data wires hold data values, band wires hold band
values, cradle to grave. That stability is what makes carrier spans
long, intervals welded, and wire-lineage/function-lifetime attacks
possible.

**Design rule: values are not resident; they are in transit.** A
carrier may sit bare on a wire only inside one window. Between uses it
travels masked:

- after its birth envelope, the carrier is re-encoded (v → v⊕r);
- before each reader, it is restored inside the reader's window and
  re-encoded again after;
- before its true overwrite it is restored one last time.

Bare residency collapses to ≤1-cut envelopes around birth, each read,
and death — each with a tiny DAG interval, hence windowable. The
mask-rotation prototype (`carrier_refresh.py`) implements exactly this
post hoc with CCX pairs (t ^= c1c2 done/undone while c1,c2 are
write-stable, rotating pairs when stability runs out, never bare at a
rotation), is function-preserving (verified at full scale, +24% gates),
and needs no free wires.

Measured end-to-end (refresh → repack, G=128): winnability rises from
86% to **93% of carried values**, and the populations no other
mechanism reaches move exactly as predicted — data exposure 82.6% →
66.7%, mid 90.0% → 81.1%. Total exposure however stayed flat (41.8%):
a migrated value carries 3–5 short spans, the packer must hide all of
them, and sibling-conflict deaths ate the gain. The binding constraint
is now the scheduler (value-unit planning, section 8), not the
material.
Wire *relocation* (move v⊕r to a different wire, retire or re-seed the
old one) is the stage-2-native form of the same discipline and
additionally breaks positional lineage; exposure-wise the two are
equivalent, so the cheap in-place form is the right prototype and the
relocating form is the right generator design.

**Honest caveat:** the prototype's masks are degree-2 functions of
values visible at the *same* boundary, so a quadratic adversary can
reconstruct them. They defeat the exact-affine and equality-cluster
metrics in use (and genuinely break cluster linkage), but production
masks must be chained band material whose own carriers rotate — the
same mask-hygiene stage 2 already applies to data values.

## 5. The stage-2 band-subsystem redesign spec

Keep: the data path. Data values have zero bare readers (1225/1234),
C-view sits at the output-only floor from stage 2 onward. Validated;
do not touch.

Redesign the band/carrier subsystem to satisfy, in order of leverage:

1. **Carrier lifetime budget**: every band/mask value is overwritten
   within a bounded DAG distance of its last use (target: interval ≤
   the window cap). Measured by `interval_census.py`.
2. **Transit re-encoding / migration**: carriers cross window
   boundaries only re-masked (section 4). Measured by the packer's
   ground-truth exposure.
3. **Carrier dedup**: no value reused across distant consumer groups;
   the ×3/×6 mask-copy discipline goes away. Target: max boundary
   equality-cluster < froth ceiling (~20). Measured by
   `boundary_cluster.py`.
4. **Entangled refresh**: masks and refresh values must depend on
   secret-carrying wires (pinnable ancilla randomness collapses to
   slice constants — the fresh-wires result).
5. **Consumer-fused corrections** where the gadget algebra allows
   (needed for the welded data/mid stragglers).
6. **Band seeding density** (RC, 2026-10-04): current band seeding is
   far too sparse; densify substantially. Separate work item — it
   interacts with (1)–(4) by shortening natural carrier gaps and
   enlarging the mask vocabulary, but is independently implementable.

Generation-time placement should make intervals small *by
construction* (birth, readers, overwrite DAG-adjacent), rather than
relying on the packer's reorder to rediscover locality.

## 6. Division of labor (updated)

- **Stage 2**: semantic hiding — which values ever exist bare
  (data path: zero bare reads; band: lifetime budget + migration +
  dedup + density).
- **Stage 3**: un-invertible respelling, dilution, DB-form digestion
  (judged by text metrics, per the stage-3 ledger).
- **Window packing (new stage 4)**: cut deletion — where anything is
  observable at all; plus canonical within-window respelling.
- **Ends (sandwich/slicing)**: exposure complexity of survivors (the
  t\*-spectrum belongs to the ends; unchanged by this program).

## 7. Implementation plan

Prototype-validated, pending Rust:

1. `window_pack` as a pipeline stage (port of `window_pack3.py`):
   epoch scheduler, winnable-value intervals, drain+burst+contagion,
   ANF-budget-adaptive close, emits reordered circuit + boundary list.
2. `anf_emit` (port of `anf_compose.py`): per-window sparse ANF,
   GF(2) toggle semantics (beware dedup-vs-cancel), per-window
   re-verification, canonical monomial/wire ordering, binary format.
3. Band-refresh/migration: near-term as a stage-3.5 (the
   `carrier_refresh.py` transform, with band-material masks instead of
   same-cut pairs); long-term inside stage-2 generation per section 5.
4. Metrics to CI: packer ground-truth exposure, boundary cluster
   maxima, ANF size/overflows.

## 8. Open questions

- Scheduler gap (now the single binding constraint): v3 hides ~58% of
  carried while the refreshed material supports 93%. Death counters
  attribute the loss to burst-time closes and, after migration, to
  sibling cascades. Scheduler v4 spec: treat a value's whole span set
  as the planning unit — reserve all of its intervals before bursting
  any, abandon the value at the first unwinnable conflict and release
  capacity immediately — plus conflict-graph pre-grouping of
  overlapping intervals.
- Mask strength: upgrade prototype masks to chained band carriers;
  quantify against the approximate/nonlinear-predictor flank (open
  thread #1 of the exposure program).
- Boundary randomization policy (boundaries are new visible structure).
- Window width vs re-synthesis: small windows are canonically
  re-synthesizable; floor on window size, or deliberate window-size
  jitter.
- Interaction with the curated-DB rethink (moot: curated DB removed 2026-10-08) and with the sliced
  deliverable / ends program.
