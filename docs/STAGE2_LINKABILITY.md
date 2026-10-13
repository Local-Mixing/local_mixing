# Stage 2 Redesign and Linkable Exposure

*2026-10-07; on `main` (2026-10-12). Every number below is
recorded in `docs/notes/ncdb_ab_20261001_RESULTS.md` (lab notebook) and was
produced by the prototypes in `scripts/windowed_delivery/`. Replicate
results: §5.*

---

## 1. The pipeline: six stages

*(Revised 2026-10-12. This document was written for the four-stage chain of
2026-10-07, which is now `run.skip_stages = [4, 5]` with
`delivery.anf_only = true`; current pipeline: `docs/tdp_pipeline.md`,
`docs/TDP_PIPELINE_REFERENCE.md`.)*

| stage | what it does | code |
|---|---|---|
| 1. Sandwich | classic sliced sandwich around the secret C | `stages/sandwich` (unchanged) |
| 2. Preprocessing | embedded masking: masked fires, band, migration, guards | `stages/preprocessing` |
| 3. Mixing | DB mixing with the **regular** frozen store | `circuit_mixer --tdp --db-mixing` |
| 4. Splitting | every complemented g57 gate split into plain conjunctions, joined to distant brackets through absorbed NOT twists | `circuit_mixer --split --split-stop` |
| 5. Crossing | the directional crossing walk under a size thermostat | `circuit_mixer --resume split.state --target-size …` |
| 6. Delivery | packing (windowed ANF delivery: pack, repair, emit), then unpacking with lattice refinement | `tdp_packing run`, `tdp_unpacking run --refine` (`src/stages/packing/`, `src/stages/unpacking/`; Python oracle `scripts/windowed_delivery`) |

The four-stage chain measured here is `run.skip_stages = [4, 5]` with
`delivery.anf_only = true`; stages 4 and 5 joined the default chain on
2026-10-11 (`docs/notes/stage45_revival_20261011.md`). History of the stage
list: `docs/local_mixing_history.md`.

## 2. What the adversary sees, and what we measure

This document measures the four-stage chain (`delivery.anf_only`), whose
shipped file is the packer's deliverable: the lattice of windows and, for
each window, the ANF of the permutation it applies. Nothing else ships
there: not C, not the sandwich or the masked circuit of stages 1-2, not the
mixed circuit of stage 3, not the packer's gate order. From the ANFs the
adversary can derive the wire state at every window boundary (on inputs of
its choice), compose adjacent windows into wider ones, and resynthesize any
window or merged window into a circuit of its own; it cannot see how the
shipped function was spelled inside a window. The metrics below are the
boundary metrics: they are computed at every cut of the **boundary lattice**
of that artifact (no cut sampling).

In the default six-stage pipeline the published object is instead the
circuit `unpacking/unpacked.mpmct1`, the canonical-order concatenation of
the windows resynthesized by the unpacking sub-step; the adversary holds that
circuit and nothing else (not the lattice, not the ANFs, not the circuits of
stages 1-5, not the packer's gate order), can cut it anywhere, compute the
function of any segment, compose segments and resynthesize them, and never
sees how a window was spelled before resynthesis. The boundary metrics then
describe the internal refined deliverable rather than the published object;
that adversary is measured by the all-cuts C-view census of the
resynthesized circuit (at n=64 the unpacked circuit, resynthesized at the
delivered lattice, exposes 5 of 936 interior C values at its own cuts; the
raw packed circuit 0) and by uniform re-cut-and-resynthesize experiments,
which never use the delivered lattice: 244 of 936 interior C values at
W=2000 on the n=64 crossing output (241 with 10× the time; canonical
reorder changes it by ≤ 2%), 0 for the raw circuit at any cut; stages 4 and
5 do not close it (`docs/notes/stubborn_windows_20261012.md`,
`docs/notes/stage45_revival_20261011.md`).

**C-view (hiding C).** For every value computed by a gate of C, the smallest
number t\* of boundary wires whose XOR (plus a constant) equals it exactly, at
any boundary (`hmap_affine --exposure --g-cuts`). The **floor** is the
outputs: the last value on each C wire is trivially visible (64 at n=64,
128 at n=128). Target: exactly the floor, i.e. no interior C value is an
affine function of any boundary state.

**Linkability (thread ends).** A *run* is a maximal stretch of consecutive
boundaries at which one wire holds the same function (up to complement). The
*cluster size* of a value is its number of runs over all wires. A value is
*proxy* if it equals the output of some stage-2 gate, *froth* otherwise
(mixer-made). Values seen once are near-innocuous; values seen in ≥ 2
separate runs are recognizable from the artifact alone. Reported
(`scripts/windowed_delivery/waist/link_census.py`; Rust:
`stages::packing::census`, `packing/link_rep512.txt`):

- **linkable values (L2)**: proxy values with cluster size ≥ 2;
- **sightings (M2)**: total runs over those values;
- **linkable froth**: froth values with cluster size ≥ 2;
- **anchors**: proxy values with cluster size ≥ 100;
- **exposed**: distinct proxy values seen at all.

Functions are compared on 1,024 shared random inputs over *all* wires.

**Span check (stage 2 vs C).** At each stage-2 cut, the number of independent
affine combinations of C values that equal an affine combination of the
cut's wire values, beyond those explained by the inputs (and outputs)
(`hmap_affine SPAN_INTERSECT`, `SPAN_OUTPUTS_TRIVIAL`).

## 3. Findings

**3.1 Stage 2 reveals nothing about C beyond the outputs.** The span check is
zero at every interior cut for all 29 stage-2 variants tested, at both
scales, and zero at every tail cut once the outputs count as trivial.

**3.2 C leaks in the delivered artifact are made by mixing.** Dense stage-2
scans find no exposed C value; the mixed circuit does, and packing only
places boundaries near them. Two classes:

- *Interior leaks*: values that depend on all inputs, exposed at 1–3
  boundaries; caused mainly by retirement (early mask closes). Cheap to
  repair.
- *Fringe leaks*: C's first-layer gates (target and controls still raw
  inputs; `V = 1 + a + y + x·y`, 3 input bits). The predictor is raw input
  copies plus a wire holding the product `x·y` (from a band-seed product on
  the same pair, or the gate's own fire ancilla). Can stay exposed across
  thousands of gates. Fixed in stage 2 by `PREMASK` (mask every data wire up
  front: no raw copies; 31–45 of 64 wires were unmasked a thousand gates into
  the compute) and `SEED_AVOID_C1` (no seed products on first-layer pairs);
  residual cases are fixed by the repair pass (§3.6).

**3.3 Celebrities are bracket restores.** Long-lived band values are re-born
every time a dirty-ancilla bracket borrows and restores them (84% of births)
or a migration transfer copies them (16%). Migration drains are 0.8%, which
is why just-in-time drains changed nothing. The number of re-births is fixed
by the number of borrows: retirement (any threshold K) and a reserved ancilla
pool only redistribute them, turning a few heavy threads into many light ones
(L2 up, M2 down, total mass constant).

**3.4 Window size is the strong cheap lever.** Each doubling of the window
(G = 128 → 256 → 512) cuts sightings ~25% and linkable froth ~60%; total
monomials fall (fusion cancels terms). G = 512 is the practical maximum:
at 1,024 some windows reach hundreds of thousands of monomials.

**3.5 Chained brackets (restore-free design).** The first three gates of a
bracket `t ^= h·y; h ^= P·x; t ^= h·y` add `P·x·y` to `t` whatever `h` holds,
so brackets can share an ancilla and defer their restores. *Full chains*
(restore once per fire) remove the 257+ class at stage 2 but grow the
delivered ANF 2.4× after mixing (not before: a stage-2 proxy shows no blow-up)
because a displaced ancilla spans the whole fire. *Bounded chains*
(`CHAIN_LEN = 8`: an ancilla's restores are flushed every 8 brackets) keep
most of the gain at roughly A's size. *Rotating segments* (a fresh ancilla
pair per segment) cut stage-2 recurrence similarly but give no delivered
gain and +48–73% size.

**3.6 Commutation-aware repair.** On the final packed order: (1) cancel pairs
of identical gates that commute adjacent, a do/undo the mixer leaves behind
(≈ 7% of all delivered gates); (2) for each interior C leak, find the gates
that complete and break its relation (the predictor's support is a fixed set
of wire segments) and slide them toward each other by commutation; (3) fall
back to boundary merging and exposure-aware ANF splitting. Every repaired
artifact is functionally equivalent to its input. Example: a 128-wire fringe
leak that needed a ~2,600-gate window disappears after two slides.

**3.7 Not everything cancels.** 85% of linkable stage-2 singletons are
one-boundary excursions (a value perturbed across one boundary, then
restored), but they are restored by *different* gates: gate-level pair
cancellation removes only ~4% of linkable values.

## 4. Results

The four-stage chain (today `run.skip_stages = [4, 5]` with
`delivery.anf_only = true`), packing = pack at G = 512 + the order repair
that every packing run includes (now the first sub-step of stage 6). A = no
chains; L8 = bounded chains. All rows: C-view at the exact floor, no window
over 65,536 monomials, spot checks and functional equivalence pass.

| | exposed | linkable | sightings | linkable froth | anchors | size |
|---|---|---|---|---|---|---|
| n=64 old baseline (G=128, merge only) | — | 8,079 | ~134k | 2,804 | ~390 | 24.9 MB |
| n=64 A | 15,410 | 6,070 | 73,763 | 396 | 319 | 22.9 MB |
| n=64 **L8** | 11,769 | 5,299 | 58,514 | 1,124 | 194 | 27.0 MB |
| n=128 A | 54,494 | 18,573 | 252,069 | 2,318 | 987 | 103.0 MB |
| n=128 **L8** | 34,544 | 16,091 | 187,710 | 5,765 | 706 | 102.7 MB |

L8 vs A: linkable −13%, sightings −21% / −26%, exposed −24% / −37%, anchors
−39% / −28%, for +0.7k / +3.4k linkable froth (almost all seen exactly
twice) and +18% / 0% size (n=64 / n=128). Sizes are ANF-composer estimates;
§5 adds the serialized deliverables.

Levers tested and dropped (n=64, G=512 unless noted):

| lever | outcome |
|---|---|
| retirement K = 16–96 (+ warmup, safe, burst, repair, pin, early opens) | redistributes rebirths; L2 +45–60%; creates interior C leaks |
| reserved ancilla pool (`ANC_RESERVE`) | removes 100+ class, leaves ~4,800 values re-born 5–49×; L2 and froth up |
| migration dose 1 / 0 | −7% stage-2 gates but +50% deliverable size |
| full chains | ANF ×2.4 after mixing |
| rotating segments | no delivered gain, +65–73% size |
| just-in-time drains | celebrities unchanged; exposure 16 → 44% |
| boundary redraw (post hoc) | impossible: celebrity liveness ~110 at every cut |

## 5. Replicate results

Three seeds per configuration per scale (the runs in §4 plus two new seeds
each), every one run end to end with `scripts/windowed_delivery/waist/val_chain.sh`
(gen → mix → pack G=512 → repair → serialize; the reference driver of the
four-stage chain, today a parity tool). In all twelve: forward and reverse
verification pass; the repaired circuit is functionally equivalent to the mixed
one; no window exceeds 65,536 monomials; the C-view is at the exact floor; the
span check is zero at every sampled stage-2 cut; and the end-to-end check
`verify_zero_slice` (C(x) against the final circuit's payload wires, 512 random
honest-slice inputs) passes. These are the reference numbers for the stage-2
linkability of the four-stage chain; the six-stage chain has no replicate
figures yet (`docs/TDP_PIPELINE_REFERENCE.md`, status block).

| | exposed | linkable | sightings | linkable froth | anchors | `deliverable.bin` | xz | eval (ms/input) |
|---|---|---|---|---|---|---|---|---|
| n=64 A (×3) | 15.4k | 6.0–6.1k | 73.2–73.8k | 373–409 | 315–321 | 26.0–27.0 MB | 2.81–2.86 MB | 0.66–0.85 |
| n=64 L8 (×3) | 11.7k | 5.27–5.37k | 58.5–58.9k | 1,110–1,126 | 185–200 | 30.2–32.8 MB | 3.05–3.10 MB | 0.79–1.00 |
| n=128 A (×3) | 54.3–54.7k | 18.5–18.6k | 250–252k | 2,219–2,318 | 987–995 | 104.7–106.9 MB | 9.71–9.81 MB | 1.81–2.04 |
| n=128 L8 (×3) | 34.5–35.0k | 16.06–16.11k | 187–188k | 5,544–5,765 | 701–718 | 102.8–106.8 MB | 9.50–9.62 MB | 1.83–1.99 |

Replicates agree within 1–2% on every metric. L8 vs A (means, n=64 / n=128):
linkable −12% / −13%, sightings −20% / −25%, exposed −24% / −36%, anchors
−39% / −28%; linkable froth ×2.8 / ×2.4 (almost all seen exactly twice);
serialized size +20% / 0% (xz +9% / −2%). Evaluation timings are from the
Python reference evaluator (1,024-lane batched, ANF form).

The figures are those of the patched `order_repair.py` (the slide step
locates the breaker before moving the completer), the version the Rust
packer ports (`stages::packing::repair`); the unpatched slide picked up the
moved gate, and the difference shows only on vL128a, which reaches the floor
with the patched version.

## 6. Recommended configuration

**Managed default (2026-10-09; updated 2026-10-12).** The stages 1–3 and
the packer of the managed driver (`tdp_gen`, config v4,
`docs/tdp_pipeline.md`) default to this configuration; the default chain
additionally runs stages 4 (splitting) and 5 (crossing) and the unpacking
sub-step, so this four-stage chain is `run.skip_stages = [4, 5]` with
`delivery.anf_only = true`: preset `L8` sets exactly the stage-2 environment
below and nothing else, schedule `validated` is the stage-3 command below
(serial), and the packing sub-step of stage 6 is `tdp_packing run` with the
parameters below. From the same seeds the
managed run is byte-identical to `val_chain.sh` CONFIG=L8 (checked on the
archived vL64a and vL128a runs and by a reproduction test;
`docs/STAGE4_RUST_DESIGN.md` §12). Config A (no chains) is
`preprocessing.preset = "A"`. The source sizes the runs used (|C| = |D| =
1000 at n=64, 3000 at n=128) are the default `source.gates`.

Stage 2 (`gen_sandwich_gadget`):

```
EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0
EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1
EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8
```

Stage 3: `circuit_mixer --tdp --db-mixing --profile 2,2,2,2,2 --moves
2·2·|G|·1.3 --p-db 1.0 --db-max-degree 9
--db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048
--no-local-verify` (regular frozen store). The replicate runs (§5) also passed
`--p-comp 1.0 --p-any 0.1`; those flags fed only the crossing walk's slot-3
thermostat, which never fires at `--p-db 1.0`, and the validated schedule
omits them (the output is byte-identical without them); stages 4 and 5 pass
`--p-comp 0 --p-any 0`.

Packing (stage 6, first sub-step; stage 4 of the four-stage chain):
`window_pack3.py A s3 512 512 24576 3000` then `repair.sh TAG 512 N`
(pair cancellation, leak slides, merge/split fallback), then ANF emission
(`anf_dump_eval.py`); in the managed pipeline `tdp_packing run` with its
defaults, a bit-exact port.

## 7. Limitations and open items

- **Anchors remain.** ~200 (n=64) / ~700 (n=128) values are still seen 100+
  times; the remaining class is migration hops, which no bracket design
  reaches.
- **Attack scope.** C-view and linkability cover exact affine predictors and
  exact value equality. Low-degree nonlinear predictors were not tested on
  the final artifacts.
- **Excursion singletons** need value-level (not gate-level) cancellation.
- **Engineering.** The new stage-2 knobs are typed `EmbeddedMaskingParams`
  fields (off in the library's base parameters) and the managed presets
  L8/A set them; chains / premask / seed-avoidance have no unit tests beyond
  end-to-end verification and the byte-identity test of the typed-knob
  refactor (`tests/integration/stage2_typed_params.rs`). Packing (the first
  sub-step of stage 6) is in Rust (`tdp_packing`) and unpacking with
  refinement in `tdp_unpacking`, with the Python prototypes as the packer's
  oracle.
- **Small n.** L8 generation works at n=8 (18,238 gates, 1,734 chained
  brackets) and n=16; the packing fixtures (`tests/fixtures/stage4/`) and
  the reproduction tests run
  the whole L8 chain at n=8 and n=16 (calibration material, not validated
  sizes).

## 8. Stage-2 knob reference

| knob | default | status |
|---|---|---|
| `EMBEDDED_MASKING_PREMASK=1` | off | **keep** (fringe fix) |
| `EMBEDDED_MASKING_SEED_AVOID_C1=1` | off | **keep** (fringe fix) |
| `EMBEDDED_MASKING_CHAIN=k`, `CHAIN_LEN=L` | off | **keep** (k=2, L=8) |
| `EMBEDDED_MASKING_CHAIN_ROTATE=1` | off | drop |
| `EMBEDDED_MASKING_RETIRE=K` (+ `_WARMUP`, `_SAFE`, `_BURST`, `_MODE=repair`, `_PIN`, `_NOREPLACE`, `_PREOPEN`) | off | drop |
| `EMBEDDED_MASKING_ANC_RESERVE=R` | off | drop |
| `EMBEDDED_MASKING_JIT=gap` | off | drop |
| `EMBEDDED_MASKING_STRADDLE_CENSUS` | off | diagnostic |
