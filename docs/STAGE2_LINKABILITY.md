# Stage 2 Redesign and Linkable Exposure

*2026-10-07. Branch `ran/stage2-windowed-delivery`. Every number below is
recorded in `docs/notes/ncdb_ab_20261001_RESULTS.md` (lab notebook) and was
produced by the prototypes in `scripts/windowed_delivery/`. Validation
replicates: §5.*

---

## 1. The pipeline: four stages

| stage | what it does | code |
|---|---|---|
| 1. Sandwich | classic sliced sandwich around the secret C | `stages/sandwich` (unchanged) |
| 2. Preprocessing | embedded masking: masked fires, band, migration, guards | `stages/preprocessing` |
| 3. Mixing | DB mixing with the **regular** frozen store | `circuit_mixer --tdp --db-mixing` |
| 4. Packing | windowed ANF delivery: pack, repair, emit | `scripts/windowed_delivery` (Rust port planned) |

The curated DB and the former stages 4–6 (split, crossing walk, fcompress)
are historic: the new pipeline does not use them (see
`docs/ANF_WINDOW_DELIVERY.md` §3 for why windowing replaced stages 4–6).

## 2. What the adversary sees, and what we measure

The deliverable is a sequence of ANF windows. An adversary sees the full wire
state at every window boundary (and the inputs), not the gates inside a
window. All metrics below are computed on the **boundary lattice** of the
final artifact, exactly (no sampling).

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
(`link_census.py`):

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

Recommended pipeline, stage 4 = pack at G = 512 + repair. A = no chains;
L8 = bounded chains. All rows: C-view at the exact floor, no window over
65,536 monomials, spot checks and functional equivalence pass.

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

## 5. Validation (replicates)

Three seeds per configuration per scale (the runs in §4 plus two new seeds
each), every one run end to end with `scripts/windowed_delivery/waist/val_chain.sh`
(gen → mix → pack G=512 → repair → serialize). All twelve: forward and reverse
verification pass; the repaired circuit is functionally equivalent to the mixed
one; no window exceeds 65,536 monomials; the C-view is at the exact floor; the
span check is zero at every sampled stage-2 cut; and the end-to-end check
`verify_zero_slice` (C(x) against the final circuit's payload wires, 512 random
honest-slice inputs) passes.

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

The validation found one bug in the prototype repair: the slide step
re-located the breaker after moving the completer and picked up the moved gate.
It is fixed in `order_repair.py`; all replicates were re-repaired with the
fixed code (identical results except vL128a, which now reaches the floor).

## 6. Recommended configuration

**Managed default (2026-10-09).** This configuration is the default of the
managed driver (`tdp_gen`, config v2, `docs/tdp_pipeline.md`): preset `L8`
sets exactly the stage-2 environment below and nothing else, schedule
`validated` is the stage-3 command below (serial), and stage 4 is
`tdp_packing run` with the parameters below. From the same seeds the
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
--no-local-verify` (regular frozen store). The validation runs also passed
`--p-comp 1.0 --p-any 0.1`; those flags fed only the crossing walk's slot-3
thermostat, which never fires at `--p-db 1.0`, and they are retired (the
current build rejects them; the output is byte-identical without them).

Stage 4: `window_pack3.py A s3 512 512 24576 3000` then `repair.sh TAG 512 N`
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
  refactor (`tests/integration/stage2_typed_params.rs`). Stage 4 is in Rust
  (`tdp_packing`), with the Python prototypes as its oracle.
- **Small n.** L8 generation works at n=8 (18,238 gates, 1,734 chained
  brackets) and n=16; the stage-4 fixtures and the reproduction tests run
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
