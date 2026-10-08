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

*Pending: two new seeds per configuration per scale (A and L8 at n=64 and
n=128), each run end to end (gen → mix → pack G=512 → repair), with the
serialized deliverable (`deliverable.bin`, gzip/xz) and evaluation timing.*

## 6. Recommended configuration

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
(pair cancellation, leak slides, merge/split fallback), then ANF emission.

## 7. Limitations and open items

- **Anchors remain.** ~200 (n=64) / ~700 (n=128) values are still seen 100+
  times; the remaining class is migration hops, which no bracket design
  reaches.
- **Attack scope.** C-view and linkability cover exact affine predictors and
  exact value equality. Low-degree nonlinear predictors were not tested on
  the final artifacts.
- **Excursion singletons** need value-level (not gate-level) cancellation.
- **Engineering.** New stage-2 knobs default off; no managed preset yet;
  chains / premask / seed-avoidance have no unit tests beyond end-to-end
  verification; stage 4 exists only as Python prototypes.

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
