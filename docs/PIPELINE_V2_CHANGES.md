# What recipe v13 changes against `main` (2026-10-12 → 2026-10-26)

This document lists, setting by setting and file by file, the differences
between this release (`ran/pipeline-v2`, recipe v13, config schema 5) and
the previous `main` (4a85b09f, recipe v12, config schema 4). The previous
release's pipeline was six stages whose stage 6 was the windowed delivery;
this release keeps that pipeline intact as options and adds four things RC
asked for on 2026-10-26: a choice of random or balanced source circuit, the
three-mask stage 2 in its "v2" configuration, the ability to leave out stage
3 and stages 4+5 (already present, now documented and tested with the new
stages), and the adaptive, C-aware stage 6 v2 with an early-stop round cap.

## 1. Summary table

| | `main` (recipe v12) | this release (recipe v13) |
| --- | --- | --- |
| Source $C$ | random g57 gates, or supplied (`source.path`) | the same, plus `source.construction = "balanced"`: RC's balanced $C$ (every internal value read exactly twice), bit-exact with the campaign's Python builder |
| Stage 2 | embedded masking (presets L8, A, custom) or nonlinear291 | the same, plus `preprocessing.mode = "three-mask"` with the `[three_mask]` table (defaults = the campaign's "three-mask modified / v2"); outer blocks fixed to L8 |
| Stages 3, 4, 5 | unchanged; optional through `run.skip_stages` | unchanged; `run.skip_stages = [3]` and `[4, 5]` are exercised by the tests and documented (skipping 3 leaves stage 4 almost nothing to do) |
| Stage 6 | windowed delivery: `tdp_packing run` + `tdp_unpacking run --refine` | `delivery.mode = "adaptive"` (default): `tdp_adaptive run`, the C-aware adaptive resynthesis v2 with the 2 % two-round stop rule and `adaptive.max_rounds` as an early stop; `"windowed"` keeps the previous stage 6 unchanged |
| Published circuit | `unpacking/unpacked.mpmct1` | `adaptive/final.mpmct1` (windowed mode: as before) |
| Config schema | `config_version = 4` | `5`; a v4 file is refused with a migration message (set 5; add `delivery.mode = "windowed"` to keep the old stage 6) |
| Recipe manifest | `tdp_command_recipe=12` | `13`; new keys `source_construction`, `three_mask_*`, `delivery_mode`, `adaptive_*`; a v12 run is refused with a pointer |
| Stage-1/2 marker | `tdp_stage12_recipe=7` | `8` (adds `source_construction=` and the three-mask knobs) |
| Stage-6 marker | `delivery_recipe=1` | adaptive mode `delivery_recipe=2` (`delivery_mode=adaptive`, the binary's hash, `adaptive_params`, the keyed tags of the seed, the three inputs, the sandwich and `final.mpmct1`, `status`, `rounds`, `stop_reason`); windowed mode unchanged (`delivery_recipe=1`) |
| Production binaries | 4 (`gen_sandwich_gadget`, `circuit_mixer`, `tdp_packing`, `tdp_unpacking`) | 5 (plus `tdp_adaptive`); all built and fingerprinted by `tdp_gen` |
| Seeds | format 3 (gen, mix, split, cross, unpack) | unchanged; `unpack_seed` is the stage-6 seed in both modes (`tdp_adaptive` derives every round's and seed's synthesizer seed from it) |
| Environment registry | `EMBEDDED_MASKING_*`, `TDP_SOURCE_C`, … | plus `TDP_SOURCE_CONSTRUCTION` (driver) and the thirteen `THREE_MASK_*` knobs (recipe; any other `THREE_MASK_*` forbidden) |
| Validated chain pin (`pinned_recipe`) | skip 4,5 + `anf_only` + defaults | the same plus `delivery.mode = "windowed"` and `source.construction = "random"` |

Nothing of the previous pipeline was removed: every stage, binary, format,
oracle and test of `main` is still there; the embedded-masking, stage-3 and
windowed-delivery bytes are byte-identical for the same seeds (pinned by
`tests/integration/stage2_typed_params.rs`, `tests/shell/tdp_stage12_parity.bash`
and `tests/shell/tdp_defaults.bash`).

## 2. Stage 1: the source construction

* `src/stages/sandwich/balanced.rs` (new): `balanced_source(n, m, seed)`,
  the port of the campaign's `balanced_c.py` (`generate` + `g57`), itself a
  port of RC's `balanced_c_build.generate`: a sequential sampler in which
  every completed internal segment is read exactly twice as a control, no
  version is read more than twice and the target is always among the
  least-written wires, with rejection of immediate dead ends and up to 1,000
  attempts; `audit_balanced` checks the properties and runs before the
  circuit is used. The sampler uses the crate's CPython-exact MT19937
  (`packing::pyrand`, extended with `random()`, `_randbelow`, `choice`,
  `randrange`, `choices` with weights), so the reference seed 20261020091
  reproduces RC's published balanced $C$ gate for gate (pinned by a hash in
  `tests/unit/stages/sandwich/balanced.rs`).
* `SourceConstruction { Random, Balanced }`; `prepare_source` takes it as a
  fifth argument (a supplied path with `balanced` is an error). The random
  path is unchanged byte for byte.
* `gen_sandwich_gadget` reads `TDP_SOURCE_CONSTRUCTION`, appends
  `source_construction=…` to its banner and prints the audit line.
* Config: `source.construction` (`"random"` | `"balanced"`); flag
  `--source-construction`; manifest key `source_construction`; stage-1/2
  marker line.

Why: exposure of $C$'s values through the pipeline is per control read; the
balanced $C$ has no read-free values (which are nearly immune) and no value
read more than twice, which halves the gates with both controls exposed and
lowers the reads-weighted exposure, at the cost of a larger exposed count
([notes/c_design_exposure_20261021.md](notes/c_design_exposure_20261021.md)).

## 3. Stage 2: the three-mask mode

* `src/stages/preprocessing/three_mask/{mod,band,fire,mask_rank}.rs` (new):
  the port of the campaign generator crate `stage2_compute_pilot`
  (`ThreeMaskParams`, `ThreeMaskParams::v2`, `compute`), reusing the crate's
  `preprocessing_shuffling` (which the generator used unchanged).
* `PreprocessingMode::ThreeMask` (`"three-mask"`), `PreprocessingParams::ThreeMask`;
  `preprocess_sandwich` builds the outer blocks through the same helper as
  embedded masking (so they are identical for the same seed) and puts the
  three-mask compute between them; the band is $\max(2n, 128)$ wires
  (rounded up to a multiple of 16), with a reversible XOR ingress for the
  wires beyond the seeded $2n$.
* `gen_sandwich_gadget` accepts `three-mask` as its mode positional, reads
  the `THREE_MASK_*` knobs, refuses `EMBEDDED_MASKING_*` in this mode, runs
  `verify_payload` as before.
* Config: `[three_mask]` (13 keys, defaults = the v2 configuration:
  `band_wires 0 (auto)`, `banks 8`, `retire_budget 2`, `max_retiring 0`,
  `rotation 1`, `maturity 2`, `visit_sweeps 2`, `growth 8`, `warmup 4`,
  `segments 6`, `mask_count 3`, `random_helpers true`, `unique_cover true`);
  flags `--three-mask-*`; manifest keys `three_mask_params`, `three_mask_*`,
  `three_mask_outer_blocks=L8`.
* Parity: the gen output for n=32, |C| = |D| = 128, seed 1, band 128 and
  guard size 160 is byte-identical to the campaign generator's
  (`tests/integration/three_mask_parity.rs`). The managed driver passes the
  guard size `10 * 2n`, as it always did, so a managed three-mask circuit
  differs from the campaign's only in its two guards.

Why: at n=64 the three-mask compute is 35 % smaller than embedded masking's
and its stage-6 output leaks no internal value of $C$ to the census (0 of
936 against 384-424 for the baseline), with the "modified / v2" levers
(random helpers, unique cover) beating the as-is generator on every
resilience measure ([THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md)).

## 4. Stages 3, 4 and 5

Unchanged. `run.skip_stages` already accepted any of 2..6; the new tests run
the adaptive and windowed stage 6 after `[3]`, `[4, 5]` and the full chain.
Noted in the manual: with stage 3 skipped the split stage finds almost no
g57-form gate to act on, and the campaign found the three-mask circuit
without stage 3 ANF-incompressible in stage 6 (the synthesizer returned
larger blocks, the 2 % rule fired after two rounds).

## 5. Stage 6: the adaptive mode

* `src/stages/adaptive/` (new): `types` (intervals, census, source,
  plan), `census` (every-cut direct/affine exposure with the fresh-input
  audit), `boundary` (the sampled public-quadratic classification),
  `planner` (incremental ANF composition, boundary choice, parallel segments,
  the refinement profile), `synth` (compose a block, synthesize it with the
  unpacking synthesizer, check, peephole), `round` (plan, synthesize every
  block with every seed under size-scaled budgets, refine, census the
  candidates, accept smaller-and-not-leakier, assemble, check), `driver` (the
  rounds, the stop rules, the final circuit, the canonical order, the
  zero-slice and payload checks, the report). The algorithm and its
  deviations from the server prototype: [ADAPTIVE_STAGE.md](ADAPTIVE_STAGE.md).
* `tdp_adaptive` (new binary, `src/programs/tdp_adaptive/`): `run` and
  `census`; no environment reads; seed from `--seed-file`; exit codes 2, 3,
  4, 7, 8, 101 like `tdp_unpacking`.
* Config: `delivery.mode` (`"adaptive"` default | `"windowed"`), `[adaptive]`
  (15 keys: `max_rounds 0`, `stop_fraction 0.02`, `seeds_per_block 2`,
  `time_limit 60`, `restarts 16`, `beam 6`, `term_cap 131072`,
  `max_block_gates 65536`, `segments 16`, `avoid_radius 64`,
  `refinement_levels 2`, `min_refine_gates 128`, `census_batches 96`,
  `accept "smaller-not-leakier"`, `canonical_order true`); the mode decides
  which family of stage-6 keys is valid; flags `--delivery-mode`,
  `--adaptive-*`; manifest keys `delivery_mode`, `adaptive_params`,
  `adaptive_*`.
* `scripts/tdp_gen.sh`: the adaptive branch of stage 6 (the binary's
  invocation, `delivery_recipe=2`, the resume of an interrupted attempt
  through `adaptive/attempt.recipe`, `delivery.failed` with
  `step=adaptive`, the exit table); the windowed branch unchanged.
* `src/tdp`: `DeliveryMode`, `AdaptiveConfig`, `DeliveryStep::Adaptive`,
  the final-output path `adaptive/final.mpmct1`, `PRODUCTION_BINARIES` with
  `tdp_adaptive`.

Why: on the server campaigns the adaptive v2 stage 6 on the three-mask
circuit publishes a circuit whose census shows no internal value of $C$,
while the windowed delivery's measured basis (the n=64 seven-stage run) left
5 of 936 interior values exposed and resynthesis of the crossing output in
uniform windows recovers 244 ([tdp_pipeline.md](tdp_pipeline.md), stage 6);
the early-stop cap (`max_rounds`) lets a run end after a prespecified number
of rounds, as RC asked, with the smallest round output as its result.

## 6. Driver, tests and documents

* Config schema 5 / recipe v13 / stage-1/2 marker 8; migration messages for
  v4 configs and v12 runs.
* `tests/unit/tdp.rs`: every schema-5 expectation, the new tables, the mode
  exclusivity, the exit tables (`exit_code_tables_agree` covers
  `tdp_adaptive`), the registry entries.
* `tests/shell/*.bash`: the mock gained `tdp_adaptive`; `tdp_flags` covers
  the new flags and their rejections; `tdp_stage_resume` the adaptive
  marker, resume, parameter change, interrupted attempt, failures and
  allowed failure; `tdp_block_size` the adaptive chain; `tdp_seeds` the seed
  hand-over to `tdp_adaptive`; `tdp_defaults` the windowed four-stage chain
  with `delivery.mode = "windowed"`.
* New tests of the ports: `tests/unit/stages/sandwich/balanced.rs`,
  `tests/unit/stages/packing/pyrand.rs` (the CPython primitives),
  `tests/unit/stages/preprocessing/three_mask.rs`,
  `tests/integration/three_mask_parity.rs`, `tests/unit/stages/adaptive/*`,
  `tests/integration/tdp_adaptive.rs`.
* Documents: [ADAPTIVE_STAGE.md](ADAPTIVE_STAGE.md),
  [THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md),
  [PIPELINE_INTRODUCTION.md](PIPELINE_INTRODUCTION.md), this file;
  [tdp_pipeline.md](tdp_pipeline.md), [configs/tdp.toml](../configs/tdp.toml)
  and the README updated throughout.

## 7. Migrating a v4 recipe

```toml
config_version = 5           # was 4
[delivery]
mode = "windowed"            # add this line to keep the previous stage 6; omit it for the adaptive stage 6
anf_only = true              # the four-stage chain, as before (windowed mode only)
```

Everything else in a v4 recipe is valid unchanged. A run directory made by
the v12 driver is refused (resume it with that checkout); start a fresh
`run.directory` with the new driver.
