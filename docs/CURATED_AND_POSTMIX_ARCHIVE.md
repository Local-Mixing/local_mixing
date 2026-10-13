# curated-and-postmix archive: what exists only there

2026-10-08 · cleanup branch `ran/cleanup-four-stage` · inventory: `docs/notes/cleanup_inventory_20261007.md`

Status (2026-10-12): archive inventory as of 2026-10-08, historical; the 2026-10-11 note below records the revival of stages 4 and 5; the pipeline is six-stage since 2026-10-12 (`docs/tdp_pipeline.md`, `docs/TDP_PIPELINE_REFERENCE.md`).

## Summary

> **2026-10-11:** the split stage and the crossing walk were revived from this archive (commit `9499f42a` on `ran/seven-stage`: the reverse of cleanup steps 3 and 4, 3-way merged onto the stage-4 port) as stages 4 and 5 of the pipeline (seven stages then; six since 2026-10-12, when packing and unpacking became the one delivery stage 6: `docs/tdp_pipeline.md`); mixer states are v6 (the v3 layout). The curated DB and `fcompress` remain archive-only. The "Old stage 4: split" and "Old stage 5: crossing walk" tables below are therefore historical: that code is in the working tree again.

TDP had four stages at the time of the cleanup: 1 sliced sandwich, 2 preprocessing (embedded masking), 3 database mixing with the **regular** frozen store (`circuit_mixer --tdp --db-mixing --p-db 1.0`), and 4 windowed ANF packing (`tdp_packing`, the Rust port of the Python oracle under `scripts/windowed_delivery/`). The curated database and the old stages 4 (split), 5 (crossing walk) and 6 (`fcompress`) are historic. This doc lists what exists only on the archive branch, how to get it back, and why it was retired.

- **Where the archive is:** branch `ran/archive/curated-and-postmix`, tip `ea2e516f` (2026-10-08, "docs: stage-2 linkability write-up, repair pass, cleanup inventory"). It is the last commit of `ran/stage2-windowed-delivery` before the cleanup. At the time of writing it is a local branch; push it before relying on it from another machine.
- **What the cleanup did:** five steps on `ran/cleanup-four-stage`, each ending in a summary commit: step 1 curated DB (`04973782`), step 2 `fcompress` (`2dfaaca2`), step 3 split (`c4e6c0d6`), step 4 crossing walk (`ece32ceb`), step 5 driver, CI and docs. 46 files were deleted outright (about 14.4k lines of source, tests and docs) and about 100 more were edited.
- **What stayed byte-identical:** `gen_sandwich_gadget` output and `circuit_mixer` output for the stage-3 command (minus the inert `--p-comp`/`--p-any`), checked after every step on 300k-move A/B runs (serial, `--leakage-repair`, `--parallel-pieces 2`) against an `ea2e516f` build, plus an end-to-end `tdp_gen` run at n=16.
- **What is archive-only:** the curated store format, builders, runtime cascade and about 20 analysis tools; the split, crossing (both revived 2026-10-11, see above) and `fcompress` stages with their mixer machinery, CLI flags and tests; three demixing/benchmark probes; the old run formats (mixer states v1/v2, TDP recipe v8).

To get a single file back without switching branches:

```
git show ran/archive/curated-and-postmix:src/stages/post-processing/crossing.rs > /tmp/crossing.rs
```

To browse or build the whole archive side by side:

```
git worktree add ../lm-curated-postmix ran/archive/curated-and-postmix
```

**Old runs.** The current build refuses TDP runs that record `tdp_command_recipe=11` or older (v9, v10 and v11 with a pointer to the checkout that created them; v8 and older as unsupported, with a pointer to this archive branch) and every mixer state other than `STATE_VERSION` 6 (the v3 layout, since the 2026-10-11 revival); the current versions are recipe 12 and state 6. Resume v1/v2 states and recipe-v8 runs, or rerun the old stages 4-6, with a build of the archive branch (it reads states v1 and v2; v1 is a strict prefix of v2). States v3, v4 and v5 were written only by the four-stage interval (2026-10-08 .. 2026-10-11); neither the archive nor the current build reads them, so rerun those from the stage-3 input. The archive's `tdp_gen.sh` runs the old six stages (4 split, 5 crossing, 6 `fcompress`) and defaults to `stop_after_stage = 6`.

## Curated database

The curated store was a second frozen table built by splitting (approximately) minimal identities: for an identity I = AB, A = B⁻¹, so every prefix/suffix split, rotation and reversal gives a structurally different spelling. In the mixer it was a cascade consulted ahead of the regular store, in both MIX and COMP rounds (`--tdp` preset: `curated`, `curated_exhaust`, `curated_in_comp`).

| Feature | Archive location | Now |
| --- | --- | --- |
| Curated store codec and validation | `src/database/codec.rs`, `src/database/validation.rs` | Gone; `FrozenDb::open(dir)` reads the regular store only |
| Curated reader API (`get_curated`, `has_curated`, `get_curated_qc`, `CuratedKeyMode`, circuit/function key modes, control convention) | `src/database/frozen.rs`, `environment.rs`, `lookup_cache.rs` (namespaced cache) | Gone; the lookup cache has one namespace |
| Environment: `FROZEN_CURATED_DIR`, `FROZEN_CURATED_KEY`, `FROZEN_CURATED_*`, reference store `FROZEN_REF_DIR` | same, `src/engine/mixer/state.rs` | Gone; an exported variable has no effect |
| Replacement cascade: `circuit_key_of_window`, `curated_armed_for`, curated/regular-fallback arguments of `db_replace*`/`db_probe`/`qc_candidates*`, `DbResult.curated_matches`/`chosen_curated`, `ReplacementOptions.curated_key`, `QcCandidate.from_curated` | `src/stages/db_mixing/replacement.rs` | Regular path only, same order (forward, reverse, min-direction probe) |
| Mixer: `try_db_splice_curated` (refused a failed curated verification instead of panicking), descent pass loop, counters `db_curated_hits`/`db_curated_rejected`, `splice_sizes_curated`, `m123_class_hist`, `bigpool_hits` | `src/engine/mixer/replacement.rs`, `state.rs`, `reporting.rs` | `try_db_splice`, single pass |
| `circuit_mixer` flags `--curated`, `--no-curated`, `--curated-exhaust`, `--curated-in-comp`, `--curated-only-mix` (and `--no-` forms), `--frozen-curated-dir` | `src/programs/circuit_mixer/cli.rs`, `mod.rs` | Gone |
| `circuit_mixer --litter-in` (per-gate litter sidecar of the input, written only by `sgdb_substitute`), `Mixer::load_litters`, the sidecar option of `shuffle_mpmct` | `src/programs/circuit_mixer/cli.rs`, `mod.rs`, `src/engine/mixer/provenance.rs`, `security_tests/fixtures/shuffle_mpmct.rs` | Gone; input gates start as singleton litters |
| TDP: `database.curated_dir`, `database.curated_control_order`, curated manifest fields | `src/tdp/`, `configs/tdp.toml`, `scripts/tdp_gen.sh` | Keys rejected as unknown |
| Builders: `build_curated_full` (v2/v3 identity-split stores), `db_gen/curated_full.rs`, `CompositeShards`, `require_curated_full_manifest`, pool swap | `db_gen/bin/build_curated_full.rs`, `db_gen/curated_full.rs`, `db_gen/frozen_build.rs` | Gone |
| `frozen_from_lmdb --curated`, `--composite`, `--min-gates` | `db_gen/bin/frozen_from_lmdb.rs` | Gone; `--bands` stays |
| Test-only `canonicalize_polys_single_neg*` and its input-negation helper `substitute_input_negation` (with its two tests) | `src/canonicalization/window.rs`, `polynomial.rs` | Gone |
| Tests: curated cascade, regular-fallback and circuit-key replacement tests, `tests/unit/db_gen/curated_full/`, `tests/unit/db_gen/bin/build_curated_full/` | `tests/` | Gone; CI no longer runs `--bin build_curated_full` |
| Write-up `CURATED_DB_V3_SUMMARY` (`.md`, `.tex`, `.pdf`) | `docs/` | Gone |

Curated analysis tools (all `db_gen/analysis/` unless noted; Cargo `[[bin]]` entries removed):

| Tool | What it did |
| --- | --- |
| `curated_coverage_census`, `curated_size_census` | Coverage and size distribution of curated keys |
| `curated_key_filter`, `curated_key_histogram`, `curated_key_structure`, `curated_recanon_probe` | Key-level structure, skew and re-canonicalization checks |
| `pool_census`, `true_friend_probe` | Candidate-pool anatomy; true-friend (tight-length) census |
| `greedy_walk` | Order-preserving verbatim-dressing walk (measured exposure-maximal) |
| `inflate_gates` | Per-gate inflation of a circuit by DB equivalents (inflated-start ablation) |
| `frozen_pool_swap` | Pool swap between stores |
| `sgdb_build`, `mgdb_build`, `sgdb_substitute` | Single-/multi-gate DBs built from the curated store, and substitution with them |
| `plot_m1_3d.py`, `plot_m1_heat.py`, `sample_similarity.py`, `compare_similarity.sh` | Plots and similarity scripts for the above |
| `benchmarks/db_mixing/db_curated_probe.rs` | Curated lookup benchmark |
| `scripts/windowed_delivery/run_stage3.sh`, `curves.py` | Stage-3 driver with curated flags; curve plotting |

Kept and adapted (they run on the regular store): `identity_shingle_sieve`, `minimal_halves_probe`, `minimal_identity_filter`, `frozen_class_census`, `identity_length_census`, `cross_gluing_probe` (without `--partner-composite`), `blocker_census`, `db_match_synth`, `far_pair_probe`, `wide_yield_probe`, and `id_rewrite` (regular store only; `--curated`/`--regular` flags gone).

## Old stage 4: split

Split replaced every complemented r57 gate `a ^= b OR NOT c` by two disjoint plain conjunctions and joined fragments to distant brackets through absorbed NOT twists. It ran as a mixer mode (`circuit_mixer --split`) on the stage-3 output.

| Feature | Archive location | Now |
| --- | --- | --- |
| Split stage | `src/stages/post-processing/splitting.rs` | Gone |
| Mixer split slot, `took_split` guards, split dispatch in `twist_round` | `src/engine/mixer/scheduling.rs` | Gone |
| Side index (`comp_ids`, `wt_buckets`), rank state, wire canaries (`Tap`, `evict_taps`, leakage-repair tap re-anchoring), `MixStop::SplitDone` | `src/engine/mixer/state.rs`, `indices.rs`, `leakage_repair.rs` | Gone |
| MixParams `split`, `split_stop`, `p_split_twist`, `p_join`, `split_fail_limit`, `split_canaries`, `split_reach_k`, `span_norm`, `rank_base`, `rank_total`; eight split counters, `split_span_hist` | `src/engine/mixer/params.rs`, `state.rs` | Gone |
| Piecewise `Policy::Split`, `TapRec`, tap transport | `src/engine/mixer/piecewise.rs` | `Policy::Profile` and fixed-length `Policy::Moves` only |
| Checkpoint split line and `staps` section | `src/engine/mixer/checkpoint.rs` | Gone (state v4 and later) |
| Flags `--split`, `--split-stop`, `--p-join`, `--split-fail-limit`, `--split-canaries`, `--split-reach-k`, `--p-split-twist`, `--split-rounds-max` | `src/programs/circuit_mixer/cli.rs` | Gone |
| Tests: `tests/unit/engine/mixer/splitting.rs`, split checkpoint round trip, `golden_split_stop` (`GOLDEN_SPLIT_STOP`), piecewise split test | `tests/` | Gone |

## Old stage 5: crossing walk

The crossing walk shot fragments to their first true collider and rewrote each collision by one of three exact case-split rules (R1-R3). A thermostat held the size near a target with two contraction channels: exact undo of isolated crossings from a journal, and a pairwise merge catalogue over same-target gates, with a short tabu for fresh siblings. In the mixer it was slot 3 of `Mixer::run`, the default walk when `p_db < 1`. Stage 3 passes `--p-db 1.0`, so slot 3 never ran there; `random_bool(1.0)` draws no randomness, so removing it left stage-3 output unchanged.

| Feature | Archive location | Now |
| --- | --- | --- |
| Crossing stage | `src/stages/post-processing/crossing.rs` | Gone (the `post-processing` module is gone) |
| Slot 3 (thermostat), `expand_move`, `db_attempt_comp`, g57-only COMP path | `scheduling.rs`, `transformations.rs`, `engine/mixer/replacement.rs` | Gone; rounds no slot takes are idle (only when `p_db < 1`) |
| Merge catalogue (`Merge`, `merge_result`, `merge_key`) and merge-partner index | `transformations.rs`, `src/engine/mixer/indices.rs` | Gone |
| Crossing rules and presplit | `src/engine/moves/rules.rs` | Only `verify_rewrite` stays |
| Undo journal (`UndoEntry`), tabu ring, per-gate events (`Meta.event`, `next_event`), min-dgen cross pool, `child_gen`, `splice_replace_one` | `state.rs`, `provenance.rs`, `sampling.rs` | Gone (`Meta.dir` stays: it is drawn from the RNG) |
| MixParams `split_damp`, `split_base`, `dir_p`, `target_size`, `temp`, `merge_reach`, `journal_len`, `undo_frac`, `tabu_moves`, `w_cross`, `w_fresh`, `w_unsub`, `w_insert`, `w_twist_neg`, `w_twist_swap`, `w_twist_cnot`, `p_comp`, `p_any`, `p_comp_g57`, `s_db_g57`, `contract_ceiling`, `gen_split_inherit`, `p_mincross`, `cross_pool_k`, `cross_rescan`; 31 crossing/merge/g57-only counters | `params.rs`, `state.rs`, `reporting.rs` | Gone |
| Flags for all of the above, including `--p-comp`, `--p-any`, `--target-size`, `--temp` | `src/programs/circuit_mixer/cli.rs` | Gone; the store opens only when `p_db > 0` or `p_bridge > 0` |
| TDP `[crossing]` section (`xr`..`xmoves`), stage 5 in `tdp_gen.sh` | `src/tdp/`, `configs/tdp.toml`, `scripts/tdp_gen.sh` | Rejected with a pointer to this branch (revived 2026-10-11: `[crossing]` with `target_size_factor`, `width_penalty_base`, `width_penalty_threshold`, `size_tolerance_divisor`, `move_budget_factor`, `move_attempts`) |
| Tests: thermostat, crossing, erosion, directional, tabu-age, mincross, cross-union and journal-prune ancestry, split-generation, merge-catalogue, presplit, `cross_random_pairs`, `fig12`; golden `GOLDEN_THERMOSTAT_WALK` | `tests/unit/` | Gone. Tests that only used the walk as a driver now run the stage-3 DB path on an in-memory fixture store (`FrozenDb::from_memory`, test-only) |

`gauntlet_gen --mix` used the thermostat walk; during the four-stage interval (2026-10-08 .. 2026-10-11) it mixed through the DB path (p_db 1.0 with the TDP MIX knobs and p_twist 0.16) and needed `FROZEN_DB_DIR`; since the revival (`9499f42a`) it runs the store-free walk again (crossings, splits, conjugation twists at p_twist 0.16, thermostat at the input size), so the mixed gauntlet results of that interval are the ones not comparable with the archive's.

## Old stage 6: fcompress

`fcompress` gathered commutable same-target gates, reduced each group's ESOP (and, for small supports, its ANF), and packed each run into one generalized gate (`esop1`).

| Feature | Archive location | Now |
| --- | --- | --- |
| `fcompress` executable | `src/programs/fcompress/` | Gone |
| Compression stage: downhill, packing, reduce, transport | `src/stages/post-processing/compression/` | Gone |
| Tests: 22 compression tests | `tests/unit/stages/post-processing/compression/compress_tests.rs` | Gone |
| Demixing probes `crossing_downhill` (downhill on crossed circuits), `output_cone` (liveness prune plus crossing rules) | `security_tests/demixing/` | Gone |
| Benchmark `pairfloat_exp` (uses the crossing rules and presplit) | `benchmarks/mixing/pairfloat_exp.rs` | Gone |
| Formats `esop1`, `anf1`, packed | `src/circuit/formats.rs` | **Read side kept**: `read_packed` and the `read_mpmct` dispatch still load archive-built packed outputs, tested on byte fixtures. The writers (`write_packed`, `write_anf1`, `write_esop1`) and the `PackedGate` helpers only they used went to the archive with their sole producer |

## Why they were retired

**Owner decision, 2026-10-07.** The four-stage pipeline does not use the curated DB, and windowed packing replaces stages 4-6. The measurements behind that decision:

- **The curated store adds nothing measurable to mixing.** In four ablations on random material (2026-10-02/03, run tree `runs/ncdb_random_20261001/RESULTS.md`), removing the curated store or restricting it to MIX rounds left affine ridge depth unchanged within seed noise (for example 0.3784 / 0.3768 / 0.3776 for full / MIX-only / none at eff 30; the regular store covered the missing splices). The strong form, curated-only MIX rounds at a matched splice budget, was slightly worse (0.3832 vs 0.3784) and narrowed the choice entropy per splice (0.288 to about 0.21 bits).
- **The curated store was expensive and fragile.** The v2 store was extremely skewed (one 1-gate key held 7.7M circuits), a builder bug dropped same-length spellings, and curated-first runs crawled because candidate selection was linear in the candidate count (mean 327k per choice). The rebuilt circuit-keyed stores ran about 50× faster but still did not separate from each other or from regular-only mixing (`docs/notes/ncdb_ab_20261001_RESULTS.md`). An exported `FROZEN_CURATED_DIR` silently switched stage 3 to curated-first, an operational hazard.
- **Stages 4-6 bought little and cost a lot.** In the stage-wise exposure study (2026-10-03), split was visibility-neutral and added carriers, the crossing walk was the only eraser (−3 to −5 points of visibility), and `fcompress` refunded 1-3 points. At TDP scale stage-2-view visibility was flat across stages 3-6 (26.5 → 26.5 → 25.6 → 27.2%), and the C-view floor is owned by stage 2. Crossing also inflated the circuit about 3.5× and stretched the median dependency interval from 6 to 26 gates, which destroys the reordering that windowing relies on: windowing the stage-3 output hides 69.6% of carried values at W=20, against 46.9% after crossing. A single windowing pass hides 58-72% of carried values (G=128-256). See `docs/ANF_WINDOW_DELIVERY.md` §3.
- **Division of labour now.** Stage 2 decides which values ever exist in the clear, stage 3 respells and dilutes, and the packer (stage 4 at the time; the first sub-step of stage 6, delivery, since 2026-10-12) decides where anything is observable at all. `docs/tdp_pipeline.md` and `docs/STAGE2_LINKABILITY.md` describe the current pipeline; the full reference is `docs/TDP_PIPELINE_REFERENCE.md` (formerly `TDP_FOUR_STAGE_PIPELINE.md`).

## Porting notes

Nothing here is scheduled for revival (the split stage and the crossing walk were revived into `circuit_mixer` on 2026-10-11, commit `9499f42a`, so the "separate post-mix tool" row below no longer applies; the curated DB and `fcompress` remain archive-only). If an experiment needs a piece:

| Piece | Lands in | Effort |
| --- | --- | --- |
| A curated-tool analysis on the regular store | `db_gen/analysis/`, adapting to `FrozenDb::open(dir)` and single-key lookups | Small |
| `crossing_downhill` / `output_cone` as demixing probes | `security_tests/`, together with the crossing rules they call | Medium (needs `rules::cross`, presplit, compression downhill) |
| The crossing walk or split as a separate post-mix tool | A new executable outside `circuit_mixer`; the mixer no longer has the merge index, journal, tabu or events | Large |
| Reading an old (state v1/v2, recipe v8) run | Build the archive branch (it reads states v1 and v2; the v3, v4 and v5 states of the four-stage interval, 2026-10-08 .. 2026-10-11, are unreadable by both builds); do not port the old formats | None |

For each port: copy from the archive with `git show ran/archive/curated-and-postmix:<path>`, adapt to the current module paths and signatures, build with the relevant `--features`, and cite the archive commit (`ea2e516f`) in the commit message.
