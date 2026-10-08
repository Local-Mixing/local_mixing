# Cleanup inventory: curated DB and old stages 4–6 (2026-10-07)

Read-only survey of branch `ran/stage2-windowed-delivery` (544be97e) for
making the curated DB and the old stages 4 (split), 5 (crossing walk) and
6 (fcompress) historic. Line numbers approximate (±2).

## Key facts

1. **Crossing = the mixer's slot-3 thermostat walk.** Stage 3 passes
   `--p-db 1.0`, so slot 2 (`scheduling.rs:526-544`) takes every non-twist
   round and slot 3 (`scheduling.rs:545-578`: `db_attempt_comp` →
   undo/merge → `expand_move` → `cross_move`) never runs; `random_bool(1.0)`
   consumes no RNG, so removing slot 3 is trajectory-neutral for stage 3.
   But `MixParams::default()` has `p_db = 0`: slot 3 is the engine's default
   walk and drives ~30 unit tests, `golden_thermostat_walk`, parts (1)/(3) of
   `tests/integration/mixer_db_replacement.rs`, `gauntlet_gen`, and the
   empty-store plumbing mode (`tdp_gen.sh:490-493`, `--p-db 0`).
2. `--p-comp 1.0 --p-any 0.1` (stage-3 recipe) act only in slot 3 (and open
   the store in `Mixer::new`, `state.rs:685-695`). Keep accepting or update
   the scripts. `p_comp_g57`/`s_db_g57`/`db_g57_only` are slot-3-only.
3. Curated removal is RNG-neutral with curated off (current state when
   `FROZEN_CURATED_DIR` unset); changes only `counters.to_line()` (state file,
   golden hash), report line, QC lookup counts.
4. `database/codec.rs`, `database/validation.rs` are curated-only.
5. Merge machinery (`transformations.rs:223-361`, index in `indices.rs:7-45`)
   is used only by crossing and fcompress.
6. Banner bug: db_mixing/TDP banners (`circuit_mixer/mod.rs:88-91,105-108`)
   print `curated=true` before the env resolution at 209-230.

## A. Curated DB

Delete: `src/database/{codec,validation}.rs` (+ `database/mod.rs:2,5`);
`db_gen/curated_full.rs` (+ `db_gen/mod.rs:3-4`); `db_gen/bin/build_curated_full.rs`;
`db_gen/analysis/`: `curated_*.rs` (coverage_census, key_filter,
key_histogram, key_structure, recanon_probe, size_census), `pool_census`,
`true_friend_probe`, `greedy_walk`, `inflate_gates`, `frozen_pool_swap`,
`sgdb_build`, `mgdb_build`, `sgdb_substitute`, `plot_m1_*.py`,
`sample_similarity.py`, `compare_similarity.sh` (check first);
`benchmarks/db_mixing/db_curated_probe.rs`; `docs/CURATED_DB_V3_SUMMARY.*`;
`scripts/windowed_delivery/{run_stage3.sh,curves.py}`; tests
`tests/unit/db_gen/curated_full/tests.rs`, `tests/unit/db_gen/bin/build_curated_full/tests.rs`.
Ambiguous (curated-motivated, run on regular store): `identity_shingle_sieve`,
`minimal_halves_probe`, `minimal_identity_filter`, `frozen_class_census`,
`identity_length_census`, `cross_gluing_probe` (`--partner-composite`).

Entangled (edit): `stages/db_mixing/replacement.rs` (`xgate_to_db_g57` 43-56,
`circuit_key_of_window` 58-96, `curated_armed_for` 829-870, `db_replace`
curated params 790-821, `curated_armed`/`regular_fallback` 872-1043, Stage A
probe 1170-1213, `CandRef.curated` 1101-1116, `catalogue` 1117-1155,
`DbResult.curated_matches/chosen_curated` 698-701, `ReplacementOptions.curated_key`
751, `restrict_curated` 1450-1452/1567-1570, `CandLen::curated` 1677-1681,
`db_probe` tuple 293-311, QC curated pieces 336-517);
`engine/mixer/replacement.rs` (240-266, 320-331, 446-466, 527-546,
`try_db_splice_curated` 604 → rename, 658-700, 781-787);
`transformations.rs` bridge_round (1478-1583); `leakage_repair.rs:153`;
`params.rs` (146-163, 363-369, 626-638, 681, 744-754); `state.rs` (63-69,
104-106, 483-490, 836-837); `checkpoint.rs` (to_line 18,30; from_line ~124,136);
`reporting.rs` (162-173, 229, 270-276, 396-409, 476-479); `piecewise.rs`
(202-203, 219, 338-339, 355); `environment.rs` 14-26; `runtime.rs` 35-40;
`database/frozen.rs` (curated fields/open/get_curated 1048-1211, ~14 `open`
callers); `database/environment.rs` (8-10, 25-30, 51-62);
`database/lookup_cache.rs` (4, 26-32, 47-48, 93-102, 141-154, 179-181);
`circuit_mixer/cli.rs` (247-280, 465-480, 523-525, docs 123-127, 453-456);
`circuit_mixer/mod.rs` (58-66, 89-109, 209-249, 370-488, 567-600);
`src/tdp/` (config/mod.rs 94-95, parse.rs 76-84, validate.rs 38-56/276-277,
paths.rs 34-52/214-224, runner.rs 382-391/441-450, manifest.rs 185-207 +
`CURRENT_RECIPE_VERSION` bump), `configs/tdp.toml` 26-30;
`db_gen/frozen_build.rs` (`CompositeShards` 537-645,
`require_curated_full_manifest` 744-762, `pool_swap*` 1093+);
`db_gen/bin/frozen_from_lmdb.rs` (`--curated/--composite`);
`canonicalization/window.rs:288-306` (`canonicalize_polys_single_neg*`, test-only);
docs (`frozen_database.md` 268-453 + scattered, `local_mixing_documentation.md`
§1078/§1198, `tdp_pipeline.md` 90-92, `README.md` 36-46/219,
`tdp_gen.sh` 96-102/449/481), CI `ci.yml:72-77`.

## B. Old stage 4: split

Delete `src/stages/post-processing/splitting.rs`. Entangled: `scheduling.rs`
(352-367, 450-474, took_split guards); `params.rs` (104-111, 481-485,
514-536, defaults); `state.rs` (225-242, 520-584 incl. `Tap`, `MixStop::SplitDone`
604-606, init 793-872); `indices.rs` (side index 50-110, `evict_taps`
112-131, `splice_replace_one` 134-148); `sampling.rs::child_gen` 264-276;
`checkpoint.rs` (split line, staps; `STATE_VERSION = 2`); `piecewise.rs`
(`Policy::Split`, `TapRec`, taps export/install, split counters);
`leakage_repair.rs` 169-180; `reporting.rs`; `circuit_mixer/cli.rs`
(190-221, 574-578, 610-613, 687-690); `circuit_mixer/mod.rs` (24, 251-254,
504-508, 620-632, 693-695, 859-891); `src/stages/post-processing/mod.rs`,
`src/stages/mod.rs:3-4`; `tdp_gen.sh` (14-16, 79-93, 120, 208, 420, 517-535);
`src/tdp/runner.rs:473-483`; `tdp_pipeline.md` §4.

## C. Old stage 5: crossing walk

Delete `src/stages/post-processing/crossing.rs`. Entangled: slot 3 in
`scheduling.rs` 545-578; `transformations.rs` (`expand_move` 477-492, merge
223-361; `mod.rs:45-49` re-exports); `engine/mixer/replacement.rs::db_attempt_comp`
26-39; `params.rs` (walk params 70-120, `p_comp/p_any` 190-194,
`contract_ceiling` 405-410, `p_comp_g57/s_db_g57` 417-418, mincross 537-551);
`state.rs` (merge/undo/cross counters, `UndoEntry`, journal, tabu, cross pool);
`provenance.rs` (tabu ring 792-815, journal walk 768-778); `checkpoint.rs`
(merge/undo/tabu/journal); `piecewise.rs` (`Policy::Moves`, `JournalRec`,
journal/tabu transport); `reporting.rs`; `engine/moves/rules.rs` 14-277
(keep `verify_rewrite`); `circuit_mixer/cli.rs` (22-73 walk flags,
163-170, 222-234, 490-518); `circuit_mixer/mod.rs`; `src/tdp/` crossing
config (`xr..xmoves`), `stop_after` default "6"; `configs/tdp.toml`
`[crossing]`; `tdp_gen.sh` stage 5; docs.

## D. Old stage 6: fcompress

Delete `src/programs/fcompress/`, `src/stages/post-processing/compression/`,
`tests/unit/stages/post-processing/compression/compress_tests.rs`, Cargo
`[[bin]] fcompress`. Keep `circuit/formats.rs` packed/anf1/esop1 (circuit CLI).
Dependents: `security_tests/demixing/crossing_downhill.rs` (downhill),
`security_tests/demixing/output_cone.rs` (`liveness_prune` + `rules::cross`);
`src/tdp/runner.rs` 184/556, `manifest.rs:61` fingerprints; `tdp_gen.sh`
stage 6; shell tests `tdp_flags.bash:72`, `tdp_block_size.bash:59,79-86`.

## Tests and goldens

Delete with features: split tests (`splitting.rs`, checkpoint split
roundtrip, piecewise split tests, tap-anchor repair test); crossing tests
(`scheduling.rs` thermostat tests, ancestry/generations/gate_algebra/engine
cross tests); 22 compress tests; curated replacement tests (cascade,
regular_fallback, circuit-key tests) and curated db_gen tests.
Need a new driver (they use the thermostat walk as driver): ancestry
sampled-tracer tests, checkpoint round-trips, all `twists.rs` run tests,
`global_shuffle_moves`, piecewise thread-independence/empty tests, a
leakage-repair metadata test, integration parts (1)/(3).
Signature churn: ~46 `db_replace*`/`qc_*` calls, frozen/lookup_cache tests,
`database_policy.rs` (`try_db_splice_curated`); QC count assertions change.
`tests/unit/tdp.rs` manifest/flag tests; shell tests.
Goldens (`serial_baselines.rs`): `GOLDEN_SPLIT_STOP` delete;
`GOLDEN_THERMOSTAT_WALK` delete or re-baseline; `GOLDEN_PROFILE_NULL_PLANT`
trajectory unaffected but hash changes if counter fields leave `to_line()`.
Unaffected: preprocessing/construct, canonicalization/g57, circuit/formats.

## Must keep

Stages 1–2, gen_sandwich_gadget, hmap_affine (independent of all of this);
frozen reader regular path, lookup cache, `FROZEN_DB_DIR`/`FROZEN_FILTER`;
canonicalization (xgate, polys blob, gate_order, db_gen xcanon re-export);
replacement regular path + QC leakage repair; mixer slots 0–2 (profile,
twist-g57, shuffle, bridge, db_attempt + splice), sampling, transport,
provenance, checkpoint save, piecewise `Policy::Profile`; `Meta.dir` (RNG-
drawn in `Mixer::new`: removing it changes trajectories); swap_words,
`rules::verify_rewrite`, arena, stats; circuit formats; TDP driver stages 1–3.

## --tdp preset

Shipped defaults `curated=true`, `curated_exhaust=true`, `curated_in_comp=true`;
`--tdp` re-asserts curated + sets `db_advance`; DB layer has nothing curated.
With `FROZEN_CURATED_DIR` unset the mixer warns and sets `curated=false`
(explicit `--curated` panics). Removing curated = today's `--tdp` behaviour
with the env unset, apart from log/counter/state-format changes.
