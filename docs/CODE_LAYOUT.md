# Current code layout

Inventory checked against the working tree on 2026-09-11: **691 files in the organized source/configuration/documentation areas**, including **107 files under `src/`** and **81 registered Rust executables**. Preserved campaign sources and the other physical workspace areas are mapped separately below; the 691 count does not include them. Generated databases, runs, reports, build outputs and archives are outside the source-area file inventory.

## Runtime flow

```text
main → commands/gss → gss/config + runner + manifest
                        → scripts/gss_mix.sh
                          → gen_sandwich_gadget → sandwich + preprocessing
                          → fmix → db_mixing + splitting + crossing
                          → fcompress → compression + packing
```

Source stage ownership is `src/stages/{sandwich,preprocessing,db_mixing,post-processing}`. `post-processing` contains splitting, crossing and compression. Both quadratic masking and nonlinear291 are ordinary-build preprocessors; historical nonlinear193/product modes are optional compatibility code.

## Where to start

For an explanation of each subsystem, important functions and optimization techniques, read the [complete code walkthrough](CODE_WALKTHROUGH.md).

| Area | Responsibility |
| --- | --- |
| `src/commands/` | Thin command definitions; separate generate/evaluate/compare files |
| `src/gss/config/` | Parsing, defaults, validation and historical recipe interpretation |
| `src/gss/` | Build/run lifecycle, paths, manifests and saved runs |
| `src/stages/sandwich/` | Source preparation and sliced sandwich |
| `src/stages/preprocessing/` | Typed construction, quadratic masking, nonlinear291, slice guards, four runtime templates |
| `src/stages/db_mixing/` | Replacement policy and leakage audit/repair |
| `src/stages/post-processing/` | Splitting, crossing and compression/transport/packing |
| `src/engine/mixer/` | State, parameters, checkpoints, provenance, scheduling, reporting and transformations |
| `src/engine/moves/` | Shared local rules and swap words |
| `src/circuit/` | Representations, permutations, evaluation, file formats and randomization |
| `src/canonicalization/` | Polynomial arithmetic, deterministic ordering, key encoding and caches |
| `src/database/` | Frozen-store reading, codecs, validation and lookup caching |
| `src/programs/` | Executable argument/environment translation and artifact I/O |
| `src/entrypoints/` | Three small `main()` adapters; executable names unchanged |
| Compatibility modules | Old Rust imports forward to canonical owners; see [retention audit](CLEANUP_AUDIT.md) |

Root ownership and workflows are documented in [the project guide](../README.md), [runtime guide](../src/README.md), [database tools](../db_gen/README.md), [tests](../tests/README.md), [security tools](../security_tests/README.md), [benchmarks](../benchmarks/README.md) and [documentation index](README.md).

## Executable registry

Cargo has `autobins = false`; only declared targets become executables. Ordinary GSS uses the main CLI and the three stage programs. Optional programs retain their established names.

| Executable | Source | Required features |
| --- | --- | --- |
| `bench_canon4` | [benchmarks/canonicalization/bench_canon4.rs](../benchmarks/canonicalization/bench_canon4.rs) | `benchmark-tools` |
| `bench_eval` | [benchmarks/circuit/bench_eval.rs](../benchmarks/circuit/bench_eval.rs) | `benchmark-tools` |
| `bench_polycanon` | [benchmarks/canonicalization/bench_polycanon.rs](../benchmarks/canonicalization/bench_polycanon.rs) | `benchmark-tools` |
| `blinded_v5_gadgetize` | [security_tests/preprocessing/quadratic_masking.rs](../security_tests/preprocessing/quadratic_masking.rs) | `security-tools` |
| `block_cipher` | [challenges/block_cipher.rs](../challenges/block_cipher.rs) | `challenge-tools` |
| `blocker_census` | [benchmarks/db_mixing/blocker_census.rs](../benchmarks/db_mixing/blocker_census.rs) | `benchmark-tools` |
| `build_curated_full` | [db_gen/bin/build_curated_full.rs](../db_gen/bin/build_curated_full.rs) | `legacy-db-tools` |
| `canon_probe` | [benchmarks/canonicalization/canon_probe.rs](../benchmarks/canonicalization/canon_probe.rs) | `benchmark-tools` |
| `check_output` | [security_tests/fixtures/check_output.rs](../security_tests/fixtures/check_output.rs) | `security-tools` |
| `commute_shuffle_exp` | [benchmarks/mixing/commute_shuffle_exp.rs](../benchmarks/mixing/commute_shuffle_exp.rs) | `benchmark-tools` |
| `corpus_stats` | [db_gen/bin/corpus_stats.rs](../db_gen/bin/corpus_stats.rs) | `legacy-db-tools` |
| `cross_gluing_probe` | [db_gen/analysis/cross_gluing_probe.rs](../db_gen/analysis/cross_gluing_probe.rs) | `legacy-db-tools` |
| `curated_coverage_census` | [db_gen/analysis/curated_coverage_census.rs](../db_gen/analysis/curated_coverage_census.rs) | `legacy-db-tools` |
| `curated_key_filter` | [db_gen/analysis/curated_key_filter.rs](../db_gen/analysis/curated_key_filter.rs) | `legacy-db-tools` |
| `curated_key_histogram` | [db_gen/analysis/curated_key_histogram.rs](../db_gen/analysis/curated_key_histogram.rs) | `legacy-db-tools` |
| `curated_key_structure` | [db_gen/analysis/curated_key_structure.rs](../db_gen/analysis/curated_key_structure.rs) | `legacy-db-tools` |
| `curated_recanon_probe` | [db_gen/analysis/curated_recanon_probe.rs](../db_gen/analysis/curated_recanon_probe.rs) | `legacy-db-tools` |
| `curated_size_census` | [db_gen/analysis/curated_size_census.rs](../db_gen/analysis/curated_size_census.rs) | `legacy-db-tools` |
| `db_curated_probe` | [benchmarks/db_mixing/db_curated_probe.rs](../benchmarks/db_mixing/db_curated_probe.rs) | `benchmark-tools` |
| `db_match_synth` | [benchmarks/db_mixing/db_match_synth.rs](../benchmarks/db_mixing/db_match_synth.rs) | `benchmark-tools` |
| `eval_point` | [security_tests/fixtures/eval_point.rs](../security_tests/fixtures/eval_point.rs) | `security-tools` |
| `far_pair_probe` | [benchmarks/db_mixing/far_pair_probe.rs](../benchmarks/db_mixing/far_pair_probe.rs) | `benchmark-tools` |
| `fcompress` | [src/entrypoints/fcompress.rs](../src/entrypoints/fcompress.rs) | `ordinary build` |
| `fcone` | [security_tests/demixing/fcone.rs](../security_tests/demixing/fcone.rs) | `security-tools` |
| `fire_corr` | [security_tests/leakage/fire_corr.rs](../security_tests/leakage/fire_corr.rs) | `security-tools` |
| `flip_match` | [security_tests/leakage/flip_match.rs](../security_tests/leakage/flip_match.rs) | `security-tools` |
| `float_histogram` | [benchmarks/mixing/float_histogram.rs](../benchmarks/mixing/float_histogram.rs) | `benchmark-tools` |
| `fmix` | [src/entrypoints/fmix.rs](../src/entrypoints/fmix.rs) | `ordinary build` |
| `fmix_downhill` | [security_tests/demixing/fmix_downhill.rs](../security_tests/demixing/fmix_downhill.rs) | `security-tools` |
| `fmix_gauge_score` | [security_tests/demixing/fmix_gauge_score.rs](../security_tests/demixing/fmix_gauge_score.rs) | `security-tools` |
| `fmix_stats` | [benchmarks/mixing/fmix_stats.rs](../benchmarks/mixing/fmix_stats.rs) | `benchmark-tools` |
| `fragment_wide` | [security_tests/experiments/circuit/fragment_wide.rs](../security_tests/experiments/circuit/fragment_wide.rs) | `security-tools` |
| `frozen_census` | [db_gen/analysis/frozen_census.rs](../db_gen/analysis/frozen_census.rs) | `legacy-db-tools` |
| `frozen_class_census` | [db_gen/analysis/frozen_class_census.rs](../db_gen/analysis/frozen_class_census.rs) | `legacy-db-tools` |
| `frozen_degree_scan` | [benchmarks/db_mixing/frozen_degree_scan.rs](../benchmarks/db_mixing/frozen_degree_scan.rs) | `benchmark-tools` |
| `frozen_filters_build` | [db_gen/bin/frozen_filters_build.rs](../db_gen/bin/frozen_filters_build.rs) | `legacy-db-tools` |
| `frozen_find_small` | [db_gen/analysis/frozen_find_small.rs](../db_gen/analysis/frozen_find_small.rs) | `legacy-db-tools` |
| `frozen_from_lmdb` | [db_gen/bin/frozen_from_lmdb.rs](../db_gen/bin/frozen_from_lmdb.rs) | `legacy-db-tools` |
| `frozen_pool_swap` | [db_gen/analysis/frozen_pool_swap.rs](../db_gen/analysis/frozen_pool_swap.rs) | `legacy-db-tools` |
| `fsplit` | [security_tests/experiments/postprocessing/fsplit.rs](../security_tests/experiments/postprocessing/fsplit.rs) | `security-tools` |
| `fsplit_trace` | [security_tests/experiments/postprocessing/fsplit_trace.rs](../security_tests/experiments/postprocessing/fsplit_trace.rs) | `security-tools` |
| `gauntlet_audit` | [security_tests/gauntlet/gauntlet_audit.rs](../security_tests/gauntlet/gauntlet_audit.rs) | `security-tools` |
| `gauntlet_gen` | [security_tests/gauntlet/gauntlet_gen.rs](../security_tests/gauntlet/gauntlet_gen.rs) | `security-tools` |
| `gen_random_mpmct` | [security_tests/fixtures/gen_random_mpmct.rs](../security_tests/fixtures/gen_random_mpmct.rs) | `security-tools` |
| `gen_sandwich_gadget` | [src/entrypoints/gen_sandwich_gadget.rs](../src/entrypoints/gen_sandwich_gadget.rs) | `ordinary build` |
| `hmap` | [security_tests/heatmaps/hmap.rs](../security_tests/heatmaps/hmap.rs) | `security-tools` |
| `hmap_affine` | [security_tests/heatmaps/hmap_affine.rs](../security_tests/heatmaps/hmap_affine.rs) | `security-tools` |
| `hmap_stat` | [security_tests/heatmaps/hmap_stat.rs](../security_tests/heatmaps/hmap_stat.rs) | `security-tools` |
| `hmap_trace_affine` | [security_tests/heatmaps/hmap_trace_affine.rs](../security_tests/heatmaps/hmap_trace_affine.rs) | `security-tools` |
| `id_rewrite` | [security_tests/fixtures/id_rewrite.rs](../security_tests/fixtures/id_rewrite.rs) | `security-tools` |
| `identity_length_census` | [db_gen/analysis/identity_length_census.rs](../db_gen/analysis/identity_length_census.rs) | `legacy-db-tools` |
| `identity_shingle_sieve` | [db_gen/analysis/identity_shingle_sieve.rs](../db_gen/analysis/identity_shingle_sieve.rs) | `legacy-db-tools` |
| `ladder_mobility` | [benchmarks/mixing/ladder_mobility.rs](../benchmarks/mixing/ladder_mobility.rs) | `benchmark-tools` |
| `leeway_by_width` | [benchmarks/mixing/leeway_by_width.rs](../benchmarks/mixing/leeway_by_width.rs) | `benchmark-tools` |
| `legacy_mixing` | [security_tests/legacy/main.rs](../security_tests/legacy/main.rs) | `legacy-tools` |
| `local_mixing_bin` | [src/main.rs](../src/main.rs) | `ordinary build` |
| `merge_rocks_parallel` | [db_gen/bin/merge_rocks_parallel.rs](../db_gen/bin/merge_rocks_parallel.rs) | `legacy-db-tools` |
| `mgdb_build` | [db_gen/analysis/mgdb_build.rs](../db_gen/analysis/mgdb_build.rs) | `legacy-db-tools` |
| `minimal_halves_probe` | [db_gen/analysis/minimal_halves_probe.rs](../db_gen/analysis/minimal_halves_probe.rs) | `legacy-db-tools` |
| `minimal_identity_filter` | [db_gen/analysis/minimal_identity_filter.rs](../db_gen/analysis/minimal_identity_filter.rs) | `legacy-db-tools` |
| `newton_feistal` | [challenges/newton_feistal.rs](../challenges/newton_feistal.rs) | `challenge-tools` |
| `oracle_gate_learn` | [security_tests/oracle/oracle_gate_learn.rs](../security_tests/oracle/oracle_gate_learn.rs) | `security-tools` |
| `oracle_preimage_game` | [security_tests/oracle/oracle_preimage_game.rs](../security_tests/oracle/oracle_preimage_game.rs) | `security-tools` |
| `pairfloat_exp` | [benchmarks/mixing/pairfloat_exp.rs](../benchmarks/mixing/pairfloat_exp.rs) | `benchmark-tools` |
| `persistence_census` | [security_tests/leakage/persistence_census.rs](../security_tests/leakage/persistence_census.rs) | `security-tools` |
| `point_function` | [challenges/point_function.rs](../challenges/point_function.rs) | `challenge-tools` |
| `poly_canon` | [challenges/poly_canon.rs](../challenges/poly_canon.rs) | `challenge-tools` |
| `prod_grid` | [security_tests/fixtures/prod_grid.rs](../security_tests/fixtures/prod_grid.rs) | `security-tools` |
| `regen_sandwich_c` | [security_tests/fixtures/regen_sandwich_c.rs](../security_tests/fixtures/regen_sandwich_c.rs) | `security-tools` |
| `sampled_trace_support` | [security_tests/leakage/sampled_trace_support.rs](../security_tests/leakage/sampled_trace_support.rs) | `security-tools` |
| `sandwich_compare` | [security_tests/fixtures/sandwich_compare.rs](../security_tests/fixtures/sandwich_compare.rs) | `security-tools` |
| `segment_deduce` | [security_tests/leakage/segment_deduce.rs](../security_tests/leakage/segment_deduce.rs) | `security-tools` |
| `sgdb_build` | [db_gen/analysis/sgdb_build.rs](../db_gen/analysis/sgdb_build.rs) | `legacy-db-tools` |
| `sgdb_substitute` | [db_gen/analysis/sgdb_substitute.rs](../db_gen/analysis/sgdb_substitute.rs) | `legacy-db-tools, legacy-tools` |
| `shuffle_mpmct` | [security_tests/fixtures/shuffle_mpmct.rs](../security_tests/fixtures/shuffle_mpmct.rs) | `security-tools` |
| `source_stats` | [security_tests/leakage/source_stats.rs](../security_tests/leakage/source_stats.rs) | `security-tools` |
| `stress_battery` | [security_tests/leakage/stress_battery.rs](../security_tests/leakage/stress_battery.rs) | `security-tools` |
| `verify_zero_slice` | [tests/manual/gss/verify_zero_slice.rs](../tests/manual/gss/verify_zero_slice.rs) | `ordinary build` |
| `wide_verify` | [db_gen/bin/wide_verify.rs](../db_gen/bin/wide_verify.rs) | `legacy-db-tools` |
| `wide_yield_probe` | [benchmarks/db_mixing/wide_yield_probe.rs](../benchmarks/db_mixing/wide_yield_probe.rs) | `benchmark-tools` |
| `window_span_stats` | [benchmarks/db_mixing/window_span_stats.rs](../benchmarks/db_mixing/window_span_stats.rs) | `benchmark-tools` |

## Other physical workspace areas

The refactor preserves campaign deployments, historical source snapshots and their artifact paths. These folders are not all output-only folders.

| Workspace location | Contents / role |
| --- | --- |
| `affine_mixing_tests/` | Campaign harness and Python/shell scripts, validation/tests, configuration, source hash manifests, packages, collections, analyses and runs |
| `experiments/` | Historical campaign scripts and results; includes Python/C++ SAT scaling and structure analysis, mixing/heatmap launchers and an experimental quadratic-masking Rust snapshot |
| `mixing_tests/` | Historical Rust tests, `heatmap.py`, circuit/benchmark material and vendored `patches/libfaster-sys/` sources |
| `sattest/` | Saved SAT challenge/results and a solver-monitor shell script |
| `1_affine_tests/` | Captured local/remote experiment outputs, circuit pieces and heatmaps |
| `red_team_tests/` | Challenge circuits, answers, archives, sanity-check results and caches |
| `sgc/`, `rantestn128m800/`, `rantestn128m900/` | Historical circuit outputs, logs, metrics and images |
| `work/` | Database build/comparison reports and source/frozen stores |
| `Local_Mixing_Documentation/`, `Local_Mixing_Documentation.zip` | Historical documentation and image archive |
| `circuits/` | Circuit-generation reference information |
| `db/`, `old_db/`, `frozen_curated_v1_native/`, `frozen_curated_v2/` | Existing database data; `frozen_curated_v2.sha256` and `db-lock` retain their existing roles |
| `runs/`, `reports/` | Generated run/checkpoint and analysis/report artifacts |
| `tools/`, `gadgetization/`, `heatmap/` | Remaining Python cache directories at the time of this inventory |
| Root CSV/log files | Historical compression/expansion measurements and database-copy log |
| `target/`, `.codex_build/`, `.venv/` | Build outputs and local Python environment |
| `.git/`, `.agents/`, `.codex/`, `.claude/` | Repository/editor/agent metadata and separate local worktrees; contents are not part of this source map |

Generated artifact contents, installed/vendored dependency trees and separate worktrees are not recursively expanded here. Some historical run/supervisor directories are permission-restricted. Their presence is recorded without claiming a complete inventory of their contents.

## Organized source-area file inventory

Root configuration and package files:

```text
Cargo.toml
Cargo.lock
rust-toolchain.toml
rustfmt.toml
pyproject.toml
.gitignore
README.md
```

### src/ — 107 files

```text
src/README.md
src/canonicalization/cache.rs
src/canonicalization/canonicalize.rs
src/canonicalization/keys.rs
src/canonicalization/legacy_environment.rs
src/canonicalization/mod.rs
src/canonicalization/options.rs
src/canonicalization/polynomial.rs
src/canonicalization/window.rs
src/canonicalization/xgate.rs
src/circuit/evaluate.rs
src/circuit/formats.rs
src/circuit/formats/g57.rs
src/circuit/g57.rs
src/circuit/mod.rs
src/circuit/operations.rs
src/circuit/permutation.rs
src/circuit/randomize.rs
src/circuit/types.rs
src/circuit/xgate.rs
src/commands/circuit/compare.rs
src/commands/circuit/evaluate.rs
src/commands/circuit/generate.rs
src/commands/circuit/mod.rs
src/commands/circuit/shared.rs
src/commands/gss.rs
src/commands/mod.rs
src/database/codec.rs
src/database/frozen.rs
src/database/legacy_environment.rs
src/database/lookup_cache.rs
src/database/mod.rs
src/database/validation.rs
src/db_mixing/mod.rs
src/engine/arena.rs
src/engine/mixer/checkpoint.rs
src/engine/mixer/indices.rs
src/engine/mixer/leakage_repair.rs
src/engine/mixer/legacy_environment.rs
src/engine/mixer/mod.rs
src/engine/mixer/params.rs
src/engine/mixer/piecewise.rs
src/engine/mixer/provenance.rs
src/engine/mixer/replacement.rs
src/engine/mixer/reporting.rs
src/engine/mixer/runtime.rs
src/engine/mixer/sampling.rs
src/engine/mixer/scheduling.rs
src/engine/mixer/state.rs
src/engine/mixer/transformations.rs
src/engine/mixer/transport.rs
src/engine/mod.rs
src/engine/moves/mod.rs
src/engine/moves/rules.rs
src/engine/moves/swap_words.rs
src/engine/stats.rs
src/entrypoints/fcompress.rs
src/entrypoints/fmix.rs
src/entrypoints/gen_sandwich_gadget.rs
src/gss/cli.rs
src/gss/config/legacy_recipe.rs
src/gss/config/mod.rs
src/gss/config/parse.rs
src/gss/config/validate.rs
src/gss/manifest.rs
src/gss/mod.rs
src/gss/paths.rs
src/gss/runner.rs
src/lib.rs
src/main.rs
src/postprocessing/mod.rs
src/preprocessing/blinded_v5.rs
src/preprocessing/gadgets.rs
src/preprocessing/mod.rs
src/programs/fcompress.rs
src/programs/fmix/cli.rs
src/programs/fmix/mod.rs
src/programs/gen_sandwich_gadget.rs
src/programs/mod.rs
src/stages/db_mixing/leakage_repair/blocks.rs
src/stages/db_mixing/leakage_repair/detect.rs
src/stages/db_mixing/leakage_repair/mod.rs
src/stages/db_mixing/legacy_environment.rs
src/stages/db_mixing/mod.rs
src/stages/db_mixing/replacement.rs
src/stages/mod.rs
src/stages/post-processing/compression/downhill.rs
src/stages/post-processing/compression/mod.rs
src/stages/post-processing/compression/packing.rs
src/stages/post-processing/compression/reduce.rs
src/stages/post-processing/compression/transport.rs
src/stages/post-processing/crossing.rs
src/stages/post-processing/mod.rs
src/stages/post-processing/splitting.rs
src/stages/preprocessing/construct.rs
src/stages/preprocessing/mod.rs
src/stages/preprocessing/nonlinear291.rs
src/stages/preprocessing/quadratic_masking.rs
src/stages/preprocessing/slice_guards.rs
src/stages/preprocessing/templates/nonlinear291_and.mpmct1
src/stages/preprocessing/templates/nonlinear291_copy.mpmct1
src/stages/preprocessing/templates/nonlinear291_nab.mpmct1
src/stages/preprocessing/templates/nonlinear291_r57.mpmct1
src/stages/preprocessing/types.rs
src/stages/preprocessing/verify.rs
src/stages/sandwich/construct.rs
src/stages/sandwich/mod.rs
```

### db_gen/ — 55 files

```text
db_gen/README.md
db_gen/analysis/compare_similarity.sh
db_gen/analysis/cross_gluing_probe.rs
db_gen/analysis/curated_coverage_census.rs
db_gen/analysis/curated_key_filter.rs
db_gen/analysis/curated_key_histogram.rs
db_gen/analysis/curated_key_structure.rs
db_gen/analysis/curated_recanon_probe.rs
db_gen/analysis/curated_size_census.rs
db_gen/analysis/frozen_census.rs
db_gen/analysis/frozen_class_census.rs
db_gen/analysis/frozen_find_small.rs
db_gen/analysis/frozen_pool_swap.rs
db_gen/analysis/identity_length_census.rs
db_gen/analysis/identity_shingle_sieve.rs
db_gen/analysis/mgdb_build.rs
db_gen/analysis/minimal_halves_probe.rs
db_gen/analysis/minimal_identity_filter.rs
db_gen/analysis/plot_m1_3d.py
db_gen/analysis/plot_m1_heat.py
db_gen/analysis/sample_similarity.py
db_gen/analysis/sgdb_build.rs
db_gen/analysis/sgdb_substitute.rs
db_gen/bin/build_curated_full.rs
db_gen/bin/corpus_stats.rs
db_gen/bin/frozen_filters_build.rs
db_gen/bin/frozen_from_lmdb.rs
db_gen/bin/merge_rocks_parallel.rs
db_gen/bin/wide_verify.rs
db_gen/commands.rs
db_gen/curated_full.rs
db_gen/frozen_build.rs
db_gen/maintenance/build_filters.sh
db_gen/maintenance/check_filters.sh
db_gen/maintenance/cleanup_old_stores.sh
db_gen/maintenance/deploy_check.sh
db_gen/maintenance/deploy_regular.sh
db_gen/maintenance/deploy_v2.sh
db_gen/maintenance/distribute_filters.sh
db_gen/maintenance/final_inventory.sh
db_gen/maintenance/final_verify.sh
db_gen/maintenance/fleet_check.sh
db_gen/maintenance/hist_m1_new.sh
db_gen/maintenance/launch_coverage.sh
db_gen/maintenance/launch_filters.sh
db_gen/maintenance/n64_keygen.sh
db_gen/maintenance/paths.sh
db_gen/maintenance/run_census_new.sh
db_gen/maintenance/verify_deploy.sh
db_gen/mod.rs
db_gen/regular.rs
db_gen/support/mpx1.rs
db_gen/support/wide_db.rs
db_gen/support/xcanon.rs
db_gen/wide_gates.rs
```

### tests/ — 74 files

```text
tests/README.md
tests/circuit_cli.rs
tests/database/frozen/tests.rs
tests/database/lookup_cache.rs
tests/db_gen/bin/build_curated_full/tests.rs
tests/db_gen/commands/tests.rs
tests/db_gen/curated_full/tests.rs
tests/db_gen/regular_validation_tests.rs
tests/db_gen/support/mpx1/tests.rs
tests/db_gen/support/wide_db/tests.rs
tests/db_gen/support/xcanon/tests.rs
tests/db_gen/wide_gates/tests.rs
tests/fmix_db_move.rs
tests/frozen_roundtrip.rs
tests/gadgetization/__init__.py
tests/gadgetization/test_nonlinear_gadgets.py
tests/gadgetization/test_topology_templates.py
tests/gss_block_size.bash
tests/gss_flags.bash
tests/manual/README.md
tests/manual/gss/verify_zero_slice.rs
tests/poly_canon_failure_case.rs
tests/poly_canon_graph.rs
tests/poly_canon_stress.rs
tests/stages/db_mixing/leakage_repair/blocks/tests.rs
tests/stages/db_mixing/leakage_repair/detect/tests.rs
tests/stages/db_mixing/leakage_repair/tests.rs
tests/stages/db_mixing/replacement/tests.rs
tests/stages/post-processing/compression/compress_tests.rs
tests/stages/preprocessing/gen_sandwich_gadget.rs
tests/stages/preprocessing/guards.rs
tests/stages/preprocessing/legacy_gadgets_cnot_gadget_tests.rs
tests/stages/preprocessing/legacy_gadgets_drip_tests.rs
tests/stages/preprocessing/legacy_gadgets_feistal_32_wire_tests.rs
tests/stages/preprocessing/legacy_gadgets_feistal_fixed_point_n_tests.rs
tests/stages/preprocessing/legacy_gadgets_feistal_property_tests.rs
tests/stages/preprocessing/legacy_gadgets_feistal_structural_tests.rs
tests/stages/preprocessing/legacy_gadgets_feistal_tests.rs
tests/stages/preprocessing/legacy_gadgets_helpers.rs
tests/stages/preprocessing/legacy_gadgets_slice_zero_random_large_wire_tests.rs
tests/stages/preprocessing/legacy_generator.rs
tests/stages/preprocessing/managed_construction.rs
tests/stages/preprocessing/nonlinear291.rs
tests/stages/preprocessing/quadratic_masking.rs
tests/stages/preprocessing/samf_tests.rs
tests/stages/sandwich/construct.rs
tests/unit/canonicalization/g57/tests.rs
tests/unit/canonicalization/options.rs
tests/unit/canonicalization/xgate/tests.rs
tests/unit/circuit/formats/tests.rs
tests/unit/circuit/operations.rs
tests/unit/circuit/randomize/tests.rs
tests/unit/circuit/wide_fragment/tests.rs
tests/unit/circuit/xgate/xgate_kernel_tests.rs
tests/unit/circuit/xgate/xgate_lane_tests.rs
tests/unit/engine.rs
tests/unit/engine/mixer/leakage_repair/tests.rs
tests/unit/engine/mixer/mix_tests.rs
tests/unit/engine/mixer/piecewise/tests.rs
tests/unit/engine/moves/rules/tests.rs
tests/unit/engine/moves/swap_words/tests.rs
tests/unit/engine/stats/stats_tests.rs
tests/unit/gss.rs
tests/unit/legacy/cli.rs
tests/unit/legacy/db_mixing/convex/tests.rs
tests/unit/legacy/db_mixing/main_mix/opt_equiv_tests.rs
tests/unit/legacy/db_mixing/main_mix_cnot/tests.rs
tests/unit/legacy/db_mixing/replace/degree_filter_tests.rs
tests/unit/legacy/db_mixing/replace/float_tests.rs
tests/unit/legacy/db_mixing/sat_score/tests.rs
tests/unit/legacy/db_mixing/segcircuit/tests.rs
tests/unit/legacy/db_mixing/transpositions/reversed_samf_tests.rs
tests/unit/legacy/db_mixing/transpositions/unsamf_scale_tests.rs
tests/unit/programs/fmix/tests.rs
```

### security_tests/ — 228 files

```text
security_tests/README.md
security_tests/__init__.py
security_tests/attacks/__init__.py
security_tests/attacks/add_centered_hamming_ball_cnf.py
security_tests/attacks/add_hamming_ball_cnf.py
security_tests/attacks/check_point_retry_sat.py
security_tests/attacks/cinv_sample_pairs.cpp
security_tests/attacks/circuit_metrics_inverse.py
security_tests/attacks/circuit_structure_scan.py
security_tests/attacks/circuit_to_cnf_F_partial_prefix.cpp
security_tests/attacks/circuit_to_cnf_F_prefix_challenge.cpp
security_tests/attacks/circuit_to_cnf_block_preimage_gadget.cpp
security_tests/attacks/circuit_to_cnf_feistel_zero_z.cpp
security_tests/attacks/circuit_to_cnf_fixed_b_zero_e.cpp
security_tests/attacks/circuit_to_cnf_fixed_yz_graph.cpp
security_tests/attacks/circuit_to_cnf_fixed_yz_target_e.cpp
security_tests/attacks/circuit_to_cnf_fixed_yz_target_e_sliced.cpp
security_tests/attacks/circuit_to_cnf_forward_lowtarget_leading0.cpp
security_tests/attacks/circuit_to_cnf_forward_lowtarget_leading0_wide.cpp
security_tests/attacks/circuit_to_cnf_forward_lowtarget_orig_lz_gadget.cpp
security_tests/attacks/circuit_to_cnf_inverse_middle_to_yz.cpp
security_tests/attacks/circuit_to_cnf_lowtarget_leading0_generic.cpp
security_tests/attacks/circuit_to_cnf_lowtarget_leading0_wide.cpp
security_tests/attacks/circuit_to_cnf_pointcase_mapped.cpp
security_tests/attacks/circuit_to_cnf_rev_lowtarget_leading0.cpp
security_tests/attacks/circuit_to_cnf_rev_lowtarget_leading0_stream.cpp
security_tests/attacks/circuit_to_cnf_rev_lowtarget_orig_lz_gadget.cpp
security_tests/attacks/circuit_to_cnf_target128.cpp
security_tests/attacks/circuit_to_cnf_xor_io_preimage_gadget.cpp
security_tests/attacks/circuit_to_cnf_zero_b_target_e.cpp
security_tests/attacks/circuit_to_cnf_zero_bz_target_e.cpp
security_tests/attacks/commute_reduce.py
security_tests/attacks/decode_block_preimage_gadget_model.cpp
security_tests/attacks/decode_fixed_yz_model.cpp
security_tests/attacks/decode_forward_lowtarget_leading0_wide_model.cpp
security_tests/attacks/decode_forward_lowtarget_model.cpp
security_tests/attacks/decode_forward_lowtarget_orig_lz_gadget_model.cpp
security_tests/attacks/decode_inverse_middle_to_yz_model.cpp
security_tests/attacks/decode_lowtarget_leading0_wide_model.cpp
security_tests/attacks/decode_rev_low64_orig_lz_gadget_model.cpp
security_tests/attacks/decode_rev_lowtarget_model.cpp
security_tests/attacks/decode_tdp0_middle_model.cpp
security_tests/attacks/decode_xor_io_preimage_gadget_model.cpp
security_tests/attacks/kissat_core_snapshot.c
security_tests/attacks/kissat_hardcore_probe.c
security_tests/attacks/kissat_progress_metrics.py
security_tests/attacks/kissat_trace_heatmap.py
security_tests/attacks/plot_lowtarget_replicates.py
security_tests/attacks/propagation_hardcore.py
security_tests/attacks/render_et_heatmaps.py
security_tests/attacks/run_input_cube_kissat.py
security_tests/attacks/run_leading_zero_sweep.py
security_tests/attacks/run_lowtarget_knee.py
security_tests/attacks/run_lowtarget_replicates.py
security_tests/attacks/run_random_circuit_et_study.py
security_tests/attacks/run_skolem_kissat.py
security_tests/attacks/search_forward_low64_anneal.cpp
security_tests/attacks/search_forward_low64_newton.cpp
security_tests/attacks/search_forward_low64_orig_lz_gadget.cpp
security_tests/attacks/search_forward_lowtarget_leading0_newton_wide.cpp
security_tests/attacks/search_rev_hamming_ball.cpp
security_tests/attacks/search_rev_preimage_anneal.cpp
security_tests/attacks/search_rev_preimage_newton.cpp
security_tests/attacks/search_reverse_low64_orig_lz_gadget.cpp
security_tests/attacks/search_reverse_lowtarget_leading0_anneal_wide.cpp
security_tests/attacks/search_reverse_lowtarget_leading0_hamming_batch.cpp
security_tests/attacks/search_reverse_lowtarget_leading0_linear_space_batch.cpp
security_tests/attacks/search_reverse_lowtarget_leading0_newton_wide.cpp
security_tests/attacks/search_reverse_lowtarget_prefix_generic.cpp
security_tests/attacks/simple_cdcl.cpp
security_tests/attacks/sls_circuit_cnf_from_u.cpp
security_tests/attacks/summarize_feistel_kissat_grid.py
security_tests/attacks/uncenter_model.py
security_tests/campaigns/__init__.py
security_tests/campaigns/legacy_runner.sh
security_tests/campaigns/mixwatch.sh
security_tests/campaigns/n128m1000_autofill_monitor.sh
security_tests/campaigns/parameter_subagent_worker.py
security_tests/campaigns/parameter_sweep.py
security_tests/campaigns/recompute_incremental_heatmaps.py
security_tests/campaigns/recompute_std_heatmaps.py
security_tests/campaigns/recompute_very_enhanced_heatmaps.py
security_tests/campaigns/run_llmtest_stolen_heatmaps.py
security_tests/campaigns/run_n128m1000_campaign_queue.sh
security_tests/campaigns/run_pf_adaptive_m1.py
security_tests/campaigns/run_pf_key7124193401.py
security_tests/campaigns/run_sr1_ta4_r2_m1_x20.py
security_tests/campaigns/snapshot_compression_and_start_sat.sh
security_tests/campaigns/stagereport.sh
security_tests/campaigns/start_detached_sat_for_circuit.sh
security_tests/campaigns/watch_existing_sss_no_timeout.sh
security_tests/demixing/fcone.rs
security_tests/demixing/fmix_downhill.rs
security_tests/demixing/fmix_gauge_score.rs
security_tests/experiments/circuit/fragment_wide.rs
security_tests/experiments/postprocessing/fsplit.rs
security_tests/experiments/postprocessing/fsplit_trace.rs
security_tests/fixtures/check_output.rs
security_tests/fixtures/eval_point.rs
security_tests/fixtures/gen_random_mpmct.rs
security_tests/fixtures/id_rewrite.rs
security_tests/fixtures/prod_grid.rs
security_tests/fixtures/regen_sandwich_c.rs
security_tests/fixtures/sandwich_compare.rs
security_tests/fixtures/shuffle_mpmct.rs
security_tests/gadgetization/README.md
security_tests/gadgetization/__init__.py
security_tests/gadgetization/export_templates.py
security_tests/gadgetization/nonlinear193.py
security_tests/gadgetization/nonlinear291.py
security_tests/gadgetization/templates/nonlinear193_and.mpmct1
security_tests/gadgetization/templates/nonlinear193_copy.mpmct1
security_tests/gadgetization/templates/nonlinear193_nab.mpmct1
security_tests/gadgetization/templates/nonlinear193_r57.mpmct1
security_tests/gauntlet/README.md
security_tests/gauntlet/TESTING_PIPELINE.md
security_tests/gauntlet/__init__.py
security_tests/gauntlet/gauntlet.py
security_tests/gauntlet/gauntlet_audit.rs
security_tests/gauntlet/gauntlet_build.py
security_tests/gauntlet/gauntlet_gen.rs
security_tests/gauntlet/gauntlet_heatmap.py
security_tests/heatmaps/__init__.py
security_tests/heatmaps/compression_heatmap.py
security_tests/heatmaps/compression_hist.py
security_tests/heatmaps/gate_plot.py
security_tests/heatmaps/heatmap.py
security_tests/heatmaps/heatmap_legacy.py
security_tests/heatmaps/hmap.rs
security_tests/heatmaps/hmap_affine.rs
security_tests/heatmaps/hmap_stat.rs
security_tests/heatmaps/hmap_trace_affine.rs
security_tests/heatmaps/mean_evo.py
security_tests/heatmaps/means.py
security_tests/heatmaps/prep_heat.sh
security_tests/heatmaps/wireplot.py
security_tests/leakage/__init__.py
security_tests/leakage/aggregate_blind_recovery.py
security_tests/leakage/bare_census.py
security_tests/leakage/blind_gate_recovery.py
security_tests/leakage/blind_trace_control.py
security_tests/leakage/carrier_orbit_trace_audit.py
security_tests/leakage/exact_trace_span.py
security_tests/leakage/fire_corr.rs
security_tests/leakage/flip_match.rs
security_tests/leakage/open_mask_profile.py
security_tests/leakage/persistence_census.rs
security_tests/leakage/sampled_trace_support.rs
security_tests/leakage/score_blind_recovery.py
security_tests/leakage/segment_deduce.rs
security_tests/leakage/show_blind_ranking.py
security_tests/leakage/source_stats.rs
security_tests/leakage/stress_battery.rs
security_tests/leakage/summarize_blind_arms.py
security_tests/leakage/trace_xor_support.py
security_tests/legacy/compress.rs
security_tests/legacy/main.rs
security_tests/legacy/shoot.rs
security_tests/legacy/shuffle.rs
security_tests/legacy/ssg.rs
security_tests/legacy/sss.rs
security_tests/oracle/oracle_gate_learn.rs
security_tests/oracle/oracle_preimage_game.rs
security_tests/orchestration/__init__.py
security_tests/orchestration/_paths.py
security_tests/orchestration/analyze_affine_plate.py
security_tests/orchestration/decode_mpmct1_zero_slice_model.py
security_tests/orchestration/make_challenge.py
security_tests/orchestration/mpmct1_zero_slice_test.mpmct1
security_tests/orchestration/mpmct1_zero_slice_to_cnf.cpp
security_tests/orchestration/prepare_c1_remote.sh
security_tests/orchestration/rebuild_challenge_indexes.py
security_tests/orchestration/render_affine_sanity.py
security_tests/orchestration/restart_c2_public_attack_verbose_remote.py
security_tests/orchestration/run_c1_affine_remote.sh
security_tests/orchestration/run_c1_kissat_verbose_restart.py
security_tests/orchestration/run_c1_public_attack_remote.py
security_tests/orchestration/run_c2_hmap_remote.py
security_tests/orchestration/run_c2_public_attack_remote.py
security_tests/orchestration/run_one_server.sh
security_tests/orchestration/verify_layout.py
security_tests/orchestration/verify_offslice.py
security_tests/preprocessing/quadratic_masking.rs
security_tests/python/heatmap.rs
security_tests/python/mod.rs
security_tests/reporting/__init__.py
security_tests/reporting/band_hardening_20260725/stat_readout.py
security_tests/reporting/band_hardening_20260725/window_census.py
security_tests/reporting/band_hardening_20260725/wire_census.py
security_tests/reporting/hmap_pixels.py
security_tests/reporting/plot_hmap_ridge.py
security_tests/reporting/plot_hmap_trace.py
security_tests/reporting/split_trials_20260805/bheavy_arms.sh
security_tests/reporting/split_trials_20260805/ext_arms.sh
security_tests/reporting/split_trials_20260805/ext_arms2.sh
security_tests/reporting/split_trials_20260805/frontier_arms.sh
security_tests/reporting/split_trials_20260805/gate_census.py
security_tests/reporting/split_trials_20260805/gssmix4.sh
security_tests/reporting/split_trials_20260805/gssmix4b.sh
security_tests/reporting/split_trials_20260805/gssmix4c.sh
security_tests/reporting/split_trials_20260805/plot_canary_flips.py
security_tests/reporting/split_trials_20260805/plot_span_compare.py
security_tests/reporting/split_trials_20260805/plot_xpanel_progress.py
security_tests/reporting/split_trials_20260805/run_xpanel.sh
security_tests/reporting/split_trials_20260805/snap_frac3.py
security_tests/reporting/split_trials_20260805/xpanel_gen.py
security_tests/reporting/split_trials_20260805/xpanel_spread.py
security_tests/support/__init__.py
security_tests/support/circuit/wide_fragment.rs
security_tests/support/db_mixing/convex.rs
security_tests/support/db_mixing/main_mix.rs
security_tests/support/db_mixing/main_mix_cnot.rs
security_tests/support/db_mixing/pairs.rs
security_tests/support/db_mixing/ranking.rs
security_tests/support/db_mixing/replace.rs
security_tests/support/db_mixing/sat_score.rs
security_tests/support/db_mixing/segcircuit.rs
security_tests/support/db_mixing/transpositions.rs
security_tests/support/db_mixing/util.rs
security_tests/support/echo_conjugate_growth_mc.py
security_tests/support/experimental/mod.rs
security_tests/support/experimental/poly_canon_graph.rs
security_tests/support/experimental/split_engine.rs
security_tests/support/preprocessing/gadgets.rs
security_tests/support/preprocessing/generator.rs
security_tests/support/preprocessing/samf.rs
security_tests/support/rank/incoming_fanout.rhai
security_tests/support/rank/outgoing_pareto.rhai
```

### challenges/ — 5 files

```text
challenges/README.md
challenges/block_cipher.rs
challenges/newton_feistal.rs
challenges/point_function.rs
challenges/poly_canon.rs
```

### benchmarks/ — 23 files

```text
benchmarks/README.md
benchmarks/bench_pipeline.sh
benchmarks/canonicalization/bench_canon4.rs
benchmarks/canonicalization/bench_polycanon.rs
benchmarks/canonicalization/canon_probe.rs
benchmarks/canonicalization/compare.py
benchmarks/canonicalization/support.rs
benchmarks/circuit/bench_eval.rs
benchmarks/db_mixing/blocker_census.rs
benchmarks/db_mixing/db_curated_probe.rs
benchmarks/db_mixing/db_match_synth.rs
benchmarks/db_mixing/far_pair_probe.rs
benchmarks/db_mixing/frozen_degree_scan.rs
benchmarks/db_mixing/wide_yield_probe.rs
benchmarks/db_mixing/window_span_stats.rs
benchmarks/mixing/bench_stats_20260715.py
benchmarks/mixing/commute_shuffle_exp.rs
benchmarks/mixing/dmr_probe.rs
benchmarks/mixing/float_histogram.rs
benchmarks/mixing/fmix_stats.rs
benchmarks/mixing/ladder_mobility.rs
benchmarks/mixing/leeway_by_width.rs
benchmarks/mixing/pairfloat_exp.rs
```

### configs/ — 2 files

```text
configs/README.md
configs/gss.toml
```

### scripts/ — 60 files

```text
scripts/README.md
scripts/bench_pipeline.sh
scripts/compare_canon_bench.py
scripts/compat/gss_mix_v3.sh
scripts/compat/gss_mix_v4.sh
scripts/compat/gss_mix_v5.sh
scripts/compat/gss_mix_v6.sh
scripts/golden_check.sh
scripts/gss_mix.sh
scripts/n128m1000_campaign_configs/autofill_llmtest.tsv
scripts/n128m1000_campaign_configs/autofill_localtest.tsv
scripts/n128m1000_campaign_configs/autofill_n64tests.tsv
scripts/n128m1000_campaign_configs/autofill_nho.tsv
scripts/n128m1000_campaign_configs/autofill_testpieces.tsv
scripts/n128m1000_campaign_configs/cr2_llmtest.tsv
scripts/n128m1000_campaign_configs/cr2_localtest.tsv
scripts/n128m1000_campaign_configs/cr2_n64tests.tsv
scripts/n128m1000_campaign_configs/cr2_nho.tsv
scripts/n128m1000_campaign_configs/cr2_testpieces.tsv
scripts/n128m1000_campaign_configs/cr2b_llmtest.tsv
scripts/n128m1000_campaign_configs/cr2b_localtest.tsv
scripts/n128m1000_campaign_configs/cr2b_n64tests.tsv
scripts/n128m1000_campaign_configs/cr2b_nho.tsv
scripts/n128m1000_campaign_configs/cr2b_testpieces.tsv
scripts/n128m1000_campaign_configs/cr2c_llmtest.tsv
scripts/n128m1000_campaign_configs/cr2c_localtest.tsv
scripts/n128m1000_campaign_configs/cr2c_n64tests.tsv
scripts/n128m1000_campaign_configs/cr2c_nho.tsv
scripts/n128m1000_campaign_configs/cr2c_testpieces.tsv
scripts/n128m1000_campaign_configs/cr2d_llmtest.tsv
scripts/n128m1000_campaign_configs/cr2d_localtest.tsv
scripts/n128m1000_campaign_configs/cr2d_n64tests.tsv
scripts/n128m1000_campaign_configs/cr2d_nho.tsv
scripts/n128m1000_campaign_configs/cr2d_testpieces.tsv
scripts/n128m1000_campaign_configs/cr2e_llmtest.tsv
scripts/n128m1000_campaign_configs/cr2e_localtest.tsv
scripts/n128m1000_campaign_configs/cr2e_n64tests.tsv
scripts/n128m1000_campaign_configs/cr2e_nho.tsv
scripts/n128m1000_campaign_configs/cr2e_testpieces.tsv
scripts/n128m1000_campaign_configs/cr2f_llmtest.tsv
scripts/n128m1000_campaign_configs/cr2f_localtest.tsv
scripts/n128m1000_campaign_configs/cr2f_n64tests.tsv
scripts/n128m1000_campaign_configs/cr2f_nho.tsv
scripts/n128m1000_campaign_configs/cr2f_testpieces.tsv
scripts/n128m1000_campaign_configs/deep_llmtest.tsv
scripts/n128m1000_campaign_configs/deep_localtest.tsv
scripts/n128m1000_campaign_configs/deep_n64tests.tsv
scripts/n128m1000_campaign_configs/deep_nho.tsv
scripts/n128m1000_campaign_configs/deep_testpieces.tsv
scripts/n128m1000_campaign_configs/llmtest.tsv
scripts/n128m1000_campaign_configs/localtest.tsv
scripts/n128m1000_campaign_configs/n64tests.tsv
scripts/n128m1000_campaign_configs/nho.tsv
scripts/n128m1000_campaign_configs/probe2_llmtest.tsv
scripts/n128m1000_campaign_configs/probe2_localtest.tsv
scripts/n128m1000_campaign_configs/probe2_n64tests.tsv
scripts/n128m1000_campaign_configs/probe2_nho.tsv
scripts/n128m1000_campaign_configs/probe2_testpieces.tsv
scripts/n128m1000_campaign_configs/testpieces.tsv
scripts/tex2md.py
```

### docs/ — 129 files

```text
docs/CLEANUP_AUDIT.md
docs/CODE_LAYOUT.md
docs/CODE_WALKTHROUGH.md
docs/DB_CONTROL_ORDER.md
docs/DB_QUALITY_CONTROL.md
docs/GSS_FLAGS.md
docs/GSS_MIX.md
docs/README.md
docs/REORGANIZATION_PLAN.md
docs/design/FCOMPRESS_TRANSPORT_AND_PACKING.md
docs/design/FCOMPRESS_TRANSPORT_AND_PACKING.pdf
docs/design/FCOMPRESS_TRANSPORT_AND_PACKING.tex
docs/design/FMIX_PARAM_PRECEDENCE.md
docs/design/FMIX_PARAM_PRECEDENCE.pdf
docs/design/FMIX_PARAM_PRECEDENCE.tex
docs/design/FMIX_PIECEWISE.md
docs/design/FMIX_SPLIT_TWIST.md
docs/design/FULL_CURATED_DB.md
docs/design/G57_TWIST_BRACKETS.md
docs/design/G57_TWIST_BRACKETS.pdf
docs/design/G57_TWIST_BRACKETS.tex
docs/design/G57_TWIST_REPLACEMENTS.txt
docs/design/HMAP_AFFINE.md
docs/design/HMAP_BANDS.md
docs/design/HMAP_BANDS.pdf
docs/design/HMAP_BANDS.tex
docs/design/QUADRATIC_MASKING.md
docs/design/QUADRATIC_MASKING.pdf
docs/design/QUADRATIC_MASKING.tex
docs/design/SLICED_SANDWICH.md
docs/design/SLICED_SANDWICH.pdf
docs/design/SLICED_SANDWICH.tex
docs/design/blindedv5_affine.png
docs/design/blindedv5_deg2.png
docs/design/blindedv5_lin_vs_quad.png
docs/design/blindedv5_linear_phaseA_ridge.png
docs/design/blindedv5_quad_pipeline.png
docs/design/gate_commutation_rules.html
docs/formats/checkpoints.md
docs/history/DELIVERABLES.md
docs/history/EXPERIMENTS.md
docs/history/GSS_MIX_LEGACY.md
docs/history/OPTIMIZATION_PLAN.md
docs/history/REORGANIZATION_PLAN_20260910.md
docs/history/SSG_README.md
docs/history/fmix_parity_plan.md
docs/history/research/ANCESTRY_INSTRUMENTATION.md
docs/history/research/ANCESTRY_INSTRUMENTATION.pdf
docs/history/research/ANCESTRY_INSTRUMENTATION.tex
docs/history/research/ANCESTRY_MONITOR_BATTERY.md
docs/history/research/ANCESTRY_MONITOR_BATTERY.pdf
docs/history/research/ANCESTRY_MONITOR_BATTERY.tex
docs/history/research/BALANCED_MASKS_AND_COVERAGE_20260907.md
docs/history/research/BALANCED_MASKS_AND_COVERAGE_20260907.pdf
docs/history/research/BALANCED_MASKS_AND_COVERAGE_20260907.tex
docs/history/research/BAND_HARDENING.md
docs/history/research/BAND_HARDENING.pdf
docs/history/research/BAND_HARDENING.tex
docs/history/research/CARRIER_GADGETIZATION_SUMMARY.tex
docs/history/research/CORRELATING_TWO_COMPUTATIONS.md
docs/history/research/CORRELATING_TWO_COMPUTATIONS.pdf
docs/history/research/CORRELATING_TWO_COMPUTATIONS.tex
docs/history/research/CURATED_DB_COMPARISON.md
docs/history/research/CURATED_DB_COMPARISON.pdf
docs/history/research/CURATED_DB_COMPARISON.tex
docs/history/research/DB_CAMPAIGN_20260805.md
docs/history/research/DB_CAMPAIGN_20260805.pdf
docs/history/research/DB_CAMPAIGN_20260805.tex
docs/history/research/DRAIN_SET.md
docs/history/research/DRAIN_SET.pdf
docs/history/research/DRAIN_SET.tex
docs/history/research/FMIX_GROW_CHURN.md
docs/history/research/FMIX_LAYER1.md
docs/history/research/FMIX_LAYER1.pdf
docs/history/research/FMIX_LAYER1.tex
docs/history/research/FMIX_LAYER2.md
docs/history/research/FMIX_LAYER2.pdf
docs/history/research/FMIX_LAYER2.tex
docs/history/research/FMIX_MENU.md
docs/history/research/FMIX_PHASE_A.md
docs/history/research/FMIX_PHASE_A.pdf
docs/history/research/FMIX_PHASE_A.tex
docs/history/research/GADGETIZE_UPDATE_20260822.md
docs/history/research/GADGETIZE_UPDATE_20260822.pdf
docs/history/research/GADGETIZE_UPDATE_20260822.tex
docs/history/research/GRAY_FOLD_CG.md
docs/history/research/GRAY_FOLD_CG.pdf
docs/history/research/GRAY_FOLD_CG.tex
docs/history/research/Mixing_Pieces_Documentation.md
docs/history/research/NEW_GADGETIZE.md
docs/history/research/NEW_GADGETIZE.pdf
docs/history/research/NEW_GADGETIZE.tex
docs/history/research/NONLINEAR_GADGETIZATION.md
docs/history/research/NONLINEAR_GADGETIZATION.pdf
docs/history/research/NONLINEAR_GADGETIZATION.tex
docs/history/research/NONLINEAR_GADGET_ANALYSIS.md
docs/history/research/NONLINEAR_MIXING.md
docs/history/research/NONLINEAR_RG_CG_MENU.md
docs/history/research/NONLINEAR_RG_CG_MENU.pdf
docs/history/research/NONLINEAR_RG_CG_MENU.tex
docs/history/research/NONLOCAL_ECHO.md
docs/history/research/NONLOCAL_PHASE_A.md
docs/history/research/NO_GRAY_PHASE_A_EXPERIMENT.md
docs/history/research/PIPELINE_OVERVIEW.md
docs/history/research/PIPELINE_OVERVIEW.pdf
docs/history/research/PIPELINE_OVERVIEW.tex
docs/history/research/POSTMIX_MANUAL.md
docs/history/research/PRODUCT_SHARE_ENCODING.md
docs/history/research/PRODUCT_SHARE_ENCODING.pdf
docs/history/research/PRODUCT_SHARE_ENCODING.tex
docs/history/research/PRODUCT_SHARE_UPDATE.md
docs/history/research/PRODUCT_SHARE_UPDATE.pdf
docs/history/research/PRODUCT_SHARE_UPDATE.tex
docs/history/research/RIDGE_COVERAGE_EXPERIMENTS.md
docs/history/research/RIDGE_COVERAGE_EXPERIMENTS.pdf
docs/history/research/RIDGE_COVERAGE_EXPERIMENTS.tex
docs/history/research/RIDGE_QUADFIRE_20260906.md
docs/history/research/SAT_ATTACK_STRUCTURAL_COMPLEXITY.md
docs/history/research/SINGLE_CARRIER_CONSTRUCTION.md
docs/history/research/SINGLE_CARRIER_CONSTRUCTION.pdf
docs/history/research/SINGLE_CARRIER_CONSTRUCTION.tex
docs/history/research/SPLIT_TWIST_REPORT.md
docs/history/research/SPLIT_TWIST_REPORT.pdf
docs/history/research/SPLIT_TWIST_REPORT.tex
docs/history/research/SSG_PARAMETERS.md
docs/history/research/SWAP_REFRESH_REDESIGN.md
docs/history/research/SWAP_REFRESH_REDESIGN.pdf
docs/history/research/SWAP_REFRESH_REDESIGN.tex
docs/history/ssg_compression_speedup_notes.md
```

### .github/ — 1 files

```text
.github/workflows/ci.yml
```

## Preserved campaign and historical source files

The bounded workspace survey additionally enumerated **356 code, script, build/configuration and Markdown files** outside the 690-file organized inventory. These original campaign paths remain in place. Some entries are captured source snapshots or historical reports rather than maintained runtime modules.

This list includes readable files within five directory levels, using source/configuration extensions and build-file names. It does not follow symlinks or expand vendored FASTER, databases, build/cache trees, known generated output trees or separate worktrees. The survey encountered 54 unreadable historical directories; those contents are not claimed as inventoried.

### 1_affine_tests/ — 4 enumerated files

```text
1_affine_tests/nonlinear_tdp4n_2233_degree2_20260726/RESULTS.md
1_affine_tests/nonlinear_tdp4n_sss5_fmix152_noopfix_v05/degree2_constructed_fixedslice/RESULTS.md
1_affine_tests/nonlinear_tdp4n_sss5_fmix152_noopfix_v05/degree2_constructed_zeroaux/RESULTS.md
1_affine_tests/sliced_sandwich_2233_degree2_20260726/RESULTS.md
```

### affine_mixing_tests/ — 138 enumerated files

```text
affine_mixing_tests/HARNESS.md
affine_mixing_tests/PLAN.md
affine_mixing_tests/SUPERVISOR_REQUIREMENTS.md
affine_mixing_tests/analysis/README.md
affine_mixing_tests/analysis/affine-mixing-final-cap-constrained-20260731T223843Z/FINAL_CONCLUSION.md
affine_mixing_tests/analysis/n16m300-final-conclusion-20260801t0315z-v1/REPORT.md
affine_mixing_tests/config/README.md
affine_mixing_tests/preflight/active_baseline_20260728/README.md
affine_mixing_tests/preflight/bj_pilot_d_timeout_20260730T163538Z/RECOVERY.md
affine_mixing_tests/preflight/concurrency_capacity_20260730/README.md
affine_mixing_tests/preflight/current_g100_failed_pilot_20260728/README.md
affine_mixing_tests/preflight/current_source/20260728T225316Z/RESULTS.md
affine_mixing_tests/preflight/current_source/20260728T225316Z/run_candidate_smokes.sh
affine_mixing_tests/preflight/current_source/20260728T233206Z/RESULTS.md
affine_mixing_tests/preflight/current_source/20260728T233206Z/run_v2_smokes.sh
affine_mixing_tests/preflight/db_fingerprints_20260728/FULL_SHARD_AUDIT.md
affine_mixing_tests/preflight/db_fingerprints_20260728/README.md
affine_mixing_tests/preflight/db_fingerprints_20260728/fingerprint_filters.sh
affine_mixing_tests/preflight/db_fingerprints_20260728/hash_all_shards.sh
affine_mixing_tests/preflight/diagnostics/run_clean_final_validation_v2.sh
affine_mixing_tests/preflight/diagnostics/run_clean_ssg_status_smoke.sh
affine_mixing_tests/preflight/diagnostics/run_curated_compat_g2.sh
affine_mixing_tests/preflight/diagnostics/run_curated_compat_smoke.sh
affine_mixing_tests/preflight/diagnostics/run_current_g100_timing_pilot.sh
affine_mixing_tests/preflight/diagnostics/run_regular_native_regression.sh
affine_mixing_tests/preflight/e_rapidgrid_profile_budget_export_recovery_20260731T171812Z/PLAN.snapshot.md
affine_mixing_tests/preflight/e_rapidgrid_profile_budget_export_recovery_20260731T171812Z/RECOVERY.md
affine_mixing_tests/preflight/e_rapidgrid_profile_budget_export_recovery_20260731T171812Z/scripts/run_one.future.sh
affine_mixing_tests/preflight/e_rapidgrid_profile_budget_export_recovery_20260731T171812Z/scripts/test_run_one_timeout_isolation.future.py
affine_mixing_tests/preflight/e_rapidgrid_profile_budget_export_recovery_20260731T171812Z/scripts/test_verify_calibration_pilot_partial.py
affine_mixing_tests/preflight/e_rapidgrid_profile_budget_export_recovery_20260731T171812Z/scripts/verify_calibration_pilot_partial.py
affine_mixing_tests/preflight/hmap_metadata_contract_failure_20260730T0000Z/RECOVERY.md
affine_mixing_tests/preflight/normalized_curated_db_20260729/BUILD_REPORT.remote.md
affine_mixing_tests/preflight/normalized_curated_db_20260729/FLEET_AUDIT.md
affine_mixing_tests/preflight/t420_export_path_contract_failure_20260730T220455Z/RECOVERY.md
affine_mixing_tests/scripts/active_hosts.py
affine_mixing_tests/scripts/atomic_append_evidence.py
affine_mixing_tests/scripts/build_package.sh
affine_mixing_tests/scripts/build_tools.sh
affine_mixing_tests/scripts/circuit_attempt_cap.py
affine_mixing_tests/scripts/collect.sh
affine_mixing_tests/scripts/collect_n16_package_supplement.py
affine_mixing_tests/scripts/common.sh
affine_mixing_tests/scripts/cpu_partition.py
affine_mixing_tests/scripts/db_shard_manifest.py
affine_mixing_tests/scripts/deploy.sh
affine_mixing_tests/scripts/enforce_db_reuse_mode.sh
affine_mixing_tests/scripts/main_scientific_fingerprint.py
affine_mixing_tests/scripts/make_assignments.py
affine_mixing_tests/scripts/make_main_successor.py
affine_mixing_tests/scripts/ms360_grid_contract.py
affine_mixing_tests/scripts/n16_package_supplement_contract_v1.py
affine_mixing_tests/scripts/n16_package_supplement_contract_v2.py
affine_mixing_tests/scripts/n16m300_fasttrack_cancel.py
affine_mixing_tests/scripts/n16m300_grid_contract.py
affine_mixing_tests/scripts/n16m300_gtoj_cli_smoke.sh
affine_mixing_tests/scripts/n16m300_gtoj_corrective_contract.py
affine_mixing_tests/scripts/n16m300_nolength_contract.py
affine_mixing_tests/scripts/n16m300_postcal_contract.py
affine_mixing_tests/scripts/n16m300_raw_successor_contract.py
affine_mixing_tests/scripts/n16m300_stop_all_execute.py
affine_mixing_tests/scripts/n16m300_stop_inventory.py
affine_mixing_tests/scripts/parse_gnu_time.py
affine_mixing_tests/scripts/poll.sh
affine_mixing_tests/scripts/prepare_n16m300_gtoj_corrective.py
affine_mixing_tests/scripts/recollect_ms360_package_evidence.py
affine_mixing_tests/scripts/remote_worker.sh
affine_mixing_tests/scripts/render_collection_pool.py
affine_mixing_tests/scripts/retire_abandoned_supervisor.py
affine_mixing_tests/scripts/retire_prelaunch_ms360_v1.py
affine_mixing_tests/scripts/retire_recovered_attention_supervisor.py
affine_mixing_tests/scripts/run_one.sh
affine_mixing_tests/scripts/run_worker_member.sh
affine_mixing_tests/scripts/safe_extract_tar.py
affine_mixing_tests/scripts/source_artifacts.py
affine_mixing_tests/scripts/start.sh
affine_mixing_tests/scripts/supervisor_collection.py
affine_mixing_tests/scripts/supervisor_common.sh
affine_mixing_tests/scripts/supervisor_enqueue.sh
affine_mixing_tests/scripts/supervisor_foreground.sh
affine_mixing_tests/scripts/supervisor_item.sh
affine_mixing_tests/scripts/supervisor_poll.py
affine_mixing_tests/scripts/supervisor_resume.sh
affine_mixing_tests/scripts/supervisor_start.sh
affine_mixing_tests/scripts/supervisor_status.sh
affine_mixing_tests/scripts/test_calibration_topology.py
affine_mixing_tests/scripts/test_circuit_attempt_cap.py
affine_mixing_tests/scripts/test_collect_n16_package_supplement.py
affine_mixing_tests/scripts/test_cpu_partition.py
affine_mixing_tests/scripts/test_db_shard_manifest.py
affine_mixing_tests/scripts/test_generation_lock.sh
affine_mixing_tests/scripts/test_hmap_metadata_contract.py
affine_mixing_tests/scripts/test_host_loop_isolation.py
affine_mixing_tests/scripts/test_main_scientific_fingerprint.py
affine_mixing_tests/scripts/test_main_successor.py
affine_mixing_tests/scripts/test_make_assignments.py
affine_mixing_tests/scripts/test_ms360_grid_contract.py
affine_mixing_tests/scripts/test_n16m300_fasttrack_cancel.py
affine_mixing_tests/scripts/test_n16m300_grid_contract.py
affine_mixing_tests/scripts/test_n16m300_gtoj_cli_smoke.sh
affine_mixing_tests/scripts/test_n16m300_gtoj_corrective_contract.py
affine_mixing_tests/scripts/test_n16m300_nolength_contract.py
affine_mixing_tests/scripts/test_n16m300_postcal_contract.py
affine_mixing_tests/scripts/test_n16m300_raw_successor_contract.py
affine_mixing_tests/scripts/test_n16m300_source.py
affine_mixing_tests/scripts/test_parse_gnu_time.py
affine_mixing_tests/scripts/test_prepare_n16m300_gtoj_corrective.py
affine_mixing_tests/scripts/test_recollect_ms360_package_evidence.py
affine_mixing_tests/scripts/test_render_collection_pool.py
affine_mixing_tests/scripts/test_required_db_reuse.sh
affine_mixing_tests/scripts/test_retire_abandoned_supervisor.py
affine_mixing_tests/scripts/test_retire_prelaunch_ms360_v1.py
affine_mixing_tests/scripts/test_retire_recovered_attention_supervisor.py
affine_mixing_tests/scripts/test_run_one_g50_incomplete.sh
affine_mixing_tests/scripts/test_run_one_timeout_isolation.py
affine_mixing_tests/scripts/test_supervisor_core.py
affine_mixing_tests/scripts/test_supervisor_orphan_adoption.sh
affine_mixing_tests/scripts/test_supervisor_wrappers.sh
affine_mixing_tests/scripts/test_validate_ssg_status.py
affine_mixing_tests/scripts/test_verify_calibration_pilot_partial.py
affine_mixing_tests/scripts/test_verify_calibration_salvage.py
affine_mixing_tests/scripts/test_verify_main_successor_parent.py
affine_mixing_tests/scripts/test_verify_ms360_cap_profile_recovery.py
affine_mixing_tests/scripts/test_verify_ms360_grid.py
affine_mixing_tests/scripts/test_verify_n16m300_calibration.py
affine_mixing_tests/scripts/test_verify_n16m300_gtoj_corrective_parent.py
affine_mixing_tests/scripts/test_worker_concurrency.py
affine_mixing_tests/scripts/validate_assignment_design.py
affine_mixing_tests/scripts/validate_map.py
affine_mixing_tests/scripts/validate_ssg_status.py
affine_mixing_tests/scripts/verify_calibration_pilot_partial.py
affine_mixing_tests/scripts/verify_calibration_salvage.py
affine_mixing_tests/scripts/verify_main_successor_parent.py
affine_mixing_tests/scripts/verify_ms360_cap_profile_recovery.py
affine_mixing_tests/scripts/verify_ms360_grid.py
affine_mixing_tests/scripts/verify_n16m300_calibration.py
affine_mixing_tests/scripts/verify_n16m300_gtoj_corrective_parent.py
affine_mixing_tests/scripts/verify_n16m300_gtoj_smoke_attestation.sh
```

### experiments/ — 188 enumerated files

```text
experiments/affine_heatmaps_compact300_20260722/AFFINE_HEATMAP_RESULTS.md
experiments/affine_heatmaps_compact300_20260722/run_all.sh
experiments/affine_heatmaps_compact300_20260722/tools/HMAP_AFFINE.md
experiments/blind_recovery_20260814/RESULTS.md
experiments/blind_recovery_20260814/aggregate_all.sh
experiments/blind_recovery_20260814/aggregate_pooled.sh
experiments/blind_recovery_20260814/run_arm.sh
experiments/blind_recovery_20260814/run_fleet.sh
experiments/blind_recovery_20260814/run_presets.sh
experiments/blinded_v5_clean_two_controls_20260908/README.md
experiments/blinded_v5_clean_two_controls_20260908/blinded-v5-two-controls.md
experiments/blinded_v5_clean_two_controls_20260908/blinded_v5_clean_two_controls.rs
experiments/cnot_painless_20260716/ACADEMIC_RESEARCH_REPORT.md
experiments/cnot_painless_20260716/CNOT_PAINLESS_RESULTS.md
experiments/cnot_painless_20260716/RESULTS_SUMMARY.md
experiments/cnot_painless_20260716/run_cnot_painless_trial.sh
experiments/cnot_painless_20260716/run_painless_target_batch.sh
experiments/fast_feistal_tdp_20260715/HARDNESS_RESULTS.md
experiments/fast_feistal_tdp_20260715/RESULTS.md
experiments/fast_feistal_tdp_20260715/SPEED_SWEEP_RESULTS.md
experiments/fast_feistal_tdp_20260715/run_fast_feistal_tdp_trial.sh
experiments/fast_feistal_tdp_20260715/run_fixed_slice_target_batch.sh
experiments/fast_feistal_tdp_20260715/run_min_gen_absorb_trial.sh
experiments/fast_feistal_tdp_20260715/stop_old_test_workloads.sh
experiments/fcompress_g57_20260715/REPORT.md
experiments/fcompress_g57_sources_20260715/REPORT.md
experiments/g57_pair_factorial_20260722/FORCED_PHASE_B_STATUS.md
experiments/g57_pair_factorial_20260722/HEARTBEAT_STATUS.md
experiments/g57_pair_factorial_20260722/HEATMAP_ANALYSIS.md
experiments/g57_pair_factorial_20260722/INTEGRATED_NEW_SSS_PROVENANCE.md
experiments/g57_pair_factorial_20260722/NEW_SSS_PHASE_A_PROVENANCE.md
experiments/g57_pair_factorial_20260722/PHASE_A_OPEN_FOLLOWON_STATUS.md
experiments/g57_pair_factorial_20260722/PHYSICAL_HOST_MAPPING.md
experiments/g57_pair_factorial_20260722/README.md
experiments/g57_pair_factorial_20260722/REMAINING_RESUME_METHOD.md
experiments/g57_pair_factorial_20260722/REMAINING_RESUME_STATUS.md
experiments/g57_pair_factorial_20260722/STRICT4N_NEW_SSS_METHOD.md
experiments/g57_pair_factorial_20260722/advance_phase_a_open_followon.sh
experiments/g57_pair_factorial_20260722/audit_remaining_remote.sh
experiments/g57_pair_factorial_20260722/collect_results.sh
experiments/g57_pair_factorial_20260722/deploy_and_start.sh
experiments/g57_pair_factorial_20260722/deploy_remaining.sh
experiments/g57_pair_factorial_20260722/hold_before_capped_phase_a.sh
experiments/g57_pair_factorial_20260722/poll_campaign.sh
experiments/g57_pair_factorial_20260722/poll_forced_phase_b.sh
experiments/g57_pair_factorial_20260722/poll_phase_a_open_followon.sh
experiments/g57_pair_factorial_20260722/poll_relocated_open_profile.sh
experiments/g57_pair_factorial_20260722/poll_remaining.sh
experiments/g57_pair_factorial_20260722/prepare_baselines.sh
experiments/g57_pair_factorial_20260722/run_analysis.sh
experiments/g57_pair_factorial_20260722/run_forced_phase_b_queue.sh
experiments/g57_pair_factorial_20260722/run_integrated_new_sss_pilot.sh
experiments/g57_pair_factorial_20260722/run_phase_a_open_followon.sh
experiments/g57_pair_factorial_20260722/run_profile.sh
experiments/g57_pair_factorial_20260722/run_profile_phase_a_open.sh
experiments/g57_pair_factorial_20260722/run_relocated_open_profile.sh
experiments/g57_pair_factorial_20260722/run_remaining.sh
experiments/g57_pair_factorial_20260722/run_shard.sh
experiments/g57_pair_factorial_20260722/stage_phase_a_open_followon.sh
experiments/g57_pair_factorial_20260722/sync_completed_affine_maps.sh
experiments/g57_pair_factorial_20260722/sync_relocated_open_profile.sh
experiments/g57_pair_factorial_20260722/terminate_legacy_n128_campaign.sh
experiments/g57_pair_factorial_20260722/terminate_legacy_n128_frozen_retry.sh
experiments/g57_pair_factorial_20260722/terminate_legacy_n128_stable_pids.sh
experiments/g57_pair_factorial_20260722/terminate_phase_a_forced_policy.sh
experiments/g57_pair_factorial_20260722/terminate_superseded_calibration.sh
experiments/gss_sat_scaling_20260810/DELIVERABLES.md
experiments/gss_sat_scaling_20260810/FINAL_SAT_SCALING_ESTIMATE.md
experiments/gss_sat_scaling_20260810/KNEE_EXTENSION_DECISION.md
experiments/gss_sat_scaling_20260810/PRELIMINARY_ESTIMATE.md
experiments/gss_sat_scaling_20260810/README.md
experiments/gss_sat_scaling_20260810/STATUS.md
experiments/gss_sat_scaling_20260810/analysis/PUBLIC_MODEL_COMPARISON.md
experiments/gss_sat_scaling_20260810/analysis/README.md
experiments/gss_sat_scaling_20260810/analysis/fit_censored_scaling.py
experiments/gss_sat_scaling_20260810/analysis/render_public_scaling_summary.py
experiments/gss_sat_scaling_20260810/analysis/test_fit_censored_scaling.py
experiments/gss_sat_scaling_20260810/benchlib.py
experiments/gss_sat_scaling_20260810/build_encoder.sh
experiments/gss_sat_scaling_20260810/calibration/collect_cross_host.py
experiments/gss_sat_scaling_20260810/calibration/run_cross_host.py
experiments/gss_sat_scaling_20260810/collect_results.py
experiments/gss_sat_scaling_20260810/fleet/README.md
experiments/gss_sat_scaling_20260810/fleet/direct_n256_r01_launch.md
experiments/gss_sat_scaling_20260810/fleet/inventory_20260813.md
experiments/gss_sat_scaling_20260810/fleet/knee_classification3_launch.md
experiments/gss_sat_scaling_20260810/fleet/run_direct_n256_r01_sattesting.sh
experiments/gss_sat_scaling_20260810/fleet/run_knee_c3_llmtest.sh
experiments/gss_sat_scaling_20260810/fleet/run_knee_c3_llmtest_r05.sh
experiments/gss_sat_scaling_20260810/fleet/run_knee_c3_nho_r06.sh
experiments/gss_sat_scaling_20260810/legacy_anchor/TERMINAL_SUMMARY.md
experiments/gss_sat_scaling_20260810/make_manifest.py
experiments/gss_sat_scaling_20260810/mpmct1_zero_slice_to_cnf.cpp
experiments/gss_sat_scaling_20260810/run_job.py
experiments/gss_sat_scaling_20260810/tests/test_cross_host_calibration.py
experiments/gss_sat_scaling_20260810/tests/test_harness.py
experiments/gss_sat_structure_20260816/README.md
experiments/gss_sat_structure_20260816/REFUTATION_STUDY.md
experiments/gss_sat_structure_20260816/scripts/build_kissat.sh
experiments/gss_sat_structure_20260816/scripts/build_repo.sh
experiments/gss_sat_structure_20260816/scripts/campaign_stats.sh
experiments/gss_sat_structure_20260816/scripts/cone.sh
experiments/gss_sat_structure_20260816/scripts/enumerate.cpp
experiments/gss_sat_structure_20260816/scripts/gen_widths.sh
experiments/gss_sat_structure_20260816/scripts/glue_raw.sh
experiments/gss_sat_structure_20260816/scripts/glue_stats.sh
experiments/gss_sat_structure_20260816/scripts/influence.cpp
experiments/gss_sat_structure_20260816/scripts/ladder.py
experiments/gss_sat_structure_20260816/scripts/lucky_test.py
experiments/gss_sat_structure_20260816/scripts/n128_msweep.py
experiments/gss_sat_structure_20260816/scripts/n128_struct.sh
experiments/gss_sat_structure_20260816/scripts/r3r4.py
experiments/gss_sat_structure_20260816/scripts/refute.py
experiments/gss_sat_structure_20260816/scripts/refute_run.py
experiments/gss_sat_structure_20260816/scripts/rng_diag.py
experiments/gss_sat_structure_20260816/scripts/run_cone.sh
experiments/gss_sat_structure_20260816/scripts/run_enum.sh
experiments/gss_sat_structure_20260816/scripts/run_influence.sh
experiments/gss_sat_structure_20260816/scripts/run_n128m6000.sh
experiments/gss_sat_structure_20260816/scripts/seedrate.cpp
experiments/gss_sat_structure_20260816/scripts/sem_converge.sh
experiments/gss_sat_structure_20260816/scripts/semantic.cpp
experiments/gss_sat_structure_20260816/scripts/setup.sh
experiments/gss_sat_structure_20260816/scripts/sim.py
experiments/gss_sat_structure_20260816/scripts/slice_probe.py
experiments/gss_sat_structure_20260816/scripts/source_baseline.py
experiments/gss_sat_structure_20260816/scripts/speedup_bound.cpp
experiments/gss_sat_structure_20260816/scripts/spotcheck.sh
experiments/gss_sat_structure_20260816/scripts/structure.py
experiments/gss_sat_structure_20260816/scripts/target_dist.py
experiments/gss_sat_structure_20260816/scripts/times.py
experiments/gss_sat_structure_20260816/scripts/verify_encoder.sh
experiments/new_sss_painless_20260717/C80_MIGRATION_STATUS.md
experiments/new_sss_painless_20260717/COMPACT300_C20_STATUS.md
experiments/new_sss_painless_20260717/COMPACT300_C30_STATUS.md
experiments/new_sss_painless_20260717/COMPACT300_FLEET_STATUS.md
experiments/new_sss_painless_20260717/NEW_SSS_MILLION_FMIX_RESULTS_DRAFT.md
experiments/new_sss_painless_20260717/NEW_SSS_RESULTS_SUMMARY.md
experiments/new_sss_painless_20260717/README.md
experiments/new_sss_painless_20260717/c80_summary_v2/C80_RESULTS_SNAPSHOT.md
experiments/new_sss_painless_20260717/canonicalize_compact300_fleet_payloads.sh
experiments/new_sss_painless_20260717/collect_c80_host.sh
experiments/new_sss_painless_20260717/collect_new_sss_host.sh
experiments/new_sss_painless_20260717/export_new_sss_candidate.sh
experiments/new_sss_painless_20260717/repair_painless_runtime.sh
experiments/new_sss_painless_20260717/run_compact300_c20_reuse_v3.sh
experiments/new_sss_painless_20260717/run_compact300_c30_trial.sh
experiments/new_sss_painless_20260717/run_compact300_fleet_paired_v2.sh
experiments/new_sss_painless_20260717/run_new_sss_painless_sat_only.sh
experiments/new_sss_painless_20260717/run_new_sss_painless_trial.sh
experiments/new_sss_painless_20260717/run_parallel_fmix_sat.sh
experiments/new_sss_painless_20260717/run_raw_first_c80_sat.sh
experiments/new_sss_painless_20260717/start_campaign_host.sh
experiments/new_sss_painless_20260717/start_compact300_c20_host.sh
experiments/new_sss_painless_20260717/start_compact300_c30_host.sh
experiments/new_sss_painless_20260717/start_compact300_fleet_host_v2.sh
experiments/new_sss_painless_20260717/start_painless_repair.sh
experiments/new_sss_painless_20260717/start_parallel_fmix_sat.sh
experiments/new_sss_painless_20260717/start_raw_first_c80_sat.sh
experiments/new_sss_painless_20260717/start_relocated_c80_sat.sh
experiments/new_sss_painless_20260717/stop_excluded_sat_only_host.sh
experiments/new_sss_painless_20260717/stop_parallel_fmix_for_raw_first.sh
experiments/new_sss_painless_20260717/supersede_c2_for_c80.sh
experiments/nonlinear_compact300_heatmap_20260722/RESULTS.md
experiments/nonlinear_compact300_heatmap_20260722/run_heatmaps.sh
experiments/nonlinear_compact300_heatmap_20260722/run_remote_build.sh
experiments/nonlinear_compact300_heatmap_20260722/run_remote_finish.sh
experiments/nonlinear_gadget_analysis_20260727/RESULTS.md
experiments/nonlinear_gadget_analysis_20260727/run_current_v05_allwire_compression_probe.sh
experiments/nonlinear_gadget_analysis_20260727/run_current_v05_compression_probe.sh
experiments/nonlinear_gadget_analysis_20260727/run_portaware_v05_audit.sh
experiments/nonlinear_gadget_analysis_20260727/run_reproducibility_probe.sh
experiments/nonlinear_mixing_pilot_20260722/PILOT_RESULTS.md
experiments/nonlinear_tdp4n_20260725/README.md
experiments/nonlinear_tdp4n_20260725/STATUS.md
experiments/nonlinear_tdp4n_20260725/V04_STATUS.md
experiments/nonlinear_tdp4n_20260725/continue_nonlinear_tdp4n_v05.sh
experiments/nonlinear_tdp4n_20260725/poll_nonlinear_tdp4n.sh
experiments/nonlinear_tdp4n_20260725/poll_nonlinear_tdp4n_v04.sh
experiments/nonlinear_tdp4n_20260725/poll_nonlinear_tdp4n_v05.sh
experiments/nonlinear_tdp4n_20260725/run_nonlinear_tdp4n.sh
experiments/nonlinear_tdp4n_20260725/run_nonlinear_tdp4n_v04.sh
experiments/nonlinear_tdp4n_2233_degree2_20260726/RESULTS.md
experiments/quick_2223_fmix_affine_20260728/README.md
experiments/sliced_sandwich_2233_degree2_20260726/RESULTS.md
experiments/sss_fmix_painless_20260716/SSS_FMIX_RESULTS_SUMMARY.md
experiments/sss_fmix_painless_20260716/SSS_SSG_HYBRID_STATUS.md
experiments/sss_fmix_painless_20260716/run_sss_fmix_painless_trial.sh
```

### Local_Mixing_Documentation/ — 1 enumerated files

```text
Local_Mixing_Documentation/main.md
```

### mixing_tests/ — 15 enumerated files

```text
mixing_tests/BENCH_PLAN.md
mixing_tests/app-cred-nihodb-openrc.sh
mixing_tests/heatmap.py
mixing_tests/histogram_folder/result.md
mixing_tests/patches/libfaster-sys/Cargo.toml
mixing_tests/patches/libfaster-sys/Cargo.toml.orig
mixing_tests/patches/libfaster-sys/build.rs
mixing_tests/patches/libfaster-sys/src/lib.rs
mixing_tests/tests/circuit.rs
mixing_tests/tests/contiguous.rs
mixing_tests/tests/genran.rs
mixing_tests/tests/poly_canon_graph.rs
mixing_tests/tests/polynomial.rs
mixing_tests/tests/samf_compress.rs
mixing_tests/tests/transforms.rs
```

### red_team_tests/ — 1 enumerated files

```text
red_team_tests/sanity_checks/README.md
```

### sattest/ — 5 enumerated files

```text
sattest/CHALLENGE.md
sattest/solver/STATUS.md
sattest/solver/monitor_agent/README.md
sattest/solver/monitor_agent/monitor_preimage.sh
sattest/solver/structure_agent/REPORT.md
```

### work/ — 4 enumerated files

```text
work/db_host_compare_20260729/RESULTS.md
work/db_host_compare_bounded_20260729/RESULTS.md
work/frozen_curated_m1_m11/BUILD_REPORT.md
work/frozen_curated_m1_m11_native/BUILD_REPORT.md
```
