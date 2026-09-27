# drip-reshuffle archive: what exists only there

2026-09-27 · Ran Canetti · live version: https://claude.ai/code/artifact/c3f95909-bf29-4208-93e7-34029e3a4e72

## Summary

Work continues on `main` as of 2026-09-27. `drip-reshuffle` is frozen as the archive; this doc lists what exists only there.

- **Where the archive is:** branch `drip-reshuffle`, tip `5f386381` (the WIP snapshot of 2026-09-27), whose parent `046337ea` is the last commit `main` was built on. `main` (`0b23da71`) adds 7 commits of reorganization by honbugit: 445 files exist only on the archive, 238 only on `main`, 76 were renamed.
- **Most committed features were carried over** under new names (e.g. `fmix` became `circuit_mixer`, `gadgets.rs` became `src/stages/preprocessing/construct.rs` + `embedded_masking.rs`). Quad-fire, balanced masks, `min_open`, straddle/repair rerand, fcompress transport + `esop1`, drain set, g57 twists, `db_advance` and generation targeting are all referenced on `main`.
- **What is archive-only:** the 2026-09-27 WIP edits (never on `main`), about 60 analysis and red-team binaries, the `tools/` folder, most write-ups in `docs/`, and experiment reports.

On GitHub the archive is `ran/archive/drip-reshuffle` (and `ran/archive/ssg-gen-mix-clean`). Older local-only branches and the stash are not on GitHub; they are in `~/local_mixing_archive_branches_20260927.bundle` on the old Mac.

To get a single file back without switching branches:

```
git show drip-reshuffle:src/engine/mix.rs > /tmp/mix.rs
```

To browse the whole archive side by side: `git worktree add ../lm-archive drip-reshuffle`.

## Preprocessing and gadgetization

The blinded-V5 construction itself survived: quad-fire, balanced masks, `min_open = 2`, coverage rules and the two-control masked fire on borrowed dirty wires all live in `src/stages/preprocessing/embedded_masking.rs` on `main`. What did not survive is the 2026-09-08 WIP and the older gadgetizers.

| Feature | Archive location | On `main` |
| --- | --- | --- |
| Shared production band fill for modules 2 and 4 (`emit_production_band_fill`, same aux-fill as product-2223; `BV5_LEGACY_SEED=1` restores the 2-CNOT seed) | `src/preprocessing/gadgets.rs`, `bin/blinded_v5_gadgetize.rs`, `bin/gen_sandwich_gadget.rs` (WIP) | Missing |
| `blinded_v5_gadgetize` stand-alone CLI | `src/preprocessing/bin/blinded_v5_gadgetize.rs` | Logic in `gen_sandwich_gadget` + `embedded_masking`; this CLI gone |
| Drip gadgetizer (`gadgetize_drip_layered`, route-then-fire, NSWITCH) | `src/preprocessing/gadgets.rs` | Missing |
| Balanced sliced sandwich (`--sandwich-balanced`) | `src/preprocessing/gadgets.rs` | Missing |
| Product-share encoding, Gray fold CG, `--prod-fill-nl` | `src/preprocessing/gadgets.rs` | Traces only (1 file each) |
| SAMF licence / litter module | `src/preprocessing/samf.rs` | Partly referenced; module gone |
| FASKRI / M-lane generators | `experimental/preprocessing/faskri_gen.rs`, `faskri_m_gen.rs` | Missing |
| `g57_to_mpmct1` converter | `src/preprocessing/bin/g57_to_mpmct1.rs` | Missing |

## Mixer (fmix and engine)

`fmix` became `src/programs/circuit_mixer`, and `src/engine/mix.rs` was split into `src/engine/mixer/*`. The WIP window samplers and the drain-set side scheduling are the notable losses.

| Feature | Archive location | On `main` |
| --- | --- | --- |
| Kill-list seeding: `--p-target` seeds DB rounds at externally measured exposure witnesses; twist variant re-encodes in place | `src/engine/mix.rs`, `src/db_mixing/bin/fmix.rs` (WIP) | Missing |
| Depth-seeking sampler (`--p-depth`, `DbSample::MaxDepth`, uniform-exploration escape) | same (WIP) | Missing |
| Depth-balancing sampler (`DbSample::DepthGain`, seeds below median depth) | same (WIP) | Missing |
| `swap_refresh` sides for the drain set (steered retirements, default 3) | `src/engine/mix.rs` | Drain set present; `swap_refresh` gone |
| HD-gate / leak-seek sampler | only in `~/local_mixing_hg` on .242 — never in this repo | Lost with .242 unless backed up |
| Legacy mixing modules: convex, pairs, ranking, `sat_score`, transpositions, `main_mix_cnot`, segcircuit | `src/db_mixing/*.rs` | Convex comp and `sat_score` referenced; the rest gone |
| Legacy CLI subcommands `ssg`, `sss`, `gss`, `shoot`, `shuffle`, `compress`, `genran`, `equal` | `src/commands/*.rs` | Gone |
| GSS-MIX 6-stage pipeline script | `scripts/gss_mix.sh` | Gone |
| Split engine, wide fragments | `src/experimental/split_engine.rs`, `src/circuit/wide_fragment.rs` | Gone |

## Post-processing, DB generation, experiments

fcompress (transport + `esop1` packing), crossing and splitting all moved to `src/stages/post-processing/`. The losses here are mostly small binaries and the store-convention fix.

| Feature | Archive location | On `main` |
| --- | --- | --- |
| Explicit value convention for pool swaps (`pool_swap_conv`, legacy-swapped vs native stores) | `src/db_generation/frozen_build.rs` (WIP) | Missing; `db_gen/frozen_build.rs` has the older version |
| `python-heatmap` feature gate (pyo3 optional so ordinary binaries link without a libpython shim) | `Cargo.toml`, `src/lib.rs` (WIP) | Solved on `main` differently (feature `python-extension`) |
| Identity minting: `extract_identities`, `mint_long_identities`, `db_unit_synth` | `experimental/db_generation/` | Missing |
| MGDB / SGDB build, `frozen_census`, `frozen_find_small` as `src/bin` binaries | `src/bin/`, `experimental/db_generation/` | Moved to `db_gen/analysis/` (renamed copies exist) |
| Legacy compress, `fsplit`, `fsplit_trace` | `src/postprocessing/compress.rs`, `experimental/postprocessing/` | Superseded by `stages/post-processing/` |
| Small experiment probes: `flip_match`, `float_histogram`, `window_span_stats`, `sandwich_compare`, `fragment_wide`, `shuffle_mpmct` | `experimental/` | Gone (`shuffle_mpmct` moved to `security_tests/fixtures`) |
| Challenge circuits: block cipher, Newton–Feistel, point function, polynomial canonicalization | `challenges/` | Gone |
| Poly-canon graph experiment + its 3 tests | `src/experimental/poly_canon_graph.rs`, `tests/poly_canon_*.rs` | Gone |
| Generators / evaluators: `eval_c`, `eval_point`, `gen_profile_random`, `gen_random_mpmct` | `src/bin/` | `eval_point`, `gen_random_mpmct` in `security_tests/fixtures`; others gone |

## Red-team and leakage tools

`red_team_tests/` became `security_tests/` (heatmaps, gauntlet, demixing, SAT, fixtures), but most of the leakage and oracle probes were not carried over. `main`'s own security binaries need `--features security-tools`.

| Tool | What it measures | Archive location |
| --- | --- | --- |
| `segment_deduce` | Recoverable source segments / gadget state at dense cuts | `red_team_tests/bin/leakage/`, `src/bin/` |
| `segment_stats` | Per-wire fanout, dead segments, firing twins | `red_team_tests/bin/leakage/` |
| `collision_blocks` | Identity brackets / removable identical-gate pairs | `red_team_tests/bin/leakage/` |
| `fire_corr` | Fire-block correlation for blinded-V5 | `red_team_tests/bin/leakage/` (a fire-corr check is referenced on `main`) |
| `stress_battery` | Progress-aligned bias battery | `red_team_tests/bin/leakage/` |
| `persistence_census`, `sampled_trace_support`, `source_stats` | Band persistence and trace support | `red_team_tests/bin/leakage/` |
| `invariant_classify`, `invariant_scan`, `quad_relations`, `wire_census` | Cross-cut affine / quadratic invariants | `red_team_tests/bin/leakage/`, `src/bin/` |
| `preimage_affine`, `preimage_brute`, `preimage_cnf` | Oracle preimage probes | `src/bin/` |
| `oracle_gate_learn`, `oracle_preimage_game` | Oracle games | `red_team_tests/bin/oracle/` |
| `slice_match`, `slice_affine` | Slice-match heatmaps (phase A tracking) | `red_team_tests/bin/heatmaps/` |
| `fmix_gauge_score` | Demixing gauge score | `red_team_tests/bin/demixing/` |
| `prod_grid`, `regen_sandwich_c` | Test fixtures | `red_team_tests/bin/fixtures/` |
| Gauntlet README and testing-pipeline notes | How to run the gauntlet arms | `tests/gauntlet/` (the gauntlet code moved to `security_tests/gauntlet`) |

## Python tools, docs and reports

`main` has no `tools/` folder and keeps only 8 docs plus images, so all of the following is archive-only.

- **PR identity generator** (`tools/pr_identity/`, 28 files): `generator.py`, `id_strings.py` (the (d,k)-strings solution), `chain.py`, `atoms.py`, audits, the repeat-free construction and the `synth_mmd` synthesis experiments.
- **Heatmap ridge reader** `reports/plot_hmap_ridge.py` — the standard way plates are read. `main` has its own `security_tests/heatmaps/` plotters but not this one.
- **SAT / preimage tooling** (about 70 files in `tools/`): the `circuit_to_cnf_*` encoders, matching `decode_*_model` decoders, `search_*` anneal/Newton/Hamming-ball searchers, kissat probes and sweep runners (`run_lowtarget_*`, `run_skolem_kissat.py`, `simple_cdcl.cpp`).
- **Analysis helpers**: `analyze_collision_blocks.py`, `analyze_segment_stats.py`, `fmix_progression.py`, `compare_summary_csv.py`, `circuit_structure_scan.py`, `commute_reduce.py`, `mixwatch.sh`, `stagereport.sh`.
- **Scripts**: `scripts/gss_mix.sh` (GSS-MIX pipeline), `parameter_sweep.py` + worker, heatmap recompute scripts.
- **Design write-ups** (about 60 in `docs/`, `.md` / `.tex` / `.pdf`): `BLINDED_V5_LGI_DESIGN`, `BALANCED_MASKS_AND_COVERAGE_20260907`, `RIDGE_QUADFIRE_20260906`, `RIDGE_COVERAGE_EXPERIMENTS`, `FCOMPRESS_TRANSPORT_AND_PACKING`, `DRAIN_SET`, `SWAP_REFRESH_REDESIGN`, `SLICED_SANDWICH`, `PRODUCT_SHARE_ENCODING` + update, `GRAY_FOLD_CG`, `NONLINEAR_GADGETIZATION`, `NEW_GADGETIZE`, `GADGETIZE_UPDATE_20260822`, `FMIX_LAYER1/2`, `FMIX_PHASE_A`, `FMIX_PARAM_PRECEDENCE`, `FMIX_MENU`, `G57_TWIST_BRACKETS`, `GSS_MIX`, `PIPELINE_OVERVIEW`, `POSTMIX_MANUAL`, `HMAP_BANDS`, `BAND_HARDENING`, `ANCESTRY_*`, `DB_CAMPAIGN_20260805`, `CURATED_DB_COMPARISON`.
- **Experiment reports** committed under `reports/` (ancestry, split trials, band hardening, prod33, ssgprod); the untracked bulk of `reports/`, `circuits/` and `work/` is outside git on either branch.

Many of the memory notes point at these docs and tools by path; on `main` those paths resolve only via `git show drip-reshuffle:<path>`.

## Porting notes

Port the WIP first: it is the only newer-than-`main` work, and it is small (about 700 changed lines across 6 files). Everything else is stable and can be pulled in when an experiment needs it.

| Priority | Piece | Lands on `main` in | Effort |
| --- | --- | --- | --- |
| 1 | Shared production band fill for modules 2/4 | `src/stages/preprocessing/construct.rs` / `embedded_masking.rs`, `src/programs/gen_sandwich_gadget` | Small |
| 1 | Kill-list, depth-seeking and depth-balancing samplers (+ flags) | `src/engine/mixer/sampling.rs`, `params.rs`; `src/programs/circuit_mixer/cli.rs` | Medium (~600 lines) |
| 1 | `pool_swap_conv` store convention | `db_gen/frozen_build.rs`, `db_gen/analysis/frozen_pool_swap.rs` | Small |
| 2 | Leakage probes used in current work: `segment_stats`, `collision_blocks`, `fire_corr`, `segment_deduce`, `invariant_classify` | `security_tests/leakage/` behind `security-tools` | Small each; mostly self-contained |
| 2 | `plot_hmap_ridge.py`, `pr_identity/` | `security_tests/heatmaps/`, a new `tools/` or `db_gen/` Python package | Copy as-is |
| 3 | Drip gadgetizer, balanced sliced sandwich, product-share / Gray fold | `src/stages/preprocessing/` | Large; only if revived |
| 3 | SAT / CNF C++ tooling, challenge circuits, legacy subcommands | `security_tests/sat_solve/` | Copy on demand |

For each port: copy from the archive with `git show drip-reshuffle:<path>`, adapt to `main`'s module paths, build with the relevant `--features`, and cite the archive commit (`5f386381`) in the commit message.
