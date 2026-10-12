# The seven-stage pipeline: what changed on 2026-10-11

*RC's request: redo the Rust pipeline with the old crossing and splitting stages back in between
DB mixing and packing, plus a new last stage, unpacking, that resynthesizes a circuit from each
delivered ANF window with a standard ANF synthesis, concatenates the pieces, puts the gates into
the canonical commuting order and outputs. Every stage after the first is optional; the default
runs all seven. RC's decisions on the two open points: stage 4 = splitting, stage 5 = crossing
(the archived, validated order), and the stage-7 synthesizer is the port of `anf_synth`.*

Branch `ran/seven-stage` (worktree `~/code/lm_worktrees/port_wt`, stacked on `ran/stage4-rust`
`0569792e`), nothing pushed.

| commit | what |
|---|---|
| `9499f42a` | stages 4+5 revived from the archive (`ea2e516f`): reverse of cleanup steps 3 and 4, 3-way merged onto the port; state v6 (= the v3 layout); resumed-run header fix |
| `856a9d10`..`f32045bf` | stage 7 (subagent): O(scan) `xgate_canonicalize`, `src/stages/unpacking/`, `tdp_unpacking`, `docs/STAGE7_UNPACKING.md` |
| `1c8fa87e` | the seven-stage driver (bash + `src/tdp`): optional stages, five seeds, markers, config v3, recipe v11 |
| `b43c5da9` | docs |
| `a8d77ac6` | review fixes (independent review of the driver: 1 bug, 2 likely bugs, 6 nits, all fixed) |

## 1. The pipeline

| stage | command | output | optional |
|---|---|---|---|
| 1 sandwich | `gen_sandwich_gadget` | `tdp.mpmct1.sandwich.mpmct1`, `.source_c.g57` | no |
| 2 preprocessing | (same process) | `tdp.mpmct1` | yes: the sandwich goes on |
| 3 DB mixing | `circuit_mixer --tdp --db-mixing --p-db 1.0` | `db_mixing.mpmct1` | yes |
| 4 splitting | `circuit_mixer --split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 --moves G+1e6` | `split.mpmct1`, `split.state` | yes |
| 5 crossing | `circuit_mixer --resume split.state --target-size round(2G) --temp max(64,target/25) --split-base 3 --split-damp 1 ... --moves state+round(F*target)`, F = `crossing.move_budget_factor` = 2 (RC 2026-10-11; the archive ran 6) (or `--input prev --seed-file` when 4 is skipped) | `crossing.mpmct1` | yes |
| 6 packing | `tdp_packing run --stage2 tdp.mpmct1 --stage3 <last>` | `packing/deliverable.tdp4` (+ `rep<G>.*`) | yes |
| 7 unpacking | `tdp_unpacking run --deliverable ... --packed rep<G>.mpmct1 --bounds ... --source-c C --n N` | `unpacking/unpacked.mpmct1` | yes (needs 6) |

Stages 4 and 5 are the archive-era driver's commands verbatim (the flags of
`docs/notes/stage45_revival_20261011.md`). `--run-skip-stages 4,5,7` / `run.skip_stages = [4, 5, 7]`
is the validated four-stage chain byte for byte: `tests/shell/tdp_defaults.bash` diffs the managed
transcript against the sandboxed `val_chain.sh` under that skip list (bash and Rust drivers).
`pinned_recipe` is recorded only for that skip list.

Seeds: `SEEDS` format 3 = gen, mix, split, cross, unpack (+ `run_key`, `seed_source`); the cross
seed is used only when stage 5 starts from a file (a resume of `split.state` continues that RNG);
every seed travels on `/dev/fd/3`; format-2 files of the four-stage driver are upgraded in place on
a standalone resume (the managed driver refuses v10 manifests). Markers: `stage4.recipe`,
`stage5.recipe` (tags `split.state` in resume mode, the cross seed in input mode),
`packing.recipe`, `unpacking.recipe`; failure markers `packing.failed`, `unpacking.failed`.

## 2. Stage 7

Port of `anf_synth` (bidirectional ANF search, same defaults: beam 6, 16 restarts, 30 s/window,
peephole). The synthesizer sees only the window spec built from the decoded deliverable
(support, changed wires, monomials) on a local wire universe with a per-window monomial width; the
packed gates are used only as the verbatim fallback of a window that does not synthesize, and only
after their ANF is checked equal to the delivered one (`--no-fallback` makes it fatal). Checks
before the output is committed: every window exact (recomposed ANF == delivered blob) and sampled;
the whole circuit against the deliverable's evaluator before and after `xgate_canonicalize`
(now O(scan) via slice rotation; 2.6 s on 0.53M gates); the zero slice against C.

Measured by the subagent on the n=64 L8 base arm's **four-stage** deliverable
(`dressed_runs/base/packing`): 2,591 windows, 0 fallback, 1 deadline hit, 1,487,674 → 526,874
gates (0.354, = `anf_synth`'s 526,958), 255 s on 8 threads.

## 3. The n=16 seven-stage run (real binaries, frozen_m1_m11, `port_runs/seven_n16`)

| stage | gates | wall |
|---|---|---|
| 1+2 | 45,632 (S: 208) | |
| 3 | 92,624 | 2.5 min |
| 4 split | 177,082 | 2 s |
| 5 crossing (target 354,164, temp 14,167, 2.13M moves) | 322,373 | 18 s |
| 6 packing | 639 windows, floor 16/0 (`final_nonoutput_visible=0`), all checks ok | 7 min |
| 7 unpacking | 621 synthesized, **18 verbatim fallbacks**, 27 deadline hits; 316,269 → 286,302 gates (0.905); all checks ok | 5.7 min (6 threads) |

Reading: on the **crossing output** the windows are much harder to resynthesize than on the stage-3
output (multi-control gates, denser ANF: p90 8.4 s/window, max 39 s, vs 1.2 s / 30 s on the
four-stage deliverable), and the gate ratio is 0.9 instead of 0.35. `status=fallback_used` is
reported, not fatal, by default.

## 4. The n=64 seven-stage run (managed driver, `port_runs/seven_n64`, `threads = 8`, nice 10)

Fresh run started 2026-10-11 22:26 after the stage-5 budget change (`crossing.move_budget_factor`
= 2, RC's default; the archive ran 6); `PIPELINE COMPLETE (stages 1-7)` at 01:39, exit 0.

| stage | gates | wall |
|---|---|---|
| 1+2 (L8, \|C\|=1000) | 775,332 (S: 2,832) | 1 min |
| 3 DB mixing (4,031,726 moves) | 1,583,698 | 29 min |
| 4 split | 2,987,961 | 1 min |
| 5 crossing (target 5,975,922, temp 239,037, budget 11.95M = 2×target) | 5,380,721 | **7 min** (46 min at 6×target in the revival study) |
| 6 packing | 10,150 windows → repair 5,153,273 gates / 10,365 bounds; floor 64/0; every check ok (reorder, pack, repair certificates, equiv, windows ANF 0/10,366 over budget, decode, eval, zero slice) | 88 min |
| 7 unpacking | 10,366 windows: 10,209 synthesized, **157 verbatim fallbacks (1.5 %)**, 371 deadline hits; 5,153,273 → 4,766,644 gates (ratio 0.925); checks ok (exact + sampled per window, equiv before/after canonical order, zero slice); canonical reorder 13 s | 68 min (8.9 CPU-hours of synthesis) |

Reading, same as at n=16: on the crossing output the resynthesis buys only 7.5 % (0.925) where the
four-stage deliverable gave 0.354, with 1.5 % of windows emitted verbatim and 3.6 % hitting the
30 s deadline (p90 6.9 s, max 200 s per window). The crossing walk with the 2×target budget stops
at 0.90 of its target (5.38M vs 5.98M) instead of settling on it; the stage took a seventh of
the time.

## 5. Tests

`cargo test --lib` 457 (goldens: `GOLDEN_PROFILE_NULL_PLANT` back at its pre-cleanup constant,
the hashed layout being v3's again; `GOLDEN_STAGE3_FIXTURE` re-pinned with its trajectory hash
unchanged; split-stop and thermostat goldens back), integration 30 (`tdp_unpacking` 4 new),
clippy and the all-targets check clean; shell: `tdp_flags`, `tdp_block_size` (four-stage chain
under the skip list + the full chain with every stage-5/7 knob and `--packing-window-gates 256`),
`tdp_seeds` (five seeds, upgrade, explicit extras), `tdp_stage_resume` (renamed; stage 4/5/7
markers, reruns, stop flag, exits), `tdp_defaults` (bash and Rust driver vs `val_chain.sh`),
`tdp_stage12_parity`.
