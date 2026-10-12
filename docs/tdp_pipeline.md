# The TDP Pipeline

This is the operational manual for the six-stage TDP pipeline: how to run
each stage, what it writes, and where its code lives. We start with a
reversible circuit $C$ on $n$ wires and end with a published circuit that
computes $C$ on a public slice.

| Stage | What it does | Runs as | Output |
| --- | --- | --- | --- |
| 1. Sandwich | Classic sliced sandwich around $C$ on $2n$ wires | `gen_sandwich_gadget` (with stage 2) | `tdp.mpmct1.sandwich.mpmct1`, `tdp.mpmct1.source_c.g57` |
| 2. Preprocessing | Embedded masking on $4n$ wires: masked fires, band, guards, persistent carrier migration (preset L8) | `gen_sandwich_gadget` | `tdp.mpmct1` |
| 3. Mixing | Database mixing with the regular frozen store | `circuit_mixer --tdp --db-mixing --p-db 1.0` | `db_mixing.mpmct1` |
| 4. Splitting | Every complemented g57 gate split into plain conjunctions, joined to distant brackets through absorbed NOT twists, to exhaustion | `circuit_mixer --split --split-stop` | `split.mpmct1`, `split.state` |
| 5. Crossing | The directional crossing walk under a size thermostat (target 2× its input, budget 2× the target) | `circuit_mixer --resume split.state --target-size …` | `crossing.mpmct1` |
| 6. Delivery | Packing (commuting reorder into ANF windows, repair, serialization), then unpacking with lattice refinement: every window resynthesized from its ANF alone, a window that does not resynthesize cut in two and both halves repacked and unpacked, the result concatenated and canonically reordered | `tdp_packing run`, then `tdp_unpacking run --refine` | `packing/deliverable.tdp4` (the packer's lattice), `unpacking/deliverable.tdp4` (the refined lattice, the shipped file), `unpacking/unpacked.mpmct1` |

```text
C → sliced sandwich → embedded masking → DB mixing → split → crossing walk → packing → unpacking with refinement
       stage 1            stage 2          stage 3     stage 4    stage 5      stage 6 (delivery)
                                                                         packing/deliverable.tdp4 → unpacking/deliverable.tdp4
                                                                                                     unpacking/unpacked.mpmct1
```

**Every stage after the first is optional** (`run.skip_stages`,
`--run-skip-stages`); the default runs all six. Each stage reads the output
of the last stage that ran: with stage 2 skipped the sandwich itself goes on,
with stage 4 skipped the crossing walk starts from the previous circuit with
its own seed, with stage 3 skipped the split runs on the stage-2 circuit, and
so on. Within stage 6, `delivery.anf_only` ends the stage after the packer's
deliverable (no resynthesis). The final artifact is the last stage's output;
`PIPELINE COMPLETE (stages 1-6): …` names it.

Stage 6 was two stages, 6 (packing) and 7 (unpacking), until 2026-10-12
(RC's decision: a window that does not resynthesize is a packing defect to
be split and repacked, not a fallback; [DELIVERY_STAGE.md](DELIVERY_STAGE.md)).
A config of that driver (`config_version = 3`, an `[unpacking]` table,
`run.skip_stages` with a 7) is refused with a pointer.

The managed `tdp_gen` command drives the stages. **The defaults of stages
1-3 and the packer reproduce the chosen validation runs** (`val_chain.sh`
CONFIG=L8, runs vL64a/b and vL128a/b): a managed run that sets only
`source.wires` (64 or 128), `run.skip_stages = [4, 5]` and `delivery.anf_only
= true` produces, from the same seeds, byte-identical stage-2, stage-3,
packed, repaired and deliverable artifacts. This is checked on the archived
runs themselves (the historic anchor) and by a reproduction test against the
reference script; see [docs/STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md) §2
and §12 (written when packing was stage 4; its stage numbers are the
four-stage ones). Stages 4 and 5 run with the flags of the archive-era driver
(`ea2e516f`), revived 2026-10-11; the unpacking sub-step dates from the same
week.

The packing sub-step of stage 6 is the Rust port (`src/stages/packing/`,
binary `tdp_packing`) of the Python prototypes in
[scripts/windowed_delivery/](../scripts/windowed_delivery/), which stay as
the canonical **oracle**: the port is bit-exact wherever the oracle is
deterministic, and CI compares the two on random inputs on every push.

The curated store and the former stage 6 (`fcompress`) are historic. They
remain on branch `ran/archive/curated-and-postmix`; see
[Curated DB and post-mixing archive](CURATED_AND_POSTMIX_ARCHIVE.md) for what
was retired, how to retrieve it, and why (the split stage and the crossing
walk listed there are back in the pipeline as stages 4 and 5).

Further reading:

- [TDP_FOUR_STAGE_PIPELINE.md](TDP_FOUR_STAGE_PIPELINE.md): the full
  reference for stages 1, 2, 3 and 6 (construction details, parameters,
  measured behaviour, pass criteria, open issues; written when packing was
  stage 4).
- [STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md): the packing port, the
  reference recipe the defaults reproduce, seeds and the environment policy,
  and the validation results.
- [DELIVERY_STAGE.md](DELIVERY_STAGE.md): the delivery stage (the decision,
  the refinement contract, the Python reference and the Rust binaries).
- [STAGE7_UNPACKING.md](STAGE7_UNPACKING.md): the unpacking sub-step of
  stage 6 (`tdp_unpacking`: its synthesizer, its checks, its exit codes and
  its measured results; written when it was stage 7).
- [STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md): the stage-2 redesign and
  the linkable-exposure metrics that the recommended configuration is chosen
  by.
- [ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md): windowed ANF packing, its
  threat model, and the deliverable format.

For first-time setup and the circuit utilities, start with the
[README](../README.md).

## Two ways to run

**The managed driver** (`tdp_gen`) is the supported entry point. It reads a
TOML recipe (config v4), builds and fingerprints the four stage binaries,
draws the seeds into the private `$RUN/SEEDS`, records a manifest (recipe
v12), and resumes runs safely. Its defaults are the recommended
configuration: stage-2 preset `L8`, stage-3 schedule `validated`, the packer
at G = 512 with a 65,536-monomial split budget, the refinement at 30 s per
window per round for up to 16 rounds.

**The reference driver**
[scripts/windowed_delivery/waist/val_chain.sh](../scripts/windowed_delivery/waist/val_chain.sh)
is the script the validation runs were made with. It is now a **parity
tool**: it hard-codes the run tree's paths, draws seeds below $2^{31}$ /
$2^{40}$ (calibration grade) and runs the Python stage 4. The reproduction
test (`tests/manual/stage4/reproduce_val_chain.sh`) runs it in a
path-rewritten sandbox next to the managed driver with the same seeds and
compares every artifact.

## Quick start

```bash
cargo build --release --locked
cp configs/tdp.toml configs/local.toml        # set database.regular_dir (and source.wires)
cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

The run ends with `[tdp_gen] PIPELINE COMPLETE (stages 1-6): <run>/unpacking/unpacked.mpmct1`
(under `delivery.anf_only`: `<run>/packing/deliverable.tdp4`). A run
directory holds:

| Stage | Files |
| --- | --- |
| 1. Sliced sandwich | `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` |
| 2. Preprocessing | `tdp.mpmct1`, `stage12.log`, `stage12.recipe`; report-only `stage2_mult.txt` (rebirth census) and `stage2_span.{tsv,txt}` (val_chain.sh's span check; the managed driver builds the `security-tools` `hmap_affine` it needs, a standalone run skips it unless `$TDP_BIN_DIR/hmap_affine` exists) |
| 3. Database mixing | `db_mixing.mpmct1` (the mixer writes `db_mixing.mpmct1.tmp`, fsyncs it and renames it, so a kill or crash never leaves a truncated output under this name), `stage3.log`, `stage3.settings` (the resolved `FROZEN_FILTER` and the mixer's `stop_reason`), `stage3.recipe` (keyed completion marker: the schedule arguments, HMAC-SHA256 tags of `tdp.mpmct1` and of `db_mixing.mpmct1`, the stop reason; written only after an accepted stop) (and `db_mixing.leakage_repair.txt` with leakage repair). A resume skips stage 3 only when `stage3.recipe` matches; a missing or stale marker (an interrupted output, a changed stage-2 input or schedule) re-runs stage 3, which then needs the store. A run made before the marker existed is adopted without re-mixing when its `delivery.recipe` (or the seven-stage driver's `packing.recipe`) carries the tags of exactly these `tdp.mpmct1` and `db_mixing.mpmct1` bytes and `stage3.settings` records an accepted stop. A chain ended by the operator stop flag (`touch $RUN/stage3.stop`) is under-dosed: its output is moved to `db_mixing.stopped.mpmct1`, the run stops with exit 2 and no deliverable, and a resume re-runs stage 3 from scratch |
| 4. Splitting | `split.mpmct1`, `split.state` (the resume point of stage 5), `stage4.log`, `stage4.recipe` (keyed completion marker); a split ended by `$RUN/stage4.stop` is kept as `split.stopped.mpmct1` |
| 5. Crossing | `crossing.mpmct1`, `stage5.log`, `stage5.recipe` |
| 6. Delivery | `packing/`: `deliverable.tdp4` (the packer's lattice), `deliverable.bin` (its raw body), `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1`, `rep512.bounds` (the repaired circuit and the lattice the refinement starts from), the last C-view scan `rep512.{exposure,allcuts,wit}.tsv` + `rep512.cview.log`, the oracle-format reports `pack_G512.txt`, `repair512.log`, `anf_rep512.txt`, `link_rep512.txt`, and `packing.report`; `unpacking/` (absent under `anf_only`): `deliverable.tdp4` (**the shipped file**: the refined lattice), `bounds.final`, `unpacked.mpmct1` (the resynthesized circuit, canonical order), `windows.tsv`, `rounds.tsv`, `unpacking.report`, `unpacking.log`; `stage6.log` (both sub-steps, a separator line between them); `delivery.recipe` (keyed completion marker); a failed sub-step leaves `delivery.failed` (`exit=<code>`, `step=packing\|unpacking`; the managed driver then reports that code's meaning for that binary, and no other failure is labelled as stage 6's); `NOT_RECOMMENDED` next to the final output after an allowed failure |
| Run | `tdp_command.conf` (recipe manifest), `tdp_gen.log`, the private `SEEDS` |

Everything except `packing.report`, `unpacking.report` and the two
`deliverable.tdp4` is private: the circuits regenerate or expose $C$, and
`SEEDS` regenerates everything.

## Where a run starts

The `tdp_gen` command enters through [src/main.rs](../src/main.rs) and
[src/commands/tdp_gen.rs](../src/commands/tdp_gen.rs). From there,
[src/tdp/config/](../src/tdp/config/) reads and checks the recipe, and
[src/tdp/runner.rs](../src/tdp/runner.rs) prepares the run (the environment
check, `SEEDS`, the manifest), checks any saved manifest, and selects the
stage binaries.

Then [scripts/tdp_gen.sh](../scripts/tdp_gen.sh) runs the stages. This is the
easiest file to read when we want to see the exact stage order, arguments, or
output names. Stages 1 and 2 share one generator process,
`gen_sandwich_gadget`; `circuit_mixer` runs stages 3, 4 and 5; stage 6 runs
`tdp_packing run` and then `tdp_unpacking run --refine`. Each executable's
`main.rs` and command implementation live together under `src/programs/`.

## Configuration reference

Use [configs/tdp.toml](../configs/tdp.toml) as both the default recipe and the
annotated template (config v4). Copy it to `configs/local.toml`, put your
database paths and settings there, then run:

```bash
cp configs/tdp.toml configs/local.toml
cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

The managed command has these flags:

| Flag | Effect |
| --- | --- |
| `--config PATH` | Read a TOML recipe; default `configs/tdp.toml`. Relative paths resolve from the repository root. |
| `--dry-run` | Parse, resolve and validate the recipe, paths and resume fingerprints; print the planned stages without building or creating a run. |
| `-h`, `--help` | Show command help. |

In TOML, `number` accepts an integer or a finite float, `integer` requires an
integer, booleans are unquoted `true`/`false`, and paths/enums are quoted
strings. Empty strings are treated as omitted. Unknown keys, wrong types,
duplicate settings (including an alias plus its canonical spelling), and
invalid combinations are rejected before starting a run.

**Presets and the schedule are authoritative.** Under `preprocessing.preset =
"L8"` or `"A"` every stage-2 knob field is an error, even when it equals the
preset's value; under `db_mixing.schedule = "validated"` the schedule fields
are errors. An old configuration therefore cannot silently produce a hybrid
recipe. A `config_version = 1`, `2` or `3` file is rejected with a migration
message (`2` was the four-stage schema; `3` the seven-stage schema with
separate packing and unpacking stages: set `4`, move `[unpacking]` to
`[delivery]`, and write the four-stage chain as `run.skip_stages = [4, 5]`
with `delivery.anf_only = true`). An `[unpacking]` table is refused with the
same pointer.

### Schema and run management

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `config_version` | integer, required `4` | TOML schema identifier, separate from the saved recipe v12. `1`, `2` and `3` are rejected with migration messages. |
| `run.directory` | string, generated directory under `runs/` | Output directory and continuation target. Omit for a fresh timestamped run; select the same directory to continue it. Within the repository it must be a dedicated subdirectory of `runs/`; dedicated external directories are also allowed. It must not overlap build or database directories. |
| `run.build_binaries` | bool, `true` | Build `gen_sandwich_gadget`, `circuit_mixer`, `tdp_packing` and `tdp_unpacking` for a fresh run. `false` requires them under the resolved build directory. |
| `run.build_directory` | string, `CARGO_TARGET_DIR` or `target` | Cargo artifact directory; stage binaries come from its host-target `release/` directory. |
| `run.skip_stages` | integer array, `[]` | Stages not run, each in `2..6` (stage 1 always runs). Recorded in the manifest (`skip_stages=`): it changes the output. `[4, 5]` with `delivery.anf_only = true` is the validated four-stage chain. |
| `run.stop_after_stage` | integer, `6` | Last stage to reach, `2..6`. |
| `run.rerun_from_stage` | integer, unset | Recompute this stage and every later reached stage, `2..6`. Regenerating a stage always reruns the later stages. |
| `run.threads` | integer, unset | Pin `RAYON_NUM_THREADS` for the stage binaries and the `--threads` of `tdp_packing` and `tdp_unpacking`. Lifecycle only: stages 1-5 do not use the pool (only the piecewise mode, which has its own setting), the packer is tested to be thread-count independent, and the unpacking sub-step's output depends on threads only through its per-window deadline (`deadline_hits` in `unpacking.report`). |
| `run.adopt_unverified_run` | bool, `false` | Retired compatibility field. `true` is rejected. |

Lifecycle options (stop/rerun stage, threads, build selection) do not change
the recipe. Everything else, `run.skip_stages` included, is recorded in
`tdp_command.conf`; changing it requires a fresh run. The manifest starts
with `tdp_command_recipe=12` (recipe v11 runs, the seven-stage driver's, v10
runs, the four-stage driver's, and v9 runs are refused: resume them with the
checkout that created them); `stage12.recipe` uses `tdp_stage12_recipe=7`.

The manifest's `pinned_recipe` names the validated chain the run reproduces,
`val_chain-L8-20261007` or `val_chain-A-20261007`, and is `none` for anything
else. It requires every recipe setting of that chain (embedded masking with
preset L8 or A, the validated |C| for a generated source, schedule
`validated` in one piece, no leakage repair, default packing, stages 4 and 5
off and stage 6 ending after the packer's deliverable: `run.skip_stages =
[4, 5]` and `delivery.anf_only = true`; the then-unused unpacking parameters
do not matter) **and** stage 3
on the store the chosen runs used, unfiltered: `frozen_db_fingerprint`
(the name and size of every file in the store plus the text of each `.ok`
checksum marker, xxh3-128; the shards themselves are not read) must equal
that of `frozen_m1_m11`, and `FROZEN_FILTER=1` on a store without
`filters.bin`. A copy of the store with other shards or markers, a curated
store, or a filtered run (`lookup_miss_filter = "on"`) records `none`.

### Source circuit

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `source.wires` | integer, `128` | Logical width $n$ of $C$, in `3..4095`; at most 256 when stage 6 runs. |
| `source.gates` | integer, the validated table | $|C| = |D|$. Default: **1000 at n=64, 3000 at n=128** (the sizes the validation runs used); for any other $n$ a size or `gates_rule = "formula"` is required. Recorded as `source_gates_rule=validated` when it equals the table, `explicit` otherwise. |
| `source.gates_rule` | string, `"validated"` | `"formula"` = `round(n·log2(n)^2)` for any $n$; recorded as `formula` and printed as `UNVALIDATED SIZE`. Exclusive with `source.gates`. |
| `source.path` | string, unset | Supply $C$ as a nonempty G57 circuit instead of generating it (the route to a $C$ with more than 64 bits of entropy, §13.1 of the design). |

### Frozen replacement stores

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `database.regular_dir` | string, `FROZEN_DB_DIR` or unset | Regular frozen replacement store, required when stage 3 runs. A TOML path overrides the environment path. Needs nonempty `tables.bin` and all 256 shard files. |
| `database.lookup_miss_filter` | string, `"auto"` | `"auto"`: `FROZEN_FILTER=1` when the store has no `filters.bin` (the reference runs' situation: the mixer notes it and runs unfiltered); when `filters.bin` exists the filter stays off until its output neutrality is recorded. `"on"` requires `filters.bin`; `"off"` never loads it. The manifest records the mode with the value it resolves to (`frozen_filter=auto(1)`), `$RUN/stage3.settings` the value stage 3 set. |
| `database.allow_no_database_for_tests` | bool, `false` | Plumbing runs with all replacement channels disabled. |

### Preprocessing

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `preprocessing.mode` | string, `"embedded-masking"` | `"embedded-masking"` or `"nonlinear291"` (no preset, no knobs). nonlinear291 circuits are wide (24n + 12·(2\|C\| + 2s + n) + 29 wires: 3,677 at n=16, \|C\|=64), and stage 6 takes at most 1,024 state wires, so the driver refuses nonlinear291 with stage 6 up front unless the planned width fits; stop before stage 6 or skip it. |
| `preprocessing.preset` | string, `"L8"` | `"L8"`: `EMBEDDED_MASKING_SHUFFLING=8 SHUFFLING_RETURN_HOME=0 PREMASK=1 SEED_AVOID_C1=1 CHAIN=2 CHAIN_LEN=8` and nothing else, exactly val_chain.sh's L8 environment ([STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md) §6). `"A"`: the same without the chains. `"custom"`: every knob below is required. |
| `preprocessing.mask_pair_wires` | integer, custom only | Band wires per mask, `2..64` (base 2). |
| `preprocessing.max_open_masks` | integer, custom only | Open-mask cap, `2..64` (base 3). |
| `preprocessing.min_open_masks` | integer, custom only | Minimum open masks, `1..63`, below `max_open_masks` (base 2). |
| `preprocessing.balanced_masks` | bool, custom only | Balanced masks (base `true`). |
| `preprocessing.shuffling_segments` | integer, custom only | `0` off, else at least `8` (base 0). |
| `preprocessing.shuffling_return_home` | bool, custom only | `false` = persistent carrier migration; the stage-2 tail restores the ports (base `true`). |
| `preprocessing.premask` | bool, custom only | Up-front masking (base `false`). |
| `preprocessing.seed_avoid_c1` | bool, custom only | Band seed avoids the first-layer control pairs (base `false`). |
| `preprocessing.chain`, `preprocessing.chain_len` | integers, custom only | Bounded bracket chains (base 0, 0 = off). |

Under `custom` a knob's variable is exported to gen only when it differs from
gen's base value; every effective value is recorded in the manifest and in
`stage12.recipe`. The embedded mask layout requires
`2 * source.wires > 2 * floor(mask_pair_wires / 2) + int(balanced_masks)`.

### Database mixing schedule

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `db_mixing.schedule` | string, `"validated"` | `"validated"`: `--profile 2,2,2,2,2 --moves int(G*2*2*1.3)` (G = the stage-2 gate count from the `tdp.mpmct1` header; Python `int` truncation), serial, default `--verify-every`/`--report-every`. `"custom"` requires the three fields below. |
| `db_mixing.growth_work_units` | number, custom only | Profile $n_0$. |
| `db_mixing.hold_work_units` | number, custom only | Profile $n_1 = n_2 = n_0 + $ hold. |
| `db_mixing.target_size_factor` | number, custom only | Profile $r_1 = r_2$, in `(1, 16]`; moves `int((n0+hold)·r·G·1.3)`. |

The mixer runs with `FROZEN_REGULAR_VALUE_CONVENTION=native` and the
cache sizes `LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256
XPOLY_CANON_CACHE_MB=1024` (exact memo caches, classified output-neutral: no
dedicated test, but stage 3 reproduced the reference runs' bytes, made with
the built-in 512 MB lookup cache; `docs/STAGE4_RUST_DESIGN.md` D10b); the canonicalization caps are unbounded, as in the
reference runs.

### Splitting and crossing (stages 4 and 5)

Stage 4 has no settings: it runs the archive-era driver's flags to exhaustion
(`--split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100
--split-canaries 256 --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12`,
move ceiling = input gates + 1,000,000; the manifest records
`stage4_split_args=archive(ea2e516f)`). Stage 5 resumes `split.state` when
stage 4 ran (the walk continues the split stage's RNG and move counter) and
otherwise starts from the previous circuit with its own `cross_seed`.

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `crossing.target_size_factor` | number, `2.0` | Stage-5 target relative to its input size, in `1..16`: `target = round(input_gates * factor)`. |
| `crossing.width_penalty_base` | number, `3.0` | Base `B` of the wide-split acceptance penalty, in `1..1_000_000`. A relevant parent/control width `c` above threshold `d` is allowed with probability `B^(-(c-d))`; `B=1` disables this damping. |
| `crossing.width_penalty_threshold` | integer, `1` | Width `d` below which this penalty does not apply, in `0..1_000_000_000`. A width bias, not a maximum legal gate width (`--k-max 12`). |
| `crossing.size_tolerance_divisor` | integer, `25` | Size-controller temperature `max(64, round(target / divisor))`, in `1..1_000_000_000`. |
| `crossing.move_budget_factor` | number, `3.0` | Move budget `round(factor * target)`, factor in `(0, 1000]`. The walk reaches its target in about `2 * target` moves; the archive-era driver ran `6 * target`; RC set the default to 2 on 2026-10-11 and to 3 on 2026-10-12. |
| `crossing.move_attempts` | integer, unset | An absolute move budget instead of the factor (the two are exclusive), in `1..10^12`. |

Either budget is added to the split stage's move counter on a resume of
`split.state`, because the mixer's resumed `--moves` ceiling is absolute.
The target, penalty and temperature defaults are the archive-era driver's
calibrated values; their promotion status was never resolved (the crossing
walk was retired on 2026-10-07 and revived on 2026-10-11 for the resynthesis
studies: `docs/notes/stage45_revival_20261011.md`).

### Packing (stage 6, the first sub-step)

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `packing.window_gates` | integer, `512` | Window gate cap G (`window_pack3.py` argv 4). |
| `packing.interval_cap` | integer, `512` | Interval cap W_cap (argv 3). |
| `packing.monomial_budget` | integer, `24576` | Soft window monomial budget (argv 5; product cap 8×). At most 262,144 (a memory bound). |
| `packing.span_cap` | integer, `3000` | Longest span considered (argv 6). |
| `packing.split_budget` | integer, `65536` | `repair.sh`'s split budget (and the final budget test). At most 262,144 (a memory bound). |
| `packing.fix_iterations`, `packing.fallback_iterations` | integers, `4`, `4` | `repair.sh`'s loop counts. |
| `packing.anf_cap_semantics` | string, `"cpython39"` | `"cpython39"` reproduces the oracle's CPython 3.9 set-order cap decisions; `"canonical"` is a recorded deviation. |
| `packing.allow_floor_miss`, `packing.allow_over_budget` | bools, `false` | Ship despite a visible non-output row / an over-budget window: the run finishes with `status=floor_miss`/`over_budget`, `packing/NOT_RECOMMENDED`, and no `PIPELINE COMPLETE` line. `packing.report`'s `status` always names every violation found (`floor_miss+over_budget` when both occur, allowed or not); the exit code is the first one not allowed (5, then 6). |

Any non-default value is recorded as `packing_params=custom`.

### Delivery (stage 6, the unpacking sub-step)

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `delivery.anf_only` | bool, `false` | `true` ends stage 6 after the packer's deliverable: no resynthesis, no refinement, no `unpacking/`; `tdp_unpacking` is not needed. The four-stage validation chain. |
| `delivery.time_limit` | number, `30` | Synthesis wall-clock limit per window per round, seconds, in `(0, 86400]`. A window that hits it keeps the best converged restart (`deadline_hits` in the report) or fails the round; the output then depends on timing. |
| `delivery.restarts` | integer, `16` | Synthesis restarts per window, `1..1000` ("less optimized" = fewer, e.g. 2). |
| `delivery.beam` | integer, `6` | Lookahead width of the search, `1..1000`. |
| `delivery.max_rounds` | integer, `16` | Refinement rounds, `1..1000`: a window that fails to resynthesize is cut in two at an exposure-aware cut and both halves are repacked and unpacked in the next round; a 1,024-gate window reaches 1-gate pieces in 10 rounds. A window still failing after the last round (or a failed 1-gate window) is emitted verbatim and counted (`verbatim` in the report, `status=verbatim_used`: reported as NOT recommended, not fatal). |

Any non-default value (including `anf_only`) is recorded as
`delivery_params=custom`. The synthesizer sees only the window ANF; see
[DELIVERY_STAGE.md](DELIVERY_STAGE.md) for the loop and
[STAGE7_UNPACKING.md](STAGE7_UNPACKING.md) for the synthesizer. The
seven-stage driver's `unpacking.allow_fallback` has no counterpart: the
refinement replaces the verbatim fallback.

### Parallel stage 3

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `parallel.pieces` | integer, `1` (serial) | Fixed partitioning for contiguous piecewise mixing, in `1..64`. A partitioned run is a different walk from the serial (validated) one. |
| `parallel.target_piece_gates` | integer, unset | Automatic piece count `max(1, current_gate_count / value)`. |
| `parallel.threads` | integer | Worker pool size in `1..1024`; requires `pieces>1` or `target_piece_gates`. |

### Leakage audit and repair

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `leakage_repair.enabled` | bool, `false` | Sampled leakage audit and function-preserving DB repairs after stage 3 (`db_mixing.leakage_repair.txt`). |
| `leakage_repair.seed` | integer, `20803` | Diagnostic/repair seed; requires `enabled=true`. |
| `leakage_repair.reference` | string, the stage-3 input | Reference circuit in matching coordinates; requires `enabled=true`. |

### Seeds and calibration

Fresh runs draw five seeds uniformly from `[0, 2^63 − 16]` with the OS
CSPRNG, plus a 256-bit `run_key`, into the private `$RUN/SEEDS` (mode 0600),
**before** anything else is written:

```text
seeds_format=3
gen_seed=<decimal>
mix_seed=<decimal>
split_seed=<decimal>
cross_seed=<decimal>
unpack_seed=<decimal>
run_key=<64 hex>
seed_source=random|explicit
```

Gen receives `gen_seed` as all three of its seeds (C, gadgetization and
sandwich), the stage-3 mixer `mix_seed`, the same values val_chain.sh passes;
the split stage `split_seed`; the crossing walk `cross_seed` only when it
starts from a circuit file (a resume of `split.state` continues that run's
RNG); the delivery stage's `tdp_unpacking run --refine` `unpack_seed` (the
fifth seed keeps its name from the seven-stage driver, so format-3 `SEEDS`
files resume unchanged). Unlike val_chain.sh, **no seed is on any
command line**: val_chain.sh puts them on gen's and the mixer's argv, where
`ps` shows them to every local user while the process runs (the gen seed
alone regenerates $C$ and the stage-2 artifact). The driver hands them over
on inherited file descriptors instead (gen `@/dev/fd/3 @/dev/fd/4
@/dev/fd/5`, the mixer and `tdp_unpacking` `--seed-file /dev/fd/3`, each fed
by a bash here-string, so nothing is written to disk), and the binaries
print `(file)` in their banners in place of the value. So no seed value
appears in `ps`, on the Rust-to-bash command line, in `tdp_gen.log`, the
stage logs, the manifest or stdout; the private `SEEDS` (and, for
calibration, your seed file) is the only copy. No unkeyed hash of a seed, of
$C$ or of a private circuit is written; resume markers use HMAC-SHA256 keyed
by `run_key` (the stage-4/5/6 markers also carry a keyed tag of their seed).
(Running gen or the mixer by hand with a numeric seed, or val_chain.sh
itself, still exposes it in `ps`.) The manifest records `seed_record=SEEDS`,
`seed_source`, `seeds_format=3` and `seed_bits=63` (a stopgap: $C$ is
generated from at most 64 bits; use `source.path` for production
deliverables until the 256-bit decision, design §13.1).

A `SEEDS` file of format 2 (gen and mix only, written by the four-stage
driver) is upgraded in place when such a run is resumed: the three new seeds
are drawn fresh and the file is rewritten as format 3 (noted in
`tdp_gen.log`).

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `calibration.enabled` | bool, `false` | Mark the run calibration-only and permit an explicit seed file. |
| `calibration.seed_file` | string, unset | A private file (no group/other permissions, at most 256 bytes) with the lines `gen_seed=<u63>` and `mix_seed=<u63>`, and optionally `split_seed=`, `cross_seed=`, `unpack_seed=` (drawn fresh when absent). A one-integer (format 1) file is rejected with a migration message. On resume every seed it names must agree with `SEEDS`. |

### Environment policy

Every stage binary starts from an **empty environment** plus `PATH`, `HOME`,
`TMPDIR`, `LANG=C`, `LC_ALL=C` (and `RAYON_NUM_THREADS` when `run.threads`
pins it) and the variables the recipe sets. The registry
[src/tdp/env_registry.rs](../src/tdp/env_registry.rs) (mirrored in
[scripts/tdp_env_registry.tsv](../scripts/tdp_env_registry.tsv), kept in sync
by a test) classifies every environment name the crate reads:

| Class | Examples | An inherited value is |
| --- | --- | --- |
| recipe | `EMBEDDED_MASKING_SHUFFLING` … `CHAIN_LEN`, `K`, `MAX_OPEN`, `MIN_OPEN`, `BALANCED` | fatal (the preset sets them) |
| driver | `FROZEN_FILTER`, `FROZEN_REGULAR_VALUE_CONVENTION`, `SANDWICH_VARIANT`, `TDP_SOURCE_C`, `MIXER_STOP_FLAG` | fatal (use the config field / flag) |
| input | `FROZEN_DB_DIR`, `TDP_BIN_DIR`, `TDP_GEN_ALLOW_EMPTY_STORE` | accepted |
| neutral | `LOOKUP_CACHE_MB`, `CANON_CACHE_MB`, `XPOLY_CANON_CACHE_MB` | ignored with a note |
| forbidden / diagnostic | every other `EMBEDDED_MASKING_*`, `SLICE_GUARD_LEGACY`, `MIXER_DB_LEN_BAND`, `MIXER_STOP_AT_*`, `TWIST_G57_*`, `MIN_DIR_LOOKUP`, `CANON_*_CAP`, `PROD_*`, `SAT_*`, `DB_QC*`, `BV5_*`, `FROZEN_CURATED_*` | fatal: `X is not honoured by the managed recipe; unset it (use flags/config)` |

A test scans `src/` for environment reads and fails on an unclassified name,
so a new knob forces a decision. The stage-2 construction knobs are typed
`EmbeddedMaskingParams` fields; only gen's adapter maps their historical
environment names (design §11.4).

### Direct driver flags

The managed command reads configuration; the Bash driver takes flags (its
`--help` lists them all):

| Recipe setting | `scripts/tdp_gen.sh` flag |
| --- | --- |
| `run.directory`, `run.skip_stages`, `run.stop_after_stage`, `run.rerun_from_stage`, `run.threads` | `--run-directory DIR`, `--run-skip-stages 4,5`, `--run-stop-after-stage K`, `--run-rerun-from-stage K`, `--threads T` |
| `source.wires`, `source.gates`, `source.gates_rule`, `source.path` | `--source-wires N`, `--source-gates M` (`-m`), `--source-gates-rule formula`, `--source-path P` |
| `preprocessing.mode`, `preprocessing.preset` | `--preprocessing-mode M`, `--preprocessing-preset L8\|A\|custom` |
| custom knobs | `--preprocessing-{mask-pair-wires,max-open-masks,min-open-masks,balanced-masks,shuffling-segments,shuffling-return-home,premask,seed-avoid-c1,chain,chain-len} V` |
| `db_mixing.*` | `--db-mixing-schedule custom`, `--db-mixing-growth-work-units`, `--db-mixing-hold-work-units`, `--db-mixing-target-size-factor` |
| `database.lookup_miss_filter` | `--database-lookup-miss-filter auto\|on\|off` |
| `crossing.*` | `--crossing-target-size-factor R`, `--crossing-width-penalty-base B`, `--crossing-width-penalty-threshold C`, `--crossing-size-tolerance-divisor D`, `--crossing-move-budget-factor F`, `--crossing-move-attempts M` |
| `packing.*` | `--packing-window-gates`, `--packing-interval-cap`, `--packing-monomial-budget`, `--packing-span-cap`, `--packing-split-budget`, `--packing-fix-iterations`, `--packing-fallback-iterations`, `--packing-anf-cap-semantics`, `--packing-allow-floor-miss`, `--packing-allow-over-budget` |
| `delivery.*` | `--delivery-anf-only`, `--delivery-time-limit S`, `--delivery-restarts R`, `--delivery-beam B`, `--delivery-max-rounds N` (the seven-stage driver's `--unpacking-*` flags are refused with a pointer) |
| `leakage_repair.*` | `--leakage-repair`, `--leakage-repair-seed S`, `--leakage-repair-reference P` |
| `calibration.seed_file` | `--seed-file PATH` (seeds on argv, `-s`, are refused) |
| `parallel.*` | `--parallel-pieces P`, `--parallel-target-piece-gates B`, `--parallel-threads T` |

Standalone, the script draws the seeds itself (`python3 -I` + `secrets`)
into `$RUN/SEEDS` under the same format and rules. The managed command never
adopts such a run directory (SEEDS without its manifest): continue it with
the script, or start a fresh managed run. It reads only
`FROZEN_DB_DIR`, `TDP_BIN_DIR` and `TDP_GEN_ALLOW_EMPTY_STORE` from its
environment. It needs **bash 5.1 or newer** and refuses an older one
(exit 2) before doing anything: associative arrays and `mapfile` need 4.0,
and 5.1 is the first bash that feeds a small here-string through a pipe
instead of a temporary file under `TMPDIR`, which keeps the seed hand-over
off disk. macOS's `/bin/bash` is 3.2 (`brew install bash`). The managed
command finds a suitable bash itself, before building or writing `SEEDS`:
the first `bash` on `PATH` that is 5.1 or newer, then
`/opt/homebrew/bin/bash`, `/usr/local/bin/bash`, `/bin/bash`,
`/usr/bin/bash`; with none it stops naming every bash it found.

## Stage 1: the sliced sandwich

We put $C$ inside a circuit on $2n$ wires. In execution order, the classic
sandwich is

$$
[C\text{ interleaved with }S_1]\ ;\ N\ ;\
[D\text{ interleaved with }S_2].
$$

$D$ is a separately sampled circuit, $N$ copies the first half into the second
with CNOTs, and the slice blocks $S_1,S_2$ use the second half as controls.
On inputs $(x,0^n)$, the second output half is $C(x)$; the first half may
contain junk. We also move the copy CNOTs through gates they commute with so
they do not remain in one obvious column.

Read [src/stages/sandwich/construct.rs](../src/stages/sandwich/construct.rs),
starting with `prepare_source`, `construct_seeded_sandwich`, and
`sliced_sandwich_cnot`.

## Stage 2: preprocessing (embedded masking)

We now hide each sandwich value behind a nonlinear mask. The default is
embedded masking: one carrier and one band wire per sandwich value, taking
us from $2n$ to $4n$ physical wires. The construction surrounds the masked
computation with slice guards and input-derived band seed/reseed blocks.

At the end, on input $(x,0^n,0^{2n})$, wires $n,\ldots,2n-1$ contain $C(x)$.
The remaining outputs are junk. The full circuit is reversible, but this is
the public slice on which we ask it to reproduce the source computation.

Start at `preprocess_sandwich` in
[src/stages/preprocessing/construct.rs](../src/stages/preprocessing/construct.rs).
The masking, band seed blocks and shuffles are in
[embedded_masking.rs](../src/stages/preprocessing/embedded_masking.rs);
the slice guards are in
[slice_guards.rs](../src/stages/preprocessing/slice_guards.rs).
[verify.rs](../src/stages/preprocessing/verify.rs) checks the promised outputs.
The optional `nonlinear291` mode has its own adapter in the same directory.
See [gadgetization](gadgetization.md) for the mask and shuffle construction.

The generator's command handling is in
[src/programs/gen_sandwich_gadget/mod.rs](../src/programs/gen_sandwich_gadget/mod.rs).
Stages 1–2 write `tdp.mpmct1`, its source/sandwich sidecars, and `stage12.log`.

## Stage 3: database mixing

We sample small circuit windows and replace them with different spellings of
the same function. To find these spellings, we compose the window's
polynomials, canonicalize its wire labels, and query the regular frozen
store. A candidate must be mapped back to the window and checked before
insertion. Stage 3 runs `circuit_mixer --tdp --db-mixing --p-db 1.0`: every
round that is not a g57 twist is a DB move, and a lookup miss makes no
replacement.

The validated schedule grows toward twice the incoming size with profile
`2,2,2,2,2` and a ceiling of `int(G·2·2·1.3)` moves (G = the stage-2 gate
count), exactly val_chain.sh's command:

```text
circuit_mixer --input tdp.mpmct1 --output db_mixing.mpmct1 --tdp --db-mixing --profile 2,2,2,2,2
    --moves int(G*2*2*1.3) --p-db 1.0 --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024
    --db-total-terms 2048 --no-local-verify --seed-file /dev/fd/3   # mix_seed; val_chain.sh: --seed <mix_seed>
```

`--verify-every` and `--report-every` stay at their defaults: the global
check draws from the mixer RNG, so its cadence is part of the trajectory.
Optional leakage repair runs here too. The output is `db_mixing.mpmct1` with
`stage3.log`.

Start with [src/programs/circuit_mixer/mod.rs](../src/programs/circuit_mixer/mod.rs) for how the
mode is selected. Follow `Mixer::run` in
[src/engine/mixer/scheduling.rs](../src/engine/mixer/scheduling.rs).
[src/engine/mixer/replacement.rs](../src/engine/mixer/replacement.rs) samples,
verifies and splices the replacement; it calls
[src/stages/db_mixing/replacement.rs](../src/stages/db_mixing/replacement.rs)
for lookup and candidate selection.
[src/canonicalization/](../src/canonicalization/) builds the keys;
[src/database/](../src/database/) reads and decodes the stored values. The
[frozen DB](frozen_database.md) and
[canonicalization](polynomial_canonicalization.md) guides explain these parts.

<a id="stage-4-packing"></a>

## Stage 4: splitting

The database leaves many gates in the r57 shape `a ⊕= b ∨ ¬c`. The split
stage rewrites each complemented firing condition into two disjoint plain
conjunctions, `{a ⊕= b, a ⊕= ¬b ∧ ¬c}`, and with probability 0.8 wraps an
absorbed pure-NOT twist on the gate's target wire between the one-control
piece and a bracket found across the circuit, finishing each move with one
crossing shot. It stops when the complemented-gate pool is exhausted or the
bracket search hits its failure limit (`--split-stop`: the stage boundary ends
the run). The driver runs

```bash
circuit_mixer --input db_mixing.mpmct1 --split --split-stop --p-join 0.8 --split-reach-k 2 \
    --split-fail-limit 100 --split-canaries 256 --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 \
    --k-max 12 --moves G+1000000 --seed-file /dev/fd/3 --report-every 1000000 \
    --state-out split.state --output split.mpmct1
```

and writes `split.mpmct1`, `split.state` (the resume point of stage 5),
`stage4.log` and the completion marker `stage4.recipe` (the arguments, the
binary's hash, keyed tags of the input, the output and the split seed, the
stop reason). A split ended by an operator stop flag (`$RUN/stage4.stop`) is
kept as `split.stopped.mpmct1`, never handed on, and re-run on resume. At
n=64 (L8 arm) the stage takes 1.60M gates to 3.03M in about 70 s; at n=128
4.59M to 8.66M in 8.5 min. Read `split_twist_move` in
[src/stages/post-processing/splitting.rs](../src/stages/post-processing/splitting.rs);
the local rewrite identities live in
[src/engine/moves/rules.rs](../src/engine/moves/rules.rs).

## Stage 5: the crossing walk

The walk shoots fragments to their first true collider and rewrites each
collision by one of three exact case-split rules so the pieces cross; these
moves introduce wider conjunctions than r57 gates (median 3 controls, up to
12). A thermostat holds the size near `target = round(input_gates ×
factor)` with two contraction channels: exact undo of isolated crossings
from a journal, and a pairwise merge catalogue over same-target gates. The
driver resumes the split state (the walk continues the split stage's RNG and
move counter; `--moves` is absolute on a resume):

```bash
circuit_mixer --resume split.state --target-size TARGET --temp max(64,round(TARGET/25)) \
    --split-base 3 --split-damp 1 --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
    --moves state_moves+3*TARGET --report-every 500000 --output crossing.mpmct1
```

With stage 4 skipped it starts from the previous circuit (`--input …
--seed-file /dev/fd/3`, the `cross_seed`) with `--moves 3*TARGET`. It writes
`crossing.mpmct1`, `stage5.log` and `stage5.recipe` (which records the
source: `resume split.state` or `input <file>`). No state file is written:
nothing downstream resumes it. At n=64 the walk takes 3.03M gates to 5.49M
in 46 min; at n=128 8.66M to 15.7M in 6.3 h; it slows several-fold under CPU
contention. Read `cross_move_on`, `undo_move` and `merge_move` in
[src/stages/post-processing/crossing.rs](../src/stages/post-processing/crossing.rs);
the scheduler and acceptance logic are in
[src/engine/mixer/scheduling.rs](../src/engine/mixer/scheduling.rs). Within
the mixer the walk is slot 3 of `Mixer::run`, the default walk when `p_db <
1`; stage 3 passes `--p-db 1.0`, so slot 3 never runs there and stage-3
output is unchanged by its presence.

## Stage 6: delivery

Delivery turns the mixed circuit into the deliverable and shows that the
deliverable unpacks. It is one stage of two sub-steps (RC, 2026-10-12;
[DELIVERY_STAGE.md](DELIVERY_STAGE.md)): the packer emits the circuit as ANF
windows on a lattice of cuts, and the unpacking sub-step resynthesizes every
window from its ANF alone, treating a window that does not resynthesize as a
packing defect: the window is cut in two, both halves are repacked and
unpacked, recursively, until every window unpacks (the size of a
resynthesized window is irrelevant). The stage's shipped file is the
deliverable on the refined lattice, `unpacking/deliverable.tdp4`; its
resynthesized circuit is `unpacking/unpacked.mpmct1`. With
`delivery.anf_only` the stage ends after the packer's
`packing/deliverable.tdp4`.

### Packing (the first sub-step)

The adversary is assumed to see the full wire state at every window
boundary and the inputs, but not the gates inside a window. The packer
reorders its input (the last circuit produced: `crossing.mpmct1` by default,
`db_mixing.mpmct1` under `run.skip_stages = [4, 5]`) along commutations so
that values are born and consumed inside one window, cuts the order into
windows of at most G gates, and publishes each window as its sparse
algebraic normal form (ANF). A commutation-aware repair then removes the
remaining boundary exposures of $C$'s values, and the result is serialized.
The design and the measurements are in
[ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md) and
[STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md); the port in
[STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md).

The driver runs

```bash
tdp_packing run --stage2 tdp.mpmct1 --stage3 crossing.mpmct1 --source-c tdp.mpmct1.source_c.g57 \
    --n N --out-dir packing [packing flags only when not at their defaults]
```

(`--stage2` is the circuit whose values the windows hide: `tdp.mpmct1`, or
the sandwich when stage 2 is skipped; `--stage3` is the previous stage's
output.) On the stage-5 output the packer needs about 4× the windows and far
longer (n=64: 2.2 h instead of 23 s; n=128: 7.7 h instead of 2 min;
`docs/notes/stage45_revival_20261011.md`).

which does, in one process, what val_chain.sh does after stage 3:

| Step | Oracle | Rust | Writes (in `packing/`) |
| --- | --- | --- | --- |
| Pack | `window_pack3.py A B 512 512 24576 3000` | `stages::packing::{fingerprint,intervals,scheduler}` | `packed_G512.mpmct1(.bounds)`, `pack_G512.txt` |
| Repair | `waist/repair.sh` (cancel, C-view fix ×4, merge/split fallback ×4, budget 65536) | `stages::packing::repair`, C-view in `stages::packing::cview` | `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `rep512.cview.log`, `repair512.log` |
| Reports | `anf_compose.py`, `link_census.py` | `stages::packing::census` | `anf_rep512.txt`, `link_rep512.txt` |
| Serialize | `waist/anf_dump_eval.py` | `stages::packing::serialize` | `deliverable.bin` (raw body), `deliverable.tdp4` (container v1) |

Every check runs on every run: an exact certificate that the packed order is
a linear extension of G3, the oracle's VERIFY, an exact cancellation replay
and repair certificate, both sampled equivalence checks, decode = window
ANF, window ANF = gates (exhaustive up to 16 support wires, sampled above),
`anf_dump_eval.py`'s evaluation check, and the zero slice of the deliverable
against $C$ (4096 inputs). Outputs are written to `*.tmp` and renamed; the
deliverable is written after every check, and `packing.report` (the
completion record) after the deliverable. Run standalone into a reused
`--out-dir`, `tdp_packing run` first removes the earlier run's outputs, so a
failed run leaves no deliverable or report from another run. Parameters are
bounded as in `[packing]` (window gates ≤ 10^6, span cap ≤ 10^9, iterations
≤ 1000; exit 2 otherwise). The monomial and split budgets are resource
bounds, at most 2^18 = 262,144: a window may hold 4× and a product 8× the
budget in monomials, and budgets of 10^6 and above exhausted host memory on
the n=8 fixture within minutes (2^18 is 5× the largest measured packer
budget and 4× the recommended split budget). The deliverable is emitted at
the same ceiling; a window above it is exit 6 (`anf_dump_eval.py` emits at
2^30; every window the repair leaves fitting emits identically).

**`packing/deliverable.tdp4`** is the packer's deliverable (the shipped file
under `anf_only`; otherwise the refinement's starting point): a 48-byte
header (`LMTDPANF`, version, source and state wires, layout, window count,
body length, CRC-32 of the body and of the header) followed by the unchanged
`anf_dump_eval.py` body, which is also kept as `deliverable.bin`.
`tdp_packing decode --format container` validates it; the refined
`unpacking/deliverable.tdp4` has the same format.

**Exit codes** (also printed by the driver with their meaning):

| Code | Meaning | Deliverable |
| --- | --- | --- |
| 0 | finished; `packing.report` `status=ok`, or `floor_miss`/`over_budget` under `--allow-*` | written |
| 2 | usage or parameter error | — |
| 3 | input error (unreadable, malformed, inconsistent) | — |
| 4 | verification failure (a certificate or check) | — |
| 5 | exposure floor missed: the final C-view has a visible non-output row | not written (intermediates kept) |
| 6 | over budget: a final window fails the split's budget test | not written (intermediates kept) |
| 7 | I/O error writing outputs | — |
| 8 | internal invariant violated | — |

Exits 5 and 6 are deliberate departures from the oracle on failure paths
only: the oracle would ship such a circuit silently.

**`packing.report`** (flat `key=value`, publishable: counts and parameters
only, no seeds and no hash of a private input) records the status, every
parameter, the pack/repair/emit counters, the linkability census (L2/M2) and
the timings.

**Speed** (Apple-silicon Mac, other jobs running): `tdp_packing run` on the
archived validation runs takes 20-45 s at n=64 (0.9-1.1 GB peak) and 75-140 s
at n=128 (1.9-2.1 GB), against about 5-6 min and 15-17 min for the Python
chain.

### Unpacking with lattice refinement (the second sub-step)

Unpacking is what a recipient of the deliverable does to obtain a circuit
again, and the refinement makes sure that it succeeds for every window. The
driver runs, right after the packer (skipped under `delivery.anf_only`),

```bash
tdp_unpacking run --deliverable packing/deliverable.tdp4 --packed packing/rep512.mpmct1 \
    --bounds packing/rep512.bounds --source-c tdp.mpmct1.source_c.g57 --n N \
    --out-dir unpacking --seed-file /dev/fd/3 --refine --max-rounds 16 \
    [--time-limit S --restarts R --beam B] [--threads T]
```

(`--refine --max-rounds` are always passed; the synthesizer knobs only when
not at their defaults.) Round 0 unpacks every window of the packer's
lattice from the deliverable alone (the synthesizer, a port of `anf_synth`,
never sees the packed gates; they define the windows and serve the exact
check). Each window ends converged (its gates verified exactly against the
delivered ANF) or failed (timeout, stall, step cap, verification). Every
failed window of two or more gates is cut in two at an exposure-aware cut
(32 candidates, one C-view scan, the one nearest the midpoint at which no
interior value of $C$ is affine; the midpoint when none is clean, counted
as an unclean cut), the deliverable is re-emitted with the two halves'
blobs in its place, and the next round unpacks the new halves; the loop
ends when nothing is pending or after `max_rounds`. The converged windows
are concatenated in lattice order and the whole circuit is put into the
canonical commuting gate order (`xgate_canonicalize`, so no window boundary
survives). Measured basis and the full contract: [DELIVERY_STAGE.md](DELIVERY_STAGE.md)
(on the n=64 seven-stage run 157 of 10,366 windows failed to unpack in 30 s;
bisection converges by about 90 % per round and ends at 15-16-gate pieces);
the synthesizer, its checks and its exit codes: [STAGE7_UNPACKING.md](STAGE7_UNPACKING.md).

It writes into `unpacking/`: `deliverable.tdp4` (the refined lattice: **the
shipped file**), `bounds.final` (its cuts), `unpacked.mpmct1` (the state
width in the header, canonical order), `windows.tsv` (one row per window of
the final lattice with its round and status), `rounds.tsv` (per round:
pending, converged, failed, split, unclean cuts, seconds), `unpacking.report`
(`status=ok|verbatim_used`, `rounds`, `windows_initial`, `windows_final`,
`splits`, `unclean_cuts`, `verbatim`, the gate counts and ratio, every
check) and `unpacking.log`; the driver appends its output to `stage6.log`
after a separator line. Before anything is committed every converged window
is checked exactly and by sampling, the refined deliverable passes the
packer's final check at every cut and evaluates equal to the packed circuit,
the circuit is checked against the deliverable before and after reordering,
and the zero slice is checked against $C$ (any failure is exit 4, nothing is
renamed). A window that still fails after the last round, or a failed
1-gate window, is emitted verbatim and counted: `status=verbatim_used`,
which the driver reports as an allowed failure (`NOT_RECOMMENDED`, no
`PIPELINE COMPLETE`), not as an error; expected 0.

**Exit codes** of `tdp_unpacking` (the driver prints them with their
meaning and `step=unpacking` in `delivery.failed`): 2 usage or parameter
error, 3 input error, 4 verification or synthesis failure, 7 I/O error
writing outputs, 8 internal invariant violated, 101 panic.

### Resume, markers and failures

After both sub-steps the driver writes `delivery.recipe`: the two binaries'
hashes, the packing parameters, `anf_only`, the unpacking parameters
(`--refine --max-rounds N …`) and the keyed tag of the delivery seed (both
omitted under `anf_only`), HMAC-SHA256 tags (keyed by `run_key`) of the three
inputs, of `packing/deliverable.tdp4` and, without `anf_only`, of
`unpacking/deliverable.tdp4` and `unpacking/unpacked.mpmct1`, then
`status=` (the sub-step statuses other than `ok`, joined by `+`),
`packing_status=` and `unpacking_status=` (`skipped` under `anf_only`). A
resume skips stage 6 only when the whole marker matches; rerunning stage 2-5,
changing any packing or delivery parameter (including `anf_only`), a binary
or an output reruns the whole stage: `packing/` and `unpacking/` are removed
and both sub-steps run again. A failed sub-step leaves `delivery.failed`
(`exit=<code>`, `step=packing|unpacking`) and no marker; the managed driver
attaches that binary's exit meaning only when the marker names the code. An
allowed failure (`floor_miss`/`over_budget` from the packer under
`--allow-*`, `verbatim_used` from the refinement) finishes the stage with
`NOT_RECOMMENDED` (the status lines) next to the final output
(`unpacking/`, or `packing/` under `anf_only`) and `PIPELINE FINISHED WITH
AN ALLOWED FAILURE` instead of `PIPELINE COMPLETE`.

### The oracle

The Python/bash prototypes in
[scripts/windowed_delivery/](../scripts/windowed_delivery/) are the canonical
oracle and are never edited. Every comparison runs them in a path-rewritten
sandbox (`tests/fixtures/stage4/oracle_sandbox.sh`), under an interpreter that
passes the CPython 3.9 set-order self-test
(`tests/fixtures/stage4/pyset_selftest.py`; `/usr/bin/python3` 3.9.6 on
macOS), with the `hmap_affine` baseline built from `492564f0`
(`tests/fixtures/stage4/README.md`). The comparisons:

- `tests/integration/tdp_packing.rs`: oracle fixtures (pack, repair, serialize,
  the whole `run` at n=8) byte for byte, exit codes, thread independence;
- the CI job `stage4-oracle` (`tests/stage4/differential.sh`): the oracle and
  `tdp_packing` on 20 random pack and repair/serialize cases per push;
- `tests/manual/stage4/{run,pack,repair,anf,cview}_parity.sh`: the 8
  archived validation runs, every artifact identical;
- `tests/manual/stage4/reproduce_val_chain.sh N [L8|A] [MC]`: the managed
  driver against a sandboxed val_chain.sh with fresh seeds;
  `tests/shell/tdp_reproduce_small.bash` runs it at n=8 when a store is
  available;
- `tests/manual/stage4/anchor.sh`: the archived runs themselves from their
  archived seeds (gen and mixer at `492564f0`, then the managed driver).

### The reference driver

`waist/val_chain.sh TAG N MC S SLICE CONFIG` runs stages 1-4 with the Python
stage 4 and seeds below $2^{31}$/$2^{40}$ in `W/<TAG>/seed.txt` (calibration
grade). It hard-codes the run tree (`/Users/rancanetti/code/local_mixing`,
`runs/ncdb_ab_20261001/waist/<TAG>`), so outside that tree run it through the
reproduction test's sandbox. Its CONFIG is `L8` (recommended) or `A`; the
managed presets of the same names set exactly its environment.

## Historic stages

The former stage 6 (`fcompress`: gather commutable same-target gates, reduce
each group's ESOP, pack runs into generalized gates) is historic, as is the
curated store. Their code and tests remain on branch
`ran/archive/curated-and-postmix`; runs that record recipe v8 need that
branch to continue. The split stage and the crossing walk, retired with them
on 2026-10-07, were revived from that branch on 2026-10-11 as stages 4 and
5 (mixer state format v6 = the pre-cleanup v3 layout; v3-v5 states of the
four-stage interval are refused). See
[CURATED_AND_POSTMIX_ARCHIVE.md](CURATED_AND_POSTMIX_ARCHIVE.md).

## The shared pieces

We do not need to understand every file to change one stage. The circuit
representation is in [src/circuit/xgate.rs](../src/circuit/xgate.rs), and its
readers/writers are in [src/circuit/formats.rs](../src/circuit/formats.rs).
The mutable gate tape used by `circuit_mixer` is in
[src/engine/arena.rs](../src/engine/arena.rs). Mixer state, sampling, checkpoint
I/O, and reporting live together in [src/engine/mixer/](../src/engine/mixer/).

Use these current module paths directly. The old top-level `preprocessing`,
`db_mixing` and `postprocessing` wrappers have been removed. Files named
`environment.rs` read settings of the stage-3 engine and the store; the
managed driver sets or forbids each of them (see the environment policy).
Stages 4 and 5 live in
[src/stages/post-processing/](../src/stages/post-processing/) as mixer modes
(their slots, indices, journal and counters in `src/engine/mixer/`). Stage 6's
packer lives in [src/stages/packing/](../src/stages/packing/) (no environment
reads) with its binary in [src/programs/tdp_packing/](../src/programs/tdp_packing/);
its unpacking sub-step in [src/stages/unpacking/](../src/stages/unpacking/)
with [src/programs/tdp_unpacking/](../src/programs/tdp_unpacking/).

Stage 3 can run on contiguous pieces in parallel; the coordinator is
[piecewise.rs](../src/engine/mixer/piecewise.rs). Correctness checks live in
[tests/](../tests/), while attacks and leakage measurements live in
[security_tests/](../security_tests/). The detailed
[current-method documentation](local_mixing_documentation.md) explains the
implemented construction and attacks. The [history](local_mixing_history.md)
preserves the motivation and earlier experiments behind this order.
