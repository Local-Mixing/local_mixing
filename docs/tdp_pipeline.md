# The TDP Pipeline

This is the operational manual for the four-stage TDP pipeline: how to run
each stage, what it writes, and where its code lives. We start with a
reversible circuit $C$ on $n$ wires and end with a published circuit that
computes $C$ on a public slice.

| Stage | What it does | Runs as | Output |
| --- | --- | --- | --- |
| 1. Sandwich | Classic sliced sandwich around $C$ on $2n$ wires | `gen_sandwich_gadget` (with stage 2) | `tdp.mpmct1.sandwich.mpmct1`, `tdp.mpmct1.source_c.g57` |
| 2. Preprocessing | Embedded masking on $4n$ wires: masked fires, band, guards, optional carrier migration | `gen_sandwich_gadget` | `tdp.mpmct1` |
| 3. Mixing | Database mixing with the regular frozen store | `circuit_mixer --tdp --db-mixing --p-db 1.0` | `db_mixing.mpmct1` |
| 4. Packing | Windowed ANF delivery: commuting reorder into windows, repair, serialization | Python prototypes in [scripts/windowed_delivery/](../scripts/windowed_delivery/) | `deliverable.bin` |

```text
C → sliced sandwich → embedded masking → DB mixing → db_mixing.mpmct1 → packing → deliverable.bin
       stage 1            stage 2          stage 3     (tdp_gen ends)      stage 4
```

The managed `tdp_gen` command drives stages 1-3 and ends at
`db_mixing.mpmct1`; it says so when it finishes. Stage 4 is not yet in Rust:
run it with the prototypes as described in [Stage 4: packing](#stage-4-packing).
A Rust port is planned.

The curated store and the former stages 4 (split), 5 (crossing walk) and
6 (`fcompress`) are historic. They remain on branch
`ran/archive/curated-and-postmix`; see
[Curated DB and post-mixing archive](CURATED_AND_POSTMIX_ARCHIVE.md) for what
was retired, how to retrieve it, and why.

Further reading:

- [TDP_FOUR_STAGE_PIPELINE.md](TDP_FOUR_STAGE_PIPELINE.md): the full
  reference for the four stages (construction details, parameters, measured
  behaviour, pass criteria, open issues).
- [STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md): the stage-2 redesign and
  the linkable-exposure metrics that the recommended configuration is chosen
  by.
- [ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md): windowed ANF packing, its
  threat model, and why it replaced the old stages 4-6.
- [windowed_delivery_design.md](windowed_delivery_design.md): the design of
  stage 4 and the plan for the Rust port.

For first-time setup and the circuit utilities, start with the
[README](../README.md).

## Two ways to run

**The managed driver** (`tdp_gen`, stages 1-3) is the supported entry point.
It reads a TOML recipe, builds and fingerprints the stage binaries, records a
manifest (recipe v9), and resumes runs safely. It runs the production
defaults: embedded masking without carrier migration, and the stage-3 profile
`3,30,30,2,2`.

**The research reference driver**
[scripts/windowed_delivery/waist/val_chain.sh](../scripts/windowed_delivery/waist/val_chain.sh)
runs all four stages with the configuration recommended by the stage-2
linkability program: persistent carrier migration
(`EMBEDDED_MASKING_SHUFFLING=8`, `RETURN_HOME=0`), `PREMASK`,
`SEED_AVOID_C1`, optionally bounded chains (config `L8`), the stage-3
profile `2,2,2,2,2`, then packing at G=512, repair and serialization. The
managed driver cannot express this configuration yet: it refuses persistent
migration, does not pass the `PREMASK`/`SEED_AVOID_C1`/`CHAIN*` controls, and
fixes the profile shape. Closing that gap is part of the Rust port of stage 4.
The gap analysis is in [TDP_FOUR_STAGE_PIPELINE.md](TDP_FOUR_STAGE_PIPELINE.md)
(Appendix A).

## Quick start (stages 1-3)

```bash
cargo build --release --locked
cp configs/tdp.toml configs/local.toml        # set database.regular_dir
cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

A run directory holds:

| Stage | Files |
| --- | --- |
| 1. Sliced sandwich | `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` |
| 2. Preprocessing | `tdp.mpmct1`, `stage12.log`, `stage12.recipe` |
| 3. Database mixing | `db_mixing.mpmct1`, `db_mixing.state`, `stage3.log` (and `db_mixing.leakage_repair.txt` with leakage repair) |
| Run | `tdp_command.conf` (recipe manifest), `tdp_gen.log`, the private `SEED` |

Then pack `db_mixing.mpmct1` as described in
[Stage 4: packing](#stage-4-packing).

## Where a run starts

The `tdp_gen` command enters through [src/main.rs](../src/main.rs) and
[src/commands/tdp_gen.rs](../src/commands/tdp_gen.rs). From there,
[src/tdp/config/](../src/tdp/config/) reads and checks the recipe, and
[src/tdp/runner.rs](../src/tdp/runner.rs) prepares the run, checks any saved
manifest, and selects the stage binaries.

Then [scripts/tdp_gen.sh](../scripts/tdp_gen.sh) runs the stages.
This is the easiest file to read when we want to see the exact stage order,
arguments, or output names. Stages 1 and 2 share one generator process,
`gen_sandwich_gadget`; `circuit_mixer` runs stage 3. Each executable's
`main.rs` and command implementation live together under `src/programs/`.

## Configuration reference

Use [configs/tdp.toml](../configs/tdp.toml) as both the default recipe and the
annotated template. There is one checked-in example. Copy it to
`configs/local.toml`, put your database paths and settings there, then run:

```bash
cp configs/tdp.toml configs/local.toml
cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

The managed command has these flags:

| Flag | Effect |
| --- | --- |
| `--config PATH` | Read a TOML recipe; default `configs/tdp.toml`. Relative paths resolve from the repository root. |
| `--dry-run` | Parse, resolve and validate the recipe, paths and resume fingerprints; print the planned stages without building or creating a run. A fresh default build need not already exist; disabling builds or resuming requires the recorded binaries. |
| `-h`, `--help` | Show command help. `cargo run --release --locked -- tdp_gen --help` prints it. |

There are **29 recognized section settings plus `config_version`**. The
following tables cover all of them. “Default” means the behavior when the
field is omitted, including defaults computed by the stage driver. In TOML,
`number` accepts an integer or a finite float, `integer` requires an integer,
booleans are unquoted `true`/`false`, and paths/enums are quoted strings.
Empty strings are treated as omitted. Unknown keys, wrong types, duplicate
settings (including an alias plus its canonical spelling), and invalid
combinations are rejected before starting a run. Paths in the recipe resolve
from the repository root, not from the config file's directory.

### Schema and run management

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `config_version` | integer, required `1` | TOML schema identifier. This is separate from saved execution recipe v9. |
| `run.directory` | string, generated directory under `runs/` | Output directory and continuation target. Omit for a fresh timestamped run; select the same directory to continue it. Within the repository it must be a dedicated subdirectory of `runs/`; dedicated external directories are also allowed. It must not overlap build or database directories. |
| `run.build_binaries` | bool, `true` | Build the stage executables (`gen_sandwich_gadget`, `circuit_mixer`) for a fresh run. `false` requires them to exist under the resolved build directory. Managed continuation always reuses matching binaries and skips rebuilding. |
| `run.build_directory` | string, `CARGO_TARGET_DIR` or `target` | Cargo artifact directory. Stage binaries are selected from its host-target `release/` directory; changing this can affect whether the saved binary fingerprints can be matched. |
| `run.stop_after_stage` | integer, `3` | Last stage to reach: `2` or `3`. Stages 1 and 2 run together; `2` is the shortest run. The historic old stages 4-6 (split, crossing walk, fcompress) are rejected. |
| `run.rerun_from_stage` | integer, unset | Recompute this stage and later stages reached by this invocation, `2` or `3`, even if their outputs exist. It does not authorize changing a saved recipe or make earlier stages run again. |
| `run.adopt_unverified_run` | bool, `false` | Retired compatibility field. `true` is rejected: existing runs need a valid current managed manifest. Leave this omitted or false. |

Run lifecycle options such as the stop stage and worker count do not change
the circuit recipe. Algorithm choices, source identity, database paths and
piece partitioning are recorded in `tdp_command.conf`; changing a recorded
choice requires a fresh run. The manifest starts with
`tdp_command_recipe=9`; `stage12.recipe` uses `tdp_stage12_recipe=6`.
Earlier pipeline artifact names are explicitly rejected rather than adopted.

### Source circuit

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `source.wires` | integer, `128` | Logical width of the source computation $C$, in `3..4095`. The sandwich uses twice this width and default embedded masking uses four times this width. The selected preprocessing layout must fit its physical-wire budget. |
| `source.gates` | integer, `round(n * log2(n)^2)` | Generated size of $C$ and $D$, in `1..1_000_000_000`. At `n=128` the default is `6272`. If `source.path` is set, the gate count is inferred; an explicit count must match the file. |
| `source.path` | string, unset | Supply $C$ as a nonempty G57 circuit instead of sampling it. All touched wires must fit `source.wires`. General `mpmct1`, `esop1` and `anf1` inputs are supported by the separate circuit utilities, not this source field. $D$ is still generated separately. |

### Frozen replacement stores

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `database.regular_dir` | string, `FROZEN_DB_DIR` or unset | Regular frozen replacement store, required when stage 3 runs normally. A TOML path overrides the environment path. Needs nonempty `tables.bin` and all 256 shard files. |
| `database.lookup_miss_filter` | string, `"auto"` | `"auto"` enables the in-memory miss filter at at least 60 GiB available RAM; `"on"` requires a nonempty `filters.bin` in the store; `"off"` avoids loading filters. Filters must belong to their exact store. |
| `database.allow_no_database_for_tests` | bool, `false` | Allow a stage-3 plumbing run with all replacement channels disabled. To set true, clear the DB path and its environment fallback. A regular store combined with this mode is rejected when stage 3 runs. |

See [Frozen Database](frozen_database.md#constructing-and-freezing-a-store)
for construction, freezing and validation commands. The curated store of
earlier recipes is historic (branch `ran/archive/curated-and-postmix`); its
TOML keys `database.curated_dir` and `database.curated_control_order` are no
longer accepted.

### Preprocessing

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `preprocessing.mode` | string, `"embedded-masking"` | Choose `"embedded-masking"` or `"nonlinear291"`. The first computes while nonlinear masks remain embedded in carrier values; the second uses encoded shares and fixed operation templates. |
| `preprocessing.mask_pair_wires` | integer, `2` | Band wires in the paired mask component, in `2..64`; odd values are rounded down to an even number for the mask layout. The optional balancing wire is additional. Embedded masking only. |
| `preprocessing.max_open_masks` | integer, `3` | Ordinary scheduling cap for masks open on one data carrier, in `2..64`. Temporary cover masks and refresh replacements can exceed it; it is not a global hard cap on every intermediate state. Embedded masking only. |
| `preprocessing.min_open_masks` | integer, `2` | Minimum open-mask coverage during covered interior intervals, in `1..63`, strictly less than `max_open_masks`. Public input/output fringes remain unmasked. Embedded masking only. |
| `preprocessing.balanced_masks` | bool, `true` | Add a separate band-variable contribution to balance the mask's Boolean function. This does not assert that all input-derived band values are independent. Embedded masking only. |
| `preprocessing.shuffling_segments` | integer, `0` | Opt-in internal preprocessing shuffling: `0` disables it; enabled values must be at least `8`. Transfers spread the active role across physical wires within each fire block. Embedded masking only. |
| `preprocessing.shuffling_return_home` | bool, `true` | Return each data role to its physical home at the end of its fire block. Must stay `true` when shuffling is enabled in TDP, whose output ports have fixed physical positions. |

The embedded mask layout requires
`2 * source.wires > 2 * floor(mask_pair_wires / 2) + int(balanced_masks)`.
For `nonlinear291`, remove all explicit mask and shuffling controls, even if they
equal their usual defaults. Its share layout has a different resource plan;
the default 128-wire, 6272-gate source exceeds its current wire capacity.
The [gadgetization guide](gadgetization.md) explains the construction and
width formula. `embedded-masking` is the only accepted name for embedded
masking; historical mode names and masking flag aliases are rejected.

For a direct `gen_sandwich_gadget` experiment, use
`EMBEDDED_MASKING_SHUFFLING=8`; `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME`
defaults to `1` and cannot be disabled for that executable's sandwich contract.
Managed TDP runs take these settings from TOML and record them in the recipe
manifest. Older manifests without the settings mean shuffling off and return
home; changing an enabled policy still changes the run identity. The wrapper
pins both child environment values even when shuffling is off. Executable
and driver fingerprints must match on resume; use a fresh run after updating
the implementation.

The Gauntlet exposes `embedded_masking_shuffled` and
`embedded_masking_shuffled_carried` as separate comparison arms. The latter
uses `gauntlet_gen --gadget embedded-masking-shuffled --shuffling-carry-layout`
and decodes outputs through the recorded final role-to-physical layout.
The existing balanced arm remains unshuffled. Shuffling is disabled by default
in the managed recipe; improved write distribution does not establish a
Gauntlet pass. (The meaning of the carried-layout arm under persistent
migration is an open question; see
[TDP_FOUR_STAGE_PIPELINE.md](TDP_FOUR_STAGE_PIPELINE.md) §2c.13.)

### Database mixing schedule

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `db_mixing.target_size_factor` | number, `2.0` | Stage-3 target relative to its incoming gate count, strictly greater than `1` and at most `16`. |
| `db_mixing.hold_work_units` | number, `27.0` | Time spent holding the expanded target, in `0..10000` effective work units. Each attempted move contributes approximately `1 / current_gate_count`; the schedule first spends 3 units growing, then holds for this duration. |

For factor `R` and hold `H`, the driver chooses profile
`3,3+H,3+H,R,R`: no final shrinking leg. The default is `3,30,30,2,2`.
The attempt ceiling is `round((3+H) * R * incoming_gate_count * 1.3)`.
These are scheduling targets and limits, not promises of an exact output size.

### Parallel stage 3

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `parallel.pieces` | integer, `1` (serial) | Fixed partitioning for contiguous piecewise mixing, in `1..64`. Mutually exclusive with `target_piece_gates`, including explicit `pieces=1`. |
| `parallel.target_piece_gates` | integer, unset | Automatically choose the round's piece count as `max(1, current_gate_count / value)` using integer division, in `2..1_000_000_000`. This is a nominal sizing divisor, not a minimum enforced for every resulting piece; shifted edge pieces can be smaller. |
| `parallel.threads` | integer, `pieces+1` for fixed pieces; available CPU count for automatic sizing | Worker pool size in `1..1024`. Requires `pieces>1` or `target_piece_gates`. This controls workers independently from circuit partitioning. |

Piecewise execution alternates contiguous partitions with shifted seams and
shares immutable stores across workers. Worker-count changes preserve the
partitioned schedule's seed assignment; a partitioned run is a different
walk from a serial run. The combined stage-3 checkpoint is written to
`db_mixing.state`.

### Leakage audit and repair

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `leakage_repair.enabled` | bool, `false` | Run the sampled leakage audit and attempt function-preserving database repairs after stage-3 mixing. Writes `db_mixing.leakage_repair.txt`. Disabled by default. |
| `leakage_repair.seed` | integer, `20803` | Separate diagnostic/repair random seed; requires `enabled=true`. Nonnegative TOML integer, at most `9223372036854775807` (TOML's signed 64-bit ceiling). This does not replace the secret construction seed. |
| `leakage_repair.reference` | string, stage-3 input `tdp.mpmct1` | Original reference circuit for the leakage analysis; requires `enabled=true`. Use an `mpmct1` circuit in matching input/wire coordinates. A reference from a differently encoded representation is not interchangeable. Its file hash is part of the recorded recipe. |

These are **all three** managed leakage-repair settings. The audit uses
sampled, held-out measurements and bounded candidate searches, so a clean
report is evidence about those tests, not a security proof. Already completed
stage-3 outputs are skipped on continuation. Decide whether to enable repair
when creating the run; `run.rerun_from_stage=3` recomputes a matching recorded
recipe but cannot change its repair settings. The standalone `circuit_mixer`
also exposes additional low-level leakage-analysis budgets and thresholds;
they are not extra TOML fields accepted by `tdp_gen`.

A `[crossing]` section, which configured the historic crossing walk, is
rejected with a pointer to the archive branch.

### Calibration

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `calibration.enabled` | bool, `false` | Mark the run as calibration-only and permit an explicit construction seed file. Enabling it alone still uses fresh randomness for a fresh run. |
| `calibration.seed_file` | string, unset | Read a fixed seed for reproducible calibration; requires `enabled=true`. A regular file of at most 64 bytes containing one unsigned decimal integer in `0..9223372036854775792`, with surrounding whitespace allowed. On Unix it must have no group/other permissions, for example mode `600`. |

Fresh normal runs obtain the construction seed from OS-backed randomness;
continuations use the run's protected `SEED` file. Store only the file path
in a recipe, not the seed itself. Explicit calibration seeds make the
construction reproducible and therefore are reserved for calibration output.

### Direct driver flags and advanced experiments

The managed command reads configuration; the Bash driver takes these flags.
This map shows the equivalent direct-driver flags for the recipe values.

| Recipe setting | Canonical `scripts/tdp_gen.sh` flag |
| --- | --- |
| `run.directory` | `--run-directory DIR` |
| `run.stop_after_stage` | `--run-stop-after-stage K` |
| `run.rerun_from_stage` | `--run-rerun-from-stage K` |
| `source.wires` | `--source-wires N` |
| `source.gates` | `--source-gates M` |
| `preprocessing.mode` | `--preprocessing-mode MODE` |
| `preprocessing.mask_pair_wires` | `--preprocessing-mask-pair-wires K` |
| `preprocessing.max_open_masks` | `--preprocessing-max-open-masks N` |
| `preprocessing.min_open_masks` | `--preprocessing-min-open-masks N` |
| `preprocessing.balanced_masks` | `--preprocessing-balanced-masks 0\|1` |
| `preprocessing.shuffling_segments` | `--preprocessing-shuffling-segments N` |
| `preprocessing.shuffling_return_home` | `--preprocessing-shuffling-return-home 0\|1` |
| `db_mixing.target_size_factor` | `--db-mixing-target-size-factor R` |
| `db_mixing.hold_work_units` | `--db-mixing-hold-work-units E` |
| `parallel.pieces` | `--parallel-pieces P` |
| `parallel.target_piece_gates` | `--parallel-target-piece-gates B` |
| `parallel.threads` | `--parallel-threads T` |
| Contents of `calibration.seed_file` | `--calibration-seed SEED` (calibration only) |

The driver's `-h`/`--help` lists these flags and its environment interface.
Build selection, source-file path, the mask and shuffling controls, DB path/filter, test-only no-store mode and leakage repair are passed by the wrapper
through its controlled environment. In particular: `TDP_BIN_DIR`, `TDP_SOURCE_C`,
`EMBEDDED_MASKING_K`, `EMBEDDED_MASKING_MAX_OPEN`, `EMBEDDED_MASKING_MIN_OPEN`, `EMBEDDED_MASKING_BALANCED`,
`EMBEDDED_MASKING_SHUFFLING`, `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME`,
`FROZEN_DB_DIR`, `FROZEN_FILTER`, `TDP_GEN_ALLOW_EMPTY_STORE`, `DB_QC`,
`DB_QC_SEED`, and `DB_QC_REFERENCE`. Direct script execution does not provide
the managed command's complete manifest validation and build workflow.
The former masking environment prefix is rejected; use `EMBEDDED_MASKING_*`.
Stage-1/2 recipes now record version 6, so use a fresh run directory after
updating the driver and generator.

For experiments with individual mixer channels,
verification cadence, or extra leakage thresholds, inspect:

```bash
cargo run --release --locked --bin circuit_mixer -- --help
bash scripts/tdp_gen.sh --help
```

Those executables have a broader interface than the managed recipe.
`tdp_gen` pins its stage presets and supported cache/search budgets; arbitrary
environment overrides are scrubbed or replaced. Do not add a standalone
mixer flag as an unrecognized TOML key. Configuration names and retained
spelling aliases are defined in
[src/tdp/config/parse.rs](../src/tdp/config/parse.rs); types and bounds are
validated in [validate.rs](../src/tdp/config/validate.rs), with cross-field
and external-path checks in the runner and [paths.rs](../src/tdp/paths.rs).

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

The managed schedule grows toward twice the incoming size and then holds it
there while continuing replacements. Its profile is `3,30,30,2,2`. There is no
final shrinking leg in this TDP stage. Optional leakage repair runs here too.
The output is `db_mixing.mpmct1`, with `db_mixing.state` and `stage3.log`.
The research reference driver runs the same command with the shorter profile
`2,2,2,2,2` and a ceiling of 5.2·|G2| moves (|G2| = stage-2 gate count).

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

## Stage 4: packing

Packing turns the mixed circuit into the deliverable. The adversary is
assumed to see the full wire state at every window boundary and the inputs,
but not the gates inside a window. The packer reorders the stage-3 circuit
along commutations so that values are born and consumed inside one window,
cuts the order into windows of at most G gates, and publishes each window as
its sparse algebraic normal form (ANF). A commutation-aware repair pass then
removes the remaining boundary exposures of $C$'s values, and the result is
serialized. The design and the measurements are in
[ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md) and
[STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md).

The prototypes are a tracked snapshot of the research run tree
`runs/ncdb_ab_20261001/`; the
[snapshot README](../scripts/windowed_delivery/README.md) lists every script.
They are the reference oracle for the Rust port and keep the run tree's
hard-coded paths: `/Users/rancanetti/code/local_mixing` as the repository,
`runs/ncdb_ab_20261001/` (`RD`) for the top-level scripts and
`runs/ncdb_ab_20261001/waist/` (`W`) for the `waist/` scripts and the run
directories `W/<TAG>`. To use them from another checkout, copy the snapshot
into that layout or edit the paths.

### The three commands

From `RD`, with `R=W/<TAG>` holding `tdp.mpmct1` (stage-2 output),
`tdp.mpmct1.source_c.g57` (the source C) and the stage-3 output:

```bash
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1 > $R/pack_G512.txt 2>&1
bash $W/repair.sh $TAG 512 $N > $R/repair512.log 2>&1
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG > $R/deliver.txt 2>&1
```

| Step | Script | Writes |
| --- | --- | --- |
| Pack | `window_pack3.py A B W_cap G B_mono span_cap label out` (A = stage-2, B = stage-3 circuit) | `packed_G512.mpmct1` and its `.bounds` window lattice; report `pack_G512.txt` |
| Repair | `waist/repair.sh TAG G N`: pair cancellation and leak slides (`order_repair.py`), C-view scans (`hmap_affine --exposure`), merge/split fallback (`merge_bounds2.py`, `anf_split2.py`), equivalence check (`verify_equiv.py`), ANF and linkability scoring | `rep512.mpmct1`, `rep512.bounds`, `rep512.cview.log`, `anf_rep512.txt`, `link_rep512.txt` |
| Serialize | `waist/anf_dump_eval.py` | `deliverable.bin` with size, compressibility and evaluation timings in `deliver.txt` |

Production parameters: G = W_cap = 512 gates per window, a soft monomial
budget B_mono = 48·G = 24,576, span cap 3000, and a 65,536-monomial
split budget inside `repair.sh`. Notes:

- **File-name contract.** `repair.sh TAG G N` takes no paths: it reads
  `W/<TAG>/packed_G<G>.mpmct1` and its `.bounds`, so the packer's output must
  use exactly that name and the same G.
- **Packing a managed run.** Make the `tdp_gen` run directory available as
  `W/<TAG>` (copy or symlink) and pass `db_mixing.mpmct1` as the packer's
  second argument. The stage-4 measurements were made on `val_chain.sh`
  outputs, whose stage-2 and stage-3 settings differ from the managed
  defaults (see [Two ways to run](#two-ways-to-run)).
- **Pass criteria.** The repaired C-view must reach the floor: the visible
  count in `rep512.cview.log` equals n (only the outputs of $C$ are
  visible), the last split pass reports `0 splits added` with no
  `WARNING: no exposure-free split`, and both `verify_equiv.py` checks print
  `EQUIVALENT`. The full checklist is in
  [TDP_FOUR_STAGE_PIPELINE.md](TDP_FOUR_STAGE_PIPELINE.md) §6.7.
- **No `set -e`.** `repair.sh` and `val_chain.sh` keep going after a failed
  step, so read every log. Never run two repairs in the same `TAG`
  directory, and never edit a driver while it runs.

### The reference driver

`waist/val_chain.sh TAG N MC S SLICE CONFIG` runs stages 1-4 in one go, with
fresh CSPRNG seeds recorded in `W/<TAG>/seed.txt`:

```bash
bash runs/ncdb_ab_20261001/waist/val_chain.sh vL64a 64 1000 384 1280 L8
bash runs/ncdb_ab_20261001/waist/val_chain.sh vL128a 128 3000 896 2560 L8
```

CONFIG is `L8` (bounded chains `CHAIN=2`, `CHAIN_LEN=8`; recommended) or `A`
(no chains). Stage 3 uses `FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11`
and `MOVES = 5.2·|G2|`. Besides the stage-4 outputs, the driver runs a
rebirth census (`mult_s2.txt`) and a stage-2 span check (`span_s2.txt`) in
the background. It needs `hmap_affine`, built with
`cargo build --release --locked --features security-tools --bin hmap_affine`.

## Historic stages

The former stages 4 (split every remaining complemented gate), 5 (the
crossing walk: shoot fragments to collisions, rewrite them so they cross, and
contract with undo/merge moves under a size thermostat) and 6 (`fcompress`)
are historic, as is the curated store. Windowed packing replaced them. Their
code and tests remain on branch `ran/archive/curated-and-postmix`; runs that
reached them, or that record recipe v8, need that branch to continue. Within
the mixer, the crossing walk was slot 3 of `Mixer::run`; stage 3 runs every
non-twist round as a DB move (`--p-db 1.0`), so slot 3 never ran there and its
removal left stage-3 output unchanged. The driver rejects the old stage
numbers 4-6, the crossing flags and the `[crossing]` section with a pointer
to the archive. See [CURATED_AND_POSTMIX_ARCHIVE.md](CURATED_AND_POSTMIX_ARCHIVE.md).

## The shared pieces

We do not need to understand every file to change one stage. The circuit
representation is in [src/circuit/xgate.rs](../src/circuit/xgate.rs), and its
readers/writers are in [src/circuit/formats.rs](../src/circuit/formats.rs).
The mutable gate tape used by `circuit_mixer` is in
[src/engine/arena.rs](../src/engine/arena.rs). Mixer state, sampling, checkpoint
I/O, and reporting live together in [src/engine/mixer/](../src/engine/mixer/).

Use these current module paths directly. The old top-level `preprocessing`,
`db_mixing` and `postprocessing` wrappers have been removed, as has the
`stages/post-processing` module of the historic stages. Files named
`environment.rs` read settings that the current TDP driver pins for the run;
they still support the current cache and mixing behavior.

Stage 3 can run on contiguous pieces in parallel; the coordinator is
[piecewise.rs](../src/engine/mixer/piecewise.rs). Correctness checks live in
[tests/](../tests/), while attacks and leakage measurements live in
[security_tests/](../security_tests/). The detailed
[current-method documentation](local_mixing_documentation.md) explains the
implemented construction and attacks. The [history](local_mixing_history.md)
preserves the motivation and earlier experiments behind this order.
