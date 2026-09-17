# The TDP Pipeline

The `tdp_gen` command constructs the current TDP pipeline. We start with a reversible circuit $C$
on $n$ wires and pass it through six steps. The first two build the
representation we want to mix. The next three change its local structure, and
the last step removes redundancy and packs the circuit.

```text
C → sliced sandwich → gadgetization → DB mixing → splitting → crossing → compress and pack → final.esop1
        step 1           step 2         step 3       step 4      step 5         step 6           output
```

This is a guide to the current code. For commands and configuration, start
with the [README](../README.md).

## Where a run starts

The `tdp_gen` command enters through [src/main.rs](../src/main.rs) and
[src/commands/tdp_gen.rs](../src/commands/tdp_gen.rs). From there,
[src/tdp/config/](../src/tdp/config/) reads and checks the recipe, and
[src/tdp/runner.rs](../src/tdp/runner.rs) prepares the run, checks any saved
manifest, and selects the stage binaries.

Then [scripts/tdp_gen.sh](../scripts/tdp_gen.sh) runs the six steps below.
This is the easiest file to read when we want to see the exact stage order,
arguments, or output names. Steps 1 and 2 share one generator process;
`circuit_mixer` runs steps 3–5, and `fcompress` runs step 6. Each executable's
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

There are **36 recognized section settings plus `config_version`**. The
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
| `config_version` | integer, required `1` | TOML schema identifier. This is separate from saved execution recipe v8. |
| `run.directory` | string, generated directory under `runs/` | Output directory and continuation target. Omit for a fresh timestamped run; select the same directory to continue it. Within the repository it must be a dedicated subdirectory of `runs/`; dedicated external directories are also allowed. It must not overlap build or database directories. |
| `run.build_binaries` | bool, `true` | Build the three stage executables for a fresh run. `false` requires them to exist under the resolved build directory. Managed continuation always reuses matching binaries and skips rebuilding. |
| `run.build_directory` | string, `CARGO_TARGET_DIR` or `target` | Cargo artifact directory. Stage binaries are selected from its host-target `release/` directory; changing this can affect whether the saved binary fingerprints can be matched. |
| `run.stop_after_stage` | integer, `6` | Last stage to reach, in `2..6`. Stages 1 and 2 run together; `2` is the shortest run. |
| `run.rerun_from_stage` | integer, unset | Recompute this stage and later stages reached by this invocation, in `2..6`, even if their outputs exist. It does not authorize changing a saved recipe or make earlier stages run again. |
| `run.adopt_unverified_run` | bool, `false` | Retired compatibility field. `true` is rejected: existing runs need a valid current managed manifest. Leave this omitted or false. |

Run lifecycle options such as the stop stage and worker count do not change
the circuit recipe. Algorithm choices, source identity, database paths and
piece partitioning are recorded in `tdp_command.conf`; changing a recorded
choice requires a fresh run. The manifest starts with
`tdp_command_recipe=8`; `stage12.recipe` uses `tdp_stage12_recipe=6`.
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
| `database.curated_dir` | string, `FROZEN_CURATED_DIR` or unset | Optional curated store, used ahead of regular lookup when the move's routing policy selects it. Requires the regular store; this is not a standalone replacement for `regular_dir`. |
| `database.curated_control_order` | string, `"native"` | Interpret decoded G57 control bytes. Allowed values: `"native"`, `"legacy-swapped-controls"`. Use the latter only for a historical store known to use that convention. Regular stores use native order. |
| `database.lookup_miss_filter` | string, `"auto"` | `"auto"` enables the in-memory miss filter at at least 60 GiB available RAM; `"on"` requires a nonempty `filters.bin` for every configured store; `"off"` avoids loading filters. Filters must belong to their exact store. |
| `database.allow_no_database_for_tests` | bool, `false` | Allow a stage-3 plumbing run with all replacement channels disabled. To set true, clear both DB paths and their environment fallbacks. A curated-only store or a regular store combined with this mode is rejected when stage 3 runs. |

See [Frozen Database](frozen_database.md#constructing-and-freezing-a-store)
for regular and curated construction, freezing and validation commands.

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
in production; improved write distribution does not establish a Gauntlet pass.

### Database mixing schedule

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `db_mixing.target_size_factor` | number, `2.0` | Stage-3 target relative to its incoming gate count, strictly greater than `1` and at most `16`. |
| `db_mixing.hold_work_units` | number, `27.0` | Time spent holding the expanded target, in `0..10000` effective work units. Each attempted move contributes approximately `1 / current_gate_count`; the schedule first spends 3 units growing, then holds for this duration. |

For factor `R` and hold `H`, the driver chooses profile
`3,3+H,3+H,R,R`: no final shrinking leg. The default is `3,30,30,2,2`.
The attempt ceiling is `round((3+H) * R * incoming_gate_count * 1.3)`.
These are scheduling targets and limits, not promises of an exact output size.

### Parallel stages 3 and 4

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `parallel.pieces` | integer, `1` (serial) | Fixed partitioning for contiguous piecewise mixing, in `1..64`. Mutually exclusive with `target_piece_gates`, including explicit `pieces=1`. |
| `parallel.target_piece_gates` | integer, unset | Automatically choose the round's piece count as `max(1, current_gate_count / value)` using integer division, in `2..1_000_000_000`. This is a nominal sizing divisor, not a minimum enforced for every resulting piece; shifted edge pieces can be smaller. |
| `parallel.threads` | integer, `pieces+1` for fixed pieces; available CPU count for automatic sizing | Worker pool size in `1..1024`. Requires `pieces>1` or `target_piece_gates`. This controls workers independently from circuit partitioning. |

Piecewise execution alternates contiguous partitions with shifted seams and
shares immutable stores across workers. Worker-count changes preserve the
partitioned schedule's seed assignment; a partitioned run is a different
walk from a serial run. Stage 5 resumes the combined split checkpoint.

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

### Crossing and final compression

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `crossing.target_size_factor` | number, `2.0` | Stage-5 target relative to split output size, in `1..16`: `target=round(split_gates * factor)`. |
| `crossing.width_penalty_base` | number, `3.0` | Base `B` of the wide-split acceptance penalty, in `1..1_000_000`. A relevant parent/control width `c` above threshold `d` is allowed with probability `B^(-(c-d))`; `B=1` disables this damping. |
| `crossing.width_penalty_threshold` | integer, `1` | Width `d` below which this penalty does not apply, in `0..1_000_000_000`. Larger values permit wider splitting more readily. This is a width bias, not a maximum legal gate width. |
| `crossing.size_tolerance_divisor` | integer, `25` | Sets size-controller temperature to `max(64, round(target / divisor))`, in `1..1_000_000_000`. Larger divisors tighten tolerance until the floor of 64 is reached. |
| `crossing.move_attempts` | integer, `6 * target` | Additional stage-5 attempt budget, in `1..1_000_000_000_000`. The driver adds the move counter from `split.state` because the standalone mixer's resumed `--moves` ceiling is absolute. |

Stage 6 uses the managed compression preset and a derived seed. There is no
`[compression]` TOML section or adjustable splitting section in the current
schema. Stop stage, rerun stage and the parallel settings still control the
relevant lifecycle and stage-4 execution.

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
| `crossing.target_size_factor` | `--crossing-target-size-factor R` |
| `crossing.width_penalty_base` | `--crossing-width-penalty-base B` |
| `crossing.width_penalty_threshold` | `--crossing-width-penalty-threshold C` |
| `crossing.size_tolerance_divisor` | `--crossing-size-tolerance-divisor D` |
| `crossing.move_attempts` | `--crossing-move-attempts M` |
| Contents of `calibration.seed_file` | `--calibration-seed SEED` (calibration only) |

The driver's `-h`/`--help` lists these flags and its environment interface.
Build selection, source-file path, the mask and shuffling controls, DB paths/filter/control
convention, test-only no-store mode and leakage repair are passed by the wrapper
through its controlled environment. In particular: `TDP_BIN_DIR`, `TDP_SOURCE_C`,
`EMBEDDED_MASKING_K`, `EMBEDDED_MASKING_MAX_OPEN`, `EMBEDDED_MASKING_MIN_OPEN`, `EMBEDDED_MASKING_BALANCED`,
`EMBEDDED_MASKING_SHUFFLING`, `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME`,
`FROZEN_DB_DIR`, `FROZEN_CURATED_DIR`, `FROZEN_FILTER`,
`FROZEN_CURATED_VALUE_CONVENTION`, `TDP_GEN_ALLOW_EMPTY_STORE`, `DB_QC`,
`DB_QC_SEED`, and `DB_QC_REFERENCE`. Direct script execution does not provide
the managed command's complete manifest validation and build workflow.
The former masking environment prefix is rejected; use `EMBEDDED_MASKING_*`.
Stage-1/2 recipes now record version 6, so use a fresh run directory after
updating the driver and generator.

For experiments with individual mixer channels, splitting controls,
verification cadence, or extra leakage thresholds, inspect:

```bash
cargo run --release --locked --bin circuit_mixer -- --help
cargo run --release --locked --bin fcompress -- --help
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

## 1. Build the sliced sandwich

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

## 2. Gadgetize the sandwich

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
Steps 1–2 write `tdp.mpmct1`, its source/sandwich sidecars, and `stage12.log`.

## 3. Mix with the frozen database

We sample small circuit windows and replace them with different spellings of
the same function. To find these spellings, we compose the window's
polynomials, canonicalize its wire labels, and query the frozen tables.
A candidate must be mapped back to the window and checked before insertion.

The default schedule grows toward twice the incoming size and then holds it
there while continuing replacements. Its profile is `3,30,30,2,2`. There is no
final shrinking leg in this TDP stage. Optional leakage repair runs here too.
The output is `db_mixing.mpmct1`, with `db_mixing.state` and `stage3.log`.

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

## 4. Split the remaining complemented gates

The database still gives us many gates with the same r57 shape. We split
their complemented firing conditions into plain conjunctions. For instance,

$$
a\mathrel{\oplus{=}}b\lor\neg c
\quad\longrightarrow\quad
\{a\mathrel{\oplus{=}}b,\quad a\mathrel{\oplus{=}}\neg b\wedge\neg c\}.
$$

The two conditions are disjoint, and their XOR is the original condition.
The stage also tries absorbed-NOT twists between compatible brackets and
shoots fragments through the circuit. It stops when the complemented-gate
pool is exhausted or repeated bracket searches reach the failure limit.
It then writes `split.mpmct1`, `split.state`, and `stage4.log`.

Read `split_twist_move` in
[src/stages/post-processing/splitting.rs](../src/stages/post-processing/splitting.rs).
The local rewrite identities live in
[src/engine/moves/rules.rs](../src/engine/moves/rules.rs).

## 5. Run the crossing walk

Now we move fragments toward collisions and rewrite the colliding gates so
their pieces can cross. These moves can introduce wider conjunctions than the
original r57 gates. Undo and merge moves remove compatible fragments again,
while the size target and width penalty control growth.

This step resumes `split.state`. It writes `crossing.mpmct1`,
`crossing.state`, and `stage5.log`. The current default target is twice the
split circuit's size, with six times that target as the move-attempt budget.

Read `cross_move_on`, `undo_move`, and `merge_move` in
[src/stages/post-processing/crossing.rs](../src/stages/post-processing/crossing.rs).
The scheduler and acceptance logic are in
[src/engine/mixer/scheduling.rs](../src/engine/mixer/scheduling.rs).

## 6. Compress and pack

Finally, we gather compatible gates with the same target and simplify their
combined firing function. We repeat this with transport and reduction passes,
then pack each consecutive same-target run into a generalized gate. Its
activation function is reduced through ANF and a deterministic ESOP spelling.

The final file is `final.esop1`, with `stage6.log`. A packed-gate count and an
expanded cube count measure different things, so keep both in mind when
reading the compression log.

Read [src/programs/fcompress/mod.rs](../src/programs/fcompress/mod.rs), then
`compress_anc` in
[src/stages/post-processing/compression/mod.rs](../src/stages/post-processing/compression/mod.rs).
The same directory separates transport, reduction, and
[packing](../src/stages/post-processing/compression/packing.rs).

## The shared pieces

We do not need to understand every file to change one stage. The circuit
representation is in [src/circuit/xgate.rs](../src/circuit/xgate.rs), and its
readers/writers are in [src/circuit/formats.rs](../src/circuit/formats.rs).
The mutable gate tape used by `circuit_mixer` is in
[src/engine/arena.rs](../src/engine/arena.rs). Mixer state, sampling, checkpoint
I/O, and reporting live together in [src/engine/mixer/](../src/engine/mixer/).

Use these current module paths directly. The old top-level `preprocessing`,
`db_mixing` and `postprocessing` wrappers have been removed. Files named
`environment.rs` read settings that the current TDP driver pins for the run;
they still support the current cache and mixing behavior.

Stages 3–4 can run on contiguous pieces in parallel; the coordinator is
[piecewise.rs](../src/engine/mixer/piecewise.rs). Correctness checks live in
[tests/](../tests/), while attacks and leakage measurements live in
[security_tests/](../security_tests/). The detailed
[current-method documentation](local_mixing_documentation.md) explains the
implemented construction and attacks. The [history](local_mixing_history.md)
preserves the motivation and earlier experiments behind this order.
