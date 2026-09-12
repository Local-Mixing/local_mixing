# GSS configuration and flag reference

`cargo run --release --locked -- gss` reads `configs/gss.toml`. Most operator
settings are TOML fields. The `gss` command itself accepts only `--config PATH`,
`--dry-run`, and `-h`/`--help`; mixing flags in the tables below belong to
`scripts/gss_mix.sh`. Environment-variable counterparts are explicitly marked
`env`; they are not command-line flags. Normal `gss` translates the config
into shell arguments and a controlled environment.

This reference describes current names for new runs (recipe v7).
Saved v3/v4/v5/v6 runs retain their original drivers and recorded binaries.

## Configuration syntax and defaults

`config_version = 1` is required at the top level. It selects the configuration
syntax version, not a gadget or checkpoint version. There is no shell flag.

Fields under `[source]` have names such as `source.wires`. Section headers
can be replaced by equivalent dotted keys (`source.wires = 128`). Omitted
fields and sections use defaults. Relative paths in TOML resolve from the
repository root, not the configuration file's directory. Unknown fields,
duplicate keys and invalid types are rejected.

## Database settings explained

The regular and optional curated frozen stores contain small equivalent
circuit replacements for db_mixing. The curated store is tried ahead of the
regular store; it does not replace the requirement for a regular store.

`database.curated_control_order` tells the reader how to interpret the
control ordering in decoded circuits from the curated store:

- `native` (default): read each gate with its stored control order. Use this
  for current database builds.
- `legacy-swapped-controls`: swap each decoded gate's two controls to match
  the convention used by the historical pre-`2ed0222a` curated store.

This is a database compatibility setting. Selecting the wrong order can change
which function a decoded circuit computes. It does not convert or rewrite the
database on disk. The regular store is read with the native convention. The four current servers
were checked against their deployed files: see [the control-order audit](DB_CONTROL_ORDER.md).

`database.lookup_miss_filter` selects an optional lookup miss filter loaded from each
store's `filters.bin`. Before an expensive store read, this in-memory structure
can say that a key is definitely absent. Possible matches still go through
the normal lookup. For a valid matching filter, it avoids work on misses
without changing which replacements are available.

- `auto` (default): the driver enables it when `/proc/meminfo` reports at least
  60 GiB available memory. Missing filter files fall back to unfiltered reads.
- `on`: request it regardless of available memory; the wrapper requires a
  nonempty `filters.bin` in each configured store.
- `off`: perform lookups without the optional miss filter.

The filter costs load time and additional RAM; the amount depends on the store.
It does not screen circuits for correctness or security and does not delete
entries from the database.

`database.allow_no_database_for_tests = true` permits pipeline plumbing tests without
any replacement database. The driver disables all db_mixing database replacement
channels. It does not create an empty database or make an incomplete database
valid. The wrapper rejects this mode with a configured regular store, including
one supplied through the environment; a curated-only store is also rejected.
Normal database mixing uses `false`.

## Run settings

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `run.directory` | Fresh timestamped directory under `runs/` | `--run-directory DIR` (required directly) | Destination for all stage circuits, checkpoints, logs, seed and manifest. Reuse with the same recipe to continue; omit for a new run each invocation. |
| `run.build_binaries` | `true` | None; wrapper-only | Build the three pipeline executables in release mode before a new run. `false` requires existing executables. Managed resumes always reuse their fingerprinted binaries. |
| `run.build_directory` | `CARGO_TARGET_DIR`, otherwise repository `target/` | env `CARGO_TARGET_DIR`; driver receives env `GSS_BIN_DIR` | Cargo build directory. The wrapper uses `<dir>/<host-triple>/release`; the direct driver's default is repository `target/release`. |
| `run.stop_after_stage` | `6` | `--run-stop-after-stage K` | Finish stage K and exit, retaining completed outputs. Valid stages are 2..6. |
| `run.rerun_from_stage` | Unset | `--run-rerun-from-stage K` | Recompute stage K and later stages reached by this invocation, replacing their outputs. Earlier stages are reused. This does not bypass managed recipe/fingerprint checks. |
| `run.adopt_unverified_run` | `false` | None; wrapper-only | Explicitly accept an old run without `gss_command.conf`, whose earlier recipe cannot be verified. Also requires calibration mode. Ordinary managed continuation does not require this. |

Stages 1-2 construct and gadgetize; stage 3 is db_mixing; stage 4 splits;
stage 5 performs the crossing walk; stage 6 compresses and packs.

## Source settings

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `source.wires` | `128` | `--source-wires N` (required directly) | Original computation C's wire count, 3..4095. Sandwich construction and gadgetization add wires. |
| `source.gates` | Inferred for a supplied source, otherwise `round(n * log2(n)^2)` | `--source-gates M`; direct default `0` means derived | Gate count of C and the generated D circuit before sandwich construction. If source.path is supplied, an explicit count must match the file. At n=128 the generated default is 6272 gates each. |
| `source.path` | Unset | env `GSS_SOURCE_C` | Load C from a nonempty G57 circuit that fits source.wires. Without it, generate a random C. The current wrapper does not accept mpmct1/esop1/anf1 here. |

## Database setting counterparts

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `database.regular_dir` | env value or unset | env `FROZEN_DB_DIR` | Regular frozen replacement store. Required when db_mixing will run, except in the explicit no-database testing mode. |
| `database.curated_dir` | env value or unset | env `FROZEN_CURATED_DIR` | Optional curated store tried ahead of the regular store. |
| `database.curated_control_order` | `native` | env `FROZEN_CURATED_VALUE_CONVENTION` | Curated-store control ordering: native or legacy-swapped-controls. See explanation above. |
| `database.lookup_miss_filter` | `auto` | env `FROZEN_FILTER`: 1=on, 0=off, unset=auto | Optional in-memory miss filter; see memory and file requirements above. |
| `database.allow_no_database_for_tests` | `false` | env `GSS_MIX_ALLOW_EMPTY_STORE=1` | Allow no-database plumbing tests with database replacement channels disabled. |

## Preprocessing settings

An open mask is a masking gadget that has been applied to a data wire and has
not yet been removed. The band is the auxiliary wire pool used by these masks.

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `preprocessing.mode` | `quadratic-masking` | `--preprocessing-mode MODE` | Two supported modes: quadratic-masking and nonlinear291, both available in ordinary builds. Aliases ran-balanced, blinded-v5 and blinded_v5 select quadratic masking. Product-2223/2223 and standalone nonlinear193 remain readable for saved runs and are rejected for fresh managed runs. |
| `preprocessing.mask_pair_wires` | `2` | `--preprocessing-mask-pair-wires K`; env `BV5_K` | Band wires in the paired part of each masking gadget, before even rounding and the extra balancing wire; 2..64 for new Quadratic masking runs. Also affects automatic refresh sizing. |
| `preprocessing.max_open_masks` | `3` | `--preprocessing-max-open-masks N`; env `BV5_MAX_OPEN` | Rolling cap on simultaneously open masks; also influences read-time masking. Range 2..64. |
| `preprocessing.min_open_masks` | `2` | `--preprocessing-min-open-masks N`; env `BV5_MIN_OPEN` | Minimum open-mask coverage between a wire's first and last mask. Range 1..63 and strictly less than max_open_masks. |
| `preprocessing.balanced_masks` | `true` | `--preprocessing-balanced-masks 0|1`; env `BV5_BALANCED` | Add one extra band-wire term to each mask; false selects the plain-mask research variant. |

The managed construction uses the classic sandwich, guards, quadratic fire,
band helper ancillas, no extra LGIs and no encoded I/O. Refresh bursts retain
the historical wire pool; the managed recipe does not enable `BV5_BURST_BANDONLY`.
Balanced masks remain the default.
For quadratic masking the band must be wide enough:
`2 * source.wires > K - (K % 2) + balanced`.

For nonlinear291, specify only `preprocessing.mode = "nonlinear291"` in this
section. Both TOML and the direct driver reject explicit mask controls for this
mode, including values equal to quadratic-mask defaults. Omitted controls
remain valid. Historical v3-v6 recipes keep their recorded behavior.
Explicit shell flags override their corresponding BV5 environment values.
The wrapper clears ambient BV5 controls and applies the recorded recipe.

## db_mixing settings

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `db_mixing.target_size_factor` | `2` | `--db-mixing-target-size-factor R` | Target gate-count multiplier relative to the gadgetized db_mixing input. The controller ramps toward this size over the first 3 effective work units. It is a target, not an exact output-size guarantee. |
| `db_mixing.hold_work_units` | `27` | `--db-mixing-hold-work-units E` | Effective work units spent holding near the expanded size. One unit is approximately one attempted move per current gate: the clock adds 1/current_gate_count each move. |

## Parallel settings

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `parallel.pieces` | `1` (serial) | `--parallel-pieces P` | Fixed nominal number of contiguous gate-sequence pieces for stages 3-4; range 1..64. Pieces are mixed, rejoined in order, then boundaries shift between rounds. |
| `parallel.target_piece_gates` | Unset | `--parallel-target-piece-gates B` | Automatic sizing divisor in gates: P=max(1,floor(current_gate_count/B)), recalculated each round; 2..1000000000. Shifted end pieces can be smaller than B. |
| `parallel.threads` | P+1 for fixed pieces; available CPU parallelism for automatic sizing | `--parallel-threads T` | Worker-pool size, 1..1024; requires pieces > 1 or automatic sizing. It stays fixed within each stage even when automatic P changes. |

Choose either pieces or target_piece_gates, including when pieces=1. Shifted rounds
with P>1 can contain P+1 pieces. More pieces than threads are queued for the
same worker pool; fewer pieces leave workers idle. These controls do not
parallelize crossing or final compression.

## Crossing settings

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `crossing.target_size_factor` | `2` | `--crossing-target-size-factor R` | Target gate-count multiplier relative to the split-stage output. |
| `crossing.width_penalty_base` | `3` | `--crossing-width-penalty-base B` | Base of the width penalty on expansion attempts: larger values more strongly discourage splitting wide gates. Width is the number of control wires. |
| `crossing.width_penalty_threshold` | `1` | `--crossing-width-penalty-threshold C` | Largest control count exempt from the width penalty. Increasing it makes expansion of wider gates easier. |
| `crossing.size_tolerance_divisor` | `25` | `--crossing-size-tolerance-divisor D` | Temperature=max(64,round(target/D)); increasing D makes gate-count steering sharper, decreasing it makes steering softer. This is not a hard size limit. |
| `crossing.move_attempts` | 6 × crossing target | `--crossing-move-attempts M` | Additional stage-5 move attempts, including unsuccessful attempts. The driver adds the checkpoint's previous move count before passing the underlying mixer's absolute budget. |

The width check passes automatically at or below C, otherwise with probability
`B^-(width-C)`. At defaults, gates with 1/2/3/4 controls pass that check with
probability 1, 1/3, 1/9 and 1/27. Other move checks still apply.

## Leakage audit and repair

`leakage_repair` enables an optional sampled leakage audit
and circuit-repair pass after db_mixing. It looks for affine relations between
internal values and reference signals, and correlated gate-firing predicates.
It attempts database replacements that preserve the exact circuit function
while removing detected signals in the screened region. This is a bounded
sampled audit, not a proof that no leakage remains. It can modify the circuit;
it is not just a report or a database-file integrity check.

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `leakage_repair.enabled` | `false` | env `DB_QC=1` → `fmix --leakage-repair` | Enable the audit-and-repair pass after db_mixing and before its output/checkpoint are saved. The report is run/db_mixing.leakage_repair.txt. |
| `leakage_repair.seed` | `20803` | env `DB_QC_SEED` → `fmix --leakage-repair-seed` | Independent randomness for audit samples and repair search. Does not replace the run's construction seed. Explicit values require enabled leakage repair. |
| `leakage_repair.reference` | db_mixing input | env `DB_QC_REFERENCE` → `fmix --leakage-repair-reference` | Circuit whose internal signals are compared with the mixed circuit. Must use matching input/wire coordinates; not automatically the original C supplied by source.path. Explicit paths require enabled leakage repair. |

The GSS driver uses an mpmct1-compatible leakage reference. The pass also works after
piecewise db_mixing; it runs serially on the assembled circuit. See
[`DB_QUALITY_CONTROL.md`](DB_QUALITY_CONTROL.md) for its sample budgets and scope.

## Calibration settings

| TOML field | Default | Current counterpart | Meaning |
| --- | --- | --- | --- |
| `calibration.enabled` | `false` | None; wrapper-only designation | Mark the run as calibration-only and permit an explicit seed file. Required when adopting an unverified old run. Enabling it alone does not select a fixed seed. |
| `calibration.seed_file` | Unset | File contents → `--calibration-seed SEED` | Read a private file containing one unsigned decimal seed. Requires calibration enabled. Without an explicit seed, new runs draw a random seed and continuing runs reuse their saved SEED. |

The seed file must be readable only by its owner on Unix (for example mode 600).
The wrapper keeps its numeric value out of the config and dry-run output.

## Main command flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | `configs/gss.toml` | Select the TOML recipe or an existing marked Markdown recipe. |
| `--dry-run` | Off | Resolve and validate the recipe and print the planned invocation, without building or creating run artifacts. |
| `-h`, `--help` | — | Show help. The direct shell driver also accepts these. |

For example, four pieces with five worker threads:

```toml
config_version = 1

[parallel]
pieces = 4
threads = 5
```

Other settings retain their defaults; db_mixing still needs a regular database
path from the configuration or FROZEN_DB_DIR. `gss --parallel-pieces 4` is not a supported
main-command flag: use the recipe or the direct shell flag.

## Compatibility aliases

Old TOML fields and shell spellings remain accepted. The canonical names above
are used by new runs and help. Do not supply both an old TOML field and its
new name: they are the same setting, and duplicate aliases are rejected.
The driver also rejects duplicate preprocessing flags after alias normalization.
Environment names are unchanged. Saved v3/v4/v5/v6 recipes use their exact
original drivers, arguments, stage filenames and fingerprinted binaries.

| Old TOML field | Canonical field |
| --- | --- |
| `run.build_release` | `run.build_binaries` |
| `run.build_target_dir` | `run.build_directory` |
| `run.stop_after` | `run.stop_after_stage` |
| `run.force_from` | `run.rerun_from_stage` |
| `run.adopt_existing_run` | `run.adopt_unverified_run` |
| `database.curated_value_convention` | `database.curated_control_order` |
| `database.filter` | `database.lookup_miss_filter` |
| `database.allow_empty_store` | `database.allow_no_database_for_tests` |
| `gadget.mode` | `preprocessing.mode` |
| `gadget.balanced_masks` | `preprocessing.balanced_masks` |
| `gadget.k`, `gadget.mask_pair_wires` | `preprocessing.mask_pair_wires` |
| `gadget.max_open`, `gadget.max_open_masks` | `preprocessing.max_open_masks` |
| `gadget.min_open`, `gadget.min_open_masks` | `preprocessing.min_open_masks` |
| `phase_a.expand` | `db_mixing.target_size_factor` |
| `phase_a.hold` | `db_mixing.hold_work_units` |
| `parallel.min_block_size` | `parallel.target_piece_gates` |
| `crossing.target_factor` | `crossing.target_size_factor` |
| `crossing.width_base` | `crossing.width_penalty_base` |
| `crossing.width_threshold` | `crossing.width_penalty_threshold` |
| `crossing.temperature_divisor` | `crossing.size_tolerance_divisor` |
| `crossing.moves` | `crossing.move_attempts` |
| `quality_control.enabled` | `leakage_repair.enabled` |
| `quality_control.seed` | `leakage_repair.seed` |
| `quality_control.reference` | `leakage_repair.reference` |

| Old shell flag | Canonical flag |
| --- | --- |
| `-n` | `--source-wires` |
| `-o` | `--run-directory` |
| `-s` | `--calibration-seed` |
| `--mcd` | `--source-gates` |
| `--gadgetization-mode`, `--gadget-mode` | `--preprocessing-mode` |
| `--bv5-k`, `--gadget-mask-pair-wires` | `--preprocessing-mask-pair-wires` |
| `--bv5-max-open`, `--gadget-max-open-masks` | `--preprocessing-max-open-masks` |
| `--bv5-min-open`, `--gadget-min-open-masks` | `--preprocessing-min-open-masks` |
| `--bv5-balanced`, `--gadget-balanced-masks` | `--preprocessing-balanced-masks` |
| `--expand` | `--db-mixing-target-size-factor` |
| `--hold` | `--db-mixing-hold-work-units` |
| `--xr` | `--crossing-target-size-factor` |
| `--xb` | `--crossing-width-penalty-base` |
| `--xc` | `--crossing-width-penalty-threshold` |
| `--xtdiv` | `--crossing-size-tolerance-divisor` |
| `--xmoves` | `--crossing-move-attempts` |
| `--stop-after` | `--run-stop-after-stage` |
| `--force-from` | `--run-rerun-from-stage` |
| `--pieces` | `--parallel-pieces` |
| `--min-block-size` | `--parallel-target-piece-gates` |
| `--piece-threads` | `--parallel-threads` |

The underlying `fmix` also accepts `--db-mixing`, `--parallel-pieces`,
`--parallel-target-piece-gates`, `--parallel-threads`, and the
`--leakage-repair` / `--leakage-repair-*` family. Its corresponding old
`--phase-a`, `--pieces`, `--min-block-size`, `--piece-threads` and
`--qc` / `--qc-*` names remain compatibility aliases.
