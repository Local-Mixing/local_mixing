# The TDP Pipeline

This is the operational manual for the four-stage TDP pipeline: how to run
each stage, what it writes, and where its code lives. We start with a
reversible circuit $C$ on $n$ wires and end with a published circuit that
computes $C$ on a public slice.

| Stage | What it does | Runs as | Output |
| --- | --- | --- | --- |
| 1. Sandwich | Classic sliced sandwich around $C$ on $2n$ wires | `gen_sandwich_gadget` (with stage 2) | `tdp.mpmct1.sandwich.mpmct1`, `tdp.mpmct1.source_c.g57` |
| 2. Preprocessing | Embedded masking on $4n$ wires: masked fires, band, guards, persistent carrier migration (preset L8) | `gen_sandwich_gadget` | `tdp.mpmct1` |
| 3. Mixing | Database mixing with the regular frozen store | `circuit_mixer --tdp --db-mixing --p-db 1.0` | `db_mixing.mpmct1` |
| 4. Packing | Windowed ANF delivery: commuting reorder into windows, repair, serialization | `tdp_packing run` | `packing/deliverable.tdp4` |

```text
C → sliced sandwich → embedded masking → DB mixing → windowed ANF delivery → packing/deliverable.tdp4
       stage 1            stage 2          stage 3          stage 4
```

The managed `tdp_gen` command drives all four stages. **Its defaults
reproduce the chosen validation runs** (`val_chain.sh` CONFIG=L8, runs
vL64a/b and vL128a/b): a managed run that sets only `source.wires` (64 or
128) produces, from the same seeds, byte-identical stage-2, stage-3, packed,
repaired and deliverable artifacts. This is checked on the archived runs
themselves (the historic anchor) and by a reproduction test against the
reference script; see
[docs/STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md) §2 and §12.

Stage 4 is the Rust port (`src/stages/packing/`, binary `tdp_packing`) of
the Python prototypes in [scripts/windowed_delivery/](../scripts/windowed_delivery/),
which stay as the canonical **oracle**: the port is bit-exact wherever the
oracle is deterministic, and CI compares the two on random inputs on every
push.

The curated store and the former stages 4 (split), 5 (crossing walk) and
6 (`fcompress`) are historic. They remain on branch
`ran/archive/curated-and-postmix`; see
[Curated DB and post-mixing archive](CURATED_AND_POSTMIX_ARCHIVE.md) for what
was retired, how to retrieve it, and why.

Further reading:

- [TDP_FOUR_STAGE_PIPELINE.md](TDP_FOUR_STAGE_PIPELINE.md): the full
  reference for the four stages (construction details, parameters, measured
  behaviour, pass criteria, open issues).
- [STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md): the stage-4 port, the
  reference recipe the defaults reproduce, seeds and the environment policy,
  and the validation results.
- [STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md): the stage-2 redesign and
  the linkable-exposure metrics that the recommended configuration is chosen
  by.
- [ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md): windowed ANF packing, its
  threat model, and the deliverable format.

For first-time setup and the circuit utilities, start with the
[README](../README.md).

## Two ways to run

**The managed driver** (`tdp_gen`) is the supported entry point. It reads a
TOML recipe (config v2), builds and fingerprints the three stage binaries,
draws the seeds into the private `$RUN/SEEDS`, records a manifest (recipe
v10), and resumes runs safely. Its defaults are the recommended
configuration: stage-2 preset `L8`, stage-3 schedule `validated`, stage 4 at
G = 512 with a 65,536-monomial split budget.

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

The run ends with `[tdp_gen] PIPELINE COMPLETE (stages 1-4): <run>/packing/deliverable.tdp4`.
A run directory holds:

| Stage | Files |
| --- | --- |
| 1. Sliced sandwich | `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` |
| 2. Preprocessing | `tdp.mpmct1`, `stage12.log`, `stage12.recipe`; report-only `stage2_mult.txt` (rebirth census) and, when a `security-tools` `hmap_affine` is built, `stage2_span.{tsv,txt}` |
| 3. Database mixing | `db_mixing.mpmct1`, `stage3.log` (and `db_mixing.leakage_repair.txt` with leakage repair) |
| 4. Packing | `packing/`: `deliverable.tdp4` (the shipped file), `deliverable.bin` (its raw body), `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1`, `rep512.bounds`, the last C-view scan `rep512.{exposure,allcuts,wit}.tsv` + `rep512.cview.log`, the oracle-format reports `pack_G512.txt`, `repair512.log`, `anf_rep512.txt`, `link_rep512.txt`, and `packing.report`; `stage4.log`; `packing.recipe` (keyed completion marker) |
| Run | `tdp_command.conf` (recipe manifest), `tdp_gen.log`, the private `SEEDS` |

Everything except `packing.report` and `deliverable.tdp4` is private: the
circuits regenerate or expose $C$, and `SEEDS` regenerates everything.

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
`gen_sandwich_gadget`; `circuit_mixer` runs stage 3; `tdp_packing run` runs
stage 4. Each executable's `main.rs` and command implementation live
together under `src/programs/`.

## Configuration reference

Use [configs/tdp.toml](../configs/tdp.toml) as both the default recipe and the
annotated template (config v2). Copy it to `configs/local.toml`, put your
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
recipe. A `config_version = 1` file is rejected with a migration message.

### Schema and run management

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `config_version` | integer, required `2` | TOML schema identifier, separate from the saved recipe v10. `1` is rejected with a migration message. |
| `run.directory` | string, generated directory under `runs/` | Output directory and continuation target. Omit for a fresh timestamped run; select the same directory to continue it. Within the repository it must be a dedicated subdirectory of `runs/`; dedicated external directories are also allowed. It must not overlap build or database directories. |
| `run.build_binaries` | bool, `true` | Build `gen_sandwich_gadget`, `circuit_mixer` and `tdp_packing` for a fresh run. `false` requires them under the resolved build directory. |
| `run.build_directory` | string, `CARGO_TARGET_DIR` or `target` | Cargo artifact directory; stage binaries come from its host-target `release/` directory. |
| `run.stop_after_stage` | integer, `4` | Last stage to reach: `2`, `3` or `4`. The historic stages 5-6 are rejected. |
| `run.rerun_from_stage` | integer, unset | Recompute this stage and every later reached stage, `2`, `3` or `4`. Regenerating stage 2 or 3 always reruns the later stages. |
| `run.threads` | integer, unset | Pin `RAYON_NUM_THREADS` for the stage binaries and `tdp_packing --threads`. Lifecycle only: stages 1-3 do not use the pool (only the piecewise mode, which has its own setting) and stage 4 is tested to be thread-count independent. |
| `run.adopt_unverified_run` | bool, `false` | Retired compatibility field. `true` is rejected. |

Lifecycle options (stop/rerun stage, threads, build selection) do not change
the recipe. Everything else is recorded in `tdp_command.conf`; changing it
requires a fresh run. The manifest starts with `tdp_command_recipe=10`
(recipe v9 runs are refused: resume them with the checkout that created
them); `stage12.recipe` uses `tdp_stage12_recipe=7`.

### Source circuit

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `source.wires` | integer, `128` | Logical width $n$ of $C$, in `3..4095`; at most 256 when stage 4 runs. |
| `source.gates` | integer, the validated table | $|C| = |D|$. Default: **1000 at n=64, 3000 at n=128** (the sizes the validation runs used); for any other $n$ a size or `gates_rule = "formula"` is required. Recorded as `source_gates_rule=validated` when it equals the table, `explicit` otherwise. |
| `source.gates_rule` | string, `"validated"` | `"formula"` = `round(n·log2(n)^2)` for any $n$; recorded as `formula` and printed as `UNVALIDATED SIZE`. Exclusive with `source.gates`. |
| `source.path` | string, unset | Supply $C$ as a nonempty G57 circuit instead of generating it (the route to a $C$ with more than 64 bits of entropy, §13.1 of the design). |

### Frozen replacement stores

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `database.regular_dir` | string, `FROZEN_DB_DIR` or unset | Regular frozen replacement store, required when stage 3 runs. A TOML path overrides the environment path. Needs nonempty `tables.bin` and all 256 shard files. |
| `database.lookup_miss_filter` | string, `"auto"` | `"auto"`: `FROZEN_FILTER=1` when the store has no `filters.bin` (the reference runs' situation: the mixer notes it and runs unfiltered); when `filters.bin` exists the filter stays off until its output neutrality is recorded. `"on"` requires `filters.bin`; `"off"` never loads it. |
| `database.allow_no_database_for_tests` | bool, `false` | Plumbing runs with all replacement channels disabled. |

### Preprocessing

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `preprocessing.mode` | string, `"embedded-masking"` | `"embedded-masking"` or `"nonlinear291"` (no preset, no knobs). |
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
output-neutral cache sizes `LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256
XPOLY_CANON_CACHE_MB=1024`; the canonicalization caps are unbounded, as in the
reference runs.

### Packing (stage 4)

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `packing.window_gates` | integer, `512` | Window gate cap G (`window_pack3.py` argv 4). |
| `packing.interval_cap` | integer, `512` | Interval cap W_cap (argv 3). |
| `packing.monomial_budget` | integer, `24576` | Soft window monomial budget (argv 5; product cap 8×). |
| `packing.span_cap` | integer, `3000` | Longest span considered (argv 6). |
| `packing.split_budget` | integer, `65536` | `repair.sh`'s split budget (and the final budget test). |
| `packing.fix_iterations`, `packing.fallback_iterations` | integers, `4`, `4` | `repair.sh`'s loop counts. |
| `packing.anf_cap_semantics` | string, `"cpython39"` | `"cpython39"` reproduces the oracle's CPython 3.9 set-order cap decisions; `"canonical"` is a recorded deviation. |
| `packing.allow_floor_miss`, `packing.allow_over_budget` | bools, `false` | Ship despite a visible non-output row / an over-budget window: the run finishes with `status=floor_miss`/`over_budget`, `packing/NOT_RECOMMENDED`, and no `PIPELINE COMPLETE` line. |

Any non-default value is recorded as `packing_params=custom`.

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

Fresh runs draw `gen_seed` and `mix_seed` uniformly from `[0, 2^63 − 16]`
with the OS CSPRNG, plus a 256-bit `run_key`, into the private
`$RUN/SEEDS` (mode 0600), **before** anything else is written:

```text
seeds_format=2
gen_seed=<decimal>
mix_seed=<decimal>
run_key=<64 hex>
seed_source=random|explicit
```

Gen receives `gen_seed` as all three of its seeds (C, gadgetization and
sandwich), the mixer `--seed mix_seed`, exactly as val_chain.sh does. No seed
travels on a command line between the Rust driver and bash, no seed value is
printed, and no unkeyed hash of a seed, of $C$ or of a private circuit is
written; resume markers use HMAC-SHA256 keyed by `run_key`. The manifest
records `seed_record=SEEDS`, `seed_source`, and `seed_bits=63` (a stopgap:
$C$ is generated from at most 64 bits; use `source.path` for production
deliverables until the 256-bit decision, design §13.1).

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `calibration.enabled` | bool, `false` | Mark the run calibration-only and permit an explicit seed file. |
| `calibration.seed_file` | string, unset | Format 2: a private file (no group/other permissions, at most 256 bytes) with the lines `gen_seed=<u63>` and `mix_seed=<u63>`. A one-integer (format 1) file is rejected with a migration message. On resume it must agree with `SEEDS`. |

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
| `run.directory`, `run.stop_after_stage`, `run.rerun_from_stage`, `run.threads` | `--run-directory DIR`, `--run-stop-after-stage K`, `--run-rerun-from-stage K`, `--threads T` |
| `source.wires`, `source.gates`, `source.gates_rule`, `source.path` | `--source-wires N`, `--source-gates M` (`-m`), `--source-gates-rule formula`, `--source-path P` |
| `preprocessing.mode`, `preprocessing.preset` | `--preprocessing-mode M`, `--preprocessing-preset L8\|A\|custom` |
| custom knobs | `--preprocessing-{mask-pair-wires,max-open-masks,min-open-masks,balanced-masks,shuffling-segments,shuffling-return-home,premask,seed-avoid-c1,chain,chain-len} V` |
| `db_mixing.*` | `--db-mixing-schedule custom`, `--db-mixing-growth-work-units`, `--db-mixing-hold-work-units`, `--db-mixing-target-size-factor` |
| `database.lookup_miss_filter` | `--database-lookup-miss-filter auto\|on\|off` |
| `packing.*` | `--packing-window-gates`, `--packing-interval-cap`, `--packing-monomial-budget`, `--packing-span-cap`, `--packing-split-budget`, `--packing-fix-iterations`, `--packing-fallback-iterations`, `--packing-anf-cap-semantics`, `--packing-allow-floor-miss`, `--packing-allow-over-budget` |
| `leakage_repair.*` | `--leakage-repair`, `--leakage-repair-seed S`, `--leakage-repair-reference P` |
| `calibration.seed_file` | `--seed-file PATH` (seeds on argv, `-s`, are refused) |
| `parallel.*` | `--parallel-pieces P`, `--parallel-target-piece-gates B`, `--parallel-threads T` |

Standalone, the script draws the seeds itself (`python3 -I` + `secrets`)
into `$RUN/SEEDS` under the same format and rules. The managed command never
adopts such a run directory (SEEDS without its manifest): continue it with
the script, or start a fresh managed run. It reads only
`FROZEN_DB_DIR`, `TDP_BIN_DIR` and `TDP_GEN_ALLOW_EMPTY_STORE` from its
environment. It needs bash 4 or newer.

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
    --db-total-terms 2048 --no-local-verify --seed <mix_seed>
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

## Stage 4: packing

Packing turns the mixed circuit into the deliverable. The adversary is
assumed to see the full wire state at every window boundary and the inputs,
but not the gates inside a window. The packer reorders the stage-3 circuit
along commutations so that values are born and consumed inside one window,
cuts the order into windows of at most G gates, and publishes each window as
its sparse algebraic normal form (ANF). A commutation-aware repair then
removes the remaining boundary exposures of $C$'s values, and the result is
serialized. The design and the measurements are in
[ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md) and
[STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md); the port in
[STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md).

The driver runs

```bash
tdp_packing run --stage2 tdp.mpmct1 --stage3 db_mixing.mpmct1 --source-c tdp.mpmct1.source_c.g57 \
    --n N --out-dir packing [packing flags only when not at their defaults]
```

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
deliverable is written last.

**`deliverable.tdp4`** is the shipped file: a 48-byte header (`LMTDPANF`,
version, source and state wires, layout, window count, body length, CRC-32 of
the body and of the header) followed by the unchanged `anf_dump_eval.py`
body, which is also kept as `deliverable.bin`. `tdp_packing decode
--format container` validates it.

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

**Resume.** After exit 0 the driver writes `packing.recipe`: the parameters,
the `tdp_packing` binary hash, HMAC-SHA256 tags (keyed by `run_key`) of the
three inputs and of `deliverable.tdp4`, and the status. A resume skips stage 4
only when the marker matches; rerunning stage 2 or 3, changing a packing
parameter, the binary or the deliverable reruns it.

**Speed** (Apple-silicon Mac, other jobs running): `tdp_packing run` on the
archived validation runs takes 20-45 s at n=64 (0.9-1.1 GB peak) and 75-140 s
at n=128 (1.9-2.1 GB), against about 5-6 min and 15-17 min for the Python
chain.

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
`environment.rs` read settings of the stage-3 engine and the store; the
managed driver sets or forbids each of them (see the environment policy).
Stage 4 lives in [src/stages/packing/](../src/stages/packing/) (no
environment reads) with its binary in
[src/programs/tdp_packing/](../src/programs/tdp_packing/).

Stage 3 can run on contiguous pieces in parallel; the coordinator is
[piecewise.rs](../src/engine/mixer/piecewise.rs). Correctness checks live in
[tests/](../tests/), while attacks and leakage measurements live in
[security_tests/](../security_tests/). The detailed
[current-method documentation](local_mixing_documentation.md) explains the
implemented construction and attacks. The [history](local_mixing_history.md)
preserves the motivation and earlier experiments behind this order.
