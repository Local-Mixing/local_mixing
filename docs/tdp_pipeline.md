# The TDP Pipeline

This is the operational manual for the six-stage TDP pipeline: how to run
each stage, what it writes, and where its code lives. We start with a secret
reversible circuit $C$ on $n$ wires and end with a published circuit, the
TDP, that computes $C$ on a public slice. In general $C$ may use any
reversible gate with up to two controls (NOT, CNOT, Toffoli with either
control polarity); the engine takes it in the g57 form (`a ^= x ∨ ¬y`, which
expresses that class) through `source.path`, or generates $C$ itself:
uniformly at random, or with RC's balanced construction
(`source.construction = "balanced"`: every internal value of $C$ is read
exactly twice as a control, write counts within one of each other). What
the pipeline then mixes is not $C$ itself but a specific circuit structure
built around it: the sliced sandwich on $2n$ wires (stage 1) and the
masked construction on $4n$ wires (stage 2: embedded masking, or the
three-mask compute inside the same outer blocks), to which stages 3-6 apply.

| Stage | What it does | Runs as | Output |
| --- | --- | --- | --- |
| 1. Sandwich | Classic sliced sandwich around $C$ on $2n$ wires; $C$ random, balanced, or supplied | `gen_sandwich_gadget` (with stage 2) | `tdp.mpmct1.sandwich.mpmct1`, `tdp.mpmct1.source_c.g57` |
| 2. Preprocessing | Embedded masking on $4n$ wires: masked fires, band, guards, persistent carrier migration (preset L8); or the three-mask compute (`preprocessing.mode = "three-mask"`: each value carried as $d_i$ plus three cubic band monomials, rotating band banks, dirty-helper chains, differential refresh, persistent migration) inside the same L8 outer blocks, on $2n + \max(2n, 128)$ wires | `gen_sandwich_gadget` | `tdp.mpmct1` |
| 3. Mixing | Database mixing with the regular frozen store | `circuit_mixer --tdp --db-mixing --p-db 1.0` | `db_mixing.mpmct1` |
| 4. Splitting | Every complemented g57 gate split into plain conjunctions, joined to distant brackets through absorbed NOT twists, to exhaustion | `circuit_mixer --split --split-stop` | `split.mpmct1`, `split.state` |
| 5. Crossing | The directional crossing walk under a size thermostat (target 2× its input, move budget 3× the target) | `circuit_mixer --resume split.state --target-size …` | `crossing.mpmct1` |
| 6. Delivery, adaptive mode (default) | The C-aware adaptive resynthesis: every round an every-cut exposure census of $C$'s internal values, large variable-length blocks with boundaries away from exposed cuts, every block resynthesized from its ANF alone with two seeds, the smallest candidate that is not leakier kept, until the size plateaus; the smallest round output in canonical gate order | `tdp_adaptive run` | `adaptive/final.mpmct1` (**the published circuit**, the TDP); internal: the per-round directories `adaptive/round<r>/`, `adaptive/adaptive.report` |
| 6. Delivery, windowed mode (`delivery.mode = "windowed"`) | Packing (commuting reorder into ANF windows, repair, serialization), then unpacking with lattice refinement: every window resynthesized from its ANF alone, a window that does not resynthesize cut in two and both halves repacked and unpacked, the result concatenated and canonically reordered | `tdp_packing run`, then `tdp_unpacking run --refine` | `unpacking/unpacked.mpmct1` (the published circuit in this mode); internal: `packing/deliverable.tdp4` (the packer's lattice; the published object only under `delivery.anf_only`), `unpacking/deliverable.tdp4` (the refined lattice) |

```text
C → sliced sandwich → embedded masking | three-mask → DB mixing → split → crossing walk → adaptive resynthesis
       stage 1               stage 2                   stage 3     stage 4    stage 5      stage 6 (delivery, default)
                                                                                           adaptive/round<r>/ (internal)
                                                                                           → adaptive/final.mpmct1 (the TDP, published)
     ... → crossing walk → packing → unpacking with refinement          stage 6, windowed mode (delivery.mode = "windowed")
                           packing/deliverable.tdp4 → unpacking/deliverable.tdp4 (internal) → unpacking/unpacked.mpmct1
```

**Every stage after the first is optional** (`run.skip_stages`,
`--run-skip-stages`); the default runs all six. Each stage reads the output
of the last stage that ran: with stage 2 skipped the sandwich itself goes on,
with stage 4 skipped the crossing walk starts from the previous circuit with
its own seed, with stage 3 skipped the split runs on the stage-2 circuit, and
so on (with stage 3 skipped the split has almost nothing to do: it acts on
the g57-form gates stage 3 produces, and the October 2026 campaign found the
three-mask circuit without stage 3 ANF-incompressible in stage 6). Stage 6
runs in one of two modes, `delivery.mode`: `"adaptive"` (the default, binary
`tdp_adaptive`, parameters `[adaptive]`) or `"windowed"` (recipe v12's stage
6, binaries `tdp_packing` + `tdp_unpacking`, parameters `[packing]` and
`[delivery]`); within the windowed mode `delivery.anf_only` ends the stage
after the packer's deliverable (no resynthesis; the recipient then does the
unpacking). The final artifact is the last stage's output; `PIPELINE COMPLETE
(stages 1-6): …` names it: by default the circuit `adaptive/final.mpmct1`,
the published object of the pipeline (windowed: `unpacking/unpacked.mpmct1`;
under `anf_only`, `packing/deliverable.tdp4`).

Stage 6 was two stages, 6 (packing) and 7 (unpacking), until 2026-10-12
(RC's decision: a window that does not resynthesize is a packing defect to
be split and repacked, not a fallback; [DELIVERY_STAGE.md](DELIVERY_STAGE.md)),
and the windowed delivery was its only mode until recipe v13 (2026-10-26:
the adaptive C-aware resynthesis validated by the server campaigns of
October 2026 became the default; [ADAPTIVE_STAGE.md](ADAPTIVE_STAGE.md); the
differences from the previous release: [PIPELINE_V2_CHANGES.md](PIPELINE_V2_CHANGES.md)).
A config of an earlier driver (`config_version = 3` or `4`, an `[unpacking]`
table, `run.skip_stages` with a 7) is refused with a pointer; a `config_version
= 4` recipe becomes a v5 one by setting `5` and, to keep the windowed
delivery, `delivery.mode = "windowed"`.

The managed `tdp_gen` command drives the stages. **The defaults of stages
1-3 and the packer reproduce the chosen validation runs** (`val_chain.sh`
CONFIG=L8, runs vL64a/b and vL128a/b): a managed run that sets only
`source.wires` (64 or 128), `run.skip_stages = [4, 5]` and `delivery.anf_only
= true` produces, from the same seeds, byte-identical stage-2, stage-3,
packed, repaired and deliverable artifacts. This is checked on the archived
runs themselves (the historic anchor) and by a reproduction test against the
reference script; see [docs/STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md) §2
and §12 (written when packing was stage 4; its stage numbers are the
four-stage ones). Stage 4 runs the archive-era driver's flags (`ea2e516f`); stage 5 runs its
target, penalty and temperature values with a 3×target move budget (the
archive-era driver ran 6×target); both were revived 2026-10-11, and the
unpacking sub-step dates from the same week.

The packing sub-step of stage 6 is the Rust port (`src/stages/packing/`,
binary `tdp_packing`) of the Python prototypes in
[scripts/windowed_delivery/](../scripts/windowed_delivery/), which stay as
the canonical **oracle**: the port is bit-exact wherever the oracle is
deterministic, and CI compares the two on random inputs on every push.

Everything described here is on `main` (2026-10-12). Retired components and
the earlier pipelines are not described here; history:
[local_mixing_history.md](local_mixing_history.md).

Further reading:

- [TDP_PIPELINE_REFERENCE.md](TDP_PIPELINE_REFERENCE.md): the full
  reference for all six stages (construction details, parameters, measured
  behaviour, rationale, pass criteria, open issues; revised 2026-10-12 for
  the six-stage pipeline; formerly `TDP_FOUR_STAGE_PIPELINE.md`).
- [STAGE4_RUST_DESIGN.md](STAGE4_RUST_DESIGN.md): the packing port, the
  reference recipe the defaults reproduce, seeds and the environment policy,
  and the parity results on the archived runs.
- [ADAPTIVE_STAGE.md](ADAPTIVE_STAGE.md): stage 6 in its adaptive mode (the
  census, the planner, the acceptance rule, the stop rules, the checks, the
  outputs and the report keys).
- [THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md): the three-mask stage 2 (the
  construction, its knobs, the parity with the campaign generator).
- [PIPELINE_V2_CHANGES.md](PIPELINE_V2_CHANGES.md): what recipe v13 changed
  against the previous release, setting by setting.
- [PIPELINE_INTRODUCTION.md](PIPELINE_INTRODUCTION.md): the pipeline from
  scratch, for a first-time reader.
- [DELIVERY_STAGE.md](DELIVERY_STAGE.md): the windowed delivery (the decision,
  the refinement contract, the Python reference and the Rust binaries).
- [STAGE7_UNPACKING.md](STAGE7_UNPACKING.md): the unpacking sub-step of
  stage 6 (`tdp_unpacking`: its synthesizer, its checks, its exit codes and
  its measured results; written when it was stage 7).
- [STAGE2_LINKABILITY.md](STAGE2_LINKABILITY.md): the stage-2 redesign and
  the linkable-exposure metrics that the recommended configuration is chosen
  by.
- [ANF_WINDOW_DELIVERY.md](ANF_WINDOW_DELIVERY.md): windowed ANF packing, its
  threat model (the `anf_only` one: the adversary holds the window ANFs), and
  the deliverable format.
- [notes/stubborn_windows_20261012.md](notes/stubborn_windows_20261012.md):
  the measured basis of the refinement policy (157 of 10,366 windows failed
  at n=64; they are ANF-dense tails, not a gate-structure class; 10× time
  rescues 15/102; bisection converges ~90 % per round and terminates; the
  complete bisection assembly has 0 interior C values at its own cuts, the
  seven-stage output with its 157 verbatim windows 5 of 936, the raw packed
  circuit 0).
- [notes/stage45_revival_20261011.md](notes/stage45_revival_20261011.md):
  stages 4 and 5 measured (the crossing output has no C value at any cut,
  but merged-window resynthesis at W=2000 still exposes 244/936 interior
  values; stages 4+5 do not close the leak).
- [notes/seven_stage_20261011.md](notes/seven_stage_20261011.md): the n=16
  and n=64 run tables of the seven-stage session.

For first-time setup and the circuit utilities, start with the
[README](../README.md).

## Two ways to run

**The managed driver** (`tdp_gen`) is the supported entry point. It reads a
TOML recipe (config v5), builds and fingerprints the five stage binaries,
draws the seeds into the private `$RUN/SEEDS`, records a manifest (recipe
v13), and resumes runs safely. Its defaults are the recommended
configuration: a random $C$, stage-2 preset `L8`, stage-3 schedule
`validated`, stages 4 and 5 on, and the adaptive stage 6 run to its plateau
(two synthesizer seeds per block, 60/120/240 s per block by size, the 2 %
two-round stop rule); the windowed mode keeps its defaults (the packer at G =
512 with a 65,536-monomial split budget, the refinement at 30 s per window
per round for up to 16 rounds).

**The reference driver**
[scripts/windowed_delivery/waist/val_chain.sh](../scripts/windowed_delivery/waist/val_chain.sh)
is the script the validation runs were made with. It is now a **parity
tool**: it hard-codes the run tree's paths, draws seeds below $2^{31}$ /
$2^{40}$ (calibration grade) and runs the Python packer (today's windowed
stage 6 under `delivery.anf_only`, with stages 4 and 5 skipped). The reproduction
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

The run ends with `[tdp_gen] PIPELINE COMPLETE (stages 1-6): <run>/adaptive/final.mpmct1`,
the published circuit (windowed mode: `<run>/unpacking/unpacked.mpmct1`;
under `delivery.anf_only`: `<run>/packing/deliverable.tdp4`). A run
directory holds:

| Stage | Files |
| --- | --- |
| 1. Sliced sandwich | `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` |
| 2. Preprocessing | `tdp.mpmct1`, `stage12.log`, `stage12.recipe` (the keyed stage-1/2 marker: mode, preset or three-mask knobs, source construction, dimensions, seeds, binary); report-only `stage2_mult.txt` (rebirth census) and `stage2_span.{tsv,txt}` (val_chain.sh's span check; the managed driver builds the `security-tools` `hmap_affine` it needs, a standalone run skips it unless `$TDP_BIN_DIR/hmap_affine` exists) |
| 3. Database mixing | `db_mixing.mpmct1` (the mixer writes `db_mixing.mpmct1.tmp`, fsyncs it and renames it, so a kill or crash never leaves a truncated output under this name), `stage3.log`, `stage3.settings` (the resolved `FROZEN_FILTER` and the mixer's `stop_reason`), `stage3.recipe` (keyed completion marker: the schedule arguments, HMAC-SHA256 tags of `tdp.mpmct1` and of `db_mixing.mpmct1`, the stop reason; written only after an accepted stop) (and `db_mixing.leakage_repair.txt` with leakage repair). A resume skips stage 3 only when `stage3.recipe` matches; a missing or stale marker (an interrupted output, a changed stage-2 input or schedule) re-runs stage 3, which then needs the store. A run made before the marker existed is adopted without re-mixing when its `delivery.recipe` (or the seven-stage driver's `packing.recipe`) carries the tags of exactly these `tdp.mpmct1` and `db_mixing.mpmct1` bytes and `stage3.settings` records an accepted stop. A chain ended by the operator stop flag (`touch $RUN/stage3.stop`) is under-dosed: its output is moved to `db_mixing.stopped.mpmct1`, the run stops with exit 2 and no deliverable, and a resume re-runs stage 3 from scratch |
| 4. Splitting | `split.mpmct1`, `split.state` (the resume point of stage 5), `stage4.log`, `stage4.recipe` (keyed completion marker); a split ended by `$RUN/stage4.stop` is kept as `split.stopped.mpmct1` |
| 5. Crossing | `crossing.mpmct1`, `stage5.log`, `stage5.recipe` |
| 6. Delivery (adaptive mode) | `adaptive/`: `final.mpmct1` (**the published circuit**: the smallest round output in canonical gate order), `adaptive.report` (key=value: status, rounds, stop reason, gate counts per round, the exposure of the published circuit, every check), `adaptive.log`, `boundary.txt` (the classification of $C$'s births: public quadratic boundary values versus deeper ones), `progress.txt`, and one directory per round `round<r>/` (the input census, the partition, the per-seed candidates' censuses, `output.mpmct1` and `result.report`; a completed round is reloaded on resume, not recomputed); `stage6.log`; `delivery.recipe` (keyed completion marker: `delivery_recipe=2`, `delivery_mode=adaptive`, the binary's hash, the non-default parameters, the keyed tags of the inputs and of `final.mpmct1`, the status, rounds and stop reason); an interrupted attempt leaves `adaptive/attempt.recipe` and resumes in place when the recipe is unchanged; a failed run leaves `delivery.failed` (`exit=<code>`, `step=adaptive`) |
| 6. Delivery (windowed mode) | `packing/`: `deliverable.tdp4` (the packer's lattice; the published object only under `anf_only`), `deliverable.bin` (its raw body), `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1`, `rep512.bounds` (the repaired circuit and the lattice the refinement starts from), the last C-view scan `rep512.{exposure,allcuts,wit}.tsv` + `rep512.cview.log`, the oracle-format reports `pack_G512.txt`, `repair512.log`, `anf_rep512.txt`, `link_rep512.txt`, and `packing.report`; `unpacking/` (absent under `anf_only`): `unpacked.mpmct1` (**the published circuit**: the resynthesized windows, canonical order), `deliverable.tdp4` (the refined lattice: the internal ANF form the circuit was resynthesized from, kept for auditing and parity), `bounds.final`, `windows.tsv`, `rounds.tsv`, `unpacking.report`, `unpacking.log`, the per-round `bounds.<r>` and `deliverable.<r>.tdp4` (round 0 = the packer's; internal: they carry the lattice and the window ANFs); `stage6.log` (both sub-steps, a separator line between them); `delivery.recipe` (keyed completion marker); a failed sub-step leaves `delivery.failed` (`exit=<code>`, `step=packing\|unpacking`; the managed driver then reports that code's meaning for that binary, and no other failure is labelled as stage 6's); `NOT_RECOMMENDED` next to the final output after an allowed failure |
| Run | `tdp_command.conf` (recipe manifest), `tdp_gen.log`, the private `SEEDS` |

Only the published circuit ships: `adaptive/final.mpmct1` (the smallest
round's assembly in canonical gate order; in the windowed mode
`unpacking/unpacked.mpmct1`, the canonical-order concatenation of the
resynthesized windows; under `anf_only`, `packing/deliverable.tdp4`: the
window ANFs, which the recipient unpacks itself). Everything else is secret
or a secret intermediate: the per-round outputs, censuses and partitions of
the adaptive mode (they locate $C$'s values), the two `deliverable.tdp4` and
the per-round containers and bounds of the windowed mode (the packer's
windowed form that the unpacking sub-step resynthesizes from, kept for
auditing and parity; they carry the lattice and the window ANFs), the packed
and repaired circuits and the stage 1-5 circuits (they regenerate or expose
$C$), `SEEDS` (it regenerates everything), the logs and the reports. The
reports hold only counts and parameters, but they are not part of the TDP.

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
`tdp_adaptive run` (adaptive mode) or `tdp_packing run` and then
`tdp_unpacking run --refine` (windowed mode). Each executable's `main.rs` and
command implementation live together under `src/programs/`.

## Configuration reference

Use [configs/tdp.toml](../configs/tdp.toml) as both the default recipe and the
annotated template (config v5). Copy it to `configs/local.toml`, put your
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

**Presets, the schedule and the stage-6 mode are authoritative.** Under
`preprocessing.preset = "L8"` or `"A"` every stage-2 knob field is an error,
even when it equals the preset's value; `[three_mask]` keys are errors unless
`preprocessing.mode = "three-mask"`; under `db_mixing.schedule =
"validated"` the schedule fields are errors; under `delivery.mode =
"adaptive"` every `[packing]` key and every windowed `[delivery]` key is an
error, and under `"windowed"` every `[adaptive]` key is. An old
configuration therefore cannot silently produce a hybrid recipe. A
`config_version = 1`, `2`, `3` or `4` file is rejected with a migration
message (`2` was the four-stage schema; `3` the seven-stage schema with
separate packing and unpacking stages; `4` the six-stage schema whose only
stage 6 was the windowed delivery: set `5`, and set `delivery.mode =
"windowed"` to keep that delivery; the four-stage chain is `run.skip_stages
= [4, 5]`, `delivery.mode = "windowed"`, `delivery.anf_only = true`). An
`[unpacking]` table is refused with the same pointer.

### Schema and run management

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `config_version` | integer, required `5` | TOML schema identifier, separate from the saved recipe v13. `1`, `2`, `3` and `4` are rejected with migration messages. |
| `run.directory` | string, generated directory under `runs/` | Output directory and continuation target. Omit for a fresh timestamped run; select the same directory to continue it. Within the repository it must be a dedicated subdirectory of `runs/`; dedicated external directories are also allowed. It must not overlap build or database directories. |
| `run.build_binaries` | bool, `true` | Build `gen_sandwich_gadget`, `circuit_mixer`, `tdp_adaptive`, `tdp_packing` and `tdp_unpacking` for a fresh run, plus `hmap_affine` (`--features security-tools`, a separate cargo build whose failure also aborts the run) for the report-only stage-2 span check. `false` requires them under the resolved build directory. |
| `run.build_directory` | string, `CARGO_TARGET_DIR` or `target` | Cargo artifact directory; stage binaries come from its host-target `release/` directory. |
| `run.skip_stages` | integer array, `[]` | Stages not run, each in `2..6` (stage 1 always runs). Recorded in the manifest (`skip_stages=`): it changes the output. `[4, 5]` with `delivery.mode = "windowed"` and `delivery.anf_only = true` is the validated four-stage chain. |
| `run.stop_after_stage` | integer, `6` | Last stage to reach, `2..6`. |
| `run.rerun_from_stage` | integer, unset | Recompute this stage and every later reached stage, `2..6`. Regenerating a stage always reruns the later stages. |
| `run.threads` | integer, unset | Pin `RAYON_NUM_THREADS` for the stage binaries and the `--threads` of `tdp_adaptive`, `tdp_packing` and `tdp_unpacking`. Lifecycle only: stages 1-5 do not use the pool (only the piecewise mode, which has its own setting), the packer is tested to be thread-count independent, and the synthesizers' output depends on threads only through their per-block or per-window deadline. |
| `run.adopt_unverified_run` | bool, `false` | Retired compatibility field. `true` is rejected. |

Lifecycle options (stop/rerun stage, threads, build selection) do not change
the recipe. Everything else, `run.skip_stages` included, is recorded in
`tdp_command.conf`; changing it requires a fresh run. The manifest starts
with `tdp_command_recipe=13` (recipe v12 runs, the previous six-stage
driver's, v11 runs, the seven-stage driver's, v10 runs, the four-stage
driver's, and v9 runs are refused: resume them with the checkout that
created them); `stage12.recipe` uses `tdp_stage12_recipe=8`.

The manifest's `pinned_recipe` names the validated chain the run reproduces,
`val_chain-L8-20261007` or `val_chain-A-20261007`, and is `none` for anything
else. It requires every recipe setting of that chain (a random generated
source, embedded masking with preset L8 or A, the validated |C|, schedule
`validated` in one piece, no leakage repair, default packing, stages 4 and 5
off and the windowed stage 6 ending after the packer's deliverable:
`run.skip_stages = [4, 5]`, `delivery.mode = "windowed"` and
`delivery.anf_only = true`; the then-unused unpacking parameters do not
matter) **and** stage 3
on the store the chosen runs used, unfiltered: `frozen_db_fingerprint`
(the name and size of every file in the store plus the text of each `.ok`
checksum marker, xxh3-128; the shards themselves are not read) must equal
that of `frozen_m1_m11`, and `FROZEN_FILTER=1` on a store without
`filters.bin`. Any other store (other shards or markers) or a filtered run
(`lookup_miss_filter = "on"`) records `none`.

### Source circuit

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `source.wires` | integer, `128` | Logical width $n$ of $C$, in `3..4095`; at most 256 when stage 6 runs. |
| `source.gates` | integer, the validated table | $|C| = |D|$. Default: **1000 at n=64, 3000 at n=128** (the sizes the validation runs used); for any other $n$ a size or `gates_rule = "formula"` is required. Recorded as `source_gates_rule=validated` when it equals the table, `explicit` otherwise. |
| `source.gates_rule` | string, `"validated"` | `"formula"` = `round(n·log2(n)^2)` for any $n$; recorded as `formula` and printed as `UNVALIDATED SIZE`. Exclusive with `source.gates`. |
| `source.path` | string, unset | Supply $C$ as a nonempty circuit in the g57 file form instead of generating it (any reversible gates with up to two controls, written as `a ^= x ∨ ¬y`; the route to a $C$ with more than 64 bits of entropy, §13.1 of the design). Exclusive with the balanced construction. |
| `source.construction` | string, `"random"` | How a generated $C$ is drawn from the gen seed: `"random"` (uniformly random g57 gates, `random_circuit`) or `"balanced"` (RC's balanced construction, `stages::sandwich::balanced_source`: every internal value of $C$ is read exactly twice as a control, no value more than twice, targets always among the least-written wires; a constrained random sampler with rejection of immediate dead ends, bit-exact with the Python reference that built the campaign's balanced $C$, audited before use). Exposure of $C$'s values through the pipeline is per control read, so the balanced $C$ redistributes it: no value is read-free, none is read more than twice ([notes/c_design_exposure_20261021.md](notes/c_design_exposure_20261021.md)). Recorded in the manifest and in `stage12.recipe`; reaches gen as `TDP_SOURCE_CONSTRUCTION`. |

### Frozen replacement stores

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `database.regular_dir` | string, `FROZEN_DB_DIR` or unset | Regular frozen replacement store, required when stage 3 runs. A TOML path overrides the environment path. Needs nonempty `tables.bin` and all 256 shard files. |
| `database.lookup_miss_filter` | string, `"auto"` | `"auto"`: `FROZEN_FILTER=1` when the store has no `filters.bin` (the reference runs' situation: the mixer notes it and runs unfiltered); when `filters.bin` exists the filter stays off until its output neutrality is recorded. `"on"` requires `filters.bin`; `"off"` never loads it. The manifest records the mode with the value it resolves to (`frozen_filter=auto(1)`), `$RUN/stage3.settings` the value stage 3 set. |
| `database.allow_no_database_for_tests` | bool, `false` | Plumbing runs with all replacement channels disabled. |

### Preprocessing

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `preprocessing.mode` | string, `"embedded-masking"` | `"embedded-masking"`, `"three-mask"` (the three-mask compute inside the embedded-masking L8 outer blocks; no preset, knobs in `[three_mask]`; [THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md)) or `"nonlinear291"` (no preset, no knobs). nonlinear291 circuits are wide (24n + 12·(2\|C\| + 2s + n) + 29 wires: 3,677 at n=16, \|C\|=64), and stage 6 takes at most 1,024 state wires, so the driver refuses nonlinear291 with stage 6 up front unless the planned width fits; stop before stage 6 or skip it. |
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

### Three-mask (stage 2, `preprocessing.mode = "three-mask"`)

The three-mask compute replaces the masked compute of embedded masking
between the same outer blocks (opening guard, band seed, …, band reseed,
closing guard, always the L8 parameters): every sandwich value is carried
as $x_i = d_i \oplus m_1 \oplus m_2 \oplus m_3$ with three cubic monomials of
band wires, the band is divided into banks that rotate through serving,
retiring and growing phases, every source gate fires through a dirty-helper
chain over the band, masks are refreshed differentially, and the carriers
migrate persistently ([THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md)). The
defaults below are the campaign configuration "three-mask modified / v2"
(random helpers, unique cover); any key is an error under another mode.

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `three_mask.band_wires` | integer, `0` | Band width; `0` = auto: the larger of $2n$ and 128, rounded up to a multiple of 16 (a 64-wire band cannot host 64 data roles with three masks each). Else at least 64 and a multiple of 16. When it exceeds $2n$ the extra wires get a reversible XOR ingress from the seeded $2n$-wire prefix. |
| `three_mask.banks` | integer, `8` | Band banks, even, at least 4; the bank size `band_wires / banks` must be even and at least 8. |
| `three_mask.retire_budget` | integer, `2` | Mask replacements per fire that drain retiring banks. |
| `three_mask.max_retiring` | integer, `0` | Banks retiring at once; `0` = auto (half the banks minus one), else below half the banks. |
| `three_mask.rotation`, `three_mask.maturity` | integers, `1`, `2` | Promotion check interval (fires; 0 = never) and the growth sweeps a bank needs before promotion. |
| `three_mask.visit_sweeps`, `three_mask.growth`, `three_mask.warmup` | integers, `2`, `8`, `4` | Growth sweeps per bank visit, growth gates per fire, warm-up sweeps before the first fire. |
| `three_mask.segments` | integer, `6` | Persistent carrier migration segments per fire box (0 = off). |
| `three_mask.mask_count` | integer, `3` | Persistent masks per value, `1..3`. |
| `three_mask.random_helpers`, `three_mask.unique_cover` | bools, `true`, `true` | The "modified / v2" levers: dirty helpers drawn from a shuffled pool of every reference-free band wire (instead of the first four wires of the lowest growing bank), and a temporary cover mask whose cubic label is never a live persistent label. `false` gives the generator's "as-is" variant. |

Every effective value is recorded in the manifest (`three_mask_*`,
`three_mask_outer_blocks=L8`) and in `stage12.recipe`; a knob reaches gen as
`THREE_MASK_*` only when it differs from its default. Measured at n=64 on the
server (October 2026): 504k gates against 777k for embedded masking L8, the
outer blocks identical; the gen output is bit-identical to the campaign
generator's for the same seed (`tests/integration/three_mask_parity.rs`).

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
| `crossing.move_budget_factor` | number, `3.0` | Move budget `round(factor * target)`, factor in `(0, 1000]`. At 2×target the n=64 walk stopped at 0.90 of its target (5.38M of 5.98M gates, 7 min; `docs/notes/seven_stage_20261011.md` §4) instead of settling on it; the archive-era driver ran `6 * target` (46 min, on target). RC set the default to 2 on 2026-10-11 and to 3 on 2026-10-12; a run at 3× is not yet measured. |
| `crossing.move_attempts` | integer, unset | An absolute move budget instead of the factor (the two are exclusive), in `1..10^12`. |

Either budget is added to the split stage's move counter on a resume of
`split.state`, because the mixer's resumed `--moves` ceiling is absolute.
The target, penalty and temperature defaults are the archive-era driver's
calibrated values; their promotion status was never resolved (the revival
measurements: `docs/notes/stage45_revival_20261011.md`; history:
[local_mixing_history.md](local_mixing_history.md)).

### Delivery mode (stage 6)

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `delivery.mode` | string, `"adaptive"` | `"adaptive"`: `tdp_adaptive run`, parameters `[adaptive]`; every `[packing]` key and every other `[delivery]` key is an error. `"windowed"`: `tdp_packing run` then `tdp_unpacking run --refine`, parameters `[packing]` and `[delivery]`; every `[adaptive]` key is an error. Recorded as `delivery_mode=` in the manifest; the stage-6 marker names the mode (`delivery_recipe=2` adaptive, `1` windowed). |

### Adaptive resynthesis (stage 6, the default mode)

The adaptive stage runs rounds of C-aware resynthesis on the stage-5
circuit until its size plateaus ([ADAPTIVE_STAGE.md](ADAPTIVE_STAGE.md)):
each round measures where $C$'s internal values are directly or affinely
recoverable from the live wires at every cut (the census, on `census_batches`
× 64 honest inputs, every entry witness re-tested on fresh and structured
inputs), partitions the circuit into large blocks whose boundaries avoid
those cuts, resynthesizes every block from its ANF alone with
`seeds_per_block` seeds, splits the blocks that do not synthesize, keeps per
block the smallest candidate that exposes no more deeper values of $C$ than
the original, and checks the assembly. The defaults are the policy of the
server campaigns of October 2026 (variant "v2", 2 % two-round stop rule).

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `adaptive.max_rounds` | integer, `0` | Rounds at most, `0..1000`; `0` = until the plateau. A capped run reports `stop_reason=capped`; its published circuit is still the smallest round output. |
| `adaptive.stop_fraction` | number, `0.02` | End of mixing when the last two rounds compress jointly by less than this fraction (`output(r) >= (1 - f) * input(r-1)`), in `[0, 1)`; two consecutive rounds without a decrease also end it. |
| `adaptive.seeds_per_block` | integer, `2` | Synthesizer seeds per block, `1..16`; every seed's converged, strictly smaller synthesis is a candidate. |
| `adaptive.time_limit` | number, `60` | Synthesis wall limit per block, seconds, for blocks of at most 1,000 gates; ×2 up to 4,000 gates, ×4 above (`(0, 86400]`). |
| `adaptive.restarts`, `adaptive.beam` | integers, `16`, `6` | The synthesizer's restarts and lookahead width (`1..1000`). |
| `adaptive.term_cap` | integer, `131072` | ANF terms at an eligible block endpoint (`64..2^24`); composition may peak at 8× while scanning. |
| `adaptive.max_block_gates` | integer, `65536` | Longest block (`2..10^8`). |
| `adaptive.segments` | integer, `16` | Planner segments, planned in parallel with seams at unexposed cuts (`1..1024`; 1 = the sequential planner). |
| `adaptive.avoid_radius` | integer, `64` | Cuts within this distance of the previous round's boundaries are penalized (`0..10^6`; 0 = off), so every former boundary falls inside a block at least every other round. |
| `adaptive.refinement_levels`, `adaptive.min_refine_gates` | integers, `2`, `128` | Split levels for a block that synthesizes under no seed (at an exposure-aware cut in its middle half), and the shortest block that is split. |
| `adaptive.census_batches` | integer, `96` | Census discovery inputs = 64 × batches (`1..4096`). |
| `adaptive.accept` | string, `"smaller-not-leakier"` | Per-block acceptance: `"smaller-not-leakier"` (the candidate assemblies are censused; a replacement must be strictly smaller and expose no more deeper values of $C$ inside the block than the original) or `"smaller"` (size only). |
| `adaptive.canonical_order` | bool, `true` | Publish the smallest round output in the canonical commuting gate order (checked equivalent); `false` publishes the assembly as is. |

Any non-default value is recorded as `adaptive_params=custom`; a knob
reaches `tdp_adaptive` only when it differs from its default. Resource
guidance: the census and the planner hold the whole circuit; the synthesis
runs one block per thread, each bounded by the term cap; `run.threads`
pins the pool.

### Packing (stage 6, windowed mode, the first sub-step)

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

### Delivery (stage 6, windowed mode, the unpacking sub-step)

| Setting | Type and default | What it changes and constraints |
| --- | --- | --- |
| `delivery.anf_only` | bool, `false` | `true` ends stage 6 after the packer's deliverable: no resynthesis, no refinement, no `unpacking/`; `tdp_unpacking` is not needed. `packing/deliverable.tdp4` is then the published object (the recipient does the unpacking, and the adversary sees the window ANFs: the four-stage pipeline's threat model). The four-stage validation chain. |
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
RNG); stage 6 `unpack_seed` (`tdp_adaptive run` derives every round's and
seed's synthesizer seed from it; `tdp_unpacking run --refine` uses it as its
seed; the fifth seed keeps its name from the seven-stage driver, so format-3
`SEEDS` files resume unchanged). Unlike val_chain.sh, **no seed is on any
command line**: val_chain.sh puts them on gen's and the mixer's argv, where
`ps` shows them to every local user while the process runs (the gen seed
alone regenerates $C$ and the stage-2 artifact). The driver hands them over
on inherited file descriptors instead (gen `@/dev/fd/3 @/dev/fd/4
@/dev/fd/5`, the mixer, `tdp_adaptive` and `tdp_unpacking` `--seed-file
/dev/fd/3`, each fed by a bash here-string, so nothing is written to disk), and the binaries
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
| recipe | `EMBEDDED_MASKING_SHUFFLING` … `CHAIN_LEN`, `K`, `MAX_OPEN`, `MIN_OPEN`, `BALANCED`; `THREE_MASK_BAND_WIRES` … `UNIQUE_COVER` | fatal (the preset / the `[three_mask]` table sets them) |
| driver | `FROZEN_FILTER`, `FROZEN_REGULAR_VALUE_CONVENTION`, `SANDWICH_VARIANT`, `TDP_SOURCE_C`, `TDP_SOURCE_CONSTRUCTION`, `MIXER_STOP_FLAG` | fatal (use the config field / flag) |
| input | `FROZEN_DB_DIR`, `TDP_BIN_DIR`, `TDP_GEN_ALLOW_EMPTY_STORE` | accepted |
| neutral | `LOOKUP_CACHE_MB`, `CANON_CACHE_MB`, `XPOLY_CANON_CACHE_MB` | ignored with a note |
| forbidden / diagnostic | every other `EMBEDDED_MASKING_*` and `THREE_MASK_*`, `SLICE_GUARD_LEGACY`, `MIXER_DB_LEN_BAND`, `MIXER_STOP_AT_*`, `TWIST_G57_*`, `MIN_DIR_LOOKUP`, `CANON_*_CAP`, `PROD_*`, `SAT_*`, `DB_QC*`, `BV5_*`, `FROZEN_CURATED_*` | fatal: `X is not honoured by the managed recipe; unset it (use flags/config)` |

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
| `source.wires`, `source.gates`, `source.gates_rule`, `source.path`, `source.construction` | `--source-wires N`, `--source-gates M` (`-m`), `--source-gates-rule formula`, `--source-path P`, `--source-construction random\|balanced` |
| `preprocessing.mode`, `preprocessing.preset` | `--preprocessing-mode M`, `--preprocessing-preset L8\|A\|custom` |
| custom knobs | `--preprocessing-{mask-pair-wires,max-open-masks,min-open-masks,balanced-masks,shuffling-segments,shuffling-return-home,premask,seed-avoid-c1,chain,chain-len} V` |
| `three_mask.*` | `--three-mask-{band-wires,banks,retire-budget,max-retiring,rotation,maturity,visit-sweeps,growth,warmup,segments,mask-count,random-helpers,unique-cover} V` |
| `db_mixing.*` | `--db-mixing-schedule custom`, `--db-mixing-growth-work-units`, `--db-mixing-hold-work-units`, `--db-mixing-target-size-factor` |
| `database.lookup_miss_filter` | `--database-lookup-miss-filter auto\|on\|off` |
| `crossing.*` | `--crossing-target-size-factor R`, `--crossing-width-penalty-base B`, `--crossing-width-penalty-threshold C`, `--crossing-size-tolerance-divisor D`, `--crossing-move-budget-factor F`, `--crossing-move-attempts M` |
| `packing.*` | `--packing-window-gates`, `--packing-interval-cap`, `--packing-monomial-budget`, `--packing-span-cap`, `--packing-split-budget`, `--packing-fix-iterations`, `--packing-fallback-iterations`, `--packing-anf-cap-semantics`, `--packing-allow-floor-miss`, `--packing-allow-over-budget` |
| `delivery.mode` | `--delivery-mode adaptive\|windowed` |
| `adaptive.*` | `--adaptive-max-rounds N`, `--adaptive-stop-fraction F`, `--adaptive-seeds-per-block K`, `--adaptive-time-limit S`, `--adaptive-restarts R`, `--adaptive-beam B`, `--adaptive-term-cap T`, `--adaptive-max-block-gates G`, `--adaptive-segments S`, `--adaptive-avoid-radius D`, `--adaptive-refinement-levels L`, `--adaptive-min-refine-gates G`, `--adaptive-census-batches B`, `--adaptive-accept A`, `--adaptive-no-canonical-order` |
| `delivery.*` (windowed) | `--delivery-anf-only`, `--delivery-time-limit S`, `--delivery-restarts R`, `--delivery-beam B`, `--delivery-max-rounds N` (the seven-stage driver's `--unpacking-*` flags are refused with a pointer) |
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

$C$ itself is supplied (`source.path`) or generated from the gen seed:
uniformly random g57 gates, or RC's balanced construction
(`source.construction = "balanced"`, [balanced.rs](../src/stages/sandwich/balanced.rs):
a sequential sampler that gives every completed internal value exactly two
control reads, never more, with the targets always among the least-written
wires; it rejects choices that would leave no legal target for the next gate
and audits the result). Read
[src/stages/sandwich/construct.rs](../src/stages/sandwich/construct.rs),
starting with `prepare_source`, `construct_seeded_sandwich`, and
`sliced_sandwich_cnot`.

## Stage 2: preprocessing (embedded masking or three-mask)

We now hide each sandwich value behind a nonlinear mask. The default is
embedded masking: one carrier and one band wire per sandwich value, taking
us from $2n$ to $4n$ physical wires. The construction surrounds the masked
computation with slice guards and input-derived band seed/reseed blocks.
The three-mask mode (`preprocessing.mode = "three-mask"`) keeps those outer
blocks and replaces the masked computation: every value is carried as its
data bit plus three cubic monomials of band wires, the band is organized in
banks that rotate through serving, retiring and growing phases (a growing
bank is rewritten by random Toffolis until it is promoted; a retiring bank
drains as its masks are replaced), every source gate fires through a
dirty-helper chain over the band, and the carriers migrate persistently; at
n=64 the compute is 35 % smaller than embedded masking's. The construction
is in [src/stages/preprocessing/three_mask/](../src/stages/preprocessing/three_mask/)
and described in [THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md).

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
The optional post-stage-3 leakage audit and DB repair (`[leakage_repair]`,
off by default; `db_mixing.leakage_repair.txt`) runs here too; it is
separate from the packer's order repair, which every stage-6 run performs.
The output is `db_mixing.mpmct1` with `stage3.log`.

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
nothing downstream resumes it. With the archive-era 6×target budget the n=64 walk took 3.03M gates to
5.49M in 46 min and the n=128 walk 8.66M to 15.7M in 6.3 h
(`docs/notes/stage45_revival_20261011.md`); with 2×target the n=64 walk
stopped at 5.38M after 7 min (`docs/notes/seven_stage_20261011.md`); the 3×
default is not yet measured. The walk slows several-fold under CPU
contention. Read `cross_move_on`, `undo_move` and `merge_move` in
[src/stages/post-processing/crossing.rs](../src/stages/post-processing/crossing.rs);
the scheduler and acceptance logic are in
[src/engine/mixer/scheduling.rs](../src/engine/mixer/scheduling.rs). Within
the mixer the walk is slot 3 of `Mixer::run`, the default walk when `p_db <
1`; stage 3 passes `--p-db 1.0`, so slot 3 never runs there and stage-3
output is unchanged by its presence.

<a id="stage-4-packing"></a>
<a id="stage-6-delivery"></a>

## Stage 6: delivery

Delivery turns the mixed circuit into the published circuit, the TDP, in one
of two modes. The **adaptive mode** (the default, `delivery.mode =
"adaptive"`, binary `tdp_adaptive`) resynthesizes the stage-5 circuit in
large, C-aware blocks, round after round, until its size plateaus; the
**windowed mode** (`"windowed"`, recipe v12's stage 6) packs the circuit
into ANF windows and unpacks them with lattice refinement. Both publish one
circuit and nothing else.

### Adaptive resynthesis (the default mode)

The driver runs, on the last circuit produced (`crossing.mpmct1` by default),

```bash
tdp_adaptive run --input crossing.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n N \
    --sandwich tdp.mpmct1.sandwich.mpmct1 --out-dir adaptive --seed-file /dev/fd/3 \
    [adaptive knobs only when not at their defaults] [--threads T]
```

Every round (docs/ADAPTIVE_STAGE.md; the server prototype `stage6_run.py`
variant v2, ported): the **census** scans every cut of the round's input on
the honest slice and records every interval of cuts at which an internal
value of $C$ (a source birth) equals a live wire up to complement (direct) or
an XOR of simultaneously live wires plus a constant (affine), with the
locations that witness it; every entry witness is re-tested on fresh random
and structured inputs. The **planner** grows blocks by exact ANF composition
(at most `term_cap` terms at an endpoint, at most `max_block_gates` gates,
256 gates of lookahead past the last eligible endpoint) and places each
boundary in the upper half of the feasible interval at the cut with the
fewest exposed deeper values of $C$ (then fewest exposed values of any kind,
away from the previous round's boundaries, largest clearance from exposure
intervals, fewest direct copies, latest), 16 segments in parallel with seams
at unexposed cuts. "Deeper" excludes the births of $C$ that are polynomials
of degree at most 2 in the public inputs or outputs (`boundary.txt`, a
sampled classification). Every block is **resynthesized** from its ANF alone
by the unpacking synthesizer (the port of `anf_synth`) with two seeds under a
size-scaled budget; a block that converges under no seed is cut in two at an
exposure-aware cut in its middle half, twice. Per block, the candidate
assemblies are censused and the smallest converged candidate that exposes no
more deeper values inside the block than the original is **kept** (a growing
or leakier synthesis is rejected, the original block stays). The assembly is
checked equivalent to the round's input on random full states and censused;
`round<r>/output.mpmct1` and `result.report` make the round resumable. The
run ends after two consecutive rounds without a decrease, after two
consecutive rounds whose joint compression is below `stop_fraction`, or after
`max_rounds`; the smallest round output is put into the canonical commuting
gate order (checked equivalent), checked on the zero slice against $C$ and
against the sandwich's payload contract in both directions, censused once
more, and written as `adaptive/final.mpmct1` with `adaptive.report`.

**What the adversary sees** in this mode: `adaptive/final.mpmct1`, nothing
else; not the rounds, the censuses, the partitions or the candidates. Measured
on the server campaigns of October 2026 at n=64 (three-mask stage 2, all
stages, adaptive v2 to its plateau): the published circuit exposes 0 of 936
internal values of $C$ directly or affinely at any cut (the embedded-masking
baseline under the same stage 6: 384 direct / 424 affine of 936); a C-blind
resynthesis of the published circuit exposes 6, and narrow 125-250-gate
rewrites still find 100-160 repeated deeper values
([notes/stage2_stage6_campaign_20261021.md](notes/stage2_stage6_campaign_20261021.md)).

**Exit codes** of `tdp_adaptive` (the driver prints them with their meaning
and `step=adaptive` in `delivery.failed`): 2 usage or parameter error, 3
input error, 4 verification or synthesis failure, 7 I/O error writing
outputs, 8 internal invariant violated, 101 panic. The completion marker
`delivery.recipe` holds `delivery_recipe=2`, `delivery_mode=adaptive`, the
binary's hash, the non-default parameters, the keyed tags of the seed, the
three inputs, the sandwich and `final.mpmct1`, then `status=`, `rounds=` and
`stop_reason=`. A resume skips the stage only when the whole marker
matches; an interrupted attempt of an unchanged recipe (no marker,
`adaptive/attempt.recipe` equal to the expected marker text) resumes in
place and keeps its completed rounds; any other rerun removes `adaptive/`.

### Windowed delivery (`delivery.mode = "windowed"`)

The windowed mode is one stage of two sub-steps (RC,
2026-10-12; [DELIVERY_STAGE.md](DELIVERY_STAGE.md)): the packer emits the
circuit as ANF windows on a lattice of cuts, and the unpacking sub-step
resynthesizes every window from its ANF alone, treating a window that does
not resynthesize as a packing defect: the window is cut in two, both halves
are repacked and unpacked, recursively, until every window unpacks (the size
of a resynthesized window is irrelevant). The stage's output, and the
pipeline's published object, is the resynthesized circuit
`unpacking/unpacked.mpmct1`: the converged windows concatenated in lattice
order and put into the canonical commuting gate order. The windowed ANF
deliverables, `packing/deliverable.tdp4` (the packer's lattice, round 0) and
`unpacking/deliverable.tdp4` (the refined lattice), are an internal
representation: the packer's windowed form that the unpacking sub-step
resynthesizes from, kept for auditing and parity. Only under
`delivery.anf_only`, where the stage ends after the packer's
`packing/deliverable.tdp4`, is that ANF file the published object: the
recipient then does the unpacking itself.

**What the adversary sees.** A circuit, nothing else:
`unpacking/unpacked.mpmct1`. Not $C$, not the sandwich or the masked circuit
of stages 1-2, not the mixed, split or crossed circuits of stages 3-5, not
the packer's gate order, not the window lattice, not the window ANFs. It can
cut the circuit anywhere, compute the function of any segment, compose
segments and resynthesize them into circuits of its own; it is never handed
the lattice or the ANFs, and it does not see how a window was spelled before
resynthesis: the packer's spelling is gone, and what it holds is the
synthesizer's spelling. The measurements that model this adversary are the
all-cuts C-view census of the resynthesized circuit and the uniform
re-cut-and-resynthesize experiments, which re-cut the circuit uniformly and
never use the delivered lattice. Measured at n=64: at its own cuts the
unpacked circuit of the seven-stage run (157 windows left verbatim) exposes
5 of 936 interior $C$ values, the raw packed circuit 0 and the complete
bisection assembly of the same run 0; merging and resynthesizing uniform
W=2000 windows of the crossing output recovers 244 of 936 interior $C$
values (241 with a 10× time budget; the canonical reorder changes these
counts by under 2 %), while the raw crossing circuit exposes none at any
cut; stages 4 and 5 do not close this leak. What windowing buys in this
mode is not the absence of interior cuts (the resynthesized windows have
them: the synthesizer's gates) but the separation guarantee: the packer's
and the mixer's spelling never ships, every window is re-spelled from its
function alone, and the refinement's exposure-aware cuts are chosen so that
no interior $C$ value is affine at a new cut, each one re-scanned. Under
`anf_only` the older framing applies: the adversary holds the window ANFs,
hence the wire state at every window boundary (on inputs of its choice); it
can compose adjacent windows into wider ones and resynthesize any window or
merged window, and a window has no interior cuts
([notes/stubborn_windows_20261012.md](notes/stubborn_windows_20261012.md),
[notes/stage45_revival_20261011.md](notes/stage45_revival_20261011.md)).

#### Packing (the first sub-step)

The packer reorders its input (the last circuit produced: `crossing.mpmct1`
by default, `db_mixing.mpmct1` under `run.skip_stages = [4, 5]`) along
commutations so that values are born and consumed inside one window, cuts
the order into windows of at most G gates, and emits each window as its
sparse algebraic normal form (ANF). The packer's order repair
(`stages::packing::repair`, the port of
`scripts/windowed_delivery/waist/order_repair.py` in its patched version)
then removes the remaining boundary exposures of $C$'s values; it runs in
every `tdp_packing run` and is not optional (the optional repair is the
separate post-stage-3 leakage audit, `[leakage_repair]`). The result is
serialized. The design and the measurements are in
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

The command does, in one process, what val_chain.sh does after stage 3:

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

**`packing/deliverable.tdp4`** is the packer's deliverable: the stage's
output, and the published object, only under `anf_only`; otherwise the
refinement's starting point (round 0), an internal record. It is a 48-byte
header (`LMTDPANF`, version, source and state wires, layout, window count,
body length, CRC-32 of the body and of the header) followed by the unchanged
`anf_dump_eval.py` body, which is also kept as `deliverable.bin` (a secret
intermediate).
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

**`packing.report`** (flat `key=value`; an internal record, not part of
the TDP: counts and parameters only, no seeds and no hash of a secret input)
records the status, every
parameter, the pack/repair/emit counters, the linkability census (L2/M2) and
the timings.

**Speed** (Apple-silicon Mac, other jobs running): `tdp_packing run` on the
archived validation runs takes 20-45 s at n=64 (0.9-1.1 GB peak) and 75-140 s
at n=128 (1.9-2.1 GB), against about 5-6 min and 15-17 min for the Python
chain.

#### Unpacking with lattice refinement (the second sub-step)

Unpacking resynthesizes a circuit from the windowed ANF form (under
`delivery.anf_only` this is what the recipient of the deliverable does
itself), and the refinement makes sure that it succeeds for every window; its
output is the published circuit. The driver runs, right after the packer
(skipped under `delivery.anf_only`),

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
blobs in its place, and the next round unpacks the new halves (a half whose
ANF exceeds the 65,536-monomial budget is emitted but not synthesized and is
split again in the next round, `over_budget` in the report; one still over
budget at the last round is a check failure, exit 4); the loop ends when
nothing is pending or after `max_rounds`. The converged windows
are concatenated in lattice order and the whole circuit is put into the
canonical commuting gate order (`xgate_canonicalize`, so no window boundary
survives). Measured basis and the full contract: [DELIVERY_STAGE.md](DELIVERY_STAGE.md)
(on the n=64 seven-stage run 157 of 10,366 windows failed to unpack in 30 s;
bisection converges by about 90 % per round and ends at 15-16-gate pieces);
the synthesizer, its checks and its exit codes: [STAGE7_UNPACKING.md](STAGE7_UNPACKING.md).
First six-stage run (n=16, 2026-10-12, `port_runs/six_n16b`, made at
`move_budget_factor = 2`): 18 min end to end; delivery 3 rounds, 678 → 696
windows, 18 clean cuts, 0 unclean cuts, 0 verbatim, every check ok.

It writes into `unpacking/`: `unpacked.mpmct1` (**the published circuit**:
the state width in the header, canonical order), `deliverable.tdp4` (the
refined lattice: the internal ANF form the circuit was resynthesized from,
kept for auditing and parity), `bounds.final` (its cuts), `windows.tsv` (one
row per window of the final lattice with its round and status), `rounds.tsv`
(per round: pending, converged, failed, split, seconds), `unpacking.report`
(`status=ok|verbatim_used`, `rounds`, `windows_initial`, `windows_final`,
`splits`, `unclean_cuts`, `over_budget`, `verbatim`, the gate counts and
ratio, every check), `unpacking.log` and the per-round `bounds.<r>` /
`deliverable.<r>.tdp4` (round 0 = the packer's; internal); the driver appends its output to `stage6.log`
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

#### Resume, markers and failures (windowed mode)

After both sub-steps the driver writes `delivery.recipe` (`delivery_recipe=1`): the two binaries'
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

#### The oracle

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
- `tests/manual/delivery_parity.sh OUT`: the Rust delivery loop
  (`tdp_unpacking run --refine`) against the Python reference `deliver.py` on
  n8_L8 with the same seed, time limit and restarts; `bounds.final`,
  `deliverable.tdp4`, `unpacked.mpmct1` and every per-round file identical
  (`TDP_UNPACKING`, `TDP_PACKING`, `HMAP_AFFINE` env);
  `tests/integration/tdp_unpacking.rs` for the binary;
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

#### The reference driver

`waist/val_chain.sh TAG N MC S SLICE CONFIG` runs stages 1-3 and the Python
packer (the four-stage chain: today's `run.skip_stages = [4, 5]` with
`delivery.mode = "windowed"` and `delivery.anf_only = true`) and seeds below $2^{31}$/$2^{40}$ in `W/<TAG>/seed.txt` (calibration
grade). It hard-codes the run tree (`/Users/rancanetti/code/local_mixing`,
`runs/ncdb_ab_20261001/waist/<TAG>`), so outside that tree run it through the
reproduction test's sandbox. Its CONFIG is `L8` (recommended) or `A`; the
managed presets of the same names set exactly its environment.

## History

Retired components and the earlier pipelines are recorded in
[local_mixing_history.md](local_mixing_history.md) (and the archive
inventories it links to); this manual describes only the current six
stages. A run that records a recipe earlier than v13, or a mixer state file
of a format other than the current v6, is refused: resume it with the
checkout that created it.

## The shared pieces

We do not need to understand every file to change one stage. The circuit
representation is in [src/circuit/xgate.rs](../src/circuit/xgate.rs), and its
readers/writers are in [src/circuit/formats.rs](../src/circuit/formats.rs).
The mutable gate tape used by `circuit_mixer` is in
[src/engine/arena.rs](../src/engine/arena.rs). Mixer state, sampling, checkpoint
I/O, and reporting live together in [src/engine/mixer/](../src/engine/mixer/).

Use these current module paths directly. Files named
`environment.rs` read settings of the stage-3 engine and the store; the
managed driver sets or forbids each of them (see the environment policy).
Stages 4 and 5 live in
[src/stages/post-processing/](../src/stages/post-processing/) as mixer modes
(their slots, indices, journal and counters in `src/engine/mixer/`). Stage 6's
adaptive mode lives in [src/stages/adaptive/](../src/stages/adaptive/) (the
census, the boundary classification, the planner, the block synthesis, the
rounds and the driver; no environment reads) with its binary in
[src/programs/tdp_adaptive/](../src/programs/tdp_adaptive/); the windowed
mode's packer in [src/stages/packing/](../src/stages/packing/) with
[src/programs/tdp_packing/](../src/programs/tdp_packing/), and its unpacking
sub-step in [src/stages/unpacking/](../src/stages/unpacking/) with
[src/programs/tdp_unpacking/](../src/programs/tdp_unpacking/) (the adaptive
mode reuses its synthesizer). The three-mask stage 2 lives in
[src/stages/preprocessing/three_mask/](../src/stages/preprocessing/three_mask/),
the balanced source in [src/stages/sandwich/balanced.rs](../src/stages/sandwich/balanced.rs).

Stage 3 can run on contiguous pieces in parallel; the coordinator is
[piecewise.rs](../src/engine/mixer/piecewise.rs). Correctness checks live in
[tests/](../tests/), while attacks and leakage measurements live in
[security_tests/](../security_tests/). The detailed
[current-method documentation](local_mixing_documentation.md) explains the
implemented construction and attacks. The [history](local_mixing_history.md)
preserves the motivation and earlier experiments behind this order.
