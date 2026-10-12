# Local Mixing

We use local mixing to obfuscate reversible circuits: replace small portions
of a circuit with equivalent ones, while preserving the function we want to
compute. TDP is the current end-to-end pipeline. It has seven stages: it
constructs a sliced sandwich (1), applies preprocessing, i.e. gadgetization
(2), mixes with the frozen database (3), splits the remaining complemented
gates (4), runs the crossing walk (5), packs the result into a windowed ANF
deliverable (6), and unpacks it again by resynthesizing every window from its
ANF into one canonically ordered circuit (7). Every stage after the first is
optional. The `tdp_gen` driver runs all seven by default and ends with
`unpacking/unpacked.mpmct1`; the defaults of stages 1-3 and 6 reproduce the
chosen validation runs (`docs/STAGE4_RUST_DESIGN.md`; `run.skip_stages = [4,
5, 7]` is that four-stage chain). Stage 6 is the Rust binary `tdp_packing`;
the Python prototypes under [`scripts/windowed_delivery/`](scripts/windowed_delivery/)
are its oracle. Stage 7 is `tdp_unpacking` (`docs/STAGE7_UNPACKING.md`). The
[TDP Pipeline](docs/tdp_pipeline.md) guide explains how to run the stages; the
full reference for stages 1-3 and 6 is
[TDP_FOUR_STAGE_PIPELINE.md](docs/TDP_FOUR_STAGE_PIPELINE.md). (The curated
database and the old `fcompress` stage are historic: branch
`ran/archive/curated-and-postmix`; see
[the archive inventory](docs/CURATED_AND_POSTMIX_ARCHIVE.md).)

## Setup

Run from the repository root on Linux or WSL, with Bash, Python 3.10 or newer,
a native compiler/linker, and Rust installed through rustup. The repository
pins Rust 1.97.1 in [`rust-toolchain.toml`](rust-toolchain.toml); rustup selects
it automatically. [`Cargo.lock`](Cargo.lock) pins the Rust dependencies.

```sh
cargo build --release --locked
cp configs/tdp.toml configs/local.toml
```

[`configs/tdp.toml`](configs/tdp.toml) is the single default recipe and fully
annotated template. Copy it to `configs/local.toml` for your database paths
and run settings; that local file is ignored by Git. Relative paths are
resolved against the repository root. The default recipe is read when
`--config` is omitted. The [pipeline configuration reference](docs/tdp_pipeline.md#configuration-reference)
explains every setting, its default, constraints and corresponding driver flag.

## Run TDP

Replace `/path/to/frozen` in the existing `[database]` section of
`configs/local.toml` with your regular frozen database directory:

```toml
[database]
regular_dir = "/path/to/frozen"
lookup_miss_filter = "auto"
allow_no_database_for_tests = false
```

Stage 3 requires the regular frozen store. (The former curated store is
historic; it lives on the branch `ran/archive/curated-and-postmix`.) A
frozen directory contains `tables.bin` and all 256 `shard_00.frz` through
`shard_ff.frz` files. `filters.bin` supplies the optional miss filter. See
[Frozen Database](docs/frozen_database.md) for the format and build tools.
`FROZEN_DB_DIR` supplies the path when the TOML field is omitted.

```sh
cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

The dry run resolves and validates the recipe without creating a run or
starting the stages. A normal fresh run builds the three stage executables
and writes a timestamped directory under `runs/`. Database files are separate
from the source checkout; building the project does not generate them.

By default we generate a source circuit $C$ on 128 wires with 3,000 gates
(the validated sizes are 1,000 gates at 64 wires and 3,000 at 128; other
wire counts need `source.gates` or `source.gates_rule = "formula"`, i.e.
$\mathrm{round}(n\log_2(n)^2)$). To use your own G57 source, edit `[source]`:

```toml
[source]
wires = 8
path = "circuits/source.g57"
```

The file must be nonempty and fit the declared wire count. Its gate count is
inferred, so omit `source.gates` unless deliberately supplying a matching
count. Without `source.path`, `source.gates` overrides the generated size.

The settings we usually change are:

- `preprocessing.preset`: `"L8"` (default; persistent carrier migration,
  premask, seed avoidance and bounded chains, the recommended configuration),
  `"A"` (the same without chains) or `"custom"` (every knob explicit).
  `preprocessing.mode = "nonlinear291"` is also supported, without a preset.
- `db_mixing.schedule`: `"validated"` (default; profile `2,2,2,2,2`) or
  `"custom"` with explicit growth, hold and size factor.
- `[packing]`: stage-4 parameters; the defaults are the validated ones.
- `parallel.pieces` or `parallel.target_piece_gates`: enable piecewise
  execution for stage 3. Choose one; omit both for serial execution.
  `parallel.threads` is available with piecewise execution.

Optional leakage repair is configured under `[leakage_repair]`. For a plumbing
test without a database, clear the DB paths and set
`database.allow_no_database_for_tests=true`; this disables DB replacements.

## Outputs and continuation

| Step | Output in the run directory |
| --- | --- |
| 1. Sliced sandwich | `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` |
| 2. Gadgetization / preprocessing | `tdp.mpmct1` |
| 3. Database mixing | `db_mixing.mpmct1` |
| 4. Packing | `packing/deliverable.tdp4` (the shipped file), `packing/packing.report`, and the private intermediates in `packing/` |

`tdp_gen.log` records overall progress; `stage12.log`, `stage3.log` and
`stage4.log` hold stage output. The
[TDP Pipeline](docs/tdp_pipeline.md) explains these steps and where to find
their code.

To continue a run, set `run.directory` to its directory and invoke the same
command again. Completed stage outputs are skipped. Set `run.stop_after_stage`
to `2` or `3` to stop early (`4`, the default, runs every stage; the historic
stages 5-6 are rejected); stages 1 and 2 execute together.
`run.rerun_from_stage` deliberately recomputes that stage and the later stages
the invocation reaches. It does not permit changing a recorded recipe.

New runs record recipe v10 in `tdp_command.conf`, including the resolved
recipe and driver/executable fingerprints (never a seed or a hash of $C$). Managed continuation reuses the exact
recorded stage binaries and skips rebuilding them. Preserve their build
directory; use a fresh run directory after changing recipe settings or stage
binaries. Continue only runs created by the matching TDP recipe; earlier
pipeline runs use different artifact names and are explicitly rejected.
The private `SEEDS` file (fresh CSPRNG seeds; an explicit seed file is for
calibration only) and `stage12.recipe` also belong to the run.

## Using circuit files

```sh
cargo run --release --locked -- circuit generate -n 8 -m 30 -d source.g57
cargo run --release --locked -- circuit evaluate -n 8 -s source.g57 --input 0
cargo run --release --locked -- circuit compare -n 8 -i 1000 -a first.g57 -b second.g57
```

Evaluation and comparison accept G57, `mpmct1`, `esop1` and `anf1`. Evaluation
also accepts `--random` instead of `--input`; explicit inputs may be decimal
or `0x`-prefixed hexadecimal. Comparison samples complete inputs and outputs,
and exits unsuccessfully if it finds a mismatch. Passing is a sampled check.

We must distinguish the source from the public TDP circuit. With default
embedded masking, an $n$-wire source becomes a $4n$-wire circuit. To compute
$C(x)$, place $x$ on wires `0..n`, set the remaining input wires to zero, and
read output wires `n..2n`. Other outputs can contain junk. Thus, comparing the
source directly with `db_mixing.mpmct1` using the full-function command above is
not the TDP payload check. See [Gadgetization](docs/gadgetization.md) for the
slice contract.

## Working in the codebase

| Location | What lives there |
| --- | --- |
| [`configs/`](configs/) | One annotated default recipe, plus your ignored local configuration |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Builds and tests on GitHub pushes and pull requests; not required for local TDP runs |
| [`src/tdp/`](src/tdp/) | Recipe parsing, validation, run management and manifests |
| [`src/stages/`](src/stages/) | The pipeline steps |
| [`src/engine/`](src/engine/) | Mixer state, moves, scheduling and checkpoints |
| [`src/circuit/`](src/circuit/) | Gates, file formats and evaluation |
| [`src/canonicalization/`](src/canonicalization/) | Polynomial composition, canonical forms and keys |
| [`src/database/`](src/database/) | Frozen-store lookup, decoding and validation |
| [`db_gen/`](db_gen/) | Offline database generation and maintenance |
| [`tests/`](tests/) | Current circuit, pipeline, gadgetization and database correctness tests |
| [`security_tests/`](security_tests/) | Heatmaps, affine/trace analysis, gadget gauntlet and SAT solving |
| [`benchmarks/`](benchmarks/) | Performance tools |

[`scripts/tdp_gen.sh`](scripts/tdp_gen.sh) runs the stage executables;
[`src/programs/`](src/programs/) contains each stage executable's `main.rs`
alongside its argument handling and artifact I/O. `circuit_mixer` is the
database-mixing executable (stage 3); `tdp_gen` is
the complete managed pipeline command.
The benchmark launchers live directly under `benchmarks/`. Python's
`__pycache__/` directories are generated locally and ignored by Git.
[Polynomial Canonicalization](docs/polynomial_canonicalization.md) explains
how different circuits reach the same database key.

Ordinary TDP needs no optional Cargo features. `db-tools` enables database
commands and native LMDB/RocksDB dependencies; building those tools needs a
C/C++ toolchain and Clang/libclang. `security-tools` and `benchmark-tools`
enable tools built on the current circuit and stage code.

Stage code lives directly under `src/stages/`, and frozen storage lives under
`src/database/`. TDP configuration uses TOML recipes based on
[`configs/tdp.toml`](configs/tdp.toml).

The correctness tests are grouped by how they run:

- [`tests/unit/`](tests/unit/) checks individual Rust modules, including private
  circuit, canonicalization, database, mixer, and stage behavior.
- [`tests/integration/`](tests/integration/) checks the circuit CLI, frozen-store
  conversion, and database replacement through their public interfaces.
- [`tests/python/`](tests/python/) checks gadget reference implementations,
  template regeneration, gauntlet orchestration, and heatmap plotting.
- [`tests/shell/`](tests/shell/) checks driver flags and piece sizing with mock
  executables.
- [`tests/manual/`](tests/manual/) contains the zero-slice validator for
  explicitly supplied generated circuits.

Run the ordinary checks from the repository root:

```sh
cargo fmt --all -- --check
rustfmt --check --edition 2024 tests/unit/engine/mixer/*.rs
cargo test --locked --lib --test circuit_cli --test frozen_roundtrip --test mixer_db_replacement
bash tests/shell/tdp_flags.bash
bash tests/shell/tdp_block_size.bash
python3 -B -m unittest discover -s tests/python -v
```

The Rust tests check functional equivalence through gadgetization, mixing,
database replacement and compression, along with circuit formats and run
management. The Python tests require NumPy and Matplotlib
(`python3 -m pip install numpy matplotlib`). The frozen integration
tests create small LMDB fixtures and convert them to frozen stores before
checking lookups and replacements.

The optional database tools have additional builder and command tests. With
the native dependencies described above installed, run the checks used in CI:

```sh
cargo test --locked --features db-tools --lib db_generation::
cargo test --locked --features db-tools --bin local_mixing_bin
cargo test --locked --features db-tools --test frozen_roundtrip
```

Security-tool checks cover the trace heatmap and SAT workflow separately:

```sh
cargo test --locked --features security-tools --bin hmap_trace_affine
python3 security_tests/heatmaps/plot_hmap_trace.py --self-test
python3 -B -m unittest security_tests.sat_solve.test_workflow -v
```

For Python analysis, install from a Python environment with
`python3 -m pip install .`. [`pyproject.toml`](pyproject.toml) uses maturin and
the `python-extension` feature, preserving `import local_mixing`.

The detailed current-method reference is
[Local Mixing Documentation](docs/local_mixing_documentation.md)
([PDF, stale](docs/local_mixing_documentation.pdf)). It explains the implemented
construction, shuffling, mixing stages, optimizations, and attacks. The
research narrative and historical experiments are preserved in
[Local Mixing History](docs/local_mixing_history.md)
([PDF, stale](docs/local_mixing_history.pdf)). The committed PDFs were built
on 2026-09-16, before the four-stage cleanup, and still describe the curated
database and the six-stage pipeline; the Markdown files are authoritative
until the PDFs are rebuilt. Install Pandoc, XeLaTeX and
`rsvg-convert` (the `librsvg2-bin` package on Debian/Ubuntu), then rebuild both
PDFs with the checked-in helper:

```sh
bash scripts/build_docs.sh
```

[`scripts/build_docs.sh`](scripts/build_docs.sh) locates the repository from
its own path, so it also works when called from another directory. It builds
both PDFs before replacing the outputs in `docs/`; `--output-dir DIRECTORY`
chooses another destination. The SVG converter preserves vector circuit
figures, and [`scripts/pdf_links.lua`](scripts/pdf_links.lua) preserves explicit
section anchors and links to the companion PDF. Plain Pandoc commands without
that filter lose those custom section destinations.
