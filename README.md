# Local Mixing

We use local mixing to obfuscate reversible circuits: replace small portions
of a circuit with equivalent ones, while preserving the function we want to
compute. GSS is the current end-to-end pipeline. It constructs a sliced
sandwich, applies gadgetization, mixes with the frozen database, then runs
splitting, crossing and final compression. The final circuit is `final.esop1`.

## Setup

Run from the repository root on Linux or WSL, with Bash, Python 3.10 or newer,
a native compiler/linker, and Rust installed through rustup. The repository
pins Rust 1.97.1 in [`rust-toolchain.toml`](rust-toolchain.toml); rustup selects
it automatically. [`Cargo.lock`](Cargo.lock) pins the Rust dependencies.

```sh
cargo build --release --locked
cp configs/gss.example.toml configs/local.toml
```

`configs/` holds the GSS mixing configuration. The
[`gss.example.toml`](configs/gss.example.toml) example runs all six steps with
the current defaults. Copy it to `configs/local.toml` for your database paths
and run settings; that local file is ignored by Git. Relative paths are
resolved against the repository root. [`gss.toml`](configs/gss.toml) is the
full template and is read by default when `--config` is omitted.

## Run GSS

Replace `/path/to/frozen` in the existing `[database]` section of
`configs/local.toml` with your regular frozen database directory:

```toml
[database]
regular_dir = "/path/to/frozen"
# curated_dir = "/path/to/curated"
lookup_miss_filter = "auto"
allow_no_database_for_tests = false
```

Stage 3 requires the regular frozen store. The curated store is optional and
is consulted before the regular store when its routing policy applies. A
frozen directory contains `tables.bin` and all 256 `shard_00.frz` through
`shard_ff.frz` files. `filters.bin` supplies the optional miss filter. See
[Frozen Database](docs/frozen_database.md) for the format and build tools.
`FROZEN_DB_DIR` and `FROZEN_CURATED_DIR` supply paths when their TOML fields
are omitted.

```sh
cargo run --release --locked -- gss --config configs/local.toml --dry-run
cargo run --release --locked -- gss --config configs/local.toml
```

The dry run resolves and validates the recipe without creating a run or
starting the stages. A normal fresh run builds the three stage executables
and writes a timestamped directory under `runs/`. Database files are separate
from the source checkout; building the project does not generate them.

By default we generate a source circuit $C$ on 128 wires with 6,272 gates,
using $\operatorname{round}(n\log_2(n)^2)$ gates. To use your own G57 source,
edit `[source]`:

```toml
[source]
wires = 8
path = "circuits/source.g57"
```

The file must be nonempty and fit the declared wire count. Its gate count is
inferred, so omit `source.gates` unless deliberately supplying a matching
count. Without `source.path`, `source.gates` overrides the generated size.

The settings we usually change are:

- `preprocessing.mode`: defaults to `"quadratic-masking"`, with balanced
  masks, `mask_pair_wires=2`, `max_open_masks=3` and `min_open_masks=2`.
  `"nonlinear291"` is also supported; remove all four mask controls when
  selecting it and use a smaller source that fits its 65,535-wire limit.
  The default source exceeds that limit; see [Gadgetization](docs/gadgetization.md)
  for the width calculation.
- `db_mixing.target_size_factor` and `hold_work_units`: default to `2.0` and
  `27.0`, controlling the expansion and holding schedule.
- `parallel.pieces` or `parallel.target_piece_gates`: enable piecewise
  execution for stages 3–4. Choose one; omit both for serial execution.
  `parallel.threads` is available with piecewise execution.
- `crossing.target_size_factor`: defaults to `2.0`. The template also exposes
  the width penalty, size tolerance and optional move limit.

Optional leakage repair is configured under `[leakage_repair]`. For a plumbing
test without a database, clear the DB paths and set
`database.allow_no_database_for_tests=true`; this disables DB replacements.

## Outputs and continuation

| Step | Output in the run directory |
| --- | --- |
| 1. Sliced sandwich | `gss.mpmct1.source_c.g57`, `gss.mpmct1.sandwich.mpmct1` |
| 2. Gadgetization / preprocessing | `gss.mpmct1` |
| 3. Database mixing | `db_mixing.mpmct1`, `db_mixing.state` |
| 4. Splitting | `split.mpmct1`, `split.state` |
| 5. Crossing | `crossing.mpmct1`, `crossing.state` |
| 6. Compression and packing | `final.esop1` |

`gss_mix.log` records overall progress; `stage12.log` and `stage3.log` through
`stage6.log` hold stage output. The
[GSS Pipeline](docs/gss_pipeline.md) explains these six steps and where to find
their code.

To continue a run, set `run.directory` to its directory and invoke the same
command again. Completed stage outputs are skipped. Set `run.stop_after_stage`
to `2`, `3`, `4`, `5` or `6` to stop early; stages 1 and 2 execute together.
`run.rerun_from_stage` deliberately recomputes that stage and the later stages
the invocation reaches. It does not permit changing a recorded recipe.

New runs record recipe v7 in `gss_command.conf`, including source hashes and
driver/executable fingerprints. Managed continuation reuses the exact
recorded stage binaries and skips rebuilding them. Preserve their build
directory; use a fresh run directory after changing recipe settings or stage
binaries. Saved runs must use recipe v7; the older v3–v6 drivers have been
retired. The private `SEED` file and `stage12.recipe` also belong to the run.

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

We must distinguish the source from the public GSS circuit. With default
quadratic masking, an $n$-wire source becomes a $4n$-wire circuit. To compute
$C(x)$, place $x$ on wires `0..n`, set the remaining input wires to zero, and
read output wires `n..2n`. Other outputs can contain junk. Thus, comparing the
source directly with `final.esop1` using the full-function command above is
not the GSS payload check. See [Gadgetization](docs/gadgetization.md) for the
slice contract.

## Working in the codebase

| Location | What lives there |
| --- | --- |
| [`configs/`](configs/) | GSS example, default recipe, and your local configuration |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Builds and tests on GitHub pushes and pull requests; not required for local GSS runs |
| [`src/gss/`](src/gss/) | Recipe parsing, validation, run management and manifests |
| [`src/stages/`](src/stages/) | The six pipeline steps |
| [`src/engine/`](src/engine/) | Mixer state, moves, scheduling and checkpoints |
| [`src/circuit/`](src/circuit/) | Gates, file formats and evaluation |
| [`src/canonicalization/`](src/canonicalization/) | Polynomial composition, canonical forms and keys |
| [`src/database/`](src/database/) | Frozen-store lookup, decoding and validation |
| [`db_gen/`](db_gen/) | Offline database generation and maintenance |
| [`tests/`](tests/) | Current circuit, pipeline, gadgetization and database correctness tests |
| [`security_tests/`](security_tests/) | Security experiments and analysis tools |
| [`challenges/`](challenges/), [`benchmarks/`](benchmarks/) | Challenge programs and performance tools |

[`scripts/gss_mix.sh`](scripts/gss_mix.sh) runs the stage executables;
[`src/programs/`](src/programs/) adapts their arguments and artifact I/O.
The benchmark launchers live directly under `benchmarks/`. Python's
`__pycache__/` directories are generated locally and ignored by Git.
[Polynomial Canonicalization](docs/polynomial_canonicalization.md) explains
how different circuits reach the same database key.

Ordinary GSS needs no optional Cargo features. `db-tools` enables database
commands and native LMDB/RocksDB dependencies; building those tools needs a
C/C++ toolchain and Clang/libclang. `security-tools`, `challenge-tools` and
`benchmark-tools` enable tools built on the current circuit and stage code.

Stage code lives directly under `src/stages/`, and frozen storage lives under
`src/database/`. The old top-level `preprocessing`, `db_mixing` and
`postprocessing` imports and the historical mixing implementations have been
removed. GSS configuration uses TOML; the old Markdown recipes are retired.

```sh
cargo run --release --locked --features db-tools -- db --help
cargo fmt --all -- --check
cargo test --locked --lib --test circuit_cli
cargo test --locked --test frozen_roundtrip --test fmix_db_move
bash tests/gss_flags.bash
bash tests/gss_block_size.bash
python3 -B -m unittest discover -s tests/gadgetization -v
```

The Rust tests check functional equivalence through gadgetization, mixing,
database replacement and compression, along with circuit formats and run
management. The Python tests check nonlinear291 and its four runtime templates
and require NumPy (`python3 -m pip install numpy`). The frozen integration
tests create small LMDB fixtures and convert them to frozen stores before
checking lookups and replacements.

For Python analysis, install from a Python environment with
`python3 -m pip install .`. [`pyproject.toml`](pyproject.toml) uses maturin and
the `python-extension` feature, preserving `import local_mixing`.

The longer research document is
[Local Mixing Documentation](docs/Local_Mixing_Documentation.pdf), with its
[editable source](docs/Local_Mixing_Documentation.md). Rebuild the PDF with
Pandoc and XeLaTeX installed:

```sh
pandoc docs/Local_Mixing_Documentation.md --from=markdown --resource-path=docs --pdf-engine=xelatex -o docs/Local_Mixing_Documentation.pdf
```
