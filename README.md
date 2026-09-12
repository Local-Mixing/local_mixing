# Local mixing

GSS mixes a reversible circuit through a sliced sandwich, preprocessing,
database mixing, splitting, crossing and final compression. The output is the
packed circuit `final.esop1`.

The default preprocessing is **quadratic masking** (formerly Ran balanced / blinded V5), with `mask_pair_wires=2`,
`max_open_masks=3`, `min_open_masks=2`, balanced masks and quadratic fire. It uses the
classic guarded sandwich, input-derived band seed/reseed, and no encoded I/O.
Both `quadratic-masking` and `nonlinear291` are available in ordinary builds.

## Run GSS

Edit [`configs/gss.toml`](configs/gss.toml), or copy it to the ignored
`configs/local.toml` for machine-specific settings. Set the frozen database
paths in that file or through `FROZEN_DB_DIR` and `FROZEN_CURATED_DIR`.

```sh
cargo run --release --locked -- gss --config configs/local.toml --dry-run
cargo run --release --locked -- gss --config configs/local.toml
```

For an existing source circuit, set `source.path` and `source.wires`. Otherwise
GSS generates a source. Paths in the configuration are relative to the repository.
Omit `run.directory` for a fresh run; set it to continue an existing managed run.

## Circuit and database commands

```sh
cargo run --release --locked -- circuit generate -n 8 -m 30 -d source.g57
cargo run --release --locked -- circuit evaluate -n 8 -s source.g57 --input 0
cargo run --release --locked -- circuit compare -n 8 -i 1000 -a first.g57 -b second.g57
cargo run --release --locked --features db-tools -- db --help
```

Evaluation and comparison accept G57, `mpmct1`, `esop1` and `anf1`. Comparison
samples inputs and returns a failure status if it finds a mismatch; it is not
an exhaustive equivalence proof.

## Code and retained tools

| Location | Contents |
| --- | --- |
| [`src/`](src/README.md) | GSS runtime, grouped CLI, circuit primitives and three stage entrypoints |
| [`configs/`](configs/README.md) | Operator recipe and configuration reference |
| [`db_gen/`](db_gen/README.md) | Regular, curated and wide database builders, freezing, filters and maintenance |
| [`tests/`](tests/README.md) | Correctness, checkpoint and command tests |
| [`security_tests/`](security_tests/README.md) | Heatmaps, affine heatmaps, gauntlet, attacks and Python nonlinear references |
| [`challenges/`](challenges/README.md) | All four existing challenge programs |
| [`benchmarks/`](benchmarks/README.md) | Performance measurements and retained probes |
| [`scripts/`](scripts/README.md) | GSS orchestration, verification helpers and historical run compatibility |
| [`docs/`](docs/README.md) | Complete file map, usage and design notes |

Optional Cargo features are `db-tools`, `security-tools`, `challenge-tools`,
`benchmark-tools`, `legacy-tools` and `python-extension`. Ordinary GSS builds
exclude the historical mixing/gadget families and Python/RocksDB dependencies.
`legacy-db-tools` remains a compatibility spelling for the existing DB feature.

Historical `sss`, `ssg`, `shoot`, `shuffle` and `compress` workflows are retained
in `legacy_mixing`, built with `--features legacy-tools --bin legacy_mixing`.
The same executable retains old `genran`, `evaluate` and `equal` utility aliases.

## Checks and Python analysis

```sh
cargo fmt --all -- --check
cargo test --locked --lib --test circuit_cli
cargo test --locked --test frozen_roundtrip --test fmix_db_move
cargo check --locked --all-targets --features "db-tools security-tools challenge-tools benchmark-tools"
python3 -m unittest discover -s tests/gadgetization -v
python3 -m pip install .
```

The Python extension keeps the existing `import local_mixing` heatmap API.
The compiler/tooling versions are pinned by `rust-toolchain.toml`, `rustfmt.toml`
and `Cargo.lock`; `pyproject.toml` owns Python packaging.

Current `.state` checkpoint readers are retained, including v1 and v2. Managed
v3 through v6 runs use their original drivers and recorded binaries; new runs use recipe v7; see
[`docs/GSS_MIX.md`](docs/GSS_MIX.md) for the resume contract. Existing databases,
run outputs and local archives remain at their original locations and are
excluded from the source push list.
