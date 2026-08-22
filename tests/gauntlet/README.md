# Gadget gauntlet

This directory contains the trace-and-audit pipeline for the
`gadgetization.nonlinear193` and `gadgetization.nonlinear291` constructions.
It also keeps three native controls (`none`, `secretshare14`, and
`bandproduct92`) so a run can demonstrate that the attacks detect known weak
encodings.

## Dependencies

- A Rust toolchain supported by this repository.
- Python 3.10 or newer with NumPy and Matplotlib. The repository's Python
  project dependencies include both packages.

Build the two test binaries from the repository root:

```sh
cargo build --release --bin gauntlet_gen --bin gauntlet_audit
```

The scripts resolve repository paths from their own location. If the script
itself is addressed by an absolute path, it can be launched from any working
directory without moving outputs or losing package imports.

## Commands

A small end-to-end smoke run is:

```sh
python tests/gauntlet/gauntlet.py all \
  --arms none,nonlinear193,nonlinear291 \
  --ks 1 --mix off --corr-samples 1024 \
  --w2-cap 128 --w3-cap 64
```

The full default matrix uses every arm, chain lengths 1/2/16, and both mixed
and unmixed modes. It is a long-running empirical campaign; the default
weight-2/weight-3 subset caps are 64/16 to keep the combinatorial scans
bounded:

```sh
python tests/gauntlet/gauntlet.py all --jobs 4
```

Larger `--w2-cap` and `--w3-cap` values deepen the scan, but work grows with
the number of feature pairs and triples, not linearly with the caps.

Stages can be run separately:

```sh
python tests/gauntlet/gauntlet.py gen
python tests/gauntlet/gauntlet.py audit
python tests/gauntlet/gauntlet.py maps
python tests/gauntlet/gauntlet.py report
python tests/gauntlet/gauntlet.py clean
```

Outputs default to `target/gauntlet`. `clean` removes that exact marked output
directory. Ownership markers are bound to the resolved output path. The runner
never removes repository source paths, and a forced reset of an unmarked
directory is limited to a child of `target/` or the system temporary directory.

Each cell records stage configuration and source/binary provenance in
`cell-config.json`. Completed stages resume only when their artifacts and
saved configuration match; every artifact is bound by its size and SHA-256
digest. A changed generation configuration requires
`--force` (or `clean`) before replacement; audit and map products are safely
regenerated when their own settings change.

See [TESTING_PIPELINE.md](TESTING_PIPELINE.md) for the bundle format, attack
coverage, limitations, and arm definitions.
