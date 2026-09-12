# Experimental Rust programs

This tree contains prototypes, diagnostics, censuses, and alternative workflows
that are not invoked by `scripts/gss_mix.sh` and do not define production
formats or correctness boundaries.

The singular `experimental/` directory contains source code; the pre-existing
plural `experiments/` directory contains dated campaign artifacts and results.

Every program remains an explicit Cargo binary, so use its existing name:

```bash
cargo run --release --bin NAME -- ARGS
```

See [`CODE_LAYOUT.md`](../CODE_LAYOUT.md) for the purpose of every file.
