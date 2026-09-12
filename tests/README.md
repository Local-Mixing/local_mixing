# Correctness tests

`cargo test --lib` runs the runtime tests, including circuit evaluation,
canonicalization, replacement-store lookups, slicing, crossing, compression,
checkpoint round trips and both supported preprocessors. Large optional or known
failure audits remain explicitly ignored where they were before this cleanup.

- `unit/`: private Rust unit-test bodies, included from their owning modules
  under `cfg(test)`. Moving them here does not make runtime internals public.
- `stages/`: sandwich, preprocessing, database mixing and post-processing correctness.
- `database/`: frozen-reader and lookup-cache checks.
- `unit/canonicalization/`: canonical-key, polynomial and G57 compatibility regressions.
- `db_gen/`: database construction validation with private builder access.
- `unit/circuit/operations.rs`: reusable circuit loading/evaluation, high-wire
  comparison semantics and state preservation.
- `circuit_cli.rs`: real generation, evaluation/comparison formats, input flags
  and mismatch exit status.
- `frozen_roundtrip.rs` and `fmix_db_move.rs`: real frozen-store integration tests.
- `poly_canon*.rs`: graph-canonicalizer comparison tests; use `--features legacy-tools`.
- `gadgetization/`: Python nonlinear193/nonlinear291 correctness and template tests.
- `manual/gss/`: circuit artifact verification (`verify_zero_slice`).
- `gss_block_size.bash`: script regression checks for automatic/fixed block sizing.
- `gss_flags.bash`: gadget flag normalization, validation, retired-mode rejection,
  environment precedence and stage-2 restart checks using mock binaries.

Run the Python checks from the repository root:

```sh
python3 -m unittest discover -s tests/gadgetization -v
```

Security measurements, heatmaps, attacks and gadget gauntlets live in
`security_tests/`. Runtime leakage repair remains part of GSS and its correctness
is tested here.
