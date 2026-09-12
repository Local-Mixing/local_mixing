# GSS runtime

The main CLI exposes `gss`, separate `circuit generate/evaluate/compare` handlers, and optional `db` commands. Reusable circuit loading, evaluation and full-width sampled comparison live in `circuit/operations.rs`.

| Directory | Responsibility |
| --- | --- |
| `gss/config/` | Typed recipes, TOML parsing, validation and historical recipe interpretation |
| `gss/` | CLI schema, paths, run/build management, manifests and continuation |
| `stages/sandwich/` | Source preparation and sliced sandwich construction |
| `stages/preprocessing/` | Quadratic masking, nonlinear291, slice guards and runtime templates |
| `stages/db_mixing/` | Equivalent database replacement and leakage repair |
| `stages/post-processing/` | Splitting, crossing and compression/transport/packing |
| `circuit/` | G57 and general gate tapes, permutations, evaluation, formats and randomization |
| `canonicalization/` | GF(2) polynomials, canonical ordering, keys and caches |
| `database/` | Immutable frozen stores, codecs, lookup cache and validation |
| `engine/mixer/` | Mutable mixer state, parameters, scheduling, provenance and checkpoint I/O |
| `engine/moves/` | Shared local transformation rules and verified wire-swap words |
| `programs/` | Executable argument/environment adapters and artifact I/O |
| `entrypoints/` | Three small `main()` functions registered by Cargo |

```text
source → sandwich → preprocessing → db_mixing → post-processing
                                               splitting → crossing → compression
```

Both supported preprocessors compile without optional features. Historical imports under `preprocessing/`, `db_mixing/`, `postprocessing/` and old engine paths delegate to the canonical owners. Optional comparison implementations remain in `security_tests/support/`; builder source stays in `db_gen/` and is excluded from ordinary runtime builds. Compatibility adapters own historical environment translation; new stage construction takes typed options.

The folder is literally `post-processing`, mapped once to Rust's `stages::post_processing`. Executable names and circuit/checkpoint formats are unchanged. See the [complete file map](../docs/CODE_LAYOUT.md) and [compatibility contract](../docs/formats/checkpoints.md).
