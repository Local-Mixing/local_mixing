# Manual validators

These are command-line acceptance checks that require user-supplied artifacts.
They are not automatically executed by `cargo test`; Cargo registers each one
explicitly as a binary.

For example:

```bash
cargo run --release --bin verify_zero_slice -- C.g57 final.mpmct1 N
```

