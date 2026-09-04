# local_mixing

Tools and research for transforming and analyzing obfuscated reversible
circuits.

- [`CODE_LAYOUT.md`](CODE_LAYOUT.md) inventories every maintained Rust source,
  executable, experiment, benchmark, validator, and red-team tool.
- [`docs/GSS_MIX.md`](docs/GSS_MIX.md) documents the supported GSS_MIX pipeline
  and contains the editable configuration used by
  `cargo run --release -- gss`.
- [`docs/BLINDED_V5_LGI_DESIGN.md`](docs/BLINDED_V5_LGI_DESIGN.md) describes the
  LGI-based (blinded-V5) compute stage — the drop-in alternative to the drip
  compute — with its rationale, parameters, and measurements.
- [`src/db_generation/README.md`](src/db_generation/README.md) documents frozen
  regular and curated database regeneration.
