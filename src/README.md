# `src/` layout

The complete per-file inventory is in [`CODE_LAYOUT.md`](../CODE_LAYOUT.md).

## Production GSS_MIX path

```text
n -> sliced sandwich -> gadgetize -> Phase A -> split -> crossing walk -> fcompress
     preprocessing/                 db_mixing/     postprocessing/
```

Only three production executables are used by `scripts/gss_mix.sh`:

| stage | executable | source |
|---|---|---|
| 1–2 | `gen_sandwich_gadget` | `preprocessing/bin/gen_sandwich_gadget.rs` |
| 3–5 | `fmix` | `db_mixing/bin/fmix.rs` |
| 6 | `fcompress` | `postprocessing/bin/fcompress.rs` |

`fmix` owns three modes: Phase A (`--gss --phase-a`), splitting
(`--split --split-stop`), and the resumed crossing walk (`--resume`). The
production splitting algorithm is implemented in `postprocessing/splitting.rs`;
the production crossing/undo/merge walk is implemented in
`postprocessing/cross_walk.rs`. Shared scheduling, arena operations, provenance,
and checkpoint state remain in `engine/mix.rs`. The standalone `fsplit` program
is experimental and lives outside `src/`.
Run the supported pipeline with `cargo run --release -- gss`; the command reads
the editable marked block in `docs/GSS_MIX.md` and delegates stage execution to
the script.

## Folder responsibilities

| folder | responsibility |
|---|---|
| `circuit/` | Shared `CircuitSeq`/`XGate` types, evaluation, canonicalization, randomization, and generic wide-gate lowering. |
| `preprocessing/` | Stages 1–2: sliced-sandwich construction and gadgetization. |
| `db_mixing/` | Frozen-store lookup and the stage 3–5 `fmix` entry point, plus maintained legacy mixing workflows. |
| `engine/` | Shared equivalence-walk state, rules, formats, ANF windows, statistics, checkpoints, and ancestry. |
| `postprocessing/` | Split/cross ownership and final compression; production split/cross execute through `fmix`. |
| `db_generation/` | Reproducible regular/curated DB construction, freezing, filter generation, and validation. |
| `commands/` | Handlers for the maintained general-purpose `local_mixing_bin` CLI. |
| `experimental/` | Reusable non-authoritative library implementations, including graph polycanon and the standalone float-and-split engine. |

The PyO3 heatmap implementation and all red-team executables now live under
`red_team_tests/`. Benchmark programs live under `benchmarks/`; standalone
probes and alternative workflows live under `experimental/`; manual artifact
validators live under `tests/manual/`.

Cargo has `autobins = false`, so every executable is explicitly classified in
`Cargo.toml`. Moving tools out of `src/` did not change their binary names.

## Important ownership distinctions

- `circuit/wide_fragment.rs` is a semantics-preserving lowering of one wide
  controlled-XOR gate into narrow gates using restored dirty helpers. It is not
  stage-4 circuit splitting. Its standalone wrapper is
  `experimental/circuit/fragment_wide.rs`.
- `postprocessing/splitting.rs` is the production Stage-4 implementation called
  by `engine::Mixer`.
- `postprocessing/cross_walk.rs` is the production Stage-5 directional
  cross/undo/merge implementation called by the shared `Mixer` scheduler.
- `experimental/split_engine.rs` is the older standalone split algorithm used
  by experiments and tests. GSS does not call it.
- `experimental/poly_canon_graph.rs` is retained for comparison and known-
  failure regression tests. `circuit.rs` canon4 remains the database-key ABI.
- `db_generation/bin/` contains only the four core rebuild/publish tools. DB
  probes and censuses live under `experimental/db_generation/`.

## Major move history

| old | current |
|---|---|
| `src/postmix/{mix,arena,rules,xpoly,format,swap_words,stats,tests}.rs` | `src/engine/` |
| `src/postmix/xgate.rs` | `src/circuit/xgate.rs` |
| `src/postmix/fragment.rs` | `src/circuit/wide_fragment.rs` |
| `src/postmix/engine.rs` | `src/experimental/split_engine.rs` |
| `src/postmix/compress.rs` | `src/postprocessing/compress.rs` |
| `src/replace/gadgets.rs` | `src/preprocessing/gadgets.rs` |
| runtime files from `src/replace/` | `src/db_mixing/` |
| DB builders from `src/rainbow_table/` and `src/replace/frozen_build.rs` | `src/db_generation/` |
| `src/random/random_data.rs` | split between `src/circuit/randomize.rs` and `src/db_mixing/convex.rs` |
| `src/circuit/poly_canon_graph.rs` | `src/experimental/poly_canon_graph.rs` |
| auxiliary GSS CLIs | `experimental/` or `tests/manual/gss/` |
| research `src/bin/*.rs` | `experimental/`, `benchmarks/`, or `red_team_tests/bin/` |
| `src/heatmap.rs` | `red_team_tests/heatmap.rs` |

Compatibility re-exports preserve selected older Rust paths, but new code
should use the current ownership paths above.
