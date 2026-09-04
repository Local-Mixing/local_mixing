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

### Compute stage: the LGI (blinded-V5) alternative

The default stage-2 compute is the drip `route_fire` gadgetizer. **Blinded-V5**
([`preprocessing/blinded_v5.rs`](preprocessing/blinded_v5.rs)) is a drop-in
alternative for *only the compute part*: it takes the `n`-wire sliced sandwich
`A` and emits an equivalent `2n`-wire circuit whose body is one very long
**locally-geodesic-identity (LGI)** with `A` embedded inside it. The rest of the
5-step gadgetization (junk-guard slice guards as modules 1 and 5, band seed as
module 2) is unchanged.

The core ideas: long masking identities are the only thing that moves the state
substantially (HD/affine heatmaps); the only long identity we know that actually
moves the state is the commuting run of a **single-active-wire** identity;
identities on different active wires are **entangled** by sharing one **band** of
control wires (they still commute because they target data and read band); random
**control-wire updates** from live data then re-randomise the band and block
naive commutation-back (STRADDLE closes straddling masks; REPAIR re-derives them
across the update with no thinning); and `A` is embedded via a **hidden
unmasking** of control wires at each gate — the operand is spread over band wires
(never bare), the gate fires as an expansion over the masked wires, then re-masks.
The LGI masks, the band updates and the A-gate placements are **co-sampled** in
one pass, so every gate's firing is hidden — its fire (split into two
independently-shuffled halves) straddles one of its active wire's LGI opens,
making the module's net XOR `Δ ⊕ secret-mask` rather than the bare gate increment
— at no size cost, reusing the wire's existing LGI budget. Each operand read is
held to a **masking floor** (`|ρ| ≥ min_mask`, auto = `max_open`), never bare and
never thinly masked. Band re-randomisation is emitted in **bursts** (one slot =
`F ≈ 8K` gates on a single band wire), rate-calibrated with no end-flush, so the
band mixes hard in few slots and carries a data-wire-like activity signature.

Production preset: `K=2` (band wires per LGI; affine/deg-2-neutral across K, so
smallest wins — read cost is quadratic in `max_open·K`), `max_open=3`, rerand
auto (`≈ m/4K` straddle slots × `F=8K`, no repair). Drive it through the pipeline with
`gss_mix.sh --gadgetization-mode blinded-v5 --bv5-k K`, or build the gadget alone
with `gen_sandwich_gadget … blinded-v5` / the `blinded_v5_gadgetize` bin. Full
rationale, parameters, and measurements:
[`docs/BLINDED_V5_LGI_DESIGN.md`](../docs/BLINDED_V5_LGI_DESIGN.md).

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
