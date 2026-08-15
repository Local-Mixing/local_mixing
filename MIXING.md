# Mixing

Mixing is the second obfuscation stage: after gadgetization has spread
every logical value across shares, the **circuit text itself** is
re-randomized by a long walk of semantics-preserving local rewrites. The
goal is that an adversary holding the final circuit cannot align it with
the pre-mixing circuit — cannot tell which gates belong to which original
gadget, where one gadget ends and the next begins, or which wires carry
which shares.

## What mixing is (and is not)

Mixing is a **move-based random walk** over equivalent circuits. Each
move rewrites a small window of the circuit using a local identity of the
eca57 calculus (collision/commutation rules, expansion, replacement
against a frozen database of circuit fragments, twists that absorb
neighboring gates, crossings that transport gates across each other).
Because every move preserves the computed function, the output is correct
by construction; because the walk is long, the output's *spelling* is
statistically decoupled from its input.

Mixing is **not** the hiding mechanism by itself — a mixed but
ungadgetized circuit still exposes every logical value on single wires
(the mixer just permutes which wires). Hiding is gadgetization-borne;
mixing destroys the *residual structure* (gadget boundaries, periodic
layout, fixed wire roles) that an attacker would use to locate and align
the shares. The complete piece-by-piece reference — every move family,
its parameters, and the theory of the eca57 gate and its rewrite rules —
is **`docs/Mixing_Pieces_Documentation.md`**.

## The production recipe (phases)

The full pipeline with every knob and its calibration rationale is
`docs/PIPELINE_OVERVIEW.md`; the runnable packaging is
`scripts/gss_mix.sh` (see `docs/GSS_MIX.md`). In brief:

1. **Phase A — DB re-encoding mix** (`fmix --gss --phase-a`).
   Expansion/contraction walk whose moves re-spell windows of the circuit
   against a frozen replacement store (MIX moves may grow the window,
   COMP moves strictly shrink), driven by a size profile
   (expand → hold → compress). g57-preserving: its job is re-encoding
   depth and spelling diversity. Deep dives: `docs/FMIX_PHASE_A.md`,
   `docs/FMIX_LAYER1.md`, `docs/FMIX_LAYER2.md`, `docs/FMIX_MENU.md`.
2. **Phase B — structure breaking.** First a *split stage* presplits
   random g57s into (CNOT/NCNOT, 2-control AND) pieces with absorbed
   pure-NOT twists, then a *crossing walk* transports gates across the
   circuit under a thermostat until the size equilibrium ("arrival").
   Its job is anti-inversion: absolute spread of each input gate's
   descendants. See `docs/FMIX_SPLIT_TWIST.md`,
   `docs/G57_TWIST_BRACKETS.md`, `docs/SPLIT_TWIST_REPORT.md`.
3. **fcompress — the honesty check.** The attacker-computable greedy
   compressor (gather → group-cap → ANF reduce) is run over the whole
   output; the residual size is the earned incompressibility (healthy
   mixed material ≳ 90% residual vs ~83% for raw split artifacts).
   Manual: `docs/POSTMIX_MANUAL.md`.

## Mixing in the testing pipeline

In the gauntlet, mixing is an **optional in-process stage**
(`gauntlet_gen.rs --mix`): after gadgetization, the trace-circuit is
randomized with the same CIFY/commutation machinery so the audit battery
runs against *mixed* gadget outputs — this is what catches attacks that
only become possible when gadget boundaries blur (e.g. the mixer's
demotion of the 193-gate gadget's weight-3 mask-cascade leak to weight 2;
see `TESTING_PIPELINE.md` §4 and `GADGETIZATION.md`).

## Reading list

| doc | content |
|---|---|
| `docs/Mixing_Pieces_Documentation.md` | the pieces reference: eca57 calculus, every move family |
| `docs/POSTMIX_MANUAL.md` | the `fmix`/`fcompress` tools, options, workflows |
| `docs/PIPELINE_OVERVIEW.md` | the production pipeline end-to-end, all parameters |
| `docs/GSS_MIX.md` | the packaged `scripts/gss_mix.sh` recipe |
