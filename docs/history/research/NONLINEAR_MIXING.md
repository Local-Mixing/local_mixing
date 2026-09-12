# Nonlinear mixing upgrade

## Motivation

The compact-300 degree-1 heatmaps showed that the existing SSS/fmix pipeline
reduces the affine reconstruction ridge but does not remove its chronology.
This is expected: wire permutations, NOTs, CNOT transvections, and SAMFs are
affine changes of basis, so a degree-1 predictor can absorb them. Local
nonlinear rewrites can enlarge the circuit without keeping a nonlinear state
encoding alive across a meaningful prefix interval.

This upgrade adds persistent, low-degree nonlinear state frames to both the
legacy all-G57 SSS path and heterogeneous `fmix`, while retaining exact
functional equivalence and leaving all historical behavior disabled by
default.

## 1. Legacy SSS: conservative nonlinear handles

After the last G57 compression, SSS can now insert nonlinear handles. A handle
is one self-inverse G57 gate `P = [target, positive, negative]`. Its Boolean
action contains a genuine quadratic term. For a window `W` in which every gate
commutes with `P`,

```text
P ; W ; P = W.
```

The circuit boundary function is unchanged, but every state inside the window
is encoded by `P`. The pass samples windows throughout the final circuit,
scores several random carriers, truncates at the first true dependency
collision, and accepts only windows meeting the minimum span. Multiple handles
may overlap, giving nonlinear carrier depth without requiring a large gate
increase: every accepted handle adds exactly two G57 gates.

For fixed-slice Feistal experiments, the CLI defaults both targets and controls
to the original input block `0..n`. This prevents a handle from becoming a
constant-only auxiliary-wire effect when public `y,z` are fixed. Explicit wire
limits can override this.

This mode is deliberately conservative. It does not claim to conjugate a G57
carrier through a real collision. Manufacturing an exposed `P;g;P` triple
would merely add decoder checkpoints, not genuinely transport the carrier.
Collision-capable nonlinear transport is delegated to the verified XGate
R-rule machinery in `fmix`.

### SSS flags

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--nonlinear-handles` | `0` | Handle attempts; zero is a strict no-op |
| `--nonlinear-handle-min-span` | `64` | Smallest accepted clean window |
| `--nonlinear-handle-max-span` | `0` | Requested span cap; zero searches to the boundary |
| `--nonlinear-handle-candidates` | `64` | Carriers scored per attempt |
| `--nonlinear-handle-max-gates` | `0` | Optional hard final gate ceiling |
| `--nonlinear-handle-seed` | `2026072201` | Deterministic selection seed |
| `--nonlinear-handle-target-wires` | `0` | `0` means original input width |
| `--nonlinear-handle-control-wires` | `0` | `0` means original input width |

The `[sss:nl]` report records attempts, candidate evaluations, accepted and
rejected handles, collision truncations, requested and delivered gate-visits,
carrier depth, span statistics, target coverage, and added gates. Generation
tags remain aligned when new-SSS tracking is enabled.

A reasonable first n=128 sweep is 256, 512, and 768 handles with minimum spans
64 and 128 and 64/128 candidates. These are experimental doses, not established
optima.

## 2. fmix: persistent nonlinear packets

One nonlinear-frame move constructs a packet

```text
P = p1 ; p2 ; ... ; pk
```

of active low-degree XGates. Each `pi` has two or three controls, so it fires on
roughly 1/4 or 1/8 of random states instead of being a nearly inert wide
conjunction. Since every XGate is self-inverse,

```text
P^-1 = reverse(P).
```

The move inserts the exact identity `P ; reverse(P)`. It then sends tagged
material from the two halves in opposite directions. Every transport step is
an existing locally verified R1/R2/R3 crossing (with required G57 presplits
tracked separately), so no new unproved rewrite rule is introduced. The
resulting descendants form a spatially distributed nonlinear state frame.

Frame identity and a protection deadline propagate through crossings,
presplits, fresh/unsubsume splits, CNOT-twist case splits, merges, and journal
restoration. Before the deadline, catalogue merges and journal undos involving
any frame descendant are refused. Other exact rewrites remain allowed, so the
frame can keep diffusing rather than becoming frozen material.

### fmix frame flags

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--w-nl-frame` | `0` | Relative expansion-move weight; zero is inert |
| `--nl-frame-min-width` | `2` | Minimum controls per packet gate |
| `--nl-frame-max-width` | `3` | Maximum controls per packet gate |
| `--nl-frame-packet-gates` | `16` | Gates in `P` (twice this many inserted initially) |
| `--nl-frame-shots` | `64` | Total transport attempts per frame |
| `--nl-frame-tenure` | `100000` | Moves before destructive contraction is admitted |

Reports separate frame attempts, accepted frames, skips, inserted packet gates,
actual R crossings, preparatory presplits, blocked shots, delivered span and
descendant counts, protection blocks, live tagged coverage, and the still-
protected subset.

## 3. Grow first, then churn

`fmix` now supports an explicit two-phase schedule on one continuous `Mixer`:

```text
--grow-moves G --churn-moves C
```

This conflicts with legacy `--moves`. The circuit arena, RNG stream,
provenance, journal, tabu state, frame metadata, and counters all survive the
phase boundary. The second stop is cumulative (`G+C`). Optional `--churn-*`
flags override the target, temperature, merge/undo/tabu controls, every move
weight, and frame geometry; unspecified values inherit.

The intended use is:

1. grow efficiently to the desired gate count with the established moves;
2. lower insertion pressure at the target;
3. spend a substantial fixed-size budget on crossings and nonlinear frames;
4. retain long enough tenure for independently placed frames to overlap.

An initial *calibration* profile is:

```text
fmix --input INPUT --output OUTPUT --target-size TARGET \
  --grow-moves G --churn-moves C \
  --w-nl-frame 0 \
  --churn-w-cross 0.55 --churn-w-fresh 0.15 \
  --churn-w-unsub 0.20 --churn-w-insert 0.10 \
  --churn-undo-frac 0.10 \
  --tabu-moves 100000 --churn-tabu-moves 100000 \
  --churn-w-nl-frame 0.0003 \
  --nl-frame-min-width 2 --nl-frame-max-width 3 \
  --nl-frame-packet-gates 16 --nl-frame-shots 64 \
  --nl-frame-tenure 100000
```

Sweep `--churn-w-nl-frame` over approximately `1e-4`, `3e-4`, and `1e-3`
before choosing a production value. Packet count, protection blocks, live
coverage, gate growth, wall time, degree-1/degree-2 heatmaps, and paired SAT
times must all be considered. A high frame weight can deliberately overwhelm
the thermostat because protected material is temporarily unavailable to the
contraction channels.

## 4. Verification and compatibility

- `w_nl_frame=0` produces no nonlinear attempts and preserves the historical
  RNG trajectory, including low-`k_max` configurations.
- Every inserted SSS handle is exhaustively checked in randomized small-circuit
  tests and uses the exact legacy collision predicate.
- Every fmix packet begins as a structural `P;reverse(P)` identity; transport
  uses the already verified R-rule catalogue.
- Sampled global equality checks remain active during both phases.
- Frame tenure blocks only destructive contraction; it is not a timeout and
  does not stop the mixer.

Integrated validation on 2026-07-22:

- 7 nonlinear-handle tests passed, including exhaustive/random equivalence,
  generation-tag alignment, cap transactionality, collision boundaries, and
  distributed starts;
- 15 fmix mixer tests passed, including exhaustive nonlinear-frame equivalence,
  tenure/journal protection, and a zero-weight deterministic regression;
- 7 fmix CLI/schedule tests passed;
- all 204 library tests ran with 201 passed and 3 pre-existing tests ignored;
- `cargo check --all-targets` and `cargo fmt --check` passed.

## 5. Small heatmap smoke test

A deliberately tiny 1,000-gate, 128-wire smoke test used 200 growth plus 200
churn moves. The nonlinear run used an intentionally excessive frame weight
`0.5`, four gates per packet, 16 shots, and tenure 1,000. It is a mechanism
test, not a production parameter recommendation.

| Circuit | Gates | Ridge depth | Contrast |
| --- | ---: | ---: | ---: |
| Original self-control | 1,000 | 0.4465 | 4.327 |
| Two-phase fmix control | 1,317 | 0.4142 | 4.316 |
| Two-phase fmix + nonlinear packets | 1,501 | 0.4079 | 4.175 |

The nonlinear run reduced ridge depth by `0.0063` (`1.52%`) relative to the
small fmix control and reduced contrast by `3.27%`. Eleven live frames covered
`0.850` of the circuit by the delivered outer-span gauge. The same run also
produced 1,021 protected-contraction blocks and 14% more gates than control,
which is exactly why production weights should be orders of magnitude lower.
The two runs do not share an identical move trajectory after the nonlinear
weight changes, so these numbers establish proof of mechanism, not an unbiased
effect size.

The next meaningful experiment is the canonical 384-wire fixed-y/z compact
circuit with paired degree-1 and degree-2 heatmaps, build/RSS telemetry, and
Painless SAT targets. No SAT-hardness improvement is claimed from this smoke
test alone.

