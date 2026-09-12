# NONLOCAL_PHASE_A — far-pair fusion and bridge fusion

Two non-local replacement moves, both experimental and default-off
(`--p-pair 0`, `--p-bridge 0`). Implemented 2026-08-15.

- **Far-pair fusion** (`--p-pair`): fuse the seed with a far COMMUTING
  partner into one 2-gate window. Free transport, g57-preserving; reach is
  bounded by the seed's commutation box.
- **Bridge fusion** (`--p-bridge`): jointly re-encode two gates that
  commutation CANNOT bring together, by conjugating the interior through a
  carrier — the interior gates are ADJUSTED to compensate (the nonlinear
  analog of a SAMF insertion), and both carrier-adjacent windows are
  re-spelled through the store. Reach is unbounded by colliders; the cost is
  a small non-g57 wake.
Companion docs: `POSTMIX_MANUAL.md` §2.1 (the DB move), `FMIX_MENU.md`
(knob catalogue), `ANCESTRY_INSTRUMENTATION.md` (span/litter meters),
`NONLINEAR_MIXING.md` (the complementary nonlinear-frame direction).

## Why

Phase A's two window geometries are both dependency-bound:

- **contiguous** takes the seed plus its physical neighbors — commuting
  gates only pair at distance 1;
- **convex** gathers by floating the block to its next **collider** and
  absorbing it — a gate that *commutes* with the block floats past and can
  never join the window.

So a window holding two mutually commuting gates that started far apart —
the "tensor-product window", whose function is the parallel composition of
its two halves — is structurally unreachable today. Meanwhile transport is
the standing phase-A gap (`FMIX_MENU.md` Open: "span is set by window
length, so s_db is the transport lever, not run length"), and the n64
poststage bundle showed the transport stage is what reduces short
whole-trace witnesses.

The far-pair move adds exactly that geometry, using only machinery the
mixer already trusts: pairwise-commutation relocation ("floating") for
transport, and the ordinary store lookup/splice for re-spelling. It is
g57-preserving by construction — the store emits g57 words and floats do
not rewrite gates — so it is legal inside the phase-A profile.

What it does **not** claim: per-move hiding. A single pair splice is as
search-reversible as any single local splice; old firings remain short
XORs of new deltas at move granularity (they do for every
function-preserving rewrite — the whole-trace span survives the entire
pipeline, `reports/global_trace_affine_postmix_20260812`). The payoff
claimed is **transport geometry**: litter unions across two commutation
boxes per move instead of one window span, i.e. heavy-tailed jumps in the
ancestry graph where the current kernel diffuses. That claim is measurable
and the acceptance test below is the measurement.

## The move

With probability `p_pair` (checked before the convex/contiguous coin; the
coin is not drawn when `p_pair = 0`, so inert configs are bit-identical to
builds without the move):

1. **Seed** `g1` via the normal seed pick (generation-pool bias applies).
2. **Box scan**: walk from `g1` in its stored direction, read-only,
   collecting gates `g1` commutes with (`!collides`), stopping at the
   first collider or after `far_scan_cap` gates. The scan is the same
   predicate the convex sampler floats with; nothing moves yet.
3. **Partner**: pick `g2` from the box — the farthest gate (`--far-pick
   far`, default: maximum transport per move) or uniform
   (`--far-pick uniform`).
4. **Fuse**: relink `g1` adjacent to `g2`. Every position crossed was
   verified commuting during the scan, so the relocation is
   function-preserving for the same reason every float is.
5. **Replace**: hand the contiguous `[g1, g2]` window to the unchanged
   descent → canonicalize → guards → curated-first lookup → mode rules →
   splice + verify path. Litter union (the two parents' litters merge into
   the fresh litter) and generation stamping come free from the splice
   code. Under a descent-armed mode a rung-2 miss descends to rung 1 as
   usual — `g2` is dropped (the gate farther from the seed) and `g1` alone
   is re-spelled *at the remote site*, which is still transport; under the
   GSS MIX profile (descent off) the round is simply a miss.
6. **Miss**: if the whole descent misses, `g1` is relinked back to its
   recorded origin (the pair-move mirror of `restore_seed`).

**Permutation ban.** For a commuting pair the reversed order `[g2, g1]`
is a distinct gate sequence computing the same function, so it passes the
gate-for-gate identity ban while being an attacker-trivial reorder (the
same hole behind ssg's measured 79.6% trivial identity/reorder rate).
Pair windows therefore reject any candidate that is a permutation of the
outgoing pair.

## Knobs

| knob | default | meaning |
|---|---|---|
| `--p-pair` | 0.0 | probability a non-COMP DB round uses the pair geometry |
| `--pair-scan-cap` | 4096 | max gates scanned for the commuting box |
| `--pair-pick-uniform` | off | pick the partner uniformly from the box instead of farthest |

COMP rounds never draw the pair coin: COMP admits only non-growing
spellings, and with both bans armed a commuting pair has no admissible
same-length spelling, so a COMP pair round could never splice.

## Meters

One `[fmix] pair` report line, printed only while `p_pair > 0`: rounds that
drew the geometry, fused windows, splices, reorder-ban rejections
(`permskip`), empty boxes (no eligible partner before the first collider),
scan truncations at the cap, and mean/max box length and fused transport
distance. Counters are session-local and report-only; state files are
unchanged and resume is unaffected.

## The bridge move — joint replacement across a causal wall

The pair move declines any partner past the seed's first collider. The
bridge move exists for exactly those pairs — two gates that CANNOT be made
adjacent through commuting swaps.

**The algebra.** Write the circuit as `A · g1 · M · g2 · B` with `M` the
interior. Any correct two-site rewrite is `X = g1·P` at the left site and
`conj_M(P⁻¹)·g2` at the right, for a free choice of `P` — that is the
entire solution space. The bridge takes `P` = one 2-control conjunction
carrier `u` (fires on a single monomial `m`) and realises the conjugation
by adjusting the interior:

```
g1·M·g2 = (g1·u) · (u·M·u) · (u·g2),      u·M·u = Π  (u·hᵢ·u)
                                                    i
```

Each interior gate's exact conjugate `u·h·u` is `[h, correction(s)]`
(`conj_wake`):

- `h` commutes with `u` (the overwhelming majority at production widths):
  unchanged, zero cost;
- `h` READS `u`'s target (literal λ there, other literals L):
  `(λ⊕m)·L = λL ⊕ m∧L` — keep `h`, add `(t_h; m∧L)`;
- `h` WRITES one of `m`'s wires (other `u`-literal ρ): the net delta on
  `t_u` is `f_h∧ρ` — keep `h`, add `(t_u; c_h∧ρ)` (two corrections for a
  comp-1 `h`, since `f_h = 1⊕mon`);
- a contradictory correction never fires and is dropped — the conjugate is
  `h` unchanged (this is also why the separation exemption self-corrects);
- mutual collision (both modes at once): the round is refused (rare).

Every non-trivial conjugate is verified exhaustively over its own support
(`rules::verify_rewrite`) before anything mutates, and the telescoping
identity makes the whole move exact.

**The move.** Pick `g1` uniformly and a log-uniform interior length; `g2`
is the gate past the interior. The carrier is sited to COLLIDE with both
endpoints — `u` reads `t_{g1}` (its first control sits on it) and `g2`
reads `t_u` (chosen as `g2`'s least-read control wire) — and to minimise
interior collisions (its free control on a least-written wire). Both
endpoint windows `[g1, u]` and `[u, g2]` are PROBED against the store
before anything mutates (a miss leaves no trace); then the wake and the two
carrier copies are inserted, the far window is spliced first (a decline
there rolls back to the exact pre-insert circuit), and the near window
last. The result: both distant sites are consumed by one correlated joint
replacement — site 1 computes `g1·u`, site 2 computes `u·g2` — whose
halves only compose to the original through the shared carrier, across a
wall no commutation could cross.

**The trade.** Wake corrections are conjunction gates outside strict g57
form: `polf > 0`, and under the production ctrl-cap they are
window-ineligible until phase B splits them — the same trade the legacy
twist packets make, bounded by `--bridge-max-colliders` and metered
(`wake=`). Expected wake at production widths is ~1–2 gates per bridge for
spans of hundreds of gates. Run the move in a profile that tolerates
shaped material, or budget it explicitly.

| knob | default | meaning |
|---|---|---|
| `--p-bridge` | 0.0 | per-round probability of one bridge round |
| `--bridge-min-span` | 16 | log-uniform interior-length draw, lower bound |
| `--bridge-max-span` | 512 | log-uniform interior-length draw, upper bound |
| `--bridge-max-colliders` | 8 | refuse rounds needing more wake than this |

Meters (`[fmix] bridge`, printed while armed): rounds, committed, half
(near window missed after the far one spliced — the bare carrier stays,
still exact), rollbacks, probe misses, refused plans, tail-clipped walks,
interior length avg/max, colliders avg, wake gates inserted.

## Store coverage — measure before dosing

For a *disjoint* pair the window function factors, and whether the store's
candidate lists for product-form keys contain **entangled** spellings
(gates whose support graph connects the two blocks) is an open empirical
question. `far_pair_probe` answers it offline:

```bash
FROZEN_DB_DIR=... FROZEN_CURATED_DIR=... \
  far_pair_probe --input phaseA.mpmct1 --samples 20000 --scan-cap 4096
```

(`benchmarks/db_mixing/far_pair_probe.rs`; add `--g-format g57` for source-form inputs,
`--uniform` for the uniform partner policy.)

reporting: box-length distribution, fused-key store hit rate
(curated/regular), candidates per hit, entangled-candidate fraction,
free-vs-pay size mix. Run it on fleet phase-A material (the store does not
work over the laptop's USB mount). If the entangled fraction is ~0 the
move still buys litter-union transport, but the case for it weakens —
decide with data.

## Acceptance experiment (fleet)

Paired-seed arms on the standard GSS pipeline, phase A only, identical
profiles except `--p-pair` ∈ {0, 0.05, 0.15}:

- **primary**: median absolute ancestry span and frac(≥3 descendants) at
  matched move budget (`ANCESTRY_INSTRUMENTATION.md` conventions — median,
  not mean);
- **witness meter**: `sampled_trace_support` per-source-firing minimum
  witness support on the stage output (the decisive rule, applied after
  composition);
- **sanity**: blind entry-time AUC (should stay in the no-signal band);
  polf — exactly 0.000 on pair arms (that move is g57-preserving), and on
  bridge arms bounded by the wake meter (each wake gate is one shaped
  gate); fcompress residual (must stay ≳ 90%).

Adopt a nonzero default only if the primary moves at equal-or-better
witness meter and unchanged sanity row.

## Known properties

- **Seed displacement on a pair miss.** `collect_pair` floats the seed up to
  the partner distance (≤ `pair_scan_cap`, default 4096), but the shared
  `restore_seed` walk is bounded at `RESTORE_HOPS` (512). A pair round that
  misses with a partner farther than 512 leaves the seed partially advanced —
  function-preserving (every hop was a commutation), but under pool targeting
  a repeatedly-drawn stubborn seed can drift forward. Set `--pair-scan-cap
  ≤ 512` if that statistical artifact matters for an arm; the default favors
  reach.

## Deferred

- **Mobility-biased candidate choice** — among admissible spellings prefer
  the one whose boundary gates float farthest (feeds `db_advance` a longer
  ballistic step). Touches the hot candidate path; separate change.
- **Pair+collider (3-gate) windows** — superseded by the bridge move, which
  couples across colliders directly; revisit only if bridge store-hit rates
  disappoint.
- **Driver integration** — `gss_mix.sh` stays pinned; arms pass flags to
  the stage-3 binary directly until the acceptance experiment justifies a
  default.
