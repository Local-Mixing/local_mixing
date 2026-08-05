# Split-stage trials — 2026-08-05

First trials of the phase-B split stage (docs/FMIX_SPLIT_TWIST.md) on local
phase-A-descended material. No true phase-A output lives on this machine
(cg1_phaseA5 is server-side), so the stand-ins are the gadgetized phase-A
input and the July two-phase pipeline outputs — all in the g57 + X-series
closure the stage requires.

All runs: `--split --split-stop --p-join 0.8 --split-fail-limit 100
--split-canaries 256 --p-db 0 --p-comp 0 --p-any 0 --k-max 12`, local_verify
ON, FMIX_STOP_FLAG/FMIX_DUMP_FLAG exported. Binary: ssg-gen-mix-clean @ the
split-twist commit. Every run ended by **exit A (g57 exhaustion)**; the
failure limit was never approached (fails=0 in all three).

| input | gates | g57 | seed | moves | prims/hspl/segs | joins | end size | growth | time |
|---|---|---|---|---|---|---|---|---|---|
| new_sliced_sandwich_gadgetized_n128 | 99,016 | 5,120 (5.2%) | 71 | 1,462 | 1462/887/2685 | 1,184 | 106,000 | ×1.07 | 1.5s |
| cg1_phaseB (slow anneal) | 1,020,340 | 705,980 (69.2%) | 72 | 5,495 | 5495/3287/696904 | 4,402 | 1,731,515 | ×1.70 | 129s |
| cg1_phaseB1Mfa (fast anneal) | 1,016,535 | 163,937 (16.1%) | 73 | 3,788 | 3788/2327/157740 | 3,055 | 1,184,051 | ×1.16 | 74s |

Headlines:

- **Growth matches the prediction band**: ×1.70 on the 69%-g57 input — the
  pair split alone accounts for +69%, cross ladders barely register. Well
  under the ×2–3 ceiling.
- **Segment splits do almost all the work**: 98.7% of g57 depletion on the
  big run came through 5a (the twist's path), not the primary picks. The
  stage is a few thousand moves of large sweeps, not a per-gate crawl.
- **The bracket cascade never fails** (0 failures anywhere): every split
  mints a permanent 1-control bracket, so the population only improves. Exit
  is always by exhaustion. Consequently `xmid == joins` everywhere — an
  other-half bracket essentially always exists, so the midpoint counter
  saturates by construction and the canaries are the real reach instrument.
- **Canary flip profiles (mean flips by ORIGINAL position decile)**:
  - gadg99k:  1.1 2.1 2.9 2.2 2.5 2.3 2.2 2.9 2.2 2.0
  - cg1 slow: 5.2 7.3 6.8 8.8 9.3 7.9 7.4 7.4 6.5 4.4
  - cg1 fast: 2.9 4.1 4.9 5.4 6.1 5.5 5.2 5.0 4.4 3.2
  Mid-heavy with symmetric edge falloff — exactly the 2x(1-x) coverage of a
  span between two near-uniform endpoints. No clustering near any origin;
  reach is absolute up to the unavoidable end effect. If flatter EDGE
  coverage is wanted, the lever would be biasing bracket choice toward the
  ends, not more twists.
- Function preservation verified: per-rewrite local_verify on every split /
  absorption / pin flip, global_check at the stage boundary, final float
  verified on every run.

Files: gadg99k_split.*, cg1_1M_split.*, cg1fa_1M_split.* (.mpmct1 + .state),
logs *.log with per-move `[fmix] split` lines and per-canary dumps.

## Round 2 — true phase-A outputs, p_join = 1 (user request)

Two phase-A outputs pulled from the second server (inputs/), run with
`--p-join 1.0` (every split carries the twist + cross), other knobs as above:

| input | gates | comp | seed | moves | segs | end size | growth | time |
|---|---|---|---|---|---|---|---|---|
| cg1_phaseA5 | 64,782 | 47,626 (73.5%) | 75 | 2,713 | 42,849 | 115,629 | ×1.78 | 4s |
| g57A/phaseA (pure-g57 recipe) | 1,664,636 | 1,389,686 (83.5%) | 74 | 4,818 | 1,381,051 | 3,060,082 | ×1.84 | 293s |

With p_join=1: joins == prims exactly (zero bracket failures anywhere), and
growth ≈ 1 + comp-fraction — the pair split IS the growth; cross ladders are
net-neutral. Segment splits carried 99%+ of the depletion. Canary deciles
keep the mid-hump shape (pA5: 2.7…5.4…2.6; g57A: 5.7…9.8…5.9).

**Gate-makeup census** (gate_census.py, input → output):

- cg1_phaseA5: comp 73.5% (48.6 opp + 24.9 same) → **0**; 1-ctrl 13.5% →
  48.5% split 24.1/24.4 CNOT/NCNOT; AND2 4.0% → 42.4% in a 1:2:1
  neg-count ratio (12418/24281/12342); X unchanged in absolute count
  (5023); small w3/w4 ladder tail (+4.4%).
- g57A: comp 83.5% → **0**; 1-ctrl 3.9% → 47.5% at 23.8/23.7; AND2 2.1% →
  46.5% at 1:2:1 (355k/713k/355k); X unchanged (35,911); the wide-conj
  tail (w4–w32, ~9% of input) unchanged in count — those gates only take
  pin flips.

The 50/50 CNOT/NCNOT balance and the binomial 1:2:1 AND2 polarity split are
the direct signature of the absorbed-NOT polarity churn composed with the
r-coin split — the anti-inversion texture the stage is for.

Note: the g57A input carries wide conjunctions up to w32; the per-gate
exhaustive verify is capped at width < 16 (verify_rewrite's support
envelope), wide flips being covered by global_check — cap added after the
first g57A attempt tripped the support assert.

## Round 3 — nR20_mixed (user request), post-review binary

`nR20_mixed` (from the second server's bigmix dir) is the cleanest phase-A
output tried: 179,132 gates, **99.3% true g57** (opp-pol) + 0.7% w3 — the
census found NO CNOT/NCNOT in it, so the stage minted its own brackets from
split pieces (as designed; joins never failed).

Run (p_join 1, seed 76, fixed binary): 3,789 moves, 22.4s,
179,132 → 361,123 (**×2.016** — the pure-g57 limit), comp → 0, joins =
prims = 3789, fails = 0, segs = 171,254. Canary deciles:
2.7 4.8 5.0 7.2 8.1 6.4 7.1 5.5 4.3 3.5.

Output census is the theoretical signature exactly: CNOT 24.6% / NCNOT
24.6% (dead-even polarity churn), AND2 12.2/24.3/12.3 (1:2:1), ladder tail
w3–w6 ≈ 2%, comp 0.

Figure: `canary_flips_by_position.png` (`plot_canary_flips.py`) — mean ±1 std
of flips per canary by original-position decile, individual canaries behind,
for the three p_join=1 phase-A runs. All three show the same shape: a broad
mid-circuit plateau at roughly 2× the edge coverage, std comparable to the
mean (spans are few and long, so per-canary counts are overdispersed), and no
asymmetry — reach is absolute up to the unavoidable span-geometry end effect.

## Round 4 — bracket-draw redesign A/B (span histograms)

The original cascade's hard other-half preference made xmid a constant 100%
(overshoot). New default: DIRECTIONAL max-of-k draw — brackets on the picked
g57's own side (its stored direction), farthest of `--split-reach-k` uniform
samples wins (k=1 uniform, 2 ≈ 2/3 of the available run, 3 ≈ 3/4; 0 keeps
the original cascade as the A/B arm). Spans now logged per twist
(`span=` on every split line, 5%-bucket histogram + mean at the boundary).

nR20_mixed, p_join 1, seed 76, both arms to exhaustion
(`span_compare_nR20.png`, `plot_span_compare.py`):

| arm | moves | joins | fails | mean span | xmid | segs/join |
|---|---|---|---|---|---|---|
| cascade (k=0) | 3,811 | 3,811 | 0 | 0.77 of circuit | 100% | 44.9 |
| directional k=2 | 6,417 | 6,280 | 137 | 0.34 of circuit | 41% | 27.0 |

Readings:
- The CASCADE's spans are extremely long (mode 95–100%!): survivors of the
  segment sweeps cluster at the circuit edges, and edge-picked primaries
  paired with other-half brackets give near-full-circuit spans —
  self-reinforcing as the stage proceeds.
- The DIRECTIONAL draw lands mean 0.34 ≈ (2/3) × E[available run 0.5] —
  exactly the max-of-2 design value — with a flat body and an honest
  midpoint-crossing rate of 41%. Failures are now real (137, all when the
  picked g57's own side holds no bracket) but nowhere near the limit.
- Cost of shorter spans: fewer segment splits per twist (27 vs 45), so the
  stage takes ~1.7× the moves — still 44s total.
- The 0–5% span spike (28% of twists) is g57s picked near their direction's
  edge, where the available run is short. If unwanted, the dials are
  --split-reach-k 3 (≈3/4 of the run) or a min-span floor; both trivial.
- Canary deciles under k=2: 2.5 3.7 3.7 4.5 6.6 5.4 4.9 4.1 3.6 2.8 —
  still mid-humped, slightly flatter-tailed than the cascade's.

**v3 (final, user decision): side drawn ∝ remaining length.** The twist
direction is no longer the g57's stored direction — it is drawn with
probability proportional to the circuit length remaining on each side, so a
short side is picked exactly as rarely as it is short (squared suppression
of tiny spans). Same seed, nR20:

| arm | mean span | 0–5% bucket | xmid | fails | moves |
|---|---|---|---|---|---|
| cascade (k=0) | 0.77 | 0.03% | 100% | 0 | 3,811 |
| own-direction k=2 | 0.34 | 27.8% | 41% | 137 | 6,417 |
| **∝-length k=2 (shipped)** | **0.57** | **2.9%** | **71%** | **1** | 5,357 |

The shipped arm's distribution is a broad gentle ramp toward long spans with
a soft top-end rolloff — no spike at either extreme. Canary deciles recover
to 3.4 5.6 6.0 8.4 9.4 7.7 7.5 6.6 5.3 3.4 (mean 6.3, the strongest edge
coverage of the three arms). The mean exceeding the naive 4/9 comes from the
survivor dynamic: late-stage primaries sit near the edges, and the
proportional coin then sends them across most of the circuit.

## Post-review fixes (same day)

The adversarial review over the diff confirmed and we fixed: (1) canary
eviction order in splice_pair and the merge splice — when both partners die
(R2 rewrites, merges), the right partner's canaries could re-anchor onto the
dying left partner and end up on a recycled slot (silent canary teleport);
evictions now run after the prior partner's unlink. (2) Resume semantics:
the state file now records the stage phase as a tri-state (never/live/ended),
so --split on a resumed non-split state arms the stage, an ended stage never
re-arms, and a live stage missing --split warns loudly. (3) The exhaustion
exit is gated on the stage being live (standalone --p-split-twist dispatch no
longer prints stage banners). (4) taps_reported persists (no duplicate canary
dump on resume). (5) After a --split run's boundary the live split-twist
dispatch is pinned to 0 regardless of --p-split-twist. Rounds 1–2 above were
re-run on the fixed binary (seed-identical trajectories; canary hygiene now
exact).
