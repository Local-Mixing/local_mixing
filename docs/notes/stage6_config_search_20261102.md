# Stage 6 configuration search and the n = 128 run (2026-11-02, overnight)

RC (2026-11-02, before sleeping): "continue on your own to determine the best configuration for stage 6;
once done, run also a 128 bit pipeline for comparison." Code: branch `ran/stage6-deep` (worktree
`s12_wt`), not pushed. Stage 6 here = `tdp_adaptive`; "phase A/B/C" and "levels" are steps INSIDE
stage 6, not pipeline stages (see the vocabulary at the end).

## The yardstick

Every size below is in the deliverable's gate set: a gate with at most 2 controls counts 1, a gate with
k > 2 controls counts 4k - 8 (its Barenco ladder, the exact lowering stage 6's phase C applies). Raw
gate counts are misleading here: stage 5 leaves millions of gates with up to 12 controls, each counted
as 1 by a raw count. Every circuit reported is checked equivalent to its input (random full states), on
the zero slice against C, and against the sandwich's payload contract; "deeper a/d" = deeper births of
C affinely / directly exposed (validated census); "early outputs" = C's output bits affinely exposed
before 90 % of the circuit (a copy of C's output before its delivery region).

## n = 64 (local run `port_runs/tdp64_default`, recipe v13; stage-5 output 3,537,140 gates)

Stage-5 output: 3,537,140 raw gates, 2,132,189 of them wide, **2-control cost 17,334,663**; deeper 0/0,
early outputs 0.

| stage 6 | 2-control cost (published) | ratio to stage 5 | deeper a/d | early outputs | time |
|---|---|---|---|---|---|
| today's stage 6, after 4 rounds (not finished) | 5,738,495 | 0.331 | 0/0 | **3** | ~16 h |
| new, phase A first (raw cost) | 2,321,808 | 0.134 | 0/0 | 0 | 65 min |
| new, fcompress first (raw cost) | 1,863,815 | 0.108 | 0/0 | 0 | 31 min |
| new, fcompress first, 2-control cost, phase A cap 4,000 (stopped at round 6, dominated) | 1,673,521 at round 6 | 0.097 | 0/0 | | 55 min to round 6 |
| new, fcompress first, 2-control cost, phase A cap 512, descent 128 / 32 | **1,369,679** | **0.079** | 0/0 | 0 | 25 min (5 threads) |
| ... descent 128 / 32 / 16 / 8 (cut at the 8-round cap after one cap-16 round) | 1,358,309 | 0.078 | 0/0 | 0 | 26 min (5 threads) |
| ... descent 128 / 32, stop fractions 0.05 (vs 0.2) | 1,363,602 | 0.079 | 0/0 | 0 | 26 min (6 threads) |
| ... descent 128 / 32, two synthesizer seeds | **1,352,375** | **0.078** | 0/0 | 0 | 35 min (6 threads) |
| **FINAL: from the raw stage-5 output, `--pre-cleanup`, descent 128 / 32 / 16, two seeds, no round cap** | **1,332,800** | **0.077** | 0/0 | 0 | 41 min (6 threads) |
| ... the same with the production synthesizer effort (16 restarts, 60 s budgets instead of 2 / 12 s) | **1,253,943** | **0.072** | 0/0 | 0 | 2.35 h (6 threads) |

Bounded settings shared by these runs (to finish overnight): phase A block cap 4,000, one synthesizer
seed, 2 restarts, 12 s base budget, stop when two consecutive rounds compress jointly by less than 20 %
(phase A and each descent level), descent caps 512 / 128 / 32, 10 fcompress iterations per cleanup,
phase C to <= 2 controls, census batches 8, 12 threads.

Findings:

1. **The cheap exact cleanup (NOT-push + fcompress) is the dominant lever on a stage-5 output**: alone,
   3,537,140 -> 1,311,745 raw gates in 78 s, census unchanged (0/0, 0 early outputs). Today's stage 6
   spent ~16 h to reach 2,855,301.
2. **Clean first.** ANF rounds run before the cleanup rewrite blocks into the synthesizer's own form and
   destroy the local crossing-walk redundancy fcompress removes; cleaning after two ANF rounds only
   reached 2.04M raw. Fcompress-first beat phase-A-first by 20 % in half the time.
3. **Judge acceptance by the 2-control cost.** Under raw-count acceptance a <= 2-control resynthesis of a
   block holding wide gates is rejected as "growing" though it is cheaper once lowered: fcompress-first
   with raw cost synthesized 99 of 1,697 blocks in its first round, and 146k wide gates reached phase C,
   which inflated the circuit by 78 %. With `--cost two-control` the first round synthesized 1,381 of
   1,697 blocks and cut the 2-control cost 3,294,230 -> 2,243,137 (-32 %) in one round; 17 blocks were
   rejected as leakier by the guard.
4. **Start small.** A phase-A block cap of 512 beats 4,000 under the 2-control cost: its first round
   synthesized 2,627 of 2,823 blocks and cut the cost 47 % (vs 32 % at cap 4,000, which was stopped at
   round 6 at 1,673,521, already 22 % above the cap-512 run's final 1,369,679). The large-block level is
   not needed.
5. **Everything else is worth about 1 %:** a finer descent (adding caps 16 and 8) -0.8 %, stop fractions
   0.05 instead of 0.2 -0.4 %, two synthesizer seeds -1.3 %; combined in the final command -2.7 %.
6. **Synthesizer effort is a real size/time dial at small blocks:** the production 16 restarts / 60 s
   budgets give another -5.9 % (1,332,800 -> 1,253,943) for 3.4x the time. (Earlier campaigns found
   synthesizer options worth ~0.2 %, but with large blocks and raw-count acceptance.)
7. **Today's stage 6 creates the payload replicas RC described.** Its round-4 output carries 3 of C's
   output bits as verbatim single-wire copies (validated) at depth 0.485-0.526, the sandwich midpoint;
   the stage-5 input has none. The old acceptance excluded outputs as public. The new early-output term
   of the leak guard rejects them; no new-pipeline output has any.

## n = 128

Baseline (server, `~/tdp_runs/tdp128_default`, recipe v13, old stage 6, 48 threads): stage-5 output
11,090,312 raw gates, 2-control cost 53,772,533; old stage 6: 9 rounds, **397,989 s = 4.6 days**, final
8,464,455 raw gates = 2-control cost 13,467,606 (761,297 wide, up to 12 controls), ratio 0.250, deeper
0/0, **8 early outputs**.

New run (local, `port_runs/tdp128_s6`): recipe v14 stages 1-5 with the stage-3 twists of 2026-11-01
(managed driver, `run.skip_stages = [6]`, frozen worktree `s6_run128` at `15f4d9b6`), then the chosen
stage 6 by hand (`d827c65e`; pre-cleanup, 2-control cost, phase A cap 512, descent 128 / 32 / 16, two
seeds, 2 restarts, 12 s budgets, stop fractions 0.2, phase C, census batches 16, 12 threads,
`--forward-only` since the v14 stage 2 has no reverse contract). Stage 2 / 3 / 4 / 5 = 1,141,924 /
2,566,339 / 4,739,227 / 8,746,933 gates (512 wires); stage-5 2-control cost 41,599,345.

| n = 128 | stage-5 2-control cost | published 2-control cost | ratio | deeper a/d | early outputs | stage-6 time |
|---|---|---|---|---|---|---|
| old: recipe v13, old stage 6 (server) | 53,772,533 | 13,467,606 | 0.250 | 0/0 | **8** | 4.6 days on 48 threads |
| **new: recipe v14 stages 1-5, new stage 6** | 41,599,345 | **4,427,214** | **0.106** | 0/0 (of 2,767) | **0** | **4.0 h on 12 threads** |

Trajectory: pre-cleanup 8,746,933 -> 3,332,296 gates (-62 %, the same as at n = 64); phase A, 3 rounds
(2-control cost 8,532,324 -> 5,246,403; 28-38 min each); cleanup; cap 128 (2 rounds), cleanup; cap 32
(2 rounds), cleanup; cap 16 (2 rounds), cleanup -> 4,401,829 gates; phase C lowered the 7,065 remaining
wide gates (up to 10 controls) -> 4,427,214, every gate at most 2 controls. Zero slice ok, payload
contract ok (forward), canonical order applied, no block or cleanup rejected as leakier. Delivered as
`circuits/tdp128_s6_20261102.mpmct1` (provenance in `circuits/CIRCUIT_GENERATION_INFO`, seeds in
`~/.ran-private` only).

n = 128 compresses less than n = 64 at the same caps (0.106 vs 0.077): twice the wires and a 3x larger C
leave less local structure per small block. The production synthesizer effort (16 restarts, 60 s) was
not tried at n = 128 (at n = 64 it gave another -5.9 % for 3.4x the time).

## Recommended stage-6 configuration

```text
tdp_adaptive run ... --pre-cleanup --cost two-control --max-block-gates 512 --descent 128,32,16 \
    --seeds-per-block 2 --lower-controls 2 --cleanup-fcompress-iters 10 \
    --stop-fraction 0.2 --descent-stop-fraction 0.2 --max-rounds 0 \
    [--restarts 16 --time-limit 60]        # the size/time dial: -6 % for ~3.4x the time
```

plus the leak guard, which is on whenever `--accept smaller-not-leakier` (the default): every block
acceptance and every cleanup is rejected if it raises the validated deeper exposure or the early C
outputs. All of this is OFF by default in the code (`descent` empty, `pre_cleanup` false, `cost =
gates`, `lower_controls = None`), so the pipeline's default stage 6 is unchanged until RC decides; making
it the default means a recipe bump (v15) and wiring the new knobs through `scripts/tdp_gen.sh`, the
config schema and the manifest, which has not been done.

## Caveats

- **The census is the weak test.** "0/0 deeper, 0 early outputs" means no C value is a wire or an affine
  combination of live wires at any cut, validated on fresh and structured inputs. The stronger attacks
  of the variants campaign (factor-window rewrites, degree-2 replica scans) have NOT been run on these
  outputs; in that campaign fcompress-shortened circuits were MORE exposed to window rewrites than the
  unshortened ones. That battery, on the server, is the next step before adopting the configuration.
- One C per n, one seed per configuration; differences under ~1 % are within run-to-run noise.
- The bounded runs used 1 or 2 seeds, 2 restarts and 12 s budgets to finish overnight; the effort A/B
  shows the production synthesizer settings compress further.
- n = 128 stages 1-5 come from recipe v14 (with the 2026-11-01 stage-3 twists), the baseline's from v13,
  hence the comparison by ratio to each run's own stage-5 cost.

## Vocabulary

Pipeline stages: 1 sandwich, 2 three-mask preprocessing, 3 DB mixing, 4 split, 5 crossing walk, 6
delivery (`tdp_adaptive`). Inside stage 6: **phase A** = large-block ANF resynthesis rounds; **phase B**
= the descent, one **level** per smaller block cap, an exact cleanup (NOT-push + fcompress, with the
leak guard) after each; **phase C** = lowering every gate with more than 2 controls (Barenco) and a
final cleanup; **pre-cleanup** = one exact cleanup before phase A. A **round** = census, plan, synthesize,
accept, assemble.
