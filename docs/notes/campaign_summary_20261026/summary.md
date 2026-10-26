# Three-mask stage 2 and adaptive stage 6: matched comparison, resilience campaigns, and shortening of the baseline output

Summary of the efforts of 2026-10-21 to 2026-10-26 and their conclusions. Full records: `docs/notes/stage2_stage6_campaign_20261021.md` (campaign note with auto-updated tables), `docs/notes/three_mask_review_20261021.md`, `docs/notes/stage6_adaptive_review_20261021.md`; data and scripts on the server under `~/threemask_cmp` and `~/shorten`.

## 1. Scope and setup

One random master seed, kept only on the server, fixed everything the arms share: the source circuit C (n = 64, 1,000 gates, of which 936 internal births and 876 "deeper" ones, i.e. not on the public-quadratic boundary), the classic sandwich (128 wires, 2,832 gates) and the seeds of every downstream stage. The three arms differ only in the stage-2 compute block on 256 wires (128 data wires plus 128 band wires):

- **baseline**: the pinned `managed_tdp` stage 2;
- **three-mask as-is**: the design document's package (`triple_dirty_v1` flags), reproduced bit for bit;
- **three-mask modified**: as-is plus the two zero-cost recommendations of the review, `--random-helpers` and `--unique-cover`.

Downstream of stage 2: stages 3 to 5 with the document's reduced recipe (mixing profile 2,2,2,2,2, split, crossing), then the adaptive stage 6 in two variants (`asis`, `v2`), then the resilience campaigns. Exposure is measured as the number of C's internal births whose value appears, directly or as an affine function, on a wire of the circuit (every-cut census on 6,144 inputs, entry witnesses re-tested on 8,192 fresh and about 8,000 structured inputs), plus two decoder families: the raw ledger (rank-10/12 blind affine observers with a direct ledger and a structured audit) and the nine-arm narrow portfolio (125/250-gate window rewrites: raw, factor, factor-shift, factor-permuted, factor-w250, pairs, and the payload-side factor-w125, factor-w250 and delta-w250 arms). Narrow and ledger counts are "repeated / once" candidate equalities on fresh inputs.

## 2. Review of the three-mask stage-2 design

Verification: the present source reproduces the fixture bit for bit. Findings, most important first:

1. **Helper wires are fixed and absorb most band writes.** The first four wires of the lowest growing bank take 79 to 84% of all band writes (about 1,000 of 2,000 writes in a 2,000-gate window), a structural beacon for any window observer. Shuffling the helper pool per fire and striding per product unit costs nothing (501,636 vs 501,725 gates) and drops the top-four write share from 14.8% to 4.5%. Adopted as `--random-helpers`.
2. **Discarding helper restoration is most of the saving, and it is data feedback.** Baseline 769,936 gates; three-mask with restored helpers 707,347 (8.1% smaller); with discarded helpers 501,725 (34.8% smaller). So 77% of the saving comes from `--discard-fire-helpers`, and the discarded garbage is data dependent (products of carriers and masks on wires later promoted to mask coordinates). A restored-versus-dirty ablation is needed before any exposure difference is attributed to the third mask.
3. **Rotation is drain limited.** 2,688 of 2,832 fires stall on the retiring cap; a serving bank lives about 103 fires, roughly 16,300 gates, during which every mask coordinate is constant. Faster turnover costs about 15 gates per replacement (retire budget 4: +17%, 8: +51%).
4. **The cover mask is not label unique**; reusing the existing uniqueness rejection fixes it at no cost. Adopted as `--unique-cover`.
5. **A complete fire inside a narrow window factors into x_u times x_v**: the 16 product units of a fire (about 158 gates) sum to the product of two decoded values, so a 125 to 250-gate window often contains a whole fire. This is the candidate mechanism for the narrow-portfolio reversal between three-mask and baseline. Interleaving product units of independent fires would push every fire outside narrow windows at zero gate cost; it needs overlapping boxes in the migration pass and was not included in the modified arm.
6. Smaller points: `helpers[0]` always receives the cubic delta; drained retiring wires are valid helpers; several statistics are misleading in cubic mode; the package should be frozen as a named preset before any port.

## 3. Review of the adaptive stage-6 notes

The construction grows variable-length blocks by exact ANF composition under term and work caps, chooses proxy-aware boundaries, splits failed blocks at proxy-aware cuts, keeps a block only if smaller, and repeats to a size plateau. Findings and what was implemented:

1. **The boundary objective has little leverage on leakage**: a round with 3,370 proxy-free boundaries still exposed 319 births, because proxies are created by the synthesizer inside blocks. Remedy: exposure-aware block acceptance (two seeds, keep smaller and not leakier, judged by a full census of each candidate assembly; sound because the state at every retained boundary is fixed). Implemented in `v2`.
2. The proxy-aware driver of the notes was not monotone (growing blocks were kept); keep-smaller per block and the zero-gain stop are in both variants.
3. Planning was the slow, sequential part (2,153 s on 5.4M gates; 2,935 of 3,032 blocks stopped at the 4M product-pair work limit). Parallel segment planning with seams at exposure-free cuts reduces it to minutes. Implemented in `v2`.
4. Boundaries should move between rounds (round 2 pinned 20 deeper ids at its boundaries); a 64-gate avoidance radius around the previous round's boundaries is in `v2`.
5. Budgets should scale with block size (25% of blocks hit the fixed 60 s deadline); 60/120/240 s in `v2`.
6. Measurement: ever-exposed births with deeper/all denominators are the primary numbers; ledger R/O counts are a separate decoder family.

Stop rules set during the campaign: no round cap; a run ends after two consecutive rounds without a size decrease or two consecutive rounds whose cumulative compression is below 1%; synthesizer seeds advance each round. The last circuit is the run's mixed output F. A third variant, `attacker`, is C-blind (empty census for planning and splitting, keep-smaller only) and is the resynthesis used by the resilience campaigns.

## 4. Matched pipeline results

| arm | stage 2 | stage 3 | stage 4 | stage 5 | stage-5 narrow (deeper R/O) |
| --- | --- | --- | --- | --- | --- |
| baseline | 779,113 | 1,598,867 | 3,018,226 | 5,508,346 | 0 / 0 |
| three-mask as-is | 505,448 | 1,056,566 | 1,899,746 | 3,516,629 | 19 / 6 |
| three-mask modified | 505,393 | 1,065,202 | 1,889,290 | 3,497,436 | 7 / 3 |

Adaptive stage 6 to plateau (deeper direct / affine births exposed in the output):

| run | rounds | stage 5 to F (gates) | gzip-6 | exposed at F |
| --- | --- | --- | --- | --- |
| baseline / asis | 24 | 5,508,346 to 666,745 | 2.1 MB | 384 / 424 |
| as-is / asis | 16 | 3,516,629 to 2,611,000 | 8.4 MB | 21 / 21 |
| as-is / v2 | 14 | 3,516,629 to 2,368,032 | 6.1 MB | 0 / 0 |
| modified / asis | 12 | 3,497,436 to 2,761,726 | 9.1 MB | 14 / 14 |
| modified / v2 | 9 | 3,497,436 to 2,706,063 | 7.8 MB | 0 / 0 |

Observations. The three-mask stage 2 is 35% smaller than the baseline and stays so through mixing. Stage 6 compresses the baseline 8.3-fold and exposes half of C while doing so (205 deeper births after round 1, 424 at the plateau); it compresses the three-mask outputs only 1.3 to 1.5-fold and exposes 0 to 21. The `v2` variant compresses more and reaches a clean census (0 exposed) on both three-mask arms. The narrow portfolio on the stage-5 outputs already reverses the ordering: 19 deeper values repeated on three-mask as-is, 7 on modified, 0 on baseline, consistent with review finding 5.

## 5. Resilience campaigns

Each mixed output F was attacked by: (a) the raw ledger and the nine narrow arms on F; (b) C-blind adaptive resynthesis of F to its own plateau (largest feasible blocks, keep smaller, fresh seeds per round, same stop rules); (c) a stage-3 remix of F (same reduced profile, new seed, 5.2 effective-work units) followed by (b); (d) a lighter remix for 2 effective-work units followed by (b); with the census after every step and the raw ledger on every plateau. Every rewrite was checked for exact 256-wire equivalence (8,192 states each direction) and the stage-1 payload contract.

| output | gates of F | census / raw on F | narrow on F | blind resynthesis | remix 5.2 + blind | remix 2 + blind |
| --- | --- | --- | --- | --- | --- | --- |
| baseline / asis | 666,745 | 424 / 348 | 444 | 30 r, 355,990 g, 503 (raw 410) | 32 r, 382,635 g, 508 (raw 415) | 41 r, 351,744 g, 505 (raw 417) |
| as-is / asis | 2,611,000 | 21 / 3 | 232 | 15 r, 2,161,429 g, 42 (raw 13) | 17 r, 2,467,113 g, 40 (raw 23) | 18 r, 2,384,372 g, 37 (raw 13) |
| as-is / v2 | 2,368,032 | 0 / 0 | 264 | 12 r, 2,155,215 g, 9 (raw 2) | 14 r, 2,474,027 g, 39 (raw 18) | 14 r, 2,364,042 g, 33 (raw 20) |
| modified / asis | 2,761,726 | 14 / 1 | 137 | 9 r, 2,513,906 g, 17 (raw 2) | 12 r, 3,090,135 g, 26 (raw 9) | 11 r, 2,790,863 g, 22 (raw 5) |
| modified / v2 | 2,706,063 | 0 / 0 | 157 | 6 r, 2,605,565 g, 6 (raw 1) | 13 r, 3,007,374 g, 24 (raw 5) | 16 r, 2,819,216 g, 19 (raw 5) |

(r = rounds to plateau, g = gates at plateau, then deeper affine births exposed at the plateau and the repeated deeper values of its raw ledger.)

Exposure since the start of each campaign, as the union over every circuit produced or attacked (stage-6 rounds that produced F excluded), of the 936 internal C values; "strongly exposed" means at least two distinct supports within one circuit:

| output | circuits examined | exposed (deeper) | strongly exposed (deeper) | by narrow arms | by raw ledgers | by census of blind rounds and remixes | only by narrow arms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline / asis | 120 | 723 (671) | 696 (646) | 481 | 646 | 723 | 0 |
| as-is / asis | 67 | 341 (316) | 297 (273) | 302 | 96 | 122 | 218 |
| as-is / v2 | 57 | 370 (346) | 316 (295) | 348 | 77 | 98 | 270 |
| modified / asis | 49 | 247 (224) | 188 (168) | 222 | 61 | 72 | 173 |
| modified / v2 | 52 | 278 (258) | 205 (188) | 249 | 53 | 69 | 206 |

Conclusions.

- **The baseline is open under every family.** Three quarters of C is exposed, all three blind lines converge to 350 to 380 thousand gates with half of C in the clear, and the narrow arms add nothing the ledgers did not already find.
- **Three-mask outputs leak in the low double digits** to census, raw ledger, blind resynthesis and remixes. Most of their campaign tally comes from the narrow 125/250-gate rewrites of F, which find values no other family does (173 to 270 values found only there).
- **Remix then repack exposes more than repack alone** on the three-mask outputs: on as-is / v2 the remix lines expose 39 and 33 against 9 for blind packing, on modified / v2 24 and 19 against 6. On the baseline the remix lines simply return to the same plateau. On the three-mask outputs the remix lines end larger than the direct lines, so remixing is an exposure lever rather than a compression lever.
- **The modified stage 2 beats the as-is package on every measure**, and modified / asis is the least exposed output overall (247 values, 188 strongly). The `v2` stage 6 gives a cleaner day-one census but a slightly higher campaign tally, because its deeper compression gives the narrow arms more to work with.
- **Compression and exposure move together** throughout: every blind round that shrinks a three-mask output raises its census, and the shortening campaign below continues the trend.

## 6. Shortening the baseline output

Question: is the 351,744-gate blind plateau of the baseline output a size floor? All three attack lines had converged there and each remix had been undone by resynthesis. The target was to go materially lower under exact 256-wire equivalence (and the payload contract), using any combination of tools.

Chain, every step verified against its input and the final circuit verified end to end against the plateau (equivalent on 8,192 states each direction; payload contract passed on 10,338 inputs per direction):

| step | gates | note |
| --- | --- | --- |
| C-blind plateau after 41 attacker rounds | 351,744 | 1,000 to 2,000-gate blocks |
| small-block rounds (512-gate cap, 5 rounds) | 332,408 | first change of scale |
| fcompress | 314,135 | gathering of same-target gates |
| cycle 1: block caps 64 to 1024, fcompress with reorderings between passes | 292,714 | 6.8% |
| cycles 2 to 9: caps 16 to 128, forward, reversed and reordered passes, fcompress between | 274,097 | 2.6, 1.15, 0.94, 0.61, 0.45, 0.34, 0.28% per cycle; stop at < 0.3% |

Final: 274,097 gates, 22.1% below the plateau; gzip 1.03 MB against 1.21 MB; census of the final circuit 525 deeper affine births (plateau 505). About 16 hours of wall time on 32 to 56 cores of a shared box.

Findings.

1. **The plateau was a block-size artefact.** The attacker's rounds used 1,000 to 2,000-gate blocks, and at that size the ANF synthesizer returned a larger circuit for 9 of 10 blocks (round 41: 176 blocks kept, 22 replaced, 176 rejected as growing). Capping blocks at 16 to 512 gates compresses again at once; each small window synthesizes in under a second, so a round takes minutes. The smallest caps stayed the most productive to the end (32-gate passes forward, reversed and on clustered orders), while 128-gate and larger passes were exhausted after cycle 1.
2. **Three rewrites keep each other going**: window resynthesis, fcompress (same-target gathering with transport, ANF compaction, exact small groups: 5.6% in 34 s on first use, then converged), and function-preserving reorderings before fcompress (ASAP and ALAP dependency layering with same-target clustering, random and target-clustered topological orders): each new order reopened fcompress for 0.1 to 0.7%, and fifteen iterations were worth about 0.8% every time they were applied. Resynthesizing the reversed circuit found different windows with similar gains.
3. **Effort inside a window is irrelevant.** Ten synthesizer settings (beam 6 to 32, restarts 16 to 96, backtracking depth, exact rescoring, degree weights, macros, group merge, noise) all gave 0.2% on 240 sampled 64-gate windows, and a run with 300 s per window and wider search on large blocks gained 0.2% in 2.5 hours. Reordering before windowing was no better than plain passes.
4. **The payload contract is not a lever here.** The deliverable only has to be correct with inputs on wires 0 to 63, zeros elsewhere, outputs on wires 64 to 127, in both directions. Constant propagation from the zero ancillas plus liveness from the payload outputs removes 0.33% when both directions are kept and 0.78% forward-only. A sampled semantic test (32,768 domain inputs per direction) finds 3 gates that never fire on the domain in both directions and about 1% forward-only. After 41 rounds every gate is active on the reachable states.
5. **It is a tool floor, again.** The gains decayed geometrically to the stop rule, but the sandwich itself (2,832 gates on 128 wires) satisfies the same contract, so the true floor is more than a hundred times lower and reachable only by structural recovery of the sandwich, which is the exposure problem rather than a compression one. For the resilience report this means the campaign's final sizes describe what each fixed attacker reached, not minimal circuits, and that any attacker who changes scale or algorithm compresses further and exposes more.

## 7. Concurrent preimage campaign (recorded in the same note)

Alongside, and sharing the campaign note, a C-blind preimage campaign on the baseline output (find x with C(x) = 0) ran SAT solvers (CryptoMiniSat with native XOR clauses) on F and on its blind plateau, two-circuit linked variants (42,771 shared internal functions between F and the plateau used as implied links), hint-augmented variants, and a prefix-only solve, together with structural walks over the vocabulary of values born in the derived circuits. As of 2026-10-26 09:00Z none of the solvers had produced a preimage after about two days, and the structural walks were shown to leave C's gates early because remix identities produce values of exactly the tested form. Its notable finding is a mid-circuit landmark: around depth 0.45 to 0.50 of the plateau circuit, 50 of the 64 output bits are sparse degree-2 polynomials of the interior wires (median 10 monomials; 4 bits are single wires), 36 of 64 on F, 20 and 16 on the remixes, with the union over cuts and circuits covering 62 of 64 bits; these relations were fed to the solvers as sound hints. The shortened circuits of Section 6 were also scanned (40 of 64 bits at depth 0.40 on a 279,263-gate intermediate).

## 8. Overall conclusions and recommendations

1. Adopt the two zero-cost stage-2 changes (random helpers, unique cover): the modified arm is smaller than the baseline by a third and least exposed under every attack family. Run the restored-versus-dirty helper ablation before crediting the third mask, and implement fire interleaving, since whole fire boxes inside 125 to 250-gate windows are the most likely source of the dominant narrow-arm exposure.
2. Judge stage 6 and its stop rule by exposure, not by size. The `v2` acceptance rule keeps the census at zero, but every further compression, by the defender or by an attacker, raises exposure; the narrow portfolio and a C-blind resynthesis to plateau should be part of the acceptance test, and the "plateau" should be reported as the fixed attacker's result rather than as a floor.
3. The attacker's strongest cheap moves are, in order: narrow-window rewrites of F, a stage-3 remix followed by repacking, and multi-scale resynthesis with gathering and reordering; all three are now scripted on the server and can be run on any future output in hours.
4. The baseline construction is broken under this attacker model and serves only as a reference.
5. The information-theoretic gap is the whole story: 274 thousand gates against a 2,832-gate sandwich. Closing it needs structural recovery, and the degree-2 midpoint landmark of the preimage campaign is the first concrete handle on that structure; whether it is exploitable remains open.

## Artifacts

- Campaign note with final tables: `docs/notes/stage2_stage6_campaign_20261021.md`; reviews: `docs/notes/three_mask_review_20261021.md`, `docs/notes/stage6_adaptive_review_20261021.md`.
- Server (`machine1`): pipeline, attacks and campaigns under `~/threemask_cmp` (`py/`, `py6/`, `run/`); shortening scripts and the final circuit under `~/shorten` (`strict_run.py`, `ms2_sched.py`, `ms3_sched.py`, `ms4_sched.py`, `fc_step.py`, `reorder.py`, `reduce_payload.py`, `reduce_semantic.py`, `final_274097.mpmct1`); fcompress built from the working branch at `~/local_mixing_wd`.
