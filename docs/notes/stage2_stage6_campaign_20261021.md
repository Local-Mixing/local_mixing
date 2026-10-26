# Stage-2 variants × adaptive stage-6 variants: matched pipeline campaign (2026-10-21, in progress)

All runs on the server, tree `~/threemask_cmp` (see memory note threemask-stage6-campaign-20261021 for paths).
One random master seed (kept only on the server) for the source C (n=64, 1,000 gates), the classic sandwich,
and the downstream stage seeds; the arms differ only in the stage-2 compute block.

## Arms
- **baseline**: pinned `managed_tdp` stage 2 (commit 4a85b09f), as in the three-mask document.
- **three-mask as-is**: the document's package (`triple_dirty_v1` flags), present source reproduces the fixture bit for bit.
- **three-mask modified**: as-is plus `--random-helpers` (dirty helpers drawn from a shuffled pool of all reference-free
  band wires, per product unit) and `--unique-cover` (temporary cover rejected if its cubic label is live).
  Not included: fire interleaving (needs overlapping boxes in the migration pass), retire-budget changes.

Downstream: stages 3–5 with the document's reduced recipe (profile 2,2,2,2,2, split, crossing), then adaptive stage 6
in two variants (`asis` = the notes' combined procedure with the frozen policy; `v2` = plus exposure-aware block
acceptance with two seeds, boundaries shifted away from the previous round's, size-scaled budgets, parallel planning;
see docs/notes/stage6_adaptive_review_20261021.md), then the exposure census of every round's output and the raw +
nine-arm narrow portfolio on the stage-5 outputs.


Stop rule (RC, 2026-10-21 16:25 UTC): no round cap; a run ends after two consecutive rounds without a size decrease. Because keep-smaller with a fixed seed would repeat a zero-gain round verbatim, the synthesizer seeds advance each round (as-is: seed r in round r; v2: seeds 2r-1, 2r). Rounds completed before the change kept seed 1 (as-is) / 1,2 (v2).


Resilience campaign (RC, 2026-10-21): each run's plateau circuit is its mixed output F. F is then attacked in several ways, with the exposure census after each: (a) the raw and nine-arm narrow portfolio ledgers on F; (b) C-blind adaptive resynthesis of F (largest feasible blocks, keep-smaller, fresh seeds per round, no proxy information) to its own plateau; (c) a stage-3 remix of F (same reduced profile, new seed, 5.2 effective-work units as in the pipeline's stage 3) followed by the same C-blind resynthesis; (d) added 2026-10-22 at RC's request: a lighter stage-3 remix of F for 2 effective-work units (moves = 2 × gates; the profile expands the circuit to about 2× by then), again followed by C-blind packing/resynthesis. Results appear in the Resilience section below as they complete.


Stop rule addition (RC, 2026-10-22 14:00 UTC): a run also ends when two consecutive rounds together compress by less than 1%. Applied retroactively to completed rounds: the mixed output is the output of the first round at which either rule holds; rounds computed beyond that point (baseline rounds 25–28) are kept in the table as extra data but are not the mixed output.


Synthesizer note (2026-10-23 05:40 UTC): on one window of the baseline remix's C-blind round 12 the ANF synthesizer's composition balloons past 150 GB and the kernel kills it (four OOM kills in the log). The driver now caps each synthesizer process at 48 GB of address space, retries a failed chunk once, and then synthesizes the chunk window by window, keeping a window whose synthesizer dies verbatim (status `crash`); such windows count as unsynthesized in the coverage figures.

## Results (auto-updated 14:51Z UTC)

Denominators: 936 internal C births, 876 deeper, 60 public-quadratic boundary. Narrow-portfolio counts are repeated/once candidate equalities (fresh 8,192 + structured inputs). Attack cases finished: 18 of 27.

### Stage-2 arms: sizes and stage-5 exposure
| arm | stage 2 | stage 3 | stage 4 | stage 5 | gzip-6 | raw deeper R/O | narrow deeper R/O (arms done) | narrow boundary R/O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 779,113 | 1,598,867 | 3,018,226 | 5,508,346 | 26.3 MB | 0/0 | 0/0 (9) | 4/0 |
| three-mask as-is | 505,448 | 1,056,566 | 1,899,746 | 3,516,629 | 17.1 MB | 0/0 | 19/6 (9) | 6/0 |
| three-mask modified | 505,393 | 1,065,202 | 1,889,290 | 3,497,436 | 18.0 MB | 0/0 | 7/3 (9) | 10/0 |

Per-arm narrow results (deeper R/O): baseline: factor 0/0, factor_perm 0/0, factor_shift63 0/0, factor_w250 0/0, pairs 0/0, raw 0/0, p-delta_w250 0/0, p-factor_w125 0/0, p-factor_w250 0/0; three-mask as-is: factor 1/0, factor_perm 1/0, factor_shift63 1/0, factor_w250 8/2, pairs 0/0, raw 0/0, p-delta_w250 9/4, p-factor_w125 1/0, p-factor_w250 8/4; three-mask modified: factor 0/0, factor_perm 0/0, factor_shift63 0/0, factor_w250 4/2, pairs 0/0, raw 0/0, p-delta_w250 2/0, p-factor_w125 1/1, p-factor_w250 3/0

### Adaptive stage 6, per round (deeper direct / affine births exposed in the output, validated; "picks" = blocks kept original / replaced; rejected growing, leakier)
| stage-2 arm | stage-6 | round | gates in → out | gzip-6 | blocks | picks | deeper direct/affine | all-internal affine | cut fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-mask as-is | asis | 1 | 3,516,629 → 3,308,752 | 14.1 MB | 5,558 | 4301/1257; 4281, 0 | 2 / 2 | 3 | 0.000000 |
| three-mask as-is | asis | 2 | 3,308,752 → 3,156,901 | 12.6 MB | 5,275 | 4118/1157; 4098, 0 | 2 / 2 | 3 | 0.000000 |
| three-mask as-is | asis | 3 | 3,156,901 → 3,059,113 | 11.7 MB | 4,401 | 3643/758; 3627, 0 | 3 / 3 | 4 | 0.000001 |
| three-mask as-is | asis | 4 | 3,059,113 → 2,981,630 | 11.1 MB | 3,856 | 3314/542; 3299, 0 | 4 / 4 | 6 | 0.000001 |
| three-mask as-is | asis | 5 | 2,981,630 → 2,914,987 | 10.6 MB | 3,882 | 3295/587; 3273, 0 | 6 / 6 | 8 | 0.000002 |
| three-mask as-is | asis | 6 | 2,914,987 → 2,858,175 | 10.1 MB | 3,593 | 3054/539; 3037, 0 | 9 / 9 | 11 | 0.000002 |
| three-mask as-is | asis | 7 | 2,858,175 → 2,811,993 | 9.8 MB | 3,351 | 2920/431; 2900, 0 | 10 / 10 | 12 | 0.000003 |
| three-mask as-is | asis | 8 | 2,811,993 → 2,775,191 | 9.5 MB | 3,178 | 2820/358; 2806, 0 | 13 / 13 | 15 | 0.000003 |
| three-mask as-is | asis | 9 | 2,775,191 → 2,742,360 | 9.3 MB | 3,060 | 2731/329; 2717, 0 | 14 / 14 | 17 | 0.000006 |
| three-mask as-is | asis | 10 | 2,742,360 → 2,716,973 | 9.1 MB | 2,980 | 2669/311; 2656, 0 | 16 / 16 | 19 | 0.000006 |
| three-mask as-is | asis | 11 | 2,716,973 → 2,693,341 | 8.9 MB | 2,839 | 2600/239; 2587, 0 | 16 / 16 | 19 | 0.000005 |
| three-mask as-is | asis | 12 | 2,693,341 → 2,667,256 | 8.8 MB | 2,754 | 2513/241; 2497, 0 | 15 / 15 | 18 | 0.000005 |
| three-mask as-is | asis | 13 | 2,667,256 → 2,648,317 | 8.7 MB | 2,725 | 2536/189; 2517, 0 | 15 / 15 | 18 | 0.000005 |
| three-mask as-is | asis | 14 | 2,648,317 → 2,633,745 | 8.6 MB | 2,753 | 2563/190; 2544, 0 | 17 / 17 | 20 | 0.000006 |
| three-mask as-is | asis | 15 | 2,633,745 → 2,620,991 | 8.5 MB | 2,640 | 2476/164; 2464, 0 | 20 / 20 | 23 | 0.000007 |
| three-mask as-is | asis | 16 (plateau) | 2,620,991 → 2,611,000 | 8.4 MB | 2,628 | 2484/144; 2464, 0 | 21 / 21 | 24 | 0.000007 |
| three-mask as-is | v2 | 1 | 3,516,629 → 3,299,400 | 14.1 MB | 5,299 | 4047/1252; 7847, 3 | 0 / 0 | 1 | 0.000000 |
| three-mask as-is | v2 | 2 | 3,299,400 → 2,976,441 | 10.4 MB | 4,485 | 2702/1783; 5311, 7 | 0 / 0 | 1 | 0.000000 |
| three-mask as-is | v2 | 3 | 2,976,441 → 2,817,013 | 9.1 MB | 2,930 | 2196/734; 4243, 9 | 0 / 0 | 1 | 0.000000 |
| three-mask as-is | v2 | 4 | 2,817,013 → 2,691,926 | 8.2 MB | 2,431 | 1817/614; 3527, 8 | 0 / 0 | 1 | 0.000000 |
| three-mask as-is | v2 | 5 | 2,691,926 → 2,632,217 | 7.7 MB | 2,018 | 1649/369; 3178, 13 | 0 / 0 | 1 | 0.000000 |
| three-mask as-is | v2 | 6 | 2,632,217 → 2,587,017 | 7.4 MB | 1,810 | 1540/270; 2945, 3 | 0 / 0 | 2 | 0.000001 |
| three-mask as-is | v2 | 7 | 2,587,017 → 2,515,542 | 6.9 MB | 2,057 | 1485/572; 2892, 5 | 0 / 0 | 2 | 0.000001 |
| three-mask as-is | v2 | 8 | 2,515,542 → 2,463,435 | 6.6 MB | 1,825 | 1421/404; 2759, 7 | 0 / 0 | 1 | 0.000001 |
| three-mask as-is | v2 | 9 | 2,463,435 → 2,438,816 | 6.5 MB | 1,403 | 1244/159; 2353, 4 | 0 / 0 | 1 | 0.000001 |
| three-mask as-is | v2 | 10 | 2,438,816 → 2,418,863 | 6.4 MB | 1,377 | 1224/153; 2314, 10 | 0 / 0 | 1 | 0.000001 |
| three-mask as-is | v2 | 11 | 2,418,863 → 2,402,800 | 6.3 MB | 1,321 | 1186/135; 2205, 5 | 0 / 0 | 1 | 0.000001 |
| three-mask as-is | v2 | 12 | 2,402,800 → 2,385,629 | 6.2 MB | 1,328 | 1184/144; 2252, 8 | 0 / 0 | 1 | 0.000001 |
| three-mask as-is | v2 | 13 | 2,385,629 → 2,375,864 | 6.1 MB | 1,242 | 1150/92; 2178, 4 | 0 / 0 | 1 | 0.000001 |
| three-mask as-is | v2 | 14 (plateau) | 2,375,864 → 2,368,032 | 6.1 MB | 1,276 | 1180/96; 2229, 2 | 0 / 0 | 1 | 0.000001 |
| baseline | asis | 1 | 5,508,346 → 2,831,405 | 9.2 MB | 3,432 | 971/2461; 963, 0 | 205 / 289 | 312 | 0.000088 |
| baseline | asis | 2 | 2,831,405 → 1,905,790 | 5.9 MB | 1,832 | 649/1183; 635, 0 | 298 / 351 | 382 | 0.000239 |
| baseline | asis | 3 | 1,905,790 → 1,462,618 | 4.4 MB | 1,091 | 436/655; 429, 0 | 325 / 379 | 408 | 0.000439 |
| baseline | asis | 4 | 1,462,618 → 1,245,723 | 3.8 MB | 810 | 392/418; 384, 0 | 349 / 394 | 423 | 0.000599 |
| baseline | asis | 5 | 1,245,723 → 1,118,532 | 3.4 MB | 677 | 373/304; 367, 0 | 352 / 400 | 430 | 0.000724 |
| baseline | asis | 6 | 1,118,532 → 1,037,077 | 3.1 MB | 640 | 387/253; 377, 0 | 358 / 402 | 434 | 0.000811 |
| baseline | asis | 7 | 1,037,077 → 980,947 | 3.0 MB | 614 | 400/214; 389, 0 | 370 / 412 | 442 | 0.000878 |
| baseline | asis | 8 | 980,947 → 929,403 | 2.8 MB | 562 | 371/191; 356, 0 | 367 / 413 | 442 | 0.000998 |
| baseline | asis | 9 | 929,403 → 889,026 | 2.7 MB | 553 | 368/185; 354, 0 | 370 / 421 | 452 | 0.001029 |
| baseline | asis | 10 | 889,026 → 854,064 | 2.6 MB | 527 | 356/171; 339, 0 | 379 / 418 | 448 | 0.001068 |
| baseline | asis | 11 | 854,064 → 823,490 | 2.5 MB | 490 | 355/135; 341, 0 | 390 / 435 | 464 | 0.001125 |
| baseline | asis | 12 | 823,490 → 800,993 | 2.5 MB | 481 | 355/126; 343, 0 | 381 / 430 | 460 | 0.001168 |
| baseline | asis | 13 | 800,993 → 782,270 | 2.4 MB | 442 | 339/103; 330, 0 | 376 / 427 | 455 | 0.001201 |
| baseline | asis | 14 | 782,270 → 767,690 | 2.4 MB | 483 | 379/104; 369, 0 | 378 / 425 | 454 | 0.001214 |
| baseline | asis | 15 | 767,690 → 746,434 | 2.3 MB | 427 | 342/85; 333, 0 | 380 / 427 | 458 | 0.001240 |
| baseline | asis | 16 | 746,434 → 729,825 | 2.3 MB | 431 | 357/74; 347, 0 | 392 / 432 | 463 | 0.001312 |
| baseline | asis | 17 | 729,825 → 716,612 | 2.2 MB | 435 | 337/98; 325, 0 | 397 / 438 | 471 | 0.001359 |
| baseline | asis | 18 | 716,612 → 703,770 | 2.2 MB | 411 | 328/83; 325, 0 | 395 / 435 | 467 | 0.001356 |
| baseline | asis | 19 | 703,770 → 693,272 | 2.2 MB | 379 | 319/60; 312, 0 | 392 / 430 | 461 | 0.001367 |
| baseline | asis | 20 | 693,272 → 684,310 | 2.2 MB | 408 | 342/66; 331, 0 | 392 / 429 | 460 | 0.001384 |
| baseline | asis | 21 | 684,310 → 677,942 | 2.1 MB | 401 | 334/67; 326, 0 | 387 / 427 | 458 | 0.001389 |
| baseline | asis | 22 | 677,942 → 671,467 | 2.1 MB | 396 | 339/57; 330, 0 | 384 / 427 | 458 | 0.001443 |
| baseline | asis | 23 | 671,467 → 668,575 | 2.1 MB | 392 | 348/44; 337, 0 | 383 / 426 | 457 | 0.001437 |
| baseline | asis | 24 (plateau) | 668,575 → 666,745 | 2.1 MB | 392 | 356/36; 346, 0 | 384 / 424 | 455 | 0.001458 |
| baseline | asis | 25 | 666,745 → 663,397 | 2.1 MB | 404 | 363/41; 354, 0 | 386 / 427 | 458 | 0.001443 |
| baseline | asis | 26 | 663,397 → 660,529 | 2.1 MB | 385 | 352/33; 343, 0 | 385 / 425 | 456 | 0.001441 |
| baseline | asis | 27 | 660,529 → 658,998 | 2.1 MB | 371 | 341/30; 333, 0 | 387 / 425 | 456 | 0.001466 |
| baseline | asis | 28 | 658,998 → 656,614 | 2.1 MB | 409 | 379/30; 366, 0 | 387 / 424 | 455 | 0.001478 |
| three-mask modified | asis | 1 | 3,497,436 → 3,222,057 | 13.5 MB | 4,414 | 3025/1389; 3012, 0 | 3 / 3 | 4 | 0.000001 |
| three-mask modified | asis | 2 | 3,222,057 → 3,076,392 | 11.9 MB | 3,954 | 2995/959; 2984, 0 | 5 / 5 | 6 | 0.000001 |
| three-mask modified | asis | 3 | 3,076,392 → 2,999,953 | 11.1 MB | 3,372 | 2758/614; 2750, 0 | 7 / 7 | 8 | 0.000001 |
| three-mask modified | asis | 4 | 2,999,953 → 2,946,908 | 10.6 MB | 3,091 | 2630/461; 2625, 0 | 7 / 7 | 8 | 0.000001 |
| three-mask modified | asis | 5 | 2,946,908 → 2,903,956 | 10.3 MB | 2,930 | 2566/364; 2558, 0 | 10 / 10 | 11 | 0.000002 |
| three-mask modified | asis | 6 | 2,903,956 → 2,872,600 | 10.0 MB | 2,798 | 2460/338; 2453, 0 | 10 / 10 | 12 | 0.000002 |
| three-mask modified | asis | 7 | 2,872,600 → 2,849,137 | 9.8 MB | 2,638 | 2366/272; 2360, 0 | 10 / 10 | 13 | 0.000003 |
| three-mask modified | asis | 8 | 2,849,137 → 2,826,522 | 9.6 MB | 2,578 | 2330/248; 2320, 0 | 10 / 10 | 13 | 0.000003 |
| three-mask modified | asis | 9 | 2,826,522 → 2,804,065 | 9.5 MB | 2,455 | 2236/219; 2231, 0 | 13 / 13 | 15 | 0.000003 |
| three-mask modified | asis | 10 | 2,804,065 → 2,782,993 | 9.3 MB | 2,412 | 2217/195; 2208, 0 | 12 / 12 | 14 | 0.000003 |
| three-mask modified | asis | 11 | 2,782,993 → 2,771,154 | 9.2 MB | 2,336 | 2178/158; 2172, 0 | 13 / 13 | 15 | 0.000003 |
| three-mask modified | asis | 12 (plateau) | 2,771,154 → 2,761,726 | 9.1 MB | 2,280 | 2152/128; 2149, 0 | 14 / 14 | 16 | 0.000004 |
| three-mask modified | v2 | 1 | 3,497,436 → 3,208,699 | 13.4 MB | 4,238 | 2832/1406; 5522, 4 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 2 | 3,208,699 → 2,971,913 | 10.3 MB | 3,244 | 1993/1251; 3943, 7 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 3 | 2,971,913 → 2,877,034 | 9.3 MB | 2,228 | 1723/505; 3351, 1 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 4 | 2,877,034 → 2,815,017 | 8.7 MB | 1,893 | 1529/364; 2928, 3 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 5 | 2,815,017 → 2,771,426 | 8.3 MB | 1,713 | 1454/259; 2779, 1 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 6 | 2,771,426 → 2,745,314 | 8.1 MB | 1,620 | 1433/187; 2740, 8 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 7 | 2,745,314 → 2,729,447 | 8.0 MB | 1,525 | 1392/133; 2656, 0 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 8 | 2,729,447 → 2,716,083 | 7.9 MB | 1,458 | 1336/122; 2538, 5 | 0 / 0 | 1 | 0.000000 |
| three-mask modified | v2 | 9 (plateau) | 2,716,083 → 2,706,063 | 7.8 MB | 1,429 | 1336/93; 2548, 0 | 0 / 0 | 1 | 0.000000 |

Runs at end of mixing (plateau): asis/asis, asis/v2, baseline/asis, v2/asis, v2/v2.

### Exposure since the start of each resilience campaign (union over every circuit produced or attacked in the campaign: raw and narrow-arm ledgers on F, blind resynthesis rounds, remixes and their rounds; the stage-6 rounds that produced F are excluded; internal C births, denominator 936 / deeper 876)
| mixed output under attack | circuits examined | exposed in at least one circuit (deeper) | strongly exposed: ≥2 distinct supports in one circuit (deeper) | found by narrow arms | by raw ledgers | by census of blind rounds and remixes | only narrow arms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline / as-is stage 6 | 120 | 723 (671) | 696 (646) | 481 | 646 | 723 | 0 |
| three-mask as-is / as-is stage 6 | 67 | 341 (316) | 297 (273) | 302 | 96 | 122 | 218 |
| three-mask as-is / v2 stage 6 | 57 | 370 (346) | 316 (295) | 348 | 77 | 98 | 270 |
| three-mask modified / as-is stage 6 | 49 | 247 (224) | 188 (168) | 222 | 61 | 72 | 173 |
| three-mask modified / v2 stage 6 | 52 | 278 (258) | 205 (188) | 249 | 53 | 69 | 206 |

### Resilience of the mixed outputs (attacks on the plateau circuit F; census = deeper direct/affine births; ledgers = deeper R/O)
| mixed output | gates | census of F | raw ledger of F | narrow portfolio of F (arms) | C-blind resynthesis of F: rounds, final gates, census | remix 5.2 effs: gates, raw ledger | C-blind resynthesis of remix | remix 2 effs: gates, raw ledger | C-blind resynthesis of 2-eff remix | done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-mask as-is / asis | 2,611,000 | 21/21 | 3/18 | 232/51 (9) | 15 rounds, 2,161,429 gates, 42/42 (max 43); raw 13/29 | 5375506, 15/6 | 17 rounds, 2,467,113 gates, 40/40 (max 40); raw 23/17 | 4514368, 12/9 | 18 rounds, 2,384,372 gates, 36/37 (max 37); raw 13/23 | yes |
| three-mask as-is / v2 | 2,368,032 | 0/0 | 0/0 | 264/63 (9) | 12 rounds, 2,155,215 gates, 9/9 (max 11); raw 2/7 | 4857451, 0/0 | 14 rounds, 2,474,027 gates, 39/39 (max 39); raw 18/21 | 3957095, 0/0 | 14 rounds, 2,364,042 gates, 33/33 (max 35); raw 20/13 | yes |
| baseline / asis | 666,745 | 384/424 | 348/49 | 444/6 (9) | 30 rounds, 355,990 gates, 460/503 (max 503); raw 410/59 | 1384798, 347/48 | 32 rounds, 382,635 gates, 466/508 (max 508); raw 415/62 | 1109439, 345/48 | 41 rounds, 351,744 gates, 458/505 (max 507); raw 417/55 | yes |
| three-mask modified / asis | 2,761,726 | 14/14 | 1/13 | 137/67 (9) | 9 rounds, 2,513,906 gates, 17/17 (max 17); raw 2/15 | 5650625, 11/3 | 12 rounds, 3,090,135 gates, 26/26 (max 28); raw 9/17 | 4486488, 6/8 | 11 rounds, 2,790,863 gates, 22/22 (max 25); raw 5/17 | yes |
| three-mask modified / v2 | 2,706,063 | 0/0 | 0/0 | 157/76 (9) | 6 rounds, 2,605,565 gates, 6/6 (max 6); raw 1/5 | 5531925, 0/0 | 13 rounds, 3,007,374 gates, 24/24 (max 24); raw 5/19 | 4333427, 0/0 | 16 rounds, 2,819,216 gates, 19/19 (max 20); raw 5/14 | yes |


## Preimage campaign on the baseline mixed output (C-blind inversion at 0^64)

Goal: find x with C(x) = 0^64 given only the declared mixed output F of the baseline / asis run (666,745 gates) and the attacker's own rewrites of it (the C-blind resynthesis rounds, the stage-3 remixes and their rounds). Neither C nor the census's identification of exposed values is used by any attack; C is used only in clearly labelled diagnostics that measure what the attack could have seen. Any candidate x is verified with F alone.

Strategies and outcomes so far:

1. SAT (CryptoMiniSat with native XOR clauses; one variable per initial wire and per gate birth; non-input wires fixed to 0 at the start, payload wires fixed to 0 at the end). Instances on F and on the 30-round C-blind plateau circuit (355,990 gates), 8 threads each, started 16:23Z. No solution after 2 h; still running.
2. Structural reconstruction from the vocabulary: every function (values on 8,192 sampled inputs, up to NOT) born on any wire of the derived circuits, with its earliest position. A walk from the inputs (forward) or from the payload (backward) applies a g57 gate when the implied value is in the vocabulary; a chain from inputs to payload would give C explicitly and hence x.
   - One circuit: forward 33 gates, backward 7 (F); 32 / 17 (plateau).
   - Seven circuits plus a one-gate lookahead (hypothesize a missing gate; keep it only if the next gate on the same wire has a present output): backward 19 gates (0 hypothesized), forward 65 gates (8 hypothesized), no meet, 21 min.
   - Diagnostic (uses C): the union of the seven circuits contains 684 of C's 1,000 gate births exactly; the missing births form runs of length 1 (151 runs), 2 (45), 3 (12), 4 (6) and 5 (3) in C's gate order. The walks stall not for lack of values over long stretches but because a present value often needs a control whose own value is stale (changed by an unexposed gate), which the same-wire continuation rule cannot see.
3. Round 3 (launched 17:58Z): vocabulary over all 65 fair-game circuits (F, 30 direct rounds, remix and its 22 rounds, remix2 and its 10 rounds), membership exact up to NOT and, as a fallback, up to affine functions of the inputs; depth-first search with backtracking and memoised states; a generalized one-gate lookahead confirmed either by a continuation on the same wire or by a gate that uses the hypothesized value as a control. Results below as they arrive.
   - Round-3 interim (18:50Z): the 65-circuit vocabulary (6.78M functions) covers 735 of C's 1,000 births (missing runs: 146 of length 1, 36 of 2, 8 of 3, 2 of 4, 3 of 5), but it also makes the walk unreliable: a state-based diagnostic (uses C) shows the backward walk leaving C's births from its first steps (27 of 64 wires off C's births after 35 direct steps), while the seven-circuit walk of round 2 had been genuine (17 of 19 backward gates). The forward walk from the inputs was spurious in every version (3 of 65 gates genuine): near the inputs the vocabulary is saturated with low-degree polynomials of the inputs (mask and band material). Cause on the payload side: stage-3 remixing inserts identity sequences whose intermediates have exactly the form "value ^ g57 gate on two other values", indistinguishable from a real C gate by membership alone. Next: a lineage rule (a value must appear in at least two of the three independent lineages: direct rounds, remix, remix2) and a degree-2 span test that already removes the spurious control-use confirmations (251k fingerprint hits -> 11 candidates).
   - Lineage rule (19:20Z): keeping only functions seen in at least two of the three lineages shrinks the vocabulary to 362,095 functions and still covers 644 of C's births, but the backward walk again leaves C's births early (14 of 64 wires off after 20 direct steps). Conclusion for this round: membership of the implied value in any derived-circuit vocabulary (7 circuits, 65 circuits, or the two-lineage subset) is not by itself evidence that a g57 step is a gate of C; the direct C-blind resyntheses and remixes contain many non-C values of exactly the tested form. No candidate x has been produced by the structural route; the SAT instances on F and on the plateau circuit are still running without a solution.

Status at 19:25Z: SAT on F and on the plateau (8 threads each) running since 16:23Z; structural reconstruction halted at this point. The five resilience campaigns continue (baseline remix round 24, remix2 round 12, modified/v2 remix2 round 10 done at this time).

Campaign note (01:55Z, 2026-10-24): the three-mask modified / asis resilience campaign had completed under the earlier three-arm script (DONE.json, no remix-2 arm). Its remix-2 arm (stage-3 remix for 2 effs, then C-blind rounds) was started by re-running the campaign script, which resumes finished arms from their saved results; DONE2.json will mark the complete campaign. Baseline / asis now has direct (round 30) and remix (round 32) at plateau with remix-2 at round 20.

4. Interior-cut / replication landmark (2026-10-24 03:4xZ). The raw sandwich computes C(x) in the middle and copies it, so an unmixed circuit has an interior cut where two disjoint 64-wire bundles carry the same 64 functions, and constraining that bundle to 0 would isolate C in the prefix. Tested C-blind on F and its three rewrites (plateau, remix, remix2) by scanning every cut for (a) functions carried by >=2 wires up to NOT [the copy signature] and (b) how many of F's own 64 output functions are simultaneously live. Result: the landmark does not survive. Duplicated-function count peaks at only 15-21 and only at depth 0.005-0.009 (the input-copy step at the start), and is 0 through the whole interior. The 64 output functions are never co-present at an interior cut: 0-2 live across the middle, rising to 44-54 only in the final window and to 64 only at the last gate; the earliest all-64 cut in F is gate 666,734/666,745 (depth 0.99998), and the last output bit is first assembled at depth 0.98. Stage-6 packing + ANF resynthesis rebuild the circuit toward the output, so there is no "first half that computes C" and no clean C(x) bundle to set to 0. Tools: py6/payload_cut.py, py6/replication_scan.py; results run/preimage/payload_cut_F.json, rep_{F,plateau,remix,remix2}.json. Prefix-SAT (py6/prefix_sat.py) was prepared but has no interior cut to attach to. Full-circuit SAT on F and on the plateau still running (~11.5 h, no solution).

5. Affine-replica scan (2026-10-24, py6/affine_replica_scan.py). At each cut, how many of F's 64 output functions lie in the GF(2)-affine span of the live wires (256 wires span <=257 of 8192 sample-dims, so a hit is not coincidental). The wire-rank saturates to ~256 by depth 0.03 and stays full. Output functions in the affine span across the interior (depth 0.03-0.97): F peaks at only 10/64, at depth 0.485 (the sandwich midpoint); remix 5/64 and remix2 5/64, both at depth ~0.48; plateau reaches 41/64 but at depth 0.97 (the resynthesis assembling the output near the end, not a mid-circuit landmark). So the midpoint carries a weak partial affine replica (5-10 of 64 output bits are linear combinations of the interior wires there) but not the full bundle, which is insufficient to invert. Confirms the exact/NOT result: no full C(x) bundle at any interior cut, affine or verbatim. Degree-2 scan launched to see whether more output bits are quadratic (not merely affine) replicas at the midpoint (py6/degree2_replica_scan.py, N2=49152 samples so the ~33k degree-2 monomials stay below the sample count and the test is non-vacuous).

6. Degree-2 replica scan (2026-10-24, py6/degree2_replica_scan.py, N2=49152->36864 samples so the ~33k degree-2 monomials stay below the sample count; RREF with provenance validated on controlled cases; membership false-positive ~2^-4000). This is the first scan to find a real interior landmark. At each cut, how many of F's 64 output functions are GF(2) degree-<=2 polynomials in the live wires. The count rises sharply toward the middle and peaks per circuit:
   | circuit | peak deg-2 outputs | depth | (profile 0.40/0.45/0.48/0.50) |
   | plateau | 50/64 | 0.45 | 13/50/30/44 |
   | F (baseline mixed output) | 36/64 | 0.48 | 3/20/36/19 |
   | remix | 20/64 | 0.48 | 2/12/20/12 |
   | remix2 | 16/64 | 0.50 | 3/8/8/16 |
   So the sandwich's mid-circuit replication of C(x) survives the pipeline not verbatim and not affinely (affine peak was only 10/64) but up to DEGREE 2: around depth 0.45-0.50 most of the output bits are quadratic functions of the interior wires, strongest in the compressed C-blind resynthesis (plateau, 50/64). The counts are non-monotonic in the exact gate (plateau 50/30/44 at 0.45/0.48/0.50), so the exact cut matters. No single cut reaches all 64, so this is a strong partial landmark, not a full bundle. Exploitability depends on whether the degree-2 representations are sparse (usable as constraints) or dense; the span is near-full-rank (dim ~32.6k of 32.9k monomials), which predicts dense representations. Extracting the explicit combinations on the plateau/0.45 cut (py6/combo_extract.py) to measure this.

7. Two-circuit linked SAT (2026-10-24, py6/combined_link_sat.py). Combine the 355,990-gate plateau (smallest equivalent) and the 666,745-gate output into ONE formula: shared 64-var input, both payloads = 0, plus linking clauses equating internal birth-variables whose functions coincide on sampled inputs (up to NOT). Such links are implied (same function of x), so they only help the solver -- they transfer deductions between the cheap small circuit and the expensive large one. Finding: the two equivalent circuits share 42,771 identical internal functions, so linking material is abundant. Variants running: link_6k (6000 links, 16384 samples), link_30k (30000 links, 32768 samples); both ~1.7M vars, ~990k XOR clauses, CryptoMiniSat with Gauss. Baselines: SAT on the plateau alone and on the 666k alone (both ~15 h, no solution). A false link (two functions agreeing on samples but differing at the preimage) could wrongly prune, so samples are set high and any SAT result is verified against the real circuit; a fast UNSAT would be the signal to suspect links. Priority shifted to the plateau as the primary target per its smaller size and stronger degree-2 exposure (50/64).

8. Sparse degree-2 combinations and hint-augmented SAT (2026-10-24, py6/combo_extract.py, plateau_hint_sat.py). Extracting the explicit degree-2 combinations at the plateau's depth-0.45 cut (gate 160195) gave a surprise: despite the near-full-rank span, the 50 degree-2 output bits have SPARSE representations -- min 1, median 10, max 50 monomials per output bit. Four output bits (17,31,41,57) equal a single interior wire verbatim (190,13,141,184) at the cut; the rest are XORs of ~10 wire/pair terms. So C's output bits are compact degree-2 functions of the interior wires there -- a genuine, usable structural weakness, not a dense artifact. Exploit: full plateau SAT + payload=0 + the 50 sparse relations imposed at the cut as implied hints (hint_plateau, 50 hints avg 11.6 terms, 601,906 vars). Sound (implied by payload=0; verified against the real circuit). Directly comparable to the plateau-alone baseline. Running alongside the two-circuit linked variants.

9. Replica classification by degree, shorter circuit (2026-10-24, py6/classify_replicas.py). Plateau at depth 0.45 (gate 160195), output bits by MINIMAL internal replica (nested: direct subset affine subset quadratic): DIRECT (= a single wire up to NOT) 4 bits {17,31,41,57}; AFFINE 5 (adds bit 3); QUADRATIC 50 (adds 45 that need a genuine product); NONE 14 {4,5,10,11,18,22,25,26,29,39,40,43,56,61}. So higher degree exposes NEW output bits, not re-representations; the affine->quadratic jump is almost all new (45/50 are quadratic-only, only 5 affine, 4 bare wires). Cross-circuit: plateau exposes 50, the 666k exposes 36, overlap only 29; plateau-only 21, F-only 7. The 7 F-only bits {5,18,22,25,39,56,61} are among the plateau's 14 unexposed -> the two circuits are COMPLEMENTARY. Union covers 57/64; only {4,10,11,26,29,40,43} have no degree-2 replica in either at these cuts. Motivates pulling degree-2 constraints from BOTH circuits (and more cuts) for a full 64-bit midpoint constraint set.

10. Multi-cut coverage and hint SAT (2026-10-24). combo_extract at plateau depth 0.50 (gate 177995): 44/64 bits, even sparser (min1/med9/max16). Union of plateau cuts 0.45 and 0.50 = 59/64 output bits (0.50 added {4,10,11,18,22,26,43,56,61}); remaining uncovered by plateau {5,25,29,39,40}. Adding the 666k best cut covers {5,25,39} -> grand union 62/64, only {29,40} with no degree-2 replica anywhere checked. So ~all output bits are compact degree-2 functions of interior wires at some midpoint cut. Attack: hint2_plateau = full plateau + payload=0 + 94 sparse hints across both cuts covering 59 bits (py6/plateau_hint_sat2.py). Six sound preimage solvers now running: sat_F, sat_plateau (baselines); link_6k, link_30k (two-circuit linked); hint_plateau (50 hints/1 cut), hint2_plateau (94 hints/2 cuts). All verify any x against the real circuit.

11. Prefix-only solve with degree-2 midpoint constraints (2026-10-24, py6/prefix_quad_sat.py). The realization of the original "cut the circuit, set the bundle to 0, solve the prefix" idea -- infeasible for exact replicas (no clean bundle) but enabled by the sparse degree-2 structure. Encode ONLY the plateau prefix up to the deeper cut (gate 177995, 177,996 of 355,990 gates), add the 94 degree-2 relations (covering 59 of 64 output bits) = 0 at their cuts, and enumerate the prefix's satisfying inputs (blocking clauses), checking each against the full plateau. 59 constraints pin the 64-bit input to ~2^5 candidates. Half the gates, uncompute half removed. 300,601 vars. If a candidate zeroes all 64 payload bits it is the preimage (verified on the real circuit). Running alongside the six full/linked/hint solvers.

## Shortening campaign on the baseline mixed output (RC, 2026-10-25: "resume attempts to further shorten the circuit")

Target: the baseline / asis mixed output's C-blind plateau, 351,744 gates (41 rounds of the 2-eff remix line), under exact 256-wire equivalence (circuit_check, 8,192 states each direction) plus the stage-1 payload contract (clean_payload). Work directory on the server: `~/shorten` (scripts `strict_run.py`, `ms2_sched.py`, `ms3_sched.py`, `fc_step.py`, `reorder.py`, `reduce_payload.py`); results under `~/threemask_cmp/run/shorten_strict`, `~/shorten/fcsteps`, `~/shorten/reorder`.

Findings so far (chronological):

1. The plateau was an artefact of block size. The attacker's rounds used 1,000–2,000-gate blocks; at that size the ANF synthesizer returned a larger circuit for 9 of 10 blocks (picks: 176 original / 22 replaced / 176 rejected-growing in round 41). Capping blocks at 128 or 512 gates compresses again at once: 351,744 → 335,361 (128-gate blocks, 4 rounds) and → 332,408 (512-gate blocks, 5 rounds); windows synthesize in well under a second, a round takes minutes.
2. fcompress (gathering of same-target runs with transport, ANF compaction, exact small groups) is complementary to window resynthesis: 335,361 → 316,639 in 34 s (verified), and 332,408 → 314,135. Its options (group cap, support cap, transport slack) do not change the result; pack-first is worse.
3. Alternating the two keeps paying: 314,135 → 308,174 (three 64-gate rounds) → 306,919 (fcompress) → 305,892 (256-gate rounds) → 305,100 → 303,902 (128-gate rounds) → 298,620 (fcompress).
4. Function-preserving reorderings reopen fcompress after it converges: ASAP/ALAP dependency layering (same-target gates clustered within a layer) and random or target-clustered topological orders each give 0.1–0.7% per pass; iterating 15 passes with fresh orders took 298,620 → 296,283 and was still gaining 0.03% per pass. The fcompress step of the schedule now tries five orderings per iteration.
5. Synthesizer settings do not matter at small scale: ten option sets (beam 6–32, restarts 16–96, back-k, exact-k 96, degree weights, macros, group merge, noise) all give 0.2% on 240 sampled 64-gate windows of the current best; disabling macros loses.
6. Large blocks with long budgets are not a lever: 300 s per window, beam 12, restarts 32 on 1,000–2,000-gate blocks gave 335,361 → 334,705 (0.2%) in 2.5 h; stopped.
7. Resynthesizing the reversed (inverse) circuit finds different windows but similar gains: 64-gate rounds on the reversed circuit 305,100 → 303,570, reversed back and verified; folded into cycle 2 as extra passes.
8. Payload-contract relaxation buys almost nothing syntactically: with wires 64–255 zero at entry and only wires 64–127 live at exit, constant propagation plus liveness removes 0.33% when both directions of the contract are kept (verified by simulation and clean_payload) and 0.78% forward-only; fcompress with `--zero-wires 64-255 --live-wires 64-127` gives 312,882 from 335,361 against 316,639 strict, but breaks the inverse contract. After 41 rounds every wire is entangled with every other.
10. Semantic version of the contract relaxation (sampled, 32,768 contract-domain inputs per direction, `reduce_semantic.py`): on the 280,598-gate circuit, gates that never fire on the domain in BOTH directions: 3 (0.00%); forward-only: 2,784 never-firing gates, 431 always-firing (→ NOT), 710 always-true literals = 0.99%. So the two-sided contract leaves no domain-dead gates at all; the circuit's gates are all active on the reachable states. The only route below the local-rewrite floor is structural (recovering the sandwich).
9. Perspective: the sandwich itself (2,832 gates on 128 wires) satisfies the same contract, so the true floor is more than 100× below these numbers; only a structural reconstruction could approach it.

Running schedule (multi-scale block caps 64 / 256 / 128 / 1024 / 32 / 512 forward, 64 and 128 on the reversed circuit, iterated multi-ordering fcompress between passes; stop when a full cycle gains < 0.3%). Current best: see the table below (updated as the campaign proceeds).

| step | gates | note |
| --- | --- | --- |
| plateau (remix-2 line, round 41) | 351,744 | start |
| 128- / 512-gate block rounds | 335,361 / 332,408 | pilots |
| fcompress | 314,135 | from the 512 pilot |
| cycle 1: 64, fc, 256, fc, 128, fc | 298,620 | fc = iterated fcompress with reorderings |
| standalone 15-pass multi-ordering fcompress | 296,283 | lever ceiling probe (not chained) |
| cycle 1 complete: + 1024, fc, 32, fc, 512, fc | 292,714 | 1024- and 512-gate passes ≈ 0; 32-gate pass 0.8%; cycle gain 6.8% |
| cycle 2 complete: 32, r32, 64, 16, 48, 128, r64, 256 with fc between | 285,224 | cycle gain 2.6%; 32/r32 0.5% each, 16 0.3%, 256 ≈ 0 |
| cycle 3 complete (same list) | 281,947 | cycle gain 1.15% |
| cycle 4 (partial: 32-gate pass + fc) → hand-over to the reordered-pass schedule | 281,369 | |
| cycle 5 complete: 32, l32, r32, c32, 16, a16, 64, n64, 48, lr32, 128 with fc between (l/a/c/n = ALAP/ASAP/clustered/random order, r = reversed) | 278,716 | cycle gain 0.94%; per step 0.04–0.17%; reordering before windowing is no better than plain passes |
| cycle 6 complete (same list) | 277,020 | cycle gain 0.61% |
| cycle 7 complete (same list) | 275,779 | cycle gain 0.45% |
| cycle 8 complete (same list) | 274,855 | cycle gain 0.34% |
| cycle 9 complete (same list) | 274,097 | cycle gain 0.28% → stop rule (< 0.3%) fired, 2026-10-26 01:20Z |

Final circuit: `~/shorten/final_274097.mpmct1` on the server (256 wires, 274,097 gates, 22.1% below the 351,744-gate plateau; gzip-6 1.03 MB vs 1.21 MB). End-to-end verification: circuit_check EQUIVALENT to the plateau on 8,192 random full-width states in each direction; clean_payload PASS (10,338 inputs per direction). Controls histogram: 2,718 NOT, 89,742 single-control, 178,364 double-control, 3,273 with three or more. Census of the final circuit: 503 internal / 466 deeper births direct, 566 / 525 affine (plateau: 505 deeper affine) — further compression exposes slightly more of C, as throughout the campaign. Cycle gains: pilots 5.5%, first fcompress 5.5%, cycles 1–9: 6.8, 2.6, 1.15, (partial), 0.94, 0.61, 0.45, 0.34, 0.28%. Total wall time about 16 h on 32–56 cores of a shared box.

12. Sub-280k equivalent circuit (2026-10-25). A shorten_strict/fcompress sweep on the baseline (run/shorten_strict/baseline/, variants fc3_*/fc4_c5_*) produced equivalent circuits at ~279-282k gates; smallest = fc4_c5_n64/round2/output.mpmct1 at 279,263 gates. Verified equivalent to the baseline C (all 64 payload wires match F on 8192 samples). It is ~21% smaller than the 356k plateau, ~58% smaller than the 666k original. Rationale to attack it: smaller SAT, and the compression-increases-exposure trend (666k->36/64, 356k->50/64 degree-2 at the midpoint) predicts higher degree-2 exposure here, possibly approaching all 64 (the threshold for a tractable prefix-only attack). Probe: degree-2 scan at depth 0.48 (run/preimage/d2_short279.json) to measure exposure before committing combos + prefix_quad to this target.
