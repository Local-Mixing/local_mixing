# Source-C design versus proxy exposure (2026-10-21, campaign in progress)

Question (RC): why does the balanced C (every internal value read exactly twice, write counts balanced) leak as
much as the random C after stage 6, is there any parameter on which it is better, and can C be redesigned to
lower exposure without growing it or losing pseudorandomness? Test with the latest stage 2 (three-mask modified)
and stage 6 (v2) on the server.

Inputs: `~/Codex/baseline64_fanout2_pipeline_20261019/C_RESTRUCTURING_EXPERIMENTS_20261021.md` and its audited
per-birth data (`source_topology_followup_20261021/{old,balanced}_features.json`, labels = validated census ids),
plus the censuses of our own matched campaign (`~/threemask_cmp`, memory note threemask-stage6-campaign-20261021).
Analysis scripts: session scratchpad `cdesign/{expo_model.py,design_sim.py,census_by_fanout.py}`
(census_by_fanout.py also at `~/cdesign/py/` on the server).

## 1. Exposure is a per-control-read phenomenon; unread values are immune

Validated affine exposure of the 936 internal births by source control fanout (number of gates that read the
value before it is overwritten):

| run (baseline stage 2) | fanout 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RC original C, adaptive pass 1 | 7/336 (2%) | 65/217 (30%) | 52/130 (40%) | 45/71 (63%) | 43/59 (73%) | 38/48 | 23/27 | 16/17 | 30/31 | 319 |
| RC original C, ordinary | 5/336 (1%) | 76/217 (35%) | 62/130 (48%) | 44/71 (62%) | 44/59 (75%) | 40/48 | 23/27 | 14/17 | 29/31 | 337 |
| our original C, asis round 1 | 4/329 (1%) | 59/201 (29%) | 64/150 (43%) | 54/88 (61%) | 36/57 (63%) | 37/42 | 21/26 | 11/13 | 26/30 | 312 |
| our original C, asis round 4 | 12/329 (4%) | 87/201 (43%) | 96/150 (64%) | 69/88 (78%) | 53/57 | 38/42 | 26/26 | 13/13 | 29/30 | 423 |
| RC balanced C, adaptive pass 1 | – | – | 455/936 (49%) | – | – | – | – | – | – | 455 |

Direct exposure has the same shape at lower levels (fanout 0: 1%, 1: 21%, 2: 28%, 5+: 62-88%). A value that is
never read as a control (36% of a random C's births) stays hidden even after four rounds; each read is roughly an
independent chance of about 0.3 (affine) / 0.2 (direct) of materialisation: 1-(1-p)^k reproduces the table to
within a few points. Lifetime adds nothing once fanout is fixed (RC's balanced analysis, AUC 0.5; in the original
fanout-2 cohort the short-lived values are if anything more exposed). The few exposures in the three-mask arms
(3-8 values after three asis rounds) are all fanout >= 2 values; none is unread.

Mechanism (consistent with the three-mask review, finding 5): the synthesizer only needs an unmasked C value when
it re-implements a gate that multiplies it; a value that is only ever a target accumulates inside its masked wire
and never has to exist in the clear.

## 2. Why the balanced C leaks as much, and what it does improve

The balanced C removes the 253 hubs (fanout >= 3, 62-97% exposed) but also removes the 336 unread values (2%) and
the 217 single-read values (30%), making all 936 values fanout-2 (40-49%). Reads per gate are fixed at two
(g57 gates), so the 2,000 reads are only redistributed. Same census, same driver (RC's adaptive pass 1):

| metric (affine; direct in parentheses) | original C | balanced C |
| --- | --- | --- |
| exposed births | 319 (239) | 455 (283) |
| reads-weighted exposure: fraction of control reads whose source value is exposed | 0.69 (0.55) | 0.49 (0.30) |
| C gates with both controls exposed (inputs count as public; floor ~29) | 462 (307) | 248 (127) |
| C gates whose increment is exposed (target value before and after both exposed) | 120 (67) | 241 (96) |

So the balanced C is worse on the count and on target-chain exposure, and about twice as good on the read side:
half as many gates have both multiplicands in the clear, and no exposed value serves more than two gates. Which
metric is the security-relevant one depends on the attacker; for reconstructing C's gates from exposed functions
the read-side metrics are the natural ones, and there the balanced C is near-optimal (k*f(k) is convex, so with
sum k fixed the all-equal distribution minimises it), while the count f(k) is concave, so the same distribution
maximises the count. This is the whole story of "balanced = same leak": the two objectives are duals.

The AUC 0.5 result for the balanced C follows: when every value has the same read count, which ones get
exposed is decided by the synthesizer's choices downstream, not by any source feature.

## 3. Design space: concentrating reads lowers the count but breaks avalanche at 1,000 gates

Simulated designs (seed 7), predicted with the fanout rates of RC's adaptive pass 1, pseudorandomness proxies on
the actual circuits (SAC = per input/output flip probability over 4,096 inputs; full-degree fraction = share of
(random 12-variable subcube, output) pairs whose restriction has degree 12, 0.5 for a random function):

| design | unread values | predicted affine count | reads-weighted | gates both | increments | SAC max bias | full-degree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random (RC's C) | 336 | 319 | 0.69 | 469 | 125 | 0.027 | 0.49 |
| balanced (RC's C) | 0 | 374 (observed 455) | 0.40 | 185 (observed 248) | 165 (observed 241) | 0.033 | 0.50 |
| burst q=0.5 (sticky targets) | 602 | 225 | 0.81 | 572 | 68 | 0.36 (dead pairs) | 0.50 |
| burst q=0.67 | 738 | 160 | 0.88 | 650 | 35 | 0.50 | 0.48 |
| phased hot set 16/100 | 809 | 140 | 0.97 | 842 | 22 | 0.50 | 0.35 |
| preferential reads w=4 | 730 | 129 | 0.89 | 699 | 30 | 0.50 | 0.05 |

Every design that lowers the count does so by leaving values unread, and a random 64-wire circuit needs about
1,000 gates (about 590 read values) before every input bit reaches every output bit (800 gates: SAC bias 0.12;
650: dead pairs). The random 1,000-gate C sits at that floor, so there is no slack to concentrate reads without
losing avalanche, short of adding gates (which adds births one-for-one). Within g57 circuits of this size, C
design redistributes exposure; it does not reduce it. The lever that does reduce it is the encoding and the
synthesizer policy: three-mask stage 2 brings one asis round from 205-312 exposed values to 2-3, and v2 (keep-
smaller-and-not-leakier acceptance) to 0 for the original C.

## 4. Server campaign (`~/cdesign` on machine1, launched 17:30 UTC)

Same SEED, D, sandwich recipe, stage-2 flags and stage 3-5 seeds as the matched campaign, generator extended
with `--source-file` (g57 triples). Arms: RC's balanced C (byte-exact reproduction, sha d0586653...) and the
burst q=0.67 C (model test only; its avalanche is broken) x stage 2 {baseline managed_tdp, three-mask modified}.
Stage-2 sizes are unchanged by the C (777k/780k baseline, 505k three-mask, vs 779k/505k for the original C).
Downstream: stages 3-5 (reduced profile), then stage 6 asis (two rounds) for the *_base arms and v2 to plateau
plus the resilience campaign for the *_v2 arms, plus the raw + nine-arm narrow portfolio on all stage-5 outputs.

Predictions (affine, one asis round; original-C reference in our pipeline 312 exposed, 412 gates-both):
balanced 374-455 exposed / 185-250 gates-both; burst67 about 160 exposed / 650 gates-both / 35 increments,
with the unread 738 values essentially absent. Under three-mask modified + v2 all C's should give 0/0; the
discriminating data there are the C-blind resilience attacks and the narrow portfolio.

## 5. Results so far (2026-10-22 05:30 UTC)

Overnight the server's disk filled (observer scratch of the narrow portfolio, ~50 GB) and killed every run on
both campaigns at about 03:30 UTC; recovered at 05:20 UTC (incomplete observers deleted, completed ones trimmed,
loops and attacks relaunched with two workers). Stage 6 round 1 of all four arms and round 2 of the two
baseline-stage-2 arms completed before the outage.

### Stage 6, baseline stage 2, asis policy (model test)
| C | round | gates in -> out | exposed direct / affine (of 936) | per-fanout affine | reads-weighted | gates both | increments |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original (matched campaign) | 1 | 5,508,346 -> 2,831,405 | 222 / 312 | 0: 4/329, 1: 59/201, 2: 64/150, 3: 54/88, 4: 36/57, 5+: 95/111 | 0.65 | 412 | 110 |
| original | 2 | 2,831,405 -> 1,905,790 | 327 / 382 | 0: 9/329, 1: 80/201, 2: 78/150, 3: 67/88, 4: 43/57, 5+: 105/111 | 0.76 | 549 | 157 |
| balanced (RC's) | 1 | 5,488,705 -> 2,830,920 | 328 / 453 | 2: 453/936 | 0.48 | 230 | 236 |
| balanced | 2 | 2,830,920 -> 1,919,514 | 464 / 582 | 2: 582/936 | 0.62 | 381 | 378 |
| burst q=0.67 | 1 | 5,506,304 -> 2,991,300 | 82 / 124 (of 937) | 0: 5/738, 1: 3/30, 2: 10/21, 3: 12/24, 4: 9/20, 5+: 85/104 | 0.79 | 560 | 26 |
| burst q=0.67 | 2 | 2,991,300 -> 1,857,096 | 145 / 165 | 0: 5/738, 1: 10/30, 2: 11/21, 3: 20/24, 4: 17/20, 5+: 102/104 | 0.94 | 712 | 38 |

Predictions from section 3 were: balanced 374-455 exposed / 185-250 gates-both (observed 453 / 230); burst about
160 exposed / 650 gates-both / 35 increments with the unread values absent (observed 124 / 560 / 26, and 5 of the
738 unread values exposed). The per-read model is causal: the same pipeline, seed, D and sandwich, only C changed.
Balanced C: +45% exposed values, half the gates with both controls exposed, twice the exposed increments.
Burst C: -60% exposed values, +36% gates with both controls exposed, a quarter of the increments, and in round 2
98% of its read values are exposed. Note the denominators: the boundary classification gives 49 public-quadratic
births for the balanced C and 73 for the burst C (60 for the original).

### Stage 6, three-mask modified stage 2, v2 policy
| C | round | gates in -> out | exposed direct / affine | note |
| --- | --- | --- | --- | --- |
| original (matched campaign) | 1 | 3,497,436 -> 3,208,699 | 0 / 0 | all-internal affine 1 |
| original | 2 | 3,208,699 -> 2,971,913 | 0 / 0 | |
| balanced | 1 | 3,608,410 -> 3,280,215 | 0 / 0 | all-internal affine 2 (boundary values) |
| balanced | 2 | 3,280,215 -> 3,030,935 | 0 / 0 | |
| balanced | 3 | 3,030,935 -> 2,927,888 | 0 / 0 | |
| balanced | 4 | 2,927,888 -> 2,866,390 | 0 / 0 | |
| burst q=0.67 | 1 | 3,517,946 -> 3,221,692 | 0 / 0 | all-internal affine 1 |
| burst q=0.67 | 2 | 3,221,692 -> 2,989,173 | 0 / 0 | |
| burst q=0.67 | 3 | 2,989,173 -> 2,875,930 | 0 / 0 | |
| burst q=0.67 | 4 | 2,875,930 -> 2,794,183 | 0 / 0 | |

As expected, the encoding plus v2 acceptance keeps every C at 0-2 exposed values; stage-5 sizes differ by 3% at
most (3.61M balanced, 3.52M burst, 3.50M original). Later rounds, the narrow portfolio and the resilience attacks
are running (see the memory note cdesign-campaign-20261021 for how to read them).

(Update 2026-10-22 17:40 UTC: the "deeper" columns above are the validated deeper counts; the 0/2 and 1/1 quoted
earlier for round 1 were all-internal counts that include public-quadratic boundary births. Rounds 2-4 keep every C
at 0/0 deeper while sizes still fall 2-3% per round; no plateau yet. In the matched campaign, baseline/asis
plateaued after 24 rounds at 666,745 gates with 384/424 deeper exposed, and three-mask-modified/asis after 12 rounds
at 2,761,726 gates with 14/14; their resilience campaigns started 14:05 UTC.)

### Narrow portfolio on the stage-5 outputs (balanced arms complete, 9 cases; burst arms 6 of 9)
| arm | raw deeper R/O | narrow deeper R/O (cases done) | narrow boundary R/O |
| --- | --- | --- | --- |
| original, baseline stage 2 | 0/0 | 0/0 (9) | 4/0 |
| original, three-mask modified | 0/0 | 7/3 (9) | 10/0 |
| balanced, baseline stage 2 | 0/0 | 1/0 (9) | 12/0 |
| balanced, three-mask modified | 0/0 | 14/2 (9) | 15/0 |
| burst, baseline stage 2 | 0/0 | 1/0 (6) | 8/0 |
| burst, three-mask modified | 0/0 | 6/1 (6) | 6/0 |

## 6. C values present at more than one location of the mixed circuit (direct copies)

Direct census of each stage-6 output (`py/locations.py` on the server): one "location" = one physical wire
version equal to the C value (or its complement) over its live range; control reads counted on the output circuit.
No run has two copies of the same value live at the same time: copies are sequential re-materialisations inside
different blocks, which is what the per-read model predicts (each read of the value is a separate chance to
rebuild it; stage 5 holds no value on any wire at block boundaries).

| C, stage 2, policy | round | directly exposed values | values at >= 2 locations | copies (versions) | max copies of one value | control reads of all copies | max reads of one copy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original, baseline, asis | 1 | 222 | 85 (38%) | 365 | 11 | 2,317 | 35 |
| original, baseline, asis | 2 | 327 | 142 (43%) | 641 | 34 | 4,158 | 68 |
| original, baseline, asis | 4 | 376 | 169 (45%) | 825 | 64 | 6,000 | 80 |
| balanced, baseline, asis | 1 | 328 | 75 (23%) | 468 | 34 | 2,714 | 32 |
| balanced, baseline, asis | 2 | 464 | 146 (31%) | 809 | 70 | 4,838 | 41 |
| burst, baseline, asis | 1 | 82 | 38 (46%) | 162 | 10 | 946 | 54 |
| burst, baseline, asis | 2 | 145 | 95 (66%) | 450 | 46 | 3,420 | 57 |
| original, three-mask, asis | 4 | 7 | 1 | 8 | 2 | 51 | 17 |
| original, three-mask, v2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| balanced, three-mask, v2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| burst, three-mask, v2 | 1 | 1 | 0 | 1 | 1 | 17 | 17 |

Copies per exposed value grow with source fanout (original C, round 1: fanout 1 values 1.8 copies and 31% at
>= 2 locations; fanout 5+ values 1.9 copies and 53%; round 4: 5+ values 3.9 copies, 80% multi). The balanced C has
the lowest multi-location fraction (23-31%) but, having 45% more exposed values, about the same number of
multi-location values as the original (75/146 vs 85/142) and more copies and copy reads in total. The burst C has
the fewest multi-location values (38/95) but the highest fraction, because what it exposes are its 8+-read hubs.
RC's balanced adaptive pass 1 (Codex driver) had 381 versions / 2,414 reads for 283 direct values, the same
1.3-1.4 copies per value as here.

## 7. C values found at more than one location of the stage-5 mixed circuits (narrow portfolio)

Six full cases per arm (raw, factor, factor_perm, factor_shift63, factor_w250, pairs) on the stage-5 outputs; a
"location" is a validated representation found by the observers (fresh 8,192 + structured inputs); R = values at
two or more locations, O = once. Boundary = public-quadratic births (49 balanced / 60 original / 73 burst).

| C, stage 2 | deeper values found | R | O | max locations | boundary values found (all R) |
| --- | --- | --- | --- | --- | --- |
| original, baseline | 0 | 0 | 0 | - | 4 |
| balanced, baseline | 1 | 1 | 0 | 4 | 12 |
| burst, baseline | 1 | 1 | 0 | 4 | 8 |
| original, three-mask modified | 6 | 4 | 2 | 9 | 10 |
| balanced, three-mask modified | 15 | 13 | 2 | 6 | 15 |
| burst, three-mask modified | 7 | 6 | 1 | 8 | 6 |
| original, three-mask as-is | 11 | 9 | 2 | 8 | 5 |

All but two of the R values come from the 250-gate factoring window (factor_w250), the case that recovers whole
fires (three-mask review, finding 5). Fanout of the deeper values found: original three-mask 1,1,1,3,5,5;
balanced all 2 (2-6 locations each); burst 2,3,12,13,14,17,30 (its hubs). The balanced C gives the narrow
observer the most repeated values because every value is read twice, so every recovered fire control has a
second fire to be matched against; the original's single-read values can only be found once.

## 8. Resilience attacks on the plateau outputs (original C, matched campaign; 2026-10-23 05:40 UTC)

Four of the five original-C stage-6 runs reached the two-zero-gain plateau and their resilience campaigns are
running (`resilience.py`: narrow portfolio on the plateau output F, C-blind adaptive resynthesis of F, stage-3
remix of F followed by C-blind resynthesis; census and raw ledger at every step). Deeper counts, 876 denominator:

| run | plateau F (gates, rounds) | census on F direct/affine | narrow portfolio on F (R/O, cases) | C-blind resynthesis of F (rounds, last exp) | remix + C-blind (rounds, last exp) |
| --- | --- | --- | --- | --- | --- |
| baseline stage 2, asis | 666,745 (24) | 384 / 424 | 436 / 7 (6) | 13+ rounds, 444 / 484 | 11+ rounds, 421 / 475 |
| three-mask modified, asis | 2,761,726 (12) | 14 / 14 | 104 / 53 (6) | 7+ rounds, 16 / 16 | 2+ rounds, 23 / 23 |
| three-mask modified, v2 | 2,706,063 (9) | 0 / 0 | **157 / 76 (9)** | 6 rounds (plateau), 6 / 6 | 2+ rounds, 14 / 14 |
| three-mask as-is, asis | 2,611,000 (16) | 21 / 21 | 3 / 18 (1 case so far) | 1+ rounds | - |

The important row is the v2 one: the all-cuts census finds nothing on its plateau output, yet the C-blind narrow
portfolio finds 233 deeper values (157 at two or more locations) on the same circuit, against 10 (7 repeated) on
the stage-5 circuit it came from. Every case contributes (factor 39/28, factor_perm 52/32, shift63 43/19, w250
74/48, payload w125 30/51, w250 61/74, delta 69/83). By source fanout the found values are 0: 7, 1: 36, 2: 45,
3: 43, 4: 26, 5+: 76, so even seven unread values are reached. Reading: the v2 acceptance rule only sees the census
family; the compact ANF-synthesised blocks it accepts are much easier for windowed factoring than the mixed
stage-5 gate stream (fire fragments shrink, so more of them fit in a 125-250-gate window). The raw ledger on the
C-blind-resynthesised F finds 1/5, on the remixed F 0/0: remixing restores the stage-5 behaviour.

For the C-design question the same measurement on the balanced and burst plateau outputs is what will decide
whether C matters under three-mask + v2; their runs are at rounds 6-7 (balanced 2.79M, burst 2.70M gates), still
shrinking 1-2% per round.

Update 2026-10-23 17:45 UTC. All five original-C runs have plateaued (asis/v2 after 14 rounds at 2,368,032 gates,
0/0) and every resilience campaign is running. Narrow portfolio on the plateau outputs, six full cases each,
deeper R/O: baseline/asis 436/7, three-mask as-is/asis 198/52, three-mask as-is/v2 229/46, three-mask
modified/asis 104/53, three-mask modified/v2 117/57 (157/76 with the three payload cases). C-blind resynthesis
plateaus: modified/v2 6/6 (6 rounds), modified/asis 17/17 (9 rounds); remix + C-blind reaches 21-27 after 7-8
rounds. The modified three-mask stage 2 roughly halves the repeated count of the as-is one on the plateau outputs.
The balanced and burst v2 runs are at round 9 (2.74M and 2.67M gates, 0/0), shrinking 0.4-0.5% per round, not yet
at the two-zero-gain plateau.

Update 2026-10-24 05:45 UTC. Balanced and burst v2 runs at round 12 (2,714,468 and 2,644,607 gates), 0/0 deeper
every round, shrinking 0.3% per round; no plateau yet. Original-C resilience: modified/asis campaign complete
(DONE); raw ledgers after C-blind resynthesis: baseline/asis 410/59 (remix 347/48, remix+resynth 415/62),
modified/v2 1/5 (remix 0/0), as-is/asis remix 15/6, as-is/v2 remix 0/0. Disk 41 GB free; the finished attack
circuits of the new arms (23 GB) are being gzip-compressed in place.

Update 2026-10-24 17:55 UTC. Balanced at round 14 (2,703,707 gates), burst at round 15 (2,628,101), 0/0 deeper in
every round, now shrinking 0.2% per round; the two-zero-gain plateau has still not triggered. To get the decisive
C comparison without waiting, the C-blind narrow portfolio (`py6/early_narrow.py`, results under
`run/early_narrow/<arm>_r<round>` and `run/ledgers/<arm>_r<round>`) was started on the round-14 balanced output
and the round-15 burst output; these are the same size as the original C's plateau output (2,706,063 gates,
117/57 R/O on six cases, 157/76 on nine). Disk back to 53 GB after compressing the finished attack circuits.

## 9. The decisive comparison: C-blind narrow portfolio on the three-mask + v2 stage-6 outputs (2026-10-24 20:15 UTC)

Nine cases each, on outputs of equal size: the original C's plateau output (round 9) and the balanced round-14
and burst round-15 outputs (not yet at plateau, 0/0 under the census like the original). Deeper denominators
876 / 887 / 864.

| C | stage-6 output (gates) | deeper values found (R / O) | by source fanout 0 / 1 / 2 / 3 / 4 / 5+ | reads-weighted | gates with both controls known | increments known |
| --- | --- | --- | --- | --- | --- | --- |
| original | 2,706,063 | 233 (157 / 76) | 7 / 36 / 45 / 43 / 26 / 76 | 0.51 | 338 | 119 |
| balanced | 2,703,707 | 265 (172 / 93) | - / - / 265 / - / - / - | 0.28 | 151 | 129 |
| burst q=0.67 | 2,628,101 | 108 (76 / 32) | 1 / 5 / 5 / 11 / 9 / 77 | 0.73 | 584 | 84 |
| (original, baseline stage 2, asis plateau) | 666,745 | 450 (444 / 6) | 23 / 96 / 103 / ... | 0.82 | 721 | 292 |

Per-fanout find rates on the original C are 2 / 18 / 30 / 49 / 46 / 68 %; the balanced C's single cohort is found at
30% (265/887), exactly the original's fanout-2 rate, and the burst C's hubs at 74% (77/104) with 1 of its 738 unread
values found. So the per-read law holds for the C-blind windowed observers on the compressed output just as it did
for the C-aware census under the baseline encoding, and the C-design trade-off is unchanged: balanced +14% count,
half the gates with both controls known; burst -54% count, 1.7x the gates with both controls known. Every case
contributes for every C (factor 43-81 R for balanced, 24-44 for burst); pairs finds 4 hubs on burst only.

Update 2026-10-25 08:20 UTC. Balanced and burst v2 runs at round 18 (2,683,646 and 2,615,496 gates), 0/0 deeper in
every round, shrinking 0.1-0.2% per round; the two-zero-gain plateau has still not triggered. All five original-C
resilience campaigns have finished (the matched tree ~/threemask_cmp is now driven by another session:
launch_resilience3.sh, a second remix step "remix2"; nothing there was touched from here). Final table, deeper
counts (876):

| run | plateau F | narrow on F (6 cases) | C-blind resynthesis of F: rounds, last census | raw ledger after | remix + C-blind: rounds, last census | raw ledger after |
| --- | --- | --- | --- | --- | --- | --- |
| baseline / asis | 666,745 | 436 / 7 | 13+, 444 / 484 | 410 / 59 | 11+, 421 / 475 | 415 / 62 |
| as-is / asis | 2,611,000 | 198 / 52 | 10+, 39 / 39 | 13 / 29 | 10+, 37 / 37 | 23 / 17 |
| as-is / v2 | 2,368,032 | 229 / 46 | 10+, 9-11 / 9-11 | 2 / 7 | 10+, 37 / 37 | 18 / 21 |
| modified / asis | 2,761,726 | 104 / 53 | 9 (plateau), 17 / 17 | 2 / 15 | 10+, 27 / 27 | 9 / 17 |
| modified / v2 | 2,706,063 | 117 / 57 | 6 (plateau), 6 / 6 | 1 / 5 | 10+, 23 / 23 | 5 / 19 |

The C-blind adaptive resynthesis (largest feasible blocks, keep-smaller) stays in single digits to the thirties for
every three-mask arm, so the census family is robust there; the windowed narrow observers are the ones that
recover hundreds of values from the compressed outputs.

Update 2026-10-25 20:20 UTC. Balanced and burst v2 runs at round 21 (2,672,729 and 2,605,062 gates), 0/0 deeper in
every round, gains now ~0.1% per round and the two-zero-gain plateau still not reached (the fresh seeds keep
finding small gains). Rather than wait, the C-blind adaptive resynthesis attack (resilience step 2, variant
`attacker`: largest feasible blocks, keep-smaller, no C information) was started on both round-21 outputs
(`run/early_resil/<arm>_r21/<arm>/attacker/round*/result.json`); the original C's reference is 3 -> 6 deeper after
6 rounds (plateau) on its 2.71M-gate output. The remix step can follow on the same outputs if wanted.

Update 2026-10-26 08:25 UTC. C-blind adaptive resynthesis of the round-21 outputs, deeper direct/affine per round:
balanced 1/1 -> 4/4 -> 9/9 (rounds 1-3, 2.67M -> 2.62M gates); burst 1/1 -> 1/1 -> 0/0 (2.61M -> 2.57M). Original C
reference: 3/3 -> 4/4 -> 6/6, plateau 6/6 after round 6. The v2 runs themselves are at rounds 23-24, still 0/0 and
still not at plateau. Disk 41 GB.
