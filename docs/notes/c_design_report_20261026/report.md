# Source-C architecture and proxy exposure: summary of the 2026-10-21 to 2026-10-26 investigation

*Prepared 2026-10-26. Campaign still running on the server (see section 7); every number below is from a
completed, independently validated measurement unless marked as in progress.*

## 1. Questions

RC asked three things after the C-restructuring experiments (`C_RESTRUCTURING_EXPERIMENTS_20261021.md`):

1. Why does the balanced C (every internal value read exactly twice as a control, write counts balanced at every
   cut) leak as many internal values after stage 6 as the random C, and is there any parameter on which it is
   better?
2. Is there a way to design C so that fewer of its internal values are exposed, without growing it or losing the
   pseudorandomness of the function it computes?
3. Test with the latest stage-2 compute (three-mask, modified) and the latest stage 6 (adaptive, v2), on the server.

Short answers: (1) exposure is a per-control-read phenomenon, the balanced C only redistributes the 2,000 reads,
and it does win on read-side metrics (gates with both controls exposed are halved) while losing on the count;
(2) within g57 circuits of 1,000 gates on 64 wires, no: every read distribution that lowers the count breaks
avalanche, because the random C already sits at the minimum number of read values that full diffusion needs;
(3) under three-mask + v2 every C gives 0/0 in the census, and the C-blind windowed observers on the stage-6
outputs reproduce exactly the same per-read law, so C design redistributes exposure there as well.

## 2. What was measured, and where in the pipeline

Pipeline per arm: stage 1 sandwich (C on 64 wires, 1,000 g57 gates, plus D and slicing, 2,832 gates) -> stage 2
compute block (either the pinned baseline `managed_tdp`, ~778k gates, or the three-mask compute with RC's
modifications `--random-helpers --unique-cover`, ~505k gates) -> stages 3-5 (DB mixing, split, crossing, reduced
profile 2,2,2,2,2; ~5.5M or ~3.5M gates) -> stage 6 adaptive resynthesis, policy `asis` (one seed, keep-smaller)
or `v2` (two seeds, keep-smaller-and-not-leakier, moved boundaries, size-scaled budgets). One master seed shared
by every arm, so D, the sandwich and all stage seeds are identical; only C (and the stage-2 block) differ.

Measurement families, all on the honest-slice inputs and all re-validated on 8,192 fresh plus structured inputs:

- **Census** (C-aware, every cut, every wire): a C birth is *directly* exposed if some wire version equals it or
  its complement; *affinely* exposed if it is an affine combination of simultaneously live wires. Reported on
  stage-6 outputs; every stage-5 circuit in this study has 0/0.
- **Narrow portfolio** (C-blind): nine windowed resynthesis attacks on a circuit (raw, factor, factor_perm,
  factor_shift63, factor_w250, pairs, payload factor_w125 / factor_w250 / delta_w250), then an observer scores the
  attacked circuit; counts are deeper C values found at >= 2 locations (R) or once (O). "Deeper" excludes the
  public-quadratic boundary births (60 for the original C, 49 balanced, 73 burst).
- **C-blind adaptive resynthesis** (resilience step): stage 6 with no C information (largest feasible blocks,
  keep-smaller) run to plateau on an output, census after each round.

Three C's: the campaign's random C (geometric read distribution: 336 of 936 internal births never read as a
control, 217 read once, 130 twice, 253 hubs with >= 3 reads), RC's balanced C (reproduced byte-exact, sha
d0586653...), and a sticky-target "burst" C (q = 0.67: 738 of 937 births unread, 104 hubs with 8+ reads) built
as a model test; its avalanche is broken on purpose, so it is a diagnostic, not a candidate.

## 3. Exposure is per control read; unread values are immune

Validated affine exposure of the internal births after one stage-6 round, by source control fanout (number of
gates that read the value before it is overwritten):

| run (baseline stage 2) | fanout 0 | 1 | 2 | 3 | 4 | 5+ | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RC original C, adaptive pass 1 | 7/336 (2%) | 65/217 (30%) | 52/130 (40%) | 45/71 (63%) | 43/59 (73%) | 107/123 (87%) | 319 |
| our original C, asis round 1 | 4/329 (1%) | 59/201 (29%) | 64/150 (43%) | 54/88 (61%) | 36/57 (63%) | 95/111 (86%) | 312 |
| our original C, asis round 4 | 12/329 (4%) | 87/201 (43%) | 96/150 (64%) | 69/88 (78%) | 53/57 (93%) | 106/111 (95%) | 423 |
| RC balanced C, adaptive pass 1 | - | - | 455/936 (49%) | - | - | - | 455 |

Direct exposure has the same shape at lower levels (1 / 21 / 28 / 42-46 / 47-51 / 62-88 %). Each read is roughly
an independent chance of about 0.3 (affine) or 0.2 (direct) that the synthesizer materialises the value: it only
needs a C value in the clear when it re-implements a gate that multiplies it, whereas a value that is only ever a
target accumulates inside its masked wire. Lifetime adds nothing once fanout is fixed (RC's 52-feature analysis of
the balanced C is at chance; in the original fanout-2 cohort the short-lived values are, if anything, more
exposed). The few values exposed in the three-mask arms (3-16 after several asis rounds) are all read values.

## 4. Why the balanced C leaks as much, and what it improves

Reads per gate are fixed at two, so a design can only redistribute 2,000 reads over 936 internal births (plus 64
inputs). The balanced C removes the 253 hubs but also the 553 values read zero or one time, moving everything to
the two-read rate. Same census, same driver (RC's adaptive pass 1), affine with direct in parentheses:

| metric | original C | balanced C |
| --- | --- | --- |
| exposed births | 319 (239) | 455 (283) |
| fraction of control reads whose source value is exposed | 0.69 (0.55) | 0.49 (0.30) |
| C gates with both controls exposed (inputs public; floor ~29) | 462 (307) | 248 (127) |
| C gates whose increment is exposed (target before and after) | 120 (67) | 241 (96) |

The count f(k) is concave in the reads k, so the equal distribution maximises it; k*f(k) is convex, so the same
distribution minimises read-weighted exposure and, approximately, the number of gates whose two multiplicands are
both in the clear. Those two objectives are duals; that is the whole story of "balanced = same leak". It also
explains the AUC 0.5: with every value at the same read count, which ones are exposed is decided downstream by
the synthesizer, not by any source feature.

## 5. The design space: concentrating reads lowers the count but breaks avalanche

Candidate designs (seed 7) with the count predicted from the fanout rates above, and pseudorandomness proxies
measured on the circuits (SAC = per input/output bit flip probability over 4,096 inputs; full-degree fraction =
share of (random 12-variable subcube, output) pairs whose restriction has degree 12, 0.5 for a random function):

| design | unread values | predicted affine count | reads-weighted | gates both | increments | SAC max bias | full-degree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random (RC's C) | 336 | 319 | 0.69 | 469 | 125 | 0.027 | 0.49 |
| balanced (RC's C) | 0 | 374 (obs. 455) | 0.40 | 185 (obs. 248) | 165 (obs. 241) | 0.033 | 0.50 |
| burst q = 0.5 | 602 | 225 | 0.81 | 572 | 68 | 0.36 (dead pairs) | 0.50 |
| burst q = 0.67 | 738 | 160 | 0.88 | 650 | 35 | 0.50 | 0.48 |
| phased hot set 16/100 | 809 | 140 | 0.97 | 842 | 22 | 0.50 | 0.35 |
| preferential reads w = 4 | 730 | 129 | 0.89 | 699 | 30 | 0.50 | 0.05 |

A random 64-wire g57 circuit needs about 1,000 gates, equivalently about 590 read values, before every input bit
reaches every output bit (650 gates: dead input/output pairs; 800 gates: SAC bias 0.12; 1,000: 0.03). The random
1,000-gate C sits at that floor. Every design that lowers the exposed count does so by leaving values unread, and
every such design that lowers it by more than a few percent loses avalanche. Adding gates to compensate adds
births one for one. Within this family, C design redistributes exposure; it does not reduce it.

## 6. Server campaign: matched arms with three C's

Tree `~/cdesign` on machine1, same SEED as the matched three-mask campaign, generator extended with
`--source-file`. Stage-2 sizes are unchanged by the C (777-780k baseline, 505k three-mask).

### 6.1 Causal test under the baseline stage 2 (asis policy)

| C | round | gates in -> out | exposed direct / affine | per-fanout affine | reads-weighted | gates both | increments |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 1 | 5,508,346 -> 2,831,405 | 222 / 312 | 4/329, 59/201, 64/150, 54/88, 36/57, 95/111 | 0.65 | 412 | 110 |
| original | 2 | 2,831,405 -> 1,905,790 | 327 / 382 | 9, 80, 78, 67, 43, 105 | 0.76 | 549 | 157 |
| balanced | 1 | 5,488,705 -> 2,830,920 | 328 / 453 | 453/936 at fanout 2 | 0.48 | 230 | 236 |
| balanced | 2 | 2,830,920 -> 1,919,514 | 464 / 582 | 582/936 | 0.62 | 381 | 378 |
| burst | 1 | 5,506,304 -> 2,991,300 | 82 / 124 | 5/738, 3/30, 10/21, 12/24, 9/20, 85/104 | 0.79 | 560 | 26 |
| burst | 2 | 2,991,300 -> 1,857,096 | 145 / 165 | 5/738, 10, 11, 20, 17, 102/104 | 0.94 | 712 | 38 |

Predictions from section 5 were 374-455 exposed / 185-250 gates-both for balanced (observed 453 / 230) and about
160 / 650 / 35 for burst with the unread values absent (observed 124 / 560 / 26, with 5 of 738 unread values
exposed). Same pipeline, seed, D and sandwich; only C changed, so the read structure is the cause.

### 6.2 Three-mask modified stage 2 with v2 stage 6

Census on every round's output: 0/0 deeper for all three C's (original: 9 rounds to plateau at 2,706,063 gates;
balanced: 37 rounds so far; burst: 38 rounds; both still shrinking 0.1% per round, the
two-zero-gain plateau not yet reached). Stage-5 sizes differ by at most 3% across the C's.

Narrow portfolio on the stage-5 outputs (nine cases): original 7/3 deeper R/O, balanced 14/2, burst 6/2 under
three-mask; 0/0, 1/0, 2/0 under baseline stage 2. The balanced C hands the windowed observer the most repeated
values because every value is read twice, so every recovered fire control has a second fire to be matched against.

### 6.3 The decisive comparison: C-blind narrow portfolio on the stage-6 outputs

Nine cases on outputs of equal size (original plateau output; balanced round 14; burst round 15), all 0/0 under
the census:

| C | gates | deeper found (R / O) | by fanout 0 / 1 / 2 / 3 / 4 / 5+ | reads-weighted | gates both controls known | increments known |
| --- | --- | --- | --- | --- | --- | --- |
| original | 2,706,063 | 233 (157 / 76) | 7 / 36 / 45 / 43 / 26 / 76 | 0.51 | 338 | 119 |
| balanced | 2,703,707 | 265 (172 / 93) | all 265 at fanout 2 | 0.28 | 151 | 129 |
| burst | 2,628,101 | 108 (76 / 32) | 1 / 5 / 5 / 11 / 9 / 77 | 0.73 | 584 | 84 |
| original, baseline stage 2, asis plateau | 666,745 | 450 (444 / 6) | 23 / 96 / 103 / ... | 0.82 | 721 | 292 |

Find rates by fanout on the original C are 2 / 18 / 30 / 49 / 46 / 68 %. The balanced C's single cohort is found
at 30%, the original's fanout-2 rate; the burst C's hubs at 74% with 1 of 738 unread values found. The per-read
law therefore holds for C-blind observers on the compressed output exactly as it does for the C-aware census under
the baseline encoding, and with the same trade-off: balanced +14% count and half the gates with both controls
known; burst -54% count and 1.7x the gates with both controls known.

C-blind adaptive resynthesis of the round-21 outputs, deeper direct/affine per round (stopped after twelve
rounds): balanced 1, 4, 9, 10, 10, 10, 12, 12, 13, 14, 16, 17; burst 1, 1, 0, 1, 3, 2, 2, 2, 2, 2, 2, 3; original
reference 3, 4, 6, 6, 6, 6 with plateau at round 6.

### 6.4 Where the values appear

Direct copies in the stage-6 outputs (baseline stage 2, asis round 1): original 222 exposed values, 85 at two or
more locations, 365 physical copies carrying 2,317 control reads; balanced 328 / 75 / 468 / 2,714; burst 82 / 38 /
162 / 946. No value is ever live at two locations at the same time in any run: copies are sequential
re-materialisations in different blocks. Copies per value grow with fanout (original round 4: 5+ values 3.9
copies, 80% multi-location).

Fire fragments: a fire is 170 stage-2 gates, about 1,200 stage-5 gates after the 7.1x expansion, five 250-gate
windows. The 20 product units are emitted in groups sharing the left factor (17-35 stage-2 gates, 120-250 at
stage 5), and any group sums to L * x_v, so one group in a window already yields x_v. That is why the 250-gate
factoring finds 4-13 repeated values on stage-5 circuits while the 125-gate window finds about one, and why the
finds are spread along a value's lifetime rather than clustered at its consumer fires.

## 7. Side findings on stages 2 and 6 (original C, matched campaign)

| stage 2 / stage 6 | plateau output (rounds) | census | narrow on output (6 cases) | C-blind resynthesis, last | raw ledger after |
| --- | --- | --- | --- | --- | --- |
| baseline / asis | 666,745 (24) | 384/424 | 436 / 7 | 444/484 | 410 / 59 |
| three-mask as-is / asis | 2,611,000 (16) | 21/21 | 198 / 52 | 39/39 | 13 / 29 |
| three-mask as-is / v2 | 2,368,032 (14) | 0/0 | 229 / 46 | 9-11 | 2 / 7 |
| three-mask modified / asis | 2,761,726 (12) | 14/14 | 104 / 53 | 17/17 (plateau) | 2 / 15 |
| three-mask modified / v2 | 2,706,063 (9) | 0/0 | 117 / 57 (157 / 76 on 9 cases) | 6/6 (plateau) | 1 / 5 |

- Stage-6 compression exposes the mixed output to the windowed observers: 157/76 repeated/once on the v2 plateau
  output against 10 (7 repeated) on the stage-5 circuit it came from. The v2 acceptance rule only sees the census
  family. Remixing the output (stage 3 again) brings the raw ledger back to 0/0.
- v2 is no better than asis against the C-blind windowed observers; it is better in the census family (0 vs
  14-21) and under C-blind adaptive resynthesis (6 vs 17-39).
- RC's two stage-2 modifications (random dirty helpers, unique covers) roughly halve the repeated count of the
  as-is three-mask compute on the plateau outputs (104-117 vs 198-229).

## 8. Conclusions

1. Exposure after resynthesis is driven by control reads of C's values: about 0.3 per read for the affine family
   under the baseline encoding, about 0.25 per read for the C-blind windowed observers on three-mask + v2 outputs,
   and essentially zero for values that are never read.
2. The balanced C does not reduce exposure because it cannot: with two reads per gate the design only chooses
   who absorbs the reads. It maximises the number of exposed values (+40-45%) and minimises the read-side
   quantities (gates with both multiplicands known halved, no exposed value serving more than two gates); which
   of these matters depends on what the attacker needs to reconstruct C.
3. No C design in this family lowers exposure without losing avalanche: the random 1,000-gate C already has the
   fewest read values compatible with full diffusion on 64 wires. Concentrating reads (bursts, hot sets,
   preferential reads) cuts the count by up to 60% in the model and in the measurement, and breaks avalanche in
   every variant tried.
4. The lever that reduces exposure is the encoding and the synthesizer policy, not C: three-mask stage 2 takes the
   census from 205-312 exposed values per asis round to 2-3, and v2 acceptance to 0. Against C-blind windowed
   observers the compressed outputs remain exposed (100-230 values) for every C and both policies; that is the
   open problem, and it belongs to stages 2 and 6.
5. Recommendation: keep C random; spend the effort on the stage-6 side (acceptance rules that also measure the
   windowed-factoring family, or a remix after the last resynthesis, which restores 0/0 on the raw ledger), and on
   the stage-2 fire structure (interleaving fires so that no factor group fits a 250-gate window).

## 9. Status, artifacts, limitations

Running on machine1 (`~/cdesign`): the balanced and burst v2 runs (rounds 37-38, plateau rule not met); their
C-blind adaptive resynthesis attacks on the round-21 outputs were stopped after 12 rounds, and the other session's work in
`~/threemask_cmp`. One incident: the disk filled on 2026-10-22 (narrow-portfolio observer scratch, ~50 GB) and
killed every run; all were resumed from their markers, a cleanup loop added, attack workers reduced to two.

Artifacts: lab note `docs/notes/c_design_exposure_20261021.md` (sections 1-9 with every table and update);
refreshed `docs/notes/stage2_stage6_campaign_20261021.md`; server scripts under `~/cdesign/py` and `~/cdesign/py6`
(`pipeline.py`, `census_by_fanout.py`, `locations.py`, `fire_windows.py`, `narrow_fanout.py`, `narrow_gates.py`,
`resil_summary.py`, `early_narrow.py`); all scripts and the two source circuits are copied under
`docs/notes/c_design_report_20261026/scripts/` (`local/expo_model.py`, `local/design_sim.py`, `server/...`); memory notes `cdesign-campaign-20261021`, `server-disk-full-observers-20261022`.

Limitations: one C per design and one master seed, so run-to-run variation is not measured (RC's and our
original-C rates differ by a few points per fanout class); the burst C is a diagnostic with broken avalanche; the
balanced and burst stage-6 comparisons are at equal size rather than at the plateau rule; all counts are validated
candidate equalities on finite samples, not Boolean proofs; the census and narrow families measure different
things and should not be added.
