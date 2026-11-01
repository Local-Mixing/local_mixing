# Single-pass ANF-resynthesis survey of the June-August TDP-like circuits (2026-10-14)

Question (RC): which of the June-August circuits with a TDP-like provenance (sandwich first stage, |C| >= 3000)
are compressible by the iterated ANF-window resynthesis used on the six-stage TDP outputs? First step: a single pass.

Method: per circuit, 12 random windows of W gates (seed 7) are composed to their ANF (2^22-monomial budget,
compose limit 300 s) and resynthesized from the ANF alone into <=2-control gates (anf_synth, time limit
max(60, 30*W/1000) s per window, 4 threads), for W = 1000 and W = 4000. `ratio` = synthesized/original gates over
the windows that succeeded (ok/12); `eff` = the ratio a real pass would give if the failed windows are emitted
verbatim, i.e. (ratio*ok + fails)/12. Windows fail by ANF overflow (dense polynomials) or by the time limit.
mean_k / k<=2% = mean controls per gate and the share of gates with <=2 controls in the input circuit.
g57 (ssg) circuits were converted to mpmct1 with `fcompress --input-format g57 --max-iters 0 --no-pack` (verified).
Provenance: pre-2026-09-11 CIRCUIT_GENERATION_INFO (`git show 242e4ee6^:circuits/CIRCUIT_GENERATION_INFO`);
sgfmix_n128_* has no record (assumed: the n=128 sliced-sandwich gadget + fmix). Excluded: fz_m3000 (feistelized),
n128m4000_sss (plain random + sss), ssgprod (C = 900), sd*/rec*/gad*/c256 (m <= 2200), slice_scramble.

| circuit | gates | mean_k | k<=2 % | W=1000 ratio (ok/12) | eff | W=4000 ratio (ok/12) | eff | ANF monomials p50 @4000 | deg p50 |
|---|---|---|---|---|---|---|---|---|---|
| old_compose_a_gadget | 67,032 | 2.0 | 100 | 0.800 (11/12) | 0.82 | fail (0/12) | 1.00 | - | - |
| new_sliced_sandwich_gadget | 99,016 | 1.3 | 100 | 0.762 (12/12) | 0.76 | fail (0/12) | 1.00 | - | - |
| cdcnot_sm50_mixed_final | 1,036,567 | 2.0 | 100 | 0.529 (12/12) | 0.53 | 0.370 (11/12) | 0.42 | 2567 | 6 |
| cdcnot_sm100_mixed_final | 1,098,254 | 2.0 | 100 | 0.580 (12/12) | 0.58 | 0.387 (9/12) | 0.54 | 2554 | 8 |
| cdcnot_sl1_mixed_final | 847,272 | 2.0 | 100 | 0.590 (12/12) | 0.59 | 0.410 (8/12) | 0.61 | 2211 | 6 |
| cdcnot_sl2_mixed_final | 851,378 | 2.0 | 100 | 0.544 (12/12) | 0.54 | 0.380 (6/12) | 0.69 | 2395 | 6 |
| cdcnot_sm50_fsplit | 1,200,000 | 2.12 | 89.2 | 0.523 (12/12) | 0.52 | 0.364 (9/12) | 0.52 | 2818 | 7 |
| cdcnot_sm50_fsplit_c1_k12 | 2,000,000 | 2.58 | 62.6 | 0.613 (12/12) | 0.61 | 0.306 (9/12) | 0.48 | 3203 | 9 |
| cdcnot_sm50_fsplit_k10 | 3,000,001 | 3.31 | 38.4 | 0.961 (12/12) | 0.96 | 0.549 (10/12) | 0.62 | 9882 | 12 |
| cdcnot_sm50_fsplit_c2_k12 | 3,000,001 | 3.75 | 32.8 | 1.285 (10/12) | 1.24 | 0.479 (9/12) | 0.61 | 4947 | 11 |
| cdcnot_sm50_fsplit_c1_k12_g57t | 3,000,000 | 2.77 | 50.2 | 0.605 (11/12) | 0.64 | 0.341 (12/12) | 0.34 | 5176 | 10 |
| cdcnot_sm100_fsplit_c1_k12_g57t | 3,000,000 | 2.75 | 51.5 | 0.618 (12/12) | 0.62 | 0.312 (12/12) | 0.31 | 5356 | 10 |
| cdcnot_sm50_fmix_s1 | 3,029,251 | 2.85 | 47.3 | 1.438 (12/12) | 1.44 | 0.610 (6/12) | 0.81 | 10341 | 11 |
| cdcnot_sm50_fmix_s2 | 3,029,006 | 2.85 | 47.3 | 1.476 (12/12) | 1.48 | 0.679 (6/12) | 0.84 | 8754 | 15 |
| cdcnot_sm50_fmixdirect_s1 | 3,007,102 | 2.68 | 52.6 | 1.379 (11/12) | 1.35 | 0.608 (7/12) | 0.77 | 11550 | 15 |
| cdcnot_sm100_fmixdirect_s1 | 3,010,131 | 2.59 | 54.7 | 1.353 (12/12) | 1.35 | 0.553 (8/12) | 0.70 | 14644 | 13 |
| cdcnot_fx2tw_fcmp | 893,884 | 7.23 | 12.3 | fail (0/12) | 1.00 | 0.464 (1/12) | 0.96 | 480 | 6 |
| cg1_phaseB | 1,020,340 | 2.13 | 92.7 | 0.572 (12/12) | 0.57 | 0.298 (3/12) | 0.82 | 118092 | 19 |
| cg1_phaseB1Mfa | 1,016,535 | 2.34 | 76.9 | 1.086 (12/12) | 1.09 | fail (0/12) | 1.00 | - | - |
| cg1_phaseB2M | 2,035,987 | 2.2 | 90.8 | 0.643 (12/12) | 0.64 | 0.292 (4/12) | 0.76 | 23532 | 20 |
| cg1_phaseB2Mfa | 1,996,578 | 3.36 | 49.2 | 2.847 (3/12) | 1.46 | fail (0/12) | 1.00 | - | - |
| cg1_phaseB3M | 3,051,468 | 2.23 | 89.8 | 0.819 (11/12) | 0.83 | 0.370 (5/12) | 0.74 | 27572 | 17 |
| cg1_phaseB3Mfa | 2,984,942 | 3.96 | 36.7 | 3.657 (4/12) | 1.89 | fail (0/12) | 1.00 | - | - |
| prodshare_n128_2233_x3_final | 1,127,387 | 4.99 | 18.4 | fail (0/12) | 1.00 | fail (0/12) | 1.00 | - | - |
| prodshare_n128_233_x3_final | 862,630 | 3.4 | 42.1 | 7.137 (4/12) | 3.05 | fail (0/12) | 1.00 | - | - |
| prodshare_n128_33_x3_final | 1,428,155 | 3.78 | 38.4 | 4.718 (5/12) | 2.55 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb01_final | 591,209 | 3.22 | 51.5 | 2.862 (6/12) | 1.93 | 0.988 (1/12) | 1.00 | 40100 | 14 |
| sgfmix_n128_pdb0_fresh_final | 572,586 | 3.41 | 46.7 | 4.246 (6/12) | 2.62 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb0_same_final | 574,553 | 3.41 | 46.8 | 2.959 (4/12) | 1.65 | fail (0/12) | 1.00 | - | - |
| cdcnot_es2_stage1_ckpt | 959,250 | 2.0 | 100 | 0.475 (12/12) | 0.47 | 0.300 (6/12) | 0.65 | 1663 | 6 |
| cdcnot_es3_stage2_ckpt | 2,006,538 | 2.0 | 100 | 0.446 (12/12) | 0.45 | 0.281 (11/12) | 0.34 | 1940 | 7 |
| cdcnot_r4_stage3_ckpt | 616,239 | 2.0 | 100 | 0.632 (12/12) | 0.63 | 0.407 (5/12) | 0.75 | 1456 | 5 |
| cdcnot_r5_stage3_ckpt | 1,152,754 | 2.0 | 100 | 0.576 (12/12) | 0.58 | 0.418 (8/12) | 0.61 | 2408 | 8 |
| cdcnot_sl1_stage3_ckpt | 799,078 | 2.0 | 100 | 0.519 (11/12) | 0.56 | 0.371 (8/12) | 0.58 | 11749 | 9 |
| cdcnot_sl1_midstage4_snapshot | 991,871 | 2.0 | 100 | 0.498 (12/12) | 0.50 | 0.353 (12/12) | 0.35 | 3117 | 7 |
| cmp1M_stall_1M_gen23 | 997,646 | 2.0 | 100 | 0.560 (12/12) | 0.56 | 0.426 (6/12) | 0.71 | 4242 | 8 |
| sm2_fmixgad_final | 2,968,524 | 6.56 | 10.3 | fail (0/12) | 1.00 | fail (0/12) | 1.00 | - | - |
| sm2_fmixgad_final_fc | 2,669,716 | 6.69 | 9.0 | fail (0/12) | 1.00 | fail (0/12) | 1.00 | - | - |
| sm2_fmixgad_snap32M | 2,965,257 | 6.28 | 10.3 | fail (0/12) | 1.00 | fail (0/12) | 1.00 | - | - |
| sm2_fmixgad_snap32M_fc | 2,546,871 | 6.47 | 9.6 | fail (0/12) | 1.00 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb01_snap1M | 500,781 | 3.81 | 38.9 | 3.245 (1/12) | 1.19 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb0_fresh_snap1M | 499,750 | 3.96 | 36.0 | 5.136 (2/12) | 1.69 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb0_same_snap1M | 499,840 | 3.96 | 36.1 | 4.066 (2/12) | 1.51 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb01_snap2M | 511,863 | ~3.9 | ~37 | 5.340 (4/12) | 2.45 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb0_fresh_snap2M | 508,717 | ~3.9 | ~37 | 4.528 (4/12) | 2.18 | fail (0/12) | 1.00 | - | - |
| sgfmix_n128_pdb0_same_snap2M | 509,527 | ~3.9 | ~37 | 4.538 (4/12) | 2.18 | fail (0/12) | 1.00 | - | - |

All 46 circuits surveyed (survey finished 2026-10-14 20:02). The sgfmix snapshots (1M/2M moves) behave like their finals
(x3.2-x5.3 at W=1000 on 1-4 of 12 windows, all fail at W=4000); the four sm2 variants (K=20, mean 6.3-6.7 controls) fail every
window at both widths (an earlier probe at W=25..250 resynthesized them to x20-x31 their size).
## Reading

1. **Compressible in one pass (eff <= ~0.5 at W=4000):** the ssg-era cdcnot circuits (2-control g57 gates):
   finals sm50 0.37 (11/12 ok), sm100 0.39, sl1 0.41, sl2 0.38; checkpoints es3 0.28 (11/12), sl1 mid-stage-4 0.35 (12/12),
   es2 0.30, r4/r5 0.41-0.42, sl1 stage 3 0.37; and the fsplit outputs with K=12/c=1 (fossil-targeted) 0.31-0.34 with
   12/12 ok, sm50_fsplit K4 0.36, c1_k12 0.31. These sit where the six-stage TDP stage outputs sit (0.13-0.22 after
   iterating; 0.2-0.4 after one pass).
2. **Marginal (eff 0.6-0.85):** the fmix outputs (K<=12, mean 2.6-2.9 controls): W=1000 EXPANDS (x1.35-1.48), W=4000
   gives 0.55-0.68 on the half of the windows that finish. The full-circuit run on sm50_fmix_s2 confirms: x1.43 after the
   W=1000 pass, x1.24 and x1.14 after two W=2000 passes, and a W=4000 pass with a 300 s limit is running at ~0.77 per window
   (=> ~0.88 of the original). fsplit k10 / c2_k12 (mean k 3.3-3.8) 0.48-0.55; cg1 slow-anneal (p_db 0.6 -> 0 over the
   full budget, 90% 2-control) 0.29-0.37 but only on 3-5 of 12 windows (eff 0.7-0.8); cmp1M 0.43 on 6/12.
3. **Not handled / expands:** everything whose gates are wide or whose window polynomials are dense: fmix K=20 (fx2tw,
   sm2: x20-x30), the cg1 fast-anneal variants (p_db annealed to 0 within 1M moves then split growth: x1.1-x3.7 at W=1000,
   all windows fail at W=4000), prodshare (32-control mask gates, mean k 3.4-5: fail / x4.7-x7.1), sgfmix (mean k 3.2-3.4:
   x2.9-x4.2 / fail), and the UNMIXED gadgetized sandwiches at W=4000 (the raw random C has an exponential ANF; only
   W=1000 works there, 0.76-0.80).

The dividing line is the gate basis the mixer left behind, not the lineage: circuits that are (nearly) all 2-control
gates compress to 0.3-0.4 in one pass; once the mean control count passes ~3 (fmix K>=10, fast-anneal split growth,
product-share masks) the window ANFs become dense (10^4-10^5 monomials, degree 11-20 at W=4000), the resynthesis
first has to pay a basis-conversion expansion, and the net is at best ~0.9.

## Caveats
- 12 sampled windows per width (seed 7); per-window ratios vary (p10-p90 typically +-30%).
- `ratio` counts only the windows that finished; `eff` charges failed windows at 1.0, which is what the driver does
  (verbatim fallback). The time limit (120 s at W=4000) is part of the verdict: cg1 slow-anneal and the ssg finals
  lose windows to time, not to overflow, and would improve with a longer limit (cf. cdcnot full run).
- Single pass only; the iterated driver typically gains another 10-40% on the compressible class (TDP stages: 0.22 -> 0.13
  on n64 s3 over 4 passes; cdcnot segments: 0.47 -> 0.44).

Files: ~/code/lm_worktrees/compress_runs/survey/{list.tsv,results.tsv,results_mw9.tsv,runs/NAME/wW/windows.tsv}.

## Calibration of the sampled ratios (2026-10-15)

A full W=4000 pass over this week's delivered n64 circuit (six_n64_a stage 6, 5,444,544 gates, 1362 windows, same
120 s budget) gave 0.515 (1247 ok, 115 verbatim, equivalent), whereas the 12-window survey sample of the same circuit
gave 0.266. The sampled windows reproduce exactly in the full pass (deterministic synthesis), so the gap is sampling:
the per-window ratio is heavy-tailed (p10 0.21, p50 0.28, p90 0.73; the first 10% of the circuit runs at 0.91) and the
gate-weighted whole-circuit ratio (0.47 over ok windows, 0.515 with the 8.4% verbatim windows) is dominated by the dense
tail that 12 windows rarely hit. Read the survey's `ratio` as a per-window median: for the compressible class the
whole-circuit single-pass ratio can be 1.5-2x larger. Rankings between families are unaffected (the dense families fail
outright).

This week's six-stage circuits at W=4000 (24-window smoke of the campaign, same budget): stage 2 0.21/0.21 (n64/n128),
stage 3 0.16/0.17, stage 4 0.13/0.14 (all 24/24 ok, 0.4-1.3 monomials per gate, degree 5-7), stage 5 0.23/0.32 (20/24,
k 2.9, degree 9), delivered stage 6 sample 0.27/0.39 (11/12) -> full passes n64 0.515 (115/1362 verbatim) and n128 0.472 (194/4315
verbatim, deciles flat 0.41-0.51, per-window p10/p50/p90 0.27/0.34/0.68, 8.5 h at 6 threads). The campaign's
stage-6 plateau 0.668 was a ladder artifact (W=2000 rejected on the original by one extra fallback, so W=4000 was never
tried on the original and no longer passes after a W=1000 pass).
