# Stages 4+5 revived: ANF growth of the crossing output, and the resynthesis adversary on it

*2026-10-11, RC's request of 2026-10-10 night: "bring back stages 4 and 5, redo the partitioning to
windows for the output of stage 5; what is the growth curve of the ANF complexity per # of gates in
the outputs of stage 5?" and then "pick the largest window size in the output of stage 5 that you can
comfortably resynthesize, resynthesize the windowed output of stage 5, and check how many C proxies
reappear (64 and 128)". Runs in `~/code/lm_worktrees/s45_runs/` (arms `base64`, `e2e128`); tools
there (`run_s45.sh`, `adversary_s5.sh`, `mult_verbatim.sh`, `growth_summary.py`) and the compose-only
tool `anf_grow` + the k-control input path of `anf_synth` in `runs/ncdb_ab_20261001/synth/` (untracked).*

## 1. What was run

- **Stages 4 (split) and 5 (crossing walk)** = the archive-era `tdp_gen.sh` stages, verbatim flags
  (split: `--split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100
  --split-canaries 256 --k-max 12`; crossing: resume of `split.state`, target = 2 x split gates,
  temp = target/25, budget 6 x target moves, `--split-base 3 --split-damp 1`), built from the main
  checkout at `ea2e516f` (= the archive branch tip). Inputs = the existing stage-3 outputs of the
  L8 arms: `dressed_runs/base/s3.mpmct1` (n=64, |C|=1000) and `port_runs/e2e128_r4/db_mixing.mpmct1`
  (n=128, |C|=3000). Fresh OS seeds, kept in private `SEEDS` files.

| | n=64 | n=128 |
|---|---|---|
| stage 3 gates | 1,603,492 | 4,593,662 |
| stage 4 (split) gates; time | 3,026,369; 70 s | 8,657,194; 8.5 min |
| stage 5 (crossing) gates; time | 5,494,528; 46 min | 15,712,838; 6.3 h (shared CPU) |
| stage-5 control-width histogram (k: gates) | 0:19k 1:759k 2:1.30M 3:1.41M 4:1.19M 5:529k 6:196k 7:62k 8:18k 9-12: 6.3k | same shape (3: 4.03M, 4: 3.40M, max 12) |

The crossing output's header says `mpmct1 1 N` (archive mixer bug on resume); a copy with the
right wire count (`crossing256.mpmct1`) feeds every tool.

- **Re-partition with the stage-4 packer** (`tdp_packing run`, G=512, repair): n=64 stage-5 input →
  10,464 windows (med 512 gates, p90 579), window monomials med 727 / p90 12,016 / max 206k, exposed
  at boundaries 13,713 (10.8 % of carried) vs 11,411 (6.9 %) for the stage-3 input, final C-view scan
  64 visible / 0 interior (floor), 2.2 h (vs 23 s on stage 3). n=128 stage-5 input → 29,969 windows (med 512, p90 577),
  window monomials med 756 / p90 9,753 / max 196k, repair 15.71M → 15.41M gates, final scan 128 visible / 0 interior
  (floor), 7.7 h at low priority.

- **ANF growth curve** (`anf_grow`): uniform windows of W gates, exact vector ANF composed under a
  2^22-monomial budget (and a 120 s limit for W ≥ 2000 on stage 5); 200 windows sampled per width
  (100 on n=128 stage 5; fewer at W ≥ 2000 on n=64 stage 5). Quantiles exclude over-budget /
  timed-out windows, so censored rows (fraction given) are lower bounds. Plots:
  `s45_runs/base64/growth.png`, `s45_runs/e2e128/growth.png`.

## 2. Growth of monomials per window (median; p90 in brackets)

n=64, L8 base arm:

| W | stage 2 (784k g) | stage 3 (1.60M) | stage 4 (3.03M) | stage 5 (5.49M) | stage-5 censored |
|---|---|---|---|---|---|
| 100 | 62 (115) | 151 (226) | 127 (188) | 247 (599) | 0/200 |
| 200 | 124 (197) | 235 (361) | 211 (302) | 392 (1,613) | 0/200 |
| 500 | 263 (357) | 437 (632) | 363 (624) | 648 (2,616) | 1/200 |
| 1,000 | 505 (704) | 714 (1,090) | 580 (937) | 1,001 (6,515) | 2/200 |
| 2,000 | 1,104 (1,704) | 1,238 (1,981) | 921 (1,580) | 2,355 (13,719) | 5/60 |
| 4,000 | 3,970 (10,489) | 2,564 (5,018) | 1,742 (2,841) | ≥3,674 (≥10,758) | 23/48 |
| 8,000 | 56,033 (450k) 6/97 | 11,118 (31k) 19/200 | 3,561 (6,655) | ≥7,561 | 45/48 |
| 16,000 | all over 48/48 | 84,889 (278k) 83/100 | 13,388 (51k) 24/189 | all over | 48/48 |

n=128, L8 e2e run:

| W | stage 3 (4.59M) | stage 4 (8.66M) | stage 5 (15.7M) | stage-5 censored |
|---|---|---|---|---|
| 100 | 151 (208) | 134 (209) | 243 (756) | 0/100 |
| 200 | 241 (344) | 215 (362) | 389 (1,233) | 0/100 |
| 500 | 451 (629) | 403 (549) | 727 (2,682) | 0/100 |
| 1,000 | 766 (1,142) | 619 (889) | 1,179 (3,358) | 1/100 |
| 2,000 | 1,329 (1,803) | 1,032 (1,426) | 1,843 (6,481) | 4/100 |
| 4,000 | 2,385 (3,270) | 1,749 (2,417) | ≥2,884 (≥6,759) | 27/100 |
| 8,000 | 5,281 (8,522) | 3,180 (4,685) | ≥4,470 | 71/100 |
| 16,000 | 24,510 (73k) 48/200 | 6,784 (11,879) | all over | 97/100 |

Reading: below ~2,000 gates every stage grows sublinearly (log-log slope 0.6–0.8; monomials per
gate FALL with W: 1.5 → 0.6 at stage 3). Stage 5 is 1.4–1.9x stage 3 at the median and 3–7x at the
p90, with degree median 7–11 (stage 3: 4–8) from the multi-control gates. Its tail is what grows:
composition under 2^22 monomials / 120 s fails for >20 % of windows from W=4,000 (n=64 and n=128),
where stage 3 still composes 90–99 % of windows at W=8,000. Split (stage 4) is the sparsest
representation per gate at every width (it doubles the gate count without adding monomials).
Per function covered the picture is milder: stage 5 has 3.4x the gates, so a 2,000-gate stage-5
window covers what a ~590-gate stage-3 window covers, and holds ~2–3x its monomials (2,355 vs
714–1,238 at W_3 = 1,000–2,000; p90 13,719 vs 1,090–1,981).

## 3. The resynthesis adversary on the stage-5 output

Largest comfortable width: smoke runs of 24 sampled windows at n=64 (4 threads, 60 s search limit):
W=500 median 2.2 s / p90 33 s / 1 fallback; W=1,000 6.8 / 53 s / 0; W=2,000 12.5 / 50 s / 0;
W=4,000 61 / 181 s / 4 fallbacks. → **W = 2,000** for both sizes. The synthesizer only sees the
window ANF; its input loader was widened to 12 controls (composition, equivalence check, verbatim
fallback), and a 300 s bound was put on per-window composition after one n=64 window ran >70 min.

**n=64** (`base64/adv_s5_2000`): 2,748 windows, 2,559 resynthesized, 189 (6.9 %) emitted verbatim
(composition overflow/timeout or non-convergence), 164 deadline hits; monomials per ok window
p50/p90/max 1,688 / 8,579 / 715k; 5.49M → 2.72M gates; wall 2.2 h on 12 threads.

| n=64, L8 base arm, resynthesized circuit, all cuts | interior C values verbatim (of 936) | repeated: ≥2 instances / ≥2 windows | C outputs early (<90 %, of 64) |
|---|---|---|---|
| stage 3 packed, delivered lattice (~590 g) [2026-10-10] | 297 | 118 / 92 | 56 |
| stage 3 packed, uniform W=2,000 (`dressed_runs/base/adv2000`, 744 windows, 4 fallback) | 434 | 225 / 198 | 61 |
| **stage 5 crossing, uniform W=2,000** | **244** | **88 / 56** | **45** |
| stage 5 crossing, raw (no resynthesis), all cuts | 0 | – | 0 |

Interior re-creations in the stage-5 synth: 157 by CNOTs, 87 by 2-control gates (none with both of
C's operands on its controls, 0 with one) — the same mask-removal factoring as on stage 3. Early
outputs: 129 early instances, first appearance p50 at 0.71 of the circuit (raw stage 5: 0.996).

**n=128** (`e2e128/adv_s5_2000`): 7,857 windows, 7,411 resynthesized, 446 (5.7 %) verbatim fallback,
635 deadline hits; monomials per ok window p50/p90/max 1,703 / 5,963 / 862k; 15.71M → 9.38M gates;
wall 8.7 h on 12 threads (shared with the packer).

| n=128, L8 e2e run, resynthesized circuit, all cuts | interior C values verbatim (of 2,872) | repeated: ≥2 instances / ≥2 windows | C outputs early (of 128) |
|---|---|---|---|
| stage 3 packed, delivered lattice (~960 g), interleaving CONTROL arm [2026-10-10] | 417 | – | 60 |
| stage 3 packed, uniform W=2,000, interleaving CONTROL arm [2026-10-10] | 1,067 | – | 110 |
| stage 3 packed, uniform W=2,000, same run (`dressed_runs/e2e128/adv2000`, 2,129 windows, 1 fallback) | 1,270 | 635 / 583 | 124 |
| **stage 5 crossing, uniform W=2,000** | **711** | **255 / 163** | **94** |
| stage 5 crossing, raw (no resynthesis), all cuts | 0 | – | 0 |

Interior re-creations in the n=128 stage-5 synth: 471 by CNOTs, 240 by 2-control gates (0 with C's operands
on the controls); early outputs 213 instances, first appearance p50 at 0.63 (raw: 0.997).


## 4. Conclusions

1. **Stages 4+5 do not close the resynthesis leak.** The crossing output has no C value on any wire at
   any cut (0 interior, 0 early outputs at both sizes), and per-window resynthesis at W=2,000 brings
   back 26 % / 25 % of C's interior values (244 of 936 at n=64, 711 of 2,872 at n=128) and 70 % / 73 %
   of the outputs early (45/64, 94/128) — the same order as the stage-3 deliverable, by the same
   mechanism (mask-removal factoring: CNOT / 2-control re-creations, none with C's operands as controls).
2. **At equal gate width** the stage-5 proxies are ~55 % of the same run's stage-3 proxies (244 vs 434
   interior at n=64; 711 vs 1,270 at n=128), repeated proxies ~40 % (88 vs 225; 255 vs 635), early
   outputs ~75 % (45 vs 61; 94 vs 124). This is the window-fraction effect: crossing stretches the
   circuit 3.4x, so a 2,000-gate stage-5 window covers what a ~590-gate stage-3 window covers, and
   against that lattice the counts are ~80 % (n=64: 244 vs 297 interior, 88 vs 118 repeated).
3. **ANF growth**: every stage grows sublinearly below ~2,000 gates (monomials per gate fall with W);
   stage 5 is 1.4–1.9x denser than stage 3 at the median and 3–7x at the p90 (degree median 7–11 vs
   4–8), and becomes un-composable (2^22 monomials / 120 s) from W≈4,000 where stage 3 holds to
   8,000–16,000. So crossing halves the adversary's composable width in gates, but the leak needs
   only the width of one fire, and in function coverage the reach is about unchanged.
4. The stage-4 packer still reaches the C-view floor on the stage-5 circuit (64/0 and 128/0 visible)
   with 4x the windows, 10.8 % boundary exposure (vs 6.9 %), and 2.2 h / 7.7 h instead of 23 s / 2 min.
5. Verbatim fallbacks (6.9 % / 5.7 % of windows) contribute nothing, so per resynthesized window the
   stage-5 numbers are ~6–7 % higher than printed.

## 5. Ops notes

- `anf_synth` without a composition bound hung >70 min on one n=64 window (13 GB RSS) and holds all
  results in memory until the end: the run was lost. `--compose-limit` (default 300 s) now bounds it.
- The n=128 crossing walk slowed 5x under CPU contention (packer repair = 9 threads, growth curves);
  renice background jobs. Packer on 15.7M gates: 7.7 h at nice 19.
- The archive mixer writes `mpmct1 1 N` on a resumed state's output; fix the header before use.
