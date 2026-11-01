# Iterated ANF-window compression of the TDP stage outputs, and what the compressed circuits expose

*2026-10-13 → 2026-10-15 (complete). RC's request (2026-10-13 22:30): "try to compress the outputs of each one of stages
2,3,4,5,6, iterating between creating ANF representation for as large blocks as you can handle and then
resynthesis, until progress stops. I'd like to compare how compressible these intermediate states are."
Follow-ups: "for each one of the plateaued compressed versions, run a best affine predictor heatmap
against the original C (in each cut of C, it suffices to predict the active wire in the cut)"; "say that a
C-value is k-predicted if k values of G were used in the linear equation that predicts the value. what
fraction of the predicted values were k-predicted for each k? how many times were they predicted?"
Runs: `port_runs/six_n64_a` (n=64, |C|=1000) and `port_runs/six_n128_a` (n=128, |C|=3000), the six-stage
TDP runs delivered 2026-10-13. Everything lives in `~/code/lm_worktrees/compress_runs/` (README there);
tables in `docs/notes/compress_iter_20261014/tables.md`, plates `plates_<run>_<stage>.png`.*

## 1. Method

**Compression** (`compress_iter.sh`, one circuit): iterate
1. choose the window width W (gates) = the largest value on the ladder 250…16000 that the synthesizer
   "handles": a smoke test of 24 sampled windows of width W (`anf_synth --sample 24`) is accepted iff at
   most 2 windows fall back (ANF composition over 2^22 monomials or 300 s, or non-convergence) and the
   median per-window time is at most TL/2, with TL = max(60, 30·W/1000) s; start at the previous pass's W
   and climb or descend;
2. cut the circuit into uniform windows of W gates (offset 0 on odd passes, W/2 on even passes, so windows
   straddle the previous cuts), compose each window's exact vector ANF and resynthesize it from the ANF
   alone with `anf_synth` (the untracked crate of the 2026-10-08 study: ≤2-control output, 16 restarts,
   beam 6, per-window verification on 256 random states; a window that fails is emitted verbatim);
3. splice, check equivalence with the ORIGINAL stage output on 8192 random full states
   (`port_target/release/examples/equiv`), log; stop when the gate count improves by < 2% or grows (keep
   the smallest), or after 8 passes.

Stage outputs: s2 = `tdp.mpmct1` (preprocessed sandwich), s3 = `db_mixing.mpmct1`, s4 = `split.mpmct1`,
s5 = `crossing.mpmct1`, s6 = `unpacking/unpacked.mpmct1` (the delivered circuit). All five compute the same
permutation (checked). Gate counts are the metric; total controls are tabulated too.

**Plates** (`cview_plates.py`): `hmap_affine --active-wire --c-step 1` (chunked over `--g-from/--g-to`,
merged): row i = C prefix i scoring only the value born there (the target of C gate i−1 = the active
wire), column j = G prefix (≈1000 cuts), cell = holdout error of the best affine predictor of that value
from the 256/512 wire values of G at the cut, exact GF(2) fit (0 = recoverable, 0.5 = hidden). C-view
convention: the same random x on wires 0..n−1 of both circuits. One plate for the plateaued compressed
circuit and one for the stage output it came from. Rendered at a fixed 0..0.5 scale (the house ridge
plotter's percentile scale degenerates on these nearly-flat plates).

**Exposure scans** (`exposure_scan.py`): `hmap_affine --exposure` at every cut (every 2nd at n=128), with
`EXPOSURE_ALLCUTS`: for every value born in C and every cut, whether an EXACT affine predictor exists
and the minimal number k of distinct non-constant G wire values it reads (complement pairs collapse;
exact coset minimisation when the nullity is ≤ 14). "k-predicted" = predicted by an equation reading k G
values. Reported per value (minimal k over cuts, number of cuts, number of contiguous runs) and per
(value, cut) pair.

## 2. Results

See `compress_iter_20261014/tables.md` for the generated tables (compressibility, plates, exposure).

### 2.1 Compressibility

| stage output | n=64 gates → best | ratio | passes (W) | n=128 gates → best | ratio | passes (W) |
|---|---|---|---|---|---|---|
| s2 preprocessed sandwich | 777,627 → 167,700 | 0.216 | 1 (4000; a W=500 2nd pass grew) | 2,242,721 → 443,994 | 0.198 | 1 (8000; 2nd pass grew) |
| s3 mixing | 1,590,313 → 211,269 | 0.133 | 4 (4000, 1000, 1000, 1000) | 4,613,083 → 640,709 | 0.139 | 4 (8000, 1000 ×3) |
| s4 split | 3,001,463 → 221,071 | 0.074 | 6 (8000, 1000 ×5) | 8,693,428 → 670,918 | 0.077 | 6 (16000, 1000 ×5) |
| s5 crossing | 5,477,201 → 1,005,737 | 0.184 | 5 (2000, 2000, 1000, 2000, 1000) | 15,853,526 → 2,799,394 | 0.177 | 8 = cap (2000, 2000, 1000 ×3, 4000, 2000, 1000) |
| s6 delivered | 5,444,544 → 3,636,640 | 0.668 | 2 (1000, 2000; a 3rd W=2000 pass grew) | 17,259,242 → 9,757,826 | 0.565 | 3 (2000, 2000, 4000; a 4th W=4000 pass grew 14%) |

Reading:
- **Stages 2–4 collapse to the same absolute size** (170–220k gates at n=64, 440–670k at n=128):
  everything mixing (×2) and splitting (×2 again) add is removed by a few passes. Stage 2 is a one-pass
  fixed point (its compressed form cannot be improved at any handled width: W=2000 overflows, W=1000 grows
  it 4%, W=500 leaves it at 0.995). Stages 3–4 keep yielding 2–15% per W=1000 pass for 4–6 passes.
- **Stage 5 (crossing) is the first representation that holds**: 0.18 of its gates survive (1.0M / 2.8M,
  4–5× the stage 2–4 plateau). The handled width drops to 2000 gates (multi-control gates, dense ANF tails:
  3–6% of windows fall back verbatim) and the plateau is **width-limited, not a fixed point**: whenever the
  width probe lets a pass run at W=2000 instead of 1000 the circuit loses 20–40% (n=64 passes 2 and 4:
  0.60, 0.81), and the single W=4000 pass that was accepted at n=128 (pass 6, median 56 s) removed 37% at a
  stroke (0.63), after three W=1000 passes had yielded 3–6% each. n=128 hit the 8-pass cap still gaining 5%.
- **Stage 6 (delivered) is by far the least compressible**: 0.67 at n=64 and 0.57 at n=128, although it has the same length
  and function as the crossing output it was unpacked from. Equal-length spans of the delivered circuit
  carry 40–55% more ANF monomials than spans of the crossing circuit (2,322 vs 1,496 per 2,000-gate window;
  1.05 vs 0.75 per gate at W=1000), and the synthesizer's output tracks the monomial count (per-window
  ratio p50 0.67 vs 0.375). The crossing walk's redundancy (brackets that cancel inside a span) is what made
  stage 5 compressible; the delivery already removed it, and the cut states inside a resynthesized window
  are denser than the mixer's states, so uniform re-windowing finds little.
- By total controls the n=64 ratios are 0.17 / 0.10 / 0.07 / 0.11 / 0.69 (s2–s6): the stage-5 output's
  2.9 controls per gate make it look less compressible by gates than it is by Toffoli cost.
- n=64 and n=128 agree to within a few percent at every stage.

### 2.2 Plates: best affine predictor of C's active wire from the compressed circuits

| compressed stage | n=64: C values exposed at interior cuts (1000 cuts; census, interior = inner 80% of rows and cols) | n=128 (every 3rd C value) | Spearman(row, best cut) | original stage output |
|---|---|---|---|---|
| s2 | 485 | 470 | 0.994 / 0.997 | 0 / 0 |
| s3 | 452 | 407 | 0.993 / 0.996 | 1 / 0 |
| s4 | 449 | 399 | 0.992 / 0.996 | 1 / 0 |
| s5 | 341 | 268 | 0.985 / 0.992 | 1 / 0 |
| s6 | 85 | 99 | 0.974 / 0.984 | 1 / 0 |

Every compressed circuit shows a clean diagonal: C's values appear in C's order over the first ~45–50% of
the compressed circuit, with all n outputs readable from there to the end (figure `plates_n64_s2.png`
etc.). Every original stage output shows nothing in the interior, only the ports (inputs at the entry,
outputs in the last 0.4%). The stage-6 plate counts (85 / 99) are artefacts of their 1-in-3,637 / 1-in-9,758 cut sampling
against values that live ~700 gates; the all-cuts scan below is the real number.

### 2.3 k-predicted census of the compressed circuits (all cuts)

| compressed | n | interior C values predicted | minimal k = 1 / 2 / 3 / 4 / ≥5 | (value, cut) pairs with k = 1 | cuts per value p10 / p50 / p90 (gates) | runs p50 | outputs: cuts p50 |
|---|---|---|---|---|---|---|---|
| s2 | 64 | 619 / 936 (66%) | 71 / 5 / 14 / 8 / 1 % | 30% | 143 / 931 / 2,662 | 2 | 9,268 |
| s3 | 64 | 575 (61%) | 78 / 7 / 7 / 7 / 1 % | 54% | 168 / 1,126 / 3,502 | 2 | 9,120 |
| s4 | 64 | 558 (60%) | 83 / 7 / 6 / 2 / 2 % | 65% | 183 / 1,346 / 3,928 | 2 | 12,676 |
| s5 | 64 | 466 (50%) | 88 / 5 / 5 / 1 / 2 % | 73% | 315 / 1,312 / 3,528 | 2 | 35,524 |
| s6 | 64 | 427 (46%) | 74 / 10 / 8 / 4 / 4 % | 60% | 97 / 728 / 2,076 | 2 | 42,950 |
| s2 | 128 | 1,938 / 2,872 (67%) | 71 / 6 / 12 / 10 / 1 % | 30% | 216 / 1,474 / 4,916 | 2 | 18,224 |
| s3 | 128 | 1,770 (62%) | 70 / 10 / 9 / 9 / 2 % | 42% | 212 / 1,402 / 4,546 | 2 | 17,144 |
| s4 | 128 | 1,737 (60%) | 76 / 8 / 9 / 6 / 2 % | 56% | 194 / 1,348 / 4,492 | 2 | 21,334 |
| s5 | 128 | 1,478 (51%) | 88 / 6 / 3 / 2 / 1 % | 71% | 512 / 2,026 / 6,210 | 2 | 98,022 |
| s6 | 128 | 1,393 (49%) | 87 / 6 / 3 / 2 / 1 % | 76% | 362 / 1,882 / 6,348 | 2 | 120,562 |

Reading:
- About two thirds of C's interior values are affinely readable from the compressed stage 2–4 circuits,
  half from compressed stage 5/6 (46–51% at both sizes); all n outputs always (verbatim, over 5–9% of the circuit at s2–s4, more
  at s5–s6).
- Of the exposed values, 70–88% sit verbatim on a wire at some cut (k = 1) and ≥97% need at most 4 G
  values at their best cut; at a random cut where a value is readable, it is verbatim 30% (s2) to 73% (s5)
  of the time, otherwise XORed with 1–3 other wire values.
- A value is typically readable over ~900–1,500 consecutive gate positions (0.2–0.6% of the circuit) in two
  contiguous runs; a verbatim value sits bare at ~400–1,100 positions.
- Trend with stage: fewer values exposed (66 → 61 → 60 → 50 → 46% at n=64) but a rising verbatim share
  (71 → 78 → 83 → 88%, then 74% at s6; n=128: 71 → 70 → 76 → 88 → 87%) and longer bare spans; n=128
  reproduces the n=64 numbers to within a few points at every stage.

## 3. Conclusions

1. Compressibility ranks s4 (0.07) < s3 (0.13–0.14) < s5 (0.18) < s2 (0.20–0.22) ≪ s6 (0.57–0.67) by ratio,
   but in absolute gates s2 ≈ s3 ≈ s4 (0.17–0.22M at n=64, 0.44–0.67M at n=128) ≪ s5 (1.0M / 2.8M) ≪ s6
   (3.6M / 9.8M): preprocessing, mixing and
   splitting are fully undone by window resynthesis; the crossing walk leaves 5× more, and the delivery
   (per-window resynthesis + canonical order) leaves a representation that uniform re-windowing barely
   touches.
2. The stage-5 and stage-6 plateaus are set by the handled window width (2000 gates at a 60 s budget),
   not by the representation: every accepted doubling of W removed 20–40% more. The crossing output's
   true window-compression limit is therefore lower than 0.18 and would need either a cheaper composition
   (the 2^22-monomial / 300 s budget) or longer synthesis budgets to reach.
3. Every plateaued compressed circuit re-exposes C: half to two thirds of C's interior values become
   affinely readable, mostly verbatim, in C's order, with all outputs readable from the C/rest junction
   onwards; no original stage output exposes anything. This is the 2026-10-08 resynthesis leak measured
   through the whole pipeline: the compression that removes the redundancy also materialises C's values.

## 4. Ops notes

- Campaign scripts: `run_campaign.sh NAME RUNDIR` (stages sequentially), `NAME/THREADS` re-read before
  every synthesizer call (rebalanced 4→8→12 as the delivery run and the n=64 campaign finished),
  `plates_watcher.sh` / `exposure_watcher.sh` (plates and scans as soon as a stage plateaus),
  `status.sh`, `make_report.py`. Wall: n=64 ≈ 18 h on 8 threads; n=128 ≈ 43 h on 4–12 threads (s5 9.6 h
  for its first pass alone), 2026-10-13 22:42 → 2026-10-15 20:37; a job from another session shared the box
  from 2026-10-14 19:00.
- `anf_synth` caps at 12 threads per process; the exposure scan at 512 wires needs ≤ 8,192 cuts per
  process (393 KB/cut); the n=64 all-cuts scans take 1.5–30 min, the n=128 ones 5–20 min at `--g-step 2`.
- The plot_hmap_ridge percentile colour scale is useless on these plates (uniform panels); use the
  fixed-scale render (`plates_fixed.png`) or the 1:1 pixel render (`*_aw.px.png`).
