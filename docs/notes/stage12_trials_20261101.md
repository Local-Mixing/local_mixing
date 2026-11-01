# Stages 1-2 update and trial sweep (n = 64, 128), 2026-11-01

Code: branch `ran/stage12-layered-v` (from `ran/review-fixes-20261028`): c31a55ef (library, gen, tests), eea26e56 and
7191de03 (managed driver, recipe v14, docs), 4d780951 (adaptive stage 6 `--forward-only`). Recommendations implemented from `docs/notes/stage2_interleaving_20261101.md`: stage 1
slicing optional, layer size w, the V gadget; stage 2 reverse-contract switch, wire turnover + interleave 4096 as default,
pre-growth of the whole band. Trial arms: `~/code/lm_worktrees/s12_runs` (make_arms.py, tables.py, blocks.py; seeds per
(n, w) in `private/`, shared by every arm with that (n, w) so the t and slicing comparisons are paired); battery on the
server under `~/variants/s2new` (scheduler `py/s2queue.py`, results in `~/variants/run/battery/<arm>/`).

## High level

1. Stage 1 now defaults to C | copy | (D beside V, V*, recovery) with no slice blocks; `SandwichShape::LEGACY` reproduces
   the old sandwich gate for gate (both parity hashes unchanged). The V gadget is built on groups of three of C's outputs,
   not pairs: the pair copy c' = c + a(b+1) + 1, d' = d + b(a+1) + 1 maps (a, b) = (0, 0) and (1, 1) to the same (1, 1),
   and every invertible map on two bits is affine, so no pairwise nonlinear copy can be undone from the register.
2. Structure: the sandwich's dependency width grows with w and t as intended (n = 64: 10.2 at w = 1, t = 1 to 31.2 at
   w = 21, t = 384; n = 128: 17.5 to 55.8), but the fires a 250-gate window of stage 2 holds barely move (9.6 to 10.3 at
   n = 64, 13.1 to 14.1 at n = 128): stage 2 is lane-bound. Lifting the lanes alone (band 512, 32 lanes) gives 14.0 /
   15.4; lanes plus a 16k interleave window give 16.0 (n = 64) / 19.0 (n = 128), of which layering contributes about +1.1
   / +1.5 over w = 1. The width is there; stage 2 does not yet use it.
3. Cost: slicing adds 20 % to stage 2 at n = 64 and 17 % at n = 128; the V gadget at t = n log2 n adds 35 % / 28 % over
   t = 1; the default (w = 1, t = n log2 n, no slicing) is 13 % / 10 % above the old rust2-like stage 2.
4. Exposure (narrow portfolio on the stage-2 circuits, C-aware labelling of C-blind observations): in progress; the
   first 125/250-gate ledgers do not separate the arms yet (section (d)).

## Details

**(a) Stage 1.** `SandwichShape { slicing, layer, v_gates }` (gen: `SANDWICH_SLICING`, `SANDWICH_LAYER`,
`SANDWICH_V_GATES`; defaults 0, 1, round(n log2 n)).

- *Layered sampling*: the random C (fastrand, source seed), D and V draw `layer` wire-disjoint gates at a time (3w distinct
  wires per layer; a layer repeating a gate of the previous one is redrawn). w = 1 keeps the old draws exactly.
- *V gadget*: C's n outputs are split by a random permutation into groups of three (the last group takes 3 to 5 wires).
  For a group (a; x_1 .. x_m) the copy XORs into the register H(a) ^= mux(a; x_1, !x_2), H(x_1) ^= mux(a; x_1, x_2),
  H(x_i) ^= mux(a; x_i, x_pi(i)) (literal polarities random), mux(a; u, v) = (!a & u) ^ (a & v): two two-control
  conjunctions per register wire, no CNOT, every register value a balanced degree-2 function of three of the group's
  values (a unit test checks that no register wire is ever an affine copy of one C output). V is t random gates on the
  register, interleaved with D's first half; V* (V reversed), then the recovery, with D's second half. The recovery acts
  on the register alone: H(a) ^= H(x_1) leaves a there, one or two CNOTs from H(a) per wire make both mux branches
  positive, and controlled swaps on a move each x_i home. Per group of three: 6 copy gates and 4-8 recovery gates. The
  copy gates float backward into C's tail (they cannot pass V). Reverse: A^-1(p, 0) leaves the copy's bijection of
  D^-1(p) on the register; with slice blocks on, V uses positive conjunctions and the copy's a = 0 branches are positive,
  so the register stays zero through V, V* and the recovery in reverse and S2 stays dead (unit-tested).
- *Slicing optional*: off by default; on, S1/S2 are drawn exactly as before.

**(b) Stage 2.** `STAGE2_REVERSE_CONTRACT` (default 1): 0 makes the opening guard cover all 2n data wires with the same
gate count (the Feistel guard's rounds alternate halves: accumulate from the high half and inject into the low half, then
the reverse, so each round's uncompute stays exact) and drops the closing guard; gen skips the reverse check. The band
reseed is kept (the interleaving note also suggested dropping it without the reverse contract; not done, say if wanted).
`ThreeMaskParams::v3` is the new default: v2 + wire turnover + interleave window 4096 + `pregrow` (the serving half of the
band rewritten `warmup_sweeps` times, 256 gates at n = 64 and 512 at n = 128, before the first masks reference it; each
gate composes a growing wire with another band wire).

**(c) Structure of the arms** (gadget = full stage-2 circuit; widths = gates / dependency depth; blk1 = up to C's last
gate including the floated copy gates, blk2 = the rest):

### n = 64

| arm | w | t | slice | sandwich | blk1 width | blk2 width | sandwich width | stage-2 gates | fires/250 | fire span | stage-2 dep. width | gadget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n64_leg | 1 | 0 | on | 2,832 | 10.6 | 11.0 | 10.9 | 448,857 | 9.7 | 2,487 | 5.7 | 454,272 |
| n64_w1_t1 | 1 | 1 | off | 2,264 | 9.6 | 10.4 | 10.2 | 375,478 | 9.6 | 2,616 | 5.6 | 380,893 |
| n64_w1_t192 | 1 | 192 | off | 2,646 | 9.6 | 14.0 | 11.9 | 441,209 | 9.8 | 2,771 | 5.7 | 446,624 |
| n64_w1_t384 | 1 | 384 | off | 3,030 | 9.6 | 17.5 | 13.6 | 506,359 | 9.9 | 2,819 | 5.8 | 511,774 |
| n64_w1_t384_b256 | 1 | 384 | off | 3,030 | 9.6 | 17.5 | 13.6 | 512,213 | 12.5 | 3,296 | 8.5 | 517,628 |
| n64_w1_t384_nopre | 1 | 384 | off | 3,030 | 9.6 | 17.5 | 13.6 | 506,438 | 9.9 | 2,820 | 5.7 | 511,853 |
| n64_w1_t384_sl | 1 | 384 | on | 3,773 | 11.2 | 17.2 | 14.3 | 608,051 | 9.7 | 2,489 | 5.7 | 613,466 |
| n64_w6_t1 | 6 | 1 | off | 2,257 | 12.0 | 13.2 | 12.5 | 375,870 | 9.7 | 2,729 | 5.7 | 381,257 |
| n64_w6_t192 | 6 | 192 | off | 2,639 | 12.0 | 17.6 | 14.7 | 439,241 | 10.0 | 2,857 | 5.8 | 444,628 |
| n64_w6_t384 | 6 | 384 | off | 3,023 | 12.0 | 22.1 | 16.8 | 506,158 | 10.1 | 2,934 | 6.0 | 511,545 |
| n64_w11_t1 | 11 | 1 | off | 2,262 | 15.8 | 16.0 | 16.0 | 375,276 | 9.9 | 2,776 | 5.8 | 380,555 |
| n64_w11_t192 | 11 | 192 | off | 2,644 | 15.8 | 21.4 | 18.8 | 441,372 | 10.2 | 2,930 | 6.0 | 446,651 |
| n64_w11_t384 | 11 | 384 | off | 3,028 | 15.8 | 26.8 | 21.5 | 506,732 | 10.2 | 3,015 | 6.0 | 512,011 |
| n64_w16_t1 | 16 | 1 | off | 2,264 | 18.8 | 19.3 | 19.2 | 375,322 | 9.9 | 2,784 | 5.9 | 380,667 |
| n64_w16_t192 | 16 | 192 | off | 2,646 | 18.8 | 25.8 | 22.4 | 439,351 | 10.2 | 2,941 | 6.0 | 444,696 |
| n64_w16_t384 | 16 | 384 | off | 3,030 | 18.8 | 32.3 | 25.7 | 506,761 | 10.3 | 3,037 | 6.1 | 512,106 |
| n64_w21_t1 | 21 | 1 | off | 2,263 | 23.0 | 23.7 | 23.3 | 375,374 | 10.0 | 2,894 | 5.9 | 380,819 |
| n64_w21_t192 | 21 | 192 | off | 2,645 | 23.0 | 31.7 | 27.3 | 440,959 | 10.2 | 2,984 | 6.0 | 446,404 |
| n64_w21_t384 | 21 | 384 | off | 3,029 | 23.0 | 39.7 | 31.2 | 506,181 | 10.3 | 3,076 | 6.0 | 511,626 |
| n64_w21_t384_b256 | 21 | 384 | off | 3,029 | 23.0 | 39.7 | 31.2 | 507,645 | 13.2 | 3,466 | 8.5 | 513,090 |
| n64_w21_t384_sl | 21 | 384 | on | 3,776 | 19.6 | 27.0 | 23.5 | 606,831 | 10.0 | 2,670 | 5.9 | 612,276 |

### n = 128

| arm | w | t | slice | sandwich | blk1 width | blk2 width | sandwich width | stage-2 gates | fires/250 | fire span | stage-2 dep. width | gadget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n128_leg | 1 | 0 | on | 7,920 | 18.7 | 19.3 | 19.1 | 1,275,100 | 13.2 | 3,307 | 9.5 | 1,285,912 |
| n128_w1_t1 | 1 | 1 | off | 6,533 | 17.3 | 17.5 | 17.5 | 1,094,842 | 13.1 | 3,427 | 9.5 | 1,105,654 |
| n128_w1_t448 | 1 | 448 | off | 7,427 | 17.3 | 22.3 | 19.9 | 1,245,508 | 13.2 | 3,455 | 9.5 | 1,256,320 |
| n128_w1_t896 | 1 | 896 | off | 8,323 | 17.3 | 27.1 | 22.3 | 1,398,855 | 13.4 | 3,498 | 9.7 | 1,409,667 |
| n128_w1_t896_sl | 1 | 896 | on | 10,058 | 19.5 | 27.2 | 23.6 | 1,635,475 | 13.3 | 3,368 | 9.8 | 1,646,287 |
| n128_w6_t1 | 6 | 1 | off | 6,517 | 20.2 | 21.1 | 20.8 | 1,091,319 | 13.2 | 3,436 | 9.6 | 1,101,998 |
| n128_w6_t448 | 6 | 448 | off | 7,411 | 20.2 | 26.8 | 23.7 | 1,244,328 | 13.4 | 3,518 | 9.9 | 1,255,007 |
| n128_w6_t896 | 6 | 896 | off | 8,307 | 20.2 | 32.6 | 26.5 | 1,395,945 | 13.5 | 3,529 | 9.9 | 1,406,624 |
| n128_w11_t1 | 11 | 1 | off | 6,521 | 22.6 | 22.9 | 22.9 | 1,091,437 | 13.4 | 3,458 | 9.8 | 1,102,160 |
| n128_w11_t448 | 11 | 448 | off | 7,415 | 22.6 | 29.1 | 26.0 | 1,245,856 | 13.4 | 3,531 | 10.0 | 1,256,579 |
| n128_w11_t896 | 11 | 896 | off | 8,311 | 22.6 | 35.4 | 29.2 | 1,396,820 | 13.5 | 3,547 | 10.0 | 1,407,543 |
| n128_w16_t1 | 16 | 1 | off | 6,512 | 25.0 | 25.9 | 25.5 | 1,092,291 | 13.3 | 3,467 | 9.8 | 1,103,090 |
| n128_w16_t448 | 16 | 448 | off | 7,406 | 25.0 | 33.0 | 29.0 | 1,242,856 | 13.5 | 3,516 | 9.9 | 1,253,655 |
| n128_w16_t896 | 16 | 896 | off | 8,302 | 25.0 | 40.1 | 32.6 | 1,396,831 | 13.7 | 3,579 | 10.2 | 1,407,630 |
| n128_w21_t1 | 21 | 1 | off | 6,527 | 28.8 | 29.5 | 29.1 | 1,091,774 | 13.6 | 3,536 | 10.1 | 1,102,495 |
| n128_w21_t448 | 21 | 448 | off | 7,421 | 28.8 | 37.6 | 33.1 | 1,246,809 | 13.7 | 3,596 | 10.2 | 1,257,530 |
| n128_w21_t896 | 21 | 896 | off | 8,317 | 28.8 | 45.6 | 37.1 | 1,397,503 | 13.9 | 3,609 | 10.5 | 1,408,224 |
| n128_w26_t1 | 26 | 1 | off | 6,512 | 30.4 | 32.9 | 31.6 | 1,091,510 | 13.7 | 3,582 | 10.3 | 1,102,253 |
| n128_w26_t448 | 26 | 448 | off | 7,406 | 30.4 | 41.9 | 36.0 | 1,243,124 | 13.7 | 3,598 | 10.1 | 1,253,867 |
| n128_w26_t896 | 26 | 896 | off | 8,302 | 30.4 | 51.0 | 40.3 | 1,395,077 | 13.9 | 3,635 | 10.3 | 1,405,820 |
| n128_w31_t1 | 31 | 1 | off | 6,525 | 35.0 | 34.8 | 35.1 | 1,092,914 | 13.7 | 3,598 | 10.2 | 1,103,585 |
| n128_w31_t448 | 31 | 448 | off | 7,419 | 35.0 | 44.3 | 39.9 | 1,244,372 | 13.7 | 3,619 | 10.4 | 1,255,043 |
| n128_w31_t896 | 31 | 896 | off | 8,315 | 35.0 | 53.9 | 44.7 | 1,397,373 | 13.9 | 3,661 | 10.5 | 1,408,044 |
| n128_w36_t1 | 36 | 1 | off | 6,525 | 38.7 | 39.0 | 38.8 | 1,092,304 | 13.8 | 3,622 | 10.4 | 1,103,014 |
| n128_w36_t448 | 36 | 448 | off | 7,419 | 38.7 | 49.6 | 44.2 | 1,245,093 | 13.9 | 3,661 | 10.4 | 1,255,803 |
| n128_w36_t896 | 36 | 896 | off | 8,315 | 38.7 | 60.3 | 49.5 | 1,397,104 | 14.1 | 3,689 | 10.6 | 1,407,814 |
| n128_w41_t1 | 41 | 1 | off | 6,524 | 43.4 | 44.2 | 43.8 | 1,092,414 | 13.9 | 3,614 | 10.4 | 1,103,194 |
| n128_w41_t448 | 41 | 448 | off | 7,418 | 43.4 | 56.3 | 49.8 | 1,244,556 | 13.9 | 3,645 | 10.4 | 1,255,336 |
| n128_w41_t896 | 41 | 896 | off | 8,314 | 43.4 | 68.4 | 55.8 | 1,395,502 | 14.0 | 3,657 | 10.4 | 1,406,282 |
| n128_w41_t896_sl | 41 | 896 | on | 10,058 | 34.0 | 48.5 | 41.4 | 1,637,885 | 13.8 | 3,563 | 10.3 | 1,648,665 |

Lanes experiment (uncommitted build with the 64-wire reserve cap lifted, structure only; same seeds):

| config | fires / 250 | fire span |
|---|---|---|
| n = 64, w = 21, t = 384, band 256 / 384 / 512, window 4096 | 13.2 / 13.4 / 14.0 | 3.5k-3.8k |
| n = 64, w = 21, band 512, window 16384 | 16.0 | 8.6k |
| n = 64, w = 1, band 512, window 16384 | 14.9 | 7.6k |
| n = 128, w = 41, t = 896, band 256 / 512 | 14.0 / 15.4 | 3.7k / 3.9k |
| n = 128, w = 41, band 512, window 16384 | 19.0 | 11.3k |
| n = 128, w = 1, band 512, window 16384 | 17.5 | 10.2k |

**(d) Battery** (per arm: validated census + audit, then the 11-case narrow portfolio: factor windows 125/250/500/1000,
shifted/permuted 125, pairs, payload factor 125/250, payload delta 250, raw; deeper = internal births minus the public
quadratic boundary; per-read recovery from the union of the ledgers):

Status at the time of writing (2026-11-01 22:30 UTC): running, n = 64 arms first, 24 jobs at nice 5 on a server
loaded to ~230 by the variants campaign. The validated census (direct and affine proxies with the witness audit) finds
0 internal C births in every arm prepared so far, as in every three-mask stage 2. First narrow ledgers (deeper births of
the 873 or 865 deep internal ones; single seed per arm, the 500- and 1000-gate windows and two payload cases still
running, so the unions are partial):

| arm | cases done | factor 125 | perm 125 | shift63 125 | w250 | payload w125 | pairs | union (deeper) |
|---|---|---|---|---|---|---|---|---|
| n64_leg | 6/11 | 23 | 28 | 27 | - | 30 | 1 | 73 |
| n64_w1_t1 | 2/11 | 13 | - | - | - | - | - | 13 |
| n64_w1_t384 | 7/11 | 30 | 33 | 23 | 74 | 32 | 2 | 104 |
| n64_w21_t384 | 6/11 | 13 | 16 | 11 | 73 | 23 | - | 91 |

Too early to rank the arms: the 125-gate windows put the default stage 1 (w = 1, t = 384) at or slightly above the old
sandwich and the layered C (w = 21) below both, while the 250-gate window finds the same ~73 in both new arms. The
strongest cases (500, 1000) are still to come. Interim tally: `python3 ~/variants/s2new/py/peek_s2.py` on the server
(writes `~/variants/s2new/PEEK.json`); complete arms get `run/battery/<arm>/{narrow,DONE}.json` and the per-read
recovery tables `attrib/<case>.txt`, `attrib/union.txt` (attrib3.py: recovery rate by reads, by dead-S1 re-encodings,
by lifetime, by depth decile, recovering windows by position decile). Local tables: `s12_runs/tables.py` after pulling
`~/variants/run/battery/s2v_*` into `s12_runs/battery/`.

## Suggestions

1. Stage 2 is the bottleneck for concurrency, not stage 1: with 8 lanes (n = 64) or 16 (n = 128) the fires per window
   stay at ~10 / ~14 for every w and t. Lifting the 64-wire reserve cap (`helper_reserve`) together with the interleave
   window (band 512 and window 16384: 16.0 / 19.0) is the lever; a band of 512 at n = 64 makes a 640-wire stage 2
   (within the ANF tools' 768 and stage 6's 1024). Not changed in the code: a decision for after the battery.
2. Keep slicing off unless the battery shows the chaff matters: it costs 17-20 % of stage 2 and gives back part of a
   layered C's width (block 1 at w = 21: 23.0 -> 19.6).
3. Choose t from the exposure results: V at t = n log2 n costs 28-35 % of stage 2 over t = 1 and raises block 2's width 1.5-1.7x.
4. Without the reverse contract the band reseed could go as well (the interleaving note's suggestion 3); kept for now.
5. Recipe v14 consequences: v13 runs are refused on resume (the message lists the settings that rebuild v13's stages
   1-2); the validated chain needs the legacy stage 1 (`slicing = true`, `v_gates = 0`); nonlinear291 takes the V gadget
   only with slicing; a forward-only stage 2 runs the adaptive stage 6 with `--forward-only`.
