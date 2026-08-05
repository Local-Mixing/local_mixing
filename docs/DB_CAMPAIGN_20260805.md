# The DB Move, Measured End to End

*2026-08-05*

> Markdown rendering of `docs/DB_CAMPAIGN_20260805.tex`. The PDF is authoritative for figures, diagrams and display math.

## Abstract
This note reports a campaign of roughly seventy fmix runs that measured the
frozen-store DB move along four axes — window *geometry* (convex vs
contiguous), *prefix descent*, *window length* s_db, and
*curation* — in both COMP and MIX modes, on material ranging from
fresh gadget output to near-minimal circuits. Four results dominate.
(1) Convex beats contiguous on every axis in COMP: 16× the gates
removed, 7× the ancestry transport, in 1/31 the wall time; the
shipped default was inverted and has been fixed. (2) Prefix descent is not a
size lever but a *transport* lever, worth 600× on ancestry while
being 6× *worse* on size per CPU-second. (3) The MIX-DB move is a
mean-reversion operator on window length with a fixed point near 5: short
windows expand, long windows compress, and s_db is therefore a
net-growth-rate knob rather than a search-width knob. (4) The curated store
is a *wide-window* store — worthless at length 1 and worth 19×
at length 20. We close with the GSS profile, the settings for running fmix on
a gadgetized sliced sandwich, and with a section on the measurement traps
this campaign walked into, which we think is the most reusable part.

## What was measured, and on what

All runs use the frozen store `frozen_m1_m11` with the curated store
`frozen_curated_m1_m11` under the `legacy-swapped-controls` value
convention, `–db-advance` on throughout, span cap 30, degree cap 9.

| tag | description | gates | character |
|---|---|---|---|
| `pre2_100k` | fresh gadget output | 100,000 | 52% g57 fossils |
| `nR20_mixed` | profile-run output, near-minimal | 179,132 | 99.3% g57, d_≈0.002 |
| `cs_p20_mixed` | p_mix 0.20 growth run, never compressed | 202,674 | MIX-grown fat |
| `cs_p35_mixed` | p_mix 0.35 growth run, never compressed | 317,813 | more fat |

*Input circuits. The last two were produced by a parallel session's
p_mix sweep and are used as "lots of easily compressible fat".*

## Part I: the 32-arm COMP factorial

### Design

A full 2^5 factorial on `pre2_100k`: s_db∈{12,13} ×
descent on/off × convex/contiguous × reshuffle on/off ×
curated on/off. 500k moves, seed 909, `–db-mode comp –p-comp 1.0
–p-any 0.0`, `–anc-samples 256`. All 16 convex arms and all 8
no-descent contiguous arms completed; the 8 descent×contiguous arms are
30× slower and are reported separately.

### Convex half: main effects

| factor (mean of 8) | gates removed | DB attempts | gates/hit | bits/splice |
|---|---|---|---|---|
| descent on / off | 15,290 / 8,748 | 4,853,272 / 469,074 | 0.060 / 0.286 | 0.232 / 0.355 |
| s_db 13 / 12 | 11,906 / 12,133 | 2,785,496 / 2,536,850 | 0.177 / 0.169 | 0.295 / 0.293 |
| reshuffle on / off | 12,284 / 11,754 | 2,641,481 / 2,680,865 | 0.172 / 0.173 | 0.288 / 0.299 |
| curated on / off | 12,077 / 11,961 | 2,629,091 / 2,693,255 | 0.159 / 0.187 | **0.562 / 0.025** |

The effects are additive with no sign reversals, and the best single cell is
also the composite of all four: `s_db 12, convex, descent, reshuffle,
curated` produced the smallest circuit of all sixteen (84,321 from 100,000)
*and* the second-highest ancestry.

### The mixing channel: descent is worth 600×

| factor (mean of 8) | est_anc | cov | ent | reach | carriers | gen_r |
|---|---|---|---|---|---|---|
| descent on / off | **26,820 / 44** | 0.456 / 0.037 | 0.668 / 0.085 | 0.448 / 0.025 | 0.807 / 0.093 | 0.859 / 0.331 |
| s_db 13 / 12 | 13,371 / 13,492 | 0.245 / 0.248 | 0.373 / 0.381 | 0.236 / 0.237 | 0.445 / 0.455 | 0.591 / 0.599 |
| reshuffle on / off | 13,972 / 12,892 | 0.251 / 0.242 | 0.383 / 0.371 | 0.240 / 0.233 | 0.457 / 0.443 | 0.603 / 0.588 |
| curated on / off | 13,955 / 12,909 | 0.255 / 0.238 | 0.383 / 0.371 | 0.244 / 0.229 | 0.459 / 0.441 | 0.609 / 0.582 |

Here `est_anc` is the mean number of distinct *input* gates each
current gate descends from (computed as desc_mean× m/size
over K=256 tracers). So the descent arms have every gate carrying lineage
from 28% of the 100k input, while the no-descent arms carry lineage
from 44 gates — essentially nothing. **Without descent, COMP is
a size-only pass.**

### Descent is a transport lever, not a size lever

The two axes point in opposite directions per unit CPU:

|  | size | ancestry |
|---|---|---|
| descent | 9.6 gates/s | 17.9 anc/s |
| no descent | **58 gates/s** | 0.29 anc/s |
| ratio | 6× worse | 60× better |

The size trajectories explain it. Descent front-loads and exhausts; no-descent
decays gently and *overtakes it on marginal rate* by move 500k:

| moves | descent removed | Δ/50k | no-descent removed | Δ/50k |
|---|---|---|---|---|
| 50k | 5,159 | 5,159 | 1,771 | 1,771 |
| 100k | 8,101 | 2,942 | 3,012 | 1,241 |
| 200k | 11,325 | 1,257 | 5,314 | 974 |
| 300k | 13,238 | 913 | 7,033 | 728 |
| 400k | 14,687 | 614 | 8,319 | 578 |
| 500k | **15,679** | **427** | 9,350 | **533** |

Consequence: for *compression at all costs* under a CPU budget, drop
descent and spend the savings on moves. For compression *without
compromising mixing*, descent is not optional.

### Curated in COMP: the contract holds on size, fails on entropy

Curated is worth +1.0% on size — confirming the "COMP is regular-only"
contract on that axis — but it also uses *2.4% fewer* attempts (it
hits earlier in the cascade), gives +8.1% ancestry in all four matched
descent pairs, and multiplies selection entropy by 22×. Its extra hits
are size-neutral re-encodings, visible in gates/hit (0.159 with vs 0.187
without). It supplies 38–41% of successful splices under descent and
61–62% without. **Recommendation: reverse the contract** — the
original was written on the size axis alone.

### Contiguous: 8 matched no-descent pairs

| cell | geom | final | rm | hit% | g/hit | bits | est_anc | carr | gen_r | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| s12 noshuf nocur | ctg | 99,586 | 414 | 2.56 | 0.033 | 0.004 | 3.5 | 0.009 | 0.142 | 3,852s |
|  | cvx | 91,520 | 8,480 | 5.86 | 0.306 | 0.035 | 37.0 | 0.079 | 0.318 | 152s |
| s12 noshuf cur | ctg | 99,578 | 422 | 1.91 | 0.045 | 1.585 | 3.2 | 0.008 | 0.155 | 3,968s |
|  | cvx | 91,395 | 8,605 | 7.30 | 0.253 | 0.677 | 49.4 | 0.105 | 0.350 | 149s |
| s12 shuf nocur | ctg | 99,273 | 727 | 3.65 | 0.041 | 0.003 | 10.0 | 0.024 | 0.212 | 4,993s |
|  | cvx | 90,790 | 9,210 | 6.63 | 0.296 | 0.034 | 42.4 | 0.093 | 0.312 | 113s |
| s12 shuf cur | ctg | 99,291 | 709 | 2.74 | 0.053 | 1.467 | 8.9 | 0.021 | 0.228 | 5,033s |
|  | cvx | 90,650 | 9,350 | 8.03 | 0.252 | 0.671 | 69.1 | 0.133 | 0.378 | 160s |
| s13 noshuf nocur | ctg | 99,580 | 420 | 2.35 | 0.037 | 0.004 | 2.6 | 0.007 | 0.100 | 4,634s |
|  | cvx | 91,877 | 8,123 | 5.35 | 0.320 | 0.034 | 25.3 | 0.057 | 0.287 | 172s |
| s13 noshuf cur | ctg | 99,568 | 432 | 1.84 | 0.048 | 1.595 | 2.7 | 0.007 | 0.111 | 4,690s |
|  | cvx | 91,693 | 8,307 | 6.64 | 0.267 | 0.684 | 41.5 | 0.089 | 0.334 | 172s |
| s13 shuf nocur | ctg | 99,414 | 586 | 3.31 | 0.037 | 0.003 | 9.2 | 0.022 | 0.218 | 5,540s |
|  | cvx | 91,132 | 8,868 | 5.87 | 0.320 | 0.036 | 34.1 | 0.079 | 0.308 | 141s |
| s13 shuf cur | ctg | 99,295 | 705 | 2.46 | 0.059 | 1.465 | 8.0 | 0.019 | 0.218 | 5,728s |
|  | cvx | 90,956 | 9,044 | 7.10 | 0.273 | 0.668 | 52.7 | 0.111 | 0.360 | 185s |
| **mean** | **ctg** |  | **552** | 2.60 | 0.044 | 0.766 | **6.0** | 0.015 | 0.173 | **4,805s** |
|  | **cvx** |  | **8,748** | 6.60 | 0.286 | 0.355 | **43.9** | 0.093 | 0.331 | **156s** |

Three points. First, convex wins size by 16× and transport by 7×
in *every* pair. Second, contiguous costs 31× the wall time at
essentially identical attempt counts (487k vs 470k), so the per-attempt
canonicalization is 30× dearer — **the damage is window
width, not the span cap**, since contiguous skips only 0.5–3.6% of attempts
at the cap. Third, contiguous's one apparent win — per-splice entropy
(0.766 vs 0.355) — **does not survive aggregation**: convex lands
2.4× more splices, so it delivers more *total* entropy in all 8
pairs (e.g. 23,021 vs 14,881 bits at s12/noshuf/cur).

### Contiguous with descent

| arm | final | removed | est_anc | carriers |
|---|---|---|---|---|
| s12 noshuf cur | 98,959 | 1,041 | 195.0 | 0.228 |
| s12 noshuf nocur | 99,027 | 973 | 221.7 | 0.248 |
| s12 shuf cur | 97,372 | 2,628 | 4,269.9 | 0.585 |
| s12 shuf nocur | 97,325 | 2,675 | **6,372.7** | 0.606 |
| s13 noshuf cur | 98,997 | 1,003 | 190.2 | 0.221 |
| s13 noshuf nocur | 99,030 | 970 | 235.5 | 0.247 |
| s13 shuf cur^ | 97,687 | 2,313 | 2,686.6 | 0.557 |
| s13 shuf nocur^ | 97,748 | 2,252 | 3,650.3 | 0.576 |

{ ^ stopped at 450k of 500k moves.}

Descent lifts contiguous a lot in relative terms (removals 2.7–3.7×,
ancestry up to 640×) but it remains far behind convex-with-descent
(14,948–15,679 removed, est_anc 23,704–29,487). Two notes worth
carrying: reshuffle is contiguous's *dominant* lever (est_anc
222→6,373, a 29× swing), and **curated *hurts*
ancestry in contiguous** — negative in all four pairs — the exact opposite
of its +8% in convex.

### A bug: the shipped COMP sampler was inverted

`–p-convex-comp` shipped at **0.1** from the 2026-08-03 defaults
commit, i.e. 90% contiguous, when the intent was the reverse. Every run
taking the COMP default before 2026-08-04 spent 90% of its COMP DB
budget on the geometry that loses on every axis above. Fixed to 0.9 in
`46350179`. MIX's `–p-convex 0.4` was left alone: no MIX run had
ever varied geometry, so there was no measurement behind it either way.

## Part II: per-length structure

### Method

With `–no-db-prefixes` the window length is drawn *uniformly* from
1..s_db, so "hit rate at length k" is a genuine per-length conversion
rate. With descent on, the same counter would be conditional on every longer
length having already failed. Every table in this part uses the uniform draw,
s_db=20, 600k moves — about 30k independent attempts per length.

### COMP, convex, four circuits: fat changes amplitude, not width

| len | base (100k fresh) | nR20 (179k minimal) | fat200 (203k) | fat318 (318k) |
|---|---|---|---|---|
| 3 | 336 | 20 | 1,348 | 2,912 |
| 4 | 541 | 49 | 1,917 | 4,027 |
| 5 | 960 | 93 | 4,163 | 8,846 |
| 6 | 1,274 | 235 | 5,438 | 11,105 |
| 7 | **1,686** | 362 | 7,689 | 15,304 |
| 8 | 1,398 | **433** | **8,769** | **17,522** |
| 9 | 812 | 178 | 6,573 | 15,403 |
| 10 | 347 | 88 | 4,620 | 12,353 |
| 12 | 52 | 42 | 2,125 | 6,883 |
| 14 | 12 | 38 | 873 | 3,642 |
| 16 | 0 | 40 | 348 | 1,851 |
| 18 | 0 | 12 | 158 | 1,049 |
| 20 | 0 | 26 | 102 | 367 |
| total | 7,638 | 1,731 | 50,029 | 122,324 |
| % of circuit | 7.6% | **1.0%** | 24.7% | **38.5%** |
| peak length | 7 | 8 | 8 | 8 |
| 50% of mass by | 7 | 8 | 8 | 9 |
| share from len >14 | 0.0% | 5.4% | 3.3% | 6.9% |

The compressible-fat hypothesis is confirmed on *volume*: MIX-grown
material is 25–38% compressible, near-minimal material 1.0%, a 38×
spread. But the *useful-width profile barely moves*: across that range
the peak shifts by one gate and the median by two. This refuted our prior
expectation that the fresh-material peak at 7 was an artifact of picked-over
material. What fat buys is amplitude, plus a modest long tail (lengths 13–20
give 14.5% of `fat318`'s removals versus 0.5% of `base`'s).

Weighting by draw cost, the efficiency-optimal s_db is **8 for lean
material and 10–12 for fat** — landing on the factorial's s_db=12 winner
and confirming it was not an artifact of one input.

### COMP, contiguous, same four circuits

| circuit | in | cvx out | cvx removed | ctg out | ctg removed | cvx/ctg |
|---|---|---|---|---|---|---|
| base | 100,000 | 92,362 | 7,638 (7.6%) | 99,612 | 388 (0.4%) | **19.7×** |
| nR20 | 179,132 | 177,401 | 1,731 (1.0%) | 178,773 | 359 (0.2%) | 4.8× |
| fat200 | 202,674 | 152,645 | 50,029 (24.7%) | 186,706 | 15,968 (7.9%) | 3.1× |
| fat318 | 317,813 | 195,489 | 122,324 (38.5%) | 290,553 | 27,260 (8.6%) | 4.5× |

Convex wins everywhere, but the gap *narrows sharply with fat*: 19.7×
on lean fresh material, 3–4.5× on MIX-grown circuits. Contiguous does
find real compression when obvious fat exists; it collapses on picked-over
material. Two structural differences from the convex ladder: contiguous
*is* span-crushed above 11 at this s_db (skips exceed 50% by
length 13–14 and reach 94–99% at 20, against convex's 7.45% maximum), and
its useful width *does* move with the material (peak at 3, 9, 11, 10
across the four circuits, against convex's steady 7, 8, 8, 8). Contiguous
wall times were 4,973–8,648s.

### MIX: the DB move is a mean-reversion operator

| len | hit% | net ( - = growth) | len | hit% | net |
|---|---|---|---|---|---|
| 1 | 99.78 | -261,276 | 11 | 15.80 | +21,964 |
| 2 | 99.02 | -182,797 | 12 | 12.62 | +21,237 |
| 3 | 97.53 | -79,246 | 13 | 10.33 | +20,719 |
| 4 | 91.15 | -27,073 | 15 | 6.60 | +17,047 |
| 5 | 75.21 | +175 | 18 | 2.98 | +10,279 |
| 6 | 67.09 | +8,477 | 20 | 1.72 | +6,949 |

*MIX, convex, curated, s_db=20, 600k moves. The fixed point is at
window length ≈5.*

Short windows almost always find a longer spelling and expand; long windows
almost always find a shorter one and compress. Over the run, lengths 1–4
added 556k gates and lengths 5–20 clawed back 250k, net +313k. So
**s_db in MIX is a net-growth-rate knob**, not a search-width knob:
raising it dilutes expansion with self-compression. This is a second lever on
growth alongside `p_mix`, which the layer-2 controller does not
currently know about.

Each geometry has its own fixed point: convex+curated ≈5,
convex-curated ≈5.5, contiguous+curated ≈6.5. That is why
contiguous grew *more* (483,839 vs 412,797 final) despite adding about
the same at lengths 1–2: it compresses less above the crossover.

At s_db=20 specifically, lengths 7–20 added **exactly zero** gates
across all fourteen of them while removing 31,372; 98.6% of all expansion
came from lengths 1–4 on 20% of the draws, and 77% from lengths 1–2 alone.

### Hit rate by length: the full mode × geometry × curation cube

All eight cells, same circuit (`pre2_100k`), same budget, uniform draw
over 1..20, ≈30k attempts per length.

|  | MIX | COMP |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| len | cvx +cur | cvx -cur | ctg +cur | ctg -cur | cvx +cur | cvx -cur | ctg +cur | ctg -cur |
| 1 | 99.78 | 99.75 | 99.79 | 99.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 99.02 | 98.83 | 98.23 | 97.95 | 0.00 | 0.00 | 1.60 | **11.95** |
| 3 | 97.53 | 96.79 | 91.45 | 83.91 | 13.87 | 6.71 | 12.33 | 9.83 |
| 4 | 91.15 | 85.47 | 79.22 | 76.71 | 19.59 | 13.64 | 5.66 | 5.37 |
| 5 | 75.21 | 70.10 | 64.23 | 61.73 | 16.64 | 16.64 | 2.62 | 2.71 |
| 6 | 67.09 | 62.43 | 54.75 | 49.87 | 17.15 | 17.32 | 1.01 | 1.03 |
| 7 | 45.50 | 39.42 | 42.89 | 39.72 | 8.39 | 8.78 | 0.52 | 0.50 |
| 8 | 35.49 | 23.57 | 32.89 | 31.47 | 2.94 | 2.87 | 0.34 | 0.25 |
| 9 | 25.52 | 15.06 | 24.36 | 24.53 | 1.17 | 1.16 | 0.12 | 0.10 |
| 10 | 20.14 | 9.72 | 15.82 | 14.24 | 0.36 | 0.39 | 0.04 | 0.05 |
| 11 | 15.80 | 6.58 | 11.77 | 8.60 | 0.16 | 0.18 | 0.01 | 0.01 |
| 12 | 12.62 | 3.85 | 7.56 | 2.42 | 0.04 | 0.03 | 0.00 | 0.00 |
| 13 | 10.33 | 2.44 | 6.33 | 1.70 | 0.02 | 0.01 | 0.00 | 0.00 |
| 14 | 8.10 | 1.55 | 4.64 | 0.75 | 0.01 | 0.00 | 0.00 | 0.00 |
| 15 | 6.60 | 0.95 | 3.81 | 0.37 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 4.76 | 0.60 | 2.77 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 |
| 17 | 3.87 | 0.45 | 2.27 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| 18 | 2.98 | 0.21 | 1.45 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| 19 | 2.15 | 0.18 | 1.20 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 20 | **1.72** | **0.09** | 0.84 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| overall | **36.27** | 30.82 | 32.33 | 29.72 | **4.02** | 3.39 | 1.22 | 1.59 |

| cell | added | removed | net | max span-skip |
|---|---|---|---|---|
| MIX cvx +cur | 562,415 | 249,618 | +312,797 | 0.9% |
| MIX cvx -cur | 324,931 | 94,309 | +230,622 | 1.9% |
| MIX ctg +cur | 493,073 | 109,234 | +383,839 | **70.4%** |
| MIX ctg -cur | 263,530 | 21,097 | +242,433 | **94.2%** |
| COMP cvx +cur | 0 | 7,638 | -7,638 | 2.3% |
| COMP cvx -cur | 0 | 7,454 | -7,454 | 2.2% |
| COMP ctg +cur | 0 | 388 | -388 | **99.3%** |
| COMP ctg -cur | 0 | 390 | -390 | **99.4%** |

MIX converts 9× more often than COMP, which is just the acceptance
rule (MIX takes any spelling, COMP requires strictly shorter; COMP's 0.00% at
lengths 1–2 is structural). Half of all successful splices come from lengths
≤4 in every MIX arm (≤5 in COMP), on 20–25% of the draws; above
length 7 the decay is close to geometric, ×0.75 per gate with curation
and ×0.6 without.

**Curation's value is entirely at the wide end, in both geometries.**
Nothing at length 1 (99.78 vs 99.75). In convex, 3.3× at length 12 and
19× at length 20; in contiguous the ratio is *larger* still —
3.1× at 12 and **84×** at 20 (0.84 vs 0.01). The curated
store is a wide-window store.

**But curation is net-negative for COMP contiguous** — 1.22% against
1.59% overall, the only cell where it lowers the aggregate hit rate. It is
driven entirely by length 2, where curation costs 11.95% → 1.60%.
Removals are unchanged (388 vs 390), so this is substitution of equals rather
than lost compression — but it is the same displacement effect that cost
ancestry in the descent×contiguous arms of Part I, now visible on a
third independent measure. Curation should be read as a convex-mode asset.

Finally, the span-skip column isolates what stops contiguous at high s_db:
70–94% of MIX contiguous attempts and *99%* of COMP contiguous
attempts are rejected at the cap, against ≤2.3% for every convex cell.

## Part III: MIX s_db× geometry at matched size

### Design

Arms are compared at *matched final size*, not matched moves: MIX with
pay-random grows, and net growth per unit work falls monotonically with
s_db, so a fixed move budget would leave every arm at a different size and
a low-s_db arm would "win" transport merely by being bigger. Each arm's
move budget was calibrated from a measured growth pass so it lands at
≈2× the 100k input.

### Results

| geo | s_db | moves | size | wall | added | removed | mv/kgate | bits/spl | tot bits | bits/kmv |
|---|---|---|---|---|---|---|---|---|---|---|
| cvx | 3 | **19,000** | 201,432 | **1s** | 101,766 | 334 | **187** | **3.789** | 70,157 | **3,692** |
| cvx | 5 | 30,000 | 202,095 | 2s | 103,501 | 1,406 | 294 | 3.100 | 80,364 | 2,679 |
| cvx | 7 | 44,000 | 204,755 | 3s | 109,719 | 4,964 | 420 | 2.845 | 91,029 | 2,069 |
| cvx | 9 | 60,000 | 205,961 | 6s | 116,686 | 10,725 | 566 | 2.752 | 100,330 | 1,672 |
| cvx | 12 | 86,000 | 205,515 | 12s | 126,403 | 20,888 | 815 | 2.690 | 111,562 | 1,297 |
| cvx | 20 | 157,000 | 204,028 | 71s | 138,910 | 34,882 | 1,509 | 2.632 | **123,075** | 784 |
| ctg | 3 | 21,000 | 199,925 | 1s | 100,011 | 86 | 210 | 3.774 | 71,664 | 3,413 |
| ctg | 5 | 34,000 | 203,909 | 7s | 104,259 | 350 | 327 | 3.055 | 75,846 | 2,231 |
| ctg | 7 | 48,000 | 205,271 | 42s | 106,370 | 1,099 | 456 | 2.795 | 77,508 | 1,615 |
| ctg | 9 | 65,000 | 211,802 | 165s | 114,305 | 2,503 | 581 | 2.676 | 83,563 | 1,286 |
| ctg | 12 | 93,000 | 217,039 | 559s | 122,293 | 5,254 | 795 | 2.610 | 89,810 | 966 |
| ctg | 20 | 171,000 | 224,427 | 1,039s | 135,165 | 10,738 | 1,374 | 2.550 | 99,570 | 582 |

### The contiguous cost curve is non-monotone

| s_db | 3 | 5 | 7 | 9 | 12 | 20 |
|---|---|---|---|---|---|---|
| ctg/cvx wall | 1.1× | 3.6× | 12.6× | 29.9× | 47.8× | 14.7× |

The penalty peaks at s_db=12 and then *falls*, because by s_db=20
the span cap rejects 70% of contiguous windows *before*
canonicalization — the cap acts as a cheap early-out. The expensive regime is
exactly s_db 9–12, which is where the shipped MIX default sat.

### Verdict, and one thing this cannot answer

Convex at every s_db (cheaper and more total entropy); low s_db for
expansion efficiency (187 vs 1,509 moves per 1,000 gates grown).

But **est_anc stays 1–10 in all twelve arms** — pure MIX expansion to
2× transports essentially no ancestry at any setting. The braiding seen
in long p_mix runs (est_anc 12k–44k, carriers 0.98) comes from sustained
churn over millions of moves, not from the expansion leg. So this sweep
answers "which s_db expands best" and *not* "which braids best at
held size"; the latter needs long runs with size pinned.

Under the phase-B reframe, where spread is judged in *absolute* gates,
reach of 0.001–0.004 on a 205k circuit is 200–820 positions, which is the
same order as the 850 that reframe calls meaningful. So these arms are
not as inert as the fractional numbers suggest.

## Part IV: re-reading the p_mix dose-response

Eight runs at 2M moves, single DB configuration, only `p_mix` varying:

| p_mix | size | est_anc | cov | reach | carriers | card | anc (mv/gate) |
|---|---|---|---|---|---|---|---|
| 0.05 | 128,355 | 44,302 | 0.519 | 0.510 | 0.984 | 115.2 | 2,840 |
| 0.10 | 150,699 | 38,844 | 0.461 | 0.446 | 0.988 | 100.6 | 2,921 |
| 0.15 | 176,429 | 32,181 | 0.381 | 0.369 | 0.991 | 83.1 | 2,848 |
| 0.20 | 202,674 | 26,141 | 0.313 | 0.300 | 0.993 | 67.4 | 2,649 |
| 0.25 | 235,577 | 22,119 | 0.268 | 0.254 | 0.994 | 57.0 | 2,605 |
| 0.30 | 277,404 | 17,719 | 0.218 | 0.204 | 0.995 | 45.6 | 2,458 |
| 0.35 | 317,813 | 14,896 | 0.187 | 0.172 | 0.996 | 38.3 | 2,368 |
| 0.40 | 369,987 | 12,220 | 0.158 | 0.141 | 0.997 | 31.4 | 2,259 |

Read raw, this looks like a 3.6× collapse in transport as p_mix rises —
a strong argument for keeping it low. **It mostly isn't.** All eight got
2M moves at different final sizes, so moves-per-gate falls 15.6→5.4,
a 2.9× dilution that nearly accounts for the whole effect. Normalised
(last column), the decline is **26% across the entire range**. The one
unambiguous result: carriers saturates at ≥0.984 everywhere, so all the
differentiation is in depth, none in breadth.

## Part V: the GSS profile

### Specification

The DB settings for running fmix on a gadgetized sliced sandwich. Exposed as
`–gss`; explicit flags win; composes with `–phase-a`.

|  | COMP-DB | MIX-DB |
|---|---|---|
| curated | on | on |
| descent | **on** | **off** |
| `p_mingen` | 0 | 0.5 |
| geometry | convex 95% / contiguous 5% | convex 50% / contiguous 50% |
| s_db convex | 12 | 6 |
| s_db contiguous | 6 | 6 |

`–gss` deliberately does *not* set `p_mix`: the MIX/COMP
balance is the layer-2 controller's lever, and the profile is intended to be
the right per-mode setting at every p_mix.

### Evidence for each setting

COMP convex s_db=12 with descent is exactly the factorial's best cell, and
s_db=12 matches the measured productive band on GSS-like material
(`nR20_mixed`: peak at 8, lengths >14 worth 5.4%). Capping contiguous
at s_db=6 keeps it inside both the affordable regime (the ctg/cvx cost
ratio is 3.6× at 5 but 47.8× at 12) and the span-safe one (skips
begin around length 10); at 5% weight it buys geometric diversity cheaply, and
on all-g57 material it still converts at 52–96% for lengths 3–9. MIX at
s_db=6 without descent converts on ≈88% of draws (99/99/97/91/75/67%
across lengths 1–6) at the cheapest width.

### The profile is g57-preserving, by design

| arm | comp= | g57= | shaped= | polf= |
|---|---|---|---|---|
| M20_cvx_cur (MIX) | 398,904 | 398,904 | 398,904 | **0.000** |
| MS_cvx_s12 (MIX) | 175,555 | 175,555 | 175,555 | **0.000** |
| LB_base (COMP) | 61,691 | 61,691 | 61,691 | **0.000** |
| LB_nR20 (COMP) | 176,118 | 176,118 | 176,118 | **0.000** |
| f32_12_desc_cvx_shuf_cur (COMP) | 69,340 | 69,340 | 69,340 | **0.000** |

`comp`=`g57`=`shaped` holds exactly in every DB-only run, and
`polf` — the structure-breakdown meter — is identically zero. The store
emits g57-form words, so every DB splice re-spells a g57 word as another g57
word. This is intentional: the profile's job is re-spelling, and breaking g57
form is a separate concern requiring the twist family, not the store.

### Implementation

Two mechanisms had to be added.

*Geometry is now drawn once per round, before the length.* It used to be
drawn inside `sample_window` *after* the length was fixed, which
made a geometry-conditional length inexpressible and let the
best-of-`litter_samples` selection compare windows drawn under different
geometries. New flags `–s-db-ctg` and `–s-db-comp-ctg`; resolution
is most-specific-first (mode+geometry → mode → base),
with 0 falling through rather than clamping.

*Descent is now per-mode*, via `–db-prefixes-mix` and
`–db-prefixes-comp` (unset inherits the global `–db-prefixes`),
because the `–p-mix` overlay runs both modes in one process and they want
opposite settings.

Cleanup in the same change: `DbSample::Mixed` and `DbSample::parse`
were removed (nothing constructed or called them since the sampler knobs were
split per mode), and the DB banner now prints the *effective* per-mode
settings rather than the base knobs.

## Part VI: measurement traps

We think this is the most reusable section. Every one of these cost real runs.

**Shipped defaults move under a rebuild.** `–db-prefixes` and
`–curated` both flipped ON between two builds of the same binary. A
launch script copied forward therefore silently changed the measurement. This
is fatal for per-length work: uniform draw gives a per-length conversion rate,
descent gives a quantity conditional on longer lengths failing. *Always
pin every knob a measurement depends on.* Sanity check without reading the
banner: under a uniform draw a 600k-move run gives ≈600k/s_db
attempts per length; descent gives ≈600k at s_db alone.

**The COMP overrides shadow the base knobs.** `s_db_comp` (12)
and `p_convex_comp` (0.9) ship as concrete values, and
`active_s_db`/`active_p_convex` prefer them whenever the live mode
is COMP — despite doc comments promising sentinel fall-through. A run passing
`–db-mode comp –s-db 20` silently gets 12. Detected because the COMP
per-length tables stopped at length 12 while the MIX ones reached 20.

**`–target-size` does not cap DB-driven growth.** With
`–p-db 1.0` the DB move fires every round and the size brake does not
gate it. A MIX sweep intended to stop at 200k ran s_db=3 to 3,726,036
gates and OOM-killed six sibling jobs. Match sizes by *budgeting moves*
from a measured growth calibration instead.

**Hit rate is not a compression proxy.** `nR20_mixed` has the
highest hit rates anywhere (47–77% at lengths 3–6) and almost no removals;
`pre2_100k` has the lowest (14–19%) and removes far more. Hit rate
tracks *g57-form fraction* (what the store can match); removal tracks
*compressible fat*. Reading one for the other ranks these two circuits
exactly backwards.

**Per-splice entropy inverts under aggregation.** Contiguous wins
bits/splice by 2.2× and loses total entropy in all 8 matched pairs,
because it lands 2.4× fewer splices. Always report both.

**Budgeting by moves flatters whichever arm is slower per move.**
Descent "wins" size by 75% at matched moves and loses by 6× at matched
CPU. State the budget axis explicitly.

## Open questions

1. **Matched-work no-descent** (≈5M moves, ≈25 min).
The trajectories predict no-descent overtakes descent on size at equal CPU;
whether its ancestry percolates is unknown and would settle the COMP
configuration.
1. **s_db 9/10 with descent in COMP.** The factorial is monotone
decreasing (12 beats 13) and 92% of convex removals come from lengths ≤9,
so the optimum may be lower than 12.
1. **MIX braiding at held size.** Part III measures expansion, not
transport. Long runs with size pinned would say which s_db braids best.
1. **Layer 2 and s_db.** MIX s_db is a second growth lever the
controller does not model.
1. **Why does curation cost COMP contiguous its length-2 hits?**
11.95% → 1.60% is too large to be noise and has no analogue in
convex. The cascade prefers a curated match when one exists; at length 2 in
contiguous that appears to displace a regular match that would have been taken.
Removals are unaffected, so nothing is lost on size — but the mechanism is
not understood, and it is the third measure on which curated hurts contiguous.
