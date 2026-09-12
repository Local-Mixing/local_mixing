# The affine ridge in blinded-V5 circuits: mechanism, knobs, and the quad-fire fix

*2026-09-06 overnight investigation (RC asked: "figure out how to optimize parameters so
that the ridge will stay low all the way through; would spacing A-gates out among the
B-gates help? more LGIs? longer LGIs?"). Runs: `.242:~/tds/ridge_20260906/` (driver
`~/ridge_exp2.sh`, float proxy `~/float_proxy.sh`), pipelines
`~/tds/bv5_k2_{quad,lin}2eff_20260906/`. Code: quad-fire is the DEFAULT read mode of
`src/preprocessing/blinded_v5.rs` since this commit; `BV5_QUAD_FIRE=0` restores the legacy
linearised read. The plates are `docs/design/blindedv5_quad_pipeline.png`,
`docs/design/blindedv5_lin_vs_quad.png`, `docs/design/blindedv5_linear_phaseA_ridge.png`.*

## 1. What the ridge is

The C-vs-G affine heatmap's diagonal consists of **exact** GF(2) relations (holdout error
0.000): one C bit equals the XOR of about seven gadget wires, the data wire plus the ~6 band
wires of its linearised masks. They are the **masked-read windows** of the construction: a
read completes every open g57 mask with its reverse, so during the window the wire carries
`operand ⊕ ρ` with `ρ` a plain XOR of band wires, which an adversary reading all wires at one
prefix cancels. Counts per stage of the K2 hold-10 arm: unmixed gadget 703, 2-eff snapshot
6,214, phase-A end 6,747, old final 5,276. The plate's depth is the count of such bits per
cell: gadget median-row depth 0.008 (≈2 bits/cell), 2-eff 0.07 (≈18 bits/cell).

## 2. What multiplies it: gate reordering

| experiment (same gadget, 563,842 gates, 481 exact relations) | relations |
|---|---|
| phase A, DB re-encoding OFF (reordering moves only), snapshot at 100k / 200k moves | 503 / 601 |
| the same run's final output after fmix's *final uniform float* (580,571 gates moved, mean displacement ≈1,800) | 4,555 |
| phase A, DB re-encoding ON, snapshots (no float) at 100k / 200k / 300k / 500k / 1M moves | 2,605 / 3,607 / 4,615 / 5,886 / 6,174 |
| phase A with no growth (profile R=1), 200k moves | 4,080 |

A read's undo gate is a write to the operand wire; it commutes with every other write to
that wire and is pinned only by the wire's **next read**. Any float therefore stretches the
affine window to the operand's idle interval. Both reordering channels do it: the DB splice
moves (which float) saturate the count by ≈1 move/gate, and the final uniform float alone
multiplies it 7.6×. Growth is irrelevant; the crossing walk floats too, so skipping phase A's
final float would not save the final. Compression/packing add ≈0.015 of depth by sharpening
the position map and create nothing.

## 3. The parameter levers

Measured on a **float-only proxy** (apply only the final float to a fresh gadget, ~1 min;
its knob ranking matches real phase-A snapshots) and confirmed in phase A at ≈1 move/gate.
Noise floor of the proxy across seeds: 4,966 / 5,449 / 5,693.

| knob | gates | float-only relations | phase A @≈1 mv/gate: relations, depth, median-row depth |
|---|---|---|---|
| baseline (K2, max_open 3, auto rerand) | 564k | ~5,400 | 5,750–6,140, 0.095, 0.066 |
| extra LGIs 200 / 600 / 1000 / 1500 per wire | 659k / 862k / 1.07M / 1.32M | 3,374 / 2,900 / 1,957 / 1,445 | xl200: 5,072 (final 4,894); xl600: 3,740–3,783, 0.078, 0.047 |
| repair slots 875 / 3500 / 8000 / 16000 / 32000 | 586k / 652k / 756k / 945k / 1.12M | 4,245 / 3,062 / 2,097 / 1,824 / 1,421 | — |
| burst F=64 (default 16) | 612k | 3,018 | — |
| max_open 4 / 6 | 828k / 1.50M | 4,084 / 2,846 | — |
| max_open 4 + repair 8000 | 1.03M | 1,647 | — |
| K=4 | 1.36M | 3,032 | (arm: depth 0.071) |
| max_open 2 | 352k | 6,887 | 8,330 (final 7,457), 0.113 |
| fewer straddle slots 100 / 200 / 400 | ~550k | 6,947 / 6,489 / 6,005 | str400: 6,126 |
| more straddle slots 2000 / 4000 | 587k / 629k | 4,235 / 4,268 | — |
| burst F=4 | 553k | 6,450 | — |

Reading: mask **width** is irrelevant to an all-wires affine reader; what counts is the
fraction of prefixes at which the operand wire carries no quadratic term. Every lever that
helps adds quadratic writes on data wires (filler LGIs, repair g57 pairs, wider masks) or
band turnover (bigger bursts) inside the smeared window; the best per-gate efficiency is
about ÷1.9 relations per +1 gate factor (repair 8000: ÷2.6 at +34%). *Spacing A-gates out*
helps only through the filler LGIs it inserts (dilution alone would lengthen idle
intervals); *longer LGIs* (higher max_open) lengthens the fire batch and helps little per
gate; *shorter* (max_open 2) is worse. No parameter removes the diagonal: the best
parameter-only setting keeps rho = 1.00.

## 4. The fix: fire from inside the quadratic masks ("quad-fire")

Never linearise. Each operand is the ANF polynomial `w ⊕ Σ_(x,y) (1 ⊕ y ⊕ xy)` over its
net-open g57 pairs, topped up with fresh quadratic pairs (a single g57 each, undone after the
fire) to `max_open` quadratic terms; the fire `c ^= comp ⊕ lit(a)·lit(b)` is the polynomial
product, one conjunction gate per monomial (degree ≤ 4 at K=2, ~64 monomials versus 49 of
degree 2 plus the linearise/undo bracket). No read ever leaves a wire exactly affine, so
there is no window for reordering to stretch. Straddled (hidden) firing is unchanged.

| | gates | gadget relations | float-only | phase A 300k / 600k / 900k / 1.2M moves (DB moves, snapshots) | phase-A output after 3 mv/gate + float | depth / median-row / rho |
|---|---|---|---|---|---|---|
| linear baseline | 563,842 | 481 | 4,966 | 4,615 / 5,753 / 6,047 / 5,995 | 5,861 (seed 2: 6,362) | 0.097 / 0.070 / 1.00 |
| **quad-fire** | **496,386 (−12%)** | 141 | **88** | **155 / 160 / 144 / 144** | **144** | **0.046 / 0.000 / 0.04** |
| quad-fire, seed 2 | 495,730 | 521 | 185 | 389 / 366 / 317 / 325 | 345 | 0.050 / 0.004 / 0.77† |
| quad-fire + max_open 4 | 720,198 | 245 | 37 | 168 / 142 / 134 / 146 | — | 0.046 / 0.000 / 0.03 |
| quad-fire + extra LGIs 600 | 794,422 | 0 | 30 | — | — |
| quad-fire + repair 8000 | 688,022 | 186 | 595 | — | (repair closes masks around bursts: counter-productive here) |

† every residual relation of both seeds is a single-wire identity in the delivered-output
rows (and, for output bits that settle early, the same bit at earlier C prefixes): the public
I/O fringe, which also sets the depth floor of ≈0.046. No masked-read relation remains.

Both verifications of the generator pass (forward payload and reverse honesty) for every
quad-fire gadget; the gadget is 12% smaller (no linearise/undo gates). Degree-2 adversary
(hmap_affine `--degree 2`, products over a 24-data + 24-band wire slice, the design doc's
lower-bound style): linear gadget 481 → after float 4,978; quad 141 → 102. Caveat: a full
degree-2 adversary with every band-pair product (≈32k regressors) cancels three open
quadratic masks exactly, in either design, between reads or during them; that is the
existing posture (bounded by sample count) and quad-fire does not change it. It removes the
affine channel, which was the open one.

Pipeline implications: gates of width up to 4 (histogram of the n=128 quad gadget: width 0
7,920 / 1 37,737 / 2 215,036 / 3 171,911 / 4 63,782); the frozen store, fmix (k_max 12),
the split and crossing stages and fcompress all take multi-control gates.

## 5. End to end

Two 2-eff pipelines (phase A stopped at the 2-eff snapshot, then split → crossing →
fcompress with packing), quad-fire and a linear control, same recipe:

| stage | linear control: gates, relations, depth / median-row / contrast | quad-fire: gates, relations, depth / median-row / contrast |
|---|---|---|
| gadget | 563,356, 834, 0.049 / 0.008 / 0.25σ | 497,548, 500, 0.048 / 0.000 / 0.35σ |
| 2-eff (phase A halted at 1.67× g_in) | 962,429, 6,032, 0.093 / 0.066 / 0.86σ | 898,416, 443, 0.050 / 0.004 / 0.27σ |
| split | 1,839,248, 5,984, 0.096 / 0.070 / 0.86σ | 1,507,931, 478, 0.049 / 0.004 / 0.28σ |
| crossing | 3,345,382, 4,574, 0.081 / 0.051 / 0.72σ | 2,738,434, 368, 0.049 / 0.004 / 0.27σ |
| final: fcompress + pack (esop1) | 1,577,408 cubes → 337,194 packed (47.2%); 7,179, 0.104 / 0.082 / 0.99σ, rho 1.00 | 1,230,751 cubes → 311,285 packed (44.9%); 400, 0.049 / 0.004 / 0.29σ |

The quad-fire plate (`docs/design/blindedv5_quad_pipeline.png`; side by side with the linear
control in `docs/design/blindedv5_lin_vs_quad.png`) shows no diagonal at any stage: only the public
input corner and the delivered-output band, the same picture the design document shows for
the unmixed gadget. The linear control carries the ridge from its 2-eff on, and its packed
final is the deepest point of its pipeline (7,179 relations, depth 0.104), consistent with
the delivered arms (0.085–0.11).
(The ridge finder's `rho` is not meaningful when there is no ridge: with only fringe cells
it still fits a line, giving 0.2–0.8 on the quad plates; read depth, median-row depth and
contrast.)

## 6. Recommendation

Quad-fire is now the default read mode of the blinded-V5 compute (both generator
binaries; `BV5_QUAD_FIRE=0` = legacy; exhaustive small-n test of both modes in
`blinded_v5.rs`; forward and reverse verify PASSED at n=128):
it is the only change found that removes the diagonal rather than diluting it, and it is
free (smaller gadget). Keep `max_open` 3 (4 gives a marginal proxy gain at +45% gates).
Do not add repair slots with quad-fire. Among parameter-only options for the linear design,
repair slots (8000) and extra filler LGIs (600+) are the efficient ones, each about ÷2 at
+35–55% gates, but neither removes the diagonal. Re-measure the delivered arms' finals with
`plot_hmap_ridge.py` (depth/rho), not mean H or exposed-row counts, which are blind to a
ridge of exact relations at 2–18 bits per cell.
