# ncdb A/B results (2026-10-01, overnight run)

Both arms completed the full mini-sandwich stage-3 profile (2,2,2,2,2 =
3,654,934 moves, same input/seed/binary as runs/canon_ab_20260928) in **27-28
minutes each, running concurrently** — against day-scale wall time for ~1M
moves with the v2 function-keyed curated store.

## Ridge table (stage-2 output rows vs stage-3 output cols, hmap/final_*)

| map      | meanH  | depth  | rho   | perm_z | contrast |
|----------|--------|--------|-------|--------|----------|
| (c)      | 0.4036 | 0.3116 | 1.000 | 14.23  | 2.830    |
| (b')     | 0.4032 | 0.3146 | 1.000 | 14.23  | 2.827    |

Reference points (runs/canon_ab_20260928/hmap): stage-2 vs itself depth
0.398; stage-2 vs the OLD baseline 1M-move snapshot depth 0.341, perm-z
14.2, contrast 2.83.

Reading: the two ncdb arms are statistically indistinguishable from each
other; both dip the diagonal slightly deeper than the old baseline reference
(0.312/0.315 vs 0.341, at 3.65M vs 1M moves). At this mini profile the TDP
diagonal remains detectable (perm-z ~14) in every configuration measured so
far — erasing it needs a longer/steeper profile, which is now cheap to run.

## Counters at matched 700k moves (vs the killed v2-store runs)

|                  | old baseline | old fixed | ncdb (c) | ncdb (b') |
|------------------|-------------|-----------|----------|-----------|
| identity skips   | 311,380     | 592,689   | 232,004  | 223,740   |
| reorder-idsk     | (unarmed)   | 153,820   | 90,716   | 88,292    |
| mean cands/choice| ~327,000    | ~327,000  | 1.5-1.8  | 2.0-2.2   |

## Curated usage (full runs)

- (c): 85,366 curated splices; equal-length substitutions dominate the
  mid-lengths (4->4: 22k, 5->5: 27k, 3->3: 5.3k by the last report block).
- (b'): 281,365 curated splices (3.3x more); 5->5 alone 163k, plus
  broader expansion traffic (2->9/2->10, 3->7/3->8).

## Stores

- frozen_tf_c_20260930: 1,957,427 keys / 3,295,831 pairs, validate PASS,
  filters FP 0.39%.
- frozen_tf_bp_20260930: 16,871,794 keys / 63,061,739 pairs, validate PASS,
  filters FP 0.39%.
- Keys are canonical-circuit hashes (FROZEN_CURATED_KEY=circuit at run
  time); composites for all four modes (a/c/b/b') retained under
  frozen_db/tf_v3_*_20260930.

Caveats: at run time the leakage-repair QC path still probed curated with
function keys — inert here in any case, since --leakage-repair was not armed
and the run logs contain no QC activity. Fixed immediately after the run:
qc_candidates now honors FROZEN_CURATED_KEY like the ordinary path (with an
explicit-mode test entry point, qc_candidates_with_key_mode). The (a) and
(b) composites are built but not frozen. See PROVENANCE.txt for tree state.

## Seed-replicated 1M matrix: five stores x two seeds (hmap/matrix_ridge.txt)

depth (seed0, seed+1000): c 0.3267/0.3334 | bp 0.3348/0.3318 | v1 (tight
friends + spread leeway) 0.3305/0.3360 | v2 (slack friends, exact spread)
0.3374/0.3367 | c1 (uniform leeway 1) 0.3357/0.3329. Old-store baseline
0.3409 (one seed). Seed-to-seed spread within a store is 0.003-0.007 -- the
same size as every between-store difference, so NO ncdb variant separates
from any other at 1M on this profile (the seed-0 "c beats bp" was noise).
The robust result: all ten ncdb runs land below the old baseline (10/10,
mean improvement ~0.007-0.008).

## Expansion-factor sweep on (c) (profile 3,5,5,F,F; hmap/factor_ridge.txt)

| F | end eff | depth  | DB splices/move | regular | curated | final size |
|---|---------|--------|-----------------|---------|---------|------------|
|1.5| 5.39    | 0.3243 | 12.2%           | 10.8%   | 1.35%   | 1,081,423  |
|2.5| 5.58    | 0.2997 | 24.6%           | 22.1%   | 2.44%   | 1,788,467  |
|3  | 5.46    | 0.2964 | 29.5%           | 26.7%   | 2.84%   | 2,165,544  |
|5  | 5.28    | 0.2814 | 40.5%           | 36.9%   | 3.63%   | 3,666,245  |

Ridge depth falls and splice success rises MONOTONICALLY with the expansion
factor; the f=1.5 -> 5 depth drop (0.043) is 6-14x the seed noise. Confirms
the historical "r is the master knob" finding on the ncdb runtime. Per-eff
snapshots for all four arms are under expfactor/c_fF/snapshots/.

Extended (2026-10-01): f=7 depth 0.2748 (splices 46.2%/move), f=10 depth
0.2728 (49.9%/move). Still falling but decelerating hard (5->7: -0.0066,
7->10: -0.0020): asymptote ~0.27. The historical 0.28 floor is nudged past
(0.2728, depthMed 0.2666) by brute dilution, at 10.7x circuit size -- the
dilution envelope saturates rather than breaking the structure, matching
the waypoint-conservation mechanism from the random-circuit panel.

## Long-hold runs to eff 30 (long30/, hmap/long_curves.{png,csv})

Both stores, profile 2,30,30,2,2 (ramp to 2x by eff 2, hold), 41.9M moves,
~7h each, stopped by watcher at eff ~30.1. Snapshots every ~2 effs.

- Depth curves are IDENTICAL within seed noise at every point: (c)
  0.3137 -> 0.2904, (b') 0.3149 -> 0.2917 over eff 2 -> 30. The store
  variants never separate, even with 15x the work.
- Hard deceleration: gains per 10 effs are ~0.012 (2->10), ~0.006 (10->20),
  ~0.003 (20->30) -- an asymptote near 0.290 on the 2x cap. perm-z stays
  ~14.2 throughout: the diagonal fades but never loses detectability.
- DB splice traffic decays toward saturation: (c) 22.7% -> 9.6% of moves,
  (b') 23.4% -> 10.5%. Curated's SHARE of DB traffic grows all run: (b')
  from ~30% to ~49% (5.2% of moves at eff 30); (c) holds ~1.1-2%.

## Synthesis across the three experiments

1. The EXPANSION FACTOR dominates everything else measured: f=5 reaches
   depth 0.2814 by eff 5.3 -- BELOW the eff-30 asymptote of the 2x-cap runs
   (0.290) -- i.e., 28 extra effs of hold cannot buy what 2.5x more size
   headroom buys by eff 5. (0.2814 also lands on the historical "0.28
   floor" from the 2026-08 sidetrip experiments.)
2. Store variant within the ncdb family is statistically irrelevant on this
   profile at every scale tested (1M, seed-replicated; eff 30). Choose on
   other grounds: (c) is the smallest and strictest; (b') carries ~5x the
   curated traffic if curated-move volume is wanted.
3. The ncdb family robustly beats the old v2 curated store (10/10 runs
   below its depth at 1M) and runs ~50x faster wall-clock.

## QC detector-as-metric + repair iteration (qc_iterate/, tool qc_scan)

Detector = the production leakage_repair scanner, reference = the stage-2
output, fixed 384-feature sample (128 value segments + 256 firings), 4096
mixed segments scanned per point. Findings:

- Hot rate ~0.9 +- 0.15% of internal segments (~12-15k of 1.4M), FLAT from
  eff 1.1 to eff 30, across stores and across expansion factors (absolute
  count ~constant; rate dilutes with size). Stage-2 scanned against itself:
  1.56% (its own affine self-redundancy) -- the mixed equilibrium sits at
  ~60% of the reference's self-redundancy.
- Hot segments are UNIFORM in circuit position and spread across wires --
  not boundary-pinned.
- Repair-and-mix iteration (3 rounds from the eff-2 snapshot, budgets 4000
  attempts / 1000 repairs, 2 effs of --db-mode stable mixing per round):
  ZERO repairs in 124 attempts; hot rate oscillates 0.8-1.1% (noise).
  Failure statuses: 77 block_limit (convex closure > 256-gate span cap),
  23 no_db_entry (block function in neither store; suspect blocks containing
  original MPMCT gates), 22 examined_replacements_hot (every equivalent
  re-spelling is itself hot -- value-forced correlation, unrepairable by
  re-spelling in principle).

Conclusion: the hot population is an equilibrium property of function +
architecture (the segment-level shadow of the ridge floor), not removable
by the current QC mechanism; levers in order of promise: raise
--qc-max-span, make the block planner avoid non-g57 gates, and accept the
value-forced class as irreducible.

## Hot-segment provenance + stage-1/2 independence check (indep_check/)

Tracing the persistent hot population to the reference (producer-gate
indices in the repair evidence + qc_scan --dump-hot over the trajectory,
~250 evidence hits across eff 2..30 and f=5):

- Wire classes of leaked features' producers: ~89% masking band
  (wires 128..255, added by gadgetization), ~10% sandwich slice/key
  machinery (0..63), ~1.6% payload (64..127) -- payload appears at its
  base rate (1.1% of gates), no enrichment. The un-erasable leakage is the
  band machine's own values, not C.
- Stage-2 positions tri-modally clustered (deciles 1-2, 4, 8 hot; 3 and
  6-7 near-empty), identical across repair-failure types, stable across
  eff and factor; mixed-side carriers stay position-uniform.

Independence double-check (post-reorg): gen_sandwich_gadget rerun with the
SAME source C (TDP_SOURCE_C) and fresh sandwich/gadget seeds
(indep_check/stage2b.mpmct1, verify PASSED). Against the original stage-2
reference: qc_scan 0 hot of 4096 scanned (vs 64 self, 37 for the eff-30
mix); hmap ridge fully null -- meanH 0.4999, depth 0.0061, depthMed 0.0000,
perm-z 0.0 (not even boundary anchoring: the keyed interfaces differ per
instance). Stages 1-2 independence holds.

Synthesis: the mixed circuit's persistent hot segments are forced by THIS
stage-2 instance's function (which stages 3-6 preserve); they identify the
instance's masking machinery, not the source payload. They are the discrete
witnesses of the same per-instance structure the ridge measures, and the
natural anchors for alignment attacks -- but carry essentially no payload.

## Ridge decomposition: zeroing the hot segments (hmap/eff30_exclusion_*)

Exhaustive hot census of the eff-30 circuit (all 1.41M segments scanned,
255+1024-feature reference sample): 31,348 hot segments (2.22%), dumped via
qc_scan --hot-out. hmap_affine gained --exclude-segments (per-column
regressor masking of live hot carriers). Result: baseline depth 0.2904;
hot-excluded 0.2751 (mean 15.5/256 wires masked per column); equal-sized
RANDOM-wire control 0.2732. The hot exclusion does no more than the random
control: the ridge is a DIFFUSE affine structure across the whole cut, not
a sum of the QC-visible near-copies. Repairing hot segments could never
have lowered the ridge; the ridge floor is not explained by any removable
segment set at this hotness threshold.

## Pure-compression descent from the mixed x5 state (compress_from_f5/)

--db-mode comp from the f=5 output (3.67M gates), 40M moves (7h): removal
collapsed from ~0.39 gates/move to a crawl; the run exhausted its budget at
2,551,794 gates (x3.63) -- a PRACTICAL COMPRESSION FLOOR far above the x2.5
and x2 targets. Waypoint-conservation again: once local slack is spent,
window-scale COMP cannot shorten further.

Ridge along the descent (vs stage-2): x5.2 start 0.2814; x4.5 0.2855;
x4.0 0.2868; x3.63 end 0.2825 -- flat at ~0.284 +- 0.003 throughout. The
direct-mix curve at the final size (~2.55M) interpolates to ~0.293, so the
descended circuit keeps ~75% of the extra depth the x5 expansion bought:
MIXING IS STICKY UNDER COMPRESSION (down to the floor at x3.63). Growing
to x5 and compressing back yields a smaller circuit that mixes like a
bigger one -- a usable schedule trick, bounded by the compression floor.

## 1M-move checkpoint comparison (added after the main run)

The arms ran straight through, so no mid-run snapshot existed; the 1M-move
states were regenerated deterministically (same archived binary, seed,
stores, flags; --moves 1000000) into c_mv1M/ and bp_mv1M/, verified by
matching the mv=1000000 report line of each rerun against the full run's
log. Ridge numbers vs the same stage-2 rows, next to the old-store
reference (runs/canon_ab_20260928, clean-main binary + v2 curated store,
1M-move dump snapshot): see hmap/mv1M_ridge.txt.

## TDP exposure study: stages 3-6, two references (post6_tdp/, hmap/exp_tdp_*, hmap/cv_*), 2026-10-03

Per-segment visibility/exposure (hmap_affine --exposure; see
runs/ncdb_random_20261001/RESULTS.md for the metric) of the real pipeline:
long30/c stage-3 output continued through split (1.41M->2.93M), crossing
(->4.87M), fcompress both orders (compress+pack: 2.45M cubes/774k packed;
pack-first: 2.50M/786k, +2% bigger, otherwise identical). Sampled grids
(stage-2 view: 7,988 of 702,872 segments, ~3k cuts; C view: ALL 1,000
segments of source_c.g57).

STAGE-2 VIEW (which stage-2 born values survive):
            mix 26.5 | split 26.5 | crossing 25.6 | c+p 27.2 | p-f 27.3
  (t*>=2:       2.1  |       2.5  |          2.5  |     2.9  |     2.9)
  Wire classes @s3: data 0-63 = 3.8%, band 128-255 = 63.8% -- the
  survivors are band filler; 96% of data-wire values erased.
  READER LAW RESHAPED: 0r 3.2 / 1r 55.4 / 2r 63.8 / >=3r 100.0 -- the
  random-material ">=2 => 100%" absolute breaks at 2 readers on masked,
  identity-dense material (the law's mechanism is semantic determination;
  embedded masking removes it). Only >=3 is absolute. Hull dominates
  instead (<=9: 21.4%, >9: 99.6%). Stage signatures replicate at
  production scale: split neutral + small t-raise, crossing = the only
  eraser (-0.9), compression refunds (+1.6), PACK ORDER IRRELEVANT.

C VIEW (the actual secret): OUTPUT-ONLY FLOOR FROM STAGE 2 ONWARD --
  6.4% (= exactly C's 64 outputs, all t*=1) at every stage; dense-grid
  verification (88k columns) confirms stage-2 exposes ZERO of C's 936
  interior values. Mixing adds a thin high-weight tail: 11 interior
  values (1.1%) at t* = [2,2,3,9,12,12,12,17,20,22,26] (median ~12) --
  the only high-t exposure population measured anywhere. Postprocessing
  never breaks the floor (6.5-6.6% throughout).

Division of labor, measured end to end: the masking preprocessing owns
C-level hiding (interior values never bare); mixing owns stage-2-
vocabulary digestion (96% of data-wire values); postprocessing is
second-order on both. Open flank unchanged: approximate and nonlinear
predictors untested; sampled grids undercount uniformly.

## Full-resolution center block (blk_s3_*, 2000 gates @350k), 2026-10-03

RC's design: ALL 2,001 born values of stage-2 gates 350,000-352,000, every
cut of the aligned stage-3 region (650k-760k, g_step 1), plus a global
sparse control on the same rows.

- LOCALITY CONFIRMED EXACTLY: 0 values visible globally but not in the
  +-50k window (208 visible only locally = short-lived carriers the
  sparse grid misses). Windowed full-resolution scans are legitimate.
- GRID ARTIFACT AT SCALE: 26.3% (sparse, = the circuit-wide sampled
  baseline) -> 36.7% true visibility (+10.4 pts), all of it in the
  band/mask classes.
- THE READER LAW RE-ASSERTS AT FULL RESOLUTION: 0r 2.7 / 1r 86.7 /
  2r 99.5 / >=3r 100.0. The sampled-grid "law breaks at 2 readers on
  masked material" was grid artifact: masking makes bare appearances
  SHORT-LIVED (between mask toggles), not absent -- dense grids find
  them.
- THE CLOSURE (cross-tab wire class x readers): data wires have 1225 of
  1234 born values with ZERO readers (the 9 read ones: 100% visible);
  band wires carry all the reads (86.6/99.7% visible). Stage 2's
  embedded masking literally implements the reader-law design corollary
  derived on random material: DATA VALUES ARE NEVER READ BARE --
  consumers read band/mask carriers -- putting the whole secret-adjacent
  population in the 0-reader erasable class, which mixing then erases to
  2.8%.
- Sober note: band/mask material is ~91% visible at full resolution (not
  64% as sampled) -- nearly fully carried. Its safety rests on
  independence from C, not on invisibility.

## Proxy census (proxy_census.py, transpose view: classify A' segments), 2026-10-03

A' born value = PROXY of A born value when equal as functions (up to
complement; exact over 2048 samples). Random pair (full) vs TDP block
(A' cuts 650k-760k; window-edge truncation deflates coverage slightly):

                         random (A=random_c)   TDP (A=stage2)
  A' segments = proxies        46.6%               21.7%
  carriers/covered value       1.12                2.43 (peaks at x3, x6
                                                   -- mask/LGI copying)
  proxy order preserved        97.3%               ~chance WITHIN window
                                                   (coarse locality holds;
                                                   A repeats 19% of its
                                                   values -> ambiguity)
  intrinsic tell (span/readers) 155/2 vs 14/0      291/2 vs 7/0

Cross-validation: random A-coverage 84% matches the exposure scan's 85%
visibility from an independent instrument. "Long-lived or read" spots
proxies from A' ALONE on both materials (function-lifetime attack,
per-segment form); on the TDP what it finds is band/mask carriers -- the
data-wire secrets are not in the proxy population to be found.

## Equality-cluster attack (RC's proposal; cluster census on born values), 2026-10-03

Proxies of one A value agree on EVERY input; clusters are pure by
construction (shared value is an A value or it is not). Measured
false-positive floor = non-proxy collision rate:

                           random pair     TDP block
  proxy segs in clusters      19.4%          69.2%
  largest proxy cluster          4            135
  non-proxy collisions         4.2%           3.1%
  largest non-proxy cluster      4              5
  (non-proxy multi-clusters mostly same-wire value-rebirths)

Verdict: high-precision proxy detector -- >=2-cluster membership is ~86%
proxy by mass (93% cross-wire on random); CLUSTER SIZE >=6 IS PURE PROXY
(froth never exceeds 5). Recall 19% (random) / 69% (TDP) of proxy
segments, plus the carrier-LINKAGE graph (same value across positions)
that intrinsic features cannot give. On the TDP the giant clusters are
recurring band/mask values (the x3/x6 copy structure at scale) -- the
data secrets stay outside the proxy population, but the attack hands the
adversary the mask plumbing's connectivity: a sharpened band-persistence
surface. Countermeasure with a built-in metric: carrier dedup /
per-use fresh band draws (drain-set, band-refresh) should collapse max
proxy-cluster size toward the froth's <=5.

UPDATE @STAGE 6 (cluster_rand_s6.txt, cluster_tdp_s6.txt; finals expanded
via 0-move mixer pass -- terminal float may perturb a few same-target
born values, identical treatment both cases): the pack step's canonical
ANF respelling MANUFACTURES froth collisions -- non-proxy collision rate
triples (random 4.2->8.1%, TDP 3.1->9.5%) and TDP froth clusters reach
19 (was 5). The >=6 purity threshold survives on random, BREAKS on TDP
(moves to ~>=20); >=2-membership precision falls to 54% on TDP (was 86%).
Mega-clusters (<=94, band/mask plumbing) remain pure and fully mapped.
Proxy multiplicity also rises (random 1.12->1.52/value). Net: stage 6
accidentally camouflages the attack below the mega-cluster scale --
amplifiable deliberately (bias the packer toward sub-spelling reuse);
carrier dedup remains the fix for the mega-clusters, target "max proxy
cluster under the froth ceiling (~19)".

## ANF-window packing program: waist/interval censuses + joint packer v1 (waist/, 2026-10-04)

Design under test (RC): replace stages 5-6 with a direct partition of the
stage-3 (or stage-5) circuit into contiguous windows (after commuting
reorder), delivering each window as the ANF of its outputs. The delivered
artifact then has NO interior cuts: a reference born value is exposed only
if its carrier span crosses a window BOUNDARY. This is cut-lattice
deletion -- the first operation that can hide READ values (the reader law
presumed gate-granularity cuts) and it sidesteps the closure law (which
constrains rewriting between cuts, not the cut set itself).

Tools (this dir): waist_census.py (full-resolution alive-carrier profile +
optimal-boundary DP), interval_census.py (reorder-aware DAG-interval
ceiling; sound for MPMCT commutation -- same-target writers commute, so a
reader pins against ALL later writers; taint scans use the
any-tainted-reader/writer-ever rule), window_pack.py (drain+burst joint
packer over an epoch-correct scheduler: reader ready iff
writes_done==epoch, writer ready iff earlier-epoch readers done; emits
reordered .mpmct1 + .bounds and re-verifies from scratch). Reference
stage2_tdp.mpmct1; candidates long30/c/stage3_out (s3), post6_tdp/split
(s4), post6_tdp/crossing (s5). 1024-sample complement-normalized
fingerprints; all cut coverage exact (g_step=1 equivalent).

1. WAIST CENSUS (delivered order): proxy spans are long (med 278 cuts s3 /
   343 s4 / 634 s5) and alive(cut) is flat (~200 of 256 wires on s3,
   p10=191). Optimal boundary placement ~= uniform (97.8 vs 97.8%
   exposed @W=10); cap-W windowing without reorder hides only 2-13% of
   carried values. The boundary-placement DP is dead weight.

2. INTERVAL CENSUS (reorder ceiling): a value's bare lifetime compresses
   to its DAG interval I(birth, next same-target writer): med 6 gates on
   s3 (x21 vs spans), 26 on s5 (x13). Per-value hideable ceiling (max
   interval over all the value's spans <= W):
     s3: 61.4 / 69.6 / 76.2 / 82.0 %  @ W=10/20/40/80
     s5: 37.3 / 46.9 / 55.2 / 63.5 %
   Classes on s3 @W=20: band 76%, data 27%, mid 18%.

3. S3-VS-S5 DECIDED: window the STAGE-3 OUTPUT; skip split+crossing+
   fcompress. Crossing's x3.45 inflation WELDS the dependency structure
   (intervals x4); its carrier erasure (105.7k -> 89.8k of 571k distinct
   stage-2 values) does not pay it back: absolute exposure at matched
   W=20 ~= 32k (s3 ceiling) vs 48k (s5 ceiling). Split alone ADDS
   carriers (105.7k -> 107.9k).

4. JOINT PACKER v1 (ground truth on emitted order, function check
   256/256 both runs):
     s3 W=20: 71,811 windows; 21.3% of carried values hidden (exposed
       83,259/105,749 = 78.7%; 14.6% of all distinct stage-2 values);
       band 23.5% hidden, data 5.8%, mid 2.8%. Max output support per
       window med 13 / p90 19 wires -- ANF budget comfortable.
     s3 W=40: 35,652 windows; 24.6% hidden (exposed 75.4%); support med
       22 / p90 30 -- near the ANF wall; W=20-30 is the operating point.
   v1 gap to ceiling (21% vs 70% @W=20) is scheduler losses: interval
   overlaps burned by bursts, drain-phase windows closing at exactly W
   mid-commitment, and multiplicity (a value needs ALL ~2.9 of its spans
   hidden). Artifacts: waist/s3_packed_W{20,40}.mpmct1 + .bounds,
   waist/pack_s3_W*.txt, waist/{s3,s4,s5}.txt, waist/iv_{s3,s5}.txt.

Next: (a) scheduler v2 -- boundary-aware drain (close windows at
protected-gate waists), value-level span grouping (hide all spans of a
value or none), min over later same-target writers when building
intervals; (b) actual sparse-ANF composition per window with a real bit
budget (replaces the support proxy, enables variable-W windows); (c)
equality-cluster attack on the boundary-value sequence as the success
metric for the band surface; (d) re-measure text-level metrics on the
windowed deliverable.

## ANF-window packing: packer v2/v2.1 + ANF composer + boundary cluster attack (waist/, 2026-10-04, part 2)

New tools (this dir): window_pack2.py (v2/v2.1: value-level all-or-
nothing -- intervals built only for winnable values, sibling spans killed
on first death; next-next-writer rescue for intervals > W, sound because
same-target writers commute; atomic bursts with soft 2W cap; v2.1 also
completes intervals incidentally committed by shared-gate bursts),
anf_compose.py (real sparse-ANF composition per window, GF(2) sets of
monomial bitmasks, per-window functional re-verification; NB
set.symmetric_difference_update(generator) DEDUPES collisions instead of
cancelling them -- explicit per-pair toggling required), and
boundary_cluster.py (equality-cluster attack on the boundary-value runs,
proxy vs froth).

Joint hiding progression at W=20 on s3 (ground truth, function check
256/256 every run; ceiling = 69.6% from interval census):
  naive uniform windows      5.4% of carried values hidden
  v1 drain+burst            21.3%
  v2 value-level            32.0%   (exposed 71,895 = 12.6% of all A)
  v2.1 + burst completion   35.1%   (exposed 68,609 = 12.0% of all A)
Band 61.1% exposed (from 97.6% naive); data/mid stay 91-96% exposed
(their carried stragglers are interval-welded, consistent with the
census). Hidden-by-construction matches ground truth EXACTLY every time
(37,118 vs 37,117): reorder-created new spans are negligible.
W=30 adds ~1 pt over W=20 and pushes max output support to med 17/p90 25
-- W=20 confirmed as the operating point (support med 13 / p90 19).

ANF composition (verified, zero functional failures in 351 spot-checked
windows; zero budget overflows at 8192 monomials): med 56 monomials per
window, total 4.04M monomials; deliverable 22.5 MB vs 5.5 MB gate list =
x4.1 inflation. ANF-packing is structure erasure, not compression,
exactly as modeled.

Boundary cluster attack (the band surface): windowing reduces the proxy
population at boundaries (100,005 uniform -> 68,609 v2.1) but does NOT
break the linkage graph: max proxy cluster 1294 -> 1228 (froth ceiling
~50). The mega-clusters are the long-interval band values windowing
cannot swallow. Confirms windowing and carrier-dedup (drain-set /
band-refresh) are complementary countermeasures, not substitutes.

Artifacts: waist/s3_packed21_W20.mpmct1 (+.bounds) is the current best
deliverable prototype; waist/pack21_s3_W20.txt, anf_s3_v21_W20.txt,
bc_s3_v21_W20.txt, bc_s3_{uniform,v1}.txt; v2 runs kept for comparison.

Remaining gap to ceiling (35% vs 70%) = scheduling conflicts among
overlapping intervals. Next levers, diminishing order: (a) conflict-graph
coloring / hull-merging of overlapping winnable intervals before
scheduling; (b) boundary placement inside drain stretches at local
alive-minima; (c) window-level ANF budget feedback (variable W per
window, split-on-overflow); (d) band-refresh integration to shorten band
intervals at the source.

## ANF-window packing part 3: budget-adaptive packer v3, ideal-refresh
## census, carrier-refresh (value-migration) prototype (waist/, 2026-10-04 night)

New tools: window_pack3.py (windows closed by INCREMENTAL SPARSE-ANF
BUDGET + gate sanity cap, contagion completion of co-committed
intervals, death-cause counters), ideal_refresh_census.py (hull of
{birth} u {readers} = ceiling under refresh-after-last-read),
carrier_refresh.py (function-preserving post-hoc value migration:
carriers kept masked between uses via CCX pairs with write-stability
reservations and pair rotation; targets = values the packer cannot
win).

1. PACKER v3 AT G=128 (ANF budget never binds -- real windows use med
   ~440 of 6144 monomials): 58.2% of carried values hidden (exposed
   44,136 = 7.7% of the 571k stage-2 vocabulary); band 97.6% -> 35.8%
   exposed. ANF x8.7 (47.7 MB), zero overflows, zero check failures.
   Death counters: burst_close 26.5k dominates = conflicting interval
   chains, not placement accidents.

2. IDEAL-REFRESH CENSUS: refresh-after-last-read raises the ceiling
   only marginally (82.1% vs ~80% @W=64) -- the unwinnable residue is
   READER-SPREAD, not writer-distance. Data/mid ceilings stay 27-39%
   under last-read refresh. Conclusion: migration must re-mask BETWEEN
   reads (bare only in per-read envelopes); its ceiling is then
   everything-but-outputs.

3. CARRIER-REFRESH PROTOTYPE (between-reads masking, 3 bugs found by
   the per-pair validator and fixed: GF(2) cancellation vs set-dedup
   in ANF mul; overlapping reservation intervals defeating bisect
   stabbing (fix: merged disjoint intervals); same-anchor tie between
   an original write and an earlier-planned inserted toggle (fix: tie
   goes to inserted, horizon-1)). Full s3: 13,971 of 14,484 unwinnable
   values refreshed (513 mega values > 24 long spans skipped
   deliberately -- the band-plumbing backbone, stage-2's job),
   +23.9% gates, function check 256/256.

4. REFRESH + REPACK @G=128: winnability jumps 90.9k -> 98.5k values
   (93% of carried; big-interval disqualified 7.9k -> 1.7k) and
   per-class exposure shifts exactly as the census predicted --
   data 82.6% -> 66.7% exposed, mid 90.0% -> 81.1% (the welded
   reader-spread population is rescued ONLY by this mechanism) -- but
   TOTAL exposure is unchanged (44,182 vs 44,136 = 41.8%): refreshed
   values now carry 3-5 short spans each, value-level all-or-nothing
   multiplies the conflict surface, and sibling deaths explode
   (2.7k -> 22.1k). The scheduler, not the material, is again the
   binding constraint. Max proxy cluster 1,228 -> 887.

Pipeline status: s3 -> [carrier_refresh] -> window_pack3 -> anf_compose
is end-to-end function-verified (256/256 at every step, every ANF
window re-evaluated). Best deliverable prototypes:
waist/s3_packed3b.mpmct1 (no refresh) and waist/s3_ref_packed.mpmct1
(refreshed; same total exposure, much better data/mid profile).

Decisive next lever (death counters + the refresh null): scheduler v4
must treat a VALUE'S SPAN SET as the planning unit (hide-all-or-
abandon-early: reserve all spans' intervals before bursting any;
abandon the value at first unwinnable conflict and release capacity),
plus conflict-graph pre-grouping. The 58%-of-carried plateau is pure
scheduling; the material now supports 93%.

## ANF-window packing part 4: G=256 scale test (waist/*256*, 2026-10-04 night)

Window-size scaling has NOT saturated and the ANF budget is still not
binding (med 738/751 of 12288 monomials at 512-gate windows; one
overflow in 3,961 windows):

  hidden (of carried)  G=64: 45.9% | G=128: 58.2% | G=256: 72.0% (refreshed)
  REFRESHED @G=256: exposed 29,543 = 28.0% of carried = 5.2% of the
    571k stage-2 vocabulary; band 23.1% / data 59.2% / mid 77.0%
    exposed; ANF x11.7 (79.1 MB), all windows verified; max proxy
    cluster 674 (was 1,228 pre-refresh @128).
  RAW s3 @G=256: 71.3% hidden, total ~same as refreshed -- but
    refresh keeps its distinctive data/mid rescue (data 59.2 vs 77.7
    exposed, mid 77.0 vs 86.2), confirming migration is the only
    mechanism for the reader-spread population while total remains
    scheduler-bound.

Night progression (ground truth, % of carried hidden): naive 5.4 ->
v1 21.3 -> v2 32.0 -> v2.1 35.1 -> v3@64 45.9 -> v3@128 58.2 ->
v3@256 72.0. Exposure of the full vocabulary: 17.5% -> 5.2%.

Interpretation of the window-size law: erasure grows ~14 pts per
doubling of G with no ANF cost increase worth noting -- the limit on G
is now security-architectural (boundary count vs the ends, window
resynthesis hardness floor, delivery-size x12), not combinatorial.
Scheduler v4 (value-unit planning) remains the orthogonal lever: at
G=256 the winnable ceiling is 94%, achieved 72%.

## Switch visibility in the windowed deliverable (switch_visibility.py, 2026-10-05)

RC's question: with carrier refresh running AFTER stage 3, how many
wire-hop switches stay visible vs vanish inside ANF windows? Measured
on the shipped G=256 refreshed artifact (deterministic re-derivation,
bounds byte-identical): 168,879 toggle pairs --

  ABSORBED (both toggles in one window): 48.8%. The pair cancels inside
    the composed polynomial; the window ANF is bit-identical to one
    without the switch. Zero trace. Only 27 pairs (0.02%) look
    reorder-collapsed (gap > 512); the packer did not undo protection.
  STRADDLING (toggles in different windows): 51.2% (86,474 pairs,
    181,968 boundary crossings, med 1 / p90 5 boundaries). These are
    the working switches: the masked value crosses boundaries. Hidden
    from the affine/equality adversaries; QUADRATIC-STRIPPABLE in the
    prototype because the pair controls sit verbatim at the same
    boundaries (the stability constraint guarantees co-location).

Design consequence (answers the "hop before stage 3" motive): gate
syntax is erased by windows either way; what stage-3 respelling would
add is destruction of the literal stable-pair mask structure. The
order that keeps targeting correct AND gets that benefit is
  mix (stage 3) -> carrier refresh -> LIGHT re-mix (small move budget,
  respells the toggles into DB vocabulary) -> window pack,
or, terminally, chained-band masks / generation-time migration per the
design doc. Note a light re-mix perturbs spans, so refresh targeting
should leave margin (W_cap slack) for post-polish drift.

## Wire-split sanity check (wire_balance.py, waist/wb_*.txt, 2026-10-05)

RC's check: does the sandwich's top/bottom wire split survive to stage
3.5? Per-wire activity census (touches, target-fraction, positional
moments) + separability AUC per class pair, full circuit and middle-80%.

1. SURPRISE AT STAGE 2: the POSITIONAL phase split is already gone at
   stage-2 output (top-half touch fraction flat 0.25 across 32 bins;
   per-wire mean-pos AUC ~ 0.49) -- preprocessing's interleaving
   erased gate-order phasing. What stage 2 DOES carry, totally, is a
   ROLE/VOLUME split: band = read-mostly (target-frac med 0.17) and
   heavy (12.1k touches), data = write-mostly (0.87) at 7.8k, mid
   nearly INERT (241 touches med). Class membership recoverable at
   AUC 1.000 from first-order stats.
2. STAGE 3 evens it out "a bit but not completely", exactly as
   suspected -- at the role level: residual AUCs 0.55-0.61
   (data-vs-band target-frac 0.610 full / 0.574 mid-80%).
3. STAGE 3.5 (carrier refresh) flattens it FURTHER: the mask toggles
   add writes to read-mostly wires -- middle-80% AUCs drop to
   0.506-0.551 for all role features; the largest residue anywhere is
   touch volume data-vs-mid 0.587 (data wires slightly busier, mid
   slightly lazier -- the stage-2 inert-mid fingerprint diluted).
4. PACKED order: statistically identical (drain is near-positional).
5. Tails: only a whisper (first/last bins 0.28 vs 0.25 at s2; 0.52 at
   s3+). RC's ideal "outside the tails all wires look alike" is NEARLY
   met after 3.5 at first-order: residual 0.51-0.59 AUCs, worst
   data-vs-mid volume.

Caveats: first-order features only (volume/role/position); pair
co-occurrence and arity-profile features untested. Stage-1 raw
sandwich not on disk in this run -- the positional split may exist
there and die in stage-2 preprocessing, not stage 3.

## Wire-split check, part 2: stages 1+2 rerun fresh (waist/s12_check/, 2026-10-05)

RC asked to rerun stages 1-2 and re-measure (the flat stage-2 profile
contradicted the expected sandwich phase split). Fresh
gen_sandwich_gadget run with the mini_sandwich recipe (n=64, m=1000,
s=384, slice_gates=1280, classic, embedded-masking defaults, fresh
seed): statistically identical to the recorded artifacts.

RESOLUTION -- the expected split is the BALANCED variant's layout, not
the classic one. Per src/stages/sandwich/construct.rs: CLASSIC puts C,
D, AND both slice blocks on the LOW half (wires 0..n-1); the high half
is a pure answer register (N-copy targets + slice readout controls
only). TDP runs use classic (stage12.recipe: sandwich_variant=classic).
Measured stage-1 sandwich accordingly: positionally FLAT (low-half
touch fraction ~0.90 in every bin, per-wire pos-std = uniform 0.289)
with a pure VOLUME/ROLE imbalance (low half 115 touches med vs high 13;
target-frac 0.38 vs 0.08). No first-half/second-half gate phasing
exists at stage 1, so none can persist later.

The corrected statement of the earlier findings: what the sandwich +
stage-2 hand to stage 3 is a per-wire VOLUME/ROLE fingerprint
(data/mid/band separable at AUC 1.0; mid near-inert), positionally
uniform. Stage 3 dilutes it to 0.55-0.61, stage 3.5 to ~0.51-0.59;
largest residue = data-vs-mid touch volume 0.59 (inert-mid legacy) --
the band-seeding densification / mid-wire mask spreading is the fix.
(Balanced-variant sandwiches, if ever used, WOULD add the positional
phase structure RC described; worth a one-off census then.)

## Carrier shuffling (EMBEDDED_MASKING_SHUFFLING) at stage 2: census (waist/s12_check/wb_shuf*.txt, 2026-10-05)

RC's two-CNOT wire-switch (b^=w; w^=b = value hop + role relabel) turns
out to be IMPLEMENTED already as the dormant preprocessing_shuffling
feature ("two-CNOT role transfer" in fire boxes), but gated: in-box
only, return-home mandatory per box, and OFF (segments=0) in every TDP
run to date. Turned on at segments=8/32 (fresh n=64 recipe):

- size +10% / +37% (703k -> 774k / 960k gates); verify PASSES.
- stage-2 wire classes remain PERFECTLY separable (touches AUC 0.000/
  1.000, target-frac 0.91-1.0); the sandwich-wire touch share drops
  0.25 -> 0.06 (transfers add band writes without evening roles).
  In-box + per-box return-home preserves data-wire identity BY DESIGN,
  so this mode cannot deliver wire indistinguishability at stage 2.

Conclusion: the primitive is right, the POLICY is what must change --
persistent data-role migration with port restoration deferred to the
circuit END (the accepted tails exception). Plumbing exists
(ShufflingOutput.final_layout carries the role map; band roles already
stay permuted between boxes); the gen_sandwich_gadget validation
currently refuses return_home=false and must be relaxed alongside a
final restoration pass + verify. Open before promoting: stage-3
digestibility under shuffling doses (quad-fire regression precedent)
and whether post-mix residue (the 0.51-0.59 AUCs) actually improves --
needs a stage-3 run on shuffled material.

## Stage-2 wire equalization: persistent migration + read-pressure + rich seed (waist/s12_check/, 2026-10-05)

RC directive: no top/bottom or data/band distinction in the mixed
circuit; data bits keep moving across wires; do it in stage 2. Three
code changes (working tree, tests 332/333 = pre-existing bash env
failure only):
1. PERSISTENT CARRIER MIGRATION: shuffling_return_home=false legalized
   -- data roles hop across fire boxes, ports restored once at the
   stream tail (3-CNOT swaps); guards assert the port contract itself.
2. READ-PRESSURE TRANSFER PARTNERS (punchlist #6 RESOLVED): selection
   by most-reads-since-last-write replaces least-load (which favored
   cold wires). Read runs: baseline 285 max / 789 >=100 ->
   pressure-32: 141 / 0 >=200 / 62 >=100.
3. RICH BAND SEED REVIVED (punchlist #5 RESOLVED): seed_band_rich =
   the pre-reorganization emit_band_fill_nl_pivots_src (recovered from
   5f386381; dropped in the Sept restructure). seed_fill_nl knob,
   default 4: 1,445 seed + 1,426 reseed gates (thin: 256; guard:
   1,280). Preset hashes re-pinned.

Measurements: stage-2 data-vs-band AUC 1.000/0.000 -> 0.51/0.50
(persistent-32 + rich seed), both verifies PASS every variant.
Post-mix A/B at full move budget (regular store): top-bottom touches
AUC 0.129 -> 0.361, data-band 0.249 -> 0.538; target-frac overshoots
band-heavy (0.859 -> 0.240, a tuning knob); digestibility parity
(g57-form end fraction 82.5 vs 82.6%). Residual outlier everywhere:
MID WIRES (classic answer register, zero at seed time, near-inert;
data-vs-mid touches ~0.85-0.88) -- structural, the next equalization
target. Combined rich+persistent post-mix scan: wb_rich32_s3mix.txt.

PUNCHLIST: items 1-6 Done; #7 (width-3 slice guards -> accumulate-and-
inject band-keyed rounds, all <=2 controls) parked for 2026-10-06.

Addendum (combined run landed): rich32+persistent post-mix at full
budget -- top-bottom touches AUC 0.405, data-band 0.549 (flat-ish),
data-mid 0.760 (from 0.954 baseline); target-frac overshoot deepens
(0.202, band now clearly MORE written -- dose knob), and a positional
mid signal appears (mean-pos AUC ~0.87 even middle-80%: mid wires get
written late). Next equalization knobs: transfer/seed dosing + early
mid coverage; pair with punchlist #7 tomorrow.

## Stage-2 redesign x windowing: the compounding result (waist/s12_check/{iv,pack}_*_s3.txt, 2026-10-05)

Matched short-budget stage-3 arms, each censused/packed against its own
stage-2 reference (G=128 packer, function checks 256/256):

                          old stage-2      NEW stage-2 (rich seed +
                          (baseline)       persistent migration +
                                           read-pressure partners)
  med max-span/value          109                 7   cuts
  med max-interval              5                 3   (p90 283 -> 11)
  ceiling @W=20              76.7%              92.4%
  winnable (packer)        106.9k/122.0k     249.0k/262.0k  (95%)
  spans hidden               73.5%              97.5%
  EXPOSED at boundaries     35.9% of carried   7.3% of carried
                           (data 81 / mid 87    (data 7.4 / mid 8.9
                            / band 30%)          / band 6.8%)

The at-source discipline SURVIVES MIXING: median bare residency drops
to 7 cuts post-mix, scheduler conflicts collapse (97.5% of spans
hidden; the conflict graph is nearly empty at interval med 3), and --
the program goal -- residual exposure is CLASS-UNIFORM at ~7-9%
across data/mid/band: no wire population is distinguishable by what
leaks. Program trajectory on exposure-of-carried: naive windows 97.8%
-> best old-material pipeline 28.0% -> 7.3%.

LAYER 3.5 VERDICT (RC's question): the bulk carrier_refresh transform
is OBSOLETE on new-stage-2 material -- its customer base shrinks to
the 12.8k remaining long-span/big-interval values (4.9% of carried,
and the exposure they cause is already inside the 7.3%). What remains
of layer 3.5: (a) an OPTIONAL thin repair pass targeting those
residuals (likely folded into the packer as refresh-on-failure),
(b) the census/verification tooling as standing metrics. The
mask-strength caveat of the prototype disappears with it.

## The final circuit (pipeline v2 assembled, 3.5 removed): format + size ladder (waist/s12_check/{anf,bc}_final_*, 2026-10-06)

Pipeline: stage 1+2 (classic sandwich; embedded masking WITH persistent
migration + read-pressure partners + rich band seed) -> stage 3
(db_mixing, tdp formula budget) -> window-pack (G=128) -> per-window
sparse ANF. No stage 4/5/6, no 3.5.

Size ladder (n=64 recipe, |C|=1,000 gates):
  C                 1,000 g /  64 w   (~2.6 KB)
  sandwich (s1)     2,832 g / 128 w
  stage 2 (new)   969,449 g / 256 w   (~3.8 MB as gates)
  stage 3       2,040,229 g           (~7.8 MB as gates)
  DELIVERABLE: 14,499 ANF windows, 2.64M monomials, 28.3 MB
    (x3.62 over its own gate list; windows med 128 gates / 163
     monomials; zero budget overflows; every spot-checked window
     re-verified; function 256/256)
Expansion attribution: gadgetization x342, mixing x2.1, ANF bytes x3.6;
total ~x2,040 in gates, ~x11,000 in bytes over C.

Window-size trade measured: G=32 -> 57,750 windows, 22.0 MB (x2.81)
but exposure 11.7% of carried; G=128 -> 28.3 MB at 7.3%. G=128
dominates (+29% size buys -4.4 pts exposure); G=128 is the operating
point on new material.

What the adversary receives: a sequence of 14,499 window functions
over 256 wires (footprint list + per-output monomial lists, canonical
order), ports at the ends. Observable state = the 14,499 boundaries:
2.6% of the stage-2 vocabulary carried across any boundary,
CLASS-UNIFORM (data 7.4 / mid 8.9 / band 6.8% of carried); C-view at
the output-only floor; within-window gate text nonexistent.

Residual attack surfaces, ranked: (1) boundary equality-cluster
linkage of the exposed residue -- recurring band values still cluster
(max 1,337 runs, p99 313 vs froth ceiling ~53): the band-persistence
surface remains THE target (dedup discipline / #7-era band hygiene);
(2) the circuit ENDS own the t*-spectrum (sliced-deliverable thread,
unchanged); (3) nonlinear/approximate-predictor flank (untested
everywhere); (4) the 12.8k long-span residuals (optional packer-
integrated refresh if ever needed).

## Heatmaps of the DELIVERED artifact (boundary cuts; hmap --g-cuts added, 2026-10-06)

hmap_affine gained --g-cuts (explicit column positions) so packed
artifacts are scanned at their real cut lattice: the window boundaries.
n=64 final stack (feistel+rich+persistent), 204 boundary columns:

  map            meanH   depth  depthMed  rho   perm_z
  packed vs s2   0.389   0.335   0.332    1.00   14.2   (raw mix: 0.343)
  packed vs C    0.4994  0.068   0.016    0.33    9.3   (raw mix: same)

HONEST READING: windowing does NOT shrink the all-wires affine ridge
against stage-2 (0.343 -> 0.335, same rho/z) -- the state-level
adversary fits affine combinations of ALL 256 boundary wires, and each
surviving boundary is as informative per-cut as a gate cut; windowing
cut the LATTICE x140 (2.0M -> 14.5k cuts) and the verbatim per-value
carriers (97.8% -> 7.1% of carried) but not positional state
recoverability. The two metrics separate exactly as the 2026-10-03
program taught (per-value vs all-wires; relinearization = fake-erasure
caveat). What the ridge tracks is the band/mask vocabulary (data wires
were 3-4% of it); its defense remains INDEPENDENCE FROM C -- and the
C-maps confirm: packed-vs-C at the hidden ceiling (0.4994), identical
to stage-2-vs-C, output-only floor, through the entire new pipeline.

## Stage-2-view per-value census, new pipeline (exp_s2view_*, 2026-10-06)

8,031 sampled stage-2 born values (c-step 120), --exposure (exact-affine
t*, nullspace-coset exact): vs the MIXED circuit (g-step 677, sampled)
and vs the DELIVERED artifact (ALL 14,572 boundaries = its complete
cut lattice):

                      visible   t*=1   t*>=2      data / mid / band
  mixed (sampled)      30.8%    24.8%   6.0     31.1 / 31.9 / 30.4 %
  delivered (exact)    32.2%    24.1%   8.1     32.2 / 34.2 / 31.7 %

Four findings:
1. CLASS-UNIFORM per-value visibility: the historical split (data 3.8%
   / band 63.8%) is GONE. Partly real equalization, partly that wire-
   class labels no longer track semantic roles at all under migration
   -- which was the goal. The SEMANTIC security statement is the
   C-view (output-only floor, unchanged).
2. FIRST REAL WEIGHT-RAISING: t*>=2 is now 8.1 pts (historically
   0.2-1.9%; the spectrum used to be binary). Masked transit makes a
   solid band of values recoverable only as multi-wire combinations
   (weights spread 2..34). Stage-2 redesign evades the closure law
   because it changes what is BORN, not how the interior respells.
3. Instance-vs-distinct reconciliation: boundaries expose only 2.5% of
   DISTINCT stage-2 values (packer ground truth) but 24.1% of born
   SEGMENTS at t*=1 -- the exposed residue is dominated by high-
   recurrence band-vocabulary values (the 1,337-run mega-clusters).
   Carrier dedup/recurrence hygiene is confirmed as the top remaining
   lever on this surface.
4. Windowing leaves the instance-weighted per-value number roughly
   flat (30.8 sampled-mixed vs 32.2 exact-boundary; the sampled grid
   understates per the known 5-10 pt artifact): the boundary lattice
   is x140 smaller but the recurring values cross it anyway.

## Recurrence anatomy of the boundary-exposed residue (recurrence_census.py, 2026-10-06)

Exact (unsampled) two-pass census, new-pipeline n=64 artifacts:
stage-2 = 963,677 born segments / 745,903 distinct values (22.6%
repeats). Boundary-exposed: 18,640 distinct (2.50%), covering 232,784
segments (24.2% -- reproduces the sampled census's 24.1% t*=1 exactly).

The exposed set splits into two unrelated populations:
  - 17,373 SINGLETONS (mult 1): one-off values that happen to cross a
    boundary; 93% of exposed VALUES but only 1.8% of segment mass.
  - ~430 CELEBRITY VALUES (mult >= 65, up to 1,713 births): 2.3% of
    exposed values but 212k segments = 22% of ALL stage-2 segment
    mass. The top ten are each born ~1,700 times and sit on some wire
    at ~13,000 of the 14,572 boundaries (89%).
RECURRENCE IMPLIES EXPOSURE DETERMINISTICALLY: the largest
multiplicity among NON-exposed values is 3 -- every value born >= 4
times crosses a boundary. The band vocabulary's celebrity values are
the whole mega-cluster/linkage surface; dedup targeting those ~430
values (or breaking their literal recurrence with per-use re-masking
in stage 2) is the single highest-leverage remaining intervention.
Open diagnostic: identify WHAT the celebrities are (seed forms, mask
pair combos, N-copy residues) -- degree/support census of the top set.

## n=128 / m=3000 full stack (waist/n128/, 2026-10-06)

Everything replicates at production scale (function check 512/512; all
ANF windows verified, zero overflows):
  C 3,000 g / 128 w -> sandwich 7,920 g -> stage-2 2,768,804 g / 512 w
  (arity census {0:5692, 1:847565, 2:1915547} -- ZERO width-3) ->
  stage-3 5,855,844 g (x2.11, 87 min) -> DELIVERABLE 39,431 windows /
  7.77M monomials / 99.1 MB (x4.41 own gate list).
Packer: winnable 95.0%, values fully hidden 686,598/713,913 (96.2% of
winnable); EXPOSED 8.6% of carried = 3.1% of the 2.1M-distinct
vocabulary (n=64: 7.1%/2.5% -- scale-invariant). Wire AUCs match n=64
(top-bottom 0.402, data-band 0.537). Gates x1,952 over C; bytes
~x12,000.

## Rebirth multiplicity vs migration dose (mult_census.py, 2026-10-06)

RC's hypothesis: lower transfer dose reduces single-value rebirths in
stage-2 output. Same gen seed across all arms:

  dose   born     distinct  repeats  top-mult  vals>=65 (segment cover)
    0   710,288   577,265    18.7%    1,065     397  (17.8%)
    1   719,165   583,054    18.9%    1,089     398  (18.0%)
    2   727,661   588,711    19.1%    1,115     401  (18.2%)
    4   744,656   600,039    19.4%    1,156     404  (18.5%)
   16   843,038   665,521    21.1%    1,387     421  (20.3%)
   32   963,677   745,903    22.6%    1,713     431  (22.0%)

CONFIRMED directionally: the transfer-induced increment (+4.2 pts of
segment coverage, top multiplicity +650) is monotone in dose and fully
recovered by dose 1-2. Mechanism: each transfer REBIRTHS the partner's
band value verbatim on the vacated wire, and read-pressure selection
concentrates picks on hot wires = the celebrity values (refinement
candidate: value-aware partner filtering / rotate-before-swap).
BUT the dose-0 floor is 17.8% coverage / top 1,065: ~80% of the
celebrity mass is INTRINSIC to the embedded-masking vocabulary (fires
re-materializing the same mask combinations ~1,000x). The dedup
intervention must target stage-2's mask-combo generation; dose tuning
alone cannot go below the floor. (Punchlist-#4-intent granularity
dose=1 measured: +1.25% gates, recurrence at floor.)

## Migration dose sweep: the decision table (waist/s12_check/dose*, 2026-10-06)

RC clarified punchlist #4's intent = ONE transfer per ORIGINAL-circuit
write (dose 1); segments floor relaxed 8 -> 1. All arms: same gen seed,
rich seed + read-pressure + feistel guards, full mix budget, G=128 pack
vs own stage-2 reference:

 dose  gate cost  exposed distinct (%vocab)  celeb cover  runs>=100/max
   0      +0%          (not packed)             17.8%        789 / 285
   1    +1.3%        41,363  (7.1%)             18.0%        439 / 237
   2    +2.4%        34,002  (5.8%)             18.2%        335 / 237
   4    +4.9%        26,963  (4.5%)             18.5%        261 / 228
   8    +9.6%        22,629  (3.6%)             ~19.6%       204 / 228
  16     +19%        19,820  (3.0%)             20.3%        146 / 193
  32     +36%        18,640  (2.5%)             22.0%         76 / 193

Shape: exposure improves steeply 1->8 (the intra-box partial sums are
real exposure carriers; one hop per original write leaves them parked),
then flattens 8->32 (-1.1 pt of vocabulary for 4x the cost) while
recurrence worsens linearly (transfer rebirths) and read-runs keep
improving. RECOMMENDED OPERATING POINT: dose 8 (knee on every axis),
revisit after the mask-vocabulary dedup fix, which attacks the
recurrence floor that dose cannot (17.8% at dose 0) and may shift the
exposure curve too.

## Celebrity mechanism found + drift experiment parked (2026-10-06 late)

Root cause of the celebrity values: the dirty-ancilla brackets in
product_fire restore each ancilla's base value verbatim (gap-2 toggle,
examples = identical 2-control gates ~100 gates apart with the
intermediate read 4-11x). 129,042 restore events on the n=64 artifact;
99.98% of intermediates ARE read, so the gates are functional -- only
the exact-restore is the problem. RC's randomize-the-undo intuition
confirmed as the right family; the zero-cost variant (drop the final
restore, let band ancillas drift) was implemented with a five-layer
constraint map (open support, cover, straddle-drawn-early, per-unit
ancilla uniqueness for shuffle commutation, in-box-write partner
exclusion) -- four layers verified by tests, but ONE band-stability
consumer remains unidentified (balanced mode, dirty-band fixtures,
reproducible from a single drifted ancilla at fire 0). Code kept
env-gated OFF (EMBEDDED_MASKING_DRIFT=1 to investigate; DRIFT_ONLY
window bisection + DUMP_GATES + DRIFT_DEBUG hooks in place). Suite
green at defaults (333/334). PUNCHLIST #8 records the handoff and the
SAFE alternative: recurrence-triggered rerand slots (the burst path
already owns the open-mask repair invariant).

## Punchlist #8 SOLVED: the drift fix completed (2026-10-07 overnight)

The residual band-stability consumer was found empirically (fire-window
bisection + leak-ANF: the leak was the product d1*b16*b29, exactly one
band pair; gate-level trace pinned an LGI open at gate 41 closed at
gate 1809 straddling the drift): the MID-FIRE `ensure_min_open!(c)` on
the TARGET -- it runs after the straddle open, after the restore-drop
pass, and opens cycles whose wires the pass could never see. Fix: the
mid-fire min-open cycles are PRE-DRAWN (same avoid set and count)
before the drop pass, excluded from drifting, and emitted at their
original position. Additionally the stability set is now computed LIVE
at drop time (the product_fire-call snapshot was stale: the target's
own bookkeeping runs in between).

Full stability-layer list, each empirically required: live open-mask
support incl. open_lin; fire cover; straddle (drawn early); pre-drawn
mid-fire opens; per-unit ancilla uniqueness (shuffled units stop
commuting when a drifted ancilla is shared); in-box-written partner
exclusion in apply_shuffling. Suite: 333/334 with DRIFT DEFAULT ON
(EMBEDDED_MASKING_DRIFT=0 = legacy restores; hashes re-pinned 4th
time). Debug kit kept env-gated: DRIFT_ONLY window, DRIFT_DEBUG,
LGI_DEBUG, DUMP_GATES.

Effectiveness validation + the stable-pipeline 64/128 runs (gen, mix,
pack, ANF, boundary clusters, wire balance, recurrence census) are in
flight: waist/final64/, waist/final128/.

## Punchlist #8 overnight status: two legs shipped, residual class found
## (2026-10-07, continued)

Burst hot-targeting added (rerand slots target the band wire with the
most bracket borrows since its last write; borrows counted from the
emitted fire units, reset on burst). Same-seed A/B at n=64 production
params: top multiplicity 1,219 -> 453. The drift leg is confirmed
correct but nearly inert at production scale (live open-mask support
covers almost the entire band; the tiny-fixture tests had eligibility,
production does not -- measured: drifted ancillas ~0 at n=64/128).

RESIDUAL CELEBRITY CLASS (new, now the #8 frontier): top values born
~440-453x across 50-77 MIXED data+band wires (the old class lived on
exactly 1 wire each), 83-89% same-wire rebirths at write-gap 2,
nonlinear. Example birth context: CNOT-from-band births on data wires
inside fire blocks adjacent to 4-gate brackets. Suspects: bracket
pair-product deltas sharing bases across wires, or transfer-drain
re-materialisations. Tools: celebrity_anatomy.py (now shows multi-wire
classes), the birth-context dumper in this section's session.

Addendum (residual class IDENTIFIED, analytic + arithmetic fit): the
multi-wire celebrities are LONG-LIVED ROLE VALUES re-materialised by
the migration drain. Every fire box ends by draining the pending
transfer masks off the carrier (carrier ^= partner), restoring the
role's bare value V on whatever physical wire it currently occupies;
a long-lived role (the near-inert answer-register mid roles above all)
is drained bare once per box across ever-different wires: ~22 boxes x
~8 transfer-drains = ~176-450 births over 50-77 wires, matching the
census. It is the SAME restore anti-pattern one level up (brackets
restored ancillas; the drain restores carriers). Fix direction for
RC's morning review: register the transfer mask as a TRACKED mask
(open_lin entry `role ^= partner`) instead of draining it at box end
-- the carrier then crosses boxes dressed and the mask machinery
closes it naturally; requires emitter/shuffling ledger integration
(the drain exists only because transfer masks are untracked).

## MORNING REPORT: stable pipeline v2.1, final 64/128 runs (waist/final64|final128, 2026-10-07)

Pipeline: classic sandwich -> embedded masking with rich band seed
(fill_nl=4), persistent carrier migration (segments=8, tail port
restoration), read-pressure transfer partners, ancilla drift
(default-on; inert at these params but correct), burst hot-targeting,
feistel slice guards (width <= 2 everywhere) -> db_mixing (full
formula budget) -> window-pack G=128 -> per-window sparse ANF.
Function checks green at every stage, both scales; all ANF windows
spot-verified, zero budget overflows; fresh CSPRNG seeds recorded.

SIZE LADDER            n=64 (|C|=1,000)      n=128 (|C|=3,000)
  stage-2                771,029 g / 256w     2,240,757 g / 512w
  stage-3 (x2.14)      1,648,679 g            4,803,546 g (76 min)
  windows                 10,702                 28,986
  deliverable           24.6 MB (x3.88)        93.7 MB (x5.06)

LEAKAGE PARAMETERS     n=64                  n=128
  celebrity top-mult      473                    494   (was ~1,065-1,713)
  exposed @boundaries   16.0% of carried       20.8% of carried
                         3.8% of vocabulary     5.0% of vocabulary
  max proxy cluster       342 (froth 36)         397 (froth 68)
                          (campaign start: ~814-1,337)
  AUC top-bottom/d-band  0.43/0.56              0.38/0.62
  winnable / hidden      90.4% / 92.9%          89.5% / 88.4%

Notes: (1) dose 8 trades exposure (16-21% of carried vs 7-8% at dose
32) for 4x lower migration cost and lower recurrence -- the dose table
(2026-10-06) remains the tuning reference; (2) the boundary-linkage
surface (max proxy cluster) improved ~4x from the #8 campaign and is
now dominated by the migration-drain rebirths of long-lived roles --
fix design pending RC (ledger-tracked transfer masks / just-in-time
drains / emitter-side migration); (3) celebrity top-mult is
scale-stable (~470-500), all remaining mass in the drain class.

## All-pair exposure/visibility + multiplicity, final 64/128
## (waist/final*/exp_*, recur.txt, compress_eval.txt; 2026-10-07)

C-VIEW (every C born value, delivered scanned at its FULL boundary
lattice -- exact, no grid caveat):
  n=64 : vs stage-2 6.4% | vs DELIVERED 6.4% = exactly the 64 outputs,
         all t*=1; ZERO interior values, zero weight escalation.
  n=128: vs stage-2 4.3% | vs DELIVERED 4.3% = the 128 outputs (t*=1)
         + 2 interior values at t*=4 (+<=4 conservative coset skips).
  The output-only floor holds EXACTLY through the entire new pipeline.

STAGE-2 VIEW (sampled born values, full boundary lattice):
  n=64 : 27.3% visible (t*=1 22.1, t*>=2 5.2); data/mid/band
         26.5/27.3/27.7 -- class-uniform.
  n=128: 28.4% visible (t*=1 23.6, t*>=2 4.7); 28.3/27.2/28.7.

MULTIPLICITY (exact two-pass, boundary-exposure linked):
                         n=64                  n=128
  born/distinct      771,029/617,980      2,240,757/1,797,858 (19.8% rep)
  exposed distinct   23,347 (3.78%)       89,072 (4.95%)
   covering segments 173,270 (22.5%)      525,295 (23.4%)
  exposed split      21,723 singletons    85,263 singletons
                     330 vals >=257 mult  966 vals >=257 mult
                      = 127,785 segs       = 393,714 segs
  non-exposed max mult = 4 (both): recurrence >= 5 implies boundary
  exposure deterministically at dose 8. Top mults 473/494 (the
  migration-drain class, see #8 addendum).

Also recorded: deliverable serialization + compressibility (27.5 ->
4.5 MB xz at n=64; 103.1 -> 15.3 MB at n=128; compressed ANF SMALLER
than compressed gate list) and evaluation timings (python 1024-lane:
0.68/1.70 ms-per-input gate-form, 0.85/2.54 ANF-form, bit-exact
match). Incompressibility-vs-rerandomized-control test = open
follow-up.

## Boundary-redraw exploration for output celebrities (2026-10-06, RC request)

Question: can a post-post-processing pass that only re-chooses window boundaries
(gates untouched) hide the exposed celebrities?

**Answer: no — structurally impossible, measured at n=64 (final64).**

1. **Pure redraw** (`waist/redraw_bounds.py`): a value is hidden iff *all* its spans
   sit inside windows. Celebrity (mult>=5) spans in the emitted order: median length
   494 gates, p75 1,087, p95 2,505, max 10,401. Protectable values @W=256: **0/551**;
   even @W=4096 still 452/551 unprotectable (`waist/span_anatomy.py`).
2. **Redraw + pull-forward reorder**: collapsing a span by scheduling the next write
   earlier requires zero intervening reads of the carrier. Only 4.4% of >254-gate
   spans are read-free (median 6 reads, p95 49). The spans are long because the role
   values are genuinely *used*, not idly parked.
3. **Redraw + targeted boundary refresh** (nonlinear toggle/untoggle pair inside the
   read-free gap containing each celebrity-crossing cut, absorbed into adjacent
   windows' ANFs): works functionally with no span-length limit, but costs
   515,109 pairs = **62.5% gate overhead** under current bounds (`waist/gap_cost.py`).
4. **Floor of the whole family** (`waist/liveness.py`): celebrity liveness per cut
   position is p5=101, median=110, p95=117 — near-uniform across the entire line
   (min 0 only at the ends). 232,144 celebrity spans cover every position ~110-deep.
   Any cut set (min ~6.4k cuts @W=256) crosses >=~650k celebrity spans; no redraw
   escapes ~35-60% refresh overhead.

**Structural fact uncovered:** at any instant ~110 wires simultaneously carry one of
the 551 drain-class role values — the band subsystem's steady state is
celebrity-dominated. Post-hoc fixes fight 232k spans; the births number only ~551
values x ~470 drains. The economical fix remains at the source (stage-2 drain
mechanics), cheapest candidate: just-in-time drains (skip the drain at fire-box end;
un-mask the carrier only immediately before its next genuine read), which
re-materializes the bare value only inside short read-adjacent stretches and costs
~0 extra gates.

## JIT drains (punchlist #8 fix candidate): implemented, measured NEGATIVE (waist/jit64, 2026-10-06)

Implementation (uncommitted, default OFF, goldens stable): transfer masks
outlive their fire box and drain at gate-granularity triggers (before any
read of the masked role / any write of the holder); bare data roles with
next touch > gap source gates get an immediate single-CNOT re-mask with the
longest-lived band holder (same-target reuse barred; cross-target allowed
after the partner-starvation fix -- first attempt collapsed transfers
22,656 -> 6,846, fix restored full dose). Knob EMBEDDED_MASKING_JIT=<gap>;
gen verify + reverse verify PASSED at both scales; unit ledger law
added = 3*transfers + 2*remasks + 3*ports.

Full-chain A/B at n=64, gap=24 (jit64 vs final64, same recipe, fresh seeds):

                           final64 (eager)    jit64 (JIT, gap=24)
  stage-2 gates              771,029            1,011,098  (+31%)
  born / distinct          771k / 618k          1,011k / 735k (repeats 19.8 -> 27.4%)
  stage-3 gates             1,648,679            2,135,936
  windows (G=128)             10,702                9,940 (med size 128 -> 256: saturated)
  exposed distinct       23,347 = 16.0%       97,821 = 43.8% of carried  (4x WORSE)
  celebrity class (>=257)  330, top 473         330, top 453  (UNCHANGED)
  new mult 5-64 class      ~700 values          6,386 values / 100k segments
  proxy clusters >= 2        8,079                21,943
  packer disq big-interval   5,778                38,290
  C-view                 6.4% = exact floor    6.6% (floor + 1 interior t*=1, 1 t*=4)

Mechanism (why drain timing CANNOT fix the celebrities): every read of a
role needs the bare value, so the drain-before-read re-bears V at every
read cluster -- the bare-rebirth MULTIPLICITY is pinned by the read count
(~450), not by drain placement. JIT only shortens the bare spans; hiding a
450-span value then requires the packer to contain ALL spans (all-or-
nothing), which fails combinatorially -- and stage-3 scatters the once-
adjacent drain/read pairs, exploding big-interval disqualifications. The
masked-transit vocabulary meanwhile adds +31% born values, and the near-
deterministic holder choice (argmax next-write) re-creates repeated V+Q
values (the new 5-64 class; fixable with key diversity, but secondary).

CONCLUSION: no drain-timing policy can kill the celebrity class; the
identity equation V-at-read must change. Ranked paths: (1) EMITTER-SIDE
dressed carriers (accumulate-and-inject / open_lin integration, RESULTS
2026-10-06 option 3): reads consume the dressed value and the fire algebra
compensates, so the bare V never recurs -- the only mechanism that reduces
multiplicity; (2) value-aware packer (scheduler v4 value-unit planning)
to exploit short spans if a JIT-like mode is kept; (3) remask key
diversity as hygiene. jit128 arm stopped after gen verify (63,360
transfers, 348,760 remasks, +30%, function PASS) -- scale confirmation of
a negative not worth the hours. The committed pipeline is UNCHANGED
(JIT opt-in, default off); final64/final128 dossiers remain the reference.

Addendum (s2-view per-value census, exact boundary lattice): jit64 46.4%
visible (t*=1 37.4, t*>=2 9.0) vs final64 27.3% (22.1 / 5.2) -- the
instance-weighted surface regresses in step with the distinct-value one.

## Birth-site decomposition + compensation pricing (waist/final64/birth_anatomy2.txt, 2026-10-07)

Tool: waist/birth_anatomy.py (classifies every birth of a mult>=5 value by
producing-gate shape; then prices RC's interleave+compensate scheme).

DECOMPOSITION (corrects the earlier "drain class" attribution):
  >=257 : 330 values / 127,785 births -> bracket restores 83%, transfer
          hops 16%, bare-CNOT rebirths (drains/closes) 1%
  65-256: 122 values / 19,375 births  -> 85 / 14 / 1%
  5-64  :  99 values /  1,930 births  -> 85 / 9 / 1%
The celebrity mass is CLASS 1 (dirty-helper do/undo restores), not drains.

PRICING of interleave+compensate (eliminate base intervals, one
compensation gate per read inside an eliminated interval):
  146,708 base intervals, 1,121,528 reads inside = +145.5% if all go;
  keep-3 (push every value to mult<=4): 1,020,205 compensations = +132.3%
  (per-value cost med 2,494 / p90 3,361 / max 5,075).
  Distribution: 0% of intervals read-free, 62% have <=2 reads, median 2,
  p90 24 -- but all-but-3 must go per value, so the mid-tail dominates.
  Widths fine: 94% 3-ctrl, 6% 2-ctrl, ZERO 4-ctrl (no factoring needed).
  ~75% of ALL stage-2 control-reads land inside celebrity base intervals:
  the celebrity vocabulary IS the working mask vocabulary, so per-read
  compensation ~ rewriting the stream.

REDIRECT (cost scaling argument): interleaving pays PER INTERVAL; a
steered RETIREMENT (2-gate nonlinear rewrite of the wire) pays once and
kills the wire's entire FUTURE chain. Borrow births ~124.6k; capping
chains at <=3 rebirths needs ~41k retirements x ~2 gates + ledger
compensation for straddling open/close pairs (~1-2 per retirement, using
RC's compensation trick -- this is what frees retirements from rerand
slots) ~= +15-20% of stage-2. Infrastructure exists: band_uses
borrow counter (hot-targeting) becomes a retire-at-threshold-4 trigger;
live-support tracker identifies ledger-safe points; compensation gates
cover the rest. Also caps transfer-class births (partners are the same
band values). Estimated result: ALL values below mult 5 (the non-exposure
threshold), at carrier-refresh-like cost but applied pre-mixing.

## THRESHOLD RETIREMENT: celebrity class ANNIHILATED at ~zero cost (2026-10-07)

Follow-up to the birth anatomy: per-change compensation prices at x10 (the
straddle census measured avg 9.79 live ledger entries per borrow, med 10,
only 2% zero -- drop-all ~ +125%, retire-at-4 with compensation ~ +63%).
The winning mechanism instead reuses the hot-targeting slot mechanics at
per-wire cadence: EMBEDDED_MASKING_RETIRE=K retires any band wire with
band_uses >= K at the next fire-box boundary via a mini-slot (early-close
the open cycles containing it -- the close is MOVED, not added; keep_covered
+ min_open reopen coverage; one fresh burst_gate write). No compensation
gates at all; ledger handled natively.

n=64 production arms (fresh seeds, gen verify + reverse PASSED all):
  arm        retirements   stage-2 gates     top mult   >=65   >=257
  baseline        --          771,029           473      452    330
  K=32          4,667         677,695 (-12%)     39        0      0
  K=16          9,161         795,766 (+3.2%)    24        0      0
  K=8          17,640       1,025,557 (+33%)     18        0      0
Repeats drop 19.8% -> 9.6-15.5%. The K=32 arm is NET NEGATIVE cost:
capped cycle lifetimes shrink the cover/repair gate load (which scales
with open-mask counts) by more than the retirement writes add.
K=2/K=4 probe running for the sub-5 (zero-exposure-threshold) regime.

## Threshold retirement FULL-CHAIN A/B: celebrity objective met, C-VIEW
## FLOOR BROKEN (waist/ret16, waist/ret32, 2026-10-07)

                        final64         ret16 (K=16)     ret32 (K=32)
  stage-2 gates          771,029          795,766          677,695
  stage-3 gates        1,648,679        1,649,322        1,447,639
  windows (G=128)         10,702            7,811            7,244
  carried values         145,481          165,971          149,101
  winnable                 90.4%            72.2%            75.2%
  big-interval disq        5,778           29,753           21,364
  exposed distinct    23,347=16.0%     78,316=47.2%     59,978=40.2%
  mult >=65 exposed     452 (top 473)        0 (top 25)       0 (top 48)
  max proxy cluster          342               31               50
  clusters >=2 (proxy)     8,079           20,402           16,664
  C-view vs s2          6.4% exact       6.4% exact       6.4% exact
  C-view vs DELIVERED   6.4% = floor     13.1% (+67 int.  9.4% (+30 int.
                                          t*=2..8)          t*=2..6)
  deliverable           27.5 MB          59.9 MB          40.7 MB

VERDICT: the retirement mechanism achieves its target (celebrity/mega-
cluster class annihilated: >=65 class 452 -> 0, max cluster 342 -> 31)
but is NOT shippable as-is:
1. C-VIEW FLOOR BROKEN in the delivered artifact (64 -> 131/94 visible;
   the extra values are EARLY C gates -- 2,3,18,22... -- at t*=2..8,
   dose-dependent). Stage-2 C-view stays exact-floor, so the leak is a
   delivered-surface phenomenon: suspected input-fringe mechanism --
   heavy retirement churn before the band accumulates nonlinear depth
   leaves early data values masked by affinely-strippable combinations.
2. Packer regression (the JIT lesson repeating): retirement material
   welds intervals (big-interval disq x4-5, winnable 90 -> 72-75%,
   windows saturate at 256) -> total exposed distinct 3x worse even as
   the linkage surface collapses.
Next probe: warmup-gated retirement (no retirement in the first ~15% of
the stream) at K=32 -- tests the input-fringe hypothesis directly.
Residual class decomposition (retK16 anatomy): the 5-64 class is 9,589
values / 107k births, still 69% bracket restores + 21% transfers --
retirement caps chain length as designed; no new mechanism appeared.

## Celebrity birth-site anatomy: ATTRIBUTION CORRECTED (waist/celebrity_births.py on final64, 2026-10-07)

Classifying every birth gate of the 330 mult>=257 values in the stage-2
stream (reciprocal-pair detection on adjacent CNOTs):

  bracket restores (multi-control do/undo)  106,852 births  83.6%
  transfer value-hops (2nd CNOT of a pair)   19,894 births  15.6%
  drains / other single CNOTs                 1,039 births   0.8%

Top celebrity: mult 473 = 357 bracket restores + 110 hops + 6 drains,
spread over 87 distinct wires. The "migration-drain rebirth" story was
WRONG (drains are 0.8% of the class); it also explains cleanly why JIT
drains left the class untouched. A celebrity is a long-lived BAND value P:
born by seed/burst write, then re-materialized once per BORROW -- every
dirty-helper bracket that restores the wire re-bears P in place, and every
migration transfer that picks the wire as partner re-bears P on the old
carrier wire (hence the 60-87 wire spread). Life ends at the next real
band write (burst); hot-targeting already capped borrows/life 1,219->473.

Implication for the fix: rotate the band value every <=4 borrows (the
recurrence>=5 => exposure law sets the budget). The borrow counter
(band_uses) already exists from hot-targeting; a 1-gate fresh write per 4
borrows costs ~33k gates (+4%) at n=64 vs JIT's +30%. Bracket borrows are
emitter-side (counter in place); transfer hops are shuffler-side (safe at
any cut where the wire is outside mask_support -- the partner-eligibility
condition already computed). Emitter-side carrier dressing is DEMOTED:
it addresses the 0.8% channel.

## Retirement DOSE SWEEP (regroup after session rewinds, 2026-10-07)

Code state re-established: threshold retirement (EMBEDDED_MASKING_RETIRE=K,
plus straddle census) re-applied from the rewound session's edit log; NEW
knob EMBEDDED_MASKING_RETIRE_WARMUP=<pct> (no retirement during the first
pct% of placed source gates; gated at the call site). Default off; full
suite 336 pass + the known permanent failure. Driver waist/run_ret.sh
(TAG N MC S SLICE K WARM). All gens verify + reverse-verify PASS.

Stage-2 multiplicity (criterion b), n=64:
  arm       K   warm  retirements  stage-2 gates   top   >=65   >=257
  final64   -    -        -          771,029       473    452    330
  ctl64     0    0        -          774,954       476    453    335  (control = baseline)
  rw32_15  32   15%     4,130        701,775       193    144      0
  rw32_30  32   30%     3,380        717,258       366    201     83
  rw48_15  48   15%     2,841        682,135       187    145      0
  rw64_0   64    0      2,463        663,007        59      0      0
  rw64_0b  64    0      2,463        663,686        62      0      0  (replicate)
  rw96_0   96    0      1,613        662,500        81    304      0

Findings so far: (1) WARMUP DEFEATS (b): values born in the unretired
window keep accumulating borrows (15% -> top 193, 30% -> top 366); the
input-fringe guard and celebrity elimination are in direct conflict.
(2) Top multiplicity tracks K; K <= 64 is required to empty the >=65
class. (3) Retirement is net gate-NEGATIVE at every dose (-9 to -14%).
Full chains (stage 3 -> pack -> C-view vs delivered) running for all 7
arms; criterion (a) = C-view vs delivered at the 64-output floor.

### C-view leak mechanism: retirement write coincides with a live mask pair (hypothesis + fix arms)

Leak profile (interior C values visible in the DELIVERED artifact; the 64
outputs are the floor and always visible):
  final64 0 | jit64 2 | ret16 67 (quarters 19/21/20/7) | ret32 30 (14/6/7/3)
  t* = 2..8. UNIFORM across C -> NOT an input-fringe effect; warmup is the
  wrong lever (and it revives celebrities, see sweep table).
Mechanism hypothesis: a retirement writes ONE gate b ^= lit(c1)&lit(c2);
burst_gate_safe only checks clashes for DATA controls, so with two BAND
controls (c1,c2) can be an open mask pair of some data wire w. Then
old_b ^ new_b = that mask product (+ affine), and old_b is a long-lived
value with copies elsewhere -> w's mask strippable by a few-wire affine
combination at a boundary (t* = w + b_new + b_old copy + ...). Regular
slots write ~16 products, diluting the difference. Leak counts scale with
retirement counts (67/30 vs 9,161/4,667 retirements).
Fix knobs (default ON when RETIRE>0): EMBEDDED_MASKING_RETIRE_SAFE redraws
such gates; EMBEDDED_MASKING_RETIRE_BURST=N writes N products.
Measured coincidence rate (redraws): K=32 45, K=64 22, K=64/burst4 94.
Arms rs32 (K=32 safe), rs64 (K=64 safe), rb64 (K=64 safe burst 4) running;
the rw*/ctl64 arms were generated pre-fix (= SAFE off) and serve as A.

### Dose sweep FULL-CHAIN results (n=64, fresh seeds, same build, 2026-10-07)

(The 7 pre-fix chains were killed after their mixers by an edit to the
running driver script -- bash reads scripts by offset; mixer outputs were
intact (header = line count, packer function check 256/256) and the chains
were resumed with waist/run_post.sh. Mixer logs for those 7 are lost.)

  arm       K   warm  safe   interior C leaks   boundary exposure (carried)
  ctl64     -    -     -       0 (EXACT floor)     16.3%
  rs32     32    0     on     32                   40.3%
  rw32_15  32   15     off    40                   36.6%
  rw32_30  32   30     off    21 (14 in Q3)        33.1%
  rw48_15  48   15     off    20                   32.7%
  rw64_0   64    0     off     8                   32.4%
  rw64_0b  64    0     off    14                   32.5%
  rs64     64    0     on      8                   32.2%
  rb64     64    0     on+B4   6                   32.4%
  rw96_0   96    0     off     4                   28.4%
  n=128: rs64_128 (K=64 safe) stage-2 top mult 61, >=65 class 0 (final128:
  494 / 1,202); 16% fewer stage-2 gates; full chain running.

Conclusions: (1) control at exact floor -> leak is caused by retirement
alone. (2) Leaks fall monotonically with K (~32 -> ~9 -> 4) while the
celebrity criterion needs K <= 64: NO dose satisfies both exactly.
(3) Leaks track retirement activity LOCALLY (W=30: they appear where
retirement switches on). (4) SAFE (mask-pair redraw) and BURST=4 have no
effect -> the retirement WRITE is not the culprit; prime suspect is the
early CLOSE of every open cycle containing the retired wire.
(5) Witness traces (waist/trace_witness.py): predictor supports read band
values whose multiplicity sits just under the cap (K=32: 16-32; K=64:
34-52) plus mixer-born values. (6) Any retirement roughly doubles distinct-
value boundary exposure (16% -> 28-40%), even at K=96 -- a third cost axis.
Next: dense C-view vs stage-2 scan to find the exact stage-2 leak cut.

### C-view leak LOCALIZED to stage 3 (mixer), and the variant search (2026-10-07)

Dense C-view scans on rs64 (K=64): stage 2 at EVERY 16th cut over the whole
C region -> 0 values visible; mixed circuit (s3) prefix cuts every 32nd ->
11 interior values visible, 7 of the 8 delivered leaks among them with
IDENTICAL predictor supports. The leak is created by the mixer; packing only
lands boundaries near those cuts. Working model: retirement places a
replacement OPEN and an early CLOSE on the same data wire next to each
other (keep_covered then close), and a local mixer rewrite merges them into
an under-masked intermediate. (Global commutation reordering is NOT the
mechanism: every fire already re-opens its target's masks unpinned and
baseline does not leak.)

Variants (n=64, gen-level; all verify + reverse-verify PASS):
  variant                         retirements  stage-2 gates  top  >=65
  close (rs64, reference)            2,458        661,272      ~60    0
  PIN=2 pinned-coverage deferral       155        765,582      420  503
  PIN=1                                639        708,257      285  618
  NOREPLACE count deferral             393        719,409      350  568
  repair re-key (rp64)               4,004        926,284       54    0
  early opens @75% K (pe64)          3,753        969,820       54    0
  early opens @50% K (pe64h)         3,939      1,014,498       54    0
  K=32: rp32 1,013,783 (top 33); pe32 1,086,254 (top 33)
Deferral rules cannot work: most data wires sit at exactly min_open masks,
so blocking a retirement until coverage allows it blocks nearly all of
them. The mask-preserving variants (repair / early opens) keep celebrity
elimination but cost +20..+41% gates. Full chains (C-view) for rp32, rp64,
pe32, pe64, pe64h and n=128 rs64_128 running.

### Leak multiplicity + merge-only packing repair (K=64 close mode, 2026-10-07)

New diagnostics: hmap_affine EXPOSURE_ALLCUTS=<path> (every cut where a row
is exactly affine); waist/leak_mult.py; waist/merge_bounds.py (delete every
boundary exposing an interior C value -- a cut-lattice SUBSET, so no value of
any kind can become newly visible).

Per-value multiplicity (delivered boundaries exposing each leaked C value):
  rs64 8 values: 1,1,1,1,2,3,3,3 | rb64 6: 1,1,1,1,4,5 | rw64_0 8: 1,1,1,1,1,2,2,3
  rw64_0b 14: mostly 1-3, one value at 9. Each value's exposing boundaries
  form ONE contiguous run. Boundaries removed: 10-25 of ~7,700 per artifact.

Merge-only repair: C-view on the merged lattice = EXACT FLOOR (64/64, all
t*=1, zero interior) on all four arms; total monomials slightly DOWN;
deliverable size unchanged (28.7-29.7 MB); spot checks 0 failures.
BUT the long runs sit in the INPUT FRINGE (first ~5k delivered gates) and
there fusing explodes the ANF: rw64_0b's 1,372-gate fused window = 707,911
monomials (deliverable 29 -> 51 MB); rb64's 1,150-gate window exceeds a
4M-monomial budget. Gate-level fringe scan: these values are exposed at
EVERY cut over long contiguous stretches (rw64_0b C20: 1,125 consecutive
positions; rb64 C7/C11: ~660 each) -> no re-cut exists; a hiding window
must span the whole stretch.
The fringe leaks look partly independent of retirement: low-rate Q1 leaks
also occur with no retirement nearby (jit64 2, final128 2, rw32_30 3 in Q1
where retirement never ran). Interior (non-fringe) leaks: all 1-3
boundaries, spans <= ~400 gates, fused windows <= 768 gates -- fully
repairable by merging.

### What in C makes the fringe values leak (waist/c_structure.py, anf_at_cut.py, trace_domain.py, seed_coincidence.py; 2026-10-07)

1. The long-run fringe leaks are C's FIRST-LAYER gates: target and both
   controls are still raw inputs, so V = a + g57(x,y) = 1 + a + y + x*y, a
   degree-2 function of 3 input bits. No other structural tell (balanced,
   lifetimes/reads typical, no duplicate C functions, no XOR relation to
   other live C values).
2. Every witness decomposes the same way: V = [wire holding g57/product of
   the SAME raw pair x,y, + linear] + [raw copies of a, x, y]. E.g.
   rw64_0b C20 (x44 ^= x27 | ~x57): wires 252 = 1+x27+x27*x57, 221 = x27,
   226 = x57, 138 = 1+x44.
3. Two producers of the product wire:
   (a) band-seed products on the same raw pair (seed_band_rich draws g57
       products from raw data wires): first-layer gates whose pair is also a
       seed pair leak 26% (8/31) vs 6.6% (25/380) unmatched -- 4x enrichment;
   (b) the C gate's own masked fire: a burst of same-target writes into a
       dirty band ancilla computing the product of the masked operands then
       cancelling cross terms; same-target writes commute, so a reordering
       can leave the ancilla holding the bare product (plausible, not yet
       verified gate-by-gate).
   Raw input copies are plentiful because data wires are first masked only
   when first used: in rw64_0b, 31 of 64 data wires are still unmasked
   1,726 gates into the compute.
4. In stage 2 none of this is visible (dense scans: 0); the product wires
   are never standalone stage-2 values -- they appear only after mixer/
   packer reordering of same-target writes.
5. Band wires themselves are NOT sparse after the seed (support 47-52 of
   64 bits), so "thin band seed" is not the cause.
n=128 K=64 (rs64_128): 33 interior leaks / 3,000 (final128: 2), of which 3
first-layer fringe + 30 interior; top mult 61, >=65 class 0; boundary
exposure 38.9% vs 20.8%.
Candidate stage-2 fixes for the fringe: (i) open min_open masks on EVERY
data wire before the first fire (no raw input copies survive into the
compute); (ii) exclude C first-layer control pairs from seed product draws.

### Fringe fixes implemented (2026-10-07; default OFF, goldens unchanged)

Fix 1 EMBEDDED_MASKING_PREMASK=1: open min_open masks on EVERY data wire
(random order) at the start of the compute (site11, on-circuit; drained by
the normal end-of-compute close). Verified: C data wires unwritten in the
first 1,000 compute gates 39-45/64 -> 0/64.
Fix 2 EMBEDDED_MASKING_SEED_AVOID_C1=1: the forward band seed redraws any
nonlinear product whose two sources form a first-layer control pair of the
sandwich (construct.rs computes the pairs; slice-block gates that read the
zero half with positive polarity are treated as dead, i.e. not writes --
the first version missed this and left 1 overlapping pair in 3 of 4 arms;
corrected rule: 0 overlaps on 4/4 fresh seeds). Reseed untouched.
Cost: none measurable (stage 2 at K=64: 660-665k gates vs 661k without).
Celebrities unchanged: K=64 + fixes top mult 58-63, zero >=65 class.
Full chains: fx1 (fix 1), fx2 (fix 2), fx12a/b/c (both), fx12ctl (both,
no retirement). Suite: 336 pass + known failure.

### Fringe fixes + merge repair: FULL RESULT at n=64 (2026-10-07)

All arms K=64 close-mode retirement unless noted; fresh seeds; all gens
forward+reverse verify PASS.
  arm      fixes      leaks (fringe/int)  merged C-view   merged ANF          boundary exp.
  fx1      premask     11 (0 / 11)        64/64 EXACT     0 overflow, 29.0 MB   32.9%
  fx2      seed-avoid  10 (0 / 10)        64/64 EXACT     1 (pre-existing)      33.0%
  fx12a    both         9 (2 / 7)         64/64 EXACT     0, max 6,885 mono     32.9%
  fx12b    both         5 (1 / 4)         64/64 EXACT     0, max 6,347          32.5%
  fx12c    both         9 (1 / 8)         64/64 EXACT     0, max 1,931          32.6%
  fx12ctl  both, no retirement: 0 leaks; top mult 470 (celebrities intact) 16.1%
Both fixes eliminate the LONG fringe runs: remaining first-layer leaks are
exposed at 1-3 boundaries (fx12a C43: 3 in 2 runs; others 1-2), so merge
repair handles them. Boundaries removed: 6-21 of ~7,600; fused windows
<= 768 gates (fx2 one 1,254); deliverable size unchanged (+<=0.2 MB);
total monomials unchanged; spot checks 0 failures.
Celebrities: top mult 58-63, >=65 class 0, stage 2 660-665k gates (-14%).
=> K=64 + PREMASK + SEED_AVOID_C1 + merge repair meets BOTH criteria at
n=64: C-view exact output floor, no celebrities. Remaining cost axis:
distinct stage-2 values exposed at boundaries 16% -> ~33% of carried
(retirement-driven; unaffected by the fringe fixes). n=128 arm fx12_128
running.

## LINKABLE-EXPOSURE optimization (RC 2026-10-07: "values that appear multiple times are thread ends")

Metric (waist/link_census.py): a value is LINKABLE iff visible in >= 2
separate boundary runs. L2 = linkable proxy (stage-2) values; M2 = linkable
mass = total runs in those clusters; froth = mixer-born values.

Baseline vs retirement (G=128):
  arm            L2      M2       mass by stage-2 birth multiplicity
  ctl64 (off)    8,052   133,685  mult-1 31,356 | 100+ 96,586 (celebrities)
  fx12a (K=64)  14,331   111,555  mult-1 27,320 | 20-49 77,107
  rs32 (K=32)   16,616   106,646  mult-1 24,814 | 10-49 78,894
  rw96_0 (K=96) 12,782   113,443  mult-1 27,848 | 50-99 77,275
=> retirement REDISTRIBUTES linkable mass (107-134k at every K) and raises
L2 by 1.6-2x: rebirth count is fixed by the number of bracket borrows; K
only picks which values carry them. ~84% of linkable VALUES are stage-2
singletons re-created by mixing/restores (mass ~25-31k).

Lever 1, window size (repack, post-hoc):
  arm    G     L2      M2       froth L2  deliverable
  ctl64  128   8,052  133,685   2,717    24.9 MB
  ctl64  256   7,240  103,362   1,184    26.8 MB
  ctl64  512   6,200   75,463     464    28.1 MB (2 windows > 65k monomials)
  fx12a  128  14,331  111,555   2,319    29.2 MB
  fx12a  256  11,883   82,293     927    30.6 MB
  fx12a  512   9,059   58,039     291    31.6 MB (2 windows > 65k)
~-25% M2 and ~-60% froth per doubling for +4-8% size (total monomials FALL).
At G=512 the residual mass is the recurring band class (71% / 71%).

Lever 2, reserved ancilla pool (EMBEDDED_MASKING_ANC_RESERVE=R, new):
R band wires are excluded from every mask cycle and fire-cover pair and
are the ONLY fire ancillas, so every bracket restore on them is droppable
(existing drift machinery) -- borrows evolve the ancilla instead of
re-bearing it. Default 0 = unchanged. Gen probes (no retirement, fixes on):
  R=0 (fx12ctl): >=257 class 332, >=65 cover 19.1%, top 470, 776k gates
  R=16:          >=257 6,   >=65 cover 13.0%, top 335, 877k
  R=32:          >=257 11,  >=65 cover  2.7%, top 276, 862k (+11%)
Lever 3, migration dose: SEG=1 cuts transfers 8x (22,656 -> 2,832) and
stage 2 by ~7%. Full chains + scoring (G128/256/512) running for sg0, sg1,
sg1k64, ar32, ar32s1, ar32s1k64, ar48s1, fx12ctl.

Window lever saturates: packer at G=1024 -> ctl64 L2 4,916 / M2 53,253;
fx12a L2 6,379 / M2 39,793 (same as greedy link-breaking deletions on
G=512, which delete ~half the boundaries); 7 windows exceed 65k monomials
(max 386k). Residual at G=1024: recurring band class 72-77% of mass;
stage-2 singletons 4,008 / 2,912 values. Singleton anatomy (ctl64 G1024):
85% same-wire excursions (value present, perturbed for ONE boundary in 81%
of cases, back again) -> do/undo pairs straddling a boundary, created by
mixing/packer reordering (not stage-2 restores). A packer that keeps such
pairs inside one window would remove most of this class (packer redesign).

### n=128 confirmation of K=64 + fringe fixes + merge repair (fx12_128)
Stage 2 1,884,665 gates; 26 interior C leaks (33 without fixes); merge
removed 70 of 19,810 boundaries (max fused window 1,536 gates) -> C-view
EXACT floor 128/128; ANF 0 overflow at 16,384; 117.2 MB; spot checks OK.

### Lever results at n=64, G=512 (scored with waist/score_arm.sh)
  arm        levers                    L2      M2      froth   MB
  fx12ctl    dose8, K off              6,276   76,368    445   24.5
  sg0        no migration              5,834   60,005    491   44.6
  sg1        dose1                     6,287   64,100    481   36.9
  sg1k64     dose1, K=64               8,355   41,421    228   40.7
  ar32       pool32, dose8            10,523   71,424  1,212   28.9
  ar32s1     pool32, dose1             8,737   47,444  2,203   42.0
  ar32s1k64  pool32, dose1, K=64       8,412   45,214  1,843   41.0
  ar48s1     pool48, dose1             9,624   44,775  1,734   42.6
Findings: (1) migration dose 8 keeps the ANF compact (dose 0/1: +25%
monomials of higher degree, +50% MB) -> keep dose 8. (2) The ancilla pool
removes the celebrities (100+ class 420 -> 0) but leaves ~4,800 values
re-born 5-49x on a single wire (restores that stay non-droppable because
pool wires are shared within a fire), raising L2 and 4-5x froth -> dead
end as built. (3) Retirement and the pool both convert a few heavy
threads into many light ones (L2 up, M2 down). Candidates: A = dose 8,
K off, fixes, G=512 (fewest threads); B = dose 8, K=64, fixes, G=512
(no celebrities, lower mass). Over-budget windows (2-3 per G=512 packing)
are split by waist/anf_split.py; C-view leaks merged by merge_bounds2.py;
waist/finalize.sh iterates both to a fixed point.

### FINAL deliverables, n=64 (waist/finalize*.sh: C-view merge repair + ANF split to a fixed point)
All: C-view EXACT output floor (64/64), 0 windows over 65,536 monomials,
spot checks OK. Linkable census on the final bounds:
  candidate                        L2      M2      froth   size
  old baseline final64 (G=128)     8,079  ~134k    2,804   24.9 MB
  A: K off, dose 8, fixes, G=256   7,349  103,976  1,248   25.0 MB
  A: K off, dose 8, fixes, G=512   6,289   76,425    450   25.0 MB  <- dominates baseline at equal size
  B: K=64, dose 8, fixes, G=512    9,068   58,053    292   32.8 MB
(B at G=256 oscillates between merge and split -- the midpoint split lands
on an exposing cut; needs a split-point search. Not pursued: B@512 converges.)
n=128 B (fx12_128): L2 43,507 -> 37,723 -> 30,609 and froth 11,649 -> 5,738
-> 2,189 at G=128/256/512, but size 116 -> 134 -> 150 MB (+29% at G=512;
+8% at n=64). A at n=128 (a128) pending.

### Exposure-aware ANF split (waist/anf_split2.py, finalize3.sh)
Midpoint splitting can re-create the exposing cut that the merge just
removed (B@256 n=64 and B@512 n=128 oscillated at floor+1). anf_split2
scans 32 candidate cuts of each over-budget window with one C-view run
and splits at an exposure-free cut nearest the middle. Both cases now
converge in 2 iterations:
  B@256 n=64  (fx12a):    C-view 64/64, 0 overflow, 31.0 MB, L2 11,874, M2 82,183, froth 923
  B@512 n=128 (fx12_128): C-view 128/128, 0 overflow, 151.1 MB, L2 30,552, M2 201,280, froth 2,175
(Ops: never run two finalizers in one arm directory concurrently -- they
share fin.* scratch files.)

n=128 scored (before finalization), G=512:
  A  (a128):       L2 19,350  M2 264,353  froth 2,812   115.4 MB (G=128: 23,496 / 436,680 / 11,400 / 94.0 MB)
  B  (fx12_128):   L2 30,609  M2 201,761  froth 2,189   150.3 MB
  pool (ar64s1_128): L2 26,024  M2 143,611  froth 8,060   174.9 MB
Window growth costs more size at n=128 (A: +23% at G=512) than at n=64
(A: +0%).

## SUMMARY for RC: minimizing linkable exposure at minimal size (2026-10-07)

FINAL deliverables (every row: C-view EXACT output floor, 0 windows over
65,536 monomials, spot checks OK; finalize3.sh = merge repair + exposure-
aware split). L2 = stage-2 values visible in >= 2 boundary runs (thread
ends), M2 = their total sightings, froth = mixer-born linkable values.

n=64                           L2      M2       froth   size     celebrities (100+ runs)
  old baseline (G=128)         8,079  ~134k     2,804   24.9 MB  ~390
  A@256                        7,349  103,976   1,248   25.0 MB   366
  A@512  <- RECOMMENDED        6,289   76,425     450   25.0 MB   323
  B@256                       11,874   82,183     923   31.0 MB     0
  B@512                        9,068   58,053     292   32.8 MB     0
n=128
  baseline-equivalent (A@128) 23,496  436,680  11,400   94.0 MB  ~1,000
  A@256                       21,683  349,024   6,135  104.4 MB  1,070
  A@512  <- RECOMMENDED       19,351  264,365   2,813  112.7 MB   994
  B@512                       30,552  201,280   2,175  151.1 MB     0
A = no retirement, migration dose 8, PREMASK + SEED_AVOID_C1, G=512 pack.
B = A + K=64 retirement.
Trade-off: A minimizes the number of threads and the size; B removes the
celebrities and lowers sightings by 24-27%, at +31-34% size and +44-58%
more linkable values. Neither retirement dose nor the ancilla pool nor
migration dose 1 improves on A in thread count; dose 1/0 costs +50% MB.
Main residual threat in A: ~320 (n=64) / ~1,000 (n=128) celebrity values,
each seen 100+ times (72% of M2), and stage-2 singletons re-appearing after
one-boundary excursions (85% of linkable singletons; packer-level fix).

## RESTORE-FREE (CHAINED) BRACKETS (EMBEDDED_MASKING_CHAIN=k, 2026-10-07)

Algebra: the first three gates of a dirty-ancilla bracket add P*x*y to the
target WHATEVER value the ancilla holds (Q_a*Q_b Gray walk: visits all four
(h1,h2) combinations -> adds A*B for any start values). So a fire's
brackets can share k ancillas with no intermediate restore; each ancilla
accumulates its terms and all restores (the brackets' final gates) run once
after the last unit, in emission order -> the ancilla passes through fresh
partial sums and its start value is re-born once per FIRE (not per bracket).
Implementation (embedded_masking.rs): after cover/straddle/mid-fire opens
are drawn, choose k ancillas from band wires no other gate of the fire
reads or writes (units, cover, straddle, mid-fire opens, cover-pair undo);
re-target every bracket (remap_gate), defer its final gate; units and
deferred restores are shuffled together (same draw sequence as
shuffle_units -> default path byte-identical, goldens unchanged); restores
emitted after the second half, before the cover closes. Chained brackets
never drift. Default 0 = off.
Gen probes (config A: no retirement, dose 8, fixes), all verify+reverse PASS:
                 >=257   >=65 cover   top   repeats   gates
  A (fx12ctl)     332     19.1%       470   19.9%     776k
  chain k=2         0      2.4%       153    7.3%     783k (+0.9%)
  chain k=4         0      3.7%       162    7.5%     779k
  chain k=8         0      5.4%       190    8.5%     786k
Full chains ch2_64, ch4_64, ch2_128 running (auto-scored).

### C-value / stage-2 affine relations: NONE beyond the trivial (all variants, 2026-10-07)

New hmap_affine mode SPAN_INTERSECT=<tsv> (with --exposure): at each stage-2
cut, the EXCESS dimension dim(U∩W) - dim(I∩W), U = span{C born values,
inputs, 1}, I = span{inputs, 1}, W = span{stage-2 wire values at the cut,
1}: the number of independent affine combinations of C values that equal
affine combinations of stage-2 wires beyond what the inputs explain.
SPAN_OUTPUTS_TRIVIAL=1 also counts C's outputs as trivial. Positive
control (C vs itself) large at every cut, 0 at the input state.
29 stage-2 variants (every lever tested today, both scales; n=64 every
16th cut ~41-60k cuts, n=128 every 128th ~15-17k cuts): excess = 0 at EVERY
interior cut; nonzero only in the final ~3.3k / ~6.6k gates, ramping to
exactly 64 / 128 as masks close and outputs emerge. Tail re-scan at every
cut with outputs counted trivial (A, B, chain, chain+K64 at n=64; A, B,
chain at n=128): 0 at all 4,001 / 8,001 cuts. => the only relations are the
outputs (and input-affine combinations of C values). dim U ~1,050 (n=64) /
~3,100 (n=128) << 6,144 samples, so rank results are not sampling-limited.

Chained brackets, full chains (n=64): linkable L2 / M2 / froth at G=512:
ch2 5,590 / 61,532 / 1,212; ch4 5,485 / 63,618 / 1,105 vs A 6,276 / 76,368 /
445 -- BUT ANF size 61.8 / 60.0 MB vs 24.5 MB (monomials GROW with window
size, 2.57M -> 3.43M; p99 window 23k vs 1.4k monomials): a displaced
ancilla lives across the whole fire. Chain+K=64 is INERT (0 retirements
with per-fire counting); dose-equivalent is K=4 (2,304 retirements;
stage 2 658k, -15%). Bounded chains EMBEDDED_MASKING_CHAIN_LEN=L (flush an
ancilla's restores after every L brackets) added to trade rebirths vs ANF.

### Rotating chain segments (EMBEDDED_MASKING_CHAIN_ROTATE=1 + CHAIN_LEN=L)
Every L brackets form ONE atomic unit on the next pair of fire-safe
ancillas, closed by its own restores (displacement <= ~4L gates, never
spans units after the shuffle); each band value serves at most once per
segment, rotating over all free band wires. Gen probes (config A), verify +
reverse PASS: rot4 >=257 0, >=65 cover 8.3%, top 246; rot8 0 / 5.6% / 190;
786k / 784k gates (A 776k). (Bounded cl4 7.0% / 227; cl8 4.8% / 184.)
Stage-2 packing proxy (no mixing, G=512, MB): A 10.6, full chain 12.4,
cl4 9.0, cl8 8.6, rot4 8.9, rot8 7.0 -> the full-chain blow-up (x2.5) does
NOT appear before mixing: it is a mixer x displaced-ancilla interaction,
so screening needs stage 3. Full chains: cl4_64, cl8_64, rot4_64, rot8_64,
rot8_128, cl8_128 running.

### Bounded chains, full chains (n=64): size blow-up largely FIXED
  arm        G     L2     M2      froth   MB    100+ anchors  distinct exposed
  A          512  6,276  76,368    445   24.5      323          15,006
  cl4 (L=4)  128  7,026  91,322  4,742   20.6
  cl4        256  6,432  75,915  2,793   23.1
  cl4        512  5,609  61,234  1,295   28.9      198          11,386
  cl8 (L=8)  512  5,498  61,017  1,281   28.4      199          11,527
  full chain 512  5,590  61,532  1,212   61.8
vs A@512: linkable proxy -11/-12%, sightings -20%, distinct exposed -24%,
100+ anchors -39% (remaining = transfer-hop class), for +16-18% size and
~+840 linkable froth (almost all clusters of size 2). Total linkable
(proxy+froth) is a wash (6.7-6.9k). Rotating variants and n=128 pending.

### Bounded vs rotating chains at n=128 (G=512), vs A
  arm        exposed  L2      M2       froth   100+   MB     stage-2 gates
  A (a128)   53,296   19,350  264,353  2,812   994   115.4   2,243,575
  cl8_128    33,988   16,723  198,267  7,144   755   115.6   2,229,229
  rot8_128   34,119   18,719  247,191  7,711   970   170.5   2,234,236
Bounded L=8 at n=128: exposed -36%, L2 -14%, M2 -25%, anchors -24% at the
SAME size (n=64: +14%); linkable froth 2.5x (+4.3k, nearly all size-2);
total linkable (proxy + froth) +8%. Rotating: no anchor gain, +48% size ->
dropped. Chaining's full-chain ANF blow-up is avoided by bounding (L=8).

## COMMUTATION-AWARE REPAIR (waist/order_repair.py, repair.sh; 2026-10-07)

On the final packed order (boundaries = markers between surviving gates):
(1) cancel identical gate pairs that commute adjacent (iterated to a fixed
point); (2) for each interior C leak, slide the relation's completer
forward / breaker backward by commutation (witness support -> segment
endpoints); (3) merge/exposure-aware split fallback. Functional
equivalence vs the unrepaired circuit on 1,024 random full states: PASS
for all four (twice each).
Cancellation removes ~7% of delivered gates (64: 110-115k of 1.6M; 128:
323-334k of 4.6-4.8M). The 128-wire bounded-chain fringe leak that needed
a ~2,600-gate window (C64: five wires XOR to V; completer = write to wire
262 at 8,507, breaker at 11,083, the only readers a commuting identical
pair) is resolved by slides alone.
Finals (all: exact C-view floor, 0 windows over 65,536, spot checks OK):
  n=64        exposed  L2     M2      froth  100+  MB
  A rep       15,410  6,070  73,763    396   319   22.9  (was 25.0)
  L=8 rep     11,769  5,299  58,514  1,124   194   27.0  (was 27.9)
  n=128
  A rep       54,494 18,573 252,069  2,318   987  103.0  (was 112.7)
  L=8 rep     34,544 16,091 187,710  5,765   706  102.7  (was 114.7)
Repair vs merge-only finalize: L2/M2 -3..-5%, froth -12..-19%, size
-3..-10%. Bounded L=8 vs A (both repaired): L2 -13%/-13%, M2 -21%/-26%,
exposed -24%/-37%, anchors -39%/-28%; froth +728/+3,447 (almost all
size 2); size +18% at n=64, equal at n=128.
Hypothesis corrected: the one-boundary-excursion singletons are mostly NOT
identical-gate pairs (L2 only -4%); they are restored by different gates.
