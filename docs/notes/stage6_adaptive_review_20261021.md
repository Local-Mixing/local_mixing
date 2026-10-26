# Review of the adaptive stage-6 notes (2026-10-21)

Reviewed: `~/Codex/output/docs/stage6_adaptive.md` and the drivers it cites
(`~/Codex/baseline64_fanout2_pipeline_20261019/{stage6.py,refine_blocks.py,driver/src/bin/adaptive_bounds.rs,
driver/src/bin/proxy_intervals.rs,audit_census.py,exposure_metrics.py}` and
`~/Codex/pipeline_exposure_multiseed_20261018/adaptive_largeblocks.py`).

## What the construction does
Variable-length blocks grown by exact ANF composition under term/work caps; the boundary is chosen in the
upper half of the feasible interval by (fewest deeper affine proxies at the cut, fewest internal affine proxies,
largest signed clearance from exposure intervals, fewest direct proxies, latest cut). Failed blocks are split at
proxy-aware cuts (two levels). The repeat driver applies keep-smaller retention and stops at the first
zero-gain pass. Leakage is measured by an every-cut affine census on 6,144 honest inputs, with entry
witnesses re-tested on 8,192 fresh and ~8,000 structured inputs.

## Findings and suggestions, most important first

### 1. The boundary objective has little leverage on the final leakage
Round 1 of the matched baseline experiment selected 3,370 boundaries with zero live internal affine proxies,
yet its output exposed 319 internal births (297 deeper) affinely, and round 2 exposed 379. The proxies are
created by the synthesizer inside blocks, not preserved at boundaries. C-information should therefore enter
at block acceptance, not only at boundary choice: synthesize each block with more than one seed, census the
candidates, and accept a replacement only if it is smaller and exposes no more deeper C births inside the block
than the block it replaces (keep-smaller-and-not-leakier). This is sound because the honest-slice state at every
retained boundary is fixed, so the exposure inside a block depends only on that block's own candidate.
Implemented in the v2 variant (two seeds; candidates evaluated by a full census of each candidate assembly).

### 2. The proxy-aware driver is not monotone
`stage6.py` emits every converged synthesis, including growing ones: in the baseline round 1, window 1 went
from 1,070 to 2,286 gates and was kept. The combined procedure must apply the keep-smaller rule per block and
the zero-gain stop; both are in the as-is implementation here.

### 3. Planning is the slow, sequential part
The planner took 2,153 s on 5.4M gates, single-threaded, and 2,935 of 3,032 blocks stopped at the product-pair
work limit (4M), not at the term cap or the 65,536-gate cap. Blocks are bounded by composition work, with a
mean of ~1,800 gates. Planning segments in parallel (seams at exposure-free cuts, 16 segments) cuts this to
minutes with only the seam cuts constrained. Implemented in v2.

### 4. Boundaries should move between rounds
Round 2 pinned 20 deeper C ids at its boundaries (round 1: 0), because the boundaries were replanned on a
circuit that now carries exposures. Penalizing cuts within a radius of the previous round's boundaries lets
every former boundary fall inside a block at least every other round. Implemented in v2 (radius 64, as a
priority key after the exposure counts).

### 5. Budget by block size
One fixed 60 s budget regardless of block length; 25% of blocks hit the deadline (p90 = 60 s). A size-scaled
budget (60/120/240 s for ≤1k/≤4k/larger blocks) converts deadline hits into converged blocks at modest cost.
Implemented in v2.

### 6. Measurement
Direct/affine ever-exposed births with deeper/all denominators and the cut-exposure fraction are the right
primary numbers; the repeat-ledger R/O counts are a different decoder family and should stay separate.
Both are reported here: the census on every round's input and output, and the raw ledger census on the
stage-5 and final outputs.

## Two implementations
`py6/stage6_run.py` on the server (`~/threemask_cmp`): variant `asis` = frozen policy (one seed, 60 s,
sequential planner, 2 refinement levels, keep-smaller, zero-gain stop); variant `v2` = as-is plus items 1, 3, 4, 5.
Both run on the stage-5 outputs of the stage-2 arms (baseline, three-mask as-is, three-mask modified).
