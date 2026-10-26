# Review of the three-mask stage-2 compute prototype (2026-10-21)

Reviewed: `~/Codex/output/pdf/stage2_three_mask_20261020/stage2_three_mask_design.md` and the generator it cites
(`~/Codex/stage2_compute_pilot_20261016/experiments/exposure_v3/repeated_decoder_followup/generator/src/{lib,fire,band,migration,main}.rs`, tests).

## Verification done

- All 21 generator tests pass (`cargo test --release`).
- A scratchpad rebuild of the present source reproduces the seed-1 fixture **bit for bit**
  (`pilot_stage2.mpmct` sha256 `2b33903a…5490`, 501,725 gates, 112 rotations). The document's
  "no bit-for-bit rebuild claimed" caveat can be dropped for seed 1.
- Algebra re-derived by hand: dirty chain B·V·B (2k−3 Toffolis, V palindromic so the reverse stream is the same map),
  differential refresh abc+def = (a+d)bc + d(b+e)c + de(c+f), migration transfer/drain bookkeeping. No correctness issues found.

## Findings, most important first

### 1. Helper wires are fixed, and they absorb most of the band writes
`Band::borrow_helpers()` returns the first four wires of the lowest-numbered growing bank. At saturation
(4 serving / 3 retiring / 1 growing, which is where the run sits 95% of the time) there is exactly one growing bank,
so the same four physical wires receive every dirty-chain write for ~25 fires.

Measured on the seed-1 logical compute: the first four wires of every bank take 79–84% of all band writes;
in a 2,000-gate window four wires receive ~1,000 of the ~2,000 writes. That is a structural beacon any window
observer can use to locate fire boxes and helper garbage.

Fix (patched in scratchpad, env `PILOT_RANDOM_HELPERS`): shuffle all growing wires once per fire and stride
through the pool per product unit (and per delta). Cost 0 gates (501,636 vs 501,725); top-4 write share 14.8% → 4.5%;
in the same window the top four writers drop to ~350 writes. Recommend adopting and re-running the exposure portfolio.

### 2. Discarded helper restoration is most of the size saving, and it is data feedback
| variant (seed 1, n=64) | stage-2 compute gates | vs baseline |
| --- | --- | --- |
| baseline | 769,936 | — |
| three-mask, helpers restored (4k−8 chain) | 707,347 | −8.1% |
| three-mask, helpers discarded (2k−3 chain, the package) | 501,725 | −34.8% |

So ~77% of the saving comes from `--discard-fire-helpers`, not from the mask schedule.
The discarded garbage is data-dependent: `XGate::conj` sorts controls, so for every carrier×mask product
`[d_u, p, q, r]` the chain writes `h0 ^= d_u·p`, `h1 ^= d_u·p·q`. These wires sit in a growing bank that is later
promoted and used as mask coordinates. The document lists "data feedback" among the disabled alternatives;
it should say instead that discarding helpers is an implicit, structured feedback channel, and the exposure
comparison needs a restored-vs-dirty ablation before any exposure difference is attributed to the third mask.

### 3. Rotation is drain-limited; serving coordinates are constants for ~16k gates
From the seed-1 run: 2,688 of 2,832 fires stall on the retiring cap; serving-bank life is 103 fires on average
(min 64, max 153) ≈ 16,300 gates; retirement waits 30–124 fires. Serving banks are never written during service
(growth targets only growing banks; the delta's temporary XORs are restored; the migration pass permutes which
wire holds a band role but not the role's value). So inside any resynthesis window of a few thousand gates every
mask coordinate is an unchanging quantity, and "nonlinear band evolution" is bank swapping at a ~25-fire cadence.

Levers measured:
| retire budget | gates | rotations | mean wait (fires) |
| --- | --- | --- | --- |
| 2 (package) | 501,725 | 112 | 76 |
| 4 | 587,228 (+17%) | 198 | 42 |
| 8 | 757,319 (+51%) | 372 | 22 |

Replacing the target mask that references a retiring bank (instead of always the oldest) gives only +5 rotations:
oldest ≈ retiring already. Faster turnover is roughly linear in gates at 15 gates per replacement; the document
should state the serving-life number because it is what "refresh" means to a window observer.

### 4. The cover mask is not label-unique
The cover is drawn with `band.sample`, not `sample_persistent`, so it can coincide with another role's live cubic
label for the duration of the fire box (d_t and d_w then share a monomial). The uniqueness rejection already
exists; reuse it for the cover. Cost: nothing.

### 5. A complete fire inside a narrow window factors into x_u·x_v
The 16 product units of a fire sum to exactly (d_u+S_u)(d_v+S_v) = x_u·x_v on the target. A fire is ~158 gates,
so a 125–250-gate window often contains one whole fire box and its net function is the product of two
decoded values; the cheapest reconstruction materializes x_u and x_v. This is a candidate mechanism for the
narrow-portfolio reversal (19–24 deeper values in three-mask vs 0–2 in baseline). Two things would test it:
(a) a per-arm deeper/boundary split in `completed_narrow64_comparison.json` (the audits only give total supports per arm);
(b) interleaving product units of independent fires. The scheduler already builds the dependency DAG; the migration
pass would need overlapping boxes. Zero gates, and it pushes any complete fire outside narrow windows.

### 6. Smaller points
- `emit_cubic_delta` always uses `helpers[0]`; the five-literal reorder avoids live cubic labels but h0 still
  receives quadratic sub-monomials of live masks (`l0·l1`). Randomized helpers (item 1) make both less visible.
- At saturation 48 of 128 band wires are retiring: neither serving nor growing. Drained retiring wires
  (`refs == 0`) are valid dirty helpers and would widen the helper pool for free.
- Stats are misleading in cubic mode: `cubic_units`, `quartic_units`, `omitted_restores` are always 0,
  `full_stage2_samples_each_direction` is hard-coded, and `compact_masks`/`factored` are reported though bypassed.
- The witness check (64 lanes per fire boundary) covers the pre-migration stream only; the migrated stream is
  verified by 1,024 random states per direction plus the payload check. Adequate, but a per-box check after
  migration would be cheap.
- `Config` has 28 flags with many mutually exclusive branches, and `lib.rs` has dense one-line blocks
  (lines 275, 285–292, 387, 439, 491). Before any port into `local_mixing`, freeze the package as a named preset and
  delete the bypassed branches (quadratic compile, forest, feedback, midfire, refresh-both); keep them in the research crate.

## Scratchpad artifacts (session only)
Patched copy and runs under the session scratchpad `gen/`, `runs/{base,randh,rett,both,restored,rb4,rb8}`;
`hist.py` computes the write histograms. Nothing in `~/Codex` was modified.
