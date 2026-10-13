# Stage 6, delivery: packing + unpacking with lattice refinement

Status: design 2026-10-12 (RC), implemented in Python (`scripts/windowed_delivery/waist/deliver.py`,
the reference) and in Rust (`tdp_unpacking run --refine`, driven by stage 6 of `tdp_gen`); on `main`
since 2026-10-12. Stage 6 of the six-stage pipeline (`docs/tdp_pipeline.md`,
`docs/TDP_PIPELINE_REFERENCE.md` §6a-§6b). Supersedes the separate packing and unpacking stages of
`docs/STAGE7_UNPACKING.md`, which still documents the unpacking kernel, its CLI and its outputs.

## 1. Decision

RC, 2026-10-12: packing and unpacking are one stage, stage 6 (delivery). The stage packs the mixed circuit into ANF
windows, unpacks every window by resynthesis, and treats a window that does not resynthesize as a
packing defect: the window is cut in two, both halves are repacked and unpacked separately,
recursively. The size of a resynthesized window is irrelevant ("windows that unpack badly are
completely fine"). An optional flag stops the stage after the ANFs are emitted (`anf_only`).

Measured basis (`docs/notes/stubborn_windows_20261012.md`): on the n=64 full-chain run (stages 1-5,
then packing), 157 of 10,366 windows failed to unpack in 30 s; they are the windows whose ANF did not cancel (mean
monomial degree 6.2 vs 2.6, the density in the last quarter: an uncancelled crossing-stage
identity), not a gate-structure or position class. Ten times the time budget rescues 15 of 102.
Bisection converges by ~90% per round and terminates: halves 302/314, quarters 21/24, eighths
5/6, sixteenths 1/2, thirty-seconds 2/2 (15- and 16-gate pieces synthesize in 0.02 s). The final
assembly is equivalent to the packed circuit and exposes 0 interior C values and 0 early outputs.

## 2. Contract

Inputs: the packed, repaired circuit `rep<G>.mpmct1` and its bounds `rep<G>.bounds` (the output of
the packer: scheduler + order repair + C-view repair loop, unchanged), the source circuit `C`, `n`.

Loop, round r = 0, 1, …:

1. **Lattice** L_r: cuts `[0] + bounds_r + [m]`. L_0 = the packer's bounds.
2. **Deliverable** D_r: the container (`LMTDPANF` v1, `docs/STAGE4_RUST_DESIGN.md` §6) whose
   windows are the ANFs of the windows of L_r. D_0 is emitted by the packer. D_{r+1} differs from
   D_r only in the windows that were split: their blobs are replaced by the two halves' blobs
   (each blob is a pure function of `gates[lo:hi]`); the header (window count, body bytes, CRCs)
   is recomputed.
3. **Unpack** the *pending* windows of L_r (round 0: all windows; round r > 0: the halves created
   in round r) from D_r alone (the synthesizer sees only the window ANF; the packed gates are read
   only to define the pending set and for the exact check). Each window ends `converged` (its gates
   verified exactly against the delivered ANF) or `failed` (`timeout`, `stalled`, `stepcap`,
   `verify-fail`, `exact-overflow`).
4. **Refine**: for every failed window [lo, hi) with hi − lo ≥ 2, choose a cut c, add it to the
   bounds, mark [lo, c) and [c, hi) pending. The cut is exposure-aware, as in the packer's budget
   split (`anf_split2.py` / `packing::repair::split`): 32 candidates `lo + (hi−lo)·i/33`, one
   C-view scan over the candidates, the candidate nearest the midpoint at which no interior C value
   is affine; if none is clean, the midpoint, and the window is counted in `refine.unclean_cuts`.
   A failed window of 1 gate is emitted verbatim (counted in `refine.verbatim`; expected 0).
   A half whose ANF exceeds the 65,536-monomial budget (the packer's cap test) is counted in
   `over_budget`, emitted into D_{r+1} (so the container stays complete) but not synthesized: it is
   split again in the next round. A delivered window above the budget is a check failure.

5. Stop when no window is pending, or after `max_rounds` (default 16: a 1,024-gate window reaches
   1-gate pieces in 10 rounds); the remaining failed windows are emitted verbatim and counted.

Outputs (in the stage directory):

| file | content |
|---|---|
| `deliverable.tdp4` | D_final, the container on the refined lattice (the refined deliverable): the windowed ANF form the circuit was resynthesized from, an internal record kept for auditing and parity, not published; `bounds.final` its bounds (one cut per line; secret intermediate) |
| `unpacked.mpmct1` | concatenation of every window's converged gates in lattice order, then `xgate_canonicalize` (the `tdp_unpacking` canonical order); header = state width. **The shipped file**: the circuit is the published object of the pipeline (the TDP) |
| `windows.tsv` | one row per window of the final lattice: `idx lo hi round orig_gates syn_gates status synth_status seconds monomials changed_wires input_vars max_degree` |
| `rounds.tsv` | per round: pending, converged, failed, split, seconds (header `round pending converged failed split seconds`). Unclean cuts are not a column: the per-round count is in the log's round line and the total is `unclean_cuts` in the report |
| `delivery.report` (Python) / `unpacking.report` (Rust: the `tdp_unpacking` report keys of `docs/STAGE7_UNPACKING.md` §4 plus these) | `key=value`: `rounds`, `windows_initial`, `windows_final`, `splits`, `unclean_cuts`, `over_budget`, `verbatim`, `gates_packed` (Rust: `gates_in`), `gates_out`, `ratio`, every check, `status=ok|verbatim_used` |

With `anf_only` the stage stops after D_0 (= the packer's deliverable, `packing/deliverable.tdp4`, then the
stage's output and the published object, which the recipient unpacks; with `run.skip_stages = [4, 5]` this is
the validated four-stage chain) and writes no circuit.

What ships is `unpacked.mpmct1` alone: the circuit, the canonical-order concatenation of the resynthesized
windows. The adversary holds exactly that circuit and nothing else: not C, not the circuits of stages 1-5, not
`rep<G>.mpmct1` or the packer's gate order, not the lattice (`bounds.final`) and not the window ANFs
(`deliverable.tdp4`, the internal record of the windowed form the circuit was resynthesized from). It can cut
the circuit anywhere, compute the function of any segment, compose segments and resynthesize them into circuits
of its own; what it never sees is how a window was spelled before resynthesis: the packer's (and the mixer's)
spelling is gone, every window was re-spelled from its function alone (the synthesizer's separation guarantee,
`docs/STAGE7_UNPACKING.md` §2), and the spelling it holds is the synthesizer's, whose gates are interior cuts
of the window. The measurements that model this adversary are the all-cuts C-view census of the resynthesized
circuit (at n=64 the unpacked circuit exposes 5 of 936 interior C values at its own cuts, the raw packed
circuit 0) and the uniform re-cut-and-resynthesize experiments, which re-cut the circuit uniformly and never
use the delivered lattice: merging and resynthesizing uniform W=2000 windows of the n=64 crossing output
recovers 244 of 936 interior C values (241 with a 10× time budget; canonical reorder changes the count by
≤ 2%), while the raw circuit exposes none at any cut, and stages 4 and 5 do not close this leak; that is why
every cut this loop adds is exposure-aware and re-scanned (`check.new_cuts`, §4;
`docs/notes/stubborn_windows_20261012.md`, `docs/notes/stage45_revival_20261011.md`). Under `anf_only` the
published object is the window ANFs on the packer's lattice and the older framing applies: the adversary holds
the lattice and, for each window, the ANF of the permutation it applies, derives from them the wire state at
every window boundary on inputs of its choice, and a window has no interior cuts (the four-stage pipeline's
threat model; `docs/TDP_PIPELINE_REFERENCE.md`, "Threat model in brief"). The reports `delivery.report` /
`unpacking.report` hold counts and parameters only; they are internal records, not part of the TDP.

Checks before the outputs are committed (any failure is fatal, nothing renamed):

- every converged window: exact (recomposed ANF == delivered blob) and sampled;
- D_final: every cut of the refined lattice passes the packer's final check (`tdp_packing
  check-final`: no interior C value affine at a cut, every window within the 65,536-monomial
  budget), and `decode(D_final)` evaluates equal to `rep<G>.mpmct1` on 4 × 1024 random states;
- `unpacked.mpmct1` ≡ D_final before and after canonical order (4 × 1024 random states), zero
  slice against C (4096 samples).

Parameters (`[delivery]` in the config; CLI of the binaries in §4):

| key | default | meaning |
|---|---|---|
| `anf_only` | false | stop after the packer's deliverable, which is then the published object (no circuit is written; the recipient does the unpacking) |
| `time_limit` | 30 | seconds per window per round |
| `restarts` | 16 | synthesizer restarts; "less optimized" = fewer restarts, e.g. 2 |
| `beam` | 6 | |
| `max_rounds` | 16 | |
| `run.threads` (not a `[delivery]` key) | all cores | the `--threads` of `tdp_packing` and `tdp_unpacking`; the output depends on it only through the per-window deadline |

## 3. Python reference

`scripts/windowed_delivery/waist/deliver.py` (orchestration in Python; kernels are the existing
binaries, as in the packer chain which already calls `hmap_affine`):

```
deliver.py --rep rep512.mpmct1 --bounds rep512.bounds --source-c C --n N --out DIR [--body deliverable.bin]
           [--anf-only] [--time-limit 30] [--restarts 16] [--beam 6] [--max-rounds 16] [--threads T] [--seed 1]
           [--budget 65536] [--synth tdp_unpacking|anf_synth] [--synth-bin PATH] [--canon-bin tdp_unpacking]
           [--hmap ~/code/local_mixing/target/release/hmap_affine] [--label tdp]
```

Per round it (1) emits D_r by blob splicing (`anf_compose.window_anf` on the new halves, the blob
layout of `anf_dump_eval.py`, the header of `tests/manual/stage4/container_header.py`), (2) runs
the synthesizer on the pending windows only (`--windows i,j,…`), (3) chooses cuts with one
`hmap_affine --g-cuts` scan over all candidates of all failed windows, (4) assembles the final
circuit from the per-round outputs and canonicalizes it (`canonicalization` via `tdp_unpacking
canon` or the Python fallback), (5) runs the checks. It is the oracle for the Rust parity test.

## 4. Rust

- `tdp_unpacking run … --refine [--max-rounds 16]` (needs `--packed/--bounds` and
  `--source-c/--n`) performs the loop in-process (`src/stages/unpacking/refine.rs`; CLI and
  outputs in detail: `docs/STAGE7_UNPACKING.md` §10): pending set, exposure-aware cut choice
  (`packing::repair::split::split_candidates` + one `CView::leaky` scan per round over the
  candidates of every failed window; the C-view built once on C with the packer's defaults), blob
  re-emission (`packing::serialize::encode_window` of the packer's `window_anf` emission, spliced
  into the container with window count, body bytes and CRCs recomputed), assembly, canonical
  order, checks, outputs as in §2. Per round it writes `bounds.<r>` and `deliverable.<r>.tdp4`
  (round 0 = the inputs), at the end `deliverable.tdp4` and `bounds.final` (the refined deliverable, an
  internal record), `unpacked.mpmct1` (the shipped circuit),
  `windows.tsv`, `rounds.tsv`, `unpacking.log` and `unpacking.report` (the `tdp_unpacking` report of `docs/STAGE7_UNPACKING.md` §4 with
  `refine=1 rounds windows_initial windows_final splits unclean_cuts over_budget verbatim` and
  `status=ok|verbatim_used`). A pending window is synthesized with its index in the current
  lattice, so the tie-break PRNG and verify seeds equal the reference's, which calls `--windows`
  on the re-emitted deliverable each round.
- Checks as in §2, with two implementation notes. (1) The final deliverable is checked by
  `decode` + `check_windows_anf` (every window blob-equal to `window_anf` of the packed gates and
  equal to them) + `check_deliverable` + `check_zero_slice` (`check.final_deliverable`), and every
  final window is within the budget by the emission bookkeeping; the C-view condition on the
  refined lattice is `check.new_cuts`: every added cut was scanned clean at choice time or counted
  in `unclean_cuts` — a per-cut C-view result does not depend on the lattice, so the packer's
  final scan of the old cuts plus the choice-time scan of the new ones is exactly `check-final`'s
  scan of the refined lattice, without a second full scan. (2) Over-budget halves: both implementations emit the half, skip its synthesis and split it again next round; the Python reference checks `final_budget` on every new window of D_final, the Rust refuses at max_rounds. Neither case occurs on n8_L8 (`over_budget=0` on both sides).
