# Stage 6, delivery: packing + unpacking with lattice refinement

Status: design 2026-10-12 (RC), implemented in Python (`scripts/windowed_delivery/waist/deliver.py`,
the reference) and in Rust (`tdp_unpacking run --refine`, driven by stage 6 of `tdp_gen`).
Supersedes the separate stages 6 (packing) and 7 (unpacking) of `docs/STAGE7_UNPACKING.md`.

## 1. Decision

RC, 2026-10-12: stages 6 and 7 become one stage. The stage packs the mixed circuit into ANF
windows, unpacks every window by resynthesis, and treats a window that does not resynthesize as a
packing defect: the window is cut in two, both halves are repacked and unpacked separately,
recursively. The size of a resynthesized window is irrelevant ("windows that unpack badly are
completely fine"). An optional flag stops the stage after the ANFs are emitted (`anf_only`).

Measured basis (`docs/notes/stubborn_windows_20261012.md`): on the n=64 seven-stage run, 157 of
10,366 windows failed to unpack in 30 s; they are the windows whose ANF did not cancel (mean
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
   A half whose ANF exceeds the 65,536-monomial budget is counted in `over_budget`; the Rust
   treats it as failed without synthesis and splits it again in the next round, so that the
   final check below holds (the Python reference synthesizes and delivers it: §4).
5. Stop when no window is pending, or after `max_rounds` (default 16: a 1,024-gate window reaches
   1-gate pieces in 10 rounds); the remaining failed windows are emitted verbatim and counted.

Outputs (in the stage directory):

| file | content |
|---|---|
| `deliverable.tdp4` | D_final, the container on the refined lattice; `rep<G>.bounds.final` its bounds |
| `unpacked.mpmct1` | concatenation of every window's converged gates in lattice order, then `xgate_canonicalize` (the stage-7 canonical order); header = state width |
| `windows.tsv` | one row per window of the final lattice: `idx lo hi round orig_gates syn_gates status synth_status seconds monomials changed_wires input_vars max_degree` |
| `rounds.tsv` | per round: pending, converged, failed, split, unclean cuts, seconds |
| `delivery.report` (Python) / `unpacking.report` (Rust: the stage-7 keys plus these) | `key=value`: `rounds`, `windows_initial`, `windows_final`, `splits`, `unclean_cuts`, `over_budget`, `verbatim`, `gates_packed` (Rust: `gates_in`), `gates_out`, `ratio`, every check, `status=ok|verbatim_used` |

With `anf_only` the stage stops after D_0 (= the packer's deliverable) and writes no circuit.

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
| `anf_only` | false | stop after the packer's deliverable |
| `time_limit` | 30 | seconds per window per round |
| `restarts` | 16 | synthesizer restarts; "less optimized" = fewer restarts, e.g. 2 |
| `beam` | 6 | |
| `max_rounds` | 16 | |
| `threads` | all | |

## 3. Python reference

`scripts/windowed_delivery/waist/deliver.py` (orchestration in Python; kernels are the existing
binaries, as in the packer chain which already calls `hmap_affine`):

```
deliver.py --rep rep512.mpmct1 --bounds rep512.bounds --body deliverable.bin --source-c C --n N
           --out DIR [--anf-only] [--time-limit 30] [--restarts 16] [--max-rounds 16] [--threads T]
           [--synth tdp_unpacking|anf_synth] [--hmap target/release/hmap_affine]
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
  (round 0 = the inputs), at the end `deliverable.tdp4`, `bounds.final`, `unpacked.mpmct1`,
  `windows.tsv`, `rounds.tsv`, `unpacking.log` and `unpacking.report` (the stage-7 report with
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
  scan of the refined lattice, without a second full scan. (2) Over-budget halves: the Rust does
  not synthesize a half above the budget; it is split again next round (§2 step 4). The Python
  reference emits it with a budget-free `window_anf`, synthesizes it and delivers it if it
  converges, which can leave a window above the budget in D_final (counted in `over_budget` but
  not refused). Neither occurs on n8_L8 (the parity run has `over_budget=0` on both sides). A
  half above 2^20 monomials cannot be emitted by the Rust composer and is a fatal error (exit 4).
- `--windows i,j,LO:HI` restricts a plain run to a window subset (0-based lattice indices, `LO:HI`
  half-open): only those windows are synthesized and emitted, as a plain concatenation in index
  order with no canonical reorder and no circuit-level check (`windows_selected=` in the report);
  the Python reference calls it per round. `tdp_unpacking canon IN OUT` applies
  `xgate_canonicalize` to a circuit, checks it equivalent on 4096 random states and writes it with
  the input's header width (the reference's final order). `--anf-only` is a driver-level flag.
- Driver (`scripts/tdp_gen.sh`, `src/tdp`): stage 6 = delivery = `tdp_packing run` then
  `tdp_unpacking run --refine`; `[delivery] anf_only = true` ends the stage after packing. Stage
  numbering becomes 1 sandwich, 2 preprocessing, 3 DB mixing, 4 splitting, 5 crossing, 6 delivery;
  `run.skip_stages` accepts 2..6; config_version 4 (3 refused with a pointer), recipe v12, SEEDS
  format 3 unchanged (the unpack seed is the delivery seed). Markers: `delivery.recipe`,
  `delivery.failed`.
- Tests: `tests/integration/tdp_unpacking.rs` runs `--windows`, `canon` and the refine path on
  the n8_L8 fixture, once with the test-only knob `--force-fail-rounds 1` (round 0 synthesizes
  nothing and fails every window, so the loop, the cut choice, the re-emission and the final checks
  are exercised: windows 73 → 146+) and once without it (against the plain run: identical circuit
  when every window converges; otherwise the plain run's fallback windows are the loop's splits);
  unit tests (`tests/unit/stages/unpacking/refine.rs`) cover the cut choice (clean, unclean, tie,
  single scan), the blob splice (byte-equal to `encode_body` of the refined lattice) and the loop
  on a tiny circuit. `tests/manual/delivery_parity.sh` runs `deliver.py` (`--synth tdp_unpacking`,
  `--canon-bin tdp_unpacking`, `hmap_affine` for the scan) and `tdp_unpacking run --refine` on
  n8_L8 with the same seed, time limit and restarts, and cmp's `bounds.final`, `deliverable.tdp4`,
  `unpacked.mpmct1` and every per-round file: identical (2026-10-12, seed 3, restarts 2: window 1
  stalls and is split, 73 → 74 windows, 34,825 → 10,913 gates on both sides). The shell tests
  follow the renumbering (driver work).
