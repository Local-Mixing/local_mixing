# Stage 7 (unpacking): resynthesis from the delivered ANF windows

> **Superseded by `docs/DELIVERY_STAGE.md` (2026-10-12).** Packing (stage 6) and
> unpacking (stage 7) are one stage, *delivery*, with a refinement loop: a window
> that does not resynthesize is cut in two and both halves are re-emitted and
> unpacked again. The binary, its per-window algorithm, outputs and checks below
> are unchanged and remain the loop's kernel; the loop itself is `tdp_unpacking
> run --refine` (§10 here, `docs/DELIVERY_STAGE.md` §2 and §4).

Status: implemented 2026-10-11, branch `ran/stage7-unpacking` (based on
`ran/seven-stage` @ 0569792e); `--refine`, `--windows` and `canon` added
2026-10-12 (`ran/seven-stage`). Owner: RC.

## 1. Purpose

Until 2026-10-12 the seven-stage TDP pipeline ended with **packing** (stage 6:
the windowed ANF deliverable `deliverable.tdp4`) and **unpacking** (stage 7).
The owner's specification of stage 7:

> unpacking: simply resynthesizes a circuit from each ANF window using a
> standard ANF synthesization, concatenates all the piecewise circuits,
> reorders the gates in the canonical order (per commuting gates), and
> outputs.

Stage 7 is what a recipient of the deliverable does to obtain a circuit again.
Its output `unpacked.mpmct1` computes exactly the deliverable's function (the
same state width, inputs on wires `0..n`, the answer on `n..2n` on the zero
slice); its gate order carries no window boundary.

## 2. Algorithm

The synthesizer is a port of `anf_synth` (standalone crate
`runs/ncdb_ab_20261001/synth/`, production bidirectional search since
2026-10-08; its `README.md` describes the search and its `REPORT.md` §1 the
measured results on the fx12ctl and a128 arms). Per window:

1. **Spec from the deliverable** (`src/stages/unpacking/spec.rs`). The decoded
   window (`packing::serialize::WindowView`: support wires, changed wires,
   monomials as masks over the support) is mapped onto a *local variable
   universe*: the support and changed wires in ascending global order plus
   the lowest `ancilla_pool` (32) other state wires, which the search may
   borrow as dirty ancillas for helper macros. The monomial width (1, 2, 4,
   8 or 16 u64 words) is chosen from the universe size, so a window on 150
   wires of a 1024-wire state searches over 4-word monomials. Local ids
   preserve global order, so every tie-break key the search sorts on is
   unchanged by the relabelling. **The synthesizer sees only this spec**: the
   `synth` module imports no circuit type and receives no gate, no file and
   no window position (the `anf_synth` separation guarantee, audited there by
   `audit/run_nopeek.sh`).
2. **Search** (`synth.rs`): forward input-side substitution moves and
   backward output-side peel moves with free estimates, exact rescoring of
   the best `exact_k` = 48 forward and `back_k` = 32 backward candidates,
   greedy choice (backward wins ties), 2-step lookahead over the top
   `beam` = 6 of each family, the Gauss–Jordan linear finisher, Barenco
   helper macros (amortised batches of up to 32) for degree ≥ 3 monomials,
   tabu perturbation with `stall_limit` = 60; 16 restarts cycling the
   degree-weight schedule `dw, dw+2, 0, dw-2` (`deg_weight` = 4, `w0` = 1,
   `w1` = 2), the shortest converged circuit kept; one wall-clock deadline of
   30 s per window shared by the restarts. Ties are broken by a splitmix64
   PRNG seeded from `(seed, window index, restart)` only, so a given `--seed`
   reproduces the output exactly unless the deadline truncates a window
   (`deadline_hit` in `windows.tsv`; such a window's output may depend on
   machine load, the result is still verified).
3. **Peephole** (`postopt.rs`): the pairwise pass of `anf_synth` (same-target
   gates brought adjacent by commutation merge or cancel when the XOR of
   their polynomials is again a gate polynomial), re-verified; kept only when
   it verifies. The opt-in group-merge variant (zero measured gain) is not
   ported.
4. **Verification** (`verify.rs`), on the gates mapped back to global wires:
   sampled (1024 random full states, every changed wire equals the window ANF
   evaluated from the decoded masks, every other wire unchanged — so a
   borrowed ancilla must be restored) and **exact**: the gates are composed
   by the packer's audited `packing::anf::window_anf` and the encoded blob must
   equal the deliverable's window byte for byte (the test `check_windows_anf`
   runs on the packing side). The composition runs at the emission budget
   2^18 and is retried once at 2^20 if it overflows (`exact_retried` in the
   report; 0 on the real run).
5. **Fallback**. A window that does not converge or fails a check is emitted
   verbatim from the packed circuit (`--packed rep512.mpmct1 --bounds
   rep512.bounds`, the packer's repaired circuit, windows `[0] + bounds + [m]`
   with the deliverable's indexing) **after that window's gates pass the same
   exact check against the deliverable** (a mismatch is exit 4: the packed
   circuit is not the deliverable's). Without `--packed`, or under
   `--no-fallback`, a failed window is exit 4. The synthesizer never sees the
   packed gates; they are read in the binary and used only for the fallback
   and the `orig_*` statistics.

Then (`run.rs`, the binary): the windows' gates are concatenated in window
order, the whole list is canonicalized with
`canonicalization::gate_order::xgate_canonicalize` (the DB builders' order:
every gate pulled as early as commutation allows when that lowers the
lexicographic order), and the circuit-level checks run (§5).

`xgate_canonicalize` was `Vec::remove` + `Vec::insert` per move (O(n) each,
unusable on 10^6 gates); it now rotates `gates[pos..=i]` in place, with
identical results (unit test against the old loop kept as the reference), so
its cost is the backward scan to the first colliding gate: 526,874 gates in
2.6 s.

## 3. CLI

```
tdp_unpacking run --deliverable <deliverable.tdp4> --out-dir DIR
                  [--packed <rep.mpmct1> --bounds <rep.bounds>]
                  [--source-c <C.g57> --n N]
                  [--seed S=1 | --seed-file PATH] [--threads K]
                  [--time-limit 30] [--restarts 16] [--beam 6]
                  [--no-fallback] [--no-postopt] [--label tdp]
                  [--windows i,j,LO:HI | --refine [--max-rounds 16]]
tdp_unpacking canon <in.mpmct1> <out.mpmct1>
tdp_unpacking selftest [--n 8] [--m 12] [--trials 40] [--seed 7] [--threads K]
```

`--windows`, `--refine` and `canon`: §10.

`--seed-file PATH` reads the decimal seed from a file (the driver passes
`/dev/fd/3`, as for `circuit_mixer`), so the seed never appears on argv;
`--seed` and `--seed-file` are mutually exclusive; the log and the report
name only the source (`argv`, `file`, `default`), never the value. The binary
reads no environment variable; `--threads` sets the rayon pool (default: all
cores). `selftest` composes random small circuits into a one-deliverable,
synthesizes them through the same pipeline with no fallback and verifies
(the `anf_synth --selftest` equivalent, for CI).

## 4. Outputs (all in `--out-dir`, created 0700; every file written to
`*.tmp` and renamed only after every check passed; stale outputs of an earlier
run are removed first)

| file | content |
|---|---|
| `unpacked.mpmct1` | the canonical circuit; header wire count = the deliverable's state width; `mpmct1` text (`t comp k c p ...`) |
| `windows.tsv` | one row per window: the `anf_synth` contract columns `idx orig_lo orig_hi orig_gates syn_gates changed_wires input_vars monomials max_degree status seconds syn_offset` (`orig_*` = −1 without `--packed`; `syn_offset` = the window's offset in the concatenation **before** canonical reordering), then `synth_status pre_postopt_gates steps note tag_greedy … tag_back universe width exact_budget restarts_used deadline_hit` |
| `unpacking.report` | `key=value`, publishable: no seed value, no hash of a private input (only the public deliverable's xxh3); windows / synthesized / fallback / deadline_hits / verify_fail / exact_overflow, `gates_in` (sum of the packed windows, `unknown` without `--packed`), `gates_out`, `gates_pre_postopt`, `gates_canonical`, `ratio`, gates by move type, seconds quantiles, restarts used, every check's result, `time.*`, `status=ok\|fallback_used` |
| `unpacking.log` | the progress lines (every 50th window and every fallback) and the summary line, also printed on stdout |

## 5. Checks (any failure is exit 4; nothing is renamed)

| check | what |
|---|---|
| `check.windows_exact` | every window's emitted gates compose (`window_anf`) to the deliverable's window, blob-equal |
| `check.windows_sampled` | every window on 1024 random full states: changed wires = ANF, other wires untouched |
| `check.equiv_concat` | the concatenation vs `Deliverable::eval_lanes16` on 4 × 1024 random all-wire states |
| `check.equiv_canonical` | the same after canonical reordering |
| `check.zero_slice` | with `--source-c C --n N`: x on wires `0..n`, every other wire 0, through the final circuit: wires `n..2n` hold C(x) (4096 samples); `skipped` otherwise |

Seeds of the sampled checks are fixed public constants (`param.verify_seed`).

## 6. Exit codes

| code | meaning |
|---|---|
| 0 | finished; `status=ok` or `status=fallback_used` in `unpacking.report` |
| 2 | usage or parameter error (also a state wider than 1024 wires) |
| 3 | input error: unreadable or malformed deliverable, packed circuit, bounds or C; packed window count ≠ deliverable's |
| 4 | verification or synthesis failure: a window failed with no fallback, a packed window ≠ the deliverable, a circuit-level check failed |
| 7 | I/O error writing an output (`*.tmp` removed) |
| 8 | internal invariant violated |
| 101 | panic |

## 7. Results (n = 64, `dressed_runs/base`, 2026-10-11)

`tdp_unpacking run` on `packing/deliverable.tdp4` + `rep512.mpmct1/.bounds`
+ `tdp.mpmct1.source_c.g57 --n 64`, 8 threads, `nice -n 10`, defaults:

| | |
|---|---|
| windows | 2591, all synthesized (0 fallback, 0 verify failures, 0 exact retries) |
| gates | 1,487,674 → 526,874 (ratio 0.3542; 530,575 before the peephole) |
| `anf_synth` on the same run (6 threads, `synth/summary.txt`) | 526,958 (0.3542) |
| seconds / window | p10 0.36, median 0.58, p90 1.18, max 30.2 (window 3: 952 gates, 28,188 monomials; its 16th restart hit the 30 s deadline, 15 converged) |
| wall | 255 s (synthesis 252 s, canonicalization 2.6 s, circuit checks 0.2 s); 271 MB RSS |
| local universe | median 156 wires → 4-word monomials (2551 windows), 2-word (40) |
| checks | windows exact + sampled ok, equiv concat ok(4096), equiv canonical ok(4096), zero slice ok(4096) |

The gate counts differ slightly from `anf_synth` because the monomial order
(and hence a few PRNG tie-breaks) differs under the per-window relabelling;
the algorithm, its defaults and its separation guarantee are the same.

## 8. Limits

- State width ≤ 1024 wires (n ≤ 256), as in the packer (`tdp_packing`; the
  exact check's monomial width).
- Convergence is heuristic; a stalled window falls back to the packed gates
  (counted, `status=fallback_used`) or fails. No fallback occurred on the real
  run or in the regression runs of `anf_synth`.
- A window that hits the deadline has a timing-dependent (still verified)
  output; `windows.tsv` flags it, the report counts it. Raise `--time-limit`
  for a bit-reproducible run.
- The ancilla pool reproduces `anf_synth`'s "lowest-numbered clean wire"
  choice as long as a helper macro needs at most 32 outside wires; beyond
  that the search picks inside the window (still exact, since every macro
  gate goes through the exact substitution).
- The exact check composes at budget 2^18, retried at 2^20; a synthesized
  window whose intermediate composition exceeds that is treated as unverified
  (`exact_overflow`, fallback).

## 9. Code and tests

```
src/stages/unpacking/{mod,types,mono,synth,postopt,spec,verify,run}.rs
src/programs/tdp_unpacking/{main,mod}.rs          [[bin]] tdp_unpacking
tests/unit/stages/unpacking/*.rs                  (repo convention, #[path])
tests/unit/canonicalization/gate_order.rs         rotation walk == remove/insert reference
tests/integration/tdp_unpacking.rs                [[test]] tdp_unpacking
```

The integration test builds the n8_L8 deliverable with `tdp_packing run` on
the committed fixture, unpacks it (fallback allowed, zero slice against C),
re-checks equivalence independently, exercises the no-fallback path, seed
reproducibility (`--seed` vs `--seed-file`), thread independence and the exit
codes on the crafted `tests/fixtures/stage4` (packing) cases. `cargo test --locked --lib --test
tdp_unpacking`.

## 10. `--refine`, `--windows`, `canon` (2026-10-12, the delivery loop)

`src/stages/unpacking/refine.rs`; contract and measured basis in
`docs/DELIVERY_STAGE.md`; Python reference `scripts/windowed_delivery/waist/deliver.py`
(parity test `tests/manual/delivery_parity.sh`: byte-identical `bounds.final`,
`deliverable.tdp4`, `unpacked.mpmct1` and every per-round file on n8_L8).

### `run … --windows i,j,LO:HI`

Synthesizes only the listed windows (0-based indices in the deliverable's lattice;
`LO:HI` is half-open; any order, duplicates dropped). Per-window checks and the
fallback are as in §2; the window index used for the tie-break PRNG and the verify
seed is the index in the lattice, so a window's result does not depend on the
selection. Output differences: `unpacked.mpmct1` is the plain concatenation of the
selected windows in index order (no canonical order, no circuit-level check — they
are meaningless on a subset); `windows.tsv` has rows only for the selected windows,
`syn_offset` relative to that file; `unpacking.report` adds `windows_selected=<k>`
and reports `check.equiv_concat`, `check.equiv_canonical`, `check.zero_slice` as
`skipped`. The Python reference calls this once per round.

### `canon <in.mpmct1> <out.mpmct1>`

Reads a circuit (strict `mpmct1`), applies `canonicalization::gate_order::
xgate_canonicalize`, checks the result equivalent to the input on 4096 random full
states (4 × 1024, fixed seeds) and writes it (via `*.tmp`) with the input's header
width. Exit 3 unreadable input, 4 the check failed.

### `run … --refine [--max-rounds 16]` (needs `--packed/--bounds` and `--source-c/--n`)

The loop of `docs/DELIVERY_STAGE.md` §2, in process:

- round 0 synthesizes every window of the deliverable; round r > 0 only the halves
  created in round r − 1, each with its index in the *current* lattice (so the PRNG
  keys match the reference, which re-emits the deliverable and calls `--windows` on
  the new lattice each round);
- a window is *failed* when its synthesis status is not `converged` or a check fails
  (`verify-fail`, `exact-overflow`); failed windows are not emitted verbatim inside
  the loop: a window of ≥ 2 gates is split, a 1-gate window is emitted verbatim
  (`verbatim`), and when `max_rounds` is reached (round r splits iff r + 1 ≤
  `max_rounds`) the remaining failures are verbatim (`status=verbatim_used`);
- cut choice: the packer's 32 split candidates `lo + (hi−lo)·i/33` minus the cuts the
  C-view scan flags (one `CView::leaky` scan over the candidates of every failed
  window of the round; the C-view is built once on C with the packer's defaults),
  the candidate nearest the midpoint, smaller on a tie; none clean: the midpoint,
  counted in `unclean_cuts`;
- re-emission: the halves' blobs by the packer's `window_anf` emission (CPython 3.9
  cap semantics); the next round's deliverable is the container with the blobs
  spliced in order (window count, body bytes, CRCs recomputed). A half whose ANF
  exceeds the packer's budget (65,536 monomials) counts in `over_budget`, is not
  synthesized and is split again in the next round (the Python reference does the
  same since `b248a0f2`; `DELIVERY_STAGE.md` §4). A half above 2^20 monomials is a
  fatal error (exit 4).

Outputs (in `--out-dir`; stale ones of an earlier run removed first): per round
`bounds.<r>` and `deliverable.<r>.tdp4` (round 0 = the inputs, re-encoded), then,
via `*.tmp` after every check passed: `deliverable.tdp4` + `bounds.final` (the
refined lattice), `unpacked.mpmct1` (every window's gates in final-lattice order,
canonical order), `windows.tsv` (columns `idx lo hi round orig_gates syn_gates status
synth_status seconds monomials changed_wires input_vars max_degree` first — a
verbatim row has round −1, status `verbatim`, the reason as `synth_status`, zeros —
then the §4 columns from `syn_offset` on), `rounds.tsv` (`round pending converged
failed split seconds`), `unpacking.log`, `unpacking.report` (the §4 keys plus
`refine=1 rounds windows_initial windows_final splits unclean_cuts over_budget
verbatim param.max_rounds param.budget`, `status=ok|verbatim_used`).

Checks (any failure exit 4, nothing renamed): per window exact + sampled (§5);
`check.final_deliverable` = decode of the final container, every window blob-equal
to `window_anf` of the packed gates and equal to them (`check_windows_anf`), equal
to the packed circuit on 1024 random states (`check_deliverable`), zero slice
against C (`check_zero_slice`), every final window within the budget (from the
emission bookkeeping); `check.new_cuts` = every added cut was scanned clean or
counted unclean (the per-cut C-view result does not depend on the lattice, so the
packer's final scan of the old cuts plus the choice-time scan of the new ones is the
final scan of the refined lattice); `check.equiv_concat`, `check.equiv_canonical`,
`check.zero_slice` as in §5.

`--force-fail-rounds N` is TEST ONLY: rounds 0..N synthesize nothing and fail every
pending window (the integration test exercises the split, the re-emission and the
next round on n8_L8 with it).
