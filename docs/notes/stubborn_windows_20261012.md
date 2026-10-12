# Stubborn windows, stage-7 fallback policy and the time budget (2026-10-12)

*RC's questions, in order: (1) emit every window in ancilla mode so that intermediate values stop
being proxies of C's; (2) forgo stage 7 and emit only windows too complex to synthesize small;
(3) take a stubborn window, factor it by the book with ancillas, does the factoring contain proxies;
(4) rerun the W=2000 merged-window resynthesis with 10x the time budget, only on the stubborn
windows; compare direct replacement with global canonical order; (5) what is special about the
stubborn windows; (6) policy: accept any synthesized size, send timed-out windows back to stage 6
to be cut in two; run the bisection to its end. Every number below was re-derived by an independent
four-lens verification pass (re-execution, code review, statistics, refutation); its corrections are
folded in and its caveats are in §7.*

Inputs: `port_runs/seven_n64` (seven-stage run; `packing/rep512.*` = packed circuit and delivered
lattice, 10,366 windows; stage 7 fell back on 157) and `s45_runs/base64` (crossing output cut at
uniform W=2000, 2,748 windows; the 60 s adversary run `adv_s5_2000` with 244 interior C values).
Census = the project's C-view convention (`critic/cview_xcheck.py`: random x on wires 0..n, zeros
elsewhere, complement-normalised values; interior = C gates that are not the last writer of their
wire, 936 of 1000; `output_proxies.py` for early outputs). Functional equivalence of every spliced
circuit against its packed source was checked on 8,192 random full states (`examples/equiv.rs`).

## 1. Ancilla-mode unpacking and by-the-book factorings (questions 1, 3)

`examples/anc_cview.rs`: for all 157 stubborn windows and 60 sampled ordinary windows, four
factorings of the delivered ANF (one multi-controlled gate per monomial into a clean ancilla, in
canonical and in random order; positive-Davio decomposition with hash-consed subpolynomials, two
variable orders), every materialised value censused against C.

| representation, all cuts (n=64) | interior C values of 936 |
|---|---|
| raw packed crossing circuit | 0 |
| stage-7 output (anf_synth) at its own lattice | 5 |
| the four factorings, 157 stubborn windows | 0 |
| the four factorings, 60 ordinary windows | 0 |

Ceiling: 43 interior C values are degree-2 polynomials (5 to 17 terms) of the entry wires of
37 of 152 tested stubborn windows (5 windows with supports above 155 wires were skipped by the
12,000-unknown cap), 37 values in 20 of 60 ordinary windows; 0 are affine (`cview_n64aff.tsv`); only
3 of the 52 stubborn (window, value) pairs have that polynomial as a single-cube sub-sum of an exit
polynomial (a lower bound on algebraic-factor occurrences). Degree-2-reachable values are *less*
frequent per stubborn window (0.34) than per ordinary window (0.62). Reading: the proxies the
adversary finds are not algebraic factors of the delivered polynomials and no textbook factoring
reaches them; they come from the in-place substitution search on merged windows. An ancilla-mode
stage 7 therefore emits a proxy-free circuit, but it inverts to the deliverable by inspection, so
the adversary's merge-and-resynthesize path is unchanged. Cost side: 99% of windows are
non-triangular (direct in-place emission impossible; the inverse ANF would have to be delivered),
forward monomials = 6.1x the packed gate count. Earlier claim retracted: a common-subexpression pass
over the ANF does *not* find the proxies.

## 2. The 10x time budget at W=2000 (question 4)

Only the 164 windows that hit the 60 s deadline can change (the others rerun the same 16 seeded
restarts), so `anf_synth --windows <164> --time-limit 600` (1.7 h on 9 threads) and splice
(`s45_runs/splice.py`, self-tested byte-for-byte).

| crossing output, W=2000 | 60 s | 600 s on the 164 |
|---|---|---|
| interior C values, direct replacement | 244 | 241 (like-for-like 242: 1 of the 3 lost values is a compose-limit artefact, windows 675/2317 kept the default `--compose-limit 300`) |
| after global canonical order | 242 | 239 |
| outputs appearing early (before 90%) | 45 of 64 | 45 of 64 |
| timeouts rescued | | 17 (median 527 s); they add **0 new values** (`compare_tl.py`: only-600 s = 0; their 1 first-creation is a re-attribution) |
| still timing out / deadline-hit | 28 / 164 | 9 / 41 |

Per-window first-creations at 60 s (`attrib_leak.py`): windows done in under 10 s 0.112, 10 to 60 s
0.084, over 60 s 0.055, deadline-hit 0.059, every fallback 0 (structural: verbatim packed gates).
The gradient is real but modest (about 2x, ordered-test p ~0.007); `seconds` is whole-window wall
time including ANF composition, so the slow buckets partly measure composition cost; synthesized gate
count is a slightly better difficulty proxy (partial Spearman with leak -0.09 vs -0.035). The 161
non-deadline fallbacks (144 composition overflows or compose timeouts, 17 stalls) were not rerun.

**Canonical order:** 244 -> 242 (crossing, W=2000, 60 s), 241 -> 239 (600 s), 434 -> 434 (stage-3
output, W=2000), 297 -> 302 (stage-3 output, delivered lattice): at most 1.7% either way. Expected:
a leaked value is written by one gate, and reordering commuting gates changes only same-target
partial sums. Per-window attribution on canonical circuits is nominal (gates cross window bounds).

## 3. What is special about the stubborn windows (question 5)

`feat_n64.json` (`stubborn.py`, `stubborn2.py`): 157 stage-7 fallbacks vs 10,209 synthesized.

| feature | ordinary median | stubborn median | AUC |
|---|---|---|---|
| mean monomial degree | 2.56 | 6.21 | 0.91 |
| monomials of degree > 3 | 25% | 92% | 0.91 |
| monomials per changed wire | 12 | 222 | 0.91 |
| monomials | 597 | 9,134 | 0.91 |
| packed gates / mean controls / max controls | 492 / 2.88 / 7 | 482 / 2.90 / 8 | 0.36 / 0.57 / 0.56 |
| support / changed wires / position | 111 / 47 / 0.50 | 106 / 44 / 0.46 | 0.38 / 0.39 / 0.48 |

Nothing structural separates them usefully (held-out AUC of all non-ANF features 0.69 vs 0.91 for
the ANF ones; the weak gate-count difference is significant but tiny). Every window nests
multi-control gates deeply (syntactic degree bound ~300, Spearman with the real density ~0), so the
normal case is massive cancellation inside the window and stubborn means cancellation failed.
Quarters (`halves_probe.rs`): quarters 1 to 3 of a stubborn window have ~230 monomials like ordinary
quarters; quarter 4 alone has median 809 (p90 3,829; ordinary 245). The density sits in the tail in
74% of stubborn windows, and the whole-window count (9,134) far exceeds the sum of quarters, so it is
also a cross-quarter composition effect: the picture is an uncancelled half of a crossing-stage
identity at the end of the window.

Time alone: on the 157 under `anf_synth` (seed 1; the stage-7 port used its own seed and ran under
load, so "157" is one synthesizer's verdict), 30 s converges 55 (at 2.2x the packed size), 300 s
converges 70; the 45 stalls at 30 s stay stalls (deterministic: stall-limit 60 ends each restart
regardless of budget); 87 of 157 resist 300 s. 160 restarts with stall-limit 600 and 300 s did not
rescue a 62-gate stalled piece.

## 4. Bisection policy (question 6)

| round | pieces | converged (30 s) | failing |
|---|---|---|---|
| halves of the 157 | 314 | 302 (145 of 157 pairs) | 12 (10 second halves) |
| quarters of the 12 | 24 | 21 | 3 (103 to 123 gates) |
| eighths of the 3 | 6 | 5 | 1 (62 gates, 671 monomials) |
| sixteenths of that one | 2 | 1 | 1 (31 gates, 299 monomials, degree 11, times out) |
| 32nds of that one | 2 | 2 (15 gates/41 monomials in 0.02 s -> 70 gates; 16 gates/120 monomials in 0.2 s -> 114 gates) | 0 |

Round 1 took 4 min on 8 threads. The round-2 assembly (`unpacking_sel157/quarters/assembled`:
10,209 packed windows, 302 halves, 21 quarters, 3 verbatim) and every spliced circuit are equivalent
to `rep512.mpmct1` and show 0 interior C values and 0 early outputs; at this lattice the stage-7 base
rate (5 values over 10,209 windows) predicts ~0.2 values for a few hundred pieces, so this is the
expected outcome, not evidence that stubborn windows are safer. The policy converges by ~90% per
round and terminates on its own (RC's bet: the 15- and 16-gate pieces pass instantly), so no
verbatim floor is needed; the complete bisection-to-the-end assembly (`unpacking_sel157/final`, no
verbatim piece inside any stubborn window) is equivalent to `rep512.mpmct1` with 0 interior C values
and 0 early outputs: final lattice 10,540 windows = 10,209 packed + 302 halves + 21 quarters +
5 eighths + 1 sixteenth + 2 thirty-seconds, 5,201,064 gates, EQUIVALENT on 8,192 random full states,
64 of 1000 C values visible (0 interior, the 64 outputs), 0 early outputs. The raw crossing circuit has 0 C
values at every cut, so a midpoint cut keeps the verbatim floor; the packer's affine cview check
must still be re-run on each new cut (not measured here).

## 5. Dropped: seam randomisation

R_i R_i^{-1} at each cut: 256-wire state needs ~1 gate per wire (+50% gates), raw boundaries already
carry 0 C values, the synthesizer's backward peel removes seam gates first, merging cancels them
exactly in the composed ANF, and the seam-growth probe hit quadratic polynomial products (9 GB).
RC dropped the idea.

## 6. Tools (untracked)

`port_wt/examples/{anc_cview,anc_probe,canon,equiv,halves_probe}.rs` (build with
`CARGO_TARGET_DIR=~/code/lm_worktrees/port_target cargo build --release --example NAME`);
`s45_runs/{adversary_s5_tl.sh,attrib_leak.py,compare_tl.py,splice.py}`;
`port_runs/seven_n64/unpacking_sel157/{synth_tl30,synth_tl300,spliced_*,halves,quarters,eighths,sixteenths}`;
`s45_runs/base64/adv_s5_2000_tl600sel/{synth,spliced,spliced_canon}`; scratchpad TSVs
`cview_n64b.tsv(.values.tsv)`, `cview_n64aff.tsv`, `probe_n64.tsv`, `halves_{fb,ok}.tsv`,
`feat_n64.json`. Verification pass outputs: scratchpad `verify/`.

## 7. Caveats from the verification pass

1. The zero-leak results for stubborn windows (§1 factorings, §4 bisection) are measured at the
   delivered lattice, where the full stage-7 output leaks only 5 of 936; they have no power to rank
   stubborn against ordinary windows. The leaky regime (W=2000, 244 values, carried by the easy
   windows) was never tested with bisection or factorings.
2. Only the verbatim/complement census was applied to synthesized circuits; affine and linkable
   exposure (`hmap_affine`, `link_rep512.txt`) exist for the packed circuit only.
3. The 60 s baseline ran under heavier CPU contention than the 600 s rerun; `--time-limit` is not
   a hard cap (max 924 s observed). Direction of the confound favours the rerun, which still did
   not reduce the leak.
4. Equivalence is sampled (8,192 states), the shared project convention.
5. The stubborn set is seed- and timing-specific to the stage-7 port (`anf_synth` converges 55 of
   157 at the same 30 s).
