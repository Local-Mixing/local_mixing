# Stage 6, adaptive mode: C-aware adaptive resynthesis

`delivery.mode = "adaptive"` (the default since recipe v13) runs stage 6 as
the adaptive, C-aware resynthesis of RC's design note `stage6_adaptive.md`
(2026-10-21) in the variant the review
([notes/stage6_adaptive_review_20261021.md](notes/stage6_adaptive_review_20261021.md))
called **v2** and the server campaigns of October 2026 ran: exposure-aware
block acceptance with two synthesizer seeds, parallel segment planning,
boundaries moved between rounds, size-scaled synthesis budgets, and RC's
stop rule of 2026-10-25 (end of mixing when two consecutive rounds compress
jointly by less than 2 %). The binary is `tdp_adaptive`
([src/programs/tdp_adaptive/](../src/programs/tdp_adaptive/)); the library
is [src/stages/adaptive/](../src/stages/adaptive/): `types` (the shared
vocabulary), `census`, `boundary`, `planner` (the analysis half), `synth`,
`round`, `driver` (the synthesis half). It reuses the unpacking synthesizer
([src/stages/unpacking/synth.rs](../src/stages/unpacking/synth.rs), the port
of `anf_synth`) and the C-view sampling of the packer
([src/stages/packing/cview.rs](../src/stages/packing/cview.rs)).

## 1. Vocabulary

* A **cut** $t$ of a circuit of $m$ gates is the state after the first $t$
  gates, $0 \le t \le m$; a **block** $[a, b)$ holds gates $a..b$ and maps the
  full wire state at cut $a$ to the full wire state at cut $b$.
* The **honest slice**: input $x$ on wires $0..n$, zeros on every other
  wire. A **birth** $j$ of $C$ is the value $c_j(x)$ written by its $j$-th
  gate; the **internal** births are those overwritten later by $C$ (936 of
  1000 at n=64).
* A birth is **directly exposed** at a cut when a live wire equals
  $c_j(x) \oplus b$ for a constant $b$, and **affinely exposed** when
  $c_j(x) = b \oplus \bigoplus_{i \in S} w_i$ for a set $S$ of simultaneously
  live wires. Direct exposure is a special case of affine exposure. An
  exposure has a **lifetime**, an interval of cuts.
* A **location** names a wire value: the initial value of wire $w$ is
  location $w$, the value written by gate $i$ is location $\text{width} + i$.
* **Deeper** births are the internal births that are not polynomials of
  degree at most 2 in the public inputs or in the public outputs of $C$
  (the "public quadratic boundary", which an adversary can derive without
  the circuit; 882 of 936 at n=64 for a random $C$).

## 2. What a round does

Input: the round's circuit $G$ (round 1: the stage-5 output, or the previous
stage's output), the source $C$, the boundary classification. Before round
1 the run checks the input itself on the zero slice against $C$ and, with
`--sandwich`, against the payload contract (`check.input_zero_slice`,
`check.input_payload_contract`), so a wrong `--n`, `--source-c` or
`--sandwich` fails in seconds (exit 4), and refuses a census sample count
too small for the circuit's width (exit 2).

1. **Census** (`census::census`). Every cut of $G$ is scanned on
   `census_batches` × 64 honest inputs (6,144 by default; seed 20261016, the
   prototype's) with an incremental GF(2) basis over the live wire values:
   for every birth of $C$ the intervals of cuts at which it is affinely
   recoverable, with the witnessing locations and the constant, and every
   direct copy with its lifetime. Every distinct interval-entry witness is
   then re-tested on 8,192 fresh random inputs and on structured inputs
   (zero, all-ones, unit vectors, pairs, and per $C$ gate eight patterns
   propagated through $C$'s prefix); a witness that fails is marked
   unvalidated. The census is complete linear-span membership on the
   discovery inputs, not a proof; the report gives the discovery and the
   validated counts separately.
2. **Plan** (`planner::plan`). The circuit is cut into `segments` segments
   (seams at unexposed cuts near $i m / S$, at least `avoid_radius` from the
   previous round's boundaries), planned in parallel. In a segment the
   planner grows a block from its start by exact incremental ANF composition
   of the gates (every wire a variable; limits: `term_cap` × 8 terms during
   composition, `term_cap` × 4 per wire, 4,000,000 product pairs per gate
   update, `max_block_gates` gates); an endpoint is eligible when the total
   fits `term_cap`; the scan continues 256 gates past the last eligible
   endpoint. Among the eligible cuts in the upper half of the feasible
   interval the boundary is the one with (in order) the fewest deeper births
   affinely exposed, the fewest internal births affinely exposed, not within
   `avoid_radius` of a previous boundary, the largest signed clearance from
   the union of the exposure intervals, the fewest direct copies, and the
   latest. A single gate that exceeds the symbolic limits is its own block
   and is counted (`single_gate_exceptions`). The output is the list of
   boundaries, `blocks.tsv` (per block: start, end, feasible end, stop
   reason, terms, the exposure counts and the clearance at the boundary) and
   a summary.
3. **Synthesize** (`synth::synthesize_block`). Every block, for every one of
   `seeds_per_block` seeds (derived from the stage seed, the round and the
   seed index; never printed), in parallel over blocks: its ANF is composed
   from its gates over the global wires (abort on `compose_cap` terms or
   after `--compose-time-limit` s, 300 by default, the prototype's
   composition budget), remapped to the local universe (the variables occurring, the
   changed wires and an ancilla pool of 32 outside wires), synthesized by the
   bidirectional ANF search with `restarts` restarts, `beam` lookahead and a
   wall limit of `time_limit` s for blocks of at most 1,000 gates, ×2 up to
   4,000 and ×4 above; the synthesized gates are checked against the original
   block on 2 × 1,024 random full states (every wire), the pairwise peephole
   pass is applied and re-checked. A block converges (`ok`) or fails
   (`compose_overflow`, `compose_timeout`, `stepcap`, `timeout`, `stalled`,
   `verify-fail`; a synthesizer panic is caught and becomes `crash`), and a
   failed block stays as it was.
4. **Refine.** A block longer than `min_refine_gates` that converged under
   no seed is cut in two at a cut of its middle half chosen by the input
   census (`planner::Profile::split`: fewest deeper, fewest internal, largest
   clearance, fewest direct, nearest the midpoint) and both halves are
   synthesized; `refinement_levels` times.
5. **Accept** (`accept = "smaller-not-leakier"`). For every seed the
   candidate assembly (its converged syntheses, the original block where it
   failed) is censused (discovery only, as the prototype did; the input and
   output censuses are audited); per block, among the original and the
   strictly smaller converged syntheses, the one that exposes the fewest
   deeper births at cuts strictly inside the block wins, size breaking
   ties; a synthesis that exposes more deeper births than the original is
   rejected as leakier, one that does not shrink as growing. The honest-slice
   state at every retained boundary is fixed, so a block's interior exposure
   depends on that block's candidate alone, which makes the per-block choice
   sound. `accept = "smaller"` keeps the smallest converged synthesis
   regardless of exposure (the "as-is" acceptance).
6. **Assemble and check.** The chosen blocks are concatenated; the output is
   checked equivalent to the round's input on 8 × 1,024 random full states
   (a mismatch is exit 4), censused and measured (`metrics`: internal and
   deeper births exposed per family, the cut-exposure fraction, discovery
   and validated), and written as `round<r>/output.mpmct1` with
   `round<r>/result.report`.

## 3. The run

Rounds continue while the circuit shrinks. The run ends (`stop_reason`):

* `two consecutive rounds without size decrease`;
* `two consecutive rounds with joint compression below f` when
  $\text{output}_r \ge (1 - f)\,\text{input}_{r-1}$ (`stop_fraction`, 2 % by
  default; the campaign's attacker rule used 1 %);
* `capped` after `max_rounds` rounds (0 = no cap).

The **final circuit** is the smallest round output (the input when no round
shrank). It is put into the canonical commuting gate order
(`xgate_canonicalize`, `canonical_order = true`; checked equivalent on
8 × 1,024 random full states) under a guard: the reordered circuit is
censused, and if its validated deeper exposure (affine or direct) exceeds
the final round's, the un-reordered round output is published instead
(`canonical_order_applied=false`, `canonical_order_reason=exposure
increased (…)`; the reordered circuit's census is kept as
`final_canonical_census/`), since the acceptance rule was evaluated on the
round output's cut structure and a commuting reorder changes which values
are simultaneously live. The published circuit is then checked on the zero slice against $C$ (4,096
inputs: $x$ on wires $0..n$, zeros elsewhere, $C(x)$ on wires $n..2n$) and
against the sandwich's payload contract in both directions (8,192 random
plus the structured inputs of the prototype's `clean_payload`: forward, $x$
on the low $n$ wires and zeros elsewhere reproduce the sandwich's upper
half; inverse, the same on the reversed gate lists), censused once more, and
written as `adaptive/final.mpmct1` with `adaptive/adaptive.report`.

Resumability: `boundary.txt` and every completed round (`result.report` +
`output.mpmct1`) are reloaded, not recomputed; `progress.txt` is rewritten
after every round. The driver (`scripts/tdp_gen.sh`) resumes an interrupted
attempt of the same recipe in place (`adaptive/attempt.recipe`) and removes
`adaptive/` for any other rerun.

## 4. Inputs, outputs, exit codes

```text
tdp_adaptive run --input G.mpmct1 --source-c C.g57 --n N --out-dir DIR [--sandwich S.mpmct1]
    [--seed S | --seed-file PATH] [--threads T] [--max-rounds N=0] [--stop-fraction F=0.02]
    [--seeds-per-block K=2] [--time-limit S=60] [--restarts R=16] [--beam B=6]
    [--term-cap 131072] [--max-block-gates 65536] [--segments 16] [--avoid-radius 64]
    [--refinement-levels 2] [--min-refine-gates 128] [--census-batches 96] [--probe-inputs 8192]
    [--accept smaller|smaller-not-leakier] [--blind] [--no-canonical-order] [--keep-round-circuits]
    [--compose-time-limit 300] [--compose-cap 1048576] [--ancilla-pool 32] [--equiv-batches 8] [--no-postopt]
tdp_adaptive census --input G.mpmct1 --source-c C.g57 --n N --out-dir DIR [--census-batches 96]
```

`--blind` plans and splits against an empty census (the C-blind attacker of
the resilience campaigns; the exposure is still measured). The binary reads
no environment variables; the seed comes from `--seed-file` (the driver
passes `/dev/fd/3`) and is never printed.

Outputs in `--out-dir`:

```text
boundary.txt                     C's birth classification (cached across runs)
round<r>/input_census/           the round's input census (reloaded when its content stamp matches)
round<r>/partition/              the plan: bounds, blocks.tsv, summary
round<r>/synth0/, refine<L>/     per (seed slot, budget) synthesis rows (.tsv) and gates (.mpmct1 cache)
round<r>/candidate_s<k>_census/  the census of seed k's candidate assembly (smaller-not-leakier)
round<r>/output_census/          the census of the round's output
round<r>/output.mpmct1           the round's output (non-final rounds pruned at the end unless --keep-round-circuits)
round<r>/bounds.out              the output's interior block cuts, one per line
round<r>/result.report           the round's completion record (a round with it is reloaded, never recomputed)
progress.txt                     per-round gate counts and deeper births, rewritten after every round
final_census/                    the census of the published circuit
final.mpmct1                     the published circuit
adaptive.report                  the completion record
adaptive.log                     every progress line, appended across resumes
```

`adaptive.report` is flat `key=value`: `adaptive_report=1`, `status`,
`tool`, the input paths and `n`, `width`, `input_gates`, `final_gates`,
`final_round` (a round number, or `input` when no round shrank), `rounds`,
`stop_reason`, `plateau`, every parameter as `param.<name>`
(`seed_source` names where the seed came from, never its value),
`boundary.{internal,excluded,deeper}`, per round
`round.<r>.{input_gates,output_gates,blocks,initial_blocks,synthesized_blocks,picks_orig,picks_seed,rejected_leakier,rejected_growing,deeper_affine,deeper_direct,reloaded,seconds}`,
the leakage of the stage input and of the published circuit
(`stage5_leakage.*`, `final_leakage.*`:
`{discovery,validated}.{direct,affine}.{all_internal_births,deeper_births,cut_exposure_fraction}`
and `audit_passed`), the checks `check.input_zero_slice`,
`check.input_payload_contract`, `check.equiv_canonical`,
`check.zero_slice`, `check.payload_contract` (`ok(N)`, `skipped`, or
`reloaded` for a completed run's rerun), `canonical_order_applied`,
`canonical_order_reason`, `final_reloaded` and `seconds`. `round<r>/result.report` holds the same per-round keys in full
(`refinement.<L>.*`, `picks.*`, `coverage.*`, `leakage_in.*`,
`leakage_out.*`, `equivalence_samples`, `output_boundaries`). No seed, no
hash of $C$ or of a secret intermediate appears in any report.

Exit codes (the driver prints them with their meaning and `step=adaptive`
in `delivery.failed`): 0 finished, 2 usage or parameter error, 3 input
error, 4 verification failure (a block that does not synthesize is not an
error: it stays as it was), 7 I/O error writing outputs, 8 internal
invariant violated, 101 panic.

## 5. What was measured

Server campaigns of October 2026 (n = 64, |C| = 1000, 160 cores shared with
other jobs; [notes/stage2_stage6_campaign_20261021.md](notes/stage2_stage6_campaign_20261021.md),
[notes/pipeline_variants_20261026.md](notes/pipeline_variants_20261026.md)):

| stage 2 / stage 6 | plateau gates | rounds | deeper exposed in the published circuit (direct / affine, of 882) | C-blind resynthesis | narrow rewrites |
| --- | ---: | ---: | --- | ---: | ---: |
| embedded masking L8 / as-is | 666,745 | 24 | 384 / 424 | 503 | 444 |
| three-mask as-is / v2 | 2,368,032 | 14 | 0 / 0 | 9 | 264 |
| three-mask modified / v2 | 2,706,063 | 9 | 0 / 0 | 6 | 157 |

A round of the v2 variant on the ~1.9-3.5M-gate stage-5 circuits took
several hours on 24 threads under the server's load (planning minutes,
synthesis the rest). Without stage 3 the three-mask circuit was
ANF-incompressible (the synthesizer returned ~3.6× larger blocks, every
candidate rejected as growing, the 2 % rule fired after two rounds). The
blind plateaus are not size floors: a C-blind shortening campaign took the
baseline's 351,744-gate blind plateau to 274,097 gates with small blocks and
`fcompress`.

## 6. Deviations from the prototype

* The boundary classification (which births are public quadratic) accepts
  a degree-≤2 candidate that survives `probe_inputs` fresh inputs (8,192 by
  default) without the prototype's SAT proof, so a birth the prototype's
  solver left `unresolved` (and therefore counted as deeper) is excluded
  here; the classification only steers the "deeper" counts and the
  planner's first key.
* The census audit draws its fresh and structured inputs over all $n$ input
  wires (the prototype's script fixed 64), so at n = 128 it is stricter and
  the validated counts can read lower than the prototype's; planning,
  acceptance and the stop rules use the discovery intervals, as the
  prototype did.
* The synthesizer runs in process (one block per thread) instead of in
  separate processes under an address-space cap; the composition is bounded
  by `compose_cap` terms (8 × `term_cap`, where the prototype allowed 32×
  for a product) and `--compose-time-limit` (300 s, the prototype's) instead;
  a synthesizer panic is isolated per block (the prototype retried the
  chunk and isolated the window).
* The per-round and per-seed synthesizer seeds derive from the stage seed
  (the prototype used 1, 2, 3, …).
* The final circuit is canonically reordered by default under the exposure
  guard above (the prototype published the raw assembly).
* The input is checked against $C$ and the sandwich before round 1 (the
  prototype ran `clean_payload` after every round); the candidate censuses
  are discovery-only like the prototype's; cached syntheses and plans are
  reused whatever the parameters (the managed driver starts afresh on any
  parameter change through its attempt marker; a standalone rerun with
  other knobs needs a fresh `--out-dir`).
* The payload contract check against the sandwich is part of the run (the
  prototype ran it as a separate step), and the equivalence checks use the
  crate's lane evaluator.
