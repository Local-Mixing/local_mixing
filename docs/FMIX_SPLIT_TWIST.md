# FMIX_SPLIT_TWIST — the split stage of phase B

Status: spec agreed 2026-08-05 (this file). Production part 1 is implemented in
`src/postprocessing/splitting.rs`; the part-2 crossing walk is implemented in
`src/postprocessing/cross_walk.rs`. Both operate on shared `engine::Mixer` state.
Companion docs: `FMIX_MENU.md` (layer-1 menu), `FMIX_LAYER1.*` (slots), the
phase-B reframe (2026-08-04): phase B's job is to BREAK the g57 structure
(anti-inversion) with absolute spread/reach — not to mix state.

Phase B is now partitioned in two parts:

- **Part 1 — the split stage** (this spec): split every g57 into a
  (CNOT/NCNOT, 2-control AND) pair while carrying long-range **pure NOT
  twists** whose brackets are *absorbed* into existing gates, plus one cross
  move per split for transport. Runs to saturation, then trips over.
- **Part 2 — the current phase B**: the existing menu (twists, DB, thermostat)
  under whatever parameters the command line set.

## 0. Input contract

The input circuit consists ONLY of g57s (`comp=1`, controls a conjunction of
polarized literals under a global NOT) and X-series gates (`comp=0`, target ^=
conjunction of polarized literals). Every move below preserves that closure:
splits emit conjunctions, twists flip literal polarities, crosses emit ladder
conjunctions (an R1 colliding g57 survives with flipped rung polarities only).
This closure is load-bearing for part 2 — it is WHY step 5a force-splits
segment g57s instead of flipping their pins in place (which `conj_by_not`
could do exactly): the stage must leave no g57 that a later pass would have to
special-case, and splitting along the twist's path is the structure-breaking
the stage exists to do.

## 1. The core identity (verified)

"Active wire" of a gate means its **target**. Two facts make the twist free:

- A gate targeting `w` never reads `w`, so it **commutes** with `X(w)`.
- Composing `X(w)` with a 1-control gate targeting `w` IS that gate with its
  control polarity flipped: `(w ^= lit(b)) · (w ^= 1) = (w ^= ¬lit(b))`.
  CNOT ↔ NCNOT. The X is absorbed, not moved.

So for brackets `g1 … h1` both targeting `w` (any distance apart):

```
g1♭ · S′ · h1♭  =  g1 · X(w) · S′ · X(w) · h1  =  g1 · S · h1
```

where `♭` flips the bracket's control polarity and `S′` is the segment with
every *w-reading* pin's polarity flipped (gates targeting `w` inside the
segment are invariant). Function preserved, ZERO synthetic gates — strictly
better than the retired free-standing NEG twist (two X gates + bracket-cancel
tabu). The value carried on `w` is complemented across the whole span: local
pieces stop being locally explicable, and the compensation may live across the
circuit midpoint.

## 2. The split-twist move

One move (`split_twist_move`), dispatched from the twist slot:

1. **Pick** a uniformly random g57 `g` (target `w`). If none exists anywhere,
   the stage is over (exit A, §4).
2. **Split** `g` by the first-failing-literal presplit with randomized literal
   order (this is exactly `rules::presplit`; the random order IS the `r` bit
   of the design). For the canonical 2-control g57 the two cases are
   `g: a ^= (b ∨ ¬c)` → `{a^=b, a^=¬b¬c}` or `{a^=¬c, a^=bc}`.
   `g1` := the 1-control piece, `g2` := the last (widest) piece.
   Directions: `g1` gets a fair random direction, subsequent pieces alternate
   (so `g2` is opposite — see §6). Meta: pieces inherit origin and litter,
   share a fresh event, `dgen = child_gen(parent)`. No birth transport: pieces
   stay at `g`'s position (deviation from the cross presplit, deliberate —
   `g1` must sit where the bracket forms; `g2`'s transport is step 6's cross).
   The size-damper (`split_allowed`) is bypassed: splitting is the point.
3. With probability `1 − p_join` the move **ends here** (no twist, no cross).
4. With probability `p_join`: attempt the **absorbed NOT twist** on wire `w`.
   Bracket 1 is `g1`. Bracket 2 comes from a **directional, length-biased
   draw** (v2, 2026-08-05 — replacing the original other-half-first cascade,
   whose hard preference made midpoint crossing a constant 100%):
   candidates are the bracket-eligible population on `w` = { g57s targeting
   `w` } ∪ { 1-control non-comp gates targeting `w` } minus `g1`, restricted
   to ONE side of `g`'s position. The side is **drawn with probability
   proportional to the circuit length remaining on it** (v3 — this replaced
   the own-stored-direction rule, whose edge-adjacent primaries produced a
   spike of near-zero spans: a tiny span now requires a short side AND the
   proportional coin to pick it, a squared suppression; expected span rises
   to ≈4/9 of the circuit under k=2). Comp and 1-control candidates compete
   equally. Among
   `split_reach_k` uniform samples from that side the **farthest** (rank
   distance) wins: k=1 is uniform, k=2 ≈ 2/3 of the available run, k=3 ≈
   3/4. A comp winner is split as in (2), its 1-control piece is `h1`; a
   1-control winner is `h1` directly. No candidate on that side → the twist
   **fails** (the split of `g` remains) and the consecutive-failure streak
   increments. Any twist success resets the streak; moves that end at step 3
   leave it unchanged. (Sides and distances read the approximate rank
   stamps; candidates born since the last stamp are invisible until the
   next, and stamps refresh on >25% growth as well as the move cadence.)
5. On success, with `L,R` the two brackets in circuit order, walk `L → R`:
   a. every g57 **reading** `w` in the open segment is split as in (2)
      (in place, no transport, alternating directions, inherited
      origin/litter, fresh event per split);
   b. every gate reading `w` in the open segment — including pieces from (a)
      that read `w` — has its `w`-pin's polarity flipped;
   and both brackets absorb: `g1`'s and `h1`'s control polarity flips.
   Gates targeting `w` inside the segment are untouched (invariant).
6. **Cross**: perform one ordinary cross move shot from `g2` (it is a pure
   conjunction, so the cross precondition holds). Runs whether or not the
   twist succeeded — but not when step 3 ended the move.

Litter/ancestry: splits carry litter and ancestry exactly as presplits always
have (inherit parent litter; no union). The polarity flips of the twist are
deliberately **not** recorded in litters/ancestry — the crosses carry
lineage; canaries (§5) are the twist's instrument. In-place pin flips bump the
node stamp (via `replace_gate`), so stale undo-journal entries die.

`local_verify` checks, per touched gate over its ≤4-wire support:
`[X(w), piece_set] ≡ [flipped piece_set]` for segment work and
`[X(w), bracket] ≡ [bracket♭]` for absorptions; presplits verify as before.

Edge cases (accepted, not special-cased): a zero-length segment is a valid
degenerate twist; bare `X(w)` gates are not bracket-eligible (the input has
none); a width-1 comp gate presplits to a single 1-control piece which serves
as `h1`.

## 3. Layer-1 / layer-2 embedding

Layer 1: the split twist is a third dispatch inside `twist_round`, selected
by `p_split_twist` (checked before the `twist_g57` / swap-family choice).

Layer 2 (`--split`): the run starts with the split stage ON, which forces
`p_twist = 1, p_split_twist = 1` — the split twist is the ONLY move running
(no brake, no overlay, no shuffle, no DB, no thermostat; the forced cross of
step 6 is part of the move). The stage ends (§4) by setting the live
`p_split_twist = 0` and releasing every slot to the command line's phase-B
parameters — the same pattern as the size brake steering `db_mode_cur`. The
zero is binding for the rest of a `--split` run: part 2 performs no further
split twists even if `--p-split-twist` was set (the CLI value is only the
standalone, no-`--split` layer-1 mode). `--split-stop` ends the RUN at the
stage boundary instead (trial mode): `MixStop::SplitDone`, with state/output
written as for any clean stop. `--split` and `--profile` are mutually
exclusive by assert — both claim round authority; run the stage as its own
invocation and profile a later one.

Resume semantics: the state file records the stage phase as a tri-state —
never armed / live / ended. A LIVE stage continues only when the resume line
repeats `--split` (a loud warning fires otherwise); an ENDED stage never
re-arms, `--split` on the resume line just continues part 2; a never-armed
state resumed with `--split` starts the stage fresh on the resumed circuit.

## 4. Stage exits

- **A. g57 exhaustion**: step 1 finds no comp gate. (In practice the dominant
  exit: segment splitting depletes g57s in large sweeps, while every split
  mints a permanent 1-control bracket, so 4e failures get RARER over time.)
- **B. failure limit**: `--split-fail-limit` consecutive step-4e failures.

Both exits print a stage summary (splits by kind, joins, midpoint crossings,
growth, canary histogram) and behave identically afterward.

## 5. Instrumentation

Counters (appended to the state counter line, zero-defaulted on old lines):
`split_prims` (step-2 splits of the picked g), `split_hsplits` (bracket
splits), `split_segs` (5a segment splits), `split_joins` (twist successes),
`split_fails` (step-4 failures), `split_xmid` (successes whose brackets sat
in different halves — informational; under the directional draw it is a
measurement, not a target), `tap_flips` (total canary flips), and
`split_span_sum` plus a 5%-bucket span histogram (span = gates strictly
between the brackets, normalized by the size at that move; summary printed
at the boundary). A `[fmix] split:` line prints every split-twist move with
its span (the stage is short — thousands of moves on production sizes, since
each long twist retires thousands of g57s).

**Canaries** (`--split-canaries K`, 0 = off): a canary sits on wire `w`
immediately to the right of an anchor gate, planted at stage start on `K`
uniformly random gates (wire drawn from the anchor's touched wires), and
remembers its original position as a permille of the circuit. A twist whose
bracket span covers the canary's position on its wire complements the value
carried there: the segment walk of step 5 bumps every covered canary (anchor
in `[L, R)`). Canaries ride the material: when their anchor dies (any splice)
they re-anchor to its live left neighbor at death time. At stage end (and in
the state file) each canary reports `(wire, original position, flips)`;
the summary prints a flips-by-original-decile histogram — the direct read on
whether twist reach is absolute or clusters, which is the reframe's question.

**Position stamps**: halves and positions come from a rank vector restamped
by an O(n) walk every 8192 moves (and at stage start). Ranks are heuristics
(candidate-half preference, crossing counter, canary positions); staleness
between stamps is accepted. Correctness never depends on ranks: the segment
between brackets is found by an alternating bidirectional walk from `g1`
(cost ≤ 2× the segment length the flip pass pays anyway).

## 6. Presplit direction convention (behavior change, all presplit sites)

Previously every presplit piece drew its direction independently from the
`dir_p` law. That was a bug: siblings could agree. The convention now, at
EVERY g57 split (cross-move shot presplit, colliding presplit, and all
split-twist splits): the first piece draws a fair random direction and
subsequent pieces alternate — a 2-control split always yields opposite
directions. `dir_p` still governs cross-rewrite fragments, but no longer
touches presplits. Pre-2026-08-05 walks are not move-for-move reproducible
under the new binary.

## 7. State file v2

`STATE_VERSION = 2`. The reader accepts v1 (missing sections default: split
stage off, no canaries) and v2; writers always emit v2; older binaries refuse
v2 by the existing version check. New in v2, after `canary_failures`:
a `split <on> <streak>` scalar line, and after the `ancsets`/`anctracers`
sections a `staps N` section of `wire orig_permille flips rank` lines
(canaries re-anchor at load by walking to `rank`). New counters append to the
counter line per the existing "append, never insert" rule.

## 8. CLI

```
--split                 arm the split stage (forces p_twist=1, p_split_twist=1 until exit)
--split-stop            end the run at the stage boundary (trial mode)
--p-join F              probability a split carries the twist+cross     [0.8]
--split-fail-limit N    consecutive bracket failures that end the stage [100]
--split-canaries K      wire canaries planted at stage start            [256]
--split-reach-k K       bracket length bias: max-of-k by distance       [2]
                        (0 = the original other-half-first cascade, kept
                        as the A/B comparison arm)
--p-split-twist F       layer-1 dispatch weight outside --split         [0.0]
```

Launch hygiene applies unchanged: export `FMIX_STOP_FLAG`/`FMIX_DUMP_FLAG`,
size `--moves` to the profile (the stage spends ONE move per split-twist:
budget a comfortable multiple of the g57 count), pin every measured knob.

## 9. Expectations to check in trials — CONFIRMED 2026-08-05

All confirmed same-day on five inputs (99k–1.66M gates, 5–99% g57):
growth = 1 + comp-fraction (×1.7–2.0 measured); exit always by exhaustion
(failure limit untouched); segs carry ~99% of depletion; canary coverage
mid-humped with no origin clustering; output = 50/50 CNOT/NCNOT + 1:2:1
AND2, zero comp. Full numbers: reports/split_trials_20260805/RESULTS.md and
docs/SPLIT_TWIST_REPORT.pdf. Original expectations follow.

- Growth ×2–3 over the stage (closer to ×2), all of it split pieces plus
  cross ladders; nothing contracts until part 2.
- Long segments retire g57s wholesale: expect FEW moves, each a large sweep;
  `split_segs ≫ split_prims`.
- Exit A (exhaustion) before exit B (failure limit) on realistic inputs.
- Canary flip counts roughly flat across original-position deciles if reach
  is genuinely absolute; a hump near the ends or middle is the thing to see.
