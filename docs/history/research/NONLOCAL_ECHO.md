# Nonlocal echo pairs: the P-explicit branch of the bridge move

Status: DESIGN (2026-08-16), adversarially reviewed same day (feasibility +
security, both SOUND_WITH_FIXES; fixes folded in below). Companion to
`docs/history/research/NONLOCAL_PHASE_A.md` (far-pair + bridge fusion, built 2026-08-15,
uncommitted). **Nothing here is implemented, and the recommendation (§8) is NOT
"build it now" — it is "fold into the bridge and gate on the bridge's own
pending measurement."** Read §1 and §8 first.

## 1. What this is, and its honest relationship to the bridge

A move that replaces two gates `g1` (position `i`) and `g2` (position `j`),
`j - i` large, where the joint replacement is a function of **both** gates and
of the interior `M = gates(i+1 .. j-1)` — while the interior itself stays
**byte-identical**. Functionality is genuinely broken at every prefix strictly
between the two sites and restored exactly at the far site.

This is not a new idea layered on the existing work — it is the **unbuilt
branch of the solution space the bridge move already lives in**. From
`NONLOCAL_PHASE_A.md` / `src/engine/mix.rs:7413-7414`, every correct two-site
rewrite has the form

```
left site:   g1  ->  g1 · P                     (any perturbation P)
right site:  g2  ->  conj_M(P^-1) · g2
```

- **Bridge (built):** `P = u`, one self-inverse 2-control conjunction carrier.
  `conj_M(u)` is kept **implicit** by conjugating the interior — every collider
  `h` gains 1–2 "wake" correction gates. Interior is adjusted, not preserved.
- **Echo (this doc):** `P` is **free** (chosen from the two endpoint gates),
  and `conj_M(P^-1)` is expanded **explicitly** into a short gate word — the
  **echo** `E` — spliced at the far seam only. Colliders distort `E` instead of
  being touched. Interior is byte-identical.

Echo's transported-`P` family (any short word) is strictly larger than the
bridge's (one 2-control conjunction), so echo is genuinely distinct, not a
subset. **But — the security review's central finding — against both currently
measured threat models, echo and bridge are equivalent** (§7). The one property
that distinguishes them, the byte-identical interior, only matters to an
*unmeasured* structural attacker, and even then its signal is likely buried
(§7). So the case for echo is real but unproven, and the sensible path is to
build it as a *branch of the bridge*, not a separate move (§8).

| | SAMF (`--single-end`) | bridge (built) | **echo (this doc)** |
|---|---|---|---|
| transported `P` | signed wire permutation (affine) | one 2-control conjunction | any short word (nonlinear, gate-dependent) |
| interior | every downstream gate relabeled | every collider gains wake gates | **byte-identical** |
| sites touched | all gates after insertion | 2 windows + all colliders | **2 windows only** |
| restore point | end of circuit | far carrier copy | far seam (chosen `j`) |
| distinct vs bridge in *measured* threat models | — | baseline | **none (§7)** |

SAMF is the degenerate case "P = affine relabeling, pushed to the end." All
three telescope off the same identity.

## 2. The algebra (machine-verified)

Circuit order (left-to-right application), all gates involutions (target never
among controls, enforced by `xpoly.rs:284` / `TargetInControls`). Let
`S = [g1] ++ M ++ [g2]`. For any perturbation word `P`:

```
[g1] ++ P ++ M ++ E ++ [g2]   ==   S      where   E = echo(P, M)
```

and `echo(P, M)` is computed operationally, without ever composing `M`:

```
E := reverse(P)                      // = P^-1 for involutions
for h in M left to right:            // h is an ordinary interior gate, KEPT
    E := conj_h(E)                   // conj_h(E) = simplify([h] ++ E ++ [h])
```

`conj_h` distributes over the word (conjugation is an automorphism), so it is
computed elementwise. For one generalized element `u = (a ^= F)` and interior
gate `h = (t ^= G)` (fires as ANF; g57 `[a,x,y]` is `a ^= 1 ⊕ y ⊕ xy` per
`circuit/circuit.rs` / `circuit/xgate.rs`), with `a ∉ vars(F)`, `t ∉ vars(G)`:

1. `t ∉ vars(F)` and `a ∉ vars(G)`: commute — `conj_h(u) = [u]`.
2. `t ∈ vars(F)`, `a ∉ vars(G)` (h writes a wire u reads):
   `conj_h(u) = [(a ^= F[x_t <- x_t ⊕ G])]` — one element, ANF substitution.
3. `a ∈ vars(G)`, `t ∉ vars(F)` (u writes a wire h reads): write `G = G0 ⊕
   x_a·G1` (`G1` a-free); `conj_h(u) = [u, (t ^= F·G1)]` — u plus one
   correction whose fire reads neither `a` nor `t`, so ordering is sound.
4. both (mutual collision): closed forms self-reference; emit the exact
   sandwich `[h, u, h]` for that element (rare: O(1/n²) per crossing).

Growth is ≤2 elements per colliding crossing (3 for mutual), zero for
non-colliding crossings. Same-target neighbors *may* collapse (see the
representation note below).

This exact case analysis and the end-to-end identity were machine-verified in
`security_tests/support/echo_conjugate_growth_mc.py`: every `conj_h` output is checked against
brute-force evaluation of `h·u·h` on random states, and the full identity
`P ++ M ++ E == M` is checked on random states across random interiors. **The
first draft of rules 3/4 was wrong and the self-check caught it — keep the
per-step verification in any Rust port.** (Feasibility review confirmed the
involution/cancellation facts: `cancel_adjacent_duplicates` circuit.rs:1500-1516,
`merge_result`::Cancel mix.rs:345-347.)

**Representation note (corrected after review).** A Rust `XGate` is
`target ^= comp ⊕ (one product of literals)` — a *single* monomial
(`xpoly.rs:8-11`). My model's `(target, fire)` element carries a whole ANF, so
one model element with `k` monomials is **`k`-ish real XGates** (an ESOP of `k`
cubes on that target, one `comp` folded in). Two same-target neighbors collapse
to fewer XGates **only** through the `Merge`/`merge_result` monomial catalogue
(Cancel / XFuse / DropLit / Subsume / Absorb, `mix.rs:304-328`), which fuses
only when the XOR of their fires is a single (possibly complemented) monomial,
and even bans some complemented-result fusions (`mix.rs:362-366`). Multi-monomial
same-target pairs stay as separate XGates. **Consequence: the real gate size of
an echo is its total monomial count, not its element count** — §3 reports the
monomial column as the honest size.

Directionality: the mirror move (insert `P` before `g2`, push the echo backward
to `g1`) is the same computation on the reversed circuit; choose by coin like
the bridge's L/R scan.

## 3. Affordability — measured echo growth (honest units)

`security_tests/support/echo_conjugate_growth_mc.py`, 400 trials/cell, exact code gate
semantics, `collision` = `Gate::collides_index` truth (target-reads-pin), not
mere support overlap. P policies: `random` = one random g57; `overlap` = one
g57 with one control on a random interior-support wire; `pair` = two same-target
g57s sharing one control (the Tier-0 `A·A'` shape, §5).

Two quantities matter, and they are **different**:

- **What the store lookup sees** = the *function* of the far window `E ++ [g2]`:
  its **degree** and **support** (wire count). This decides whether the window
  keys into the frozen store and clears the `--db-max-degree 9` / ≤64-wire /
  span-30-distinct-wires guards. Measured p90 degree ≤7, median support 3–10 at
  accepted spans — under the cap, thin headroom.
- **The verbatim scar if the store misses** = the *gate size* of `E`, which is
  the **monomial count** (§2 note), not the element count.

Key cells (`hit%` = ≥1 real collision along the span; `mono` = total monomials
= real XGate size of the verbatim echo; `deg`/`sup` = far-window function
degree/support that the store guards actually test):

| n | span | policy | hit% | med mono | p90 mono | p90 deg | med sup |
|---|---|---|---|---|---|---|---|
| 256 | 16 | random | 17.0 | 3 | — | 3 | 3 |
| 256 | 32 | random | 39.5 | 3 | — | 4 | 3 |
| 256 | 64 | random | 67.0 | 7 | — | 5 | 5 |
| 256 | 64 | pair | 72.0 | 10 | — | 5 | 6 |
| 128 | 64 | random | 86.5 | 16 | — | 7 | 9 |
| 64 | 32 | random | 83.8 | 15 | 7 (p90 deg) | 7 | 7 |
| 64 | 64 | random | 98.8 | 186 | 12 (p90 deg) | 12 | 19 |

Read: the multiplicative blow-up wall sits at span ≈ `n` interior gates. Below
span ≈ `n/4` the far-window function stays low-degree (≤5) and small-support
(3–6 wires) — comfortably inside the store guards — and the verbatim scar is a
~3–10-cube ESOP, not a single gate. (An earlier draft said "median 1-gate
echo"; that conflated model elements with XGates and was wrong — the honest
verbatim size at span 64 / 256 wires is ~7 cubes.) Production phase-A material
is 4n wires (n=64 source → 256 wires, ~30k gates), so the usable regime is
spans in the low tens, where the move is genuinely affordable and ~40–70% of
random pairs collide en route. The `overlap`/`pair` policies raise the collision
rate; combined with the zero-collision reject (§4), the move is never *formally*
"just commutation" — though at production width the modal accepted move is a
single-collision echo, whose peel-resistance is exactly the unmeasured
structural-attacker question (§7).

## 4. The move, step by step

Slots into the fmix/postmix engine (what `scripts/gss_mix.sh` stage 3 drives) as
a new round-slot beside pair/bridge, `--p-echo` default 0, RNG-parity when off —
same discipline as `NONLOCAL_PHASE_A.md`. If built as a bridge branch (§8), it
shares the site-pick/probe/commit code and only the P-choice + interior-walk are
new.

1. **Pick sites.** Seed `g1` via `pick_seed`; span log-uniform in
   `[echo_min_span, echo_max_span]` (defaults 8..48 at 4n=256 wires — keep the
   span well inside the width wall, §3). `g2` = first gate past the span, like
   `bridge_plan` (`mix.rs:7444-7475`).
2. **Choose P** from the policy mix (§5) — a word of 1–2 XGates whose support
   overlaps `g1` (and, for teleport, `g2`).
3. **Compute the echo, with an explicit support guard (NEW code).** Walk the
   interior once, applying rules 1–4 over a `Vec<XGate>`. Per crossing, **first
   check `joint_support(before, after) ≤ 24` and refuse the move if not** —
   `rules::verify_rewrite` *panics* on support > 24 (`assert!` at
   `rules.rs:241`), it does **not** return false, and the bridge stays under the
   cap only structurally (2-control carrier + `k_max`-bounded corrections), so
   there is no existing refusal to inherit. For wide crossings prefer the
   `db_replace::polys_equivalent` ANF check (the `try_db_splice_curated`
   fallback pattern, `mix.rs:6636-6637`) over `verify_rewrite`. Refuse the whole
   move if: monomial count exceeds `echo_max_monos` (default ~8), OR zero
   collisions occurred (pure commutation — the trivial-transport analog of the
   phase-A reorder ban), OR the echo equals `reverse(P)` as ANF (collisions
   canceled).
4. **Probe both seams before mutating** (`bridge_round` discipline,
   `mix.rs:7591-7636`): near window `[g1] ++ P` and far window `E ++ [g2]` are
   XGate windows; `db_replace` canonicalizes any XGate window — sampled or
   constructed, no provenance distinction — via
   `canonicalize_xgates_single_capped(window, dir, budget, guard.max_degree)`
   (`db_replace.rs:604,691`), keying `xxh3_128(polys_repr_blob(polys))`
   (`db_replace.rs:37-39`), byte-compatible with the g57 store (`xpoly.rs:13-15`,
   test at `xpoly.rs:542-554`). A width/degree/budget overflow is a **clean
   miss** (`Err` arms `db_replace.rs:606-607,693-694`), not a panic, so a miss
   on either seam leaves no trace.
5. **Commit far-first with rollback** (`bridge_round` pattern,
   `mix.rs:7645-7679`: commit the far window `w2v` first, and on a far decline
   unlink/free every inserted gate for exact rollback; `bridge_insert`
   `mix.rs:7548-7584` only does the far-before-near *insert* ordering). If the
   **near** respell misses post-commit, the perturbation `P` was already spliced,
   so it simply stays verbatim — exact but unhidden, exactly as the bridge
   leaves its bare carrier as `bridge_half` (`mix.rs:7689-7691`). Meter it. Note
   Tier 0 (§5) has **no near splice at all** (the near site is an in-place
   `g1 -> g1'` edit), so no verbatim fallback applies there.
6. **Verify.** The move is exact by construction; run `global_check` sampling in
   tests, and keep the per-crossing check always-on (cheap at these supports).

Tag/generation bookkeeping: both seam windows mint tags as any local respell
does; additionally log `(i, j, span, collisions, |E|_monos)` — the ancestry-span
instrumentation the phase-A acceptance experiment already wants.

## 5. Choosing P from the two gates ("depends on both gates")

Policy mix, sampled per move:

- **Tier 0 — near-collapse (single-gate edit).** Choose a replacement gate
  `g1'` sharing wires with `g1`. Set `P = [g1, g1']`: then
  `[g1] ++ P = [g1, g1, g1'] = [g1']` exactly (involution cancellation,
  confirmed `cancel_adjacent_duplicates` circuit.rs:1500-1516) — the near site
  is a **one-gate in-place edit, no splice, no lookup, no fallback**, and the
  entire cost lands in the far seam `E ++ [g2]`. The perturbation's fire is
  `F_{g1} ⊕ F_{g1'}`, compact for a shared control — this is the cheap `pair`
  row in §3. To an observer: gate `i` silently became a *wrong* gate, and the
  interior-distorted difference is absorbed at `j`.
- **Tier 0b — pure pair swap (opportunistic).** After computing `E` for a Tier-0
  draw, probe the far window for a **1-gate** candidate (`E ++ [g2] -> [g2']`).
  On success the whole move is `g1 -> g1'`, `g2 -> g2'`, interior untouched,
  nothing else — the purest form of the requested primitive. Expected rare; take
  it when the store offers it.
- **Tier 1 — free dictionary.** `P` = one random XGate/short word with a literal
  on a `g1` wire and one on a wire some interior gate writes (the `overlap`
  policy) — maximizes genuine collisions at short spans. Both seams respell.
- **Tier 2 — teleport (UNMEASURED, see §7).** `P` = the backward-pushed image of
  `g2`, so `g2` is deleted at `j` and its interior-conjugated image fuses into
  the near window. Maximal dependence. **Caveat:** this makes *both* seams
  constructed multi-gate windows, so it doubles the unmeasured store-hit
  question of §8.1 — the §3 MC measured single-hop echo *size* only, not the
  near-window hit rate. Do not treat as free.

## 6. Relays and crossing spans (UNMEASURED extrapolation)

The §3 MC measured single-hop echo size only. The following are plausible but
**not measured**, and hiding degrades *geometrically* (not linearly) with
per-hop store-miss probability, because each hop is an independent constructed
window that can fall back to verbatim:

- **Relay chains.** For spans past the width wall, pick relay sites every
  ~`n/4` gates: at relay `r`, replace `E_in ++ [r] ++ P_next` by a store
  respell; the perturbation mutates at each relay, per-hop echo size resets,
  reach grows. But relays are *correlated* (`E_in` at hop `j` = `P_next` from hop
  `j-1`) and each carries its own miss probability, so the fraction of fully
  *hidden* chains falls off geometrically in the relay count — a relay is only
  "an ordinary local respell" when it hits. Measure per-relay hit rate before
  claiming affordable unbounded reach.
- **Crossing spans.** Echo moves compose sequentially on the current exact
  circuit, so interleaved spans `i1 < i2 < j1 < j2` need no new machinery — allow
  overlap in the span sampler. The composed circuit has no contiguous window
  containing a canceling pair, the structure pairwise-cancellation analysis
  cannot peel — but see §7 on whether that helps any *measured* attacker.

## 7. Security framing (hypotheses, and where the skeptic pushed back)

**No per-move hiding claim.** The whole-trace GF(2) affine span survives every
function-preserving rewrite, echo included
(`reports/global_trace_affine_postmix_20260812/REPORT.md:9`). What echo does and
does not buy, after review:

- **Entry-time / order channel — NOT an echo buy.** The blind entry-time attack
  is *already dead* after ordinary phase-A mixing, with no echo and no bridge:
  AUC 0.885 → 0.475 → 0.448, and "it never recovers"
  (`experiments/blind_recovery_20260814/RESULTS.md:71-75,116`), even with the
  store off. Echo cannot improve a metric already at the noise floor; echo's own
  acceptance treats entry-AUC as a *sanity* row that must merely *stay* in the
  no-signal band. Correct statement: **echo does not regress the entry-time
  channel that ordinary mixing already closes.**
- **Transport geometry — shared with the bridge, unproven.** Median ancestry
  span and `frac(≥3 descendants)` are echo's plausible payoff — but they are
  *identical* to the bridge's primary metric (`NONLOCAL_PHASE_A.md:48-51,205-207`)
  and that metric has **not yet been shown to move for the bridge either**.
- **Byte-identical interior — the only distinct property, and it is a hypothesis
  against an UNMEASURED attacker.** Against the two *measured* threat models it
  buys nothing over the bridge. Whole-trace affine: `REPORT.md:75` gives the
  exact reason echo cannot hide — "two identical nonlinear toggles implement the
  identity but expose the nonlinear firing function between the toggles," which
  is *literally* echo's structure (P after g1, E before g2, perturbed interior
  between); the untouched interior is exactly the original features the
  saturated-rank-2^n span already sees. Entry-time: both moves break every
  interior prefix. The byte-identical interior bites *only* against a structural
  attacker that fingerprints spans by counting non-g57/polf gates — and even
  there the bridge's marginal 1–2 wake gates per span are buried under the
  pipeline's bulk non-g57 gates (twist packets, split residues, fcompress). This
  attacker is not in the measured set, so the advantage is **unproven**.
- **polf accounting.** Store-respelled seams are pure g57 → polf 0; the
  verbatim-P fallback (§4.5) and raw XGate seams meter like bridge wake. Target:
  polf 0 on full-commit moves.

Sanity rows unchanged from phase A: blind entry-AUC stays in the no-signal band,
fcompress residual ≥90%, RNG parity when `--p-echo 0`.

## 8. Recommendation (revised after review): fold into the bridge, gate on its probe

**Do not build echo as a standalone move now.** The bridge is built but its
acceptance experiment has never run (no `far_pair_probe` or acceptance artifacts
exist under `reports/`/`experiments/`), and echo shares the bridge's single
decisive unknown: **store coverage of *constructed* product-form / seam
windows** (`§8.1` here = `NONLOCAL_PHASE_A.md:179-199` there — the same
question). Sequence:

1. **Run the bridge's `far_pair_probe` + acceptance experiment first** (fleet;
   store unusable over the laptop USB mount). It answers echo's #1 risk for free
   — if constructed windows miss the store, echo's far windows miss too and echo
   degrades to verbatim-unhidden seams (§10). It also settles whether the shared
   transport metric moves at all.
2. **If the bridge acceptance passes, add echo as a P-branch of the bridge**, not
   a separate subsystem: a per-move flag choosing P-implicit-wake (current
   bridge) vs P-explicit-echo, reusing `bridge_plan` site pick, both-window
   probe, far-first commit, and rollback (`mix.rs:7586-7659`). Only `conj_echo`
   (the general-P interior walk, §9) and the P-choice policy (§5) are new. This
   makes the interior-fingerprint hypothesis (§7) a clean **A/B on identical site
   plans**: wake-interior arm vs echo-interior arm, same seeds, measured against
   the structural meters — the only experiment that can actually decide echo's
   distinct value.
3. **Drop echo** if `far_pair_probe` shows constructed windows miss the store, or
   if the bridge's transport metric does not move, or if the A/B shows no
   structural-meter difference between wake-interior and echo-interior.

The immediate deliverable is therefore the recon + this design + the verified
`conj_echo` algebra, *pending* the bridge measurement — not a new binary.

## 9. If/when built: implementation notes (fmix/postmix, as a bridge branch)

1. `conj_echo(h: &XGate, word: &mut Vec<XGate>)` in `src/engine/mix.rs`: rules
   1–4 of §2 over ANF fires; mutual case emits `[h, u, h]`; same-target collapse
   via the existing `Merge`/`merge_result` **monomial catalogue** (Cancel /
   XFuse / DropLit / Subsume / Absorb — *not* a general ESOP merge; multi-monomial
   XORs stay as separate XGates, §2 note); **explicit `≤24`-support guard before
   every `verify_rewrite`** (it panics, §4.3), with `polys_equivalent` fallback
   for wide crossings. This is genuinely new code — `conj_wake`
   (`mix.rs:204-246`) is hard-specialized to a 2-control non-`comp` carrier
   (`debug_assert!` at `mix.rs:205`) and conjugates in the opposite role
   (`u·h·u`, not `h·E·h`), so it does not cover the general-`h` case. Budget
   ~150 lines + a test mirroring the `conj_wake` 6000-pair check.
2. `echo_plan` / `echo_insert` / `echo_round` modeled on
   `bridge_plan`/`bridge_insert`/`bridge_round`, or better, added as branches
   inside them (§8.2).
3. Flags in `src/bin/fmix.rs`: `--p-echo`, `--echo-min-span`, `--echo-max-span`,
   `--echo-max-monos`, `--echo-policy-weights`, `--echo-relay-stride`
   (0 = single-hop). Default-off; RNG-parity test.
4. A new `experimental/db_mixing/echo_probe.rs` (with an explicit Cargo target)
   — or an extension of `far_pair_probe` — for §8.1 store coverage of
   constructed near+far windows and the Tier-0b 1-gate-collapse rate.
5. Tests: §2 identity on random circuits; empty-store no-trace; `global_check`
   battery; RNG parity.
6. Later, only if the A/B justifies it: relays/crossing spans (§6), Tier 0b
   opportunistic probe.

**Out of scope for any v1:** the sss/ssg engine — its splice/Tag machinery is
strictly contiguous (`contiguous_convex` `security_tests/support/db_mixing/convex.rs:790-878` evicts interior
non-members to make a single interval, tags in lockstep at `:840,854`; no
two-site/far-seam concept), so retrofitting two-site transactions there is
invasive. Also out: fragmentation-stage interaction, and any claim about the
whole-trace affine channel.

## 10. Risks / open questions

- **Store misses on constructed windows (§8.1) — the main unknown, shared with
  the bridge.** If far windows rarely hit, echo degrades to verbatim-P seams
  (exact, interior-untouched, but polf>0 and more visible) that later rounds must
  smear. Mitigation: Tier-0 near-collapse halves the lookups; keep `P` small so
  `E` stays a low-degree few-cube function in well-populated store classes.
- **Degree cap headroom is thin.** Far-window function degree reaches ~7 at p90
  for the larger accepted spans; the guard is 9 — meter degree-cap refusals and
  keep spans low (§3).
- **`verify_rewrite` panics past 24 wires** (`rules.rs:241`) — the explicit guard
  in §4.3/§9.1 is mandatory new code, not inherited from the bridge.
- **Mutual-collision sandwiches** copy an interior gate into `E` (3-element
  emission); hot-wire pileups could inflate `E` past budget — the monomial-budget
  refusal handles it; meter the rate.
- **Single-collision modal move (§3).** At production width the typical accepted
  echo has one collision and a ~3-cube far window — formally non-trivial but
  structurally thin; whether that resists a span-locating attacker is the same
  unmeasured question as §7. Consider requiring ≥k collisions or a minimum
  far-window degree if the A/B shows single-collision moves are peelable.
