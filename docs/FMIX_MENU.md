# fmix: the round menu (layer 1)

*Design spec, 2026-07-27. Covers the per-round move menu only. The phase
overlay — the rule engine that changes parameters mid-run — is layer 2 and is
specified separately.*

This replaces the round scheduler documented in `FMIX_PHASE_A.md` §3 and the
`fmix` CLI in `POSTMIX_MANUAL.md` §2.1. Where a value below differs from the
current implementation, the current one is noted in §9.

---

## 1. The round

Every move is one round. Four slots, evaluated in order; the first that fires
consumes the round.

| # | slot | fires with | effect on size |
|---|---|---|---|
| 0 | **conditions** | always | none |
| 1 | **twist** | `p_twist` | +2 / +6 before absorption (§3) |
| 2 | **DB move** | `p_db` | depends on `db_mode` |
| 3 | **thermostat** | otherwise | ±small |

Slot 3 runs **iff the slot-2 coin did not fire**. A slot-2 round whose descent
found nothing is spent — there is no fallthrough. So the thermostat receives
exactly `(1 − p_twist)(1 − p_db)` of rounds, independent of how hard the
material is.

**Slot 0 — conditions.** Evaluate the rule list. Each rule is a predicate over
run state with one of three actions: *stop*, *advance to phase P*, or *set
parameter X = v*. Predicates compose with AND/OR. Cheap predicates (counters)
may be evaluated every round; predicates requiring an O(size) census run on
their own cadence. The full condition vocabulary belongs to layer 2; §6 defines
the one predicate intrinsic to the menu.

**Slot 1 — twist.** One conjugation twist. The type is drawn from
`w_twist_neg` / `w_twist_swap` / `w_twist_cnot` as ratios (neg/swap 50/50 when
all are zero). Window length is log-uniform over
`[twist_min_len, |circuit|]`, with the virtual start drawn symmetrically so
head and tail accumulate coverage equally. Endpoint *placement* is not uniform
— see §3.

**Slot 2 — the DB move.** §2.

**Slot 3 — the thermostat.** §4.

---

## 2. The DB move

One operation. Three admission rules, selected by `db_mode` at slot 2 and
fixed by role on the thermostat branches.

### 2.1 Window selection

1. Draw a seed gate `g` (§5).
2. Sample a window of `s_db` gates around `g`: **convex** with probability
   `p_convex`, else **contiguous**. Both gather the block until it is
   contiguous in the arena, so arena order totally orders it.
3. **Eligibility truncation.** A gate with `≥ w_window` controls is ineligible.
   If the window contains one, truncate to the prefix before the first
   ineligible gate.

### 2.2 The descent

Try the window; on failure drop one gate and retry, down to length 1 or an
empty window.

**The gate dropped is the one farthest from the seed `g`.** The window stays a
contiguous arena run either way, so splicing is unaffected, and `g` survives to
length 1. This matters because the descent's purpose under biased seeding is to
re-encode *that* gate: dropping from a fixed end instead would walk away from
the seed on roughly half of contiguous windows and every convex one.

With `s_db = 5` the descent gives `g` five chances: lengths 5, 4, 3, 2, 1.

**The bottom rung always pays.** A single XGate's permutation has exactly one
one-gate implementation — itself. So at length 1 the free branch is
unreachable: MIX-DB either finds nothing at all (a true store miss) or pays the
smallest non-identical spelling. Every descent therefore terminates in a paid
expansion or a genuine miss, which is what makes per-gate miss accounting
unnecessary (§5).

### 2.3 The identity guard

A candidate is *identical* to the outgoing window when their gate sequences
agree after mapping both through the canonical wire relabelling
(`CanonicalXPolys::order`, already computed for the lookup). Identical
candidates are never spliced, in **all three modes**.

Note this is a circuit-level comparison in a common wire frame, not a key
comparison: the store is keyed on the canonicalized *function*, so every
candidate shares the window's key by construction. Trivial reorderings that map
to different sequences compare as different and are taken — no benefit, no
harm.

### 2.4 The three modes

| mode | admission | selection |
|---|---|---|
| **ANY-DB** | any size | uniform over all non-identical equivalents |
| **MIX-DB** | free if possible, else pay | uniform over non-identical matches **≤ window**; if none exist, uniform over the **smallest** non-identical |
| **COMP-DB** | only if some match ≤ window | uniform over the **minimum-size** ones |

The asymmetry between MIX-DB and COMP-DB is deliberate. COMP-DB is a
contraction move, so minimum size is its job; MIX-DB is a re-encoding move, so
entropy is. MIX-DB therefore maximises the draw pool exactly when re-encoding
is free, and minimises cost only when it is not.

MIX-DB makes the ingest-versus-pay decision **per window, at the point of
use**, from the match list the lookup already returned. That is what replaces
the per-gate cheap/hard tier machinery entirely (§9).

**Where each mode is used:**

- **slot 2** — `db_mode`, a deterministic three-way flag settable by slot-0
  rules (§7.2). Not a probabilistic choice.
- **contract branch** — always COMP-DB. Forced: the branch must not grow.
- **expand branch** — always ANY-DB. Forced: the branch exists to grow.
- **twist placement** — a mode parameter of its own (§3.3).

### 2.5 CURATED

The curated store holds circuits with a minimality property — every strict
subcircuit is shortest — obtained by splitting minimal identities at every
point. A curated replacement is therefore one whose internal pieces are not
locally compressible, i.e. one that survives `fcompress`, the
attacker-computable compressor. That makes the substitution more likely to be a
*meaningfully different route* to the same permutation.

`curated` is a boolean.

- **COMP-DB ignores it** — regular store only.
- **off** — regular store only, in all modes.
- **on** — probe both stores and **always prefer a non-identical curated match
  to a non-identical regular one, regardless of size**; the mode's own
  selection rule then applies within the winning class.

Curated-ness is thus a lexicographic first key. Because the curated store is
*built* to contain longer equivalents, "prefer curated regardless of size"
systematically prefers growth. This is intentional: it buys re-encoding quality
with size.

Probe **both** canonical directions. (The ssg path in `replace/pairs.rs` probes
curated forward-only; that restriction is not carried over — both keys are
already computed here, so the second probe is free.)

The store is `/home/cc/frozen_curated_m1_m11`, 6.1 GB, 256 shards — 2% the size
of the regular store, and now co-located with it.

### 2.6 Litter rules (designed in ssg, not scheduled here)

The known fix for re-encoding *count* diverging from re-encoding *displacement*
(§7.2). Carried from ssg commit `80a2c1d2`; recorded here so the port is
specified when the need arises.

**Mechanism.** Every gate carries a **litter id** — the replacement event that
created it — plus that litter's size; input gates are singleton litters. Then:

- **Full-litter ban.** A window that is *exactly one complete litter* is
  refused. Such a window is the unit some earlier replacement emitted, so it is
  precisely where the store can hand the original spelling straight back:
  `A → B → A`.
- **Diversity preference.** Sample several candidate windows and take the one
  spanning the **most distinct litters** (ssg: `LITTER_WINDOW_SAMPLES`, default
  4; measured tier-1 average 3.29 distinct litters, ~165 full-litter bans per
  sweep).

**Why it matters.** In the controlled ssg pair, litter rules + slow compression
reached **floor generation 100 at 95.1% of gates** — no prior run had ever
passed floor 45–54 — while the legacy control on the same input sat at **floor
~19 with %-at-100 stuck near 30%**. Identical clock, different displacement.

**What fmix already has.** `Meta::event` tags split events, and the report
separates `sib=` (sibling merges within one split) from `xorig=`. So the
litter-aligned-churn signal is *measured* today and simply not acted on.

**What the port needs.**

1. **Tag DB splice products with a shared litter id and size.** The splice path
   currently sets `event: 0` ([mix.rs:2577](../src/postmix/mix.rs)), so slot 2
   — the dominant channel under this menu — emits untagged material.
2. **Reject full-litter prefixes in the descent** (§2.2), which composes
   naturally: a rejected prefix just descends one rung further.
3. **Prefer max-distinct-litter windows** among sampled candidates in §2.1.

**Two caveats on the port.**

- ssg's escape hatch was the *SAMF license*: a full-litter window could commit
  if the step landed a hidden SAMF, i.e. if fresh entropy came with it. fmix
  slot 2 injects no fresh entropy, so there is no analogue — either ban
  outright, or license a full-litter window when the replacement comes from the
  **curated** store, which is a structurally different route by construction.
- ssg's tags were **RAM-only and reset on resume**. Under §11's checkpoint
  requirement they must be persisted, or every resume silently re-baselines
  every litter to singleton and disables the ban.

**Relation to the identity guard.** Complementary, not redundant. The identity
guard refuses replacing a window with itself; the full-litter ban refuses
replacing a *unit that was emitted as a unit*, which is the case where the
store most easily offers the original back. Both were live in ssg and both
fired heavily — identity skips 89/128, full-litter bans ~165/sweep.

---

## 3. Twist placement

### 3.1 Why placement is not free

Twist brackets are the binding constraint on `p_twist`. Measured on cg1, the
walk's ability to absorb them is a **rate ceiling, not a fraction**:

| run | bracket injection | absorbed | outcome |
|---|---|---|---|
| attempt 5 (`p_twist` 0.0016, neg-only) | 0.0033 g/move | 0.0030 (91%) | passed, drift +0.00027 |
| A4 (`p_twist` 0.002, mixed) | 0.0080 g/move | 0.0020 (25%) | **failed, +22%** |

Absorption saturated near 0.002–0.003 g/move in both, so injection above that
accumulates linearly. Uniform-random endpoint placement therefore caps
`p_twist` around 0.0015–0.002 for neg-only, and lower for swap.

That cap is the problem, because the dose target is **per gate per
generation**, not a one-time saturation. Gates relabelled per twist is
`E[|W|] × (1+k̄)/n ≈ 770` at n=558, m=1.2M, `twist_min_len` 256, so a full
sweep is `m/770 ≈ 1,600` twists — independent of `m`, since per-twist coverage
scales with the circuit. Over a 30M-move phase A:

| `p_twist` | twists | relabels per gate | at G=80, one twist every… |
|---|---|---|---|
| 0.0016 | 48,000 | 31 | 2.6 generations |
| **0.002** | 60,000 | 38 | **2.1 generations** |
| 0.01 | 300,000 | 192 | 0.4 generations |

**But the fixed-size absorption ceiling is not the operative constraint.**
Under the breathing model (§7.2) accumulated bracket growth is shed by the COMP
brake, which sheds at a far higher rate than the merge catalogue absorbs:
measured at `w_db=1, p_db=0` over 2M moves, **−0.06 to −0.12 gates/move**
(net −23.1% on cg1_mix). Against 0.02 g/move of injection at `p_twist = 0.01`,
that is a **~20% duty cycle in COMP** — and our COMP mode should shed faster
still, since at `p_db = 1` essentially every round is a COMP-DB attempt versus
roughly half in that benchmark. So the sustainable `p_twist` is set by duty
cycle and compute, not by the 0.002–0.003 g/move figure above.

⚠️ One caveat on that shed rate: 79.6% of its hits were trivial
identity/reorder splices, which the identity guard (§2.3) now refuses. Whether
that lowers the shed rate or redirects those rounds into real shrinks further
down the descent is unmeasured.

What placement buys is therefore **efficiency and fingerprint erasure**
(§3.3), not size stability. The compute cost — the O(|W|) relabel pass, ~1,400
gate visits per move at `p_twist = 0.01` — becomes the binding constraint
instead, which makes the deferred-relabel ledger (§3.4) the more important of
the two.

### 3.2 Negation twists: merge absorption

A negation bracket is `x_gate(w)` = `{comp: false, ctrls: []}`, i.e. `w ^= 1`.
Composed with an adjacent same-target gate `h = comp ⊕ AND(S)` it gives
`1 ⊕ comp ⊕ AND(S)` — always a single gate, `h` with its comp bit flipped. So
absorption is *always* mathematically available.

`merge_result` currently performs it in only two cases:

- `|S| = 0` → Cancel / XFuse ([mix.rs:90](../src/postmix/mix.rs));
- `|S| = 1` → Subsume, absorbing the flip into the literal's **polarity**
  rather than the comp bit (`1 ⊕ x_a^p = x_a^¬p`), so comp is untouched.

For `|S| ≥ 2` it bails at [mix.rs:99](../src/postmix/mix.rs) on `g.comp !=
h.comp`. That guard keeps the fossil count monotone and is correct for
`h.comp = 0`, where the result would be comp=1 and *create* a fossil. But for
**`h.comp = 1` the result is comp=0** — fossil-*eroding*, the allowed
direction — and it is being refused along with the banned case.

**Required extension:** allow `NOT + (comp=1, any width) → (comp=0, same
ctrls)`. It only ever decreases the comp population, so monotonicity is
preserved. The case it unlocks is exactly the g57 (comp=1, two controls), which
is 17–69% of phase-A material.

**Placement rule.** Draw a neighbourhood at random, then choose each endpoint
adjacent to an absorbing partner within it — a same-target gate that is
comp=1 (any width) or width ≤ 1 (any comp). Candidate density in a
neighbourhood of `L` gates is `L/n` gates targeting `w`:

| | candidates at L = 5,000, n = 558 |
|---|---|
| gates targeting `w` | ~9 |
| absorbable **today** (width ≤ 1) | ~1–2 |
| absorbable **with the extension** | ~5–8 at 50% g57 density |

**Entropy cost is negligible.** Endpoints snap to the nearest of ~5–10
candidates, perturbing each boundary by ~`n/2 ≈ 280` gates against windows
averaging ~142,000 — a 0.2% perturbation. Wire choice, window length and
neighbourhood location all stay uniform.

### 3.3 Swap and transvection twists: DB absorption

Merge absorption does not apply to CNOT packets (a CNOT bracket absorbs only
into an identical CNOT, via Cancel). Instead, use the store: form a window from
the packet plus adjacent gates and look it up. A hit replaces packet-plus-
neighbourhood with ordinary material.

This buys three things, not one:

1. **Size** — a replacement no longer than the window costs nothing.
2. **Dose** — it is a genuine DB re-encoding of that neighbourhood.
3. **Fingerprint erasure** — a 3-CNOT packet is a distinctive syntactic shape,
   and this destroys it at creation rather than waiting for churn. That targets
   the static/syntactic distinguisher front directly.

It also **re-enables the swap twist**, which was priced out on bracket
efficiency (negs 0.5 flips per bracket gate vs swaps 0.33). Swap rotations
route material through a fresh physical wire — a genuinely different rotation
from polarity flips — and become affordable once the packet is near-free.

**Split the packet.** CNOT-bearing windows match only when short. Measured:
`c=1` intruders at m=3 hit **78–98%**, while the m=6 cliff drops *any*
non-g57-bearing 6-gate window to **≤7%** (a CNOT costs 6 g57s, blowing the
complete-to-6 horizon). A 3-CNOT packet plus two neighbours is already m=5.

So break the 3-CNOT family into two unequal parts — 1 + 2 — and host each part
in its own neighbourhood on its own side, giving two windows at m ≈ 3 instead
of one at m ≈ 5:

| arrangement | window sizes | expected hit |
|---|---|---|
| packet + 2 neighbours, one window | 5 | low |
| **split 1 + 2, one window each** | **3 and 3–4** | **78–98% each** |

**Both ends must find homes.** A twist has two brackets; an arrangement is
complete only when every part of both is hosted. Partial absorption still helps
size, but leaves a visible packet, so the fingerprint goal is all-or-nothing
while the size goal is graded. Score candidate arrangements by residual
unabsorbed gates and accept the best; require zero residue when fingerprint
erasure is the objective. On failure, draw a different neighbourhood and retry.

**Acceptance is a mode choice** — the same three modes as §2.4, as a separate
parameter `twist_db_mode`. COMP guarantees no growth but declines often; MIX is
the natural default; ANY accepts anything. Note that **accepting longer
replacements does not solve the size problem** — it relabels bracket growth as
DB growth. If the goal is to lift the twist ceiling, acceptance must be
non-growing or at worst MIX.

**Cost and correctness.** At a 30–50% per-arrangement hit rate, 2–3 probes per
twist; at `p_twist = 0.002` against `p_db = 1` that is ~0.005 extra lookups per
move — free. Build each candidate window as a plain `&[XGate]` list and probe
it **without touching the arena**, committing only the winner; `db_replace`
needs no arena residency, so there is no rollback path to get wrong. A DB
replacement is function-preserving by construction, so replacing any contiguous
chunk containing a bracket preserves the conjugation identity with no extra
reasoning.

### 3.4 Deferred relabelling (sketch)

Placement removes the bracket cost but not the interior cost: the twist still
walks its window flipping every literal on `w`, an O(|W|) pass. Batching that
across twists is worthwhile — `B` twists cost `B·m/ln(m/twist_min_len)`
individually versus `O(m)` for one combined pass, so batching wins once
`B > ln(m/twist_min_len) ≈ 8.5`, which at `p_twist = 0.002` is about one
10,000-move interval.

The representation is a ledger of pending `(wire, span)` flips, with the flip
applied lazily at every read site and materialised in one left-to-right pass at
flush. Constraints that must be settled before building it:

- **Spans need arena-id anchors, not positions** — positions shift under every
  splice.
- **Every read site must apply pending flips.** DB lookups can do so on the
  window copy before canonicalising and invert on the replacement (windows are
  ≤5 gates, so this is cheap); `global_check` must apply them during
  evaluation. The merge index is keyed on `(target, wire-set)` and so is
  unaffected, but `merge_result` compares polarities: two gates inside the same
  pending span compare correctly (both flipped), while a merge straddling a
  span boundary would not — those must be suppressed or flushed.
- **Flush before** any checkpoint, and before any condition that reads the
  circuit's function.

Left as a sketch; the placement work in §3.2–3.3 is independent of it and
should land first.

### 3.5 Parallelisation

Deferred. The opportunity: twists with disjoint spans are independent, and the
relabel pass is embarrassingly parallel over gates. Revisit once placement and
the ledger are settled.

---

## 4. The thermostat (slot 3)

```
p_contract = sigmoid((size − target_size) / temp),  clamped
```

- **`target_size`** sets *where* the equilibrium sits — roughly the gate count
  the walk is held at.
- **`temp`** sets *how tightly* it is held: it is the width of the transition
  band, not a rate. Small `temp` gives a sharp switch — saturated expansion
  below target, saturated contraction above — hence the fastest approach and
  the tightest regulation. Large `temp` gives a soft response near target:
  slower convergence, larger size breathing, and (measured) faster fossil
  erosion at equal moves, because breathing means more crossings.

**Contract** — with probability `p_comp` try COMP-DB first (the only
contraction channel that can shrink non-ladder material); otherwise, and on a
COMP-DB miss, fall through to a journal undo (with probability `undo_frac`) and
the pairwise merge catalogue.

**Expand** — with probability `p_any` an ANY-DB move, else a **cross**. Nothing
else. Unsubsume, insert and fresh-split are removed (§9).

**The contraction clamp is a parameter, not a constant.** Today the ceiling is
0.98 except under `p_db_steer`, where it is 0.9995 — because the 2% expansion
floor is a structural growth source measured at +0.007 gates/move, which over
a 15M-move phase B is ~105k gates. `p_db_steer` is removed, so the tighter
clamp must survive on its own.

**Caveat: the thermostat only controls size when `p_db` is small.** Slot 2 runs
ahead of it and can grow, so at phase-A rates `target_size`/`temp` set the
contract/expand mix, not the size. Size control in that regime comes from
`db_mode` (§7.2). The report must not present `target=` as a size guarantee.

---

## 5. Seeding and the generation pool

Every gate carries a **generation**: input gates 0; a DB splice stamps
products with the outgoing window's upper-median generation + 1; splits stamp
children with parent + 1; merges take the min; undos restore recorded values;
twist bracket packets are born-random MAXGEN.

**Seed selection.** Flip a coin with expectation `p_mingen`:

- **heads** — draw uniformly from the **generation pool**;
- **tails** — draw uniformly from the whole circuit.

**The generation pool** is the `K` lowest-generation gates among gates that are
(a) **pool-eligible** — fewer than `w_pool` controls — and (b) **below the
generation goal**.

Both filters are load-bearing:

- *Eligible only.* An ineligible gate can never be re-encoded — a window seeded
  on one truncates to empty — so its generation is pinned at 0 forever, and an
  unfiltered pool converges on exactly the ineligible set and stays there.
- *Below goal only.* Without it, a late-run pool is padded with ordinary
  low-but-done gates that re-encode fine, so the canary (§6) can never fire.
  With it, the pool shrinks to the genuinely stuck residue at exactly the point
  the canary is meant to detect.

**Two eligibility thresholds, not one.** `w_window` (§2.1) governs what may sit
inside a window; `w_pool` governs what may seed one and count toward the dose.
They want different values because the measurements differ: width-3 gates match
in context often enough to be worth admitting to windows, but their end-to-end
per-gate re-encode rate is 0.41% against 98.98% for width ≤ 2, so at a shared
threshold they accumulate at the bottom of the pool with nothing to eject them.

`K` is a **count, not a fraction**: the drain rate is set by the move economy
(`gen_rescan × p_db × p_mingen`) and is independent of circuit size, so a
percentage over-provisions as the circuit grows and under-provisions on small
circuits. `K` must exceed the draws taken between rebuilds or the pool empties
and the biased coin silently degrades to uniform.

The pool is rebuilt by an O(size) scan every `gen_rescan` moves, with stale
entries pruned at draw time.

**No per-gate miss counter.** The bottom rung of the descent (§2.2) advances
any pool-eligible gate the store can spell at all, so the stuck population is
not "gates with no cheap spelling" (large — measured 25–35% under compress-only
regimes) but "gates the store cannot spell at any length" (small). The residual
risk is covered by the canary rather than per-gate state.

---

## 6. The canary

A single global meter: over a ring buffer of the last `W` **qualifying**
rounds, the fraction that **failed at every rung of the descent**.

- A round qualifies iff the `p_mingen` coin came up heads **and the pool was
  non-empty**, so the seed genuinely came from the pool.
- Heads-rounds where the pool had drained are counted **separately** as the
  *fall-through* rate. Conflating the two would mix "the material is
  unreachable" (stop the run) with "`gen_rescan` is too slow" (scan more
  often) — opposite remedies. The fall-through rate is the tuning signal for
  `K` and `gen_rescan`.
- Because the descent retains the seed at every rung (§2.2), "failed at every
  rung" ≡ "the seed was not consumed"; no per-gate id or stamp tracking is
  needed.

The condition fires when the buffer is **full** and the failure fraction
exceeds `θ`. Buffer-fullness supplies the minimum-sample guard, and `W` is
denominated in qualifying rounds rather than moves — the right units, since
the qualifying rate varies with the fall-through rate.

Healthy failure rates should sit well under 0.2 (five rungs against ~99%
per-window hit rates on width-≤2 material) while the pathological case drives
toward 1.0, so `θ = 0.9` sits in a wide gap and `W = 2000` gives a sampling σ
of ~0.007 at that threshold.

**The canary sleeps while `db_mode == COMP`.** COMP declines far more often by
construction, and mixing those samples in reads as "the pool is unspellable"
when the truth is "the brake is on". Since the canary is a *stop* condition,
that would end runs spuriously.

Available to layer 2 as either a stop or an advance condition. As an advance
condition it means "this phase has extracted what it can from this material" —
the honest version of what a run needs when its dose is complete but its move
budget is not.

---

## 7. Size control

Two independent mechanisms, used in different regimes.

### 7.1 The thermostat

§4. Effective when `p_db` is small — the phase-B regime.

### 7.2 The mode brake

`db_mode` is a deterministic three-way throttle on slot 2:

- **ANY** — accelerator (uniform over all equivalents; the store holds more
  long spellings than short, so it grows in expectation);
- **MIX** — cruise;
- **COMP** — brake (refuses to grow).

Slot-0 rules switch it on size predicates. Controlling *mode* dominates
controlling *rate* as a brake: lowering `p_db` slows growth and dose
proportionally, whereas COMP stops growth while still delivering dose, since
COMP hits still stamp generations.

**COMP restores the thermostat rather than bypassing it.** In MIX the
thermostat is overwhelmed by an uncontrolled channel; in COMP slot 2 stops
adding and the contract branch is back in authority. It is one controller with
a bypass switch, not two controllers.

### The breathing cycle is the point, not a side effect

Grow until the high-water mark, brake to the low-water mark, repeat. Prior
sss/ssg-era experiments found **repeated expand-and-compress cycling more
effective for mixing than holding stable or growing slowly**, and the fmix-era
measurements agree from two directions: higher `temp` (looser regulation,
larger breathing) gives measurably faster fossil erosion at equal moves, and
positional transport happens almost entirely during growth — a 44× growth run
reached `odiff 0.20 / oadj 0.34` where fixed-size runs sat at `0.01 / 0.96`.
Fixed size churns composition fast but barely moves material.

Two consequences:

- **Make the band wide.** Narrow marks give shallow, frequent cycles; wide
  marks give deep growth phases, which is where transport happens. The band is
  a mixing parameter, not a tolerance.
- **Size accumulation is not a failure mode** in phase A. Growth that the
  thermostat cannot absorb is shed by the next COMP leg, which sheds at
  −0.06 to −0.12 g/move (§3.1) — one to two orders above the merge
  catalogue's fixed-size absorption.

### But a COMP leg that runs too long makes negative progress

Also from the sss/ssg era, and it is what actually limits the band. Two
mechanisms, both measurable:

- **Starvation.** As the circuit approaches local minimality the COMP-DB hit
  rate collapses, and 79.6% of the remaining hits on mixed material were
  trivial identity/reorder splices — which the identity guard (§2.3) refuses,
  so those rounds are pure waste.
- **Entropy loss.** MIX-DB draws uniformly over *all* free equivalents;
  COMP-DB draws only over the minimum-size ones. A long COMP leg is many
  splices all pulling toward one small corner of each equivalence class,
  drifting the circuit toward its locally-minimal spelling — which is where
  `fcompress`, the attacker-computable compressor, would put it anyway. COMP
  does not merely stop paying; it spends re-encoding diversity.

**So release on productivity, not on size:**

```
arm COMP:      size ≥ high-water
release COMP:  size ≤ low-water  OR  COMP shed rate over the last N rounds < ε
```

The second disjunct is what makes wide bands safe. The risk in a wide band was
never the width — it was sitting in COMP past its usefulness. Guard that
directly and the band stops needing delicate tuning: a too-wide band costs
nothing, because the productivity guard releases early on its own.

### Signals for an eventual self-controller

The long-term goal is a controller that reads the system and sets its own
parameters. The rule engine (§1 slot 0) is already the right substrate — a rule
is *condition → set parameter* — so what is missing is calibration, not
architecture. The meters it would read, all cheap per-round counters except
where noted:

| signal | reads |
|---|---|
| `d(size)/dmove` over a trailing window | which leg of the cycle, and how fast |
| COMP-DB hit rate; strict-shrink fraction of hits | whether the brake is still paying |
| MIX-DB hit rate; free-branch vs pay-branch split; `paid=` | where growth is coming from |
| canary failure fraction; fall-through rate (§6) | whether targeting still has reachable work |
| `cov` (twist coverage) | twist dose |
| fossil count | erosion progress |
| `odiff` / `oadj` (already in the report line) | positional transport — the thing growth legs are *for* |
| generation quantiles | dose — O(size), needs a cadence |

Two meters the panel is missing, both worth building before trusting a
controller:

- **Distance from minimal.** The adversary-aligned quantity. Prolonged COMP
  drifts the circuit toward its locally-minimal spelling, and nothing currently
  reports how close it is. A cheap live proxy: sample windows, ask COMP-DB
  whether a strictly shorter spelling exists, report the fraction. That is a
  sampled `fcompress` residual without running `fcompress`.
- **Re-encoding diversity, not count.** A generation counts splice *events*.
  The identity guard forbids no-op splices but not `A → B → A → B` cycling, so
  a gate can reach generation 80 having visited two spellings. What the dose is
  supposed to buy is coverage of the equivalence class, and nothing measures
  that. **The known fix is litter rules (§2.6)** — in the controlled ssg pair
  they were the difference between floor generation 100 and floor ~19 at the
  same nominal dose. The measurable proxy already exists in fmix as the
  `sib=`/`xorig=` split; it is simply not acted on.

⚠️ A controller that adapts to circuit-specific signals makes the parameter
trajectory a function of the circuit. The trajectory is not part of the
artifact, and a converged loop plausibly yields *more* uniform statistics
rather than less — but it is worth checking rather than assuming, given this
project's history with lifetime- and persistence-based attacks.

**Requirements:**

- **Hysteresis, not a threshold.** A single size threshold chatters. Use a
  high-water mark to arm COMP and a distinctly lower low-water mark to release
  it.
- **COMP stops growth at slot 2; shrinking comes from COMP-DB hits and the
  contract branch.** Non-growing includes equal-size, and only ~13% of COMP-DB
  hits strictly shrink — so the low-water mark is reached by accumulation of
  many small wins, not by a fast collapse. Size the band and the duty cycle
  against that rate.
- **Size sawtooths** between the marks by design. Phase A's exit size is phase
  B's growth base, so add "near low-water" to the exit condition if the
  delivered size needs to be predictable; otherwise accept ±band.
- **In COMP:** generations keep growing and the pool machinery stays live; only
  the canary sleeps. `p_mingen` is expected to be re-set by the same rule that
  changes the mode.
- **COMP silently disables `curated`** (§2.5), so the brake drops growth and
  curated quality together.

---

## 8. Parameters

### 8.1 The menu

| name | default | controls |
|---|---|---|
| `p_twist` | 0.002 | probability a round is a twist |
| `p_db` | 1.0 | probability a round is a slot-2 DB move |
| `db_mode` | MIX | slot-2 admission rule; set by slot-0 rules |
| `p_comp` | 1.0 | probability a contraction tries COMP-DB before undo/merge |
| `p_any` | 0.1 | probability an expansion is ANY-DB rather than a cross |
| `s_db` | 5 | window length the descent starts from |
| `p_convex` | 0.5 | probability the sampler is convex rather than contiguous |
| `db_convex_p` | 0.75 | *within* convex growth, probability of floating in `g1`'s direction |
| `w_window` | 4 | a gate with this many controls or more may not sit in a window |
| `w_pool` | 3 | a gate with this many controls or more may not seed one or count toward the dose |
| `curated` | off | prefer curated matches (§2.5) |
| `twist_db_mode` | MIX | acceptance rule for twist-packet DB absorption (§3.3) |
| `p_mingen` | 0.8 | probability a seed comes from the generation pool |
| `K` | 20,000 | pool size, in gates |
| `gen_rescan` | 10,000 | pool rebuild cadence, in moves |
| `θ` | 0.9 | canary failure fraction that fires the condition |
| `W` | 2000 | canary ring buffer, in qualifying rounds |
| `target_size` | input size | thermostat setpoint (§4) |
| `temp` | max(target/100, 64) | thermostat band width (§4) |

Pool headroom check: `K / (gen_rescan × p_db × p_mingen) = 20,000/8,000 =
2.5×`. Adequate but not generous — a splice can drain several pool members at
once, so watch the fall-through counter.

### 8.2 Store guards

Set to the values the current store requires; ideally read from store metadata
rather than retyped per run.

| name | default | controls |
|---|---|---|
| `db_max_degree` | 9 | window ANF degree above which a match is impossible |
| `db_max_span` | 30 | distinct wires above which a match is impossible |
| `db_wire_terms` | 1024 | per-wire polynomial term budget |
| `db_total_terms` | 2048 | summed term budget across the window |
| `db_degree_probes` | 6 | random subcubes probed per direction by the degree guard |

### 8.3 Carried over unchanged

`k_max` 12, `split_damp` 2, `split_base` 2.0, `dir_p` 0.75, `dir_q` 0.85,
`merge_reach` 4096, `journal_len` 262144, `undo_frac` 0.5, `tabu_moves` 2000,
`twist_min_len` 64, the three `w_twist_*` ratios, `seed`, `verify_every`,
`report_every`, `no_local_verify`, `skip_final_float`.

---

## 9. Removed

Relative to the current implementation:

**Moves.** Unsubsume (`w_unsub`), insert (`w_insert`), fresh split (`w_fresh`).
Expansion is cross-or-ANY-DB. The syntactic variety those supplied is supplied
better by DB re-spelling. One consequence: inserts were the only source of
material not descended from the input, so the `GEN_FRESH`/MAXGEN "born-random"
case now survives only for twist bracket packets.

**Channels.** `p_db_ingest`, `p_db_hard` as top-level coins; `p_db`,
`p_db_final`, `p_db_steer`. The ingest/pay semantics survive as *modes of the
one DB move*; annealing and steering become slot-0 rules.

**Tier machinery.** `lag_cheap`/`lag_hard`, the tier split in
`rebuild_laggards`, `Meta::miss`, `gen_miss_budget`, `gen_giveup`, `bump_miss`,
`last_seed` stamp tracking, `SeedPool`. Replaced by one pool, one probability,
one canary.

**Window parameters.** `db_min_window`/`db_max_window` → `s_db`;
`db_prefixes` → always on; `db_sample` (enum) → `p_convex` (scalar);
`db_ctrl_cap` → `w_window` + `w_pool`.

**Second consumers that must be re-homed, not deleted:**

- `db_hard_added` — the `paid=` growth ledger. Re-attach to MIX-DB's pay
  branch; it is the only accounting of where growth went.
- `db_ing_*` / `db_hard_*` round and hit counters — merge into one pair, which
  is the canary's numerator and denominator.
- `GenStats::wlag` — below-goal gates that are pool-ineligible. Survives and
  matters more: it is the census of the population the pool must exclude.
- `gen_stats`'s `unreach` bucket removed `gen_giveup` write-offs from the
  dose-stop denominator and the `G=` percentile population. With the counter
  gone those gates return to both; the stop tolerance absorbs them (prod33: 52
  written off against ~95k targetable, versus a 2% tolerance).

---

## 10. Open

- **Transport under a DB-dominated schedule — resolved by `db_advance`, still
  to be measured.** The DB move already uses direction in three places: the
  contiguous sampler extends the window along `g1`'s own direction, the convex
  sampler floats `g1` and then the block along it, and the splice assigns
  product directions by pivot from `g1dir`. What was missing is the ballistic
  **birth-advance** — `advance_births` fires at all five split sites but not at
  the splice, so products were handed a direction nothing ever acted on.

  The obvious alternative, restoring crossings by setting `p_db < 1`, is the
  wrong fix: crossings widen material, and width is what kills DB matching (one
  width-3 gate takes a window from 36% to 0.2% across m = 3…6, and ~62% of a
  product-share gadget is already width ≥ 3). Phase A would be manufacturing
  exactly what the store cannot digest.

  `--db-advance` gives splice products the same birth-advance split pieces get.
  Float-only, so the function is preserved by construction, and it scatters the
  litter as a side effect — making a later window less likely to be exactly one
  earlier replacement. **Off by default because it changes trajectories**; the
  A/B is one flag, read on `odiff`/`oadj`, and it belongs in calibration stage 2.
  The menu spec wants it on.
- **`p_twist` is under-set at 0.002.** With the COMP brake as the absorber
  rather than the merge catalogue (§3.1), 0.01 costs only a ~20% COMP duty
  cycle and delivers a twist per gate every 0.4 generations. 0.002 is a safe
  starting point, not a ceiling; the ceiling is now compute, addressed by
  §3.4.
- **Duty cycle in COMP** — the fraction of phase A spent braking is a real
  cost (dose accrues more slowly there) and is set by the band width. A
  layer-2 number, but worth measuring early.
- **`target_size` / `temp`** — kept as parameters; whether the report's
  `target=` field is renamed to stop implying size control is undecided.
- **Hysteresis marks** for the mode brake — layer 2.
- **Condition evaluation cadence** — cheap counters per round, O(size) census
  on its own cadence; numbers are layer 2.

---

## 11. Implementation notes

- **Extend `merge_result`** with `NOT + (comp=1, any width) → (comp=0, same
  ctrls)` (§3.2). Without it the neg-twist placement search finds a partner
  only ~1–2 times per 5,000-gate neighbourhood instead of ~5–8.
- **The store-open guard must include `p_db`.** Today the store opens only when
  `w_db || p_db || p_db_final` is positive; under the new naming the primary DB
  channel is not in that list, which yields an empty store, zero re-encoding,
  and no diagnostic.
- **Abort when `curated` is on and no curated store is present.** Do not
  degrade silently to regular-only. `probes.bin` is *not* required — it appears
  nowhere in `src/`; `Frozen::open` needs `tables.bin` and the shards, and
  loads `filters.bin` only under `FROZEN_FILTER=1`.
- **`FROZEN_DB_DIR` is env-only and in no rc file**; detached runs must export
  it or abort instantly. Worth promoting to a flag.
- **On COMP → MIX, force a pool rebuild** before the first biased draw. The
  pool goes stale during a brake because crossings and merges keep changing
  which gates sit at the bottom.
- **Checkpoint and resume.** On any stop, persist enough to resume seamlessly:
  per-gate `dgen`, `origin`, `dir` (no sidecar today, and load-bearing for
  transport) and `event`; the **undo journal**, whose entries reference arena
  ids and stamps and therefore require the checkpoint to preserve arena
  identity, not just gate order; the **original circuit**, which is the
  fidelity anchor `global_check` compares against; `moves_done`, `next_event`,
  the tabu queue, `twspan`, the canary buffer, `db_mode`, the phase id, the
  pool scan due-time, and any pending twist ledger (§3.4). `StdRng` is not
  serialisable — draw a fresh `u64` at checkpoint and reseed from it. Write it
  as circuit + separate state file so the circuit stays readable by
  `fmix_stats`/`fcompress`/the hmap tools, and version-stamp the state file so
  checkpoints from an older parameter set refuse to resume rather than being
  silently reinterpreted.

---

## 12. Saturation: when phase A is done

The observed behaviour to explain: after some number of breathing cycles the
process is evidently saturated — generations keep climbing but no further
mixing happens. That is the clock-versus-displacement gap again, so the stop
rule must be built on **displacement** meters, not on generation.

Several already exist, each with a **known asymptote**. The right-hand column
is a fresh n=128 gadget after 60k fixed-size moves — how far from saturation
the starting point is:

| meter | asymptote | fresh gadget, 60k moves |
|---|---|---|
| `oadj` — adjacent-origin autocorrelation | **0** | 0.9999 |
| `odiff` — origin diffusion | **0.2887** (1/√12) | 0.0009 |
| `disp` — mean normalised displacement | **1/3** | 0.0054 |
| `owin` — distinct origins per 32 gates | 32 | 30.5 |
| `cov` — twist coverage | ~600 at 256 wires; scale by `n` | 55.9 |
| `comp` — g57 fossils | 0 | 4116 |
| **litter distinct per window** | **`s_db`** | — (needs a store) |
| **litter full-splices** | **0** | — |

Two readings from that column. `oadj = 0.9999` after 60k fixed-size moves —
*with* crossings enabled — is a direct demonstration that fixed size churns
composition while the conveyor belt of the original gate order stays intact.
And **`owin` saturates far too early to be a criterion**: it is already at
30.5/32 when nothing has moved. Choose meters still far from their asymptote.

**The litter meters are the most direct saturation signal**, because they
measure whether re-encoding is still *braiding* or merely churning. Mean
distinct litters per window reaching `s_db`, with full-litter splices at zero,
means every window is drawn from `s_db` different replacement events — no
coherent unit remains for a splice to undo.

**Three rules for reading them:**

1. **Plateau on the derivative, not the level.** The achievable asymptote is
   material-dependent; absolute targets are guidance, slopes are the criterion.
2. **Sample once per breathing cycle, at the same phase of the cycle** (e.g.
   every low-water crossing). Within a cycle the meters race during growth and
   stall during COMP, so intra-cycle sampling mostly reports where in the cycle
   you are. Fixed-phase, cycle-to-cycle differencing removes that component.
3. **Stop when a whole cycle buys less than ε on every axis**, for N
   consecutive cycles — loop-until-dry, not a single threshold.

⚠️ These are syntactic and positional proxies. The security question is the
heatmap, which is expensive. **Validate the proxy once**: run
`hmap_affine`/`hmap_stat` mid-plateau and well past it, and confirm the heatmap
has genuinely stopped improving where the cheap meters say so. Otherwise the
exit rule is calibrated against an unverified correlate.

---

## 13. Phase-A starting point and calibration

Input: gadgetized sliced sandwich, n = 128 — roughly 247k gates / 558 wires,
~17% g57, ~38% of gates at width ≤ 2.

```
p_twist 0.002    w_twist_neg 1.0 (neg-only)    twist_min_len 256
p_db 1.0         db_mode MIX     s_db 5    p_convex 0.5
p_comp 1.0       p_any 0.1
w_window 4       w_pool 3
p_mingen 0.8     K 20000         gen_rescan 10000
theta 0.9        W 2000
curated OFF initially
band: low-water 250k, high-water 600k, plus the productivity release (§7.2)
guards 9 / 30 / 1024 / 2048
```

`curated` starts off deliberately: it prefers growth by construction (§2.5), so
it confounds band tuning. Turn it on once the band is settled. `p_db` starts at
1.0 as specified, but see §10 — both the transport and absorption arguments
point below 1, and this is the first thing stage 2 should test.

**Calibration, cheap-first, one loop at a time:**

- **Stage 0 — machinery.** Small n, minutes. Confirm every parameter *took
  effect by reading the banner back*. This project has repeatedly lost runs to
  silently dropped parameters (shell word-splitting reverting an encoding; the
  targetable-generation fix reverting at `db_ctrl_cap 0`). Check the litter
  counters are non-zero, the canary quiet, verification failures zero.
- **Stage 1 — measure, don't tune.** One production-shape run, no stop
  conditions armed beyond a move ceiling. Read every meter in §12. This says
  which axes actually move before anything is tuned — and specifically whether
  litters fragment fast enough that the §2.6 ban would ever fire.
- **Stage 2 — the breathing band.** The fastest feedback loop, so tune it
  first: high/low water, the productivity ε, and `p_db`. Measure COMP duty
  cycle, shed rate per leg, and whether wider bands move `odiff`/`oadj` faster
  *per move*.
- **Stage 3 — `p_twist` upward.** Raise until `cov` saturates or compute bites.
  With the band settled, bracket growth simply shows up as duty cycle.
- **Stage 4 — the exit rule**, set from the plateau observed in stages 2–3 and
  validated once against a heatmap (§12).

Throughout: **one parameter per run, same C, same seed** — the A/B discipline
behind every reliable finding in this project's record.

---

## 14. Pending measurements

Everything above is specified; almost none of it is measured. These five gate
decisions currently being made on inference. Run order matters — stage 1 gives
the baselines the rest are read against.

| # | experiment | read | decides |
|---|---|---|---|
| 1 | **`--db-advance` off vs on**, same C and seed | `odiff`, `oadj` trajectories | whether the flag becomes the default (the spec wants it on) |
| 2 | **`full=` baseline**, one production-shape run, `db_advance` OFF | `litter distinct=` / `full=` | whether the §2.6 litter ban is worth building at all |
| 3 | **Curated hit rate per rung**, `--db-dry-run` with the recorder | curated vs regular matches at window 1–5 | whether `curated` belongs on by default |
| 4 | **COMP shed rate under the identity guard** | gates/move over a COMP leg | band width and COMP duty cycle (§7.2) |
| 5 | **Saturation proxy vs heatmap**, mid-plateau and well past | `hmap_affine` / `hmap_stat` | whether the §12 exit rule tracks the thing it stands in for |

⚠️ Run 2 with `--db-advance` **off**. The advance scatters litters, which
directly suppresses `full=` — measuring both at once confounds them.

The −0.06…−0.12 g/move figure the band is sized against (experiment 4) comes
from a benchmark in which 79.6% of hits were trivial identity/reorder splices,
which the identity guard (§2.3) now refuses. Whether that lowers the shed rate
or redirects those rounds into real shrinks further down the descent is exactly
what the experiment settles.
