# Implementation plan: bring `src/replace/fragment.rs` up to `fmix` / XGate parity

> **Archived / superseded.** This plan describes the pre-reorganization tree
> and a prototype that has since been retired. Paths and proposed work below
> are preserved as historical context, not current implementation guidance.
> See [`src/README.md`](src/README.md) for the maintained module and pipeline map.

## What this is

This is a **reversible Boolean circuit rewriting** research project. We have a
prototype (`src/replace/fragment.rs`) that transforms a circuit into
functionally-equivalent variants by breaking gates into smaller pieces, moving
those pieces around by commutation, and recombining them. A more mature
reference design (the `fmix` / `fcompress` / `fmix_stats` toolkit, documented
separately — ask the user for `fmix_docs.md`) implements the same idea far more
completely. This plan lists the work to bring the prototype up to that design.

Everything here is pure algorithm work on Boolean circuits: gate algebra,
equivalence-preserving rewrites, a deterministic compression pass, and
read-only mixing-quality metrics. There is no I/O with external systems and
nothing adversarial — the "mixing quality" metrics simply measure how far a
circuit's gate sequence has been randomized away from its input description,
which is the research object.

## Current state of `src/replace/fragment.rs` (the prototype)

- `CubeGate { target: u16, lits: Vec<(u16, bool)> }` — a **conjunction-only**
  gate: `x_target ^= AND over lits of (wire == polarity)`. It is the `comp = 0`
  special case of the reference `XGate`.
- Semantics/util: `fires`, `apply`, `arity`, `mass_exp`.
- Algebra: `commute` (the two-clause law: same target, OR cubes never both fire,
  OR neither target in the other's controls), `split` (Shannon cofactor),
  `cross` (conjugation-with-correction, returns `None` on mutual dependency).
- g57 interop: `from_g57` (splits a g57 into its two conjunction pieces),
  `expand_g57_circuit`.
- Compression: `try_combine` (Cancel + sibling-merge), `recombine`,
  `reassemble_g57` (pairs pieces back into g57 gates).
- Polynomials: `poly_mul`, `poly_not`, `fragment_polys` (ANF/monomial form,
  byte-identical to `CircuitSeq::to_polynomial` — verified).
- Drivers: `transport_mix`, `shoot_and_reassemble`, `collision_shoot`.
- `Φ` = summed transposition mass (`phi`); brute-force equivalence checkers.
- A `src/bin/frozen_key_check.rs` binary (DB-key verification — see "keep/retire").

The unit tests in that file (`cargo test --lib fragment::`) are the correctness
harness; every new primitive below must land with a brute-force test in the same
style (verify over all `2^n` inputs for small `n`).

## Design gap in one sentence

The prototype **always expands g57 gates into conjunction pieces and tries to
collapse back to pure g57**, whereas the reference design **keeps a richer gate
type (with a complement bit) and compresses locally with an exclusive-sum-of-
products (ESOP) pass** — it never needs to return to g57. Adopting the richer
gate type and the local compression pass is the central change; most other
tasks build on it.

---

## Task 1 (foundational): generalize `CubeGate` → `XGate` with a complement bit

**Goal.** Add a boolean `comp` field so the gate becomes
`x_target ^= comp XOR AND over ctrls of (wire == polarity)`. This is the
reference `XGate`. A g57 gate becomes a **single** gate with `comp = 1` and
controls `{(pos_ctrl, false), (neg_ctrl, true)}` (verify the exact polarities
against `evaluate_index` — the g57 fires on `pos=1 OR neg=0`).

**Subtasks.**
1. Add `comp: bool` to the struct (consider renaming `CubeGate` → `XGate`,
   `lits` → `ctrls`). Keep `lits`/`ctrls` sorted by wire, target absent.
2. Update `fires`/`apply`: `fires = comp XOR AND(...)`.
3. Update `from_g57` to produce **one** `comp=1` gate. Move the old
   two-piece decomposition into a separate function `presplit(g)` that returns
   the two `comp=0` conjunction pieces (this becomes a *move*, Task 4, not the
   default representation).
4. Update `commute` for the complement bit: **`comp=1` gates never satisfy the
   "cubes never both fire" clause** (their firing set is the complement of a
   subcube, not a subcube), so they only commute via the structural clause
   (disjoint read/write wires) or same target. Two `comp=0` gates keep the
   existing separation exemption.
5. Update `fragment_polys` for the complement bit (`comp=1` XORs an extra
   constant `1` into the target's polynomial).

**Verify.** Brute-force `commute` against actual order-independence over
`comp ∈ {0,1}` and random controls (extend the existing
`commute_law_matches_brute_force`). Confirm `from_g57` (now one gate) still
reproduces the g57 permutation.

---

## Task 2: complete the pairwise merge catalogue + the complement guard

The prototype only has `Cancel` and one merge (sibling `DropLit`). Implement the
full four-case catalogue for two same-target gates `g, h`, where each case is
the situation in which `f_g XOR f_h` is again a single (possibly complemented)
gate:

| Case | Condition | Result |
|---|---|---|
| `Cancel` | identical gates | both vanish |
| `XFuse` | same controls, **opposite** `comp` | a NOT on the target (`comp=1`, empty controls) |
| `DropLit` | same wire set, exactly one polarity flipped, equal `comp` | drop that wire |
| `Subsume` | wire sets differ by one literal, shared literals equal, equal `comp` | flip that literal |

**Complement guard.** Refuse any fusion whose *result* would be complemented
**when that result is the recombination of a g57's two presplit pieces** — i.e.
keep the count of surviving `comp=1` (original-g57) gates monotone
non-increasing. This is a policy toggle on the merge step, not a change to what
is mathematically legal. Track a running "surviving-original count" for metrics.

**Verify.** Each catalogue case: brute-force that replacing `[g, h]` by the
result preserves the function over all inputs. Add a test that the guard makes
the surviving-original count non-increasing across a mixing run.

---

## Task 3 (high value): deterministic ESOP/ANF compression pass (`fcompress`)

This **replaces** the "reassemble to pure g57 then look up in the DB" approach.
It is a fixed, deterministic, self-contained reducer. Because it is a public,
reproducible algorithm, its output gate-count is the fair number to report as a
circuit's *effective size*, and applying it to periodic snapshots of a mixing
run gives a monotone mixing clock (the greedy-recoverable fraction shrinks with
churn).

**Algorithm (iterate to a fixed point, `--max-iters`):**
1. **Gather** — one forward sweep keeping an open group per target wire. Close a
   group when (a) any gate *reads* that target wire (a reader pins the
   accumulated value), or (b) any gate *writes* a wire in the group's
   union-of-member-controls. Closures cascade; emit groups in ascending
   last-member order. These two rules make it legal for a group's members to
   float together to the close point.
2. **Reduce** — each group is `t ^= f1 XOR ... XOR fk`, an ESOP over the
   `XGate` firing functions. Apply the Task-2 catalogue to a fixed point; if the
   group's support fits within `--anf-support-cap` bits, also compute the exact
   ANF (use the `fragment_polys` monomial machinery — duplicate monomials
   annihilate) and keep whichever is smaller.
3. **Re-emit** the survivors as consecutive `XGate`s.

**Verify.** Output equivalent to input (brute force small `n`, sampled 64-lane
check large `n`); running it twice is idempotent within a tolerance; report
`gates_in -> gates_out` and literal counts per iteration.

**Optional (lower priority): dead-cone pruning.** For circuits where equality is
only required on a designated subset of output wires, add one backward liveness
pass in the XOR-accumulate model (a gate is removable iff its target is not live
at its position; a kept gate marks its controls live and its target stays live).
Default to "all wires live." Keep behind a `--live-wires` option.

---

## Task 4: the directional mixing walk (`fmix` core)

A random walk over functionally-equivalent circuits whose objective is to churn
the gate sequence as far from its input description as possible while a size
controller holds the gate count near a target. Build on Tasks 1–2.

**Subtasks.**
1. **Persistent direction.** Give each gate a `direction` (left/right), drawn
   uniformly at load and inherited through splits. Transport moves a gate along
   its *own* direction (ballistic), not a fresh random direction each time
   (the prototype's `collision_shoot` re-randomizes — replace that).
2. **Crossing move.** Float a gate along its direction to its first collision,
   then split it past the collider by one conjugation rung (the prototype's
   `cross` is one such rung; the reference has three variants R1/R2/R3 — add the
   missing ones). A shot g57 first presplits (`presplit`, Task 1) into its two
   pieces. Each produced piece inherits the shot gate's direction with
   probability `dir_p` (else the opposite) and advances `floor(dir_q * slack)`
   gates in its direction at birth, where `slack` is its free run to the next
   collider. A crossing that is declined (width-damped, capped, or at the
   boundary) **retreats** `floor((1 - dir_q) * distance_floated)` rather than
   parking at the collision.
3. **Width damping.** A produced piece with `c` controls is admitted outright if
   `c <= split_damp`, else with probability `split_base^-(c - split_damp)`.
   Hard-cap controls at `k_max`.
4. **Size controller (thermostat).** Each move chooses expansion vs contraction
   with `p(contraction) = sigmoid((size - target) / temp)`, clamped to
   `[0.02, 0.98]`. Note the reference finding: the *growth phase* (controller
   pinned toward expansion) does the most transport — so a run typically targets
   a size well above the input.
5. **Contraction moves.** (a) A journal undo: exactly reverse a recorded
   crossing while all its pieces are still alive (stamp-validated); crossings are
   the one expansion the pairwise catalogue cannot invert, so an undo journal is
   required or size creeps up. (b) A catalogue merge: pick a gate, find the
   nearest reachable same-`(target, wire-set)` partner via a hash index within
   `merge_reach`, float them adjacent (incremental wall check), apply the
   catalogue.
6. **Other expansion moves.** `insert` (insert an adjacent identity pair of a
   fresh random conjunction, give the two copies opposite directions, shoot each
   once) and `unsubsume` (inverse of `Subsume`). Weighted selection with the
   reference default weights.
7. **Tabu.** A freshly split pair may not be undone or sibling-merged until
   `tabu_moves` moves have passed.
8. **Provenance.** Each gate carries `(origin, event)` — the input-gate index its
   material descends from and the split event that created it. Splits pass the
   parent's origin to both pieces; merges keep the survivor's. Synthetic material
   (from `insert`) uses a sentinel origin.

**Verify.** Three layers, all default-on: (1) every move exhaustively verified on
its support before commit; (2) a sampled 64-lane global equality check every
`verify_every` moves; (3) the final result verified before write. A failure
should panic. Add the size controller and directional transport incrementally,
each with its own test.

---

## Task 5 (advanced / optional): state-trajectory twists

A distinct mixing axis. A twist picks a window `W` and an involution `P` (a wire
negation, a wire swap realized as 3 CNOTs/side, or a transvection `x_a ^= x_b`,
one CNOT/side) and rewrites `P · (P W P) · P ≡ W`: it conjugates every interior
gate in place and brackets the window with one `P`-packet per side. The function
and everything outside `W` are unchanged, but every intermediate *state* becomes
its image under `P`, which rotates the intermediate-state trajectory — the one
kind of mixing that support-local moves cannot produce.

Implement the negation twist first (simplest: flip the polarity of the twisted
wire in every interior gate that reads it, add a NOT bracket each side). The
transvection twist requires case-splitting interior readers of `a` on `b` (each
such gate doubles, width +1) and needs `b` unwritten in the window. Keep twist
weights small (~1e-3): each twist rewrites `O(window)` gates and each wide-gate
relabel costs a truth-table check exponential in its support.

**Verify.** Brackets restore the function exactly (brute force on the window
support); the whole-circuit function is unchanged.

---

## Task 6: mixing-quality metrics (`fmix_stats`)

A read-only analyzer. The prototype already borrows `leeway` from
`src/replace/disperse.rs`; add the rest of the stationarity signature. Given a
provenance sidecar (Task 4), also compute the positional metrics — these are the
ones that actually measure how far material has been transported, which the
prototype only approximated with "distinct output count."

Metrics to compute (all read-only, one grep-able line per family):
- **Static:** gate/width profile, fanout, leeway (float-box size), per-wire and
  wire-pair co-occurrence entropy, window wire-span.
- **Positional (needs origins):** `disp` (mean normalized displacement of
  material from its origin position; 0 = unmoved, 1/3 = independent), `odiff`
  (per-origin position spread; 0 = clumped, 1/√12 ≈ 0.289 = uniformly
  dispersed), `oadj` (autocorrelation of adjacent gates' origins; 1 = original
  order preserved, 0 = ancestry-independent neighbours), `owin` (distinct
  origins per 32-gate window).

Reference finding to reproduce as a sanity check: **positional transport is the
slow mode** — a fixed-size run churns composition quickly but barely moves
material, while a growth run moves material a lot (`odiff` ~0.20, `oadj` ~0.34
for a large-growth run vs ~0.01 / ~0.96 fixed-size).

---

## Task 7: file format + CLI binaries (tooling parity)

- **`mpmct1` text format** reader/writer: header `mpmct1 <num_wires> <num_gates>`,
  then one gate per line `target comp k wire pol wire pol ...`. Also read the
  existing base-83 `CircuitSeq` (g57) input format.
- **Binaries** under `src/bin/`: `fmix` (Task 4), `fcompress` (Task 3),
  `fmix_stats` (Task 6), each with the CLI flags from the reference docs.
- **Pause-free control** for long runs: environment flags checked at each report
  point — a stop flag (graceful finish: final pass, verify, write, exit) and a
  dump flag (write a verified snapshot mid-run and continue). These are simple
  `touch <path>` file checks; no external services.

---

## Ordering & priorities

1. **Task 1** (XGate + comp bit) — everything depends on it.
2. **Task 2** (merge catalogue + guard) — needed by compression and contraction.
3. **Task 3** (ESOP/ANF `fcompress`) — the highest-value single piece; it retires
   the DB-reassembly detour and gives the honest effective-size metric.
4. **Task 4** (directional mixing walk) — the main engine.
5. **Task 6** (metrics) — do alongside Task 4 to measure it.
6. **Task 5** (twists) and **Task 7** (format/CLI) — parity polish, lower urgency.

## What to keep vs retire

- **Retire (or demote to a demo):** `reassemble_g57`, `g57_pair`, `slot_commute`,
  `shoot_and_reassemble`, `collision_shoot`, `transport_mix`. These assume the
  "collapse back to pure g57" model that Task 3 replaces. Keep them until Task 3
  + Task 4 are working, then remove.
- **Keep:** `commute`/`split`/`cross`/`fires`/`apply`/`fragment_polys`/`phi` and
  all their tests — these are the correct primitives to build `XGate` on.
- **Orthogonal, leave alone:** `src/bin/frozen_key_check.rs` and the
  `fragment_polys` → DB-key path verify that fragment polynomials match the
  database keys used by the *separate* `sss` expansion pipeline. That work is
  correct and unrelated to `fmix`; don't delete it, just don't build `fmix` on
  top of it.

## Verification conventions (apply to every task)

- Every new primitive lands with a brute-force test over all `2^n` inputs for
  small `n` (n ≤ ~10), in the style of the existing `fragment::tests`.
- Every mixing move is exhaustively verified on its local support before commit,
  plus a sampled global equality check against the untouched input.
- Run `cargo test --lib fragment::` (and any new bin's tests) after each task;
  keep the suite green before moving on.
