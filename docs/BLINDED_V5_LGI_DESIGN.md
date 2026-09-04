# Blinded-V5: the LGI-based compute module

*Design rationale, construction, parameters, and measurements for the
locally-geodesic-identity (LGI) computation stage.*

## Where it fits in the pipeline

A secret circuit `C` on `n` wires is protected in two steps: it is first
**gadgetized** into an equivalent but structurally-scrambled circuit on `2n`
wires, which is then **mixed** (fmix). Blinded-V5 lives inside the gadgetizer,
and this document assumes you know the earlier gadgetize designs but nothing
about V5.

The **gadgetize module** is a **5-stage** pipeline, each stage on `2n` wires
(data `0..n`, band/junk `n..2n`):

1. **slice** — a junk-guard zero-slice keyed on the band (dead at the input
   port, where the band is 0);
2. **seed the band wires** — each band wire set to `x_i ∧ ¬x_j` from the honest
   inputs;
3. **compute** — realise the input circuit `A` on the `2n` wires so the data
   half carries `A`'s output and the band is only read;
4. **re-seed / re-randomise the band wires** — band turnover;
5. **re-slice** — the closing junk-guard slice (fires at the output port).

One term to keep straight: the circuit `A` that the gadgetizer computes is **not
`C` itself**. It is the **sliced sandwich of `C`** — the output of a *separate*
**sandwich module** that wraps `C` (with a random `D`, interleaved slice blocks,
a floated column) into a `2n`-wire circuit with `A(x,0) = (junk, C(x))`. The
sandwich *generates* the circuit that the gadgetize module then computes and
mixes; it is a different module and is not one of the five stages above.

**Blinded-V5 is a drop-in replacement for stage 3 — the compute.** Where the
earlier compute stages (the drip `route_fire`, the product-share carriers)
realise `A` by *routing* each operand into position and firing a gadget there,
Blinded-V5 instead embeds `A` **inside one enormous masking identity**: the
compute's body is a cloud of long masking identities on the band, and `A`'s gates
fire from *within* the mask by a momentary, reversible unmasking of their
operands. It takes `A` and emits an equivalent `A'` on the `2n` wires; the other
four stages are unchanged. (V5 re-randomises the band *internally* as part of its
own masking hygiene — §2 — which is distinct from the separate stage-4 re-seed.)

Source: [`src/preprocessing/blinded_v5.rs`](../src/preprocessing/blinded_v5.rs).
Driver: `gen_sandwich_gadget … blinded-v5`; pipeline:
`gss_mix.sh --gadgetization-mode blinded-v5`.

---

## 1. Why this design — the rationale

The compute stage must move the circuit's intermediate state as far as possible
from the plaintext computation `A`, while keeping `A'` exactly equivalent to `A`
on the data wires. The design is a chain of forced moves.

**(a) Long identities are the only thing that moves the state substantially.**
Across the HD (Hamming-distance) and affine heatmaps, short local edits barely
displace the state: the ridge of recoverable structure only recedes when the
masking identity woven through a wire is *long* — comparable to the circuit
itself. Short masks are peeled off locally; a long identity has no local seam.

**(b) The only long identity we know that actually moves the state is the
commutative structure of an identity with a single active wire.** Take one data
("active") wire `w` and a set of control wires; a run of gates that all target
`w` and read only the controls is an identity iff the per-gate increments XOR to
zero, and — crucially — *all of its gates commute*. Commutativity is what lets
the identity be arbitrarily long and be reordered or interleaved freely; a single
active wire makes the whole run a clean identity on that wire.

**(c) We entangle identities on different active wires by separating the control
wires from the active wires and sharing the same control wires across many
identities — again via commutativity.** Every identity targets a data wire and
reads only the common **band** (`n..2n`), so identities on *different* active
wires also commute. The scaffold is one large commuting cloud — many long
identities, on many active wires, sharing controls — and a band wire's value
participates in the masks of many data wires at once.

**(d) After entangling by commutation, we add random updates of the control
wires from the active wires — in bursts.** Band updates **re-randomise** the
control wires (fresh band entropy throughout the run, not just at the seed) and
**break naive commutation-back** (once a band wire is updated mid-run, the
identities reading it before and after no longer commute past it) — *without
ruining correctness*, because each straddling identity is either closed before
the update (STRADDLE) or re-derived across it (REPAIR). The updates are emitted
in **bursts**: one *slot* is `F ≈ 8K` gates all targeting a single band wire `b`,
each control drawn independently as a live-data or a band wire (so a burst spans
all of `b ^= data ∧ aux`, `b ^= aux ∧ aux`, `b ^= data ∧ data`). Bursting serves
two ends beyond a lone update: it **concentrates** the band mixing into few slots
— so fewer straddle-closures interfere with the LGIs — and it gives each band
wire a **data-wire-like burst-of-activity signature** (a data wire fires in a
burst of tens of gates whenever it is written; a single band update would not).

**(e) We build one identity longer than the whole input circuit and embed `A`
inside it via a hidden unmasking of the control wires.** The scaffold is, in
effect, a single enormous LGI over all wires. `A`'s gates fire *from inside the
mask*: at each gate the operands are momentarily and reversibly **unmasked** into
a linear combination of band wires (never bare), the gate fires as an expansion
over those masked wires, and re-masks.

That is the whole design: a long, entangled, self-updating masking cloud with
`A` firing from inside it. Everything in §2 is *how* to realise (a)–(e) as a
single correct circuit.

---

## 2. The construction

`A` has `n` wires and `m` gates. `A'` has `2n` wires. Let `u_w` be the number of
times wire `w` is used (target or control) in `A`, and `w_w` the number of times
it is written (a target).

**Masking atom.** `g57(w,x,y) = w ^= 1 ^ (¬x ∧ y)` (comp = 1; data target, band
controls). A **disjoint-pair LGI** on `w` is `⊕ᵢ g57(w, cy[2i], cy[2i+1])`, a
deg-2 mask (the optimal sparse shape). A `g57` and its reverse **linearise**:
`g57(w,r1,r2) ⊕ g57(w,r2,r1) = w ⊕ r1 ⊕ r2` — the read exploits this (AND
monomials are symmetric and do *not* linearise). A K-*cycle* would telescope to 0
under the read's pair-completion (a bare operand), so disjoint **pairs** are used;
`K ≥ 2`.

**Why one pass — co-sampling.** The masks, the band updates, and `A`'s gates are
**co-sampled**: produced together in a single forward pass, rather than laying a
fixed masking scaffold first and threading `A` through it afterwards. The reason
is hidden firing (§3): every `A`-gate must be straddled by a *real*,
rerand-protected masking identity opened at the exact instant the gate fires.
Only a co-sampled pass can open that identity on demand, so the firing-hiding is
complete while reusing the wire's existing mask budget — at no size cost. (A
fixed scaffold, filled in afterwards, starves at the tail and covers only 60–70%
of gates; §3.)

**The algorithm.** A cheap setup, then a forward pass.

*Setup.*

1. **Mask budget.** Give each active wire `w` exactly `u_w+1` masking identities,
   split into `w_w` **straddle opens** (one per write to `w`, opened *on demand*
   at the gate that writes it — for hidden firing) and `reads_w+1` **filler
   opens** (opened between gates to keep `w` masked while it is read). At most
   `max_open` are open on a wire at once. This is the same budget a fixed
   scaffold would use, so the statistics are unchanged.
2. **Order.** Build the data-hazard DAG of `A` (`compute_deps`): a read is
   ordered after every earlier write of the wire, and a write after every earlier
   read — but same-target XOR writes commute, so there is *no* write-after-write
   edge. Any linear extension preserves `A`'s function; a Kahn **ready queue**
   holds the gates whose predecessors are all placed.
3. **Rerand plan.** A shuffled list of `straddle_slots + repair_slots` band-update
   *slots* (§1d; auto `straddle_slots ≈ m/(4K)`, `repair_slots = 0`).
4. **Rate.** Fire one rerand slot every `total_steps / #slots` primitive steps (a
   *step* = one `A`-gate placement or one filler open; `total_steps = m + Σ
   filler`), so the slots spread evenly and the last lands *during* the pass — no
   end-flush (a surplus of requested slots over `total_steps` warns rather than
   truncating silently).

*Forward pass* — repeat until all `m` gates are placed:

1. **Pop** a ready `A`-gate `g` (target wire `c`).
2. **Masked read.** For each operand wire, *linearise* its currently-open masks —
   complete each open `g57` with its reverse so the wire carries `operand ⊕ ρ`,
   `ρ` a linear XOR of band wires — topping up with fresh disjoint pairs until
   `|ρ| ≥ min_mask` (a **hard floor**: never bare, and never thinly masked even
   when the open pairs happen to cancel). Remember the reverse gates as the *undo*.
3. **Fire from inside the mask.** Expand `c ^= comp ⊕ lit(a)∧lit(b)` as
   `(a'⊕ρ_a)(b'⊕ρ_b)` into a batch of monomials over ONLY the masked control
   wires and band wires — never bare `a`, `b`, or `a⊕b` (0/1/2 controls and all
   polarities; the 0-control fire is `¬comp`).
4. **Straddle (hidden firing, §3).** Split the monomial batch into two halves,
   shuffle each **independently** (they all target `c` and commute), emit the
   first half, **open a fresh LGI on `c` here** (one of its straddle opens,
   generated on demand), then emit the second half. The mid-fire open makes the
   module's net XOR on `c` equal `Δ ⊕ secret-mask`, not the bare increment `Δ`.
5. **De-linearise** — emit the undo gates, restoring the operands' masks.
6. **Maybe rerand** — if the calibrated rate says so, emit one band-update slot
   (below).
7. **Release & fill** — decrement the in-degree of `g`'s dependents, moving any
   that reach zero into the ready queue; then open a few **filler** LGIs on wires
   chosen by remaining filler budget.

*Finish.* Drain any remaining filler opens, then close every still-open LGI so
the data wires end holding `A`'s true output.

**The rerand slot** (step 6) is a §1d burst: one band wire `b`, `F ≈ 8K` updates.
A STRADDLE slot first **closes** the masks reading `b` (this is what thins masking
past the ≈1024 knee — hence keeping `straddle_slots` at that budget); a REPAIR
slot brackets the burst with each such mask's `b`-reading `g57` (old `b` cancels,
new `b` re-adds, so the mask stays open — no thinning). Running in the same pass,
the band updates protect the straddle opens automatically.

**Correctness** is verified exhaustively over all `2^n` inputs × many band
settings for `n ≤ 6`, `K ∈ 2..=5`, all `max_open` and both rerand kinds
(`scratchpad/v6`, 891 gadgets, 0 mismatches — plus a 360-case check that the
gate reordering preserves `A`), and end-to-end in `gen_sandwich_gadget` (forward
+ reverse-honesty sample-verify PASS).

---

## 3. Hidden firing

An **atomic**, mask-restoring gate module would leak *which gate fired*: the
active wire's before/after XOR across it equals the true gate increment
`Δ = comp ⊕ lit(a)∧lit(b)`. That per-module increment is a robust invariant
(it survives the downstream mixing), so an adversary who segments the mixed
circuit at the module boundaries reads off `A`'s gate list one gate at a time.

The fix uses the abundant masking gates on the **same active wire**: split the
gate's fire and emit one of `c`'s LGI **opens between the halves**. That mask
toggles `c` mid-fire and stays open past the module, so the module's net XOR on
`c` is `Δ ⊕ (that LGI's secret band-mask)` — never the bare increment. Because
the LGIs, rerand, and placement are co-sampled, the straddling LGI is a *real*
scaffold LGI (rerand-protected) opened exactly when the gate is placed, so
**every** gate is covered, and it reuses the wire's existing `u_w+1` LGI budget,
so there is **no size cost**. (Measured on the n=128 sandwich: 7920/7920 gates'
firing hidden.)

Earlier attempts and why they were dropped: same-active-wire *A-gate* weaving
(only ~23% of gates have a woven partner); an *injected* per-wire firing mask
(100% coverage but +43% size, because it keeps every wire extra-masked and needs
its own rerand repair); straddling *pre-existing fixed-scaffold* slots (60–70% —
the tail of each wire is slot-starved). Generating the slot on demand in a
co-sampled pass is what makes coverage complete at no cost.

---

## 4. Parameters and why

| knob | prod. | meaning | why this value |
|---|---|---|---|
| `K` | **2** | band wires per LGI = per-LGI **mask width** (⌊K/2⌋ disjoint pairs; `\|ρ\| ≈ max_open·K`) — *not* the identity's temporal length | affine- and deg-2-neutral across `K` (§5); smallest is best. Size grows ~linearly in `K`, read cost quadratically in `\|ρ\|`, so large `K` explodes (K16 ≈ 18M). Odd `K` wastes a wire (K3 ≡ K2). |
| `max_open` | **3** | rolling cap on simultaneously-open LGIs per wire | wider `ρ` = more local hiding, but read cost is quadratic in `\|ρ\|`; 3 is the knee. |
| `min_mask` | **auto = `max_open` = 3** | **hard floor** on `\|ρ\|` (masking wires) per operand read | guarantees no operand is ever read under fewer than 3 masking wires, even in a rare draw where the open pairs cancel (measured worst read `\|ρ\|` rises 2 → 4; mean `\|ρ\|` ≈ 5.9 unchanged; +0.2% gates). |
| `rerand_level` (straddle slots) | **auto = `m/(4K)`** (≈875) | close-straddling-masks band-update **slots** | at the safe side of the ≈1024 thinning knee; the *slot* count (not the gate count) is what thins. |
| `rerand_repair` (repair slots) | **0** | re-derive-across-update band-update slots (no thinning) | off by default; add slots for extra band turnover at no masking cost. |
| `rerand_burst` (`F`) | **auto = `8K`** (=16 at K=2) | band-update gates **per slot** (the burst) | comparable to a data wire's write-burst (tens of gates), so band wires carry a data-wire-like activity signature; `slots × F ≈ 2m`. |

`K` sets each LGI's **mask width** (control wires per LGI), *not* its temporal
length — the identity's circuit-spanning length is a property of the whole
scaffold (§1e), which is on by construction.

---

## 5. Measurements

All measurements are on the current co-sampled build with **burst rerand + the
masking floor** (n=128, `|C|=|D|=3000`, K=2). They match the pre-firing-fix
baseline to within noise — neither the hidden-firing redesign nor the burst/floor
refinements moved them (the burst change touches only the band wires, so it
cannot move the data-wire ridge): degree-1 `meanH = 0.4966`, degree-2
`meanH = 0.4967`, exposed-C `< 0.35` ≈ 5.6–5.7%.

### 5.1 Affine ridge (linear recovery of source-C segments)

Adversary: least-squares recovery of each source-`C` wire-segment from a bounded
set of gadget wires (`hmap_affine --degree 1`; `meanH`: 0 = recoverable, 0.5 =
hidden).

![affine heatmap](blindedv5_affine.png)

- **`meanH ≈ 0.498`; ridge depth ≈ 0.05** — C-segments hidden overall.
- The **interior (~78% of rows) is fully hidden** (`H ≈ 0.49`). The only recovery
  is a thin fringe at the two **endpoints**.

### 5.2 The exposed fringe is public I/O — input segments are *not* leaked

The `≈ 6–10%` of source-C segments that are linearly exposed
(`min-H < 0.25–0.35`) are **not scattered**: they are the two corners of the
diagonal. The **input side is masked as soon as the compute begins** — the plate
shows the top rows almost entirely blue, with recovery confined to the extreme
top-left corner, i.e. the raw **input port** where `x` is public *before* any
masking is applied. The output side (bottom-right) is the necessarily-delivered
`C(x)`. In other words, the input **wire segments show a lack of linear
exposure** beyond the trivial public input; the boundary-cushion knobs measured
no effect because the fringe is unavoidable public I/O, not a leak.

### 5.3 Degree-2 recovery

A quadratic adversary (`hmap_affine --degree 2`, products over a data+band slice
— a lower bound on the deg-2 leak):

![degree-2 heatmap](blindedv5_deg2.png)

- **`meanH ≈ 0.497`; depth ≈ 0.05.** The quadratic adversary reaches only ~0.01
  deeper than affine, and only into the same low-degree I/O fringe; the interior
  stays fully hidden. Degree-2 is flat across `K`, like affine.

### 5.4 Rerand: bursts, straddle vs repair (n_A = 256, affine deg-1)

Each rerand slot is a **burst** of `F ≈ 8K` gates on one band wire; the **slot**
count — not the gate count — drives the effect, since a straddle slot closes the
masks reading `b` exactly once per slot. An early single-gate-dose study fixes
where the thinning knee is (dose = total straddle/repair *gates*):

| straddle | repair | gadget | meanH | affDepth |
|---|---|---|---|---|
| 0 | 0 | 670k | 0.4979 | 0.0550 |
| 1000 | 3000 | 668k | 0.4959 | 0.0514 |
| 8000 | 0 | 607k | **0.4878** | **0.0681** |
| 0 | 8000 | 780k | 0.4985 | 0.0549 |

Heavy **straddle thins** (meanH ↓, ridge ↑, gadget *shrinks* as masks close);
heavy **repair does not** (both = baseline, gadget *grows* from the pre/post
pairs). The production plan keeps straddle **slots** on the safe side of the
≈1024 knee: auto `≈ m/(4K)` (~875 for the half-size sandwich) `× F = 8K` ≈ `2m`
band-refresh gates, **no repair**. Fewer, fatter slots mix the band as hard as
many single updates while interfering less with the LGIs (and give the band its
data-wire signature), and — because the thinning is driven by the *slot* count —
they sit at the baseline floor (deg-1 `meanH = 0.4966`, deg-2 `0.4967`).

### 5.5 Through the GSS pipeline (K = 2 vs K = 4)

Four arms through `gss_mix.sh --gadgetization-mode blinded-v5` (n = 128,
half-size |C| = |D| = 3000, `--expand 2`, fresh independent CSPRNG seeds),
measured at the **2-eff snapshot** (the phase-A grow to ≈1.7×, an early mixing
checkpoint; `hmap_affine --degree 1`, 92 C-segments):

| arm | gadget `g_in` | 2-eff gates | meanH | exposed-C `<0.25` | `<0.35` |
|---|---|---|---|---|---|
| K = 2, hold 10 | 563,946 | 962,598 | 0.4945 | 6/92 | 11/92 |
| K = 2, hold 20 | 564,394 | 969,380 | 0.4942 | 6/92 | 11/92 |
| K = 2, hold 30 | 564,674 | 960,670 | 0.4951 | 6/92 | 9/92 |
| K = 4, hold 10 | 1,362,010 | 2,300,885 | 0.4965 | 6/92 | 10/92 |

- **meanH ≈ 0.494–0.497 for both K** — affine-neutral across `K` through the
  pipeline, matching the raw-gadget measurements (§5.1) and the pre-firing-fix
  baseline. (These are also the corrected build with burst rerand + the masking
  floor + the even-filler / no-drain schedule; the pipeline statistics are
  unchanged from the earlier build, as the band-only refinements predict.)
- The exposed-C count (6/92 at `<0.25`, ~9–11/92 at `<0.35`) is the same public
  **I/O boundary** of §5.2, confirmed by the per-segment profile: min-H ≈ 0 only
  at the input port (raw `x`) and output port (`C(x)`), climbing onto the hidden
  plateau within ~5 % of `C` — the interior is not recovered.
- The three K = 2 arms agree because the 2-eff snapshot **precedes the hold
  phase**, so it is hold-independent — a seed-level consistency check.

*End-of-pipeline numbers (post split / crossing / fcompress, and the deeper
hold-20 / hold-30 mixing) are pending — the arms are mid phase-A; this section
will be completed when they finish.*

---

## 6. Tradeoffs and the current choices

- **Mask width `K` vs size.** Larger `K` widens each per-LGI mask
  (`|ρ| ≈ max_open·K`), not the temporal length. Hiding is **flat in `K`** while
  size grows ~linearly and the masked-read cost grows quadratically in `|ρ|`, so
  **K = 2** wins; K = 4 only for a wider read-mask.
- **Band turnover vs size (bursts, slots vs `F`).** More band updates = more
  entropy and stronger resistance to commutation-back, but STRADDLE **slots** thin
  the masking past ≈1024 and REPAIR slots grow the gadget. Bursting decouples the
  two knobs: `F` (gates/slot) sets how hard each band wire is stirred and how
  data-like its activity looks, while the **slot count** (`≈ m/4K`, held under the
  knee) sets the thinning. Auto `m/4K` slots × `F = 8K`, no repair, sits at the
  baseline floor.
- **Masking floor is a cheap hedge.** `min_mask` guarantees every operand is read
  under ≥ 3 masking wires (worst-case `|ρ|` 2 → 4; mean unchanged); it fires only
  on the rare under-floor draw, costing ~0.2% gates — a hedge against a
  low-probability thin-masking event, not a routine cost.
- **Firing-hiding is free** in the co-sampled build (it reuses the existing LGI
  budget); the earlier fixed-scaffold approaches paid either coverage (23–70%)
  or +43% size for it.
- **The residual fringe is I/O, not a knob.** The ~6–10% exposed segments are the
  public input/output boundary; no parameter removes them.

---

## 7. Running it

```bash
# one pipeline arm, LGI compute, K=2, half-size n=128 sandwich
scripts/gss_mix.sh -n 128 --mcd 3000 \
  --gadgetization-mode blinded-v5 --bv5-k 2 \
  --expand 2 --hold 10 -o RUNDIR
# (K=4: --bv5-k 4). fmix stages require FROZEN_DB_DIR (+ FROZEN_CURATED_DIR).

# just the gadget (stage 2), or the standalone driver:
scripts/gss_mix.sh -n 128 --mcd 3000 --gadgetization-mode blinded-v5 \
  --bv5-k 2 --stop-after 2 -o RUNDIR
blinded_v5_gadgetize src.mpmct1 out.mpmct1 <K> 0 <seed> \
  <straddle_slots=0auto> 3 <active> <extra_lgis> \
  <repair_slots=0> <burst_F=0auto8K> <min_mask=0auto>
```

All rerand knobs default to auto (`straddle_slots = m/4K`, `F = 8K`,
`repair_slots = 0`, `min_mask = max_open`); pass `0` to keep the auto value.
The `gen_sandwich_gadget`/pipeline path exposes the same knobs as the env vars
`BV5_K`, `BV5_RERAND` (straddle slots), `BV5_REPAIR`, `BV5_BURST`, `BV5_MIN_MASK`,
`BV5_MAX_OPEN`, `BV5_EXTRA_LGIS`.
