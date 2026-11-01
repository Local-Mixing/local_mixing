# Stage 2 design investigation: interleaving fires (n = 64, 128), 2026-10-28 to 11-01

Code: branch `ran/review-fixes-20261028` (commits 766e3fc4 .. 83e42b28, 2f8f43cd), knobs
`three_mask.turnover` and `three_mask.interleave` (docs/THREE_MASK_STAGE2.md, "Turnover modes and
fire interleaving"). Campaign arms (variants session, machine1 `~/variants`): `rust` = wire turnover
with the first, ineffective interleave (5b18bff3), `rust2` = wire turnover with the effective
interleave (4a5fe63d), both on base's C and sandwich. Attribution scripts of this note:
`class_depth2.py`, `dead_rebirth.py`, `attrib.py`, `attrib2.py`, `locate_replicas.py` (server
`~/tdp_runs`), `depth.py`, `critical.py` (dependency depth and critical path of an mpmct circuit).

## High level

1. A fire (one sandwich gate executed on masked values, ~170 gates for a two-control gate, ~70
   for a one-control gate) used to be contiguous in the stage-2 output, so a 1000-gate window of
   the mixed circuit F (one fire's footprint, F being ~6.6x stage 2) held a whole fire, and the
   narrow attack factored it. Interleaving spreads a fire over ~2,500 stage-2 gates (~15,000 of F)
   and puts fragments of ~10 fires in any 250-gate window, at zero gate cost.
2. Making this possible meant removing three things in the compute that chained every fire to
   the next: the mask refresh writing serving coordinates, dirty helpers shared with growth, and
   carrier migration inside every fire. The slicing and C's own structure were not the cause: the
   sandwich's dependency width (11 at n = 64, ~19 at n = 128) is the ceiling on fires in flight,
   and we now sit at it.
3. Effect on exposure: the narrow attack on F drops threefold (deeper C births 254 -> 89; per
   window 231 -> 81 at 1000 gates), twofold on the attacker's shortened circuit (251 -> 129). The
   per-read recovery probability of a value, the right stage-2 metric, falls from 18 % to 7 %.
   Wire turnover alone does nothing for exposure (it only reorganizes bank limbo); interleaving is
   the change that matters.
4. What is left: half of rust2's remaining recoveries are in the first tenth of the circuit,
   where the initial serving banks still carry the raw band seed; dead slice gates re-encode live
   values and count like extra reads; the payload register carries C(x) for the second half of
   the circuit (the degree-2 midpoint replica), which is a stage-1 matter (the V gadget).

## Details

**(a) Mechanics and parameters.** `three_mask.interleave = W` (env `THREE_MASK_INTERLEAVE`,
flag `--three-mask-interleave`; default 0) reorders the finished compute by a dependency-preserving
random topological order: a gate follows every earlier gate that writes a wire it reads or reads
the wire it writes, XOR writes into one wire commute, and no gate is emitted more than W original
positions ahead of the earliest gate still waiting. Function-preserving, no gates added, the
default output byte-identical (parity test). W = 4096 is the knee: 1024 gives a span of 1,022
and 7.4 fires per 250-gate window, 4096 gives 2,503 (p50 2,330, p90 4,336) and 9.6, 16384 gives
4,402 (p50 2,341) and 9.7. Nonzero W also switches on three changes inside the compute:

- the differential cubic refresh becomes read-only (`emit_cubic_delta_readonly`: the same 15
  gates in the bracket form b1 b2 a b1 b2, no band coordinate written; before, ~9 serving writes
  per fire ordered every fire against its neighbours);
- dirty helpers come from per-fire lanes of a reserve that growth never writes (bank mode: the
  drained wires of retiring banks; wire mode: the most rewritten growing wires, skipped by growth),
  with sticky slots so a wire keeps its lane; the reserve is a quarter of the band (16..64 wires),
  four per lane: 8 lanes at a 128-wire band (n = 64), 16 at 256 (n = 128);
- the persistent carrier migration transfers roles in one fire box per lane cycle (every 8th
  fire at n = 64, 16th at n = 128) instead of every fire.

`three_mask.turnover = "wire"` (default `"bank"`) turns band coordinates over per wire: promotion
cohorts of 16 take the bank's roles (cohort-local masks, oldest cohort retires as a unit) but a
drained wire returns to growth as soon as its last reference is replaced, least-referenced first.
Same gates per fire; the coordinate lifetime stays budget-bound (one replacement retires one of
the 3 x 2n masks: ~192 fires at n = 64, ~384 at n = 128 with budget 2); per-wire limbo 60 fires
instead of 76 for a bank.

Costs: no gates added; migration gates fall from 50k to 6.5k at n = 64 (stage 2 456k instead of
504k), roles hop 8x less often; the 32-wire reserve lowers rotations from 112 to 91 (serving
coordinates live ~20 % longer); stage 6 compresses the interleaved circuit less (0.866 vs 0.778,
F 3.0M vs 2.7M). The compute's dependency depth is 17 % of its gates (56 % before); the sandwich
alone has depth 261 of 2,832 at n = 64 (C alone 111 of 1,000; balanced C 108), 407 of 7,920 at
n = 128. Fires per 250-gate window: n = 64: 9.6 (group 8), 10.7 (group 32), 11.2 (no migration),
10.4 with 16 lanes; n = 128: 11.1 (group 8), 13.1 (group 16, now the default), 14.3 (32), 15.5
(none). One-control fires span 1,492 (83 contiguous), two-control 2,628 (186).

**(b) Effect on exposure** (variants campaign, same C; R/O = deeper C births recovered,
repeated/once):

| narrow factor window on F | base | rust | rust2 |
|---|---|---|---|
| 125 / 250 / 500 / 1000 / 2000 | 67 / 114 / 170 / 231 / 232 | 110 / 193 / 237 / 281 / 284 | 26 / 45 / 67 / 81 / 85 |
| union of the narrow portfolio | 254/61 | 322/65 | 101/27 |
| shortened circuit, union | 251/79 | 294/66 | 129/47 |

The 2000-gate windows add nothing because 43 % of them fail the synthesizer's composition or
ancilla budgets (30 % at 1000, 13 % at 500); the productive window is budget-limited, and
interleaving raises the per-window ANF cost. F is ~6.6x stage 2 and the shortened circuit ~3.4x,
so a 1000-gate window is one fire of contiguous material in F and two after shortening; the
protection halves there. The validated census on F finds no internal C birth in any arm; its
interior exposures are 10 D births, 4 payload replicas and one sum of payload bits; the rest are
the first 1 % (plain inputs on wires dead S1 gates re-birth) and the last 1 % (plain outputs
after the mask flush). Payload "replicas" of the status table are the tail (169 of 171 births in
the last decile); the structural leak is the degree-2 midpoint replica of the preimage campaign.

Attribution of the window recoveries: recovery rate by number of reads of a value (fires
consuming it), base vs rust2: 1 read 18 % / 7 %, 2 reads 34 / 13, 3 reads 45 / 15, 6+ reads
76 / 30, never read 0 / 0 (287 of 936 births); a per-read probability p of ~18 % (base) and ~7 %
(rust2). A dead S1 slice gate targeting a live value re-encodes it and counts like a read (rust2:
8 % with none, 15 with one, 21 with two, 38 with three; a quarter of births have one). Balancing
C's reads to exactly two removes the never-read third and raises exposure. rust2's recovering
windows: 193 of 388 in the first tenth of the circuit (base: even over C's half); its first-decile
births are recovered at 21 % against 9 elsewhere.

## Suggestions

Clear-cut (implement, or implemented):

1. Adopt `turnover = "wire"`, `interleave = 4096` as the default stage-2 policy once rust2's blind,
   remix and inversion results are in; lanes and the migration group follow the band (done).
2. Pre-grow the whole band in the warm-up, not only the growing half: the initial serving banks
   carry raw seed values for 100-190 fires, where half of rust2's recoveries sit. A few hundred
   gates; measure with one arm.
3. Stage 1: the V gadget on the payload register (secret bijection after the copy, inverse before
   the end; V(0) = 0 only if the reverse contract is kept), and drop the output-port guard and band
   reseed with the reverse contract (no forward downside found).
4. Stage 6: make the acceptance rule count every sandwich birth (D, slicing, copies) and degree-2
   payload replicas at interior cuts; the only census-level interior exposure left is created by
   our own compression.
5. Use the per-read recovery probability p from the attack ledgers as the stage-2 metric.

Pending measurement:

1. Migration every 32nd fire or none (+10-15 % concurrency against fewer carrier hops).
2. Dead-fire treatment: skip the refresh of a target whose value a dead slice gate did not change
   (function-preserving; removes the re-encoding channel).
3. A wider C (layered, width ~20 at n = 64) with `band_wires = 256` for 16 lanes: doubles the span
   and the roles per window; the balanced construction does not widen C (width 9.3 vs 9.0).
4. Replacement budget 4 (coordinate lifetime 96 fires; +17 % gates), and at n = 128 budget 4 just
   to match n = 64's lifetime.
5. An n = 128 arm (nothing at n = 128 has been attacked).
6. One-control fire absorption (`d_t ^= d_u` plus deferred mask retirement, ~3 % gates) only if the
   attribution singles out CNOT reads; the present data does not.

## Sandwich redesign for gate mobility (stage 1)

The fires in flight cannot exceed the dependency width of a sandwich block, and the blocks
(C with S1, then D with S2) are in series because D consumes C's outputs, so the sandwich's width
is the block's width: 10.9 measured at n = 64, i.e. C's own 9 plus a little from the slicing.
Simulated block-1 widths at n = 64 (1,000 C gates, 384 S1 gates; `mobility.py` in the session
scratchpad; the random case reproduces the measured depth):

| C | no slicing | S1 random interleave (today) | S1 in idle slots | S1 random + E(300) on the high half |
|---|---|---|---|---|
| random | 9.4 (depth 106) | 11.8 (117) | 12.9 (107) | 14.5 (116) |
| layered, 16 gates per layer | 16.7 (60) | 16.3 (85) | 22.3 (62) | 18.9 (89) |
| layered, 21 gates per layer | 20.8 (48) | 18.2 (76) | 21.3 (65) | 21.9 (77) |

1. **Layered C and D.** Sample C as layers of disjoint three-wire gates (random partial
   matchings, 16 to 21 gates per layer at n = 64) instead of a random sequence: width 17 to 21
   against 9, no extra gates, the same g57 gate form; D likewise. Whether the layered
   distribution is acceptable for the trapdoor is RC's call. The balanced construction does not
   widen C (9.3). At n = 128 the random C already has width ~19; layering would give ~40.
2. **Slice gates in idle slots.** The random interleave of S1 with C costs a layered C 40 % of
   its gain (depth 60 to 85) because a slice gate's low-half target pins it between C's accesses
   of that wire. Choosing each slice gate's target and low control among wires the current C
   layer does not touch, and emitting it in that layer, keeps the depth at C's own (62). The
   slice gates have this freedom by design: their on-slice function is the identity and their
   off-slice junk is arbitrary.
3. **Slice gates on dead values.** A dead S1 gate on a live C value re-encodes it in stage 2 and
   counts like an extra read (a quarter of births get one). Targets chosen among wires whose
   current value has no future read remove that channel at no cost.
4. **A companion E on the high half during block 1** (positive-polarity gates only, so E(0) = 0
   keeps the forward slice honest and S1 dead) adds concurrency and decoy fires, but at the cost
   of its own fires (300 gates = +10 % of stage 2 for +2 to +3 of width); layering is the
   cheaper lever. In block 2 the V gadget plays this role for free, since V (high half) and D
   (low half) are independent and interleave.
5. **The midpoint.** D depends on all of C's outputs, so fires are most pinned around the copy
   column; this is also where the degree-2 payload replica lives. V helps on both counts: it
   adds independent gates there and removes the replica. Keep the copy gates floating backward
   only (they cannot pass V).
6. **Moving S1/S2 to the ports** (guards keyed on the high half, like the band guard) would
   remove 27 % of the fires and the dead-fire channel entirely and leave C's full width, at the
   price of no longer hiding C's gates among slice gates inside the block; a reduced count of
   slice gates (s = 128) placed as in 2 and 3 is the intermediate option.

With a layered C and slice gates in idle slots, the n = 64 block width goes from 11 to ~22, so
the span of a pinned fire (width times fire length) doubles to ~5,000 stage-2 gates and a
250-gate window holds ~20 fires; stage 2 then needs 16 lanes (`band_wires = 256`) and the
migration group follows.

## Stage-2 concurrency bounds and the band width

Independently of stage 1, two stage-2 mechanisms bound the fires in flight, both set by the band
width: the helper lanes (reserve = band/4, four growth-free wires per lane, so lanes = band/16:
8 at a 128-wire band, 16 at 256; fires sharing a lane are ordered through its dirty helpers, and
the window holds at most about "lanes plus two" fires) and the migration group (one migrating
fire per lane cycle, a barrier for the coordinates it moves, worth about 15 % of concurrency).
Measured fires per 250-gate window: n = 64, band 128: 9.6 (group 8), 10.7 (group 32), 11.2 (no
migration; the sandwich width is 11); band 256 with 16 lanes: 10.4. n = 128, band 256: 11.1
(group 8), 13.1 (group 16, default), 14.3 (32), 15.5 (none; sandwich width ~19, so the 16 lanes
bind). Switching the migration off (relying on stage 3's twists for wire hopping) lifts the
barrier but adds no lanes: the migration never holds band wires beyond a fire box.

Widening the band buys lanes (band/16) and, second-order, more serving coordinates (half the
band: fewer mask references per coordinate, a larger mask vocabulary) and a larger growing pool;
it does not shorten the coordinate lifetime (budget-bound). It costs width in every later stage
(the adaptive census per cut, the planner's ANF, stage 6's 1,024-wire cap: band <= 512 at
n = 256), about five ingress gates per extra wire, and growth diluted over more wires (scale the
8 growth gates per fire with the band, ~5 % of gates per 128 wires, to keep the coordinates'
degree); the reserve cap of 64 wires in the code must be lifted beyond 16 lanes. Recommendation:
lanes matched to the sandwich width, about 16 band wires per lane, and no wider:

| configuration | sandwich width | band | lanes |
|---|---|---|---|
| n = 64, random C (today) | 11 | 128 (keep) | 8 |
| n = 64, layered C with slice gates in idle slots | ~22 | 256 to 384 | 16 to 24 |
| n = 128, random C | ~19 | 256 to 384 | 16 to 24 |
| n = 128, layered C | ~40 | 512 to 640 | 32 to 40 |

## Keying at the port, no S1, and the clean high half (stage 1 + stage 2)

RC's preferred variant of suggestion 6 above: key the input-port guard on the n top wires (the
high half) instead of interleaving S1 with C, and remove S1. Block 1 is then C alone (27 % fewer
fires, no dead-fire re-encodings, C's full width), and the high half is untouched, hence zero on
the honest slice, throughout block 1: n effectively clean ancillas during the computation of C.
The price is the chaff S1 provided inside block 1 (38 % decoy fires among C's), unmeasured against
the attacks we run. What the clean half is and is not good for:

- **Not a cheap source of concurrency.** A clean ancilla is restored at the end of a fire and the
  next fire's read must follow that write, so fires sharing a wire stay ordered; lanes still come
  from disjoint helper sets. The high half would add n/4 lanes (16 at n = 64, 32 at n = 128) in
  block 1, but as clean helpers, with the restoring product chains that cost about 40 % more fire
  gates (restored helpers 707k vs dirty 502k at n = 64); the band gives a lane for four wires at
  about five gates each. Dirty use of the high half is out (data-dependent garbage before the
  copy).
- **No masks on the zero roles during block 1.** Today the 64 high roles carry three masks each
  while holding nothing: half of the 1,152 references the replacement budget drains belong to
  zeros. Dropping them halves the reference load of block 1: coordinate lifetime ~192 to ~96
  fires at the same budget, or budget 1 at today's lifetime (about 8 % of the gates). It also
  removes 64 known-plaintext relations (a masked zero satisfies carrier = sum of its monomials at
  every cut).
- **A one-gate copy at the midpoint.** With y_i a plain zero, the copy is a single CNOT of the
  carrier, y_i inheriting x'_i's masks until its first refresh separates them: no four-unit
  linear expansion of a payload bit at the centre, and a shorter midpoint bottleneck.
- **Free landing wires** for carrier hops in block 1, if the migration is kept.

Cost: stage 2's contract becomes "correct on the honest slice" for the high half (the witness
checks and the every-band-state verification take it as zero), the high roles get their masks at
the copy, and the off-slice function changes, which the port keying makes irrelevant. None of it
touches the fire design, so it combines with the lanes and the layered C.

## The second half of the sandwich: S2, V and the reverse contract

**The reverse contract can live at the ports.** With both port guards keyed on the high half
(dead when it is zero, junking the low half otherwise), S1 and S2 are both unnecessary. Forward
on (x, 0): the input guard is dead, C runs, the copy puts C(x) up, V re-encodes it, D junks the
low half, V inverse restores C(x), the output guard sees a nonzero high half and junks the low
half further. Reverse on (a, 0): the output guard is dead, V maps zero to zero, D inverse runs
clean, V inverse maps zero to zero, the copy writes D inverse of a into the high half, C inverse
runs, and the input guard, seeing a nonzero high half, junks the low half, which is what S1 did
in reverse. The symmetric design is therefore: guards keyed on the high half at both ports (plus
the band guards), C | N | (D on the low half, V on the high half) | V inverse, with V(0) = 0
(positive-polarity gates, free). Without the reverse contract the output guard and the V(0) = 0
condition go; nothing else changes.

**Why S2 should not read the re-encoded register.** Every read of a value is a fire exposing it
with probability ~7 % per read in the interleaved design; S2 reads each register bit about six
times, so with V in place it would hand the attacker about a third of V(C(x))'s bits per sampled
input: V's input-output behaviour, and V is small. Without S2 the high half between V and V
inverse is read by nothing but V inverse, so V's values are exposed only through its own fires.
Chaff in block 2, if wanted, should read the low half (D's state), not the register.

**Resilience versus efficiency (n = 64):**

| | fires | fixes | risks |
|---|---|---|---|
| today | 2,832 | | midpoint degree-2 replica, S2 reads of the payload, dead-S1 channel |
| guards at the ports, no S1/S2, V of 300 gates | 2,664 (-6 %) | the replica between V and V inverse, both slice channels, C's full width and the clean high half in block 1 | no chaff among C's and D's fires; V learnable if too small |
| the same with V of 500 gates | 3,064 (+8 %) | the same, V harder to learn | |

A net gain in both columns for a V of a few hundred gates, with one thing to measure first: the
structural attacks (reconstruction walks, slice matching) on a block 1 that is C alone, since the
38 % decoy fires are what is given up. V inverse should be emitted interleaved with D's tail, not
as a separate block before the mask flush.

**No clean half in block 2.** The clean high half exists only in block 1: from V onward the high
half carries V(C(x)) and the low half D's state, so all 2n roles are live and masked, the
reference load is the full 1,152 and the coordinate lifetime the budget-bound 192 fires. The
register's roles are idle between V and V inverse but not clean (dropping their masks would put
V(C(x)) in the clear). The block-1 gains (halved references, no known-plaintext relations, the
one-gate copy) stay block-1 gains. Two paid levers if block 2 should match: carry the idle
register with one mask instead of three (references 1,152 to 1,024) or a larger replacement
budget in block 2 only (budget 4 there: 96 fires, about 8 % more gates overall); measure the
block-1 effect before buying either.
