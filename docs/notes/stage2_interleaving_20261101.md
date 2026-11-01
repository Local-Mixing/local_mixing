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
