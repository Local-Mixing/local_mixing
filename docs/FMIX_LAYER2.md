# fmix Layer 2: A Best-Effort Size Contract for Phase A

*2026-07-31*

> Markdown rendering of `docs/FMIX_LAYER2.tex`. The PDF is authoritative for figures, diagrams and display math.

## Abstract
Layer 2 turns phase A from a set of hand-tuned rates into a *contract*:
a caller states a size trajectory — expand to R_1× the input by
N_0 units of effective work, hold until N_1, compress toward
R_2× within a further budget — and a controller delivers it, or
says clearly why it cannot. The single lever is `p_mix`, the per-round
MIX/COMP coin; the controller identifies the plant from the monitors the
run already keeps, and steers. Crucially it does *not* model twists:
it measures their effect as a disturbance, which is what lets a profile run
at any twist rate without being told the rate. This note gives the
mechanism, the design decisions (especially the anti-over-steer measures),
what validation established, and the eight-arm experiment now running on a
100k-gate, 512-wire slice. Headline from the runs so far: the four
R_1 arms track their setpoints inside ± 3% across expansion and
hold, and there is a hard *twist ceiling* — at
`p_twist` =5×10_-3 no size contract is achievable at any
lever setting.

## What layer 2 is

### The phase-A defaults (`–phase-a`)

A single flag fixes the phase-A operating point: g57-word twist brackets
(`–twist-g57`) at `p_twist` =5×10_-4;
`–db-advance` on (per the standing directive — DB splice products
must ride their assigned direction); equal parts convex and contiguous
window sampling (`p_convex` =0.5); MIX with
`p_mingen` =0.6 and the curated cascade
(`–curated`, `–mix-pay-random`); COMP with
`p_mingen` =0 and the regular store only. Any flag given
explicitly on the command line wins, and the startup banner prints the
*resolved* values, so a run states what it actually is.

**`–mix-pay-random`** is the one selection change: when a MIX
window's store answer contains only *larger* spellings, take a
uniformly random one rather than a minimal one. This buys growth and
spelling diversity per paid splice — and gives the controller a stronger
up-lever.

### The size profile (`–profile N0,N1,N2,R1,R2`)

Three phases, in effective-work units E (cumulative moves per gate —
the same clock all our analysis uses, so the contract is size-independent):

1. **expand** to R_1 · S_in by E = N_0;
1. **hold** near R_1 · S_in until E = N_1;
1. **compress** toward R_2 · S_in, ending on arrival
or at N_2 — whichever comes first.

The setpoint S^*(E) is the corresponding piecewise-linear trajectory.
N_2 may be an absolute mark (N_2 ≥ N_1) or the compression leg's
*budget*; a value below N_1 can only mean the latter, and the run
logs the absolute end mark it derived. A completed schedule *ends the
run* (`MixStop::ProfileDone`) rather than burning the remaining move
budget outside any setpoint — which, at a twist rate the lever cannot
offset, actively undoes the compression leg (measured: +309% over target
before this stop existed).

### One size authority

`fmix` previously had three mechanisms pushing size around — the
thermostat's static `target_size`, the size brake's
`size_hi`/`size_lo` guard rails, and the `p_mix` coin.
Adding a controller as a fourth would have produced two controllers on one
plant: the profile pressing the accelerator through phase 1 while the brake
(seeing "runaway growth") and the thermostat (still targeting the input
size) pressed back — and the plant estimates would have been polluted by
their interference. So while a profile is active it is the *only* size
authority: it owns `target_size`, continuously setting it to
S^*(E) so the thermostat is conscripted rather than disabled, and the
static brake is inert. Passing `–target-size`, `–size-hi`,
`–size-lo` or `–p-mix` alongside `–profile` is refused
outright rather than silently overridden.

## The controller

### Plant identification

Every `–prof-cadence-eff` of effective work (default 0.5) the
controller reads the run's own counters and estimates three quantities, all
in *gates per move*:

- ^ g — drift if the lever were pinned at 1 (all MIX);
- ^ s — removal rate if the lever were pinned at 0 (all COMP);
- ^ d — the **disturbance**: the residual between the
observed total drift and the drift the DB move accounts for.

^ g and ^ s come from per-mode gate-delta counters added for this
purpose, normalised by the lever actually in force over the interval (an
arm that saw too little of the interval keeps its previous estimate rather
than dividing by ≈ 0). All three are EWMA-smoothed
(`–prof-ewma`).

### Why the disturbance term is the whole trick

The controller cannot steer the twist rate — twists are a layer-1
mechanism with their own purpose — so it must *cope* with it. Rather
than modelling twist growth (which would need the rate, the bracket cost,
the absorption statistics…), the controller measures what it does:
everything that changes size and is not the DB move lands in ^ d.
Twists dominate that residual; expansion moves and thermostat contractions
also live there. The lever then solves

```tex
p ^ g - (1-p) ^ s + ^ d = v^*,
```

where v^* is the slope that lands on the setpoint one cadence ahead, plus
a small integral term on the tracking error. A twist-heavy run simply gets
a lower `p_mix`, automatically, with no knowledge of the twist rate.

The running experiments validate the mechanism directly:
^ d = +0.0000 with twists off, +0.0042 at
`p_twist` =5×10_-4, +0.0388 at 5×10_-3.

### Not over-sampling, not over-steering

Measurement is O(1) counter reads — there is no census cost — so
"over-sampling" is purely a statistical concern, addressed by the cadence
and the EWMA. Over-steering is the real risk, and four guards address it:

- **cadence in effective-work units**, so control frequency scales
with circuit size and each interval contains tens of thousands of rounds;
- **deadband** (`–prof-deadband`, default 2%): no lever
change while the size is close to setpoint;
- **rate limit** (`–prof-dp-max`, default 0.1 per update)
— with an *escape hatch*: the limit is lifted while more than four
deadbands from the setpoint, because a 0.1-per-step crawl cannot catch a
ramp only six effective-work units long. Gentle where gentleness matters,
free where it does not;
- **phase hysteresis**: the integral term and saturation counter
reset at each phase boundary.

### Best effort, stated honestly

Not every profile is reachable. When the lever pins and the size is still
clearly off setpoint, the controller logs `profile: SATURATED` and
holds the pinned lever — it does not thrash, and it does not abort. Phase
3 additionally diagnoses the specific infeasibility that matters:
when ^ d > ^ s, the disturbance exceeds the maximum available
removal, so the circuit grows with the lever at 0 no matter what, and the
log says exactly that (*lower the twist rate or relax R_2*). Every
control point prints `phase`, `eff`, `size`, S^*,
`pmix`, ^ g, ^ s, ^ d, the integrator and the
saturation counter, so a run's adherence to its contract is auditable after
the fact.

## What validation established

### The plant

On the 100k gadget at the phase-A operating point: ^ g ≈ 1.7–1.8
gates/move at full lever — consistent with the parallel session's
independent `p_mix` sweep (final size ratio 1.28× → 3.27×
over `p_mix` 0.05 → 0.40, monotone and smooth). Critically,
^ s *decays* as material hardens: from ≈ 0.5 early to
≈ 0.005 late in a 20k run. A static feed-forward table calibrated on
early drift would therefore overshoot badly late in a run — live
estimation is not a refinement here, it is a requirement.

### Two defects the first runs exposed

**Early overshoot.** A `p_mix`=1.0 prior blew past the
expansion ramp before the first update (measured 185k against a 118k
setpoint). Fixed by a moderate 0.5 prior, a first update at quarter
cadence, and the escape hatch above.

**Idling past completion.** Without the profile-complete stop, a
finished schedule kept mixing outside any setpoint; at a twist rate the
lever could not offset, it grew to +309% over target.

### Two preset defects found by reading a live banner

`–phase-a` decided "did the user specify this?" by comparing each
knob to its default *value*, so `–p-twist 0` — a real request,
and also the default — was silently overwritten with 5×10_-4: an
intended no-twist arm ran as an exact duplicate of its baseline. It now
keys on whether the flag appeared on the command line. Separately,
`p_mingen` ships at 0.8, not the 0.6 phase A calls for, and a
stale comment led the preset to skip setting it. Both fixed; the banner now
prints resolved values.

## The running experiments

Eight arms, all from the same 100k-gate, 512-wire slice, sampled ancestry
(K = 256), measurements every 200k moves. Four are the requested R_1
sweep; four isolate one axis each against the `nR20` baseline.

| arm | profile | `p_twist` | what it isolates |
|---|---|---|---|
| `nR15` | 5,50,+20, 1.5,1.2 | 5 × 10_-4 | expansion ratio |
| `nR20` | 5,50,+20, 2.0,1.2 | 5 × 10_-4 | baseline |
| `nR25` | 5,50,+20, 2.5,1.2 | 5 × 10_-4 | expansion ratio |
| `nR30` | 5,50,+20, 3.0,1.2 | 5 × 10_-4 | expansion ratio |
| `notwist` | 5,50,+20, 2.0,1.2 | 0 | twist axis |
| `nR20_tw5` | 5,50,+20, 2.0,1.2 | 5 × 10_-3 | twist axis |
| `nR20_fast` | 5,20,+20, 2.0,1.2 | 5 × 10_-4 | hold duration |
| `nR20_slow` | 5,100,+20, 2.0,1.2 | 5 × 10_-4 | hold duration |

*The question is which shape buys the most mixing quality per unit
of work — read from the sampled-ancestry measures (`desc`,
`cov`, `ent`) along the way and from affine-reconstruction plates
on the finals.*

### Tracking, in progress

| arm | phase | E | size | S^* | error | `p_mix` |
|---|---|---|---|---|---|---|
| `nR15` | hold | 42.2 | 147,766 | 150,000 | -1.5% | 0.000 |
| `nR20` | hold | 28.2 | 193,975 | 200,000 | -3.0% | 0.091 |
| `nR25` | hold | 22.2 | 245,538 | 250,000 | -1.8% | 0.111 |
| `nR30` | hold | 18.3 | 295,169 | 300,000 | -1.6% | 0.085 |
| `nR20_slow` | hold | 28.2 | 193,975 | 200,000 | -3.0% | 0.091 |
| `nR20_fast` | compress | 28.9 | 168,584 | 164,473 | +2.5% | 0.000 |
| `nR20_tw5` | hold | 39.1 | 368,552 | 200,000 | +84% | 0.000 |

Six of seven track inside ± 3%, including `nR20_fast` descending
its compression ramp. (`nR20` and `nR20_slow` agree exactly:
same seed, and their setpoints coincide until E = 50 — a useful
determinism check.)

### The twist ceiling

`nR20_tw5` is the informative failure. At
`p_twist` =5×10_-3 the controller cannot even *hold*:
the lever is pinned at 0 and the circuit still runs 84% above
setpoint, because ^ d = +0.039 gates/move exceeds the maximum
available removal ^ s = +0.021. This reproduces a 20k-scale result
exactly, and it is a genuine constraint rather than a controller
deficiency: **a size contract and a high twist rate are mutually
exclusive**. Choose the twist rate first, then ask for a profile the plant
can deliver.

![(a) size against effective work, actual (solid) versus setpoint
(dotted), for the four R_1 arms; (b) tracking error, shaded band
± 3%; (c) the lever the controller chose; (d) the twist axis — same
profile at three twist rates, with the measured disturbance in the legend.](../reports/ancestry_20260728/layer2_tracking_20260731.png)

*(a) size against effective work, actual (solid) versus setpoint
(dotted), for the four R_1 arms; (b) tracking error, shaded band
± 3%; (c) the lever the controller chose; (d) the twist axis — same
profile at three twist rates, with the measured disturbance in the legend.*

## Caveats and open items

- The running arms use `p_mingen` =0.8, not the phase-A
0.6, because the preset defect above was found after they launched. Every
arm shares it, so the *comparison between profiles* is sound; the
absolute operating point is not exactly phase A. The intended remedy is to
rerun only the winning profile at the exact spec.
- A profile is a whole-run construct: its effective-work clock is not
serialised, so `–profile` on a `–resume` restarts the schedule
at E = 0 (warned at startup).
- Single trajectories, one seed per arm — these are engineering
measurements, not replicated statistics.
- The mixing-quality comparison (which profile shape is best) is the
point of the campaign and is still pending: the arms are mid-flight.
