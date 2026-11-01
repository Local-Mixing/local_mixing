# Stage 2, three-mask mode

`preprocessing.mode = "three-mask"`, the default stage 2 since recipe v13,
replaces the masked compute of embedded masking by RC's three-mask
construction (design note
`stage2_three_mask_design.md`, 2026-10-20; review
[notes/three_mask_review_20261021.md](notes/three_mask_review_20261021.md)),
in the configuration the server campaigns of October 2026 ran as
"three-mask modified / v2", since recipe v14 (2026-11-01) with per-wire
turnover, fire interleaving in a 4096-gate window and pre-growth on top
(gen's `ThreeMaskParams::v3`; the sections at the end). The outer blocks are
embedded masking's, with the L8 parameters: the opening slice guard, the band seed, the compute, the band
reseed and the closing guard; only the compute differs. The code is
[src/stages/preprocessing/three_mask/](../src/stages/preprocessing/three_mask/)
(`mod.rs` the compute and its configuration, `band.rs` the banks,
`fire.rs` the fires and mask emitters, `mask_rank.rs` the mask forest), a
port of the campaign generator crate; the migration pass is the crate's own
`preprocessing_shuffling` module, which that generator used unchanged.

## The construction

**Carriers.** Every sandwich value $d_i$ (a data role; the sandwich has $2n$
of them) is carried as

$$x_i = d_i \oplus b_{a_1} b_{a_2} b_{a_3} \oplus b_{b_1} b_{b_2} b_{b_3} \oplus b_{c_1} b_{c_2} b_{c_3},$$

three cubic monomials of band wires (`mask_count = 3`; the three persistent
masks of a role never share a bank pair with each other's labels: the mask
forest). A mask is applied and removed by a borrowed-helper Toffoli ladder
(`emit_cubic_mask`: a reversible multiple-control NOT over four dirty band
helpers, or a non-restoring chain of 2k-3 gates with `discard_fire_helpers`,
which leaves garbage only in reference-free growing-bank wires).

**The band** (described in the bank mode, `turnover = "bank"`; the default
since recipe v14 is the per-wire turnover below). $r$ band wires (`band_wires`: $\max(2n, 128)$ rounded up to a
multiple of 32 by default) are divided into `banks` banks (auto: banks of
16 wires, 8 at n=64 and 16 at n=128, the design's geometry; the rounding
to 32 keeps the auto bank count even, which the compute requires). Half the
banks start *serving* (their wires may be mask operands; with `pregrow`, the
default since recipe v14, rewritten `warmup` times before the first masks);
the others are *growing*: between fires they are rewritten by random two-control Toffolis
whose controls lie in the other half of the same bank, or, with
`cross_bank_growth`, in the next bank (`growth` gates per fire, `warmup`
sweeps before the first fire, `visit_sweeps` sweeps per bank visit). Every
`rotation` fires, a growing bank that has seen `maturity` sweeps is
*promoted* to serving and the oldest serving bank starts *retiring*: it is
drained by replacing the masks that reference it (`retire_budget`
replacements per fire, oldest retiring bank first, `max_retiring` banks at
once), and becomes growing again when no mask references it. The band is
seeded by the outer blocks over the first $2n$ wires; when the band is wider,
the extra wires get a reversible XOR ingress from the seeded prefix (the
campaign's `--band-wires 128` at n=64 was exactly $2n$; narrower bands fail
at n=32 because 64 data roles with three masks each exhaust the serving
wires).

**Fires.** A source gate $t \mathrel{\oplus}= f(u, v)$ is compiled into
products of the operands' carriers and masks (`compile_cubic_options_dirty`):
each literal $u$ expands to $x_u$ plus its three cubic masks, the product of
the two expansions is a sum of up to 16 units, each a conjunction of up to
six band literals and two data literals, emitted as a dirty-helper chain on
borrowed band wires (with `random_helpers`, drawn from a shuffled pool of
every reference-free band wire: the generator's "as-is" choice took the
first four wires of the lowest growing bank, which put 80 % of the band
writes on four wires). A five- or six-literal unit is split so that no
helper ever holds exactly a live cubic mask. Around the units the target's
masks are refreshed: a temporary cover mask is applied before the fire and
removed after it (with `unique_cover`, its label is never a live persistent
label), and the oldest persistent mask of the target is replaced
differentially (`emit_cubic_delta`: `temporal_masks`, the telescoping
coordinate replacement that XORs the old and the new cubic into the target
without materializing either), so that a mask never leaves the target bare.
The units are shuffled; every fire ends with the bank growth and, through
`retire_budget`, the drain of retiring banks. A bit-sliced witness checks
every fire boundary during construction.

**Migration.** The fire boxes are handed to the persistent carrier migration
(`segments` segments per box; `preprocessing_shuffling::apply_shuffling`
with `return_home = false`): the data roles hop across band wires for the
whole stream and are swapped back to their ports once, at the tail, so the
sandwich's output contract holds. The outer blocks and the migration are the
same code as embedded masking's.

At the end the construction removes every mask (the roles are bare on their
ports) and the closing guard follows. The generator verified the compute
against the sandwich in both directions on arbitrary band values, and the
whole stage-2 circuit with `verify_payload` (forward payload and reverse
honesty); the gen program runs the same `verify_payload`.

## Sizes and exposure (n = 64, |C| = 1000, server campaigns, October 2026)

| | embedded masking L8 | three-mask as-is | three-mask modified (v2) |
| --- | ---: | ---: | ---: |
| stage-2 gates | 777k | 505k | 504k |
| adaptive stage-6 plateau (gates / rounds; the baseline ran the one-seed "as-is" stage-6 variant, the three-mask arms the v2 variant) | 667k / 24 | 2.37M / 14 | 2.71M / 9 |
| deeper values of $C$ exposed in the published circuit (direct / affine, of 936) | 384 / 424 | 0 / 0 | 0 / 0 |
| C-blind resynthesis of the published circuit (affine) | 503 | 9 | 6 |
| narrow 125-250-gate rewrites (repeated deeper values) | 444 | 264 | 157 |

(From [notes/stage2_stage6_campaign_20261021.md](notes/stage2_stage6_campaign_20261021.md):
the "modified" stage 2 beats "as-is" on every resilience measure; the
three-mask outputs are 3-4× larger than the baseline's because the
synthesizer compresses them far less.) The review's measured levers:
77 % of the three-mask saving comes from the dirty helper chains (implicit
data feedback); 95 % of the promotions stall on the retiring budget; a whole
fire (~158 gates) factors inside a 125-250-gate window, which is what the
narrow attacks exploit.

## Configuration

The `[three_mask]` table (manual: [tdp_pipeline.md](tdp_pipeline.md), "Three-mask");
every key is an error under another mode, and `preprocessing.preset` and the
embedded-masking knobs are errors under this one (the outer blocks are
always L8). The defaults are gen's `ThreeMaskParams::v3` since recipe v14:
the v2 configuration with `turnover = "wire"`, `interleave = 4096` and
`pregrow = true`; `turnover = "bank"`, `interleave = 0`, `pregrow = false`
give v2 (recipe v13's default), and `random_helpers = false` and
`unique_cover = false` on top the "as-is" generator. The gen program takes
the mode as its gadgetization-mode positional (`three-mask`) and the knobs
as `THREE_MASK_*` variables (recipe class in the environment registry; the
driver exports only non-defaults; every `EMBEDDED_MASKING_*` variable is an
error in this mode). `stage12.recipe` and the manifest record every
effective value and `three_mask_outer_blocks=L8`.

## Parity with the campaign generator

The campaign generator (crate `stage2_compute_pilot`, built against this
crate at `4a85b09f`) produced, for n=32, |C| = |D| = 128, seed 1, band 128
and the v2 flags, a 103,605-gate stage-2 circuit on 192 wires. The gen
program of this crate, run as
`gen_sandwich_gadget OUT 32 128 128 160 1 160 1 1 1 three-mask` with
`THREE_MASK_BAND_WIRES=128`, writes the identical bytes (the generator passed
the sandwich's slice size, 160, also as the guard size; the managed driver
passes `10 * 2n`, so the managed stage-2 circuit differs from the
campaign's only in the size of the two guards). The integration test
`tests/integration/three_mask_parity.rs` pins the sha256 of that circuit,
of the sandwich and of the source.


## Turnover modes and fire interleaving (added 2026-10-28)

Two knobs added after the release, off until recipe v14 made both the
default (`turnover = "wire"`, `interleave = 4096`; `stage2_three_mask_parity`
pins the v2 output by setting them, and pre-growth, off):

**`turnover = "wire"`** turns band coordinates over per wire instead of per
bank. Phases are per wire; a *cohort* (16 wires promoted together) takes the
bank's two remaining roles: masks are drawn inside one serving cohort (the
freshest, as `prefer_fresh` does with banks), and the oldest serving cohort
retires as a unit, so every budgeted replacement frees three references of
the cohort being drained, as with bank-local masks. What changes is what
happens to a drained wire: it returns to growth the moment its last reference
is replaced, instead of waiting for the slowest wire of its bank, and
promotion cohorts form from the most rewritten mature growing wires, eight of
each parity, wherever they sit in the band. Growth rewrites growing wires
round-robin with one control among the other growing wires (a rewritten one
when any exists) and one anywhere in the band. `banks` must be 0;
`max_retiring` counts retiring cohorts (0 = half the cohorts minus one);
`maturity` is rewrites per wire. The gate count per fire is unchanged, and so
is the coordinate lifetime floor set by `retire_budget` (one replacement
retires one of the $3 \cdot 2n$ masks); the gain is the capacity that bank
limbo wasted (48 of 128 wires at n = 64) and the end of the positional bank
structure. The report prints the mean and maximum fires a drained wire
waited in its retiring cohort.

**`interleave = W`** reorders the finished compute (after carrier migration)
by a dependency-preserving random topological order: a gate must follow
every earlier gate that writes a wire it reads or reads the wire it writes,
XOR writes into one wire with no read between them commute, and no gate is
emitted more than `W` original positions ahead of the earliest gate still
waiting. Gates of independent fires interleave within the window, so a
125 to 250-gate window of the output holds fragments of many fires rather
than one complete fire (the factoring $(d_u+S_u)(d_v+S_v)$ a window
synthesizer could exploit). No gates are added; the function is unchanged.
The report prints the mean displacement and the mean spread of original
positions in a 250-gate output window (250 without interleaving). The
reordering is positional only: an attacker who clusters gates by
dependencies can undo it on the stage-2 circuit itself, so its value has to
be measured on the mixed output, where stage 3's local identities straddle
gates of different fires.

### What interleaving changes inside the compute (2026-10-28/29)

A dependency-preserving reordering only helps if consecutive fires are
independent, and in the design as released they are not: the compute's
dependency depth is 40 % of its gates (the sandwich itself, C plus slicing
gates, has depth 260 of 2,832 gates, so the serialization is the compute's
own). Three mechanisms chained every fire to the next, and `interleave > 0`
switches each one off; at `interleave = 0` (recipe v13's default) the output
is unchanged:

1. **Mask refreshes wrote serving coordinates.** The differential cubic
   refresh mixed the new coordinate into the current coordinate wire and
   undid it, about nine serving writes per fire; every fire reading those
   coordinates was ordered against it. `emit_cubic_delta_readonly` adds
   the same telescoping terms through the dirty-helper bracket
   `b1 b2 a b1 b2` (the helper's garbage cancels), same gate count, no
   band write.
2. **Dirty helpers and growth shared wires.** Fire $i$ now takes its helpers
   from lane $i \bmod L$ of a reserve that growth never writes (bank mode:
   drained wires of retiring banks; wire mode: the most rewritten growing
   wires, skipped by growth). The reserve is a quarter of the band, 16 to
   64 wires, four per lane: $L = 8$ at a 128-wire band (n = 64), 16 at
   256 (n = 128). A wire keeps its reserve slot, hence its lane, until it
   leaves the candidate set, so lanes stay disjoint across consecutive
   fires. Fires sharing a lane ($L$ apart) are still ordered through that
   lane's garbage.
3. **Carrier migration at every fire.** The persistent migration placed up
   to six transfers inside every fire box, so the coordinates a fire reads
   had just been moved. Under interleaving only every $L$-th fire box
   migrates (`migration_group`, one box per lane cycle): at n = 64, 2,081
   transfers instead of 16,756 and 6.5k migration gates instead of 50k, so
   the stage-2 circuit is about 9 % smaller; roles hop $L$ times less
   often, which is the trade-off to keep in mind.

Measured at n = 64 (one seed, wire turnover; the gen program reports these
as "three-mask fires", traced through the migration by its origin map):

| | fire span in the output (mean / p50 / p90 gates) | distinct fires in a 250-gate window |
|---|---|---|
| no interleaving | 174 / 186 / 188 (a fire is contiguous: 159 compute gates plus its migration transfers) | 2.4 |
| window 1024 | 1,022 / 1,065 / 1,371 | 7.4 |
| window 4096 | 2,503 / 2,330 / 4,336 | 9.6 |
| window 16384 | 4,402 / 2,341 / 11,979 | 9.7 |

The fires a window holds saturate near ten, so a larger window spreads
fires further without mixing more of them; 4096 is the knee. The released
design under the same reordering gave a mean displacement of 31 gates and
755 original positions per 250-gate window (342 and 3,070 now).

**What bounds the parallelism.** Fires in flight cannot exceed the
dependency width of the sandwich (C plus slicing gates): depth 261 of 2,832
gates at n = 64, so about 11 fires per layer (C alone: depth 111 of 1,000,
width 9); depth 407 of 7,920 at n = 128, width about 19. Against that
ceiling, at n = 64 the lanes (8) and the migration group (8) cost little:
without any migration a 250-gate window holds 11.2 fires, with sixteen
lanes 10.4. At n = 128 the migration group was the binding limit: 11.1
fires per window with a group of 8, 14.3 with 32, 15.5 without migration
(16.7 with window 16384, where a fire spans 8,800 output gates). The
group now follows the lane count (16 at n = 128). Going beyond the
sandwich's width needs a C or a slicing with more independent gates per
layer, not more lanes.


## Recipe v14 defaults: pre-growth and the reverse contract (2026-11-01)

Gen's default policy is `ThreeMaskParams::v3`: v2 plus wire turnover, the
4096-gate interleaving window (the knee of the measured spans above) and
**pre-growth** (`pregrow`, `THREE_MASK_PREGROW`), the clear-cut suggestions
1 and 2 of [notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md).
Without pre-growth the warm-up grows only the growing half, so the first
serving coordinates are the raw band seed for 100 to 190 fires, where half
of rust2's recoveries sat. With it, before the first mask the serving half
is rewritten `warmup` times (one two-control band gate per serving wire per
sweep: `warmup` × r/2 gates, 256 at a 128-wire band), in both turnover
modes; the warm-up then grows the growing half as before. The compute still
decodes every fire (its internal witness) and realizes the input both ways
(`pregrow_rewrites_the_serving_half_before_the_first_masks`).

**The reverse contract** (`preprocessing.reverse_contract`,
`STAGE2_REVERSE_CONTRACT`; embedded masking and three-mask only) is a
property of the outer blocks: the opening and closing guards each junk only
the low data half when the band is not zero, so the reversed circuit still
carries $D^{-1}$ of its input on wires $n..2n$, and gen verifies that
(`reverse verify PASSED`) when it is `true`. `false`, the default in the
guarded modes since RC's 2026-11-01 decision, keeps the forward payload
only: the opening guard covers all data wires with the same gate count (its
Feistel rounds alternate halves, accumulating from one and injecting into
the other, so each round's uncompute stays exact), and the closing guard is
dropped; the band seed and reseed stay at both ends; the reversed artifact
then junks both halves, and gen skips the reverse verification. The managed
driver then runs the adaptive stage 6 with `--forward-only` (its payload
contract against the sandwich checks the forward direction only).
