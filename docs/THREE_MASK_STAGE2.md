# Stage 2, three-mask mode

`preprocessing.mode = "three-mask"`, the default stage 2 since recipe v13,
replaces the masked compute of embedded masking by RC's three-mask
construction (design note
`stage2_three_mask_design.md`, 2026-10-20; review
[notes/three_mask_review_20261021.md](notes/three_mask_review_20261021.md)),
in the configuration the server campaigns of October 2026 ran as
"three-mask modified / v2". The outer blocks are embedded masking's, with the
L8 parameters: the opening slice guard, the band seed, the compute, the band
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

**The band.** $r$ band wires (`band_wires`: $\max(2n, 128)$ rounded up to a
multiple of 16 by default) are divided into `banks` banks (auto: banks of
16 wires, 8 at n=64 and 16 at n=128, the design's geometry). Half the
banks start *serving* (their wires may be mask operands); the others are
*growing*: between fires they are rewritten by random two-control Toffolis
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
always L8). The defaults are the v2 configuration; `random_helpers = false`
and `unique_cover = false` give the "as-is" generator. The gen program takes
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
