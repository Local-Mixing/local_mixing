# Pipeline changes of 2026-11-02 (recipe v15)

For everyone who knows recipe v14 (`main` at 2b4b6374). Recipe v15 changes three stages: stage 2's opening guard
and band seeds, stage 3's twists and stage 6; stages 1, 4 and 5 are v14's. The pipeline step by step, with every default:
[PIPELINE_V15_STEPS.md](PIPELINE_V15_STEPS.md); the settings: [tdp_pipeline.md](tdp_pipeline.md).

| Area | Recipe v14 (main 2b4b6374) | Recipe v15 | Why (evidence) |
|---|---|---|---|
| Stage-2 opening guard (no reverse contract, the default) | keyed on the band only, over all data wires, 20n gates: the identity whenever the band is zero, whatever the register B holds | the **two-layer guard** keyed on B and the band: the identity on the data exactly on the honest slice; block 1 spreads the band key and randomizes B (and A), every B bit liable to flip for any one-hot band key; block 2 randomizes A from B, one A bit flipping on every input for any one-hot B key; 9,380 gates at n = 128; not configurable | with the band zero and B nonzero the v14 circuit evaluated the sandwich off its slice: in `circuits/tdp128_s6_20261102.mpmct1` one bit of B changed 2-3 payload bits and none of wires 0..n-1 (section 1) |
| Stage-2 band seed and reseed (no reverse contract) | a private pivot CNOT per band wire, 1-7 more CNOTs from the input, 4 two-literal products: about 11 gates per band wire, 2,800 at n = 128 | 8·n·⌈log2 n⌉ random g57 gates writing the band, any two other wires as controls (7,168 at n = 128), minus pairs a first-layer sandwich gate reads raw; the seed non-constant on every band wire | RC: the band should be a pseudorandom-like function R(A, B) of the data (section 2) |
| Stage-3 twists | rate 0.0005 per mixer round, spans log-uniform from 64 gates (circuit_mixer's preset) | rate 0.001, spans uniform from 100,000 gates to the circuit size, every window inside the circuit | compensates the carrier hops the interleaved stage 2 gave up; no twist fits inside one stage-6 block ([PIPELINE_V14_CHANGES.md](PIPELINE_V14_CHANGES.md) section 2.5) |
| Stage 6 | large-block rounds only (blocks of up to 65,536 gates), judged by the raw gate count, stop fraction 0.02, wide gates kept | pre-cleanup (NOT-push + fcompress), blocks of up to 512 gates judged by the 2-control cost, stop fraction 0.2, descent 128 / 32 / 16 with a cleanup after each, every gate lowered to at most 2 controls; the leak guard on every acceptance and cleanup, now including early C outputs | n = 64: 2-control cost 0.072 of stage 5 (v14: 0.331 after ~16 h, unfinished, 3 early outputs); n = 128: 0.106 in 4.0 h on 12 threads with the bounded effort (v13's stage 6: 0.250 in 4.6 days on 48 threads, 8 early outputs) ([stage6_config_search_20261102.md](notes/stage6_config_search_20261102.md)) |
| Recipe bookkeeping | `tdp_command_recipe=14`, `tdp_stage12_recipe=9` | `tdp_command_recipe=15`, `tdp_stage12_recipe=10`; config schema still 5; a v14 run directory is refused on resume, with a message naming the settings that rebuild main's v14 stages 3 and 6 | |

## 1. The stage-2 opening guard

**What.** The honest slice is (x, 0, 0): x on A = wires 0..n-1, zero on the register B = wires n..2n-1 and on the
band. v14's opening guard (`slice_zero_feistel_guard_dims_all`) was keyed on the band only, so on any input with the
band zero it was the identity, whatever B held, and the rest of the circuit then computed the sandwich on (x, B).
RC (2026-11-02): the guard must be the identity only on the honest slice and pseudorandom elsewhere, G(A, B, C, D) =
(f_{B,C,D}(A), h_{C,D}(B), C, D) with f_0 and h_0 the identity, keyed on three registers for A and two for B. In v13
the slice blocks of stage 1 (S1, keyed on B) had covered B; v14 dropped them and nothing keyed on B any more.

**How.** `two_layer_opening_guard_dims` in `src/stages/preprocessing/slice_guards.rs`, built as RC specified on
2026-11-02, in two blocks:
- **Block 1 randomizes B (and A) whenever the band is nonzero.** First the band key is spread, universal-hashing
  style: 3·⌈log2 m⌉ rounds of CNOTs between disjoint random pairs of band wires (m = 2n band wires), a random linear
  map with positive controls only (dead on the zero band), never undone. A one-hot band key then sets about half of
  the band (after 2·⌈log2 m⌉ rounds, 1 % of one-hot keys still set only 13-15 %). Then 2·n·⌈log2 n⌉ three-control
  gates into B and n·⌈log2 n⌉ into A, keyed on band wires, each opened into two two-control gates through a band
  helper that is not restored (a prologue first makes every helper wire a key once). Coverage: for every one-hot band
  key, whatever B holds, every B bit must be liable to flip; a missed B wire gets one more gate keyed on a wire the
  spread key sets. Block 1 also writing A closes a corner: if it wrote only B, then for each nonzero band there would
  be one B input mapped to B' = 0, and block 2 would leave A untouched on it.
- **Block 2 randomizes A whenever B is nonzero:** on every B wire a partition triple, `t1 ^= b·u·v`, `t2 ^= b·u·¬v`,
  `t3 ^= b·¬u` on three different A wires: for a one-hot B key exactly one fires on every input, so one A bit flips
  whatever A holds (a first draft with two random gates per B wire flipped none on 56 % of inputs), and C's ripple
  changes every output bit. The two three-control gates are opened through a B helper that is restored (B carries the
  payload): 7 gates per B wire.

Every draw is checked, exhaustively up to 20 wires and by sampling above, and redrawn on failure. Details:
[PIPELINE_V15_STEPS.md](PIPELINE_V15_STEPS.md) §2 step 2. RC: no option builds the v14 guard (a security hole), so
the guard is not a setting; gen's banner (`opening_guard=`), the stage-1/2 marker and the manifest
(`preprocessing_opening_guard=`: `two-layer`, `junk-half` under the reverse contract, `none` in nonlinear291) record
what was built, and the driver refuses a gen that reports another guard. Gen's own forward payload check now samples
the honest slice (x on A, zeros elsewhere), since off it the circuit no longer computes the sandwich.

**History of the day.** The first version (pushed as `main` e41b5bb9, stage-1/2 marker 10) had n·⌈log2 n⌉ gates per
layer and no key spreading. A gate fires only when its own key bit is set, so a key with one nonzero bit armed only
the gates keyed on that wire, and no ripple could cross the dead gates: a one-hot B key changed 5-7 of A's 128 wires,
a one-hot band key 1-16 (RC: "is there a ripple effect?"). An intermediate version spread both keys (about 18,500
gates); RC's design above needs spreading in block 1 only. The stage-1/2 marker is now `tdp_stage12_recipe=11`.

**Evidence.** The same probe, 64 random x per row, one bit set, before and after:

| circuit | one bit of B set, band zero | one band bit set |
|---|---|---|
| v14 stage 2 (published `tdp128_s6_20261102`, all six stages) | wires 0..n-1 identical to the honest run; 2-3 payload wires differ | everything differs, about half of 512 bits |
| v15 stage 2, first version of the guard (fresh n = 128 stage 2, 1,145,611 gates) | A, B and band all differ: median 250-257 of 512 bits per sample (min 112-124) | the same |
| v15 stage 2, final guard and seeds (fresh n = 128 stage 2, 1,159,814 gates) | A, B and band all differ: median 255-257 of 512 bits per sample, min 224-237 | median 256-257, min 227-230 |

**Cost.** 9,380 guard gates at n = 128 instead of 2,560, 4,120 at n = 64 instead of 1,280. Measured on 256 random
inputs per key: a one-hot band key changes all B wires (and nearly all A wires); a one-hot B key flips one A bit on
every input.

## 2. The band seed and reseed

**What.** Stage 2 fills the band from the input after the guard (the seed, module 2) and refills it at the output
end (the reseed, module 4). v14's seed gave each band wire a private pivot CNOT from an input wire, 1-7 more CNOTs and
4 products of two literals: about 11 gates per band wire, 2,800 at n = 128. RC: the seed should implement F(A, B) =
(A, R(A, B)) with R pseudorandom-like, using only g57 gates, completely random except that the target is a band
wire, 8·n·log n of them.

**How.** `seed_band_g57` in `src/stages/preprocessing/embedded_masking.rs`: 8·n·⌈log2 n⌉ gates `t ^= a OR NOT b`, t a
band wire, a and b any two other wires (7,168 at n = 128, 3,072 at n = 64), for the seed and, independently drawn,
for the reseed. Two constraints kept from the old seed: no gate whose two controls are a pair of input wires that a
first-layer sandwich gate reads raw (`seed_avoid_c1`, which closed the first-layer "fringe" leak), and the seed is
redrawn until every band wire comes out non-constant on the honest slice behind the guard. Under the reverse contract
the old seed and reseed stay, so the validated chain still reproduces byte for byte. Tests: `two_layer_guard_is_the_identity_exactly_on_the_honest_slice` (every state at n = 3, 4),
`two_layer_guard_at_n64_restores_b_and_leaves_products_on_band_helpers`,
`no_reverse_contract_two_layer_guard_keys_on_the_register_and_the_band` (a whole stage 2: payload on the slice, every
single bit of B moves the payload off the sandwich's, the reverse payload junked).

## 3. Stage 6: the leak guard on the canonical order and on phase C (code review, 2026-11-02)

A review found that the canonical-order guard compared only deeper births, and against the last round's output rather
than the circuit being reordered (which by then has been through the cleanups and phase C): on the reviewer's probe
(n = 6, a 1-gate C) canonical ordering moved a C output's first exposure from cut 56 to cut 1 and was accepted with
`status=ok`. Phase C had no guard at all: on a second probe its lowering, with or without the fcompress cleanup,
exposed a C output early. Now both compare the same triple as the cleanups (validated deeper births, affine and
direct, and early C outputs) against the circuit they change: the canonical order is refused when it exposes more,
and phase C retries the lowering with random helper wires (up to 8) and, when none exposes no more, publishes the
least exposed one with `status=leakier_lowering`. On both probes the published circuit now has 0 early outputs
(phase C found a clean lowering with random helpers). Tests:
`canonical_order_and_phase_c_never_add_early_outputs` (the reviewer's probes, `tests/fixtures/adaptive_guard`),
`exposure_counts_early_outputs_as_leakage`, `seeded_helpers_stay_exact_and_vary`.

## Settings

- Stage 2: no new setting (the opening guard follows from `preprocessing.reverse_contract`); `tdp_stage12_recipe=10`
  with `preprocessing_opening_guard=`; gen's banner ends with `opening_guard=`.

- `[adaptive]`: new keys `pre_cleanup` (true), `cost` ("two-control"), `descent` ("128,32,16"),
  `descent_stop_fraction` (0.2), `cleanup_fcompress_iters` (10), `lower_controls` (2); new defaults
  `stop_fraction` 0.2 (was 0.02) and `max_block_gates` 512 (was 65536). Script flags `--adaptive-no-pre-cleanup`,
  `--adaptive-cost`, `--adaptive-descent`, `--adaptive-descent-stop-fraction`,
  `--adaptive-cleanup-fcompress-iters`, `--adaptive-lower-controls`. All are recorded in the manifest
  (`adaptive_*`).
- `adaptive.census_batches`: default `auto` (was 96), the census batch rule ⌈(W + 512) / 64⌉ for the W-wire
  stage-2 circuit, resolved by `tdp_adaptive --census-batches auto` (16 at n = 128, 12 at n = 64). An exact exposure
  fits every sample, so fewer samples never miss one; the 512-sample margin keeps chance fits near 2^-511, and every
  hit is re-tested on fresh and structured inputs. The census dominated stage 6 (2.8 of 4.0 h at n = 128 with 16
  batches); 96 batches would have made it 6 times longer. An explicit integer overrides the rule.
- `tdp_adaptive`'s own defaults are unchanged (recipe v14's stage 6), so the driver passes the v15 settings
  explicitly and the stage-6 marker lists them (`adaptive_params=--stop-fraction 0.2 --pre-cleanup ...`).
- The synthesizer effort is unchanged (16 restarts, 60 s budgets). `restarts = 2`, `time_limit = 12` (the
  bounded effort of the n = 128 run) is about 3.4x faster and about 6 % larger at n = 64.
- `[db_mixing]` `twist_rate`, `twist_min_span`, `twist_spans` as in v14 section 2.5.

To rebuild main's v14 stages 3 and 6 in a fresh run (its stage-2 guard cannot be rebuilt by any setting; use the
checkout of `main` at 2b4b6374 to reproduce a v14 circuit):

```toml
[db_mixing]               # circuit_mixer's own twist preset
twist_rate = 0.0005
twist_min_span = 64
twist_spans = "log"
[adaptive]                # tdp_adaptive's own defaults
pre_cleanup = false
cost = "gates"
max_block_gates = 65536
stop_fraction = 0.02
descent = "none"
lower_controls = 0
```

## Open

- The attack battery (factor-window rewrites, degree-2 replica scans) has not been run on v15 stage-6 outputs;
  the census (0 deeper, 0 early outputs) is the weak test. In the variants campaign fcompress-shortened circuits
  were more exposed to window rewrites than unshortened ones.
- Every circuit published before recipe v15 that kept no reverse contract (recipe v14 stage 2, including
  `circuits/tdp128_s6_20261102.mpmct1`) has the unguarded register B described in section 1.
- The key spreading is a fixed public bijection: a band key chosen as the preimage of a one-hot spread key arms only
  the few block-1 gates of one wire again. Finding it requires isolating the guard inside the published circuit, in
  which case the guard could be removed as well.
- Under the reverse contract (`preprocessing.reverse_contract = true`) both guards are the junk-half guards, keyed on
  the band only; there B is guarded only by stage 1's slice blocks (`sandwich.slicing = true`, as in v13 and the
  validated chain). `reverse_contract = true` with `slicing = false` leaves B unguarded.
