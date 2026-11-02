# Pipeline changes of 2026-11-02 (recipe v15)

For everyone who knows recipe v14 (`main` at 2b4b6374). Recipe v15 changes three stages: stage 2's opening guard,
stage 3's twists and stage 6; stages 1, 4 and 5 are v14's. The pipeline step by step, with every default:
[PIPELINE_V15_STEPS.md](PIPELINE_V15_STEPS.md); the settings: [tdp_pipeline.md](tdp_pipeline.md).

| Area | Recipe v14 (main 2b4b6374) | Recipe v15 | Why (evidence) |
|---|---|---|---|
| Stage-2 opening guard (no reverse contract, the default) | keyed on the band only, over all data wires, 20n gates: the identity whenever the band is zero, whatever the register B holds | the **two-layer guard** keyed on B and the band: the identity on the data exactly on the honest slice; n·⌈log2 n⌉ three-control gates per layer opened into two-control gates, 4,512 gates at n = 128; not configurable | with the band zero and B nonzero the v14 circuit evaluated the sandwich off its slice: in `circuits/tdp128_s6_20261102.mpmct1` one bit of B changed 2-3 payload bits and none of wires 0..n-1 (section 1) |
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

**How.** `two_layer_opening_guard_dims` in `src/stages/preprocessing/slice_guards.rs`, built as RC specified:
first gates with targets in A and B, positive controls in the band and controls of either polarity in A and B; then
gates with targets in A, positive controls in B and controls of either polarity in A. Each layer holds n·⌈log2 n⌉
three-control gates (at least twice its key wires), each opened into two-control gates through a helper wire: a band
helper in layer 1, not restored (the band is junk; a prologue first makes every helper wire a key once, into A and
into B), a B helper in layer 2, restored (B carries the payload). Every draw is checked, exhaustively up to 20 wires
and by sampling above, and redrawn on failure. Details: [PIPELINE_V15_STEPS.md](PIPELINE_V15_STEPS.md) §2 step 2.
RC: no option builds the v14 guard (a security hole), so the guard is not a setting; gen's banner (`opening_guard=`),
the stage-1/2 marker and the manifest (`preprocessing_opening_guard=`: `two-layer`, `junk-half` under the reverse
contract, `none` in nonlinear291) record what was built, and the driver refuses a gen that reports another guard.
Gen's own forward payload check now samples the honest slice (x on A, zeros elsewhere), since off it the circuit
no longer computes the sandwich.

**Evidence.** The same probe, 64 random x per row, one bit set, before and after:

| circuit | one bit of B set, band zero | one band bit set |
|---|---|---|
| v14 stage 2 (published `tdp128_s6_20261102`, all six stages) | wires 0..n-1 identical to the honest run; 2-3 payload wires differ | everything differs, about half of 512 bits |
| v15 stage 2 (fresh n = 128 stage 2, 1,145,611 gates) | A, B and band all differ: median 250-257 of 512 bits per sample | the same |

**Cost.** 4,512 guard gates at n = 128 instead of 2,560 (+1,952 gates of a 1.14M-gate stage 2), 1,936 at n = 64
instead of 1,280. Tests: `two_layer_guard_is_the_identity_exactly_on_the_honest_slice` (every state at n = 3, 4),
`two_layer_guard_at_n64_restores_b_and_leaves_products_on_band_helpers`,
`no_reverse_contract_two_layer_guard_keys_on_the_register_and_the_band` (a whole stage 2: payload on the slice, every
single bit of B moves the payload off the sandwich's, the reverse payload junked).

## Settings

- Stage 2: no new setting (the opening guard follows from `preprocessing.reverse_contract`); `tdp_stage12_recipe=10`
  with `preprocessing_opening_guard=`; gen's banner ends with `opening_guard=`.

- `[adaptive]`: new keys `pre_cleanup` (true), `cost` ("two-control"), `descent` ("128,32,16"),
  `descent_stop_fraction` (0.2), `cleanup_fcompress_iters` (10), `lower_controls` (2); new defaults
  `stop_fraction` 0.2 (was 0.02) and `max_block_gates` 512 (was 65536). Script flags `--adaptive-no-pre-cleanup`,
  `--adaptive-cost`, `--adaptive-descent`, `--adaptive-descent-stop-fraction`,
  `--adaptive-cleanup-fcompress-iters`, `--adaptive-lower-controls`. All are recorded in the manifest
  (`adaptive_*`).
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
- A key with one nonzero bit fires only the three-control gates it controls: about 3.7 per band key wire and 7 per
  register wire at n = 128 (of 896 per layer), so f_k and h_k of a low-weight key are short circuits.
- Under the reverse contract (`preprocessing.reverse_contract = true`) both guards are the junk-half guards, keyed on
  the band only; there B is guarded only by stage 1's slice blocks (`sandwich.slicing = true`, as in v13 and the
  validated chain). `reverse_contract = true` with `slicing = false` leaves B unguarded.
