# Pipeline changes of 2026-11-02 (recipe v15)

For everyone who knows recipe v14 (`main` at 2b4b6374). Recipe v15 changes two stages; stages 1, 2, 4 and 5 are
v14's. The pipeline step by step: [PIPELINE_V15_STEPS.md](PIPELINE_V15_STEPS.md); the settings:
[tdp_pipeline.md](tdp_pipeline.md).

| Area | Recipe v14 (main 2b4b6374) | Recipe v15 | Why (evidence) |
|---|---|---|---|
| Stage-3 twists | rate 0.0005 per mixer round, spans log-uniform from 64 gates (circuit_mixer's preset) | rate 0.001, spans uniform from 100,000 gates to the circuit size, every window inside the circuit | compensates the carrier hops the interleaved stage 2 gave up; no twist fits inside one stage-6 block ([PIPELINE_V14_CHANGES.md](PIPELINE_V14_CHANGES.md) section 2.5) |
| Stage 6 | large-block rounds only (blocks of up to 65,536 gates), judged by the raw gate count, stop fraction 0.02, wide gates kept | pre-cleanup (NOT-push + fcompress), blocks of up to 512 gates judged by the 2-control cost, stop fraction 0.2, descent 128 / 32 / 16 with a cleanup after each, every gate lowered to at most 2 controls; the leak guard on every acceptance and cleanup, now including early C outputs | n = 64: 2-control cost 0.072 of stage 5 (v14: 0.331 after ~16 h, unfinished, 3 early outputs); n = 128: 0.106 in 4.0 h on 12 threads with the bounded effort (v13's stage 6: 0.250 in 4.6 days on 48 threads, 8 early outputs) ([stage6_config_search_20261102.md](notes/stage6_config_search_20261102.md)) |
| Recipe bookkeeping | `tdp_command_recipe=14` | `tdp_command_recipe=15`; config schema still 5; a v14 run directory is refused on resume, with a message naming the settings that rebuild main's v14 | |

## Settings

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

To rebuild main's v14 in a fresh run:

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
- The stage-2 opening guard is keyed on the band only: with the band zero it is the identity for every value of
  the register B (wires n..2n-1), so an input (x, B, 0, 0) evaluates the sandwich off its slice (2026-11-02 probe
  of `circuits/tdp128_s6_20261102.mpmct1`: one bit of B set changes 2-3 payload output wires and none of wires
  0..n-1). Under discussion with RC.
