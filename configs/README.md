# GSS configuration

`gss` reads `configs/gss.toml` by default. Use `--config PATH` for another recipe
and `--dry-run` to validate it without creating a run or building binaries.
See [`docs/GSS_FLAGS.md`](../docs/GSS_FLAGS.md) for every flag and its TOML mapping.
Copy the template to `configs/local.toml` for machine-specific paths; this local
filename is ignored by Git. Relative paths resolve against the repository root.

The TOML parser rejects unknown sections/keys, duplicate keys and wrong value
types. Omit optional fields instead of assigning empty strings.

| Section | Fields and defaults |
| --- | --- |
| `run` | `directory` (fresh timestamped run if omitted), `build_binaries=true`, `build_directory` (environment or `target`), `stop_after_stage=6`, optional `rerun_from_stage`, optional `adopt_unverified_run=false` |
| `source` | `wires=128`, optional `gates`, optional G57 `path`; supplied sources determine the gate count, otherwise it is derived from the wire count |
| `database` | `regular_dir`, optional `curated_dir`, `curated_control_order="native"`, `lookup_miss_filter="auto"` (`auto/on/off`), `allow_no_database_for_tests=false` |
| `preprocessing` | `mode="quadratic-masking"` (aliases `ran-balanced`, `blinded-v5`, `blinded_v5`), `mask_pair_wires=2`, `max_open_masks=3`, `min_open_masks=2`, `balanced_masks=true`; second supported mode `nonlinear291` |
| `db_mixing` | `target_size_factor=2`, `hold_work_units=27`; the resulting profile is `3,30,30,2,2` |
| `parallel` | `pieces=1` (serial, range 1..64) or optional `target_piece_gates` (2..1000000000), mutually exclusive; optional `threads` (1..1024, default P+1 for fixed pieces or available CPU count for automatic sizing) |
| `leakage_repair` | Optional leakage audit and equivalent circuit repair after db_mixing: `enabled=false`, `seed=20803`, optional matching-coordinate `reference`; seed/reference require enabled leakage repair |
| `crossing` | `target_size_factor=2`, `width_penalty_base=3`, `width_penalty_threshold=1`, `size_tolerance_divisor=25`, optional `move_attempts` (default six times the target) |
| `calibration` | `enabled=false`, optional private `seed_file`; an explicit seed requires calibration mode |

The database compatibility, miss-filter and no-database testing switches are
explained in detail in [`docs/GSS_FLAGS.md`](../docs/GSS_FLAGS.md#database-settings-explained).
`leakage_repair` samples internal leakage signals and attempts equivalent
replacements; its reference defaults to the db_mixing input, not the source C.

Normal Quadratic masking uses the classic sandwich, balanced masks, quadratic fire,
band helper ancillas, `extra_lgis=0` and `encoded_io=false`. Refresh bursts
retain the historical wire pool (`burst_band_only=false`). These choices are
recorded in the run recipe. For new runs, `mask_pair_wires` is 2..64, `max_open_masks` is 2..64,
and `min_open_masks` is 1..63 and smaller than `max_open_masks`. The band has
`2 * source.wires` wires; this must exceed `mask_pair_wires` rounded down to an even number
plus the balancing wire. Explicit `parallel.threads` requires either
`parallel.pieces > 1` or `parallel.target_piece_gates`.

Database environment variables remain supported when the corresponding path
is omitted: `FROZEN_DB_DIR`, `FROZEN_CURATED_DIR`. `CARGO_TARGET_DIR` similarly
supplies the build directory. Ambient experimental gadget/QC overrides are
cleared before the runner applies the recipe's settings.

Use `[preprocessing]` for stage 2. `[gadget]` and its older field names remain
accepted aliases; specifying the same setting through both names is an error.
Both quadratic masking and nonlinear291 are included in ordinary builds. For
nonlinear291, omit all four mask controls and set only `mode = "nonlinear291"`.
Explicit mask controls are rejected for this mode, even when equal to defaults.

Existing marked Markdown recipes remain readable through `--config old.md`.
They retain their old defaults. Product-2223/2223 and standalone nonlinear193
can continue recorded runs but cannot start fresh managed GSS runs.
New runs write recipe v7; saved v3/v4/v5/v6 manifests use their original
drivers and exact recorded binaries, without rewriting manifests or rebuilding
those binaries. Preprocessing naming does not change checkpoint formats.
