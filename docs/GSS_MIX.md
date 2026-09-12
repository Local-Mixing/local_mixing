# GSS mixing

Run `cargo run --release --locked -- gss` with the recipe in
[`configs/gss.toml`](../configs/gss.toml). Use `--config PATH` to select a local
recipe and `--dry-run` to inspect its resolution. The complete field reference
is in [`configs/README.md`](../configs/README.md). All CLI and shell flags, their
TOML equivalents and defaults are listed in [`GSS_FLAGS.md`](GSS_FLAGS.md).

## Pipeline and outputs

| Stage | Implementation | Output |
| --- | --- | --- |
| 1–2 | `gen_sandwich_gadget`: source, classic sliced sandwich, guards and selected preprocessing | `gss.mpmct1`, source C and sandwich sidecars |
| 3 | `fmix --gss --db-mixing`: regular/curated frozen DB replacement and optional leakage repair | `db_mixing.mpmct1` |
| 4 | `fmix --split --split-stop`: splitting to exhaustion | `split.mpmct1`, `split.state` |
| 5 | `fmix --resume`: crossing walk | `crossing.mpmct1`, `crossing.state` |
| 6 | `fcompress`: compression and canonical packing | `final.esop1` |

Quadratic masking (formerly Ran balanced / blinded V5) uses `mask_pair_wires=2`, `max_open_masks=3`, `min_open_masks=2`, balanced
masks and quadratic fire. The classic sandwich preserves the existing forward
and reverse zero-slice contracts. The full GSS construction retains opening
and closing guards, input-derived band seed/reseed, `extra_lgis=0` and
`encoded_io=false`. The standalone security gadgetizer has its own comparison
options and is not substituted for this construction.

Set `source.path` to mix a supplied G57 circuit and set `source.wires` to its
logical input width. The file must be nonempty and fit that width; its gate
count is inferred unless explicitly supplied, in which case it must match.
Without a source file, C and D use the established derived size convention.

db_mixing requires the regular frozen store; the curated store is an optional
cascade before it. `database.allow_no_database_for_tests=true` retains the existing
plumbing-test mode. Automatic block sizing and fixed piece counts remain
mutually exclusive, including an explicit fixed count of one.

## Run identity and continuation

Omit `run.directory` for a fresh run. Set it to the same directory and rerun
the same recipe to skip completed stages. `run.rerun_from_stage` deliberately
recomputes that stage and downstream stages. A missing gadget invalidates
cached downstream stages, so db_mixing store prerequisites are checked again.

New runs write a version-7 `gss_command.conf` containing normalized settings,
source/leakage-reference hashes, and script/executable fingerprints. `stage12.recipe`
records the gadget implementation and selected controls. An explicit `hold=27`
and the omitted default now describe the same recipe. Changing a source,
gadget recipe or binary causes a managed resume to fail before mixing
provenance. Lifecycle settings such as stopping early are not recipe changes.

Existing version-3 through version-6 managed runs remain readable, including
product-2223. Select their original configuration with `gss --config PATH` and
retain their original build directory/binaries. The wrapper checks the saved
manifest against the byte-preserved [`scripts/compat/gss_mix_v3.sh`](../scripts/compat/gss_mix_v3.sh)
[`scripts/compat/gss_mix_v4.sh`](../scripts/compat/gss_mix_v4.sh),
[`scripts/compat/gss_mix_v5.sh`](../scripts/compat/gss_mix_v5.sh), or
[`scripts/compat/gss_mix_v6.sh`](../scripts/compat/gss_mix_v6.sh), respectively,
and skips rebuilding those binaries. It does not relabel an older product
construction as quadratic masking. Product-2223, its `2223` alias, and standalone nonlinear193 are rejected
for new runs, in both the wrapper and current Bash driver. A copy of the pre-cleanup manual is preserved
in [`history/GSS_MIX_LEGACY.md`](history/GSS_MIX_LEGACY.md).

The mixer still loads version-1, early version-2 and current version-2 `.state`
files. For a direct continuation, the existing `fmix --resume CHECKPOINT`
interface remains available with the desired crossing/stop/output options;
see `fmix --help`. New GSS runs still checkpoint between stages 4 and 5.
A checkpoint preserves the circuit, original reference, provenance and walk
state. The existing writer saves fresh RNG seeds rather than a complete RNG
snapshot, so continuation is not a promise of byte-identical uninterrupted
replay.

Runs created directly by an old Bash driver may lack a wrapper manifest.
Their original driver and `fmix --resume` interfaces remain available. The
existing explicit adoption path retains its calibration-only restriction
because earlier recipe provenance is not recoverable from those artifacts.

## Lower-level and comparison tools

`scripts/gss_mix.sh --help` lists the direct driver options. Its default is now
quadratic masking. Both `quadratic-masking` and `nonlinear291` compile in ordinary
builds. Select them with `[preprocessing] mode = "..."`; old `[gadget]` and
Ran balanced spellings remain aliases. Explicit quadratic-mask controls are
rejected for nonlinear291. Historical product and standalone nonlinear193
generation remain in `legacy-tools` and compatibility drivers.
All historical mixing commands are under `legacy_mixing`; the supported
operator utility names are `circuit generate`, `circuit evaluate` and
`circuit compare`. Database generation is grouped under `db` when built with
`db-tools`.

Security measurements and nonlinear291 comparisons live in
[`security_tests/`](../security_tests/README.md). They are separate from the
correctness checks in [`tests/`](../tests/README.md).

New recipe-v7 runs name stage 3 outputs `db_mixing.mpmct1`,
`db_mixing.state`, and optionally `db_mixing.leakage_repair.txt`. Resuming
a v3/v4/v5 run keeps that driver's `phaseA`, `splitB` and `crossB` filenames.
New runs use `split` and `crossing` for the later stage filenames. The checkpoint
serialization is unchanged; `config_version` remains 1.
