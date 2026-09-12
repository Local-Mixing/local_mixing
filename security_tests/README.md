# Security analysis and historical comparisons

Security source is consolidated here. Existing circuits, solver installations,
private answers, logs, and figures stay in their original artifact directories.
The four standalone programs in root `challenges/` retain their existing paths.

| Directory | Purpose |
| --- | --- |
| `demixing/`, `oracle/`, `fixtures/`, `preprocessing/` | Rust attack tools, circuit fixtures, and the standalone quadratic-mask adapter; stable Cargo binary names |
| `campaigns/` | Historical SSS campaigns, heatmap recomputation, and monitors |
| `orchestration/` | Challenge construction, verification, isolated attack supervisors |
| `heatmaps/` | Rust heatmap measurements and Python plotters, including both frontends |
| `gauntlet/` | Generation, tracing, attacks, provenance, and reporting |
| `gadgetization/` | Python nonlinear291 and nonlinear193 references, exporter, and four historical nonlinear193 templates; the four nonlinear291 runtime templates live in `src/stages/preprocessing/templates/` |
| `attacks/`, `leakage/` | SAT/search tools and trace/recovery analysis |
| `reporting/` | Reusable plotters and dated report recipes |
| `support/` | Optional Rust comparison implementations and analysis helpers |
| `python/` | Optional PyO3 implementation and registration |

Run Python modules from the repository root, or invoke the entry point by its
absolute filename. The gauntlet and template exporter support both forms:

```sh
python3 -m security_tests.gadgetization.export_templates --check
python3 -m security_tests.gauntlet.gauntlet --help
python3 -m security_tests.orchestration.verify_offslice --help
```

Python comparisons require NumPy. Plotting additionally uses Matplotlib, Pandas,
and Pillow as appropriate. Heatmap frontends import the optional Rust extension
as `local_mixing`. Its eight exports remain `heatmap`, `heatmap_subsampled`,
`heatmap_incremental`, `heatmap_small`, `heatmap_slice`, `heatmap_mini_slice`,
`heatmap_corner`, and `heatmap_corner_at`; their signatures are unchanged.
`heatmaps/heatmap.py` retains the enhanced interface, and
`heatmaps/heatmap_legacy.py` retains the distinct older interface.

Build security/comparison binaries with `--features security-tools`, the four
root challenge binaries with `--features challenge-tools`, and the PyO3
extension with `--features python-extension` (also selected by maturin).
These commands are optional; the production default build excludes them.

Historical SSS and genran campaigns use the optional `legacy_mixing` binary with
the `legacy-tools` feature. `campaigns/legacy_runner.sh` passes the original CLI
arguments unchanged and accepts `SECURITY_LEGACY_BIN=/absolute/path/to/archived/runner`.
The queue accepts `BIN` or `SECURITY_LEGACY_BIN`; random-source studies retain
their explicit `--genran` executable option. The parent integration supplies the
Cargo target and legacy aliases. Deploy the relocated `security_tests` package
and launcher alongside any campaign run on a remote repository.

The C1/C2 Python supervisors accept `--artifact-root`, `--run-dir`, `--tools-dir`,
and `--decoder`. Defaults retain the old checkout locations:
`red_team_tests/` for C1 and `red_team_tests/_orchestration/` for C2. Existing
deployed runs should pass their actual run and compiled-tool directories.
The decoder defaults to its relocated source sibling. Deploy `_paths.py` with
the supervisors, or deploy the complete package. Index rebuilding accepts
`--artifact-root` and still defaults to `red_team_tests/`, including private
answers; source consolidation never rewrites those records.

`SECURITY_REPO_ROOT` and `SECURITY_ARTIFACT_ROOT` can override these orchestration
defaults. The C1 affine shell runner accepts run/tools directories as its first
two arguments, or `SECURITY_RUN_DIR` and `SECURITY_TOOLS_DIR`.

Dated split-trial recipes default to `reports/split_trials_20260805/`, regardless
of source location; `SECURITY_REPORT_DIR` selects an existing report directory.
`snap_frac3.py` and `xpanel_spread.py` also accept `--report-dir`.
Reusable report plotters retain explicit input stems and `--out` filenames.
`heatmaps/prep_heat.sh [artifact-directory]` uses the specified directory, then
`SECURITY_ARTIFACT_DIR`, then the invocation directory, preserving its original
`start.txt`, `recent_circuit.txt`, `circuits1.txt`, and `circuits2.txt` filenames.

Pinned historical recipes such as `prepare_c1_remote.sh` and `run_one_server.sh`
retain original hashes, external tool trees, timing, and store conventions.
They require their archived deployment and do not certify a newly rebuilt
runner. The external `mixing_tests/sat_testing/solve_feistal_preimage.py` solver
wrapper is not part of this snapshot. C/C++ Kissat probes require the matching
external Kissat headers; compiled solver/output locations are unchanged.

Build Rust security programs with `cargo build --release --features security-tools`.
Build the Python extension with `python -m pip install .`. Correctness tests for
the retained nonlinear templates live under `tests/gadgetization/`.
