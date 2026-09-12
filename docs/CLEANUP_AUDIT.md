# Cleanup and retention audit

This audit covers the approved source refactor on `drip-reshuffle`, including normal GSS, optional Cargo targets, private/integration tests, Python references, shell launchers and saved-run compatibility. It does not treat an excluded default-build branch as unused everywhere.

## Removed or replaced source

- The combined `engine/mix.rs` and `circuit/circuit.rs` files were replaced by focused modules; algorithms have one authoritative implementation.
- Runtime nonlinear291 and its guard dependencies were extracted from security support. The former nonlinear adapter file was removed; historical Rust paths forward to the same shared implementation.
- The private root configuration façade was removed. Parsing, defaults, validation and legacy recipes are owned by `gss/config/`.
- Former source paths for stage executables, storage, stage algorithms and security tools were removed after their Cargo/import/include consumers were updated.
- Unneeded private preprocessing forwarding files were collapsed into the compatibility module. No algorithm copies are retained in old locations.
- Removed the unread historical `ProdLedger::next_retire` field and an unused private integration-test helper; neither had readers/callers. Removed an unused test import and two unnecessary mutable bindings.

## Retained code and why

| Code | Consumer / reason |
| --- | --- |
| `preprocessing`, `postprocessing`, `db_mixing`, and old engine exports | Existing library callers and retained tools; forward to canonical owners |
| `security_tests/support/preprocessing/gadgets.rs` | Historical product/carrier/drip comparison modes and their tests, gated by `legacy-tools` |
| Shared nonlinear193 Rust branch | Historical mode/recipe callers, gated by `legacy-tools`; not a fresh managed GSS choice |
| Python nonlinear193 | Required by the nonlinear291 reference/decomposition implementation; also validates four historical templates |
| `scripts/compat/gss_mix_v3.sh` through `v6.sh` | Exact saved driver fingerprints, recipe serialization and continuation |
| Legacy checkpoint/control-order readers | v1/early-v2/current-v2 states and historical frozen stores |
| `security_tests/legacy` and historical campaign scripts | Explicit legacy executable and archived deployments, outside ordinary runtime builds |
| `db_gen/` | Rebuilding regular/curated/wide databases and synthetic integration-test fixtures; builders excluded from ordinary runtime builds |
| Unregistered benchmark scratch probes | Research references documented in benchmark ownership, not silently promoted to executable targets |
| Dated campaign configs and artifact paths | Archived or external consumers; source cleanup does not rewrite deployment/run artifacts |
| Old benchmark launchers | Small forwarding adapters for existing scripts; implementation lives under `benchmarks/` |
| `legacy_environment` adapters | Preserve historical variable parsing, lazy reads and process-wide defaults while explicit computation/storage APIs accept typed options |

The retained historical algorithms have known comparison/compatibility consumers. No further algorithm deletion was justified by this audit. Supporting tests and public aliases are deliberately preserved.

## Consumer and artifact checks

Cargo metadata resolves every registered target. Rust builds validate module/test includes; the template exporter verifies all eight templates at their authoritative paths. Current Markdown links are checked after inventory regeneration. Checked-in campaign/deployment consumers retain executable names, configuration/artifact locations and compatibility launchers. No remote deployment was modified by this refactor.

Before editing, 632 source-area files and their hashes were archived under `/tmp/gss-refactor-20260910`. The new v6 driver matches the old active driver byte for byte, and v3–v5 snapshots are unchanged. The Git index is unchanged by this refactor. Existing database directories, generated circuits, run outputs, reports and archives were outside the edited source set.
