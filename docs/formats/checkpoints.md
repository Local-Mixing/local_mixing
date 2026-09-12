# Saved runs and checkpoints

Managed recipe v7 uses `quadratic-masking` / `nonlinear291`, the `[preprocessing]` section and descriptive flags. Old preprocessing names remain input aliases; fresh `product-2223` and `nonlinear193` modes are rejected before creating run artifacts.

Managed v3, v4, v5 and v6 runs retain their original manifest serialization, driver fingerprints and recorded binaries. The runner dispatches each to `scripts/compat/gss_mix_vN.sh`; those snapshots must not be edited. A continuation does not rebuild over recorded binaries or rewrite its manifest.

Use the managed `gss` command to select the appropriate snapshot automatically. For a direct historical-script invocation, explicitly set `GSS_BIN_DIR` to the original binary directory: relocating an immutable script changes how its old relative fallback would resolve. For example, `GSS_BIN_DIR=/absolute/path/to/original/release bash scripts/compat/gss_mix_v6.sh ...`. Keep that run's original flags and artifacts.

The mixer still reads `.state` v1, early v2 and current v2, preserving serialized field ordering and checkpoint/provenance semantics. Recipe v7 changes operator naming, not checkpoint encoding or the stage boundaries. Stages remain 2 (preprocessing), 3 (database mixing), 4 (splitting), 5 (crossing), 6 (compression).

Compatibility imports delegate to one implementation. Old Rust paths under `engine::mix`, `engine::format`, `engine::xpoly`, `db_mixing`, `postprocessing`, and `preprocessing` remain available where existing callers need them. Historical product and nonlinear193 implementations remain gated by `legacy-tools`; both supported preprocessors compile in ordinary GSS builds.
