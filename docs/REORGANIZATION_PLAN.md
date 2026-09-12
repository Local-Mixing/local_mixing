# GSS refactoring record

Status: completed and verified, 2026-09-10.

The user approved the full plan, including ordinary-build nonlinear291 and `src/entrypoints/`. The [original detailed plan](history/REORGANIZATION_PLAN_20260910.md) is retained as design history; [CODE_LAYOUT.md](CODE_LAYOUT.md) describes the current source.

Implemented ownership follows `sandwich → preprocessing → db_mixing → post-processing`, with the literal post-processing folder containing splitting, crossing and compression. The mixer is split by state, parameters, checkpoints, provenance, scheduling, reporting and transformations. Circuit formats, polynomial canonicalization and immutable database storage have dedicated owners. Reusable executable setup lives in `programs/`; three small entrypoints retain the executable names.

Quadratic masking replaces Ran balanced as the canonical name. Both it and nonlinear291 are normal-build preprocessors. Nonlinear runtime guards and the four nonlinear291 templates have a single owner; Python references and historical 193 comparison code remain outside the runtime. Typed construction preserves the effective managed defaults and raw-constructor behavior.

New recipe v7 uses `[preprocessing]` and `--preprocessing-*`, retains older aliases, rejects conflicting settings and rejects explicit mask controls for nonlinear291. Immutable v3–v6 drivers, versioned manifest serialization, recorded binaries and checkpoint encodings preserve continuation. Read the [compatibility contract](formats/checkpoints.md).

Canonicalization, store opening, lookup caching and replacement selection expose explicit options. Historical environment controls live in compatibility adapters that retain their independent first-use reads and defaults. Explicit canonicalization calls bypass the historical process-wide caches so one call's limits cannot reuse another call's result. No checkpoint fields or algorithm defaults changed.

Tests, optional tools and documentation are organized by purpose. The [cleanup audit](CLEANUP_AUDIT.md) records why remaining compatibility and research code stays. Existing generated artifacts, archives and staged changes are preserved. No commit or push is part of this refactor.

## Validation

- Integrated ordinary-build run: 302 library tests passed, one existing golden-regeneration test ignored, five circuit CLI tests passed, and both real frozen-store integration tests passed. These cover LMDB conversion/lookup round trips and actual database replacement with equivalence checks.
- After the final explicit-option extraction: 42 canonicalization tests passed (one existing ignored), and all 36 focused storage/replacement/mixer regressions passed. These include recursive budgets, per-call and thread isolation, lazy environment reads, cache isolation, control conventions, replacement RNG behavior and the mixer goldens. The native291 stage-contract regression also passed.
- Configuration and compatibility: 37 ordinary and 38 legacy GSS tests passed, both Bash flag/block-size suites passed, and all 11 legacy-command tests passed. Historical drivers v3–v5 are unchanged; v6 exactly matches the previous active driver.
- Preprocessing evidence: three quadratic construction artifacts match the captured baseline; nine nonlinear source/seed cases preserve every gate, physical width and RNG continuation. All eight exported templates validate, and all ten Python reference tests passed.
- Matched-seed gauntlet: six before/after cases cover the plain control, balanced quadratic masking and nonlinear291, each with zero or 16 mixing moves. Every generated circuit, sampled trace, target, metadata file, witness file and audit output matches byte for byte. The comparison uses a one-gate, three-wire r57 source, sample seed 19, gadget seed 7, mixing seed 11, 64 correlation samples, weight-2/3 caps 8/4 and an exact-linear feature cap of 4096. This is bounded comparative regression evidence.
- The final combined database/security/challenge/benchmark target check passed. The Python wheel built and all eight heatmap exports imported. Clippy passed with remaining style warnings; formatting, diff whitespace, current Markdown links, Cargo target paths and default runtime dependency exclusions passed. The Git index remains unchanged.

Validation logs, the original source archive and the matched-seed comparison are retained locally under `/tmp/gss-refactor-20260910`; source ownership and retained compatibility code are documented in [CODE_LAYOUT.md](CODE_LAYOUT.md) and [CLEANUP_AUDIT.md](CLEANUP_AUDIT.md).
