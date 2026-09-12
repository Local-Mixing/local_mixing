# GSS refactoring plan

Status: planning, revised 2026-09-10. Runtime source has not been reorganized.
The [complete current map](CODE_LAYOUT.md) remains the inventory of what is on
disk. This document describes the proposed destination and the work needed to
get there without changing circuit behavior or breaking retained runs.

## Agreed structure and working choices

The main source layout follows the GSS stages, with reusable components beside
them. Keep one Cargo package for this refactor. Keep the established root
folders: `src/`, `db_gen/`, `tests/`, `security_tests/`, `challenges/`,
`benchmarks/`, `configs/`, `scripts/`, `docs/`, and `.github/`.

The requested stage grouping is:

```text
sandwich → preprocessing → db_mixing → post-processing
                                            ├── splitting
                                            ├── crossing
                                            └── compression
```

`preprocessing/` owns the two supported gadgetization implementations:
**quadratic masking**, the proposed name for Ran balanced, and **nonlinear291**.
The word preprocessing names stage 2; sandwich construction remains stage 1.
`post-processing/` groups stages 4, 5 and 6 without combining their algorithms,
stage numbers, checkpoints or configuration settings.

Working choices pending the user's answers:

| Choice | Recommendation used in this draft | Alternative |
| --- | --- | --- |
| Availability of nonlinear291 | Include both supported preprocessing modes in ordinary GSS builds after extracting their legacy dependencies | Give nonlinear291 a dedicated optional feature |
| Entrypoint folder | Use `src/entrypoints/` for the three stage programs | Keep the conventional `src/bin/` name |

These are planning defaults, not recorded approvals. Neither choice changes
which algorithms belong in `preprocessing/` or requires renaming executables.

Preserve the separate `generate.rs`, `evaluate.rs` and `compare.rs` command
files. Preserve databases, circuits, run outputs, checkpoints, report artifacts
and archives. Keep correctness tests in root `tests/` and all security
measurements, heatmaps, affine heatmaps, attacks and gauntlets in
`security_tests/`. Runtime leakage repair remains part of database mixing.

## Naming the default preprocessing algorithm

Use **quadratic masking** as the algorithm name, `quadratic-masking` as its
canonical mode value, and `quadratic_masking.rs` as its source filename.

The implementation masks data values using quadratic terms from auxiliary
band wires, executes gates through the masked controls, and manages opening,
closing and refreshing masks around the computation. Its current GSS preset
uses balanced masks: an extra band wire is XORed into each mask. This names
what the code does and avoids an author's shorthand or implementation version.
It is a descriptive name, not a security guarantee. The balancing property
assumes the relevant distribution of band values; the name does not certify
that distribution or the complete pipeline.

Keep balanced masks as the managed GSS default. A research run with balancing
disabled still uses the quadratic masking family, which is why the family
name should not hard-code the balanced setting. The old linear-read branch
remains an explicitly historical comparison option, outside the two supported
new-run choices; extracting it must preserve its existing tests and callers.

| Surface | Proposed canonical form | Compatibility requirement |
| --- | --- | --- |
| Display name | Quadratic masking | Explain once that this was Ran balanced / blinded V5 |
| Mode string | `quadratic-masking` | Accept `ran-balanced`, `blinded-v5`, `blinded_v5` as aliases |
| Runtime implementation | `preprocessing/quadratic_masking.rs` | Keep old Rust exports as forwarding aliases while retained callers migrate |
| Types and construction | `QuadraticMaskingParams`, `QuadraticMaskingOutput`, `preprocess_quadratic_masking` | Old public names can forward; preserve raw-constructor behavior |
| Second supported mode | `nonlinear291` | Preserve existing spelling |
| Current mode enum | `PreprocessingMode` with two supported variants | Parse historical modes through explicit compatibility handling |
| Configuration section | Recommend `[preprocessing]` | Continue accepting `[gadget]`; reject conflicting or duplicate aliases |
| Direct driver flags | Recommend `--preprocessing-mode`, `--preprocessing-mask-pair-wires`, `--preprocessing-max-open-masks`, `--preprocessing-min-open-masks`, `--preprocessing-balanced-masks` | Keep the corresponding current `--gadget-*` flags and older aliases |

The configuration/flag spelling changes are a separate implementation slice,
after source ownership is stable. The public `gss` command still accepts its
configuration file; it does not need a new command for every stage. Keep all
other recently clarified names, including `db_mixing` and `leakage_repair`.

Example of the proposed configuration spelling, not yet accepted by current
code:

```toml
[preprocessing]
mode = "quadratic-masking"
mask_pair_wires = 2
max_open_masks = 3
min_open_masks = 2
```

For nonlinear291, the equivalent section needs only `mode = "nonlinear291"`.
Explicit quadratic-mask controls should produce a clear validation error for
that mode. Omitted controls should not become user-supplied overrides merely
because the parser fills defaults.

## Proposed code layout

This sketch shows ownership, not a mandate to create many tiny files. Keep
small cohesive implementations together. Split large implementations only at
real boundaries. Module registration files are omitted where they add noise.

```text
src/
├── main.rs                         # Main CLI dispatch
├── lib.rs                          # Deliberate library exports
├── commands/
│   ├── gss.rs                      # Thin GSS command adapter
│   └── circuit/
│       ├── generate.rs
│       ├── evaluate.rs
│       ├── compare.rs
│       └── shared.rs               # CLI formatting/validation only
├── gss/
│   ├── cli.rs
│   ├── config/                     # Types, parsing, defaults and validation
│   │   ├── parse.rs
│   │   ├── validate.rs
│   │   └── legacy_recipe.rs
│   ├── runner.rs                   # Build/run management
│   ├── manifest.rs                 # Versioned fingerprints and resume contract
│   └── paths.rs
├── stages/
│   ├── mod.rs                      # Stage exports and post-processing path mapping
│   ├── sandwich/
│   │   └── construct.rs            # Source preparation and sliced sandwich
│   ├── preprocessing/
│   │   ├── mod.rs                  # Two supported modes and their dispatch
│   │   ├── quadratic_masking.rs    # Current Ran balanced algorithm
│   │   ├── nonlinear291.rs         # Native GSS adapter, layout and capacity checks
│   │   ├── slice_guards.rs         # Guards shared by supported preprocessors
│   │   ├── types.rs                # Preprocessing options/results and contracts
│   │   └── templates/
│   │       ├── nonlinear291_r57.mpmct1
│   │       ├── nonlinear291_nab.mpmct1
│   │       ├── nonlinear291_and.mpmct1
│   │       └── nonlinear291_copy.mpmct1
│   ├── db_mixing/
│   │   ├── replacement.rs
│   │   └── leakage_repair/         # Runtime detection/repair and engine adapter
│   └── post-processing/
│       ├── mod.rs
│       ├── splitting.rs
│       ├── crossing.rs
│       └── compression/
│           ├── mod.rs
│           ├── reduce.rs
│           ├── transport.rs
│           ├── packing.rs
│           └── downhill.rs
├── circuit/
│   ├── g57.rs                      # G57 representation
│   ├── xgate.rs                    # General mixed-polarity gates
│   ├── evaluate.rs                 # Evaluation kernels
│   ├── formats.rs                  # G57/mpmct1/esop1/anf1 I/O
│   ├── operations.rs               # Reusable loading/evaluation/comparison
│   └── randomize.rs
├── canonicalization/
│   ├── polynomial.rs
│   ├── canonicalize.rs
│   ├── keys.rs
│   └── cache.rs
├── database/
│   ├── frozen.rs                   # Immutable store reading
│   ├── codec.rs                    # Shared storage-format primitives
│   ├── lookup_cache.rs
│   └── validation.rs
├── engine/
│   ├── arena.rs
│   ├── mixer/
│   │   ├── mod.rs
│   │   ├── state.rs
│   │   ├── params.rs
│   │   ├── checkpoint.rs
│   │   ├── scheduling.rs
│   │   ├── provenance.rs
│   │   └── reporting.rs
│   ├── moves/                      # Shared primitive transformations
│   └── stats.rs
└── entrypoints/                    # Working choice; bin/ also works
    ├── gen_sandwich_gadget.rs
    ├── fmix.rs
    └── fcompress.rs
```

Keep the literal folder spelling `post-processing`. Rust identifiers cannot
contain that hyphen, so `src/stages/mod.rs` will declare:

```rust
#[path = "post-processing/mod.rs"]
pub mod post_processing;
```

Imports will use `stages::post_processing`. This single visible path mapping
honors the requested folder name. Avoid scattering path redirects through
individual algorithm files. A small standalone Rust 2024 compile/run verified
this layout during planning.

`entrypoints/` contains executable source, not compiled binaries. A file there
defines a `main()` that parses arguments, calls library code and reports the
result. The current GSS command launches these programs indirectly through
its driver. Cargo already has `autobins = false`, so its explicit target paths
can point to `entrypoints/` without changing any executable name:

| Entrypoint | Executable name retained | Responsibility |
| --- | --- | --- |
| `src/main.rs` | `local_mixing_bin` | `gss`, grouped `circuit`, optional grouped `db` |
| `src/entrypoints/gen_sandwich_gadget.rs` | `gen_sandwich_gadget` | Sandwich and preprocessing |
| `src/entrypoints/fmix.rs` | `fmix` | Database mixing, splitting, crossing and resume |
| `src/entrypoints/fcompress.rs` | `fcompress` | Final compression/packing |

The compiled executables remain in the selected Cargo target directory.
Optional tool programs stay in their owning root folders. They need no new
`bin/` directory merely to be executable.

## Dependency findings that shape the migration

### Nonlinear291 is entangled with historical code

The native adapter currently lives in
`security_tests/support/preprocessing/nonlinear_gss.rs`. It imports
`CnotCircuit`, `SLICE_ZERO_CCNOT_GATES_PER_WIRE` and
`try_nonlinear_slice_zero_preblock_dims` through the old `gadgets` facade.
The latter two are in the very large historical `gadgets.rs` implementation.
Its required guard constructor and indexed slice checker can be extracted
without importing the unrelated product, carrier and drip algorithms.

The extraction must include the constructor's random-wire helper, exact
small-slice checker and indexed wide-slice checker. Preserve the order and
number of random draws, checked capacity arithmetic, fan-in-two decomposition,
restored scratch wires and output layout. Reuse the existing guard/helper
implementations instead of creating a second subtly different version.

Move the four nonlinear291 templates beside the native implementation. Keep
one authoritative copy of each template. The Python exporter must know the
new destination and `--check` must still regenerate identical content.

The Python nonlinear291 reference imports nonlinear193 to construct and
decompose the reference gadgets. Retain both Python files and their reference
checks outside `src/`, with the comparison/template tooling. They are not
invoked by ordinary Rust GSS runs. Nonlinear193's standalone historical Rust
mode and four templates remain available to retained legacy workflows, but
will not be a third supported choice for new managed GSS runs. Extract any
shared encoding internals deliberately; do not duplicate the entire adapter.

This puts both supported runtime gadgetizations and their required runtime
assets in preprocessing, while keeping comparison/test tooling in its agreed
root folder. Old `security_tests.gadgetization` imports need adapters if that
Python package is later reorganized.

### Defaults currently have more than one owner

`BlindedV5Params::production()` is not the complete managed GSS recipe. For
example, its band-only refresh field defaults to false, while the managed
runner pins the band-only environment setting to true. The generator also
sets active wires, builds guards, seeds/reseeds the band and applies overrides.

Create one explicit managed-GSS preset that reproduces today's effective
recipe. Keep raw construction options for tests/research distinct. Do not
silently change every library caller to the managed preset. Capture existing
outputs before extraction, and verify seeded equivalence after extraction.
Keep legacy linear reads, encoded-I/O experiments and balancing overrides
available to their retained callers; they are not extra new-run modes.

### Post-processing already shares mutable mixer state

Splitting and crossing currently contain `impl Mixer` blocks and call shared
arena, RNG, provenance and undo operations. Splitting also performs crossing
shots. Keep those relationships explicit inside the new common folder.
Compression and packing use different structures; a common parent folder
does not mean forcing all three stages to accept one oversized state object.

First divide `engine/mix.rs` internally, keeping related private methods in
child modules. Then extract narrow stage-facing operations. Do not make all
mixer fields public, clone its state, add one trait per stage or introduce a
plugin registry just to make file moves compile. Two preprocessing choices
can use a small enum and ordinary function dispatch.

### Representations, storage and builders need separate owners

`src/circuit/operations.rs` currently imports formats from `engine`. Move
circuit I/O beside circuit representations. Split the polynomial/key logic
from the 2,788-line `circuit/circuit.rs` and `engine/xpoly.rs` while preserving
canonical keys, wire ordering, bounded-search failures and cache semantics.

`db_mixing/` currently owns both immutable storage and replacement policy.
Move storage under `database/`; keep replacement and leakage repair with the
stage. `src/lib.rs` unconditionally includes `db_gen/mod.rs`, so some builder
code is part of the default library even when no builder runs. Separate
shared codecs/validation from builder-only code before tightening features.
The default runtime must not acquire RocksDB, Rhai or Python merely because
nonlinear291 becomes a supported preprocessor.

## Stage contracts and coding conventions

| Owner | Input / responsibility | Output / boundary to preserve |
| --- | --- | --- |
| GSS management | Validated recipe, paths, stage selection and recorded run state | Chooses the driver and exact programs; algorithms receive resolved options |
| Sandwich | Source circuit, companion construction, seeds and source width | Gate sequence with explicit physical width and documented payload ports |
| Preprocessing | Sandwich gates, mode-specific parameters and logical/physical layout | Gadgetized gates, physical width and the documented zero-slice/payload contract |
| Database mixing | Mutable circuit, store handles, replacement policy and RNG/state | Equivalent logical computation, replacement statistics and resumable state |
| Splitting | Shared mixer state and split settings | Split circuit and explicit stopping reason/checkpoint; crossing helpers may be used |
| Crossing | Split checkpoint/state, walk settings and work limits | Rewritten circuit with preserved provenance/undo invariants |
| Compression | Gate sequence and compression/packing settings | Final packed representation that evaluates according to the input contract |
| Circuit/storage libraries | Typed data and explicit options | Reusable computation or I/O; no command dispatch or implicit stage selection |

Document which wires carry logical inputs/outputs and which auxiliary wires
must start at zero. Nonlinear291 and quadratic masking have different physical
layouts. Their end-to-end equivalence checks must compare the documented
logical computation on the proper slice, not require equality on every
auxiliary output. Raw-core tests can exercise stronger contracts separately.
Do not use general full-width circuit comparison as a substitute for this
stage-specific verification.

Keep algorithms free of argument parsing, process spawning and direct
configuration-environment reads. Entrypoints own those adapters. Prefer small
concrete structs/enums and functions; use `pub(crate)` or narrower visibility
unless a retained tool needs a public API. Keep module comments short and
specific: purpose, inputs, outputs, invariants and the owning tests. Explain
compatibility code where it is declared, with a link to the format/recipe
contract. Keep dated experiments in research documentation.

Move error handling with its owner. Constructors should report invalid modes,
capacity or shapes before allocating large outputs; entrypoints turn those
errors into stable exit statuses. Preserve current error behavior during the
mechanical split and improve inconsistent behavior in a separately tested
change. A folder should have a concrete purpose; avoid adding miscellaneous
`utils`, `common` or `shared` directories as destinations for unresolved code.
The existing small command `shared.rs` can remain because its scope is clear.

## Source migration ledger

| Current code | Destination / action | What must stay stable |
| --- | --- | --- |
| `src/config.rs` | `src/gss/config/` | Defaults, validation, old inputs and duplicate-alias rejection |
| `src/preprocessing/sandwich.rs` | `stages/sandwich/construct.rs` | Source/sandwich seeds, gate order and payload placement |
| `src/preprocessing/blinded_v5.rs` | `stages/preprocessing/quadratic_masking.rs` | Algorithm and old raw constructor behavior |
| Stage-2 assembly in `src/bin/gen_sandwich_gadget.rs` | Preprocessing construction/dispatch | Guards, band seed/reseed, forward/reverse behavior and stage artifacts |
| `security_tests/support/preprocessing/nonlinear_gss.rs` | Native nonlinear291 under preprocessing; historical facade retained | Capacity planning, templates, layout, slice semantics and fan-in limit |
| Required guard code in historical `gadgets.rs` | `stages/preprocessing/slice_guards.rs` | RNG consumption, checks and scratch restoration |
| `src/preprocessing/guards.rs` | `stages/preprocessing/slice_guards.rs` | Guard semantics and port placement |
| `src/preprocessing/types.rs` | Stage-specific options/results stay with preprocessing; the generic gate-list/width container belongs with circuit types | Avoid representing all circuits as CNOT-only; keep `CnotCircuit` alias while callers migrate |
| `src/preprocessing/shared.rs` | Put the generic wire sampler with circuit randomization if both sandwich and preprocessing use it | Same rejection-sampling sequence |
| Four nonlinear291 template files | `stages/preprocessing/templates/` | Identical template bytes and one authoritative destination |
| `src/db_mixing/db_replace.rs` | `stages/db_mixing/replacement.rs` | Acceptance policy, geometry, remapping and key/control order |
| `src/db_mixing/frozen.rs`, `lookup_cache.rs` | `database/` | Stored formats, native/swapped compatibility and lookup semantics |
| Runtime `quality` modules and mixer integration | `stages/db_mixing/leakage_repair/` | Repair configuration and behavior |
| `src/postprocessing/splitting.rs` | `stages/post-processing/splitting.rs` | Split moves, stopping and checkpoint transition |
| `src/postprocessing/cross_walk.rs` | `stages/post-processing/crossing.rs` | Crossing, undo, provenance and RNG |
| `src/postprocessing/compress.rs`, `downhill.rs` | `stages/post-processing/compression/` | Reduction, transport, canonical packing and output evaluation |
| `src/engine/format.rs` | `circuit/formats.rs` | All existing readers/writers and packed formats |
| Canonicalization sections of `circuit.rs`, `engine/xpoly.rs` | `canonicalization/` | Database keys and budget/cap behavior |
| State/parameter/checkpoint/scheduling/reporting sections of `engine/mix.rs` | `engine/mixer/` | Public API initially, `.state` readers and exact serialized field ordering |
| Shared transformations in `mix.rs`, `rules.rs`, `swap_words.rs` | `engine/moves/` | Function preservation and verified transformation words |
| Three `src/bin/` files | Thin adapters in the chosen entrypoint directory | Cargo executable names, flags, exit codes and scripts |

Destinations abbreviated above are under `src/`. Keep forwarding exports at
explicit compatibility boundaries; do not keep copies of algorithms in both
old and new locations. The full current file map remains linked at the top.

## Configuration, aliases and saved runs

Changing the canonical mode string changes manifest text. Changing the driver
changes its fingerprint. The next managed recipe should therefore be version
7 when these user-facing changes land, with this sequence:

1. Save the exact current v6 driver as `scripts/compat/gss_mix_v6.sh` before
   editing the active driver. Preserve existing v3/v4/v5 snapshots byte-for-byte.
2. Parse old configuration spellings into typed values, but serialize old
   recipes using their version's original mode strings, field names and order.
   A global replacement of `as_str()` is insufficient.
3. Dispatch v3/v4/v5/v6 resumes to their original driver and recorded binaries.
   Do not rebuild over recorded binaries or rewrite existing manifests.
4. Use `quadratic-masking` and the new supported-mode validation for v7 runs.
   Existing mode aliases select the same new algorithm for fresh runs.
5. Keep product-2223 and standalone nonlinear193 available to supported old-run
   and legacy-tool paths. Reject them as fresh managed GSS choices before
   creating run artifacts or building stage binaries.
6. Keep `.state` v1, early-v2 and current-v2 readers and writers stable. The
   recipe version change does not require a checkpoint-format change.

Keep the current `[gadget]` and `--gadget-*` inputs as aliases if the proposed
preprocessing spellings are introduced. Do not rename all `BV5_*` variables
at the same time: isolate their translation at the old driver/environment
boundary first. Old direct research invocations and recorded recipe controls
need their original meaning. New core algorithm functions should receive
typed options, with environment reading confined to command/compatibility
adapters. Cosmetic naming must not change effective flags or randomness.

## Organizing the retained root folders

| Folder | Plan |
| --- | --- |
| `db_gen/` | Keep regular/curated/wide/frozen construction, analysis and maintenance. Separate heavy builders from shared runtime formats before changing feature gates. |
| `tests/` | Mirror runtime ownership; group stage correctness under `tests/stages/`, including preprocessing and post-processing. Preserve private test inclusion. Keep integration tests at Cargo-discovered paths or register moved targets explicitly. |
| `security_tests/` | Group heatmaps/affine heatmaps, gauntlet, attacks, leakage, comparisons, campaigns and reporting by purpose. Rust/Python/C++ tools for the same analysis should be co-located. Runtime nonlinear291 moves out; Python references, historical comparisons and experiment drivers remain. |
| `challenges/` | Keep all four programs. Add per-challenge folders only when shared files or fixtures justify them. |
| `benchmarks/` | Keep canonicalization, circuit, database and mixing groups. Move benchmark-only launchers beside their consumers after checking paths. |
| `configs/` | Keep small current examples and one settings reference. Machine-local overrides remain ignored. Move campaign TSVs beside their campaign only after auditing consumers. |
| `scripts/` | Keep the active driver, immutable compatibility snapshots and general verification/build helpers. Domain-specific launchers can live with their owner. |
| `docs/` | Separate current usage, architecture/design and format contracts from historical research. Keep related Markdown/TeX/PDF/figures together and repair links. |
| Root Cargo/TOML files | Keep package/lockfile, toolchain pin, formatting and Python packaging. Update executable paths/features and package inclusions as needed; a module move alone does not justify changing compiler versions. |
| `.github/` | Verify default runtime, optional tools, correctness and Python packaging as separate concerns. |

## Ordered implementation slices

Each slice should compile and have its relevant regressions passing before
starting the next. These are logical review boundaries, not authorization to
commit, publish or discard existing user changes.

1. **Freeze the baseline and compatibility contracts.** Record existing source
   and driver hashes, relevant test results and small deterministic fixtures.
   Keep a separate build directory. Inventory imports, `include_str!` assets,
   tests, Cargo targets, Python imports, script paths and documentation links.
2. **Split the mixer internally.** The current file is 8,094 lines. Extract
   parameters, counters/state, checkpoint serialization, provenance, reporting,
   scheduling and moves one responsibility at a time. Initially retain
   `engine::mix::Mixer` and private child-module access. Preserve struct field
   order where serialization or fixtures rely on it.
3. **Clarify circuit and storage ownership.** Move formats next to circuits;
   extract canonicalization and storage modules with forwarding exports.
   Preserve exact database key behavior and all circuit formats. Separate
   builder-only imports after shared primitives have an owner.
4. **Establish the requested stage folders.** Move sandwich, create
   preprocessing, and place splitting/crossing/compression under the literal
   `post-processing/` directory. Initially keep algorithms unchanged. Update
   private unit-test paths and all related imports together.
5. **Extract supported preprocessing implementations.** Move quadratic masking
   and native nonlinear291 into their owner; extract nonlinear slice guards
   from historical code; move the four runtime templates and repair exporter
   paths. Provide one explicit managed preset and a small supported-mode enum.
   Keep historical adapters and reference code functional. Apply the chosen
   nonlinear291 build policy without enabling unrelated legacy dependencies.
6. **Make entrypoints small.** Move reusable setup and construction out of the
   three stage `main()` functions. If selected, relocate their files to
   `entrypoints/` and update Cargo paths. Keep current program names, flags,
   exit statuses, artifact names and Bash orchestration.
7. **Apply public naming with recipe compatibility.** Introduce the new mode
   string and, if adopted, preprocessing configuration/flag spellings. Add the
   v6 driver snapshot and v7 recipe handling in the same slice. Validate fresh
   runs, resume, aliases and rejected retired modes together.
8. **Organize optional tools and documentation.** Update Cargo features,
   Python package/import paths, template exports, CI, campaign launchers and
   docs together. Check remote deployment consumers before changing their
   paths. Keep current research artifacts and archived scripts intact.
9. **Audit removal candidates separately.** Check normal GSS, optional tools,
   tests, public callers, Python and compatibility consumers before deleting.
   Classify unexplained legacy code with the user. Do not equate an unused
   default-build branch with a globally unused implementation.

Do not combine this with replacing the Bash pipeline, splitting into many
Cargo packages, changing defaults, altering checkpoint formats, redesigning
algorithms or raising the toolchain version. Each can be considered later on
its own merits. The initial refactor should improve navigation and ownership
while preserving results.

## Verification by risk

| Area | Required evidence when implementing |
| --- | --- |
| Mixer split | Existing golden DB/split/crossing runs; checkpoint chain state; provenance sidecars; piecewise seed and thread-count invariance |
| Checkpoint/recipe compatibility | v1/early-v2/current-v2 loading; v3/v4/v5/v6 manifests and driver hashes; wrong-binary rejection; resume skipping completed DB stages |
| Quadratic masking | Existing exhaustive-data/sampled-band evaluation in retained read/mask modes; encoded-I/O compatibility; unchanged seeded managed outputs and masks/refresh defaults |
| Nonlinear291 | Template hashes/shapes, every operation on the zero slice, heterogeneous chains, layout/resource bounds, unsupported-gate errors, fan-in-two output and deterministic seeds |
| Circuit formats/canonicalization | Real circuit CLI tests, full-width comparison, untouched-state behavior, format round trips, exact canonical keys and budget failures |
| Storage/builders | Real frozen round trips and DB moves, key/control-order conventions, builder feature compilation and default dependency inspection |
| Driver/config rename | Current and new aliases normalize consistently; duplicate aliases rejected; mode-specific controls validated; fresh retired modes rejected before writes; stage restart detection |
| Optional tools | Security/challenge/benchmark/DB feature combinations compile; old CLI aliases parse; gauntlet and Python reference imports work; eight extension exports still exist |
| Folder/path moves | Cargo metadata target paths, `#[path]` test modules, template includes, Python imports, driver calls, documentation links, formatting and CI |

Maintain source RNG draw order during mechanical extraction. Bitwise output
comparison is appropriate for deterministic fixtures. For genuinely timed
runs, compare the applicable functional/state invariants rather than promise
identical scheduling. Security measurements are comparative evidence; passing
correctness tests is not a security proof. Run a small matched-seed gauntlet
comparison when algorithm wiring moves, using the same recipe and inputs.

## Planning baseline and known limits

On 2026-09-10, the existing code passed **73 selected Rust tests**, using the
`legacy-tools` library build to cover the currently gated nonlinear adapter:

- 44 GSS configuration, normalization, environment and resume tests.
- 11 nonlinear adapter tests, including zero-slice behavior, capacity,
  deterministic construction, templates and complete fan-in-two output.
- One quadratic-mask regression covering multiple read/mask/encoded-I/O
  settings, exhaustive data inputs and sampled band states.
- Two checkpoint regressions, including v1/early-v2 loading and continuation.
- Three golden mixer regressions.
- Six piecewise mixer integration regressions and six scheduling unit tests.

`export_templates --check` also verified all eight current nonlinear templates.
The hyphenated module path compiled and ran in an isolated Rust example.
Build output went to `/tmp/gss-config-check-29im0rkj`, preserving recorded
runtime binaries. This is a targeted starting baseline, not a full security
campaign or a claim that the proposed code has been implemented and tested.
Four existing legacy-build warnings remain: an unused import, two unnecessary
`mut` bindings and an unread historical ledger field. Address them with their
owning cleanup; do not run broad automatic fixes over the dirty working tree.

A textual impact inventory found 34 files mentioning the old default gadget
names, seven mentioning the nonlinear runtime interface, 23 with selected
postprocessing/entrypoint paths, and 38 with nonlinear reference/template
names. These groups overlap and include historical documents/snapshots. They
are a starting checklist, not a global replacement list or a dead-code proof.

The current source tree and index already contain earlier authorized cleanup.
Use guarded edits and preserve that work. No source files, Cargo settings,
configuration defaults, executable names or Git index entries were changed
by this planning revision.
