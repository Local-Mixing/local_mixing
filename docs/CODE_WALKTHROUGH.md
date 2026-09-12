# Explaining Local Mixing: architecture, algorithms and optimizations

This walkthrough describes the working tree inspected on 2026-09-11. It is written for a programmer who understands loops, arrays and basic Boolean logic but has not worked on this repository. Read it alongside [the file inventory](CODE_LAYOUT.md): the inventory answers “where is everything?”, while this guide explains the computations, important functions and design decisions. Source references identify the implementation inspected; line numbers can change after subsequent edits.

The project transforms a reversible Boolean circuit into a larger, differently structured implementation of its documented logical computation. GSS means **gadgetized sliced sandwich**. The construction adds structure and auxiliary wires; database mixing, splitting and crossing rewrite that structure; final compression removes algebraic redundancy and packs the result. Separate tools measure what an observer can still recover from intermediate states and traces.

Two objectives must be kept separate when explaining the code. A rewrite must preserve the applicable circuit function. The resulting representation is also intended to obscure useful structure. Equivalence tests address the first objective; leakage and attack measurements provide evidence about the second. A functionally correct rewrite does not, by itself, establish an obfuscation guarantee.


## Reading route

- [Stage entry functions to follow in the source](#stage-entry-functions-to-follow-in-the-source)
- [Optimization map](#optimization-map)
- [Circuit representations and the mathematical model](#circuit-representations-and-the-mathematical-model)
- [How a managed run reaches the algorithms](#how-a-managed-run-reaches-the-algorithms)
- [Sandwich and preprocessing](#sandwich-and-preprocessing)
- [The mixer: a function-preserving random walk over circuit descriptions](#the-mixer-a-function-preserving-random-walk-over-circuit-descriptions)
- [Canonicalization, database lookup, and offline construction](#canonicalization-database-lookup-and-offline-construction)
- [Stages 4–5: split structure, then walk through collisions](#stages-45-split-structure-then-walk-through-collisions)
- [Stage 6: reduce functions and remove representation history](#stage-6-reduce-functions-and-remove-representation-history)
- [Analysis tools, optional programs and preserved campaigns](#analysis-tools-optional-programs-and-preserved-campaigns)
- [How the repository establishes correctness](#how-the-repository-establishes-correctness)
- [How to teach, investigate and extend the code](#how-to-teach-investigate-and-extend-the-code)


A useful initial presentation is:

```text
operator recipe + source circuit
           |
           v
managed runner: validate settings, choose binaries, check provenance
           |
           v
sandwich → preprocessing → database mixing → splitting → crossing
                                                         |
                                                         v
                                            compression and packing
                                                         |
                                                         v
                                                   final.esop1

Shared support: circuit representations, polynomial canonicalization,
                frozen databases, mutable mixer state and checkpoints.
Separate consumers: correctness tests, security analysis, benchmarks,
                    offline database builders and historical campaigns.
```

The source-stage names describe ownership, not separate processes. One generator process constructs both the sandwich and preprocessing. The `fmix` executable runs database mixing, splitting or crossing depending on its arguments. `fcompress` handles the last stage.

## Stage entry functions to follow in the source

The `Mixer` methods below are crate-internal operations on shared walk state; the construction and compression functions are reusable library interfaces. Links are relative to this guide's location in `docs/`.

| Stage | Entry function | Role |
| --- | --- | --- |
| Sandwich | [`prepare_source`](../src/stages/sandwich/construct.rs#L64) | Validate a supplied G57 circuit or sample one, preserving the historical source-seed behavior. |
| Sandwich | [`construct_seeded_sandwich`](../src/stages/sandwich/construct.rs#L94) | Build the selected classic or balanced sandwich using its dedicated random stream. |
| Preprocessing | [`preprocess_sandwich`](../src/stages/preprocessing/construct.rs#L13) | Dispatch to the selected preprocessor and assemble the guards, seeds and output contract appropriate to it. |
| Quadratic masking | [`preprocess_quadratic_masking_with_execution`](../src/stages/preprocessing/quadratic_masking.rs#L889) | Schedule masks, source operations and band refresh from resolved construction and execution parameters. |
| Nonlinear291 | [`nonlinear_gss_resource_plan`](../src/stages/preprocessing/nonlinear291.rs#L611) | Compute the exact physical width and a conservative gate-count bound before allocating the adapter. |
| Nonlinear291 | [`preprocess_nonlinear291`](../src/stages/preprocessing/nonlinear291.rs#L835) | Construct the supported nonlinear291 encoding/template/decoding pipeline with zero auxiliary inputs. |
| Preprocessing verification | [`verify_payload`](../src/stages/preprocessing/verify.rs#L10) | Compare the promised forward outputs on 256 bit-sliced samples and additionally check the classic guarded reverse payload. |
| Splitting | [`Mixer::split_twist_move`](../src/stages/post-processing/splitting.rs#L41) | Split a complemented gate and, when the join coin succeeds, attempt an absorbed NOT twist and a crossing shot. |
| Crossing | [`Mixer::cross_move_on`](../src/stages/post-processing/crossing.rs#L68) | Float a chosen gate toward a collision, then split, rewrite or retreat according to the crossing rules and limits. |
| Crossing contraction | [`Mixer::undo_move`](../src/stages/post-processing/crossing.rs#L349) | Find a valid journal entry and try to gather its descendants for exact reversal of the recorded crossing. |
| Crossing contraction | [`Mixer::merge_move`](../src/stages/post-processing/crossing.rs#L532) | Use gate indexes to find a reachable partner and apply an exact contraction identity. |
| Compression | [`compress_anc`](../src/stages/post-processing/compression/mod.rs#L143) | Iterate optional slice/liveness pruning, gathering and reduction while carrying aligned ancestry sets. |
| Packing | [`pack`](../src/stages/post-processing/compression/packing.rs#L105) | Turn each consecutive same-target run into one packed activation function in canonical ANF. |
| Packed compaction | [`compact_gate`](../src/stages/post-processing/compression/packing.rs#L66) | Derive a deterministic mixed-polarity ESOP from a packed ANF, retaining ANF when support exceeds 63 wires. |

## Optimization map

These are the main techniques to point out during a code review. “Avoided work” describes the mechanism, not a newly measured speedup. Search limits and sampling also change how much of a problem is explored; they should not be confused with an exact implementation of unlimited search.

| Expensive operation | Technique and avoided work | Where to look / important condition |
| --- | --- | --- |
| Evaluate a wide state | Direct limb/bit addressing and in-place mutation avoid whole-integer shifts and copies | [General gate kernels](../src/circuit/xgate.rs#L225); dispatcher must choose sufficient width |
| Evaluate many inputs | Pack 64 samples into bits; carry four batches together so one gate traversal serves 256 inputs | [Bit-sliced kernels](../src/circuit/xgate.rs#L159); samples and wires are different dimensions |
| Parse millions of gates | Byte scanners, a static base-83 decoder, reusable buffers and direct decimal output | [Formats](../src/circuit/formats.rs#L20), [G57 encoding](../src/circuit/formats/g57.rs#L11) |
| Store and sort small control sets | Inline `SmallVec` storage, insertion sort and direct two-control ordering | [Literal representation](../src/circuit/xgate.rs#L12) |
| Edit the middle of a large tape | Integer-ID arena with linked execution order avoids shifting the remaining tape | [Arena](../src/engine/arena.rs#L88); liveness and mutation stamps distinguish reused IDs |
| Find collisions and merge partners | Cached wire masks, target/support buckets and bucket-position bookkeeping avoid general scans | [Collision fast path](../src/engine/arena.rs#L145), [indexes](../src/engine/mixer/indices.rs#L7); wide cases fall back to general logic |
| Select low-generation material | Cadenced pool rebuild, partial selection and lazy stale-entry removal avoid sorting or rebuilding every round | [Sampler pool](../src/engine/mixer/sampling.rs#L111); cadence and selection policy remain part of the algorithm |
| Build preprocessing operations | Reusable parsed templates, checked preallocation, dirty-helper identities and bounded mask counts | [Nonlinear adapter](../src/stages/preprocessing/nonlinear291.rs#L835), [quadratic emitter](../src/stages/preprocessing/quadratic_masking.rs#L557) |
| Canonicalize tied variables | Structural refinement, exact compact rank keys, automorphism pruning and reusable recursive scratch reduce backtracking work | [Canonicalizer](../src/canonicalization/canonicalize.rs#L777); compact keys have an eligibility condition |
| Compose large polynomials | Sorted-vector algebra and reusable multiplication/merge buffers reduce allocation; budgets stop excessive growth | [Window composition](../src/canonicalization/window.rs#L11), [XGate budgets](../src/canonicalization/xgate.rs#L29); exceeding a budget is a skipped/undecided case |
| Repeat canonical queries | Exact normalized-window caches, shared values and precomputed hashes avoid recomposition and rehashing | [G57 cache](../src/canonicalization/cache.rs), [XGate cache](../src/canonicalization/xgate.rs#L543); explicit options bypass historical caches |
| Read absent or repeated DB keys | BinaryFuse filters, response caches and positional bucket reads reduce storage traffic | [Frozen lookup](../src/database/frozen.rs#L812), [lookup cache](../src/database/lookup_cache.rs#L143); filter positives require lookup and keys are fingerprints |
| Decode a large candidate pool | Keep byte offsets/lengths, choose using cheap metadata, decode the selected spelling | [Replacement core](../src/stages/db_mixing/replacement.rs#L897); rejected choices can cause another draw/decode |
| Decode compressed values | Short-code Huffman lookup table, Elias–Fano indexing and skip-without-materialization avoid reconstructing unrelated records | [Huffman table](../src/database/frozen.rs#L109), [value skipping](../src/database/frozen.rs#L313) |
| Simplify local expressions | Small-support exact tables, bounded ANF expansion, grouped gathering and dependency indexes avoid unconstrained global search | [Compression](../src/stages/post-processing/compression/mod.rs), [reducer](../src/stages/post-processing/compression/reduce.rs) |
| Mix a long tape across workers | Independent contiguous-piece rounds share immutable stores and collect results in piece order | [Piecewise runner](../src/engine/mixer/piecewise.rs#L1082); fixed partitions/configuration define the reproducibility promise |
| Produce state-comparison grids | Parallel input evaluation, snapshot transposition and disjoint row tasks improve locality and avoid output contention | [Python heatmap core](../security_tests/python/heatmap.rs#L83); grid size can still dominate memory/work |
| Re-run completed pipeline work | Versioned recipe and artifact checks skip valid completed stages | [Managed runner](../src/gss/runner.rs#L94); a changed implementation must not silently reuse incompatible state |

## Circuit representations and the mathematical model

A gate toggles one target bit based on other bits. For a general `XGate`, the firing function is

```text
fire = comp XOR product(literal(control_i))
target = target XOR fire

literal(w, true)  = bit[w]
literal(w, false) = 1 XOR bit[w]
```

Here product means Boolean AND, while XOR is addition modulo two. The target is excluded from its own controls. Consequently applying the same gate twice restores the input: the controls are unchanged by the first application. Every gate is an involution, and reversing the order of a gate list computes its inverse.

`XGate` stores a target, a complement bit and sorted control literals ([`src/circuit/xgate.rs:15`](../src/circuit/xgate.rs#L15)). A plain conjunction with no controls is a NOT gate because the empty product is one. `XGate::conj` merges duplicate equal literals and rejects contradictory controls; the `mpmct1` reader rejects malformed duplicate/self-target controls before they can violate downstream assumptions. Sorted, unique control wires are an invariant used throughout the mixer, not merely a formatting preference.

The repository also retains a specialized G57 representation. `CircuitSeq` stores triples `[target, x, y]`, with the action `target ^= x OR NOT(y)` ([`src/circuit/g57.rs:15`](../src/circuit/g57.rs#L15), [`src/circuit/evaluate.rs:10`](../src/circuit/evaluate.rs#L10)). `XGate::from_g57` translates this to `comp=true` with the conjunction `NOT(x) AND y`, since `1 XOR (NOT(x) AND y) = x OR NOT(y)`. Be particularly careful about the order of the two control columns when working with historical stores.

The generic `Circuit` holds `Vec<XGate>` and an explicit physical wire count ([`src/circuit/types.rs:5`](../src/circuit/types.rs#L5)). `CnotCircuit` is a compatibility name for this generic type; it does not mean that every gate is a CNOT. `Permutation` represents wire orderings used to move between original, dense and canonical wire labels ([`src/circuit/permutation.rs:7`](../src/circuit/permutation.rs#L7)). Wire renaming and reordering gate execution are different operations.

Three width limits have different meanings. Wires in gate representations use `u16` labels. The file-oriented `CircuitSource` utility supports at most 1024 wires. Polynomial lookup windows can touch at most 64 distinct wires because a monomial uses a `u64` bitset. Thus a large physical circuit can still use canonical database lookup on small local windows; the 64-variable canonicalization limit is not a whole-circuit limit.

| Read here | Important entry points | Responsibility |
| --- | --- | --- |
| [`src/circuit/xgate.rs:22`](../src/circuit/xgate.rs#L22) | `conj`, `x_gate`, `cnot`, `from_g57`, `collides` | General gate construction and commutation/dependency logic |
| [`src/circuit/evaluate.rs:179`](../src/circuit/evaluate.rs#L179) | `CircuitSeq::evaluate_64/128/256/512/1024` | Specialized G57 execution |
| [`src/circuit/xgate.rs:280`](../src/circuit/xgate.rs#L280) | `eval_lanes`, `eval_lanes4`, `eval_limbs` | Generalized gate execution |
| [`src/circuit/operations.rs:38`](../src/circuit/operations.rs#L38) | `CircuitSource::read`, `evaluate`, `compare_sampled` | Reusable file-based circuit utilities |
| [`src/circuit/randomize.rs:6`](../src/circuit/randomize.rs#L6) | `random_circuit`, `shoot_random_gate` | Source generation and retained randomization helpers |
| [`src/circuit/formats.rs:302`](../src/circuit/formats.rs#L302) | `read_mpmct` | General/packed circuit loading |
| [`src/circuit/formats/g57.rs:47`](../src/circuit/formats/g57.rs#L47) | `CircuitSeq` blob/text methods | Compact historical G57 encodings |

### Vocabulary used below

| Term | Meaning in this code |
| --- | --- |
| Literal / cube | A literal is a wire value or its negation; a cube is an AND of literals. A generalized gate condition adds the complement bit to one cube. |
| Support / degree | Support is the set of wires used by an expression or window. Polynomial degree is the largest number of variables in any remaining monomial. They measure different limits. |
| ANF | Algebraic normal form: an XOR of positive-variable monomials. With fixed input labels, every Boolean function has one such form after duplicate terms cancel. |
| ESOP | Exclusive sum of products: an XOR of cubes that may contain negated variables. Several different ESOPs can represent the same function. |
| GF(2) / affine | GF(2) arithmetic uses XOR for addition and AND for multiplication. An affine function is a constant XORed with selected input bits. |
| Ancilla / clean / dirty helper | An ancillary wire supplies extra workspace. A clean helper starts at a known value, usually zero. A dirty helper can start arbitrarily, but its borrowing identity must restore it. |
| Commutation / conjugation | Commuting operations can exchange order without changing the function. A conjugation rewrites an interior operation in the frame created by surrounding inverse operations. |
| Canonical form / fingerprint | A canonical form deterministically removes the supported labeling choices. A fingerprint is a compact hash of that form; comparing hashes is weaker than comparing complete forms. |
| Window / candidate | A window is the current local subcircuit selected for rewriting. A candidate is an alternative stored spelling to map back onto that window's wires. |
| Provenance / litter | Provenance records how current gates arose. A litter groups gates produced together in one rewrite; exact ancestry records sets of contributing original gates. |

### Evaluation and parsing optimizations

**Scalar versus bit-sliced state.** In scalar evaluation, a `u64` contains one sample with up to 64 wires. In bit-sliced evaluation, `state[wire]` is a `u64` whose bits represent 64 independent samples of that wire. One AND/XOR instruction then evaluates the same logical operation for all 64 samples. `apply_lanes4` carries four such words per wire, evaluating 256 samples while traversing each gate and its controls once ([`src/circuit/xgate.rs:159`](../src/circuit/xgate.rs#L159)). This is software batching; it should not be described as an explicit hardware-SIMD implementation without examining the generated machine code.

**Direct limb access.** Wide scalar evaluators address a wire as `(wire >> 6, wire & 63)` rather than shifting an entire 256/1024-bit integer per control. The gate-list evaluators mutate one limb array in place, avoiding repeated wide-state copies ([`src/circuit/xgate.rs:225`](../src/circuit/xgate.rs#L225), [`src/circuit/evaluate.rs:156`](../src/circuit/evaluate.rs#L156)). G57 dispatch chooses the narrowest appropriate kernel. Some fixed-size kernels use power-of-two index masks plus debug assertions so the compiler can eliminate redundant bounds checks; callers must still establish the in-range invariant.

**Branchless polarity.** Polarity and complement bits become all-zero or all-one masks. XORing a sample word with the polarity mask chooses the positive or negative literal without a data-dependent branch. The control loop remains a sequence of AND operations ([`src/circuit/xgate.rs:270`](../src/circuit/xgate.rs#L270)).

**Small and ordered control lists.** `Lits = SmallVec<[(u16, bool); 6]>` keeps up to six controls inline. `sort_lits` uses insertion sort because lists are short and usually already sorted. The two-control G57 conversion directly emits the correct order instead of invoking a general sorting routine ([`src/circuit/xgate.rs:12`](../src/circuit/xgate.rs#L12), [`src/circuit/xgate.rs:255`](../src/circuit/xgate.rs#L255)). These techniques reduce allocation and fixed setup costs in loops executed millions of times.

**Byte-oriented formats.** The G57 parser uses a compile-time 256-entry character lookup table for the base-83 encoding ([`src/circuit/formats/g57.rs:11`](../src/circuit/formats/g57.rs#L11)). General and packed readers scan decimal fields directly from bytes. Writers reuse a line buffer, append decimal digits directly and use a large `BufWriter`, avoiding a temporary formatted string per field ([`src/circuit/formats.rs:20`](../src/circuit/formats.rs#L20)). These are performance techniques, not a reason to weaken format validation.

### Formats and comparison contracts

`mpmct1` stores one generalized cube gate per line. `PackedGate` stores an XOR of terms for one target. `anf1` restricts those terms to positive literals; `esop1` permits positive and negative literals. `PackedGate::expand` emits one `XGate` per term, so existing consumers can read packed circuits through `read_mpmct` without changing the represented function ([`src/circuit/formats.rs:146`](../src/circuit/formats.rs#L146)). A packed-gate count, an expanded cube count and a literal count measure different things.

`CircuitSource::compare_sampled` samples all wires up to the maximum requested/declared/touched width and compares complete output states. It rejects zero samples and preserves untouched state bits. Its absence of a counterexample is a sampled result. It must not replace a preprocessing test that specifies zero auxiliary inputs and only particular logical output ports; those are different equivalence contracts.

## How a managed run reaches the algorithms

The top-level `main` creates three command groups: `gss`, `circuit`, and the feature-gated `db` group ([`src/main.rs:4`](../src/main.rs#L4)). Files under `commands/` define arguments and dispatch. The three stage entrypoints contain small `main` adapters; `programs/` owns their argument/environment translation, reporting and artifact I/O. This lets reusable algorithm modules operate on typed circuits and parameters.

The main orchestration function is `run_inner` in [`src/gss/runner.rs:94`](../src/gss/runner.rs#L94). Read it as a sequence of boundaries:

1. Locate the repository, read the selected configuration and parse it.
2. Resolve typed values, defaults, environment-backed paths and path separation constraints.
3. Determine whether the directory is a fresh run, a managed resume or a historical pre-wrapper run.
4. Choose the recipe-version-specific Bash driver and validate its settings and recorded fingerprints.
5. Build the three stage binaries for a fresh run if requested. A managed resume uses the recorded binaries instead of rebuilding over them.
6. Prepare the manifest, configure the child environment and invoke the driver.

`parse_config`, `parse_toml` and `parse_markdown` preserve both current TOML and the historical fenced configuration format ([`src/gss/config/parse.rs:127`](../src/gss/config/parse.rs#L127)). Aliases are normalized to a single internal setting; specifying both aliases for the same setting is an error. `resolve_config` performs value/range/path validation before creating run output ([`src/gss/config/validate.rs:4`](../src/gss/config/validate.rs#L4)). `RawConfig` keeps input information; `ResolvedConfig` is the effective recipe. A sourced path also records whether its value came from configuration or the environment, which makes diagnostics explainable ([`src/gss/config/mod.rs:82`](../src/gss/config/mod.rs#L82)).

New recipes are version 7 and expose `[preprocessing]` and `--preprocessing-*`. Existing recipe versions 3–6 keep their original driver bytes and serialization conventions. `recipe_manifest`, `compare_recipe_manifest` and `script_for_recipe` implement this contract ([`src/gss/manifest.rs:164`](../src/gss/manifest.rs#L164)). The main manifest uses streamed XXH3-128 fingerprints of the three executables and driver; these identify recorded implementations rather than providing signed software authenticity. The stage-1/2 marker separately uses SHA-256 over the generator, selected controls and supplied source contents.

`configure_environment` removes research override namespaces and sets the supported recipe values ([`src/gss/runner.rs:402`](../src/gss/runner.rs#L402)). For example, the managed runner pins canonicalization budgets and cache capacities; direct research calls can retain historical environment behavior through compatibility adapters. A parser alias is therefore only one part of compatibility: recipe serialization, binary selection and environment interpretation must agree too.

### The six stages and their artifacts

The authoritative sequence is in [`scripts/gss_mix.sh:381`](../scripts/gss_mix.sh#L381):

| Stage | Process and input | Main artifact / reason |
| --- | --- | --- |
| 1 + 2 | `gen_sandwich_gadget` constructs the sandwich and selected preprocessor | `gss.mpmct1`, sandwich/source sidecars and `stage12.recipe` |
| 3 | `fmix --gss --db-mixing` reads the preprocessed tape | `db_mixing.mpmct1` and `db_mixing.state`; grows/holds the representation according to the profile |
| 4 | `fmix --split --split-stop` reads the mixed tape | `split.mpmct1` and `split.state`; runs the splitting stage to its stopping condition |
| 5 | `fmix --resume split.state` runs crossing | `crossing.mpmct1` and `crossing.state`; continues the state needed for the walk |
| 6 | `fcompress` reads the crossing output | `final.esop1`; algebraic reduction and packed output |

The current stage-3 profile has an expansion leg and a hold, with no separate compression leg. Stages 3 and 4 can run piecewise; stage 5 resumes the split checkpoint and passes an **absolute** move ceiling that includes moves already recorded. A plain tape file carries gates; a mixer checkpoint carries the additional state needed for continuation. They are not interchangeable.

Completed artifacts are skipped subject to the driver's recipe/provenance checks; an explicit rerun stage invalidates the appropriate continuation. The driver persists a run seed and derives stage seeds by fixed offsets. Fresh default seeds come from Python's `secrets` draw; explicit seeds are a calibration mode. The subsequent algorithms use deterministic seeded streams, so preserving RNG draw order matters for reproducibility. No actual run seed is needed to explain the implementation.

The defaults and every public flag are documented separately in [GSS_FLAGS.md](GSS_FLAGS.md). Use that reference for parameter values; this walkthrough explains what the parameters control.

## Sandwich and preprocessing

A slice contract specifies behavior when designated input wires start zero. “Junk” means that an output is unconstrained by that contract; the complete physical circuit remains reversible. The sandwich establishes logical ports, and preprocessing must preserve the appropriate ports.

### Stage 1: arrange two computations around a copy layer

Let the original computation be `C` on `n` wires and `D` a freshly sampled companion. The sandwich has two registers, `x` and `y`, each of width `n`. Its classic layout executes:

`[C interleaved with S1]; [y ^= x]; [D interleaved with S2]`.

Both computations target `x`. Every slice gate in `S1` and `S2` targets `x` and positively reads `y`, so these gates are inactive when `y=0`. On entry `(x,0)`, the first block computes `C(x)`, the copy layer preserves it in `y`, and the second block can scramble only `x`. Thus `A(x,0)=(junk,C(x))`. Reversing the gate list similarly gives `A^-1(p,0)=(junk,D^-1(p))`. These are two computations available through the same zero-register interface, not an assertion that `D=C`. The [construction’s algebra](../src/stages/sandwich/construct.rs#L215) spells this out.

The balanced variant places `D` on `y`, copies downward with `x ^= y`, and mirrors `S2`. Its forward payload remains on `x`: `A(x,0)=(C(x),junk)`. Its inverse uses a different slice: `A^-1(0,q)=(junk,D^-1(q))`. Confusing the forward and reverse zero registers breaks verification.

Construction preserves the order within each computation while selecting a uniformly random order-preserving interleaving with slice gates. Slice gates are roughly one-third CNOTs and otherwise two-control gates. The copy layer’s gates then float through commuting neighbors in independently assigned directions. Only those mutually commuting copy gates move, so [one extremal pass suffices](../src/stages/sandwich/construct.rs#L386); repeatedly scanning to a fixpoint would add work without additional travel. Recorded copy-gate identities matter because balanced slice gates can have the same shape.

Default slice length scales as `n log2(n)` and companion length as `n log2(n)^2`, each floored at `n`. Source sampling, sandwich sampling, and later preprocessing use separate seed streams, making comparisons reproducible.

### Stage 2: hide the logical computation inside a larger reversible circuit

New managed recipes support quadratic masking and nonlinear291. Historical names remain compatibility aliases; historical nonlinear193 is feature-gated. Explicit stage parameters avoid reading process environment inside these computations.

#### Quadratic masking

For a logical circuit of width `N`—the sandwich has `N=2n`—the computation adds a band of `R` wires, normally `R=N`. A data wire holds its logical value XORed with masks built from band wires. The basic mask gate is

`w ^= 1 ^ (!x & y) = 1 ^ y ^ x*y`.

Balanced masks add `w ^= z`, giving `1 ^ y ^ x*y ^ z`. A uniform independent `z` balances this term while preserving its quadratic part. This is a property of the mask expression under the stated distribution, not a general security proof about the complete generated trace.

The builder jointly schedules mask openings, source gates, and band refresh. A mask’s opening and closing cancel if its inputs are unchanged; refresh handling maintains the corresponding algebra when band values change. Source gates become ready through a [dependency graph](../src/stages/preprocessing/quadratic_masking.rs#L807) containing read-after-write and write-after-read hazards. Same-target controlled XORs commute, so unnecessary write-after-write edges are omitted. Full read/write history is retained to avoid moving a writer before a read of the initial value.

At a source gate, each operand is expressed as its currently masked wire XOR its open mask polynomial. The default quadratic firing multiplies these expressions directly, rather than temporarily making an operand affine in band wires. Two quadratic operands produce terms of degree at most four. The emitter realizes them with gates of at most two controls: cubic terms use four-gate dirty-helper brackets, quartic terms use eight-gate brackets with two helpers. For example,

`t ^= h*y; h ^= P*x; t ^= h*y; h ^= P*x`

has net effect `t ^= P*x*y` for any initial `h`, and restores `h`. This avoids allocating clean scratch wires and keeps emitted operations compatible with the low-fan-in mixing vocabulary. [The emitter](../src/stages/preprocessing/quadratic_masking.rs#L557) prefers helpers outside the operands’ masks and avoids selected partial products that would cancel another data wire’s open mask. Each bracket remains intact while commuting firing units are shuffled.

A fresh target mask is inserted between portions of a firing, so the target’s before/after XOR across that portion does not directly equal the source gate’s increment. Production uses `K=2`, balanced masks, at most three ordinary open masks, and a minimum of two between a wire’s first opening and final closing. Top-ups and replacement masks maintain coverage. Sampling prefers disjoint mask wires; small bands relax that preference after bounded retries, so small exhaustive examples do not reproduce every statistical property of production-size bands.

Refresh comes in two forms. A straddle slot closes masks reading the refreshed band wire, opening replacements first where necessary. A repair slot removes the affected terms and reapplies them using the updated band value, retaining the open masks. Bursts amortize this maintenance across multiple updates to one band wire. Defaults request about `m/(4K)` slots of `8K` updates, roughly `2m` refresh gates. Even scheduling puts refresh inside the computation; oversized explicit schedules report undelivered slots instead of appending a misleading final flush.

[Full delivery](../src/stages/preprocessing/construct.rs#L13) consists of opening guard, input-derived band seed, masked computation, independent band reseed, and closing guard. The reseed is a separate module, not the inverse of the initial seed. The opening guard is inactive when auxiliaries start zero. The closing guard can fire, but targets only the sandwich’s forward-junk half: low for classic, high for balanced. Consequently only the designated payload half is promised after this assembly. The classic construction also has a checked reverse-payload contract; the implementation does not claim the same check for balanced delivery.

The schedule is built in one forward construction, but this is not a blanket linear-time algorithm: collecting every hazard edge can be quadratic, and firing expansion grows with the number of open mask terms. Small `K` and bounded mask counts control that cost.

Two easily confused settings are deliberately different: managed refresh bursts may read data and band wires (`burst_band_only=false`); dirty firing helpers default to band wires only (`ancilla_band_only=true`). The raw masked computation restores every logical output, whereas its closing guard intentionally narrows that guarantee. Encoded-I/O experiments return separate off-circuit encoding/decoding gates; production exposes ordinary logical I/O. Optional `hot_intervals` identifies affine exposure intervals for diagnostics and is not deliverable circuit data.

#### Native nonlinear291

Here each logical bit is represented by two five-wire blocks, decoded as `E(S1) XOR E(S2)`, where `E(x)=x0 XOR x1 XOR maj(x2,x3,x4)`. Raw logical wires remain a separate public prefix. Ingress derives carrier masks from the stable original-data prefix and solves one coordinate so the decoded bit equals its raw input. Each source operation updates the target’s encoding by assigning a fresh share and three fresh majority coordinates; later operations follow that new physical mapping. Egress uses a clean delta wire to replace each raw value with its final decode. Every auxiliary input must start zero; every logical output is preserved, while auxiliary outputs may remain junk.

The [adapter](../src/stages/preprocessing/nonlinear291.rs#L684) accepts complemented mixed-polarity r57, uncomplemented `!a & b`, positive AND, and positive copy. Their template bodies contain 291, 288, 291, and 127 gates respectively. “291” therefore names a particular operation body, not the complete circuit’s gate count. Physical width is `12N + 12m + 29`: ten carrier wires and one delta per logical bit, twelve per source operation, and shared scratch/decomposition resources in addition to the raw prefix.

The four templates are embedded text parsed once with `OnceLock`; shape, fan-in, indices, and trailing tokens are checked. Ordinary Rust generation needs no Python runtime. The [Python reference](../security_tests/gadgetization/nonlinear291.py#L116) explains how the topology was produced: accumulate mask controls before operand controls, reuse safe mask-prefix products, invalidate them before a mask changes, then uncompute caches in reverse construction order. Shared scratch is restored between operations. This reduces the reference r57 decomposition from its documented naïve 505 gates to 291; the runtime consumes the already generated topology.

Checked resource planning precedes classification and allocation, enforces the `u16` wire ceiling, and computes reservation sizes. A slice preblock covers auxiliary controls; three-control macros use dirty-scratch decompositions to keep the complete adapter at fan-in two. The checker indexes macros by slice control for singleton/pair probes and packs random probes into bit lanes. Eight fixed commuting-swap passes mix seams with linear work. The zero-slice identity follows gate algebra; rejection sampling’s “every nonzero slice disturbed” condition is exhaustive only for small layouts and sampled for larger ones. Disturbing a slice means finding at least one data assignment changed on that slice, not proving that every off-slice input is changed.

## The mixer: a function-preserving random walk over circuit descriptions

The mixer changes how a reversible Boolean function is written. Its state is an ordered tape of `XGate`s plus the metadata needed to choose, undo, measure, and resume rewrites. A gate flips its target when `comp XOR product(control literals)` is true. The target is absent from its own controls, so applying that gate twice cancels. This involution property underlies cancellation, reverse-direction crossing, and conjugation. Circuit order still matters: two individually reversible gates need not commute.

The main owner is [Mixer](../src/engine/mixer/state.rs#L389); construction in [new_with_shared_db](../src/engine/mixer/state.rs#L717) retains the original circuit, creates the mutable tape, initializes directions and provenance, and accepts a shared immutable database. Move randomness and several diagnostic metrics use separate seeded generators. This separation is selective: global functional verification deliberately consumes the move generator, so verification cadence remains part of reproducible configuration.

### The mutable tape and its invariants

[Arena](../src/engine/arena.rs#L88) stores gates, previous/next links, liveness, and mutation stamps in parallel vectors addressed by integer node IDs. Local insertion or removal changes links instead of shifting every subsequent gate. Freed slots can be reused; the stamp distinguishes a current gate from an older occupant of the same ID. The linked list determines execution order, while vectors provide direct access.

The mixer maintains additional indexes rather than repeatedly scanning the tape. [index_add](../src/engine/mixer/indices.rs#L7) and [index_remove](../src/engine/mixer/indices.rs#L24) organize potential merge partners by target and control-wire set. Per-node bucket positions permit constant-time `swap_remove`; when the last bucket member moves, its recorded position is repaired. This removal deliberately changes bucket order. A candidate still passes the actual merge predicate, so a hash match alone never authorizes a rewrite. Separate side indexes track complemented gates and eligible targets.

Collision checks also have a specialized representation. [collides_ids](../src/engine/arena.rs#L145) uses cached control and polarity masks covering 128 wires. If neither gate reads the other's target, they commute. For uncomplemented gates, shared controls of opposite polarity provide another commuting case. Complemented predicates require the guarded path. The arena falls back to the general predicate when wires exceed mask capacity. [The mask equivalence test](../tests/unit/engine/mixer/mix_tests.rs#L650) compares these implementations; this optimization accelerates an existing predicate rather than discovering every possible semantic commutation.

### How a round chooses work

[Mixer::run](../src/engine/mixer/scheduling.rs#L425) is the dispatch loop. A live splitting stage owns its round. Otherwise, the mixer updates either the size-profile controller or the static size brake, then tries channels in a fixed order: twist, commuting shuffle, optional bridge, database replacement, and finally the size thermostat.

The shuffle probability is `clamp(shuffle_rate / current_gate_count, 0, 1)`. A selected database slot consumes the round even on a miss. It does not fall through to an unrelated move: otherwise database hit rate would silently change the requested proportions of work.

The thermostat turns excess size into a logistic contraction probability, bounded by configured limits. A contraction round may try compressing database replacement, then undo and merge in a randomized preference order. The other branch expands. These are probabilistic controls, so an individual round need not move size toward its target.

[prof_target](../src/engine/mixer/scheduling.rs#L7) describes the optional size trajectory: ramp toward the first target, hold, then ramp toward the second. Feedback estimates observed growth, compression, and disturbance and adjusts channel settings. Profile mode owns size control instead of simultaneously applying the static brake. This is an empirical controller, not an optimization proof.

Every completed round advances the work clock; effective work accumulates approximately one divided by current size. Stops include budgets, stage boundaries, empty circuits, explicit flags, and configured dose/canary conditions. [global_check](../src/engine/mixer/reporting.rs#L7) compares against the retained original using 256 random inputs. Four batches of 64 bit lanes are evaluated together in one arena traversal, avoiding four dependent linked-list walks while preserving random draw order. This check is sampled, not exhaustive. Local verification and database verification are separately configurable; the presence of a verifier does not mean every rewrite is exhaustively checked in every configuration.

### Why the moves preserve the function

Commuting transport moves a gate or block only through neighbors permitted by the collision predicate. [collect_convex](../src/engine/mixer/sampling.rs#L337), for example, repeatedly moves a block through commuting surroundings and absorbs the next dependency. An arbitrary reorder would not have this justification.

The crossing algebra lives in [rules::cross](../src/engine/moves/rules.rs#L76). R0 swaps a commuting pair. R1 splits the moving gate into cases according to whether its collider changes a control. R2 splits the collider when the moving gate changes its control. R3 handles mutual dependence by leaving a sensitive residue and crossing the remaining cases; some configurations are blocked. Required widths are checked before emitting changes. Complemented conjunctions first use [presplit](../src/engine/moves/rules.rs#L63): the negation of a conjunction becomes disjoint cases for the first failing literal. Exactly one case fires whenever the original predicate fires. Randomizing the literal order changes the spelling without changing that partition. Leftward crossing reverses the corresponding rightward rewrite because each gate is an involution.

[merge_result](../src/engine/mixer/transformations.rs#L255) implements a small, guarded catalogue. Equal gates cancel. Equal controls with opposite complement bits yield an unconditional target flip. The identity `xF XOR (!x)F = F` removes a differing literal. Subset-related predicates support another contraction. Complement guards prevent these rules from manufacturing forbidden complemented residues. [Catalogue tests](../tests/unit/engine/mixer/mix_tests.rs#L695) check soundness and those guards.

Expansion runs related identities in the other direction and can insert cancelling pairs. [twist_move](../src/engine/mixer/transformations.rs#L545) uses brackets and conjugates their interior by a wire swap or negation, restoring the external function. The G57 variant can absorb nearby gates into synthesized seams. Its [SwapWordEngine::solve](../src/engine/moves/swap_words.rs#L270) represents a four-wire function as its complete 16-state permutation, packed into a `u64`. A breadth-first table of short words meets an inverse frontier to search bounded-length decompositions; adjacent identical involutions are pruned. The shared engine and memoized seam results avoid rebuilding pure searches. This is bounded synthesis over a particular gate alphabet, not global minimum-circuit synthesis.

The optional [bridge insertion](../src/engine/mixer/transformations.rs#L1416) places two carrier copies near distant endpoints and adds corrections beside intervening colliders. The corrections account for moving the carrier through those gates, giving a telescoping functional identity. Inserted material is tracked so a declined far-window replacement can roll the insertions back.

[verify_rewrite](../src/engine/moves/rules.rs#L278) gives exact local checking on supported sizes: densely remap touched wires, evaluate every input assignment, and compare both sequences. Sixty-four assignments occupy the bits of each machine word, sharing Boolean operations across lanes. Reused state buffers and fixed low-wire patterns avoid rebuilding scalar inputs. The function asserts support at most 24 wires; exhaustive verification still grows exponentially.

### Database windows and candidate choice

The mixer first chooses a window geometry and its permitted size. [collect_contiguous](../src/engine/mixer/sampling.rs#L284) extends from a seed in its stored direction, with bounded attempts to move obstructing wide gates aside. Convex sampling gathers a dependency-connected block through commuting moves. [collect_pair](../src/engine/mixer/sampling.rs#L409) instead searches a commutation box for a distant partner. These geometries expose different equivalent subcircuits to the same replacement mechanism.

[sample_best_window](../src/engine/mixer/sampling.rs#L63) can choose among several trials by distinct litter count, encouraging material from different earlier replacement events to meet. Generation-biased sampling uses [rebuild_pool](../src/engine/mixer/sampling.rs#L111): a cadenced scan selects the lowest-generation eligible gates with `select_nth_unstable`, avoiding a complete sort. Random draws lazily remove stale or newly ineligible entries. Support and width caps prevent unusable gates from permanently dominating this pool. Generation and litter diversity describe rewrite history, not demonstrated secrecy.

Replacement's [shared core](../src/stages/db_mixing/replacement.rs#L897) composes and canonicalizes the window, checks degree limits, and queries equivalent spellings. Curated lookup, when armed, comes first in the forward direction. Regular lookup follows a complete curated miss only when fallback is enabled. Finding curated entries but no size-eligible choice does not automatically authorize regular fallback. Reverse canonicalization is delayed until regular lookup needs it. The default regular min-key route depends on the database builder storing the hash of the minimum forward/reverse canonical polynomial vector; alternate legacy and validation policies exist.

Candidate handling avoids decoding every stored circuit. It records offsets, byte lengths, gate counts, store origin, and direction into retained value buffers. [choose_ref_with_options](../src/stages/db_mixing/replacement.rs#L1291) scans these small references to establish eligibility, counts the pool, draws one uniform rank, and finds that member. Only the selected spelling is decoded and mapped from canonical wires back to the circuit. Scratch wires are assigned from available wires; invalid, identity, or banned reorder choices can be rejected and retried.

Selection policy matters. `Mix` prefers uniformly among no-larger candidates; if none exist it normally chooses among the shortest, unless configured to broaden that pool. `Compressing` chooses the shortest candidate no longer than the outgoing window: equal-length respelling is allowed. Other modes accept arbitrary sizes, minimum size, exact size, or a narrow size neighborhood. Incoming absolute bands apply before these rules. The [selector oracle test](../tests/stages/db_mixing/replacement/tests.rs#L89) checks both the selected result and resulting RNG state against the older allocation-heavy selector.

Typed [db_replace_with_options](../src/stages/db_mixing/replacement.rs#L821) and [db_replace_options](../src/stages/db_mixing/replacement.rs#L858) expose explicit replacement policy. Ordinary mixer database walks retain compatibility replacement/cache policy. `MixRuntimeOptions` resolves its documented mixer controls; it is not a declaration that every database operation underneath a mixer ignores legacy environment settings.

### Provenance, undo, and continuation

Per-gate [Meta](../src/engine/mixer/state.rs#L298) records origin, event, direction, generation, litter, and litter size. A litter groups outputs born in one rewrite. Exact ancestry separately unions bitsets of contributing original gates; sampled ancestry tracks a selected tracer subset. A single origin tag cannot represent a merged gate's entire ancestry. Dead ancestry can be pruned while preserving records referenced by usable undo entries.

[try_undo](../src/stages/post-processing/crossing.rs#L388) requires every recorded fragment to remain linked with its saved stamp. It can gather fragments by commuting them back together before restoring the previous pair and metadata. Stamps prevent a recycled arena slot from accidentally satisfying an old journal entry. Undo is therefore conditional, not an unconditional stack operation. Recent event IDs can also be tabu for a configured move interval. [is_tabu](../src/engine/mixer/provenance.rs#L805) uses binary search because events enter the queue in increasing order; age checks and front removal maintain the bounded history.

[save_state](../src/engine/mixer/checkpoint.rs#L283) saves the original reference, current tape, counters, provenance, directions, and usable undo data. IDs are renumbered into tape order and journal references remapped. [resume_state](../src/engine/mixer/checkpoint.rs#L463) accepts current version 2 and legacy layouts. However, `StdRng` state is not serialized: saving draws continuation seeds from both generators. Saving consumes RNG draws, and resumed execution is not bit-identical to uninterrupted execution. [Round-trip](../tests/unit/engine/mixer/mix_tests.rs#L2503) and [legacy-format](../tests/unit/engine/mixer/mix_tests.rs#L3292) tests protect chain state and compatibility, not that stronger replay claim.

### Piecewise execution

[run_piecewise](../src/engine/mixer/piecewise.rs#L1082) cuts the tape into contiguous pieces, mixes them independently, concatenates their results, and changes the cuts across rounds so earlier seams enter later interiors. [run_round](../src/engine/mixer/piecewise.rs#L956) shares immutable database storage, derives seeds from master seed/round/piece, separates event/litter ID ranges, and collects outputs in piece order. Worker scheduling does not choose the seeds or concatenation order.

Each piece verifies against its own incoming function, while the whole mixer retains its true original. Counters and metadata are folded back into the whole-circuit owner, which retains the controller. Effective-work budgets account for differing piece sizes. [The thread-count test](../tests/unit/engine/mixer/mix_tests.rs#L264) checks deterministic output for the same partition/configuration across worker counts. Changing partitioning changes the walk; piecewise execution is not promised to reproduce a serial trajectory, and time-driven stops weaken reproducibility.

### Leakage detection and guarded repair

[run_quality_control](../src/stages/db_mixing/leakage_repair/mod.rs#L127) is a separate bounded detector/repair loop. Its detector samples original-circuit features and current writer-to-writer segments using packed Boolean traces. [AffineBasis](../src/stages/db_mixing/leakage_repair/detect.rs#L585) learns GF(2) affine relations on training samples and evaluates the same coefficients on held-out samples; a constant column prevents trivial constants from masquerading as reference relationships. Firing evidence uses binary correlation with variation and sign checks, including terminal boundary firings.

For a hot segment, [plan_convex_block](../src/stages/db_mixing/leakage_repair/blocks.rs#L56) finds an order-convex closure in the forward collision DAG under explicit span, gate, and support limits. Bounded database candidates are tried in a cloned circuit. Screening includes both replacement seams and affected borrowed wires; truncated screening cannot approve a repair. The first passing candidate reaches [apply_quality_repair](../src/engine/mixer/leakage_repair.rs#L33), which verifies the gathering permutation through commuting swaps and checks functional equivalence before touching live state.

Successful repair uses the shared mutation machinery, preserves unaffected IDs/metadata, and keeps the normal mixer random stream unchanged. [Atomic refusal tests](../tests/unit/engine/mixer/leakage_repair/tests.rs#L203) and [state-preservation tests](../tests/unit/engine/mixer/leakage_repair/tests.rs#L50) cover these invariants. After each splice the controller rescans, preventing stale segment indices; a final independent sample seed provides a fresh audit after adaptive repairs. Results remain evidence from configured samples, features, and budgets. A clean report is not a proof of security or absence of every form of leakage.

## Canonicalization, database lookup, and offline construction

This describes the current source as inspected on 2026-09-11. The central runtime path is: express a window as Boolean output polynomials, choose a deterministic wire ordering, fingerprint that canonical form, find stored equivalent spellings, and map a selected spelling back onto the circuit. Offline programs construct and validate the candidate stores. They are separate from ordinary runtime builds: [`src/lib.rs:4`](../src/lib.rs#L4) gates `db_generation` behind tests or the database/benchmark tool features.

### Exact Boolean algebra and bounded composition

[`polynomial.rs:4`](../src/canonicalization/polynomial.rs#L4) represents a monomial as a `u64` variable mask and a polynomial as a sorted vector of masks. Mask zero means the constant one; an empty vector means zero. Computation occurs in the Boolean polynomial ring over GF(2): addition is XOR, duplicate monomials cancel in pairs, and multiplication unions masks because `x*x=x`. `normalize_polynomial` sorts and retains odd multiplicities. `poly_xor_assign` performs a sorted symmetric-difference merge. `substitute_input_negation` replaces `x_w` with `x_w+1` by generating the corresponding terms with bit `w` removed and merging them into the original polynomial.

[`CircuitSeq::to_polynomial`](../src/canonicalization/window.rs#L11) starts with identity output polynomials and composes each G57 gate. For triple `[a,b,c]`, the target update is `P_a ^= 1 + P_c*(1+P_b)`, matching the gate's `b OR NOT c` toggle condition. The companion capped method returns failure when its term bound is exceeded. Scratch multiplication and merge vectors retain allocation capacity across gates. Dense wire remapping limits polynomial width to the wires actually touched; windows with more than 64 distinct wires cannot use this representation and are skipped.

[`xgates_to_polynomial`](../src/canonicalization/xgate.rs#L198) generalizes composition to arbitrary supported XGates: a gate toggles its target by the XOR of its complement bit and the product of its positive or negative control literals. `XPolyBudget` at line 29 bounds raw multiplication work, each reduced polynomial, and total live terms. Its defaults are respectively 1,048,576, 262,144, and 1,048,576 terms. Invalid wires, target-in-controls, excessive support, and budget exhaustion are errors. The optional degree cap is checked on the composed polynomials before canonical ordering; it is an exact degree test, although using it to skip database lookup assumes the store's advertised degree coverage is valid.

The algebra provides a mathematical guarantee: normalized output polynomials on the same physical input variables uniquely describe the Boolean function. [`polys_equivalent`](../src/stages/db_mixing/replacement.rs#L60) therefore compares both circuits after mapping their combined support into one coordinate system. It returns `Some(true)` or `Some(false)` when composition succeeds, and `None` when resource limits prevent a decision. This is stronger than agreement on sampled inputs; a budget failure is not permission to accept a rewrite.

### Deterministic ordering and its optimizations

[`canonicalize_polys_4_using`](../src/canonicalization/canonicalize.rs#L1209) normalizes the input and groups output wires by degree profiles. Each group's class polynomial counts monomial occurrences using natural-number multiplicities, deliberately retaining information that XOR cancellation would discard. All wires initially share a rank. The refinement loop in [`canon4_run_inner`](../src/canonicalization/canonicalize.rs#L777) repeatedly splits tied wires using ranked monomial levels, occurrence frequencies, individual polynomial keys, and dynamically formed rank-class polynomials. These are deterministic structural invariants used to reduce the remaining search.

When refinement leaves a tie, Rule L individually promotes each candidate in the first tied group, recursively completes the ordering, and selects the lexicographically smallest completed canonical form. If Rule L is disabled, unresolved ties fail. Equal completed forms reveal automorphisms; the implementation can skip another branch only when a known automorphism preserves the current coloring and connects it to an already explored branch. That pruning preserves the selected form. The shared per-call branch counter charges candidate groups at recursive nodes; exceeding its cap aborts rather than returning an approximate key. The final permutation remaps both output positions and monomial variables, and [`trim_canonicalized`](../src/canonicalization/canonicalize.rs#L65) removes only trailing identity outputs that earlier outputs do not reference. Thus canonical comparison is in a relabeled coordinate system, not direct equality on the original numbered wires.

Several optimizations target work and allocation without changing that ordering. Monomial rank keys carry a packed 128-bit prefix, with a full representation available when necessary. The compact scanning path is used only when every monomial has degree at most 16, making that representation exact. Per-class wire-union masks skip classes unrelated to current ties; clean flags avoid repeating unaffected scans; tied-group masks narrow frequency counting. [`Canon4Frame`](../src/canonicalization/canonicalize.rs#L719) supplies reusable per-thread scratch frames at recursion depth, so vectors can be cleared without discarding capacity. The branch budget is ordinary per-call state, separate from this scratch pool. These are implementation techniques, not claims of a particular measured speedup.

### Fingerprints, direction conventions, and caches

[`polys_repr_blob`](../src/canonicalization/keys.rs#L5) emits sorted monomials as little-endian `u64` bytes with a `u64::MAX` separator after each polynomial. The lookup key is `xxh3_128(blob).to_le_bytes()`. This serialization and canonical ordering are persisted compatibility boundaries: changing them can make existing entries unreachable even if the new algorithm is mathematically reasonable.

Direction matters because every supported gate is an involution, so reversing the sequence implements the inverse function. The regular G57 builder's [`canonicalize_bidirectional`](../db_gen/regular.rs#L98) computes both canonical polynomial vectors and stores the smaller vector's direction; ties additionally select a deterministic gate spelling. Runtime replacement uses the same polynomial-vector comparison, not the smaller hash. `MinDirLookup::Min` consequently needs only the minimum-direction regular probe when the store satisfies that contract; `Legacy` retains the historical probing policy, and `Validate` additionally checks the alternate key on a minimum-key miss and counts violations. Curated lookup initially uses the forward key. The separate offline wide-gate format compares serialized polynomial blobs instead; its verifier at [`wide_verify.rs:13`](../db_gen/bin/wide_verify.rs#L13) preserves that distinct convention.

Legacy G57 canonicalization caches successful results by exact dense normalized gate sequences, including the negation variant's distinct key encoding. Entries share polynomials, permutation, and precomputed hash through `Arc`. XGate canonicalization additionally caches failures with its explicit composition budget and degree cap included in the key. Caps approximate memory and use whole-cache clearing rather than LRU eviction. [`LookupCache`](../src/database/lookup_cache.rs#L143) is a separate cache of database responses: a namespace byte separates regular and curated keys, misses are cached, and regular positive results are shared. Positive curated values are deliberately not retained in this general cache because they can be large.

The compatibility boundary is explicit. [`canonicalization/legacy_environment.rs`](../src/canonicalization/legacy_environment.rs#L1) preserves historical parsing, defaults, and independent first-read `OnceLock` behavior; existing public wrappers retain it. New `*_with_options` methods accept concrete options and bypass those process-wide canonicalization caches, preventing one call's budget from contaminating another. [`FrozenDb::open_with_options`](../src/database/frozen.rs#L1100) resolves control convention and filter loading without environment reads; caller-owned `LookupCache` binds its lifetime and state to one database. [`ReplacementOptions`](../src/stages/db_mixing/replacement.rs#L621) similarly supplies direction, incoming-length band, and canonical-search controls. Old wrappers still use their legacy adapters; merely adding these APIs did not change all historical callers' effective configuration.

### Frozen storage and candidate selection

[`split_key`](../src/database/frozen.rs#L351) divides the first 76 bits of the key's byte sequence into an 8-bit shard, 20-bit bucket, and 48-bit tail. The remaining 52 bits are not stored. There are 256 shard files, each with a header and `2^20+1` packed 40-bit offsets identifying bucket byte ranges. Buckets contain sorted tails encoded in Elias–Fano form: unary upper parts followed by fixed-width lower parts. Values use canonical Huffman tables, including gate-position/width contexts and escape forms. [`rebuild_canonical`](../src/database/frozen.rs#L109) constructs a 12-bit decoding lookup table for short codes and retains a longer-code fallback.

After a decoded-pool cache miss, [`get_capped`](../src/database/frozen.rs#L812) checks an optional per-shard BinaryFuse8 filter. A valid matching filter can reject an absent fingerprint without disk access; a positive result still requires lookup. The reader obtains only the selected bucket with `read_exact_at`, allowing independent concurrent positional reads, and reuses a thread-local buffer. It scans Elias–Fano upper parts to identify the possible range, compares only those lower parts, skips predecessor values without materializing their output circuits, and decodes the selected value. Control-order compatibility swapping happens at one decoding boundary, including raw escape blocks. A separate per-store cache retains fully decoded pools exceeding 20 candidates, up to 4,096 keys; bounded QC prefixes never populate it.

This index is a fingerprint lookup, not a mathematical equality proof. Both the 128-bit hash and the stored 76-bit prefix entail assumed accidental-collision risk. [`stage_write`](../db_gen/frozen_build.rs#L735) refuses two source keys that collapse to the same stored address, but that does not rule out a future unrelated query sharing the prefix. Normal `db_replace` selection itself trusts the stored mapping and does not perform an ANF proof before returning its choice; callers' verification policy therefore matters.

Stored legacy records are `[byte_length][G57 triples]`. Replacement catalogues record byte offsets, gate counts, source store, and direction; selection then chooses using length and curated preference policies. Only the selected friend is converted to gates; an unplaceable, identical, or forbidden reorder candidate is removed and selection retried. [`friend_to_xgates`](../src/stages/db_mixing/replacement.rs#L122) reverses a friend when needed, undoes the canonical permutation and dense remapping, and draws available scratch wires if the friend needs additional slots. Those selection preferences are search heuristics, not guarantees of minimum gate count or leakage resistance.

QC has a deliberately separate bounded path. [`get_qc`](../src/database/frozen.rs#L712) checks bucket size before allocation and uses a strict reader with a shared work budget that also charges skipped predecessors. [`QcCandidates`](../src/stages/db_mixing/replacement.rs#L278) distinguishes missing entries, malformed values, truncation, unplaceable candidates, and undecidable equivalence. The runtime bucket bound is 16 MiB. Candidate enumeration considers both directions and both stores round-robin, bounds examined records, gates, and support, and proves every returned candidate with `polys_equivalent`. Exhausting its search cannot establish that no acceptable equivalent exists.

### Offline workflows and what validation establishes

Regular construction starts with [`build_m1`](../db_gen/regular.rs#L1436), extends previous size classes in [`build_from_rocks`](../db_gen/regular.rs#L1058), or combines two classes in [`build_from_2rocks`](../db_gen/regular.rs#L1751). Abstract fresh-wire representatives avoid enumerating redundant assignments; canonicalization, deduplication, buffered sorted writes, and SST ingestion organize the results. Requested wire/gate bands and any permitted unresolved-Rule-L skipping determine coverage. Builders reject environment canonicalization caps and existing output paths. Regular intermediates do not currently have the curated completion manifest: interrupted output is not a resumable completed corpus, and structural merge/export validation is not a universal semantic re-audit of every regular candidate.

Curated construction derives alternatives from identities, imports older curated material, or takes a regular-store shortcut with different provenance. [`derive_identity_candidates_where`](../db_gen/curated_full.rs#L114) visits both orientations, every cyclic rotation, and each accepted split. For identity `AB=I`, prefix `A` equals reversed suffix `B^-1`; both are mapped into the prefix's canonical wire space. Each emitted candidate passes [`validate_and_emit`](../src/database/validation.rs#L12), which recanonicalizes and checks its full key. This is a strong consistency check subject to the hash assumption. Composite RocksDB keys combine the full function key and candidate bytes, so deduplication does not require a historically capped list in one value. Finalization writes exact key/candidate counts and a digest; [`verify_manifest`](../db_gen/bin/build_curated_full.rs#L1687) recomputes that structural audit, detecting an incomplete or altered snapshot rather than re-proving every circuit.

Freezing runs `stage_tables`, `stage_write`, and [`stage_validate`](../db_gen/frozen_build.rs#L822) against the same source. The last compares decoded frozen entries and values with the source, establishing byte-preserving export. Composite sources can feed the encoder directly instead of exceeding the legacy LMDB per-value limit. Filter generation scans the actual frozen address set, checks every inserted fingerprint before and after serialization, and publishes atomically while refusing overwrite. Wide builders at [`regular.rs:336`](../db_gen/regular.rs#L336) and line 514 add one or two wide gates, use the separate MPX1 heterogeneous encoding and offline reader, and have a verifier that decodes every stored candidate and recomputes its directional key. That remains an offline development path, not a promise that ordinary frozen G57 lookup accepts MPX1 values.

Relevant existing evidence includes [golden canonical hashes](../tests/unit/canonicalization/g57/tests.rs#L915), [relabeling invariance](../tests/unit/canonicalization/g57/tests.rs#L1173), [XGate scalar/algebra agreement](../tests/unit/canonicalization/xgate/tests.rs#L73), [recursive budget and per-call isolation](../tests/unit/canonicalization/options.rs#L6), [legacy lazy environment reads](../tests/unit/canonicalization/options.rs#L116), [Huffman reference equivalence](../tests/database/frozen/tests.rs#L235), [QC read/decode bounds](../tests/database/frozen/tests.rs#L44), [LMDB/frozen lookup round-trips](../tests/frozen_roundtrip.rs#L46), [regular direction compatibility](../tests/db_gen/regular_validation_tests.rs#L53), [identity-derived candidate keys](../tests/db_gen/curated_full/tests.rs#L59), and [interrupted curated-store rejection](../tests/db_gen/bin/build_curated_full/tests.rs#L69). These are source references inspected for this explanation, not claims that new tests or production database builds were run during this read-only pass.

## Stages 4–5: split structure, then walk through collisions

These stages operate on the existing `Mixer`, retaining arena node identities, direction metadata, indexes, ancestry, and checkpoint state. Their rewrites preserve the full function, beyond any input-slice promise used earlier.

Splitting replaces a complemented conjunction with a randomized first-failing-literal decomposition. Pieces get alternating directions. An optional [absorbed NOT twist](../src/stages/post-processing/splitting.rs#L289) selects another bracket on the same target wire, flips both brackets’ literal polarities, and flips every intervening read of that wire. The implicit pair of NOTs cancels algebraically, so no explicit NOT padding is needed. Encountered complemented readers are themselves split. Bracket selection uses target-wire buckets and a tournament favoring distant candidates; approximate ranks are refreshed periodically and after substantial growth. Direction probabilities account for remaining length on each side, avoiding a bias toward tiny edge spans.

Crossing floats a selected gate to its next real collision, then applies the exact rule catalogue, possibly splitting either participant. Width caps and probabilistic damping limit expansion. Failed shots retreat along their already traversed commuting path. Optional least-split-lineage sampling uses a periodically rebuilt pool, linear selection, and lazy removal of dead entries; with that option disabled no additional random draws occur.

Contraction uses exact undo and indexed merges. Undo journals exist only when enabled, use node stamps to reject stale events, and bound retries. Gather searches try a fragment’s stored direction first. Merge candidates come from matching support indexes; nearest reachable partners are preferred, and collider lists are built only if a candidate is reached, avoiding collision tests on fruitless scans. Cross outputs receive the union of both parents’ ancestry, including a surviving pivot; undo retains the earlier ancestry needed to reverse this bookkeeping.

## Stage 6: reduce functions and remove representation history

Compression optionally specializes known-zero inputs with three-valued constant propagation, then prunes dead output cones backwards. These options change the promise: specialization guarantees equivalence only on the chosen input slice; liveness guarantees only selected outputs. Neither means arbitrary wire deletion is valid.

A forward sweep gathers same-target gates while controls remain valid. Separated readers can commute past a group. [Transport](../src/stages/post-processing/compression/transport.rs#L61) crosses a control writer by substituting its update into the group’s ESOP, accepting a non-worsening cost by default. Changed groups record dependencies on the writer’s group; cycle detection and dependency-first emission preserve the frame in which their new expressions are valid. Target and control indexes avoid repeatedly scanning every open group, and a group cap bounds local work.

Reduction first applies cancellation/drop-literal/subsumption rules, then considers canonical ANF expansion. At support size at most four, lazily built BFS tables give minimum-cube ESOPs. Larger cases compare greedy subcube covering plus maximum bipartite matching against matching alone; oversized matching falls back to greedy pairing. Support and expansion budgets prevent unbounded expansion: default ANF support is 40, the hard mask limit is 63, and expansion is capped at `2^18` generated monomials. The largest exact table contains 65,536 functions and 81 candidate cubes; caching amortizes that startup cost. Repeated pairwise catalogue scans can still be expensive, so the default 64-member group cap matters. Reverse gathering finds leftward opportunities; downhill conjugation applies a nonoverlapping best-first set of profitable local moves in one linear rebuild. Iterations stop on unchanged `(gate count,literal count)` and reject regression.

Verification skips unchanged reductions, compiles literal tests into bitmasks, checks small supports exhaustively, and samples larger ones. Ancestry follows moved gates; reduced groups stamp every survivor with their member union. Downhill preserves the crossed neighbor’s tag and unions the rewritten block’s tags. These are diagnostic derivation sets, not a minimal algebraic dependency proof.

Finally, packing replaces each consecutive same-target run by one activation function in canonical ANF. This may increase term count: its purpose is to erase the history carried by a particular cube decomposition. Deterministic compaction derives a smaller ESOP from that ANF alone, retaining one spelling per activation function within a fixed variable labeling. It does not canonicalize an entire circuit or quotient arbitrary wire renamings. Packing bounds negative-literal expansion; compaction leaves supports above 63 wires in ANF.

Useful executable examples are [sandwich slice/inverse tests](../tests/stages/sandwich/construct.rs#L56), [exhaustive masking tests](../tests/stages/preprocessing/quadratic_masking.rs#L23), [nonlinear shape/chain/resource tests](../tests/stages/preprocessing/nonlinear291.rs#L61), [seeded managed fixtures](../tests/stages/preprocessing/managed_construction.rs#L33), and [compression equivalence, frame-order, ancestry, and packing tests](../tests/stages/post-processing/compression/compress_tests.rs#L93).

## Analysis tools, optional programs and preserved campaigns

The production pipeline and its analysis consumers share circuit primitives, but they answer different questions. A compressor asks whether a local expression can be shortened without changing its function. A heatmap asks how executions relate. An attack tool asks whether a particular observer can recover information. Explain these independently instead of treating every measurement program as another pipeline stage.

### Gauntlet: reproducible trace-and-attack experiments

The gauntlet's Python orchestrator is [`security_tests/gauntlet/gauntlet.py:878`](../security_tests/gauntlet/gauntlet.py#L878). Its `generate_cell`, `stage_is_current`, `artifact_signatures` and `build_report` functions manage a matrix of construction/input-policy/mixing cases ([`security_tests/gauntlet/gauntlet.py:410`](../security_tests/gauntlet/gauntlet.py#L410)). The executable `gauntlet_gen` constructs or reads a gadget, optionally applies the shared mixer, evaluates bit-sliced samples and records initial wire values plus each gate's flip and resulting target value ([`security_tests/gauntlet/gauntlet_gen.rs:220`](../security_tests/gauntlet/gauntlet_gen.rs#L220)). Python `gauntlet_build.py` supplies encoded inputs and decode metadata for nonlinear reference constructions.

`gauntlet_audit` reads that bundle and runs several observers ([`security_tests/gauntlet/gauntlet_audit.rs:338`](../security_tests/gauntlet/gauntlet_audit.rs#L338)): direct feature matching; affine recovery from all wires at one intermediate circuit state; affine recovery from the full trace span (represented by all initial wire values plus every gate flip); and covariance with one-, two- and three-feature Boolean expressions. Affine models are fitted on one sample region and checked on a held-out region; correlations use a separate tail and a NULL reference. The fit/holdout split helps distinguish an actual relation from an accidental fit to too few samples.

The weight-2 and weight-3 scans use capped feature subsets, so pair/triple work is bounded rather than exhaustively covering a large trace. Their work grows combinatorially with the feature cap. The global affine attack has a feature-size guard and can explicitly report that it was skipped. A skipped attack is not a successful zero-leak result.

The current orchestrator's enabled arms are the unprotected control, balanced quadratic masking, its wide-band variant and nonlinear291 ([`security_tests/gauntlet/gauntlet.py:58`](../security_tests/gauntlet/gauntlet.py#L58)). Some older construction modes still exist in the native generator or Python builder but are not enabled entries in this matrix. Historical testing documents describe additional arms; the current `ARMS` dictionary is the selection source of truth. The encoded-I/O/random-band quadratic gauntlet arm is also not identical to the managed sandwich's input-seeded, ordinary-I/O delivery. An analysis result must name the tested contract.

Cell manifests keep generation, audit and rendering records separate. Artifact hashes and configuration records determine whether a stage can be reused. Each mixed file-mode case probes its own dimensions before sampling, avoiding a shared temporary probe race. Python coordinates case-level concurrency; Rust performs the computation. This organization improves throughput and reproducibility without pretending that an empirical battery proves security.

### Heatmaps and other observers

[`security_tests/heatmaps/hmap_affine.rs:1`](../security_tests/heatmaps/hmap_affine.rs#L1) fits a GF(2)-affine predictor for a source intermediate bit from a wider implementation's state, then measures its held-out error. Its degree-2 mode additionally supplies products of selected wires as regressors. This can detect a nonlinear relation that an affine-only fit misses, but regressor count and basis memory grow rapidly; restricting the product wires restricts the observer's coverage.

The optional Python extension registers eight stable functions in [`security_tests/python/mod.rs:6`](../security_tests/python/mod.rs#L6). Its `compute_grid_parallel` computes circuit evolutions over inputs in parallel, transposes snapshots to `[position][input]` so each cell traverses contiguous samples, and assigns disjoint output rows to Rayon workers ([`security_tests/python/heatmap.rs:83`](../security_tests/python/heatmap.rs#L83)). Inputs are generated sequentially before parallel evaluation so worker scheduling does not reorder RNG draws. Depending on options, a cell reports normalized Hamming distance, the mean normalized absolute Hamming-weight difference (averaged across inputs and divided by the selected wire count), or a transformed distance. These are measurements of states, not equivalence checks. Snapshot-only corner sampling avoids retaining every intermediate state when only selected positions are needed ([`security_tests/python/heatmap.rs:727`](../security_tests/python/heatmap.rs#L727)).

Other directories provide specialized observers:

| Area | What belongs there |
| --- | --- |
| `security_tests/demixing/` | Attempts to simplify or recover structure from a released circuit. `fmix_downhill` reuses the compressor's exact conjugation/reduction core; it does not need the source seed or mixer journal. |
| `security_tests/oracle/` | Oracle-query learning and preimage experiments, including SAT-based candidate learning. |
| `security_tests/attacks/` | Circuit-to-CNF encoders, SAT-related helpers and solver-trace analyses; external solvers are separate dependencies. |
| `security_tests/leakage/` | Source/trace correlation, gate-flip recovery, persistence and structural measurements. |
| `security_tests/fixtures/` | Circuit generators, transformations and verification fixtures used by experiments. |
| `security_tests/orchestration/`, `campaigns/` | Challenge preparation, supervisors, historical execution recipes and artifact management. |
| `security_tests/reporting/` | Plotting and report generation; some recipes deliberately retain dated artifact paths. |
| `security_tests/support/`, `legacy/`, `experiments/` | Retained comparison algorithms and the historical `legacy_mixing` command family. |
| `security_tests/gadgetization/` | Python nonlinear reference implementations and the template exporter. |

The four `challenges/` programs are `block_cipher`, `point_function`, `poly_canon` and `newton_feistal`; they cover cipher/entropy experiments, point-function construction, graph-canonicalization comparison and Newton/Feistel experiments respectively. Their existing names remain stable.

`benchmarks/` contains evaluation and canonicalization measurements plus mixing and database probes. Use `bench_eval` for evaluation kernels, the canonicalization probes for key-computation cost, and the database/mixing probes for candidate geometry, degree, span, hit rate or mobility. End-to-end runtime combines allocation, algebra and storage latency, so a faster evaluator does not imply the same proportional improvement to a database-bound run. Report cache temperature, workload, parameters and feature set when comparing measurements.

### Build boundaries and the historical workspace

`Cargo.toml` explicitly registers binaries (`autobins=false`). Dropping a Rust file into a directory does not make it a supported executable. Default GSS compiles the runtime, while feature gates add offline database tools, security/legacy tools, challenges, benchmarks or Python extension linking. The default runtime excludes RocksDB, LMDB, Rhai, PyO3 and NumPy; test fixtures may use LMDB without making it a runtime dependency. The library builds as both an `rlib` and a `cdylib`; maturin selects the extension linking feature for Python packaging. Thin LTO and one release codegen unit allow compiler optimization across more of the program, with a compile-time cost (`Cargo.toml:1`).

`mod.rs` files primarily establish module ownership, reexports and test inclusions. The directory literally named `post-processing` is mapped once to the Rust module `post_processing`. Compatibility modules and aliases forward old imports to current implementations; they do not imply a second copy of the main algorithm. Explicit canonicalization, storage and replacement APIs accept typed options. Legacy environment adapters preserve the old lazy or open-time reads. `MixRuntimeOptions` resolves its documented mixer controls; ordinary DB-backed mixer walks still use the historical replacement/cache policy unless the explicit replacement interfaces are used.

The physical workspace also retains substantial campaign code outside the organized source tree. `affine_mixing_tests/` contains its own harness, configuration and supervisors; `experiments/` contains SAT scaling/structure tools and captured study sources; `mixing_tests/` retains historical tests and a vendored FASTER-related tree. These paths coexist with generated databases, run directories and reports because saved deployments refer to them. Explain them as campaign/snapshot consumers, not additional owners of the current production algorithms. [CODE_LAYOUT.md](CODE_LAYOUT.md) distinguishes enumerated source files, preserved sources and folders whose generated or restricted contents are not expanded.

## How the repository establishes correctness

Tests mirror ownership. Many bodies live under `tests/unit/` or `tests/stages/` but are included from the owning module with `#[cfg(test)]` and `#[path]`. This gives tests access to private implementation details without exposing internals as public APIs. Files directly registered as integration tests exercise the external interfaces and real processes.

| Question | Best evidence to read |
| --- | --- |
| Does a local identity or optimized evaluator preserve the same function? | Circuit/move unit tests, exhaustive small cases, randomized equivalence and old-versus-new kernel comparisons |
| Does a preprocessor preserve the intended ports on the intended slice? | `tests/stages/preprocessing/` and sandwich tests; inspect initialization and output projection explicitly |
| Do canonical keys remain stable? | `tests/unit/canonicalization/`: golden keys, G57/XGate agreement, budget and cache-isolation cases |
| Do real store files decode and replace correctly? | `tests/frozen_roundtrip.rs` and `tests/fmix_db_move.rs`, which build synthetic LMDB sources and exercise actual frozen files |
| Does a checkpoint retain its documented state? | `tests/unit/engine/mixer/mix_tests.rs`: version compatibility, chain state, provenance and piecewise cases |
| Do commands reject or report invalid requests correctly? | `tests/circuit_cli.rs`, `tests/unit/gss.rs`, `tests/gss_flags.bash` and `tests/gss_block_size.bash` |
| Does the nonlinear reference agree with its templates? | `tests/gadgetization/` and the exporter's `--check` path |
| Do retained tools still build? | Optional-feature checks in `.github/workflows/ci.yml` |
| Did a refactor change a deterministic experiment? | Matched-seed circuit/trace comparisons, alongside the above functional tests |

The current CI also checks formatting, Clippy, default dependency exclusions, legacy command compatibility, Python packaging and all eight heatmap exports. The previous cleanup's executed results are recorded in [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md); this guide was assembled by source inspection rather than by rerunning that suite.

## How to teach, investigate and extend the code

A productive first session starts with `XGate` and one scalar evaluation. Translate a G57 gate by hand. Then demonstrate two exact identities: applying a gate twice cancels it, and two gates with the same target commute when neither reads that target. Next show how the sandwich chooses payload ports and why auxiliary initialization is part of the contract. Only then walk through a database lookup and a mixer move; the state and mapping machinery will have a reason to exist.

For a second session, follow one move through outgoing-window selection, dense remapping, canonical key computation, store lookup, candidate choice, placement and mutation. Track **three orders** separately: gate execution order, physical wire labels and canonical variable order. Look at an undo record and an ancestry set to see why a gate tape alone cannot stand in for mixer state. Finish with compression to distinguish simplifying a function from merely serializing it in fewer packed records.

When changing code, start with the owner of the changed behavior:

- A new operator setting belongs in config/CLI parsing, typed resolution and validation, driver translation and versioned recipe serialization, with alias/resume tests.
- A gate semantic change belongs in circuit primitives, with format, evaluator, collision and polynomial agreement checks before changing the mixer.
- A new replacement policy belongs in stage replacement logic; preserve placement maps, selection probabilities and RNG continuation when the intended change is only a speed optimization.
- An arena mutation must update every index, direction/stamp/provenance structure that relies on node identity. Read the existing mutation helpers before adding another mutation path.
- A faster canonicalizer must preserve complete keys and budget failure behavior. Shortcuts that change canonical order also change database compatibility.
- A compression change must preserve the right frame across transported writers and the promised live-output/slice contract. Measure expanded cubes and literals in addition to packed gate count.
- A checkpoint change requires explicit format/version handling. A recipe change may require a new driver/manifest version even when the checkpoint layout does not change.

The recurring optimization principle is to avoid doing general, allocating work in a tiny hot operation: keep small data inline, reuse scratch buffers, visit only candidate nodes, cache exact repeat computations, postpone decoding until selection, and batch independent Boolean evaluations. Equally important are the limits that contain work—canonical search budgets, monomial caps, group caps, candidate limits and stopping rules. Some improve raw execution cost without changing results; others deliberately limit the search and therefore affect which useful rewrites can be discovered. Preserve that distinction when explaining or benchmarking them.
