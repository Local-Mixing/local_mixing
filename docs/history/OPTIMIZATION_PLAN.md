# Local Mixing — Whole-Codebase Speed Optimization Plan

> **Archived / superseded.** This plan predates the source-tree reorganization
> and records assumptions that are no longer current. Paths, profile settings,
> and proposed work below are preserved as historical context, not active
> instructions. See [`src/README.md`](src/README.md) for the maintained module
> and pipeline map.

**Goal:** Make the codebase as fast as possible **without changing any observable behavior**.
Functionality preservation is the top priority; speed is second. Every change must be
proven behavior-neutral before it is kept.

**Audience:** This plan is for an autonomous coding agent (Codex) to execute. It is written
to be self-contained: it states the invariants, the order of work, the verification gates,
and the specific files/functions to touch.

**Non-negotiable rules (carry these into every step):**
- This repo's hard rule: `git pull` is the ONLY mutating git command allowed. **Never** `git
  add`, `commit`, `push`, `checkout -b`, `stash`, `reset`, or `rebase`. Leave changes in the
  working tree for the user to commit.
- The project uses a pinned toolchain (`rust-toolchain.toml`, channel `1.93.1`, edition 2024).
  Do not bump the toolchain.
- Do not change CLI surface, file formats, RNG seeding semantics, or numeric outputs of any
  command. The obfuscation pipeline is randomized but *seedable*; identical seeds must produce
  identical circuits before and after every change (see "Determinism oracle" below).

---

## 0. Context: what this code is and where time goes

Rust crate (`local_mixing`) implementing a reversible-circuit "shuffle-shoot-shuffle"
obfuscation/compression pipeline. ~15.5k LOC. Built as both a `cdylib`/`rlib` (PyO3 bindings)
and a CLI binary (`local_mixing_bin`).

Hot subsystems by size and by role (confirm with profiling in Phase 2 before trusting this):

| File | LOC | Role / why it's hot |
|---|---|---|
| `src/circuit/circuit.rs` | 2826 | Core gate/circuit/permutation types; evaluation; `to_polynomial`; canonicalization (`canonicalize_polys`, `canonicalize_polys_4`, `canonicalize_inner`); compression. |
| `src/replace/transpositions.rs` | 2584 | Transposition/identity machinery; `canonicalize`. |
| `src/replace/gadgets.rs` | 1835 | SAMF/SG/RG gadget construction & evaluation. |
| `src/replace/replace.rs` | 1418 | `compress_loop`, `compress_lmdb`, `compress_big_ancillas`; rayon fan-out; xxh3 keying; LMDB lookups. |
| `src/replace/sat_score.rs` | 956 | SAT-hardness scoring. |
| `src/random/random_data.rs` | 1170 | Random circuit generation. |
| `src/rainbow/canonical.rs` | 685 | Canonicalization with a global `Mutex<LruCache>`. |
| `src/replace/main_mix.rs` | 890 | Top-level mixing loop driving `sss`. |

Key data types (`circuit.rs`):
- `CircuitSeq { gates: Vec<[u16;3]> }` — gate = `[active, pos_ctrl, neg_ctrl]`.
- Evaluation kernels: `evaluate_index` (usize/≤64 wires), `_256` (U256), `_512` (U512),
  `_1024` (U1024 via `uint::construct_uint`). The 256/512/1024 paths use `primitive-types`/
  `uint` bignums — these are the slow lane.
- `Polynomial = Vec<u64>` (monomials), heavily allocated/collected.

The probabilistic correctness oracle already exists: `CircuitSeq::probably_equal(&other, n,
iters)` (driven by the `equal` CLI subcommand). This is the backbone of all verification below.

### Known, high-confidence opportunities (the agent should still confirm via profiling)
1. **`[profile.release] overflow-checks = true`** — every `+ - *` carries a branch+panic check
   in release. This is almost certainly the single biggest free win. Must be audited (some code
   may *rely* on panicking on overflow as a correctness assertion) then disabled or scoped.
2. **No LTO / no `codegen-units=1` / no `target-cpu`** — no release tuning at all.
3. **`std::collections::HashMap/HashSet` (SipHash)** used in ~22 sites; `rustc-hash` and
   `xxhash-rust` are already dependencies. Hot maps keyed by small integers/tuples should use
   `FxHashMap`.
4. **~168 `.collect()` + ~78 `.clone()`** — allocation churn, especially around `Polynomial`
   and `Vec<[u16;3]>` in inner loops.
5. **~149 `println!/eprintln!`** in `src/` — some are inside hot loops; unbuffered stdout +
   formatting is non-trivial overhead at scale.
6. **Bignum eval paths (U256/U512/U1024)** — used even when wire counts fit in `u64`/`u128`.
7. **`rainbow/canonical.rs`: global `Mutex<LruCache<String, _>>`** keyed by `String` — both a
   contention point and a per-lookup `String` allocation/hash.
8. **~56 `Instant::now()`** calls — confirm none are in the inner loop unconditionally.

---

## 1. Working method & guardrails (do this BEFORE any optimization)

The agent must establish a reproducible measure-and-verify harness first. Optimization without
a baseline is forbidden.

### 1.1 Determinism oracle (functionality preservation)
Behavior preservation is verified two ways; **both** must pass after every change that is kept:

**A. Golden-output determinism test.** The pipeline is seeded (`fastrand::seed`, and `rand`).
Build a script `scripts/golden_check.sh` that:
- Runs a fixed battery of commands with fixed seeds and fixed inputs into a temp dir, e.g.:
  - `genran` → small/medium/large circuits at a few `(n,m)`.
  - `sss` with a representative flag matrix (at least: plain; `--feistalize`;
    `--gadgetize`; `--full-shuffle`; `--expansion_game`; `--single-end`) on each generated
    circuit, fixed `-n -m -x -r` and fixed `--source`.
  - `compress`, `shuffle`, `shoot`, `evaluate`.
- Captures the **exact output circuit files** and stdout, hashes them (`sha256`), and writes a
  manifest `golden/<commit>.sha256`.
- Re-running the script must reproduce identical hashes.

  ⚠️ First confirm the pipeline is actually deterministic under fixed seeds. If any command
  pulls entropy from a non-seeded source (system RNG, time, thread-scheduling-dependent rayon
  reductions), that command is verified with oracle **B** only, and the non-determinism source
  must be documented (do NOT "fix" it as part of this work — it may be intentional).

**B. Functional-equivalence oracle.** For every `(input_circuit, output_circuit)` produced by
the battery, run `local_mixing_bin equal -n <wires> -i <iters> -a in -b out` — but note the
pipeline *intentionally* transforms circuits, so equivalence is only expected where the command
is supposed to be functionality-preserving (e.g. `compress`, `shuffle`, identity insertions).
For transforming commands, the invariant is "before vs after the optimization, same seed ⇒ byte-
identical output" (oracle A), not equivalence to the input.

Concretely: **oracle A is the primary gate** (same seed ⇒ identical bytes, pre vs post change).
Oracle B is the secondary gate for the subset of operations that are equivalence-preserving by
design, and as a sanity net that the canonicalizer/compressor still produces equivalent circuits.

### 1.2 Benchmark harness (speed measurement)
- Use the existing `BENCH_CANON` / `BENCH_N` / `BENCH_MAX_N` env hooks and the `bench_canon4` /
  `bench_polycanon` bins (`src/bench/`) as a starting point.
- Add a top-level wall-clock benchmark script `benchmarks/bench_pipeline.sh` that runs the same
  battery as the golden script (without hashing) under `hyperfine` (or `/usr/bin/time -v` if
  hyperfine is unavailable), 3 warmup + 10 measured runs, and records median wall time +
  max-RSS per command into `reports/bench_baseline.json`.
- Pick at least one **large** representative config that dominates real runtime (the `rantestn128m800` /
  `rantestn128m900` dirs and `mixing_tests/` suggest n≈128, m≈800–1000 is the real workload —
  use one of those as the headline benchmark).
- Record the baseline commit hash and `rustc -vV` / CPU model in the report.

### 1.3 Profiling
- Build a profiling profile (inherits release + debuginfo) and profile the headline benchmark
  with `perf record`/`perf report` (or `samply` for flamegraphs; or `cargo flamegraph` if the
  tooling is present). If none are installed, fall back to `perf stat` + targeted
  `Instant`-based timing already present in the code (`CANON4_CORE_TIME`, `COMPRESSION_TRACE`).
- Produce `reports/profile_baseline.md`: top 20 functions by self-time, top allocation sites,
  cache-miss / branch-miss summary from `perf stat`, and an explicit hot-path ranking. **All
  later algorithmic work is prioritized by this file, not by guesswork.**

### 1.4 Per-change protocol (apply to EVERY optimization commit-candidate)
1. State the hypothesis ("X is hot because profile line N; change Y should cut Z").
2. Make the smallest change that tests the hypothesis.
3. `cargo build --release` must succeed with **zero new warnings**.
4. Run `scripts/golden_check.sh` → hashes must match baseline (oracle A). If a hash changes,
   the change altered behavior → **revert** unless the change is in the explicitly-allowed
   "behavior may change" set (only: logging/formatting wording, and only with user note).
5. Run `cargo test` (the `tests/poly_canon_*` suite + unit tests in `circuit.rs`) → all pass.
6. Run `benchmarks/bench_pipeline.sh` → record delta. Keep only if neutral-or-faster on the
   headline benchmark and no individual command regresses >3%.
7. Append a one-line entry to `reports/optimization_log.md`:
   `<area> | <hypothesis> | <Δ median wall> | <Δ max-RSS> | kept/reverted`.

---

## 2. Phased optimization roadmap

Phases are ordered cheapest-and-safest → most-invasive. **Do not** start a later phase until
the harness (Phase 1) is green and the profile exists. Within a phase, re-run the per-change
protocol per item.

### Phase A — Build & compiler configuration (cheapest, highest leverage)
Do these first; each is a few lines and individually measured.

A1. **Audit overflow reliance, then tune `[profile.release]`.**
   - Grep for arithmetic that could legitimately overflow and is used as an assertion. The
     evaluation kernels use only `^ & << >>` (unaffected by overflow-checks), so the risk is
     confined to counters, indices, and size math. Check `random_data.rs`, `main_mix.rs`,
     `replace.rs`, `sat_score.rs` for `+`/`*` on `usize`/`u64` that could exceed bounds at the
     n≈128/m≈1000 scale.
   - If audit is clean, set:
     ```toml
     [profile.release]
     overflow-checks = false      # default; remove the explicit true after audit
     lto = "thin"                 # try "fat" too; measure both
     codegen-units = 1            # measure; trades build time for runtime
     panic = "abort"              # ONLY if no code relies on unwinding; PyO3 cdylib may need unwind — verify first
     ```
     Treat each line as a **separate** measured change (A1a overflow, A1b lto, A1c codegen-units,
     A1d panic). `panic = "abort"` is risky with the `cdylib`/PyO3 target and `ctrlc`/`signal-hook`
     handlers — verify the Python extension still imports and the CLI still handles Ctrl-C before
     keeping it; if in doubt, skip it.
   - Keep a `[profile.bench]`/`[profile.profiling]` with `debug = true` for future profiling.

A2. **`target-cpu`.** Add `.cargo/config.toml`:
   ```toml
   [build]
   rustflags = ["-C", "target-cpu=native"]
   ```
   ⚠️ `target-cpu=native` makes binaries non-portable. The remote server (cc@129.114.109.44) may
   have a different CPU than the dev box. Confirm with the user whether builds are per-machine.
   If binaries are shipped between machines, use a conservative baseline (e.g. `x86-64-v3`)
   instead, or gate native behind an env/profile. **Ask before committing this one.**

A3. **PyO3 / dependency feature audit.** Confirm no debug-heavy features are enabled. Ensure
   `rand`/`fastrand` aren't pulling unused features. Low priority; measure only if profile shows
   dependency hot spots.

### Phase B — Allocation & hashing (broad, mechanical, low-risk)
Prioritize by the allocation-site ranking from `reports/profile_baseline.md`.

B1. **Swap SipHash maps for `FxHashMap`/`FxHashSet`** at integer/tuple-keyed hot sites only
   (~22 candidate sites). `rustc-hash` is already a dependency. Do NOT change maps keyed by
   attacker-controlled or security-sensitive data (none expected here, but check). Each swap is
   behavior-neutral for lookups but changes iteration order — **verify no code depends on
   HashMap iteration order** (grep for `for .. in <map>` then collect/serialize). If iteration
   order leaks into output, that map must keep a deterministic type (`BTreeMap`) — note it.

B2. **Kill allocation churn in inner loops.** Using the profile's top allocation sites
   (expected: `Polynomial`/`Vec<u64>` in `to_polynomial`/canonicalization, and `Vec<[u16;3]>`
   clones in `replace.rs`/`gadgets.rs`):
   - Reuse scratch buffers (`Vec::clear()` + reuse) instead of re-allocating per iteration.
   - Replace `.collect()` into a fresh `Vec` followed by immediate consumption with iterator
     chains where possible.
   - Use `SmallVec` (already a dependency) for the many tiny `[u16;3]`/short-gate vectors where
     length is usually small and known-bounded.
   - Replace `.clone()` of large `CircuitSeq`/`Polynomial` with borrows or `Cow` where the clone
     is defensive and not mutated.
   - `String`-keyed `LruCache` in `rainbow/canonical.rs`: replace the `String` key with a cheap
     integer/`u128` hash key (xxh3 is already used elsewhere) to drop per-lookup allocation; and
     evaluate whether the global `Mutex` can become a sharded/`DashMap` cache (DashMap is already
     a dep) to cut contention. Verify cache semantics (eviction, hit-rate) are preserved with a
     before/after hit-rate counter.

B3. **Buffer and gate logging.** Wrap stdout in a `BufWriter` for bulk output paths; gate
   per-iteration `println!/eprintln!` behind an existing verbosity/env flag (the code already
   uses `COMPRESSION_TRACE`, `BENCH_CANON` patterns — reuse that idiom). Behavior note: this
   *can* change stdout interleaving/wording → only acceptable if the golden script's hashed
   stdout is updated deliberately and the user is told. Safer default: keep output identical,
   just buffer it.

### Phase C — Numeric & kernel optimization (medium risk, needs careful equivalence)
C1. **Right-size the evaluation kernel.** Dispatch to the narrowest integer type for the wire
   count: `u64` for n≤64, `u128` for n≤128, and only fall to U256/U512/U1024 when truly needed.
   The headline workload is n≈128 → a native `u128` path (`evaluate_index_128`) would replace
   the U256 bignum path for the dominant case. This is the highest-value algorithmic kernel win
   if the profile confirms eval dominates.
   - Implement `evaluate_index_128` mirroring `evaluate_index_256` exactly, add a dispatcher
     that picks the kernel by `n`, and prove bit-exact equality against the existing 256 path
     over a large random battery (add a `#[test]` that fuzzes `state`/`gate` and asserts the
     128 and 256 results agree for n≤128).
C2. **`probably_equal` / evaluation batching.** If the oracle itself is a runtime cost (it runs
   inside `--equality_check`), vectorize the per-input loop and ensure it's the rayon path
   (`circuit.rs:602/623` already parallelize) with good chunking.
C3. **Canonicalization hot loop** (`canonicalize_polys_4`, `canonicalize_inner`, transpositions
   `canonicalize`). Only after the profile confirms it's hot: look for repeated re-sorting,
   redundant `to_polynomial` recomputation, and `HashMap`-per-call patterns that could be
   precomputed once. These are the most algorithmically subtle — change one rule at a time,
   guarded by the `tests/poly_canon_*` suite which exists specifically for this.

### Phase D — Parallelism & scheduling (medium risk)
D1. **Rayon chunking review.** `replace.rs` uses `4 * current_num_threads()` chunking
   (lines ~262/381). Validate chunk sizes against the profile; oversized chunks underutilize,
   undersized chunks thrash. Tune empirically on the headline benchmark.
D2. **Confirm no accidental serialization.** The global `Mutex<LruCache>` (B2) and any
   `DashMap` write contention (`COMPRESSION_HISTOGRAM` etc.) can serialize parallel work — check
   with `perf` whether threads stall on locks; shard or make per-thread+merge if so.
D3. **Determinism caution.** Any change that alters reduction order across threads can change
   output bytes (oracle A) even while preserving equivalence (oracle B). For seed-deterministic
   commands, parallel reductions must be order-stable (collect-then-reduce in index order) — do
   NOT trade determinism for speed unless the user explicitly allows it.

### Phase E — Micro-optimization & cleanup (lowest leverage, do last)
- `#[inline]` tuning on tiny hot fns flagged by the profile (don't carpet-bomb `inline(always)`).
- Bounds-check elision via iterators / `get_unchecked` **only** in proven-hot, proven-safe inner
  loops, each with a comment proving the invariant. Prefer safe iterator forms first.
- Remove dead `Instant::now()`/timing in hot paths or gate behind the bench env flag.
- Strip the binary (`strip = true` in release profile) for faster load if relevant.

---

## 3. Suggested subagent / parallel-workstream decomposition

If Codex supports parallel sub-agents, split along these **independent, non-overlapping file
boundaries** to avoid merge conflicts. Each sub-agent owns its files end-to-end and runs the
full per-change protocol (§1.4) on its own slice. A single **integrator** agent owns the shared
harness and the merge/verify gate.

| Sub-agent | Owns (files) | Scope |
|---|---|---|
| **harness/integrator** (do first, blocks others) | `scripts/golden_check.sh`, `benchmarks/bench_pipeline.sh`, `reports/*`, `Cargo.toml` profile, `.cargo/config.toml` | Builds determinism oracle + bench + profile (Phase 1, A). Owns all config changes (A1–A3). Final merge & full-battery verification. |
| **build-config** | (folded into integrator — config is too cross-cutting to parallelize) | Phase A. |
| **alloc-hashing** | `replace/replace.rs`, `replace/pairs.rs`, `replace/gadgets.rs`, `rainbow/canonical.rs` | Phase B (Fx maps, scratch buffers, LRU key, SmallVec). |
| **numeric-kernels** | `circuit/circuit.rs` (eval kernels + dispatch), `circuit/poly_canon_graph.rs` | Phase C1/C2. |
| **canonicalization** | `circuit/circuit.rs` (canon fns), `replace/transpositions.rs` | Phase C3 — guarded by `tests/poly_canon_*`. |
| **parallelism** | rayon call-sites across `replace/replace.rs`, `lib.rs`, `random/random_data.rs` | Phase D. |

⚠️ `circuit/circuit.rs` is touched by both **numeric-kernels** and **canonicalization**. Either
serialize those two agents, or split `circuit.rs` mentally into "eval kernels (lines ~90–260,
the `Gate` impl)" vs "canonicalization (lines ~1374+)" and assign disjoint line ranges. The
integrator resolves any overlap. **Do not run two agents writing `circuit.rs` concurrently
without a line-range contract.**

A separate **profiling agent** can re-profile after each phase and keep `reports/profile_*.md`
current so later phases retarget the now-hottest code (hot paths shift as earlier wins land).

---

## 4. Verification gate (definition of done)

A change ships (stays in the working tree) only if ALL hold:
1. `cargo build --release` clean, no new warnings.
2. `cargo test` green (incl. `tests/poly_canon_failure_case`, `poly_canon_graph`,
   `poly_canon_stress`, and `circuit.rs` unit tests).
3. Golden determinism oracle (A): every battery output byte-identical to baseline (or, for any
   intentionally-changed output, explicitly approved by the user and re-baselined).
4. Functional-equivalence oracle (B): equivalence-preserving ops still report equal at high
   iteration count.
5. Python extension still imports and a smoke test of the PyO3 path runs (the crate is a
   `cdylib`; profile/panic changes can break it).
6. Headline benchmark median wall-time is **≤ baseline** and no command regressed >3%; max-RSS
   not significantly worse (note any RSS/throughput trade-offs for the user to decide).
7. `reports/optimization_log.md` updated.

**Overall acceptance:** a cumulative speedup report (`reports/final_summary.md`) with per-phase
contribution, the final vs baseline numbers on the headline benchmark, confirmation that all
golden hashes are reproduced, and a list of anything deferred or needing user decision
(`target-cpu`, `panic=abort`, any logging/wording changes).

---

## 5. Explicit "ask the user before doing" list
- `target-cpu=native` vs portable baseline (A2) — depends on whether binaries move between the
  dev box and the remote server.
- `panic = "abort"` (A1d) — interaction with PyO3 `cdylib`, `ctrlc`, `signal-hook`.
- Any change that alters stdout wording/format (would change golden hashes).
- Any change that trades cross-thread determinism for speed (would change seeded outputs).
- Disabling `overflow-checks` if the audit finds any site that relies on the panic as a
  correctness guard.

---

## 6. Quick-start checklist for the executing agent
1. Read this file fully. Confirm the no-`git`-mutation rule.
2. Build & confirm baseline: `cargo build --release`, `cargo test`.
3. Build the harness (§1.1–1.3); capture baselines into `reports/`.
4. Phase A (config), one line at a time, measured.
5. Re-profile. Phase B. Re-profile. Phase C. Re-profile. Phase D. Phase E.
6. Produce `reports/final_summary.md`. Leave everything in the working tree; do not commit.
