# `ssg` — Generation-Mixing Obfuscation: Parameters & Environment Variables

`ssg` is the generation-mixing variant of `sss` (shuffle-shoot-shuffle). It takes a
source circuit, optionally gadgetizes/feistalizes it, then runs shooting-game
expansion + SAMF hiding + compression, writing the mixed circuit to `--destination`.

There are two scheduling modes:
- **Fixed rounds** (default): run `--rounds` rounds; per-round files are
  `<destination-stem>round<k>.txt`.
- **Stage D — size-threshold cadence** (`--grow-threshold > 0`): ignore `--rounds` and
  instead alternate shoot/compress *stages* until the min-gen condition is met. Each
  stage shoots until the circuit grows by a set factor, then compresses back down; per-stage
  files are `<destination-stem>stage<k>.txt`. See §2b.

```
cargo run --release -- ssg -n <wires> -m <gates> -x <..> -s <src> -d <dst> -r <rounds> [flags]
```

Binary: `target/release/local_mixing_bin ssg ...`

---

## 0. What's new in this version

**`ssg` vs. plain `sss`** — the generation-mixing changes that define this branch:

- **Generation tags.** Every gate carries a *generation*; a replacement's new gates are tagged
  `floor(median(removed-window generations)) + 1`. The mix is driven by these generations rather
  than a fixed pass count.
- **Min-gen-driven shooting (Stage B).** Shoot until `--min-gen-fraction` of gates reach
  `--min-gen`, using min-gen-anchored, length-bounded passes (`--pass-length`, `--max-passes`).
- **Pluggable selection ranking.** Outgoing windows (#11) and incoming replacements (#9) are chosen
  by a `Ranker` — built-in Pareto/fanout heuristics, or a runtime Rhai script
  (`--outgoing-rank-script` / `--incoming-rank-script`), no recompile.
- **Global SAMF propagation (Stage C).** Partial/bidirectional passes keep a segmented circuit + a
  SAMF ledger so the body stays consistent; the net relabeling is undone once before compression.
- **Generation-aware extras:** `CENTRALIZE` (central-band bidirectional passes), feistelization
  refinements, `SymmetricCD`/`SymmetricG`, fanout/leeway shaping (`--max-fanout`,
  `--min-median-leeway`), and the `>64`-wire canonicalization guard (fixes an occasional
  equivalence break).

**Newest additions (this version, vs. the earlier `ssg`):**

- **Stage D — size-threshold compression cadence** (`--grow-threshold`, `--compress-fraction`):
  schedule the mix by circuit *size* instead of a fixed `--rounds` count, stopping on the min-gen
  condition, saving each compression stage. See §2b.
- **`compress --target-fraction`:** the standalone `compress` subcommand can stop early at a chosen
  fraction of the initial size.
- **Fraction-based no-progress guard:** Stage D's stall detector tracks the *fraction* of
  sub-min-gen gates on the compressed circuit (not an absolute count), so it no longer false-trips
  as the circuit grows.
- **Read-only DB open:** the curated LMDB is opened `READ_ONLY | NO_LOCK`, so a run can share a DB
  owned by another user.

---

## 1. Core CLI parameters (shared with `sss`, via `add_shoot_args`)

| Flag | Req | Default | Effect |
|------|-----|---------|--------|
| `-n, --n <usize>` | **yes** | — | Number of wires in the source circuit. |
| `-m, --m <usize>` | **yes** | — | Plain-SAMF count: SAMFs inserted across the whole circuit per shooting round (`m=0` disables plain-SAMF insertion). |
| `-x, --x <usize>` | **yes** | — | Shooting/expansion intensity parameter (per-pass expansion budget). |
| `-s, --source <path>` | **yes** | — | Path to the source circuit file. |
| `-d, --destination <path>` | **yes** | — | Output circuit path. Per-round files written as `<stem>round<k>.txt`. |
| `-r, --rounds <usize>` | **yes** | — | Number of mixing rounds. |

### Front-end transforms (applied once, before the main loop)

| Flag | Effect |
|------|--------|
| `--gadgetize` | Gadgetize at start; input grows to **2n** wires. |
| `--feistalize` | Feistalize at start; input grows to **3n** wires (the production mode for the n=128 runs). |
| `--interleave` | Use interleaving. |
| `--slice-zero` | Before feistalization, insert M preserving the `(y,z)=0` slice and randomizing x off it. |
| `--slice-zero-random` | Same, preserving a public **random** `(y,z)` slice. |
| `--slice-zero-hardcoded` | Insert hardcoded M preserving the `(y,z)=0` slice. |
| `--slice-zero-random-gates <n>` | # random M gates for `--slice-zero-random` (default `32n`). |
| `--slice-zero-hardcoded-rounds <n>` | # hardcoded M rounds for `--slice-zero-hardcoded` (default `1`). |
| `--gadget_path <path>` | Where to write the gadgetized/feistalized circuit (default `./gadgetized/<source filename>`). |

### Shooting / SAMF-hiding controls

| Flag | Default | Effect |
|------|---------|--------|
| `--shooting_times <n>` | `1` | # shooting rounds; each = one collision game + one plain-SAMF insertion, then final unsamf. |
| `--type_attempts <n>` | `1` | Distinct SAMF gate types tried per collision before giving up. |
| `--rg-frequency <n>` | `2` | SG gadgets between each RG gadget (2 = two SGs then one RG). |
| `--gates_ahead_expand <n>` | `2` | Gates per curated **expansion** window, anchored at the colliding pair (>2 shrinks by 1 on a curated-DB miss, down to the pair). |
| `--gates_ahead_samf <n>` | `3` | Context gates prepended to the 3 SAMF gates when hiding a SAMF. |
| `--full-shuffle` | off | Insert n SAMFs between every gate after each round's shooting insertion, before compression. |
| `--full-shuffle-early` | off | Insert n SAMFs between every gate once, after gadgetization/feistalization, before the main loop. |
| `--egg` | off | Use expansion game (`expand_loop` 2×) instead of the shuffled shooting game. |
| `--single-end` | off | Accumulate SAMFs/NOTs across **all** rounds (functionality broken between rounds) and undo in a single pass after the last round, before its compression. |

### Compression / bookkeeping

| Flag | Default | Effect |
|------|---------|--------|
| `-l, --light-compression` | off | Between rounds, stop compressing once circuit ≤ half its max (post-shooting) size. |
| `--equality_check` | off | Run probabilistic equality/functionality checks after each round and at the end. |
| `--record` | off | Record every expansion/compression replacement to `<destination>.replacements`. |
| `--track-survivors` | off | Record pre-mixing gates never part of any replacement to `<destination>.survivors`. |

---

## 2. `ssg`-specific parameters (generation/fanout selection)

| Flag | Default | Effect |
|------|---------|--------|
| `--max-fanout <usize>` | `50` | Hard cap on per-gate fanout (gen mode). |
| `--min-median-leeway <usize>` | `10` | Raise low-leeway gates when median leeway < this (gen mode). |
| `--min-gen <usize>` | `1` | **Stage B**: keep shooting passes until every gate's *generation* ≥ this. |
| `--min-gen-fraction <f64>` | `0.99` | **Stage B**: stop once this fraction of gates reach `--min-gen`. |
| `--pass-length <usize>` | `0` | **Stage B**: max successful replacements per shooting pass (`0` = unbounded). |
| `--max-passes <usize>` | `100000` | **Stage B**: safety cap on shooting passes per round. |
| `--samf-target <usize>` | `0` | If a round hides ≥ this many SAMFs, skip plain-SAMF insertion (`m→0`) for later rounds (`0` = disabled). |
| `--outgoing-rank-script <path>` | — | Rhai script providing `rank(cands)` for outgoing window selection (#11). |
| `--incoming-rank-script <path>` | — | Rhai script providing `rank(cands)` for incoming replacement selection (#9). |

---

## 2b. Stage D — size-threshold compression cadence

Instead of a fixed `--rounds` count, Stage D drives the mix by **circuit size** and stops on the
**min-gen condition**. When `--grow-threshold > 0`, `--rounds` is ignored and the run becomes a
loop of *stages*: shoot until the working circuit is `--grow-threshold` percent larger than the
size at the end of the previous compression, then compress (down to `--compress-fraction` of the
post-shooting size, or fully). The loop stops once `--min-gen-fraction` of gates reach `--min-gen`.
Each stage's compressed circuit is saved to `<destination-stem>stage<k>.txt`.

| Flag | Default | Effect |
|------|---------|--------|
| `--grow-threshold <f64>` | `0` | Percent growth per stage that triggers compression. `0` = off (use fixed `--rounds`). E.g. `100` = shoot until the circuit doubles relative to the previous compressed size. |
| `--compress-fraction <f64>` | `0` | Stage D only: compress each stage down to this fraction of the **post-shooting** size, instead of fully. E.g. with `--grow-threshold 100`, `0.55` nets ≈ +10% size per round; `0.60` ≈ +20%. `0` = compress fully each stage. |

Notes:
- **Stop rule** = the min-gen condition, evaluated on the *compressed* circuit as the **fraction**
  of gates below `--min-gen` (robust to the circuit growing each stage). A no-progress guard stops
  the cadence if that fraction fails to improve for 8 consecutive stages.
- The size cap is checked **per shooting pass**, so a stage can overshoot the target by up to one
  pass's growth. Use a smaller `--pass-length` for tighter size control.
- Large feistalized circuits are nearly incompressible, so `--compress-fraction` may be unreachable;
  each stage's compression then ends on the no-progress (`STABLE_MAX`) stop instead. Lowering
  `STABLE_MAX` (e.g. to `3`) and/or raising `--compress-fraction` (e.g. to `0.60`) speeds this up.

Example (no feistalize, ~10%/round growth, stop at min-gen 10 over 99% of gates):
```
local_mixing_bin ssg -n 128 -m 0 -x 10 -s src.txt -d mixed.txt -r 1 \
    --min-gen 10 --min-gen-fraction 0.99 --pass-length 100 \
    --grow-threshold 100 --compress-fraction 0.55 \
    --outgoing-rank-script rank/outgoing_pareto.rhai \
    --incoming-rank-script rank/incoming_fanout.rhai --equality_check
```

### Standalone compression with an early-stop target (`compress` subcommand)

The `compress` subcommand now accepts `--target-fraction <f>`: stop compressing early once the
circuit reaches that fraction of its **initial** size (in addition to the usual no-progress stop).
```
local_mixing_bin compress -n <wires> -s in.txt -d out.txt --target-fraction 0.5
```

---

## 3. Environment variables

### Correctness / debugging

| Var | Effect |
|-----|--------|
| `STAGEC_CHECK` | (set = on) Enables Stage-C equivalence instrumentation: per-pass, per-flush, and end-of-Stage-C invariant checks that materialize the current ledger state and compare (sampled, U1024) against the input. On a break it logs which pass/flush first corrupted the global function and `exit(8/9)`. **Off by default** (non-perturbing only when on; used to localize the occasional equivalence-failure bug). |
| `ABSORB_NOTS` | (set = on) Enables NOT-absorption (#10 / Stage F) — absorb pending NOTs into the curated lookup instead of emitting NOT gadgets. Off by default (was gated off due to a latent correctness bug). |
| `COMPRESSION_TRACE` | (set = on) Emit a per-replacement compression trace. |
| `COMPRESSION_TRACE_MS <ms>` | Throttle the compression trace to at most one line per `<ms>` milliseconds. |
| `SURVIVOR_LOG_EVERY <n>` | Log survivor stats every `n` events. |

### Selection / shuffle behavior

| Var | Effect |
|-----|--------|
| `CENTRALIZE <pct>` | Bias window/anchor selection toward the circuit center by `<pct>` percent. |
| `STABLE_MAX <n>` | Cap on consecutive "stable" (no-progress) iterations before early-stopping a pass. |
| `SymmetricCD` | (set = on) Use symmetric C/D construction in gadget selection. |
| `SymmetricG` | (set = on) Use symmetric G (only takes effect together with `SymmetricCD`). |

### SAT-probe scoring (advanced; tunes which collisions/windows are chosen via a SAT solver)

| Var | Effect |
|-----|--------|
| `SAT_PROBE_SOLVER` | Solver binary for SAT probes (default `kissat`). |
| `SAT_PROBE_FREQUENCY <n>` | How often to run a SAT probe (every n-th candidate/pass). |
| `SAT_PROBE_WINDOW_GATES <n>` | Window size (in gates) handed to the SAT probe. |
| `SAT_HIDDEN_SAMF_CANDIDATES <n>` | # hidden-SAMF candidates the probe scores per collision. |
| `SAT_CONE_MIN_FRACTION <f>` | Minimum SAT cone fraction for a candidate to qualify. |
| `SAT_BCP_MIN_RESISTANCE <n>` | Minimum BCP "resistance" threshold for scoring. |
| `SAT_EXPAND_MIN_DELTA <n>` | Minimum gate-delta for an expansion to be accepted by the probe. |
| `SAT_COMPRESS_PRESERVE_DELTA <n>` | Compression delta the probe tries to preserve. |
| `SAT_SCORE_SEED <n>` | RNG seed for SAT-probe scoring (reproducibility). |
| `SAT_SCORE_SLACK <n>` | Slack allowed around the best SAT score when selecting. |

### Benchmark-only (not used in production runs)

`BENCH_CANON`, `BENCH_N`, `BENCH_MAX_N` — enable/parameterize canonicalization micro-benchmarks.

---

## 4. End-of-run diagnostics

- `Oversized-canon (>64-wire) lookups skipped: <N>` — count of replacement windows whose
  canonicalization was skipped because they touched >64 distinct wires (the `Monomial = u64`
  overflow guard added to fix the occasional equivalence break). A nonzero value is normal and
  safe; it means those windows were left unchanged rather than risking a wrong DB match.
- `Final len: <N>` — final gate count.

---

## 5. Notes

- **Read-only DB.** `sss`/`ssg`/`compress` open the curated replacement LMDB with
  `READ_ONLY | NO_LOCK` (the mixing path only reads it), so a run can share a DB owned by another
  user without write access to the lock file.
- **Slice metadata.** With `--slice-zero-random`, the public `(y,z)` slice is printed to the log
  (`slice_zero_random public slice: y=0x… z=0x…`) and written to
  `<gadget_path>.slice_zero_random`. Record `y`/`z` alongside the mixed circuit (e.g. as a
  `# y=0x…` / `# z=0x…` header) so the preserved slice is reproducible.
- **Equality.** Pass `--equality_check` to verify functional equivalence after each stage/round and
  at the end (plain circuits via direct sampling; feistalized via the preserved middle block). A
  mismatch panics with `The functionality has changed`.
