# Stage 4 (packing) in Rust: design

Status: design revision 2 (2026-10-08), **implemented** (P0.5-P9, 2026-10-09; results in §12), branch `ran/stage4-rust` (based on `492564f0`). Owner: RC.

> **2026-10-11:** packing is now **stage 6** of a seven-stage pipeline (4 splitting, 5 crossing walk, 7 unpacking; `docs/tdp_pipeline.md`). This document keeps the four-stage numbering; the driver facts it states (config v2, recipe v10, SEEDS format 2, `stage4.failed`) are now config v3, recipe v11, SEEDS format 3 (five seeds) and `packing.failed`, and the chain it reproduces is `run.skip_stages = [4, 5, 7]`.

This document does two things:

- It specifies how stage 4 of the TDP pipeline is ported to Rust.
- It specifies how the managed driver changes so that one managed run with **default settings** reproduces the latest chosen Python runs (vL64a/b, vL128a/b, CONFIG=L8). The output must be byte-for-byte what the Python reference chain produces from the same seeds.

Revision 2 addresses the 37 review issues on revision 1. Appendix C lists each issue and how it is resolved.

Sources of truth, in order of precedence:

1. The owner's requirements, restated in §1.
2. The Python/bash oracle in `scripts/windowed_delivery/`. These files are byte-identical to the run-tree copies, except that `val_chain.sh` drops the inert `--p-comp/--p-any` (§2.1). Abbreviations used in this document:
   - `WP` = `window_pack3.py`;
   - `AC` = `anf_compose.py`;
   - `W/` = `waist/`, which holds `order_repair.py`, `merge_bounds2.py`, `anf_split2.py`, `verify_equiv.py`, `anf_dump_eval.py` (`DE`), `link_census.py`, `mult_census.py`, `repair.sh` and `val_chain.sh`;
   - `HA` = `security_tests/heatmaps/hmap_affine.rs`.

   **The oracle is never edited.** Every run of it, in fixtures, CI or reproduction, uses a path-rewritten sandbox copy.
3. The validation runs `runs/ncdb_ab_20261001/waist/{vL64a,vL64b,vL128a,vL128b}` (CONFIG=L8) and the matching `vA*` runs. They are read-only. Every default in §2 was checked against their `seed.txt`, `gen.log` and `s3.log`; where docs disagree with these files, the files win.

Everything not marked "deviation" or "decision" is meant to be bit-exact.

---

## 1. Requirements

**Goal (RC).** Stage 4 becomes a Rust production stage, integrated into the TDP driver (`scripts/tdp_gen.sh` and `src/tdp`). It must be faithful to the Python oracle, bit-exact wherever the oracle is deterministic, and validated on real artifacts.

**Defaults requirement (RC, 2026-10-08): "the default in the rust driver should match the latest chosen runs in python".** A managed run that sets nothing but `source.wires` (64 or 128) and its seeds must reproduce `W/val_chain.sh TAG N MC S SLICE L8` exactly as it was run for vL64a, vL64b, vL128a and vL128b. That covers:

- the source size: `|C| = |D| = 1000` at n=64 and `3000` at n=128 (§2.4);
- stage 1+2 argv and the output-affecting environment;
- stage 3 argv and the output-affecting environment, run serially;
- stage 4: pack `512 512 24576 3000`; repair as `repair.sh` (4 fix iterations, 4 fallback iterations, ANF budget 65536); serialization as `DE`.

"Output-affecting environment" is defined by the environment registry (§11.5). Variables classified output-neutral (the three cache sizes) may differ from the reference. No dedicated test proves their neutrality; the evidence is stage-3 bytes identical to the reference runs (D10b).

Further requirements:

- Config A is selectable by one option.
- Gen and mix seeds are random per run and recorded privately. Explicit seeds are accepted for calibration and reproduction only.
- Old configurations cannot silently produce a hybrid recipe (§11.2: config_version 2, authoritative presets).
- No ambient environment variable can change a managed run's bytes (§11.5).

**Acceptance.**

- **Historic anchor (required, §10.8).** For vL64a and vL128a, the managed driver with the archived seeds reproduces the archived `tdp.mpmct1`, `s3.mpmct1`, `packed_G512*`, `rep512.*` and `deliverable.bin` (payload). If a binary change since Oct 7 breaks this, the break is documented and the new reference is defined explicitly (§10.8).
- **Reproduction test (§10.7).** The managed driver and a sandboxed `val_chain.sh` (L8 and A) with fresh seeds produce byte-identical stage-2, stage-3, packed, repaired and payload artifacts. Exits 5 and 6 are handled as in §10.7.
- **CI-sized checks (§10.9)**, including a CI differential test of the Python oracle against the Rust port (§10.6).
- Gates G1-G7, `cargo fmt --check` and the other CI checks in `.github/workflows/ci.yml`.

---

## 2. The reference recipe, verified, and how today's driver differs

### 2.1 What `val_chain.sh TAG N MC S SLICE L8` ran

Verified against the script, `seed.txt` and the gen/s3 logs of all four L8 runs.

| item | value in the reference runs | evidence |
|---|---|---|
| gen argv | `tdp.mpmct1 N MC MC S 1 SLICE SEED SEED SEED`. There is no mode positional (the default is `embedded-masking`) and no variant positional (the default is `classic`) | `val_chain.sh:15-16`; gen.log `seed=X gadget_seed=X sandwich_seed=X` |
| sizes | n=64: MC=1000, S=384, SLICE=1280. n=128: MC=3000, S=896, SLICE=2560. S = `sandwich_default_s(n)` = `max(n, round(n·log2 n))` and SLICE = `10·2n`; these equal the driver's derivation (`tdp_gen.sh:336-341`). MC was passed explicitly and is **not** the formula (2304 / 6272) | gen.log line 1 |
| stage-2 environment | exactly `EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8`. Every other `EMBEDDED_MASKING_*` variable is unset, so gen uses its base `EmbeddedMaskingParams::managed_tdp(gadget_seed, n)` (K=2, max_open=3, min_open=2, balanced, quad_fire) | `val_chain.sh:13,15`; gen `mod.rs:240-260` |
| gen seed | `secrets.randbelow(2**31)` | `val_chain.sh:11` |
| mix seed | `secrets.randbelow(2**40)` | `val_chain.sh:12` |
| stage-3 argv | `--input tdp.mpmct1 --output s3.mpmct1 --tdp --db-mixing --profile 2,2,2,2,2 --moves int(G*2*2*1.3) --p-db 1.0 --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 --no-local-verify --seed MSEED`. G is field 3 of the `tdp.mpmct1` header, and `int` truncates. The runs also passed `--p-comp 1.0 --p-any 0.1`, which equal the CLI defaults and are inert at `p_db=1.0`. They were removed on this branch (TDP doc §3.2), and the tracked `val_chain.sh` omits them | `val_chain.sh:17-23`; s3.log `profile ON: n=[2.0, 2.0, 2.0] r=[2.0, 2.0]`, `moves=4051814` |
| stage-3 environment | `FROZEN_DB_DIR=<regular store> FROZEN_FILTER=1`, and nothing else from the recipe: `CANON_*` caps unset (unbounded), cache sizes at their built-in defaults, `--verify-every` and `--report-every` at their CLI defaults (10 000 and 50 000). The local store has no `filters.bin`, so the filter was off. The interactive shell's other variables were inherited and are unknown; the historic anchor (§10.8) is what proves they were inert | s3.log `FROZEN_FILTER=1 but no filters.bin ... running unfiltered`; reports every 50 000 moves |
| stage-3 mode | serial: no `--parallel-*` | `val_chain.sh` |
| side reports | `mult_census.py tdp.mpmct1` → `mult_s2.txt`; `hmap_affine` SPAN on stage 2 (`SPAN_OUTPUTS_TRIVIAL=1`, `--g-step 64` at n=64 or `256` otherwise) → `span_s2.{tsv,txt}`. Both are report-only | `val_chain.sh:24-27` |
| stage 4 | `window_pack3.py tdp s3 512 512 24576 3000 TAG-G512 packed_G512.mpmct1`, then `repair.sh TAG 512 N`, then `anf_dump_eval.py rep512.mpmct1 rep512.bounds deliverable.bin TAG` | `val_chain.sh:29-31` |
| repair version | All eight replicates were re-repaired on Oct 8 with the canonical `order_repair.py` (breaker located before the completer moves) and re-serialized. The earlier repairs are in `rep512_v1/`, so TDP doc §4.7 is stale. The parity targets are the current `rep512.*` and `deliverable.bin` (Appendix B), subject to the binary-provenance check of §10.5 step 0 | file times; `repair512.log` |
| binaries | stage-1/2/3 artifacts are timestamped 2026-10-07 23:23 (gen) to 2026-10-08 01:01 (vL128a s3), three minutes after commit `e720d5c0` (23:20). The stage-2 fix `d9631d0e` (Oct 8 18:54) and the stage-3 cleanup commits `8070f5a0..b5608cd9` (Oct 8) came **after** the runs | file mtimes; `git log` |

### 2.2 Differences in today's managed driver that must change

Each row changes output bytes, could change them, or could let an ambient setting change them. The driver must match the reference column.

| # | today (`tdp_gen.sh`, `src/tdp/runner.rs`) | reference / new rule | why it matters |
|---|---|---|---|
| D1 | `gadget_seed = SEED+1` (`tdp_gen.sh:419`) | `gadget_seed = sandwich_seed = seed = GEN_SEED` | gadgetization RNG stream (`StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E)`) |
| D2 | `shuffling_segments` defaults to 0, and `return_home=0` is **rejected** (`tdp_gen.sh:254-261`, `validate.rs:101-108`) | 8 and 0, set by the preset | stage-2 construction |
| D3 | PREMASK, SEED_AVOID_C1, CHAIN and CHAIN_LEN cannot be expressed: they are scrubbed (`runner.rs:338`) and have no field | 1, 1, 2, 8 (L8); 1, 1, unset, unset (A) | stage-2 construction |
| D4 | always exports `EMBEDDED_MASKING_{K,MAX_OPEN,MIN_OPEN,BALANCED,QUAD_FIRE}` | not set under the L8/A presets | value-neutral today, but the recipe sets no other stage-2 variable |
| D5 | profile `3,30,30,2,2` (n0=3 hard-coded, hold=27) | `2,2,2,2,2` | stage-3 schedule |
| D6 | moves = `round(n2·r1·G·1.3)` | `int(G·2·2·1.3)` (truncation) | the last move whenever the fraction is ≥ .5 |
| D7 | mix seed = `SEED+2` | an independent random value | stage-3 RNG |
| D8 | `--verify-every 2000000` | default 10 000 | **`global_check` draws `4·num_wires` u64 values from the mixer RNG** (`reporting.rs:17-23`), so the cadence changes the trajectory. This adds about 400 global checks at n=128 (§9) |
| D9 | `--report-every 100000 --state-out` | defaults; no state-out | the report cadence runs `anc_prune` and the flag and snapshot checks. Not proven trajectory-neutral, so it is dropped |
| D10 | exports `CANON_RULE_L_BRANCH_CAP=512 CANON_MONOMIAL_CAP=200000` | **unset** (unbounded) | the caps change canonicalization results whenever they bind |
| D10b | exports `CANON_CACHE_MB=256 XPOLY_CANON_CACHE_MB=1024 LOOKUP_CACHE_MB=2048` | **kept** at the driver's values | these are exact memo caches (`lookup_cache.rs:15-23`), and the default 512 MB lookup cache epoch-resets on runs of a million moves or more. They are classified output-neutral. Evidence: the n=16 `cache_neutrality` test (§10.9 item 5) was never run; what shows neutrality is that stage 3 under the driver's sizes produced bytes identical to the reference runs (built-in 512 MB lookup cache) on the historic anchor (vL64a, vL128a) and on the same-seed reproductions at n=64 (§12). This is the one stage-3 environment difference from `val_chain.sh` |
| D11 | `FROZEN_FILTER` auto-detection reads `/proc/meminfo` (Linux only) and requires 60 GiB available | `auto`: `FROZEN_FILTER=1` when the store has no `filters.bin`, the exact reference situation. When `filters.bin` exists, `FROZEN_FILTER=0` until the filtered/unfiltered equivalence of §10.9 item 6 is recorded; after that, `1` only behind a 60 GiB available-RAM gate that also works on macOS | the filter is meant to be output-neutral but needs about 25.5 GB RAM (`frozen.rs:6`); neutrality has never been measured |
| D12 | gate count via `wc -l` minus 1 | header field 3 | equal for consistent files; use the header field |
| D13 | the driver ends at stage 3 | stage 4 | — |
| D14 | the gen mode positional `embedded-masking` is passed | **omitted** for embedded-masking (passed only for nonlinear291) | neutral, but omitting it makes the argv identical to the reference, so the transcript test (§10.9 item 1) needs no normalization |
| D15 | environment is scrubbed by a deny-list (`PROD_`, `SAT_`, `EMBEDDED_MASKING_`, `DB_QC*`, `SCRUBBED_ENV`), which lets through `MIXER_DB_LEN_BAND`, `SLICE_GUARD_LEGACY`, `MIXER_STOP_AT_*`, `TWIST_G57_*` and `MIN_DIR_LOOKUP` | **allow-list**: stage binaries start from an empty environment plus the allow-list and recipe variables. Every inherited registry variable that changes output is a fatal error (§11.5) | any of these silently changes or truncates stage 2 or stage 3 |
| D16 | the C and D size defaults to `round(n·log2²n)` | the validated table: 1000 at n=64, 3000 at n=128. Other n require an explicit size or the opt-in `formula` rule (§2.4) | every downstream byte |
| D17 | gen parses positionals with `.parse().ok().unwrap_or(default)` (`gen mod.rs:100-117`): an unparsable seed becomes **1** | strict parsing: a present but unparsable positional is a fatal error. The driver also checks that gen's banner seeds equal `GEN_SEED` (§8.4) | a parse slip gives a public, fixed C that passes every later check |
| D18 | `HOLD` default 27 in the script, `"30"` in the manifest text (`manifest.rs:197`) | the manifest prints resolved values only | latent inconsistency |

`FROZEN_REGULAR_VALUE_CONVENTION=native` stays exported: the reference log shows `regular=native`, which is the built-in default, so the effective value is equal (§11.5 compares effective values). `SANDWICH_VARIANT` is not exported (as implemented; `GEN_ENV` holds only `EMBEDDED_MASKING_*` values, and `tests/shell/tdp_flags.bash` asserts that no `SANDWICH_*` variable reaches gen): its built-in default is `classic`, the reference's value, so the effective value is equal; the registry classifies it `driver` (inherited → fatal) and the stage-2 marker records `sandwich_variant=classic`.

### 2.3 Seeds, entropy and hygiene (deviation; decision for RC)

The reference draws the gen seed below 2^31 and the mix seed below 2^40. **The gen seed regenerates C** (`prepare_source(n, m_c, seed)`), and generating C takes microseconds. A 31-bit seed therefore makes C recoverable by brute force against any artifact. The validation runs are calibration-grade material.

The managed default draws **both** seeds uniformly below `2^63 − 16` from the OS CSPRNG. For any given seed, every computation is the one the reference performs, and explicit seeds in the reference range are accepted, so reproduction is unaffected. The manifest records this as `seed_bits=63`.

**63 bits is a stopgap, not a security margin.** C is generated through `fastrand::seed(seed)` (wyrand, 64-bit state; `sandwich/construct.rs:70`), the gadget stream through `StdRng::seed_from_u64`, and the mixer from a u64. Whatever width the driver draws, C never carries more than 64 bits of entropy, and about 2^63 cheap generations is within reach of a well-resourced attacker. This is recorded as open decision §13.1, with two routes:

- **(a)** Derive C and the gadget streams from a 256-bit seed (ChaCha20 via `rand_chacha`, already in `Cargo.lock`). This changes bytes, so it cannot be the default while the defaults requirement holds without RC re-validating.
- **(b)** Supply a CSPRNG-generated C through `source.path` (`TDP_SOURCE_C`), which works today.

**Hygiene rules.**

- No unkeyed hash of C, of `tdp.mpmct1`, of `db_mixing.mpmct1`/`s3.mpmct1` or of the seeds is written anywhere: not in `packing.report`, not in `tdp_command.conf`, not in `packing.recipe`, not in committed files.
- Integrity pins on private inputs use HMAC-SHA256 keyed by a per-run 256-bit `run_key`, which lives only in the private `$RUN/SEEDS` (§8). The MACs are computed in bash with `python3 -I` (`hmac`, `hashlib`), so Rust needs no new crypto dependency.
- **Publishable files:** `packing.report` (§5.5) and `tdp_command.conf` contain no seeds, no seed hashes and no hashes of private circuits. Their only content hash is the hash of the deliverable, which is public. The private files are `$RUN/SEEDS`, `$RUN/packing.recipe` (MACs), `stage12.log` (gen banner) and every circuit file.

### 2.4 Source size (resolved by the request)

The request makes the validated sizes the default:

| `source.wires` | default `source.gates` | `source_gates_rule` |
|---|---|---|
| 64 | 1000 | `validated` |
| 128 | 3000 | `validated` |
| any other n, `source.gates` given | as given | `explicit` |
| any other n, `source.gates_rule = "formula"` | `round(n·log2²n)` | `formula`; the manifest and log print `UNVALIDATED SIZE` |
| any other n, neither | — | **error**, exit 2: "no validated source size for n=…; set source.gates or source.gates_rule = \"formula\"" |

An explicit `source.gates` at n=64 or n=128 equal to the table value is recorded as `validated`; any other value is recorded as `explicit`. In `tdp_gen.sh` the same rule applies to `-m/--source-gates` and the new `--source-gates-rule formula`. The C-view floor and the L2/M2 numbers were measured only at the table sizes.

---

## 3. Architecture

```
     stage 2 G2 (tdp.mpmct1)   stage 3 G3 (db_mixing.mpmct1)        C (tdp.mpmct1.source_c.g57)
                \                     /                                    |
                 v                   v                                     |
  pack  (fingerprints MT19937/1024 + blake2b-128 value keys -> proxy spans |
         -> intervals -> scheduler, windowed ANF caps on a CPython-3.9 set)|
     -> packed_G512.mpmct1 + .bounds   [certificate: linear extension of G3]
                 |                                                         v
  repair: cancel -> [C-view scan -> fix]x<=4 -> [C-view -> merge? -> split]x<=4 -> final scan (report)
          (in-process exposure engine; per-cut memo)  [certificate: cancel pairs + trace equivalence]
     -> rep512.mpmct1 + rep512.bounds (+ rep512.{exposure,allcuts,wit}.tsv)
                 |
  emit:   canonical per-window ANF -> DE body -> deliverable.bin (raw)
                                             -> deliverable.tdp4 (48-byte header + same body)
  verify: exact certificates; sampled equivalence; exact decode == ANF; exhaustive or sampled window ANF;
          zero-slice deliverable(x) == C(x)
```

There is one library module tree, `src/stages/packing/`, with **no environment reads**, following the `src/stages` contract (`preprocessing/types.rs:31`). Its one production adapter is the binary `tdp_packing`. The exposure engine moves out of `hmap_affine` into this library and is shared by both.

---

## 4. Library layout: `src/stages/packing/`

Add `pub mod packing;` to `src/stages/mod.rs`. Its doc comment becomes: "TDP stage ownership: sandwich, preprocessing, database mixing and packing (stage 4)." Unit tests attach by the repo convention:

```rust
#[cfg(test)] #[path = "../../../tests/unit/stages/packing/<module>.rs"] mod tests;
```

The `repair/` submodules use one more `../`.

**No new crate dependency.** The following are in-house, each under about 200 lines, and each tested against `hashlib`/`zlib` vectors:

- MT19937;
- the CPython set emulator;
- blake2b-128 (RFC 7693);
- CRC-32.

`rayon` and `xxhash-rust` are already dependencies.

**Supported range.** Stage 4 supports n ≤ 256, i.e. a state of at most 1024 wires, so the monomial width is W ∈ {1, 2, 4, 8, 16}. Config validation rejects n > 256 when `stop_after_stage = 4`, and `tdp_packing` rejects it with exit 2.

```
src/stages/packing/
  mod.rs          run_stage4(), re-exports
  types.rs        params, CapSemantics, RepairMode, reports, Stage4Error + exit codes
  oracle_io.rs    oracle-compatible readers/writers (mpmct1 width rules, k=0 trailing space, bounds)
  pyrand.rs       CPython random.Random: MT19937 init_by_array + getrandbits word order
  blake2b.rs      blake2b with digest_size 16 (value identity exactly as WP)
  pyset.rs        CPython 3.9 setobject emulator (slot table of arena indices + cached hashes)
  mono.rs         monomial keys [u64; W], Mersenne-fold CPython int hash, ordering, width dispatch
  anf.rs          packer compose + window_anf (split / emission), cap semantics
  fingerprint.rs  1024-sample bit-sliced evaluation, complement normalisation, proxy_spans
  intervals.rs    interval_set + next-next-writer rescue, winnable values, iid numbering
  scheduler.rs    the window_pack3 main loop
  gf2.rs          GF(2) basis with coefficient tracking (shared by cview)
  cview.rs        the C-view exposure engine (from HA), streaming over cuts, memoized per cut
  certify.rs      exact release-mode certificates (trace equivalence, cancellation)
  repair/mod.rs   repair loop (repair.sh), RepairMode
  repair/seq.rs   gate/marker sequence: to_seq, from_seq, commute, slide
  repair/cancel.rs cancel_pass, run_cancel
  repair/fix.rs   run_fix (canonical breaker handling)
  repair/merge.rs merge_bounds2
  repair/split.rs anf_split2 (candidate rule, recursion, fit predicate)
  serialize.rs    DE body encoder, container v1, CRC-32, flat-arena decoder, evaluator
  verify.rs       verify_equiv (seed 77), window checks, decode checks, zero-slice vs C
  census.rs       link_census (L2/M2) and the AC size report
```

### 4.1 `types.rs`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapSemantics { CPython39, Canonical }        // §7.2; default CPython39
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepairMode { Oracle }                        // later fixes add variants behind flags

#[derive(Clone, Debug)]
pub struct PackParams {
    pub w_cap: usize,        // 512  (WP argv3)
    pub g: usize,            // 512  (argv4)
    pub b_mono: usize,       // 24576 (argv5); MUL_CAP = 8*b_mono
    pub span_cap: usize,     // 3000 (argv6)
    pub pad: usize,          // 4    (WP const)
    pub cmax: usize,         // 24   (WP const)
    pub fp_seed: u64,        // 20261004
    pub cap_semantics: CapSemantics,
}
impl PackParams { pub const PRODUCTION: PackParams = /* above, CPython39 */; pub fn mul_cap(&self) -> usize { 8 * self.b_mono } }

#[derive(Clone, Debug)]
pub struct RepairParams {
    pub split_budget: usize,         // 65536
    pub fix_iterations: usize,       // 4
    pub fallback_iterations: usize,  // 4
    pub cview: CviewParams,          // batches 96, seed 12345
    pub cap_semantics: CapSemantics, // CPython39 (window_anf in the split predicate)
    pub mode: RepairMode,
}

#[derive(Clone, Debug)]
pub struct Stage4Params {
    pub pack: PackParams, pub repair: RepairParams,
    pub verify_seed: u64,            // 77   (verify_equiv)
    pub zero_slice_samples: usize,   // 4096 (new check; no oracle counterpart)
    pub exhaustive_support_max: usize, // 16 (windows with support <= this are checked exhaustively)
    pub allow_floor_miss: bool,      // false
    pub allow_over_budget: bool,     // false
    pub max_concurrent_anf: usize,   // 8 (bounds concurrent window_anf evaluations; §4.4 memory)
}

pub struct PackReport { /* every number in the WP log lines, plus mul_calls, ambiguous, compose_aborts, order_dependent */ }
pub struct RepairReport { /* cancel passes, fix trajectory, moved per pass, fallback trajectory, merges, splits,
                             forced splits, fix_exhausted, final scan, path counters (§10.3) */ }
pub struct EmitReport { /* windows, monomials, max per window, body bytes, deliverable xxh3 */ }

#[derive(Debug)]
pub enum Stage4Error {
    Usage(String),                                  // exit 2
    Input(String),                                  // exit 3
    Verify(String),                                 // exit 4
    Floor { nonoutput_visible: Vec<usize> },        // exit 5
    OverBudget { windows: Vec<(usize, usize)> },    // exit 6
    Io { path: PathBuf, source: io::Error },        // exit 7 (disk full, permission, rename)
    Internal(String),                               // exit 8 (invariant violated without panicking)
}
impl Stage4Error { pub fn exit_code(&self) -> i32; pub fn meaning(code: i32) -> &'static str; }
```

`Stage4Error::meaning` is the single table of exit-code meanings (§5.4). `runner.rs` and `tdp_gen.sh` print it.

### 4.2 `oracle_io.rs`: circuits as the oracle reads them

The oracle and Rust read circuits differently. Python keeps controls in file order and accepts malformed gates, while `formats::read_mpmct` sorts and validates. The Python scripts also use different widths. This module makes those differences explicit.

```rust
pub struct OracleCircuit {
    pub header_n: usize,     // hdr[1]       (order_repair, verify_equiv use this)
    pub header_m: usize,     // hdr[2]       (DE uses this as m)
    pub computed_n: usize,   // max(header_n, maxwire+1) (WP, AC use this)
    pub gates: Vec<XGate>,
}
pub fn read_oracle_mpmct(path: &Path) -> Result<OracleCircuit, Stage4Error>;
pub fn write_pack_mpmct(w: &mut impl Write, n: usize, gates: &[XGate]) -> io::Result<()>;   // WP:476-481 (no trailing space)
pub fn write_repair_mpmct(w: &mut impl Write, n: usize, gates: &[XGate]) -> io::Result<()>; // order_repair.save: k=0 -> "t c 0 \n"
pub fn read_bounds(path: &Path) -> Result<Vec<usize>, Stage4Error>;   // whitespace-split ints, input order kept
pub fn write_bounds(w: &mut impl Write, b: &[usize]) -> io::Result<()>; // "\n".join + "\n" ("\n" when empty)
```

`read_oracle_mpmct` is strict. It accepts only files on which `XGate` equality coincides with the Python tuple:

- controls already ascending in the file;
- no duplicate control;
- no control on the target;
- comp and polarity in {0, 1};
- header m equal to the number of gate lines;
- no extra tokens on a line.

Every Rust-written artifact satisfies this (checked on vA64a). Any violation is an `Input` error and is never silently canonicalized.

Width rules, applied where each oracle step uses them:

- pack: `n = max(computed_n(A), computed_n(B))`, and the output header is `computed_n(B)`;
- repair: the header n is carried through;
- emission: m is the header m.

`commute(a, b)` lives in `repair/seq.rs` and is `!a.reads(b.target) && !b.reads(a.target)`, where `reads` means "has as a control". It must **not** reuse `XGate::collides`, which exempts polarity-separated conjunctions. Two gates on the same target with neither controlling on the other's target commute, because XOR targets add.

### 4.3 `pyrand.rs`: CPython `random.Random`

```rust
pub struct PyRandom { mt: [u32; 624], idx: usize }
impl PyRandom {
    pub fn new(seed: u64) -> Self;                       // key = 32-bit LE words of |seed| (one word if < 2^32), init_by_array
    pub fn genrand_u32(&mut self) -> u32;
    pub fn getrandbits_words(&mut self, k: usize) -> Vec<u32>; // ceil(k/32) words, least-significant first; partial top word = u32 >> (32 - r)
    pub fn getrandbits_lanes<const L: usize>(&mut self) -> [u64; L]; // k = 64L; lane i = w[2i] | w[2i+1] << 32
    pub fn getrandbit(&mut self) -> bool;                // getrandbits(1) = top bit of one output
}
```

| user | seed | draw |
|---|---|---|
| packer and link census | 20261004 | 16 lanes × n |
| `verify_equiv` | 77 | 16 lanes × n |
| AC spot checks | 20261004 | 4 lanes |
| DE evaluation | 1 | 16 lanes |

Golden vectors are in Appendix A.

### 4.4 `pyset.rs`: CPython 3.9 set emulator

This module is what makes the packer and the split predicate bit-exact. `mul` aborts when a **prefix** of the smaller operand's iteration order pushes `|res|` above the cap (WP:135-150, AC:49-64), and that order is CPython's table order. All 8 measured compose aborts, and the AC case vA64a span `[492039,492540)`, have a full product within the cap; they fire only because of the order. Under sorted order, 1 of 2 (vA64a) and 1 of 6 (vL128b) disappear.

The order is version-dependent. On this machine, `{1,2,3}; discard 2; add 10` iterates as `[10, 1, 3]` on CPython 3.9.6, which does not reuse the dummy slot. A reported 3.13 run iterates `[1, 10, 3]`. The emulator therefore targets **3.9 `setobject.c` semantics**, and the oracle runtime is a pinned CPython 3.9 that passes the trace self-test (§10.1, traces in §10.3). It is never "whatever `python3` is installed".

```rust
pub trait PyIntKey: Copy + Eq { fn py_hash(&self) -> u64; }  // nonnegative int: value mod (2^61-1)
pub struct PySet<K: PyIntKey> {
    slots: Vec<Slot>,          // Slot { hash: u64, idx: u32 }, idx = NULL (u32::MAX) | DUMMY (u32::MAX-1) | arena index
    arena: Vec<K>,             // keys; discarded keys stay as garbage until the next resize
    mask: usize, fill: usize, used: usize,
}
impl<K: PyIntKey> PySet<K> {
    pub fn new() -> Self;                         // size 8 (PySet_MINSIZE)
    pub fn singleton(k: K) -> Self;               // BUILD_SET {x}: new() then add
    pub fn len(&self) -> usize;
    pub fn contains(&self, k: &K) -> bool;
    pub fn add(&mut self, k: K);                  // first NULL on the probe path; dummies NOT reused (3.9)
    pub fn discard(&mut self, k: &K) -> bool;     // slot -> DUMMY, used -= 1, never shrinks
    pub fn toggle(&mut self, k: K);               // `if v in res: discard else add`
    pub fn copy_of(other: &Self) -> Self;         // set(other) == set_merge(empty, other), incl. verbatim fast path
    pub fn xor(a: &Self, b: &Self) -> Self;       // a ^ b == copy_of(b) then toggle a's keys in a's table order
    pub fn xor_assign(&mut self, b: &Self);       // a ^= b: toggle b's keys in b's table order
    pub fn iter(&self) -> impl Iterator<Item = &K>; // table order, skipping NULL/DUMMY
}
```

Rules to transcribe (packer spec §9; validated empirically and by the traces of §10.3):

- **Probe:** `i = h & mask`, `perturb = h`. Examine slot i and, when `i + 9 <= mask`, slots `i+1..i+9` (LINEAR_PROBES = 9). Then `perturb >>= 5` and `i = (5i + 1 + perturb) & mask`.
- **Growth:** after an insert, if `fill*5 >= mask*3`, resize to `used > 50000 ? used*2 : used*4`.
- **`resize(minused)`:** start from 8 and double while `<= minused`. Re-insert the active entries in old table order with `insert_clean` (first NULL on the probe path, no equality test). This includes the **small-table purge**: newsize == 8 with `fill > used`, which rebuilds in place.
- **Merge into an empty set:** if `(fill + other.used)*5 >= mask*3`, first resize to `(used + other.used)*2`. Then copy verbatim when `fill == 0`, `mask == other.mask` and `other` has no dummies; otherwise `insert_clean` in `other`'s table order.

**Memory layout.** Iteration order is a function of the slot table only, so storing a 12-byte `(hash, idx)` slot plus an arena of keys leaves the order semantics unchanged. Resize compacts the arena in the new table order.

At n=128 (W=8, 64-byte keys), a 2^21-slot table costs about 32 MB of slots plus the live keys, instead of about 150 MB for inline keys.

*Implemented (P3):* the slot is a 16-byte `(hash: u64, idx: u32)` record (padding included), so one probe step touches one record, and a linear group of 10 slots is scanned branch-free (candidates = NULL slots and active slots with an equal hash, examined in slot order). On vL64a this took the packer's scheduler from 24.5 s (two parallel arrays, slot-by-slot loop) to 13.7 s; a 2^21-slot table is 32 MB of records. Probe chains are long because CPython's int hash of a monomial is its value mod 2^61-1, so clusters share low bits: vL64a makes 336 M probes visiting 13 G slots (39 per probe), which the emulator must visit because the table order depends on them. A tag-byte filter (SwissTable style) gave no further gain and was not kept. Concurrent `window_anf` evaluations are bounded by `max_concurrent_anf` (default 8). The memory test is §10.5 step 7.

The emulator is THE set type in both the packer's compose and `window_anf`, whatever the `CapSemantics`. Table order only matters for the prefix abort, but one code path is simpler to test.

### 4.5 `mono.rs`

```rust
pub trait MonoKey: PyIntKey + Ord + Hash + Default {
    const WORDS: usize;
    fn bit(w: usize) -> Self; fn or(&self, o: &Self) -> Self; fn is_zero(&self) -> bool;
    fn bits(&self) -> impl Iterator<Item = usize>;
}
impl<const W: usize> MonoKey for [u64; W] { /* Ord compares from the most significant word = Python int order */ }
/// Calls `f::<[u64; W]>()` for the smallest W in {1,2,4,8,16} with 64·W >= nw; errors above 1024 wires.
pub fn with_mono_width<R>(nw: usize, f: impl MonoWidthFn<R>) -> R;
```

At n=64 the state is 256 wires, so W=4. At n=128 it is 512 wires, so W=8.

Const-generic width is used only where it pays: `PySet`, `mul`, the hash, `WindowState` and `window_anf`. The scheduler, repair, emission framing and census take `&dyn` access or a runtime width. Five instantiations of the hot core are compiled, not nine instantiations of all of stage 4.

**CPython int hash with Mersenne folding** (no u128 division on the hot path). With `M = 2^61 − 1`, we have `2^64 ≡ 8 (mod M)` and `2^61 ≡ 1`. Folding from the most significant limb down, starting at `h = 0`:

```
rot8(h)  = ((h << 3) & M) | (h >> 58)          // 8·h mod M, for h < M
x        = rot8(h) + (limb >> 61) + (limb & M)  // < 2^63
h        = (x & M) + (x >> 61); if h >= M { h -= M }
```

This is exact and branch-light. Tests:

- `int_hash_matches_cpython`: recorded 3.9 hashes for keys with top bits at 0, 60, 61, 63, 64, 127, 128, 255, 511 and 1023, including `1<<61` (hash 1) and `(1<<122)` (hash 1);
- the `py_hash` microbenchmark in `benches/` (target ≤ 5 ns at W=8).

### 4.6 `anf.rs`

```rust
/// The packer's incremental window ANF (WP compose, 324-344). Monomials are over GLOBAL wire ids.
pub struct WindowState<M: MonoKey> { polys: FxHashMap<u16, PySet<M>>, monos: usize, mul_cap: usize }
impl<M: MonoKey> WindowState<M> {
    pub fn new(mul_cap: usize) -> Self;
    pub fn compose(&mut self, g: &XGate, sem: CapSemantics, stats: &mut MulStats) -> bool; // false = abort, nothing mutated
    pub fn monos(&self) -> usize;   // == sum(len) over every written wire (maintained incrementally)
    pub fn reset(&mut self);
}
pub fn mul<M: MonoKey>(a: &PySet<M>, b: &PySet<M>, cap: usize, sem: CapSemantics, stats: &mut MulStats) -> Option<PySet<M>>;

#[derive(Debug)] pub enum AnfOverflow { Product, RunningSum, Final }
pub struct WindowAnf<M> { pub changed: Vec<(u16, Vec<M>)>, pub total: usize } // wires ascending, monomials ascending
/// AC window_anf (66-91): product cap 8·budget checked per outer row; running sum > 4·budget after each gate
/// over ALL entries (identity-valued ones included); final total > budget.
pub fn window_anf<M: MonoKey>(gates: &[XGate], budget: usize, sem: CapSemantics, stats: &mut MulStats) -> Result<WindowAnf<M>, AnfOverflow>;
pub struct MulStats { pub calls: u64, pub ambiguous: u64, pub aborts: u64, pub aborts_order_dependent: u64 }
```

`mul` semantics:

- Swap so the smaller operand is outer, and only when `len(a) > len(b)`. On a tie, the accumulated product stays outer.
- Toggle per pair: XOR, not dedup.
- Check `len(res) > cap` after each outer row.
- In the ambiguous zone `(|small|-1)·|large| > cap`:
  - in `Canonical` mode, complete the product and abort only if the final size exceeds the cap;
  - in `CPython39` mode, the prefix decides.

  Both modes count `aborts_order_dependent`, the aborts whose full product would fit. In `CPython39` mode that requires finishing the product on a scratch set after an abort, which happens only on aborts.

### 4.7 `fingerprint.rs`

```rust
pub type Fp = [u64; 16];                                  // 1024 samples
pub fn inputs(n: usize, seed: u64) -> Vec<Fp>;            // PyRandom(seed).getrandbits(1024) per wire, wire order
pub fn normalize(v: &Fp) -> Option<Fp>;                   // complement if sample 0 is set; None if constant
pub fn value_key(v: &Fp) -> [u8; 16];                     // blake2b(int.to_bytes(128,'little'), digest_size=16), exactly WP
pub struct Spans { pub spans: Vec<(u32 /*b*/, u32 /*e*/, u32 /*a*/)>, pub forced: FxHashSet<u32>, pub agree: usize,
                   pub a_wire: FxHashMap<u32, u16>, pub carried: usize }
pub fn proxy_spans(a: &[XGate], b: &[XGate], n: usize, na: usize, nb: usize, inputs: &[Fp]) -> Spans; // WP:63-90
```

The evaluation adds `XGate::apply_lanes16(&self, &mut [[u64; 16]])` to `xgate.rs`, bit-identical per word to `apply_lanes`.

**Value identity uses the oracle's own key**, blake2b-128 of the normalized value, computed by the in-house `blake2b.rs` (golden vector in Appendix A). The index is `FxHashMap<[u8;16], u32>` from the key to the first birth index in A. Revision 1 keyed on the full 128-byte `Fp` and listed a 2^-128 collision deviation. Using the oracle's key removes that deviation (even a blake2b collision behaves identically) and saves about 250 MB at n=128. Cost: about 4.6M hashes of 128 bytes at n=128, about 1-2 s.

The pending-close rule is always applied: the span pops on **every** write, including constant and equal values.

### 4.8 `intervals.rs`

```rust
pub fn interval_set(b_gates: &[XGate], b: usize, e: usize, n: usize, cap: usize) -> Option<Vec<u32>>; // WP:92-133
pub struct Intervals {
    pub imem: Vec<Vec<u32>>, pub ival: Vec<u32>, pub ib: Vec<u32>, pub ie: Vec<u32>,  // by iid, mem sorted
    pub value_iids: Vec<(u32 /*a*/, Vec<u32>)>,     // first-appearance order of a
    pub member_of: Vec<SmallVec<[u32; 4]>>,         // per gate, ascending iid
    pub stats: WinnableStats,                        // carried, winnable, disq_long, disq_big, rescued
}
pub fn build_intervals(b_gates: &[XGate], spans: &Spans, n: usize, w_cap: usize, span_cap: usize) -> Intervals; // WP:175-226
```

The order of `by_val` is the first appearance in `spans`, which is e-ascending. It is reproduced with a first-appearance `Vec` plus an index map. The numeric value of `a` never orders anything.

### 4.9 `scheduler.rs`

```rust
pub struct Packed { pub n_out: usize, pub order: Vec<u32>, pub bounds: Vec<usize>, pub report: PackReport }
pub fn pack(a: &OracleCircuit, b: &OracleCircuit, p: &PackParams, log: &mut dyn Write) -> Result<Packed, Stage4Error>;
```

Faithful to WP:228-460, following packer spec §§6-11:

- **Epoch readiness:** per-wire `Vec<Vec<u32>>` buckets indexed by write and read counts.
- **Heaps:** `filler` and `anyready` are `BinaryHeap<Reverse<u32>>` with lazy deletion.
- **`committed`:** an order-preserving map, a `Vec<(key, val, live)>` plus an `FxHashMap` from key to slot. Iteration skips dead entries, and it compacts when more than half the slots are dead.
- **Sorts:** stable sorts for `by_end` and the contagion snapshot.

Quirks are reproduced, not fixed:

- On a compose failure the window closes **after** g is appended and **before** g's interval bookkeeping.
- The contagion snapshot goes stale within a pass.
- A burst continues through a mid-burst close.
- Caps are tested before a gate is added, so windows overshoot.
- The pre-emptive close is unconditional on `win_load`, so it can produce a duplicate or zero boundary.
- The final filter keeps `0 < b < m` without dedup.
- The output header is `computed_n(B)`.

The unreachable paths (`burst stuck`, `scheduler stuck`, `resumed_later`) and the lemma "the minimum non-scheduled entry of anyready is the lowest unscheduled gate" are **release-mode checks** returning `Internal` (exit 8), not `debug_assert!`s.

`order` is certified exactly after packing (§4.11a): it is a permutation of G3's indices and a linear extension of G3's dependency order. WP's sampled VERIFY is kept as an additional check.

*Implementation notes (P3).* The signature is `pack(a, b, p, label, log) -> Result<Packed, Stage4Error>` (the label is WP argv 7). The window ANF is a width-erased `Box<dyn Composer>` (the `WindowState<W>` of §4.6 picked by `with_mono_width`), so only the ANF core is monomorphized. Readiness buckets are compressed rows keyed by a dense id per (wire, count) event; `committed` is a `Vec<(iid, rem)>` with a per-iid slot index, compacted when more than half its entries are dead; heaps are `BinaryHeap<Reverse<u32>>`. Intervals are built in parallel per value (pure, merged in first-appearance order). `PackReport` adds `compose_aborts`, `bounds`, `duplicate_bounds` and per-phase wall times. Failure-path differences (D-b): an empty B is an `Input` error (WP writes its outputs and then fails on its window statistics); a VERIFY FAIL prints WP's full log and then exits 4 without writing any output (WP writes before verifying).

Text log lines reproduce WP's format: header, winnable, intervals, scheduled, the `death causes` dict repr in first-increment order, VERIFY, RESULT, class lines and window stats. They are compared on fixtures after path normalization (§10.3). Python `:.1f` versus Rust `{:.1}` can differ only on exact binary ties (D-c).

### 4.10 `gf2.rs` and `cview.rs`: the exposure engine, moved out of `hmap_affine`

```rust
pub const DEFAULT_SEED: u64 = 12345; pub const DEFAULT_BATCHES: usize = 96; pub const NULLITY_CAP: usize = 14;
#[derive(Clone, Debug)] pub struct CviewParams { pub batches: usize, pub seed: u64 }
pub struct HonestSamples { pub n: usize, pub batches: usize, x: Vec<Vec<u64>> }          // [batch][wire < n]
impl HonestSamples { pub fn draw(n: usize, batches: usize, seed: u64) -> Self; }         // StdRng::seed_from_u64(seed); per batch, n x random::<u64>()
pub fn prefix_indices(end: usize, step: usize) -> Vec<usize>;                            // == HA indices()
pub fn parse_cut_file(text: &str, g_len: usize) -> Result<Vec<usize>, String>;           // HA:328-341
pub struct BornRow { pub prefix: usize, pub gate: usize, pub wire: usize }
pub struct CRows { pub rows: Vec<BornRow>, vecs: Vec<Vec<u64>> }                         // [row][batch]
pub fn born_rows(c: &[XGate], n: usize, prefixes: &[usize], s: &HonestSamples) -> CRows;
pub fn output_gates(c: &[XGate], n: usize) -> FxHashSet<usize>;                          // last writer per wire < n
pub struct CutColumns { /* reps, rep_wires, basis, nulls */ }
impl CutColumns { pub fn build(state: &[Vec<u64>], nw: usize) -> Self; pub fn predict(&self, row: &[u64]) -> Option<CutFit>; }
pub struct CutResult { pub cut: usize, pub fits: Vec<(u32 /*row*/, CutFit)> }            // rows ascending
pub struct CutOracle<'a> { /* P, nw_g, samples, rows, checkpoints every 16384 gates, memo FxHashMap<cut, Arc<CutResult>> */ }
impl<'a> CutOracle<'a> {
    pub fn new(p: &'a [XGate], nw_g: usize, rows: &'a CRows, s: &'a HonestSamples) -> Self;
    pub fn results(&mut self, cuts: &[usize]) -> Vec<Arc<CutResult>>;                    // missing cuts in parallel, returned in cut order
}
pub struct ExposureOptions { pub collect_allcuts: bool, pub collect_witness: bool, pub skip_settled: bool }
pub fn assemble(rows: &CRows, cuts: &[usize], per_cut: &[Arc<CutResult>], o: &ExposureOptions) -> ExposureReport; // sequential, cut order
pub fn exposure_scan(c_rows: &CRows, g: &[XGate], nw_g: usize, s: &HonestSamples, cuts: &[usize], o: &ExposureOptions) -> ExposureReport;
impl ExposureReport {
    pub fn visible(&self) -> usize;
    pub fn summary_lines(&self) -> Vec<String>;
    pub fn write_exposure_tsv(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn write_allcuts_tsv(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn write_witness_tsv(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn nonoutput_visible(&self, outs: &FxHashSet<usize>) -> Vec<usize>;  // identity floor test
}
```

**Cut independence.** A `(row, cut)` fit depends only on `(C, P, cut, n, seed, batches)`, never on which other cuts are scanned, so `CutResult` is memoized per cut. Semantics that span several cuts are replayed **sequentially in cut order** in `assemble`:

- strict-< best weight;
- earliest-cut witness;
- `skip_settled` and its `approx` accounting.

Within a cut, the following match HA exactly:

- the representative is the lowest wire of each complement class;
- reps are inserted in wire order, with the constant last;
- forward reduction runs in basis insertion order;
- the Gray-code walk keeps the first strict minimum;
- nullity above 14 counts as approximate.

The engine **streams** P with one `[batch][wire]` state, 393 KB at n=128. It keeps checkpoints every 16 384 gates so it can re-simulate candidate cuts, instead of HA's materialized 2.8 GB `gs[batch][cut][wire]`.

**`hmap_affine` after the move.** Exposure mode becomes `HonestSamples::draw`, then `born_rows`, then `exposure_scan` with `skip_settled = allcuts_path.is_none()`, then printing and writing in the same order. Line 1, every assert and every panic message stay. Heatmap and SPAN modes keep their code; SPAN moves later and separately.

**Engine parity is never self-referential.** It is checked against a **pinned baseline `hmap_affine` built from `492564f0`** (§10.1), never against a binary built from the tree under test.

The engine is not feature-gated, because stage 4 must build with `cargo build --release`. `hmap_affine` stays under `security-tools`.

**As implemented (P2).** Three details differ from the sketch above; none changes an output byte.

- `gf2::TrackedBasis` keeps the basis in **reduced** row-echelon form, so `solve` selects the basis vectors from the pivot bits of the row and tests membership word by word, rejecting a hidden row after a few hundred word operations instead of a full reduction. HA's basis is not reduced (pivot = lowest set bit). Both give the same independent columns (greedy, in insertion order), the same null vectors (the unique `e_d + c_I` with `R_I c_I = R_d`) in the same order, and the same particular solution (the unique one supported on the independent columns), so the Gray walk starts from the same vector and walks the same coset. A unit test checks this against a transcription of HA's basis.
- The state of P (and of C in `born_rows`) is **wire-major**, `[wire][batch]`, so a cut's column for wire k is a contiguous slice. `CutColumns::build(cols, nw, batches)` takes that layout.
- `parse_cut_file` returns `CutParseError { line, source }`; HA maps it back to its old `expect("cut index")` panic, so the message is unchanged.

`CutOracle` computes missing cuts in chunks of `2 x threads` snapshots with rayon. `with_parallel(false)` and `with_checkpoint_every(k)` exist for tests. `ExposureOptions::historic(allcuts, witness)` is HA's flag combination (`skip_settled = !allcuts`). In HA, `SPAN_INTERSECT` and the heatmap still materialize `[batch][cut][wire]` with their old code; only the exposure scan streams.

### 4.11 `repair/`: the oracle repair in-process

```rust
// seq.rs
pub enum Item { Gate { gate: XGate, origin: u32 /* index in packed */ }, Marker }
pub fn to_seq(gates: &[(XGate, u32)], bounds: &[usize]) -> Vec<Item>;     // dedup, keep 0 < b < m
pub fn from_seq(seq: &[Item]) -> (Vec<(XGate, u32)>, Vec<usize>);        // merge adjacent markers, drop leading/trailing
pub fn commute(a: &XGate, b: &XGate) -> bool;
pub fn slide(seq: &mut Vec<Item>, pos: usize, forward: bool) -> usize;    // incl. the dest==pos no-move quirk
// cancel.rs
pub fn cancel_pass(gates: &[XGate]) -> (Vec<bool> /*dead*/, Vec<(u32, u32)> /*pairs by origin*/); // dead gates still block
pub fn run_cancel(seq: &mut Vec<Item>) -> Vec<Vec<(u32, u32)>>;           // pairs per pass, to the fixed point
// fix.rs
pub struct FixRow { pub c_gate: usize, pub cut: usize, pub support: Vec<u16> }
pub fn run_fix(seq: &mut Vec<Item>, rows: &[FixRow], outs: &FxHashSet<usize>) -> FixReport; // rows in ascending C gate
// merge.rs
pub fn merge_bounds(bounds: &[usize], allcuts: &[(u32, usize, u32)], rows: &[BornRow], outs: &FxHashSet<usize>) -> (Vec<usize>, usize);
// split.rs
pub fn split_candidates(lo: usize, hi: usize) -> Vec<usize>;              // {lo + (hi-lo)*i/33 : i=1..32} \ {lo,hi}, sorted
pub fn fits<M: MonoKey>(window: &[XGate], budget: usize, sem: CapSemantics, stats: &mut MulStats) -> bool; // == anf_split2 fit()
pub fn split_pass<M: MonoKey>(p: &[XGate], bounds: &[usize], budget: usize, sem: CapSemantics,
                              leaky: &mut dyn FnMut(&[usize]) -> FxHashSet<usize>, log: &mut dyn Write) -> SplitReport;
// mod.rs
pub struct Repaired { pub n_header: usize, pub gates: Vec<XGate>, pub origins: Vec<u32>, pub bounds: Vec<usize>,
                      pub last_scan: ExposureReport, pub final_scan: Option<ExposureReport>, pub cancelled: Vec<(u32, u32)>,
                      pub report: RepairReport }
pub fn repair(packed: &[XGate], packed_n: usize, bounds0: &[usize], c: &[XGate], n: usize,
              p: &RepairParams, log: &mut dyn Write) -> Result<Repaired, Stage4Error>;
```

Gates carry their packed index (`origin`) through every sequence operation. Nothing in the oracle algorithm reads it; it exists for the certificates.

Control flow of `repair` in `RepairMode::Oracle`, which is exactly `repair.sh`:

1. `seq = to_seq(packed, bounds0)`, then `run_cancel`, then `(gates, bounds) = from_seq(seq)`.
2. `verify_equiv(packed, gates)` against the immutable in-memory packed circuit. There is no file race.
3. **Fix loop**, for `it` in `1..=fix_iterations`. **Every iteration starts from `seq = to_seq(gates, bounds)`**, where `gates, bounds` come from the previous `from_seq`, exactly as the subprocess chain reloads the files (`order_repair.py:177-179`). Markers are therefore normalized at the start of each iteration: adjacent and duplicate markers merged, leading and trailing markers dropped. Then:
   - scan `(gates, bounds)`;
   - log `fix iter {it}: C-view visible {v} (floor {n})`;
   - if `v <= n`, break;
   - otherwise `run_fix(&mut seq, witness rows, outs)` and `(gates, bounds) = from_seq(seq)`.

   There is no rescan after the last fix. If all four iterations run without meeting the floor, `fix_exhausted = true` is reported.
4. `verify_equiv` again.
5. **Fallback loop**, for `it` in `1..=fallback_iterations`:
   - `v` = scan of the current lattice;
   - if `v > n`, `merge_bounds` with this scan's allcuts;
   - `split_pass` with budget 65536 and `leaky` = `CutOracle` on the unchanged rep circuit (cuts with a fit row outside `outs`);
   - break iff `v <= n` and 0 splits were added.
6. **Final scan (new, report-only).** Scan the final lattice and compute `nonoutput_visible(outs)` and the over-budget windows (`!fits(window)` with the configured `CapSemantics`). This changes no output byte. It decides the exit status (§5.4).

**Single window.** If the bounds are empty at any scan (a legitimately short G3, or merges that remove every boundary), there is no interior cut:

- the scan is skipped and recorded as `visible=0`, `final_visible=0`;
- the fallback loop breaks;
- the circuit is serialized as one window.

The oracle panics here (HA "empty plate"), so this is a defined behaviour on an oracle failure path (D-b), covered by the `single_window` fixture and unit test.

**Memoization.** The rep circuit is constant across the fallback loop, because merge and split touch only bounds. Every scan and every split predicate in that loop share one `CutOracle`. Each fix iteration changes the circuit and gets a fresh one.

The merge reads the outputs from C via `output_gates(c, n)`. The oracle derives them from `exposure.tsv` (`last[w] = max gate`). Both are equal when c-step is 1 and every wire is written; the reader asserts this equality on the rows.

**Implementation notes (P4).** The code follows the sketch above, with these interface differences (none changes an output byte):

- A sequence item is a `u32`: the gate's origin (its packed index; the packed circuit is the arena every gate is looked up in) or `seq::MARKER`. `to_seq(origins, bounds: &[i64])`, `from_seq(seq) -> (origins, bounds)`, `slide(seq, arena, pos, forward)`.
- `cancel_pass(&[&XGate]) -> (dead, pairs)` uses per-wire CSR position lists and a next-identical-gate chain; a unit test compares it with a literal transcription of the Python on 300 random circuits. `run_cancel(seq, arena)` returns the pairs per pass as origins.
- `repair(packed, packed_n, bounds0, &CView, &RepairParams, rep_label, log)`. `CView` holds the honest samples, C's born rows and the output gates, built once; it scans a lattice (`hmap_affine --g-cuts` rules: sorted, deduplicated, cuts beyond the circuit dropped), answers the split's leak query and renders the `cview.log` lines. The output gates are derived from the born rows exactly as the oracle derives them from `exposure.tsv` (largest C gate per wire), which equals `output_gates(c, n)` at c-step 1.
- `split::FitCache` memoizes the split predicate per window of the fixed fallback circuit; the top-level windows of a pass are evaluated in parallel, the recursion is sequential (DFS order, so the `WARNING` lines come out in the oracle's order). The final budget test reuses it. A window of at most 2 gates is not evaluated during the split (the oracle evaluates it and returns either way).
- **Lattice without an interior cut (D-b).** The scan is "zero cuts": `visible = 0` (logged as the oracle line plus `  (single window: no interior cut, scan skipped)`). The loop rules then apply unchanged: the fix loop stops, the fallback still runs the split (so an over-budget single window is split), and the loop breaks when the split adds nothing. If every scan of a run was skipped, the TSVs describe the zero-cut scan (every row `-1`).
- `tdp_packing repair` writes `rep<G>.*` and `repair<G>.log` even when the final tests fail (exit 5 or 6): they are intermediates the oracle also writes, and parity on failure paths needs them; `run` (P6) will not write a deliverable in that case. `check-final` runs the final tests on any repaired circuit.

#### 4.11a `certify.rs`: exact certificates (release mode)

Every stage-4 transformation is a reorder by commutation or a cancellation, so equivalence can be certified exactly in O(gate-wire incidences). These checks run in release builds on every run, and a failure is exit 4.

- **`certify_reorder(g3, order)`.** `order` is a permutation of `0..|G3|`, and for every wire w the **run-canonical form** of the access sequence is equal for G3 and for the packed order. The access sequence is the gate ids in order, each labelled `W` (writes w) or `R` (controls on w). The run-canonical form is the sequence of maximal same-label runs, each as a sorted id set.

  Two orders have equal forms on every wire iff they order every conflicting pair (one writes w, the other reads w) identically. That is iff they are linear extensions of the same dependency partial order, which implies functional equality, because non-conflicting gates commute (§4.2). Writes on the same wire commute with each other (XOR), and so do reads.
- **`certify_cancel(packed_order, passes)`.** Replay the recorded pairs pass by pass on the packed sequence of origins. For each pair `(i, j)`, check that the gates are identical and that every **live** gate strictly between them commutes with gate i. Then delete both. This uses the pairs' semantics, not `cancel_pass`'s code: `g X g = X` when g commutes with every gate of X.
- **`certify_repair(packed, rep)`.** Apply `certify_cancel`, then `certify_reorder` between the remaining sequence and `rep.origins`. Fix slides are pure commutations, so they are covered.
- **Window ANF.** `check_windows_anf` compares each decoded window's monomial sets **exactly** with `window_anf` (an equality of sorted lists, not sampling). The ANF engine itself is checked against the gates:
  - exhaustively for every window whose support is at most `exhaustive_support_max` = 16 wires (all 2^f inputs, bit-sliced);
  - with 1024 random full states for every other window.

The sampled checks (`verify_equiv` with seed 77, WP VERIFY, zero slice) are kept as additional checks.

### 4.12 `serialize.rs`: body, container, decoder

```rust
pub const MAGIC: [u8; 8] = *b"LMTDPANF"; pub const FORMAT_VERSION: u16 = 1; pub const HEADER_BYTES: u16 = 48;
pub const IO_LAYOUT_CLASSIC_ZERO_SLICE: u32 = 1;   // inputs 0..n, answer n..2n, all other wires 0
pub struct ContainerHeader { pub source_wires: u32, pub state_wires: u32, pub io_layout: u32, pub window_count: u32, pub body_bytes: u64, pub body_crc32: u32 }
pub fn encode_body(gates: &[XGate], bounds: &[usize], m_header: usize, nw: usize, out: &mut impl Write) -> Result<EmitReport, Stage4Error>; // streamed
pub fn write_container_header(h: &ContainerHeader, out: &mut impl Write) -> io::Result<()>;
pub fn crc32_ieee(bytes: &[u8]) -> u32;                     // == zlib.crc32
pub enum Format { Raw { source_wires: u32 }, Container }
pub struct Deliverable {                                    // flat arena: no per-monomial allocation
    pub header: Option<ContainerHeader>,
    bytes: Vec<u8>,                                         // all mask bytes, contiguous
    windows: Vec<WindowIdx>,                                // support range, changed-wire range
    wires: Vec<WireIdx>,                                    // (wire id, f, mask_bytes, offset into bytes, count)
    supports: Vec<u16>,
}
pub fn decode(bytes: &[u8], format: Format) -> Result<Deliverable, Stage4Error>;   // explicit format; validates every invariant
impl Deliverable { pub fn eval_lanes16(&self, state: &mut [[u64; 16]]); pub fn eval_u64(&self, state: &mut [u64]); }
```

`encode_body` follows `DE:14-50` byte for byte:

- `cuts = [0] + sorted(0 < b < m_header) + [m_header]`, without dedup;
- each window comes from `window_anf(…, EMIT_BUDGET)`. The oracle uses 1<<30, where no cap fires; the port uses `EMIT_BUDGET = MAX_MONOMIAL_BUDGET` (2^18) as a resource ceiling (review round 3). A cap only ever aborts, so a window that emits is the oracle's canonical one; every window the repair leaves (fitting a split budget ≤ 2^18, or of at most 2 gates) emits; a larger one (only reachable with hand-made bounds) is `OverBudget`, exit 6, instead of an unbounded allocation;
- the support is ascending;
- the record header is `<HH`;
- the support ids are `<H` each;
- each changed wire, in ascending order, is `<HI` followed by its local masks: `ceil(f/8)` bytes little-endian per mask, monomials in ascending global value;
- each blob is preceded by `<I len`.

Overflows (a wire above 65535, f or c above 65535, a count or length beyond u32) are errors, never a wrap.

Windows are encoded in parallel in bounded batches (`max_concurrent_anf`) and **streamed** to `deliverable.bin.tmp` in window order, while the CRC-32 is computed on the fly. There is no in-memory body. `deliverable.tdp4` is the 48-byte header followed by a copy of the body file (`io::copy`).

`decode` never guesses the format: `Format::Raw` requires the source width, and `Format::Container` requires the magic.

### 4.13 `verify.rs`

```rust
pub fn verify_equiv(a: &[XGate], b: &[XGate], n: usize, seed: u64) -> Result<(), Stage4Error>;      // verify_equiv.py: PyRandom(77), 1024 all-wire states
pub fn check_windows_anf(gates: &[XGate], bounds: &[usize], d: &Deliverable, exhaustive_max: usize, seed: u64) -> Result<usize, Stage4Error>; // §4.11a
pub fn check_deliverable(gates: &[XGate], d: &Deliverable, seed: u64) -> Result<(), Stage4Error>;   // decode(file) == circuit, sampled end to end
pub fn check_zero_slice(c: &[XGate], d: &Deliverable, n: usize, samples: usize, seed: u64) -> Result<(), Stage4Error>; // deliverable(x on 0..n, 0 elsewhere)[n..2n] == C(x)
```

`check_windows_anf` replaces AC's spot checks, which only tested windows with `wi % 500 == 0`. AC's report numbers are reproduced by `census.rs`. `check_zero_slice` is the honest-slice end-to-end check that the reference chain never ran automatically.

### 4.14 `census.rs` (report only)

```rust
pub fn link_census(a: &OracleCircuit, p: &[XGate], bounds: &[usize]) -> LinkCensus;   // link_census.py, PyRandom(20261004) 1024 samples
pub fn anf_size_report(gates: &[XGate], bounds: &[usize], budget: usize, label: &str, sem: CapSemantics) -> String; // AC report lines (no spot checks)
```

These reproduce `link_rep512.txt` (L2, M2, size histograms) and the first three lines of `anf_rep512.txt`, and their parity is gated (§10.5 step 5). They are not on the deliverable path.

### 4.15 `mod.rs`

```rust
pub struct Stage4Inputs<'a> { pub g2: &'a OracleCircuit, pub g3: &'a OracleCircuit, pub c: &'a [XGate], pub n: usize }
pub struct Stage4Output { pub packed: Packed, pub repaired: Repaired, pub emit: EmitReport, pub report: Stage4Report } // body is on disk
pub fn run_stage4(inp: &Stage4Inputs, p: &Stage4Params, out_dir: &Path, log: &mut dyn Write) -> Result<Stage4Output, Stage4Error>;
```

---

## 5. The production binary `tdp_packing`

```toml
[[bin]]
name = "tdp_packing"
path = "src/programs/tdp_packing/main.rs"
```

The binary has no required features. It lives in `src/programs/tdp_packing/{main.rs,mod.rs,run.rs}` (`run.rs` is the production `run` command; `mod.rs` the parity subcommands, argument parsing and staged output). It reads **no environment variables**; threads are set by `--threads` only.

### 5.1 CLI

```
tdp_packing run   --stage2 <tdp.mpmct1> --stage3 <db_mixing.mpmct1> --source-c <tdp.mpmct1.source_c.g57>
                  --n <N> --out-dir <DIR> [--label tdp]
                  [--window-gates 512] [--interval-cap 512] [--monomial-budget 24576] [--span-cap 3000]
                  [--split-budget 65536] [--fix-iterations 4] [--fallback-iterations 4]
                  [--anf-cap-semantics cpython39|canonical]
                  [--allow-floor-miss] [--allow-over-budget] [--threads K] [--no-oracle-files]
tdp_packing pack      <A> <B> <W_cap> <G> <B_mono> <span_cap> <label> <out.mpmct1>      # WP argv, parity mode
tdp_packing repair    --packed P --bounds B --source-c C --n N --out-dir DIR [repair flags] # repair.sh parity mode
tdp_packing serialize <rep.mpmct1> <rep.bounds> <out.bin> <label>                        # DE parity mode (raw body)
tdp_packing check-final --rep R --bounds B --source-c C --n N [--split-budget 65536] [--anf-cap-semantics …]
                                                     # final identity + budget test on any (oracle or Rust) repair
tdp_packing decode    <file> --format raw --n N | --format container [--stats]
tdp_packing eval      <file> --format … --input <hex|file>
tdp_packing census    <A> <P> <B> <label>                                                # link_census parity
```

`--label` is the text prefix of the report lines (`<label>-G512`, `<label>-rep512`; the oracle's `TAG`); it reaches no binary artifact. The zero-slice check always draws 4096 samples (the constant `ZERO_SLICE_SAMPLES`; there is no flag). The parameters are bounded as in the managed driver's `[packing]` (`PackParams::validate`, `RepairParams::validate`): window gates and interval cap 1..=10^6, span cap 1..=10^9, monomial budget and split budget 1..=2^18 (`MAX_MONOMIAL_BUDGET`), iterations 0..=1000; anything else is exit 2. The gate bounds keep the scheduler's `2·G` and `G − load` from wrapping in release builds; the budget bound is a resource limit (review round 3): a window holds up to 4·budget monomials and a product 8·budget, budgets of 10^6-10^9 reached 5-9 GB on the n=8 fixture within two minutes, 2^20 still held 2-3 GB at one monomial word, and 2^18 peaked at 1.2 GB (G = W_cap = 10^6, 156 s). 2^18 is 5× the largest measured packer budget (49,152 at G=1024) and 4× the recommended split budget.

### 5.2 Inputs

| input | notes |
|---|---|
| `tdp.mpmct1` | the stage-2 circuit |
| `db_mixing.mpmct1` | the stage-3 circuit; `s3.mpmct1` in the reference |
| `tdp.mpmct1.source_c.g57` | the source C, read only by repair and verify. It never reaches a shipped byte |
| `--n` | the source wires. It must equal the width of C, `4n` must be at most the state width, and n ≤ 256 |

### 5.3 Outputs, all in `--out-dir`

The directory is created with mode 0700, and every file is created under umask 077. Each file is written to `*.tmp` and renamed. A rename or write failure is `Io` (exit 7).

| file | content | ships? |
|---|---|---|
| `deliverable.bin` | the **raw DE body**, byte-identical to the reference's `deliverable.bin`. Existing readers (`decode_check.py`, `finalize*.sh`, `score_arm.sh`) keep working | no |
| `deliverable.tdp4` | container v1 (§6): 48-byte header plus the same body | **yes, the shipped file** |
| `packed_G{G}.mpmct1`, `packed_G{G}.mpmct1.bounds` | packer output, WP format | no |
| `rep{G}.mpmct1`, `rep{G}.bounds` | repaired circuit and bounds, `order_repair` format | no |
| `rep{G}.exposure.tsv`, `rep{G}.allcuts.tsv`, `rep{G}.wit.tsv`, `rep{G}.cview.log` | the last in-loop scan, exactly as the oracle leaves them; omitted with `--no-oracle-files` | no (C-keyed) |
| `pack_G{G}.txt`, `repair{G}.log`, `anf_rep{G}.txt`, `link_rep{G}.txt` | oracle-format text reports | no |
| `packing.report` | machine-readable report (§5.5), publishable | no |

Nothing is written outside `--out-dir`.

A reused `--out-dir`: once the inputs have loaded, `run` removes every file of the table above (for this `G`, and its `*.tmp`) before writing anything, so a failed run never leaves a deliverable, `packing.report` or intermediate that belongs to an earlier run (an input error, exit 2 or 3, touches nothing). On success the commit order is `deliverable.bin`, `deliverable.tdp4`, then `packing.report`: the report is the completion record, and a directory without one, or with a status other than `ok`, holds no recommended deliverable. Each file is renamed separately (renames cannot be undone as a group); after a failed rename the remaining `*.tmp` files are removed.

### 5.4 Exit codes

| code | meaning | `deliverable.*` |
|---|---|---|
| 0 | finished. `packing.report` `status=ok`, or, only under an `--allow-*` flag, `status=floor_miss` or `status=over_budget` | written |
| 2 | usage or parameter error (including n > 256) | — |
| 3 | input error: unreadable, malformed, inconsistent header, non-canonical gate line, width mismatch | — |
| 4 | verification failure: a certificate (§4.11a), pack VERIFY, `verify_equiv`, ANF ≠ gates on any window, decode ≠ ANF, zero slice ≠ C | not written |
| 5 | exposure floor missed: the final scan has a visible non-output row (identity test) | not written |
| 6 | over budget: some final window fails `split::fits(window, split_budget, cap_semantics)`, the exact predicate the split uses | not written |
| 7 | I/O error writing outputs (disk full, permission, rename) | partial `*.tmp` removed |
| 8 | internal invariant violated (non-panicking) | — |
| 101 | panic | — |

Under `--allow-floor-miss` or `--allow-over-budget`, the corresponding failure exits **0** with the deliverable written and `status=floor_miss` or `over_budget` in `packing.report`. A caller must read `status`; the driver does (§11.1).

Exits 5 and 6 are deliberate departures from the oracle on failure paths only. The oracle silently shipped vL128a v1 at 129 visible, and it keeps over-budget 1-2 gate windows silently. On every path where the oracle's output meets both tests, the bytes are unchanged.

### 5.5 `packing.report`

A flat `key=value` file in fixed key order. **Every key is publishable:** no seeds, no hashes of private inputs, nothing derived from C except counts.

```
packing_report=2
status=ok|floor_miss|over_budget
tool=tdp_packing <crate version> <git describe if embedded>
binary_xxh3=<128-bit hex>
n=64  state_wires=256
param.window_gates=512 param.interval_cap=512 param.monomial_budget=24576 param.mul_cap=196608 param.span_cap=3000
param.pad=4 param.cmax=24 param.fp_seed=20261004 param.fp_samples=1024 param.split_budget=65536
param.fix_iterations=4 param.fallback_iterations=4 param.cview_batches=96 param.cview_seed=12345
param.verify_seed=77 param.anf_cap_semantics=cpython39 param.repair_mode=oracle param.exhaustive_support_max=16
pack.windows= pack.spans= pack.carried= pack.forced= pack.winnable= pack.disq_long= pack.disq_big= pack.rescued=
pack.intervals= pack.hidden_spans= pack.hidden_values= pack.death.burst_close= pack.death.sibling= pack.death.drain_close=
pack.exposed= pack.mul_calls= pack.mul_ambiguous= pack.compose_aborts= pack.compose_aborts_order_dependent=
repair.cancel_passes=… repair.cancel_pairs= repair.fix_trajectory=… repair.fix_moved=… repair.fix_exhausted=
repair.fallback_trajectory=… repair.merged= repair.splits=… repair.forced_splits= repair.single_window=
repair.final_visible= repair.final_nonoutput_visible= repair.final_over_budget= repair.split_mul_ambiguous=
repair.split_order_dependent=
emit.windows= emit.monomials= emit.max_window_monomials= emit.body_bytes= emit.container_bytes= emit.deliverable_xxh3=
census.proxy_values= census.proxy_L2= census.proxy_M2= census.froth_L2= census.froth_M2=
check.reorder_cert=ok check.repair_cert=ok check.pack_verify=ok check.equiv=ok
check.windows_anf=ok(<exhaustive>/<sampled>) check.decode=ok check.zero_slice=ok(4096)
time.pack_s= time.repair_s= time.emit_s= time.verify_s= peak_rss_mb=
```

---

## 6. Deliverable container v1 (`deliverable.tdp4`)

The DE body has no header, so it carries no n, no version and no integrity check (TDP doc §6.12). Container v1 wraps the **unchanged** body under a **distinct name**. The raw `deliverable.bin` keeps its existing meaning.

| offset | size | field | value |
|---|---|---|---|
| 0 | 8 | magic | `LMTDPANF` |
| 8 | u16 | format_version | 1 |
| 10 | u16 | header_bytes | 48 |
| 12 | u32 | flags | 0 (reserved; readers reject unknown bits) |
| 16 | u32 | source_wires | n |
| 20 | u32 | state_wires | header n of the repaired circuit (4n); at most 65536 |
| 24 | u32 | io_layout | 1 = classic zero slice: x on wires 0..n, every other wire 0, answer on wires n..2n |
| 28 | u32 | window_count | |
| 32 | u64 | body_bytes | |
| 40 | u32 | body_crc32 | zlib/IEEE CRC-32 of the body |
| 44 | u32 | header_crc32 | CRC-32 of bytes 0..44 |

All fields are little-endian. `deliverable.tdp4[48:] == deliverable.bin`, which is byte-identical to the reference chain's `deliverable.bin`. That body is the "payload" in every parity statement.

**The oracle is not edited.** Parity compares `deliverable.bin` (and `deliverable.tdp4[48:]`) with the oracle's raw body. The header is tested by:

- decoder unit tests;
- a 15-line standalone writer, `tests/manual/stage4/container_header.py` (`struct` + `zlib.crc32`), which `anf_parity.sh`, `run_parity.sh` and `reproduce_val_chain.sh` use to rebuild the header from the oracle body and `cmp` the whole container.

The decoder (`serialize::decode`, `tdp_packing decode`) validates the following (the same set `port_runs/anf_spec/decode_check.py` verified on vA64a):

- magic, version and both CRCs (container);
- blob lengths;
- support strictly ascending;
- changed wires strictly ascending;
- masks strictly ascending;
- no mask bit at or above f;
- every support index used;
- f = 0 iff c = 0.

---

## 7. Exactness plan

### 7.1 Per component

| component | oracle | target | how it is achieved and tested |
|---|---|---|---|
| MT19937 / getrandbits | `random.Random` | bit-exact | golden vectors (Appendix A) |
| value identity | WP blake2b-128 | **bit-exact** (same key) | in-house blake2b, hashlib vectors |
| fingerprints, spans, forced, agree | WP:52-90 | bit-exact | `apply_lanes16` |
| winnable values, intervals, iid order | WP:175-226 | bit-exact | first-appearance order; rescue rule |
| scheduler | WP:228-460 | **byte-exact `packed_G512.mpmct1` + `.bounds`** | ordered `committed`, stable sorts, quirks kept; sha256 against the 8 archived packs |
| packer ANF caps | WP:135-150, 324-344 | bit-exact | `PySet` (`CPython39`); every compose abort reproduced (2 on vA64a, 6 on vL128b); large-table traces |
| pack log text | WP prints | equal after path normalization; `:.1f` ties tolerated (D-c) | |
| cancel / slide / fix | `order_repair.py` (canonical) | bit-exact | stale-blocker rule, greedy ascending, slide landing rules, gate-count cut, breaker survival, `to_seq` per fix iteration |
| C-view TSVs, visible counts | HA exposure | **byte-exact TSVs** | against the **pinned baseline** HA (§10.1), never the tree under test |
| merge | `merge_bounds2.py` | bit-exact | outputs from C, asserted equal to the TSV rule |
| split | `anf_split2.py` | bit-exact | `fits` on `PySet` (`CPython39`); per-cut leak memo; candidate, tie and recursion rules |
| loop control | `repair.sh` | bit-exact | count test `vis <= N`, 4+4 iterations, break rules; final scan report-only |
| `rep512.mpmct1` + `.bounds` | `order_repair.save` | **byte-exact** | k=0 trailing space; header n carried |
| body bytes | DE | **byte-exact payload** | canonical (the oracle's budget 1<<30; the port refuses windows above `EMIT_BUDGET` = 2^18 with exit 6, never emits a different one); sha256 against the 8 archived `deliverable.bin` |
| container | none (new) | header unit tests + standalone writer | |
| equivalence | sampled in the oracle | **exact certificates** + sampled | §4.11a |
| link census / AC report | `link_census.py`, AC | numbers equal, gated | §10.5 step 5 |

### 7.2 `CapSemantics` (decision recorded)

The default is **`CPython39`**. The defaults requirement and the reproduction test both require it: without it, the packer and split decisions diverge on real runs. It applies in the packer (MUL_CAP 196 608) and in the split predicate (8·65 536).

`Canonical` (`--anf-cap-semantics canonical`) aborts a product only when the completed product exceeds the cap. It is an explicit, recorded deviation, never the default. The `*_order_dependent` counters show on every run how many decisions depended on the emulator.

### 7.3 Allowed differences

- **D-b, failure paths:**
  - hard errors (exits 3-8) where the oracle continues, writes on `VERIFY FAIL`, panics ("empty plate"), or silently keeps an over-budget window;
  - the defined single-window behaviour;
  - no background verify race;
  - no pid-named scratch files.

  On paths where the oracle's result meets the identity and budget tests, there is no byte difference.
- **D-c.** Log float formatting on exact `.x5` binary ties (log only).
- **D-d.** The shipped file is `deliverable.tdp4`. `deliverable.bin` is the unchanged raw body.
- **D-e.** The oracle's dead DE placeholder pass and its timings are not ported (DE:70-90).

Revision 1's D-a (full-value keys) no longer exists (§4.7). Sorted iteration in place of CPython order is not allowed in default mode.

---

## 8. Determinism and seeds

### 8.1 Stage 4

Stage 4 is a deterministic function of (G2, G3, C, n, parameters). Its fixed analysis constants are:

- the fingerprint seed 20261004;
- the C-view seed 12345 with 96 batches;
- the verify seed 77.

They are recorded in `packing.report`. Rayon parallelism is restricted to pure per-cut and per-window functions merged in fixed order, so a test runs with 1 and 8 threads and compares the outputs.

### 8.2 Who draws the seeds, and where they live

**Managed path (Rust).** `runner.rs` resolves the seeds before anything else is written to the run directory:

- **random** (default): `gen_seed` and `mix_seed` uniform in `[0, 2^63 − 16]` from the OS CSPRNG (`rand::rngs::OsRng`, by rejection sampling), plus a 256-bit `run_key`;
- **explicit** (calibration only): read from `[calibration] seed_file`, plus a fresh random `run_key`.

On a fresh run, Rust creates `$RUN/SEEDS` with `create_new` and mode 0600, **then** writes `tdp_command.conf`. On a resume, `SEEDS` must already exist and is never rewritten. An explicit `seed_file` whose values differ from the existing `SEEDS` is fatal; the comparison is done in Rust and never printed.

Seeds never travel on argv between Rust and bash: bash only reads `$RUN/SEEDS`.

`$RUN/SEEDS` format, version 2:

```
seeds_format=2
gen_seed=<decimal>
mix_seed=<decimal>
run_key=<64 hex>
seed_source=random|explicit
```

**Standalone path (bash).** When `tdp_gen.sh` runs without Rust and `$RUN/SEEDS` is absent, bash draws the same three values with `python3 -I -c 'import secrets; …'` and writes the file under `umask 077` with `set -o noclobber`. Explicit seeds in standalone mode are given only as `--seed-file PATH`: a private file (mode 0600, at most 256 bytes) with `gen_seed=` and `mix_seed=` lines.

### 8.3 Legacy seed inputs

These are rejected with migration messages; there is no silent reinterpretation:

- A one-integer `seed_file` / calibration file (recipe v9): "seed file format 1 holds one seed; recipe v10 needs `gen_seed=` and `mix_seed=` lines (the old seed was the gen seed; the old mix seed was gen+2)".
- `-s` / `--calibration-seed` on the `tdp_gen.sh` command line: "seeds on argv are no longer accepted (they appear in ps); use --seed-file".
- A v9 `$RUN/SEED` file: "created by recipe v9; resume with the checkout that created it".

Calibration mode never draws a random mix seed.

### 8.4 Validation and redaction

- `tdp_gen.sh` parses `SEEDS` strictly: exactly the five keys, `gen_seed` and `mix_seed` through `bounded_uint 0 9223372036854775792`, and `run_key` as 64 lowercase hex characters. A malformed file is fatal (exit 2).
- **Gen banner check.** After stage 2, bash requires line 1 of `stage12.log` to show `seed=(file) gadget_seed=(file) sandwich_seed=(file)`, i.e. that gen took all three seeds from the driver's descriptors (§8.5). It prints only `gen seeds: verified` or a fatal "gen used a different seed than SEEDS". With strict gen parsing (D17) this is a second line of defence.
- `render_command` (`runner.rs:542`) redacts the value after **every** flag whose name contains `seed`, except `--seed-file`, whose path is not secret. Seed values are never printed to `tdp_gen.log`, stdout or the dry-run output.
- `tdp_command.conf` records `seed_record=SEEDS`, `seed_source`, `seed_bits=63` and `calibration_only`, and nothing derived from the seeds. The stage-12 marker records `seeds_mac = HMAC-SHA256(run_key, "gen_seed=…\nmix_seed=…\n")`, so a swapped `SEEDS` file is detected on resume without publishing a brute-forceable hash.

### 8.5 Use of the seeds

- Gen receives `GEN_SEED` as all three positional seeds: `seed`, `gadget_seed` and `sandwich_seed` (D1).
- Stage 3 receives `MIX_SEED` as the mixer seed (D7).
- **Neither seed is on a stage binary's argv** (review round 1). `val_chain.sh` passes them as arguments, where `ps` shows them to every local user for the life of the process (seconds for gen, 25 min to hours for the mixer), and the gen seed alone regenerates the secret C and the whole stage-2 artifact. The driver instead hands each value over on an inherited file descriptor: gen gets `@/dev/fd/3 @/dev/fd/4 @/dev/fd/5` (a positional seed of the form `@PATH` is read from PATH), the mixer `--seed-file /dev/fd/3`, each fed by a bash here-string (`3<<< "$GEN_SEED"`; a builtin, so the value is never on any argv; nothing is written to disk). The values reaching the binaries are unchanged, so every output byte is too. Their banners print `(file)` in place of a file-borne seed, so `stage12.log` and `stage3.log` hold no seed value either. The only on-disk copy is the private `$RUN/SEEDS` (and, for a calibration run, the operator's seed file). Running `val_chain.sh` itself, or a stage binary by hand with `--seed N`, still exposes the seed in `ps`; the `ps-seed-leak` hygiene rule applies there.

---

## 9. Performance targets

Reference points (Python/bash, Apple-silicon Mac). Stages 1-3 ran with 8 replicates concurrently.

| step | n=64 | n=128 |
|---|---|---|
| stage 1+2 (gen) | seconds | seconds |
| stage 3 (mixer, verify-every 10 000) | about 34 min (vL64a) | about 98 min (vL128a) |
| pack | 84 s | about 3.2 min |
| repair | about 2-4 min | about 10-12 min |
| DE serialize | — | 107 s for 106.8 MB |
| one HA scan | about 13 s | about 2.4 min, single-threaded |

Targets on a 10-core Apple-silicon Mac (release build):

| step | n=64 | n=128 | notes |
|---|---|---|---|
| stage 3 | within +10% of a `val_chain.sh` run on the same machine and seeds | same | D8 brings back about 400 global checks at n=128, which parity requires. D10b keeps the cache sizes. Measured in the reproduction test |
| pack | ≤ 15 s | ≤ 60 s | Mersenne hash (§4.5); arena sets; incremental `win_monos`; 9.1M mul calls at n=128 |
| cancel + certificates | ≤ 3 s | ≤ 15 s | O(gate-wire incidences) |
| one full C-view scan | ≤ 5 s | ≤ 30 s | parallel over cuts; about 7 000 cuts × 3 000 rows at n=128 |
| repair total | ≤ 1 min | ≤ 4 min | at most 4 fix scans with a fresh oracle each; the fallback loop and every split share one memo |
| emit + all checks | ≤ 20 s | ≤ 90 s | streamed body; flat-arena decode; exhaustive windows ≤ 16 support |
| **stage 4 wall** | **≤ 2 min** | **≤ 6 min, hard ceiling 10 min** | `time.*_s` in `packing.report`; measured (not asserted) by `run_parity.sh` and the acceptance runs (§12) |
| **stage 4 peak RSS** | ≤ 1.5 GB | **≤ 3 GB** | G3 as XGate (4.6M gates, about 180 MB); blake2b value index (about 2M × 20 B, about 40 MB); intervals ≤ 100 MB; arena sets ≤ 8 concurrent × ≤ 60 MB; C-view streaming plus checkpoints and memo ≤ 200 MB; decoded deliverable as a flat arena (about 110 MB) |

---

## 10. Fixtures and tests

### 10.1 Pinned reference binaries and interpreter

- **Baseline `hmap_affine`:** built from `492564f0` with `--features security-tools` into `port_runs/baseline_target` (durable disk) via a detached `git worktree` at that commit. Its sha256 is recorded in `tests/fixtures/stage4/README.md`. The scripts take it as `HMAP_BASELINE=<path>` and verify the sha256 before use.
  - Every fixture regeneration (`regen.sh`), the reference side of `reproduce_val_chain.sh` and `tdp_reproduce_small.bash`, and the CI differential job (§10.6) use this binary. The CI job builds it from `492564f0` itself and caches it by commit.
  - `regen.sh` **refuses** to run with an HA binary whose sha256 differs from the recorded one, so fixture TSVs can never be produced by a binary that links the new library.
- **Archive-era `hmap_affine`:** the main checkout's `target/release/hmap_affine` that produced the Oct 8 re-repairs. Today it has sha256 `94757413406e58c9fe40694e392213da015b198a0cfb22a83f4b9b79ca8a84f3` and mtime 2026-10-07 18:54:28. At P2 start it is **copied** (read-only use) to `port_runs/archive_hmap_affine` so a later rebuild of the main checkout cannot lose it.
- **Oracle interpreter:** a uv-managed CPython 3.9 (`uv python install 3.9.6`; `ORACLE_PY=$(uv python find 3.9.6)`), not `/usr/bin/python3`, which an OS or Command Line Tools update can replace.
  - Every script that runs the oracle first runs `tests/fixtures/stage4/pyset_selftest.py` under `$ORACLE_PY`, replaying the recorded traces (§10.2), and refuses to continue if any order differs.
  - The gate is the **self-test**, not `sys.version`. Any 3.9.x build whose set behaviour matches is acceptable, and `/usr/bin/python3` 3.9.6 remains usable when it passes.

### 10.2 Unit tests (`tests/unit/stages/packing/*.rs`)

| file | tests |
|---|---|
| `pyrand.rs` | `mt_first_words_seed_20261004`, `getrandbits_1024_lane_layout`, `seed_77_and_1_low128`, `getrandbit_is_top_bit` |
| `blake2b.rs` | `rfc7693_vectors`, `wire0_value_key_matches_hashlib` |
| `pyset.rs` | `replays_cpython39_traces` (`pyset39_traces.txt`), `replays_large_traces_checkpoints` (`pyset39_large.txt`), `dummy_not_reused_on_add`, `resize_used_le_50000_quadruples`, `resize_used_gt_50000_doubles`, `small_table_purge_newsize_8`, `resize_reinserts_in_old_table_order`, `merge_fast_path_copies_slots_verbatim`, `merge_slow_path_insert_clean`, `xor_is_copy_b_then_toggle_a`, `arena_compaction_keeps_order`, `order_with_keys_bit_ge_61_and_ge_128` |
| `mono.rs` | `int_hash_matches_cpython` (bits 0..1023, including the M61 wrap), `mersenne_fold_equals_u128_mod`, `ord_matches_python_int_order_multi_limb` |
| `anf.rs` | `worked_example_doc_4_6`, `pair_toggle_not_dedup`, `swap_only_when_strictly_larger`, `running_sum_counts_identity_entries`, `prefix_abort_cpython_vs_canonical`, `compose_failure_mutates_nothing`, `order_dependent_counter` |
| `fingerprint.rs` | `complement_normalization_and_constants`, `span_closes_on_every_write`, `first_birth_is_value_id` |
| `intervals.rs` | `interval_set_matches_bruteforce_taint`, `cap_returns_none_at_cap_minus_2`, `rescue_next_next_writer`, `iid_order_is_first_appearance` |
| `scheduler.rs` | `lowest_unscheduled_is_ready`, `committed_delete_preserves_order`, `compose_failure_boundary_after_gate`, `preemptive_close_can_emit_duplicate_boundary`, `contagion_snapshot_is_stale` |
| `certify.rs` | `reorder_cert_accepts_commuting_swap`, `reorder_cert_rejects_conflicting_swap` (including one that differs on only a 2^-k fraction of inputs, and one that differs only off the zero slice), `cancel_cert_rejects_blocked_pair`, `cancel_cert_rejects_nonidentical_pair`, `same_target_writes_commute` |
| `cview.rs` | `complement_pair_collapses`, `constant_value_tstar_zero`, `nullity_le_14_exact_minimum`, `nullity_gt_14_counts_approx`, `earliest_cut_wins_tie`, `cut_file_parse_rules`, `samples_pinned_seed_12345`, `memo_equals_full_scan`, `parallel_equals_serial` |
| `repair/*.rs` | `to_from_seq_merges_and_drops_markers`, `fix_iteration_restarts_from_to_seq` (adjacent markers created by iteration 1 change a slide in iteration 2), `cancel_dead_gates_still_block`, `cancel_reaches_fixed_point`, `slide_no_move_when_only_markers_before_blocker`, `slide_lands_before_markers_forward_after_backward`, `fix_breaker_located_before_completer_moves`, `fix_uses_stale_gate_count_cut`, `merge_keeps_input_order`, `split_candidates_and_smaller_tie`, `split_never_splits_two_gate_windows`, `split_forced_when_no_clean_candidate`, `oracle_loop_break_rules`, `fix_exhausted_without_rescan`, `single_window_skips_scans` |
| `serialize.rs` | `blob_23_byte_example`, `constant_monomial_first`, `crc32_matches_zlib_vectors`, `container_header_fields`, `decode_requires_explicit_format`, `decoder_rejects_each_invariant_violation`, `decode_eval_equals_gates`, `emission_body_keys_bit_ge_61_and_ge_128` (recorded 3.9 bytes) |
| `verify.rs` | `zero_slice_detects_single_flip`, `equiv_detects_noncommuting_swap`, `exhaustive_window_detects_rare_difference` |
| `types.rs` | `exit_codes_and_meanings_are_a_bijection` |

### 10.3 Committed fixtures: `tests/fixtures/stage4/` (total ≤ 4 MB)

| path | content |
|---|---|
| `README.md` | provenance: worktree commit; baseline HA sha256; oracle interpreter and its self-test result; seeds marked **CALIBRATION, never deliverables**; the `regen.sh` command; sizes; the coverage table produced by `regen.sh` |
| `regen.sh` | regeneration (below) |
| `pyset_selftest.py` | replays both trace files under the oracle interpreter; exit 0 iff every order matches |
| `pyset39_traces.txt` | about 100 KB: random add/discard/copy/xor/literal traces with keys of 0..1024 bits (many with bits ≥ 61, ≥ 128, ≥ 256), with the iteration order after each step |
| `pyset39_large.txt` | ≤ 10 KB. Large traces are **regenerated** from a recorded op generator (`PyRandom` seed per trace): mul-style toggle loops growing to 2×10^5, 4×10^5 and 6×10^5 elements (the `used > 50000` branch, dummy-heavy tables of 2^19 to 2^21 slots), and patterns that force the small-table purge. Only the sha256 of the iteration order at fixed checkpoints is stored |
| `n8_L8/` | real chain at n=8, C and D of 16 gates, preset L8: `g2.mpmct1`, `g3.mpmct1`, `c.g57`, `params`, and the oracle outputs `packed.mpmct1`, `packed.bounds`, `pack.txt`, `rep.mpmct1`, `rep.bounds`, `rep.{exposure,allcuts,wit}.tsv`, `repair.txt`, `deliverable.bin` |
| `crafted/<case>/` | synthetic inputs from the committed generator `tests/stage4/synth.py` (seeded), each with the oracle outputs as above. The cases are listed below |
| `vA64a_window/` | the gate slice of the real ambiguous window vA64a `[492039,492540)` (packed order) with the expected abort or no-abort decision of `fits` and `window_anf` under `CPython39` and under `Canonical`. CALIBRATION material (31-bit seed run); no C, no G2 |

Crafted cases:

- `caps_aborts`: production n8 inputs with small caps (pack `32 32 48 300`) **plus** a crafted G3 tail that forces order-dependent compose aborts;
- `split_forced`: small split budget with a planted leak at every candidate;
- `fix_exhausted`;
- `merge`;
- `single_window`;
- `wide300`: a synthetic 300-wire G2/G3/C (W=8, wire ids ≥ 61 and ≥ 256);
- `wide128`: n=32, state 128, W=2.

The crafted G3s are G2 with random legal commutations, inserted cancelling pairs, and planted copy/uncopy gadgets that expose chosen C values at chosen cuts.

**L8 at small n is proven to run** (`port_runs/gen_feas`, 2026-10-08, calibration seeds):

| n | C gates | G2 gates | state wires | gen log |
|---|---|---|---|---|
| 8 | 16 | 18 238 | 32 | `[chain] k=2: 1734 brackets chained`; 160 transfers, 16 ports restored |
| 16 | 64 | 65 537 | 64 | |

Stage 3 at n=8 on the local store is proven as the first step of P6. If it fails, the n8 case moves to n=16 with `|C|=16`.

**`regen.sh`.**

1. Verifies the baseline HA sha256 and runs `pyset_selftest.py` under `$ORACLE_PY`.
2. Builds `gen_sandwich_gadget`, `circuit_mixer` and `tdp_packing` from the worktree into `$CARGO_TARGET_DIR`.
3. Copies `scripts/windowed_delivery/` into a fresh sandbox `port_runs/fixture_<stamp>/`. With `sed`, each rewrite asserted to apply **exactly once**, it rewrites:
   - `ROOT`, `RD`, `W`, the `sys.path.insert` lines and the `target/release/hmap_affine` path (to `$HMAP_BASELINE`);
   - `python3` (to `$ORACLE_PY`);
   - the split budget literal `65536` at `repair.sh:31` (to `$SPLIT_BUDGET`);
   - the two `for it in 1 2 3 4` loops (to `$FIX_ITERS` and `$FALLBACK_ITERS`);
   - the pack arguments.

   The tracked oracle is never edited.
4. Runs stages 1-3 for `n8_L8` (L8 recipe, local regular store, fixed calibration seeds), and `synth.py` for the crafted cases.
5. Runs the oracle stage 4 and `tdp_packing` on every case and compares the outputs (§10.4).
6. **Coverage assertions.** Across the fixture set, from the oracle logs and both sides' `RepairReport`/`PackReport` counters, all of the following must hold:
   - cancel passes ≥ 2 in at least one case;
   - fix moved > 0 in ≥ 2 iterations of one case;
   - `fix_exhausted` in one case;
   - merge removed > 0;
   - ≥ 1 clean split;
   - ≥ 1 forced split (oracle `WARNING` line);
   - one `single_window` case;
   - ≥ 1 packer abort and ≥ 1 split decision with `order_dependent > 0`. This is cross-checked once against the instrumented oracle copies in `port_runs/spec_pack3`, which report CPython-versus-sorted disagreements.

   If an assertion fails, `regen.sh` searches crafting seeds up to a bound and then fails loudly. The seeds found are recorded in `README.md` as calibration-only.
7. Copies the outputs into the fixture tree, prints the coverage table and the sizes, and asserts the 4 MB budget.

### 10.4 Integration tests (CI): `tests/integration/tdp_packing.rs`

Add `[[test]] name = "tdp_packing"`. It uses `env!("CARGO_BIN_EXE_tdp_packing")`.

**Comparison rule.** Circuits, bounds, TSVs and `deliverable.bin` are compared **byte for byte**. Text logs (`pack.txt`, `repair.txt`, `cview.log`) are compared after normalization: the absolute sandbox or out-dir prefix becomes `<DIR>`, and `:.1f` ties are tolerated (D-c).

Tests:

- `pack_matches_oracle_<case>` and `repair_matches_oracle_<case>` for every fixture case;
- `serialize_matches_oracle_<case>`;
- `run_end_to_end_n8` (exit 0, `status=ok`);
- `vA64a_window_decisions` (both `CapSemantics`);
- `thread_count_does_not_change_output`;
- `single_window_case`;
- `exit_codes`:
  - a malformed gate line gives 3;
  - a G3 with one gate flipped gives 4 (from the reorder certificate);
  - a planted interior-visible value gives 5, and with `--allow-floor-miss` gives 0 with `status=floor_miss`;
  - a tiny split budget with a 1-gate over-budget window gives 6, and with `--allow-over-budget` gives 0;
  - an unwritable out-dir gives 7.

### 10.5 Real-artifact parity (manual): `tests/manual/stage4/*_parity.sh`

As built, the plan below is split over five scripts, one per component: `cview_parity.sh` (step 0 and the HA comparisons), `pack_parity.sh` (step 1; `pack_replay.sh` for step 6), `repair_parity.sh` (step 2), `anf_parity.sh` (steps 3 and 5) and `run_parity.sh` (the whole `tdp_packing run`); their results are recorded in §12. They read `waist/<tag>/` read-only, write under `port_runs/`, and never run in CI. Time and peak RSS are measured with `/usr/bin/time -l` and recorded, not asserted (step 7). For each of the 8 tags:

0. **Provenance of the archived C-view.** Run the baseline HA and the archive-era HA (§10.1) on the archived `rep512.mpmct1` + `rep512.bounds`, and compare both with the archived `rep512.{exposure,allcuts,wit}.tsv` and line 2 of `rep512.cview.log`.
   - **If the baseline reproduces the archive:** the targets are Appendix B.
   - **If it does not (fallback, planned):** re-run the Python repair and serialization (`repair.sh` + DE, sandboxed, `$ORACLE_PY`, baseline HA) on the archived `packed_G512.mpmct1` + `.bounds`, into `port_runs/retarget_<tag>/`. Those outputs become the parity targets, and their sha256s are added as Appendix B′. The report lists the diff against the archived `rep512`/`deliverable.bin`, including the visible count per iteration and whether the final visible set changes. The pack targets are unaffected, because the packer does not use HA.

   Then compare the refactored HA with the baseline on rep512, on packed + pre-repair bounds, on a split-candidate file, without ALLCUTS, on SPAN and on a small heatmap run: `cmp` every file and stdout.
1. `tdp_packing pack tdp.mpmct1 s3.mpmct1 512 512 24576 3000 …`: the sha256 of `packed_G512.mpmct1` and `.bounds` must equal Appendix B.
2. `tdp_packing repair` on `packed_G512` gives a byte-identical `rep512.mpmct1` and `rep512.bounds`, and TSVs identical to the targets of step 0.
3. `tdp_packing serialize`: the body sha256 equals the target `deliverable.bin`. `container_header.py` rebuilds the container from the oracle body, and it must be `cmp`-equal to `deliverable.tdp4`.
4. `check_zero_slice` of the archived `deliverable.bin` against C, and `tdp_packing check-final` on the archived rep512, which reports the final identity and budget status.
5. **Census parity (gated):** `tdp_packing census` output against the archived `link_rep512.txt` (all lines), and `anf_size_report` against the first three lines of `anf_rep512.txt`.
6. Replay the window ANF of `packed_G512` and its bounds on vA64a and vL128b: every compose abort is reproduced (2 and 6), with `order_dependent` counts equal to the instrumented oracle's.
7. Record times and peak RSS (`/usr/bin/time -l`) and assert the §9 targets. Include a dedicated memory run of `fits` on the vA64a ambiguous window.

### 10.6 CI differential test of the oracle (new CI job `stage4-oracle`)

Stage 4 needs no frozen store, so the oracle itself can run in CI. The job:

1. `actions/checkout` with `fetch-depth: 0`. Installs uv and `uv python install 3.9.6`, then runs `pyset_selftest.py`. **The job fails if the self-test fails**, which pins set semantics, not the version string.
2. Builds the baseline HA from `492564f0` in a `git worktree`, caches it by commit, and checks it against the recorded sha256. The sha256 is recorded per platform: `README.md` holds a Linux and a macOS value.
3. Builds `tdp_packing` (release).
4. Runs `tests/stage4/differential.sh`:
   - every committed fixture input;
   - **K = 20 random small triples** from `synth.py`. The seeds are derived from the CI run id, printed and labelled calibration, and span n ∈ {4, 6, 8, 12}, random caps in and around the ambiguous zone, and random budgets.

   For each, it runs the sandboxed oracle (pack, `repair.sh` with rewritten literals, DE) and `tdp_packing`, and `cmp`s every artifact under the §10.4 rule. On a mismatch it uploads the failing triple as a CI artifact.

This replaces revision 1's claim that "CI cannot run the Python stage-4 oracle". Rust/oracle parity is then exercised on fresh random inputs on every push, not only on frozen fixtures.

### 10.7 Reproduction test (required acceptance): `tests/manual/stage4/reproduce_val_chain.sh`

Usage: `reproduce_val_chain.sh N [L8|A]`. MC defaults to the §2.4 table and is required for other n.

1. Draw `GEN` and `MIX` from the CSPRNG (calibration values). Keep them in the private `port_runs/repro_<stamp>/SEEDS` (format 2) and never echo them.
2. **Reference side.** A sandbox copy of `W/val_chain.sh`, with the paths rewritten as in `regen.sh` and HA set to the **baseline**. Its two seed lines are replaced by `SEED=$VAL_GEN_SEED` and `MSEED=$VAL_MIX_SEED` (each asserted to match once). It runs under `env -i HOME PATH VAL_* FROZEN_DB_DIR`, with arguments `TAG N MC sandwich_default_s(N) 20N CFG` and `python3` set to `$ORACLE_PY`.
3. **Managed side.** `local_mixing_bin tdp_gen --config <generated v2 toml>` with:
   - `source.wires = N` (and `source.gates` only when N is not in the table);
   - `[calibration] enabled = true` and the seed file;
   - **nothing else**, so preset L8 or A and every other setting is at its default.

   The harness exports a set of decoy variables (`MIXER_DB_LEN_BAND`, `SLICE_GUARD_LEGACY`, `EMBEDDED_MASKING_RETIRE`, …) in the caller and expects the driver to refuse them (§11.5). It then reruns with them removed.
4. Compare:

   | reference | managed |
   |---|---|
   | `tdp.mpmct1`, `.source_c.g57`, `.sandwich.mpmct1` | the same files |
   | `s3.mpmct1` | `db_mixing.mpmct1` |
   | `packed_G512.mpmct1(.bounds)` | `packing/packed_G512.mpmct1(.bounds)` |
   | `rep512.mpmct1`, `rep512.bounds` | `packing/rep512.*` |
   | `rep512.{exposure,allcuts,wit}.tsv` | `packing/` copies |
   | `deliverable.bin` | `packing/deliverable.bin` and `packing/deliverable.tdp4[48:]` |
   | `link_rep512.txt`, `anf_rep512.txt` head | `packing/` copies |
   | `span_s2.txt`, `mult_s2.txt` | `stage2_span.txt`, `stage2_mult.txt` (§11.1) |

   `pack_G512.txt` is compared line by line after normalization.
5. **Exits 5 and 6.** The managed side may exit 5 or 6 where the oracle shipped (count floor met but identity failed, or an over-budget 1-2 gate window kept). Then:
   - every pre-decision artifact (packed, rep, bounds, last-scan TSVs) must be identical;
   - `tdp_packing check-final` on the **oracle's** rep must report the same violation;
   - the harness reruns stage 4 alone with the matching `--allow-*` flag, and `deliverable.bin` must equal the oracle's.

   That outcome counts as a pass, labelled `faithful (failure path)`. Any other nonzero exit is a failure.
6. Required runs: `N=16 MC=64 L8`, `N=16 MC=64 A`, `N=64 L8` (MC 1000 by default, about 40 min per side). The script prints a summary table, including the stage-3 wall-time ratio (§9), which RC keeps in the PR description. The table contains no seeds and no hashes of private files.

### 10.8 Historic anchor (required): `tests/manual/stage4/anchor.sh`

The anchor proves that the managed defaults reproduce the **chosen runs themselves**, not just today's `val_chain.sh`.

**A. Pre-P7 binary check, cheap and done first.** Build `gen_sandwich_gadget` at `492564f0`, which includes `d9631d0e`. For all 8 tags, run gen with the archived seeds (read from `seed.txt` in a subshell, never printed) under `env -i` and the tag's CONFIG environment, and `cmp` with the archived `tdp.mpmct1`, `.source_c.g57` and `.sandwich.mpmct1`.

- `d9631d0e`'s message argues that generated sources never contain zero-control gates, so no bytes should change. This step proves or refutes that.
- Then run `circuit_mixer` at `492564f0` on the archived `tdp.mpmct1` of vL64a and vL128a with the archived mix seed, and `cmp` with `s3.mpmct1`. This covers the Oct 8 mixer cleanup commits `8070f5a0..b5608cd9`, which re-baselined goldens.

**B. After P7.** Run the managed driver with the archived seeds (calibration seed file, default size table) for **vL64a and vL128a**, and `cmp` every artifact against the archive: `tdp.mpmct1`, `s3.mpmct1` vs `db_mixing.mpmct1`, `packed_G512*`, `rep512.*`, and `deliverable.bin` (or the Appendix B′ targets if §10.5 step 0 fell back).

**If A or B mismatches.** Bisect over `e720d5c0..492564f0` (the run binaries were built from `e720d5c0`) to the responsible commit, and record it in §2.1 and in TDP doc §4.7. State plainly that the chosen runs are no longer reproducible by the current binaries. The **new reference** is then the oracle chain run with the archived seeds on the current binaries, regenerated into `port_runs/anchor_<tag>/` and with sha256s recorded as Appendix B″. RC decides whether to re-validate (L2/M2 and floor) on it. P7 does not start until A is resolved.

### 10.9 CI-sized version of the reproduction claim

1. **`tests/shell/tdp_defaults.bash`** (mocks). This is a transcript diff against the **oracle script itself**, not a hand transcription:
   - Run a path-rewritten sandbox copy of `val_chain.sh` (the `regen.sh` rewrite rules) with `target/release/{gen_sandwich_gadget,circuit_mixer,hmap_affine}` replaced by recording mocks, and the stage-4 Python scripts replaced by recording stubs.
   - Run `tdp_gen.sh` (and, in a second case, `local_mixing_bin tdp_gen` with a v2 TOML) with the same seeds against the same mocks.
   - Each mock records its argv as an option set (positional order kept) and its **complete** environment.
   - The test asserts:
     - (a) gen argv identical;
     - (b) mixer argv identical as an option set;
     - (c) for every **output-affecting** registry variable (§11.5), the effective value (unset ≡ the registry's built-in default) is identical;
     - (d) the managed mocks' environment is exactly allow-list ∪ recipe ∪ driver-set;
     - (e) the stage-4 stub's WP/repair/DE arguments map to `tdp_packing run` parameters that are all at their defaults (`PACKFLAGS` empty).
   - A literal expected transcript is kept as a secondary pin.
   - Cases: L8, A, G values where `4·G·1.3` has a fractional part ≥ .5 (floor versus round), n=64 and n=128 default sizes, n=32 without size (fatal), and one case per forbidden variable exported in the caller (fatal with a message naming it).
2. **`tests/shell/tdp_stage12_parity.bash`** (real debug binaries). At n=16 with `|C|=|D|=32`, run gen exactly as `val_chain.sh` would (sandboxed env and argv) and `tdp_gen.sh --run-stop-after-stage 2 --seed-file …`, then `cmp` the three artifacts.
3. **`tests/integration/tdp_packing.rs`** (§10.4) and the **`stage4-oracle` job** (§10.6).
4. **`tests/shell/tdp_reproduce_small.bash`:** the full §10.7 at `N=8 MC=16` for real. It skips with a clear message when there is no frozen store (as in CI); the CI job covers stage 4. RC runs it locally before merging.
5. **`cache_neutrality`** (planned, never run; see §2.2 D10b for the evidence used instead) (local, once per P7): n=16, same seeds, with `LOOKUP_CACHE_MB=0 CANON_CACHE_MB=0 XPOLY_CANON_CACHE_MB=0` versus the driver values. `db_mixing.mpmct1` must be identical.
6. **`filter_neutrality`** (once, on a host whose store has `filters.bin`; recorded in §2.2 D11): same seed with `FROZEN_FILTER=0` and `=1`, and `db_mixing.mpmct1` must be identical. Until this is recorded, `auto` never loads a filter.
7. **`tests/shell/tdp_seeds.bash`:**
   - malformed `SEEDS` (missing key, non-decimal, out of range, short `run_key`) is fatal;
   - a one-line legacy seed file is rejected with the migration text;
   - `-s` is rejected;
   - a gen mock whose banner reports a different seed is fatal;
   - no seed value appears in `tdp_gen.log`, `stage12.log`, `stage3.log`, stdout or the dry-run output;
   - no seed value is on any stage binary's raw argv: gen gets `@/dev/fd/3..5`, the mixer `--seed-file /dev/fd/3` (the mocks resolve them and record both forms).
   - Every assertion is its own command (errexit ignores all but the last command of an `&&` list and any `!`-negated command).
8. **`tests/shell/tdp_stage4_resume.bash`:**
   - a resume after an `--allow-floor-miss` run does not print `PIPELINE COMPLETE`;
   - regenerating stage 3 forces stage 4;
   - a changed packing parameter forces stage 4;
   - a tampered `packing.recipe` forces stage 4;
   - exits 5, 6 and 7 are reported with their meanings;
   - every check is a plain command or an explicit `if … then exit 1` (not `!` or `&&`, which errexit does not enforce).

The CI workflow adds:

- `--test tdp_packing` and `--test stage2_typed_params` (gen's L8/A output bytes pinned to the validated binary; the only CI guard of stage-2 bytes, since `tdp_stage12_parity.bash` compares one binary with itself) to "Runtime and command correctness";
- `tdp_defaults.bash`, `tdp_stage12_parity.bash`, `tdp_seeds.bash` and `tdp_stage4_resume.bash` to "TDP script flags and block sizing";
- `--features security-tools --test hmap_affine_exposure` to "Trace-affine heatmap". This test runs the **refactored** HA on the n8 fixture and compares with TSVs produced by the **baseline** HA;
- the new `stage4-oracle` job (§10.6).

---

## 11. Driver integration

### 11.1 `scripts/tdp_gen.sh`

**Flags.**

- `--preprocessing-preset L8|A|custom`, default L8. Not applicable to `nonlinear291`: an explicit preset there is fatal.
- `--source-gates N` (alias `-m`) and `--source-gates-rule validated|formula`.
- `--seed-file PATH`.
- `--packing-window-gates`, `--packing-split-budget`, `--packing-anf-cap-semantics`, `--packing-allow-floor-miss`, `--packing-allow-over-budget`.
- The stop/rerun range is `2..4`; `5` and `6` keep the "historic" error.

**Environment discipline** (§11.5).

- At start, the script checks the inherited environment against the registry. Any inherited **output-affecting** variable is fatal, with the message "X is not honoured by the managed recipe; unset it (use flags/config)". This includes every `EMBEDDED_MASKING_*`, `CANON_RULE_L_BRANCH_CAP`, `CANON_MONOMIAL_CAP`, `MIXER_DB_LEN_BAND`, `MIXER_STOP_AT_*`, `SLICE_GUARD_*`, `TWIST_G57_*`, `MIN_DIR_LOOKUP`, `PROD_*`, `SAT_*`, `DB_QC*`, `TDP_SOURCE_C` and `SANDWICH_VARIANT`.
- **This replaces today's inherited-default semantics.** `${EMBEDDED_MASKING_K_ARG:-${EMBEDDED_MASKING_K:-2}}` becomes `${EMBEDDED_MASKING_K_ARG:-2}`. Flags remain the way to set custom values (preset `custom`). `tdp_flags.bash`'s `EMBEDDED_MASKING_K=1 reject` and `SHUFFLING=8 RETURN_HOME=false reject` cases still reject, now with the inherited-variable message. The `precedence` case (line 87) becomes "inherited values are rejected even when flags are given".
- Every stage binary is launched through one helper, `stage_exec`, as `env -i "${ALLOW_ENV[@]}" "${RECIPE_ENV[@]}" "$BIN/…" …`:
  - `ALLOW_ENV` = `PATH HOME TMPDIR LANG=C LC_ALL=C`, plus `RAYON_NUM_THREADS` only when the config pins threads;
  - `RECIPE_ENV` is built per stage from the registry.

**Stage 2.**

- The preset sets `RECIPE_ENV`: `SHUFFLING=8`, `SHUFFLING_RETURN_HOME=0`, `PREMASK=1`, `SEED_AVOID_C1=1`, plus for L8 `CHAIN=2` and `CHAIN_LEN=8`. Nothing else, so K, MAX_OPEN, MIN_OPEN, BALANCED and QUAD_FIRE stay unset.
- Under preset `custom`, the flag values are exported only when they differ from gen's base parameters, and every value is recorded in the marker.
- Under `nonlinear291`, no `EMBEDDED_MASKING_*` variable is exported.
- The `return_home=0` rejection (`tdp_gen.sh:257-260`) is removed. Persistent migration restores the ports at the stage-2 tail.
- Gen call (D14: no mode positional for embedded-masking): `stage_exec "$GADGET_BIN" "$GADGET" "$N" "$M_CD" "$M_CD" "$S_SL" 1 "$SLICE_G" "$GEN_SEED" "$GEN_SEED" "$GEN_SEED"`.
- After gen: the banner seed check (§8.4).
- **Side reports**, report-only and parallel with stage 3, written to `$RUN`:
  - `stage2_mult.txt`, from `mult_census.py` under `python3 -I` (copied into `scripts/tdp_reports/`, unchanged);
  - `stage2_span.{tsv,txt}`, from `hmap_affine` SPAN mode with val_chain's arguments, **when a `security-tools` build of `hmap_affine` is present**; otherwise the note "span_s2 skipped (hmap_affine not built)".

  A failure of either is a note, never fatal.
- Stage-12 marker `tdp_stage12_recipe=7`: adds `preprocessing_preset`, every effective stage-2 knob, `gadget_seed_rule=gen`, `seeds_mac` and `source_gates_rule`.
- **If stage 2 actually runs** (fresh, forced, or a missing artifact), then `FORCE_FROM=min(FORCE_FROM,3)`.

**Stage 3.**

- `G_IN` comes from the header field.
- The profile comes from the schedule. The `validated` schedule is fixed at `2,2,2,2,2` (n0=2, hold 0, expand 2). The `custom` schedule takes `growth/hold/expand`.
- `A_MOVES = int(n2*r1*G*1.3)`, computed with Python `int()`.
- `--seed "$MIX_SEED"`.
- `--verify-every 2000000 --report-every 100000 --state-out …` are deleted.
- `CANON_RULE_L_BRANCH_CAP` and `CANON_MONOMIAL_CAP` are no longer exported. The three cache sizes stay (D10b).
- `FROZEN_FILTER` follows D11:
  - `auto` = `1` when `$FROZEN_DB_DIR/filters.bin` is absent; when it is present, `0` until filter neutrality is recorded, then `1` iff available RAM ≥ 60 GiB (Linux `MemAvailable`; macOS `vm_stat` free + inactive + speculative pages × page size);
  - `on` and `off` force it.
- `MIXER_STOP_FLAG` and `MIXER_DUMP_FLAG` are driver-set paths (no RNG effect when the files are absent) and stay.
- **If stage 3 actually runs**, then `FORCE_FROM=min(FORCE_FROM,4)`.

**Stage 4.** A completion marker makes resume safe:

```bash
PACK_DIR=$RUN/packing; MARKER=$RUN/packing.recipe
expected_marker() {   # params + MACs of the three inputs, keyed by run_key; never unkeyed hashes
  python3 -I "$LIB/packing_marker.py" expect "$RUN/SEEDS" "$GADGET" "$DB_MIXING" "$GADGET.source_c.g57" \
          "$BIN/tdp_packing" "${PACKING_RECIPE[@]}"
}
stage4_done=0
if [ "$FORCE_FROM" -gt 4 ] && [ -s "$MARKER" ]; then
  if python3 -I "$LIB/packing_marker.py" check "$MARKER" "$PACK_DIR" < <(expected_marker); then stage4_done=1
  else note "stage 4: completion marker does not match inputs/parameters/deliverable; re-running stage 4"; fi
fi
if [ "$stage4_done" -eq 0 ]; then
  rm -f "$MARKER"; rm -rf "$PACK_DIR"
  note "stage 4: tdp_packing run (G=$PACK_G, split budget $SPLIT_BUDGET, caps $CAP_SEM)"
  set +e
  stage_exec "$BIN/tdp_packing" run --stage2 "$GADGET" --stage3 "$DB_MIXING" --source-c "$GADGET.source_c.g57" \
      --n "$N" --out-dir "$PACK_DIR" ${PACKFLAGS[@]+"${PACKFLAGS[@]}"} > "$RUN/stage4.log" 2>&1
  rc=$?; set -e
  if [ "$rc" -ne 0 ]; then
    note "stage 4 FAILED: exit $rc ($(stage4_exit_meaning "$rc")); see $RUN/stage4.log"; exit "$rc"
  fi
  status=$(kv status "$PACK_DIR/packing.report")
  # atomic: write .tmp then mv; records status, params, input MACs, MAC of deliverable.tdp4
  expected_marker | python3 -I "$LIB/packing_marker.py" write "$MARKER" "$PACK_DIR" "$status"
fi
status=$(kv status "$MARKER")
if [ "$status" = ok ]; then
  note "PIPELINE COMPLETE (stages 1-4): $PACK_DIR/deliverable.tdp4"
else
  : > "$PACK_DIR/NOT_RECOMMENDED"
  note "PIPELINE FINISHED WITH AN ALLOWED FAILURE (status=$status): $PACK_DIR/deliverable.tdp4 is NOT a recommended deliverable"
fi
```

- `PACKFLAGS` is empty at the defaults.
- `stage4_exit_meaning` mirrors `Stage4Error::meaning`, and a unit test checks that the two tables agree.
- `packing_marker.py` (new, in `scripts/tdp_reports/`) computes HMAC-SHA256 with `run_key`.
- The marker is written **only after exit 0**, and `status` is either `ok` or `allowed_floor_miss`/`allowed_over_budget`.
- On a skip path the marker is **re-checked** against the current inputs, parameters, binary and deliverable.
- `tdp_packing` is added to the binary presence check, and the "stage 4 is not run" note is deleted.

### 11.2 `src/tdp` (managed path): config schema v2

**`config_version = 2` is required.** A v1 file fails with this migration message:

> config_version 1 is not supported by recipe v10: defaults now reproduce the chosen L8 runs. Remove shuffling_segments, shuffling_return_home, hold_work_units and stop_after_stage = 3 copied from the old template; use preprocessing.preset / db_mixing.schedule; seed files need gen_seed=/mix_seed= lines. See docs/STAGE4_RUST_DESIGN.md §11.2.

**Authoritative presets.**

- `preprocessing.preset = "L8" | "A" | "custom"`, default `"L8"`.
  - Under L8 or A, **any** of `mask_pair_wires`, `max_open_masks`, `min_open_masks`, `balanced_masks`, `shuffling_segments`, `shuffling_return_home`, `premask`, `seed_avoid_c1`, `chain`, `chain_len` that is present is an error, even when it equals the preset: "field X is set by preset L8; remove it or use preset = \"custom\"".
  - Under `custom` they are all required.
  - With `mode = "nonlinear291"`, `preset` must be absent.
- `db_mixing.schedule = "validated" | "custom"`, default `"validated"` (profile 2,2,2,2,2, moves `int(4·G·1.3)`).
  - Under `validated`, the fields `growth_work_units`, `hold_work_units` and `target_size_factor` must be absent.
  - Under `custom` they are required. `growth_work_units` is new in v2 (it is the profile's n0), and `hold_work_units` keeps its meaning only under `custom`.

**Other fields.**

- `source.gates`: optional, default per §2.4. `source.gates_rule = "validated" | "formula"`, default `validated`.
- `[packing]`: `window_gates=512`, `interval_cap=512`, `monomial_budget=24576`, `span_cap=3000`, `split_budget=65536`, `fix_iterations=4`, `fallback_iterations=4`, `anf_cap_semantics="cpython39"`, `allow_floor_miss=false`, `allow_over_budget=false`. The section is added to the list at `parse.rs:255-266`. Any non-default value is recorded as `packing_params=custom`.
- `[calibration] seed_file`: format 2 (§8.2).
- `run.stop_after_stage`: default 4, range 2..4.
- `run.threads`: optional; when present it pins `RAYON_NUM_THREADS` and `--threads`.

**`config/validate.rs`.**

- Delete the `return_home` rejection.
- Validate the preset and schedule exclusivity rules.
- n ≤ 256 when stop_after_stage is 4.
- Packing integers positive; `monomial_budget` and `split_budget` at most 2^18 (a resource bound, `MAX_MONOMIAL_BUDGET`, shared with `tdp_packing`); with `preprocessing.mode = "nonlinear291"` and stage 4 on, the planned stage-2 width (`nonlinear_tdp_resource_plan`) at most 1024 state wires (review round 3).
- The `2*n > K + balanced` rule stays.
- `read_calibration_seed` becomes `read_seed_file`: format 2 only, with the §8.3 legacy message.

**`ResolvedConfig`** gains:

- `preprocessing_preset: Preset { L8, A, Custom(Stage2Knobs) }`;
- `schedule: Schedule { Validated, Custom { growth, hold, expand } }`;
- `source_gates_rule`;
- `packing: PackingConfig`;
- `seeds: SeedPlan { Random | Explicit(path) }`.

**`runner.rs`.**

- Resolve the seeds and create `$RUN/SEEDS` (create_new, 0600) **before** `prepare_recipe_manifest` (§8.2).
- `configure_environment` is **allow-list based**: `command.env_clear()`, then the inherited `ALLOW_ENV` variables (PATH, HOME, TMPDIR), `LANG=C`/`LC_ALL=C`, `TDP_BIN_DIR`, `FROZEN_DB_DIR` (when a store is set) and `TDP_GEN_ALLOW_EMPTY_STORE=1` (plumbing tests). Nothing else: every recipe setting (preset and knobs, schedule, packing parameters, filter mode, pieces, leakage repair) travels as a `tdp_gen.sh` flag (`script_args`), and bash builds each stage binary's environment itself (`stage_exec`: `FROZEN_FILTER` per D11, `FROZEN_REGULAR_VALUE_CONVENTION`, the cache sizes, the `EMBEDDED_MASKING_*` preset variables, `TDP_SOURCE_C` when `source.path`).
- Before launching, the inherited environment is checked against the registry and refused with the same message as bash.
- `render_command` redaction per §8.4.
- `script_args`: `--preprocessing-preset`, `--source-gates-rule`, the `--packing-*` flags. No seed flags.
- Map `tdp_gen.sh`'s exit codes to messages via the shared table (`Stage4Error::meaning` for 2-8 and 101).
- The build and dry-run lists add `--bin tdp_packing`. Delete the "not driven yet" message.
- A fresh run with `build_binaries = true` also builds the report-only `hmap_affine` in a separate `cargo build --features security-tools --bin hmap_affine` (same target directory and triple, so the feature never reaches the production binaries). `tdp_gen.sh` runs val_chain.sh's stage-2 span check with it (`$RUN/stage2_span.{txt,tsv}`); without it the report is skipped with a note.

**`paths.rs`.** Stage-4 prerequisites for `rerun_from_stage = 4`.

**`manifest.rs`.**

- `CURRENT_RECIPE_VERSION = 10`. `read_recipe_version` accepts only 10; v9 gives "recipe v9 run: resume with the checkout that created it".
- `production_binary_fingerprints` adds `tdp_packing`; `run_binary_fingerprints` adds `hmap_affine` (`binary_hmap_affine_xxh3`) when it is in the binary directory, and the manifest records `stage2_span_report=hmap_affine|skipped`.
- `frozen_filter=<mode>(<resolved>)`: the `FROZEN_FILTER` value stage 3 will use, resolved as `tdp_gen.sh` does (`auto`: 1 without `filters.bin`, 0 with one while no neutrality comparison is recorded; `unset` without a store). The script records the value it actually set in `$RUN/stage3.settings`.
- `pinned_recipe` is derived, not constant: `val_chain-L8-20261007` or `val_chain-A-20261007` only for embedded masking with preset L8/A, the validated |C| for a generated source, schedule validated in one piece, `FROZEN_FILTER` resolving to 1 on a real store, no leakage repair and default packing; otherwise `none`.
- The manifest prints **resolved** values only, which removes the `hold` "30" text (D18):

```
preprocessing_preset=L8
preprocessing_shuffling=8  preprocessing_return_home=0  preprocessing_premask=1
preprocessing_seed_avoid_c1=1  preprocessing_chain=2  preprocessing_chain_len=8
gadget_seed_rule=gen
db_mixing_schedule=validated  stage3_profile=2,2,2,2,2  stage3_moves_rule=int(n2*r1*G*1.3)
stage3_verify_every=default(10000)  stage3_report_every=default(50000)
source_gates=1000  source_gates_rule=validated|explicit|formula
seed_record=SEEDS  seed_source=random|explicit  seed_bits=63  calibration_only=false
env_policy=allowlist-v1  cache_mb=lookup:2048,canon:256,xpoly:1024  canon_caps=unbounded
frozen_filter=auto(<resolved>)        # e.g. auto(1); stage3.settings records the value set
packing_params=default  packing_window_gates=512  packing_interval_cap=512  packing_monomial_budget=24576
packing_span_cap=3000  packing_split_budget=65536  packing_fix_iterations=4  packing_fallback_iterations=4
packing_anf_cap_semantics=cpython39  packing_allow_floor_miss=false  packing_allow_over_budget=false
pinned_recipe=val_chain-L8-20261007   # or val_chain-A-20261007, or none (derived)
stage2_span_report=hmap_affine|skipped
```

**`configs/tdp.toml`** becomes a v2 template. `wires = 128`, and no stage-2 or stage-3 knob appears uncommented. A comment says: "defaults reproduce val_chain.sh CONFIG=L8 as run for vL64a/b, vL128a/b (docs/STAGE4_RUST_DESIGN.md §2); source.gates defaults to 1000 (n=64) / 3000 (n=128)". `lookup_miss_filter` documents the D11 rule.

### 11.3 Test updates

**`tests/unit/tdp.rs`:**

- stop/rerun 4 accepted, 5 and 6 rejected;
- `return_home=false` accepted under `custom`;
- the exact default manifest text;
- three mock binaries; a v10 fresh run.

New tests:

- `v1_config_rejected_with_migration`;
- `old_template_with_preset_l8_rejected` (the v1 template text with `config_version` bumped to 2 is rejected for its explicit knobs);
- `preset_field_conflict_rejected`;
- `schedule_field_conflict_rejected`;
- `preset_rejected_for_nonlinear291`;
- `default_source_gates_table` (64 → 1000, 128 → 3000, 32 → error, 32 + formula → 1280 recorded `formula`);
- `default_profile_is_2_2_2_2_2`;
- `moves_floor_not_round`;
- `preset_a_drops_chain`;
- `seeds_file_created_before_manifest`;
- `seed_file_format_1_rejected`;
- `all_seed_flags_redacted`;
- `seeds_never_in_narrative_log`;
- `env_is_allowlist_only`;
- `inherited_output_affecting_var_rejected` (one per registry class);
- `env_registry_covers_every_env_read` (§11.5);
- `exit_code_tables_agree`;
- `v9_manifest_rejected_with_message`.

**`tests/shell/tdp_flags.bash`:**

- lines 40-41: `return_home 0` accepted under `--preprocessing-preset custom`;
- the inherited-variable cases now expect the inherited-variable message;
- the mock key list at line 64 expects exactly the preset's variables;
- new cases: an explicit preset with `nonlinear291` is fatal, and `nonlinear291` exports no `EMBEDDED_MASKING_*`.

**`tests/shell/tdp_block_size.bash`:** lines 35-40 reject only 5 and 6; line 89 has three stage calls; line 96 is the "PIPELINE COMPLETE (stages 1-4)" line; add a `tdp_packing` mock. Callers that relied on the derived source size pass `-m`.

**New:** `tdp_defaults.bash`, `tdp_stage12_parity.bash`, `tdp_seeds.bash`, `tdp_stage4_resume.bash` and `tdp_reproduce_small.bash` (§10.9).

**Gen strict parsing (D17).**

- `gen_sandwich_gadget` positionals use `parse_positional::<T>(name, a.next())`: absent means the default, **present and unparsable (including empty) is fatal** with exit 2 and a message naming the argument.
- P7 greps `scripts/`, `tests/` and `security_tests/` for gen invocations that relied on the fallback (for example a placeholder `""` to reach later positionals) and fixes them.
- Unit test `gen_rejects_unparsable_positionals`.

### 11.4 Stage-2 knobs become typed parameters (part of P7)

The `src/stages` contract says construction never reads process configuration (`preprocessing/types.rs:31`). In **P7** (not optional), every output-affecting stage-2 environment read moves into typed fields:

- `EmbeddedMaskingParams { premask, seed_avoid_c1, chain_k, chain_len, chain_rotate, retire: RetireParams, drift: DriftParams, anc_reserve }`;
- the preprocessing slice guard `legacy_slice_guard` (`SLICE_GUARD_LEGACY`, `construct.rs:67`).

These are today's reads at `construct.rs:89`, `embedded_masking.rs:1214, 1611-1668, 1927, 2409-2410` and `construct.rs:67`.

- Gen's adapter (`gen mod.rs`) is the only place that maps the environment names to these fields.
- Diagnostic-only reads (`LGI_DEBUG`, `STRADDLE_CENSUS`, `DRIFT_DEBUG`, `DUMP_GATES`) move into a `Diagnostics` field passed the same way.

Acceptance is `stage2_typed_params_byte_identical`: for 3 fixed calibration seeds at n=16, in A, L8 and one RETIRE configuration, the sha256 of the outputs from the pre-refactor binary is committed, and the refactored output must match.

The stage-3 engine reads (`MIXER_STOP_AT_*`, `TWIST_G57_*`, `MIXER_DB_LEN_BAND`, `MIN_DIR_LOOKUP`) are outside the stage-4 scope. They are classified `forbidden` in the registry, so the driver can never pass them, and their typed migration is recorded as a follow-up (§13.4).

### 11.5 Environment registry

New `src/tdp/env_registry.rs` holds one table: `(name or prefix, class, built-in default)`. The classes are (as implemented; `scripts/tdp_env_registry.tsv` is the generated copy):

| class | meaning | examples | managed handling |
|---|---|---|---|
| `recipe` | output-affecting, set by preset or schedule | `EMBEDDED_MASKING_SHUFFLING…CHAIN_LEN`, `EMBEDDED_MASKING_K…QUAD_FIRE` (custom only) | set by driver only; inherited → fatal |
| `driver` | output-affecting or run plumbing, set by driver from config | `FROZEN_REGULAR_VALUE_CONVENTION`, `SANDWICH_VARIANT` (never exported; built-in `classic`), `TDP_SOURCE_C`, `FROZEN_FILTER`, `MIXER_STOP_FLAG`, `MIXER_DUMP_FLAG` | set by driver only; inherited → fatal |
| `input` | the driver's own inputs | `FROZEN_DB_DIR` (the store path), `TDP_BIN_DIR`, `TDP_GEN_ALLOW_EMPTY_STORE` | inherited accepted; never passed to a stage binary as such (the store path reaches the mixer as the driver's own `FROZEN_DB_DIR`) |
| `neutral` | classified output-neutral (exact memo caches; evidence: identical stage-3 bytes against the reference runs, D10b; no dedicated test) | `LOOKUP_CACHE_MB`, `CANON_CACHE_MB`, `XPOLY_CANON_CACHE_MB` | driver values; inherited ignored with a note |
| `forbidden` | output-affecting, not part of any recipe | every other `EMBEDDED_MASKING_*`, `CANON_RULE_L_BRANCH_CAP`, `CANON_MONOMIAL_CAP`, `MIXER_DUMP_OUT`, `MIXER_DB_LEN_BAND`, `MIXER_STOP_AT_PHASE`, `MIXER_STOP_AT_SPLICES`, `SLICE_GUARD_LEGACY`, `TWIST_G57_NO_RETRY`, `TWIST_G57_NO_SLIDE`, `MIN_DIR_LOOKUP`, `PROD_*`, `SAT_*`, `DB_QC*`, `BV5_*`, `FROZEN_CURATED_*`, `FROZEN_REF_DIR` | inherited → fatal |
| `diagnostic` | diagnostics of construction experiments | `EMBEDDED_MASKING_LGI_DEBUG`, `…_STRADDLE_CENSUS`, `…_DRIFT_DEBUG`, `…_DUMP_GATES`, `…_DIAG`, `…_HOT_MANIFEST` | inherited → fatal (never passed) |
| `tooling` | not read by stage binaries | `CARGO`, `CARGO_TARGET_DIR`, `RUSTC`, `RAYON_NUM_THREADS`, `COMPRESSION_TRACE*` | ignored; never passed |

`env_registry_covers_every_env_read` scans `src/**/*.rs` for string literals passed to `env::var`, `env::var_os`, `envu`, `value_convention` and `env_flag`. It fails if a name is not covered by the table. A new knob anywhere in the crate therefore forces a classification decision.

`tdp_gen.sh` reads the same table from a generated file `scripts/tdp_env_registry.tsv`, which `cargo test` checks is in sync. The transcript test (§10.9 item 1) compares effective values of the `recipe` and `driver` classes.

---

## 12. Implementation phases and commits

Each phase ends with G1-G7, `cargo fmt --check` and clippy green, and a WIP commit.

| phase | content | acceptance |
|---|---|---|
| P0 | this document | committed |
| P0.5 | **anchor A** (§10.8): gen at `492564f0` on all 8 archived seeds; mixer on vL64a and vL128a. Pin the baseline and archive-era HA (§10.1); uv CPython 3.9.6 and the trace recording | results recorded in §2.1; P7 blocked until resolved |
| P1 | `oracle_io`, `pyrand`, `blake2b`, `pyset` (arena) + traces (small and large), `mono` (Mersenne hash), `anf`; `apply_lanes16` | unit tests; trace replays; vA64a window fixture; window-ANF replay of the vA64a and vL128b packs reproduces all 8 aborts |
| P2 | `gf2` + `cview` extraction; `hmap_affine` refactor | §10.5 step 0 (provenance, fallback if needed); refactored HA byte-identical to the **baseline** on vA64a and vA128a |
| P3 | `fingerprint`, `intervals`, `scheduler`, `certify_reorder`; `tdp_packing pack` | sha256 of the 8 packs and bounds; `pack.txt` lines |
| P4 | `repair/*`, `certify_cancel`/`certify_repair`, single window, `check-final`; `tdp_packing repair` | byte-identical `rep512.*` and TSVs against the §10.5 step-0 targets on the 8 replicates |
| P5 | `serialize` (streamed, flat-arena decoder), `verify`, container, `container_header.py` | 8 payload sha256s; container `cmp`; zero slice |
| P6 | `run`, report, exit codes 0-8; `census`; fixtures + `regen.sh` with coverage assertions; integration tests; `stage4-oracle` CI job | `tests/integration/tdp_packing.rs` and the differential job green in CI; census parity on 8 tags |
| P7 | driver: D1-D18, config v2, presets and schedule, env registry and allow-list, seeds (§8), gen strict parsing, stage-4 step with marker, manifest v10, side reports, typed stage-2 knobs (§11.4), shell and unit tests; cache neutrality | G4 + new shell tests; `stage2_typed_params_byte_identical`; `cache_neutrality` NOT run (neutrality rests on identical stage-3 bytes against the reference runs, D10b) |
| P8 | reproduction tests §10.7 (N=16 L8, N=16 A, N=64 L8); **anchor B** for vL64a and vL128a; filter neutrality when a host with `filters.bin` is available | all comparisons identical (or `faithful (failure path)`); §9 times and RSS |
| P9 | docs (§14) | — |

**P2 result (2026-10-08).** `src/stages/packing/{gf2,cview}.rs`; `hmap_affine --exposure` runs on them.

- Baseline HA built from `492564f0` (`git archive` export, `port_runs/baseline_target`), sha256 `ecf87728491d2ce5c66e06b4fbb319dbc9b2221263d1a30e980b088b18538089` (macOS arm64); archive-era HA copied to `port_runs/archive_hmap_affine`.
- **§10.5 step 0 (provenance): resolved, no fallback.** On all 8 tags the baseline HA on the archived `rep512.mpmct1` + `rep512.bounds` reproduces the archived `rep512.{exposure,allcuts,wit}.tsv` and line 2 of `rep512.cview.log`, and the archived `split_cands_*.{exposure,allcuts}.tsv` (the four vL128a files from 01:18-01:30 reproduce on `rep512_v1/rep512.mpmct1`, the P they were scanned on). The archive-era HA equals the baseline on vA64a. The targets stay Appendix B.
- **Refactored HA = baseline, byte for byte** (stdout and every output file, `tests/manual/stage4/cview_parity.sh`, output in `port_runs/p2_parity/`): on all 8 tags for rep512 + bounds with ALLCUTS and WITNESS, with WITNESS only (`skip_settled` path), on `packed_G512` + pre-repair bounds, and on every archived split-candidate file; on vA64a also SPAN and a small heatmap. Time and peak RSS, refactored vs baseline: n=64 rep512 0.4-0.6 s / 140-155 MB vs 8.2-9.0 s / 640-690 MB; n=128 4.3-5.7 s / 440-480 MB vs 120-173 s / 3.2-3.4 GB.
- CI: `--features security-tools --test hmap_affine_exposure` compares the refactored HA with baseline outputs on `tests/fixtures/stage4/cview_n8` (9 cases: a real n=8 G2 and a synthetic pair with t* up to 3 and nullity above 14). Unit tests: `tests/unit/stages/packing/{gf2,cview}.rs`, including a transcription of HA's scan compared on random circuits with all four option combinations.

**ANF / deliverable result (2026-10-08; the P1 ANF part and the P5 serialization part).** `src/stages/packing/{types,oracle_io,pyrand,mono,pyset,anf,serialize,verify,census}.rs`, `XGate::apply_lanes16`, and the binary `tdp_packing` with `serialize`, `anf-report`, `window-anf`, `pack-replay`, `decode`, `eval`, `container` and `verify-equiv` (`run`, `pack`, `repair`, `check-final`, `census` exit 2 until their phases).

- **Bit-exact on all 8 archived runs** (`tests/manual/stage4/anf_parity.sh`, output `port_runs/anf_parity/summary.txt`): `tdp_packing serialize` on `rep512.mpmct1` + `rep512.bounds` gives a raw body whose sha256 equals Appendix B and the archived `deliverable.bin`; `deliverable.tdp4` equals the container that the standalone `container_header.py` builds from the archived body; `tdp_packing anf-report … 65536 TAG-rep512 500` reproduces `anf_rep512.txt` byte for byte (overflow count, monomial median/p90/p99/max/total, both MB sizes, the spot-check line: the spot checks draw the same `Random(20261004)` 256-bit states). Every run also passed decode, the exact window check (decoded window == `window_anf`, ANF == gates on 1024 states per window), DE's 1024-lane evaluation check (seed 1) and the zero slice against C (4096 samples). Time / peak RSS of `serialize` including all checks: n=64 4.8-16 s / 0.3-0.8 GB; n=128 14-36 s / 1.0-1.4 GB (Python: 23 s for vA64a and 107 s for 106.8 MB at n=128, without checks).
- **Packer compose caps on real data** (`tests/manual/stage4/pack_replay.sh`): replaying `WindowState` (MUL_CAP 196 608, `CPython39`) over the archived `packed_G512` windows reproduces the instrumented oracle's MULSTATS exactly: vA64a 3 267 293 products, 40 ambiguous, 2 aborts, both order-dependent; vL128b 9 104 992 products, 242 ambiguous, 6 aborts, all order-dependent, 152 windows with an ambiguous product. Every abort is the window's last gate. Under `Canonical` none of the 8 aborts happens.
- **Set emulator.** `pyset.rs` replays `pyset39_traces.txt` (1 469 recorded CPython 3.9.6 iteration orders, keys up to 300 bits, all set operations the oracle uses) and the checkpoints of `pyset39_large.txt` (toggle loops to about 200 000 elements, tables up to 2^19 slots, the `used > 50000` growth branch), with zero mismatches.
- **Fixtures** (`tests/fixtures/stage4/`, `regen.sh`, oracle in a sandbox from `oracle_sandbox.sh`, interpreter `/usr/bin/python3` 3.9.6 after the self-test): `n8_L8/` is a real L8 chain at n=8 (gen seed 11, mix seed 1013, calibration only) run through `window_pack3.py`, `repair.sh` (with one exposure-aware split) and `anf_dump_eval.py`; `anf/` holds 11 crafted cases (state widths 8 to 1024, so W = 1, 2, 4, 8, 16; empty windows; duplicate and out-of-range bounds; one window; no gates; `orderdep`, whose products abort only under CPython order). `tests/integration/tdp_packing.rs` checks every body, report and per-window decision (both semantics, via `anf_windows.py` on the oracle's own `window_anf`) byte for byte; the fixtures exercise all three overflow kinds and 14 order-dependent product aborts.
- **The oracle is not edited.** The container header is written by the Rust serializer and, independently, by `tests/manual/stage4/container_header.py` (`struct` + `zlib`), which is how parity rebuilds the container from an oracle body (§6). `anf_dump_eval.py` keeps producing the raw body.
- **Differences, all on failure paths or new outputs (D-b, D-d):** `serialize` writes `*.tmp` and renames only after every check passed (exit 4 on a failed check, 7 on an I/O error, 3 on a non-canonical input line); `anf-report` refuses `check_every = 0` (Python divides by zero) and exits 4 when a spot check fails (Python prints and exits 0). Python's `:.1f`/`:.2f` formatting and Rust's agree on exact ties (both round half to even on the binary value), so D-c never triggers.

**P3 result: the packer (2026-10-08).** `src/stages/packing/{blake2b,fingerprint,intervals,scheduler,certify}.rs` and `tdp_packing pack` (WP argv and defaults; `--order-out` = WP's `PACK_DUMP_ORDER`; outputs written via `*.tmp` only after VERIFY and the reorder certificate pass).

- **Bit-exact on all 8 archived runs** (`tests/manual/stage4/pack_parity.sh`, output `port_runs/p3_parity_final/summary.txt`): `packed_G512.mpmct1` and `.bounds` have the Appendix B sha256s and `cmp`-equal the archive; stdout equals the archived `pack_G512.txt` byte for byte (every count, percentage, progress line and the death-cause dict order); `--threads 1` gives identical outputs. Compose aborts (all order-dependent, decided by the CPython 3.9 table order): vA64a 2, vL64a 5, vL64b 3, vA128a 3, vA128b 2, vL128a 5, vL128b 6 (vA64b none), matching the instrumented oracle where it was recorded (vA64a, vL128b).
- **Oracle fixtures** (`tests/fixtures/stage4/pack/`, `regen.sh pack`): 15 cases (the real n=8 G2/G3 pair at 6 argv sets; synthetic `wide300`, `wide128` (B wider than A), `dense16`, `w1024` (W = 16)); stdout, bounds and blake2b-128 digests of the packed circuit and the emitted order all equal (`pack_matches_oracle`). Coverage: 7 compose aborts (4 order-dependent), a duplicate boundary and an empty first window, rescues, both disqualifications, all death causes and none, the mid/band classes.
- **Exact certificate:** every pack is checked by `certify_reorder` (permutation + per-wire run-canonical access sequence), in addition to WP's sampled VERIFY.
- **Speed and memory** (Apple-silicon Mac; another session was using 10-13 cores at the time, load average 12 to 50, so times are inflated): vL64a 13.7 s at load ~12 and 32 s at load ~50, peak RSS 0.82 GB; the Python oracle on the same artifact and machine took 53.4 s and 1.74 GB (load ~12; its outputs equal the archive). At load ~50: n=64 15-32 s / 0.6-0.85 GB, n=128 43-81 s / 1.46-1.54 GB. The scheduler's window ANF is 97% of the time (fingerprints + spans 0.3 s, intervals < 0.1 s, VERIFY 0.2 s, certificate 0.05 s at n=64); see §4.4 for the probe-chain measurement. The §9 target (pack ≤ 15 s at n=64) is met by the vA runs at moderate load and not by vL64a/vL64b under contention.

**P4 result: the repair (2026-10-09).** `src/stages/packing/repair/{mod,seq,cancel,fix,merge,split}.rs`, `certify::{certify_cancel, certify_repair, certify_reorder_subset}`, `tdp_packing repair` and `tdp_packing check-final`.

- **Bit-exact on all 8 archived runs** (`tests/manual/stage4/repair_parity.sh`, output `port_runs/p4_parity/summary.txt`): from the archived `packed_G512.mpmct1` and its unrepaired bounds, `rep512.mpmct1` and `rep512.bounds` have the Appendix B sha256s and `cmp`-equal the archive; `rep512.{exposure,allcuts,wit}.tsv` `cmp`-equal the archive (the §10.5 step-0 targets); `rep512.cview.log` and the repair log (`repair512.log` up to its report part) are equal after the directory prefix is normalized, the two `verify_equiv` lines compared as a set. The runs cover every path of the loop on real data: 4-6 cancel passes; the floor met at fix iteration 1 (vA64a, vA128a, vL128b) or 2 (vA64b, vL64a); the fix loop exhausted (vL64b, vA128b, vL128a, which moved gates in all four iterations: 5, 2, 1, 1); merges (vL64b, vA128b, vL128a); clean splits; and product aborts in the split predicate that depend on the CPython 3.9 set order (one on vA64a, vL64a, vL64b and vA128a, two on vA128b). `--threads 1` gives the same files (checked on vL64b and vL128a; 48 s and 7.8 min single-threaded).
- **Oracle fixtures** (`tests/fixtures/stage4/repair/`, `regen.sh repair`, oracle `repair.sh` in per-case sandboxes with the split budget and iteration counts rewritten): the real n=8 pack and 5 synthetic cases; circuits, bounds, TSVs, `cview.log` and the log all equal (`repair_matches_oracle`). Coverage asserted by `regen.sh`: ≥ 2 cancel passes, fix moves in ≥ 2 iterations, the fix loop exhausted, merges, clean and forced splits, order-dependent split decisions (6 splits under `cpython39`, none under `canonical`), non-default budget and iteration counts, and runs ending with visible non-output rows (the oracle's count test passes on `sparse` while the identity test does not).
- **Exact certificates on every repair:** the cancellations replay on the packed order (each pair identical, every live gate between them commuting), then the repaired order is a linear extension of the survivors' dependency order. Both sampled `verify_equiv` checks (seed 77) also run.
- **Speed and memory** (another session was using 10-13 cores, load average 24-36): n=64 6-9 s, 0.5-1.0 GB peak RSS; n=128 28-57 s, 1.4-1.8 GB. The oracle takes about 2-4 min (n=64) and 10-12 min (n=128) (§9). The §9 targets (repair ≤ 1 min / ≤ 4 min) are met.
- **Differences, all on failure paths (D-b):** a lattice without an interior cut is scanned as zero cuts (the oracle's `hmap_affine` panics, and `repair.sh` goes on with an empty count); a failed `verify_equiv` or certificate is exit 4 with nothing written (the oracle prints `MISMATCH` and continues); the final tests give exit 5 (a visible non-output row) and exit 6 (a window failing the split's budget test) where the oracle exits 0; the first `verify_equiv` line has a fixed place in the log (right after the cancel; the oracle runs it in the background).

**P0.5 result: historic anchor step A (2026-10-09).** `tests/manual/stage4/anchor.sh gen|mix` with the baseline binaries built from `492564f0` (`port_runs/baseline_target`), the archived seeds read inside the script (never printed), `env -i` plus the tag's CONFIG environment:

- **gen, all 8 tags:** `tdp.mpmct1`, `.source_c.g57` and `.sandwich.mpmct1` identical to the archive; `gen.log` identical after line 1 with the directory normalized. `d9631d0e` (the zero-control fire fix) changes no generated byte, as its message argued.
- **mixer, vL64a and vL128a:** `s3.mpmct1` identical (2,052 s / 5,112 s single-run wall; 6.5 GB peak at n=128). The Oct 8 mixer cleanup commits change no byte either.

So the chosen runs are reproducible by the current binaries, and the targets stay Appendix B.

**P6 result: `tdp_packing run` (2026-10-09).** `src/programs/tdp_packing/run.rs`: pack, repair, the `anf_compose`/`link_census` reports and `repair.sh`'s report part, the deliverable (raw body and container, written last after every check) and `packing.report`, with the exit codes of §5.4. `stages::packing::census::{link_census, mult_census}` and the commands `census` and `mult-census`. `RepairReport::status` now reports an over-budget window even when a floor miss is allowed (`floor_miss+over_budget` when both are allowed).

- **Bit-exact on all 8 archived runs** (`tests/manual/stage4/run_parity.sh`, `port_runs/p6_parity/summary.txt`, "ALL RUNS IDENTICAL"): one `tdp_packing run` from the archived `tdp.mpmct1`, `s3.mpmct1` and C reproduces `packed_G512.mpmct1(.bounds)`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, the three TSVs, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` byte for byte; `deliverable.tdp4` equals `container_header.py` applied to the archived body; `rep512.cview.log` and the whole `repair512.log` (including its report part) are equal after path normalization (verify lines as a set). Wall / peak RSS: n=64 20-44 s / 0.85-1.1 GB, n=128 75-137 s / 1.9-2.1 GB (§9 targets: 2 min / 6 min, 1.5 GB / 3 GB).
- `mult-census` reproduces the archived `mult_s2.txt` (checked on vA64a, vL64a, vL128b).
- **Fixture** `tests/fixtures/stage4/run/n8_L8` (new `regen.sh run` part: the oracle stage 4 exactly as `val_chain.sh` runs it on the committed n=8 inputs). Integration tests `run_end_to_end_n8` (every artifact, the container, file modes, the census command), `run_thread_count_does_not_change_output`, `run_exit_codes` (2, 3, 4 from a flipped G3 gate via the packer's VERIFY, 5 with and without `--allow-floor-miss`, 6 with and without `--allow-over-budget`, an allowed floor miss that does not hide an over-budget window, 7).
- **CI job `stage4-oracle`** (§10.6): CPython 3.9 with the set-order self-test, the baseline `hmap_affine` built from `492564f0` in the job (cached by commit; the job appends its own Linux sha256 to the pin list), then `tests/stage4/differential.sh <run id> 20`: 20 random pack cases (`synth.py`: the fixture generators with randomized sizes, gate mixes and packer caps) and 20 random repair + serialize cases (n ∈ {4, 6, 8, 12}, budgets and iteration counts randomized), oracle against Rust byte for byte. A case whose merge leaves a single window takes the documented D-b path (the oracle's `hmap_affine` panics): only `rep512.*` is compared there. An oracle repair over 600 s is reported as skipped, never as a pass, and more than a quarter skipped fails the job. Locally on four seeds: see the P8 result.

**P7 result: the managed driver (2026-10-09).** D1-D18 are implemented:

- `scripts/tdp_gen.sh` (stages 1-4): presets L8/A/custom, the `validated` schedule (`int(G*2*2*1.3)` with Python `int`, G from the header), the validated size table, seeds in the private `SEEDS` (format 2; legacy inputs refused with migration messages; the gen banner check), every stage binary under `env -i` + allow-list + recipe variables (`stage_exec`), inherited output-affecting variables fatal per `scripts/tdp_env_registry.tsv`, `FROZEN_FILTER` per D11, no canonicalization caps, the stage-4 step with the keyed completion marker (`packing.recipe`, HMAC-SHA256 with `run_key` via `scripts/tdp_reports/run_mac.py`), `NOT_RECOMMENDED` on allowed failures, the report-only side reports (`stage2_mult.txt` via `tdp_packing mult-census`; `stage2_span.*` when a `security-tools` `hmap_affine` is present).
- `src/tdp`: config v2 (v1 refused with the migration message; authoritative preset and schedule; `[packing]`; `source.gates_rule`; `run.threads`), `SEEDS` created with `create_new` and mode 0600 before the manifest, the allow-list environment (`env_clear`), the registry `src/tdp/env_registry.rs`, recipe v10 with resolved values only, `tdp_packing` built and fingerprinted, seed-flag redaction, exit-code meanings.
- **Gen strict parsing (D17):** an unparsable positional exits 2 naming it.
- **Typed stage-2 knobs (§11.4):** `EmbeddedMaskingParams { premask, seed_avoid_c1, chain_k, chain_len, chain_rotate, retire: RetireParams, drift, drift_window, anc_reserve, legacy_slice_guard }` and the diagnostics in `EmbeddedMaskingExecution`; gen's adapter (`stage2_knobs_from_env`, `execution_from_env`, reused by the two security tools) is the only mapping from the environment names. `stage2_typed_params_byte_identical` (`tests/integration/stage2_typed_params.rs`): 6 configurations (A, L8, two retirement configurations, chains/rotation/ancilla reserve/drift window/legacy slice guard, drift off) × 3 calibration seeds at n=16, blake2b-256 of the three outputs against the pre-refactor binary's: all 18 identical. `src/stages` reads no environment except the stage-3 engine's `db_mixing/environment.rs` (§13.4), asserted by a test.
- **Tests:** `tests/unit/tdp.rs` (43 tests: the default L8 resolution, the size table, preset/schedule conflicts, v1/v9 refusals, the exact default manifest text, seeds created before the manifest and absent from every log, the allow-list environment, refusals per registry class, `env_registry_covers_every_env_read`, `env_registry_tsv_in_sync`, `stages_library_reads_no_environment`, `exit_code_tables_agree`, ...); `tests/shell/tdp_flags.bash`, `tdp_block_size.bash` (rewritten), and the new `tdp_defaults.bash` (§10.9 item 1: a sandboxed copy of `val_chain.sh` on recording mocks against `tdp_gen.sh` and the Rust driver, same seeds: gen argv, mixer argv, effective recipe/driver environment, allow-list, stage-4 calls; the `int`-not-`round` case; n=64/128 default sizes; n=32 refused; 16 forbidden variables refused by name), `tdp_seeds.bash`, `tdp_stage4_resume.bash`, `tdp_stage12_parity.bash` (real gen at n=16, L8 and A, both ways) and `tdp_reproduce_small.bash`.
- **D10b cache neutrality:** not run separately. The reference runs used the built-in cache sizes (512 MB lookup) and the managed runs use 2048 MB; the historic anchor (vL64a, vL128a) and the reproduction runs produced identical stage-3 bytes under the two settings, which is the property `cache_neutrality` was to establish (at full scale rather than n=16).
- **D11 filter neutrality:** not recorded (the local store has no `filters.bin`); `auto` keeps the filter off whenever `filters.bin` exists.

**P8 result: reproduction and anchor B (2026-10-09).** All runs used the snapshot of the tree at `e4743fcf` (so the scripts could not change under a running driver), `/usr/bin/python3` 3.9.6 and the baseline `hmap_affine`; seeds fresh from the CSPRNG (calibration, private).

| run | result | wall (val_chain.sh / managed) |
|---|---|---|
| reproduction n=8, \|C\|=16, L8 (`tdp_reproduce_small.bash`) | ALL IDENTICAL | 51 s / 15 s |
| reproduction n=16, \|C\|=64, L8 | ALL IDENTICAL | 182 s / 73 s |
| reproduction n=16, \|C\|=64, A | ALL IDENTICAL | 158 s / 64 s |
| reproduction n=64, L8 (\|C\| = 1000 by default) | ALL IDENTICAL | 1,908 s / 1,492 s (stage 3: 1,457 s / 1,601 s, +9.9%, sequential runs on a shared machine) |
| anchor B vL64a (archived seeds) | every artifact identical to the archive, `status=ok` | managed 1,542 s (5.8 GB peak, the mixer) |
| anchor B vL128a (archived seeds) | every artifact identical to the archive, `status=ok` | managed 4,499 s (7.4 GB peak, the mixer) |

"ALL IDENTICAL" covers: `tdp.mpmct1`, the source C, the sandwich, `s3.mpmct1` = `db_mixing.mpmct1`, the stage-2 rebirth census, `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1`, `rep512.bounds`, the three TSVs, `pack_G512.txt`, `anf_rep512.txt` and `link_rep512.txt` (labels normalized), `deliverable.bin`, `deliverable.tdp4[48:]`, and `deliverable.tdp4` against `container_header.py` of the oracle body; the driver refused four decoy variables (`MIXER_DB_LEN_BAND`, `SLICE_GUARD_LEGACY`, `EMBEDDED_MASKING_RETIRE`, `CANON_MONOMIAL_CAP`) before each managed run. The anchor rows compare against the archived files of the chosen runs themselves (Appendix B).

The CI differential (`tests/stage4/differential.sh`) on four local seeds: 160 cases (seeds 777, 1009, 2018 and 3027; 80 pack and 80 repair + serialize cases), all identical, none skipped. An earlier trial seed produced the single-window D-b case (the merge removed every boundary; the oracle's `hmap_affine` panicked): `rep512.mpmct1` and `rep512.bounds` identical. That trial also showed that a 16-wire state with budget 65,536 makes the oracle's uncapped Python products run for hours, so `synth.py` keeps budget 65,536 for n ≥ 8.

**P9:** the documentation of §14 is updated.

**Acceptance round 1, vL64a (2026-10-09, at `b77753ad`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64` with no packing flag (the defaults are the recommended G=512 chain: `window_pack3.py 512 512 24576 3000`, repair budget 65536, 4+4 iterations, cpython39 caps); exit 0, `status=ok`. Against the archived oracle files: `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv` and `deliverable.bin` byte-identical; `deliverable.tdp4[48:]` = the archived (pre-header) `deliverable.bin`, and `tdp_packing decode --format container` accepts it; `pack_G512.txt`, `anf_rep512.txt`, `link_rep512.txt` identical up to the run label (`tdp-` for `vL64a-`), `rep512.cview.log` and `repair512.log` up to output paths. Independent checks on the Rust outputs: `hmap_affine --exposure --g-cuts rep512.bounds` reports 64 visible of 1000 (the exact floor; the 64 are the last writer of each of the 64 wires) with TSVs identical to the oracle's; `verify_zero_slice` PASS with 512 samples on `rep512.mpmct1` and `packed_G512.mpmct1`; the oracle's `link_census.py` on the Rust `rep512` gives the archived `link_rep512.txt` byte for byte (proxy L2 5366 / M2 58598, froth L2 1126 / M2 2330). A fresh sandboxed oracle rerun reproduced the archive byte for byte.

| stage 4 on vL64a | wall | peak RSS |
|---|---|---|
| Rust `tdp_packing run` (pack, repair, reports, serialize, every check incl. 4096-sample zero slice) | 31.3 s | 1.10 GB |
| Python oracle: `window_pack3.py` / `repair.sh` / `anf_dump_eval.py` | 67.4 s / 213.7 s / 49.4 s = 330.5 s | 1.75 GB (the packer) |

**Acceptance round 1, vA128a (2026-10-09, at `a64b61ec`; outputs `port_runs/vA128a_r1/`).** Same command at n=128, every packing parameter passed explicitly at its recommended value (`--window-gates 512 --interval-cap 512 --monomial-budget 24576 --span-cap 3000 --split-budget 65536 --fix-iterations 4 --fallback-iterations 4 --anf-cap-semantics cpython39`, all equal to the defaults); exit 0, `status=ok`: pack 4,757,723 gates in 7,376 windows, repair 4,437,893 gates and 7,376 boundaries (cancel 159,915 pairs, fix already at the floor, one fallback split), final scan 128 visible / 0 non-output, 0 windows over budget, 7,377 ANF windows with 4,185,577 monomials (max 41,026). Against the archive, with `--label vA128a`, every file is byte-identical with no normalization beyond output paths: `packed_G512.mpmct1(.bounds)`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt`, `deliverable.bin` (106,890,630 bytes); `rep512.cview.log` and `repair512.log` identical after path normalization, line order included. `deliverable.tdp4` = `container_header.py` applied to the archived pre-header body (48-byte header + the identical payload), and `decode --format container` accepts it. Independent checks on the Rust outputs: `hmap_affine --exposure --g-cuts rep512.bounds` (fresh build) reports 3000 born values, 128 visible (4.3 %, all at t*=1): the exact floor, TSVs identical to the run's; `verify_zero_slice` PASS with 512 samples (random seed) on `rep512.mpmct1` and on `packed_G512.mpmct1`; the oracle's `link_census.py` on the Rust `rep512` equals both the archived and the Rust `link_rep512.txt` (proxy 54,345 values, L2 18,521 / M2 250,448; froth 280,727, L2 2,219 / M2 4,661). A fresh sandboxed oracle rerun (CPython 3.9.6, pinned `hmap_affine`) reproduced all eleven files byte for byte.

| stage 4 on vA128a | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (pack 25.5 s, repair 24.5 s, reports 4.4 s, serialize + checks 17.0 s) | 72.9 s | 303.5 s | 2.10 GB |
| Python oracle: `window_pack3.py` / `repair.sh` / `anf_dump_eval.py` | 141 s / 4753 s / 1057 s = 5951 s | 136.8 s / 760.8 s / 82.9 s = 980.5 s | 5.14 GB (the packer; repair 3.37 GB, serializer 1.98 GB) |

The oracle's wall time is inflated by contention (it ran in the background next to another n=128 oracle chain and two `circuit_mixer` jobs; `anf_dump_eval.py` used 83 s of CPU in 1057 s, against 65 s wall for its serialization in the archived run); its CPU time is the fair comparison: about 3.2x the Rust CPU, and the Rust wall is 12-14x shorter than the 15-17 min the uncontended oracle chain takes at n=128 (§12 P6). The Rust peak (2.1 GB) is below the 3 GB n=128 target of §9 and 2.4x below the oracle's packer peak.

**Acceptance round 1, vL128a (n=128, L8; 2026-10-09).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 128` with every parameter at its default (= the recommended configuration: G=512, interval cap 512, monomial budget 24576, span cap 3000, split budget 65536, 4+4 iterations), on the archived inputs; outputs in `port_runs/vL128a_r1/`.

| artifact | vs archived oracle output |
|---|---|
| `packed_G512.mpmct1`, `.bounds` | identical |
| `rep512.mpmct1`, `rep512.bounds` | identical (4 596 808 → 4 259 448 gates, 7 058 boundaries) |
| `rep512.exposure.tsv`, `.allcuts.tsv`, `.wit.tsv` | identical |
| `deliverable.bin` | identical; `deliverable.tdp4[48:]` = `deliverable.bin` (the 48-byte container header is the only addition) |
| `anf_rep512.txt`, `link_rep512.txt`, `pack_G512.txt` | identical |
| `repair512.log`, `rep512.cview.log` | identical except the absolute output paths printed in them |

Independent checks on the Rust outputs: the standalone `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (current tree) reports **128 visible of 3 000 born values (the exact floor)** and reproduces the three TSVs byte for byte; `verify_zero_slice C rep512.mpmct1 128 512` PASS; the Python `link_census.py` on the Rust `rep512.*` equals the oracle `link_rep512.txt` (proxy L2 16 106, M2 187 248).

Cost on this Mac (machine shared with another session, load about 5):

| | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks included) | 119 s | 627 s | 2.04 GB |
| Python oracle, sandboxed (CPython 3.9.6, pinned baseline HA) | 6 921 s | 1 943 s | 5.08 GB |
| — `window_pack3.py` | 176 s | 171 s | 5.08 GB |
| — `repair.sh` (incl. `anf_compose.py`, `link_census.py`) | 6 418 s | 1 445 s | 3.40 GB |
| — `anf_dump_eval.py` | 328 s | 327 s | 2.00 GB |

The sandboxed oracle rerun reproduced the archived outputs byte for byte. Its repair wall time is far above its CPU time (the machine was shared during the four fix scans and the fallback), so the ratio is about 58× in wall time and 3× in CPU time. Both §9 n=128 targets hold (wall ≤ 6 min, peak RSS ≤ 3 GB).

**Review fixes round 1 (2026-10-09).** Seeds no longer travel on stage-binary argv (gen `@/dev/fd/N`, mixer `--seed-file /dev/fd/3`; banners print `(file)`; §8.5); the managed build adds the report-only `hmap_affine` (fingerprinted; `stage2_span_report=`); the manifest records `frozen_filter=<mode>(<resolved>)` and a derived `pinned_recipe`, and `$RUN/stage3.settings` records the `FROZEN_FILTER` set; `tdp_packing run` clears a reused `--out-dir`, commits the report last and bounds its parameters (§5); the shell tests no longer rely on `!`/`&&` under errexit; CI runs `stage2_typed_params`. Checked: all gates; `reproduce_val_chain.sh 8 L8 16` at the fix commit (every artifact identical, failure path faithful, span report identical to val_chain.sh's, no seed value in any log); a default managed build run produces `stage2_span.*`. The n=64 L8 reproduction at the fix commit was started in `port_runs/fix_r1/repro_n64_L8/`; it, and the later attempts `port_runs/repro_r1` and `repro_r2`, were killed with the agent sessions that started them and produced no result (no `summary.txt`). The same-seed n=64 reproduction after these fixes is in round 2 below.

**Acceptance round 2, vA64a (n=64, config A; 2026-10-09, at `4dc934e9`, after review fixes round 1; outputs `port_runs/vA64a_r2/`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64 --label vA64a` with no packing flag; exit 0, `status=ok` (pack 1,647,349 gates in 2,774 windows; repair 54,318 cancelled pairs, floor at fix iteration 1, two fallback splits, 1,538,713 gates and 2,775 boundaries; final 64 visible, 0 non-output, 0 over budget). A second run with every packing flag passed explicitly at its recommended value gives the same files (`packing.report` differs only in `time.*`), so the defaults are the chosen G=512 chain. Against the archive: `packed_G512.mpmct1(.bounds)`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` byte-identical; `deliverable.tdp4[48:]` = the archived pre-header body and `deliverable.tdp4` = `container_header.py` of it; `rep512.cview.log` and `repair512.log` equal after path normalization. Independent checks: `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (HEAD build) reports 64 visible of 1000 (the exact floor; 64 distinct wires, each value the wire's last writer), TSVs identical; `verify_zero_slice` PASS with 512 samples on `rep512.mpmct1` and `packed_G512.mpmct1`; `link_census.py` on the Rust `rep512` = archived `link_rep512.txt` = Rust's (proxy 15,455, L2 6,039 / M2 73,178; froth 74,595, L2 373 / M2 780). A fresh sandboxed oracle rerun reproduced all eleven files byte for byte.

| stage 4 on vA64a (load average about 16, both runs) | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks included) | 45.2 s | 148.9 s | 1.17 GB |
| Python oracle: `window_pack3.py` / `repair.sh` / `anf_dump_eval.py` | 126.7 s / 496.6 s / 54.3 s = 677.6 s | 104.8 s / 397.0 s / 53.7 s = 555.5 s | 1.71 GB (the packer) |

Round 1 on an idle machine measured 18.2 s for Rust and 219.9 s for the oracle; the ratio (about 12-15× in wall time) holds under load.

**Acceptance round 2, vL64a (2026-10-09, at `4dc934e9`, after review fixes round 1; outputs `port_runs/vL64a_r2/`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64 --label vL64a` with every packing parameter at its default (the report records `window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`, i.e. the recommended G=512 chain); exit 0, `status=ok`: pack 1,593,054 gates in 2,586 windows, repair 1,476,974 gates and 2,587 boundaries (cancel 58,040 pairs, fix 65 → 64, fallback 2 splits), final scan 64 visible / 0 non-output, 0 windows over budget. Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt`, `deliverable.bin` (32,783,935 bytes) are byte-identical; `rep512.cview.log` and `repair512.log` differ only in the absolute output paths printed in them; `deliverable.tdp4` = 48-byte container header + the archived (pre-header) `deliverable.bin`, accepted by `decode --format container` (2,588 windows, 1,659,662 monomials). `packing.report` equals round 1's except `binary_xxh3`. Independent checks on the Rust outputs: `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (current tree, `security-tools`) reports 1000 born values, 64 visible (6.4 %, all t*=1), each the last writer of one of the 64 wires: the exact floor, TSVs identical to the oracle's; `verify_zero_slice C rep512.mpmct1 64 512` and the same on `packed_G512.mpmct1` PASS; the oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte (proxy 11,743 values, L2 5,366 / M2 58,598; froth 78,179, L2 1,126 / M2 2,330).

| stage 4 on vL64a, round 2 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load 6-20 (another session building) | 72-122 s | 241-339 s | 1.02-1.09 GB |
| Rust, round 1 (`b77753ad`, quiet machine) | 31.3 s | 110.9 s | 1.10 GB |
| Python oracle (round-1 sandboxed rerun): pack / repair / serialize | 67.4 / 213.7 / 49.4 = 330.5 s | 328.4 s | 1.75 GB (packer) |

The round-2 Rust times are contention, not a regression: run back to back under the same load, the pre-fix binary (`snap_target`, 00:55) took 116 s wall / 341 s CPU and the `4dc934e9` binary 122 s / 339 s, with identical deliverables. Peak RSS is unchanged (about 1.1 GB, 0.6x the oracle's packer peak).

**Acceptance round 2, vL128a (n=128, L8; 2026-10-09, binary built from `4dc934e9`; outputs `port_runs/vL128a_r2/`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 128 --label vL128a` with every packing parameter at its default (the report records the recommended G=512 chain: `window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`); exit 0, `status=ok`: pack 4,596,808 gates in 7,059 windows, repair 4,259,448 gates and 7,058 boundaries (cancel 168,680 pairs, fix 132 → 130 then exhausted, fallback merge + 1 split 130 → 128), final scan 128 visible / 0 non-output, 0 windows over budget.

- Against the archived oracle, no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (102,818,808 bytes) are byte-identical (sha256 = Appendix B). `rep512.cview.log` and `repair512.log` differ only in the absolute output paths (equal after path normalization, line order included). `deliverable.tdp4[48:]` = the archived pre-header `deliverable.bin`; `decode --format container --stats` accepts it (7,059 windows, 4,065,912 monomials). `packing.report` equals round 1's except `binary_xxh3` and the `time.*` keys. A second run (`out2/`) gave the same deliverable, repair and bounds.
- Independent checks on the Rust outputs (`port_runs/vL128a_r2/check/`): `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (current tree, `security-tools`) reports 3,000 born values, **128 visible** (4.3 %, all t*=1): the exact floor, with TSVs byte-identical to the oracle's; `verify_zero_slice C rep512.mpmct1 128 512` and the same on `packed_G512.mpmct1` PASS; the oracle's `link_census.py` on the Rust `rep512` equals the archived `link_rep512.txt` byte for byte (proxy 34,958 values, L2 16,106 / M2 187,248; froth 285,325, L2 5,579 / M2 11,622).
- Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN` set (the Rust driver case included) passes for d16, d64A, d64L8 and d128: gen argv, mixer argv, effective recipe env, allow-list and the stage-4 call (`tdp_packing run` at its defaults ≙ `window_pack3.py … 512 512 24576 3000`, `repair.sh TAG 512 N`, `anf_dump_eval.py`) are identical to `val_chain.sh`'s. `tdp_flags.bash` and `tdp_block_size.bash` pass.

| stage 4 on vL128a | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run`, round 2 (all checks; load 12-19, another session running `circuit_mixer`/`hmap_affine`) | 335 / 359 s | 1,508 / 1,617 s | 2.02 / 1.87 GB |
| — pack / repair / reports / emit+checks (first run) | 109 / 142 / 19 / 63 s | | |
| Rust, round 1 (`b77753ad`, load about 5) | 119 s | 627 s | 2.04 GB |
| Python oracle (round-1 sandboxed rerun): pack / repair / serialize | 176 / 6,418 / 328 = 6,921 s | 1,943 s | 5.08 GB (packer) |

The oracle scripts are unchanged since round 1 (no diff under `scripts/windowed_delivery/` since `ad9eedfe`), so its round-1 measurement stands. The round-2 Rust times are contention: total CPU time rose with wall time (627 s → 1,508 s), every phase slowed, including the single-threaded pack (43 s → 109 s wall), and the review fixes touched no packing, repair or emission code path that runs here (outputs and counters are identical). Under that load the Rust stage 4 is still about 19× faster in wall time than the oracle and below the §9 n=128 targets (≤ 6 min, ≤ 3 GB) — barely on wall time; peak RSS is 0.4× the oracle's.

**Acceptance round 2, vA128a (n=128, config A; 2026-10-09, binary built from `4dc934e9`; outputs `port_runs/vA128a_r2/`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 128 --label vA128a` with **no packing flag** (round 1 passed every flag explicitly). The report records the recommended G=512 chain as the defaults (`window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`); exit 0, `status=ok`. Pack 4,757,723 gates in 7,376 windows. Repair: 159,915 cancelled pairs, the floor already reached at fix iteration 1, one fallback split, 4,437,893 gates and 7,376 boundaries. Final scan 128 visible / 0 non-output, 0 windows over budget. 7,377 ANF windows, 4,185,577 monomials (max 41,026). The counters are identical to round 1's, and `packing.report` differs from round 1's only in `binary_xxh3` and `time.*`.

- Against the archived oracle, with no normalization, these files are byte-identical: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (106,890,630 bytes). `rep512.cview.log` and `repair512.log` are identical once the absolute output paths are normalized, with line order unchanged. The payload: `deliverable.tdp4[48:]` equals the archived pre-header `deliverable.bin`, `deliverable.tdp4` equals `container_header.py` applied to that body, and `decode --format container --stats` accepts it (7,377 windows, 4,185,577 monomials).
- Independent checks on the Rust outputs (`checks/`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (HEAD build, `security-tools`) reports 3,000 born values and **128 visible** (4.3 %, all t*=1). This is the exact floor. Its TSVs are identical to both the run's and the oracle's.
  - `verify_zero_slice C rep512.mpmct1 128 512` and the same check on `packed_G512.mpmct1` both PASS (random seeds, kept private).
  - The oracle's `link_census.py` on the Rust `rep512` gives exactly the archived and the Rust `link_rep512.txt`: proxy 54,345 values, L2 18,521 / M2 250,448; froth 280,727 values, L2 2,219 / M2 4,661.
- A second Rust run (`packing_t2/`) produced the same files. A fresh sandboxed oracle chain (`oracle_sb/`, CPython 3.9.6, pinned `hmap_affine`, `oracle_chain.sh`) ran at the same time, and it again reproduced all eleven archived files byte for byte.

| stage 4 on vA128a, round 2 (same session, load 4-14) | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks; two runs) | 128 / 139 s | 608 / 705 s | 1.90 / 2.13 GB |
| — pack / repair / reports / emit+checks (first run) | 47.5 / 49.0 / 7.5 / 22.6 s | | |
| Rust, round 1 (`a64b61ec`) | 72.9 s | 303.5 s | 2.10 GB |
| Python oracle, round 2 (fresh sandboxed rerun): `window_pack3.py` / `repair.sh` / `anf_dump_eval.py` | 366 / 1,847 / 160 = 2,373 s | 275 / 1,715 / 159 = 2,149 s | 5.50 GB (the packer; repair 3.54 GB, serializer 2.08 GB) |

The machine was slower than in round 1 for both implementations alike, so the round-2 Rust times do not show a regression:
- the Rust run retired the same number of instructions (2.884 × 10¹² in both rounds) at about half the effective clock (1.28 × 10¹² cycles over 608 s of CPU, against 1.19 × 10¹² over 303 s);
- the oracle's single-threaded packer slowed by the same factor (134 s → 273 s user).

Measured side by side under the same conditions, the Rust stage 4 takes 18.5× less wall time than the oracle and 3.5× less CPU. Its peak RSS is 0.35× the oracle packer's. It stays within the §9 n=128 targets (≤ 6 min, ≤ 3 GB).

**End-to-end managed run, round 2, n=64 L8 (2026-10-09, binaries built from `4dc934e9`; outputs `port_runs/e2e64_r2/`).** `tdp_gen` with a config that sets only `source.wires = 64`, the run and build directories and `database.regular_dir` (the frozen store `frozen_m1_m11`); every other setting is the managed default (preset L8, schedule `validated`, stage 4 at G = 512, split budget 65,536, cpython39 caps). Seeds are fresh CSPRNG draws, kept in the private `run/SEEDS`; the zero-slice check seed is in the private `checks/SEEDS.checks`. No seed value appears in any log, the manifest or `packing.report`. The driver exited 0 with `status=ok`.

- Stages 1+2: 778,166 gates (sandwich 2,832), with 109,331 chained brackets. Forward and reverse verification PASS (256 samples each). The stage-2 span check (`stage2_span.txt`) is 0 at all 12,160 cuts.
- Stage 3: `--moves 4046463` (= int(2·2·|G|·1.3)) with `FROZEN_FILTER=1` (auto, no `filters.bin`); output 1,592,588 gates.
- Stage 4:
  - packing: 2,579 windows; 11,511 distinct values exposed at window boundaries;
  - repair: cancelled 57,986 pairs, leaving 1,476,616 gates; the fix loop was already at the floor (64); the fallback added 4 splits, giving 2,582 boundaries;
  - result: 0 windows over budget; 2,583 ANF windows with 1,477,260 monomials (window max 71,243); every certificate and check in the report is `ok`, including the zero slice with 4,096 samples.
- Independent checks (`checks/`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` reports 1,000 born values and **64 visible**, all at t* = 1. That is the exact floor, with 0 non-output values visible. Its TSVs are identical to the run's.
  - `verify_zero_slice C rep512.mpmct1 64 512` PASS, and the same check PASS on `packed_G512.mpmct1`.
  - The oracle's `link_census.py` on the Rust `rep512` gives the run's `link_rep512.txt` byte for byte.
  - The oracle's `anf_dump_eval.py` on the Rust `rep512` gives the Rust `deliverable.bin` byte for byte. `deliverable.tdp4` is a 48-byte header followed by that body.

| n=64 L8 metric | e2e64_r2 | e2e64_r1 | STAGE2_LINKABILITY §5 range |
|---|---|---|---|
| exposed | 11,761 | 11,799 | 11.7k |
| linkable (L2) | 5,305 | 5,371 | 5.27–5.37k |
| sightings (M2) | 58,360 | 58,793 | 58.5–58.9k (r2 is 0.2 % below) |
| linkable froth | 1,114 | 1,080 | 1,110–1,126 (r1 is below) |
| anchors (100+) | 194 | 183 | 185–200 (r1 is below) |
| `deliverable.bin` | 27,959,712 B (28.0 MB) | 28.9 MB | 30.2–32.8 MB (both runs are below) |
| xz -9 | 2.91 MB | 2.96 MB | 3.05–3.10 MB (both runs are below) |
| eval, ANF form (Python, 1,024 lanes) | 1.21 ms/input | 1.00 | 0.79–1.00 (r2 is above; load average ~12) |

Both managed runs have smaller deliverables than the §5 replicates. The size spread across seeds is wider than the "1–2 %" stated in §5: the `anf_compose` size estimates are 26.6 MB (r2), 27.5 MB (r1), 28.7 MB (vL64b) and 31.2 MB (vL64a).

| e2e64_r2 wall time (load average 11-14, other sessions building and running) | wall |
|---|---|
| stages 1+2 (`gen_sandwich_gadget`) | ≈ 1 s (the span report then ran in the background, 220 s) |
| stage 3 (`circuit_mixer`; chain 2,557 s) | 2,560 s |
| stage 4 (`tdp_packing run`: pack 25.7 / repair 12.1 / reports 5.8 / emit+checks 16.7 s) | 61 s |
| driver total, resumed attempt (build skipped) | 2,622 s (CPU 1,816 s) |

The first attempt was lost to an environment problem. The driver launches `bash` from `PATH`, and that `PATH` started with `/bin/bash` 3.2. After a 9-minute wait on the shared build lock, the run stopped at `scripts/tdp_gen.sh` line 138 with `declare: -A: invalid option` and exit 2. It had already written `SEEDS` and the manifest. With Homebrew bash first on `PATH`, the resume matched the manifest, kept the seeds, and skipped the build. Neither the runner nor `tdp_gen.sh` checked the bash version (closed in review fixes round 2: the runner resolves a bash >= 5.1 before building or writing anything, and `tdp_gen.sh` refuses an older bash first).

**End-to-end managed run, round 2, n=128 L8 (2026-10-09, binaries built from `4dc934e9`; outputs `port_runs/e2e128_r2/`, checks in `port_runs/e2e128_r2_cfg/checks/`).** `tdp_gen` with a config that sets only `source.wires = 128`, the run and build directories and the regular store `frozen_m1_m11`. Every other value is the managed default: preset L8, schedule `validated`, packing `default`. The manifest records `pinned_recipe=val_chain-L8-20261007`, `seed_source=random` and `packing_params=default`. The driver exited 0 with `status=ok`.

- Stages 1+2: 2,231,499 gates (sandwich 7,920), with 316,328 chained brackets. Forward and reverse verification PASS (256 samples each). The stage-2 span check is 0 at all 8,718 sampled cuts.
- Stage 3: `--moves 11603794` (= int(2·2·|G|·1.3)) with `FROZEN_FILTER=1` (auto); output 4,595,765 gates.
- Stage 4:
  - packing: 7,066 windows; 34,377 distinct values exposed at window boundaries;
  - repair: cancelled 168,520 pairs, leaving 4,258,725 gates; the fix loop was already at the floor (128); the fallback added no split, giving 7,065 boundaries;
  - result: 0 windows over budget; 7,066 ANF windows with 4,187,470 monomials (window max 57,489); every certificate and check in the report is `ok`, including the zero slice with 4,096 samples.
- Independent checks:
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` reports 3,000 born values and **128 visible** (4.3 %), all at t* = 1. That is the exact floor, with 0 non-output values visible. Its three TSVs are identical to the run's.
  - `verify_zero_slice C rep512.mpmct1 128 512` PASS (random seed, recorded in `checks/check_seeds.txt`), and the same check PASS on `packed_G512.mpmct1`.
  - The oracle's `link_census.py` (sandboxed) on the Rust `rep512` gives the run's `link_rep512.txt` byte for byte.
  - The oracle's `anf_dump_eval.py` (sandboxed) on the Rust `rep512` gives the Rust `deliverable.bin` byte for byte. `deliverable.tdp4` is a 48-byte header followed by that body.

| n=128 L8 metric | e2e128_r2 | e2e128_r1 | STAGE2_LINKABILITY §5 range |
|---|---|---|---|
| exposed | 34,947 | 34,603 | 34.5–35.0k |
| linkable (L2) | 16,201 | 15,992 | 16.06–16.11k (r2 is 0.6 % above, r1 0.4 % below) |
| sightings (M2) | 187,800 | 188,350 | 187–188k (r1 is 0.2 % above) |
| linkable froth | 5,575 | 5,747 | 5,544–5,765 |
| anchors (100+) | 719 | 719 | 701–718 (both runs one above) |
| `deliverable.bin` | 107,027,606 B (107.0 MB) | 108.7 MB | 102.8–106.8 MB (r2 is 0.2 % above, r1 1.8 % above) |
| xz -9 | 9.62 MB | 9.69 MB | 9.50–9.62 MB |
| eval, ANF form (Python, 1,024 lanes) | 1.94 ms/input | not measured | 1.83–1.99 |

Every metric is within 2 % of the §5 range. Where a value falls outside the range, it does so by less than the 1–2 % spread between replicates that §5 reports.

| e2e128_r2 time | wall | awake time (from the tools' own timers) |
|---|---|---|
| stages 1+2 (`gen_sandwich_gadget`) | ≈ 6 s (the span report then ran in the background, about 10 min) | |
| stage 3 (`circuit_mixer`) | 28,142 s | chain 5,708.6 s + final float 4.7 s |
| stage 4 (`tdp_packing run`: pack 38.2 / repair 14.5 / reports 4.4 / emit+checks 18.9 s) | 3,909 s | 76.6 s |
| driver total (incl. a build that waited on the shared lock) | 32,615 s | CPU 4,176 s; peak RSS 7.5 GB (the mixer) |

The wall times are not representative. From about 08:00 the laptop's lid was closed (`AppleClamshellState = Yes`, and `pmset` logs show repeated maintenance sleeps), so the processes ran only during dark wakes. The internal timers use `Instant`, which on macOS does not advance during sleep, so they count only time awake. The awake times are comparable with round 1 (chain 4,829 s) and with the n=128 stage-4 figures in this section.


**Review fixes round 2 (2026-10-09; WIP commits `e553b804`, `6400aa53`).**

- **bash.** The runner resolves a bash >= 5.1 (`MIN_BASH`; the first one on `PATH`, then `/opt/homebrew/bin`, `/usr/local/bin`, `/bin`, `/usr/bin`) before any build, `SEEDS` or manifest, and runs `tdp_gen.sh` with that path (also in the dry run); with none it exits 2 naming every bash it found. `tdp_gen.sh` refuses bash < 5.1 first, in syntax 3.2 parses. 5.1 rather than 4: the seed here-strings go through a pipe, not a temporary file, only from 5.1.
- **Stop flag.** `tdp_gen.sh` reads the mixer's `chain done ... : <reason>` line and records `stop_reason=` in `stage3.settings`. A stop-flag finish (or no such line) is under-dosed: the output is moved to `db_mixing.stopped.mpmct1`, the driver exits 2 without stage 4, and a resume re-runs stage 3. Other reasons than the move budget or the profile's end are noted as warnings.
- **Exit meanings.** A failed `tdp_packing run` writes `$RUN/stage4.failed` (`exit=<code>`); the runner removes it before starting and attaches a stage-4 meaning only when it names the pipeline's exit code.
- **`pinned_recipe`.** The manifest records `frozen_db_fingerprint` (xxh3-128 of every file's name and size plus each `.ok` checksum marker's text; the shards are not read). The recipe label needs that fingerprint to equal `frozen_m1_m11`'s (`VALIDATED_REGULAR_STORE`) and an unfiltered stage 3 (`FROZEN_FILTER=1`, no `filters.bin`); a filtered `on` run, another store or a copy with other shards records `none`.
- **`tdp_packing`.** `packing.report`'s `status` names every violation (`floor_miss+over_budget`) whatever is allowed; `run` removes the outputs of a run with any window size from a reused `--out-dir`; `census`, `mult-census` and `verify-equiv` refuse circuits wider than 65,536 wires (exit 3), `anf-report` wider than 1,024 (exit 2), `eval --format raw` a `--state-wires` below 2n (exit 2).
- **Tests.** Unit tests for the bash resolution, the stage-4 marker and the store-gated `pinned_recipe` (including `on` with `filters.bin`); integration tests for the four width cases, `eval` raw, both-violation status and other-G stale outputs; `tdp_stage4_resume.bash` covers the stop flag and the marker; the Rust-driver case of `tdp_defaults.bash` now runs d64L8, d64A (TOML `preset = "A"`) and d128 (validated size, no `--mcd`) against the oracle transcripts. `docs/TDP_FOUR_STAGE_PIPELINE.md` marks its pre-port passages as superseded; D10b and review row 37 state the actual cache-neutrality evidence.

**Reproduction round 2 (§10.7), n=64, L8 and A, at `6400aa53`** (binaries copied to `port_runs/repro_r3_{L8,A}/bin/` with sha256 `PROVENANCE`; both runs detached from the session with `setsid`; fresh private seeds; `/usr/bin/python3` 3.9.6; baseline `hmap_affine`). The managed driver ran with `PATH=<cargo>:/usr/bin:/bin`, so `/bin/bash` 3.2 came first: it skipped it and ran `/opt/homebrew/bin/bash` 5.3. Both: **RESULT: ALL IDENTICAL**. Stage 2 (`tdp.mpmct1`, source C, sandwich), stage 3 (`s3.mpmct1` = `db_mixing.mpmct1`), the stage-2 reports (mult, span txt/tsv), every stage-4 file (packed and bounds, `rep512.mpmct1`, bounds, the three TSVs, the pack/anf/link reports), `deliverable.bin`, `deliverable.tdp4[48:]` and `deliverable.tdp4` = `container_header.py` of the oracle body; `stop_reason=moves budget spent`, `frozen_filter=1`, `pinned_recipe=val_chain-L8-20261007` / `val_chain-A-20261007` (store fingerprint `fd1983677e4d5b74267d3802c9c34642`); four inherited decoys refused; no seed value in any log. Wall time, val_chain.sh / managed: L8 1,842 / 1,493 s, A 1,832 / 1,535 s. The n=8 reproductions (`port_runs/fix_r2/repro_n8_{L8,A}`) also pass: A all identical; L8 the faithful failure path (exit 5, `stage 4 tdp_packing: exposure floor missed` from the marker, the oracle's `rep512` fails `check-final` the same way, `--allow-floor-miss` payload identical). This is also the evidence for D10b at n=64 in config A.

**Acceptance round 3, vL64a (n=64, L8; 2026-10-09, binary built from `09552766` = review fixes round 2, clean tree; outputs `port_runs/vL64a_r3/`, binary copy and sha256 in `bin/`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64 --label vL64a` with **no packing flag**; the report records the recommended G=512 chain as the defaults (`window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`), i.e. `window_pack3.py 512 512 24576 3000` + `repair.sh` (budget 65536) + `anf_dump_eval.py`, as `val_chain.sh` runs them. Exit 0, `status=ok`: pack 1,593,054 gates in 2,586 windows; repair cancels 58,040 pairs, fix 65 → 64, fallback 2 + 0 splits, 1,476,974 gates and 2,587 boundaries; final scan 64 visible / 0 non-output, 0 windows over budget; 2,588 ANF windows, 1,659,662 monomials. Every counter equals round 2's. Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (32,783,935 bytes) are byte-identical; `rep512.cview.log` and `repair512.log` are identical after replacing the output directory. Payload: `deliverable.tdp4` = 48-byte `LMTDPANF` v1 header + a body byte-identical to the archived pre-header `deliverable.bin`; `decode --format container` and `decode --format raw --n 64` (on the archive) give the same stats (2,588 windows, 1,659,662 monomials, max support 253). There is no difference to explain. A fresh sandboxed oracle rerun (`port_runs/vL64a_r3_oracle/`, `/usr/bin/python3` 3.9.6, baseline `hmap_affine`) reproduced the same eleven files byte for byte (its `deliver.txt` differs only in timings and the gz/xz lines `val_chain.sh` appends). Independent checks on the Rust outputs: `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (HEAD, `security-tools`) reports 1000 born values, 64 visible (6.4 %, all t*=1), on 64 distinct wires, each the wire's last writer in C: the exact floor; TSVs identical to the oracle's. `verify_zero_slice C rep512.mpmct1 64 512 <random seed>` and the same on `packed_G512.mpmct1` PASS. The oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte (proxy 11,743 values, L2 5,366 / M2 58,598; froth 78,179, L2 1,126 / M2 2,330). Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN` set passes both the `tdp_gen.sh` cases (d64L8, d64A, d128, d16) and the Rust-driver cases (d64L8, d64A, d128): gen argv, mixer argv, recipe environment, allow-list and stage-4 calls identical to the `val_chain.sh` transcripts. A second Rust run under load gives identical files (`packing.report` differs only in `time.*`).

| stage 4 on vL64a, round 3 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load 1-2 | 32.0 s | 112.0 s | 1.05 GB |
| Rust, same binary, load 7-9 (sibling sessions running) | 42.4 s | 126.4 s | 1.04 GB |
| Python oracle rerun, load 7-14: pack / repair / serialize | 76.6 / 236.8 / 58.5 = 371.9 s | 346.7 s | 1.76 GB (packer; repair 0.90, serialize 0.61) |

On the quiet machine the Rust stage 4, its checks included, takes 0.09x the oracle's wall time (0.32x its CPU) and 0.6x its peak memory. The oracle's time was measured under load, but it is mostly single-threaded and its CPU sum matches round 1's quiet rerun (327 s), so the ratio holds within about 10 %.

**Acceptance round 3, vA64a (n=64, config A; 2026-10-09, binary built from `09552766` = review fixes round 2, clean tree; outputs `port_runs/vA64a_r3/`, sha256 of the three binaries in `logs/binary.sha256`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64 --label vA64a` with **no packing flag** (the report records the recommended G=512 chain as the defaults); exit 0, `status=ok`: pack 1,647,349 gates in 2,774 windows; repair cancels 54,318 pairs, floor at fix iteration 1, fallback 2 + 0 splits, 1,538,713 gates and 2,775 boundaries; final scan 64 visible / 0 non-output, 0 windows over budget; 2,776 ANF windows, 1,412,026 monomials. A second run with every packing flag passed explicitly at its recommended value (`--window-gates 512 --interval-cap 512 --monomial-budget 24576 --span-cap 3000 --split-budget 65536 --fix-iterations 4 --fallback-iterations 4 --anf-cap-semantics cpython39`) gives the same files (`packing.report` differs only in `time.*`). Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (26,036,498 bytes) are byte-identical; `rep512.cview.log` and `repair512.log` identical after replacing the output directory. Payload: `deliverable.tdp4` = 48-byte `LMTDPANF` v1 header + a body byte-identical to the archived pre-header `deliverable.bin`, and equal to `container_header.py` applied to the archived body; `decode --format container` and `decode --format raw --n 64` on the archive give the same stats (2,776 windows, 1,412,026 monomials, max support 255). There is no difference to explain; every output also equals round 2's except `packing.report` (`binary_xxh3`, `time.*`). A fresh sandboxed oracle rerun (`port_runs/vA64a_r3_oracle/`, `/usr/bin/python3` 3.9, baseline `hmap_affine`, HEAD oracle scripts with path rewrites only) reproduced all eleven files byte for byte. Independent checks on the Rust outputs: `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (HEAD, `security-tools`) reports 1000 born values, 64 visible (6.4 %, all t*=1) on 64 distinct wires, each the wire's last writer in C: the exact floor; TSVs identical to the oracle's. `verify_zero_slice C rep512.mpmct1 64 512 <random seed>` and the same on `packed_G512.mpmct1` PASS. The oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte (proxy 15,455 values, L2 6,039 / M2 73,178; froth 74,595, L2 373 / M2 780).

| stage 4 on vA64a, round 3 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load about 1 at start | 21.5 s | 66.8 s | 1.18 GB |
| Rust, same binary, load about 9 (sibling sessions running) | 23.5 s | 73.6 s | 1.19 GB |
| Python oracle rerun, load 7-9: pack / repair / serialize | 49.7 / 153.8 / 37.5 = 241.0 s | 232.2 s | 1.68 GB (packer; repair 0.92, serialize 0.58) |

At comparable load the Rust stage 4, its checks included, takes about 0.10x the oracle's wall time (0.32x its CPU) and 0.7x its peak memory.


**Acceptance round 4, vA64a (n=64, config A; 2026-10-10, binaries built from `a525f894` = review fixes round 3, clean tree; outputs `port_runs/vA64a_r4/`, sha256 of the three binaries in `logs/binary.sha256`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64 --label vA64a` with **no packing flag**; exit 0, `status=ok`: pack 1,647,349 gates in 2,774 windows; repair 1,538,713 gates and 2,775 boundaries; final scan 64 visible / 0 non-output, 0 windows over budget. A second run with every packing flag passed explicitly at its recommended value gives the same files (`packing.report` differs only in `time.*`). Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (26,036,498 bytes) are byte-identical; `rep512.cview.log` and `repair512.log` are identical after replacing the output directory. `deliverable.tdp4[48:]` equals the archived pre-header `deliverable.bin`, and the whole container equals `container_header.py` applied to the archived body; `decode --format container` and `decode --format raw --n 64` (archive) agree (2,776 windows, 1,412,026 monomials). There is no difference to explain; every output equals round 3's except `packing.report` (`binary_xxh3`, `time.*`). A fresh sandboxed oracle rerun (`port_runs/vA64a_r4_oracle/`, `/usr/bin/python3` 3.9, baseline `hmap_affine`, HEAD oracle scripts with path rewrites only) reproduced all eleven files byte for byte. Independent checks on the Rust outputs: `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (`a525f894`, `security-tools`): 1000 born, 64 visible (6.4 %, all t*=1) on 64 distinct wires, each the wire's last writer in C, the exact floor; TSVs identical to the oracle's. `verify_zero_slice C rep512.mpmct1 64 512 <random seed>` and the same on `packed_G512.mpmct1` PASS. `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte (proxy 15,455, L2 6,039 / M2 73,178; froth 74,595, L2 373 / M2 780). Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN=target/release/local_mixing_bin` passes (tdp_gen.sh d16/d64L8/d64A/d128, Rust driver d64L8/d64A/d128 identical to the `val_chain.sh` transcripts).

| stage 4 on vA64a, round 4 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (defaults, all checks), load about 10 | 25.8 s | 75.8 s | 1.17 GB |
| Rust, explicit flags, same load | 31.0 s | 72.9 s | 1.15 GB |
| Python oracle rerun, load 6-12: pack / repair / serialize | 52.5 / 153.9 / 29.9 = 236.3 s | 231.5 s | 1.68 GB (packer; repair 0.88, serialize 0.61) |

At comparable load the Rust stage 4, its checks included, takes 0.11x the oracle's wall time (0.33x its CPU) and 0.70x its peak memory.

**Acceptance round 4, vL64a (n=64, L8; 2026-10-10, binaries built from `a525f894` = review fixes round 3, clean tree; outputs `port_runs/vL64a_r4/`, copies of `tdp_packing`, `verify_zero_slice`, `hmap_affine` and their sha256 in `bin/`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 64 --label vL64a --out-dir packing` with **no packing flag** (the defaults are the recommended G=512 chain: `window_pack3.py 512 512 24576 3000` + `repair.sh` with split budget 65536 + `anf_dump_eval.py`). Exit 0, `status=ok`: pack 1,593,054 gates in 2,586 windows; repair cancels 58,040 pairs, fix 65 → 64, fallback 2 + 0 splits, 1,476,974 gates and 2,587 boundaries; final scan 64 visible / 0 non-output, 0 windows over budget; 2,588 ANF windows, 1,659,662 monomials. Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (32,783,935 bytes) are byte-identical; `rep512.cview.log` and `repair512.log` are identical after replacing the output directory. Payload: `deliverable.tdp4[48:]` (after the `LMTDPANF` v1 header) is byte-identical to the archived pre-header `deliverable.bin`; `decode --format container` and `decode --format raw --n 64` (archive) agree (2,588 windows, 1,659,662 monomials, max support 253, max changed 215). There is no difference to explain; every output equals round 3's except `packing.report` (`binary_xxh3`, `time.*`). A fresh sandboxed oracle rerun (`port_runs/vL64a_r4_oracle/`, `/usr/bin/python3` 3.9.6, baseline `hmap_affine`, round 3's sandbox scripts with the path rewritten) reproduced all eleven files byte for byte. Independent checks on the Rust outputs: `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (`a525f894`, `security-tools`; `cview_check/`) reports 1000 born values, 64 visible (6.4 %, all t*=1) on 64 distinct wires, each the wire's last writer in C: the exact floor; TSVs and log (path-normalized) identical to the oracle's. `verify_zero_slice C rep512.mpmct1 64 512 <random seed>` and the same on `packed_G512.mpmct1` PASS (seeds not recorded). The oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte (proxy 11,743 values, L2 5,366 / M2 58,598; froth 78,179, L2 1,126 / M2 2,330). Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN=target/release/local_mixing_bin` passes (tdp_gen.sh d16/d64L8/d64A/d128, Rust driver d64L8/d64A/d128 identical to the `val_chain.sh` transcripts). A second Rust run (`packing2/`) gives identical files (`packing.report` differs only in `time.*`, the two logs only in the output directory).

| stage 4 on vL64a, round 4 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load 10-21 (sibling sessions) | 56.8 s | 133.3 s | 1.04 GB |
| Rust, second run, load about 5: pack / repair / reports / emit+checks | 32.6 s (17.0 / 4.2 / 2.0 / 9.1) | 115.5 s | 1.05 GB |
| Python oracle rerun, load 6-13: pack / repair / serialize | 76.3 / 219.3 / 48.5 = 344.0 s | 71.4 / 218.6 / 48.3 = 338.4 s | 1.66 GB (packer; repair 0.90, serialize 0.61) |

At comparable load the Rust stage 4, its checks included, takes 0.095x the oracle's wall time (0.34x its CPU) and 0.63x its peak memory, as in round 3.
**Acceptance round 3, vA128a (n=128, config A; 2026-10-09, binary built from `09552766` = review fixes round 2, clean tree; outputs `port_runs/vA128a_r3/`, sha256 of the three binaries in `binary.sha256`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 128 --label vA128a` with **no packing flag**. The report records the recommended G=512 chain as the defaults (`window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`), which is `window_pack3.py 512 512 24576 3000` + `repair.sh` (budget 65536) + `anf_dump_eval.py` as `val_chain.sh` runs them. Exit 0, `status=ok`. Pack: 4,757,723 gates in 7,376 windows. Repair: 159,915 cancelled pairs, the floor reached at fix iteration 1, fallback 1 + 0 splits, 4,437,893 gates and 7,376 boundaries. Final scan: 128 visible / 0 non-output, 0 windows over budget. 7,377 ANF windows, 4,185,577 monomials. Every counter equals rounds 1 and 2, and `packing.report` equals round 2's except `binary_xxh3` and `time.*`.
- Against the archived oracle, with no normalization, these files are byte-identical: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (106,890,630 bytes). `rep512.cview.log` and `repair512.log` are identical after replacing the output directory.
- Payload: `deliverable.tdp4` (106,890,678 bytes) is a 48-byte `LMTDPANF` v1 header + a body byte-identical to the archived pre-header `deliverable.bin`. It also equals `container_header.py` applied to the archived body, and `decode --format container --stats` accepts it (7,377 windows, 4,185,577 monomials, max support 489). There is no difference to explain.
- A fresh sandboxed oracle rerun (`oracle_sb/`, `/usr/bin/python3` 3.9.6, baseline `hmap_affine`, the main checkout's oracle scripts with path rewrites only, identical after rewriting to round 2's sandbox) reproduced all eleven files byte for byte. They also equal the Rust files. Its `deliver.txt` differs from the archive only in timings and in the gz/xz size lines.
- Independent checks on the Rust outputs (`checks/`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (HEAD, `security-tools`) reports 3,000 born values and **128 visible** (4.3 %, all t*=1). This is the exact floor, and the TSVs are identical to the run's and the oracle's.
  - `verify_zero_slice C rep512.mpmct1 128 512 <random seed>` and the same check on `packed_G512.mpmct1` both PASS.
  - The oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte: proxy 54,345 values, L2 18,521 / M2 250,448; froth 280,727 values, L2 2,219 / M2 4,661.
- Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN` set (HEAD build) passes the `tdp_gen.sh` cases (d64L8, d64A, d128, d16) and the Rust-driver cases (d64L8, d64A, d128). Gen argv, mixer argv, recipe environment, allow-list and stage-4 calls are identical to the `val_chain.sh` transcripts. `tdp_flags.bash` and `tdp_block_size.bash` pass.

| stage 4 on vA128a, round 3 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load about 1 at start | 55.0 s | 291.9 s | 2.07 GB |
| — pack / repair / reports / emit+checks | 25.4 / 16.0 / 2.6 / 10.1 s | | |
| Python oracle rerun, load 5-9: pack / repair / serialize | 155.8 / 770.5 / 81.1 = 1,007 s | 146.2 / 783.4 / 81.0 = 1,010.6 s | 5.32 GB (packer; repair 3.54, serialize 2.08) |

The Rust stage 4, its checks included, takes 0.055x the oracle's wall time (18x faster), 0.29x its CPU and 0.39x its peak memory. It retires the same 2.885 × 10¹² instructions as in rounds 1 and 2. The oracle is mostly single-threaded, so the higher load during its rerun hurts it less than it would hurt Rust. The Rust run is within the §9 n=128 targets (≤ 6 min, ≤ 3 GB).

**Acceptance round 3, vL128a (n=128, L8; 2026-10-09, binaries built from `09552766` = review fixes round 2, clean tree; outputs `port_runs/vL128a_r3/`, binary copies `*.bin` and their sha256 in `binary.sha256`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 128 --label vL128a` with **no packing flag**; the report records the recommended G=512 chain as the defaults (`window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`). Exit 0, `status=ok`: pack 4,596,808 gates in 7,059 windows; repair cancels 168,680 pairs (6 passes), fix 132 → 130 then exhausted (4 iterations), fallback merge + 1 split 130 → 128, 4,259,448 gates and 7,058 boundaries; final scan 128 visible / 0 non-output, 0 windows over budget; 7,059 ANF windows, 4,065,912 monomials (max support 491). Every check in the report is `ok` (reorder and repair certificates, pack verify, equivalence, windows = window ANF, decode, eval, zero slice with 4,096 samples). `packing.report` equals round 2's except `binary_xxh3` and `time.*`.

- Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (102,818,808 bytes) are byte-identical, with the Appendix B sha256s. `rep512.cview.log` and `repair512.log` are identical after replacing the output directory, line order included.
- Payload: `deliverable.tdp4` (102,818,856 bytes) is a 48-byte `LMTDPANF` v1 header + a body byte-identical to the archived pre-header `deliverable.bin`, and equals `container_header.py` applied to the archived body (n=128, 512 state wires); `decode --format container --stats` accepts it (7,059 windows, 4,065,912 monomials). There is no difference to explain.
- A fresh sandboxed oracle rerun (`oracle_sb/`, `/usr/bin/python3` 3.9.6, baseline `hmap_affine`, HEAD oracle scripts with path rewrites only, `oracle_chain.sh`) reproduced all eleven files byte for byte, and the two logs after path normalization.
- Independent checks on the Rust outputs (`checks/`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (HEAD, `security-tools`) reports 3,000 born values and **128 visible** (4.3 %, all t*=1), on 128 distinct wires, each the wire's last writer in C: the exact floor. Its three TSVs are identical to the run's and the oracle's.
  - `verify_zero_slice C rep512.mpmct1 128 512 <random seed>` and the same check on `packed_G512.mpmct1` PASS (seeds only in the private `checks/vzs_seed.txt`).
  - The oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte: proxy 34,958 values, L2 16,106 / M2 187,248; froth 285,325 values, L2 5,579 / M2 11,622.
- Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN` set (HEAD build) passes the `tdp_gen.sh` cases (d16, d64L8, d64A, d128) and the Rust-driver cases (d64L8, d64A, d128): gen argv, mixer argv, recipe environment, allow-list and the stage-4 call identical to the `val_chain.sh` transcripts. `tdp_flags.bash` and `tdp_block_size.bash` pass.
- A second Rust run (`packing_t2/`) gives identical files (`packing.report` differs only in `time.*`, the two logs only in the output directory).

| stage 4 on vL128a, round 3 | wall | CPU (user+sys) | peak RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load 1-8 at start | 105.2 s | 590.7 s | 2.06 GB |
| — pack / repair / reports / emit+checks | 43.6 / 36.0 / 4.2 / 20.5 s | | |
| Rust, second run, load about 4 at start | 96.7 s | 586.9 s | 2.11 GB |
| Python oracle rerun, load 4-11: pack / repair / serialize | 200.1 / 1,296.7 / 118.6 = 1,615 s | 188.7 / 1,306.2 / 118.4 = 1,613 s | 4.89 GB (packer; repair 3.40, serialize 2.00) |

The Rust stage 4, its checks included, takes 0.065x the oracle's wall time (15x faster), 0.37x its CPU and 0.42x its peak memory. This time the oracle ran nearly uncontended (CPU ≈ wall), so the ratio is fair; the round-1 6,921 s wall was contention. Both §9 n=128 targets hold (≤ 6 min, ≤ 3 GB).


**Acceptance round 4, vL128a (n=128, L8; 2026-10-10, binaries built from `a525f894` = review fixes round 3, clean tree; outputs `port_runs/vL128a_r4/`, binary copies `*.bin` and their sha256 in `binary.sha256`).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 s3.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n 128 --label vL128a` with **no packing flag** (the defaults are the recommended G=512 chain: `window_gates=512 interval_cap=512 monomial_budget=24576 span_cap=3000 split_budget=65536 fix_iterations=4 fallback_iterations=4 anf_cap_semantics=cpython39`). Exit 0, `status=ok`: pack 4,596,808 gates in 7,059 windows; repair 4,259,448 gates and 7,058 boundaries; final scan 128 visible / 0 non-output, 0 windows over budget. Every check in the report is `ok` (reorder and repair certificates, pack verify, equivalence, windows = window ANF 0/7059, decode, eval, zero slice with 4,096 samples). `packing.report` equals round 3's except `binary_xxh3` and `time.*`.

- Against the archived oracle, with no normalization: `packed_G512.mpmct1`, `.bounds`, `pack_G512.txt`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `anf_rep512.txt`, `link_rep512.txt` and `deliverable.bin` (102,818,808 bytes) are byte-identical. `rep512.cview.log` and `repair512.log` are identical after replacing the output directory.
- Payload: `deliverable.tdp4` (102,818,856 bytes) = 48-byte `LMTDPANF` v1 header + a body byte-identical to the archived pre-header `deliverable.bin`; `decode --format container --stats` accepts it (7,059 windows, 4,065,912 monomials, max support 491, 128 source / 512 state wires). There is no difference to explain.
- A fresh sandboxed oracle rerun (`oracle_sb/`, round 3's sandbox scripts with the directory renamed, `/usr/bin/python3` 3.9.6, baseline `hmap_affine`; the oracle scripts and the `hmap_affine` sources are unchanged since `09552766`) reproduced all eleven files byte for byte and the two logs after path normalization.
- Independent checks on the Rust outputs (`checks/`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` (`a525f894`, `security-tools`): 3,000 born values, **128 visible** (4.3 %, all t*=1), on 128 distinct wires, each the wire's last writer in C: the exact floor. Its three TSVs are identical to the run's and the oracle's.
  - `verify_zero_slice C rep512.mpmct1 128 512 <random seed>` and the same on `packed_G512.mpmct1` PASS (seeds only in the private `checks/vzs_seed.txt`).
  - The oracle's `link_census.py` on the Rust `rep512` equals the archived and the Rust `link_rep512.txt` byte for byte: proxy 34,958 values, L2 16,106 / M2 187,248; froth 285,325 values, L2 5,579 / M2 11,622.
- Driver defaults: `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN=target/release/local_mixing_bin` (`a525f894`) passes the `tdp_gen.sh` cases (d64L8, d64A, d128, d16) and the Rust-driver cases (d64L8, d64A, d128): gen argv, mixer argv, recipe environment, allow-list and the stage-4 call are identical to the `val_chain.sh` transcripts, i.e. the Rust driver's defaults are the configuration of the latest chosen Python runs. `tdp_flags.bash`, `tdp_block_size.bash` and `cargo fmt --check` pass.

| stage 4 on vL128a, round 4 | wall | CPU (user+sys) | max RSS |
|---|---|---|---|
| Rust `tdp_packing run` (all checks), load 1.6 at start | 99.7 s | 590.1 s | 1.86 GB (footprint 1.74) |
| — pack / repair / reports / emit+checks | 41.5 / 33.2 / 4.3 / 19.7 s | | |
| Python oracle rerun, load 4-15: pack / repair / serialize | 207.4 / 1,254.6 / 123.1 = 1,585 s | 191.9 / 1,273.1 / 121.8 = 1,587 s | 4.95 GB (packer; repair 3.40, serialize 2.00) |

The Rust stage 4, its checks included, takes 0.063x the oracle's wall time (16x faster), 0.37x its CPU and 0.38x its peak memory; the oracle ran nearly uncontended (CPU ≈ wall). Both §9 n=128 targets hold (≤ 6 min, ≤ 3 GB).

---

**Review fixes round 3 (2026-10-10; WIP commit `52c65427`, then the docs).** Nine minor findings, each reproduced before fixing; the acceptance round 3 cases all passed and needed no change.

- **nonlinear291 with stage 4.** Its stage-2 width is 24n + 12·(2|C| + 2s + n) + 29 (`nonlinear_tdp_wire_count`; 3,677 at n=16, |C|=64), beyond stage 4's 1,024 state wires for any realistic size, so a nonlinear291 run used to fail in stage 4 after stages 1-3. The validator now checks the width `nonlinear_tdp_resource_plan` plans against `MAX_STATE_WIRES` (new in `stages::packing::types`) whenever stage 4 is on, and `tdp_gen.sh` does the same with the same formula; both refuse before anything runs and point at `run.stop_after_stage = 3`. Small plans that fit (n=4, |C|=1: 389 wires) stay accepted.
- **Stage-3 completion marker.** `$RUN/stage3.recipe` (`tdp_stage3_recipe=1`, the schedule arguments, HMAC tags of `tdp.mpmct1` and `db_mixing.mpmct1`, the stop reason) is written only after an accepted stop; a resume skips stage 3 only when it matches, so a truncated or interrupted output, a changed stage-2 input or a changed schedule re-runs stage 3 instead of failing stage 4 on every resume. The store is not part of it (a resume after stage 3 needs no store, as before). A run from before the marker is adopted without re-mixing when `packing.recipe` carries the tags of exactly these stage-2 and stage-3 bytes and `stage3.settings` records an accepted stop. `circuit_mixer` writes `--output` through `<path>.tmp`, fsync, rename (`formats::write_mpmct_atomic`). The Rust driver's store prediction (`validate_external_paths`) counts `db_mixing.mpmct1` as done only next to a marker.
- **`--source-wires` overflow.** `container` and `serialize` bound n (1..=65536) before `2·n`, which wrapped at 2^63 in release builds and wrote `source_wires=0`; `eval --format raw` saturates its `4·n` default.
- **Resource bounds.** The monomial and split budgets are capped at 2^18 = 262,144 (`MAX_MONOMIAL_BUDGET`; Rust config, `tdp_gen.sh`, `tdp_packing`), measured: 10^6-10^9 grew to 5-9 GB, 2^20 held 2-3 GB at one monomial word, 2^18 peaked at 1.2 GB (G = W_cap = 10^6, n=8 fixture, 156 s). Emission uses the same ceiling (`EMIT_BUDGET`) and returns `OverBudget` (exit 6) instead of panicking or allocating without limit: serializing the whole n=8 chain as one window now exits 6 in 0.9 s at 0.3 GB (it reached 5.5 GB before). Bit-exactness is kept: a cap only ever aborts, and every window the repair leaves fits a split budget ≤ 2^18 or has at most 2 gates. The `OverBudget` message names the first window.
- **Docs.** TDP_FOUR_STAGE_PIPELINE §1.8 open issues carry status notes (seeds, argv, |C| default, presets/`val_chain.sh` resolved; pins, `rg_freq`, D⁻¹ test still open); §3 move budget and §6.3 library size describe the pre-port driver as such; the stage-4 snapshot note records that the oracle snapshot equals the run-tree copies (`cmp` at HEAD) and has the patched slide. The cache sizes are described as classified neutral with the evidence actually available (identical stage-3 bytes against the reference runs; `cache_neutrality` never run) in `env_registry.rs`, §2.2, §11.5, the P7 row, `tdp_gen.sh` and `tdp_pipeline.md`. §2.2 says `SANDWICH_VARIANT` is not exported (built-in `classic`); §11.5 lists the implemented classes (`FROZEN_DB_DIR` is `input`, `MIXER_DUMP_OUT` `forbidden`, plus `input` and `diagnostic`).
- **Defaults tie.** `packing_defaults_are_the_stage_four_binary_defaults` (unit) ties `PackingConfig::default()` to `PackParams::PRODUCTION`, `RepairParams::default()`, the literal `val_chain.sh` values and `tdp_gen.sh`'s `PACK_*` initial values and `pf` omission defaults; `run_end_to_end_n8` (integration) pins all nine `param.*` defaults of a flagless `tdp_packing run`.
- **Tests added.** `nonlinear_width_beyond_stage_four_is_refused_up_front`, `packing_budgets_are_bounded_by_the_resource_limit`, the defaults tie above, marker cases in the two resume unit tests; `tdp_flags.bash` (nonlinear291 n=16 refused before any directory, budgets 262,145 refused); `tdp_stage4_resume.bash` (stage-3 skip on a matching marker, truncated output and changed schedule re-run, legacy adoption, no marker and no stage-4 marker re-run); `tdp_packing` integration (`--source-wires` 0, 65537, 2^32, 2^63 refused by `container` and `serialize` with nothing written; whole-chain emission exit 6; budgets 262,145 exit 2).
- **Checked.** G1-G7, `cargo fmt --check` (incl. the mixer topic files), clippy (no new warnings), `cargo test --features security-tools --test hmap_affine_exposure`, the CI shell set (`tdp_defaults.bash` with `LOCAL_MIXING_BIN`, `tdp_seeds`, `tdp_stage4_resume`, `tdp_stage12_parity`); `tdp_reproduce_small.bash` run for real (regular store, pinned baseline `hmap_affine`): ALL IDENTICAL; `tests/stage4/differential.sh` with a random seed, K=12: 24/24 identical. G5 has the 3 baseline numpy import errors only.

## 13. Open decisions for RC

1. **Seed entropy beyond 64 bits** (§2.3): route (a) a 256-bit ChaCha-derived C and gadget streams, which changes bytes and needs re-validation, or route (b) a CSPRNG-generated C via `source.path` for production deliverables. Until RC decides, the default is the reference algorithm with 63-bit seeds, recorded as a stopgap.
2. **Sizes for other n.** Today other n require an explicit size or the `formula` opt-in (§2.4). RC may add validated rows (for example after new runs at n=96 or n=256).
3. **Repair improvements** (TDP doc §4.9): slide across markers, marker-identity cuts, all exposing cuts, finer split candidates. Each would be a new `RepairMode` behind a flag, never the default until RC re-validates.
4. **Typed stage-3 knobs** (`MIXER_STOP_AT_*`, `TWIST_G57_*`, `MIXER_DB_LEN_BAND`, `MIN_DIR_LOOKUP`): a follow-up after the port. The driver already forbids them.
5. **`CapSemantics::Canonical` as the default:** never, unless RC re-validates. It changes bytes relative to the oracle on real runs.

Revision 1's open items "source size" and "where the seed values live" are resolved: §2.4, and the private `$RUN/SEEDS` with no seed-derived value in any publishable file (§8).

## 14. Documentation to update (P9)

- `docs/TDP_FOUR_STAGE_PIPELINE.md`:
  - §4.5 and line 171: one canonical `order_repair.py`;
  - §4.7: all 8 replicates re-repaired, and the anchor result;
  - §4.9 and §5.7: port status, linking here;
  - §6.9: HA provenance result;
  - §6.10 and §6.12: decisions taken;
  - Appendix A: gaps closed by D1-D18.
- `docs/tdp_pipeline.md`:
  - quick start for stages 1-4;
  - config v2 (presets, schedule, size table, seeds format 2, `[packing]`);
  - the environment policy;
  - "Stage 4: packing" (`tdp_packing`, outputs, exit codes, `deliverable.tdp4`);
  - the reference driver becomes a parity tool.
- `docs/ANF_WINDOW_DELIVERY.md`: §1.2 wrongly says every window is re-verified; the format section adds container v1 under its own name.
- `docs/windowed_delivery_design.md` §7: replaced by a pointer here.
- `docs/STAGE2_LINKABILITY.md` §6: the managed default is L8; §7: small-n L8 gen feasibility.
- `scripts/windowed_delivery/README.md`: the oracle role, the sandbox rule, the pinned interpreter and baseline HA.
- `configs/tdp.toml`: the v2 template.

---

## Appendix A. Golden vectors (from CPython 3.9.6)

**`random.Random(20261004)`:**

- first four `genrand_uint32` outputs: `749935133, 232581420, 3240862439, 1300416022`;
- `getrandbits(1024)` for wire 0: low64 `0x0ddce92c2cb31a1d`, high64 `0x3f688f70a42f573a`, low128 `0x4d82c616c12ba2e70ddce92c2cb31a1d`;
- wire 1: low64 `0x12e4aa97d52ee91c`;
- `blake2b(digest_size=16)` of wire 0 (`to_bytes(128,'little')`): `10443c04d722b9ea2873b4564780ebb1`. *Precisely:* this is the key of the NORMALIZED value (wire 0 has sample 0 set, so it is complemented first), which is what the packer hashes; the raw value's key is `69f9d20df3b9c38754a0159c292000f0` (both checked in `tests/unit/stages/packing/blake2b.rs`).

**`getrandbits(1024)`, low128 of the first draw:**

- `Random(77)`: `0x32833106536e95df40b2d002cc92d33d`;
- `Random(1)`: `0xcd613e30d8f16adf91b7584a2265b1f5`.

**Set order** (`{1,2,3}`, discard 2, add 10): `[10, 1, 3]` on 3.9.6, because 3.9 does not reuse the dummy slot.

## Appendix B. Parity targets (sha256 of archived files in `runs/ncdb_ab_20261001/waist/<tag>/`)

Hashes of the source C, `tdp.mpmct1` and `s3.mpmct1` are intentionally not listed (§2.3). If §10.5 step 0 or §10.8 falls back, Appendix B′ or B″ will be added; these tables are not edited.

| tag | `packed_G512.mpmct1` | `packed_G512.mpmct1.bounds` |
|---|---|---|
| vA64a | `bde5e0fc6f170b725be484433e50e3a1d1a245fdaec17da6e5918c36b23419cc` | `18138db7c3a4c1287ef98ac73ba56e126369174bec93a6740591d83514a9247e` |
| vA64b | `699d3c6eed512ba1af47bee929022d0cfdf3ffbe782f66d8cfed7f0be2fceb61` | `2a8624a5b8084ceef8d7ab706bf3b799bae2399d20d19bb04c6b49adacc1e98c` |
| vL64a | `f2c6b53af825f84b6698bf95c01f653d087f7f1e30bf678279a194e3a720f69e` | `a5cbe27daee2e9b638ee30ecd64d309f65740b89ec50aed53d5c749411658076` |
| vL64b | `fe4d3d7db336a1ff33bc8b91ab9453cbced29e42dc7f8204ad4f3bebe3b30ab0` | `01f3ce503e98c7352890118c9b911d5d2aacdba88ef85317a3dbb82fc17dec23` |
| vA128a | `8ed20637c33b3dfe709af129976a8f5e10b44ae1d44fb69dd62d8073f2d051a1` | `09654ee2e91efca4c77a37cff34d81f8eb0c9b9dabe41356044338e92019796c` |
| vA128b | `deac6d86150738c304ee7a92dd9af22f282def29f24680487e6aec8bd7c3a28f` | `7ccc526dc4563a3d9781ad9f48265d6974d28f7e45a563f154be1ccbf633c19a` |
| vL128a | `932fc7b45794ee04519a07c82fe9d70d10ef0a8cc4e90867f560ad0116aa528b` | `bf19c8a1434ea4b99b2c642b72f2e9436341f137f7dde44fb293c3c90f208832` |
| vL128b | `f9d22195aeeec4b629cfdc196e69155dffafac92824502770e328fd961ee59d9` | `8a19eb139ff6c18522917a1810306187e7d595efc9ccf8a2ed14b1c595d639f7` |

| tag | `rep512.mpmct1` | `rep512.bounds` | `deliverable.bin` (raw body) |
|---|---|---|---|
| vA64a | `fcec8a2cc34f0d3250b24104e5b1d54f1cfb9ddc2047c10f7b6505564ab25b1a` | `489606afcb18fc989ed16a9e588229c6ec3b28f93106bfef20e403a2f190c800` | `430c6e3b249629f14508467ae33175ced9918efd7fc85886a1a3d2693c4ef93a` |
| vA64b | `dbe5220b1273fe8e6b9968e6d696f2b9a026f719493af5d47325d4b9bd2783f4` | `21e3ca2d9738fd361bf41ac80e094dad82949b614707e2d42d8fb3cdcfbde9c5` | `9b3c0a593ad36dab7bf111f2c0a510c5a28878fc831e2c249fe96e3bbf4f8a8a` |
| vL64a | `274570154da5ec5152e8f520d3c552b9fdbee4ebe023c1915e30e82e1441210c` | `4e9ccd63faa03dc18021debe54d43a3cce510490cd80289cbcffcd90d225b90d` | `e32e0cf271373d6bd8046a527cbdce6ad1404d60dce75a6234b9d94adfd6cc7b` |
| vL64b | `fd9e4bb5d86cc6a22ccb6f156ce1a78dd902461ea1ec9503d25f388362f2e321` | `47acd95851c85cd9a58331903e1ac156301960cdbaa1c07bb77c9f8ede537dc8` | `a9e5ebc4df0e710f19eebac7b44fcd55185b0ef93dc7c67edebcd67acbacdf25` |
| vA128a | `d335e9868f25ebcc6ee738be4e791f433e202b9436c1446a975b4666f8f2aba5` | `55819935cedc23705a55223484bad33cae81b9caa5830abb62d2897981abafc6` | `7b1ad3801169ff65bb27fd308d035f07ce1132d18308ae6884e362b807ebf26d` |
| vA128b | `4ae19d27b7b66a23d4db7a5646338a77f46332b8cd6851d7f1b2cf99d58b13d0` | `a6384f2a4fccd8f56c6b68cb2274476403cfd9a8bd6d07b4dae120c1e6546003` | `73fb706e9d5558be291aa75d4d2307fef6ddb7c9545eaf7dd75cb6eab109ae9b` |
| vL128a | `e987ce0407b5160ff0359f08d4480ead39c63625b2c97b1235d6728afdb8131f` | `9ed0c259af91a4cd6c1a2bf882c86804409a6cc9b5a13d3cfc9c75a649895f99` | `aa2d8845db0be1c367c2e345334f1b9a2792695b726c73950990076ae437b11e` |
| vL128b | `6ed654388b6e52951eb0165c536c5a119e137eb08746daf4b26cb51b4abf2b1a` | `3150c3a4d8fc3abbc6c4c86bc358793a4c15fda95c1c92acfe3ef1284697dfa8` | `710baf59423c7f2a5ab2cb84d24d432f206c334c6ec39b7e4bebda2edc6bf68c` |

The `rep512.*` and `deliverable.bin` files are the Oct 8 re-repairs made with the canonical `order_repair.py`; the superseded versions are in `rep512_v1/`. None of the eight `.bounds` files contains a duplicate.

Archive-era `hmap_affine` (main checkout, mtime 2026-10-07 18:54:28): `94757413406e58c9fe40694e392213da015b198a0cfb22a83f4b9b79ca8a84f3`.

## Appendix C. Review issues (revision 1) and resolutions

All 37 issues are accepted; none is rejected. Duplicates are grouped. Numbers are the review's order.

| # | issue | resolution |
|---|---|---|
| 1, 20 | default source size is the formula, not the validated 1000/3000 | §2.4: the validated table is the default; other n need an explicit size or the `formula` opt-in; `source_gates_rule` is recorded. D16 |
| 2, 22 | old `local.toml` (v1 template) silently overrides the new preset; hold "30" in the manifest versus 27 in the script | §11.2: `config_version = 2` with a migration error; presets and schedule are authoritative (explicit knobs are rejected unless `custom`); `growth_work_units` is v2 and custom-only; the manifest prints resolved values only (D18); tests `v1_config_rejected_with_migration`, `old_template_with_preset_l8_rejected`, `preset_field_conflict_rejected` |
| 3, 25 | deny-list scrub misses `MIXER_DB_LEN_BAND`, `SLICE_GUARD_LEGACY`, `MIXER_STOP_AT_*`; the test filter cannot see them | §11.5: registry + allow-list launch (`env_clear` / `env -i`); inherited output-affecting variables are fatal; `env_registry_covers_every_env_read` covers every env read in `src/`; the mocks dump and assert the complete environment (§10.9 item 1). D15 |
| 4 | gen silently turns an unparsable seed into 1 | D17 and §11.3: strict positional parsing; `SEEDS` validated with `bounded_uint`; gen banner seed check without printing; `tdp_seeds.bash`. Verified 2026-10-08: an unparsable `s` positional silently became the default 24 |
| 5 | exposure-engine parity becomes self-referential after P2 | §10.1: baseline HA built from `492564f0`, sha256 pinned; used for regen, reproduction, `tdp_reproduce_small.bash` and the CI differential job; `regen.sh` refuses any other HA binary |
| 6 | historic anchor optional; post-Oct-7 changes (d9631d0e, Oct 8 mixer cleanup) could break "defaults reproduce the chosen runs" | §10.8: anchor required for vL64a and vL128a; step A (P0.5, before P7) checks gen on all 8 tags and the mixer on 2; bisect, document and re-baseline (Appendix B″) on mismatch |
| 7 | fixtures never exercise W ≥ 2, wires ≥ 61, multi-limb hash and ordering | `crafted/wide128` (W=2) and `crafted/wide300` (W=8); unit tests of hashes, PySet order and emission bytes for keys with bits ≥ 61 and ≥ 128 against recorded 3.9 output (§10.2) |
| 8 | small pyset traces cannot cover the large-table regime where aborts happen | `pyset39_large.txt` (regenerated ops, checkpoint sha256s up to 6×10^5 elements), one unit test per resize branch including the small-table purge, and the `vA64a_window` replay fixture under both `CapSemantics` |
| 9, 30 | CI repair coverage may be empty; small-cap claim unproven; budget literal; n=8 L8 feasibility; path-bearing logs | §10.3: `regen.sh` coverage assertions with a seed search; crafted cases; split budget and iteration literals rewritten with exactly-once asserts; L8 gen at n=8 and n=16 proven; logs compared after path normalization, artifacts byte for byte |
| 10 | move legality only checked by debug asserts, sampling and a gate multiset | §4.11a: exact release-mode certificates (per-wire run-canonical trace equivalence for pack and repair; cancellation replay; exact decode = ANF; exhaustive check of windows with support ≤ 16); sampled checks kept |
| 11 | exits 5/6 make reproduction seed-dependent; exit-6 predicate undefined; `--allow-*` still exits 5/6 | §5.4: exit 6 is exactly `split::fits` with the configured `CapSemantics`; `--allow-*` exits 0 with `status=floor_miss/over_budget`; §10.7 step 5 compares pre-decision artifacts and confirms the oracle's state violates the same test |
| 12 | secret-derived hashes in `packing.report`, `packing.recipe`, `tdp_command.conf` | §2.3 and §5.5: no unkeyed hash of C, G2, G3 or the seeds anywhere; input pins are HMAC-SHA256 keyed by `run_key` in private files; publishable files are listed explicitly |
| 13 | the baseline-mismatch path is a dead end; C-view provenance unresolved | §10.5 step 0: planned fallback (re-run the oracle repair and serialization on archived packs with the baseline HA, giving Appendix B′ and a diff report including visible counts and the final visible set); archive-era HA sha256 recorded (Appendix B) and copied at P2 start |
| 14 | fix-loop sequence state underspecified | §4.11 step 3: every iteration restarts from `to_seq(gates, bounds)` of the previous `from_seq`; test `fix_iteration_restarts_from_to_seq` |
| 15 | census not compared; span_s2 and mult_census dropped | §10.5 step 5 gates census parity on all 8 tags; §11.1 side reports `stage2_mult` and `stage2_span` (report-only); compared in §10.7 |
| 16, 24 | `FROZEN_FILTER` auto removes the RAM guard; neutrality unmeasured | D11: `1` when there is no `filters.bin` (the exact reference line); `0` when present until `filter_neutrality` is recorded; then a 60 GiB gate that also works on macOS (`vm_stat`) |
| 17 | container reuses the name `deliverable.bin`; format auto-detection is ambiguous | §5.3 and §6: `deliverable.bin` stays raw; the container is `deliverable.tdp4`; `decode` takes an explicit `--format raw --n N` or `--format container` |
| 18 | single-window lattice treated as a hard error | §4.11: defined behaviour (scans skipped, `final_visible=0`, one window); `single_window` fixture and unit test |
| 19 | PySet memory not budgeted | §4.4: 12-byte slots with arena indices and cached hashes (order semantics unchanged); `max_concurrent_anf`; memory run (§10.5 step 7); §9 RSS revised |
| 21 | seed plumbing: manifest written before seeds exist; `--mix-seed` on argv not redacted; two-line seed file breaks old calibration files | §8.2-8.4: Rust draws seeds or reads the seed file and creates `$RUN/SEEDS` (create_new, 0600) before the manifest; bash only reads `SEEDS`; no seeds on argv between Rust and bash; standalone bash writes `SEEDS` itself; every seed flag redacted; legacy one-line files and `-s` rejected with migration messages; calibration never draws a random mix seed |
| 23 | stage-4 block resume and invalidation holes; allow flags can never succeed | §11.1: atomic, MAC-keyed completion marker written only after exit 0 and re-checked on skip; `FORCE_FROM` propagates from stages 2 and 3; allowed failures get `NOT_RECOMMENDED` and no COMPLETE line; `tdp_stage4_resume.bash` |
| 26 | CI transcript compared with a hand transcription | §10.9 item 1: the sandboxed `val_chain.sh` runs against the same mocks and the transcripts are diffed; the literal transcript is a secondary pin |
| 27 | "CI cannot run the oracle" is weak; brittle pin on `/usr/bin/python3` and `sys.version` | §10.6: `stage4-oracle` CI job with uv CPython 3.9.6, gated by the set-trace self-test, K random triples; §10.1: pinned uv interpreter as the oracle runtime |
| 28 | u128-modulo hash too slow; fingerprint index too large | §4.5: exact Mersenne folding and a microbenchmark; §4.7: the value index is keyed on the oracle's own blake2b-128 (exact, and about 250 MB smaller) |
| 29 | 63-bit seeds are not a fix (C has at most 64 bits); hashes contradict hygiene | §2.3: stated as a stopgap; routes (a) 256-bit ChaCha and (b) CSPRNG C via `source.path` in §13.1; hashes removed per issue 12 |
| 31 | nine monomorphized widths; config allows n up to 4095 | §4: supported range n ≤ 256 (W ∈ {1,2,4,8,16}), validated in config and CLI; const generics only on the hot set, mul and ANF code |
| 32 | wasteful decoded-window and output layouts | §4.12: flat-arena `Deliverable`; streamed body; no in-memory body or container copy |
| 33 | no I/O or internal exit code; codes not mapped to meanings | §4.1: `Io` (7) and `Internal` (8); one meaning table used by `tdp_packing`, `runner.rs` and `tdp_gen.sh`, with a test that they agree |
| 34 | editing the oracle for `--container` | §6: the oracle is untouched; standalone `container_header.py` plus decoder unit tests |
| 35 | L8/A knobs stay environment reads inside library code; typed params optional | §11.4: typed parameters in P7 (required), covering all output-affecting stage-2 reads including `SLICE_GUARD_LEGACY`; `stage2_typed_params_byte_identical` |
| 36 | unsetting inherited `EMBEDDED_MASKING_*` changes standalone semantics; nonlinear291 interaction | §11.1: decision: inherited values are a hard error, and the tests are updated (`tdp_flags.bash` cases listed); the preset is not applicable to nonlinear291 (explicit preset rejected, nothing exported) |
| 37 | dropping cache sizes slows stage 3 with no output change; no stage-3 budget | D10b keeps the three cache sizes; their neutrality rests on identical stage-3 bytes against the reference runs (historic anchor vL64a/vL128a, n=64 reproductions; §12), not on `cache_neutrality`, which was never run; caps stay unset (output-affecting); §9 adds the stage-3 wall-time expectation |

**End-to-end managed run, round 3, n=64 L8 (2026-10-09, binaries built from `09552766`; outputs `port_runs/e2e64_r3/`).** `tdp_gen` with the same minimal config as round 2: `source.wires = 64`, the run and build directories and `database.regular_dir` (`frozen_m1_m11`). Every other setting is the managed default. The dry run and manifest show preset L8, schedule `validated`, packing `default` (G = 512, split budget 65,536, cpython39 caps) and `pinned_recipe=val_chain-L8-20261007`. Seeds are fresh CSPRNG draws, kept in the private `run/SEEDS`. The zero-slice seed is in the private `checks/SEEDS.checks`. The runner's bash resolution picked `/opt/homebrew/bin/bash` 5.3, and the run started first time. The driver exited 0 with `status=ok`.

- Stages 1+2: 782,261 gates (sandwich 2,832), with 110,119 chained brackets. Forward and reverse verification PASS (256 samples each). The span check is 0 at all 12,224 cuts.
- Stage 3: `--moves 4067757` (= int(2·2·|G|·1.3)) with `FROZEN_FILTER=1` (auto). The chain stopped on the move budget. Output: 1,601,663 gates.
- Stage 4:
  - packing: 2,584 windows; 11,463 distinct values exposed at window boundaries;
  - repair: cancelled 57,679 pairs, leaving 1,486,305 gates. The fix loop went from 65 to 64 visible (1 gate moved). The fallback added 0 splits, so there are 2,583 boundaries;
  - result: 0 windows over budget; 2,584 ANF windows with 1,664,346 monomials (window max 61,854). Every check in the report is `ok`, including the zero slice with 4,096 samples.
- Independent checks (`checks/`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` reports 1,000 born values and **64 visible**, all at t* = 1. That is the exact floor, with 0 non-output values visible. Its exposure, all-cuts and witness TSVs are identical to the run's.
  - `verify_zero_slice C rep512.mpmct1 64 512` PASS. The same check on `packed_G512.mpmct1` also PASS.
  - The oracle's `link_census.py` on the Rust `rep512` gives the run's `link_rep512.txt` byte for byte.
  - The oracle's `anf_dump_eval.py` on the Rust `rep512` gives the Rust `deliverable.bin` byte for byte. `deliverable.tdp4` is a 48-byte header followed by that body.

| n=64 L8 metric | e2e64_r3 | e2e64_r2 | STAGE2_LINKABILITY §5 range |
|---|---|---|---|
| exposed | 11,679 | 11,761 | 11.7k |
| linkable (L2) | 5,361 | 5,305 | 5.27–5.37k |
| sightings (M2) | 59,076 | 58,360 | 58.5–58.9k (r3 is 0.3 % above) |
| linkable froth | 1,089 | 1,114 | 1,110–1,126 (r3 is 1.9 % below) |
| anchors (100+) | 188 | 194 | 185–200 |
| `deliverable.bin` | 32,789,118 B (32.8 MB) | 28.0 MB | 30.2–32.8 MB |
| xz -9 | 3.13 MB | 2.91 MB | 3.05–3.10 MB (r3 is 0.8 % above) |
| eval, ANF form (Python, 1,024 lanes) | 0.88 ms/input | 1.21 | 0.79–1.00 |

Every metric is within 2 % of the §5 range. The deliverable size again varies more across seeds than the other metrics: 28.0 MB (r2), 28.9 MB (r1) and 32.8 MB (r3).

| e2e64_r3 wall time (load average 5–9) | wall |
|---|---|
| build (incremental, including a wait on the shared build lock) | 155 s |
| stages 1+2 (`gen_sandwich_gadget`) | ≈ 1 s (the span report then ran in the background, 57 s) |
| stage 3 (`circuit_mixer`; chain 1,548.3 s, final float 0.8 s) | 1,550 s |
| stage 4 (`tdp_packing run`: pack 13.4 / repair 3.4 / reports 1.5 / emit+checks 6.2 s) | 24 s |
| driver total | 1,730 s (CPU 1,018 s) |
| independent checks (`checks.sh`; oracle serializer 40 s) | 82 s |

**End-to-end managed run, round 3, n=128 L8 (2026-10-09, binaries built from `09552766`; outputs `port_runs/e2e128_r3/`, config, driver log and checks in `port_runs/e2e128_r3_cfg/`).** `tdp_gen` with the same minimal config as round 2: `source.wires = 128`, the run and build directories and `database.regular_dir` (`frozen_m1_m11`). Every other setting is the managed default. The dry run and manifest show preset L8 (the six `EMBEDDED_MASKING_*` values of STAGE2_LINKABILITY §6), schedule `validated` (profile 2,2,2,2,2, moves int(2·2·|G|·1.3), `--db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 --no-local-verify`), packing `default` (G = 512, 512/24576/3000, split budget 65,536, cpython39 caps), `pinned_recipe=val_chain-L8-20261007`, store fingerprint `fd1983677e4d5b74267d3802c9c34642`, `seed_source=random`. Seeds are fresh CSPRNG draws in the private `run/SEEDS` (never printed); the zero-slice check seed is in `checks/check_seeds.txt`. The run started first time; the driver exited 0 with `status=ok`. The lid was closed but the machine was on AC and held awake (`caffeinate -ims -w <driver pid>`), so unlike round 2 the wall times are real.

- Stages 1+2: 2,224,109 gates (sandwich 7,920), with 315,321 chained brackets. Forward and reverse verification PASS (256 samples each). The stage-2 span check is 0 at all 8,689 sampled cuts.
- Stage 3: `--moves 11565366` (= int(2·2·|G|·1.3)) with `FROZEN_FILTER=1` (auto; the store has no `filters.bin`, so unfiltered). `stop_reason=moves budget spent`; output 4,576,658 gates (4,035,543 g57 fossils), final float verified.
- Stage 4:
  - packing: 7,070 windows; 34,133 distinct values exposed at window boundaries;
  - repair: cancelled 166,940 pairs (6 passes), leaving 4,242,778 gates; the fix loop went 129 → 129 → 128 (2 gates moved); the fallback added 2 + 0 splits, giving 7,071 boundaries;
  - result: 0 windows over budget; 7,072 ANF windows with 4,213,660 monomials (window max 63,925); every certificate and check in the report is `ok` (reorder and repair certificates, pack verify, equivalence, windows = window ANF, decode, eval, zero slice with 4,096 samples).
- Independent checks (`checks.sh`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` reports 3,000 born values and **128 visible** (4.3 %), all at t* = 1: the exact floor, with 0 non-output values visible. Its exposure, all-cuts and witness TSVs are identical to the run's.
  - `verify_zero_slice C rep512.mpmct1 128 512` PASS (random seed, recorded), and the same check PASS on `packed_G512.mpmct1`.
  - The oracle's `link_census.py` (sandboxed HEAD scripts, path rewrites only) on the Rust `rep512` gives the run's `link_rep512.txt` byte for byte.
  - The oracle's `anf_dump_eval.py` (sandboxed) on the Rust `rep512` gives the Rust `deliverable.bin` byte for byte. `deliverable.tdp4` (106,299,269 B) is a 48-byte header followed by that body.

| n=128 L8 metric | e2e128_r3 | e2e128_r2 | e2e128_r1 | STAGE2_LINKABILITY §5 range |
|---|---|---|---|---|
| exposed | 34,692 | 34,947 | 34,603 | 34.5–35.0k |
| linkable (L2) | 15,897 | 16,201 | 15,992 | 16.06–16.11k (r3 is 1.0 % below) |
| sightings (M2) | 186,983 | 187,800 | 188,350 | 187–188k (r3 is 0.01 % below) |
| linkable froth | 5,668 | 5,575 | 5,747 | 5,544–5,765 |
| anchors (100+) | 711 | 719 | 719 | 701–718 |
| `deliverable.bin` | 106,299,221 B (106.3 MB) | 107.0 MB | 108.7 MB | 102.8–106.8 MB |
| xz -9 | 9.62 MB (9,620,688 B) | 9.62 MB | 9.69 MB | 9.50–9.62 MB |
| eval, ANF form (Python, 1,024 lanes) | 1.96 ms/input | 1.94 | not measured | 1.83–1.99 |

Every metric is inside the §5 range except L2 (1.0 % below) and M2 (0.01 % below), both well within the 1–2 % replicate spread §5 reports. This is the first managed n=128 run with the deliverable inside the §5 size range.

| e2e128_r3 wall time (load average 1–2 at the end) | wall |
|---|---|
| build (incremental, including a wait on the shared build lock) | 155 s |
| stages 1+2 (`gen_sandwich_gadget`) | ≈ 2 s (the span report then ran in the background, 204 s) |
| stage 3 (`circuit_mixer`; chain 4,416.9 s, final float 4.6 s) | 4,423 s |
| stage 4 (`tdp_packing run`: pack 40.3 / repair 23.7 / reports 4.2 / emit+checks 21.2 s) | 90 s |
| driver total | 4,670 s (CPU 3,016 s; peak RSS 7.1 GB, the mixer) |
| independent checks (`checks.sh`; oracle link census 17 s, oracle serializer 119 s) | ≈ 150 s |

**End-to-end managed run, round 4, n=64 L8 (2026-10-10, binaries built from `a525f894` = review fixes round 3, clean tree; outputs `port_runs/e2e64_r4/`).** `tdp_gen` with the same minimal config as rounds 2 and 3: `source.wires = 64`, the run and build directories and `database.regular_dir` (`frozen_m1_m11`). Every other setting is the managed default. The dry run (`dryrun.out`) and manifest show preset L8, schedule `validated` (profile 2,2,2,2,2, moves int(2·2·|G|·1.3)), packing `default` (G = 512, 512/24576/3000, split budget 65,536, cpython39 caps), `pinned_recipe=val_chain-L8-20261007`, store fingerprint `fd1983677e4d5b74267d3802c9c34642`, `seed_source=random`. Before the run, `tests/shell/tdp_defaults.bash` with `LOCAL_MIXING_BIN` set passed (`tdp_defaults.out`): the Rust driver's gen argv, mixer argv, recipe environment, allow-list and stage-4 call are identical to the `val_chain.sh` transcripts for d64L8, d64A and d128. The only difference between the worktree `val_chain.sh` and the copy in the main checkout's `runs/ncdb_ab_20261001/waist/` is the retired `--p-comp 1.0 --p-any 0.1` (STAGE2_LINKABILITY §6). The stage-4 scripts (`window_pack3.py`, `repair.sh`, `order_repair.py`, `merge_bounds2.py`, `anf_split2.py`, `anf_dump_eval.py`, `link_census.py`, `verify_equiv.py`) are identical to that copy. Seeds are fresh CSPRNG draws in the private `run/SEEDS` (mode 0600). The zero-slice seed is in the private `checks/SEEDS.checks`. None of the four values occurs in any log, recipe, manifest or report. The run started first time under `caffeinate -ims`. The driver exited 0 with `status=ok`.

- Stages 1+2: 777,752 gates (sandwich 2,832), with 109,347 chained brackets. Forward and reverse verification PASS (256 samples each). The span check is 0 at all 12,154 cuts.
- Stage 3: `--moves 4044310` with `FROZEN_FILTER=1` (auto). The chain stopped on the move budget after 1,490.1 s. Output: 1,595,222 gates.
- Stage 4:
  - packing: 2,605 windows; 11,419 distinct values exposed at window boundaries;
  - repair: cancelled 57,653 pairs (5 passes), leaving 1,479,916 gates. The C-view was already at the floor (64) after cancellation, so the fix loop moved nothing. The fallback added 2 + 0 splits, giving 2,606 boundaries;
  - result: 0 windows over budget; 2,607 ANF windows with 1,535,097 monomials (window max 59,875). Every check in the report is `ok`: reorder and repair certificates, pack verify, equivalence, windows = window ANF, decode, eval, and the zero slice with 4,096 samples.
- Independent checks (`checks.sh`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` reports 1,000 born values and **64 visible** (6.4 %), all at t* = 1. That is the exact floor, with 0 non-output values visible. Its exposure, all-cuts and witness TSVs are identical to the run's.
  - `verify_zero_slice C rep512.mpmct1 64 512` PASS (random seed, recorded). The same check on `packed_G512.mpmct1` also PASS.
  - The oracle's `link_census.py` on the Rust `rep512` gives the run's `link_rep512.txt` byte for byte.
  - The oracle's `anf_dump_eval.py` on the Rust `rep512` gives the Rust `deliverable.bin` byte for byte. `deliverable.tdp4` (29,270,616 B) is a 48-byte header followed by that body.

| n=64 L8 metric | e2e64_r4 | e2e64_r3 | e2e64_r2 | STAGE2_LINKABILITY §5 range |
|---|---|---|---|---|
| exposed | 11,660 | 11,679 | 11,761 | 11.7k |
| linkable (L2) | 5,450 | 5,361 | 5,305 | 5.27–5.37k (r4 is 1.5 % above) |
| sightings (M2) | 59,101 | 59,076 | 58,360 | 58.5–58.9k (r4 is 0.3 % above) |
| linkable froth | 1,137 | 1,089 | 1,114 | 1,110–1,126 (r4 is 1.0 % above) |
| anchors (100+) | 184 | 188 | 194 | 185–200 (r4 is 0.5 % below) |
| `deliverable.bin` | 29,270,568 B (29.3 MB) | 32.8 MB | 28.0 MB | 30.2–32.8 MB (r4 is 3.1 % below) |
| xz -9 | 2.97 MB (2,973,164 B) | 3.13 MB | 2.91 MB | 3.05–3.10 MB (r4 is 2.5 % below) |
| eval, ANF form (Python, 1,024 lanes) | 0.79 ms/input | 0.88 | 1.21 | 0.79–1.00 |

The C-view, the equivalence and zero-slice checks, and the window budget are exact requirements, and all of them hold. The linkability metrics are within 1.5 % of the §5 range, inside the 1–2 % replicate spread that §5 reports. Again, only the deliverable size falls clearly outside the range. Across the managed n=64 runs it has been 28.9 (r1), 28.0 (r2), 32.8 (r3) and 29.3 MB (r4), so the three §5 replicates under-sample the seed variance of the ANF size. This is not a port difference: the oracle serializer reproduces the Rust body byte for byte.

| e2e64_r4 wall time (load average 3–4) | wall |
|---|---|
| build (incremental) | 136 s |
| stages 1+2 (`gen_sandwich_gadget`) | ≈ 1 s (the span report then ran in the background, 57 s) |
| stage 3 (`circuit_mixer`; chain 1,490.1 s, final float 0.8 s) | 1,491 s |
| stage 4 (`tdp_packing run`: pack 16.0 / repair 3.8 / reports 1.9 / emit+checks 10.6 s) | 33 s |
| driver total | 1,661 s (CPU 907 s) |
| independent checks (`checks.sh`; oracle serializer 50 s) | 88 s |

**End-to-end managed run, round 4, n=128 L8 (2026-10-10, binaries built from `a525f894` = review fixes round 3, tree at `6c61b8d3` with docs-only commits since; outputs `port_runs/e2e128_r4/`, config, driver log and checks in `port_runs/e2e128_r4_cfg/`).** `tdp_gen` with the same minimal config as rounds 2 and 3: `source.wires = 128`, the run and build directories and `database.regular_dir` (`frozen_m1_m11`). Every other setting is the managed default. The dry run (`dryrun.out`) and manifest show `source.gates=3000 (validated)`, preset L8 (the six `EMBEDDED_MASKING_*` values of STAGE2_LINKABILITY §6), schedule `validated` (profile 2,2,2,2,2, moves int(2·2·|G|·1.3)), packing `default` (G = 512, 512/24576/3000, split budget 65,536, cpython39 caps), `pinned_recipe=val_chain-L8-20261007`, store fingerprint `fd1983677e4d5b74267d3802c9c34642`, `seed_source=random`. Seeds are fresh CSPRNG draws in the private `run/SEEDS` (mode 0600, never printed); the zero-slice check seed is in `checks/check_seeds.txt`. The run started first time; the driver exited 0 with `status=ok`. The session that launched the run ended before the independent checks ran; they were run afterwards on the untouched outputs (`checks.out`). Per-stage wall times are from file modification times and agree with the driver's `time` total.

- Stages 1+2: 2,230,884 gates (sandwich 7,920), with 316,299 chained brackets. Forward and reverse verification PASS (256 samples each). The stage-2 span check is 0 at all 8,716 sampled cuts.
- Stage 3: `--moves 11600596` (= int(2·2·|G|·1.3)) with `FROZEN_FILTER=1` (auto). `stop_reason=moves budget spent`; output 4,593,662 gates (4,052,223 g57 fossils), final float verified.
- Stage 4:
  - packing: 7,142 windows; 34,221 distinct values exposed at window boundaries;
  - repair: cancelled 168,103 pairs (6 passes), leaving 4,257,456 gates. The fix loop stayed at 130 through all four iterations (3, 1, 1, 1 gates moved; `repair.fix_exhausted=1`), the first managed run in which it did not reach the floor by itself. The fallback then merged 3 boundaries and added 2 + 0 splits, taking the C-view 130 → 128 at 7,140 boundaries. So this run exercises the merge/split fallback end to end, with the same outcome as the oracle's `repair.sh` would give (the repair is bit-exact to it, §P4);
  - result: 0 windows over budget; 7,141 ANF windows with 4,127,841 monomials (window max 60,491); every certificate and check in the report is `ok` (reorder and repair certificates, pack verify, equivalence, windows = window ANF, decode, eval, zero slice with 4,096 samples).
- Independent checks (`checks.sh`):
  - `hmap_affine --exposure --c-step 1 --g-cuts rep512.bounds` reports 3,000 born values and **128 visible** (4.3 %), all at t* = 1: the exact floor, with 0 non-output values visible. Its exposure, all-cuts and witness TSVs are identical to the run's.
  - `verify_zero_slice C rep512.mpmct1 128 512` PASS (random seed, recorded), and the same check PASS on `packed_G512.mpmct1`.
  - The oracle's `link_census.py` (sandboxed scripts) on the Rust `rep512` gives the run's `link_rep512.txt` byte for byte.
  - The oracle's `anf_dump_eval.py` (sandboxed) on the Rust `rep512` gives the Rust `deliverable.bin` byte for byte. `deliverable.tdp4` (103,098,222 B) is a 48-byte header followed by that body.

| n=128 L8 metric | e2e128_r4 | e2e128_r3 | e2e128_r2 | STAGE2_LINKABILITY §5 range |
|---|---|---|---|---|
| exposed | 34,746 | 34,692 | 34,947 | 34.5–35.0k |
| linkable (L2) | 16,102 | 15,897 | 16,201 | 16.06–16.11k |
| sightings (M2) | 188,336 | 186,983 | 187,800 | 187–188k (r4 is 0.2 % above) |
| linkable froth | 5,614 | 5,668 | 5,575 | 5,544–5,765 |
| anchors (100+) | 698 | 711 | 719 | 701–718 (r4 is 0.4 % below) |
| `deliverable.bin` | 103,098,174 B (103.1 MB) | 106.3 MB | 107.0 MB | 102.8–106.8 MB |
| xz -9 | 9.55 MB (9,546,172 B) | 9.62 MB | 9.62 MB | 9.50–9.62 MB |
| eval, ANF form (Python, 1,024 lanes) | 1.98 ms/input | 1.96 | 1.94 | 1.83–1.99 |

Every metric is inside the §5 range except M2 (0.2 % above) and anchors (0.4 % below), both well within the 1–2 % replicate spread §5 reports. This closes acceptance round 4: with binaries built from `a525f894`, the Rust defaults reproduce every oracle file on the four archived runs (vA64a, vL64a, vA128a, vL128a), and fresh managed runs at both scales are at the exact floor with the oracle's census and serializer agreeing byte for byte on the Rust outputs.

| e2e128_r4 wall time | wall |
|---|---|
| build (incremental, including a wait on the shared build lock; cargo 88 s + 54 s) | 143 s |
| stages 1+2 (`gen_sandwich_gadget`) | ≈ 2 s (the span report then ran in the background, 212 s) |
| stage 3 (`circuit_mixer`; chain 4,369.0 s, final float 4.5 s) | 4,375 s |
| stage 4 (`tdp_packing run`: pack 35.3 / repair 27.6 / reports 3.3 / emit+checks 19.9 s) | 87 s |
| driver total | 4,607 s (CPU 3,168 s; peak RSS 7.0 GB, the mixer) |
| independent checks (`checks.sh`; C-view 4 s, oracle link census 16 s, oracle serializer 109 s, xz) | ≈ 240 s |
