# Stage 4 (packing) in Rust: design

Status: design, 2026-10-08, branch `ran/stage4-rust` (based on `492564f0`). Owner: RC.
This document specifies how stage 4 of the TDP pipeline is ported to Rust and how the managed driver is changed so that one managed run, with default settings, produces the recommended deliverable. That deliverable must be byte-for-byte the one the Python reference chain produces.

Sources of truth, in order of precedence:

1. The owner's requirements, restated in §1.
2. The Python/bash oracle in `scripts/windowed_delivery/`. These files are byte-identical to the run-tree copies, except that `val_chain.sh` drops the inert `--p-comp/--p-any`, see §2.1. This document cites the oracle as follows:
   - `WP` = `window_pack3.py`;
   - `AC` = `anf_compose.py`;
   - `W/` = `waist/` (`order_repair.py`, `merge_bounds2.py`, `anf_split2.py`, `verify_equiv.py`, `anf_dump_eval.py` (`DE`), `link_census.py`, `repair.sh`, `val_chain.sh`);
   - `HA` = `security_tests/heatmaps/hmap_affine.rs`.
3. The validation runs `runs/ncdb_ab_20261001/waist/{vL64a,vL64b,vL128a,vL128b}` (CONFIG=L8) and the matching `vA*` runs. They are read-only. Their `seed.txt`, `gen.log` and `s3.log` were read to verify every default in §2. Docs were not trusted where these disagree.

Everything not marked "deviation" or "decision" is meant to be bit-exact.

---

## 1. Requirements

**Goal (RC).** Stage 4 becomes a Rust production stage, integrated into the TDP driver (`scripts/tdp_gen.sh` and `src/tdp`). It must be faithful to the Python oracle, bit-exact wherever the oracle is deterministic, and validated on real artifacts.

**Defaults requirement (RC, 2026-10-08).** The managed driver's defaults reproduce `W/val_chain.sh` with CONFIG=L8 exactly, as it was run for vL64a, vL64b, vL128a and vL128b:

- stages 1+2 arguments and environment;
- stage 3 argv and environment, run serially;
- stage 4 parameters: pack `512 512 24576 3000`; repair as `repair.sh` (4 fix iterations, 4 fallback iterations, ANF budget 65536); serialization as `DE`.

Further requirements:

- Config A is selectable by one option.
- Gen and mix seeds are random per run and recorded. Explicit per-stage seeds are accepted, so that a run can be reproduced.
- The source size keeps the formula default `round(n·log2²n)` for now, but as one obvious option recorded in the manifest. RC has been asked about this.

**Acceptance.**

- Reproduction test, §10.6: the managed driver and `val_chain.sh` (L8), run with the same source size and the same seeds, produce byte-identical output for:
  - the stage-2 circuit;
  - the stage-3 circuit;
  - the repaired circuit and its bounds;
  - the deliverable payload.
- A CI-sized version of the same check, §10.7.
- Gates G1-G7, `cargo fmt --check` and the CI checks in `.github/workflows/ci.yml`.

---

## 2. The reference recipe, verified, and how today's driver differs

### 2.1 What `val_chain.sh TAG N MC S SLICE L8` ran

Verified against the script, `seed.txt` and the gen/s3 logs of all four L8 runs.

| item | value in the reference runs | evidence |
|---|---|---|
| gen argv | `tdp.mpmct1 N MC MC S 1 SLICE SEED SEED SEED`, with no mode argument (the default is `embedded-masking`) and no variant argument (the default is `classic`) | `val_chain.sh:15-16`; gen.log `seed=X gadget_seed=X sandwich_seed=X` |
| sizes | n=64: MC=1000, S=384, SLICE=1280. n=128: MC=3000, S=896, SLICE=2560. S equals `sandwich_default_s(n)` and SLICE equals `10·2n`. MC was passed explicitly; it is not the formula (2304 / 6272) | gen.log line 1 |
| stage-2 environment | exactly `EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8`. Every other `EMBEDDED_MASKING_*` variable is unset, so gen uses `EmbeddedMaskingParams::production` (K=2, max_open=3, min_open=2, balanced, quad_fire) | `val_chain.sh:13,15`; `embedded_masking.rs:196-216` |
| gen seed | `secrets.randbelow(2**31)` | `val_chain.sh:11` |
| mix seed | `secrets.randbelow(2**40)` | `val_chain.sh:12` |
| stage-3 argv | `--input tdp.mpmct1 --output s3.mpmct1 --tdp --db-mixing --profile 2,2,2,2,2 --moves int(G*2*2*1.3) --p-db 1.0 --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 --no-local-verify --seed MSEED`. G is field 3 of the `tdp.mpmct1` header. `int` truncates. The runs also passed `--p-comp 1.0 --p-any 0.1`; these equal the CLI defaults, are inert at `p_db=1.0`, and were removed on this branch (TDP doc §3.2). The tracked `val_chain.sh` already omits them | `val_chain.sh:17-22`; s3.log `profile ON: n=[2.0, 2.0, 2.0] r=[2.0, 2.0]`, `moves=4051814` |
| stage-3 environment | `FROZEN_DB_DIR=<regular store> FROZEN_FILTER=1`. Nothing else: `CANON_*` caps unset (unbounded), cache sizes at their built-in defaults, `--verify-every` and `--report-every` at their CLI defaults (10 000 and 50 000). The local store has no `filters.bin`, so the filter ran off | s3.log `FROZEN_FILTER=1 but no filters.bin ... running unfiltered`; reports every 50 000 moves |
| stage-3 mode | serial: no `--parallel-*` | `val_chain.sh` |
| stage 4 | `window_pack3.py tdp s3 512 512 24576 3000 TAG-G512 packed_G512.mpmct1`, then `repair.sh TAG 512 N`, then `anf_dump_eval.py rep512.mpmct1 rep512.bounds deliverable.bin TAG` | `val_chain.sh:28-31` |
| repair version | All eight replicates were re-repaired on Oct 8 with the canonical `order_repair.py` (breaker located before the completer moves) and re-serialized; the earlier repairs are kept in `rep512_v1/`. So TDP doc §4.7 "only the vL128a rerun used the canonical version / not re-serialized" is stale. The parity targets are the current `rep512.*` and `deliverable.bin` (hashes in Appendix B) | file times; `repair512.log` |

### 2.2 Differences in today's managed driver that must change

Every row below changes the output bytes, or could change them. The driver must match the reference column.

| # | today (`tdp_gen.sh`, `src/tdp/runner.rs`) | reference | why it matters |
|---|---|---|---|
| D1 | `gadget_seed = SEED+1` (`tdp_gen.sh:419`) | `gadget_seed = SEED` | gadgetization RNG stream (`gen mod.rs: StdRng::seed_from_u64(gadget_seed ^ 0x6AD6_E75E)`) |
| D2 | `shuffling_segments` defaults to 0; `return_home=0` is **rejected** (`tdp_gen.sh:254-261`, `validate.rs:101-108`) | 8, 0 | stage-2 construction |
| D3 | PREMASK, SEED_AVOID_C1, CHAIN and CHAIN_LEN cannot be expressed: they are scrubbed (`runner.rs:338`) and there is no field for them | 1, 1, 2, 8 | stage-2 construction |
| D4 | always exports `EMBEDDED_MASKING_{K,MAX_OPEN,MIN_OPEN,BALANCED,QUAD_FIRE}` | not set | value-neutral today (they equal the preset), but the requirement is "no other `EMBEDDED_MASKING_*` variables" |
| D5 | profile `3,30,30,2,2` (n0 = 3 hard-coded, hold = 27) | `2,2,2,2,2` | stage-3 schedule |
| D6 | moves = `round(n2·r1·G·1.3)` | `int(G·2·2·1.3)` (floor) | the last move, when the fraction is ≥ .5 |
| D7 | mix seed = `SEED+2` (a 63-bit derived value) | independent random value | stage-3 RNG |
| D8 | `--verify-every 2000000` | default 10 000 | **`global_check` draws `4·num_wires` u64 values from the mixer RNG** (`reporting.rs:17-23`), so the cadence changes the trajectory |
| D9 | `--report-every 100000 --state-out` | defaults; no state-out | report cadence runs `anc_prune` and the flag and snapshot checks. Not proven trajectory-neutral, so it is dropped |
| D10 | exports `CANON_RULE_L_BRANCH_CAP=512 CANON_MONOMIAL_CAP=200000 CANON_CACHE_MB=256 XPOLY_CANON_CACHE_MB=1024 LOOKUP_CACHE_MB=2048` (`runner.rs:383-387`, `tdp_gen.sh:446-450`) | all unset | the caps change canonicalization results whenever they bind. The cache sizes should be pure memoization, but the reference did not set them |
| D11 | `FROZEN_FILTER` auto-detection reads `/proc/meminfo`, which is Linux-only and therefore off on macOS | `FROZEN_FILTER=1` | should be output-neutral (a miss filter only). Matched anyway |
| D12 | gate count via `wc -l` minus 1 | header field 3 | equal for consistent files; use the header field |
| D13 | the driver ends at stage 3 | stage 4 | — |
| D14 | the gen mode positional `embedded-masking` is passed | omitted | neutral (it is the default); kept, and documented |

`FROZEN_REGULAR_VALUE_CONVENTION=native` stays: the reference log shows `regular=native`, which is the built-in default. `SANDWICH_VARIANT=classic` stays for the same reason.

### 2.3 Seed width (deviation; decision for RC)

The reference draws the gen seed below 2^31 and the mix seed below 2^40. **The gen seed regenerates C** (`prepare_source(n, m_c, seed)`), and generating C is microseconds. A 31-bit seed therefore makes C recoverable by brute force against any artifact: about 2^31 C generations, each followed by a functional comparison. The validation runs are calibration-grade material for this reason.

The managed default draws **both** seeds uniformly below 2^63 from the OS CSPRNG. This changes only the distribution the seed is drawn from. For any given seed, every computation is the one the reference performs, and explicit seeds in the reference range are accepted, so the reproduction test is unaffected.

This is the only intended difference between the reference recipe and the managed defaults. It is recorded in the manifest as `seed_bits=63`.

Hygiene: hashes of `tdp.mpmct1`, the source C and `s3.mpmct1` are deliberately **not** committed anywhere. With a 31-bit seed space they would be cheap verification oracles for a brute-force search.

---

## 3. Architecture

```
          stage 2 G2 (tdp.mpmct1)  stage 3 G3 (db_mixing.mpmct1)        C (tdp.mpmct1.source_c.g57)
                     \                   /                                     |
                      v                 v                                      |
  pack  (fingerprints MT19937/1024 -> proxy spans -> intervals -> scheduler    |
         with the windowed ANF caps on a CPython-3.9 set emulator)             |
          -> packed_G512.mpmct1 + .bounds                                      |
                      |                                                        v
  repair: cancel -> [C-view scan -> fix]x<=4 -> [C-view -> merge? -> split]x<=4 -> final scan (report)
          (in-process exposure engine; the leaky-cut predicate is memoized per cut)
          -> rep512.mpmct1 + rep512.bounds (+ rep512.{exposure,allcuts,wit}.tsv)
                      |
  emit:   canonical per-window ANF -> body (DE format) -> container v1 -> deliverable.bin
  verify: pack VERIFY, sampled equivalence, per-window ANF == gates, decode(deliverable) == circuit,
          zero-slice deliverable(x) == C(x)
```

One library module tree, `src/stages/packing/`, has no environment reads, following the `src/stages` convention (`preprocessing/types.rs:31`). One production binary, `tdp_packing`, is its adapter. The exposure engine leaves `hmap_affine` and becomes library code shared by both.

---

## 4. Library layout: `src/stages/packing/`

Add `pub mod packing;` to `src/stages/mod.rs`. Its doc comment becomes: "TDP stage ownership: sandwich, preprocessing, database mixing and packing (stage 4)."

Each module attaches its unit tests by the repo convention:

```rust
#[cfg(test)] #[path = "../../../tests/unit/stages/packing/<module>.rs"] mod tests;
```

The `repair/` submodules use one more `../`.

The new code adds **no new crate dependency**:

- MT19937, the CPython set emulator and CRC-32 are in-house (each under about 200 lines);
- the ordered map is `Vec` plus `FxHashMap`;
- `rand 0.9.2` is used only by the exposure engine (unchanged);
- `rayon` and `xxhash-rust` are already dependencies.

```
src/stages/packing/
  mod.rs          run_stage4(), re-exports
  types.rs        PackParams, RepairParams, EmitParams, CapSemantics, RepairMode, reports, errors
  oracle_io.rs    oracle-compatible readers/writers (mpmct1 width rules, k=0 trailing space, bounds files)
  pyrand.rs       CPython random.Random: MT19937 init_by_array + getrandbits word order
  pyset.rs        CPython 3.9.6 setobject emulator (table order, probing, resize, merge fast path)
  mono.rs         monomial keys [u64; W] (W in {1,2,4,8,16,...,256}), CPython int hash, ordering, width dispatch
  anf.rs          window state (packer compose) + window_anf (split / emission), cap semantics
  fingerprint.rs  1024-sample bit-sliced evaluation, complement normalisation, proxy_spans
  intervals.rs    interval_set + next-next-writer rescue, winnable values, iid numbering
  scheduler.rs    the window_pack3 main loop (epochs, filler/anyready, burst, contagion, close_window)
  gf2.rs          GF(2) basis with coefficient tracking (shared by cview)
  cview.rs        the C-view exposure engine (from HA), streaming over cuts, memoizable per cut
  repair/mod.rs   repair loop (repair.sh), RepairMode
  repair/seq.rs   gate/marker sequence: to_seq, from_seq, commute, slide
  repair/cancel.rs cancel_pass, run_cancel
  repair/fix.rs   run_fix (canonical breaker handling)
  repair/merge.rs merge_bounds2
  repair/split.rs anf_split2 (candidate rule, recursion, fit predicate)
  serialize.rs    DE body encoder, container v1, CRC-32, decoder, evaluator
  verify.rs       verify_equiv (seed 77), all-window ANF check, decode round-trip, zero-slice vs C
  census.rs       link_census (L2/M2, proxy/froth) and the packer RESULT on the delivered lattice
```

### 4.1 `types.rs`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapSemantics { CPython39, Canonical }        // §7.2; default CPython39

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepairMode { Oracle }                        // §4.12; later fixes add variants behind flags

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
impl PackParams { pub const PRODUCTION: PackParams = /* the values above, CPython39 */; pub fn mul_cap(&self) -> usize { 8 * self.b_mono } }

#[derive(Clone, Debug)]
pub struct RepairParams {
    pub split_budget: usize,         // 65536
    pub fix_iterations: usize,       // 4
    pub fallback_iterations: usize,  // 4
    pub cview: CviewParams,          // batches 96, seed 12345
    pub cap_semantics: CapSemantics, // CPython39 (window_anf in the split predicate)
    pub mode: RepairMode,            // Oracle
}

#[derive(Clone, Debug)]
pub struct Stage4Params {
    pub pack: PackParams, pub repair: RepairParams,
    pub verify_seed: u64,            // 77   (verify_equiv)
    pub zero_slice_samples: usize,   // 4096 (new check; no oracle counterpart)
    pub allow_floor_miss: bool,      // false
    pub allow_over_budget: bool,     // false
}

pub struct PackReport { /* every number in the WP log lines, plus compose_aborts, ambiguous_muls */ }
pub struct RepairReport { /* cancel passes, fix trajectory, moved per pass, fallback trajectory, merges, splits, forced splits, final scan */ }
pub struct EmitReport { /* windows, monomials, max per window, body bytes, container bytes, xxh3 */ }

#[derive(Debug)] pub enum Stage4Error { Input(String), Verify(String), Floor { nonoutput_visible: Vec<usize> }, OverBudget { windows: Vec<(usize, usize)> }, Internal(String) }
```

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

`read_oracle_mpmct` is strict. It accepts only what makes `XGate` equality coincide with the Python tuple:

- controls already **ascending** in the file;
- no duplicate control;
- no control on the target;
- comp and polarity in {0, 1};
- header m equal to the number of gate lines;
- extra tokens on a line rejected.

All Rust-written artifacts satisfy this (vA64a was checked). Any violation is an `Input` error, exit 3, and never a silent canonicalization.

The width rule of each oracle step is applied explicitly where the step uses it:

- pack: `n = max(computed_n(A), computed_n(B))`, and the output header is `computed_n(B)`;
- repair: the header n is carried through;
- emission: m is the header m.

`commute(a, b)` is defined in `repair/seq.rs` as `!a.reads(b.target) && !b.reads(a.target)`. It must **not** reuse `XGate::collides`, which exempts polarity-separated conjunctions.

### 4.3 `pyrand.rs`: CPython `random.Random`

```rust
pub struct PyRandom { mt: [u32; 624], idx: usize }
impl PyRandom {
    pub fn new(seed: u64) -> Self;                       // key = 32-bit LE words of |seed| (one word if < 2^32), init_by_array
    pub fn genrand_u32(&mut self) -> u32;
    pub fn getrandbits_words(&mut self, k: usize) -> Vec<u32>; // ceil(k/32) words, least-significant first; partial top word = top bits of the output (u32 >> (32 - r))
    pub fn getrandbits_lanes<const L: usize>(&mut self) -> [u64; L]; // k = 64L; lane i = w[2i] | w[2i+1] << 32
    pub fn getrandbit(&mut self) -> bool;                // getrandbits(1) = top bit of one output (DE single-input)
}
```

Users and their seeds:

| user | seed | draw |
|---|---|---|
| packer and link census | 20261004 | 16 lanes × n |
| `verify_equiv` | 77 | 16 lanes × n |
| AC spot checks | 20261004 | 4 lanes |
| DE evaluation | 1 | 16 lanes |

Golden vectors are listed in Appendix A.

### 4.4 `pyset.rs`: CPython 3.9.6 set emulator

This module is the reason the packer and the split predicate can be bit-exact. `mul` aborts when a **prefix** of the smaller operand's iteration order pushes `|res|` above the cap (WP:135-150, AC:49-64), and that order is CPython's table order. All 8 measured compose aborts, and the AC case vA64a span `[492039,492540)`, have a full product within the cap. They fire only because of the order. Under sorted order 1 of 2 (vA64a) and 1 of 6 (vL128b) disappear.

```rust
pub trait PyIntKey: Copy + Eq { fn py_hash(&self) -> u64; }  // nonnegative int: value mod (2^61-1)
pub struct PySet<K: PyIntKey> { table: Vec<Slot<K>>, mask: usize, fill: usize, used: usize }
enum Slot<K> { Null, Dummy, Active { key: K, hash: u64 } }
impl<K: PyIntKey> PySet<K> {
    pub fn new() -> Self;                         // size 8 (PySet_MINSIZE)
    pub fn singleton(k: K) -> Self;               // BUILD_SET {x}: new() then add
    pub fn len(&self) -> usize;
    pub fn contains(&self, k: &K) -> bool;
    pub fn add(&mut self, k: K);                  // first NULL on the probe path; dummies NOT reused (3.9)
    pub fn discard(&mut self, k: &K) -> bool;     // slot -> Dummy, used -= 1, never shrinks
    pub fn toggle(&mut self, k: K);               // `if v in res: discard else add`
    pub fn copy_of(other: &Self) -> Self;         // set(other) == set_merge(empty, other) incl. verbatim fast path
    pub fn xor(a: &Self, b: &Self) -> Self;       // a ^ b == copy_of(b) then toggle a's keys in a's table order
    pub fn xor_assign(&mut self, b: &Self);       // a ^= b: toggle b's keys in b's table order
    pub fn iter(&self) -> impl Iterator<Item = &K>; // table order, skipping Null/Dummy
}
```

Rules to transcribe (packer spec §9; validated empirically):

- **Probe:** `i = h & mask`, `perturb = h`. Examine slot i and, when `i + 9 <= mask`, slots `i+1..i+9` (LINEAR_PROBES = 9). Then `perturb >>= 5` and `i = (5i + 1 + perturb) & mask`.
- **Growth:** after an insert, if `fill*5 >= mask*3`, resize to `used > 50000 ? used*2 : used*4`.
- **`resize(minused)`:** start from 8 and double while `<= minused`. Re-insert the active entries in old table order with `insert_clean`, the first NULL on the probe path, with no equality test.
- **Merge into an empty set:** if `(fill + other.used)*5 >= mask*3`, first resize to `(used + other.used)*2`. Then copy verbatim when `fill == 0`, `mask == other.mask` and other has no dummies; otherwise `insert_clean` in other's table order.

Monomial keys implement `PyIntKey` in `mono.rs`. The hash is folded from the most significant limb down with 128-bit arithmetic: `h = ((h as u128) << 64 | limb as u128) % (2^61 - 1)`.

The emulator is THE set type in both the packer's compose and `window_anf`, whatever the `CapSemantics`. The table order only matters for the prefix abort, but one code path is simpler to test.

Validation obligations:

- a Rust replay of committed traces recorded on `/usr/bin/python3` 3.9.6 (§10.2);
- a replay of the window ANF of `packed_G512` plus its bounds on vA64a and vL128b, which must reproduce every abort (manual parity, §10.5).

The traces record `sys.version`. CI never trusts the CI runner's Python for set order.

### 4.5 `mono.rs`

```rust
pub trait MonoKey: PyIntKey + Ord + Hash + Default {
    const WORDS: usize;
    fn bit(w: usize) -> Self; fn or(&self, o: &Self) -> Self; fn is_zero(&self) -> bool;
    fn bits(&self) -> impl Iterator<Item = usize>;
}
impl<const W: usize> MonoKey for [u64; W] { /* Ord compares from the most significant word = Python int order */ }
/// Calls `f::<[u64; W]>()` for the smallest W in {1,2,4,8,16,32,64,128,256} with 64·W >= nw.
pub fn with_mono_width<R>(nw: usize, f: impl MonoWidthFn<R>) -> R;
```

At n=64 the state is 256 wires, so W=4. At n=128 it is 512 wires, so W=8. The upper limit of 16 384 wires matches the u16 wire ids of the format.

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
pub fn window_anf<M: MonoKey>(gates: &[XGate], budget: usize, sem: CapSemantics) -> Result<WindowAnf<M>, AnfOverflow>;
pub struct MulStats { pub calls: u64, pub ambiguous: u64, pub aborts: u64, pub aborts_full_product_fits: u64 }
```

`mul` semantics:

- Swap so the smaller operand is outer, and only when `len(a) > len(b)`. On a tie the accumulated product stays outer.
- Toggle per pair: XOR, not dedup.
- Check `len(res) > cap` after each outer row.
- In the ambiguous zone `(|small|-1)·|large| > cap`, `Canonical` mode completes the product and aborts only if the final size exceeds the cap. In `CPython39` mode the prefix decides.
- `MulStats` counts both outcomes, so a run can report how many decisions depended on the emulator.

### 4.7 `fingerprint.rs`

```rust
pub type Fp = [u64; 16];                                  // 1024 samples
pub fn inputs(n: usize, seed: u64) -> Vec<Fp>;            // PyRandom(seed).getrandbits(1024) per wire, wire order
pub fn normalize(v: &Fp) -> Option<Fp>;                   // complement if sample 0 is set; None if constant
pub struct Spans { pub spans: Vec<(u32 /*b*/, u32 /*e*/, u32 /*a*/)>, pub forced: FxHashSet<u32>, pub agree: usize,
                   pub a_wire: FxHashMap<u32, u16>, pub carried: usize }
pub fn proxy_spans(a: &[XGate], b: &[XGate], n: usize, na: usize, nb: usize, inputs: &[Fp]) -> Spans; // WP:63-90
```

The evaluation adds `XGate::apply_lanes16(&self, &mut [[u64; 16]])` to `xgate.rs`, bit-identical per word to `apply_lanes`.

The value index is keyed on the **full normalized `Fp`**, with the first birth index in A as the value id. The oracle keys on blake2b-128 of the same value. The two are equivalent except under a 2^-128 collision (deviation D-a, §7.3).

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

- **Epoch readiness:** per-wire `Vec<Vec<u32>>` buckets indexed by the write and read counts.
- **Heaps:** `filler` and `anyready` are `BinaryHeap<Reverse<u32>>` with lazy deletion.
- **`committed`:** `OrderedMap<u32, u32>`, a `Vec<(key, val, live)>` plus an `FxHashMap` from key to slot. Deletion preserves order. Iteration skips dead entries, and the structure compacts when more than half the slots are dead.
- **Sorts:** `by_end` and the contagion snapshot use stable sorts.

Quirks are reproduced, not fixed:

- On a compose failure the window closes **after** g is appended and **before** g's interval bookkeeping.
- The contagion snapshot goes stale within a pass.
- A burst continues through a mid-burst close.
- Caps are tested before a gate is added, so windows overshoot.
- The pre-emptive close is unconditional on `win_load`, so it can produce a duplicate or zero boundary.
- The final filter keeps `0 < b < m` without dedup.
- The output header is `computed_n(B)`.

The unreachable paths (`burst stuck`, `scheduler stuck`, `resumed_later`) become `debug_assert!`s. The lemma "the minimum non-scheduled entry of anyready is the lowest unscheduled gate" is a `debug_assert!` as well.

The text log lines reproduce WP's format (header, winnable, intervals, scheduled, the `death causes` dict repr in first-increment order, VERIFY, RESULT, class lines, window stats). They are compared byte for byte on fixtures, where `:.1f` rounding ties are the only tolerated difference (§7.3).

### 4.10 `gf2.rs` and `cview.rs`: the exposure engine, moved out of `hmap_affine`

The API follows the exposure spec §9:

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
pub struct CutColumns { reps: Vec<Vec<u64>>, rep_wires: Vec<u16>, basis: Vec<(usize, Vec<u64>, Vec<u64>)>, nulls: Vec<Vec<u64>> }
impl CutColumns { pub fn build(state: &[Vec<u64>] /*[batch][wire]*/, nw: usize) -> Self;
                  pub fn predict(&self, row: &[u64]) -> Option<CutFit>; }                 // CutFit { weight: u32, support: Vec<u16>, approx: bool }
pub struct CutResult { pub cut: usize, pub fits: Vec<(u32 /*row*/, CutFit)> }            // rows ascending
pub struct CutOracle<'a> { /* P, nw_g, samples, rows, checkpoints every 16384 gates, memo FxHashMap<cut, Arc<CutResult>> */ }
impl<'a> CutOracle<'a> {
    pub fn new(p: &'a [XGate], nw_g: usize, rows: &'a CRows, s: &'a HonestSamples) -> Self;
    pub fn results(&mut self, cuts: &[usize]) -> Vec<Arc<CutResult>>;                    // computes missing cuts in parallel (rayon), returns in cut order
}
pub struct ExposureOptions { pub collect_allcuts: bool, pub collect_witness: bool, pub skip_settled: bool }
pub struct Witness { pub cut: usize, pub support: Vec<u16> }
pub struct ExposureReport { pub rows: Vec<BornRow>, pub cuts: Vec<usize>, pub tstar: Vec<i64>, pub approx: usize,
                            pub witness: Vec<Option<Witness>>, pub allcuts: Vec<(u32, usize, u32)> }
pub fn assemble(rows: &CRows, cuts: &[usize], per_cut: &[Arc<CutResult>], o: &ExposureOptions) -> ExposureReport; // sequential, cut order
pub fn exposure_scan(c_rows: &CRows, g: &[XGate], nw_g: usize, s: &HonestSamples, cuts: &[usize], o: &ExposureOptions) -> ExposureReport;
impl ExposureReport {
    pub fn visible(&self) -> usize;
    pub fn summary_lines(&self) -> Vec<String>;          // exact HA L2 + histogram + "wrote" lines
    pub fn write_exposure_tsv(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn write_allcuts_tsv(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn write_witness_tsv(&self, w: &mut impl Write) -> io::Result<()>;
    pub fn nonoutput_visible(&self, outs: &FxHashSet<usize>) -> Vec<usize>;  // identity floor test
}
```

The key property: a `(row, cut)` fit depends only on `(C, P, cut, n, seed, batches)`, never on which other cuts are scanned. So `CutResult` is memoized per cut. All semantics that span several cuts are replayed **sequentially in cut order** in `assemble`:

- the strict-< best weight;
- the earliest-cut witness;
- the `skip_settled` rule and its `approx` accounting.

Within a cut, the following match HA exactly:

- the representative is the lowest wire of each complement class;
- reps are inserted in wire order, with the constant inserted last;
- forward reduction runs in basis insertion order;
- the Gray-code walk keeps the first strict minimum;
- nullity above 14 counts as approximate.

The engine **streams** P with one `[batch][wire]` state of 96 × nw u64 words. At n=128 that is 393 KB. It stores checkpoints every 16 384 gates so that candidate cuts can be re-simulated on demand. This replaces HA's materialized `gs[batch][cut][wire]`, which takes about 2.8 GB at n=128.

**`hmap_affine` after the move.** Exposure mode becomes `HonestSamples::draw`, then `born_rows`, then `exposure_scan` with `skip_settled = allcuts_path.is_none()`, then printing and writing in the same order. Line 1, every assert (including `tb*64 > n_reg+64`) and every panic message stay. Heatmap and SPAN modes keep their code. They use the shared sample function only where the RNG order is provably identical; SPAN moves later and separately.

The engine is not feature-gated, because stage 4 must build with `cargo build --release`. `hmap_affine` stays under `security-tools`.

### 4.11 `repair/`: the oracle repair in-process

```rust
// seq.rs
pub enum Item { Gate(XGate), Marker }
pub fn to_seq(gates: &[XGate], bounds: &[usize]) -> Vec<Item>;            // dedup, keep 0 < b < m
pub fn from_seq(seq: &[Item]) -> (Vec<XGate>, Vec<usize>);                  // merge adjacent markers, drop leading/trailing
pub fn commute(a: &XGate, b: &XGate) -> bool;
pub fn slide(seq: &mut Vec<Item>, pos: usize, forward: bool) -> usize;      // incl. the dest==pos no-move quirk
// cancel.rs
pub fn cancel_pass(gates: &[XGate]) -> (Vec<bool> /*dead*/, usize /*pairs*/); // pass-start indexes: dead gates still block
pub fn run_cancel(seq: &mut Vec<Item>) -> Vec<usize>;                        // pairs per pass, to the fixed point
// fix.rs
pub struct FixRow { pub c_gate: usize, pub cut: usize, pub support: Vec<u16> }
pub fn run_fix(seq: &mut Vec<Item>, rows: &[FixRow], outs: &FxHashSet<usize>) -> FixReport; // rows in ascending C gate
// merge.rs
pub fn merge_bounds(bounds: &[usize], allcuts: &[(u32, usize, u32)], rows: &[BornRow], outs: &FxHashSet<usize>) -> (Vec<usize>, usize);
// split.rs
pub fn split_candidates(lo: usize, hi: usize) -> Vec<usize>;                // {lo + (hi-lo)*i/33 : i=1..32} \ {lo,hi}, sorted
pub fn split_pass<M: MonoKey>(p: &[XGate], bounds: &[usize], budget: usize, sem: CapSemantics,
                              leaky: &mut dyn FnMut(&[usize]) -> FxHashSet<usize>, log: &mut dyn Write) -> SplitReport;
// mod.rs
pub struct Repaired { pub n_header: usize, pub gates: Vec<XGate>, pub bounds: Vec<usize>, pub last_scan: ExposureReport,
                      pub final_scan: ExposureReport, pub report: RepairReport }
pub fn repair(packed: &[XGate], packed_n: usize, bounds0: &[usize], c: &[XGate], n: usize,
              p: &RepairParams, log: &mut dyn Write) -> Result<Repaired, Stage4Error>;
```

Control flow of `repair` in `RepairMode::Oracle`, which is exactly `repair.sh`:

1. `seq = to_seq(packed, bounds0)`, then `run_cancel`, then `from_seq`.
2. `verify_equiv(packed, rep)` against the immutable in-memory packed circuit. There is no file race.
3. Fix loop, for `it` in `1..=fix_iterations`:
   - scan the current `(rep, bounds)`;
   - log `fix iter {it}: C-view visible {v} (floor {n})`;
   - if `v <= n`, break;
   - otherwise call `run_fix` with the scan's witness rows and the outputs, then `from_seq`.
   After the last fix there is no rescan inside this loop.
4. `verify_equiv` again.
5. Fallback loop, for `it` in `1..=fallback_iterations`:
   - `v` = scan of the current lattice;
   - if `v > n`, `merge_bounds` using this scan's allcuts;
   - `split_pass` with budget 65536. Its `leaky` closure is `CutOracle` on the unchanged rep circuit, giving cuts with a fit row outside `outs`.
   - Break iff `v <= n` and 0 splits were added.
6. **Final scan, new and report-only.** Scan the final lattice and compute the identity test `nonoutput_visible(outs)`. It does not change any byte of the output. It decides the exit status (§5.4) and fills in the "final lattice never rescanned" gap.

Memoization makes this cheap. The rep circuit is constant across the whole fallback loop, because merge and split only touch bounds. So every scan and every split-candidate predicate in the fallback loop shares one `CutOracle`. Fix iterations change the circuit and get a fresh `CutOracle` each.

The merge reads the outputs from C via `output_gates(c, n)`. The oracle derives them from `exposure.tsv` (`last[w] = max gate`). The two are equal when c-step is 1 and every wire is written; the reader asserts this equality on the rows. This drops the `SPLIT_EXP` dependency.

Other output-neutral hardening:

- Legality is asserted per move: every crossed gate commutes, and each cancelled pair is identical with no blocker in between. These are debug asserts, plus a cheap release check of the gate multiset (`rep = packed - pairs`).
- Every error the oracle swallows is a hard error here: an empty lattice (HA panics on "empty plate"), a failed scan, a failed split.

### 4.12 `serialize.rs`: body, container, decoder

```rust
pub const MAGIC: [u8; 8] = *b"LMTDPANF"; pub const FORMAT_VERSION: u16 = 1; pub const HEADER_BYTES: u16 = 48;
pub const IO_LAYOUT_CLASSIC_ZERO_SLICE: u32 = 1;   // inputs 0..n, answer n..2n, all other wires 0 (verify_zero_slice.rs)
pub struct ContainerHeader { pub source_wires: u32, pub state_wires: u32, pub io_layout: u32, pub window_count: u32, pub body_bytes: u64, pub body_crc32: u32 }
pub fn encode_body(gates: &[XGate], bounds: &[usize], m_header: usize, nw: usize) -> Result<(Vec<u8>, EmitReport), Stage4Error>;
pub fn encode_container(h: &ContainerHeader, body: &[u8]) -> Vec<u8>;
pub fn crc32_ieee(bytes: &[u8]) -> u32;                     // == zlib.crc32
pub struct DecodedWindow { pub support: Vec<u16>, pub changed: Vec<(u16, Vec<Vec<u8>>)> }
pub struct Deliverable { pub header: Option<ContainerHeader>, pub windows: Vec<DecodedWindow> }
pub fn decode(bytes: &[u8]) -> Result<Deliverable, Stage4Error>;      // container or raw body; validates every invariant
impl Deliverable { pub fn eval_lanes16(&self, state: &mut [[u64; 16]]); pub fn eval_u64(&self, state: &mut [u64]); }
```

`encode_body` follows `DE:14-50` byte for byte:

- `cuts = [0] + sorted(0 < b < m_header) + [m_header]`, without dedup;
- each window comes from `window_anf(…, 1<<30)`, so the caps can never fire, and the result is canonical;
- the support is ascending;
- the record header is `<HH`;
- the support ids are `<H` each;
- each changed wire, in ascending order, is `<HI` followed by its local masks: `ceil(f/8)` bytes little-endian per mask, monomials in ascending global value;
- each blob is preceded by `<I len`.

Overflows (a wire above 65535, f or c above 65535, or a count or length beyond u32) are errors, never a wrap.

Windows are encoded in parallel and concatenated in window order.

The container layout is in §6.

### 4.13 `verify.rs`

```rust
pub fn verify_equiv(a: &[XGate], b: &[XGate], n: usize, seed: u64) -> Result<(), Stage4Error>;      // verify_equiv.py: PyRandom(77), 1024 all-wire states
pub fn check_windows_anf(gates: &[XGate], bounds: &[usize], body: &Deliverable, seed: u64) -> Result<usize, Stage4Error>; // EVERY window, 1024 full random states
pub fn check_deliverable(gates: &[XGate], d: &Deliverable, seed: u64) -> Result<(), Stage4Error>;   // decode(file) == circuit (DE match=True, but on the decoded FILE)
pub fn check_zero_slice(c: &[XGate], d: &Deliverable, n: usize, samples: usize, seed: u64) -> Result<(), Stage4Error>; // deliverable(x on 0..n, 0 elsewhere)[n..2n] == C(x)
```

`check_windows_anf` replaces AC's spot checks, which only test windows with `wi % 500 == 0`. Spot-check parity is not needed; AC's report numbers are reproduced by `census.rs`, see below.

`check_zero_slice` is the honest-slice end-to-end check that the reference chain never ran automatically.

### 4.14 `census.rs` (report only)

```rust
pub fn link_census(a: &OracleCircuit, p: &[XGate], bounds: &[usize]) -> LinkCensus;   // link_census.py, PyRandom(20261004) 1024 samples
pub fn anf_size_report(gates: &[XGate], bounds: &[usize], budget: usize, label: &str, sem: CapSemantics) -> String; // AC report lines (no spot checks)
```

These reproduce the numbers in `link_rep512.txt` (L2, M2, size histograms) and the first three lines of `anf_rep512.txt`, so that runs are judged by the program's metrics. They are not on the deliverable path.

### 4.15 `mod.rs`

```rust
pub struct Stage4Inputs<'a> { pub g2: &'a OracleCircuit, pub g3: &'a OracleCircuit, pub c: &'a [XGate], pub n: usize }
pub struct Stage4Output { pub packed: Packed, pub repaired: Repaired, pub body: Vec<u8>, pub container: Vec<u8>, pub report: Stage4Report }
pub fn run_stage4(inp: &Stage4Inputs, p: &Stage4Params, log: &mut dyn Write) -> Result<Stage4Output, Stage4Error>;
```

`run_stage4` dispatches the monomial width once with `with_mono_width(computed_n, …)`.

---

## 5. The production binary `tdp_packing`

Add to `Cargo.toml`:

```toml
[[bin]]
name = "tdp_packing"
path = "src/programs/tdp_packing/main.rs"
```

It has no required features. The adapter lives in `src/programs/tdp_packing/{main.rs,mod.rs,cli.rs}`: `main.rs` is three lines, `cli.rs` holds the clap derive, and `mod.rs` holds `run()` with I/O and logs. The binary reads no environment variables, except that `RAYON_NUM_THREADS` is honoured through `--threads`.

### 5.1 CLI

```
tdp_packing run   --stage2 <tdp.mpmct1> --stage3 <db_mixing.mpmct1> --source-c <tdp.mpmct1.source_c.g57>
                  --n <N> --out-dir <DIR>
                  [--window-gates 512] [--interval-cap 512] [--monomial-budget 24576] [--span-cap 3000]
                  [--split-budget 65536] [--fix-iterations 4] [--fallback-iterations 4]
                  [--anf-cap-semantics cpython39|canonical] [--zero-slice-samples 4096]
                  [--allow-floor-miss] [--allow-over-budget] [--threads K] [--no-oracle-files] [--raw-body]
tdp_packing pack      <A> <B> <W_cap> <G> <B_mono> <span_cap> <label> <out.mpmct1>   # WP argv, byte-parity mode
tdp_packing repair    --packed P --bounds B --source-c C --n N --out-dir DIR [repair flags]   # repair.sh parity mode
tdp_packing serialize <rep.mpmct1> <rep.bounds> <out.bin> <label> [--container N]          # DE parity mode
tdp_packing exposure  <hmap_affine exposure argv subset>   # optional convenience; hmap_affine stays the reference tool
tdp_packing decode    <deliverable.bin> [--stats]          # validate + summary
tdp_packing eval      <deliverable.bin> --input <hex|file> # reference evaluator (zero-slice or full state)
tdp_packing census    <A> <P> <B> <label>                  # link_census parity
```

The subcommands `pack`, `repair` and `serialize` accept the oracle's positional shape, so `tests/manual/stage4/parity.sh` can call either implementation the same way.

### 5.2 Inputs

| input | notes |
|---|---|
| `tdp.mpmct1` | the stage-2 circuit |
| `db_mixing.mpmct1` | the stage-3 circuit; `s3.mpmct1` in the reference |
| `tdp.mpmct1.source_c.g57` | the source C. Read only by repair and verify; it never reaches a shipped byte |
| `--n` | the source wires. It must equal the width of C, and `4n` must be at most the state width |

All three files are read through `read_oracle_mpmct` or `read_g57_file` with the strict checks of §4.2.

### 5.3 Outputs, all in `--out-dir`

The directory is created with mode 0700, and every file inside it is created under umask 077.

| file | content | ships? |
|---|---|---|
| `deliverable.bin` | container v1 (§6). Written to `deliverable.bin.tmp`, decoded and checked, then renamed. Only on success or an allowed failure | **yes, the only shipped file** |
| `deliverable.body.bin` | raw DE body, with `--raw-body` (parity) | no |
| `packed_G{G}.mpmct1`, `packed_G{G}.mpmct1.bounds` | packer output, WP format | no |
| `rep{G}.mpmct1`, `rep{G}.bounds` | repaired circuit and bounds, order_repair format (k=0 trailing space; header n carried over) | no |
| `rep{G}.exposure.tsv`, `rep{G}.allcuts.tsv`, `rep{G}.wit.tsv`, `rep{G}.cview.log` | the last in-loop scan, exactly as the oracle leaves them; omitted with `--no-oracle-files` | no (C-keyed, never ship; doc §4.8) |
| `pack_G{G}.txt`, `repair{G}.log`, `anf_rep{G}.txt`, `link_rep{G}.txt` | oracle-format text reports | no |
| `packing.report` | machine-readable report (§5.5) | no |

Nothing is written outside `--out-dir`. There are no TAG-derived paths and no pid-named scratch files.

### 5.4 Exit codes

| code | meaning | `deliverable.bin` |
|---|---|---|
| 0 | success, all checks passed | written |
| 2 | usage or parameter error | — |
| 3 | input error: unreadable, malformed, inconsistent header, non-canonical gate line, width mismatch | — |
| 4 | verification failure: pack VERIFY, `verify_equiv` mismatch, illegal move, ANF ≠ gates on any window, decode ≠ circuit, zero-slice ≠ C | not written |
| 5 | exposure floor missed: the final scan has a visible non-output row (identity test) | not written, unless `--allow-floor-miss`, which still exits 5 |
| 6 | window over the split budget after fallback (only 1-2 gate windows can remain) | not written, unless `--allow-over-budget`, which still exits 6 |
| 101 | panic (internal invariant) | — |

Exits 5 and 6 are deliberate departures from the oracle on failure paths only. The oracle silently shipped vL128a v1 at 129 visible. On every success path the bytes are unchanged.

### 5.5 `packing.report`

A flat `key=value` file, one key per line, in fixed key order. It parses trivially in bash and Python, and in Rust tests. It never contains seeds; stage 4 has none that are secret.

```
packing_report=1
status=ok|verify_fail|floor_miss|over_budget
tool=tdp_packing <crate version> <git describe if embedded>
binary_xxh3=<128-bit hex>
input.stage2_xxh3=…  input.stage3_xxh3=…  input.source_c_xxh3=…     (private run dir only)
n=64  state_wires=256
param.window_gates=512 param.interval_cap=512 param.monomial_budget=24576 param.mul_cap=196608 param.span_cap=3000
param.pad=4 param.cmax=24 param.fp_seed=20261004 param.fp_samples=1024 param.split_budget=65536
param.fix_iterations=4 param.fallback_iterations=4 param.cview_batches=96 param.cview_seed=12345
param.verify_seed=77 param.anf_cap_semantics=cpython39 param.repair_mode=oracle
pack.windows= pack.spans= pack.carried= pack.forced= pack.winnable= pack.disq_long= pack.disq_big= pack.rescued=
pack.intervals= pack.hidden_spans= pack.hidden_values= pack.death.burst_close= pack.death.sibling= pack.death.drain_close=
pack.exposed= pack.mul_calls= pack.mul_ambiguous= pack.compose_aborts= pack.compose_aborts_order_dependent=
repair.cancel_passes=51574,2543,191,10 repair.cancel_pairs= repair.fix_trajectory=65,64 repair.fix_moved=…
repair.fallback_trajectory=64,64 repair.merged=0 repair.splits=2,0 repair.forced_splits=0
repair.final_visible= repair.final_nonoutput_visible= repair.split_mul_ambiguous=
emit.windows= emit.monomials= emit.max_window_monomials= emit.body_bytes= emit.container_bytes= emit.deliverable_xxh3=
census.proxy_values= census.proxy_L2= census.proxy_M2= census.froth_L2= census.froth_M2=
check.pack_verify=ok check.equiv=ok check.windows_anf=ok(2776) check.decode=ok check.zero_slice=ok(4096)
time.pack_s= time.repair_s= time.emit_s= time.verify_s= peak_rss_mb=
```

---

## 6. Deliverable container v1

The DE body has no header, so it carries no n, no version and no integrity check. TDP doc §6.12 lists a container as an open decision. Container v1 wraps the **unchanged** DE body:

| offset | size | field | value |
|---|---|---|---|
| 0 | 8 | magic | `LMTDPANF` |
| 8 | u16 | format_version | 1 |
| 10 | u16 | header_bytes | 48 |
| 12 | u32 | flags | 0 (reserved; readers reject unknown bits) |
| 16 | u32 | source_wires | n |
| 20 | u32 | state_wires | header n of the repaired circuit (4n). Must be at most 65536 |
| 24 | u32 | io_layout | 1 = classic zero slice: x on wires 0..n, every other wire 0, answer on wires n..2n |
| 28 | u32 | window_count | |
| 32 | u64 | body_bytes | |
| 40 | u32 | body_crc32 | zlib/IEEE CRC-32 of the body |
| 44 | u32 | header_crc32 | CRC-32 of bytes 0..44 |

All fields are little-endian. The body is the DE byte stream, `<I len` plus blob per window.

Hence `deliverable.bin[48:]` is **byte-identical** to the reference chain's `deliverable.bin`. This is the "payload" in every parity statement.

**Python oracle update**, made in the same commit as the Rust serializer. `W/anf_dump_eval.py` gains an optional trailing `--container N`. Without it the output is unchanged, which keeps the archived runs and `val_chain.sh` reproducible. With it, the script writes the same 48-byte header, computed with `struct` and `zlib.crc32`, in front of the body. The header needs the window count, which is `len(blobs)`, and the state wires, which is the header n of the repaired circuit.

`scripts/windowed_delivery/README.md` documents the flag. `tests/manual/stage4/parity.sh` compares the full container produced by both sides, and the body against the archived files.

The decoder (`serialize::decode`, `tdp_packing decode`) validates the following, all of which `port_runs/anf_spec/decode_check.py` verified on vA64a:

- magic, version and both CRCs;
- blob lengths;
- support strictly ascending;
- changed wires strictly ascending;
- masks strictly ascending;
- no mask bit at or above f;
- every support index used;
- f = 0 if and only if c = 0.

---

## 7. Exactness plan

### 7.1 Per component

| component | oracle | target | how it is achieved and tested |
|---|---|---|---|
| CPython MT19937 / getrandbits | `random.Random` | bit-exact | golden vectors (Appendix A) |
| 1024-sample fingerprints, spans, forced, agree | WP:52-90 | bit-exact | `apply_lanes16`; full-value keys (D-a) |
| winnable values, intervals, iid order | WP:175-226 | bit-exact | first-appearance order; rescue rule |
| scheduler | WP:228-460 | **byte-exact `packed_G512.mpmct1` + `.bounds`** | ordered `committed`, stable sorts, quirks kept; sha256 against the 8 archived packs (Appendix B) |
| packer ANF caps | WP:135-150, 324-344 | bit-exact | `PySet` emulator (`CapSemantics::CPython39`); every compose abort reproduced (2 on vA64a, 6 on vL128b) |
| pack log text | WP prints | byte-exact on fixtures; replicates by line | Python `:.1f` vs Rust `{:.1}` can differ only on exact binary ties (D-c) |
| cancel / slide / fix | `order_repair.py` (canonical) | bit-exact | stale-blocker rule, greedy ascending, slide landing rules, gate-count cut, breaker survival |
| C-view TSVs and visible counts | HA exposure | **byte-exact TSVs** | library extraction proven against a pre-refactor `hmap_affine` built from `492564f0` (§10.5 step 1) |
| merge | `merge_bounds2.py` | bit-exact | outputs from C, asserted equal to the TSV rule |
| split | `anf_split2.py` | bit-exact | `window_anf` on `PySet` (CPython39); per-cut leak memo (exact by the cut-independence property); candidate, tie and recursion rules |
| loop control | `repair.sh` | bit-exact | count test `vis <= N`, 4+4 iterations, break rules; final scan report-only |
| `rep512.mpmct1` + `.bounds` | `order_repair.save` | **byte-exact** | k=0 trailing space; header n carried |
| per-window ANF and body bytes | DE | **byte-exact payload** | canonical (budget 1<<30); sha256 against the 8 archived `deliverable.bin` |
| container | new | byte-exact against the updated DE `--container` | |
| verify_equiv | seed 77 | verdict equal | sampled; plus per-move legality and exact all-window checks |
| link census / AC report | `link_census.py`, AC | numbers equal | report only |
| timings | wall clock | not compared | |

### 7.2 `CapSemantics` (decision recorded)

The default is **`CPython39`**. It is required by the defaults requirement and by the reproduction test, because without it the packer and split decisions diverge on real runs. It applies in the packer (MUL_CAP 196 608) and in the split predicate (8·65 536).

`Canonical` (`--anf-cap-semantics canonical`) aborts on a product only when the completed product exceeds the cap. It is an explicit, manifest-recorded deviation for anyone who wants an implementation-independent definition. It is never the default. The report's `*_ambiguous` and `*_order_dependent` counters show on every run how many decisions depended on the emulator.

### 7.3 Allowed differences

- **D-a.** Value identity is keyed on the full normalized 1024-bit value instead of its blake2b-128 digest. They differ only if two distinct values collide in blake2b-128 (probability about 2^-128). No observable difference is expected.
- **D-b.** Failure paths:
  - a hard error (exit 3, 4, 5 or 6) where the oracle continues, writes on `VERIFY FAIL`, or panics;
  - no background verify race;
  - no pid-named scratch files.
  On success paths there is no byte difference.
- **D-c.** Log float formatting on exact `.x5` binary ties (log only).
- **D-d.** `deliverable.bin` is the container. The payload is identical to the reference's raw file, and `--raw-body` writes the raw body as well.
- **D-e.** The oracle's dead DE placeholder pass and its timings are not ported (DE:70-90).

There is no other allowed difference. In particular, sorted iteration in place of CPython order is not allowed in default mode.

---

## 8. Determinism and seeds

**Stage 4 is a deterministic function of (G2, G3, C, n, parameters).** It uses fixed analysis constants that are part of the algorithm:

- the fingerprint seed 20261004;
- the C-view seed 12345 with 96 batches;
- the verify seed 77.

They are recorded in `packing.report`. Rayon parallelism is restricted to pure per-cut and per-window functions, merged in fixed order. Output therefore does not depend on the thread count; a test runs with 1 and 8 threads and compares.

**Gen and mix seeds (RC directive: always random, recorded).**

- The driver draws `gen_seed` and `mix_seed` independently from the OS CSPRNG, uniformly below 2^63 (§2.3).
- It writes both to `$RUN/SEEDS` with mode 0600, in the format `gen_seed=<u64>\nmix_seed=<u64>\nseed_source=random|explicit\n`.
- The manifest `tdp_command.conf` records `seed_record=SEEDS`, `seed_source`, `seed_bits=63` and the xxh3-128 of `SEEDS`. A resume therefore cannot silently swap seeds.
- **Seed values are never printed to the narrative log `tdp_gen.log` or to stdout.**

The seed values live in the private run directory. Writing the values themselves into `tdp_command.conf` would put a secret into the file that people paste when they compare recipes, so the record is the hash-pinned `SEEDS` file next to it. This is a decision for RC (§13, item 2). If RC prefers the values inside `tdp_command.conf`, that is a one-line change.

The stage logs (`stage12.log` echoes `seed=` through gen's banner) stay inside the private run directory, as today.

**Explicit seeds** are for calibration and reproduction only, and the run is marked `calibration_only=true`:

- TOML: `[calibration] enabled = true` and `seed_file = "<path>"`, where the file contains `gen_seed=…` and `mix_seed=…`;
- bash: `-s <gen> --mix-seed <mix>`.

Gen uses the single gen seed for all three positional seeds, `seed`, `gadget_seed` and `sandwich_seed` (D1). Stage 3 uses `--seed <mix_seed>` (D7).

---

## 9. Performance targets

Reference points (Python/bash):

| step | n=64 | n=128 |
|---|---|---|
| pack | 84 s | about 3.2 min |
| repair | about 2-4 min | about 10-12 min |
| DE serialize | — | 107 s for 106.8 MB |
| one HA scan | about 13 s | about 2.4 min, single-threaded |

Targets for the Rust stage 4 on a 10-core Apple-silicon Mac (release build):

| step | n=64 | n=128 | notes |
|---|---|---|---|
| pack | ≤ 15 s | ≤ 60 s | the emulator set for ANF; incremental `win_monos`; 9.1M mul calls at n=128 |
| cancel | ≤ 2 s | ≤ 10 s | |
| one full C-view scan | ≤ 5 s | ≤ 30 s | parallel over cuts; about 7 000 cuts × 3 000 rows at n=128 |
| repair total | ≤ 1 min | ≤ 4 min | at most 4 fix scans with a fresh oracle each; the fallback loop and every split share one memo |
| emit + all checks | ≤ 15 s | ≤ 60 s | parallel windows; decode and evaluate the file |
| **stage 4 wall** | **≤ 2 min** | **≤ 6 min, hard ceiling 10 min** | measured in `packing.report` and asserted by `parity.sh` |
| **peak RSS** | ≤ 1.5 GB | **≤ 4 GB (goal 2.5 GB)** | G3 as XGate (4.6M gates, about 180 MB); fingerprint index (about 2M × 136 B, about 300 MB); intervals and member lists (≤ 100 MB); C-view streaming (≤ 50 MB of states plus about 100 MB of checkpoints plus the memo); no materialized `gs` (that alone was 2.8 GB) |

---

## 10. Fixtures and tests

### 10.1 Unit tests (`tests/unit/stages/packing/*.rs`, synthetic, debug-mode fast)

| file | tests |
|---|---|
| `pyrand.rs` | `mt_first_words_seed_20261004`, `getrandbits_1024_lane_layout`, `seed_77_and_1_low128`, `getrandbit_is_top_bit` |
| `pyset.rs` | `replays_cpython396_traces` (reads `tests/fixtures/stage4/pyset39_traces.txt`), `dummy_not_reused_on_add`, `resize_reinserts_in_old_table_order`, `merge_fast_path_copies_slots_verbatim`, `xor_is_copy_b_then_toggle_a`, `int_hash_mod_m61_multi_limb` |
| `anf.rs` | `worked_example_doc_4_6` (gates `2 0 2 0 1 1 1`, `3 0 1 2 0`), `pair_toggle_not_dedup`, `swap_only_when_strictly_larger`, `running_sum_counts_identity_entries`, `prefix_abort_cpython_vs_canonical`, `compose_failure_mutates_nothing` |
| `fingerprint.rs` | `complement_normalization_and_constants`, `span_closes_on_every_write`, `first_birth_is_value_id` |
| `intervals.rs` | `interval_set_matches_bruteforce_taint`, `cap_returns_none_at_cap_minus_2`, `rescue_next_next_writer`, `iid_order_is_first_appearance` |
| `scheduler.rs` | `lowest_unscheduled_is_ready`, `committed_delete_preserves_order`, `compose_failure_boundary_after_gate`, `preemptive_close_can_emit_duplicate_boundary`, `contagion_snapshot_is_stale` |
| `cview.rs` | `complement_pair_collapses`, `constant_value_tstar_zero`, `nullity_le_14_exact_minimum`, `nullity_gt_14_counts_approx`, `earliest_cut_wins_tie`, `cut_file_parse_rules`, `samples_pinned_seed_12345`, `memo_equals_full_scan`, `parallel_equals_serial` |
| `repair/*.rs` | `to_from_seq_merges_and_drops_markers`, `cancel_dead_gates_still_block`, `cancel_reaches_fixed_point`, `slide_no_move_when_only_markers_before_blocker`, `slide_lands_before_markers_forward_after_backward`, `fix_breaker_located_before_completer_moves`, `fix_uses_stale_gate_count_cut`, `merge_keeps_input_order`, `split_candidates_and_smaller_tie`, `split_never_splits_two_gate_windows`, `split_forced_when_no_clean_candidate`, `oracle_loop_break_rules` |
| `serialize.rs` | `blob_23_byte_example`, `constant_monomial_first`, `crc32_matches_zlib_vectors`, `container_roundtrip`, `decoder_rejects_each_invariant_violation`, `decode_eval_equals_gates` |
| `verify.rs` | `zero_slice_detects_single_flip`, `equiv_detects_noncommuting_swap` |

### 10.2 Committed fixtures: `tests/fixtures/stage4/` (total under 2 MB)

| path | content |
|---|---|
| `README.md` | provenance: worktree commit; `sys.version` of the oracle interpreter (`/usr/bin/python3` 3.9.6); seeds marked CALIBRATION, never deliverables; `regen.sh` command line; file sizes |
| `regen.sh` | regeneration (below) |
| `pyset39_traces.txt` | about 100 KB: random add/discard/copy/xor/literal traces with 0..512-bit keys, and the expected iteration order after each step, recorded on 3.9.6 |
| `n8_L8/` | `g2.mpmct1`, `g3.mpmct1`, `c.g57`, `params` (n and the production pack and repair parameters), then the oracle outputs: `packed.mpmct1`, `packed.bounds`, `pack.txt`, `rep.mpmct1`, `rep.bounds`, `rep.exposure.tsv`, `rep.allcuts.tsv`, `rep.wit.tsv`, `repair.txt`, `deliverable.raw.bin`, `deliverable.tdp4.bin` |
| `n8_caps/` | the same inputs (or a sibling run), packed and repaired with **small caps** (pack `32 32 48 300`, split budget 64), so that compose prefix aborts and order-dependent split decisions happen in CI. Production caps never hit the ambiguous zone at n=8 |

`regen.sh`:

- builds `gen_sandwich_gadget`, `circuit_mixer`, `hmap_affine` (`--features security-tools`) and `tdp_packing` into `$CARGO_TARGET_DIR`;
- copies `scripts/windowed_delivery/` into a fresh `port_runs/fixture_<stamp>/` sandbox, rewriting `ROOT`, `RD`, `W`, the hard-coded `sys.path.insert` lines and `target/release` paths there with `sed`. Each rewrite must apply exactly once (asserted), and the tracked oracle is never edited;
- runs stages 1-3 with the L8 recipe on the local regular store, `n=8` and `|C|=|D|=16`, with fixed calibration seeds;
- runs the oracle stage 4 with `/usr/bin/python3` (the script refuses any other `sys.version`);
- copies the outputs into the fixture tree and prints the sizes.

### 10.3 Integration test (CI): `tests/integration/tdp_packing.rs`

Add `[[test]] name = "tdp_packing"`. It uses `env!("CARGO_BIN_EXE_tdp_packing")`:

- `pack_matches_oracle_n8_l8`, `pack_matches_oracle_n8_caps`: `packed.mpmct1`, `packed.bounds` and `pack.txt` compared byte for byte;
- `repair_matches_oracle_n8_l8`, `repair_matches_oracle_n8_caps`: rep circuit, bounds and the three TSVs, byte for byte;
- `serialize_matches_oracle_n8`: raw body and container, byte for byte;
- `run_end_to_end_n8`: `run` reproduces every file above and exits 0;
- `thread_count_does_not_change_output`;
- `exit_codes`: a malformed gate line gives 3; a tampered G3 (one gate flipped) gives 4; a forced floor miss (synthetic C with a planted interior-visible value) gives 5.

### 10.4 hmap_affine regression (CI)

The existing `cargo test --features security-tools --bin hmap_trace_affine` step stays. Add `tests/integration/hmap_affine_exposure.rs`, `required-features = ["security-tools"]`, which runs `hmap_affine --exposure` on the n8 fixture and compares its TSVs with `rep.*.tsv`.

### 10.5 Real-artifact parity (manual): `tests/manual/stage4/parity.sh`

This script reads `waist/<tag>/` read-only, writes to `port_runs/parity_<tag>/`, and is never run in CI. For each of the 8 tags:

1. **Rebuild `hmap_affine` at `492564f0`** into `port_runs/baseline_target`. Re-run the last repair scan (`rep512.mpmct1` + `rep512.bounds`) and compare with the archived `rep512.{exposure,allcuts,wit}.tsv` and line 2 of `rep512.cview.log`. This also settles the binary-provenance question of TDP doc §6.9. If they differ, report it and stop; do not chase the old binary. Then compare the refactored `hmap_affine` against the baseline on rep512, on packed + pre-repair bounds, on a split-candidate file, on a run without ALLCUTS, on SPAN, and on a small heatmap run (`cmp` of every file and of stdout).
2. `tdp_packing pack tdp.mpmct1 s3.mpmct1 512 512 24576 3000 …` must give a sha256 of `packed_G512.mpmct1` and `.bounds` equal to Appendix B.
3. `tdp_packing repair` on `packed_G512` must give a byte-identical `rep512.mpmct1` and `rep512.bounds`, plus identical TSVs against the archived last-scan files.
4. `tdp_packing serialize rep512.mpmct1 rep512.bounds` must give a body with the sha256 of the archived `deliverable.bin`. With `--container 64|128`, the output must be byte-identical to the updated DE `--container`, run in the sandbox.
5. `check_zero_slice` of the archived `deliverable.bin` against C.
6. Record times and peak RSS (`/usr/bin/time -l`) and assert the §9 targets.

### 10.6 Reproduction test (required acceptance): `tests/manual/stage4/reproduce_val_chain.sh`

Usage: `reproduce_val_chain.sh N MC [L8|A]`.

1. Draw `GEN` and `MIX` from the CSPRNG; these are calibration values. Keep them in the private `port_runs/repro_<stamp>/SEEDS` and never echo them.
2. **Reference side.** Run a sandbox copy of `W/val_chain.sh` (paths rewritten as in `regen.sh`). Its two seed lines are replaced by `SEED=$VAL_GEN_SEED` and `MSEED=$VAL_MIX_SEED` (each asserted to match once), and it is run under `env -i` with only `HOME`, `PATH`, `VAL_*` and `FROZEN_DB_DIR`. Arguments: `TAG N MC sandwich_default_s(N) 20N CFG`. Stage 4 runs with `/usr/bin/python3` 3.9.6. The DE call is made a second time with `--container N`.
3. **Managed side.** `local_mixing_bin tdp_gen --config <generated toml>` with:
   - `source.wires = N` and `source.gates = MC`;
   - `[calibration] enabled = true` with the seed file holding `GEN` and `MIX`;
   - every other setting at its default (preset L8 or A).
4. `cmp` each pair; all must be identical:

   | reference | managed |
   |---|---|
   | `tdp.mpmct1`, `.source_c.g57`, `.sandwich.mpmct1` | the same files |
   | `s3.mpmct1` | `db_mixing.mpmct1` |
   | `packed_G512.mpmct1(.bounds)` | `packing/packed_G512.mpmct1(.bounds)` |
   | `rep512.mpmct1`, `rep512.bounds` | `packing/rep512.mpmct1`, `packing/rep512.bounds` |
   | `rep512.{exposure,allcuts,wit}.tsv` | `packing/` copies |
   | `deliverable.bin` (raw) | `packing/deliverable.bin[48:]` |
   | container from DE `--container` | `packing/deliverable.bin` |

   Also compare `pack_G512.txt` line by line. Exit 0 only if everything is identical.
5. Required runs:
   - `N=16 MC=64 L8` (quick, minutes);
   - `N=64 MC=1000 L8`, the scale of vL64a/b (about 40 min per side);
   - one `A` run at N=16.
   The script prints a summary table; RC keeps it in the PR description.
6. **Historic anchor (optional, strongly recommended).** Re-run the managed driver with vL64a's archived seeds (read from `seed.txt` in a subshell, never printed) and `MC=1000`. Compare against the archived `tdp.mpmct1`, `s3.mpmct1`, `packed_G512*`, `rep512*` and `deliverable.bin`. A match proves the managed defaults reproduce the actual validation run, not just today's `val_chain.sh`. A mismatch in stages 1-3 only means the binaries changed since Oct 7; the cleanup branch claims they did not.

### 10.7 CI-sized version

The full chain needs the frozen store, which CI does not have, and CPython 3.9.6 for the Python packer, which CI's Ubuntu image does not guarantee. The required equivalence is therefore split into three CI checks whose composition is the reproduction claim, plus a local small end-to-end run:

1. **`tests/shell/tdp_defaults.bash`** (new, mocks). Mock `gen_sandwich_gadget`, `circuit_mixer` and `tdp_packing` record their argv and the filtered environment (`EMBEDDED_MASKING_*`, `CANON_*`, `*_CACHE_MB`, `FROZEN_*`, `SANDWICH_VARIANT`, `TDP_SOURCE_C`). The test asserts exact equality with the expected transcript literally derived from `val_chain.sh`:

   ```
   gen: <out> N MC MC S 1 SLICE GEN GEN GEN embedded-masking
   gen env: EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0
            EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1
            EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8
            SANDWICH_VARIANT=classic   (+ nothing else in the filtered set)
   mixer: --input … --output … --tdp --db-mixing --profile 2,2,2,2,2 --moves floor(4·G·1.3)
          --p-db 1.0 --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048
          --no-local-verify --seed MIX            (as a set of options; clap is order-insensitive)
   mixer env: FROZEN_DB_DIR=… FROZEN_FILTER=1 FROZEN_REGULAR_VALUE_CONVENTION=native (no CANON_*, no *_CACHE_MB)
   packing: run --stage2 … --stage3 … --source-c … --n N --out-dir …/packing   (no parameter flags = defaults)
   ```

   It includes an `A` case with no `CHAIN*`, a case with `EMBEDDED_MASKING_RETIRE=1` exported in the caller to prove it is scrubbed, and `G` values where `4·G·1.3` has a fractional part of at least .5, to pin floor against round.
2. **`tests/shell/tdp_stage12_parity.bash`** (new, real debug binaries). At n=16 with `|C|=|D|=32`, run gen exactly as `val_chain.sh` would (sandboxed env and argv) and `tdp_gen.sh --run-stop-after-stage 2` with explicit seeds, then `cmp` the three artifacts. This is the real-binary proof for stages 1+2. The stage-1+2 identity at small n relies on the same code path as at scale.
3. **`tests/integration/tdp_packing.rs`** (§10.3). Rust stage 4 against the committed Python outputs, including the cap-abort case.
4. **`tests/shell/tdp_reproduce_small.bash`**: the full §10.6 at `N=8 MC=16`, run for real. It **skips with a clear message** when no frozen store or no Python 3.9.6 is available, which is the case in CI. RC runs it locally before merging.

The CI workflow adds:

- `--test tdp_packing` to "Runtime and command correctness";
- `bash tests/shell/tdp_defaults.bash && bash tests/shell/tdp_stage12_parity.bash` to "TDP script flags and block sizing";
- `--features security-tools --test hmap_affine_exposure` to "Trace-affine heatmap".

---

## 11. Driver integration

### 11.1 `scripts/tdp_gen.sh`

**Usage and flags.** New flags:

- `--preprocessing-preset L8|A`, default L8;
- `--mix-seed N`;
- `--source-gates N`, an alias of today's `-m/--source-gates`, documented as the one size option;
- `--packing-window-gates`, `--packing-split-budget`, `--packing-anf-cap-semantics`, `--packing-allow-floor-miss`.

The stop/rerun range becomes `2..4`. `5` and `6` keep the "historic" error. The error message for `4` used to say "historic split" and now explains that 4 is packing.

**Stage-2 environment.**

1. Unset every inherited `EMBEDDED_MASKING_*` variable: `while read v; do unset "$v"; done < <(compgen -A variable EMBEDDED_MASKING_)`.
2. Export only the recipe set:
   - `SHUFFLING=8`, `SHUFFLING_RETURN_HOME=0`, `PREMASK=1`, `SEED_AVOID_C1=1`;
   - for L8 also `CHAIN=2` and `CHAIN_LEN=8`;
   - `K`, `MAX_OPEN`, `MIN_OPEN` and `BALANCED` **only when they differ from the production preset** (2, 3, 2, 1). They are always recorded in the marker and the manifest.
   - `QUAD_FIRE` is no longer exported; the preset value is true.

Remove the `return_home=0` rejection (`tdp_gen.sh:257-260`). Persistent migration restores the ports at the stage-2 tail (gen `mod.rs:62-70`).

**Seeds.** `GEN_SEED` and `MIX_SEED` replace `SEED`. Files and rules:

- `$RUN/SEEDS` holds both seeds, chmod 600. A v9 `$RUN/SEED` is rejected with "created by recipe v9; resume with the checkout that created it".
- An explicit `-s` or `--mix-seed` that conflicts with an existing `SEEDS` file is fatal, as today.
- Gen is called as `"$GADGET" "$N" "$M_CD" "$M_CD" "$S_SL" 1 "$SLICE_G" "$GEN_SEED" "$GEN_SEED" "$GEN_SEED" "$PREPROCESSING_MODE"`.

**Stage-12 marker.** `tdp_stage12_recipe=7` adds `preprocessing_preset`, `premask=1`, `seed_avoid_c1=1`, `chain`, `chain_len` and `gadget_seed_rule=gen`.

**Stage 3.**

- `G_IN` comes from the header field.
- The profile comes from `n0=GROWTH n1=n0+HOLD n2=n1 r1=EXPAND r2=r1`. With the defaults (GROWTH=2, HOLD=0, EXPAND=2) it is `2,2,2,2,2`.
- `A_MOVES = int(n2*r1*G*1.3)`, computed with Python `int()`, i.e. truncation.
- `--seed "$MIX_SEED"`.
- Delete `--verify-every 2000000 --report-every 100000 --state-out …`.
- Delete the `CANON_*` and `LOOKUP_CACHE_MB` exports.
- `FROZEN_FILTER`: `auto` exports `1`, with no `/proc/meminfo` gate and a background prewarm when `filters.bin` exists. The mixer itself falls back to unfiltered when the file is absent. `on` and `off` behave as today.
- Keep `MIXER_STOP_FLAG` and `MIXER_DUMP_FLAG`: they are polled only at report cadence and have no RNG effect when the files are absent.

**Stage 4** is a new block after stage 3:

```bash
PACK_DIR=$RUN/packing
if [ "$FORCE_FROM" -le 4 ] || [ ! -s "$PACK_DIR/deliverable.bin" ]; then
  rm -rf "$PACK_DIR"
  note "stage 4: tdp_packing run (G=$PACK_G, split budget $SPLIT_BUDGET, caps $CAP_SEM)"
  set +e
  "$BIN/tdp_packing" run --stage2 "$GADGET" --stage3 "$DB_MIXING" --source-c "$GADGET.source_c.g57" \
      --n "$N" --out-dir "$PACK_DIR" ${PACKFLAGS[@]+"${PACKFLAGS[@]}"} > "$RUN/stage4.log" 2>&1
  rc=$?; set -e
  [ "$rc" -eq 0 ] || { note "stage 4 FAILED (exit $rc; see $RUN/stage4.log, $PACK_DIR/packing.report)"; exit "$rc"; }
  printf '%s\n' "${PACKING_RECIPE[@]}" > "$RUN/packing.recipe"
fi
note "PIPELINE COMPLETE (stages 1-4): $PACK_DIR/deliverable.bin"
```

- `PACKFLAGS` is empty at the defaults, so the call carries exactly the defaults.
- `packing.recipe` holds `tdp_packing_recipe=1`, the binary sha256, the parameters, and the xxh3 of `tdp.mpmct1` and `db_mixing.mpmct1`. A mismatch on resume is fatal unless the run is rerun from 4.
- Add `tdp_packing` to the binary presence check (`tdp_gen.sh:292-294`). Delete the "stage 4 is not run" note.

### 11.2 `src/tdp` (managed path)

**`config/parse.rs` (`TOML_FIELDS`, sections).**

- `preprocessing.preset`: string, `"L8"` or `"A"`, default `"L8"`.
- `preprocessing.shuffling_segments`: default 8 (was 0).
- `preprocessing.shuffling_return_home`: default false (was true).
- `db_mixing.growth_work_units` (new): number, default 2.0. It is the profile's n0.
- `db_mixing.hold_work_units`: default 0.0 (was 27.0).
- `db_mixing.target_size_factor`: stays 2.0.
- A new `[packing]` section with:
  - `window_gates=512`, `interval_cap=512`, `monomial_budget=24576`, `span_cap=3000`;
  - `split_budget=65536`, `fix_iterations=4`, `fallback_iterations=4`;
  - `anf_cap_semantics="cpython39"`, `allow_floor_miss=false`.
  Add it to the section list (`parse.rs:255-266`).
- `[calibration] seed_file`: a two-line `gen_seed=`/`mix_seed=` file.
- `run.stop_after_stage`: default 4, range 2..4.

**`config/validate.rs`.** Delete the `return_home` rejection. Validate the preset. Validate that the packing integers are positive and that `monomial_budget·8` fits in usize. The `2*n > K + balanced` rule stays.

**`ResolvedConfig`** gains:

- `preprocessing_preset: PreprocessingPreset { L8, A }`;
- `growth: String`;
- `packing: PackingConfig`;
- `seeds: SeedPlan { Random | Explicit { gen, mix } }`.

**`runner.rs`.**

- `configure_environment`: stop setting `EMBEDDED_MASKING_{K,MAX_OPEN,MIN_OPEN,BALANCED,QUAD_FIRE}`. Instead pass the resolved values as `TDP_PREPROCESSING_*` script variables, which `tdp_gen.sh` translates following the rule in §11.1. Stop setting `CANON_*` and `*_CACHE_MB`. Set `FROZEN_FILTER` per §11.1.
- `script_args`: add `--preprocessing-preset`, the `--packing-*` flags and `--mix-seed`.
- The build list and the dry-run list (`runner.rs:185-188`, `521-524`) add `--bin tdp_packing`.
- Delete the "not driven yet" message (`215-220`).

**`paths.rs`.** Add the stage-4 prerequisites (`tdp.mpmct1`, `db_mixing.mpmct1`, `tdp.mpmct1.source_c.g57`) for `rerun_from_stage = 4`.

**`manifest.rs`.**

- `CURRENT_RECIPE_VERSION = 10`.
- `read_recipe_version` accepts only 10. A v9 manifest gives the explicit error "recipe v9 run: resume with the checkout that created it".
- `production_binary_fingerprints` adds `tdp_packing`.
- `recipe_manifest` adds:

  ```
  preprocessing_preset=L8
  preprocessing_premask=1
  preprocessing_seed_avoid_c1=1
  preprocessing_chain=2
  preprocessing_chain_len=8
  gadget_seed_rule=gen
  growth=2
  stage3_profile=2,2,2,2,2
  stage3_moves_rule=floor(n2*r1*G*1.3)
  stage3_verify_every=default(10000)
  stage3_report_every=default(50000)
  source_gates=<resolved number>
  source_gates_rule=derived|explicit
  seed_record=SEEDS
  seed_source=random|explicit
  seed_bits=63
  seeds_xxh3=…
  packing_window_gates=512  packing_interval_cap=512  packing_monomial_budget=24576
  packing_span_cap=3000  packing_split_budget=65536
  packing_fix_iterations=4  packing_fallback_iterations=4
  packing_anf_cap_semantics=cpython39  packing_allow_floor_miss=false
  ```

  The pinned line becomes `pinned_recipe=canon-unbounded-cache-default-native-regular`.

**`configs/tdp.toml`.** Shows the new defaults and the `[packing]` section, with comments: "defaults reproduce `val_chain.sh` CONFIG=L8 (docs/STAGE4_RUST_DESIGN.md §2)" and "`source.gates`: the validation runs used 1000 (n=64) and 3000 (n=128)".

### 11.3 Test updates

**`tests/unit/tdp.rs`:**

- `243-284`: stop/rerun 4 is now accepted; 5 and 6 are rejected.
- `913-949`: `return_home=false` is now accepted; the rejection test is replaced by "shuffling 1..7 rejected".
- `986-1007`: the new exact default manifest text.
- `1009-1067`: three mock binaries; a v10 fresh run.
- New tests:
  - `default_profile_is_2_2_2_2_2`;
  - `moves_floor_not_round`;
  - `preset_a_drops_chain`;
  - `v9_manifest_rejected_with_message`;
  - `explicit_seed_file_two_lines`;
  - `seeds_never_in_narrative_log`;
  - `env_has_no_canon_caps`.

**`tests/shell/tdp_flags.bash`.** Lines 40-41 now expect `return_home 0` to be accepted. The mock key list at line 64 gains PREMASK, SEED_AVOID_C1, CHAIN and CHAIN_LEN, and drops K, MAX_OPEN, MIN_OPEN, BALANCED and QUAD_FIRE at the defaults.

**`tests/shell/tdp_block_size.bash`.** Lines 35-40: only 5 and 6 are rejected. Line 89: three stage calls. Line 96: the "PIPELINE COMPLETE (stages 1-4)" line. Add a `tdp_packing` mock.

**New:** `tests/shell/tdp_defaults.bash`, `tests/shell/tdp_stage12_parity.bash` and `tests/shell/tdp_reproduce_small.bash` (§10.7).

### 11.4 Stage-2 knobs: environment now, typed parameters later

The four knobs that distinguish L8 and A are read from the environment **inside stage code**:

- `construct.rs:89`: `SEED_AVOID_C1`;
- `embedded_masking.rs:1611-1620`: `CHAIN`, `CHAIN_LEN`;
- `embedded_masking.rs:1927`: `PREMASK`.

Phase 1 of the port (§12) keeps them there and drives them by the exact environment of §11.1. That is literally what `val_chain.sh` does, and it is proven by `tdp_stage12_parity.bash`.

Phase 2 (optional, separate commit) moves them into `EmbeddedMaskingParams { premask, seed_avoid_c1, chain_k, chain_len }`, with the gen adapter keeping the env names. Its acceptance test is `stage2_typed_params_byte_identical`: for 3 fixed seeds at n=16, each in A and in L8, the sha256 of the outputs from the pre-refactor binary is committed, and the refactored output must match.

---

## 12. Implementation phases and commits

Each phase ends with gates G1-G7, `cargo fmt --check` and clippy green, and a WIP commit.

| phase | content | acceptance |
|---|---|---|
| P0 | this document | committed |
| P1 | `oracle_io`, `pyrand`, `pyset` (+ traces), `mono`, `anf`; `apply_lanes16` | unit tests; trace replay; window-ANF replay of vA64a and vL128b packs reproduces all 8 aborts |
| P2 | `gf2` + `cview` extraction; `hmap_affine` refactor | byte-identical against the baseline `hmap_affine` (§10.5 step 1) on vA64a and vA128a |
| P3 | `fingerprint`, `intervals`, `scheduler`; `tdp_packing pack` | sha256 of the 8 packs and bounds (Appendix B); `pack.txt` lines |
| P4 | `repair/*`; `tdp_packing repair` | byte-identical `rep512.*` and TSVs on the 8 replicates |
| P5 | `serialize`, `verify`, container; DE `--container` | sha256 of the 8 `deliverable.bin`; container equals DE; zero slice |
| P6 | `run`, report, exit codes; `census`; fixtures + `regen.sh`; integration tests | `tests/integration/tdp_packing.rs` in CI |
| P7 | driver defaults (D1-D14), presets, seeds, stage-4 step, manifest v10, shell and unit tests | G4 and the new shell tests |
| P8 | reproduction tests §10.6 (N=16 L8, N=16 A, N=64 L8), historic anchor | all `cmp` identical; times and RSS within §9 |
| P9 | docs (§14) | — |
| P10 (optional) | typed stage-2 knobs (§11.4) | `stage2_typed_params_byte_identical` |

---

## 13. Open decisions for RC

1. **Source size default:** formula (2304 at n=64, 6272 at n=128) versus the validated 1000 and 3000. The formula is kept until RC decides. It is one option, `source.gates`, and is recorded.
2. **Where the seed values live:** the hash-pinned private `$RUN/SEEDS` (proposed), or inside `tdp_command.conf`.
3. **Seed width:** 63-bit (proposed, §2.3) versus the reference's 31-bit gen and 40-bit mix.
4. **Repair improvements** (TDP doc §4.9): slide across markers, marker-identity cuts, all exposing cuts, finer split candidates against leaky-split cycles. Each would be a new `RepairMode` variant behind a flag, never the default, until RC re-validates.
5. **Canonical caps:** whether to ever switch the default to `CapSemantics::Canonical`. That would change bytes relative to the oracle on real runs.

## 14. Documentation to update (P9)

- `docs/TDP_FOUR_STAGE_PIPELINE.md`:
  - §4.5 and line 171: the snapshot and canonical `order_repair.py` are now one file;
  - §4.7: all 8 replicates were re-repaired with the canonical version and re-serialized;
  - §4.9 and §5.7: the port status, linking here;
  - §6.10 and §6.12: decisions taken;
  - Appendix A: the gaps closed by D1-D14.
- `docs/tdp_pipeline.md`:
  - the quick start runs stages 1-4;
  - the configuration reference gains the preset, seeds, `growth_work_units` and `[packing]`;
  - "Stage 4: packing" describes `tdp_packing`, the outputs, the exit codes and the container;
  - the reference driver becomes a parity tool.
- `docs/ANF_WINDOW_DELIVERY.md`: §1.2 wrongly says every window is re-verified; the format section adds container v1.
- `docs/windowed_delivery_design.md` §7: the plan is replaced by a pointer here.
- `docs/STAGE2_LINKABILITY.md` §6: the managed default is L8.
- `scripts/windowed_delivery/README.md`: the oracle role, the `--container` flag, the sandbox rule.
- `configs/tdp.toml` comments.

---

## Appendix A. Golden vectors (from `/usr/bin/python3` 3.9.6)

**`random.Random(20261004)`:**

- first four `genrand_uint32` outputs: `749935133, 232581420, 3240862439, 1300416022`;
- `getrandbits(1024)` for wire 0: low64 `0x0ddce92c2cb31a1d`, high64 `0x3f688f70a42f573a`, low128 `0x4d82c616c12ba2e70ddce92c2cb31a1d`;
- wire 1: low64 `0x12e4aa97d52ee91c`;
- `blake2b(digest_size=16)` of wire 0 (`to_bytes(128,'little')`): `10443c04d722b9ea2873b4564780ebb1`.

**`getrandbits(1024)`, low128 of the first draw:**

- `Random(77)`: `0x32833106536e95df40b2d002cc92d33d`;
- `Random(1)`: `0xcd613e30d8f16adf91b7584a2265b1f5`.

## Appendix B. Parity targets (sha256 of archived files in `runs/ncdb_ab_20261001/waist/<tag>/`)

Hashes of the source C, `tdp.mpmct1` and `s3.mpmct1` are intentionally not listed (§2.3).

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

The `rep512.*` and `deliverable.bin` files are the Oct 8 re-repairs made with the canonical `order_repair.py`. The superseded versions are in `rep512_v1/`. None of the eight `packed_G512.mpmct1.bounds` files contains a duplicate.
