# The Four-Stage TDP Pipeline

**Status (2026-10-09).** Written on branch `ran/stage2-windowed-delivery`; committed on `ran/cleanup-four-stage`; stage 4 ported on `ran/stage4-rust`. Line numbers in tracked files refer to commit `ea2e516f` (before the cleanup that removed the historic code; stage-3 line numbers have since shifted). Line numbers in stage-4 scripts refer to the untracked run tree `runs/ncdb_ab_20261001/` (see "Conventions").

- **Rust: all four stages.** Stages 1 and 2 run in one process, `gen_sandwich_gadget`. Stage 3 is `circuit_mixer --tdp --db-mixing` with the regular frozen store only. Stage 4 is `tdp_packing run` (`src/stages/packing/`), a bit-exact port of the Python prototypes (`docs/STAGE4_RUST_DESIGN.md`).
- **The managed driver runs the recommended configuration by default.** `tdp_gen` with nothing set but `source.wires` (64 or 128) reproduces `val_chain.sh` CONFIG=L8, byte for byte from the same seeds (stage-2 preset L8, schedule `validated`, stage 4 at G=512): proven on the archived runs vL64a and vL128a themselves (the historic anchor, `STAGE4_RUST_DESIGN.md` §12) and by the reproduction test against a sandboxed `val_chain.sh` (n=8, n=16 L8 and A, n=64 L8). Config A is `preprocessing.preset = "A"`.
- **Oracle:** the Python/bash stage-4 scripts (tracked snapshot `scripts/windowed_delivery/`) are the canonical oracle of the port. They are never edited; CI runs them against `tdp_packing` on random inputs (§4.9).
- **Historic:** the curated DB and the old stages 4-6 (see "Historic components").
- **Recommended configuration:** config L8 (bounded chains). Config A (no chains) remains a supported alternative (smaller at n=64, more linkable values; §5.6).
- **Canonical repair code:** `scripts/windowed_delivery/waist/order_repair.py` (the patched version; the
  slide step locates the breaker before moving the completer, §4.5), ported as `stages::packing::repair`.
- **Validation (closed 2026-10-08).** Twelve end-to-end runs (configs A and L8, n=64 and n=128, three seeds
  each) all reach the exact C-view floor, are functionally equivalent after repair, and pass the end-to-end
  check `verify_zero_slice` (C(x) against the final circuit's payload wires, 512 samples, fresh seeds). All
  eight replicates were re-repaired with the canonical code and re-serialized (§4.7).
  Results: `docs/STAGE2_LINKABILITY.md` §5 and `docs/notes/ncdb_ab_20261001_RESULTS.md`. The two items that
  were open are closed:
  1. **`hmap_affine` provenance (§6.9):** `hmap_affine` rebuilt at `492564f0` reproduces every archived C-view TSV and log line on all eight replicates; the refactored tool (library exposure engine) is byte-identical to it.
  2. **Managed driver (§6.10):** it now expresses, and defaults to, the recommended configuration.

## Overview

### What the pipeline produces

The input is a secret reversible circuit C on n wires, made of g57 gates. The output is one file, `deliverable.bin`. It is a sequence of **windows**, and each window ships only as the canonical algebraic normal form (ANF) of the permutation it applies. Composed, the windows compute a 4n-wire permutation with a public **zero slice**:

```text
𝒢(x, 0ⁿ, 0²ⁿ)   = (junk, C(x), junk)             forward: C(x) appears on wires n..2n
𝒢⁻¹(a, 0ⁿ, 0²ⁿ) carries D⁻¹(a) on wires n..2n    backward: an unrelated random function
```

Here 𝒢 is the deliverable permutation, the function computed by `deliverable.bin`. The letter G on its own is kept for the stage-4 window gate cap (§4.3).

D is a random decoy circuit with the same design and size as C (§1.4, §1.5). Stage 2 preserves the sandwich's payload half on the band-zero hyperplane (§2a.7). Every later stage preserves the full 4n-wire permutation of the stage-2 circuit G2.

### Notation

| symbol | meaning |
|---|---|
| n | width of C. Wires `0..n` = x (input; junk at the output); `n..2n` = y (answer register; 0 on the slice); `2n..4n` = band (0 on the slice). Ranges are half-open |
| C, m_C | the secret source circuit and its gate count: \|C\| = 1,000 at n=64 and 3,000 at n=128 in the pipeline |
| D, m_D | the decoy (§1.4); m_D = m_C |
| A | the stage-1 **sliced sandwich**, on 2n wires |
| G2, \|G2\| | the stage-2 circuit, or **gadget** (`tdp.mpmct1`), on 4n wires, and its gate count. Code blocks hold \|G2\| in the shell variable `NG2` |
| G3 | the stage-3 mixed circuit (`s3.mpmct1`), on 4n wires |
| 𝒢 | the deliverable permutation (Overview) |
| G | the stage-4 **window gate cap**, 512 in production (packer argument 4, §4.3). In `repair.sh TAG G N`, G is only the file-name key `<G>` of `packed_G<G>` / `rep<G>` (§4.5) |
| P_a, P_b | the two operand polynomials of a stage-2 fire (§2b.5) |
| B_mono | the packer's soft monomial budget per window (§4.3) |
| K (`k`) | band wires per mask cycle (`EMBEDDED_MASKING_K`, production 2; gen `mod.rs:254`, §2b.2). The dropped retirement lever is always written "K=64 retirement" (`EMBEDDED_MASKING_RETIRE=64`, §2a.5) |
| `packed_G512`, `rep512` | the stage-4 packed circuit, and the repaired circuit with its window lattice (`rep512.mpmct1` + `rep512.bounds`) |
| config A / config L8 | the recommended stage-2 environment without / with bounded chains (`CHAIN=2 CHAIN_LEN=8`) |
| vA64a … vL128b | the validation replicates `runs/ncdb_ab_20261001/waist/v{A,L}{64,128}{a,b}/` |

The prototype scripts name their stage-2 and stage-3 inputs `A` and `B`. `repair.sh` also uses `B` for its bounds path. This document calls the circuits G2 and G3 throughout, and keeps A for the sandwich.

### Data flow

```text
                               n = 64                         n = 128
C (secret, .source_c.g57)      64 wires, 1,000 g57            128 wires, 3,000 g57
  │  stage 1  sliced sandwich (gen_sandwich_gadget, Rust)
A  (.sandwich.mpmct1)          128 wires, 2,832 gates         256 wires, 7,920 gates
  │  stage 2  preprocessing: guards, band seed, masked compute,
  │           carrier migration, band reseed (same process)
G2 (tdp.mpmct1)                256 wires, 0.777-0.779 M       512 wires, 2.22-2.24 M
                               (~14 MB text)                  (~41 MB text)
  │  stage 3  database mixing (circuit_mixer --tdp --db-mixing, regular store)
G3 (s3.mpmct1)                 1.59-1.66 M (×2.04-2.13)       4.59-4.79 M (×2.05-2.14)
                               (~30 MB text)                  (~90 MB text)
  │  stage 4  pack (window_pack3.py) → repair (repair.sh: cancel ~7%, slides,
  │           merge/split) → serialize (anf_dump_eval.py)          [Python prototypes]
deliverable.bin                2,588-2,793 windows            7,053-7,416 windows
                               26.0-32.8 MB (xz 2.8-3.1 MB)   103-107 MB (xz 9.5-9.8 MB)
```

The figures come from the eight validation replicates (§4.7, §6.9). With 8 jobs running concurrently, a run took about 40 min at n=64 and about 2 h at n=128, and stage 3 took 74-88% of it (§6.9).

### Threat model in brief

- **What ships.** Only the window ANFs: no gate list, no stage provenance, and no metadata beyond the windows themselves (§4.1, §4.6).
- **What the adversary sees.** The inputs and the full wire state at every window boundary. Nothing inside a window.
- **Security targets** (§4.1, §5.3, §5.5):
  1. **C-view at the floor.** No interior value of C is an exact affine function of any boundary state. Only the n outputs are visible, and those are visible by definition.
  2. **Low linkability.** Few stage-2 values can be recognized at more than one place in the artifact (L2, M2 and anchors). This target is informational: there is no acceptance threshold. §6.7 gives the replicate reference ranges.
- **What is not measured.** Only exact affine predictors and exact equality are measured. Low-degree nonlinear predictors and approximate predictors have not been run on final artifacts. The adversary's input domain (honest slice or all wires) is not yet fixed (§4.1, §5.1).
- **Secrets.** C is a deterministic function of the generator seed. Every file other than `deliverable.bin` is secret or a secret intermediate, and stage-4 repair reads C. So all four stages run in the secret-holding environment (§1.8, §4.8, §6.8).

### What each stage contributes

| stage | contributes | leaves open |
|---|---|---|
| 1 sliced sandwich | Fixes the problem shape: a public zero slice and an answer register. Reversal yields the random D⁻¹, not C⁻¹, which removes the "reverse the gate list" attack | No hiding at all: on its slice, A runs C gate by gate in the clear |
| 2 preprocessing (embedded masking) | Fires each sandwich gate under ≥ 2 balanced quadratic masks keyed on a band that is filled from the input, with a mid-fire straddle mask that hides which gate fired. Every gate has ≤ 2 controls, the vocabulary the store digests. Slice guards sit on both ports. Carrier migration (dose 8, persistent) erases the data/band wire classes. PREMASK and SEED_AVOID_C1 close the first-layer fringe, and L8 chains cut bracket rebirths. No affine relation to C beyond the outputs was found at any scanned stage-2 cut. The scans were strided: every 16th / 128th cut (n=64 / n=128) in the 29-variant study and every 64th / 256th in the replicates, with dense scans of the final ~4k / ~8k gates on a subset of the study variants (§5.6) | Verbatim rebirths of band values (bracket restores, migration hops); recognizable unshuffled guard and tail blocks; cuts between scan points are untested |
| 3 database mixing | Un-invertible respelling by exactly verified store splices; dilution to about 2× size; digestion into about 88% g57 gates (L8); long-range wire-frame twists | Fringe C-view leaks appear only after reordering (localized to the mixer on one artifact, §5.6); duplicate commuting gate pairs (~7% of gates, source unknown) |
| 4 windowed ANF packing | Deletes cuts, so a value alive only inside one window is never observable. Window text collapses into a canonical ANF. Repair cancels duplicate pairs, slides completers and breakers, and merges and splits boundaries, trying to bring the C-view to the floor. It runs at most 4 fix and 4 fallback iterations, and success is not enforced (§4.5) | The window lattice is visible and deterministic; repair needs C; floor attainment is not guaranteed (vL128a's first repair ended at 129 and was still serialized) |

### Recommended configuration at a glance

The supported way to run it is the managed driver, whose defaults are this
configuration (`docs/tdp_pipeline.md`):

```sh
cp configs/tdp.toml configs/local.toml        # set database.regular_dir and source.wires (64 or 128)
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

The reference driver below is what the validation runs used; it is now a
parity tool (the reproduction test runs it in a sandbox):

```sh
bash runs/ncdb_ab_20261001/waist/val_chain.sh TAG N MC S SLICE CFG   # CFG = L8 (recommended) or A
```

with (N, MC, S, SLICE) = (64, 1000, 384, 1280) or (128, 3000, 896, 2560) (§6.3). §6.4-§6.7 explain each step and the pass criteria.

The block below is the driver body (`W/val_chain.sh:6-35`), self-contained, with two changes. `env -u FROZEN_CURATED_DIR` is added, because without it an exported variable silently turns stage 3 curated-first (§3.1). The inert `--p-comp 1.0 --p-any 0.1` is dropped: those flags acted only in the removed crossing walk and are being removed (§3.2).

```sh
set -uo pipefail                                         # no -e: failures do not stop the run (§5.2)
ROOT=/Users/rancanetti/code/local_mixing                 # hard-coded in the stage-4 scripts (§6.1)
RD=$ROOT/runs/ncdb_ab_20261001; W=$RD/waist
TAG=vL64x; N=64; MC=1000; S=384; SLICE=1280; CFG=L8      # n=128: N=128 MC=3000 S=896 SLICE=2560
R=$W/$TAG; mkdir -p $R; cd $ROOT                         # R must be exactly $W/$TAG: repair.sh and
                                                         # anf_split2.py rebuild it from TAG
SEED=$(python3 -c 'import secrets;print(secrets.randbelow(2**31))')    # gen seed (all three)
MSEED=$(python3 -c 'import secrets;print(secrets.randbelow(2**40))')   # mixer seed
CH=""; [ "$CFG" = L8 ] && CH="EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8"
{ echo "$TAG gen seed: $SEED"; echo "mix seed: $MSEED"; echo "config $CFG: PREMASK SEED_AVOID_C1 SHUFFLING=8 RETURN_HOME=0 $CH"; } > $R/seed.txt

# stages 1+2 (one process)
env EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 $CH \
  target/release/gen_sandwich_gadget $R/tdp.mpmct1 $N $MC $MC $S 1 $SLICE $SEED $SEED $SEED > $R/gen.log 2>&1

# stage 3 (regular store only)
NG2=$(head -1 $R/tdp.mpmct1 | awk '{print $3}')                # |G2| from the mpmct1 header
MOVES=$(python3 -c "print(int($NG2*2*2*1.3))")                 # ⌊N2·R1·1.3·|G2|⌋ = ⌊5.2·|G2|⌋ (§3.1)
env -u FROZEN_CURATED_DIR FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  target/release/circuit_mixer --input $R/tdp.mpmct1 --output $R/s3.mpmct1 --tdp --db-mixing \
  --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
  --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
  --no-local-verify --seed $MSEED > $R/s3.log 2>&1

# background stage-2 checks: rebirth census and span check (§6.6)
python3 $W/mult_census.py $R/tdp.mpmct1 $TAG > $R/mult_s2.txt 2>&1 &
STEP=$([ $N = 64 ] && echo 64 || echo 256)
SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=$R/span_s2.tsv target/release/hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 \
  --g $R/tdp.mpmct1 --n $N --exposure --c-step 1 --g-step $STEP --out $R/span_s2 2>&1 | grep "\[span\]" > $R/span_s2.txt &

# stage 4: pack (run from RD), repair, serialize
cd $RD
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1 > $R/pack_G512.txt 2>&1
bash $W/repair.sh $TAG 512 $N > $R/repair512.log 2>&1      # 512 = the packed_G512 file-name key
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG > $R/deliver.txt 2>&1
gzip -9 -kf $R/deliverable.bin; xz -9 -kf $R/deliverable.bin
ls -la $R/deliverable.bin* | awk '{print $5, $9}' >> $R/deliver.txt
wait                                                     # for the two background checks
echo "$TAG VALIDATION DONE"
```

Seeds are fresh CSPRNG draws (§6.2). They are passed on command lines and so are visible in `ps`, which is an open gap. The §6.7 pass criteria are greps of exactly the files this block writes: `gen.log`, `s3.log`, `mult_s2.txt`, `span_s2.txt`, `pack_G512.txt`, `repair512.log` (plus `repair.sh`'s own `rep512.cview.log`, `anf_rep512.txt` and `link_rep512.txt`) and `deliver.txt`.

### Conventions of this document

- **The code is the authority.** Where a code comment or another doc disagrees with the code, this document follows the code and marks the other text **[stale]**. Claims labelled *derived* or *inferred* are reasoned from the code and were not measured or stated by the authors.
- **Sections.** §1-§4 are the four stages (stage 2 is split into §2a-§2c). §5 covers verification, metrics and attack tooling, and §6 how to run the pipeline. Appendix A holds the managed-driver gap analysis.
- **Paths** are relative to the repository root. Abbreviations used across sections:

| short form | path |
|---|---|
| `RD/` | `runs/ncdb_ab_20261001/` |
| `W/` | `runs/ncdb_ab_20261001/waist/` |
| `R` | a run directory `W/<TAG>` |
| `EM` | `src/stages/preprocessing/embedded_masking.rs` |
| `HA` | `security_tests/heatmaps/hmap_affine.rs` |
| `RESULTS.md` | `docs/notes/ncdb_ab_20261001_RESULTS.md` (lab notebook) |
| `STAGE2_LINKABILITY.md` | `docs/STAGE2_LINKABILITY.md` |

Two files are both called `mod.rs`. In §1, §2a and §2b, a bare `mod.rs` (or `gen:`) is `src/programs/gen_sandwich_gadget/mod.rs`. In §3, a bare `mod.rs` is `src/programs/circuit_mixer/mod.rs`. Sections with further abbreviations list them at their start.

**Stage-4 scripts: one convention for the whole document.** The run-tree copies are authoritative until the snapshot is refreshed. A stage-4 script path, whether written `RD/x.py`, `W/x.py`, `waist/x.py` or as a bare script name, means the run-tree copy: `RD/` for top-level scripts, `W/` (= `waist/`) for the others. The tracked snapshot `scripts/windowed_delivery/` mirrors that layout, with `waist/` = `W/`. Every snapshot file is byte-identical to its run-tree copy except `waist/order_repair.py`, and that file is identical through line 162. Its later lines are shifted by one in the live copy (§4.5). So every cited line number holds for both copies, except `order_repair.py` lines ≥ 163, which refer to the live copy. `W/val_chain.sh` exists only in the run tree.


## Stage 1: the sliced sandwich

Stage 1 takes the secret source circuit C (n wires, g57 gates) and builds the **sliced sandwich** A, a reversible circuit on 2n wires. On a public zero slice, A computes C. Run in reverse on that same slice, it computes an unrelated random function.

Stage 1 adds no hiding: on its slice, A still runs C in the clear, gate by gate. Its job is to fix the problem shape (which inputs are free, which are pinned to zero, and where the answer appears) and to remove the trivial "reverse the gate list" attack (`docs/local_mixing_documentation.md:1262-1273`, `:1402-1405`). Stages 2 to 4 do the hiding.

### 1.1 Code location and invocation

All of the logic is in `src/stages/sandwich/construct.rs`, re-exported by `src/stages/sandwich/mod.rs:1-4` ("Stage 1: source preparation and sliced-sandwich construction"). Stage 1 has no binary of its own. `gen_sandwich_gadget` (`Cargo.toml:92-93`, `src/programs/gen_sandwich_gadget/main.rs:3-5`) runs stages 1 and 2 in one process:

- stage 1 is `src/programs/gen_sandwich_gadget/mod.rs:96-232`;
- the stage-2 handoff is `:236-295`;
- payload verification is `:356-369`;
- the final gadget is written at `:371`.

In the current pipeline the reference call is `W/val_chain.sh:15-16`:

```sh
env EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 \
    EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 [$CH] \
  target/release/gen_sandwich_gadget $R/tdp.mpmct1 $N $MC $MC $S 1 $SLICE $SEED $SEED $SEED
```

The `EMBEDDED_MASKING_*` variables configure stage 2, and `$CH` holds the optional bounded-chain pair (`val_chain.sh:13`). Stage 1 reads none of them.

**[stale]** The module doc at `mod.rs:20-22` still says `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME` defaults to true and "is required when enabled here". The parser accepts `0`/`false` (`mod.rs:61-73`), which selects persistent carrier migration: data roles are swapped back to their physical ports once, at the stage-2 tail (§2c.7). `src/stages/preprocessing/construct.rs:39-48` asserts that the roles end up back on their home ports.

The arguments are positional only (`mod.rs:11-15`, parsed at `mod.rs:96-140`). This is the canonical argument table for the whole document:

| # | name | binary default | current pipeline (n=64 / n=128) |
|---|---|---|---|
| 1 | `out` | required | `$R/tdp.mpmct1` |
| 2 | `n` | 128 (`mod.rs:100`) | 64 / 128 |
| 3 | `m_C` (\|C\|) | 3000 (`:101`) | 1000 / 3000 |
| 4 | `m_D` (\|D\|) | 3000 (`:102`) | same as `m_C` |
| 5 | `s` (gates per slice block) | `sandwich_default_s(n)` = max(n, round(n·log₂n)) (`:103-106`; `construct.rs:51-54`) | 384 / 896 (equal to the default) |
| 6 | `rg_freq` | 1 (`:107`) | 1, unused (see 1.8) |
| 7 | `slice_gates` | `10·2n` (`:108-112`) | 1280 / 2560 (equal to the default; the stage-2 guard size per side, §2a.6) |
| 8 | `seed` | 1 (`:113`) | `SEED` |
| 9 | `gadget_seed` | `seed` (`:114`) | `SEED` (drives all of stage 2, §2a.2) |
| 10 | `sandwich_seed` | `seed` (`:115`) | `SEED` |
| 11 | `preprocessing_mode` | `embedded-masking` (`:116-123`; `src/stages/preprocessing/types.rs:8-12`) | omitted |
| 12 | `sandwich_variant` | `classic`, or env `SANDWICH_VARIANT` (`:127-136`) | omitted, so `classic` unless the caller's environment sets `SANDWICH_VARIANT` |

`val_chain.sh` neither pins nor unsets `SANDWICH_VARIANT`. If the variable is exported as `balanced` or `mirror`, the run silently builds the balanced sandwich, and the reverse check is then skipped (`src/stages/preprocessing/verify.rs:71`). The driver should pin it.

Two parsing behaviours matter.

- **Silent fallback.** Every numeric positional is parsed with `.and_then(|s| s.parse().ok()).unwrap_or(default)`. A malformed number therefore falls back to its default without any error.
- **What panics during parsing.** Only four things: a missing `<out>` (`:97-99`), an unknown mode (`:121-123`), an unknown variant (`:132-134`), or a 13th argument (`:137-140`).

Before it writes anything, the generator does three things:

1. It asserts that no `BV5_*` or `PROD_*` environment variables are set (`mod.rs:150-171`).
2. It validates the stage-2 shuffling controls (`:173-182`; §2c.1).
3. It prints a header line, `[gen] n=… |C|=… |D|=… s=… rg_freq=… slice_gates=… seed=… gadget_seed=… sandwich_seed=… gadgetization_mode=… sandwich_variant=…` (`:184-188`). A seed given as `@PATH` (read from PATH; the managed driver passes `@/dev/fd/N` so no seed is on argv) is printed as `(file)`.

**The header contains the raw seeds, so `gen.log` is secret material** (see 1.8).

The validated parameter sets are taken from the `gen.log` headers of `W/{vA64a,vA128a,vL128a}` and from `docs/ANF_WINDOW_DELIVERY.md:190-192`:

| n | \|C\| = \|D\| | s | slice_gates | \|A\| | wires |
|---|---|---|---|---|---|
| 64 | 1,000 | 384 | 1,280 | 2,832 | 128 |
| 128 | 3,000 | 896 | 2,560 | 7,920 | 256 |

### 1.2 The source circuit C

**Gate semantics.** C is a `CircuitSeq { gates: Vec<[u16;3]> }` (`src/circuit/g57.rs:12-17`). A g57 gate `[a, b, c]` (target, positive control, negative control) computes `a ^= b ∨ ¬c`, which is `a ^= 1 ⊕ c ⊕ bc`. It flips `a` unless `b=0` and `c=1` (`g57.rs:4-5`; `docs/local_mixing_documentation.md:694-701`). The scalar evaluator is `state ^ (c1 | !c2) << a`, with wire i stored as bit i (`src/circuit/evaluate.rs:13-17`).

**Generated C** (the default) is `random_circuit(n, m_C)` (`construct.rs:88`; `src/circuit/randomize.rs:6-47`). It asserts `3 ≤ n ≤ 65536` (`randomize.rs:15-19`). For each gate it draws three distinct pins in the order target, positive control, negative control, each by `fastrand::usize(..n)`, redrawing on a repeat (`:24-34`). It also rejects a gate identical to the one immediately before it, since the pair would cancel (`:36-42`). Each gate is therefore a uniform ordered triple of distinct wires, and no two adjacent gates are equal.

`prepare_source` (`construct.rs:62-90`) always calls `fastrand::seed(seed)` first, even when C is supplied. Its doc comment says this "preserves historical RNG behavior" (`:63`, `:70`). fastrand 2.3.0 is a thread-local generator with a 64-bit state (`Cargo.toml:56`, `Cargo.lock:359-360`).

**Supplied C.** If `TDP_SOURCE_C` names a file, its raw bytes are decoded with `CircuitSeq::from_bytes` (`mod.rs:190-201`). `prepare_source` then requires three things:

- the circuit is non-empty (`construct.rs:73-75`);
- `max_wire()+1 ≤ n` (`:76-79`);
- the gate count equals `m_C` exactly (`:80-85`).

On success it logs `[gen] source C LOADED from <path> (<g> gates, <w> wires)` (`mod.rs:202-208`). D is still generated independently. Pin distinctness is not checked on this path (see 1.8).

**Sidecar.** C is always written to `<out>.source_c.g57` with `c.repr()` (base-83, no trailing newline), and the log line is `[gen] wrote source C …` (`mod.rs:209-217`). The code comment notes that regenerating with the same seed reproduces C and the whole gadget bit for bit.

### 1.3 Formats at the stage-1 boundary

**g57 base-83 text** (`src/circuit/formats/g57.rs:57-198`). Each gate is written as three wire tokens followed by `;`.

- **Digits.** The alphabet is `0-9 a-z A-Z ! @ # $ % ^ & * ( ) - _ = + [ ] { } < > ?` (83 symbols, `:11-12`).
- **Wire tokens.** Wire `w` is written as `⌊w/83⌋` copies of `~` followed by the digit for `w mod 83` (`:95-101`). At n=128, for example, the gate `[100, 5, 90]` is written `~h5~7;`.
- **Parser.** It trims ASCII whitespace at the two ends only (`:138-152`) and accepts an unterminated final gate. It panics in three cases (`:34-43`, `:127-198`):
  - an invalid byte, including whitespace between gates such as a newline (`:189`);
  - a non-empty gate whose wire count is not 3;
  - a dangling `~`.

  Empty `;` segments (`;;` or a leading `;`) are skipped silently (`:181-187`). A wire value above 65535 is silently truncated by the `as u16` cast (`:175`).

**XGate lift.** Stage 1 outputs `Circuit { gates: Vec<XGate>, num_wires: 2n }` (`src/circuit/types.rs:4-8`). An `XGate {target, comp, ctrls}` applies `target ^= comp ⊕ ∧(literals)`, where an empty AND is 1. Its invariant is that controls are sorted by wire, there is at most one literal per wire, and the target is never a control (`src/circuit/xgate.rs:14-20`). `XGate::from_g57([a,x,y])` produces `comp=1` with monomial `¬x ∧ y`, controls emitted in wire order (`xgate.rs:65-87`). If `x==y` it becomes an unconditional X gate (`:67-70`). Every XGate is an involution, so a reversed gate list is the inverse circuit.

**mpmct1 text** (`src/circuit/formats.rs:3-5`, writer `:20-50`). This is the gate-list format of every stage. The header is `mpmct1 <num_wires> <num_gates>`. Each gate line is `<target> <comp> <k> (<wire> <pol>)×k`, with `pol=1` a positive literal and `pol=0` a negated one. The reader rejects a control on the gate's own target, a duplicate control wire, and a mismatch between the header's gate count and the body (`formats.rs:302-375`).

Worked example at n=3 (x = wires 0..2, y = wires 3..5):

| gate | meaning | mpmct1 line |
|---|---|---|
| g57 `[0,1,2]` | `x0 ^= x1 ∨ ¬x2` | `0 1 2 1 0 2 1` |
| g57 `[2,1,0]` | `x2 ^= x1 ∨ ¬x0` | `2 1 2 0 1 1 0` |
| slice CNOT | `x1 ^= y2` | `1 0 1 5 1` |
| slice CCNOT | `x0 ^= x2 ∧ y1` | `0 0 2 2 1 4 1` |
| N CNOT | `y1 ^= x1` | `4 0 1 1 1` |

The g57 positive control is written with polarity 0 and the negative control with polarity 1, because the complement flag inverts the monomial.

### 1.4 Construction of the classic sandwich

`construct_seeded_sandwich(C, n, m_D, s, variant, sandwich_seed)` (`construct.rs:94-105`) creates its own stream, `StdRng::seed_from_u64(sandwich_seed ^ 0x5150_1CED)` (rand 0.9.2). It then calls `sliced_sandwich_cnot`, which draws D (`:268-278`) and builds the circuit in `sliced_sandwich_build` (`:303-411`). The builder asserts `n ≥ 3`, `2n ≤ u16::MAX`, and that every wire of C and D is `< n` (`:311-322`).

With x the low half (wires 0..n) and y the high half (wires n..2n), the layout is (`construct.rs:218-226`):

```text
A = [ C ⧢ S1 ] ; N ; [ D ⧢ S2 ]        (⧢ = order-preserving random merge)
```

| block | gates | shape | writes | reads |
|---|---|---|---|---|
| C | m_C | g57 lift, `comp=1`, literals `(x_b,¬)` and `(x_c,+)` (`:330`) | x | x |
| S1 | s | ~1/3 CNOT `x_t ^= y_r`; otherwise CCNOT `x_t ^= x_j ∧ y_r`, j≠t (`:107-144`) | x | exactly one y wire (positive), plus one x wire for CCNOTs |
| N | n | `y_i ^= x_i`, i = 0..n−1 (`:338-345`) | y_i | x_i |
| D | m_D | random g57: a uniform, x≠a, y∉{a,x} (`:158-170`) | x | x |
| S2 | s | independent draw of the S1 shape (`:351`) | x | as S1 |

In detail:

- **Slice blocks.** All slice gates have `comp=0` and only positive literals, and each reads exactly one y wire. Each gate, and so each block, is therefore the identity whenever y = 0 (`:107-111`).
- **D.** D has the same distribution as C ("same design as the C (source) block", `:158-159`). Unlike C, it has no adjacent-duplicate rejection.
- **The high half.** It is a pure answer register: the N CNOTs are the only gates that write it, and it is read only as slice-gate controls (`:236`).

**Interleaving.** `random_interleave` (`construct.rs:172-194`) takes the next computation gate with probability `rem_c/(rem_c+rem_s)`. This samples uniformly among all order-preserving merges. The computation's order is a hard constraint. The slice block's order does not matter on the zero slice.

**N-float.** Before the float, N is a contiguous column at the C|N|D seam, which the code calls "the sandwich's most structure-revealing part" (`:359-360`). The float works as follows (`:377-399`):

1. Each N CNOT is assigned an independent random direction (`random_bool(0.5)`).
2. The processing order of the n floaters is shuffled.
3. Each gate in turn slides by adjacent swaps (`float_extremal`, `:196-213`) until its neighbour `collides` or it reaches the end of the circuit. The recorded positions of the other floaters are updated after each move.

One pass reaches the fixpoint, because the blockers are static and the floaters commute with each other (`:381-384`).

**The collision rule** (`xgate.rs:108-149`). Two gates collide if and only if one reads the other's target. The one exception is two `comp=0` gates with opposite polarities on a shared control, which are treated as not colliding. A `comp=1` (g57) gate never gets this exemption. The same rule governs every commutation move in stage 3 (§3.5). Reading `x_i` does not block `y_i ^= x_i`. The only things that stop it are:

- a gate that writes `x_i` (from C, D, or a slice block);
- a slice gate that reads `y_i`;
- the end of the circuit.

Every hop is a commuting swap, so the function of A is unchanged. The per-gate slice structure is also unchanged, as shown in 1.5. The column is tracked by its insertion range, not by target half, so classic seeds still reproduce bit for bit (`:372-376`).

The unit test asserts only that the final span of the column exceeds n (`tests/unit/stages/sandwich/construct.rs:91-139`). The three validation sandwiches were recounted read-only for this document (`W/*/tdp.mpmct1.sandwich.mpmct1`):

- the N CNOTs are the gates with target ≥ n, which is exact for classic (`:110-118`);
- a gate's travel is |(non-N gates before it) − (m_C + s)|;
- the span is the last N position minus the first.

| run | median travel | max travel | N CNOTs that did not move | left / right | final column span |
|---|---|---|---|---|---|
| vA64a (n=64) | 38.5 | 194 | 1 | 33 / 30 | 432 (positions 1190–1622 of 2,832) |
| vA128a (n=128) | 56.5 | 571 | 1 | 56 / 71 | 1,024 (positions 3570–4594 of 7,920) |
| vL128a (n=128) | 66.5 | 385 | 0 | 52 / 76 | 883 (positions 3525–4408 of 7,920) |

Left and right moves were close to balanced at n=64. At n=128 they leaned right: 71 and 76 of 128 moved right. The float widens the seam into a band of about 430 gates at n=64 and 880–1,025 gates at n=128, which is 11–15% of |A|. It does not spread the column through the whole circuit.

*Derived* estimate of the blocker density at n=128, per block, for each wire i:

- about 23 gates that write `x_i`, from the block's computation (C in block 1, D in block 2; m/n = 3000/128 ≈ 23.4);
- about 7 slice gates that target `x_i` (s/n = 7);
- about 7 slice gates that read `y_i`.

That is roughly 37 blockers in 3,896 gates, or one every ~105 gates. It agrees with the measured mean travel at n=128 (94.5 for vA128a, 105.5 for vL128a).

**RNG consumption order** on the sandwich stream (`:276`, `:331-332`, `:351`, `:357`, `:377-380`):

1. D
2. S1
3. merge(C, S1)
4. S2
5. merge(D, S2)
6. the n float directions
7. the float shuffle

**Size.** `|A| = m_C + m_D + 2s + n` (computed the same way at `mod.rs:142-146`). This gives 2,832 at n=64 and 7,920 at n=128, which matches the logs.

### 1.5 The slice contract

**Forward, on y = 0** (`construct.rs:228-236`; `docs/local_mixing_documentation.md:1307-1325`):

```text
(x, 0) → (C(x), 0) → (C(x), C(x)) → (junk, C(x)),    so A(x,0) = (junk, C(x)).
```

S1 is dead throughout. S2 is live after N, because y = C(x), but it only targets the low half, which is junk by then. The float preserves this gate by gate, not just functionally:

- `N_k = (y_k ^= x_k)` cannot pass a slice gate that reads `y_k`. That gate reads N_k's target, both gates are `comp=0`, and all their shared literals are positive, so the exemption does not apply. Every S1 gate that reads `y_k` therefore stays before `N_k` and sees `y_k = 0`.
- `N_k` cannot pass a gate that writes `x_k`, so it still copies the final block-1 value `C(x)_k`.

**Reverse, on the same slice** (`construct.rs:231-233`; `docs/local_mixing_documentation.md:1334-1346`). Every XGate is an involution, so the reversed gate list is A⁻¹ (`src/stages/preprocessing/verify.rs:69-70`):

```text
(p, 0) → (D⁻¹(p), 0) → (D⁻¹(p), D⁻¹(p)) → (junk, D⁻¹(p)),    so A⁻¹(p,0) = (junk, D⁻¹(p)).
```

Here S2 is dead, and S1 fires but only junks the low half. The same blocking argument applies in reverse. Every S2 gate that reads `y_j` stays after `N_j` in A, so in A⁻¹ it runs before `N_j` and sees `y_j = 0`.

**Why the layout is this way.** Both ports use the same slice (y = 0, data on the low half) and the same readout register (the high half). The code comment states the aim: "neither direction hands out its computation on a wrong slice, and neither reveals the other's function in the clear" (`construct.rs:233-235`). Reversing the published circuit on the slice gives the random D⁻¹, not C⁻¹. D is given C's design and size "so that later mixing has two similar computations to work with" (`docs/local_mixing_documentation.md:1347-1350`).

**Off the slice.** The high-half output is `y ⊕ C'(x,y)`, where C' is the computation as disturbed by S1 (`docs/local_mixing_documentation.md:1389-1392`). There is no proof that every off-slice input avoids producing a clean C(x). The unit test checks only that some input differs (`tests/unit/stages/sandwich/construct.rs:78-87`).

**Scope.** Slicing defines the preimage problem and removes reversal as an attack. It is not a defense against affine reconstruction (`docs/local_mixing_documentation.md:1402-1405`); embedded masking (stage 2) is.

**Unit-tested invariants.** Each test uses a single small width (n = 3, 4 or 6), 8 seeds, and a fixed toy C: 4 or 2 hand-picked gates at n=3, and a cyclic 24-gate pattern at n=6. The evaluation checks are exhaustive over inputs at that width. The column test is structural.

- **Slice blocks (n=4, exhaustive over the low half).** They have `comp=0`, all-positive controls and at least one read-half control, and they are the identity when the read half is zero (`tests/unit/stages/sandwich/construct.rs:5-26`).
- **Forward map (n=3, exhaustive).** A(x,0) carries C(x) on the high half, and A is a bijection on {0,1}^{2n} (`:55-89`).
- **Column (n=6, structural).** The classic N column is exactly the set of gates targeting ≥ n, each is `y_i ^= x_i`, and the span is > n (`:91-139`, column identity at `:110-118`).
- **Reverse map (n=3, exhaustive over p).** The reversed circuit's high half is a bijection on (p,0) (`:141-165`). This test checks bijectivity, not equality with D⁻¹.

Nothing else checks the classic D⁻¹ port either. The gadget-level reverse check (`verify.rs:66-92`) proves only that the reversed G2 matches the reversed A on wires n..2n. D is never built or evaluated there. So the claim that A⁻¹(p,0) carries D⁻¹(p) in the classic variant rests on the gate-by-gate argument above and is not tested. Only the balanced variant has a D⁻¹ equality test (`tests/unit/stages/sandwich/construct.rs:204-238`). The generator's message "D^-1 emerges under backward evaluation" (`mod.rs:361-364`) says more than is checked (see 1.8).

**Balanced variant (not used).** `SandwichVariant::Balanced` (parsed from `balanced` or `mirror`, `construct.rs:21-49`) changes three things: N becomes `x_i ^= y_i`, D moves to wires n..2n, and S2 is mirrored (`:238-263`, `:340-357`). Forward, it gives A(x,0) = (C(x), junk). Its inverse is sliced at x = 0. The generator states that it "does not provide the classic reverse-port guarantee" (`mod.rs:17-18`), and its gadget-level reverse verification is skipped as a TODO (`mod.rs:365-368`; `verify.rs:71`). TDP runs use classic (`RESULTS.md:617-621`). Despite the name, this variant has nothing to do with stage 2's balanced masks.

### 1.6 Seeds and random streams

| seed | stream | drives |
|---|---|---|
| `seed` | `fastrand::seed(seed)` (`construct.rs:70`); verify stream `StdRng(seed ^ 0xA11CE)` (`verify.rs:30`) | C; verification samples |
| `sandwich_seed` | `StdRng(sandwich_seed ^ 0x5150_1CED)` (`construct.rs:103`) | D, S1, S2, both merges, N-float |
| `gadget_seed` | `StdRng(gadget_seed)` and further derived streams; the full list is the stream table in §2a.2 | all of stage 2 |

These are domain-separated PRNG streams, not independent secrets. With `val_chain.sh`'s single 31-bit draw, one value determines every seeded stream of stages 1 and 2 (see 1.8, open issue 1).

To get a "fresh gadgetization of the identical sandwich" (`mod.rs:234-235`), keep both `seed` and `sandwich_seed` fixed and change `gadget_seed`. The code comment mentions only `seed`. It predates the `sandwich_seed` split and holds as written only when positional 10 is omitted, so that it defaults to `seed` (`:115`).

An empirical check found no sign that the streams are related in practice. The same C (via `TDP_SOURCE_C`) with fresh sandwich and gadget seeds gave a stage-2 output with 0 of 4,096 hot in `qc_scan` and a null ridge against the reference (`RESULTS.md:161-167`). Both tests are defined in §5.9.

All three seeds are `u64`. The `val_chain.sh` driver passes a single 31-bit draw for all three (`val_chain.sh:11`, `:16`). The managed driver `scripts/tdp_gen.sh` handles seeds differently (Appendix A). Seed handling across the whole pipeline is summarized in §6.2.

### 1.7 Outputs, verification and handoff

**Files written before stage 2 runs:**

- `<out>.source_c.g57`, the secret C;
- `<out>.sandwich.mpmct1`, A on 2n wires (`mod.rs:209-232`). It is logged as `[gen] classic sliced sandwich: <|A|> gates, <2n> wires (payload on wires n..2n on the zero slice)`.

**Handoff.** Stage 2 starts with `preprocess_sandwich(&sandwich, n, slice_gates, variant, &options, &mut rng)` (`mod.rs:292`).

- `slice_gates` is a stage-2 parameter. Each outer zero-slice guard gets `gc = max(slice_gates, band width)` gates (`src/stages/preprocessing/construct.rs:57`). The logs show 2,560 per side at n=128.
- `EMBEDDED_MASKING_SEED_AVOID_C1` depends on the stage-1 gate shapes. It treats every gate with `comp=0` and a positive control ≥ n as a slice gate that is dead on the honest slice (`preprocessing/construct.rs:89-110`; §2a.5).

**Verification** (`verify_payload`, `verify.rs:10-94`). This runs after stage 2, as two checks of 256 bit-sliced samples each (4 rounds of 64 lanes); details are in §2a.7.

- **Forward check** (`:43-65`). It puts random values on all 2n sandwich wires, including y, sets the band to zero, and requires G2 = A on wires n..2n. This is stronger than the zero-slice contract: with the band at zero, G2 must reproduce A's high half for every y.
- **Reverse check** (`:66-92`). It puts random values on wires 0..n, zeros everywhere else, and requires reversed G2 = reversed A on n..2n. This is the gadget-level image of the D⁻¹ port. It compares against reversed A, never against D⁻¹ itself (see 1.5).

The resulting gadget-level contract is G2(x, 0ⁿ, 0²ⁿ) = (junk, C(x), junk), and G2⁻¹(p, 0, 0) carries D⁻¹(p) on the high sandwich half (`docs/gadgetization.md:271-291`). The D⁻¹ part combines the tested equality with reversed A and the untested sandwich-level argument in 1.5. A failed check panics before G2 is written (`mod.rs:357`, `:371`), so the sidecars are left behind with no gadget.

**Stage 4 depends on the sidecar.** `repair.sh` scores the stage-4 output against C by running `hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 … --exposure --g-cuts` (`W/repair.sh:17-21`). The target is a C-view visible count ≤ n, which it pursues in two capped loops (§4.5):

- up to 4 `order_repair` fix iterations (`:22-26`);
- then up to 4 `merge_bounds2`/`anf_split2` fallback iterations (`:28-33`).

Reaching the floor is not enforced. The script continues to `anf_compose` and `link_census` without a status check (`:35-39`). The floor of n is the output values themselves, which are trivially visible because C(x) sits on the answer register (`STAGE2_LINKABILITY.md:29-33`). `anf_split2.py` also runs `hmap_affine` on the sidecar, and `val_chain.sh:26-27` uses it for the stage-2 span check (§5.3).

`source_c.g57`, `sandwich.mpmct1`, `gen.log` and `seed.txt` (`val_chain.sh:14`) are plaintext secrets. They must survive until stage 4 finishes and must never ship (§6.8).

### 1.8 Measured profile and open issues

**Measured profile.** The notebook measured the stage-1 sandwich at n=64 (m=1000, s=384) (`RESULTS.md:611-627`):

- **Positionally flat.** The low-half touch fraction is about 0.90 in every bin. The per-wire position standard deviation is 0.289 on every wire. That is 1/√12, the value for a position spread uniformly over the circuit (`:622-623`).
- **Volume and role imbalance.** Median touches are 115 per low wire against 13 per high wire. The target fraction is 0.38 against 0.08.

These values follow from the construction (*derived*):

| | formula | value at n=64 | target fraction |
|---|---|---|---|
| high wire touches | `1 + 2s/n` | 13 | 1/13 ≈ 0.077 |
| low wire touches | `(3(m_C+m_D) + (5/3)·2s + n)/n` | ≈ 114.75 | `(m_C+m_D+2s)/7344` ≈ 0.377 |

**Open issues** (as found on 2026-10-07, against `val_chain.sh` and the pre-port managed driver). Items 1, 2, 5 and 6 are resolved in the managed driver on `ran/stage4-rust` (status notes in brackets); `val_chain.sh` itself, the historic reference, is unchanged. Items 3, 4, 7 and 8 are still open.

1. **Seed entropy.** C is a deterministic function of a `u64` seed. `val_chain.sh` passes the same 31-bit draw as all three seeds, so enumerating that one value recovers C, D, S1/S2, both merges, the float and every seeded stage-2 stream (masks, guards, band seed and reseed). A candidate C can be tested with a few slice evaluations of the published deliverable. Even a full-width seed caps C at 64 bits of entropy (the `fastrand` state). The binary's default `seed=1` is also accepted silently. [Managed driver: two independent seeds (gen, mix) of up to 63 bits drawn from the OS CSPRNG (`seed_bits=63`), kept in the private `$RUN/SEEDS`; `val_chain.sh` still draws one 31-bit gen seed. The 64-bit `fastrand` cap on C remains.]
2. **Seeds in logs and on the command line.** The `[gen]` header prints all three seeds (`mod.rs:184-188`). They land in `gen.log` under `val_chain.sh` and in `$RUN/stage12.log` under `tdp_gen.sh`. Both drivers also pass the seeds as argv (`val_chain.sh:16`, plus the stage-3 seed at `:23`; `tdp_gen.sh:434-435`), where other local users can read them with `ps`. `val_chain.sh` sets no umask: in `vA128a`, `gen.log`, `seed.txt` and `tdp.mpmct1.source_c.g57` are mode 644. [Managed driver: no seed is on argv or in a log; `tdp_gen.sh` hands the three gen seeds over as `@/dev/fd/3..5` here-strings and the mix seed as `--seed-file /dev/fd/3` (`scripts/tdp_gen.sh`, stage 1+2 and stage 3 blocks), gen prints them as `(file)`, and the run directory is written under `umask 077`.]
3. **Unchecked pins on supplied C.** `TDP_SOURCE_C` does not check that pins are distinct. A target that equals a control produces an XGate that breaks the invariant at `xgate.rs:18` and is not reversible. A gate with `x==y` silently becomes a NOT gate, which the masked compute then lowers incorrectly (§2b.11, defect 1).
4. **Dead parameter.** `rg_freq` is parsed and printed but never read. Removing it would shift every positional argument after it.
5. **Canonical |C|.** The pipeline runs 1000 at n=64 and 3000 at n=128. The library convention `sandwich_default_m(n) = max(n, round(n·log₂²n))` (`construct.rs:56-60`) gives 6272 at n=128. In Rust it is called only by the nonlinear291 capacity check (`src/tdp/config/validate.rs:160-165`). The managed driver computes its own default, `round(n·log₂²n)` with no floor, in Python (`scripts/tdp_gen.sh:348-357`). That default is documented as `source.gates` in `docs/tdp_pipeline.md:84`. [Managed driver: the default is now the validated table, 1000 at n=64 and 3000 at n=128 (`validated_gates` in `scripts/tdp_gen.sh`, `VALIDATED_SOURCE_GATES` in Rust; `source_gates_rule=validated`); other n need `source.gates` or the opt-in `source.gates_rule = "formula"`, which is recorded as an unvalidated size.]
6. **Managed path cannot run the current configuration.** The managed config rejects `shuffling_return_home=false` whenever shuffling is on (`src/tdp/config/validate.rs:118-124`; shuffling is either 0 or at least 8, `:111-117`). `val_chain.sh` is gitignored (`.gitignore:3`), is not in the `scripts/windowed_delivery/` snapshot, and still passes `--p-comp`/`--p-any`. See §6.10. [Resolved: presets L8 (the default) and A express the recommended stage-2 environment exactly, and `val_chain.sh` is tracked as `scripts/windowed_delivery/waist/val_chain.sh` (without the inert `--p-comp`/`--p-any`, the only difference from the run-tree copy).]
7. **SEED_AVOID_C1 classification.** It treats S2 gates as dead, but they are live on the forward pass. This is harmless, because by block 2 every low wire has almost certainly already been written by C: at n=128, a given wire escapes all 3,000 C targets with probability about e^−23.5. §2a.5 confirms this from the code.
8. **No classic D⁻¹ test.** The classic reverse port is backed only by the argument in 1.5 and by the gadget-vs-A reverse check. A classic analogue of the balanced D⁻¹ test (`tests/unit/stages/sandwich/construct.rs:204-238`, building A with `sliced_sandwich_with_d` and checking that the high half of A⁻¹(p,0) equals D⁻¹(p)) would close the gap. The `[gen]` reverse-verify message (`mod.rs:361-364`) should then say what is actually checked.


## Stage 2a: preprocessing structure, band seeding and slice guards

Stage 2 (preprocessing) takes the stage-1 sliced sandwich A, which has 2n wires, and produces the 4n-wire gadget G2, which computes the same payload on the public slice. Stage 2 has three parts:

- **2a (this section)** covers the frame around the masked computation: the entry point and its parameters, the five-module layout, the band seed derived from the input and the band reseed, the first-layer pair exclusion (`SEED_AVOID_C1`), the two zero-slice guards, and the verification that closes the stage.
- **2b** covers the masked computation itself (LGIs, quad fire, brackets, rerand bursts, `PREMASK`, chains).
- **2c** covers carrier migration, the last transform inside the masked computation.

**Where the numbers come from.** Gate counts, gate censuses and log lines come from the two validation runs `W/vL64a/` and `W/vL128a/`. Their `seed.txt` files show that both used the recommended configuration with bounded chains (config L8). The leakage and AUC figures were measured earlier, on configurations that predate the fringe fixes or chains. They come from `RESULTS.md`, and each is tagged below with its run.

### 2a.1 Conventions

**Gate model and file format.** Every gate is an `XGate`, `target ^= comp ⊕ AND(lits)`, in mpmct1 text (§1.3). The bit-sliced evaluator is at `src/stages/preprocessing/verify.rs:20-29`. Three constructors matter here:

- `XGate::cnot(t, c)` is `t ^= c` (`src/circuit/xgate.rs:61-63`).
- `XGate::conj(t, lits)` has comp = 0 and sorted literals (`xgate.rs:26-49`).
- `XGate::from_g57([w, x, y])` is `w ^= 1 ⊕ ¬x·y`, which is comp = 1 with literals `(x, false), (y, true)`. When x = y it degenerates to an X gate (`xgate.rs:65-87`).

For example, the n = 64 gadget `vL64a/tdp.mpmct1` opens with:

- `4 0 1 180 1`, an opening-guard CNOT `x4 ^= b180`;
- `29 0 2 103 1 173 1`, which is `x29 ^= y103 ∧ b173`.

**Widths.** Two different meanings of "n" appear in the code:

- In `construct.rs` (`src/stages/preprocessing/construct.rs`), `n = source_wires` is the width of C, and `np = sandwich.num_wires = 2n` (`construct.rs:24,55`).
- The masked compute and the guards are called with data width `np`.

The band is `np..np+r`, where `r = 0` means `r = np` (`EM:1178`). `production()` sets `r = 0` (`EM:199`), and `gen_sandwich_gadget` exposes no override. The band therefore has 2n wires and G2 has 4n (`construct.rs:164`). For the classic variant at n = 128:

| wires | input on the public slice | output on the public slice |
|---|---|---|
| 0..128 (low half, x) | x | junk |
| 128..256 (high half, y) | 0 | C(x) |
| 256..512 (band) | 0 | junk |

Under carrier migration (§2c), data and band are **roles**, not fixed wires. Inside the compute a data role can sit on any physical wire; at the ports, data roles are home.

**Why a slice.** G2 is reversible on all 4n inputs, and nothing inside it forces y or the band to zero. The challenge fixes them, and that defines the public slice (`docs/local_mixing_documentation.md:1370-1373`). Slicing "defines the preimage problem and prevents the immediate reverse-circuit attack"; it is explicitly not the defence against affine reconstruction (`docs/local_mixing_documentation.md:1402-1403`). The guards described below make a wrong slice, meaning a nonzero band, scramble the data instead of computing anything useful.

### 2a.2 Entry point and parameters

`gen_sandwich_gadget` builds stages 1 and 2 in one process. Its positional arguments are in the table in §1.1. Three of them matter to stage 2:

- `slice_gates` (argument 7, default `10·2n` = 20n) is the guard size per side.
- `gadget_seed` (argument 9) drives all stage-2 construction randomness.
- `rg_freq` (argument 6) is parsed at `mod.rs:107` and printed at `:185`, but otherwise unused.

`seed` fixes C, and `sandwich_seed` drives D, the slice blocks and the N float (`mod.rs:24-27`). All stage-2 construction randomness derives from `gadget_seed`. The verifier is the exception: it draws its samples from `seed`. This is the canonical stream table for stage 2:

| consumer | RNG | reference |
|---|---|---|
| both guards (open drawn first, then close) | `StdRng(gadget_seed ^ 0x6AD6_E75E)` | `mod.rs:236` |
| masked compute (including the `PREMASK` order) | `StdRng(gadget_seed)` | `EM:1256` |
| carrier-migration routing (shuffling > 0, as in the recommended dose 8) | `StdRng(xxh3_64_with_seed(b"embedded-masking/preprocessing-shuffling/v1", gadget_seed))` | `preprocessing_shuffling.rs:492-494`, called with `p.seed` at `EM:2682-2690` |
| JIT re-masking (only when `EMBEDDED_MASKING_JIT` > 0; off in the recommended configuration) | `StdRng(xxh3_64_with_seed(b".../preprocessing-shuffling/jit/v1", route_seed))` | `preprocessing_shuffling.rs:275-278, 510` |
| ancilla reserve (only when `EMBEDDED_MASKING_ANC_RESERVE` > 0; off by default) | `StdRng(gadget_seed ^ 0xA9C1_11A5)` | `EM:1213-1219` |
| band seed (module 2) | `StdRng(gadget_seed ^ 0x5EED_B00C)` | `construct.rs:111-122` |
| band reseed (module 4) | `StdRng(gadget_seed ^ 0xB00C_5EED)` | `construct.rs:123-139` |
| verification | `StdRng(seed ^ 0xA11CE)` (the C seed) | `verify.rs:30`, called with `seed` at `mod.rs:357` |

Any `BV5_*` or `PROD_*` environment variable aborts the run (`mod.rs:151-170`). The shuffling controls are validated before any file is written (`parse_shuffling_controls`, `mod.rs:42-93`; details in §2c.1).

The parameters start from `EmbeddedMaskingParams::managed_tdp(gadget_seed, n)`, which is `production()` with `active_wires = n` (`EM:183-217`, applied at `mod.rs:248-273`). The full stage-2 knob table is in §2b.2. The knobs relevant to this section:

| knob | default | effect |
|---|---|---|
| `k`, `max_open`, `min_open` | 2, 3, 2 | mask shape (§2b); pinned by `tests/unit/programs/gen_sandwich_gadget.rs:77-89` |
| `r` | 0 → `np` = 2n | band width |
| `active_wires` | n | restricts burst data sources to the low half |
| `EMBEDDED_MASKING_SEED_NL` → `seed_fill_nl` | 4 | rich seed; 0 = legacy thin seed (`mod.rs:274-290`) |
| `EMBEDDED_MASKING_BAL_SEED` → `balanced_seed` | `params.balanced` = true | read only by the legacy thin seed |
| `EMBEDDED_MASKING_SEED_AVOID_C1` | off | first-layer pair exclusion, read directly at `construct.rs:89` |
| `SLICE_GUARD_LEGACY` | off | width-3 legacy guard (`construct.rs:67`) |

The recommended invocation is `W/val_chain.sh:13-16` (shown in §1.1 and in the Overview). Dropping the two `CHAIN` variables gives config A. Today this configuration can only be run by direct invocation, because the managed runner scrubs `PREMASK`, `SEED_AVOID_C1` and `CHAIN` and its validator rejects persistent migration (§6.10; "no managed preset yet", `STAGE2_LINKABILITY.md:182`).

### 2a.3 The five modules

`preprocess_sandwich` (`construct.rs:16-23`) emits the modules in this order (`construct.rs:156-160`):

```
open guard ; band seed ; masked compute ; band reseed ; close guard
```

| # | module | writes | reads | default size |
|---|---|---|---|---|
| 1 | opening guard | low half; band (temporarily) | data, band | `gc = max(slice_gates, 2n)` = 20n |
| 2 | band seed | band | x (0..n), earlier band wires | ≈ 22n |
| 3 | masked compute | data, band | data, band | §2b, §2c |
| 4 | band reseed | band | low half, earlier band wires | ≈ 22n |
| 5 | closing guard | junk half; band (temporarily) | data, band | `gc` |

The modules are built in a different order from the one in which they are emitted:

1. The masked compute is built first (`construct.rs:33-38`), because the guards and seeds need its band width `r_used`.
2. The code asserts that `final_layout[..2n]` is the identity: "data roles must be physically home after preprocessing" (`construct.rs:42-48`). Persistent migration meets this with 3-CNOT swaps at the end of the stream (§2c.7).
3. The two guards are drawn, open first and then close.
4. If `SEED_AVOID_C1` is on, the first-layer pairs are collected.
5. The seed and reseed are generated.

Both guards have exactly `gc` gates, which is why the log says "each side".

**Why this order.**

- The opening guard comes before the seed so that it sees the band at its port value 0. On the public slice it is therefore the identity (`construct.rs:49-54`).
- The compute needs a seeded band. Every mask term (`1 ⊕ B_y ⊕ B_x·B_y ⊕ B_z` for balanced masks) is a function of band wires, so on a zero band the masks would be constants.
- The seed must come from the input. The slice pins the band to 0, so on the slice the only entropy is the secret-carrying input: "pinnable ancilla randomness collapses to slice constants" (`docs/windowed_delivery_design.md:152-154`).
- The reseed is a separate module, not the inverse of the seed. Because "the band is junk at BOTH ports", it draws a different seed. The compute's rerand bursts do not replace it (`construct.rs:123-129`).

**Reverse direction.** Backward evaluation on `(a, 0, 0)` goes as follows:

1. close⁻¹ sees a zero band and is the identity;
2. reseed⁻¹ fills the band from the low half, which holds `a`;
3. compute⁻¹ then runs on that band.

So module 4 plays the band-seed role for `D⁻¹`, but the mirror is not exact. *Derived from the code, not measured:*

- The reseed is emitted one band wire at a time, in increasing index, and the products for wire `j` may read band wires `i < j`.
- In reverse, wire `j` is processed while every `i < j` still holds 0. A positive band literal then kills its product, a negative one reduces it to a single data literal, and a band×band product becomes a constant.
- Backward, each band wire is therefore a function of `a` of degree at most 2, with no cascade. Each wire is still individually balanced, by the pivot argument below.

### 2a.4 Band seed: `seed_band_rich`

`seed_band_rich(np, r, active_wires, seed, fill_nl, avoid)` (`EM:951-1056`) is the rich seed of punchlist #5. It was restored from the pre-reorganisation `emit_band_fill_nl_pivots_src` (commit `5f386381`, `src/preprocessing/gadgets.rs`), which had been "measured leakage-neutral" (`EM:942-946`). Data sources are `0..active_wires`, that is, the input x. Each band wire `B_j`, for j = 0..r−1 in order, is built in three steps:

1. **Pivot.** `B_j ^= x_p`, with `p` uniform in 0..n.
2. **Linear part.** `lin_w = 1 + U[0, min(n−1, 7))` distinct data wires other than `p`, one CNOT each (`:998-1009`). This is uniform in 1..7, with mean 4.
3. **Products.** `fill_nl = 4` products of two literals (`:1010-1052`).
   - Each source is drawn as follows. With probability 1/2 (when the eligible set is non-empty) it is a random *eligible* earlier band wire, meaning one whose transitive data support excludes `p`. Otherwise it is a random data wire other than `p`.
   - The two sources must be distinct. If `avoid` contains `(min, max)` of the pair, both are redrawn (`:1030-1036`).
   - Polarities are uniform, and `emit_g57_form` (`:896-940`) emits the product as follows:

| literals (after a random swap) | gates | net effect on `B_j` |
|---|---|---|
| `(¬x, y)` | `g57[B_j,x,y]` | `⊕ 1 ⊕ ¬x·y` |
| `(x, ¬y)` | `g57[B_j,y,x]` | `⊕ 1 ⊕ x·¬y` |
| `(x, y)` | `g57[B_j,x,y]`; `CNOT(B_j,y)` | `⊕ 1 ⊕ x·y` |
| `(¬x, ¬y)` | `g57[B_j,x,y]`; `CNOT(B_j,x)` | `⊕ ¬x·¬y` |

The resulting band wire is:

`B_j = x_p ⊕ ⊕_{l∈L_j} x_l ⊕ ⊕_{k=1..4} (ℓ_{k,1} ∧ ℓ_{k,2}) ⊕ c_j`

Here the product literals range over data wires and eligible earlier band wires. The constant `c_j` only complements a junk wire (`:892-893`), and it does not affect balance (`:1038`). For example, a wire could be `b200 = x17 ⊕ x3 ⊕ x41 ⊕ ¬x5·x9 ⊕ b150·x22 ⊕ ¬b131·¬x60 ⊕ x8·x33 ⊕ c`. This example is illustrative; for it to be valid, `b150` and `b131` must not depend on `x17`.

**Balance.** The pivot `x_p` occurs nowhere else in `B_j`:

- the linear sources exclude it;
- data product sources exclude it;
- band product sources must be free of it.

Hence `B_j = x_p ⊕ f(x without x_p)`, which is balanced over uniform x. This holds for each wire individually; there is no joint claim. The private-pivot branch (`reserve = r + r/3 + 2 ≤ n`, `:971`) would make the map triangular, but it never applies in production: `r = 2n > n`, so pivots are drawn with replacement. In any case, 2n band wires that are functions of n bits cannot be jointly uniform.

**Shape and cost.**

- Seed gates write only band wires.
- They read only data wires and earlier band wires.
- Every gate is either a CNOT or a comp=1 g57, so the width is at most 2.
- The block is emitted contiguously and unshuffled.
- Because products can read earlier band wires, the degree is unbounded in principle.

The expected cost is 1 + 4 + 4 × 1.5 = 11 gates per band wire, about 22n in total:

| run | seed / reseed gates | 11r |
|---|---|---|
| n=64 (`vL64a/gen.log`) | 1,407 / 1,396 | 1,408 |
| n=128 (`vL128a/gen.log`) | 2,823 / 2,872 | 2,816 |

A census of the vL64a seed block (gates 1,281–2,687) gives 895 single-control CNOTs and 512 two-control comp=1 g57s (128 wires × 4 products). All of them target band wires.

**Why a rich seed.**

- RC's directive asks for no data/band distinction (`RESULTS.md:665`).
- The legacy seed `seed_band_mode` (`EM:861-889`) writes each band wire as `x_i ⊕ x_j` with 2 CNOTs. At n=64 that is 256 gates, against 1,280 per guard (`RESULTS.md:679`, run `s12_check`).
- The design note asked to "densify substantially" (`docs/windowed_delivery_design.md:157-160`).
- With the rich seed and persistent migration at dose 32 (not the recommended dose 8), the stage-2 data-vs-band touch AUC (§5.9; 0.5 means indistinguishable) went from 1.000/0.000 to 0.51/0.50 (`RESULTS.md:682`). This was run `W/s12_check`, 2026-10-05, before PREMASK, SEED_AVOID_C1 and chains existed, and it has not been re-measured at dose 8.
- After seeding, band wires depend on 47–52 of the 64 input bits (`RESULTS.md:1454-1455`). This was measured in the 2026-10-07 fringe diagnosis on n = 64 runs, before the fringe fixes.
- The seed reads only x. The y half is zero at seed time and remains the residual outlier (`RESULTS.md:688`, `s12_check`).

### 2a.5 `SEED_AVOID_C1`: no seed products on first-layer pairs

`EMBEDDED_MASKING_SEED_AVOID_C1` is off by default and recommended on. When it is set and not "0", `construct.rs:92-110` makes one pass over the sandwich, with `written[2n]` initially all false:

1. **Skip non-writes.** Skip the gate if `!comp` and some control `(c, true)` has `c ≥ n` (`:95-100`). A skipped gate does not mark its target as written.
   - In the classic variant, the matching gates are exactly the S1/S2 slice-block gates `t ^= y_r` and `t ^= t_j ∧ y_r` (`src/stages/sandwich/construct.rs:107-145`). They are the identity while y = 0.
   - In the balanced variant, the rule also matches the N-column CNOTs `x_i ^= y_i` (`stages/sandwich/construct.rs:339-344`) and the CCNOTs of the mirrored S2, whose `t_j` control lies on the high half. It does not match the mirrored S2's CNOTs `t ^= x_r`.
2. **Detect raw gates.** A gate is *raw* if its target and all of its controls are still unwritten.
3. **Record pairs.** A raw gate with exactly two controls contributes the pair `(min, max)` of its control wires, with polarity ignored.
4. **Mark the target written.**

"First layer" therefore means value-raw, not depth 1. This matches `W/seed_coincidence.py:1-17`, which applies the same test in C's own gate order. Only the forward seed receives the set; the reseed gets an empty one (`construct.rs:135`). Band sources are ≥ np, so only data-data pairs can ever match.

**Why it exists.** The evidence comes from the 2026-10-07 fringe diagnosis on n = 64 runs before the fringe fixes (for example `rw64_0b`, `RESULTS.md:1428-1455`):

- A raw C gate `a ^= g57(x, y)` creates `V = 1 ⊕ a ⊕ y ⊕ x·y`, a degree-2 function of three input bits.
- Every leaked witness of such a value was built from raw copies of `a`, `x` and `y`, plus one wire holding a product of the same raw pair (`RESULTS.md:1430-1440`).
- A seed product on that pair is "a ready-made copy of that gate's nonlinear term" (`EM:948-950`).
- In the `seed_coincidence.py` census, first-layer gates whose pair was also a seed pair leaked 26% of the time (8/31). Unmatched gates leaked 6.6% of the time (25/380) (`RESULTS.md:1441-1442`).
- These values are invisible in stage 2 itself and appear only after the mixer and packer reorder the gates (`RESULTS.md:1451-1453`). This recombination is the **fringe leak**.
- The companion fix `PREMASK` (`EM:1922-1947`; §2b.3) removes the raw input copies.

**The skip rule matters.** S1 is interleaved with C. The first version counted S1 gates as writes and left one overlapping pair in 3 of 4 arms. The corrected rule gave 0 overlaps on 4 of 4 seeds, with no measurable cost: 660–665k stage-2 gates against 661k without it. These were the fringe-fix arms `fx1`/`fx2`/`fx12`, which used K=64 retirement (`RESULTS.md:1463-1478`). **K=64 retirement** means `EMBEDDED_MASKING_RETIRE=64` (EM:1602-1609, 1620-1670): a band wire borrowed at least 64 times since its last write gets a fresh nonlinear value at the next box boundary. The lever has since been dropped because it raises L2 and creates interior C leaks (§2c.11). It is unrelated to the mask parameter K = 2.

In the classic variant the rule also skips S2 gates, which are live during D. This is harmless, because a D gate can only be raw if C never wrote any of its three wires (§1.8, issue 7). Stated precisely, the rule treats "comp=0 gates with a positive control on the y half" as non-writes, rather than identifying "dead" gates.

### 2a.6 Slice guards (modules 1 and 5)

Both guard generators share one contract (`src/stages/preprocessing/slice_guards.rs:8-28`, `:212-238`):

- The guard is the identity exactly when the band is 0, for any data. The band is wires `n_g..n_g+nondata`, with `n_g = np`.
- Every nonzero band perturbs the data.
- Data targets lie only in the junk half.

**Symmetric ports.**

- At the output port, the forward run arrives with a junk band, so the closing guard fires. It must not touch the payload half.
- The opening guard is the identity going forward. In backward evaluation, however, its inverse runs last, on a band that is junk by then. A high-half target would destroy `D⁻¹(a)`.
- Because both guards are confined to the junk half, the same artifact computes `C(x)` forward and `D⁻¹(a)` backward, each on its own zero slice.
- In the classic variant, both guards target `0..n`.
- In the balanced variant, the closing guard targets `n..2n` instead (`construct.rs:73-78`). The default is `slice_zero_feistel_guard_dims_high` (`slice_guards.rs:388-397`); the legacy `_high` generator and its rationale are at `:39-56`. Reverse verification for balanced is not implemented (TODO).

**Default guard: accumulate-and-inject** (`try_slice_zero_feistel_dims`, `slice_guards.rs:239-375`, punchlist #7; called "feistel" in the code). It is designed so that there is "NO gate wider than two controls", and so that the block "lives in the vocabulary db_mixing digests" (`:212-216`).

Preconditions (`:247-269`):

- `n_g ≥ 3`;
- `3 ≤ target_hi ≤ n_g` and `target_lo < target_hi`;
- the total wire count `n_g + nondata` fits in `u16`;
- `gate_count ≥ nondata`.

Parameters:

- `sources` are the data wires outside the target range (`:271-274`). For the classic variant at n = 128, these are wires 128..256, the payload half.
- `acc_size = clamp(nondata/8, 2, 16)`, which is 16 in production.
- Rounds are possible only if there are at least 2 sources and `nondata > acc_size`. Otherwise the whole guard is prologue.

The gate budget (`:282-304`) is allocated as follows:

1. `prologue = min(max(gc/3, nondata), gc)` and `round_floor = 2·acc_size + 8`.
2. The prologue absorbs `(gc − prologue) mod round_floor`.
3. While `left ≥ round_floor`, plan a round with a fresh accumulator set (the first `acc_size` wires of a reshuffled band order) and `inj = min(left − 2·acc_size, 6·acc_size)` injections.
4. Any remainder joins the prologue, so the total is exactly `gc`.

The gates are emitted in this order, with no final shuffle:

- **Prologue** (`:306-320`).
  - Slice controls `s = n_g + (i mod nondata)` are shuffled, so every band wire is read at least once.
  - Each target `d` is uniform in the target range.
  - When `i mod 3 = 0` the gate is `d ^= s`. Otherwise it is `d ^= c ∧ s`, with `c` any data wire other than `d` (`src/circuit/randomize.rs:142-153`).
- **Rounds** (`:322-356`). Each round has three phases:
  - ACCUMULATE: `t ^= d_i ∧ d_j` for each accumulator `t`, with `d_i ≠ d_j` drawn from `sources`;
  - INJECT, `inj` times: `d ^= t ∧ s`, where `s` is a band wire outside the current accumulator set (a *trigger*) and `d` is in the target range;
  - UNCOMPUTE: the accumulator gates replayed in reverse.

**Why it is the identity on the zero slice** (`:218-233`):

- Prologue gates carry a positive band literal, so they never fire on a zero band.
- Accumulators fire on any input, but they are restored exactly, because the guard never writes `sources`.
- Each trigger is either untouched or an earlier, already uncomputed accumulator, so it holds 0, and every injection is the identity.

On a nonzero band, an injection deposits `(s_t ⊕ d_i·d_j)·s` on a junk wire. These are degree-3 data×data×band terms, "the same algebraic degree the width-3 quads used to provide", but only where the source half is live. For the classic opening guard, the sources are the y half, and no earlier gate has written y. Take a wrong slice with y = 0 and band ≠ 0. There, every `d_i·d_j` is 0, so the opening guard's injections carry only band×band terms `s_t·s`, and its data dependence comes from the prologue's `d ^= c∧s` gates, each one data literal times band wires. The degree-3 terms act at the output port, in the closing guard (y = C(x)), and in backward evaluation, in the opening guard's inverse (y = D⁻¹(a)). The code comment's "disjoint accumulator sets per round" holds only within a round: each round redraws its set independently, and rounds are separated in time by the uncompute.

Sizes, derived from the budget code and confirmed by a census of `vL64a` (and of the first and last 20n gates of `vL128a`):

| | n = 64 (`gc` = 1,280) | n = 128 (`gc` = 2,560) |
|---|---|---|
| prologue | 440 (147 CNOT + 293 two-control) | 896 (299 + 597) |
| full rounds (16 acc + 96 inj + 16 uncompute) | 6 | 13 |
| short round | 1 × (16 + 40 + 16) | none |

In vL64a, each guard has 147 single-control and 909 (= 293 + 616) two-control gates on the low half, plus 224 accumulate/uncompute gates that target the band. No guard gate targets the high half. Neither validation output contains a gate wider than two controls. The arity censuses are vL64a {0: 1,946, 1: 117,245, 2: 660,004} and vL128a {0: 5,648, 1: 321,185, 2: 1,908,464}. The earlier n=128 run `W/n128` showed the same (`RESULTS.md:856-861`).

**Acceptance check.** If a draw fails the check, the generator draws again. After 1,000 rejected draws, `try_slice_zero_feistel_dims` returns `Err` (`:371-374`), and the wrapper panics (`:384-385`; `:395-396` for `_high`). The check itself depends on width:

- If `nondata + n_g ≤ 20`, the check is exhaustive (`:154-163`).
- Otherwise it is a spot check (`:169-210`): every single-hot band wire, 512 weight-2 slices and 512 random subsets, each tested against 64 random data lanes. The code calls it "a spot check, not a proof".

At production widths only the spot check runs, and it consumes guard randomness.

**Legacy guard** (`SLICE_GUARD_LEGACY=1`, `slice_guards.rs:58-148`). It mixes three gate shapes: `gc/3` gates `d ^= s`, width-3 gates `d ^= c1∧c2∧s`, and gates `d ^= c∧s`. The shape mix is generated at `:98-128` and finished with a full shuffle at `:129`. It is kept only for comparison arms.

**Unit test.** `tests/unit/stages/preprocessing/slice_guards.rs:102-137` runs at `n_g` = 8, nondata = 6 and 40 gates, with seeds 1, 7 and 99. It checks that the width is at most 2 and that targets are placed correctly. It checks exhaustively that every nonzero band slice disturbs the data. It checks the zero-slice data pass-through on only 4 sample inputs.

### 2a.7 Verification and log

`verify_payload` (`verify.rs:10-94`) uses 64-lane bit-sliced evaluation:

- **Forward.** 256 samples with all 2n sandwich wires random (so y is not pinned) and the band at 0. G2 must match the sandwich on the payload half: n..2n for classic, 0..n for balanced, all 2n wires for unguarded gadgets (`:43-49`).
- **Reverse (classic only).** 256 samples of the reversed gate lists on `(a, 0, 0)`, compared on n..2n, where `D⁻¹(a)` emerges. Every `XGate` is an involution, so the reversed list is the inverse circuit. For the balanced variant this check is skipped (TODO, `mod.rs:365-368`).
- **Not checked.** Auxiliary outputs are "intentionally unconstrained" (`verify.rs:7-9`), so band wires need not return to 0.

With guards, only the payload half is promised (`types.rs:86-95`). The files `<out>.source_c.g57` and `<out>.sandwich.mpmct1` are written before preprocessing. `<out>` is written only after verification passes (`mod.rs:212-232, 371`).

The stage-2 log lines from `vL128a/gen.log` (lines 5-8) are:

```
[chain] k=2: 316961 brackets chained
[gen] embedded-masking: 23302 atoms, band=256 wires, 2823 band-seed + 2872 band-reseed + 2560 slice-guard gates each side
[gen] compute: 2033637 fire gates + 190845 shuffling gates (63360 transfers, 0 skipped cuts, 255 ports restored, 0 jit remasks, 63360 drain/other) ; 15840 rerand bursts
[gen] diversified gadget: 2235297 gates, 512 wires
```

followed by `[gen] verify PASSED (256 bit-sliced samples, payload wires 128..256)` and `[gen] reverse verify PASSED (…: D^-1 emerges under backward evaluation)`. The `[chain]` line appears only when CHAIN is set (vL64a: 109,464 brackets). It goes to stderr from `EM:2703` and comes before the report lines; it appears in `gen.log` because `val_chain.sh` redirects `2>&1`. The next two lines come from `mod.rs:297-317`, and the gadget line from `mod.rs:350-354`. The first log line prints the seeds, so gen logs must be handled like seed files.

The totals check exactly: 2 × 2,560 + 2,823 + 2,033,637 + 190,845 + 2,872 = 2,235,297. The frame (modules 1, 2, 4 and 5) is therefore about 0.5% of stage 2. Two labels are misleading:

- "fire gates" is the whole compute length before carrier migration, including mask opens and closes and rerand bursts.
- "rerand bursts" counts burst *gates*: 990 slots × 16.

### 2a.8 Known gaps and stale text

**Stale comments and docs.**

- **[stale]** `construct.rs:51-53` says the band is n..2n; it is 2n..4n.
- **[stale]** `construct.rs:80` and `EM:850-853` say the compute only reads the band. In fact it writes the band (bursts) and borrows band wires as dirty ancillas (§2b.1). `EM:851-852` also describes the thin seed `x_i & !x_j`.
- **[stale]** `EM:126-127` says "The band seed becomes `x_i ⊕ x_j` (balanced)". That is the legacy thin seed, not the default rich seed.
- **[stale]** `types.rs:31` says "construction never reads process configuration". Yet construction reads `SLICE_GUARD_LEGACY` (`construct.rs:67`), `EMBEDDED_MASKING_SEED_AVOID_C1` (`construct.rs:89`), `EMBEDDED_MASKING_ANC_RESERVE` (`EM:1213`), `EMBEDDED_MASKING_CHAIN`/`CHAIN_LEN` (`EM:1610-1620`) and `EMBEDDED_MASKING_PREMASK` (`EM:1926`).
- **[stale]** `verify.rs:54` says "upper wires stay pinned to 0", but the loop at `:53-55` randomizes all 2n sandwich wires.
- **[stale]** `mod.rs:20-22` says RETURN_HOME is required (§1.1).
- **[stale]** Thin seed and reseed: `docs/gadgetization.md:61` and `:68` and `docs/local_mixing_documentation.md:1628-1632` still describe the thin `x_a ⊕ x_b` seed and the two-CNOT reseed.
- **[stale]** Three-control guard: `docs/gadgetization.md:167` and `docs/local_mixing_documentation.md:1547-1549` say the guards can contain three-control gates.
- **[stale]** Guard shuffling: `docs/local_mixing_documentation.md:1636-1637` describes a shuffled guard, but the default guard is unshuffled.
- **[stale]** The comment at `tests/unit/stages/preprocessing/slice_guards.rs:129-130` claims a band pass-through check that the test does not perform.

**Test coverage.**

- `seed_band_rich` is exercised only indirectly, through three `preprocess_sandwich` tests that always use an empty avoid set: the golden hash, port preservation and persistent migration (`tests/unit/stages/preprocessing/construct.rs:33`, `:111`, `:210`).
- The `SEED_AVOID_C1` pair collection and redraw path is exercised by no test. No test sets the variable, and a grep of `tests/` for `AVOID_C1`, `first_layer` and `seed_band_rich` finds nothing.
- End-to-end verification cannot catch an error in this path, because every avoid choice preserves the function. The only check on the exclusion is the offline `seed_coincidence.py` census (`RESULTS.md:1469-1475`; see also `STAGE2_LINKABILITY.md:182-183`).
- No log line reports how many first-layer pairs were found or how many seed redraws they caused.

**Uncalled code.** `PreprocessingParams::managed_tdp` (`types.rs:47-56`) and `seed_band` (`EM:855-857`) have no callers.

**Unproved guard property.** At production widths, the property that every nonzero band disturbs the data is spot-checked, not proved.

### 2a.9 Open questions

1. **Reverse-direction leak.** Is there a backward analogue of the first-layer leak in C, built on D's last-layer pairs? The reseed has an empty avoid set and collapses to degree ≤ 2 backward. Whether this matters depends on whether hiding D is in the threat model.
2. **Recognisable guard pattern.** The feistel guard is unshuffled. Its rounds (accumulate from the payload half, inject, uncompute in mirror order) form a recognisable stage-2 pattern. Whether stage-3 mixing hides it has not been measured.
3. **Managed preset.** The runner and the validator need a preset for the recommended configuration (§6.10).


## Stage 2b: the masked compute (masks, fires, brackets, chains)

Stage 2b evaluates the sliced sandwich A gate by gate inside a cloud of open quadratic masks. It is the middle part of stage 2's five-module envelope: opening guard, band seed, **masked compute**, band reseed, closing guard (`src/stages/preprocessing/construct.rs:156-160`). The entry point is `preprocess_embedded_masking_with_execution(&sandwich.gates, sandwich.num_wires, params, execution)` (`construct.rs:33-38`). Unless a path is given, line references below point into `EM`; `gen:` is `src/programs/gen_sandwich_gadget/mod.rs`.

### 2b.1 Inputs, outputs, wire roles

| item | value | reference |
|---|---|---|
| source `src` | sandwich gates: 7,920 at n=128 (\|C\|=\|D\|=3000, two slice blocks S1/S2 of s=896 each, plus the 128-CNOT copy column: 6000 + 2·896 + 128); 2,832 at n=64 (\|C\|=\|D\|=1000, s=384: 2000 + 2·384 + 64) | `W/vA128a/gen.log`, `W/vA64a/gen.log`; count formula `gen:142-145` |
| source gate shapes | C, D: g57 (2 controls, `comp=1`). S1, S2: 1/3 CNOT, 2/3 positive-control CCNOT. Copy column: CNOT. Generated sources have no 0-control gates (C draws 3 distinct pins, D distinct wires); a supplied `TDP_SOURCE_C` can contain one (§2b.11, defect 1) | `src/stages/sandwich/construct.rs:119-143` (slice blocks), `:158-170` (D), `:220-226` (assembly), `:330` (C via `from_g57`); `src/circuit/randomize.rs:22-34` |
| data width `np` | 2n: the whole sliced sandwich is the source | `construct.rs:33-38` |
| band width `r` | `np` when `p.r == 0` | EM:1178 |
| total width | `np + r` = 4n: 512 wires at n=128, 256 at n=64 | gen.log |
| wire roles | data `0..np`, band `np..np+r`. The compute does not seed the band: the seed and reseed are separate modules emitted around it (`construct.rs:111-139, 157-159`). It does write band wires: refresh bursts (EM:1584-1590), bracket ancilla toggles `h ^= p·x` (EM:719-721, 825-834), and drifted or chained ancillas left changed or displaced (EM:2443, 2310-2330). The band ends dirty (EM:1154) | EM:1-5, 1151-1157, 1206 |

The output, `EmbeddedMaskingOutput` (EM:240-273), has these fields:

- `gates`.
- `num_wires = np + r`.
- `atoms = Σ_w (u_w + 1 + extra_lgis)` (EM:1274). **[stale]** The doc comment "3m+n" is exact only if every source gate has two controls.
- `rerand_done`, which counts burst *gates*.
- `pre_gates` and `post_gates`, which are non-empty only with `encoded_io`.
- `scratch_wires = 0`, `r_used`, `final_layout`, `shuffling_stats`, `role_hot_intervals`.

The contract is that the data roles end holding A's output and the band is left dirty. `construct.rs:39-48` asserts that every data role is physically home.

Gate semantics come from `XGate`: `x[t] ^= comp ⊕ AND_i lit_i`, with `AND(∅) = 1` (`src/circuit/xgate.rs:1-8, 202-212`). Hence `x_gate(t)` is `comp=0` with no controls, i.e. an always-firing NOT (`xgate.rs:51-58`). `from_g57([a,x,y])` is `comp=1` with literals `(¬x, y)`, i.e. `a ^= 1 ⊕ ¬x·y = 1 ⊕ y ⊕ x·y` (`xgate.rs:65-87`). A degenerate `[a,x,x]` becomes `x_gate(a)` (`xgate.rs:67-70`).

### 2b.2 Parameters

`EmbeddedMaskingParams::production` (EM:196-217) supplies the defaults. `managed_tdp(seed, n)` is the same preset with `active_wires = n` (EM:183-188). The gen adapter (`gen:248-290`) builds the params from `base = managed_tdp(gadget_seed, n)`. Most `EMBEDDED_MASKING_*` variables fall back to `base`, with three exceptions: `min_open` falls back to the literal 2 (gen:268-271), `burst_band_only` to `false` (gen:265-266), and `encoded_io` is hard-coded `false` (gen:267). All three match production today. This is the canonical stage-2 knob table:

| field | default | env (`EMBEDDED_MASKING` + suffix) | meaning |
|---|---|---|---|
| `k` | 2 | `_K` | Band wires per mask cycle before the balancing `z`, rounded down to disjoint g57 pairs (K=2 and K=3 give one pair, K=4 two). Effective `k = max(k,2)` (EM:1177, 1181). |
| `r` | 0 → `np` | — | Band width |
| `rerand_level` | 0 → `m/(4k)` | `_RERAND` | Straddle refresh slots |
| `rerand_repair` | 0 | `_REPAIR` | Repair refresh slots |
| `rerand_burst` | 0 → `8k` = 16 | `_BURST` | Gates per slot |
| `max_open` | 3 | `_MAX_OPEN` | Soft rolling cap of open masks per data wire |
| `min_open` | 2 | `_MIN_OPEN` | Hard floor. Clamped to `[1, max_open-1]` and asserted `< max_open` (EM:1232-1237). |
| `min_mask` | 0 → `max_open` | `_MIN_MASK` | Legacy linear read only (EM:1243-1249) |
| `active_wires` | n (gen:250) | — | In the compute, only the pool for burst data controls (EM:1251-1255, 309-331). **[stale]** The field doc at EM:99 ("seed the band"): the seed takes `n` directly (`construct.rs:111-121`). |
| `extra_lgis` | 0 | `_EXTRA_LGIS` | Added to the *filler* budget (EM:1279). **[stale]** The doc at EM:101-104 says "straddle slots". |
| `quad_fire` | true | `_QUAD_FIRE` | Quadratic (non-linearising) read |
| `balanced` | true | `_BALANCED` | Adds a `z` wire to each mask |
| `burst_band_only` | false | `_BURST_BANDONLY` | Burst controls drawn from the band only |
| `encoded_io` | false | — | Test harness: off-circuit I/O masks |
| `shuffling_*` | 0 / true / 0 | `_SHUFFLING`, `_SHUFFLING_RETURN_HOME`, `_JIT` | Carrier migration, §2c.1 (gen:42-93, 174-181) |

Several knobs are read from the environment inside EM itself rather than through the adapter:

| variable (`EMBEDDED_MASKING_` + name) | default | reference |
|---|---|---|
| `PREMASK` | off | EM:1926 |
| `CHAIN` | 0 | EM:1610-1613 |
| `CHAIN_LEN` | 0 | EM:1616-1619 |
| `CHAIN_ROTATE` | off | EM:1615 |
| `DRIFT` | **on** unless `"0"` | EM:2408 |
| `DRIFT_ONLY=lo:hi` | — | EM:2409-2414 |
| `ANC_RESERVE` | 0, capped at r/2 | EM:1213-1217 |
| `RETIRE*` | off | EM:1620-1670 |
| `STRADDLE_CENSUS`, `LGI_DEBUG`, `DRIFT_DEBUG`, `DUMP_GATES` | — | diagnostics |

Read elsewhere: `ANC_POOL` (default band, gen:284), `DIAG` (gen:286) and `HOT_MANIFEST` (gen:287, 318) in the adapter; on the seed side, `BAL_SEED` (default = `balanced`, gen:275-276), `SEED_NL` (default 4, gen:279-282), and `SEED_AVOID_C1` (`construct.rs:89`).

**Current pipeline configuration** (`W/val_chain.sh:13-16`; `STAGE2_LINKABILITY.md:157-163`):

- Stage 2b knobs: `PREMASK=1`, plus `CHAIN=2 CHAIN_LEN=8` (config L8).
- Knobs consumed by neighbouring parts: `SEED_AVOID_C1=1` (band seed, §2a.5) and `SHUFFLING=8 SHUFFLING_RETURN_HOME=0` (carrier migration, §2c).
- Everything else stays at the defaults above, with DRIFT on (but see 2b.6: under L8 it has almost nothing to act on).
- Dropping `CHAIN*` gives config A. It remains a supported alternative until the replicate validation is finished.
- `CHAIN_ROTATE`, `RETIRE*`, `ANC_RESERVE` and `JIT` were tested and dropped (`STAGE2_LINKABILITY.md:186-197`), and stay off.

This configuration is reachable only by calling `gen_sandwich_gadget` directly: the managed runner scrubs `PREMASK`, `CHAIN*` and `SEED_AVOID_C1` and rejects persistent migration (§6.10).

Derived values and preconditions:

- `lgi_k = (k − k%2) + balanced = 3` band wires per mask, written `[x, y, z]` (EM:1181).
- The code asserts `r > lgi_k`, `np + r ≥ 8` under quad fire, `np ≥ 2` and `np + r < u16::MAX` (EM:1186-1205).
- `rng = StdRng::seed_from_u64(p.seed)` with `p.seed = gadget_seed` (EM:1256).

### 2b.3 Masks (LGIs) and their bookkeeping

**Mask term.** A mask on data wire `w` with band wires `[x, y, z]` is emitted as two gates, `g57(w,x,y); CNOT(w,z)` (`emit_lgi`, EM:1113-1134; `cycle_g57`, EM:281-291). Together they XOR the term

    M = 1 ⊕ y ⊕ x·y ⊕ z

onto the wire. Re-emitting the same two gates while `x, y, z` are unchanged removes the mask. The code calls such an open/close pair an **LGI** ("locally geodesic identity", `docs/gadgetization.md:102-105`). `emit_lgi` *toggles* the ledgers, so it is self-inverse.

**Why `z` is there.** The bare g57 term is 1 with probability 3/4. A wire under one or two such terms therefore keeps a linear correlation of 1/2 to 1/4 with its plaintext, which statistical tests detect (fire_corr, gauntlet w1; both in §5.9). The fresh `z` balances the term and keeps it quadratic. The cost is that each operand polynomial grows from 8 to 11 monomials at `max_open=3` (EM:116-128). **[stale]** The "+92% gates" in that comment (EM:125-126) assumed one gate per product monomial. With the current bracket emission (2b.5), and ignoring constant terms, a fire costs `1 + 6(q_a+q_b) + 28·q_a·q_b` gates balanced against `1 + 5(q_a+q_b) + 17·q_a·q_b` unbalanced (where `|L| = q`, EM:477-490). That is 137 vs 89 gates at q=2/2 (+54%) and 289 vs 184 at q=3/3 (+57%) (*derived*).

**Disjoint pairs.** With K>2, pairs are disjoint rather than forming a cycle. Under the legacy read's pair completion, a cycle telescopes to a bare operand (EM:282-285).

**Ledgers** (EM:1319-1321):

- `open_cy[w]` lists the persistent open masks, oldest first. The rolling cap closes `remove(0)`; the final drain closes with `pop()`.
- `open_pairs[w]` is the set of ordered `(x,y)` pairs currently toggled on.
- `open_lin[w]` is the set of `z`'s currently on.

The temporary *fire cover* is toggled into `open_pairs`/`open_lin` but is never pushed to `open_cy` (EM:2164-2177).

**Decode invariant.** With `V_w` the logical value and `W_w` the carrier value, `V_w = W_w ⊕ ⊕_{j∈open(w)} M_j(B)` (`docs/gadgetization.md:83-90`). For a read, `Operand {w, pol, cycles}` (EM:459-498) decomposes the literal as

    lit(V) = W ⊕ c ⊕ L ⊕ Q,   c = (q mod 2) ⊕ ¬pol,   L = Σ_j (y_j ⊕ z_j),   Q = Σ_j x_j·y_j

where `q` is the number of open masks. So `|L| = 2q` and `|Q| = q`.

**Fresh-mask sampler** `sample_fresh!(w, avoid, extra)` (EM:1423-1455):

- It draws `sample_k(mask_band, 3)`: three distinct band wires, uniformly, in order.
- A draw containing `avoid` (a band wire about to be refreshed) is always rejected. The sampler asserts after 2^16 tries.
- Otherwise it prefers draws disjoint from all wires of `w`'s open masks ∪ `extra`.
- After 256 tries it accepts a draw with no identical pair (either order) and no identical `z`. After 4,096 tries it accepts anything. Every accepted clash increments `relaxed`.

Disjointness matters because identical pairs cancel and identical `z`'s cancel, and because a `z` equal to another pair's wire folds the terms into an OR (`x ⊕ ¬x·y = x ∨ y`, biased 3:1). Each of these was measured at φ≈0.25, where φ is the binary (Pearson) correlation coefficient of §5.9 (EM:1406-1419). The comment says production bands never relax. **[stale]** Its "r = n ≥ 128" (EM:1419): the band is now `r = np = 2n` (EM:1178).

**Coverage rules:**

- `max_open = 3` is a soft cap: covers and refresh replacements may exceed it.
- `min_open = 2` holds at every instant between a wire's first and last mask (EM:144-154). With one mask, a single visible monomial can cancel the wire's only uniform term (gauntlet w2, §5.9). Before this rule, 26% of covered wire-time sat at one mask.
- `ensure_min_open!` opens fresh masks (site2) until the floor holds (EM:1851-1868).
- `filler_open!(w)` (EM:1870-1901) does the following, in order:
  1. close the oldest mask if at cap (site3);
  2. open a fresh mask (site4);
  3. run `ensure_min_open`;
  4. decrement `filler_left[w]`;
  5. call `maybe_rerand`.
- Consequence for reads: in production every operand has `q ∈ {2,3}` open masks (read-cover floor 2; every open path keeps `|open_cy[w]| ≤ 3`).

**Budget** (EM:1259-1280). Let `u_w` be the writes plus control reads of `w`.

- Straddle opens: one per write.
- Filler opens: `filler_left[w] = u_w + 1 + extra_lgis − writes[w]`, i.e. reads + 1.
- Atoms: 23,273 in vA128a and 8,299 in vA64a (gen.log), so there are ≈15.4k fillers at n=128.
- Scheduling: fillers are spread evenly. After `placed` source gates the code catches up to `placed·total_fillers/m` opens, picking each wire weighted by `filler_left` (EM:1136-1149, 2573-2584).
- A safety drain follows (EM:2590-2596). In practice it emits nothing. **[stale]** Its comment "emits no rerand": `filler_open!` calls `maybe_rerand`.

**PREMASK** (EM:1922-1947):

- *What it does.* Before the first fire, every data wire is visited in a uniformly shuffled order and receives `min_open` fresh masks (site11). These are on-circuit, extra to the budget, and otherwise ordinary persistent masks. As the oldest entries of `open_cy[w]`, they are the first masks closed by the rolling cap (site3, EM:1876-1886; site6, EM:2129-2139), and a refresh slot whose band wire they contain closes them too (site1, EM:1565-1583). Only on wires with too few later opens do they reach the final drain. **[stale]** The notebook's "drained by the normal end-of-compute close" (`RESULTS.md:1466-1467`) is inaccurate. The shuffle consumes the main RNG, so turning PREMASK on changes the whole downstream trajectory.
- *The problem it fixes.* Without it a wire is first masked at its first use. Raw input copies then survive deep into the compute: 39–45 of 64 C wires were still unwritten after 1,000 compute gates, and 0/64 with PREMASK (`RESULTS.md:1465-1468`). After mixing, such copies combine with a wire holding `x·y` (a seed product or the gate's own fire ancilla) into a first-layer C value `V = 1 + a + y + x·y`. This is the fringe leak (`STAGE2_LINKABILITY.md:71-78`). `SEED_AVOID_C1` is the companion fix on the band-seed side (§2a.5).
- *Cost.* No measurable gate cost (`RESULTS.md:1475`), measured in arms that used K=64 retirement (§2a.5).
- *A derived consequence.* Under PREMASK, persistent coverage never drops below 2 before the final drain. Every decrease is either a cap close, which fires only at ≥3 open masks and so leaves ≥2 (EM:1876, 2129), or a refresh close preceded by replacements (`keep_covered!`, EM:1456-1483). The read-cover and mid-fire `ensure_min_open` calls therefore become dormant fallbacks.

### 2b.4 Scheduling the source gates

`compute_deps` (EM:1058-1093) builds the data-hazard DAG:

- RAW: a read follows every earlier write of that wire.
- WAR: a write follows every earlier read of its target.
- There are no WAW edges, because same-target XOR writes commute.
- The per-wire histories are never cleared, so a later writer cannot slip ahead of a read of the initial value.

Placement is Kahn's algorithm over a FIFO `VecDeque` seeded with the indegree-0 gates in index order (EM:1282-1292, 2567-2572). The order is deterministic and BFS-like. All randomness comes from masks, refreshes and unit shuffles (`docs/gadgetization.md:214-220`).

After each fire the code runs, in order:

1. `straddles_left[c]−−` and `placed++`;
2. `retire_sweep!`, which is off by default;
3. `maybe_rerand!`;
4. release of dependents;
5. filler catch-up (EM:2561-2584).

### 2b.5 The masked fire of one source gate

**Why quad fire.** The legacy read linearised each operand: `g57(w,r1,r2) ⊕ g57(w,r2,r1) = w ⊕ r1 ⊕ r2`. That left the wire exactly affine in band wires until the undo, and every later reordering stage stretched that window. It was measured as the C-vs-G affine ridge, about 6k relations (EM:12-28). Quad fire never linearises. Operands stay quadratic, and the fire is the product polynomial (degree ≤ 4).

The legacy path (`linearize`, `masked_fire`, EM:366-457; `QUAD_FIRE=0`) remains only for comparison and tests. **[stale]** The text describing a "top up to `max_open`, undone after the fire" appears in three places (EM:22-23, EM:37-42, and the `quad_fire` field doc at EM:107-109). Quad mode only enforces `min_open` through the read cover (EM:1967-1971), and `undo` stays empty (EM:1957).

For source gate `gi` with target `c`, controls `(w, pol)` and `comp`, the steps are (EM:1949-2560):

1. **Read cover** (EM:1965-1986). For each control, run `ensure_min_open!(w)` so that no wire is ever read bare, then snapshot `Operand{w, pol, open_cy[w]}`.
2. **`product_fire`** (EM:2078-2112) builds the fire as a list of commuting *units*. Nothing is emitted yet. Ancilla draws happen here.
3. **Fire box.** If `shuffling_segments ≠ 0`, open a `FireBox` for carrier migration (EM:2118-2128; §2c.2).
4. **Rolling cap.** If `|open_cy[c]| ≥ 3`, close the oldest mask (site6).
5. **Fire cover** (site7). Open a temporary mask on `c`, sampled away from every band wire of the operand masks (EM:2141-2177). Without it, an operand monomial sharing a wire with one of `c`'s masks cancels or folds that mask's uniform term (`z⊕z`, `z⊕z·u`), and the partial sums on `c` become biased toward `c_new`. At 256 band wires a collision occurs at about 2/3 of fires. The cover costs 4 gates per fire.
6. **Straddle draw.** Draw the persistent mask `cy` away from the operand and cover wires (EM:2178-2182). Then pre-draw `max(0, min_open − (|open_cy[c]|+1))` mid-fire opens (EM:2183-2193). Pre-drawing fixed the last drift leak (`RESULTS.md:942-960`).
7. **Chains** (2b.7), then **drift** (2b.6).
8. **Emission** (EM:2463-2555). Let `cut = ⌈N/2⌉` for `N` units. Each half is Fisher–Yates shuffled independently, with its deferred restores attached. The resulting block is:

```
[site6  close oldest mask of c, if at cap]
site7   open temporary cover on c                 (g57 + CNOT)
        units[0..cut]  in shuffled order            (+ chain flushes)
site8   open persistent straddle mask cy on c
site2-mid extra opens if c was thin               (dormant under PREMASK)
        units[cut..N]  in shuffled order            (+ chain flushes)
        flush remaining deferred restores
site9   close cover
```

The straddle in the middle implements **hidden firing** (EM:44-48). Across any block that contains the whole fire, the net XOR on `c` is `Δ ⊕ M_cy`, never the bare gate increment that would reveal which gate of A fired.

**Term classes.** Write the two operands as `P_a = a' ⊕ c_a ⊕ L_a ⊕ Q_a` and `P_b = b' ⊕ c_b ⊕ L_b ⊕ Q_b`. The fire is `comp ⊕ P_a·P_b`, and its 16 cross terms are emitted as follows, in this push order (EM:639-846; the census indices are the DIAG classes, EM:2643-2663):

| # | term | emission (target `t = c`) |
|---|---|---|
| 0 | `a'b'` | `and2(t, a', b')` |
| 1 | wire × linear | `and2(t, a', l)` for each `l ∈ L_b`, and the mirror |
| 2 | linear × linear | `and2(t, l_i, l_j)` (`and2` turns `u==v` into a CNOT, EM:561-568) |
| 3 | constant × all | if `c_a`: `CNOT(t,b')`, a CNOT per `L_b`, `and2` per `Q_b`; mirrored for `c_b` |
| 4 | wire × quadratic | 4-gate bracket per `(x,y) ∈ Q_b`, and the mirror |
| 5 | linear × quadratic | 4-gate bracket per `(l, (x,y))` |
| 6 | quadratic × quadratic | 8-gate Gray walk per `(Q_a term, Q_b term)` |
| 7 | constant flip | `X` iff `comp ⊕ (c_a ∧ c_b)` |

A 1-control source gate emits `CNOT(t,a')`, a CNOT per `L_a`, an `and2` per `Q_a`, and `X` iff `comp ⊕ c_a` (EM:626-638).

**The 4-gate bracket** (`and3`, EM:719-721) is

    t ^= h·y;  h ^= p·x;  t ^= h·y;  h ^= p·x

For any prior `h₀` the net effect on `t` is `h₀y ⊕ (h₀ ⊕ px)y = p·x·y`, and `h` is restored. The prior value also blinds the intermediate.

**The 8-gate Gray walk** (EM:796-841) computes `(xy)·(x'y')`.

- Regrouping: the product is regrouped cross-operand as `P1·P2`. The candidates are `((x,x'),(y,y'))` and `((x,y'),(y,x'))`. The first candidate whose pairs are open on no data wire is used; otherwise `cands[0]` is used and `qq_blinded++` is counted (EM:798-810).
- Gates, with `fa`/`sa` a random assignment of the two ancillas `h1`/`h2`:

```
t^=h1h2; fa^=Pf; t^=h1h2; sa^=Ps; t^=h1h2; fa^=Pf; t^=h1h2; sa^=Ps
```

- Net effect: the four reads visit all `(h1,h2)` offsets, so the net is `t ^= P1·P2`.
- Restores: only the final gate is a pure restore. The mid-walk restore of `fa` is an inherent rebirth of its prior value, and the random choice of `fa` spreads it across wires (EM:814-824).

**Why there are no clean ancillas and every gate has ≤2 controls** (EM:570-597):

- The stage-3 store is a g57 2-control identity ball, so a gate with ≥3 controls would never be spliced and would carry C through verbatim.
- A clean scratch wire is 0 everywhere outside fires. That is a function-level invariant no mixing can hide, and its non-zero stretches would delimit the fire blocks.
- The unit test asserts `ctrls.len() ≤ 2` for all quad-fire output (`tests/unit/stages/preprocessing/embedded_masking.rs:95-100`).

**Ancilla choice** (`pick`, EM:644-703):

- One ancilla is chosen per term, uniformly from `anc_band`.
- Excluded: `t`, `a'`, `b'` and the term's own wires.
- Preference goes to wires outside the operands' mask wires.
- `ANC_POOL=all` would also admit data wires. It was rejected: a product XORed onto a masked data wire can cancel part of its mask (φ 0.134, gauntlet w1 flags on 6 targets versus none with the band pool, §5.9; EM:660-680).

**Per-fire size (*derived* from the code).** Take `q_a`, `q_b` open masks, balanced K=2, and ignore the constant-dependent terms. Then:

- 4-gate brackets: `q_a + q_b + 4·q_a·q_b`.
- 8-gate brackets: `q_a·q_b`.
- Total gates: `1 + 6(q_a+q_b) + 28·q_a·q_b`.

That gives 137 gates and 24 brackets at q=2/2, and 289 gates and 51 brackets at q=3/3. Brackets arise only for 2-control source gates: C and D (6,000 at n=128) plus about 2/3 of the 1,792 slice gates. The slice CNOTs (probability 1/3, `sandwich/construct.rs:129`) and the 128 copy CNOTs have one control. Observed (gen.log `[chain]` lines): 316,961 chained brackets in vL128a, ≈40 per source gate or ≈44 per 2-control gate; 109,464 in vL64a, ≈39 per gate or ≈44 per 2-control gate. Both lie between the q=2/2 and q=3/3 values.

### 2b.6 Restore-drop ("drift"), default on

**Motivation** (EM:704-718, 2333-2344). A bracket restore re-materialises the ancilla's prior value verbatim once per bracket. That is about 1,000 times per hot band wire, the **celebrity value** surface. Bracket restores were 84% of recurring-value births (`STAGE2_LINKABILITY.md:80-86`). If the restore is dropped, the ancilla drifts by `p·x`, so consecutive borrows differ by data quadratics rather than by linear relations.

**Rule** (EM:2432-2446). The final gate of a droppable `(unit, h)` with `h ≥ np` is popped only if all of the following hold:

1. `h` is not in the *live* support of all open masks on all data wires ∪ `open_lin`, recomputed at drop time and unioned with the `product_fire` snapshot (EM:2386-2398). The stale snapshot alone caused the 2026-10-06 leak.
2. `h` is not in the cover.
3. `h` is not in the straddle `cy`.
4. `h` is not in the pre-drawn mid-fire opens.
5. `h` is used by exactly this unit (`band_unit_of`, EM:2364-2384). A shared drifted ancilla would break unit commutation.

Carrier migration adds a sixth layer: band roles written inside the box are excluded as transfer partners (`preprocessing_shuffling.rs:539-552`; §2c.5). Drifted wires are also appended to every `mask_support` entry of the box (EM:2453-2462).

**Status.** The code comment says "RC sign-off pending" (EM:2399). Drift is never applied to chained brackets, and in the recommended L8 configuration it has almost nothing to act on: whenever chaining activates (2b.7), the droppable list is cleared (EM:2309, 2328) before the drop pass (EM:2432). Drift then acts only on fires where chaining did not activate (fewer than two fire-safe band wires), which essentially never happens at n=64/128. Under config A the mechanism is **nearly inert**: live mask support covers almost the whole band, and about 0 ancillas drift at n=64/128 (`RESULTS.md:972-978`).

### 2b.7 Chained brackets (`CHAIN=k`, `CHAIN_LEN=L`)

**Algebra.** The first three gates of a 4-gate bracket add `p·x·y` to `t` for any value of `h`, and the Gray walk adds `P1·P2` for any start values (EM:2194-2204; `RESULTS.md:1640-1656`). A fire's brackets can therefore share `k` ancillas, accumulate partial sums on them, and defer their restores.

Small example on one chain ancilla with base value `h₀`:

1. Bracket 1 leaves `h = h₀ ⊕ p₁x₁` after adding `p₁x₁y₁` to `t`.
2. Bracket 2 reads that value. It still adds exactly `p₂x₂y₂` and leaves `h = h₀ ⊕ p₁x₁ ⊕ p₂x₂`.
3. The deferred restores `h ^= p₁x₁; h ^= p₂x₂` later bring back `h₀` once.

This is valid only if every `p_i`, `x_i` is unchanged until its restore runs. The candidate rule below guarantees that.

**Activation.** Chaining applies when `chain_k > 0`, the fire has droppable brackets, and at least two candidates exist (EM:2208, 2253).

**Fire-safe candidates** (EM:2209-2257):

- `reads` = `{c}` ∪ every wire of every unit except that unit's own bracket ancillas ∪ operand-mask and cover wires ∪ `cy` ∪ mid-fire opens ∪ `undo` (EM:2226-2247). It does not contain the persistent masks of `c` or of any other data wire (2b.11, defect 4).
- `cands` = band wires not in `reads`, Fisher–Yates shuffled.

**Re-targeting** (non-rotate path, EM:2310-2330):

1. Truncate `cands` to `max(k,2)`, so `|cands| = min(#fire-safe, max(k,2))` (EM:2311).
2. For each bracket, draw `first ∈ [0, |cands|)` and map its ancillas to `cands[(first+j) mod |cands|]` (EM:2315-2317). A 4-gate bracket uses one chain ancilla; a Gray walk uses two distinct ones.
3. Rewrite the unit with `remap_gate` (`preprocessing_shuffling.rs:80-92`) and pop its final restore into `chain_restore[ui]`.

**Restore queue** (EM:2479-2501, 2535-2540):

- Deferred restores are queued per ancilla. Units and their restores are shuffled together, so restores follow emission order.
- With `L > 0`, an ancilla's queue is flushed as soon as it holds `L` restores. Everything left is flushed after the second half, before the cover closes.
- Restores of one ancilla commute, since they XOR stable products into it.
- With full chains (`L=0`) the base value comes back once per fire; with bounded chains, roughly once per `L` brackets on that ancilla.

**Measured** (stage-2 recurrence census, `W/v*/mult_s2.txt`; "≥65" counts values born at least 65 times, with their share of all births):

| run | values born ≥65× (share of births) | ≥257× | top multiplicity | repeats |
|---|---|---|---|---|
| vA64a | 451 (19.1%) | 329 | 468 | 19.9% |
| vL64a | 351 (4.8%) | 0 | 186 | 8.6% |
| vA128a | 1,199 (19.2%) | 968 | 517 | 19.8% |
| vL128a | 989 (4.9%) | 0 | 214 | 8.6% |

- **Gate cost.** +0.9% stage-2 gates for full chains at k=2, from a single n=64 gen probe (783k vs 776k, `RESULTS.md:1659-1660`). The bounded L8 validation builds are within +0.3–0.6% of config A: +0.6% at n=128 (2,235,297 vs 2,222,024) and +0.25% at n=64 (779,195 vs 777,215), from the gen.log "diversified gadget" lines, with different seeds and C's.
- **Full chains.** The delivered ANF grows 2.4–2.5× after mixing, because a displaced ancilla spans the whole fire (`RESULTS.md:1683-1691`). This is why the recommended setting bounds chains at `L = 8`.
- **Bounded `L=8` against config A after stage-4 repair.** The replicate deltas are tabulated once, in §5.6 ("L8 against A"). In brief (n=64 / n=128): L2 −12% / −13%, M2 −20% / −25%, proxy values (link census) −24% / −36%, anchors −39% / −29%, `deliverable.bin` size +19% / −1%. Linkable froth grows by +0.7k / +3.3k values, almost all of them seen exactly twice. The slightly different figures in `STAGE2_LINKABILITY.md:132-135` come from the earlier finals, not the replicates.
- **`CHAIN_ROTATE=1`**, which gives each `L`-bracket segment its own atomic unit (EM:2258-2309), was dropped: no delivered gain, +48–73% size.

**Interaction with carrier migration.** **[stale]** The `FireBox` doc says band writes must be restored by the next unit boundary (`preprocessing_shuffling.rs:17-18`). Chains and drift violate this literally. Safety comes from the in-box written-partner exclusion (§2c.5).

### 2b.8 Band refresh: rerand slots with hot-targeting

**Plan** (EM:1294-1317):

- `straddle_slots = rerand_level` if set, else `m/(4k) = m/8`: 990 slots at n=128 and 354 at n=64. **[stale]** The "~875" in the comments at EM:80 and EM:192.
- Each slot is a **rerand burst** of 16 gates on one band wire: 15,840 and 5,664 gates (gen.log, under the misleading label "rerand bursts").
- The slot plan is shuffled. The slot count is what thins masking, with a knee near 1,024 (EM:77-89).

**Pacing.** One slot fires every `rgap = (m + total_fillers)/slots` steps, where a step is a placement or a filler open (EM:1372-1383, 1594-1601). That is about 23 in vA128a (*derived*). There is no end flush; undelivered slots produce a WARN (EM:2597-2612). `emit_slot` asserts it never runs inside a fire box (EM:1517-1520).

**Hot-targeting** (EM:1358-1364, 1527-1540). The slot's wire `b` is drawn uniformly among band wires with maximal `band_uses`, the bracket borrows since that wire's last burst, and `band_uses[b]` is then reset. This lowered the top multiplicity from 1,219 to 453 at n=64 (`RESULTS.md:972-975`).

**Straddle slot** (default, EM:1564-1591):

1. For each data wire, in index order (EM:1565-1583): `keep_covered!(w,b)` opens replacements away from `b` if closing would leave fewer than `min_open` masks (site0). Then every mask on `w` containing `b` is closed (site1).
2. After the loop, the 16-gate burst on `b` is emitted once (EM:1584-1590).

Because same-target XORs commute, opening before closing never leaves a bare instant. Before this rule there were about 450 interior bare intervals per build, median about 10k gates (EM:1390-1399).

**Burst gate** (EM:309-331). Each gate is `b ^= lit(c₁)·lit(c₂)`. Each control is, with probability 1/2, a data wire in `0..n`, otherwise a band wire. Polarities are random and `c₁ ≠ c₂ ≠ b`. `burst_gate_safe!` redraws, up to 256 times, when one control lies in the other's mask wires or when two data controls share a mask wire (φ 0.14–0.26 otherwise; EM:1484-1514).

Repair slots (EM:1541-1563) are off by default.

### 2b.9 End of compute and diagnostics

**Final drain** (EM:2613-2621). Each data wire, in index order, pops and closes all its still-open masks on-circuit. The whole stream, including the drain, is then handed to carrier migration (EM:2672-2701; §2c.1).

**Hot-value manifest** (`hot_intervals`, EM:500-554). It records the intervals during which a data wire has no open pair, i.e. its value is affine in plaintext and band. It is computed on the pre-migration stream. With a correct build only the I/O fringes appear; anything else is a defect. It is written only when `EMBEDDED_MASKING_HOT_MANIFEST=path` (gen:287, 318-334) and must never ship with a deliverable (EM:512-513).

**`EMBEDDED_MASKING_DIAG`** (EM:2623-2663) prints:

- read counts, bare reads and `|ρ|` (= 2q);
- slot counts;
- replacement, fire-cover and min-open counts;
- `qq_blinded` and `relaxed`;
- the 8-class fire census.

The "straddled=m unplaced=0" field is hard-coded, and "read-cover LGIs" is always 0 (EM:1401). In gen.log, "fire gates" is the entire pre-migration compute stream (`preprocessing_shuffling.rs:481, 503`), not only fire units.

### 2b.10 Correctness evidence

- **Exhaustive unit test** (`tests/unit/stages/preprocessing/embedded_masking.rs:22-103`): n=6 with r=6, 24 random g57 gates, 6 seeds, all 64 data inputs × 8 band states. It covers quad/legacy reads, balanced on/off, `encoded_io`, K=2/3/4 and repair slots.
- **Not covered by unit tests:** PREMASK, chains and seed avoidance; these have only end-to-end checks (`STAGE2_LINKABILITY.md:182-184`). The env-read knobs also make test behaviour depend on the shell environment (observation).
- **Validation runs:** all eight validation builds (vA/vL × 64/128 × a/b) pass forward and reverse verification on 256 bit-sliced samples (gen.log).

### 2b.11 Known defects and open questions

1. **Inverted 0-operand fire (latent).** `product_fire` emits `X` when `comp` is true (EM:621-625). A 0-control XGate fires when `comp=0` (`xgate.rs:51-58`), and the legacy path does this correctly (EM:403-407). The branch is unreachable for generated sources: C draws 3 distinct pins (`randomize.rs:22-34`) and D distinct wires (`sandwich/construct.rs:160-170`). It is reachable through `TDP_SOURCE_C`. A supplied degenerate g57 `[a,x,x]` passes `prepare_source` (`sandwich/construct.rs:64-90`, which checks only emptiness, width and count) and the g57 decoder (`src/circuit/formats/g57.rs:27-44, 127ff`, which checks only 3 wires per gate). `from_g57` then turns it into `x_gate(a)` (`xgate.rs:67-70`, `sandwich/construct.rs:330`), and `product_fire` emits nothing: a wrong circuit that only gen's payload verification would catch. Fix the branch to `if !comp` and/or reject `x == y` in `prepare_source`.
2. **Chain borrow undercount.** Chain ancillas count +2 per fire (EM:2342-2358), but under `CHAIN_LEN=L` the base value is re-born at every flush (about every `L` brackets per ancilla). Hot-targeting therefore undercounts them.
3. **Mid-fire opens ignore `cy`.** The pre-drawn mid-fire opens are sampled without `cy` in their avoid set (EM:2188-2193), so they may share wires with it. This is dormant under PREMASK.
4. **Chain candidates may be live mask wires.** Candidates (EM:2248-2252) are filtered against neither other data wires' open masks nor the target's own persistent masks: `reads` (EM:2226-2247) holds the operand masks and cover (`fire_wires`, EM:2152-2176) but not `open_cy[c]`. A chain ancilla can therefore be a band wire, even the `z`, of one of `c`'s open masks. That mask's term is then displaced by partial products during the fire. This is the situation the fire cover guards against (EM:2141-2151), now on the target's own masks; the cover itself is in `reads`, so `c` stays covered. Drift excludes all such wires through `live_support` (EM:2391-2398); chains do not. Correctness still holds: the deferred flush runs before the cover closes, and no mask containing a chain ancilla is opened or closed between site6 and the flush (cover, `cy` and mid-fire opens are all in `reads`). The security effect is unexamined.
5. **The open-pair rule is only partly enforced.** EM:596-597 claims that no ancilla holds a mask's own quadratic term. For Gray-walk groupings this is checked, with a counted fallback: when neither grouping passes, `cands[0]` is used anyway and `qq_blinded` is incremented (EM:803-810). The 4-gate brackets are unchecked. EM:691 compares a normalised pair with ordered pairs; this matters only for `ANC_POOL=all`.
6. **Positional half split (observation).** `product_fire` pushes every single-gate unit (classes 0–3) before any bracket (classes 4–6), and `cut = ⌈N/2⌉` (EM:2465). With `q_a, q_b ∈ {2,3}` there are `1 + 2(q_a+q_b) + 4q_aq_b` single-gate units (plus class-3 terms) and `q_a+q_b+5q_aq_b` brackets. So the first half contains no bracket at q=2/2, at most one at 2/3, and at most two (of 51) at 3/3. Essentially every bracket follows the straddle open, which therefore sits at a predictable seam between the single-gate terms and the brackets. The non-rotate chain path keeps unit order, so this also holds under L8. Whether it survives carrier migration and stage 3 is unexamined.
7. **Deterministic orders.** Placement order (FIFO Kahn) and drain order (wire index) are deterministic. Whether either leaves a signature that survives mixing is unchecked.
8. **Burst controls.** `burst_band_only=false` is the default despite the warning at EM:129-133 that data-control bursts can correlate with a C fire. `burst_gate_safe` is the only mitigation.
9. **Drift default.** Drift is default-on and not yet signed off (EM:2399). Under config A it is nearly inert (`RESULTS.md:972-978`). Under L8, chaining clears the droppable list first (EM:2309, 2328), so the flag is effectively a no-op. The decision is whether the documented pipeline keeps the flag at all, or keeps it only for config A.
10. **Cleanup before the Rust/main push.** Candidates:
    - the legacy linear read and `min_mask`;
    - unused `shuffle_units`/`shuffle_slice`;
    - `assert!(true)` (EM:1173);
    - the per-emission `LGI_DEBUG` env read with hard-coded wires 7/29 (EM:1106);
    - the retirement, `ANC_RESERVE`, `CHAIN_ROTATE` and JIT code (marked "drop", `STAGE2_LINKABILITY.md:186-197`);
    - the **[stale]** comments listed above: EM:22-23, 37-42 and 107-109 (quad-fire top-up/undo); EM:80 and 192 ("~875" slots; the actual count is 990); EM:99 (`active_wires` "seeds the band"); EM:101-104 (`extra_lgis`); EM:125-126 ("+92%"); EM:245 ("3m+n"); EM:852-853 and EM:1152 ("the compute only READS the band"); EM:1419 ("r = n"); the safety-drain "emits no rerand"; `construct.rs:79-80` ("compute (which only reads the band)"); gen `mod.rs:20-22` (RETURN_HOME "is required"; `parse_shuffling_controls` accepts 0, gen:61-69, and the recommended configuration uses it);
    - plus `docs/gadgetization.md:60-63`, which describes the thin seed rather than the default rich seed (`SEED_NL=4`), and `RESULTS.md:1466-1467` (PREMASK "drained by the end-of-compute close").


## Stage 2c: carrier migration (preprocessing shuffling)

**Carrier migration** (called "preprocessing shuffling" in the code) rewrites the masked compute stream (§2b) after it has been emitted. While a data role is being written, the pass moves it off its physical wire. In the recommended mode the role keeps moving for the rest of the computation. The emitter works in *logical roles*:

- data roles `0..np`, where `np = 2n` is the sandwich width;
- band roles `np..np+r`, with `r = np` by default (EM:1178), so `total = np + r = 4n` wires (EM:1204).

With migration enabled, the emitter also records one **fire box** per sandwich gate. The pass `apply_shuffling` then rewrites the whole stream:

1. Inside each box, at up to `segments` unit boundaries (the **dose**), it moves the box's target role onto a band role's physical wire with **two CNOTs** (a **transfer**; the band role is the **partner**).
2. The partner's value rides on the target as a temporary XOR **escort** until the box ends. It is drained there.
3. Every gate, inside or outside a box, is emitted through the evolving role→wire map. The physical wire currently holding a data role is its **carrier**.
4. In the recommended **persistent** mode (`return_home = false`), data roles are never sent home between boxes. They are swapped back to their ports once, at the compute tail.

File abbreviations used in this section, in addition to `EM` and `gen:`:

| short | path |
|---|---|
| `sh.rs` | `src/stages/preprocessing/preprocessing_shuffling.rs` |
| `construct.rs` | `src/stages/preprocessing/construct.rs` |
| `verify.rs` | `src/stages/preprocessing/verify.rs` |
| `sh_test.rs` | `tests/unit/stages/preprocessing/preprocessing_shuffling.rs` |
| `construct_test.rs` | `tests/unit/stages/preprocessing/construct.rs` |
| `s12/` | `W/s12_check/` |

### 2c.1 Configuration and placement

| env var (`gen_sandwich_gadget`) | field of `EmbeddedMaskingParams` | default | recommended | meaning |
|---|---|---|---|---|
| `EMBEDDED_MASKING_SHUFFLING` | `shuffling_segments` | `0` (off) | `8` | dose: maximum transfers per fire box |
| `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME` | `shuffling_return_home` | `true` | `0` | `false` selects persistent migration |
| `EMBEDDED_MASKING_JIT` | `shuffling_jit_gap` | `0` (`0`/`off`) | unset | just-in-time drains; measured negative and not used |

`gen_sandwich_gadget` parses these knobs only in `parse_shuffling_controls` (gen:42-93). It calls the parser before writing any artifact (gen:173-182). The parser applies these rules:

- Any of the three is rejected unless the mode is embedded masking (`:48-52`).
- `SHUFFLING` accepts any `usize` (`:53-55`).
- `RETURN_HOME` accepts `0/1/false/true` (`:61-69`).
- `JIT` accepts `0`, `off` or a gap (`:74-80`). It requires `SHUFFLING > 0` and `RETURN_HOME = 0` (`:84-91`).

Two other paths read or set the same variables:

- The managed TDP runner exports both from TOML (`src/tdp/runner.rs:368-379`).
- The standalone security binary has its own parser, which still uses the old rules (`security_tests/preprocessing/embedded_masking.rs:54-75`).

The values are placed over the `managed_tdp(gadget_seed, n)` base (gen:248-253). The library defaults are off (`production()`, EM:196-217), and the parameters are documented at EM:155-176.

The reference invocation is `W/val_chain.sh:15-16`. It sets `SHUFFLING=8 RETURN_HOME=0` together with `PREMASK=1 SEED_AVOID_C1=1`. For config L8 it also sets `CHAIN=2 CHAIN_LEN=8` (`:13`).

The pass is called once, at EM:2672-2701, on the **entire** emitted compute. That covers premask opens, every fire, rerand bursts, fillers, retirement sweeps and the final close of every open LGI. The comment at EM:2672-2673 gives the reason: "even with data returned home, later bursts, fillers and decoder gates must map their BAND controls."

Hot-interval diagnostics are computed beforehand, in logical coordinates. This happens only when `EMBEDDED_MASKING_HOT_MANIFEST` is set (EM:2664-2671; gen:287).

`construct.rs:156-160` then assembles `open guard ++ band seed ++ migrated compute ++ band reseed ++ close guard`. The persistent port-restoration tail is therefore the last part of the compute: it comes after the final LGI drain and before the band reseed.

### 2c.2 The fire-box ledger (emitter side)

`FireBox` (`sh.rs:17-27`) has these fields:

- `target`: the data role the fire writes.
- `start` and `end`: absolute indices into the unmigrated stream.
- `unit_cuts`: every legal cut. A cut is a boundary between **emitted units**: LGI toggles, fire units and bracket bodies. The first cut is `start` and the last is `end`.
- `mask_support`: snapshots `(cut, band roles)` of the target's current mask support, beginning at `start`.

`record_box_boundary!(masks_changed)` (EM:1332-1348) appends `out.len()` to `unit_cuts`. When masks changed, it also snapshots `open_cy[target] ∪ box_cover ∪ box_drifted`. Duplicate cuts are removed when the box closes (EM:2558).

A box opens (EM:2118-2128) after the operands' read preparation and before any gate touches the target. It closes after the fire (EM:2556-2560). In between, it contains, in order:

| # | content | cut recorded |
|---|---|---|
| 1 | overflow close (`site6`) of the oldest LGI if `open_cy[c].len() ≥ max_open` | mask change (`:2129-2140`) |
| 2 | fire-cover open (`site7`) | mask change (`:2164-2175`) |
| 3 | first-half fire units. Under L8, a unit may be followed by a flush of `CHAIN_LEN` queued restores of one chain ancilla (`:2483-2500`) | plain, after each unit and its flush (`:2502-2506`) |
| 4 | mid-fire straddle open (`site8`) | mask change (`:2507-2516`) |
| 5 | pre-drawn mid-fire min-open opens (`site2-mid`) | mask change, each (`:2517-2529`) |
| 6 | second-half fire units, with the same bounded flushes | plain, after each (`:2530-2534`) |
| 7 | flush of the *remaining* deferred chain restores | plain, only if any are pending (`:2535-2540`) |
| 8 | cover close (`site9`) | mask change (`:2542-2551`) |
| 9 | legacy `undo` gates (empty under `quad_fire`) | plain (`:2552-2555`) |

**Cuts are not restore points.** A cut does not mean every borrowed band ancilla has been restored:

- **Under L8**, each chained bracket's restore is detached from its unit (`chain_restore[ui] = units[ui].pop()`, EM:2325). The restore is queued per ancilla and flushed only when `CHAIN_LEN` restores have queued up (EM:2495-2498). So fewer than 8 restores per chain ancilla can still be pending across a cut.
- **Under config A**, drift is on by default (EM:2408). It drops the restore entirely (EM:2443), so the ancilla stays displaced for good.

**[stale]** The FireBox doc sentence "Band writes inside a unit must restore their borrowed values before the next unit boundary" (`sh.rs:17-18`) therefore holds in neither recommended configuration.

What keeps escort partners constant is `written_in_box` (`sh.rs:539-552`, `:598`). It removes every band role written anywhere in the box from the partner pool, and so covers chain ancillas and drifted ancillas alike. The code comment there: "mid-bracket toggles are only epoch-stable at unit granularity".

Three emitter rules keep the boxes self-contained:

- **Drifted ancillas are also excluded through `mask_support`.** Band ancillas whose bracket restore was dropped are added retroactively to every `mask_support` entry of the current box (EM:2453-2460). The comment at EM:2333-2341 explains this as "a pending transfer mask must stay constant to its drain".
- **No band refresh inside a box.** `emit_slot!` asserts this (EM:1515-1520).
- **Retirement and rerand run after the box closes** (EM:2562-2566).

So there is exactly one box per sandwich gate. The observed counts are 2,832 boxes at n=64 and 7,920 at n=128, matching the sandwich sizes in every `W/v*/gen.log`.

`validate_ledger` (`sh.rs:95-135`) checks:

- `np ≤ total ≤ 65536`, and all wires are below `total`;
- each target is a data role;
- boxes are ordered and disjoint;
- `unit_cuts` are strictly increasing from `start` to `end`;
- every `mask_support` cut is a unit cut, and every support role is a band role.

It also enforces the key invariant: **no gate in `[start, end)` reads the target** (`sh.rs:127-132`). Inside a box the target's wire carries `value ⊕ escorts`, so a read would see the wrong value. The tests are `target_read_in_box_is_rejected` and `mask_snapshot_inside_atomic_unit_is_rejected` (`sh_test.rs:560-586`).

### 2c.3 Cut selection: the dose is a quantile of target writes

`selected_cuts` (`sh.rs:137-196`) works as follows.

1. Let `W` be the number of gates in the box with `gate.target == target`. These are LGI toggles, fire monomials and bracket target writes.
2. If `W = 0`, it returns no cuts.
3. Otherwise it starts with `[start]`. If there is no interior cut, it stops there. For each `s = 1..segments-1` it adds the interior unit cut that minimises `|prefix(cut)·segments − W·s|`, where `prefix(cut)` counts target writes before `cut`. The arithmetic is exact `u128`, and ties go to the lower cut.
4. It sorts and deduplicates the result.

A box therefore gets at most `min(segments, 1 + #interior cuts)` cuts, and `end` is never among them. In persistent mode, dose 1 means one transfer before the box's first gate, held until the box ends. That is the "one carrier transfer per ORIGINAL-circuit write (3 gates per fire box)" intent recorded at gen:56-59. With dose 8, the box's target-write sequence is split into up to 8 roughly equal segments, each carried on a different physical wire.

For very large doses (`segments > 2·|interior|`), only the floor and ceiling quantiles next to each boundary's write count are evaluated. The comment at `:156-160` says this preserves the exact selection. `sh_test.rs:505-549` checks equality with the naive selection for segments 8–100, which covers the bounded path. At `segments = usize::MAX` it checks only structural sanity: at most `|unit_cuts|−1` cuts, first cut `start`, strictly increasing unit cuts, and no `end`.

### 2c.4 The two-CNOT role transfer

`XGate::cnot(t, c)` means `t ^= c` (`src/circuit/xgate.rs:60-63`). Consider a transfer of target role `t` (on wire `p = layout[t]`) to partner `a` (on wire `q = layout[a]`). The pass emits `cnot(q, p); cnot(p, q)` and then swaps the two layout entries (`sh.rs:621-632`).

Example with persistent mode, dose 2 and partners `a`, `b`. `X` is whatever role `t` carries at box start (its value under its open LGI masks), and `m1`, `m2` are XOR updates from fire writes:

```
start cut   q ^= p ; p ^= q     p: a            q: X⊕a        t→q, a→p
fire        q ^= m1             q: X⊕m1⊕a
cut c1      s ^= q ; q ^= s     q: b            s: X⊕m1⊕a⊕b   t→s, b→q
fire        s ^= m2             s: X⊕m1⊕m2⊕a⊕b
drain       s ^= p ; s ^= q     s: X⊕m1⊕m2                    (sh.rs:678-683)
```

Each transfer adds three gates. When the box ends, `t` lives on `s`, and the next box for `t` starts from there. The second CNOT of each pair re-writes the partner's band value **verbatim** on the vacated wire (see 2c.11).

All other gates, inside and outside boxes, are emitted through `remap_gate` (`sh.rs:80-93`), which maps the target and controls through `layout`, keeps polarities and `comp`, and re-sorts controls into canonical order.

### 2c.5 Partner selection

At a non-return cut (`sh.rs:584-620`), the candidates are band roles `np..total` that pass every exclusion below:

| exclusion | code | stated or inferred reason |
|---|---|---|
| `used`: already a partner in this box | `:585`, `:639` | Re-use would cancel the escort (`w ⊕ w`). Code comment at `:594-595`: "re-use would cancel its standing mask". |
| current `mask_support` snapshot (open LGI cycle wires, fire cover, drifted ancillas) | `:586-588` | Drifted ancillas change value inside the box (EM:2333-2341). For cycle and cover wires, the same `:594-595` comment applies. The mechanism is *inferred*: an escort equal to a mask wire folds or cancels that mask's uniform term (`z ⊕ z`, `z ⊕ z∧u`, as in the fire-cover comment at EM:2141-2151). |
| `last_read_end[role] > cut`: read by any remaining unit that writes the target, including dirty-helper controls | `:597`, `target_unit_read_ends` `:442-455` | The net target update must not depend on the escort. Test comment at `sh_test.rs:343-344`: "It cannot be an escort." Reads go through the layout, so the function would survive. This is a masking-quality rule (*inferred*). |
| `written_in_box` | `:539-552`, `:598` | Correctness: the drain XORs the partner's *current* value, which must equal its value at the cut (`:539-543`). This also covers chain ancillas with pending restores and drifted ancillas. |
| JIT `holds_for(role, t)` | `:599` | JIT only. |

Among the survivors, the pass keeps those with maximal **read pressure** `rsw[layout[role]]` and picks one uniformly (`:606-619`). Read pressure is the number of reads since the last write on the role's *current physical wire*.

`note_reads` (`:207-212`) updates `rsw` on every emitted gate: out-of-box gates, both transfer CNOTs, in-box gates and drains. The rationale (`:506-508`, `:603-605`) is that "the heavy readers are exactly the wires that need a refreshing write", so every transfer also caps a read run.

If no candidate survives, the cut is skipped and `skipped_cut_count` is incremented. The source gate is still emitted (`:655-676`; test `sh_test.rs:319-339`).

The RNG is `StdRng` seeded with `xxh3_64_with_seed(b"embedded-masking/preprocessing-shuffling/v1", p.seed)` (`sh.rs:492-494`), where `p.seed` is the gadget seed. Because the pass has its own seeded stream, turning migration on does not disturb the emitter's pre-migration stream. `construct_test.rs:147-152` asserts identical hot intervals and `original_compute_gates == baseline compute_gates`. With `segments = 0`, EM:2674-2680 skips the pass. `apply_shuffling` itself also returns the input unchanged (`sh.rs:476-486`; `sh_test.rs:551-558`).

### 2c.6 Box end: persistent drains vs. return home

In **persistent** mode, every pending partner is drained in push order with `cnot(layout[t], layout[partner])` (`sh.rs:678-683`). Added gates per box = `3 × transfers`.

In **return-home** mode (the library default):

1. The last selected cut is overwritten with `end` (`sh.rs:553-558`). The return must follow the final target write, otherwise "an earlier return leaves the last segment unfenced".
2. The return transfer takes `pending_masks[0]` as partner (`:582-583`). Under return home, that partner sits on the target's home wire. The same two CNOTs move the target home and remove the first escort (`:633-637`).
3. The remaining `k−2` escorts are drained.

A box with `k` transfers, including the return, costs `2k + (k−2)` gates. Test `sh_test.rs:438-472` checks 8 transfers → 22 gates.

Edge case: with return home and only the start cut selected, that cut is moved to `end` and becomes the return. No escort is pending, so the partner is `None` and the box gets **no** transfer (one skipped cut, `:655-657`). This happens in two cases:

- **Dose 1.** The quantile range `1..segments` is empty (`sh.rs:161-162`), so no interior cut is added. This case is not covered by a test.
- **No interior cut**, at any dose. Test `sh_test.rs:474-487` covers this at dose 8.

Return home keeps data-wire identity by construction, which is why it is not the recommended mode (2c.10).

### 2c.7 Persistent tail: port restoration

After the last source gate, the pass restores the data ports when `return_home` is false (`sh.rs:722-748`). For each data role `ρ` not on its home wire, it emits a 3-CNOT swap `cnot(ρ,w); cnot(w,ρ); cnot(ρ,w)` with `w = layout[ρ]`. Whichever role was displaced from wire `ρ` (data or band) moves to `w`.

Each swap counts one `restored_ports`. One swap can send two data roles home, when the displaced role is a data role whose home is `w`. So `restored_ports = np` proves every data role was off-home, while `np − 1` means `np − 1` or `np` were. These swaps do not update `rsw`.

`sh.rs:749-755` then asserts that `layout[..np]` is the identity in **both** modes, and `construct.rs:42-48` asserts the same contract again. Band roles remain permuted over the physical band wires `np..total`. The band reseed and closing guard act on physical wires, and the band is not part of the output promise ("only the sandwich's payload half is promised", `src/stages/preprocessing/types.rs:86-88`).

Because the tail follows the final LGI drain, these swaps move **unmasked output values**. The notebook calls this "the accepted tails exception" (`RESULTS.md:653-654`).

### 2c.8 Why this is correct

Within a box, two facts hold between consecutive emitted gates:

- **(i)** The wire `layout[t]` holds the unmigrated value of `t` XOR the pending escorts.
- **(ii)** Every other role `ρ` holds exactly its unmigrated value on `layout[ρ]`.

Fact (ii) holds for three reasons. The second CNOT leaves the partner's exact value on the vacated wire. Partners are never written in the box (`written_in_box`, which also covers chain ancillas with deferred restores and drifted ancillas). And partners are never moved again (`used`).

Fact (i) holds because every in-box write of `t` is XOR-additive and does not have `t` among its controls, and nothing in the box reads `t` (`validate_ledger`).

Draining with the unchanged partner values clears the escorts. So at every box end, the stream equals the unmigrated stream composed with the role permutation. The tail swaps then fix the data part of that permutation.

Test coverage:

- `transfers_replay_and_preserve_4096_dirty_states_and_inverses` (`sh_test.rs:162-209`) checks this for both modes against `final_layout`, including the inverse circuit. The fixture has `np = 3`, `total = 32`.
- `managed_preprocessing_shuffling_preserves_classic_and_balanced_ports` (`construct_test.rs:110-207`) checks both sandwich variants at n=6 with `r = 32` and dose 8. It covers all `2^(2n)` sandwich inputs on the payload half (`from..to`), against both the sandwich and the unmigrated baseline, plus the full inverse (`:172-203`). It does not exercise the `PREMASK` or `CHAIN` env knobs.

At build time, `gen_sandwich_gadget` runs `verify_payload` forward on the payload half and the reverse check for guarded classic sandwiches (gen:356-369; `verify.rs:43-93`; §2a.7). The port contract itself is enforced by the asserts at `sh.rs:749-755` and `construct.rs:42-48`.

### 2c.9 Cost and observed behaviour (recommended configuration)

The figures below are from the eight validation replicates, two seeds per configuration per scale (`STAGE2_LINKABILITY.md:151-153`), in `W/v*/gen.log`. The log's "fire gates" label is actually the whole pre-migration compute (`construct.rs:147`).

| run | config | n | boxes | pre-migration compute | transfers | skipped | ports restored | added | overhead |
|---|---|---|---|---|---|---|---|---|---|
| vA64a | A | 64 | 2,832 | 703,430 | 22,656 | 0 | 128 | 68,352 | 9.7% |
| vA64b | A | 64 | 2,832 | 704,373 | 22,656 | 0 | 127 | 68,349 | 9.7% |
| vL64a | L8 | 64 | 2,832 | 705,480 | 22,656 | 0 | 128 | 68,352 | 9.7% |
| vL64b | L8 | 64 | 2,832 | 705,211 | 22,656 | 0 | 128 | 68,352 | 9.7% |
| vA128a | A | 128 | 7,920 | 2,020,503 | 63,360 | 0 | 256 | 190,848 | 9.4% |
| vA128b | A | 128 | 7,920 | 2,033,178 | 63,360 | 0 | 255 | 190,845 | 9.4% |
| vL128a | L8 | 128 | 7,920 | 2,033,637 | 63,360 | 0 | 255 | 190,845 | 9.4% |
| vL128b | L8 | 128 | 7,920 | 2,030,942 | 63,360 | 0 | 256 | 190,848 | 9.4% |

Observations across the eight runs:

- `added = 3·transfers + 3·restored_ports` exactly in every run.
- Every box received the full 8 transfers (the transfer counts are exactly 8 × boxes).
- Ports restored were 127–128 of 128 at n=64 and 255–256 of 256 at n=128.
- All eight passed forward and reverse verify (all are classic sandwiches).

The log line is printed at gen:305-317, where `drain/other = added − 2·transfers − 3·ports`.

### 2c.10 Why it exists

**Value/wire separation.** Without migration, or with per-box return home, each data role occupies one wire for the whole compute. With return home at doses 8 and 32, the stage-2 wire classes stayed perfectly separable (touch AUC 0.000/1.000). That mode "preserves data-wire identity BY DESIGN" (`RESULTS.md:645-652`). RC's directive was "no top/bottom or data/band distinction in the mixed circuit; data bits keep moving across wires; do it in stage 2" (`RESULTS.md:665-666`).

Persistent migration landed in the same change set as read-pressure partners and the rich band seed (`RESULTS.md:663-680`). The n=64 `W/s12_check/` scans partly separate the three changes.

Stage-2 data-vs-band touch AUC (§5.9; line 14 of each `RD/wire_balance.py` report in `W/s12_check/`; 0.5 = indistinguishable, 0.000 or 1.000 = perfectly separable):

| arm | AUC | source |
|---|---|---|
| unmigrated | 0.000 | `wb_fresh_s2.txt` |
| return home, dose 32 | 0.000 | `wb_shuf32.txt` |
| persistent, thin seed (256-gate seed, `gen_pers8.log`), dose 8 | 0.447 | `wb_pers8.txt` |
| persistent, thin seed, dose 32 | 0.434 | `wb_pers32.txt` |
| persistent, dose 32, rich seed (1,445 gates, `gen_rich32.log`) | 0.514 | `wb_rich32.txt`; `RESULTS.md:682-683` rounds to 0.51/0.50 |

After full-budget mixing (regular store), persistent dose 8 with the thin seed moved the top-bottom touch AUC from 0.129 to 0.361 and data-band from 0.249 to 0.538. Compare `W/s12_check/wb_fresh_s3mix.txt` with `W/s12_check/wb_pers8_s3mix.txt`, lines 13-14; `RESULTS.md:684-686`. The combined rich-seed + persistent dose-32 run gave 0.405 / 0.549 on the middle 80% and 0.406 / 0.558 in full (`RESULTS.md:695-697`; `W/s12_check/wb_rich32_s3mix.txt`). No post-mix AUC has been measured on the exact recommended configuration.

Migration therefore carries most of the stage-2 equalization (0.000 → ~0.44), and the rich seed adds about 0.08 at dose 32. No recorded arm separates read pressure from migration.

Migration moves where a value lives, not whether it is masked. After a box's drain, the role is bare on its new wire, apart from its own LGI masks. Migration implements the location half of the design rule "values are not resident; they are in transit" (`docs/windowed_delivery_design.md:96`).

**Intra-box partial sums (the dose).** During a fire the target accumulates partial sums of monomials. A higher dose spreads those partial sums over more wires and adds more escorts. Dose sweep at n=64, G=128 pack (`RESULTS.md:896-919`). The exposure column is the packer's RESULT count ("exposed (packer)", §4.1), given as a percentage of the vocabulary, i.e. of all distinct stage-2 values:

| dose | gate cost | exposed (packer), % of vocabulary | read runs ≥100 / max |
|---|---|---|---|
| 0 | +0% | (not packed) | 789 / 285 |
| 1 | +1.3% | 7.1% | 439 / 237 |
| 2 | +2.4% | 5.8% | 335 / 237 |
| 4 | +4.9% | 4.5% | 261 / 228 |
| 8 | +9.6% | 3.6% | 204 / 228 |
| 16 | +19% | 3.0% | 146 / 193 |
| 32 | +36% | 2.5% | 76 / 193 |

Exposure "improves steeply 1->8 … then flattens", while recurrence worsens linearly (`RESULTS.md:912-915`). The notebook's "RECOMMENDED OPERATING POINT: dose 8" is at `RESULTS.md:916`. Dose 0 or 1 also leaves the delivered ANF larger: +25% monomials of higher degree and +50% MB (`RESULTS.md:1568-1569`; `STAGE2_LINKABILITY.md:143`).

**Read pressure.** The stated motivation is that least-load selection "favored cold wires" (`RESULTS.md:672-674`). The recorded comparison is against the unmigrated stream: dose 0 has 789 read runs ≥100 and a maximum run of 285 (`RESULTS.md:674`, `:904`). Persistent migration with read-pressure partners at dose 32 gave 62 runs ≥100 and a maximum of 141 (`RESULTS.md:674-675`). That comparison mixes migration with the partner rule. No least-load vs read-pressure A/B at equal dose is recorded.

### 2c.11 Known side effects

- **Verbatim rebirths.** Every transfer re-writes the partner's band value on the vacated wire. On `final64` (no chains), births of celebrity values (multiplicity ≥257) split as follows: transfer hops 15.6%, bracket restores 83.6%, drains 0.8% (`RESULTS.md:1260-1267`). With L8 bounded chains, 100+ anchors fall by 39%, and the remaining class is the transfer-hop class (`RESULTS.md:1713-1714`). Read pressure steers picks toward hot wires, which carry the celebrity values (`RESULTS.md:885-888`).
- **Residual anchors.** Values seen 100+ times: ~200 (n=64) and ~700 (n=128) with L8, against ~320 / ~990 under config A (`RESULTS.md:1743-1748`; replicate values in §5.6). `STAGE2_LINKABILITY.md:175-177` attributes this residual class to "migration hops, which no bracket design reaches".
- **Unimplemented fixes.** Two fixes are proposed but not built: value-aware partner filtering / rotate-before-swap (`RESULTS.md:888`), and shuffler-side band rotation at mask-support-free cuts (`RESULTS.md:1279-1286`). The latter understates its own safety condition: `mask_support` holds only the box target's masks, so a value write on a band role is safe only if every open LGI on *any* data wire that references it is closed or re-keyed, as threshold retirement does (EM:1791-1830). Retirement itself (`EMBEDDED_MASKING_RETIRE=K`) was swept and dropped because it raises L2 and creates interior C leaks (`STAGE2_LINKABILITY.md:141`, `:194`).

**JIT drains (present, unused).** `JitState` (`sh.rs:214-436`) keeps escorts alive past box end. It drains them just before a read of the masked role or a write of the holder, and re-masks idle bare roles. At n=64 with gap 24 the result was negative: +31% stage-2 gates, exposed (packer, % of carried) 16.0% → 43.8%, celebrity class unchanged (`RESULTS.md:1109-1149`). The verdict is "drop" (`STAGE2_LINKABILITY.md:196`).

### 2c.12 Inconsistent validators and stale text (the code is authoritative)

- **Managed TDP config** (`src/tdp/config/validate.rs:110-125`) rejects doses 1–7 and rejects `shuffling_return_home = false` whenever `segments > 0`; `scripts/tdp_gen.sh:267-276` and `tests/shell/tdp_flags.bash:38,40-41` enforce and expect the same. The recommended configuration is therefore unreachable through the managed TDP config ("no managed preset yet", `STAGE2_LINKABILITY.md:182`; §6.10). Persistent migration is reachable in three other ways:
  - direct `gen_sandwich_gadget`;
  - the `preprocess_sandwich` API, which has a unit test for it (`construct_test.rs:209-238`);
  - the Gauntlet `--shuffling-carry-layout` arm, which enforces dose ≥8 (`security_tests/gauntlet/gauntlet_gen.rs:272-275`) and sets `shuffling_return_home: !args.shuffling_carry_layout` (`:280`).
- **Stale docs and comments:**
  - **[stale]** `docs/tdp_pipeline.md:109-110` (floor of 8, return home mandatory) and `:121-123` ("cannot be disabled for that executable");
  - **[stale]** `docs/gadgetization.md:227` ("at least eight"), `:232` ("least-written eligible band destination"), `:244-258` (TDP requires return home; "the full sandwich constructor rejects this mode");
  - **[stale]** gen:20-22 (module doc: return home "is required when enabled here");
  - **[stale]** EM:1154-1157 ("Carried-layout shuffling instead returns their physical locations in `final_layout`", contradicted by `sh.rs:749-755` and `construct.rs:42-48`);
  - **[stale]** `sh.rs:17-18` (FireBox: band writes restored "before the next unit boundary"; false under drift and chains);
  - **[stale]** `src/tdp/config/validate.rs:123` (error text: "carried layouts are supported only by the embedded-masking API and Gauntlet");
  - **[stale]** `security_tests/gauntlet/gauntlet_gen.rs:26-29` (module doc: "carried output roles are decoded through final_layout").
- **Stale tests:** `security_tests/preprocessing/embedded_masking.rs:54-75` is a separate parser that still requires `≥8` (`:61-64`) and return home (`:72-75`). `security_tests/gauntlet/gauntlet_gen.rs:272-275` still requires 8.
- **Vacuous assert:** EM:1173-1176 contains `assert!(true, …)`, left over from relaxing the floor.

### 2c.13 Open questions

1. Should the persistent configuration become a managed TDP preset? If so, which segment floor is canonical: 1 (gen) or 8 (TDP config, security tests, Gauntlet)?
2. RC to confirm the wording for the `mask_support` and `last_read_end` exclusions. Their security rationale above is inferred.
3. The 3-CNOT port-restoration block (up to `3·np` CNOTs on bare outputs) is a recognisable stage-2 structure. No measurement shows whether stage-3 mixing or stage-4 packing hides it.
4. Zero skipped cuts and 8 transfers per box were observed in all eight replicates at n=64 and n=128 with `r = np`. Neither is guaranteed. Small `n`, a smaller band, or the reserved ancilla pool could produce skips, and this has not been characterised.
5. The Gauntlet `--shuffling-carry-layout` arm (`docs/tdp_pipeline.md:131-134`) no longer carries a layout. It sets `return_home = false` (`gauntlet_gen.rs:280`), which is now persistent migration with an identity data layout, because `apply_shuffling` always restores data ports. It also runs with `encoded_io: true` (`gauntlet_gen.rs:281`), which skips `PREMASK` (EM:1927). Its meaning needs to be re-stated, or the arm retired.
6. No unit test covers the `written_in_box` exclusion, or a ledger in which a cut separates a chained bracket's body from its deferred restore. Chains, PREMASK and seed avoidance have only end-to-end verification (`STAGE2_LINKABILITY.md:182-184`).


## Stage 3: database mixing

Stage 3 rewrites the stage-2 circuit G2 into an equivalent circuit G3 about twice as large, almost all of it spelled in g57 gates. It does this by repeating one move:

1. Take a few gates that are adjacent once commuting gates are moved aside.
2. Compute a canonical key for the function they compute together.
3. Look that key up in the regular frozen store of g57 circuits.
4. Splice in a randomly chosen stored circuit that computes the same function.

A size controller decides, round by round, whether a splice may grow the circuit (**MIX**) or may only shrink it or keep its length (**COMP**). A rare second move, the **g57 twist**, swaps the labels of two wires across a long window.

The design documents give stage 3 the role of "un-invertible respelling, dilution, DB-form digestion" (`docs/ANF_WINDOW_DELIVERY.md:374`, `docs/windowed_delivery_design.md:171-172`). It is not a semantic-hiding stage. Stage 2 fixes which values ever appear in the clear, and stage 4 fixes where anything can be observed. The curated store, the slot-3 thermostat/crossing walk and the old stages 4–6 are historic and play no part here.

The bare names `scheduling.rs`, `sampling.rs`, `transport.rs`, `transformations.rs`, `state.rs`, `params.rs`, `reporting.rs` and `provenance.rs` are in `src/engine/mixer/`, and in this section `mod.rs` and `cli.rs` are in `src/programs/circuit_mixer/`. Two files share the name `replacement.rs`: **`db/replacement.rs`** is `src/stages/db_mixing/replacement.rs` (lookup and choice), and **`mixer/replacement.rs`** is `src/engine/mixer/replacement.rs` (attempt, descent and splice).

Run statistics come from the eight validation runs (`W/v{A,L}{64,128}{a,b}/s3.log`). The worked case is `vL64a`: n = 64, config L8.

### 3.1 Inputs, command and outputs

**Input.** The input is G2, `tdp.mpmct1` (mpmct1 format, §1.3; gate `t ^= comp ⊕ ∧ᵢ ℓᵢ`). A g57 triple `[a,b,c]` is the gate `a ^= b ∨ ¬c`, i.e. the comp = 1 gate with monomial `¬b ∧ c` (`src/circuit/xgate.rs:65-88`; module note at `db/replacement.rs:5-16`). n = 64 gives 256 wires and n = 128 gives 512. Every stage-2 gate has at most two controls.

| vL64a input (256 wires) | gates |
|---|---|
| NOT (0 controls) | 1,946 |
| width 1 (CNOT type) | 117,245 |
| width-2 conjunctions, comp = 0 | 633,856 |
| g57 (comp = 1, width 2) | 26,148 |
| total | 779,195 |

**Command.** This is the recommended invocation:

```
NG2=$(head -1 tdp.mpmct1 | awk '{print $3}')                 # |G2|
MOVES=$(python3 -c "print(int($NG2*2*2*1.3))")               # ⌊N2·R1·1.3·|G2|⌋ = ⌊5.2·|G2|⌋
env -u FROZEN_CURATED_DIR \
  FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  circuit_mixer --input tdp.mpmct1 --output s3.mpmct1 --tdp --db-mixing \
    --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
    --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
    --no-local-verify --seed $MSEED
```

- **`env -u FROZEN_CURATED_DIR`.** This prefix is required for a regular-only run (see "Environment"). The reference driver does not have it: it sets only `FROZEN_DB_DIR` and `FROZEN_FILTER` inline and inherits everything else from the calling shell (`W/val_chain.sh:19-23`). Check the log for the `WARNING: curated-first ... curated OFF` line and for `curated=false` on the `DB ON` line. All eight validation logs show both.
- **Seed.** `MSEED` is a fresh 40-bit draw from Python's `secrets` module (`val_chain.sh:12`).
- **Driver extras.** The reference driver also passes `--p-comp 1.0 --p-any 0.1`. These equal the CLI defaults (`cli.rs:164-170`), so leaving them out changes nothing (see 3.2).
- **Move budget.** The general rule is `moves = ⌊N2·R1·1.3·|G2|⌋` for a profile `N0,N1,N2,R1,R2`. For `2,2,2,2,2` this is ⌊5.2·|G2|⌋ (`W/val_chain.sh:17-18`). The pre-port managed driver used `round(N2·R1·|G2|·1.3)` (`tdp_gen.sh:456-462` at `ea2e516f`), a difference of at most one move; the managed driver now computes `int(G*2*2*1.3)` with Python, exactly as `val_chain.sh` does (`stage3_moves_rule=int(G*2*2*1.3)` in the manifest). `scripts/windowed_delivery/run_stage3.sh:18` still rounds. Since eff (§3.4) counts passes over a circuit that grows to about R1·|G2|, the budget buys about N2·1.3 passes over the target-size circuit: in practice eff ≈ 3.03-3.05. That is past the ramp (eff 2) but before the eff-3.125 controller update that would enter phase 4. So the run ends on the budget in phase 3, at 2.04-2.14× (§3.4). The pre-port managed driver's comment called the formula a "move ceiling: effs x peak size x margin" (`tdp_gen.sh:461` at `ea2e516f`), i.e. a safety cap rather than the intended stop. With N0 = N1 = N2 = 2, however, the one-transition-per-update rule pushes phase 4 to eff 3.125, beyond the cap. Whether stopping in phase 3 is intended, or the budget should let `ProfileDone` fire, is open (§3.10). Anyone who changes `--profile` should recompute the budget with the general rule.

**Output.** The output is G3, `s3.mpmct1`, with the same wire count (`mod.rs:955-962`). The log goes to stdout/stderr.

- **No sidecars.** The reference driver leaves `--state-out`, `--origins-out`, `--gens-out` and `--anc-out` unset.
- **Consumer.** Stage 4 reads both G2 and G3 (`val_chain.sh:29`).
- **Composition.** The vL64a output has 1,593,054 gates: 1,409,244 g57 (88.5%), 159,971 width-2 conjunctions, 23,428 width-1 gates and 411 NOTs. Every gate still has at most two controls.

**Environment.**

| variable | reference value | effect |
|---|---|---|
| `FROZEN_DB_DIR` | `frozen_m1_m11` | Required; the mixer panics if it is absent (`src/database/environment.rs:25-30`). |
| `FROZEN_FILTER` | `1` | Loads `<dir>/filters.bin` if present. `frozen_m1_m11` on this Mac has no such file, so the log prints "running unfiltered" and the flag does nothing (`src/database/frozen.rs:661-696`). |
| `FROZEN_CURATED_DIR` | **must be unset** | Only this variable keeps stage 3 regular-only. `--tdp` sets `curated = true` again unless `--curated` is typed (`mod.rs:95-101`). This happens after `--no-curated` has been applied (`mod.rs:58-60`), so `--no-curated` has no effect under `--tdp`. Curated is switched off only when `FROZEN_CURATED_DIR` is absent and `--frozen-curated-dir` is not passed: the mixer then prints a WARNING and sets `curated = false` (`mod.rs:214-230`). If the variable is exported, the recommended command silently runs curated-first. |
| `FROZEN_REGULAR_VALUE_CONVENTION` | `native` (default) | `src/database/environment.rs:11-19` |
| `MIN_DIR_LOOKUP` | `Min` (default) | See "Lookup" in 3.5; `src/database/environment.rs:42-49`. |
| `LOOKUP_CACHE_MB`, `XPOLY_CANON_CACHE_MB` | 512, 1024 (defaults) | Exact caches, so their size affects only speed (`src/database/environment.rs:31-40`; `src/canonicalization/environment.rs:52-61`). |
| `CANON_RULE_L_BRANCH_CAP` | unset (no cap) | `src/canonicalization/environment.rs:6-14`; applied in `canonicalize.rs:1225`. |
| `MIXER_DB_LEN_BAND` | unset (no band) | `lo,hi` drops candidates outside that length range before any mode rule (`src/stages/db_mixing/environment.rs:4-11`; `db/replacement.rs:1419-1456`). |
| `TWIST_G57_NO_SLIDE`, `TWIST_G57_NO_RETRY` | unset (slides and redraws on) | Turn off twist slides and twist redraws (`src/engine/mixer/environment.rs:4-12`). |
| `MIXER_STOP_AT_PHASE`, `MIXER_STOP_AT_SPLICES`, `MIXER_STOP_FLAG` | unset | Extra stop rules: a profile phase, a splice count, or a stop-file path (`src/engine/mixer/environment.rs:28-47`; `mod.rs:826-831`). |
| `CANON_MONOMIAL_CAP`, `CANON_CACHE_MB` | unset | Read only on the g57 window path (`src/canonicalization/window.rs:202-268,353-423`), not on the XGate path that stage 3 uses. They are inert here, although the TDP driver pins them. |

The TDP driver pins different values: a Rule-L cap of 512 and `LOOKUP_CACHE_MB=2048` (`src/tdp/runner.rs:392-396`; also `scripts/tdp_gen.sh:466-470` and `run_stage3.sh:24`). Unlike the reference driver, it also sets or removes `FROZEN_CURATED_DIR` explicitly (`runner.rs:381-386`). The reference driver inherits these variables, so "the measured runs used the defaults" holds only if the calling shell was clean. The logs cannot confirm this for the Rule-L cap: a cap failure on this path is not counted (see "Canonical key" in 3.5), so `rule_l_skips=0` in all eight logs proves nothing about it. The cache sizes cannot change results. A Rule-L cap can, because a window that hits the cap silently becomes a miss. Which setting is canonical is still open.

### 3.2 Parameter resolution

Options resolve in three layers: an explicitly typed flag, then the preset (`--tdp` above `--db-mixing`), then the shipped defaults (`mod.rs:111-165`, with the DB stack at `mod.rs:140-152`; `cli.rs:713-813`). A flag counts as typed only if it appears on the command line (`mod.rs:19-20`).

- **`--db-mixing`** sets `twist_g57 = true`, `p_twist = 0.0005`, `db_advance = true` and `mix_pay_random = true` (`mod.rs:73-92`).
- **`--tdp`** sets `curated = true` and `curated_in_comp = true` unless they are typed, which overrides `--no-curated` (see 3.1). It also sets `db_advance = true` and deliberately leaves `p_mix` unset (`mod.rs:95-109`; `cli.rs:453-464`). It supplies the DB preset layer `DbLayer::tdp()` (`cli.rs:789-802`), which is the source of every value in the per-mode table below. That layer outranks the `p_mingen` 0.6 of `DbLayer::db_mixing()` (`cli.rs:806-812`) and the shipped defaults `s_db` 9, `p_convex` 0.4, `p_mingen` 0.8, `s_db_comp` 12 and `p_convex_comp` 0.9 (`cli.rs:775-785`).
- **Per-mode DB knobs** are resolved when used by `MixParams::db_knobs`, from mode plus geometry, then mode, then base (`params.rs:566-588`). The log prints the result in the `DB effective per mode` banner (`mod.rs:459-471`):

| | MIX | COMP |
|---|---|---|
| P(convex window) | 0.5 | 0.95 |
| maximum length, convex | 6 | 12 |
| maximum length, contiguous | 6 | 6 |
| prefix descent | off | on |
| `p_mingen` | 0.5 (inert) | 0 |

These values are also in force:

| knob | value | where |
|---|---|---|
| `p_db` | 1.0 | CLI |
| `p_mix` | unset (−1); `--profile` requires this | `mod.rs:300-307` |
| `db_convex_p` (convex growth keeps the seed's direction) | 0.75 | `cli.rs:303-306` |
| `w_window` / `w_pool` | 4 / 3 | `cli.rs:294-302` |
| DB splice verification | on | `cli.rs:307-311` |
| `dir_q` (birth-advance fraction) | 0.85 | `cli.rs:45-49` |
| `twist_min_len` | 64 | `cli.rs:120-122` |
| `verify_every` / `report_every` | 10,000 / 50,000 | `cli.rs:616-621` |
| `shuffle_rate`, `p_pair`, `p_bridge`, `gen_target`, `db_min_window` | 0 | `cli.rs:235-246,286-289,371-393,526-533` |

Constraints on the flags:

- `--profile` is the only size authority. The mixer refuses `--target-size`, `--size-hi`, `--size-lo`, `--p-mix` and `--split` alongside it (`mod.rs:251-254,300-307`).
- `--db-max-degree` must be at most 11 (`mod.rs:518-523`).
- With `p_db > 0` and `p_mix` unset, any typed COMP-mode override is a hard error (exit 2). The affected flags are `--s-db-comp`, `--s-db-comp-ctg`, `--p-convex-comp`, `--p-mingen-comp` and `--db-prefixes-comp` (`mod.rs:171-207`; `db_mode` defaults to `mix`, `cli.rs:174-175`). The guard tests only `p_mix < 0`, so it also fires under `--profile`, where COMP rounds do run (`scheduling.rs:181-190`). With `--db-mode comp`, the MIX overrides are refused instead. In stage 3, therefore, the COMP knobs come only from `DbLayer::tdp()` and cannot be changed from the CLI (see 3.10).

**Inert and historic switches.**

- `p_mingen` acts only when `gen_target > 0` (`sampling.rs:242-251`).
- `w_window` and `w_pool` never bind, because no gate has more than two controls.
- `--db-degree-probes` is vestigial, because the degree guard is now exact (`db/replacement.rs:1072-1077`).
- `--p-comp`, `--p-any`, `temp`, `target_size` and `dir_p` are read only by the historic slot-3 walk (`scheduling.rs:545-578`; `transformations.rs:484-490`). `p_comp > 0` also gates opening the store (`state.rs:685-694`), and `p_comp`/`p_any` gate the `DB ON` banner (`mod.rs:426`). Both are already implied by `p_db = 1.0`. They are scheduled for removal together with slot 3.
- The preset banners print `curated=true` because they run before the curated setting is resolved. The `DB ON` line shows the effective `curated=false` (`mod.rs:88-91,105-108`).

### 3.3 Round structure

The circuit lives in an index-linked arena. The original circuit is kept for the periodic equivalence check (`state.rs:796,813`).

- **Per-gate metadata** (`state.rs:733-745`). Each input gate gets an origin (its index), generation 0, a litter id, and a direction (L or R) drawn uniformly.
- **RNG.** The main RNG is `StdRng::seed_from_u64(seed)` (`state.rs:731-732`).
- **ANF budget.** At most 2^20 terms per multiply, 1,024 per output wire and 2,048 in total (`state.rs:783-792`).

Each loop iteration is one move (`scheduling.rs:425-651`):

1. **Slot 0: profile tick.** Add 1/size to `eff`, the measure of effective work (`scheduling.rs:162-168`). If a controller update is due, run it. Then draw the round's mode: MIX with probability `pmix`, otherwise COMP (`scheduling.rs:182-190`). If the controller has reached phase 4, report and return `ProfileDone` (`scheduling.rs:474-479`).
2. **Slot 1: twist.** With probability 0.0005, run one g57 twist (`scheduling.rs:351-373,486-492`).
3. **Slot 2: DB move.** In every round in which no twist ran, make one DB attempt in the round's mode (`scheduling.rs:526-544`). Because `p_db = 1.0`, the coin short-circuits and draws no randomness. A failed attempt still uses up the round. In vL64a, the 4,051,814 moves were 4,049,743 DB rounds plus 2,071 twists.

The global shuffle and bridge fusion are off and draw no randomness. Slot 3, the historic walk, runs only in rounds where no other slot ran. With `p_db = 1.0` it is unreachable, so removing it leaves the trajectory unchanged (`docs/notes/cleanup_inventory_20261007.md`, key fact 1).

`global_check` runs every 10,000 moves, and a report is printed every 50,000 (`scheduling.rs:612-647`). The run ends in one of two ways:

- **`MovesBudget`**: the move budget is spent, and a final `global_check` runs (`scheduling.rs:649-650`).
- **`ProfileDone`**: the profile completes. This exit returns without a `global_check` and relies on the post-float check (`mod.rs:901-904`).

Two other exits exist: `StopFlag` (a stop file or `MIXER_STOP_AT_*`) and `CircuitEmpty`. Both run a `global_check` first (`scheduling.rs:438-441,595-622`; `state.rs:586-612`).

### 3.4 Size profile controller

**Target.** `--profile N0,N1,N2,R1,R2` defines a target size S\*(eff), where eff is the sum of 1/size over moves, i.e. roughly the number of passes over the circuit (`scheduling.rs:7-21`). N1 and N2 are absolute eff thresholds, not durations. The target has four legs, with `s_in` the input gate count:

| eff range | S\*(eff) |
|---|---|
| [0, N0) | rises linearly from `s_in` to `R1·s_in` |
| [N0, N1) | holds at `R1·s_in` |
| [N1, N2) | moves linearly to `R2·s_in` |
| N2 and beyond | stays at `R2·s_in` |

For `2,2,2,2,2` this is a linear ramp to `2·s_in` over eff ∈ [0, 2], then a flat target. The hold and compression legs have zero length.

**Schedule.** The controller (`scheduling.rs:95-345`) starts in phase 1 with `pmix = 0.5`. It first updates at eff = 0.125 and then every 0.5 eff.

**Rate estimates.** At each update it estimates three rates, in gates per move, from the counters since the last update. Each is smoothed with an EWMA of weight 0.3:

- `ghat`: net growth per MIX round;
- `shat`: net removal per COMP round;
- `dhat`: drift not caused by DB splices, which is mainly twists.

**Control law.** The controller solves `p·ghat − (1−p)·shat + dhat = v*` for p, where `v*` is the slope that reaches S\* one update ahead. It adds an integral term: with err = (s − S\*)/S\*, the term changes by −0.05·err per update and is clamped to ±0.3. Steps are limited to 0.1 per update, or 1.0 when |err| > 0.08, and there is a 0.02 deadband (`scheduling.rs:301-327`).

**Phases.** At most one phase transition happens per update (`scheduling.rs:262-289`):

- 1→2 when eff ≥ N0 or s ≥ 0.99·R1·s_in;
- 2→3 when eff ≥ N1;
- 3→4 when eff ≥ N2 or s ≤ 1.01·R2·s_in.

The three thresholds coincide here, so phases 2 and 3 each last at least one update interval.

**Observed in vL64a.**

- During the ramp, `pmix` went 0.5 → 0.4 → 0.1–0.24.
- Phase 1→2 came at eff 2.13 (size 1.53M ≈ 1.97×).
- Phase 2→3 came at eff 2.63 (1.72M ≈ 2.21×, an overshoot; vA64a peaked at 1,748,654 ≈ 2.25× at the same transition).
- Phase 3 ran COMP only (`pmix = 0`; twist rounds still run) and pulled the size back toward 2×.
- Final estimates: `ghat` = +1.79 gates per MIX round, `shat` = 0.198 gates removed per COMP round, `dhat` = +0.0047 gates per move. The `dhat` value matches the twist contribution, 0.0005 × 9.18 ≈ 0.0046.

**Stopping in practice.** A budget of 5.2·|G2| moves corresponds to eff ≈ 3.03–3.05 (3.03 for the config-A runs, 3.05 for L8). All eight validation runs therefore stopped on "moves budget spent" in phase 3, just before the eff 3.125 update that would have entered phase 4, so the final ratio is 2.04–2.14× rather than 2.0×. A run whose size arrives early ends on `ProfileDone` instead. In `W/rot8_128/s3.log`, the 1→2 transition fired at eff 1.63 on size arrival, 2→3 came at 2.13 and 3→4 at 2.63, and the run ended at 2.18×. That input was a rotating-segment stage-2 circuit, not the recommended configuration, and the notebook dropped that variant (`RESULTS.md:1722-1726`); the example shows only the mechanism. As run today, stage 3 is "about three passes, stopped by the budget", not "profile complete".

### 3.5 The DB move

#### Seed and window

**Geometry, length and seed.** A DB attempt (`mixer/replacement.rs:41-187`) draws in three steps:

1. **Geometry.** One coin per round: convex with probability 0.5 (MIX) or 0.95 (COMP), otherwise contiguous.
2. **Length.** COMP starts at the maximum (12 convex, 6 contiguous) and descends from there. MIX draws a length uniformly from {1,…,6}.
3. **Seed.** The seed is a uniformly random gate. The mixer makes up to 8 draws, looking for a gate whose width is below `w_window` (`sampling.rs:180-259`).

The seed's home position is recorded. If the attempt fails, `restore_seed` tries to bring the seed back. It searches left, then right, up to 512 commuting hops in each direction, and stops when home is its left neighbour or when a colliding gate blocks it (`sampling.rs:189-230`). A seed whose home lies to its right therefore first moves away from it. Gates moved aside while the window was built are not put back. Those moves preserve the function, so even a failed round can reorder the circuit.

**When two gates commute** (`src/circuit/xgate.rs:124-150`; the collision rule of §1.4). Two gates commute unless one reads the other's target. Even then, two comp = 0 conjunctions commute if they share a wire with opposite polarities, because they can never both fire. A g57 (comp = 1) collides whenever such a read occurs.

**Contiguous windows** (`sampling.rs:284-331`). The window extends from the seed g1 in g1's stored direction, switching once to the other side at the circuit end. A neighbour too wide to include is moved out of the way. If it cannot be moved, the window grows the other way. Each build allows at most 128 such evasions.

**Convex windows** (`sampling.rs:337-397`). These are built by repeated absorption:

1. Move g1 in its direction until it meets its first collider, and absorb that gate.
2. Find the next gate that collides with any member of the block. Move every commuting gate in between across the block (`transport.rs:116-128`), then absorb that gate.
3. Repeat step 2. The first growth step goes in g1's direction; later steps keep that direction with probability 0.75.

If both sides reach the circuit boundary, growth stops early. The result is a cluster of mutually dependent gates made contiguous by commutation. A contiguous window may contain unrelated commuting gates; a convex window concentrates on one dependency cluster.

#### Canonical key

The key depends on the window's function, not on its spelling or its wire labels (`src/canonicalization/xgate.rs:399-482`; `db/replacement.rs:756-775`):

1. **Relabel.** Collect the touched wires (more than 64 is an error) and renumber them 0..k−1. For the reverse direction, reverse the gate list: each gate is its own inverse, so the reversed list computes the inverse function.
2. **Compose the ANF exactly** (`src/canonicalization/xgate.rs:198-267`). Start with P_w = x_w. For each gate, update P_t ^= comp ⊕ ∏ literals, where a negative literal is 1 ⊕ P_w. Monomials are stored as u64 bitmasks. Exceeding the term budget gives `BudgetExceeded`.
3. **Degree check.** Reject the window if its maximum monomial degree exceeds `--db-max-degree 9` (`src/canonicalization/xgate.rs:501-509`).
4. **Canonical wire order** (`src/canonicalization/canonicalize.rs:1206-1311`). Iterative refinement on degree profiles, monomial frequencies and class polynomials splits the wires into ordered classes. Any remaining ties go to Rule L, which tries each member of the first tied class as the next wire, recurses, keeps the lexicographically smallest result, and prunes by automorphisms it discovers (`canonicalize.rs:1046-1173`). Trailing outputs with P_i = x_i, whose variable appears nowhere else, are trimmed (`canonicalize.rs:62-97`).
5. **Key.** Each output's sorted monomials are serialized as u64 little-endian, and each output is terminated by `u64::MAX` (`src/canonicalization/keys.rs:5-27`). The key is `xxh3_128` of that blob, taken as little-endian bytes (`db/replacement.rs:103-105`).

An exact in-process cache stores the results of steps 1–4, errors included (`src/canonicalization/xgate.rs:423-481`). A degree rejection is counted (`dsk`). Any other error silently drops that direction and is not counted (`db/replacement.rs:1083-1088,1213-1218`). That covers the term budget, more than 64 wires, and a canonicalization failure such as a Rule-L cap. `rule_l_skips` counts only the g57 path (`src/canonicalization/window.rs:241,409`). In every validation run `dsk = 0` and `ssk = 0`, so the degree cap and the span cap of 30 wires never fired. Whether the term-budget caps ever fired is unknown.

#### Lookup

Both directions are canonicalized. With `MIN_DIR_LOOKUP=Min` the mixer probes only the key of the lexicographically smaller of the two canonical forms, because the regular store is keyed by that minimum (`db/replacement.rs:1213-1282`; `src/database/lookup_cache.rs:112-136`). If only one direction composed, that one is probed.

The probe first goes through an exact in-process cache, which also stores misses and is cleared wholesale when it exceeds `LOOKUP_CACHE_MB` (`lookup_cache.rs:58-110`). It then reaches the frozen store (`src/database/frozen.rs:812-918`):

- **Key split.** The key splits into an 8-bit shard, a 20-bit bucket and a 48-bit tail, so 76 of the 128 bits are used (`frozen.rs:350-359`).
- **Bucket read.** Each shard has a table of 2^20+1 40-bit offsets, and bucket i spans `off[i]..off[i+1]` (`frozen.rs:12,385-388`). Equal offsets mean an empty bucket, which is a miss with no bucket read (`frozen.rs:824-828`). Otherwise the bucket is read from disk, its Elias–Fano-coded tails are scanned, and the matching value is Huffman-decoded.
- **Values.** A value is a chain of records `[len = 3m][m triples (t,b,c)]`, each record a g57 circuit (`docs/frozen_database.md:42-75`).
- **Candidates.** Candidates are listed as references without decoding them (`db/replacement.rs:1101-1155`).

#### Choosing a replacement

The choice is one uniform draw (`choose_ref_with_options`, `db/replacement.rs:1439-1552`):

- **MIX ("free, else pay at random").** If any candidate has at most as many gates as the window, choose uniformly among all of those. Otherwise choose uniformly among all candidates, whatever their length (`mix_pay_random = true`).
- **COMP.** Among candidates no longer than the window, let L be the shortest length, and choose uniformly among candidates of exactly length L. If no candidate is short enough, this attempt misses. COMP therefore never grows the circuit: it shrinks when it can and otherwise respells at the same length.

**Decoding** (`db/replacement.rs:188-272`, called at `1326-1336`):

1. If the probed key came from the reverse direction, the stored circuit spells the inverse function, so its gate list is reversed first (`db/replacement.rs:201-203`). Each gate is an involution, so this gives the forward function.
2. The circuit is mapped back through the canonical order and the window's wire list.
3. Canonical positions beyond the window's own wires are assigned to random wires the window does not touch.
4. A candidate whose target is one of its own controls is rejected.

**Identity guard.** A candidate is skipped if it equals the window exactly, or equals it after canonical gate ordering, i.e. it is just a commuting reorder (`db/replacement.rs:1315-1367`; `src/canonicalization/gate_order.rs:26-81`). After a skip, the choice is redrawn from the remaining candidates, and the free-versus-pay decision is made again. The guard exists because a no-op splice still uses up a round and stamps a new generation (`db/replacement.rs:685-691`). It fires often: vL64a skipped 5,763,348 candidates, 2,879,880 of them as reorders.

#### COMP prefix descent and MIX single shot

**COMP descent.** A COMP round tries a sequence of nested sub-windows ("rungs"), from the full window down to the seed alone (`mixer/replacement.rs:223-404`). Each step drops the end farther from the seed; on a tie it drops the left end. The round stops at the first rung that splices.

Example: in a 12-gate window with the seed at index 3, the right end is dropped five times ([0,11] down to [0,6]). After that the two ends alternate ([1,6], [1,5], [2,5], [2,4], [3,4]), and the last rung is [3,3], the seed alone. Other sub-windows that contain the seed, such as [1,11], are never tried.

So a COMP round replaces the first rung on this one fixed chain that has a non-identical, non-growing respelling. A rung also counts as a miss when it exceeds the span cap, or when every candidate is an identity or a reorder. In vL64a there were about 3.24M COMP rounds (4,049,743 DB rounds minus 808,897 MIX rounds). They produced 33.6M rung misses and 0.65M hits: about 10.4 failed rungs, or 10.6 rungs (lookups), per COMP round.

**MIX.** A MIX round makes a single attempt on the sampled window (`mixer/replacement.rs:406-555`).

Both modes skip windows that span more than 30 wires (`mixer/replacement.rs:291-312,411-433`).

#### Verification, splice and birth advance

**Verification.** Every candidate is proved equivalent before insertion (`mixer/replacement.rs:604-677`).

- If the window and candidate together touch at most 24 wires, the check is exhaustive over all 2^k inputs, evaluated 64 at a time (`src/engine/moves/rules.rs:278`).
- Otherwise it is an exact ANF comparison. A comparison that cannot be decided within budget declines the splice.
- A definite mismatch on a regular candidate panics with "DB replacement verification failed" (`mixer/replacement.rs:668-671`), because it would mean the store or the canonicalization is wrong. Older documents describe this case as a decline, which is out of date.

Every validation run had all supports at 24 wires or fewer (`wide = 0`, `wpoly = 0`), and verification cost 1.8 s in total in vL64a.

**Splice** (`mixer/replacement.rs:720-807`). The window's gates are unlinked, and the m replacement gates are inserted after the window's left neighbour. Each new gate gets:

- origin `ORIGIN_SYNTH`, unless the whole window shared one origin;
- generation = (upper median of the window's generations) + 1;
- a fresh litter id;
- a direction, split at a pivot.

The pivot is ⌊2m/3⌋ if the seed headed L and ⌊m/3⌋ if it headed R; gate i heads L if and only if i ≤ pivot (`mixer/replacement.rs:763,794`). The inclusive rule is stated in the code comment (`mixer/replacement.rs:758-762`). It biases products toward L: for m = 6 the split is 5 L / 1 R or 3 L / 3 R, and a single gate always heads L.

**Birth advance** (`db_advance`; `transport.rs:87-114`). Each new gate then moves away from the splice site. Its slack is the number of gates it can pass in its own direction before the first one it does not commute with. It moves ⌊0.85·slack⌋ positions, so a gate with slack 10 moves 8.

The parameter documentation gives two reasons for this step (`params.rs:325-334`):

- Without it, the directions assigned at the splice would never be used.
- It moves material without widening gates (the historic crossing walk widened them), and width is what makes store lookups miss.

The splice comment adds a third: scattering the litter "makes a later window less likely to be exactly this replacement" (`mixer/replacement.rs:808-811`). It does not prevent this. In vL64a, 2,225 of the 1,155,495 splices still took exactly one complete earlier replacement as their window (`litter … full=`; counted at `mixer/replacement.rs:748-751`, census at `provenance.rs:213-226`).

### 3.6 The g57 twist

With probability 0.0005 per round, a twist (`transformations.rs:967-1268`) swaps the labels of two wires a and b inside a long window.

**Window and wire pair.** The window length is drawn log-uniform on [64, size], then rounded and clamped. Windows that would run past the circuit head are truncated symmetrically there. There are at most 4 draws in total (`TG_RETRIES = 4`, `transformations.rs:180,985-1008`). Each draw builds its candidate pairs (a, b) from the pins of the two gates just outside the window edges. The wire `a` must touch the window, and at most 6 such boundary pairs are kept. One further pair (a0, c) is added, with a0 drawn from the wires the window touches and c uniform over all wires (`transformations.rs:1031-1063`).

**Seams.** Each end of the relabelled stretch is closed by an all-g57 "seam" word. S_ab is the swap of a and b. h_1…h_k (k ≤ 3, h_1 nearest the window, support at most 4 wires) are neighbouring gates absorbed into the seam. In circuit order:

- the left seam spells R_L = (h_k…h_1)·S_ab;
- the right seam spells R_R = S_ab·(h_1…h_k).

See `src/engine/moves/swap_words.rs:6-8` and `transformations.rs:777-888,956-958`.

**Seam solver.** The word is found by a meet-in-the-middle search over the 24 g57 gates on 4 abstract wires. The precomputed ball has radius 4 (165,443 permutations), words have at most 7 gates, and solutions are memoized (`src/engine/moves/swap_words.rs:1-56,181-296`). A bare seam with nothing absorbed costs 6 gates, since S_ab needs 6 g57 gates.

**Placement** (`transformations.rs:896-951,1098-1117`). The placer prefers the smallest net cost |R| − k. Sliding applies only to a seam with net cost ≥ 6. Such a seam may slide up to 512 gates outward (at most 3 candidate positions). The target must be a g57-shaped gate (comp = 1, two controls) whose pins include both a and b. The first slide reaching net ≤ 4 is taken. A plan is accepted outright at total net cost ≤ 8; otherwise the best plan of the 4 draws is used.

**Verification.** Inserted gates take the birth advance. The interior relabels (`conj_by_swap`) and the seam words are correct by construction from the swap-word BFS tables. Their local checks, the interior check (`transformations.rs:1146-1155`) and the seam checks (`1168-1183`), both run only with `local_verify`. `--no-local-verify` (`mod.rs:638`) turns them off, so in the recommended configuration twists are checked only by the sampled `global_check`.

In vL64a, 2,071 twists removed 4,934 gates and inserted 23,941, a net +19,007. That is about 9.2 gates per twist and 2.3% of the stage's net growth of 813,859 gates. The purpose is to change the physical wire frame over long stretches without inserting a recognizable swap gadget.

### 3.7 Final float, output and correctness

**Final float.** Unless `--skip-final-float` is given, each gate in turn is placed uniformly at random within the range it can reach by commutation (`transport.rs:49-74,163-174`), and then `global_check` runs (`mod.rs:901-911`). In vL64a this moved 1,174,172 gates in 1.4 s.

**Correctness** rests on four facts:

- every DB splice is proved equivalent before insertion;
- every other change of position (window builds, seed restores, birth advance, final float) is a move past commuting gates, which preserves the function by construction;
- twist relabels and seam words are correct by construction, and under `--no-local-verify` they are checked only by `global_check`;
- `global_check` runs every 10,000 moves and at the end. "At the end" means the `MovesBudget` exit or the post-float check. A `ProfileDone` exit relies on the post-float check alone, so with `--skip-final-float` it gets no final check.

`global_check` is a sampled test, not a proof (`reporting.rs:7-49`). It asserts that the merge index's gate count equals the arena length. It then evaluates the original and current circuits on 256 random full-width inputs and panics with "FUNCTIONALITY BROKEN" on any mismatch. It draws its inputs from the main RNG, so even with the same seed, changing `--verify-every` changes the run.

### 3.8 Measured behaviour

| runs | stage-2 config | gates in → out | ratio | wall time |
|---|---|---|---|---|
| vL64a / b | L8 | 779,195 → 1,593,054 / 778,926 → 1,602,569 | 2.04 / 2.06 | 2,052 / 2,049 s |
| vA64a / b | A | 777,215 → 1,647,349 / 778,107 → 1,660,680 | 2.12 / 2.13 | 2,114 / 2,108 s |
| vL128a / b | L8 | 2,235,297 → 4,596,808 / 2,232,492 → 4,586,553 | 2.06 / 2.05 | 5,902 / 5,896 s |
| vA128a / b | A | 2,222,024 → 4,757,723 / 2,234,767 → 4,787,862 | 2.14 / 2.14 | 5,945 / 5,976 s |

The `final64` and `final128` runs fall in the same range: 771,029 → 1,648,679 in 1,584 s, and 2,240,757 → 4,803,546 in 4,554 s. The config-A runs end larger because COMP removes less from their material (`shat` ≈ 0.14 versus 0.20).

| vL64a at end of run | value |
|---|---|
| moves / DB rounds | 4,051,814 / 4,049,743 |
| DB hits | 1,155,495 (28.5%) |
| MIX hits / rounds | 502,906 / 808,897 (62%) |
| COMP hits / rounds | 652,589 / 3,240,846 (≈20%) |
| splice growth / shrink (sum of per-splice net changes) | +1,460,377 / −665,525 |
| candidate choice: mean set size, P(more than one), mean log2 bits | 1.44, 0.322, 0.382 |
| time in canonicalization | 459 s canonical ordering + 119 s ANF, of 2,052 s; 56.7M calls, 68.3M Rule-L calls |

The accounting closes exactly: 779,195 + 1,460,377 − 665,525 + 19,007 (twists) = 1,593,054.

**Hit rate by window length.** MIX windows of length 1 to 5 hit 100%, 99.6%, 79.7%, 53.0% and 27.3% of the time. Length 6, which mixes MIX rounds with contiguous COMP rounds, hits 15.6%, and 12-gate convex COMP windows hit 20.3%.

**Single-gate windows** were always replaced by 4, 5 or 6 g57 gates (15,703 / 59,749 / 59,249 splices), never by 2–3 or 7 or more. Whether this reflects how `frozen_m1_m11` was built is not documented (`docs/frozen_database.md:85-89`).

**Choice entropy is low.** About 68% of splices had exactly one eligible candidate.

**More work buys little.** In the lab notebook's long-hold runs (profile `2,30,30,2,2`, under an earlier store configuration), heatmap ridge depth (§5.9; a null ridge is ≈ 0) fell only from about 0.314 to an asymptote near 0.290 between eff 2 and 30. The expansion factor mattered far more (`RESULTS.md:89-117`).

**Stage-3 artifacts handled downstream.**

- **Duplicate pairs.** About 7% of delivered gates are pairs of identical gates that commute next to each other; stage-4 repair cancels them (`STAGE2_LINKABILITY.md:103-106`; `RESULTS.md:1728-1741`). Which mechanism produces them has not been established.
- **C-view leaks.** On one artifact (rs64, K=64 retirement), strided C-view scans placed these leaks after stage 2. No C value was visible at every 16th stage-2 cut, while every 32nd cut of the mixed circuit showed them (`RESULTS.md:1371-1376`; §5.6). This was not repeated on the recommended configuration. Without retirement, which the recommended configuration does not use, the remaining fringe leaks come from raw input copies and band-seed products on C's first-layer control pairs. Mixer or packer reordering of same-target writes then exposes them. The notebook calls this mechanism plausible but not verified gate by gate (`RESULTS.md:1424-1452`). The leaks are mitigated by stage 2's `PREMASK` and `SEED_AVOID_C1` settings (`RESULTS.md:1462-1475`) and by stage-4 repair.

### 3.9 Reading the log

The main report line (`reporting.rs:228-373`) includes these fields:

| field | meaning |
|---|---|
| `comp=N` (first field, after `target=`) | number of comp = 1 (g57) gates in the circuit; equals `g57=` |
| `slot2=` | DB hits/rounds |
| `comp=h/m` (in the `db` block) | COMP hits / misses, with misses counted per rung |
| `agn=` | MIX hits/misses |
| `rm` / `add` | sum of per-splice net shrink / growth |
| `idsk` | identity-guard skips |
| `dsk` / `ssk` | degree-cap / span-cap skips (term-budget overflows are not counted) |
| `wide` / `wpoly` | verification counters for supports over 24 wires |
| `tsw` | twists |
| `osyn` | fraction of gates with synthetic origin (0.861 in vL64a) |
| `gen G=` | 5th-percentile generation |
| `litter … full=` | splices whose window was exactly one complete earlier replacement (counted, not enforced) |
| `choice …` | candidate-choice entropy |

After it come the `dminh`, `reorder-idsk` (cumulative reorder skips), `geo`, `splice sizes out->in`, `profile:`, per-length and `twist-g57` lines (`reporting.rs:384-474`). At the end of the run come the `canon caps:` line (g57-path counters only, so always zero on this path), `chain done` and `final float` (`mod.rs:872-911`).

Counter caveats:

- **`geo`.** Hit counts include only MIX hits, but they are divided by MIX plus COMP attempts.
- **Per-length table.** `attempts` and `hits` are keyed by the sampled window length, while `removed` and `added` are keyed by the length of the rung that spliced.
- **`dmin` and `choice`.** These can count identity candidates that were never drawn.
- **`G = 0`.** Every run ends with `G = 0`, meaning more than 5% of gates still have generation 0. With `gen_target = 0`, nothing steers seeds toward material that has never been rewritten, so stage 3 guarantees no full coverage. **[stale]** The statement in `docs/local_mixing_documentation.md` that MIX samples from a minimum-generation pool half the time does not hold for this configuration.

### 3.10 Open points

- **Stopping rule.** Runs stop either on the budget at eff ≈ 3 or on `ProfileDone`, and the `ProfileDone` exit skips the in-loop final `global_check`. Whether the ⌊N2·R1·1.3·|G2|⌋ budget is meant to stop the run in phase 3 is undecided (§3.1). The Rust port should define a single rule with a guaranteed final check.
- **Regular-only depends on the environment.** `--tdp` turns curated back on and overrides `--no-curated`, so regular-only holds only while `FROZEN_CURATED_DIR` is unset. The port should make the store choice explicit.
- **COMP knobs are not settable from the CLI** (defect, `mod.rs:171-207`). The inert-flag guard ignores `--profile`, so under the profile controller every typed COMP override exits with code 2, although COMP rounds run. The port should make the guard aware of `--profile`.
- **Direction pivot.** The inclusive `i ≤ pivot` is intentional (`mixer/replacement.rs:758-762`) and biases new gates toward heading left; a single gate always heads left. The effect on positional drift has not been measured.
- **Rule-L cap and cache sizes.** The reference driver inherits them from the shell, and the TDP driver pins them. Rule-L cap failures on this path are not counted.
- **Generation coverage.** Either the port enables generation targeting, or the documentation should state that stage 3 has no coverage guarantee.
- **Duplicate pairs.** Their source is unknown, and it is open whether stage 3 should cancel them itself.
- **Store coverage.** The code documents the caps as the store's census maxima from `frozen_degree_scan` (`params.rs:305-318`; `cli.rs:322-327`), but the census itself is not recorded in `docs/`. What `frozen_m1_m11` covers is stated inconsistently: `docs/CURATED_DB_V3_SUMMARY.md:14` calls it "the complete g57 replacement corpus through 11 gates", while `docs/frozen_database.md:84-86` warns that the directory name does not establish exhaustive coverage (§6.1). The degree and span caps never fired (`dsk = ssk = 0`). Term-budget overflows are not counted, so whether the 1,024/2,048 caps bind is unknown.
- **`FROZEN_FILTER=1`.** It does nothing on this machine because there is no `filters.bin`. §6.1 gives the build route.
- **Scheduled removals.** These are listed in `docs/notes/cleanup_inventory_20261007.md:1-28,149-153`: slot 3, `--p-comp`/`--p-any`, `temp`/`target_size`/`dir_p`, `--db-degree-probes` and the dead curated code paths.


## Stage 4: windowed ANF packing and repair

Stage 4 turns the mixed circuit G3 (`s3.mpmct1`) into the shipped artifact. That artifact is a sequence of windows. Each window is delivered as the canonical vector ANF of its permutation, and nothing else ships (`docs/ANF_WINDOW_DELIVERY.md:147-152`). The stage has three steps:

1. A joint reorder-and-cut pass, `window_pack3.py` (4.4).
2. A function-preserving repair pass that brings the C-view down to its floor, `repair.sh` (4.5).
3. Serialization, `anf_dump_eval.py` (4.6).

*Superseded (2026-10-09): stage 4 is now ported to Rust as `tdp_packing run` (`src/stages/packing/`, `docs/STAGE4_RUST_DESIGN.md`), bit-exact against these prototypes, which remain its canonical oracle. The rest of this paragraph describes the prototypes as they stood before the port.* Before the port, stage 4 existed only as Python/bash prototypes. The committed snapshot `scripts/windowed_delivery/` (commit `7163e457`) is described as "the reference oracle for the Rust port" (`scripts/windowed_delivery/README.md:1-9`). The snapshot and run-tree copies of every stage-4 script are byte-identical except `W/order_repair.py`, which was edited after the snapshot (4.5). `W/val_chain.sh` exists only in the run tree. Script paths follow the document-wide convention (Overview, "Conventions"): they always mean the run-tree copies under `RD/` and `W/`, which are authoritative.

### 4.1 Threat model and what windows buy

The adversary sees the full wire state at every window boundary, plus the inputs. It does not see the gates inside a window. All metrics are evaluated at every boundary of the final artifact, with no cut sampling (`STAGE2_LINKABILITY.md:25-28`). Equality and affine membership, however, are tested on random samples, not symbolically:

- the C-view uses 96×64 = 6,144 honest-slice samples with seed 12345 (`HA:112-117`, `HA:367-375`; §5.3);
- the packer, link census and `verify_equiv.py` use 1,024 all-wire samples (`RD/window_pack3.py:22-26,165-166`; `W/link_census.py:25-26`; `W/verify_equiv.py:5-9`).

Every exposure metric quantifies over the cuts of the delivered spelling, and a window has no interior cuts. So a value is exposed only if it is alive across a boundary (`docs/ANF_WINDOW_DELIVERY.md:225-234`). This has three effects:

- Read values become hideable. The earlier "reader law" assumed a cut between every pair of gates.
- The closure law is sidestepped, because cuts are deleted rather than rewritten.
- Templates, k-grams, gate provenance and the gate count all disappear into a canonical polynomial.

Four things remain observable: values alive at boundaries, the equality-linkage graph between boundary values, the outputs, and the window lattice itself. The lattice is new visible structure. The so-called recurrence law is an empirical observation, not a theorem. At migration dose 8, on one n=64 run and one n=128 run, every value with a stage-2 birth multiplicity of 5 or more was boundary-exposed; the largest multiplicity among non-exposed values was 4 (`docs/ANF_WINDOW_DELIVERY.md:290-294`; `RESULTS.md:1064-1066`).

Stage 4 is judged by three metrics. Their tooling is described in §5.

| metric | definition (as coded) | target | tool |
|---|---|---|---|
| C-view | for each value born by a gate of C: t\* = the fewest boundary wires whose XOR (plus a constant) equals it on the samples, minimized over the scanned cuts. t\* is exact when the nullity is ≤ 14 and an upper bound otherwise (`HA:597-616`) | exactly the floor: the n outputs (64 / 128) | `hmap_affine --exposure --g-cuts` (§5.3) |
| stage-2 exposure: **exposed (packer)** | distinct stage-2 values with a span crossing a boundary, or alive at the end, on the **pre-repair** pack | minimize (informational) | `window_pack3.py:488-502` |
| stage-2 exposure: **proxy values (link census)** | distinct stage-2 values seen in at least one run at the input cut or an interior boundary of the **delivered** lattice | minimize (informational) | `W/link_census.py:47-74,86` |
| linkability | per wire, a run is a maximal stretch of consecutive cuts in {input cut} ∪ interior boundaries where the wire holds the same function up to complement. The output cut is excluded: gates after the last boundary are never applied. L2 = proxy values (equal to some stage-2 gate output) with ≥2 runs; M2 = total runs over those values; linkable froth = non-proxy values with ≥2 runs. Froth includes input values that G2 never re-births, so froth is not strictly "mixer-made" | minimize (informational; reference ranges in §6.7) | `W/link_census.py:34-74` (§5.5) |

The documented definitions are at `STAGE2_LINKABILITY.md:30-51`. Where they differ, the code is the authority.

**"Exposed" has several senses; this document labels every figure with one of them.**

- **Vocabulary.** The set of distinct nonconstant stage-2 values, with v and ¬v counted once, as fingerprinted on 1,024 all-wire samples. In the packer this is `index`; its RESULT line prints it as "% of all distinct A" (`RD/window_pack3.py:63-71,498-502`). It is about 623k values (config A) or 712k (L8) at n=64, and 1.78-1.79M / 2.04M at n=128 (`W/v*/mult_s2.txt`, `distinct`).
- **Exposed (packer).** The RESULT count: carried stage-2 values with some boundary p, b+1 ≤ p ≤ e, inside one of their spans (birth b, next write e), plus every value forced at the output (`RD/window_pack3.py:488-497`). It is reported as "% of carried" or "% of vocabulary". It describes the pack **before repair**. Repair cancellation crosses boundaries and changes boundary states (§4.5), so this figure is not the delivered exposure.
- **Proxy values (link census).** The delivered stage-2 exposure: the number of distinct stage-2 values that occupy at least one run on `rep512` (`W/link_census.py:73,86`). It differs from the packer count by definition as well as by repair. It **includes** the input cut, where an input word that some stage-2 gate re-creates counts. It **excludes** the output cut, so values forced at the output count only if they are also seen at an interior boundary. `STAGE2_LINKABILITY.md`'s "exposed" column is this count.

Example, vL64a: 11,539 exposed (packer, pre-repair; `W/vL64a/pack_G512.txt`) against 11,743 proxy values (`W/vL64a/link_rep512.txt`). The two counts have not been reconciled value by value. A re-run of the packer's RESULT computation on `rep512` would separate the effect of repair from the effect of the definitions.

Scope limits:

- Only exact affine predictors and exact equality are measured. Low-degree nonlinear predictors have not been tested on final artifacts (`STAGE2_LINKABILITY.md:178-180`).
- Small windows are canonically re-synthesizable (`docs/windowed_delivery_design.md:208-210`).
- **Open modeling point.** The two kinds of measurement use different evaluation domains. The C-view evaluates the scanned circuit with x on wires 0..n and **zeros elsewhere** (`HA:3-8`, `HA:367-375`). Packer fingerprints, the link census, `verify_equiv.py` and `anf_dump_eval.py` use random values on **all** wires (`RD/window_pack3.py:165-166`; `W/verify_equiv.py:5-9`). An adversary holding the ANF can evaluate any full state. Whether the C-view should also be measured off the honest slice has not been decided (§5.1).

### 4.2 Inputs, outputs and formats

Inputs, per run directory (`W/val_chain.sh:15-29`). All three are secret (4.8).

| file | content | used by |
|---|---|---|
| `tdp.mpmct1` | G2 (unmixed) | packer value reference (`RD/window_pack3.py:162,168`), link census (`W/link_census.py:20-21`) |
| `s3.mpmct1` | G3 | packer input |
| `tdp.mpmct1.source_c.g57` | source C, written by `gen_sandwich_gadget` | C-view in repair (`W/repair.sh:18-19`; `W/anf_split2.py:26-29`) and the stage-2 span check (`W/val_chain.sh:26-27`) |

**`mpmct1`** is the format of §1.3; its action is `st[t] ^= comp XOR AND_i (p_i ? st[c_i] : NOT st[c_i])` (`src/circuit/xgate.rs:159-166`). The Python loaders do not agree on n:

- `window_pack3.py`, `anf_compose.py` (and so `anf_split2.py` and `anf_dump_eval.py`, which import it) and `boundary_cluster.py` (used by `link_census.py`) take n = max(header n, max wire + 1) (`RD/window_pack3.py:47`; `RD/anf_compose.py:47`).
- `order_repair.py` takes the header n and writes it back unchanged. `verify_equiv.py` imports that loader (`W/order_repair.py:25-44`; `W/verify_equiv.py:4`).

**`.bounds` format.** One integer p per line, with 0 < p < m. A boundary at p sits between emitted gates p-1 and p, and cuts 0 and m are implicit (`RD/window_pack3.py:305-322, 461`). The windows are `[cut_i, cut_{i+1})` over `[0] + bounds + [m]` (`RD/anf_compose.py:129-131`). A value born at gate b and overwritten at gate e is therefore exposed iff some boundary satisfies b+1 ≤ p ≤ e (`RD/window_pack3.py:493-495`). Values alive at the end are added at lines 496-497.

Outputs:

- `packed_G512.mpmct1` and `packed_G512.mpmct1.bounds`, the pack. The packer writes its lattice to `<out>.bounds` (`RD/window_pack3.py:483`).
- `rep512.mpmct1` and `rep512.bounds` (not `rep512.mpmct1.bounds`), the repaired circuit and its lattice (`W/repair.sh:11-12`).
- `deliverable.bin`, plus `.gz`/`.xz` copies (`W/val_chain.sh:29-33`). These are the only files that may ship.
- Logs: `pack_G512.txt`, `repair512.log`, `anf_rep512.txt`, `link_rep512.txt`, `deliver.txt`.

The scratch files `rep512.{wit,allcuts,exposure}.tsv`, `rep512.cview.log` and `split_cands_<pid>.{txt,allcuts.tsv,exposure.tsv}` are derived from C. The full never-ship list is in 4.8.

### 4.3 Reference driver and parameters

`W/val_chain.sh TAG N MC S SLICE CONFIG` runs the whole pipeline. CONFIG is `A` or `L8` (`W/val_chain.sh:2-5`). Seeds are drawn fresh from `secrets` (lines 11-12). The stage-4 part is three lines, run from `RD/` (`W/val_chain.sh:28-31`):

```
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1
bash $W/repair.sh $TAG 512 $N
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG
```

The driver still passes `--p-comp 1.0 --p-any 0.1` to stage 3 (line 21). Those flags are being removed (§3.2).

| parameter | CLI pos. | code default | production | role |
|---|---|---|---|---|
| `W_cap` | 3 | 64 | 512 | max interval size, in gates (b and e included) |
| `G` | 4 | 64 | 512 | soft gate cap per window (hard 2G inside a burst) |
| `B_mono` | 5 | 3072 | 24,576 (= 48·G) | soft monomial budget (hard 2·B_mono inside a burst) |
| `span_cap` | 6 | 3000 | 3000 | max spelled span length of a winnable value |
| `label` | 7 | `pack3-G<G>` | `$TAG-G512` | log label only: prefixes the first log line (`RD/window_pack3.py:158,171`) |
| `out_path` | 8 | none | `$R/packed_G512.mpmct1` | output circuit. The lattice is written to `out_path + ".bounds"`. **Without this argument nothing is written** (`RD/window_pack3.py:159,475-484`) |
| `MUL_CAP` | derived | 8·B_mono | 196,608 | product-size abort in `compose` |
| `PAD`, `CMAX` | const | 4, 24 | — | pre-emptive-close slack; contagion threshold |
| `LANES` | const | 16 | — | 1,024 fingerprint samples |
| `SEED` | const | 20261004 | — | fingerprint RNG (fixed, not secret) |
| split/scoring budget | `repair.sh` | — | 65,536 monomials per window | enforced only by `anf_split2.py` and reported by `anf_compose.py` (`W/repair.sh:31,36`). Not checked by `anf_dump_eval.py`, which serializes with budget `1<<30` (`W/anf_dump_eval.py:20`). |

Code defaults are at `RD/window_pack3.py:22-28, 153-159`. The B_mono = 48·G convention is from `W/score_arm.sh:10`.

**File-name contract between packer and repair.** `repair.sh TAG G N` takes no input paths. It rebuilds `R=$W/$TAG` and reads `$R/packed_G$G.mpmct1` and `$R/packed_G$G.mpmct1.bounds` (`W/repair.sh:9-14`). Its G is therefore only a file-name key: it must equal the suffix in the packer's argument 8, spelled exactly `$R/packed_G<G>.mpmct1`. It sets no window parameter. The split and scoring budget of 65,536 is a literal in `repair.sh` (`:31,36`) and does not depend on G. Outputs go to `$R/rep<G>.mpmct1` and `$R/rep<G>.bounds` (`:11-12`).

**Packer failure modes.** The scheduler can raise `RuntimeError("burst stuck")` (`RD/window_pack3.py:395`) or `RuntimeError("scheduler stuck: no ready gates")` (`:456`). Either aborts before any output is written, and neither appeared in the eight validation packs. Under `val_chain.sh`, which has no `set -e`, `repair.sh` would then run on a missing or stale `packed_G512.mpmct1`. Check `pack_G512.txt` for a traceback (§6.7).

**Why these values.**

- **G = W_cap = 512.** In the G sweep (packer v3, n=64, link-scored, `RESULTS.md:1520-1529`), each doubling of G from 128 to 512 cost "+4-8% size" and cut M2 by about 25% and linkable froth by about 60%. At n=128, a K=64-retirement arm cut L2 from 43,507 to 30,609 between G=128 and G=512, for +29% size (`RESULTS.md:1590-1592`). The earlier hidden-fraction law was about +14 points of carried values hidden per doubling of G (`RESULTS.md:537-542`). The lever saturates at G=1024: L2/M2 match greedy link-breaking deletions at G=512, and 7 windows exceed 65k monomials (max 386k) (`RESULTS.md:1543-1546`). W_cap = G lets an interval fill one window.
- **B_mono = 48·G = 24,576.** This keeps the monomial budget proportional to the gate cap (`W/score_arm.sh:10`). The notebook records that the ANF budget barely binds at the smaller caps: no overflow at G=128 and one in 3,961 windows at G=256 (`RESULTS.md:472`, `:522-524`). No sweep of the 48 factor is recorded.
- **Split and scoring budget 65,536.** No derivation is recorded. The G=128-era emission budget was 16,384 (`docs/ANF_WINDOW_DELIVERY.md:118`). By the G=512 sweep, 65,536 was in use, and 2-3 packed windows per G=512 run exceeded it (`RESULTS.md:1524-1527,1576`). *Inferred:* it is kept separate from B_mono because B_mono is only a soft scheduling budget that single gates overshoot (§4.4, "Caps can be overshot").
- **span_cap = 3000.** This is the packer's code default (`RD/window_pack3.py:156`). No sweep or rationale is recorded. At G=512 it disqualifies 7.1-7.9k values at n=64 and 24.8-27.9k at n=128 as "long-span" (line 2 of `W/v*/pack_G512.txt`).

### 4.4 Packing: `window_pack3.py`

**Values and spans.** `eval_circuit` evaluates G2 and G3 gate by gate on bit-sliced 1,024-bit integers. Each written value is complement-normalized: if bit 0 (the first sample) is set, the value is complemented, so v and ¬v count as one value (`RD/window_pack3.py:52-61`). Constants are skipped.

A value is keyed by the blake2b-128 digest of its 1,024-bit value. The value id is its **first** birth position in G2 (`RD/window_pack3.py:63-71`).

During the pass over G3, every write to wire t closes the pending span on t, recording `(birth_pos, i, a_id)`. If the new value matches a G2 value, a new span opens. Spans still open at the end are `forced`, meaning alive at the output (`RD/window_pack3.py:76-88`). `agree` counts the wires whose final value in G3 equals the one in G2 (line 89), so "512/512" in the log counts wires, not samples.

A value is hidden iff none of its spans contains a boundary. This is all-or-nothing per value.

**Intervals.** `interval_set(B, b, e, n, cap)` (`RD/window_pack3.py:92-133`, with `B` = G3) computes the gates that must lie between b and e in every commuting reorder. That set is the value's minimal bare residency, its **DAG interval**. On stage-3 material the median is 6 gates, 21 times shorter than the spelled span (`docs/ANF_WINDOW_DELIVERY.md:250-258`). The computation is a two-sided taint closure:

- **Forward sweep from b.** Start with TW = {target(b)} and TR = controls(b). A gate in (b,e) is tainted if its target is in TR (WAR) or any of its controls is in TW (RAW). Two gates writing the same target do not taint each other, because same-target writers commute.
- **Backward sweep from e**, over forward-tainted gates only. Start with TRb = {target(e)} and TWb = controls(e).
- **Result.** Members are forward ∩ backward, plus b and e. The function returns `None` once the member count would exceed `cap` (line 127).

A value is **winnable** if it is not forced and every one of its spans has length ≤ `span_cap` and an interval within `W_cap`. When an interval is too big, one rescue is tried: extend e to the next writer of the same wire after e, provided the extended span is still ≤ `span_cap` (`RD/window_pack3.py:179-213`).

**Commutation and readiness.** Pairwise RAW/WAR edges are unsound for MPMCT gates (`docs/ANF_WINDOW_DELIVERY.md:25-31`), so readiness is tracked by epoch (`RD/window_pack3.py:228-246`):

- A reader of wire c waits until `writes_done[c]` reaches the number of writes to c that preceded it in G3.
- A writer of t waits until every earlier reader of t has been emitted.

Any ready gate may be emitted at any time without changing the function.

**Scheduling.** The scheduler state is set up at `RD/window_pack3.py:248-273`. `prot[g]` counts the living, unstarted intervals that contain g. The heap `filler` holds ready gates with `prot == 0`, and `anyready` holds all ready gates, keyed by original index.

Intervals are processed in ascending order of their end e (`RD/window_pack3.py:412-445`). For each one:

1. **Skip** it if it is dead. If it was already started by an earlier burst and is still committed, finish it with a burst plus contagion.
2. **Drain.** Emit `filler` gates with index ≤ e under soft caps.
3. **Pre-emptive close.** Close the window if `G − win_load < isize + PAD` or `win_monos > ¾·B_mono` (line 436). This keeps the coming burst from being split by a scheduled close.
4. **Atomic burst.** Emit all members in ascending order. When a member is not ready, pop the lowest-index ready gate from `anyready` and emit it first. This dependency pull can start other intervals (`RD/window_pack3.py:387-398`).
5. **Contagion.** Repeatedly burst committed intervals with ≤ `CMAX` = 24 gates remaining, smallest first, stopping if a window closes (`RD/window_pack3.py:400-410`).

After the interval loop comes a sixth step:

6. **Tail flush.** All remaining gates are emitted in lowest-index ready order under soft caps. `anyready` is rebuilt by a full scan if it runs empty. The last window is then closed with reason `final` (`RD/window_pack3.py:446-460`).

Window closing is budget-adaptive. Before a gate is added, and only if the window is non-empty, `schedule` closes the window if `win_monos` exceeds B_mono (2·B_mono inside a burst) or if `win_load` reaches G (2G inside a burst) (`RD/window_pack3.py:346-353`).

`win_monos` is the monomial count of the window's incrementally composed sparse ANF. `compose` represents a monomial as an integer bitmask over global wires (`RD/window_pack3.py:324-344`):

- A negative control toggles the constant monomial; so does `comp`.
- The target becomes `tgt XOR prod`.
- `mul` toggles each product monomial pairwise and gives up past `MUL_CAP` (lines 135-150). In that case the window is closed right after the gate.

`close_window` kills every committed interval that is still incomplete, together with all sibling spans of its value (`RD/window_pack3.py:296-322`). The log attributes each death:

- `burst_close`: the window closed inside a burst.
- `drain_close`: any other close. This covers a close during a drain, the pre-emptive close (line 437), and a compose-failure close outside a burst (line 358).
- `sibling` (line 301): the value lost another span. The module docstring calls this `sibling_kill` (line 14); the port should use one name.
- `final` (line 460) and `resumed_later` (lines 372-375; `docs/ANF_WINDOW_DELIVERY.md:84-87`) are still counted in code, but neither appears in the eight validation logs. `resumed_later` appears unreachable in v3.

**Verification and statistics.** After emission, `proxy_spans(A, newB)` (G2 against the reordered G3) is re-run and the final states are compared. This prints `VERIFY: … OK/FAIL` but does not abort, and the outputs are already written by then (`RD/window_pack3.py:470-487`). `RESULT:` reports "exposed (packer)" values against this ground truth, for the emitted pre-repair order (§4.1). Repair later changes boundary states (§4.5).

The v3 logs do not print a like-for-like comparison of the two counts:

- "fully hidden by construction" counts winnable values only;
- carried − exposed counts all values carried in the emitted order, including non-winnable values that happen to be hidden.

Across the eight validation packs, carried − exposed differs from the by-construction count by −149 to +795 values, in both directions. That is at most about 0.2% of carried. Reordering also shifts the carried set itself by −118 to +753 values (for example vA128a: 423,655 → 424,408) (`W/v*/pack_G512.txt`). The often-quoted exact match, 37,118 vs 37,117, was measured on packer v2.1 at W=20, not on v3 (`RESULTS.md:408,422-431`).

Typical production output, from `W/vL128b/pack_G512.txt`:

- 446,017 of 473,945 values with interior spans are winnable (473,992 carried in total, including output-forced values);
- 439,877 are fully hidden by construction;
- 34,148 are exposed (packer, pre-repair; 7.2% of carried);
- window gates median 525, p90 1,024, max 1,024;
- window monomials median 544, max 92,364.

**Quirks.**

- **Wire classes.** `wire_class` hard-codes data < 64, mid < 128, band ≥ 128 (lines 49-50). The per-class lines are therefore mislabeled at n=128.
- **Compose-failure close.** The close happens after g is appended but before g's interval bookkeeping (lines 353-371). This makes hidden-by-construction inexact in both directions for g's intervals:
  - Intervals that g starts are counted in the new window although g precedes the boundary. This is optimistic.
  - Intervals for which g is the last member are killed although they completed. This is pessimistic.
  - The VERIFY ground truth re-measures both cases.
- **Duplicate boundary.** A pre-emptive close on an empty window appends a duplicate boundary (lines 436-437).
  - It is reachable only for intervals with ≥ G − PAD + 1 = 509 members, which W_cap = 512 allows.
  - It was not observed: none of the eight pack `.bounds` files contains a duplicate.
  - It is harmless in the current pipeline, because repair output is deduplicated (`W/order_repair.py:48,60`; `W/anf_split2.py:55`).
  - Run on a raw pack, `anf_compose`/`anf_dump_eval` would emit an empty window (a 4-byte blob).
- **Caps can be overshot.** Caps are checked before a gate is added, so one gate can push a window past 2·B_mono = 49,152 monomials (vA64a's packed windows reached 77,252). Repair splits only windows above the 65,536-monomial split budget, and never windows of ≤ 2 gates (`W/anf_split2.py:38`). Windows between 49,152 and 65,536 ship as they are.
- **Determinism.** The pack is a deterministic function of (G2, G3). Ties are broken by dict-insertion order and min-index heaps. Boundary randomization or window-size jitter is listed as undecided (`docs/windowed_delivery_design.md:207-210`).

### 4.5 Repair: `W/repair.sh TAG G N`

The packer works only on stage-2 values. The residual C leaks appear only after reordering, in stage 3 and/or the packer. On one artifact (rs64, which used K=64 retirement) they were localized to the mixer. This was not re-checked on the recommended configuration (§3.8, §5.6; `RESULTS.md:1373-1376`). In the 29-variant study, the span check was zero at every scanned stage-2 cut: every 16th cut at n=64 and every 128th at n=128 (`W/span_jobs.txt`; `W/span_all.sh:6`; `STAGE2_LINKABILITY.md:60-67`; §5.6). The replicate driver repeats the check with strides 64/256 (`W/val_chain.sh:25-27`).

Repair removes the residue. It is organized as a sequence of function-preserving rewrites of the packed order, in which boundaries are markers that stay between the same surviving gates (`W/order_repair.py:1-17`). Two gates commute iff neither's target is a control of the other (lines 66-67). "The same surviving gates" does not mean the same boundary **states**. Cancellation and slides move or delete gates across markers, so the wire state at a boundary can change (step 1).

The script uses `set -uo pipefail` without `-e` and hard-codes `ROOT` and `W` (`W/repair.sh:5-7`). Its steps:

1. **Pair cancellation** (`order_repair.py … cancel`, line 14). For each gate g at position i, `blk` is the first later gate that reads g's target or writes one of g's controls (`W/order_repair.py:88-90`). The next identical gate j with i < j < blk is cancelled together with g, since g is self-inverse and everything between them commutes with it. Passes repeat to a fixed point (lines 101-112). This removes about 7% of delivered gates. For example, vL128a went from 4,596,808 to 4,259,448 gates (168,680 pairs) (`W/vL128a/repair512.log`).
   - **Cancellation crosses boundaries.** `cancel_pass` runs on the bare gate list. `run_cancel` strips the markers before calling it and only afterwards drops the dead gates from the marker sequence (`W/order_repair.py:69-99,101-112`). A pair is therefore cancelled even when boundaries lie between i and j. At every such boundary the state changes: g's increment, applied at i and undone at j, is no longer present. The overall function is unchanged.
   - **Consequence.** The packer's "fully hidden by construction" and RESULT "exposed (packer)" figures (§4.4; "pack exposed" in §4.7) describe the **pre-repair** lattice, not the shipped one. The delivered stage-2 exposure is the link census's `proxy values` on `rep512` (§4.1, §5.5, §5.6). Re-running the packer's RESULT computation on `rep512` would give a like-for-like post-repair count, but this has not been done.
2. **C-view scan.** `cview()` (lines 17-21) runs `hmap_affine --exposure --c-step 1 --g-cuts rep$G.bounds` with `EXPOSURE_WITNESS` and `EXPOSURE_ALLCUTS` set (§5.3). That binary requires `--features security-tools` (`Cargo.toml:308-311`). It uses 6,144 samples, seed 12345 (`HA:112-117`). Per cut, it deduplicates the scanned circuit's wire columns under complement and builds a GF(2) basis; a C value is visible iff it reduces to zero (`HA:512-590`). t\* is minimized over the nullspace coset by Gray code when the nullity is ≤ 14; above that, the value is an upper bound and is reported as "coset-minimizations skipped" (vL128a v1 skipped 1; `HA:597-616`). Only the best cut per value is kept as witness, with ties going to the earliest cut (`HA:619-633`). `cview()` returns the number parsed from line 2 of the log.
3. **Fix loop**, at most 4 iterations (lines 22-26). It stops when `vis ≤ N`. Otherwise it runs `order_repair.py … fix wit.tsv exposure.tsv | tail -1`. The `tail -1` keeps only the gate/boundary count line and drops the `moved N gates` line (`W/order_repair.py:185,189`), so the logs cannot show whether a stalled pass moved anything.
4. **Synchronous equivalence check** (line 27; see below).
5. **Fallback loop**, at most 4 iterations (lines 28-33). If `vis > N`, run `merge_bounds2.py`; then always run `anf_split2.py` with budget 65,536. Stop when `vis ≤ N` and zero splits were added. The `fallback iter` lines of `repair512.log` are the ones to check.
   - **`anf_split2.py` interface.** The call is `anf_split2.py TAG G N BUDGET bounds_in bounds_out` (`W/anf_split2.py:5,9`), but two required inputs come from the environment, not the CLI. `SPLIT_P` is the circuit whose windows are split; it must be the **repaired** circuit `rep<G>.mpmct1`. `SPLIT_EXP` is the C-view `exposure.tsv` from which the C outputs (the last C gate per wire) are taken; it must be the **current** `rep<G>.exposure.tsv`. `repair.sh` sets both (`W/repair.sh:31`).
   - **Dangerous defaults.** Without these variables, the script falls back to `$R/packed_G<G>.mpmct1`, the unrepaired pack (`W/anf_split2.py:12`), and to `$R/fin.exposure.tsv` (`:18`), the output map of the old `finalize3.sh` flow, which is stale or missing in a new run directory. Called outside `repair.sh`, or ported from its CLI alone, it would split windows of the wrong circuit against the wrong output map. `R` is rebuilt from TAG as `$W/$TAG` (`:11`), and scratch files `split_cands_<pid>.*` are written there (`:23-28`).
6. **Scoring.** `anf_compose.py … 65536 … 500` and `link_census.py` (lines 36-37).

`verify_equiv.py` checks the packed circuit against the repaired one on 1,024 random full states. It runs twice: in the background right after cancellation, and in the foreground after the fix loop (lines 15-16, 27). The background check is best-effort only. It reads `$P` while the fix loop may be rewriting `$P` in place, and `save` truncates the file on open (`W/order_repair.py:40-44`). It can therefore see a fix-modified or partly written file. The fallback changes only bounds, so it needs no equivalence check.

**Completer/breaker slides (fix mode).** Each witness row gives a C value, its best cut (as a gate count), and a predictor support S (`W/order_repair.py:148-152`). The relation holds from the **completer** (the last write to S before the cut) to the **breaker** (the first write to S after it). The fix slides the completer forward and the breaker backward as far as commutation allows (`W/order_repair.py:114-137`).

`slide` stops at the first non-commuting gate. It crosses boundary markers only when it also passes at least one commuting gate. A gate whose blocker sits right after a marker is not moved across that marker (`dest == pos`, lines 127-128), even though crossing it would break the relation at the cut. This is a candidate cause of fix-loop stalls and should be fixed in the port.

Example: an n=128 bounded-chain fringe leak (five wires XOR to the C value) had its completer at gate 8,507 and its breaker at gate 11,083. The leak needed a window of about 2,600 gates and was resolved by slides alone (`RESULTS.md:1737-1741`).

Output gates of C are excluded by identity: `outs` is the last C gate per wire, taken from `exposure.tsv` (lines 140-146).

There are two versions of fix mode:

- **Snapshot.** `scripts/windowed_delivery/waist/order_repair.py:155-170` re-finds the breaker in `gpos[cut:]` after the completer move (lines 163-170). When the completer has crossed the cut, that gate-count index is stale by one, so the completer itself can be re-selected as "breaker" and slid back.
- **Live.** The run-tree copy `W/order_repair.py` (last modified 2026-10-08 08:24) keeps the pre-move breaker index `brk_i`, which stays valid because the pop and insert shifts cancel past the completer's landing point, and drops it if the completer passed the breaker (`W/order_repair.py:158-171`). `val_chain.sh` now calls this copy, but only the vL128a rerun ran with it (4.7).

Both versions share three further limits:

- **Cross-row cut staleness.** `cut` is a gate count read from the witness file of the preceding scan (line 152). Earlier rows in the same pass can slide gates across a later row's marker, so `gpos[:cut]`/`gpos[cut:]` (lines 156-159) can be off.
- **One cut per value per pass.** Fix acts only on each value's single witness cut, even when `allcuts.tsv` shows the value at several cuts.
- **Lost move count.** The moved count is discarded by `tail -1`, as noted in step 3.

**Canonical version: the live `W/order_repair.py`.** It is the copy the reference driver calls (`W/repair.sh:14,25`), and the snapshot's breaker re-find is a known bug. Its validation is pending. Of the replicate figures in §4.7 and §5.6, only the vL128a rerun was produced with it; the other seven used the snapshot. The validating step is to re-repair those seven with the live version (`repair.sh`, then `anf_dump_eval.py`) and compare. Stalls occur under both versions: the snapshot stalled at 65 (vL64b), 129 (vA128b) and 131 (vL128a v1); the live version stalled at 130 (vL128a rerun).

**Merge and exposure-aware split.**

- `merge_bounds2.py` deletes every boundary at which a non-output C value is exactly affine (`W/merge_bounds2.py:6-19`). Removing cuts is a lattice subset, so nothing can become newly visible (`RESULTS.md:1400-1406`); the cost is larger windows.
- `anf_split2.py` splits each window whose ANF exceeds the budget (product > 8B, running total > 4B, or final > B). Windows of ≤ 2 gates are never split. It takes 32 evenly spaced candidates `lo+(hi−lo)·i//33`, runs one C-view on them, and picks the exposure-free candidate nearest the midpoint, then recurses (`W/anf_split2.py:22-55`).
- If no candidate is clean, it prints `WARNING: no exposure-free split` and splits anyway (`:44-46`).
- The earlier midpoint splitter (`anf_split.py`) re-created exposing cuts and oscillated (`RESULTS.md:1593-1600`).

**Failure modes.**

- **Count-based stop test.** The test is `vis ≤ N` (lines 24, 32), and the scans cover only interior bounds (`RD/window_pack3.py:461`; `HA:331-340`).
  - `vis ≤ N` certifies the floor only if all N outputs are visible at some interior boundary. Otherwise a non-output leak can hide under the count: for example, N−1 visible outputs plus 1 leak still passes.
  - An identity-based test, "no visible row whose gate is not in `outs`", is already computed by `merge_bounds2.py:6-15` and `anf_split2.py:17-21,34`. It should replace the count.
  - In all eight final scans, the visible set was exactly the N last-writer outputs, all with t\*=1. vL128a v1's last scan had 128 outputs plus 1 interior value.
- **No final rescan when the fallback is exhausted.** On a normal break (vis ≤ N and 0 splits) the last scan covers the final lattice. If the 4 fallback iterations run out, the final merge/split is never rescanned and the script still scores and emits. The "129" that `repair.sh:39` prints for vL128a v1 is from the scan before the last merge and split.
- **Silent scan failure.** Without `-e`, a failed `hmap_affine` leaves `vis` empty. The `[ "$vis" -le "$N" ]` test then errors and evaluates false, so the loops go on to fix, merge or split using stale TSVs.
- **Failed original run.** In the original vL128a run (`W/vL128a/rep512_v1/repair512.log`):
  - the fix loop went 132, 131, 131, 131;
  - fallback iteration 1 (131 visible) merged 3 boundaries and added 3 splits, one of them forced (`WARNING: no exposure-free split in [7349,8962)`); iterations 2-4 each scored 129, merged 1 boundary and re-added a forced split in the same range;
  - the last scan showed 129 with 1 coset skip, still above the floor, and the final lattice was never scanned;
  - a `deliverable.bin` was produced anyway.
- **Rerun.** The rerun with the live `order_repair.py` finished at 08:44 (`W/vL128a/repair512.log`):
  - cancellation was identical, and the fix loop went 132, 130, 130, 130;
  - fallback iteration 1 merged 1 boundary and added 1 clean split, and iteration 2 scanned 128 with 0 splits, so the floor was reached via merge;
  - ANF size 98.2 MB; L2 16,106; M2 187,248; froth L2 5,579.
- **Stale deliverable.** `W/vL128a/deliverable.bin` (01:35) still comes from the v1 lattice. It has not been re-serialized.
- **Concurrent runs.** Two repairs of the same TAG/G collide on the `rep$G.*` scratch names.

### 4.6 Serialization and evaluation

**Window ANF.** `window_anf(gates, lo, hi, budget)` (`RD/anf_compose.py:66-91`) composes each touched wire's new value as a polynomial over the **window-input** values. It aborts on a product larger than 8·budget, a running total larger than 4·budget, or a final total larger than budget.

`changed` keeps only wires whose polynomial differs from {x_w} (`RD/anf_compose.py:87`), so a wire that is written and then restored inside the window is not emitted. Pairwise toggling in `mul` is essential: the older `symmetric_difference_update(generator)` deduplicated instead of cancelling (`RESULTS.md:416-419`).

Worked example: a window with gates `2 0 2 0 1 1 1` then `3 0 1 2 0`.

- The first gate sets wire 2 to x2 + x0·x1.
- The second gate has a negative control on wire 2. Its product is 1 + x2 + x0·x1, so wire 3 becomes x3 + 1 + x2 + x0·x1.
- Result: changed = {2, 3}, support = {0, 1, 2, 3}.

**`deliverable.bin` format** (`W/anf_dump_eval.py:14-50`, little-endian). The file is a concatenation of `[u32 blob_len][blob]`, one per window, with no file header: no n, no window count and no format version. Each blob contains, in order:

1. `u16 f` (support size: the union of wires that appear in any monomial) and `u16 c` (changed count);
2. f × `u16` support wire ids, ascending;
3. for each changed wire, ascending: `u16 wire`, `u32 count`, then count × ⌈f/8⌉-byte little-endian local bitmasks over support indices.

Monomials are sorted by global mask. Because the global-to-local map is monotone, that is the same as sorting by local mask. The all-zero mask is the constant 1. Every field is a function of the window's permutation, so the blob is canonical.

f = 0 occurs only with c = 0, for an identity or empty window. A changed wire with empty support would be a constant output coordinate, which a permutation cannot have. So the `max(f,1)` mask width (lines 31-32) is never used.

Worked example: a window whose only change is `x17 ← x17 ⊕ x5·x130 ⊕ 1`.

| field | bytes |
|---|---|
| `f=3`, `c=1` | 4 |
| wire ids 5, 17, 130 | 6 |
| `wire=17`, `count=3` | 6 |
| masks: `00` (the 1), `02` (x17), `05` (x5·x130) | 3 |

The blob is 19 bytes, so the window takes 23 bytes with its length prefix.

To evaluate a window, snapshot the support values, set each changed wire to the XOR over its monomials of the AND of the snapshot bits, and leave all other wires unchanged (lines 92-108).

**Evaluating `deliverable.bin` (consumer side).** No decoder exists in the repository. The procedure below is *derived* from the writer (`W/anf_dump_eval.py:14-50`) and the in-memory evaluator (`:92-108`). It has never been run on a decoded file.

1. **Inputs needed out of band.** n. The file has no header, so the width 4n cannot be read from it. The largest wire id that appears need not be 4n−1, because a wire no window touches never appears. Allocate a state of 4n bits; wires that never appear stay 0 and are never read.
2. **Slice setup.** Put x on wires `0..n`, and set wires `n..4n` (the answer register y and the band) to 0.
3. **Decode loop, in file order** (little-endian throughout). Repeat until end of file:
   1. Read `u32 L` and then the L-byte blob. Inside the blob, read `u16 f` and `u16 c`, then f × `u16` support ids `s_0 < … < s_{f−1}`.
   2. Snapshot `v_i = state[s_i]` for all i.
   3. For each of the c changed wires: read `u16 w` and `u32 k`, then k masks of ⌈f/8⌉ bytes each. Bit i of a mask selects `v_i`, and the all-zero mask is the constant 1. Set `state[w]` to the XOR over the k masks of the AND of the selected `v_i`.
   4. Every wire outside the c changed wires keeps its value. A blob with f = c = 0 is an identity window, and empty windows only arise from the duplicate-boundary quirk (§4.4).
4. **Readout.** Wires `n..2n` hold C(x). Wires `0..n` and `2n..4n` are junk.

A versioned header carrying n and the window count, a reference decoder, and a round-trip test against `rep512.mpmct1` and against C (`verify_zero_slice`) are pending (§4.9, §5.7, §6.12).

**Checks and size model.**

- `anf_dump_eval.py` compares the gate-form and ANF-form final states on 1,024 random full states (`match=`, line 110). It does this on the in-memory windows, not on a decode of `deliverable.bin`, so the format has no round-trip test and no decoder exists. Lines 69-90 run a full placeholder ANF evaluation pass whose result and timing (`ta_wrong`) are discarded (`st2` is reset at line 92).
- `anf_compose.py` spot-checks only windows with index ≡ 0 mod 500 (line 159), on 256 samples. A failure prints but does not abort.
- Its size model is `8 + 8f + 24·|changed| + total·f` bits per window (line 155). It underestimates the real bytes for two reasons. It counts masks in bits rather than whole bytes. It also assumes 8-bit ids, 8+16-bit per-changed-wire fields, and no length prefix; at n=128 (512 wires) an 8-bit id could not even address the wires. Mask padding is the largest term.
  - For vA64a (decode of `W/vA64a/deliverable.bin`): the model gives 24.60 MB; the real file is 26.04 MB = masks 24.38 + per-changed-wire fields 0.94 + support ids 0.69 + length prefixes and headers 0.02.
  - The 1.44 MB gap ≈ mask padding 0.60 + per-changed fields 0.47 + support ids 0.34 + headers 0.02.

### 4.7 Validation replicates (G=512)

**Update (2026-10-08/09).** All eight replicates were re-repaired with the
canonical `order_repair.py` and re-serialized on Oct 8; the earlier repairs
are in `W/<TAG>/rep512_v1/`. The parity targets of the Rust port are the
current `rep512.*` and `deliverable.bin` (`STAGE4_RUST_DESIGN.md`
Appendix B), and `tdp_packing run` reproduces every one of them byte for byte
from the archived `tdp.mpmct1` and `s3.mpmct1` (`tests/manual/stage4/run_parity.sh`).
The **historic anchor** re-ran the chosen runs from their archived seeds:
gen at `492564f0` reproduces `tdp.mpmct1`, the source and the sandwich of all
eight tags; the mixer at `492564f0` reproduces `s3.mpmct1` of vL64a and
vL128a; and the managed driver on this branch reproduces every artifact of
vL64a and vL128a down to `deliverable.bin`. The fix-mode paragraph below
describes the first repairs and is superseded.

All runs used the recommended stages 1-3 and `val_chain.sh` (logs in `W/v{A,L}{64,128}{a,b}/`). Across all eight runs:

- the packer's input and emitted function checks passed (256/256 or 512/512 wires);
- `verify_equiv.py` reported EQUIVALENT;
- 0 windows exceeded 65,536 monomials after repair;
- spot checks had 0 failures;
- ANF evaluation reported `match=True`.

**Fix-mode version.** Judging by file times, every repair except the vL128a rerun used the snapshot fix mode (`scripts/windowed_delivery/waist/order_repair.py:155-170`). Those repairs finished on Oct 8 between 00:01 and about 01:35 (the n=64 runs by 00:04; the vL128a v1 deliverable at 01:35), after the snapshot commit (`7163e457`, Oct 7 23:21) and before the live `W/order_repair.py` was last modified (Oct 8 08:24). The vL128a rerun (08:44) used the live version. Its cancellation output is identical to v1 but its fix trajectory differs (132→130 vs 132→131), which confirms that the code changed between the two.

Paths to the floor:

- floor at the first scan: vA64a, vA128a, vL128b;
- by fix: vA64b (iteration 3), vL64a (iteration 2);
- fix stalled, then merge: vL64b, vA128b, vL128a rerun;
- failed: vL128a v1.

Linkability results for the same runs, including the delivered stage-2 exposure (proxy values), are in §5.6. Stage-1 to stage-3 figures are in §6.9. Evaluation times are Python timings, not performance claims. Only the vL128a rerun row used the canonical (live) `order_repair.py` (§4.5).

| run | pack windows | exposed (packer, pre-repair; % carried) | C-view trajectory → final | final windows | ANF est. MB | `.bin` / `.xz` MB | ms/input gate / ANF |
|---|---|---|---|---|---|---|---|
| vA64a | 2,774 | 15,060 (10.3%) | 64 → 64 | 2,776 | 24.6 | 26.0 / 2.81 | 0.72 / 0.77 |
| vA64b | 2,793 | 15,054 (10.3%) | 65, 65, 64 → 64 | 2,793 | 25.6 | 27.0 / 2.86 | 0.75 / 0.85 |
| vL64a | 2,586 | 11,539 (7.0%) | 65, 64 → 64 | 2,588 | 31.2 | 32.8 / 3.10 | 0.68 / 1.00 |
| vL64b | 2,600 | 11,512 (7.0%) | 66, 65, 65, 65, merge → 64 | 2,600 | 28.7 | 30.2 / 3.06 | 0.63 / 0.88 |
| vA128a | 7,376 | 53,168 (12.5%) | 128 → 128 | 7,377 | 102.1 | 106.9 / 9.81 | 1.65 / 1.86 |
| vA128b | 7,415 | 53,558 (12.5%) | 129 ×4, merge → 128 | 7,416 | 99.9 | 104.7 / 9.71 | 1.73 / 2.04 |
| vL128a v1 | 7,059 | 34,424 (7.2%) | 132, 131 ×3, fallback → **129 at last scan (fail; final lattice unscanned)** | — | 98.7 | 103.2 / 9.52 (**stale**) | 1.52 / 1.77 |
| vL128a rerun | (same pack) | (same pack) | 132, 130 ×3, merge → 128 | 7,059 | 98.2 | not re-serialized | — |
| vL128b | 7,051 | 34,148 (7.2%) | 128 → 128 | 7,053 | 102.2 | 106.8 / 9.62 | 1.66 / 1.95 |

The C-view rows are C's gates: 1,000 born values at n=64 and 3,000 at n=128. `STAGE2_LINKABILITY.md:149-153` still marks this table as pending.

### 4.8 Trust boundary

Every step of stage 4 reads secret material:

- the packer and the link census read G2 = `tdp.mpmct1`, the unmixed stage-2 circuit that stage 3 exists to hide (`RD/window_pack3.py:153,162,168`; `W/link_census.py:20-21`);
- repair and split evaluate C's interior values and test their exact affine membership against boundary states (`W/repair.sh:18-19`; `W/anf_split2.py:26-29`);
- every intermediate circuit is a gate list of G3, which exposes all interior cuts.

*Inference:* all of stage 4 must run in the secret-holding environment, and only `deliverable.bin` (or its `.gz`/`.xz`) may leave. The run directory holds at least the following never-ship files (classified in §6.8):

- `tdp.mpmct1` (G2);
- `tdp.mpmct1.source_c.g57` and `tdp.mpmct1.sandwich.mpmct1`, written beside the output (`src/programs/gen_sandwich_gadget/mod.rs:27,212,230`);
- `seed.txt`, `gen.log` and `s3.log`. These record the gen seed and the mix seed (`W/val_chain.sh:14`). `val_chain.sh:16` passes the one gen seed as seed, sandwich seed and gadget seed, and that seed regenerates C and the whole gadget bit for bit (gen `mod.rs:209-211`);
- `s3.mpmct1`, `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1` and `rep512.bounds`;
- `span_s2.tsv` and `span_s2.txt` (from `hmap_affine` on C, `W/val_chain.sh:26-27`);
- `rep512.{wit,allcuts,exposure}.tsv`, `rep512.cview.log` and `split_cands_<pid>.*`.

### 4.9 Open issues for the Rust port

**Status (2026-10-09): ported.** Stage 4 is `tdp_packing run`
(`src/stages/packing/`, `src/programs/tdp_packing/`), integrated in the
managed driver; the design, the exactness plan and the validation are in
`docs/STAGE4_RUST_DESIGN.md`. Decisions taken there for the items below:
repair is ported in-process with the C-view engine moved out of
`hmap_affine` into the library; every check is a hard error (exit 4); the
final scan is the identity test (exit 5 on a visible non-output row) plus the
split's budget test (exit 6), with `--allow-*` as recorded escapes; the
deliverable has a versioned container (`deliverable.tdp4`) and a validating
decoder with a read-back check; repair improvements (slides across markers,
marker-identity cuts, all exposing cuts) stay future options behind flags,
never the default without re-validation. The text below is the analysis as it
stood before the port.

The documented plan (`docs/windowed_delivery_design.md:178-192`) has four items:

1. `window_pack`;
2. `anf_emit`, described as a port of `anf_compose.py` that includes the binary format;
3. band-refresh/migration as a stage 3.5, now superseded by stage-2 migration at dose 8 (`W/val_chain.sh:4`);
4. metrics in CI.

The plan has two gaps:

- **It omits repair entirely:** pair cancellation and fix slides (`order_repair.py`), `merge_bounds2.py`, `anf_split2.py`, and the C-view scan in the loop. Repair is a required step of the recommended stage 4 and the only step that needs C. It must be ported, or kept as an oracle-backed step, inside the secret-holding environment. The split step's full interface is `TAG G N BUDGET bounds_in bounds_out` on the CLI, plus two environment inputs: `SPLIT_P` (the repaired circuit) and `SPLIT_EXP` (the current `exposure.tsv`). Their defaults point at the unrepaired pack and at `fin.exposure.tsv` (`W/anf_split2.py:12,18`; §4.5). The port should make both explicit arguments.
- **The binary format to port is `anf_dump_eval.py`'s** (`W/anf_dump_eval.py:14-50`). `anf_compose.py` has only the size model (`RD/anf_compose.py:155`).

The port can reuse `XGate::apply_lanes`, the `formats.rs` I/O and `hmap_affine`. The existing `anf1`/`esop1` format is a per-target format, not this multi-output window format.

Policy and engineering decisions are still open on:

- hard failure on a VERIFY mismatch, a failed spot check, a failed or missing C-view scan, or a C-view above the floor;
- an identity-based floor test in place of `vis ≤ N`;
- what to do when no exposure-free split exists;
- a final rescan after the last merge/split;
- verifying against a snapshot copy, not a file the fix loop is rewriting;
- fix mode:
  - validate the canonical (live) breaker handling by re-repairing the seven snapshot-repaired replicates (§4.5);
  - let `slide` cross markers when only markers separate a gate from its blocker;
  - track cuts by marker identity rather than gate count;
  - consider all exposing cuts, not just the witness;
  - log the moved count;
- boundary deduplication;
- death-cause naming (`sibling` vs `sibling_kill`; `final`; pre-emptive closes booked as `drain_close`);
- wire-class thresholds;
- a round-trip decode test, a reference decoder (procedure in §4.6), and a versioned container header (n, window count);
- the pre-repair packer exposure versus the delivered proxy count (§4.1, §4.5): report the delivered figure;
- reporting real bytes instead of the size model;
- location-independent paths: `W/val_chain.sh:7,19`, `W/repair.sh:6-7`, `W/score_arm.sh:6`, `W/anf_split2.py:7,10`, `W/anf_dump_eval.py:5`, `W/verify_equiv.py:3`, `W/link_census.py:17`.

Bit-exact parity with the Python oracle requires replicating its dict-insertion order and heap tie-breaks, including the tail-flush order.


## 5. Verification, metrics and attack tooling

This section (§5) covers three families of tools:

1. **Functional checks.** Each stage must preserve the function of the circuit it was given. Every runtime check is *relative*: it compares a stage with its own input (G2 vs sandwich, G3 vs G2, packed vs G2, repaired vs packed, ANF vs gates). No step of the reference driver compares the deliverable with C(x).
2. **The C-view.** It asks whether any value computed by a gate of the source circuit C is an exact affine function of a delivered boundary state.
3. **Linkability censuses.** They count values that can be recognised at more than one window boundary.

The checks for stages 2-3 are Rust hard assertions: a failure panics, and the binary exits non-zero. Stage 1 (sandwich vs C) has **no runtime functional check**. `construct_seeded_sandwich` asserts only parameters (`src/stages/sandwich/construct.rs:161,311-317`). Its correctness rests on unit tests such as `sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice` (`tests/unit/stages/sandwich/construct.rs:56,168`; §1.5). The checks for stage 4 and the linkability metrics are Python prototypes. The C-view itself is a Rust binary.

The tracked copies in `scripts/windowed_delivery/` are byte-identical to the run-tree copies except `waist/order_repair.py` (§4.5). The snapshot scripts still `sys.path.insert` the hard-coded run-tree directory (e.g. `scripts/windowed_delivery/waist/link_census.py:17`, `waist/verify_equiv.py:3-4`), so they import run-tree modules, including the patched `order_repair.py`. Script paths in §5 follow the document-wide convention (Overview, "Conventions"): `waist/x.py`, `W/x.py` and bare script names all mean the authoritative run-tree copies. Every cited line number also holds for the snapshot, except `order_repair.py` lines ≥ 163.

### 5.1 Shared conventions

**Gate model.** Every tool implements the same gate, `st[t] ^= (AND_i lit(c_i, p_i)) XOR comp`, with `p=1` the positive literal and `p=0` the negated one (Rust: `src/circuit/xgate.rs:159-166`; Python: for example `waist/verify_equiv.py:10-18` and `boundary_cluster.py:67-73`). An empty control list makes the AND all-ones, so with `comp=0` the gate is a NOT; with `comp=1` it would be the identity. The canonical NOT, `XGate::x_gate`, has `comp: false` (`src/circuit/xgate.rs:51-57`). Every gate is an involution, so the reversed gate list is the inverse circuit (`src/stages/preprocessing/verify.rs:66-70`). The g57 semantics and the mpmct1 format are in §1.3; the `.bounds` format and the loaders' disagreement on n are in §4.2. `order_repair.load` uses the header width only, and `verify_equiv.py` imports that loader, so the repair equivalence check also uses header widths (`waist/order_repair.py:25-38`; `waist/verify_equiv.py:3-4`).

**Three input models.** The tools draw random inputs in three different ways. The difference matters when reading the metrics.

| model | inputs | used by |
|---|---|---|
| honest slice | random `x` on wires 0..n, every other wire 0 | `hmap_affine` (C-view and span check), `verify_zero_slice`, gen reverse check (random `a` on 0..n, run on the reversed circuits) |
| sandwich-wide | random on sandwich wires 0..2n, band wires 2n..4n pinned to 0 | gen forward check |
| all-wire | an independent random word on every wire, band and auxiliary wires included | mixer global check, packer checks, `verify_equiv`, ANF checks, `link_census`, `boundary_cluster`, `mult_census` |

The all-wire model checks the whole permutation. That is stronger than delivery needs, since only the slice (x, 0ⁿ, 0²ⁿ) ↦ C(x) on wires n..2n is contractual (`docs/tdp_pipeline.md:297-299`). For linkability, however, the all-wire model means that two values which agree only on the honest slice are *not* linked. Whether the all-wire model is the intended linkability adversary is an open question.

### 5.2 Correctness chain

| check | location | samples | compares | on failure |
|---|---|---|---|---|
| stage 1 | none at run time | — | (unit tests only) | — |
| gen forward | `src/stages/preprocessing/verify.rs:50-65` | 4×64 = 256, RNG `seed ^ 0xA11CE` (`:30`), sandwich-wide | G2 vs sandwich, payload half | `assert_eq!` panic |
| gen reverse | `verify.rs:71-92` | 256, honest slice | reversed G2 vs reversed sandwich, wires n..2n | panic |
| mixer per-splice | `src/engine/mixer/replacement.rs:614-679` | exhaustive (≤24 wires) or ANF proof (≤64) | DB window vs replacement | panic (regular store) |
| mixer global | `src/engine/mixer/reporting.rs:7-48` | 256, all wires | current arena vs mixer input | panic `FUNCTIONALITY BROKEN` |
| packer input / emitted | `window_pack3.py:63-90,168-173,485-487` | 1,024, all wires, seed 20261004 | final states of G2 vs G3, then G2 vs reordered G3 | input: prints the count only; emitted: prints `OK`/`FAIL`; neither stops the script |
| repair equivalence | `waist/verify_equiv.py:1-21` | 1,024, all wires, seed 77 | unrepaired vs repaired packed circuit | prints `MISMATCH` only |
| ANF spot checks | `anf_compose.py:93-121,159-163` | 256 (`LANES=4`, `:25`), every 500th window | window ANF vs gate evaluation | prints `CHECK FAILED` only |
| ANF full evaluation | `waist/anf_dump_eval.py:91-124` | 1,024, seed 1 | all-window ANF vs gate evaluation of the whole circuit | prints `match=False` only |
| end-to-end (manual, **not run**) | `tests/manual/tdp/verify_zero_slice.rs:1-13,64-76` | default 200, seed 1, honest slice | C(x) vs final mpmct1 on wires n..2n | prints `FAIL`, exits 1 |

**The drivers ignore failures.** `val_chain.sh` and `repair.sh` both run with `set -uo pipefail` and without `-e` (`val_chain.sh:6`, `repair.sh:5`). Neither checks exit status. If gen or the mixer panics, the later stages run on missing or stale files (`val_chain.sh:15-23`).

**Stages 1-2: `verify_payload`** (§2a.7). gen calls it once, after preprocessing (`gen_sandwich_gadget/mod.rs:356-368`). It compares G2 with the sandwich and never evaluates C. The forward check randomizes both sandwich halves and pins the band to 0; the reverse check (classic guarded only) is the gadget-level mirror of `A⁻¹(a,0) = (junk, D⁻¹(a))`. Auxiliary outputs are "intentionally unconstrained" (`verify.rs:7-9`). All eight validation replicates log both lines, e.g. `[gen] verify PASSED (256 bit-sliced samples, payload wires 64..128)` and `… reverse verify PASSED … D^-1 emerges under backward evaluation` (`W/vA64a/gen.log`).

**End-to-end check (exists, unused).** The manual binary `verify_zero_slice <C.g57> <final.mpmct1> <n> [samples=200] [seed=1]` (`Cargo.toml:257-259`) puts x on wires 0..n with every other wire 0. It compares C(x) with wires n..2n of the final circuit and requires at most 1,024 wires (`verify_zero_slice.rs:11-13,37-43`). Its header says it closes what is "otherwise only a transitive argument" (`:6-9`), and `verify.rs:36` points to it. No driver calls it.

**Stage 3: mixer** (§3.5, §3.7).

- **Per-splice DB verification** is on by default. `MixParams::default` sets `db_verify: true` (`src/engine/mixer/params.rs:665`), and circuit_mixer sets `db_verify: !args.no_db_verify` (`src/programs/circuit_mixer/mod.rs:576`). Only `--no-db-verify` turns it off (`src/programs/circuit_mixer/cli.rs:307-311`), and the recommended command does not pass that flag. The check works as follows (`replacement.rs:614-679`):
  1. The support is the union of all targets and controls of the outgoing window and of the replacement. Both sides are the identity outside the support, so equivalence on the support is equivalence of permutations.
  2. If the support has at most 24 wires, `rules::verify_rewrite` evaluates both sides on all 2^k assignments (`src/engine/moves/rules.rs:278-318`).
  3. If the support is wider, `polys_equivalent` composes both sides to canonical ANF and compares them (`src/stages/db_mixing/replacement.rs:107-145`). This is a proof, not a sample.
  4. When `polys_equivalent` is undecided (support above 64 wires, or the term budget is hit), the splice is declined and counted (`db_wide_skip`).
  5. A regular-store replacement that fails panics with "DB replacement verification failed". The historic curated path refuses the replacement instead of panicking (`replacement.rs:664-672`).
- **Regular store only.** The recommended run uses the regular store only because `FROZEN_CURATED_DIR` is unset (§3.1). All replicates log `verify=true curated=false` (e.g. `vA64a/s3.log:3,9`).
- **Global check.** It first asserts that the merge index matches the arena size. It then draws 4 batches × 64 lanes on *every* wire from the mixer RNG, evaluates the original input and the current arena, and panics on any difference. It runs every `verify_every` moves (default 10,000; `cli.rs:616-618`; `scheduling.rs:612-614`), at the stop-flag, eff-budget, canary, dose and moves-budget exits (`scheduling.rs:603-649`), and after the final float unless `--skip-final-float` is set (`circuit_mixer/mod.rs:901-904`). The ProfileDone exit (`scheduling.rs:476-478`) has no check of its own. The replicates ended on the moves budget (`vA64a/s3.log:1234`).
- **`--no-local-verify`** (`cli.rs:622-624`) disables only the per-move exhaustive checks inside transformations. In the classic twist these are packet invertibility (`src/engine/mixer/transformations.rs:652-659`) and the per-gate `P^-1·g·P == g'` (`:689-699`). On the recommended path (`twist-g57=true`, `vA64a/s3.log:1`) they are the g57-twist conjugation check (`:1146-1153`) and the seam checks (`:1168-1186`). Correctness of the mixer output therefore rests on the per-splice proof and the sampled global check.

**Stage 4: packer.** `proxy_spans` simulates G2 and G3 on the same 1,024 all-wire inputs and returns `agree`, the number of wires `w < min(na,nb)` whose final words match (`window_pack3.py:63-90`). The log prints `input function check X/Y` (`:171-173`) with no OK/FAIL marker; Y = Y shows G3 ≡ G2 on the full permutation. After scheduling, the packer re-runs `proxy_spans` on the emitted order and prints `VERIFY: emitted function check X/Y wires (OK|FAIL)` (`:485-487`). Neither check stops the script.

**Stage 4: repair and emission** (§4.5, §4.6). `verify_equiv` compares every final wire on 1,024 all-wire states (`verify_equiv.py:5-21`), once in the background and once after the fix loop (`repair.sh:15-16,27`); the background run races the fix loop's in-place rewrites (§4.5). `anf_compose.check_window` runs on every window whose index is divisible by 500: it draws 256 fresh all-wire samples, evaluates the gate window, compares every changed wire against its ANF and requires every unchanged wire to be untouched (`anf_compose.py:93-121`). At n=64 (about 2,600-2,800 windows) this re-evaluates 6 windows; at n=128 (about 7,000-7,400 windows) it re-evaluates 15. `anf_dump_eval.py` then evaluates every window over 1,024 lanes and prints `match=True/False` (`:91-110,124`), but on the in-memory windows, never on a decode of `deliverable.bin` (§4.6).

### 5.3 C-view: `hmap_affine --exposure`

The binary is `security_tests/heatmaps/hmap_affine.rs` (`HA`). Building it requires the `security-tools` feature (`Cargo.toml:308-311`). In this section **P** is the circuit passed as `--g` (G2, the repaired pack, or a split candidate) and |P| its gate count. The binary that produced the replicate numbers predates the committed `HA` source (§6.9, "Build provenance").

**Definition.** For every value born by a gate of C, t\* is the minimum, over the scanned cuts of P, of the number of distinct nonconstant P wire values whose XOR plus a constant equals that value on all samples. Wires holding the same value or its complement count once, and constants are free. If no cut admits such a predictor, t\* = −1 (`HA:137-148`).

**Example.** Suppose that at some cut wire 7 holds `y ⊕ m ⊕ 1` and wire 12 holds `m`, where m is a mask.

- Then `y = w7 ⊕ w12 ⊕ 1`, which gives t\* = 2.
- If wire 30 also holds `¬m`, it collapses into the class of wire 12, and t\* stays 2.
- If some wire holds `y` itself, t\* = 1.

**Sampling.** x is drawn uniformly on P's wires `0..n`, every other wire is 0, and C's rows are its wires `0..n` (`HA:362-375`). The defaults are `--batches 96` (6,144 samples), `--seed 12345`, `--c-step 10` and `--g-step 50` (`HA:99-117`). The exposure scan uses all 96 batches (`HA:379`); the train/holdout split applies only to the heatmap mode.

**Cuts and rows.**

- `--g-cuts FILE` replaces the stride. Its entries above |P| are dropped, and the list is sorted and deduplicated (`HA:328-341`). Cuts 0 and |P| are *not* added (only `indices()` adds them in stride mode, `HA:153-162`), so a bounds-file scan never examines the input or output state.
- The omitted end states are still covered. The input state is identical in every stage. The final state of every artifact equals the stage-2 final state as a permutation, and the span check covers that state.
- With `--c-step 1`, row i ≥ 1 is the value born by C gate i-1 on its target (`HA:381-391`). That gives |C| rows: 1,000 at n=64 and 3,000 at n=128.

**Algorithm, per cut** (`HA:511-634`).

1. **Deduplicate columns.** Each 6,144-bit wire column is complemented if its sample-0 bit is 1. Columns that become zero are constants and are skipped. Only the first wire of each class is kept as its representative (`HA:512-533`).
2. **Build a basis.** Gaussian elimination runs over the m representatives plus a constant column, tracking coefficients. Dependent columns become nullspace vectors (`HA:534-563`).
3. **Test each row.** Each row's sample vector is reduced. A nonzero residual means the value is not affine at this cut (`HA:579-593`).
4. **Weigh the predictor.** The weight is the popcount of the coefficient vector, excluding the constant bit. It is minimised over the nullspace coset by Gray-code enumeration of all 2^k combinations when the nullity k ≤ 14. Otherwise the particular solution is kept as an upper bound and the `approx` counter is incremented (`HA:596-617`).

**Outputs.**

- Line 2 of stdout: `[exposure] T born values, V visible (p%), A coset-minimizations skipped (nullity>14)`. A t\* histogram follows (`HA:641-658`).
- `<out>.exposure.tsv`: `gate wire tstar`.
- `EXPOSURE_ALLCUTS=<path>`: every `(gate, wire, cut, weight)` at which a value is exactly affine.
- `EXPOSURE_WITNESS=<path>`: for each visible value, the first minimising cut and its predictor support. The support lists *representative* wires only (`HA:659-699`).

**Error direction** (*derived*). "Exact" means equality on all 6,144 samples. A true affine relation always passes, so visibility has no false negatives at a scanned cut. A value that differs from some affine function on only a tiny fraction of inputs can pass and be reported as visible. This error is in the conservative direction for a hiding metric. t\* is exact only when `approx = 0`.

**The floor.** The last value on each C wire is C's output, and it is trivially visible: 64 values at n=64 (6.4% of 1,000) and 128 at n=128 (4.3% of 3,000) (`STAGE2_LINKABILITY.md:30-35`). In every repaired replicate all output rows reach t\* = 1 at interior cuts near the end, 3,066-8,292 gates before the end at n=64 and 7,510-19,065 at n=128 (column 4 of `rep512.wit.tsv`). For example, the output witnesses of vA128a sit at cuts 4,422,431-4,429,005 of 4,437,893 gates. A run "meets the floor" if `visible == N`. Visibility is decided by the residual test (step 3) alone. A skipped coset minimization (`approx > 0`) only makes that value's t\* an upper bound; it does not change whether the value is visible.

- `repair.sh:20-24` tests this by *count* only: it parses line 2 and checks `vis ≤ N`.
- `merge_bounds2.py:6-10`, `anf_split2.py:17-21` and `order_repair.py:140-146` instead identify outputs as "the last writer gate per wire" and act on non-output rows.

**Span check: `SPAN_INTERSECT` / `SPAN_OUTPUTS_TRIVIAL`** (`HA:392-500`). This mode runs inside `--exposure` and returns before the t\* scan. It needs snapshot 0 among the rows (`HA:427-430`). It works with three spaces:

- I = span{1, x_0..x_{n-1}}, extended with C's outputs when `SPAN_OUTPUTS_TRIVIAL ≠ 0` (`HA:437-454`);
- U = I + span{all C born values};
- W = span{1, all wires of P} at each cut.

The reported excess is `rank(W mod I) − rank(W mod U)` = dim(U∩W) − dim(I∩W) (`HA:482-483`). It counts the independent affine combinations of C values that equal affine combinations of the cut's wires, beyond what the inputs (and, if enabled, the outputs) explain. `val_chain.sh:26-27` runs it as `SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=…` on every stage-2 output with `--g-step 64` (n=64) or `256` (n=128). Stride mode includes cuts 0 and |P|, so the outputs-trivial setting is what makes the final cut score 0 rather than 64 / 128.

### 5.4 Repair loop: how the C-view drives stage 4

The repair loop is described in §4.5. In summary, the C-view scan is run with `EXPOSURE_WITNESS` and `EXPOSURE_ALLCUTS` up to 4 times in the fix loop, where each non-output witness drives a completer/breaker slide (the **completer** is the last gate before the witness cut that writes a wire of the predictor support S; the **breaker** is the first such gate at or after it). It then runs up to 4 more times in the fallback loop, where `merge_bounds2.py` deletes exposing boundaries (a cut-lattice subset, so nothing can become newly visible; `RESULTS.md:1400-1406`) and `anf_split2.py` spends one C-view run per split decision to choose an exposure-free cut.

Two properties of the loop matter for interpreting results:

- "Before c" means the first c gates of the *current* sequence (`order_repair.py:153,156-159`), with c taken from the witness file of the pre-pass scan. Slides for earlier witnesses in the same pass move gates across markers (`:114-137`), so for later witnesses gate rank c can drift from the boundary it named.
- If the fallback loop exhausts its 4 iterations, the bounds after the last merge and split are never re-scored, and the figure printed at `repair.sh:39` predates them. This is what happened to vL128a v1 (§4.5).

The 2026-10-08 patch to the run-tree `order_repair.py` (old and new breaker handling, and its effect on vL128a) is described in §4.5.

### 5.5 Linkability metrics

**Fingerprints.**

- **Inputs.** `LANES=16`, giving 1,024 shared all-wire inputs with `SEED=20261004` (`boundary_cluster.py:22-26`).
- **Normalisation.** A value word v is normalised up to complement, `fp = v ^ MASK if v & 1 else v`. Constants map to 0 and are skipped.
- **Key.** The key is a 16-byte blake2b digest (`waist/link_census.py:28-32`).

**Runs.**

- The input state seeds one run per nonconstant wire (`link_census.py:47-52`).
- At each boundary only the wires written in the window are re-digested. A new run starts when a wire's digest differs from its digest at the previous boundary (`:53-71`).
- **Cluster size** is the total number of runs of a value over all wires. Each run is one **sighting**.

**Example.** Suppose one wire shows, at the input and boundaries 1-5, the sequence `a, a, b, a, const, a`. Then `a` gets 3 runs and `b` gets 1. If `a` also occupies two runs on another wire, its cluster size is 5.

**Proxy and froth.** `amult[d]` counts how often stage-2 gates output digest d (`:34-45`). A value is **proxy** if `d ∈ amult`. Every other value is **froth** (`:73-74`). Froth means values output by no stage-2 gate: mixer-made values, plus any input word that no stage-2 gate re-creates, since input words seed runs (`:47-52`) but never enter `amult`.

**`waist/link_census.py G2 packed bounds label`.** For proxy and froth separately it prints:

- `values`, the number of distinct values with at least one run. For proxy, this is the delivered stage-2 exposure ("proxy values (link census)", §4.1). `STAGE2_LINKABILITY.md`'s "exposed" column is this count, not the packer's;
- `L2`, the number of values with cluster size ≥ 2 (**linkable values**);
- `M2`, the total runs over those values (**sightings**);
- a size histogram in buckets 1, 2, 3-4, 5-9, 10-19, 20-49, 50-99, 100+. **Anchors** are the proxy 100+ bucket.

It also breaks linkable proxy values down by stage-2 birth multiplicity, both by count and by mass (`:75-96`). It does *not* append the output cut m (`:23`), so gates after the last boundary are never simulated. `boundary_cluster.py:54` does append it.

**Other censuses.**

- **`boundary_cluster.py G2 packed bounds`** is the earlier equality-cluster attack. It reports median/p99/max cluster size and the proxy precision `p/(p+f)` at thresholds 2, 3, 5, 6, 10 and 20 (`:105-121`). On born values, "cluster size ≥ 6 is pure proxy (froth never exceeds 5)" (`RESULTS.md:302-317`).
- **`waist/mult_census.py G2 label`** is the stage-2 rebirth census. It prints `born`, `distinct`, `repeats = (born−distinct)/born`, the top multiplicity, the count and cover of values with multiplicity ≥ 65, and the count with multiplicity ≥ 257 (`:1-25`).
- **The packer's own `RESULT` line** (`window_pack3.py:488-503`) counts carried G2 values with a boundary in (birth, next write], plus values forced at the output. It classifies wires with thresholds hard-coded to 64/128 (`:49-50`).
- **Leak-investigation diagnostics** in `waist/`: `trace_witness.py`, `leak_mult.py`, `anf_at_cut.py`, `seed_coincidence.py`, `singleton_runs.py`, `liveness.py` and `link_merge.py`. `link_merge.py` deletes boundaries in a subset-safe way: deleting a boundary never increases any value's run count.

### 5.6 Current measurements

**Threat model.** The adversary sees the full wire state at every boundary and at the input, but never the gates inside a window (`STAGE2_LINKABILITY.md:23-28`). The cut lattice is used exhaustively. Function equality is sampled: 1,024 inputs for linkability and 6,144 for the C-view.

**No affine relation to C beyond the outputs was found in stage 2, at the cuts scanned.** The span check was run on 29 stage-2 variants, with outputs *not* counted trivial. At n=64 it scanned every 16th cut (about 41-60k cuts); at n=128 every 128th cut (about 15-17k cuts).

- The excess was 0 at every scanned interior cut; cuts between the stride points were not tested. It became nonzero only in the final ~3.3k / ~6.6k gates, ramping to exactly 64 / 128 as the outputs emerged.
- A dense tail re-scan with outputs counted trivial covered a subset: A, B, chain and chain+K64 at n=64; A, B and chain at n=128. It found 0 at all 4,001 / 8,001 tail cuts.
- dim U ≈ 1,050 / 3,100, which is far below 6,144 samples, so the result is not sampling-limited.
- Positive control: C against itself.

Sources: `RESULTS.md:1665-1681`. The mixer's role was localized on one artifact, rs64 (K=64 retirement, §2a.5, a lever since dropped for creating interior C leaks, `STAGE2_LINKABILITY.md:141`). There, stage 2 scanned at every 16th cut showed 0 visible values. The mixed circuit scanned at every 32nd prefix cut showed 11 interior values, including 7 of the 8 delivered leaks with identical predictor supports. Those C leaks were therefore made by the mixer (`RESULTS.md:1373-1376`). This localization was not repeated on the recommended configuration.

**Validation replicates.** These are the outputs on disk in `W/v{A,L}{64,128}{a,b}/`; `STAGE2_LINKABILITY.md:149-153` still says "Pending". The following passed in all eight runs:

- gen forward and reverse verify;
- the packer's input and emitted checks (256/256 or 512/512);
- `verify_equiv` EQUIVALENT, twice;
- 0 windows over 65,536 monomials;
- 0 spot-check failures;
- span excess 0 at every scanned cut, outputs counted trivial: stride 64 / 256 including cuts 0 and |P|, 12,145-12,176 cuts at n=64 and 8,681-8,733 at n=128; dim U = 1,041-1,051 at n=64 and 3,104-3,113 at n=128. Each `span_s2.txt` is one line, for example vL64a: `[span] dim span(C values,1) = 1049 over 6144 samples; cuts by intersection dim: {0: 12176}; nonzero from cut None`. The span scans are strided like the study's; no dense tail scan was run on the replicates.

`verify_zero_slice` was not run on any replicate. Cancellation removed about 6.6-7.3% of gates; for example vA128a went from 4,757,723 to 4,437,893. Pack, repair and serialization figures are in §4.7.

| run | C-view final | proxy values (delivered stage-2 exposure) | L2 | M2 | froth L2 | anchors (100+) |
|---|---|---|---|---|---|---|
| vA64a | 64 | 15,455 | 6,039 | 73,178 | 373 | 315 |
| vA64b | 64 | 15,454 | 6,007 | 73,428 | 409 | 321 |
| vL64a | 64 | 11,743 | 5,366 | 58,598 | 1,126 | 185 |
| vL64b | 64 | 11,735 | 5,274 | 58,895 | 1,110 | 200 |
| vA128a | 128 | 54,345 | 18,521 | 250,448 | 2,219 | 992 |
| vA128b | 128 | 54,715 | 18,608 | 250,150 | 2,275 | 995 |
| vL128a (rerun) | 128 | 34,958 | 16,106 | 187,248 | 5,579 | 718 |
| vL128b | 128 | 34,751 | 16,062 | 187,496 | 5,544 | 701 |

vL128a's first repair (v1, last scan at 129) gave L2 / M2 / froth L2 = 16,107 / 187,247 / 5,578. Its `deliverable.bin`, `deliver.txt` and timings were written at 01:35-01:36, before the patched rerun (08:24-08:44), describe the v1 artifact, and must be regenerated.

The stage-2 rebirth census (`mult_s2.txt`, §2b.7) is consistent across seeds: config A has repeats 19.8-19.9% and 329-335 (n=64) / 968-973 (n=128) values with multiplicity ≥ 257; config L8 has repeats 8.6-8.7%, none at ≥ 257, and a top multiplicity of 186-214.

**L8 against A.** This is the single reference table of L8-versus-A deltas. Each cell is the mean of the two replicates per configuration; sizes come from §4.7.

| metric | n=64: A → L8 | Δ | n=128: A → L8 | Δ |
|---|---|---|---|---|
| L2 | 6,023 → 5,320 | −11.7% | 18,565 → 16,084 | −13.4% |
| M2 | 73,303 → 58,747 | −19.9% | 250,299 → 187,372 | −25.1% |
| proxy values (link census) | 15,455 → 11,739 | −24.0% | 54,530 → 34,855 | −36.1% |
| exposed (packer, pre-repair) | 15,057 → 11,526 | −23.5% | 53,363 → 34,286 | −35.7% |
| anchors (100+) | 318 → 193 | −39.5% | 994 → 710 | −28.6% |
| froth L2 | 391 → 1,118 | ×2.9 | 2,247 → 5,562 | ×2.5 |
| `deliverable.bin` MB | 26.5 → 31.5 | +18.9% | 105.8 → 105.0 | −0.8% |
| ANF size estimate MB | 25.1 → 30.0 | +19.3% | 101.0 → 100.2 | −0.8% |

The n=128 L8 row mixes the patched vL128a rerun with vL128b, and its `.bin` mean uses vL128a's stale v1 file (§4.7). The doc-level summary in `STAGE2_LINKABILITY.md:120-135` predates the replicates. It uses the earlier finals (`RESULTS.md:1742-1751`) and gives L2 −13%, M2 −21% / −26%, "exposed" (= proxy values) −24% / −37%, anchors −39% / −28% and size +18% / 0% (n=64 / n=128). Those are earlier-final figures and should not be quoted as replicate results.

### 5.7 Limitations and requirements for the Rust port

**Status (2026-10-09).** The port (`docs/STAGE4_RUST_DESIGN.md`) addresses the
end-to-end check (zero slice of the decoded deliverable against C on every
run), hard failures, re-scoring (a final scan, identity test), serialization
(decode round trip, container header) and memory (a streamed C-view with
checkpoints: 0.4-0.5 GB at n=128). Attack scope, witness cut drift, duplicate
wires in witnesses and the input models are unchanged (the port reproduces
the oracle's repair exactly).

- **Attack scope.** Only exact affine predictors and exact value equality are measured. Low-degree nonlinear predictors were not run on the final artifacts (`STAGE2_LINKABILITY.md:178-180`). The C-view is also strictly per-cut: it does not combine wires from different boundaries.
- **End-to-end correctness.** Add an honest-slice check of the deliverable against C: input (x, 0, 0, 0), compare wires n..2n with C(x). Today stage 1 has no runtime check and every other check is relative. At minimum the driver should run `verify_zero_slice` on `s3.mpmct1` and `rep512.mpmct1`. The port should run the equivalent on the ANF-decoded deliverable.
- **Hard failures.** Make every Python check a hard error. `FAIL`, `MISMATCH`, `CHECK FAILED` and `match=False` only print today, and neither `val_chain.sh` nor `repair.sh` greps for them. The drivers also ignore exit status: there is no `set -e` and no status checks, so a panicking gen or mixer does not halt `val_chain.sh`. If `hmap_affine` fails inside `repair.sh` (e.g. OOM), `vis` is empty and `[ "$vis" -le "$N" ]` errors, which counts as false. `order_repair fix` and `merge_bounds2` then run on the stale witness and all-cuts files of the previous scan (`repair.sh:17-25,29-30`). Abort on any non-zero exit, and on a missing or non-numeric C-view result.
- **Re-scoring.** Re-score the C-view after the final fallback step. Test floor attainment by identity (no non-output row with t\* ≥ 0) rather than by count.
- **Witness cut drift.** `run_fix` locates the completer and breaker by gate rank `cut` from the pre-pass scan (`order_repair.py:153,156-159`). Locate each boundary by its marker in the current sequence, or recompute cuts after each slide.
- **Duplicate wires in witnesses.** Witness supports list representative wires only, so the completer/breaker search ignores duplicate copies of a value. Whether a relation can survive the slides through a duplicate copy is untested.
- **Serialization.** Add a decode round-trip test for `deliverable.bin`, and decide on a global header (n, window count).
- **Input models.** Unify the input models (honest slice, sandwich-wide, all-wire), and decide whether `link_census` should count the output state.
- **Memory.** The C-view stores `batches × cuts × wires` u64 words. At n=128 with 7,058 cuts the snapshots of P alone are ≈ 2.8 GB.

### 5.8 Stale documentation (the code is the authority)

- **[stale] packer sample counts.** `docs/ANF_WINDOW_DELIVERY.md:31` gives 256 samples for packer re-verification, and `:35` and `:185` give 4,096 fingerprint inputs/lanes. The code uses 1,024 for both (`window_pack3.py:22-23`).
- **[stale] "every window re-verified".** `ANF_WINDOW_DELIVERY.md:102-105`, `:120-122` and `:388-389` say every window is re-verified before serialization. `anf_compose` spot-checks every 500th window with 256 samples; only `anf_dump_eval`'s full-state evaluation covers every window.
- **[stale] ANF budget.** The 16,384-monomial budget in the same doc (`:118,184`) belongs to the earlier G=128 pipeline; `repair.sh` uses 65,536. Its parameter table (`:183-187`) is from the G=128 era.
- **[stale] "exactly (no sampling)".** `STAGE2_LINKABILITY.md:27-28` says the metrics are computed "exactly (no sampling)". The cut lattice is exact, but value equality is sampled.

### 5.9 Auxiliary statistical tests (cited in design rationale)

The stage-2 and stage-3 design justifications cite several statistical tests that are not part of the pass criteria and were not run on the validation replicates. Each is defined below with its tool. A "safe" value is what a construction with no detectable leak would show.

| test | definition | tool | safe value |
|---|---|---|---|
| **φ** (phi) | Pearson correlation of two binary sample vectors a, b over n samples: `(n·#(a∧b) − #a·#b) / √(#a(n−#a)·#b(n−#b))`, clamped to [−1, 1]. It is undefined (skipped) when either vector lacks a minimum number of 0s and 1s | `phi`, `src/stages/db_mixing/leakage_repair/detect.rs:681-696` | ≈ 0 (\|φ\| near the 1/√n noise floor). The EM comments' φ 0.13-0.26 (§2b.3, §2b.5, §2b.8) are leaks |
| **gauntlet w1 / w2 / w3** | Attack A4 of the gadget gauntlet, a bounded correlation scan of each target bit against trace features: w1 = every single feature; w2 = pairs on a capped, strided feature subset, combined by xor/and/or/andnot; w3 = triples (xor3/and3/or3/(a∧b)⊕c/(a∨b)⊕c). A target is flagged when its best ±1 covariance exceeds max(the NULL noise target's best, 6σ), with σ = 1/√samples. The other attacks are A1 (direct wire match), A2 (state-affine) and A3 (trace-affine) | `security_tests/gauntlet/gauntlet_audit.rs:1-21,1045-1075`, driven by `security_tests/gauntlet/gauntlet.py`; `--features security-tools` (`Cargo.toml:323-326`) | 0 flags on every arm except the unprotected `none` positive control, which must flag (`gauntlet.py:42-51`) |
| **fire_corr** | Fire-block correlation for the blinded-V5 construction, cited in EM's comment on balanced masks (EM:118-128) | Not in this tree. `docs/DRIP_RESHUFFLE_ARCHIVE.md:79,113` lists it as missing, under `red_team_tests/bin/leakage/` on another branch | cannot be re-run here; treat the EM statement as historical |
| **qc_scan "hot"** | A segment of the scanned circuit is hot if it nearly determines a sampled reference feature: \|φ\| ≥ 0.98 in both sample partitions, or an exact affine train fit validating at 100% on held-out samples. The tool prints hot counts per circuit against a fixed reference | `db_gen/analysis/qc_scan.rs:1-19` (plain release build, `Cargo.toml:123-125`); threshold `min_abs_correlation: 0.98` (`detect.rs:34-35,54`) | 0 hot. Reference against itself: 64 hot; independent stage-2 instance: 0 of 4,096 (`RESULTS.md:161-167`) |
| **heatmap ridge depth, null ridge** | From `hmap_affine` heatmap mode (no `--exposure`): H(i,j) = held-out GF(2)-affine reconstruction error of C's prefix-i state from the scanned circuit's prefix-j wires, where 0 = recoverable and 0.5 = hidden (`HA:1-12`; outputs `<out>.bin` and `<out>.meta.json`, `HA:915-917`). With R = 0.5 − H, **depth** = mean over rows of R(ridge cell) − median_j R(i,j). **rho** = Spearman correlation of row index and ridge column. **perm z** = z-score of rho against shuffled ridge columns. A **null ridge** has depth ≈ 0 and perm z ≈ 0 | `security_tests/heatmaps/plot_hmap_ridge.py:1-33,80-82` | depth ≈ 0, perm z ≈ 0. The independence check gave depth 0.0061 and perm z 0.0 (`RESULTS.md:161-167`); mixed circuits sat at 0.29-0.31 (§3.8) |
| **touch AUC** | Per-wire features (touch count, mean position, target fraction) are compared between two wire classes (top vs bottom half; data `0..n` vs band) by the rank statistic AUC = P(x > y) + ½P(x = y). The "touch AUC" is the `touches=` entry, and "line 14" of a report is `AUC data vs band` | `RD/wire_balance.py:1-17,41-55,154-163` (`wire_balance.py circuit.mpmct1 [label] [bins]`); reports `W/s12_check/wb_*.txt` | 0.5 (indistinguishable). Both 0.000 and 1.000 mean perfectly separable |

None of these tests has been run on the recommended configuration's delivered artifact.


## 6. Running the pipeline end to end

This section (§6) explains how to go from nothing to a shipped artifact: what to install, which seeds to draw and where they go, the exact command for each stage, how to check that a run passed, every file a run leaves behind, and what the eight validation replicates measured. In this section, "gen `mod.rs`" is `src/programs/gen_sandwich_gadget/mod.rs` and "mixer `mod.rs`" / `cli.rs` are in `src/programs/circuit_mixer/`.

A run turns a fresh secret circuit C into one shipped file, `deliverable.bin`, in three programs:

- **Stages 1 (sandwich) and 2 (preprocessing)** run as one Rust process, `gen_sandwich_gadget`.
- **Stage 3 (mixing)** runs as one Rust process, `circuit_mixer --tdp --db-mixing`, against the regular frozen store.
- **Stage 4 (packing)** *(superseded 2026-10-09: ported to Rust as `tdp_packing run`, which the managed driver runs; the prototypes below are its oracle)* existed only as Python/bash prototypes: `window_pack3.py`, then `repair.sh`, then `anf_dump_eval.py`. The snapshot in `scripts/windowed_delivery/` (README: the reference oracle for the Rust port, `scripts/windowed_delivery/README.md:1-8`) was first taken at commit `7163e457` (2026-10-07 23:21), before the leak-slide fix to `W/order_repair.py:158-171` (2026-10-08 08:24), and without the reference driver `val_chain.sh`. Both gaps are closed: commit `5619b867` refreshed it, and at `ran/stage4-rust` HEAD `waist/order_repair.py`, `repair.sh`, `merge_bounds2.py`, `anf_split2.py`, `verify_equiv.py`, `anf_dump_eval.py`, `link_census.py`, `window_pack3.py` and `anf_compose.py` are byte-identical to the live copies in `W/` and `RD/` (checked with `cmp`), `waist/order_repair.py` has the patched slide (§4.5), and `waist/val_chain.sh` is tracked (it differs from the run-tree copy only by dropping the inert `--p-comp 1.0 --p-any 0.1` on line 21). The tracked snapshot, which CI runs against `tdp_packing` (`tests/stage4/differential.sh`), is the oracle.

The reference end-to-end driver is `W/val_chain.sh` (`val_chain.sh:1-35`; a self-contained copy of its body is in the Overview). Every validated number in this section comes from it. *Superseded (2026-10-09):* the managed driver, `scripts/tdp_gen.sh` together with `src/tdp/`, used to implement the earlier six-step pipeline and could not express the recommended configuration (Appendix A, kept as it stood). It now runs the four stages and defaults to config L8, reproducing `val_chain.sh` from the same seeds (§6.10; `docs/tdp_pipeline.md`).

### 6.1 Prerequisites

**Toolchain.** The repository pins its Rust toolchain in `rust-toolchain.toml`: channel `1.97.1`, profile `minimal`, components `rustfmt` and `clippy`. `cargo` picks this up automatically through rustup. The Python scripts use the standard library only.

**Binaries.**

- `target/release/gen_sandwich_gadget` and `target/release/circuit_mixer` come from a plain `cargo build --release` (`Cargo.toml:88-94`).
- Stage 4's C-view scans also need `target/release/hmap_affine`. That binary is declared with `required-features = ["security-tools"]` (`Cargo.toml:308-311`), so a plain release build does not rebuild it. Build it explicitly with `--features security-tools --bin hmap_affine`. A stale `hmap_affine` has been seen to silently omit the `EXPOSURE_WITNESS` output that the repair step depends on.

**Frozen store.**

- `FROZEN_DB_DIR` must point at `frozen_db/frozen_m1_m11`. That directory holds 256 `shard_XX.frz` files plus `tables.bin` and `probes.bin`, about 298 GiB in total. The environment contract is in `src/database/frozen.rs:4-8`, and `src/database/environment.rs:25-29` requires the variable to be set.

**Obtaining the store.** Stage 3 cannot be reproduced without it.

- **Download.** The local copy was fetched by `frozen_db/fetch_frozen_db.sh`. That script is untracked, because the repository's root `.gitignore` rule `/*` excludes `frozen_db/` (`.gitignore:3`). It downloads the 258 store files (`tables.bin`, `probes.bin`, `shard_00.frz` … `shard_ff.frz`) from the project's object store and resumes on restart. It checks each file against the server's MD5 ETag and records the verified digest in `<file>.ok`. It does not fetch `filters.bin`.
- **Identity check.** A correct copy has 258 `*.ok` files, and every file's `md5 -q` equals its `.ok`. In particular, `tables.bin` is 1,915,737 bytes with MD5 `ba4b7ace3027c59c9f3ef2d0b754325b`, and `probes.bin` is 8,380,071 bytes with MD5 `30e525ec0a8e59b0eaef024e08b31b6b` (`frozen_db/frozen_m1_m11/*.ok`). The reader itself checks only the shard count and total entry count, which is not a content digest (`docs/frozen_database.md:194-196`).
- **Rebuilding from scratch.** See `docs/frozen_database.md:223-260`. Build the offline tools with `cargo build --release --locked --features db-tools` (`:228`). Generate regular tables with `db build-regular -m N`, which extends the preceding gate count, or with `db combine-regular` (`:232-239`). Merge with `merge_rocks_parallel`, run `db export-lmdb`, then run `frozen_from_lmdb tables|write|validate` (`:241-249`).
- **Filters.** To make `FROZEN_FILTER=1` meaningful, build `filters.bin` with `frozen_filters_build from-frozen DIR` as part of a fresh build (`:250`). For an existing store, run `bash db_gen/maintenance/build_filters.sh FROZEN_DIR`, which refuses to overwrite and validates key membership, and check it with `check_filters.sh FROZEN_DIR [EXPECTED_SHA256]` (`:203-213`). Filters cost about 25.5 GB of RAM per process (`frozen.rs:7`).
- **Coverage (m range).** *Inferred* from the name and the build command: the store holds regular tables built for gate counts m = 1..11. The construction bounds are not recorded with the store. These are `--min_n`/`--max_n`, which restrict the wire counts used (`:236-238`), and whether each length was generated completely. `docs/CURATED_DB_V3_SUMMARY.md:14` calls it "the complete g57 replacement corpus through 11 gates", while `docs/frozen_database.md:84-86` warns that the directory name does not establish exhaustive coverage. Treat coverage as undocumented (§3.10).
- The local copy has no `filters.bin`. So `FROZEN_FILTER=1`, which would otherwise load about 25.5 GB of filters (`frozen.rs:662-690`), does nothing on this machine. Every `s3.log` prints `FROZEN_FILTER=1 but no filters.bin ...; running unfiltered`.
- `FROZEN_CURATED_DIR` must stay **unset** (§3.1). `--tdp` turns curated lookups on unless `--curated` is given (mixer `mod.rs:94-109`). With the variable unset, the mixer falls back to the regular store only and prints a WARNING (mixer `mod.rs:217-229`). The log header still shows `TDP profile ON: curated=true` (the value before the fallback). The later lines show `curated=false` and `value conventions: regular=native, curated=-`.
- Passing `--curated` explicitly turns the missing store into a hard error (mixer `mod.rs:218-222`), so do not pass it.

**Python.** Stage 4 needs `python3` with the standard library only.

**Memory and concurrency.** No peak RSS was recorded for any stage. The figures below are *derived* from the code.

- **Mixer.** It keeps the 256 shard offset tables resident, (2^20+1) × 5 B ≈ 5.2 MB per shard, ≈ 1.3 GB per process (`frozen.rs:373-379,636-660`). On top of that come the exact caches, by default `LOOKUP_CACHE_MB` 512 + `XPOLY_CANON_CACHE_MB` 1024 (§3.1), and the circuit arena of up to ~4.8M gates. The store is not memory-mapped: buckets are read with `pread` through the OS page cache, which concurrent mixers share. `filters.bin` would add about 25.5 GB per process.
- **`hmap_affine`.** It stores `batches × cuts × wires` u64 words (§5.7). That is ≈ 2.8 GB for an n=128 repair scan (7,058 cuts), ≈ 3.4 GB for the n=128 stage-2 span check (8,733 cuts × 512 wires) and ≈ 2.4 GB at n=64 (12,176 cuts × 256 wires). In `val_chain.sh` the span check runs in the background while the packer and repair run, so two scans can overlap.
- **Validated concurrency.** All eight replicates (4 at n=64, 4 at n=128) ran together on an 18-core, 128 GiB host (§6.9). Higher concurrency is untested. Treat 8 jobs per 128 GiB as the recommended maximum.
- **No `set -e`.** Neither driver uses it, so an out-of-memory kill does not stop the run. Later steps then run on missing or stale files, which is why the §6.7 checks must all be read.

**Hard-coded locations.** The stage-4 scripts assume the run directory is `W/<TAG>` under `/Users/rancanetti/code/local_mixing`. The absolute paths are fixed in `repair.sh:6-9`, `val_chain.sh:7-8`, `anf_split2.py:7,10-12`, `verify_equiv.py:3`, `anf_dump_eval.py:5`, `link_census.py:17` and `mult_census.py:3` (run by `val_chain.sh:24`). Two more scripts are run by a path relative to a working directory that the caller sets with `cd`: `repair.sh:35-36` runs `anf_compose.py` after `cd $ROOT/runs/ncdb_ab_20261001`, and `val_chain.sh:28-29` runs `window_pack3.py` the same way. All of these are to be made location-independent during the cleanup (`scripts/windowed_delivery/README.md:7-8`).

### 6.2 Seeds: always random, always recorded, always secret

**Why the seed is secret.** `prepare_source` calls `fastrand::seed(seed)` and generates C from it (`src/stages/sandwich/construct.rs:64-70`; gen `mod.rs:196`). So `(n, m_C, seed)` regenerates C. With all three generator seeds and the logged parameters (gen.log line 1 records n, |C|, |D|, s and slice_gates; `seed.txt` records the `EMBEDDED_MASKING_*` configuration), the whole gadget is reproduced bit for bit (gen `mod.rs:209-211`). In `val_chain.sh` all three seeds are the same value, so the one recorded gen seed is enough. The streams derived from the three seeds are listed in §1.6 and §2a.2. The positional defaults are `seed=1` and `gadget_seed = sandwich_seed = seed` (gen `mod.rs:113-115`). Never rely on them.

**What the reference driver does.**

- `val_chain.sh:11` draws `secrets.randbelow(2**31)` and passes that one value as all three generator seeds (`:16`).
- `val_chain.sh:12` draws a separate mixer seed, `secrets.randbelow(2**40)`.
- Both seeds and the configuration line are written to `R/seed.txt` (`:14`). Redacted, the format is three lines:
  - `<TAG> gen seed: <n>`
  - `mix seed: <n>`
  - `config L8: PREMASK SEED_AVOID_C1 SHUFFLING=8 RETURN_HOME=0 EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8`
- `circuit_mixer --seed` defaults to 0 (`cli.rs:632-633`), so the mixer seed must always be passed.

**The managed driver** (`scripts/tdp_gen.sh`) draws and derives seeds differently; see Appendix A.

**Gaps to fix before production** (also §1.8, issues 1-2).

1. **31 bits is too few.** In `val_chain.sh`, C is a deterministic function of a 31-bit value. A wider draw still cannot give C more than 64 bits of entropy, because fastrand's state is 64 bits.
2. **File modes are world-readable.** `seed.txt` and `tdp.mpmct1.source_c.g57` (C itself) are written with mode 0644.
3. **The logs record the seeds.** `gen.log` line 1 records all three generator seeds (gen `mod.rs:185`), and `s3.log` line 4 records the mixer seed. Both logs are therefore as secret as `seed.txt`.
4. **Seeds are visible in `ps`.** Both drivers pass seeds on command lines. The generator seeds, which are the secret that regenerates C, go to `gen_sandwich_gadget` as positional arguments (`val_chain.sh:16`; `tdp_gen.sh:434-435`; parsed at gen `mod.rs:113-115`) and are visible for the 1–2 s that stages 1+2 run. The mixer seed is on the mixer's command line for the whole of stage 3 (`val_chain.sh:23`; `tdp_gen.sh:504`). Seeds should be passed in by file or environment instead.

**Stage 4 adds no secret randomness.**

- `window_pack3.py` fingerprints values with the fixed seed `SEED = 20261004` (`window_pack3.py:26,165-166`). `link_census.py` fingerprints with the same value, imported from `boundary_cluster.py:26` (`link_census.py:18,25`). `mult_census.py` also uses `Random(20261004)` (`:7`).
- `anf_compose.py` uses the same `SEED` (`:28`) only to draw its spot-check inputs (`:93-94,132`).
- `anf_dump_eval.py` uses `Random(1)` for its timing and match inputs (`:54`).
- `verify_equiv.py` uses seed 77 (`:8`).
- `hmap_affine --seed` defaults to 12345.

Packing and repair are therefore deterministic given G2, G3 and C. Whether window boundaries should be randomized is an open design question (`docs/windowed_delivery_design.md` §8).

### 6.3 Scale and launches

All validated runs use the research scale: |C| = |D| = 1000 at n = 64 and 3000 at n = 128. This is **not** the library convention `round(n·log2(n)²)`, which gives 2304 and 6272 (`src/stages/sandwich/construct.rs:56-60`). The pre-port `tdp_gen.sh` used that formula; the managed driver now defaults to the validated sizes (1000 / 3000, `source_gates_rule=validated`) and keeps the formula only as the opt-in `source.gates_rule = "formula"` (§6.10, §6.12). The recommended configuration has not been run at the library size.

Two other sizes follow from n:

- `s = max(n, round(n·log2 n))` = 384 / 896 (`construct.rs:51-54`).
- `slice_gates = 10·2n` = 1280 / 2560 (gen `mod.rs:109-112`).

The validated launches were:

```
bash W/val_chain.sh <TAG> 64 1000 384 1280 L8      # or A
bash W/val_chain.sh <TAG> 128 3000 896 2560 L8
```

### 6.4 Stages 1 and 2: `gen_sandwich_gadget`

```
EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 \
EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 \
EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8 \
  target/release/gen_sandwich_gadget $R/tdp.mpmct1 64 1000 1000 384 1 1280 $SEED $SEED $SEED > $R/gen.log 2>&1
```

This is `val_chain.sh:13-16`. Config A is the same command without the `CHAIN` / `CHAIN_LEN` pair. Pin `SANDWICH_VARIANT=classic` or unset it (§1.1).

| env knob | value | default | effect |
|---|---|---|---|
| `EMBEDDED_MASKING_SHUFFLING` | 8 | 0 | Carrier transfers per fire box: the migration dose (§2c.3). |
| `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME` | 0 | 1 | Persistent migration: roles hop between wires for the whole stream and are swapped home once at the stream tail (gen `mod.rs:61-73`; §2c.7). |
| `EMBEDDED_MASKING_PREMASK` | 1 | off | Before the first fire, each data wire (in random order) gets `min_open` = 2 fresh LGI masks (`EM:1922-1947`; §2b.3). |
| `EMBEDDED_MASKING_SEED_AVOID_C1` | 1 | off | Products in the forward band seed avoid the control pairs of the sandwich's first-layer gates (`preprocessing/construct.rs:84-120`; §2a.5). |
| `EMBEDDED_MASKING_CHAIN` | 2 | 0 | Applies only to the droppable (band-ancilla) brackets that `product_fire` returns on the quad-fire path (`EM:2076-2110`). If at least 2 band wires exist that no other gate of the fire touches, the fire's droppable brackets are re-targeted onto `max(CHAIN, 2)` of them, and each bracket's restore is deferred and queued per ancilla (`EM:2208`, `:2250`, `:2311-2330`). With fewer than 2 such wires, chaining is skipped for that fire (§2b.7). |
| `EMBEDDED_MASKING_CHAIN_LEN` | 8 | 0 (flush once per fire) | An ancilla's queued restores are flushed after every 8 of its brackets (`EM:2478-2500`). |

The remaining stage-2 settings are the production defaults (`EM:196-217`; gen `mod.rs:248-290`; full table in §2b.2): K = 2, `max_open` 3, `min_open` 2, quad fire on, balanced masks on, rich seed `SEED_NL` = 4.

**Positional arguments** are in the table in §1.1. More than 12 arguments trips an assert, and any `BV5_*` or `PROD_*` environment variable aborts the run (gen `mod.rs:150-171`).

**Outputs.** The run writes three files:

- `tdp.mpmct1`: G2, on 4n wires.
- `tdp.mpmct1.source_c.g57`: C, in g57 text.
- `tdp.mpmct1.sandwich.mpmct1`: the sliced sandwich A on 2n wires.

**Built-in checks.** Before G2 is written, the generator runs a forward payload check and a reverse check on 256 bit-sliced samples each (gen `mod.rs:356-372`; §2a.7).

**Gate counts.** The sandwich has `m_C + m_D + 2s + n` gates: 2,832 / 7,920. G2 is made of guard open + band seed + compute (pre-migration compute + migration) + band reseed + guard close (`preprocessing/construct.rs:156-160`). For vA64a this is 1,280 + 1,450 + (703,430 + 68,352) + 1,423 + 1,280 = 777,215.

### 6.5 Stage 3: `circuit_mixer`

```
NG2=$(head -1 $R/tdp.mpmct1 | awk '{print $3}')     # |G2|, from the mpmct1 header
MOVES=$(python3 -c "print(int($NG2*2*2*1.3))")      # ⌊N2·R1·1.3·|G2|⌋ = ⌊5.2·|G2|⌋ (§3.1)
env -u FROZEN_CURATED_DIR \
FROZEN_DB_DIR=.../frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  target/release/circuit_mixer --input $R/tdp.mpmct1 --output $R/s3.mpmct1 --tdp --db-mixing \
  --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
  --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
  --no-local-verify --seed $MSEED > $R/s3.log 2>&1
```

This is `val_chain.sh:17-23` with the `env -u FROZEN_CURATED_DIR` guard added (§3.1); `val_chain.sh` names the gate-count variable `G`. `val_chain.sh` also passes `--p-comp 1.0 --p-any 0.1`. Those values equal the defaults and have no effect here, because they act only in slot 3 (`scheduling.rs:545-575`), which is never reached when `p_db = 1.0` (§3.2). They are being removed.

| flag | value | default | role |
|---|---|---|---|
| `--tdp` | on | off | Selects the `DbLayer::tdp()` block (`cli.rs:788-802`). It also turns on `curated`, `curated_in_comp` and `db_advance` unless they are given (mixer `mod.rs:94-109`). |
| `--db-mixing` | on | off | Preset: g57 twists at `p_twist=0.0005`, `db_advance`, `mix_pay_random` (mixer `mod.rs:73-93`). |
| `--profile` | `2,2,2,2,2` | off | N0,N1,N2,R1,R2. The size controller is the sole size authority (§3.4). |
| `--moves` | ⌊5.2·\|G2\|⌋ | 1,000,000 | Binding stop (`cli.rs:28-30`). General rule ⌊N2·R1·1.3·\|G2\|⌋ (§3.1). |
| `--p-db` | 1.0 | 0.0 | Every round that is not a twist is a slot-2 DB move (`scheduling.rs:522-544`). |
| `--db-max-degree` | 9 | off | Skips windows whose ANF degree exceeds the store's maximum degree. |
| `--db-max-span` | 30 | off | A window touching more than 30 wires counts as a miss, without canonicalization. |
| `--db-wire-terms` / `--db-total-terms` | 1024 / 2048 | 2¹⁸ / 2²⁰ | Polynomial term caps (`cli.rs:341-350`). |
| `--no-local-verify` | on | off | Disables only the per-move exhaustive local checks (`cli.rs:622-624`; §5.2). |

**What the run does** (§3.4). The target ramps linearly to 2·|G2| by eff = 2 and then holds. Size peaks near 2.2-2.25× at the 2→3 transition (vA64a: 1,748,654 gates; log lines `phase 1 -> 2 at eff=2.13` and `2 -> 3 at eff=2.63`). The controller then saturates at `pmix=0.000`, so every DB round runs in COMP mode while twist rounds still run, and the size compresses toward S\*. The run stops on `moves budget spent` at eff ≈ 3.03–3.05, before the profile completes, so the final ratio is 2.04–2.14×.

**Verification during the run.** The per-splice DB equivalence check (`verify=true` in the banner) and a global sampled check every 10,000 moves (`cli.rs:616-618`) stay on. At the end, the log reports `final float: … (verified)` and `wrote N gates`.

**Canonicalization caps.** `val_chain.sh` does not set `CANON_RULE_L_BRANCH_CAP` or `CANON_MONOMIAL_CAP`; the readers return `None` when the variables are unset (`src/canonicalization/environment.rs:6-24`), so canonicalization ran uncapped provided the calling shell did not export them (§3.1). `CANON_MONOMIAL_CAP` is read only on the g57 window path and is inert here. The `canon caps: rule_l_skips=0 monomial_skips=0` line counts only that g57 path (`src/canonicalization/window.rs:222-241`), so it is zero by construction and proves nothing about the XGate path. Whether the managed driver's caps (512 / 200000; `tdp_gen.sh:466-467`, `runner.rs:392-393`) would bind at this configuration has not been tested.

### 6.6 Background stage-2 checks, then stage 4: pack, repair, serialize

**Background stage-2 checks** (`val_chain.sh:24-27`). These start from `$ROOT` right after stage 3 and run in the background while stage 4 proceeds. The driver's final `wait` (`:34`) collects them.

```
python3 $W/mult_census.py $R/tdp.mpmct1 $TAG > $R/mult_s2.txt 2>&1 &
STEP=$([ $N = 64 ] && echo 64 || echo 256)
SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=$R/span_s2.tsv target/release/hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 \
  --g $R/tdp.mpmct1 --n $N --exposure --c-step 1 --g-step $STEP --out $R/span_s2 2>&1 | grep "\[span\]" > $R/span_s2.txt &
```

- **Rebirth census** (`mult_s2.txt`, §5.5). This prints one line: `born`, `distinct`, `repeats`, `top`, `vals>=65` (with cover) and `>=257` (`W/mult_census.py:23-25`). It is informational; §6.7 gives the reference values.
- **Stage-2 span check** (`span_s2.txt`, §5.3). This is the only runtime evidence in a run that stage 2 itself exposes no affine C relation at the scanned cuts. It scans G2 against C at every 64th (n=64) or 256th (n=128) cut, including cuts 0 and |G2|, with the outputs counted trivial. The binary writes `span_s2.tsv` and the driver keeps only the `[span]` summary line. A pass reads `cuts by intersection dim: {0: <number of scanned cuts>}; nonzero from cut None`. Any nonzero key, or a cut number in place of `None`, means some scanned stage-2 cut carries an affine combination of C values beyond the inputs and outputs. Cuts between the stride points are not tested.

**Stage 4.** These commands run from the working directory `RD/` (`val_chain.sh:28-33`):

```
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1 > $R/pack_G512.txt 2>&1
bash $W/repair.sh $TAG 512 $N > $R/repair512.log 2>&1
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG > $R/deliver.txt 2>&1
gzip -9 -kf $R/deliverable.bin; xz -9 -kf $R/deliverable.bin
ls -la $R/deliverable.bin* | awk '{print $5, $9}' >> $R/deliver.txt
wait
```

The packer's argument 8 must be exactly `$R/packed_G<G>.mpmct1`, and `repair.sh`'s second argument must be the same `<G>` (§4.3, "File-name contract").

- **Pack** (§4.4; arguments in §4.3). `window_pack3.py` reorders G3 by commutation (same gates, new order) and chooses window boundaries so that stage-2 values stay inside windows. Its monomial accounting does not enforce the delivery budget: vA64a's packed windows reached 77,252 monomials. The 65,536 budget is enforced later, by the split step in repair.
- **Repair** (§4.5). `repair.sh TAG G N` cancels commuting duplicate pairs (typically about 7% of gates; vL128a went from 4,596,808 to 4,259,448) and checks equivalence in the background. It then runs up to 4 fix iterations, each a C-view scan (`hmap_affine --c source_c.g57 --c-format g57 --g P --n N --exposure --c-step 1 --g-cuts B`, 6,144 honest-slice samples, §5.3) followed, if `visible > N`, by completer/breaker slides. It re-checks equivalence synchronously, runs up to 4 fallback iterations of merge (if `visible > N`) and exposure-aware split at 65,536 monomials, and finally scores with `anf_compose.py P B 65536 TAG-rep512 500` and `link_census.py`. A split with no exposure-free candidate prints `WARNING: no exposure-free split in [lo,hi)`; this is what kept vL128a's first repair at 129.
- **Serialize** (§4.6). `anf_dump_eval.py` writes `deliverable.bin`, then times evaluation of the gate form against the ANF form.

### 6.7 Pass criteria per run

| file | required line |
|---|---|
| `gen.log` | `verify PASSED (256 bit-sliced samples, …)` and `reverse verify PASSED` |
| `s3.log` | `chain done in …: moves budget spent` **or** `chain done in …: profile complete (size schedule finished)` (mixer `mod.rs:882-899`), **together with** `final float: … (verified)` (`:901-911`; never pass `--skip-final-float`, because the `ProfileDone` exit relies on this check, §3.3) and `wrote N gates`; `curated=false` on the `DB ON` line |
| `span_s2.txt` | `cuts by intersection dim: {0: <all scanned cuts>}` and `nonzero from cut None` (§6.6) |
| `pack_G512.txt` | `input function check 4n/4n` and `VERIFY: emitted function check 4n/4n wires (OK)`; no Python traceback (`burst stuck` / `scheduler stuck`, §4.3) |
| `repair512.log` | `EQUIVALENT (1024 random states)` (twice); a last fallback iteration showing `C-view visible n` followed by `0 splits added`; no `WARNING: no exposure-free split` line |
| `rep512.cview.log` line 2 | `n visible`. `0 coset-minimizations skipped` is advisory: a skip only makes one t\* an upper bound and does not affect visibility (§5.3). It is required only when t\* values are quoted |
| `rep512.exposure.tsv` (recommended identity test) | no visible row (t\* ≥ 0) whose gate is not the last C gate on its wire (command below) |
| `anf_rep512.txt` | `overflow 0 (0.00%)` and `spot checks: k windows re-evaluated, 0 failures (OK)` |
| `deliver.txt` | `EVAL ANF-form … (match=True)` |
| `link_rep512.txt` lines 2-3 and `mult_s2.txt` | **informational, no gate.** Linkability has no acceptance threshold. Reference ranges from the two replicates per cell are in the table below. A value far outside them is a signal to investigate, not a failure |

**Identity-based floor test** (*derived*; the drivers do not run it). The count test `vis ≤ N` can hide a leak behind a missing output (§4.5). This command counts visible non-output rows, and it must print 0:

```
awk -F'\t' '!/^#/{g[NR]=$1; w[NR]=$2; t[NR]=$3; if(!($2 in L) || $1+0>L[$2]+0) L[$2]=$1}
  END{b=0; for(i in t) if(t[i]>=0 && g[i]!=L[w[i]]) b++; print b}' $R/rep512.exposure.tsv
```

It prints 0 on all eight replicates and 1 on vL128a v1 (`W/vL128a/rep512_v1/`). `exposure.tsv` reflects the last scan, so the test is valid only when the fallback loop ended on `0 splits added` (§4.5).

Linkability and rebirth reference ranges (`W/v*/link_rep512.txt`, `W/v*/mult_s2.txt`):

| config, n | proxy values | L2 | M2 | anchors (100+) | froth L2 | stage-2 repeats | values born ≥ 257× |
|---|---|---|---|---|---|---|---|
| A, 64 | 15,454-15,455 | 6,007-6,039 | 73,178-73,428 | 315-321 | 373-409 | 19.9% | 329-335 |
| L8, 64 | 11,735-11,743 | 5,274-5,366 | 58,598-58,895 | 185-200 | 1,110-1,126 | 8.6-8.7% | 0 |
| A, 128 | 54,345-54,715 | 18,521-18,608 | 250,150-250,448 | 992-995 | 2,219-2,275 | 19.8% | 968-973 |
| L8, 128 | 34,751-34,958 | 16,062-16,106 | 187,248-187,496 | 701-718 | 5,544-5,579 | 8.6% | 0 |

**The chain of equality checks** (§5.2). No single check is end to end. Each link covers one hop, and every link is sampled rather than proved:

1. **Generator.** Forward and reverse verify compare G2 with the sandwich payload on 256 samples (gen `mod.rs:356-372`).
2. **Mixer.** A per-splice DB proof, a global sampled check every 10,000 moves, and `final float … (verified)`.
3. **Packer.** The input function check compares G3 with G2, and the emitted function check compares the packed order with G2. Both cover all 4n wires on 1,024 samples (`window_pack3.py:85-90,168-172,485-487`).
4. **Repair.** `verify_equiv` compares `rep512` with `packed_G512` on all wires, 1,024 states, seed 77 (`verify_equiv.py:1-21`; `repair.sh:15,27`).
5. **Spot checks.** `anf_compose` re-evaluates every 500th window against its gates.
6. **Serialization.** `match=True` compares the ANF form, built in memory from the `rep512` gates, with the gate form of `rep512` on 1,024 inputs (`anf_dump_eval.py:59-110`).

The serialized bytes of `deliverable.bin` are never read back or verified, and C itself is never compared with the deliverable (§5.7).

### 6.8 Artifacts and formats

Each run lives in `W/<TAG>` and takes about 132–137 MiB at n = 64 and 427–446 MiB at n = 128 (`du`). vL128a is about 499 MiB because it also holds the `rep512_v1/` archive. **Treat every file except `deliverable.bin` as confidential.** Most of them are secret-equivalent:

| class | files |
|---|---|
| **Secret: seed-equivalent** | `seed.txt`; `tdp.mpmct1.source_c.g57` (C: 4,000 B / ~15.2 KB); `tdp.mpmct1.sandwich.mpmct1` (contains C); `gen.log` and `s3.log` (both record seeds) |
| **Secret: gate-level circuits** | `tdp.mpmct1` (~13.9 / 41 MB; G2, the input before obfuscation); `s3.mpmct1` (~30 / 88–92 MB); `packed_G512.mpmct1` + `.bounds`; `rep512.{mpmct1,bounds}`. The gate lists expose the window internals that the ANF delivery exists to hide. |
| **Secret: C-keyed leak maps** | `rep512.{exposure,wit,allcuts}.tsv` and `rep512.cview.log` (per C gate: its wire, t\*, the cut and the predictor support; `exposure.tsv` alone lists C's full per-gate wire sequence); `split_cands_<pid>.*`; `span_s2.{tsv,txt}` (C-versus-G2 span scan) |
| Confidential reports | `pack_G512.txt`; `repair512.log`; `anf_rep512.txt`; `link_rep512.txt`; `mult_s2.txt`; `deliver.txt` |
| **Deliverable** | `deliverable.bin` only. The `.gz` / `.xz` copies exist only to measure compressibility. |

**Formats.** mpmct1 is in §1.3 (for example, the line `61 1 2 191 0 222 1` means `st[61] ^= ¬(¬st[191] ∧ st[222])`); `.bounds` is in §4.2; `deliverable.bin` is in §4.6, with a worked example. The `deliverable.bin` wire count, window count and format version are all implicit, and no decoder exists. The "size: ANF X MB" line in `anf_rep512.txt` is a model estimate (`anf_compose.py:154-155`); the real file is slightly larger (vA64a: 24.6 MB estimated, 26.0 MB on disk; §4.6).

### 6.9 Reference measurements (validation replicates, 2026-10-07/08)

Eight runs were launched together on an 18-core, 128 GiB Apple M5 Max. Each mixer ran single-threaded. These results are not yet recorded in `RESULTS.md` or in `STAGE2_LINKABILITY.md` §5. C-view, window, size and evaluation figures are in §4.7; linkability figures in §5.6.

| run | G2 | G3 (×) | stage-3 wall | total wall |
|---|---|---|---|---|
| vA64a | 777,215 | 1,647,349 (2.12) | 2,114 s | 40.3 min |
| vA64b | 778,107 | 1,660,680 (2.13) | 2,108 s | 39.9 min |
| vL64a | 779,195 | 1,593,054 (2.04) | 2,052 s | 41.6 min |
| vL64b | 778,926 | 1,602,569 (2.06) | 2,049 s | 41.4 min |
| vA128a | 2,222,024 | 4,757,723 (2.14) | 5,945 s | 117 min |
| vA128b | 2,234,767 | 4,787,862 (2.14) | 5,976 s | 124 min |
| vL128a | 2,235,297 | 4,596,808 (2.06) | 5,902 s | 133 min |
| vL128b | 2,232,492 | 4,586,553 (2.05) | 5,896 s | 119 min |

**vL128a.** The first repair stopped at 129, one above the floor of 128. All four of its fallback iterations printed `WARNING: no exposure-free split in [7349,8962)`. It is archived in `W/vL128a/rep512_v1/`. The leak-slide breaker-index bookkeeping in `W/order_repair.py:158-171` was then patched (the live copy now differs from the snapshot), and the run was re-repaired:

- The fix iterations went 132 → 130 → 130 → 130.
- Fallback iteration 1 merged 1 boundary and split 1 window. Iteration 2 reached **128** with 0 splits, with no warning.
- After the rerun: 7,059 windows, proxy L2 / M2 = 16,106 / 187,248, froth L2 5,579, estimated ANF size 98.2 MB.
- `deliverable.bin` has **not** been re-serialized; the 103.2 MB file is from the first repair.

The other seven runs have not been re-repaired with the patched script.

**Phase timings** (measured with 8 jobs running concurrently; single-job timings were not recorded):

| phase | n = 64 | n = 128 |
|---|---|---|
| stages 1+2 | < 1 s | ~2 s |
| stage 3 | 82–88% of wall time (A 87–88%, L8 82%) | 74–85% of wall time (A 80–85%, L8 74–83%) |
| pack | 55–87 s | 159–205 s |
| repair | 162–276 s | 741–1,705 s |
| serialize | 31–53 s | 70–113 s |

**Evaluation speed.** In Python, batched over 1,024 inputs, the ANF form costs 0.77–1.00 ms per input at n = 64 and 1.77–2.04 ms at n = 128. The gate form costs 0.63–0.75 and 1.52–1.73 ms. No repaired window exceeded the 65,536-monomial budget; the median window holds 419–491 monomials.

**L8 against A.** See the single reference table in §5.6. In brief: L2 −12% / −13%, M2 −20% / −25%, `deliverable.bin` size +19% / −1% (n=64 / n=128), and froth L2 ×2.9 / ×2.5.

**Stage-3 share of wall time** is stage-3 wall time divided by total wall time, per run: 87.4, 88.1, 82.2, 82.5% at n=64 (vA64a, vA64b, vL64a, vL64b) and 84.7, 80.3, 74.0, 82.6% at n=128 (vA128a, vA128b, vL128a, vL128b). vL128a's total includes its first, failed repair.

**Build provenance (resolved 2026-10-08).** `hmap_affine` rebuilt from
`492564f0` reproduces the archived `rep512.{exposure,allcuts,wit}.tsv`,
line 2 of `rep512.cview.log` and the split-candidate TSVs on all eight tags
(`STAGE4_RUST_DESIGN.md` §12, P2), so the numbers below stand. Gen and the
mixer at `492564f0` reproduce the archived stage-2 artifacts of all eight
tags and `s3.mpmct1` of vL64a and vL128a (the historic anchor). The original
note follows.

No commit hash was recorded for the binaries. File times on this machine:

| binary | built | newer sources? |
|---|---|---|
| `target/release/gen_sandwich_gadget` | 2026-10-07 19:18:36 | none: no file under `src/`, `Cargo.toml` or `Cargo.lock` is newer |
| `target/release/circuit_mixer` | 2026-10-07 19:18:35 | none |
| `target/release/hmap_affine` | 2026-10-07 18:54:28 | **yes**: `security_tests/heatmaps/hmap_affine.rs` was last modified at 19:17:46 and committed in `7163e457` (23:21) |

The replicates were launched at 23:23 (`W/v*/seed.txt`), so gen and the mixer match the tree as of 19:18. Every C-view, span-check and repair number (§4.5, §4.7, §5.6, §6.6) comes from an `hmap_affine` binary that predates its committed source. So the `HA` line numbers cited in this document cannot be confirmed to describe the code that ran. To close this item, rebuild `hmap_affine` at `ea2e516f` (`cargo build --release --features security-tools --bin hmap_affine`) and re-run at least one replicate's repair C-view (the `cview()` command of `W/repair.sh:18-19` on `rep512`) and its span check (`W/val_chain.sh:26-27`). Then confirm that `rep512.cview.log` line 2, `rep512.exposure.tsv` and `span_s2.txt` are identical.

### 6.10 The managed driver today

The managed driver (`scripts/tdp_gen.sh` with `src/tdp/`, run through `tdp_gen --config`) runs all four stages, and its **defaults are the recommended configuration**: config v2 with preset `L8` (preset `A` selectable), schedule `validated` (profile `2,2,2,2,2`, moves `int(G·2·2·1.3)`, serial), the validated source sizes (1000 at n=64, 3000 at n=128), and `tdp_packing run` at G=512 with split budget 65536. Every gap of Appendix A is closed (`docs/STAGE4_RUST_DESIGN.md` §2.2, D1-D18). Use it; `W/val_chain.sh` is a parity tool.

### 6.11 Pitfalls

- **File-name contract.** `repair.sh TAG G N` finds its input only as `W/TAG/packed_G<G>.mpmct1` (+ `.bounds`), so the packer's output argument must use that exact name and the same `<G>` (§4.3). `anf_split2.py` run outside `repair.sh` silently defaults to the unrepaired pack and to `fin.exposure.tsv` unless `SPLIT_P` and `SPLIT_EXP` are set (§4.5).
- **Concurrency.** Never run two repairs in the same TAG directory: they share the `rep<G>.*` scratch files (`repair.sh:10-19`). The earlier finalizer `W/finalize3.sh` (not used by `val_chain.sh`) likewise shares `fin.*` and `final_G<G>.bounds` (`finalize3.sh:10-18`). Never edit a bash driver while instances are running it, because bash reads scripts by byte offset.
- **No `set -e`.** `repair.sh` and `val_chain.sh` use `set -uo pipefail` without `-e`. If `hmap_affine` is missing or fails, `cview` returns an empty string, the `-le` test errors, and the script keeps going. `cview` also depends on line 2 of the hmap log (`repair.sh:20`).
- **The fallback loop can exit unchecked.** After its fourth iteration it exits without rescanning, so the final printed C-view can predate the last merge or split (`repair.sh:28-33,39`). Always confirm the `0 splits added` line and the absence of `WARNING: no exposure-free split`.
- **Curated leaks back in.** An exported `FROZEN_CURATED_DIR` silently turns stage 3 curated-first (§3.1). Check `curated=false` in `s3.log`.
- **Pack report labels are wrong at n = 128.** `window_pack3.py:49-50` hard-codes the wire-class thresholds at 64/128, so the per-class exposure lines in `pack_G512.txt` are mislabeled for n = 128 runs.
- **Stale documentation.** `docs/ANF_WINDOW_DELIVERY.md` describes 4,096 fingerprint lanes, per-window verification and a G = 128-era parameter table (`:183-187`; §5.8). The code uses 1,024-sample (16 × 64-bit) fingerprints, checks every 500th window and runs at G = 512. *(Superseded 2026-10-09: `docs/tdp_pipeline.md` used to describe six stages and the profile `3,30,30,2,2`; it now describes the four stages and the `validated` schedule, profile `2,2,2,2,2`.)*

### 6.12 Open decisions

Decided by the port (`docs/STAGE4_RUST_DESIGN.md`): the default |C| is the validated table (1000 / 3000; other n need an explicit size or the opt-in formula); seeds are drawn by the driver into a private `SEEDS` file, which `tdp_gen.sh` reads, and never travel on any argv: gen and the mixer read them from inherited descriptors (`@/dev/fd/3..5` for gen, `--seed-file /dev/fd/3` for the mixer; their banners print `(file)`), unlike `val_chain.sh`, which passes them on argv (the here-strings feeding those descriptors stay off disk because the driver requires bash >= 5.1); L8 is the default; the container is `deliverable.tdp4` (version 1) with a validating decoder. Still open:

- **Seed entropy:** C is generated from at most 64 bits whatever the seed width; route (a) a 256-bit ChaCha-derived C (changes bytes, needs re-validation) or (b) a CSPRNG-generated C through `source.path` (design §13.1).
- **Boundaries:** whether to randomize window boundaries.
- **Timings:** solo stage-3 timings and validation of `--parallel-pieces` with this configuration. (The managed driver no longer sets the 512/200000 canonicalization caps: the reference runs had them unbounded.)


## Historic components

The curated frozen store and the old stages 4-6 (split, the slot-3 crossing walk and fcompress) are historic and play no part in this pipeline. They were removed on `ran/cleanup-four-stage` (stage-2 and stage-3 outputs verified byte-identical before and after) and are preserved on `ran/archive/curated-and-postmix`; see `docs/CURATED_AND_POSTMIX_ARCHIVE.md`.

## Appendix A. The managed driver: gap analysis

**Closed (2026-10-09).** Every item below is resolved on `ran/stage4-rust`: persistent migration and the new stage-2 knobs are the L8/A presets (custom preset for anything else); the profile `2,2,2,2,2` is the `validated` schedule; the historic stages and the curated store are gone; stage 4 is `tdp_packing run`; seeds are `gen_seed` (all three gen seeds) and an independent `mix_seed` in the private `SEEDS` file. The full list of differences and their resolutions is `docs/STAGE4_RUST_DESIGN.md` §2.2 (D1-D18). The analysis below is kept as it stood.

This appendix collects the managed driver's differences from the recommended pipeline. None of them affects a run made with `W/val_chain.sh`.

`configs/` ships only `tdp.toml`, which is also the default config (`src/tdp/cli.rs:6,12`). The documented workflow first runs `cp configs/tdp.toml configs/local.toml` and edits the database paths (`README.md:18`; `docs/local_mixing_documentation.md:2285-2291`), then runs `cargo run --release --locked -- tdp_gen --config configs/local.toml`. The command builds the binaries, writes a `tdp_command.conf` manifest, scrubs the environment and execs `scripts/tdp_gen.sh` (`src/tdp/runner.rs:93-212,330-411`). It cannot run the recommended configuration until the cleanup changes the following:

1. **Persistent migration is rejected.** `SHUFFLING > 0` with `RETURN_HOME=0` is refused by `tdp_gen.sh:272-275` and `src/tdp/config/validate.rs:118-125` (and expected to be refused by `tests/shell/tdp_flags.bash:40-41`). The generator itself preserves ports in both modes (gen `mod.rs:70-73`). The driver also requires `segments ≥ 8`, whereas the generator accepts any positive value (§2c.12).
2. **The new stage-2 knobs are scrubbed.** The wrapper removes every `PROD_*`, `SAT_*`, `EMBEDDED_MASKING_*` and `DB_QC*` variable (`runner.rs:334-343`). It then re-sets only K, `MAX_OPEN`, `MIN_OPEN`, `BALANCED`, `QUAD_FIRE`, `SHUFFLING` and `SHUFFLING_RETURN_HOME` (`runner.rs:353-380`). Neither the TOML nor the manifest has fields for `PREMASK`, `SEED_AVOID_C1`, `CHAIN` or `CHAIN_LEN`.
3. **Profile `2,2,2,2,2` cannot be expressed.** `tdp_gen.sh:451-464` hard-codes `n0=3` and `n1=n2=3+HOLD`. Its default (HOLD = 27, EXPAND = 2; `:115`) amounts to 78·|G2| moves, against the recommended 5.2·|G2|.
4. **Inert DB flags remain.** `DBFLAGS` still carries the inert `--p-comp` / `--p-any` (`tdp_gen.sh:490`).
5. **Curated is still promoted.** The driver still describes curated as "recommended" (`tdp_gen.sh:96-102`), and the wrapper exports `FROZEN_CURATED_DIR` when it is configured (`runner.rs:382-386`).
6. **The historic stages are still on by default.** Old stages 4–6 still run by default (`stop_after_stage = 6`; `tdp_gen.sh:517-587`), and `fcompress` is a required binary.
7. **There is no new stage 4.** Neither `tdp_gen.sh` nor the Rust `tdp_gen` wrapper has a windowed-delivery (`deliverable.bin`) step. Only the research driver `val_chain.sh` has one (`val_chain.sh:29-33`).
8. **Seeds** are handled differently (below), and `docs/tdp_pipeline.md` still describes six stages and the profile `3,30,30,2,2`.

**Seed handling in the managed driver.**

- `tdp_gen.sh:342-345` draws `secrets.randbelow(2**63 - 16)`. The code comment explains that a full 64-bit value overflows bash's signed arithmetic.
- It then uses `seed=SEED`, `gadget_seed=SEED+1`, `sandwich_seed=SEED` (`:434-435`) and the mixer seed `SEED+2` (`:504`).
- The seed goes to `RUN/SEED` under `umask 077` / `chmod 600` and is kept out of the narrative log (`:414-417`).
- The `[gen]` header still writes all three seeds in clear into `$RUN/stage12.log` (`:434-435`; gen `mod.rs:184-188`). That log is owner-only only because the `umask 077` at `:417` comes before it (§1.8, open issue 2).

## Glossary

| term | meaning | see |
|---|---|---|
| A | The stage-1 sliced sandwich on 2n wires: `[C ⧢ S1] ; N ; [D ⧢ S2]` | §1.4 |
| anchor | A proxy value with 100+ runs (sightings) in the link census. The residual class under L8 is carrier-migration hops | §5.5, §2c.11 |
| ANF | Algebraic normal form: a GF(2) polynomial written as an XOR of monomials (ANDs of variables). Each window ships as the canonical vector ANF of its permutation | §4.6 |
| atom | Stage-2 mask-budget unit, `Σ_w (u_w + 1 + extra_lgis)` over data wires (u_w = writes plus control reads of w) | §2b.1, §2b.3 |
| band | The 2n auxiliary wires `2n..4n` (roles `np..np+r`). 0 on the public slice, filled from the input by the band seed, keyed into every mask, junk at both ports | §2a.1 |
| band seed / band reseed | Modules 2 and 4 of stage 2: `seed_band_rich` fills each band wire with a pivot, 1-7 linear terms and 4 products. The reseed plays the seed's role for backward evaluation | §2a.3, §2a.4 |
| birth, born value | A value written by a gate. A born value of C is the value a C gate leaves on its target; these are the C-view rows | §5.3 |
| birth advance | Stage-3 post-splice slide of each new gate by ⌊0.85·slack⌋ in its direction | §3.5 |
| boundary | An interior cut of the delivered window lattice, listed in `.bounds`. The adversary sees the full wire state at every boundary | §4.1, §4.2 |
| bracket | A dirty-ancilla gadget that adds a product term to the fire target using only ≤2-control gates: the 4-gate `and3` (adds p·x·y) or the 8-gate Gray walk (adds P1·P2). Its final gate restores the ancilla | §2b.5 |
| breaker / completer | For a C-view witness at cut c with predictor support S: the completer is the last gate before c that writes S, the breaker the first at or after c. Fix mode slides them apart | §4.5 |
| burst (rerand burst), slot | A 16-gate refresh of one band wire, `b ^= lit·lit`; one burst per slot, m/8 slots per build. Not to be confused with the packer's atomic burst of an interval | §2b.8, §4.4 |
| carrier | The physical wire currently holding a data role | §2c |
| carrier migration | The stage-2 transform (code: preprocessing shuffling) that moves the written data role onto band roles' wires with two-CNOT transfers inside each fire box. In persistent mode, roles stay off home until the tail swaps | §2c |
| celebrity value | A value reborn very many times (multiplicity ≥ 257), typically an ancilla's prior value restored by brackets or a migration hop | §2b.6, §2c.11 |
| chain (chained brackets) | `CHAIN=k`: a fire's brackets share `max(k,2)` fire-safe band ancillas and defer their restores. `CHAIN_LEN=L` flushes an ancilla's queue every L restores | §2b.7 |
| config A / config L8 | The recommended stage-2 environment without / with `CHAIN=2 CHAIN_LEN=8`. L8 is recommended; A remains supported | §2b.2 |
| C-view | For each value born by a C gate, t\* = the fewest boundary wires whose XOR plus a constant equals it on 6,144 honest-slice samples. A value is visible if some boundary admits such a predictor | §5.3 |
| COMP / MIX | Stage-3 round modes. MIX may grow the window's spelling ("free, else pay at random"); COMP only shrinks or respells at equal length, by prefix descent | §3.3, §3.5 |
| cut | A position between two gates, i.e. the state after the first j gates. Boundaries are the cuts kept in the delivered lattice | §5.1 |
| D | The decoy: a random g57 circuit with C's design and size. Reversal on the slice yields D⁻¹, not C⁻¹ | §1.4, §1.5 |
| DAG interval | The gates that must lie between a value's birth and its overwrite in every commuting reorder; the unit the packer keeps inside one window | §4.4 |
| dose | The maximum number of carrier transfers per fire box (`EMBEDDED_MASKING_SHUFFLING`; recommended 8) | §2c.3 |
| drift (restore-drop) | Dropping a bracket's final restore so the ancilla drifts by p·x. Default on, inert under L8 and nearly inert under A | §2b.6 |
| eff | Stage-3 effective work: the sum of 1/size over moves, roughly the number of passes over the circuit | §3.4 |
| escort | A partner's value riding XORed on the fire target from its transfer to the box drain | §2c.4 |
| exposed (packer) | The packer's RESULT count of stage-2 values alive across a boundary of the **pre-repair** pack, or forced at the output | §4.1, §4.4 |
| fire | The masked evaluation of one sandwich gate: read cover, product units, fire cover, straddle | §2b.5 |
| fire box | The stretch of the stage-2 stream holding one fire; the unit inside which carrier migration places transfers | §2c.2 |
| fire cover | A temporary mask opened on the fire target for the duration of the fire, sampled away from the operand masks | §2b.5 |
| first-layer gate, fringe leak | A C gate whose target and controls still hold raw inputs. Its value `1 ⊕ a ⊕ y ⊕ x·y` can become affine-visible after reordering (the fringe leak); PREMASK and SEED_AVOID_C1 close it | §2a.5, §2b.3 |
| floor | The n output values of C, trivially visible on the answer register; the C-view target | §5.3 |
| froth | Values with ≥2 boundary runs that no stage-2 gate outputs: mixer-made values plus input words never reborn in G2. "Linkable froth" is froth L2 | §5.5 |
| G (window cap), 𝒢 | G is the stage-4 window gate cap (512); 𝒢 is the deliverable permutation | Overview, §4.3 |
| G2, G3 | The stage-2 gadget (`tdp.mpmct1`) and the stage-3 mixed circuit (`s3.mpmct1`), both on 4n wires. \|G2\| is G2's gate count | Overview |
| g57 gate | `[a,b,c]`: `a ^= b ∨ ¬c` (= `1 ⊕ c ⊕ bc`), the XGate with comp = 1 and monomial `¬b ∧ c` | §1.2 |
| g57 twist | A rare stage-3 move that swaps two wire labels across a long window, closing each end with an all-g57 seam word | §3.6 |
| hidden firing, straddle | The persistent mask opened on the target mid-fire, so that no block containing the fire shows the bare gate increment | §2b.5 |
| hot-targeting | Choosing each rerand slot's band wire among those with the most bracket borrows since their last burst | §2b.8 |
| identity guard | Stage-3 rule that skips a store candidate equal to the window or to a commuting reorder of it | §3.5 |
| LGI | "Locally geodesic identity": the open/close pair of one mask, `g57(w,x,y); CNOT(w,z)`, which XORs `M = 1 ⊕ y ⊕ x·y ⊕ z` onto w and removes it when re-emitted | §2b.3 |
| linkable value (L2) | A proxy value seen in ≥ 2 separate runs at boundaries | §5.5 |
| M2, sightings | A sighting is one run of a value; M2 is the total number of runs over all linkable proxy values | §5.5 |
| mask | A balanced quadratic band term held on a data wire; each data wire carries 2-3 open masks from PREMASK to the final drain | §2b.3 |
| mpmct1 | The gate-list text format of every stage: header `mpmct1 <wires> <gates>`, one `t comp k (wire pol)×k` line per gate | §1.3 |
| multiplicity | The number of times a value is born in G2 (`mult_census.py`) | §5.5 |
| N column, N-float | The n CNOTs `y_i ^= x_i` that copy C(x) into the answer register, floated left or right by commutation | §1.4 |
| partner | The band role whose wire receives the fire target at a transfer; chosen by read pressure among non-excluded roles | §2c.5 |
| payload half | The sandwich half that carries the answer on the slice: wires n..2n for classic | §2a.7 |
| φ (phi) | Pearson correlation of two binary sample vectors; the statistic behind several design-rationale measurements | §5.9 |
| PREMASK | Opening `min_open` masks on every data wire before the first fire, so no raw input copy survives into the compute | §2b.3 |
| proxy value | A boundary value equal to the output of some stage-2 gate. The count of distinct proxy values on `rep512` is the delivered stage-2 exposure ("proxy values (link census)") | §4.1, §5.5 |
| public slice, zero slice | The inputs with y = 0 and band = 0, on which G2 (and the deliverable) computes C(x) forward and D⁻¹ backward | §1.5, §2a.1 |
| read pressure | Reads since the last write on a physical wire; transfer partners are chosen among the maximal ones | §2c.5 |
| regular frozen store | `frozen_m1_m11`, the g57 store keyed by canonical function key; the only store stage 3 uses | §3.1, §3.5 |
| role | A logical data or band identity, as opposed to a physical wire | §2c |
| run | A maximal stretch of consecutive cuts (input cut and interior boundaries) at which a wire holds the same value up to complement | §5.5 |
| rung | One nested sub-window tried in a COMP prefix descent | §3.5 |
| SEED_AVOID_C1 | Band-seed rule that redraws any product on the control pair of a first-layer sandwich gate | §2a.5 |
| slice blocks S1, S2 | The comp = 0 gates that read one y wire and are the identity on y = 0, merged into C and D | §1.4 |
| slice guard | Modules 1 and 5 of stage 2: identity exactly when the band is 0, junk-half targets only, ≤ 2 controls (accumulate-and-inject) | §2a.6 |
| span check (excess) | `dim(U∩W) − dim(I∩W)`: affine combinations of C values available at a cut beyond inputs (and outputs) | §5.3 |
| t\* | The C-view predictor weight of a born value; 1 means a single boundary wire holds it | §5.3 |
| touch AUC | Rank-statistic separability of two wire classes by per-wire touch count; 0.5 = indistinguishable | §5.9 |
| transfer | The two-CNOT role swap `cnot(q,p); cnot(p,q)` that moves the fire target onto a partner's wire | §2c.4 |
| vocabulary | The set of distinct nonconstant stage-2 values (up to complement), the denominator of "% of vocabulary" | §4.1 |
| window | The gates between two consecutive boundaries; it ships only as the canonical ANF of its permutation | §4 |
| winnable | A stage-2 value that is not output-forced and whose spans and DAG intervals fit `span_cap` and `W_cap`, so the packer can try to hide it | §4.4 |
