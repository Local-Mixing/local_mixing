# The TDP Pipeline: Reference

**Status (2026-10-12).** Everything described here is on `main` (2026-10-12; the commit pins below name commits on `main`). The pipeline has **six stages**: 1 sliced sandwich, 2 preprocessing (embedded masking), 3 database mixing, 4 splitting, 5 the crossing walk, 6 delivery = packing (commuting reorder into ANF windows, order repair, C-view repair loop, serialization of the container) followed by unpacking with lattice refinement (every window resynthesized from its ANF alone; a window that fails to resynthesize is cut in two at an exposure-aware cut, both halves re-emitted into the deliverable and unpacked separately, recursively). Stages 4 and 5 were revived from the archive on 2026-10-11 and stage 6 became one delivery stage on 2026-10-12 (commits `9f2adcd9`, `7be75f96`, `87a88e7e`, `b248a0f2`, `83e44ec5`). This document was written as the four-stage reference on 2026-10-09 (the packer was ported to Rust the same week), and the four-stage chain it validated is today's `run.skip_stages = [4, 5]` with `delivery.anf_only = true` (bash: `--run-skip-stages 4,5 --delivery-anf-only`). Revised 2026-10-12 for the six-stage pipeline. Code pointers into tracked files are pinned as follows: §1-§3, §6a, §7 and §8 at commit `ea2e516f` unless marked "at HEAD" (the two most-cited files, gen `mod.rs` and `EM`, and the mixer files have drifted since; the anchor table under "Conventions" gives the current lines of every anchor this document's text depends on, and the pointers it re-pins are marked), §4, §5 and §6b at `83e44ec5`. Script line numbers refer to the tracked snapshot `scripts/windowed_delivery/` (see "Conventions").

- **Rust: all six stages.** Stages 1 and 2 run in one process, `gen_sandwich_gadget`. Stage 3 is `circuit_mixer --tdp --db-mixing --p-db 1.0` with the regular frozen store only. Stage 4 is `circuit_mixer --split --split-stop` (the archive-era flags), stage 5 the crossing walk `circuit_mixer --resume split.state --target-size …`. Stage 6 (delivery) is `tdp_packing run` (`src/stages/packing/`, a bit-exact port of the Python prototypes, `docs/STAGE4_RUST_DESIGN.md`) followed by `tdp_unpacking run --refine --max-rounds N` (`src/stages/unpacking/`; `docs/DELIVERY_STAGE.md`, `docs/STAGE7_UNPACKING.md`).
- **The managed driver.** `tdp_gen` with `source.wires` (64 or 128), `run.skip_stages = [4, 5]` and `delivery.anf_only = true` (bash: `scripts/tdp_gen.sh --run-skip-stages 4,5 --delivery-anf-only`) reproduces `val_chain.sh` CONFIG=L8 byte for byte from the same seeds (stage-2 preset L8, schedule `validated`, the packer at G=512 with split budget 65536): proven on the archived runs vL64a and vL128a themselves (the historic anchor, `STAGE4_RUST_DESIGN.md` §12) and by the reproduction test against a sandboxed `val_chain.sh` (n=8, n=16 L8 and A, n=64 L8; `tests/shell/tdp_defaults.bash`). `pinned_recipe` (`val_chain-L8-20261007` / `val_chain-A-20261007`) is recorded only for that chain. With nothing set the driver runs all six stages and ends with the published circuit `unpacking/unpacked.mpmct1` (beside the internal refined deliverable `unpacking/deliverable.tdp4`). Config A is `preprocessing.preset = "A"`.
- **Oracles:** the Python/bash packing scripts (tracked snapshot `scripts/windowed_delivery/`) are the canonical oracle of `tdp_packing`. They are never edited; CI runs them against `tdp_packing` on random inputs (§6a.9). `scripts/windowed_delivery/waist/deliver.py` is the Python reference of the delivery loop (`tests/manual/delivery_parity.sh`: bounds, every per-round deliverable, the final deliverable and the canonical circuit byte-identical on the n8_L8 fixture).
- **Recommended configuration:** config L8 (bounded chains). Config A (no chains; `preprocessing.preset = "A"`) remains a supported alternative (smaller at n=64, more linkable values; §7.6).
- **Repair code.** The packer's order repair (`stages::packing::repair`, the port of `scripts/windowed_delivery/waist/order_repair.py` in its patched version: the slide step locates the breaker before moving the completer, §6a.5) runs in every `tdp_packing run`; it is not optional. The separate post-stage-3 leakage audit and DB repair (`[leakage_repair]`, `db_mixing.leakage_repair.txt`) is off by default.
- **Six-stage chain: no replicate validation yet.** The measured runs so far are n=16 (`port_runs/six_n16b`, 2026-10-12: 18 min end to end, delivery 3 refinement rounds, 678 → 696 windows, 18 clean cuts, 0 unclean cuts, 0 verbatim, every check ok), the n=64 seven-stage run (`port_runs/seven_n64`, 2026-10-11, pre-refinement fallback policy: 10,366 windows, 157 fallbacks before refinement, gate ratio 0.925) and its bisection-to-the-end assembly (10,540 windows, 0 verbatim, 0 interior C values; `docs/notes/stubborn_windows_20261012.md`). Both ran at `crossing.move_budget_factor = 2`, not the current default 3 (§5).

## Overview

### What the pipeline produces

The input is a secret reversible circuit C on n wires. In general C may use any reversible gate with up to two controls (NOT, CNOT, Toffoli with either control polarity); the engine takes it in the g57 form (`a ^= x ∨ ¬y`, which expresses that class) through `source.path`, or generates a random C. What the pipeline then mixes is not C itself but a specific circuit structure built around it: the sliced sandwich on 2n wires (stage 1) and the embedded-masking construction on 4n wires (stage 2), to which stages 3-6 apply. The published object of the pipeline, the TDP, is a circuit: `unpacking/unpacked.mpmct1`, the canonical-order concatenation of the resynthesized windows (the driver names it the final artifact). It is made from an internal windowed representation: `packing/deliverable.tdp4`, the packer's container (a 48-byte `LMTDPANF` v1 header and a body that is a sequence of **windows** on the packer's lattice, each represented only by the canonical algebraic normal form (ANF) of the permutation it applies), and `unpacking/deliverable.tdp4`, the same container on the refined lattice. The unpacking sub-step resynthesizes every window from its ANF alone and writes the circuit in canonical commuting order; the two deliverables are checked equivalent to the circuit, are kept for auditing and parity, and do not ship. Only under `delivery.anf_only` (the validated four-stage chain) is the packer's container `packing/deliverable.tdp4` (raw body `deliverable.bin`) the published object, and the recipient then does the unpacking itself (RC, 2026-10-13). The circuit, and window by window the deliverables, compute a 4n-wire permutation with a public **zero slice**:

```text
𝒢(x, 0ⁿ, 0²ⁿ)   = (junk, C(x), junk)             forward: C(x) appears on wires n..2n
𝒢⁻¹(a, 0ⁿ, 0²ⁿ) carries D⁻¹(a) on wires n..2n    backward: an unrelated random function
```

Here 𝒢 is the published permutation, the function computed by `unpacking/unpacked.mpmct1` (and, equivalently, by the internal `unpacking/deliverable.tdp4`; the unpacking sub-step checks the two equal on 4 × 1,024 random states). The letter G on its own is kept for the packer's window gate cap (`packing.window_gates`, §6a.3).

D is a random decoy circuit with the same design and size as C (§1.4, §1.5). Stage 2 preserves the sandwich's payload half on the band-zero hyperplane (§2a.7). Every later stage preserves the full 4n-wire permutation of the stage-2 circuit G2.

### The six stages

| stage | what it does | runs as | output |
|---|---|---|---|
| 1 sliced sandwich | the classic sliced sandwich A around C on 2n wires | `gen_sandwich_gadget` (with stage 2) | `tdp.mpmct1.sandwich.mpmct1`, `tdp.mpmct1.source_c.g57` |
| 2 preprocessing | embedded masking on 4n wires: masked fires, band, guards, persistent carrier migration (preset L8) | `gen_sandwich_gadget` | `tdp.mpmct1` (G2) |
| 3 database mixing | respelling by exactly verified store splices against the regular frozen store, dilution to ~2× | `circuit_mixer --tdp --db-mixing --p-db 1.0` | `db_mixing.mpmct1` (G3) |
| 4 splitting | every complemented g57 gate split into plain conjunctions, joined to distant brackets through absorbed NOT twists, to exhaustion | `circuit_mixer --split --split-stop` | `split.mpmct1` (G4), `split.state` |
| 5 crossing walk | the directional crossing walk under a size thermostat (target 2× its input, move budget 3× the target) | `circuit_mixer --resume split.state --target-size …` | `crossing.mpmct1` (G5) |
| 6 delivery | packing (commuting reorder into ANF windows, repair, serialization), then unpacking with lattice refinement | `tdp_packing run`, then `tdp_unpacking run --refine` | `unpacking/unpacked.mpmct1` (**the published circuit**); internal: `packing/deliverable.tdp4` (the packer's lattice; the published object only under `anf_only`) and `unpacking/deliverable.tdp4` (the refined lattice) |

```text
C → sliced sandwich → embedded masking → DB mixing → split → crossing walk → packing → unpacking with refinement
       stage 1            stage 2          stage 3     stage 4    stage 5      stage 6 (delivery)
                                                                         packing/deliverable.tdp4 → unpacking/deliverable.tdp4   (internal)
                                                                                                     unpacking/unpacked.mpmct1   (published)
```

**Every stage after the first is optional** (`run.skip_stages` / `--run-skip-stages`, values 2..6; `run.stop_after_stage` and `run.rerun_from_stage` in 2..6). Each stage reads the output of the last stage that ran: with stage 2 skipped the sandwich itself goes on and becomes the packer's `--stage2` (what the windows hide); with stage 4 skipped the crossing walk starts from the previous circuit with its own `cross_seed`. Within stage 6, `delivery.anf_only` ends the stage after the packer's deliverable, which is then the published object. Stage 6 requires `source.wires ≤ 256` (4n ≤ 1,024 state wires; nonlinear291 widths are refused when stage 6 runs). The validated four-stage chain is `run.skip_stages = [4, 5]` + `delivery.anf_only = true`, byte-identical to `val_chain.sh` (`tests/shell/tdp_defaults.bash`); `pinned_recipe` is recorded only for that chain.

**Driver facts.** `config_version` 4 (1, 2 and 3 are refused with migration messages; an `[unpacking]` table or `--unpacking-*` flag is refused with a pointer to `[delivery]`); recipe v12 in `tdp_command.conf` (v9-v11 runs are refused on resume); `SEEDS` format 3 (gen, mix, split, cross, unpack = the delivery seed, plus `run_key` and `seed_source`; a format-2 file is upgraded in place); keyed completion markers `stage12.recipe` (`tdp_stage12_recipe=7`), `stage3.recipe`, `stage4.recipe`, `stage5.recipe`, `delivery.recipe` (HMAC-SHA256 tags keyed by `run_key`); failure marker `delivery.failed` (`exit=`, `step=packing|unpacking`); logs `stage12.log`, `stage3.log`, `stage4.log`, `stage5.log`, `stage6.log` (both sub-steps, with a separator line); directories `packing/` and `unpacking/`; `NOT_RECOMMENDED` next to the final output after an allowed failure; the final line `PIPELINE COMPLETE (stages 1-6): <run>/unpacking/unpacked.mpmct1` (under `anf_only` `<run>/packing/deliverable.tdp4`), or `PIPELINE FINISHED WITH AN ALLOWED FAILURE`; `run.threads` pins `RAYON_NUM_THREADS` and both stage-6 binaries' `--threads`. Both drivers, `scripts/tdp_gen.sh` (bash) and `src/tdp` (`local_mixing_bin tdp_gen --config x.toml`), are in parity; the operational manual is `docs/tdp_pipeline.md`.

### Notation

| symbol | meaning |
|---|---|
| n | width of C. Wires `0..n` = x (input; junk at the output); `n..2n` = y (answer register; 0 on the slice); `2n..4n` = band (0 on the slice). Ranges are half-open |
| C, m_C | the secret source circuit and its gate count: \|C\| = 1,000 at n=64 and 3,000 at n=128 in the pipeline |
| D, m_D | the decoy (§1.4); m_D = m_C |
| A | the stage-1 **sliced sandwich**, on 2n wires |
| G2, \|G2\| | the stage-2 circuit, or **gadget** (`tdp.mpmct1`), on 4n wires, and its gate count. Code blocks hold \|G2\| in the shell variable `NG2` |
| G3 | the stage-3 mixed circuit (`db_mixing.mpmct1`; `s3.mpmct1` under val_chain.sh), on 4n wires |
| G4 | the split circuit (`split.mpmct1`, with the mixer checkpoint `split.state`) |
| G5 | the crossing-walk circuit (`crossing.mpmct1`); the packer's input is the last circuit produced |
| L_r, D_r | the lattice of cuts and the deliverable of refinement round r; L_0 = the packer's `rep512.bounds`, D_0 = `packing/deliverable.tdp4` (§6b) |
| 𝒢 | the published permutation: the function of `unpacked.mpmct1` and, equivalently, of the ANF deliverables (Overview) |
| G | the packer's (stage 6) **window gate cap**, 512 in production (`packing.window_gates`; `window_pack3.py` argument 4, §6a.3). In `repair.sh TAG G N`, G is only the file-name key `<G>` of `packed_G<G>` / `rep<G>` (§6a.5) |
| P_a, P_b | the two operand polynomials of a stage-2 fire (§2b.5) |
| B_mono | the packer's soft monomial budget per window (§6a.3) |
| K (`k`) | band wires per mask cycle (`EMBEDDED_MASKING_K`, production 2; gen `mod.rs:254` at `ea2e516f`, §2b.2). The dropped retirement lever is always written "K=64 retirement" (`EMBEDDED_MASKING_RETIRE=64`, §2a.5) |
| `packing/packed_G512`, `packing/rep512` | the packer's packed circuit, and the repaired circuit with its window lattice (`rep512.mpmct1` + `rep512.bounds` = L_0, the lattice the refinement starts from; the refined lattice is `unpacking/bounds.final`) |
| config A / config L8 | the recommended stage-2 environment without / with bounded chains (`CHAIN=2 CHAIN_LEN=8`); the managed presets `preprocessing.preset = "A"` / `"L8"` (default) |
| vA64a … vL128b | the validation replicates `runs/ncdb_ab_20261001/waist/v{A,L}{64,128}{a,b}/` |
| stubborn window | a packed window that does not resynthesize from its ANF within the delivery budget (§6b); pending / converged / failed are the per-window states of a refinement round |
| delivery seed | `unpack_seed`, the fifth entry of `SEEDS`, read by `tdp_unpacking run --refine` |

The prototype scripts name their stage-2 and stage-3 inputs `A` and `B`. `repair.sh` also uses `B` for its bounds path. This document calls the circuits G2 and G3 throughout, and keeps A for the sandwich.

### Data flow

The figures of the first block are those of the eight validation replicates of the four-stage chain (`skip_stages = [4, 5]`, `anf_only`; §6a.7, §8.9):

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
G3 (db_mixing.mpmct1)          1.59-1.66 M (×2.04-2.13)       4.59-4.79 M (×2.05-2.14)
                               (~30 MB text)                  (~90 MB text)
  │  stage 6  packing (tdp_packing run; oracle: window_pack3.py → repair.sh: cancel ~7%,
  │           slides, merge/split → anf_dump_eval.py)
packing/deliverable.tdp4       2,588-2,793 windows            7,053-7,416 windows
  (four-stage chain)           26.0-32.8 MB (xz 2.8-3.1 MB)   103-107 MB (xz 9.5-9.8 MB)
```

The six-stage chain continues from G3 (figures from `docs/notes/stage45_revival_20261011.md` §1, `docs/notes/seven_stage_20261011.md` §4 and `docs/notes/stubborn_windows_20261012.md` §4; the n=64 L8 runs; the move budget of the measured walks is given, since the default is now 3×target and no run at 3× is recorded yet):

```text
                               n = 64                         n = 128
  │  stage 4  splitting (circuit_mixer --split --split-stop)
G4 (split.mpmct1)              3.03 M (70 s)                  8.66 M (8.5 min)
  │  stage 5  crossing walk (circuit_mixer --resume split.state --target-size 2×)
G5 (crossing.mpmct1)           5.49 M in 46 min at 6×target;  15.7 M in 6.3 h at 6×target
                               5.38 M in 7 min at 2×target
  │  stage 6  packing (tdp_packing run)
packing/deliverable.tdp4       ~10.4k windows, 2.2 h          29,969 windows, 7.7 h
  │  stage 6  unpacking with lattice refinement (tdp_unpacking run --refine)
unpacking/unpacked.mpmct1      seven-stage n=64 run: 10,366 windows, 5,153,273 → 4,766,644 gates (0.925),
  (published; internal:        157 windows failed before refinement; bisected to the end: 10,540 windows,
  unpacking/deliverable.tdp4)  0 verbatim. n=16 six-stage run: 678 → 696 windows in 3 rounds.
```

With 8 jobs running concurrently, a four-stage run took about 40 min at n=64 and about 2 h at n=128, and stage 3 took 74-88% of it (§8.9). In the six-stage chain at n=64 the walk takes 7 min (2×target budget) to 46 min (6×), the packer 88 min to 2.2 h on the crossing output and the unpacking about 1 h on 8 threads; at n=128 the walk alone took 6.3 h and the packer 7.7 h.

### Threat model in brief

- **What ships.** The circuit `unpacking/unpacked.mpmct1`: the canonical-order concatenation of the resynthesized windows, a plain `mpmct1` gate list with no window lattice, no ANF and no stage provenance. Under `delivery.anf_only` the published object is instead `packing/deliverable.tdp4`, the window ANFs on the packer's lattice behind a 48-byte container header (window count, widths, CRCs), and the recipient does the unpacking. Everything else is an internal record: the two ANF deliverables (in the default mode), the bounds, the per-round containers, the packed and repaired circuits, the reports `packing.report` and `unpacking.report` (counts and parameters only; not part of the TDP) and the logs (§6a.1, §6a.8, §6b).
- **What the adversary sees** (default mode). A circuit, nothing else. It can cut the circuit anywhere, compute the function of any segment, compose segments and resynthesize them into circuits of its own; it is never handed the window lattice or the ANFs, and it does not see how a window was spelled before resynthesis: the packer's spelling is gone, and the synthesizer's spelling is what it holds. Nothing else ships: not C, not the sandwich or the masked circuit of stages 1-2, not the mixed, split or crossed circuits of stages 3-5, not the packer's gate order or its lattice. The measurements that model this adversary are the all-cuts census of the resynthesized circuit against C (at n=64 the unpacked circuit materialises 5 of 936 interior C values at its own cuts; the raw packed circuit 0) and the uniform re-cut-and-resynthesize experiments, which re-cut the circuit uniformly and never use the delivered lattice: W=2000 on the n=64 crossing output recovers 244 of 936 interior C values (241 with a 10× time budget; canonical reorder changes it by ≤ 2%), while the raw circuit exposes none at any cut; stages 4 and 5 do not close this leak (§6b, §7.6; `docs/notes/stubborn_windows_20261012.md`, `docs/notes/stage45_revival_20261011.md`). Under `anf_only` the older framing applies: the adversary holds the window ANFs, hence the wire state at every window boundary on inputs of its choice, and a window has no interior cuts; it can compose adjacent windows into wider ones and resynthesize any window or merged window into a circuit of its own, but cannot see how the function was spelled inside a window. In both modes the unpacking sub-step's synthesizer sees only the window ANF (the separation guarantee, §6b), and the refinement's exposure-aware cuts are chosen so that no interior C value is affine at a new cut, each one re-scanned (`check.new_cuts`).
- **Security targets** (§6a.1, §7.3, §7.5):
  1. **C-view at the floor.** No interior value of C is an exact affine function of any boundary state. Only the n outputs are visible, and those are visible by definition.
  2. **Low linkability.** Few stage-2 values can be recognized at more than one place in the artifact (L2, M2 and anchors). This target is informational: there is no acceptance threshold. §8.7 gives the replicate reference ranges.
- **What is not measured.** Only exact affine predictors, exact equality and (on one n=64 run) degree-2 reachability from window entry wires have been measured; approximate predictors have not. The C-view, the proxy census and linkability are evaluated at the cuts of the internal window lattice, that is, on the ANF representation; on the published circuit only the verbatim census has been run (the all-cuts census above and the W=2000 re-cut), and affine and linkable exposure of the resynthesized circuit at its own cuts have not (§6b.5). The adversary's input domain (honest slice or all wires) is not yet fixed (§6a.1, §7.1).
- **Secrets.** C is a deterministic function of the generator seed. Every file other than the published circuit (under `anf_only`, the packer's `deliverable.tdp4`) is secret or a secret intermediate, the ANF deliverables, the reports and the logs included, and stage 6 reads C in both sub-steps (the packer's repair and final C-view scan; the refinement's cut choice, one C-view scan per round, and its zero-slice check). So all six stages run in the secret-holding environment (§1.8, §6a.8, §8.8).

### What each stage contributes

| stage | contributes | leaves open |
|---|---|---|
| 1 sliced sandwich | Fixes the problem shape: a public zero slice and an answer register. Reversal yields the random D⁻¹, not C⁻¹, which removes the "reverse the gate list" attack | No hiding at all: on its slice, A runs C gate by gate in the clear |
| 2 preprocessing (embedded masking) | Fires each sandwich gate under ≥ 2 balanced quadratic masks keyed on a band that is filled from the input, with a mid-fire straddle mask that hides which gate fired. Every gate has ≤ 2 controls, the vocabulary the store digests. Slice guards sit on both ports. Carrier migration (dose 8, persistent) erases the data/band wire classes. PREMASK and SEED_AVOID_C1 close the first-layer fringe, and L8 chains cut bracket rebirths. No affine relation to C beyond the outputs was found at any scanned stage-2 cut. The scans were strided: every 16th / 128th cut (n=64 / n=128) in the 29-variant study and every 64th / 256th in the replicates, with dense scans of the final ~4k / ~8k gates on a subset of the study variants (§7.6) | Verbatim rebirths of band values (bracket restores, migration hops); recognizable unshuffled guard and tail blocks; cuts between scan points are untested |
| 3 database mixing | Un-invertible respelling by exactly verified store splices; dilution to about 2× size; digestion into about 88% g57 gates (L8); long-range wire-frame twists | Fringe C-view leaks appear only after reordering (localized to the mixer on one artifact, §7.6); duplicate commuting gate pairs (~7% of gates, source unknown) |
| 4 splitting | Every complemented g57 gate `a ⊕= b ∨ ¬c` → `{a ⊕= b, a ⊕= ¬b ∧ ¬c}`; with p=0.8 an absorbed NOT twist joins the one-control piece to a distant bracket; runs to exhaustion; ×1.9 gates; the sparsest ANF per gate of any stage | Adds no hiding by itself (0 C values at any cut before and after); the merged-window resynthesis leak is unchanged |
| 5 crossing walk | Fragments shot to their first collider, three exact case-split rules, thermostat at 2× its input (temperature target/25), budget 3× target; multi-control gates (median 3 controls, up to 12); ×1.8 gates | ANF-dense tails (p90 monomials 3-7× stage 3); the packer needs ~4× the windows and 100× the time; merged-window resynthesis still exposes 244/936 interior values at W=2000 |
| 6 delivery: packing | Replaces the mixer's gate list by windows, each collapsed into the canonical ANF of its permutation, so the packer's and the mixer's spelling never leave the stage; on that windowed representation (the `anf_only` artifact) a value alive only inside one window sits at no cut. Repair cancels duplicate pairs, slides completers and breakers, and merges and splits boundaries, trying to bring the C-view to the floor. It runs at most 4 fix and 4 fallback iterations (`packing.fix_iterations`, `fallback_iterations`); the Rust packer refuses to pass on a floor miss (exit 5) or an over-budget window (exit 6) unless `packing.allow_floor_miss` / `allow_over_budget` (§6a.5) | Under `anf_only` the window lattice is visible and deterministic given the run; in the default mode the resynthesized windows have interior cuts again (§6b); repair needs C; the oracle serialized a floor miss (vL128a's first repair ended at 129) |
| 6 delivery: unpacking with lattice refinement | Every window resynthesized from its ANF alone by a synthesizer that sees no gate and no window position (the separation guarantee); a window that fails is cut in two at an exposure-aware cut and both halves re-emitted and unpacked, recursively (~90% converge per round; 157/10,366 failed at n=64 and bisection ends with 0 verbatim); outputs the published canonical-order circuit beside the refined container it was resynthesized from | The refined lattice is finer where windows were ANF-dense; a window still failing after `max_rounds` is emitted verbatim (`status=verbatim_used`, NOT recommended); the published circuit has interior cuts of its own: 5 of 936 interior C values at its cuts (n=64) and 244 of 936 under uniform W=2000 re-cutting and resynthesis (§7.6) |

### Recommended configuration at a glance

The supported way to run the pipeline is the managed driver. Its defaults run all six stages (`docs/tdp_pipeline.md`):

```sh
cp configs/tdp.toml configs/local.toml        # set database.regular_dir and source.wires (64 or 128)
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

The validated four-stage configuration is the same with `run.skip_stages = [4, 5]` and `delivery.anf_only = true` in `[run]` / `[delivery]` (bash: `scripts/tdp_gen.sh --source-wires N --run-directory DIR --run-skip-stages 4,5 --delivery-anf-only`).

The reference driver below is what the validation runs used; it is now a
parity tool (the reproduction test runs it in a sandbox):

```sh
bash scripts/windowed_delivery/waist/val_chain.sh TAG N MC S SLICE CFG   # parity tool; CFG = L8 (recommended) or A;
                                                                          # outside the hard-coded run tree run it via tests/manual/stage4/reproduce_val_chain.sh
```

with (N, MC, S, SLICE) = (64, 1000, 384, 1280) or (128, 3000, 896, 2560) (§8.3). §8.4-§8.7 explain each step and the pass criteria.

The block below is the driver body (`W/val_chain.sh:6-35`), self-contained, with one change. The inert `--p-comp 1.0 --p-any 0.1` is dropped: they govern slot-3 (crossing-walk) moves, which stage 3 never runs at `--p-db 1.0`; stages 4 and 5 set them to 0 explicitly (§3.2, §4, §5).

```sh
set -uo pipefail                                         # no -e: failures do not stop the run (§7.2)
ROOT=/Users/rancanetti/code/local_mixing                 # hard-coded in the packing scripts (§8.1)
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
FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  target/release/circuit_mixer --input $R/tdp.mpmct1 --output $R/s3.mpmct1 --tdp --db-mixing \
  --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
  --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
  --no-local-verify --seed $MSEED > $R/s3.log 2>&1

# background stage-2 checks: rebirth census and span check (§8.6)
python3 $W/mult_census.py $R/tdp.mpmct1 $TAG > $R/mult_s2.txt 2>&1 &
STEP=$([ $N = 64 ] && echo 64 || echo 256)
SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=$R/span_s2.tsv target/release/hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 \
  --g $R/tdp.mpmct1 --n $N --exposure --c-step 1 --g-step $STEP --out $R/span_s2 2>&1 | grep "\[span\]" > $R/span_s2.txt &

# packing (stage 6's first sub-step; "stage 4" in val_chain.sh): pack (run from RD), repair, serialize.
# Managed equivalent: tdp_packing run --stage2 tdp.mpmct1 --stage3 db_mixing.mpmct1 --source-c tdp.mpmct1.source_c.g57 --n N --out-dir packing
cd $RD
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1 > $R/pack_G512.txt 2>&1
bash $W/repair.sh $TAG 512 $N > $R/repair512.log 2>&1      # 512 = the packed_G512 file-name key
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG > $R/deliver.txt 2>&1
gzip -9 -kf $R/deliverable.bin; xz -9 -kf $R/deliverable.bin
ls -la $R/deliverable.bin* | awk '{print $5, $9}' >> $R/deliver.txt
wait                                                     # for the two background checks
echo "$TAG VALIDATION DONE"
```

Seeds are fresh CSPRNG draws (§8.2). They are passed on command lines and so are visible in `ps`: an open gap of `val_chain.sh` (the managed driver hands every seed over on `/dev/fd/N`, §8.2). The §8.7 pass criteria are greps of exactly the files this block writes: `gen.log`, `s3.log`, `mult_s2.txt`, `span_s2.txt`, `pack_G512.txt`, `repair512.log` (plus `repair.sh`'s own `rep512.cview.log`, `anf_rep512.txt` and `link_rep512.txt`) and `deliver.txt`. The managed driver writes the same content as `stage12.log`, `stage3.log`, `stage2_mult.txt`, `stage2_span.txt`, `packing/pack_G512.txt`, `packing/repair512.log`, `packing/rep512.cview.log`, `packing/anf_rep512.txt`, `packing/link_rep512.txt` and `packing/packing.report` (the content of `deliver.txt`), plus `stage4.log`, `stage5.log`, `stage6.log` and `unpacking/unpacking.report` for stages 4-6.

### Conventions of this document

- **The code is the authority.** Where a code comment or another doc disagrees with the code, this document follows the code and marks the other text **[stale]**. Claims labelled *derived* or *inferred* are reasoned from the code and were not measured or stated by the authors.
- **Sections.** §1-§6 are the six stages (stage 2 is split into §2a-§2c; stage 6 into §6a, packing, and §6b, unpacking with lattice refinement). §7 covers verification, metrics and attack tooling, and §8 how to run the pipeline. The managed-driver gap analysis that this document once carried as Appendix A is closed; its content is in `docs/local_mixing_history.md` (October 8, 2026). The 2026-10-12 renumbering moved the old §4 (packing) to §6a, §5 to §7 and §6 to §8; every cross-reference in this document was moved with it.
- **Paths** are relative to the repository root. Abbreviations used across sections:

| short form | path |
|---|---|
| `RD/` | `runs/ncdb_ab_20261001/` |
| `W/` | `runs/ncdb_ab_20261001/waist/` |
| `R` | a run directory `W/<TAG>` |
| `SW/` | `scripts/windowed_delivery/` (the tracked oracle snapshot; `SW/waist/` = `W/`) |
| `EM` | `src/stages/preprocessing/embedded_masking.rs` |
| `HA` | `security_tests/heatmaps/hmap_affine.rs` |
| `RESULTS.md` | `docs/notes/ncdb_ab_20261001_RESULTS.md` (lab notebook) |
| `STAGE2_LINKABILITY.md` | `docs/STAGE2_LINKABILITY.md` |

Two files are both called `mod.rs`. In §1, §2a and §2b, a bare `mod.rs` (or `gen:`) is `src/programs/gen_sandwich_gadget/mod.rs`. In §3, a bare `mod.rs` is `src/programs/circuit_mixer/mod.rs`. Sections with further abbreviations list them at their start.

**Code-pointer pins.** Pointers in §1-§3, §6a, §7 and §8 are as of commit `ea2e516f` unless marked "at HEAD"; §4, §5 and §6b are at `83e44ec5`. Since `ea2e516f`, `EM` grew by 172 lines (typed `EmbeddedMaskingParams` fields at EM:174-305; EM pointers below 213 shift by about +53, 213-1607 by +73..+79, 1607-2444 by +55..+63, later ones by +57..+60), gen `mod.rs` by about 106 lines after its line 182 (the `stage2_knobs_from_env` adapter at gen:103-183; pointers up to gen:92 shift by +3), and the curated-store removal and the port moved most lines of `circuit_mixer/mod.rs` and `cli.rs`, `src/stages/db_mixing/replacement.rs` (1,685 to 1,315 lines: 503 deleted, 133 added), `src/engine/mixer/replacement.rs`, `frozen.rs`, `lookup_cache.rs`, `params.rs`, `reporting.rs`, `transformations.rs`, `state.rs`, `gate_order.rs` and `canonicalization/window.rs`; `preprocessing_shuffling.rs` (`sh.rs`), `scheduling.rs`, `sampling.rs`, `transport.rs`, `provenance.rs`, `canonicalization/xgate.rs`, `keys.rs` and `rules.rs` are unchanged and `construct.rs` moved by at most 4 lines. Current anchors (HEAD `83e44ec5`):

| anchor | at `ea2e516f` | at HEAD |
|---|---|---|
| gen `run()`; `[gen]` header; `wrote source C`; `preprocess_sandwich` calls; `verify_payload` | `mod.rs:96`; `:184-188`; `:209-217`; `:292`; `:357` | `mod.rs:184`; `:290`; `:323`; `:401` / `:447`; `:466` |
| gen `parse_shuffling_controls`; its call | `mod.rs:42-93`; `:173-182` | `mod.rs:45-96`; `:279-287` |
| gen `stage2_knobs_from_env` (the only reader of the `EMBEDDED_MASKING_*` names) | — | `mod.rs:103-183` |
| EM `production()`; `managed_tdp`; `seed_band_rich`; `r = 0 → np` | `:199`; `:183`; `:951`; `:1178` | `:249`; `:236`; `:1029`; `:1257` |
| EM `hot_intervals`; `product_fire`; `emit_slot!`; "RC sign-off pending"; `apply_shuffling` call | `:500-554`; `:619`; `:1517`; `:2399`; `:2672` | `:591`; `:676`; `:1591`; `:2462`; `:2742` |
| mixer `--tdp` block; COMP-override guard; curated warning; `canon caps`; stop text; final float | `mod.rs:95-109`; `:171-207`; `:214-230`; `:872`; `:882-899`; `:901` | `mod.rs:90-100`; `:159-200`; `:203-221`; `:808`; `:812-820`; `:830-835` |
| `DbLayer::tdp()`; `DbLayer::db_mixing()` | `cli.rs:789`; `:806` | `cli.rs:735`; `:752` |
| slot-3 walk in `Mixer::run` | `scheduling.rs:545-578` | `scheduling.rs:546-577` (unchanged file) |

**Packing scripts: one convention for the whole document.** The tracked snapshot `SW/` = `scripts/windowed_delivery/` is authoritative (the oracle, never edited; every comparison runs it in a path-rewritten sandbox, `tests/fixtures/stage4/oracle_sandbox.sh`). A packing-script path, whether written `RD/x.py`, `W/x.py`, `waist/x.py` or as a bare script name, means that file in the snapshot layout: `SW/` for top-level scripts, `SW/waist/` (= `W/`) for the others. The run-tree copies under `runs/ncdb_ab_20261001/` are byte-identical to the snapshot (refreshed from the run tree in `5619b867`, 2026-10-08), with two recorded differences: `waist/order_repair.py`'s run-tree copy was reported shifted by one line after line 162 relative to the first snapshot (`7163e457`) and the snapshot now carries the patched (live) version (§6a.5), and `waist/val_chain.sh`'s run-tree copy still passes the inert `--p-comp 1.0 --p-any 0.1`. `waist/deliver.py` (the Python reference of the delivery loop) exists only in the snapshot. (The run tree is outside the worktree; the byte identity was not re-verified for this revision.)


## Stage 1: the sliced sandwich

Stage 1 takes the secret source circuit C (n wires; any gates with up to two controls, supplied in g57 form) and builds the **sliced sandwich** A, a reversible circuit on 2n wires. On a public zero slice, A computes C. Run in reverse on that same slice, it computes an unrelated random function.

Stage 1 adds no hiding: on its slice, A still runs C in the clear, gate by gate. Its job is to fix the problem shape (which inputs are free, which are pinned to zero, and where the answer appears) and to remove the trivial "reverse the gate list" attack (`docs/local_mixing_documentation.md:1262-1273`, `:1402-1405`). Stages 2 to 4 do the hiding.

### 1.1 Code location and invocation

All of the logic is in `src/stages/sandwich/construct.rs`, re-exported by `src/stages/sandwich/mod.rs:1-4` ("Stage 1: source preparation and sliced-sandwich construction"). Stage 1 has no binary of its own. `gen_sandwich_gadget` (`Cargo.toml:87-89` at HEAD `83e44ec5`; `src/programs/gen_sandwich_gadget/main.rs:3-5`) runs stages 1 and 2 in one process (pointers at `ea2e516f`, the HEAD location in brackets):

- stage 1 is `src/programs/gen_sandwich_gadget/mod.rs:96-232` [HEAD: `run()` starts at `:184`, stage 1 ≈ `:184-335`];
- the stage-2 handoff is `:236-295` [HEAD: ≈ `:340-455`, `preprocess_sandwich` at `:401` / `:447`];
- payload verification is `:356-369` [HEAD: `verify_payload` at `:466`];
- the final gadget is written at `:371` [HEAD: after `:466`].

The managed driver's call (`scripts/tdp_gen.sh`, stage 1+2 block, `:716-722`) is `gen_sandwich_gadget $RUN/tdp.mpmct1 $N $M_CD $M_CD $S_SL 1 $SLICE_G @/dev/fd/3 @/dev/fd/4 @/dev/fd/5` under `env -i …` plus the preset's six variables, `EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8` (preset L8; `scripts/tdp_gen.sh:392-403`). The parity reference is `W/val_chain.sh:15-16`:

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

`val_chain.sh` neither pins nor unsets `SANDWICH_VARIANT`. If the variable is exported as `balanced` or `mirror`, the run silently builds the balanced sandwich, and the reverse check is then skipped (`src/stages/preprocessing/verify.rs:71`). The driver should pin it. [Managed driver: resolved. Every stage binary runs under `env -i` plus an allow-list; an inherited `SANDWICH_VARIANT` is refused (`driver` class in `scripts/tdp_env_registry.tsv`), and `stage12.recipe` records `sandwich_variant=classic`.]

Two parsing behaviours matter.

- **Silent fallback.** Every numeric positional is parsed with `.and_then(|s| s.parse().ok()).unwrap_or(default)`. A malformed number therefore falls back to its default without any error.
- **What panics during parsing.** Only four things: a missing `<out>` (`:97-99`), an unknown mode (`:121-123`), an unknown variant (`:132-134`), or a 13th argument (`:137-140`).

Before it writes anything, the generator does three things:

1. It asserts that no `BV5_*` or `PROD_*` environment variables are set (`mod.rs:150-171`).
2. It validates the stage-2 shuffling controls (`:173-182`; §2c.1).
3. It prints a header line, `[gen] n=… |C|=… |D|=… s=… rg_freq=… slice_gates=… seed=… gadget_seed=… sandwich_seed=… gadgetization_mode=… sandwich_variant=…` (`:184-188`). A seed given as `@PATH` (read from PATH; the managed driver passes `@/dev/fd/N` so no seed is on argv) is printed as `(file)`.

**Under val_chain.sh the header contains the raw seeds, so `gen.log` is secret material**; the managed driver passes `@/dev/fd/N` seeds, the banner prints `seed=(file) gadget_seed=(file) sandwich_seed=(file)` (checked by the driver, which removes the artifact otherwise, `scripts/tdp_gen.sh:725-729`), and `stage12.log` is not seed material (see 1.8; the secret is `$RUN/SEEDS`).

The validated parameter sets are taken from the `gen.log` headers of `W/{vA64a,vA128a,vL128a}` and from `docs/STAGE4_RUST_DESIGN.md` §2.1 (the "sizes" row):

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

An empirical check found no sign that the streams are related in practice. The same C (via `TDP_SOURCE_C`) with fresh sandwich and gadget seeds gave a stage-2 output with 0 of 4,096 hot in `qc_scan` and a null ridge against the reference (`RESULTS.md:161-167`). Both tests are defined in §7.9.

All three seeds are `u64`. The `val_chain.sh` driver passes a single 31-bit draw for all three (`val_chain.sh:11`, `:16`). The managed driver draws five seeds (gen, mix, split, cross, unpack = the delivery seed; each ≤ 2^63−16) plus a 256-bit `run_key` into the private `$RUN/SEEDS` (format 3, mode 0600) and passes gen's three as `@/dev/fd/3..5`, all equal to `gen_seed`; the other binaries read theirs from `--seed-file /dev/fd/3`. Seed handling across the whole pipeline is summarized in §8.2.

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

**Stage 6 depends on the sidecar.** The packer scores its output against C (Rust: `stages::packing::cview` in the repair loop and the final scan; oracle: `repair.sh` running `hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 … --exposure --g-cuts`, `W/repair.sh:17-21`), and the unpacking sub-step reads it again (`tdp_unpacking run --refine --source-c`): one `CView::leaky` scan per round over the 32 candidate cuts of every failed window, and the final zero-slice check (§6b). The oracle's target is a C-view visible count ≤ n, which it pursues in two capped loops (§6a.5):

- up to 4 `order_repair` fix iterations (`:22-26`);
- then up to 4 `merge_bounds2`/`anf_split2` fallback iterations (`:28-33`).

Reaching the floor is not enforced by the oracle: the script continues to `anf_compose` and `link_census` without a status check (`:35-39`). In the Rust port reaching the floor is enforced: a visible non-output row is exit 5 and an over-budget window exit 6, with no deliverable written, unless `packing.allow_floor_miss` / `allow_over_budget` are set (then `status=floor_miss`/`over_budget`, `NOT_RECOMMENDED`). The floor of n is the output values themselves, which are trivially visible because C(x) sits on the answer register (`STAGE2_LINKABILITY.md` §2, "C-view"). `anf_split2.py` also runs `hmap_affine` on the sidecar, and `val_chain.sh:26-27` uses it for the stage-2 span check (§7.3).

`source_c.g57`, `sandwich.mpmct1`, `gen.log` and `seed.txt` (`val_chain.sh:14`) are plaintext secrets under val_chain.sh (managed driver: `SEEDS`, `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` and every circuit; `stage12.log` holds no seed). `source_c.g57` must survive until stage 6 finishes (both sub-steps read it) and must never ship (§8.8).

### 1.8 Measured profile and open issues

**Measured profile.** The notebook measured the stage-1 sandwich at n=64 (m=1000, s=384) (`RESULTS.md:611-627`):

- **Positionally flat.** The low-half touch fraction is about 0.90 in every bin. The per-wire position standard deviation is 0.289 on every wire. That is 1/√12, the value for a position spread uniformly over the circuit (`:622-623`).
- **Volume and role imbalance.** Median touches are 115 per low wire against 13 per high wire. The target fraction is 0.38 against 0.08.

These values follow from the construction (*derived*):

| | formula | value at n=64 | target fraction |
|---|---|---|---|
| high wire touches | `1 + 2s/n` | 13 | 1/13 ≈ 0.077 |
| low wire touches | `(3(m_C+m_D) + (5/3)·2s + n)/n` | ≈ 114.75 | `(m_C+m_D+2s)/7344` ≈ 0.377 |

**Open issues** (as found on 2026-10-07, against `val_chain.sh` and the pre-port managed driver). Items 1, 2, 5 and 6 are resolved in the managed driver (on `main` since 2026-10-12; status notes in brackets); `val_chain.sh` itself, the historic reference, is unchanged. Items 3, 4, 7 and 8 are still open.

1. **Seed entropy.** C is a deterministic function of a `u64` seed. `val_chain.sh` passes the same 31-bit draw as all three seeds, so enumerating that one value recovers C, D, S1/S2, both merges, the float and every seeded stage-2 stream (masks, guards, band seed and reseed). A candidate C can be tested with a few slice evaluations of the published circuit. Even a full-width seed caps C at 64 bits of entropy (the `fastrand` state). The binary's default `seed=1` is also accepted silently. [Managed driver: five independent seeds (gen, mix, split, cross, unpack = the delivery seed) of up to 63 bits from the OS CSPRNG (`seed_bits=63`) and a 256-bit `run_key` for the HMAC markers, kept in the private `$RUN/SEEDS` (format 3); `val_chain.sh` still draws one 31-bit gen seed. The 64-bit `fastrand` cap on C remains (use `source.path` for production, design §13.1).]
2. **Seeds in logs and on the command line.** The `[gen]` header prints all three seeds (`mod.rs:184-188`). They land in `gen.log` under `val_chain.sh` and in `$RUN/stage12.log` under `tdp_gen.sh`. Both drivers passed the seeds as argv (`val_chain.sh:16`, plus the stage-3 seed at `:23`; the pre-port `tdp_gen.sh`; today's stage 1+2 block, `scripts/tdp_gen.sh:716-722`, passes `@/dev/fd/3..5`), where other local users can read them with `ps`. `val_chain.sh` sets no umask: in `vA128a`, `gen.log`, `seed.txt` and `tdp.mpmct1.source_c.g57` are mode 644. [Managed driver: no seed is on argv or in a log; `tdp_gen.sh` hands the three gen seeds over as `@/dev/fd/3..5` here-strings and the mix seed as `--seed-file /dev/fd/3` (`scripts/tdp_gen.sh`, stage 1+2 and stage 3 blocks), gen prints them as `(file)`, and the run directory is written under `umask 077`; the split seed, the cross seed (when stage 5 starts from a file) and the delivery seed likewise reach `circuit_mixer` and `tdp_unpacking` as `--seed-file /dev/fd/3` (`:979`, `:1018`, `:1142`).]
3. **Unchecked pins on supplied C.** `TDP_SOURCE_C` does not check that pins are distinct. A target that equals a control produces an XGate that breaks the invariant at `xgate.rs:18` and is not reversible. A gate with `x==y` silently becomes a NOT gate, which the masked compute then lowers incorrectly (§2b.11, defect 1).
4. **Dead parameter.** `rg_freq` is parsed and printed but never read. Removing it would shift every positional argument after it.
5. **Canonical |C|.** The pipeline runs 1000 at n=64 and 3000 at n=128. The library convention `sandwich_default_m(n) = max(n, round(n·log₂²n))` (`construct.rs:56-60`) gives 6272 at n=128. In Rust it was called only by the nonlinear291 capacity check (`src/tdp/config/validate.rs:160-165` at `ea2e516f`; at HEAD that block is `validate.rs:256-282` and uses `sandwich_default_s`). The pre-port managed driver computed its own default, `round(n·log₂²n)` with no floor, in Python (`scripts/tdp_gen.sh:348-357` at `ea2e516f`); today the bash default is `validated_gates`, with the formula as the Python-heredoc fallback (`scripts/tdp_gen.sh:503-517`), the Rust rule is `validate.rs:231-246` (`round_ties_even`, no floor) and the table is `VALIDATED_SOURCE_GATES` (`src/tdp/config/mod.rs:20`). That default was documented as `source.gates` in `docs/tdp_pipeline.md:84` (now its `:240`). [Managed driver: the default is now the validated table, 1000 at n=64 and 3000 at n=128 (`validated_gates` in `scripts/tdp_gen.sh`, `VALIDATED_SOURCE_GATES` in Rust; `source_gates_rule=validated`); other n need `source.gates` or the opt-in `source.gates_rule = "formula"`, which is recorded as an unvalidated size.]
6. **Managed path could not run the current configuration (pre-port).** The pre-port managed config rejected `shuffling_return_home=false` whenever shuffling was on (`src/tdp/config/validate.rs:118-124` at `ea2e516f`); today the only custom-preset constraints are `min_open_masks < max_open_masks` and `shuffling_segments ∈ {0} ∪ [8, ∞)` (`validate.rs:168-177`), and the presets L8/A set `shuffling_return_home = false` (`src/tdp/config/mod.rs:129-140`). `val_chain.sh` was gitignored (`.gitignore:3`), not in the `scripts/windowed_delivery/` snapshot, and still passed `--p-comp`/`--p-any`. See §8.10. [Resolved: presets L8 (the default) and A express the recommended stage-2 environment exactly, and `val_chain.sh` is tracked as `scripts/windowed_delivery/waist/val_chain.sh` (without the inert `--p-comp`/`--p-any`, the only difference from the run-tree copy).]
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

The reference invocation is `W/val_chain.sh:13-16` (shown in §1.1 and in the Overview). Dropping the two `CHAIN` variables gives config A. This configuration is the managed default: `preprocessing.preset = "L8"` (`--preprocessing-preset L8`) exports exactly `EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8` and nothing else (`scripts/tdp_gen.sh:392-403`; `Stage2Knobs::L8`, `src/tdp/config/mod.rs:129-140`); `"A"` is the same without the chains (§8.10; `STAGE2_LINKABILITY.md` §6).

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
- With the rich seed and persistent migration at dose 32 (not the recommended dose 8), the stage-2 data-vs-band touch AUC (§7.9; 0.5 means indistinguishable) went from 1.000/0.000 to 0.51/0.50 (`RESULTS.md:682`). This was run `W/s12_check`, 2026-10-05, before PREMASK, SEED_AVOID_C1 and chains existed, and it has not been re-measured at dose 8.
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

The stage-2 log lines from `vL128a/gen.log` (lines 5-8; the managed driver writes the same stream to `stage12.log`) are:

```
[chain] k=2: 316961 brackets chained
[gen] embedded-masking: 23302 atoms, band=256 wires, 2823 band-seed + 2872 band-reseed + 2560 slice-guard gates each side
[gen] compute: 2033637 fire gates + 190845 shuffling gates (63360 transfers, 0 skipped cuts, 255 ports restored, 0 jit remasks, 63360 drain/other) ; 15840 rerand bursts
[gen] diversified gadget: 2235297 gates, 512 wires
```

followed by `[gen] verify PASSED (256 bit-sliced samples, payload wires 128..256)` and `[gen] reverse verify PASSED (…: D^-1 emerges under backward evaluation)`. The `[chain]` line appears only when CHAIN is set (vL64a: 109,464 brackets). It goes to stderr from `EM:2703` and comes before the report lines; it appears in `gen.log` because `val_chain.sh` redirects `2>&1`. The next two lines come from `mod.rs:297-317`, and the gadget line from `mod.rs:350-354`. Under val_chain.sh the first log line prints the seeds, so its gen logs must be handled like seed files; under the managed driver it prints `seed=(file) gadget_seed=(file) sandwich_seed=(file)` and `stage12.log` is not seed material.

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
3. ~~**Managed preset.**~~ Closed 2026-10-09: presets `L8` (default) and `A` (§8.10).


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

`EmbeddedMaskingParams::production` (EM:196-217) supplies the defaults. `managed_tdp(seed, n)` is the same preset with `active_wires = n` (EM:183-188). The gen adapter (`gen:248-290`) builds the params from `base = managed_tdp(gadget_seed, n)`. Most `EMBEDDED_MASKING_*` variables fall back to `base`, with three exceptions: `min_open` falls back to the literal 2 (gen:268-271), `burst_band_only` to `false` (gen:265-266), and `encoded_io` is hard-coded `false` (gen:267). All three match production today. At HEAD `83e44ec5` the library takes every knob as a typed field (`EmbeddedMaskingParams`, EM:174-305) and only gen's adapter `stage2_knobs_from_env` (gen:103-183) reads the `EMBEDDED_MASKING_*` names (design §11.4); the managed driver sets them from the preset (L8/A) or, under `preprocessing.preset = "custom"`, from the `preprocessing.*` keys, and refuses every inherited `EMBEDDED_MASKING_*` variable (`scripts/tdp_env_registry.tsv`), the diagnostic ones (`HOT_MANIFEST`, `DIAG`, `LGI_DEBUG`, `STRADDLE_CENSUS`, `DRIFT_DEBUG`, `DUMP_GATES`) included, which are therefore available only from a standalone gen run. This is the canonical stage-2 knob table (EM pointers at `ea2e516f`):

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

**Current pipeline configuration** (preset `L8`: `configs/tdp.toml` `[preprocessing]`, `scripts/tdp_gen.sh:392-403`; reference `W/val_chain.sh:13-16`; `STAGE2_LINKABILITY.md` §6):

- Stage 2b knobs: `PREMASK=1`, plus `CHAIN=2 CHAIN_LEN=8` (config L8).
- Knobs consumed by neighbouring parts: `SEED_AVOID_C1=1` (band seed, §2a.5) and `SHUFFLING=8 SHUFFLING_RETURN_HOME=0` (carrier migration, §2c).
- Everything else stays at the defaults above, with DRIFT on (but see 2b.6: under L8 it has almost nothing to act on).
- Dropping `CHAIN*` gives config A (`preprocessing.preset = "A"`), a supported alternative (validated alongside L8, 2026-10-08).
- `CHAIN_ROTATE`, `RETIRE*`, `ANC_RESERVE` and `JIT` were tested and dropped (`STAGE2_LINKABILITY.md:239-242`), and stay off.

This configuration is preset `L8` of the managed driver (§8.10); a direct `gen_sandwich_gadget` call has to export the six variables by hand.

Derived values and preconditions:

- `lgi_k = (k − k%2) + balanced = 3` band wires per mask, written `[x, y, z]` (EM:1181).
- The code asserts `r > lgi_k`, `np + r ≥ 8` under quad fire, `np ≥ 2` and `np + r < u16::MAX` (EM:1186-1205).
- `rng = StdRng::seed_from_u64(p.seed)` with `p.seed = gadget_seed` (EM:1256).

### 2b.3 Masks (LGIs) and their bookkeeping

**Mask term.** A mask on data wire `w` with band wires `[x, y, z]` is emitted as two gates, `g57(w,x,y); CNOT(w,z)` (`emit_lgi`, EM:1113-1134; `cycle_g57`, EM:281-291). Together they XOR the term

    M = 1 ⊕ y ⊕ x·y ⊕ z

onto the wire. Re-emitting the same two gates while `x, y, z` are unchanged removes the mask. The code calls such an open/close pair an **LGI** ("locally geodesic identity", `docs/gadgetization.md:102-105`). `emit_lgi` *toggles* the ledgers, so it is self-inverse.

**Why `z` is there.** The bare g57 term is 1 with probability 3/4. A wire under one or two such terms therefore keeps a linear correlation of 1/2 to 1/4 with its plaintext, which statistical tests detect (fire_corr, gauntlet w1; both in §7.9). The fresh `z` balances the term and keeps it quadratic. The cost is that each operand polynomial grows from 8 to 11 monomials at `max_open=3` (EM:116-128). **[stale]** The "+92% gates" in that comment (EM:125-126) assumed one gate per product monomial. With the current bracket emission (2b.5), and ignoring constant terms, a fire costs `1 + 6(q_a+q_b) + 28·q_a·q_b` gates balanced against `1 + 5(q_a+q_b) + 17·q_a·q_b` unbalanced (where `|L| = q`, EM:477-490). That is 137 vs 89 gates at q=2/2 (+54%) and 289 vs 184 at q=3/3 (+57%) (*derived*).

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

Disjointness matters because identical pairs cancel and identical `z`'s cancel, and because a `z` equal to another pair's wire folds the terms into an OR (`x ⊕ ¬x·y = x ∨ y`, biased 3:1). Each of these was measured at φ≈0.25, where φ is the binary (Pearson) correlation coefficient of §7.9 (EM:1406-1419). The comment says production bands never relax. **[stale]** Its "r = n ≥ 128" (EM:1419): the band is now `r = np = 2n` (EM:1178).

**Coverage rules:**

- `max_open = 3` is a soft cap: covers and refresh replacements may exceed it.
- `min_open = 2` holds at every instant between a wire's first and last mask (EM:144-154). With one mask, a single visible monomial can cancel the wire's only uniform term (gauntlet w2, §7.9). Before this rule, 26% of covered wire-time sat at one mask.
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

For source gate `gi` with target `c`, controls `(w, pol)` and `comp`, the steps are (EM:1949-2560 at `ea2e516f`; the EM pointers of §2b.5-§2c are at that commit and sit about +55..+79 lines later at HEAD, see the anchor table under "Conventions"):

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
- `ANC_POOL=all` would also admit data wires. It was rejected: a product XORed onto a masked data wire can cancel part of its mask (φ 0.134, gauntlet w1 flags on 6 targets versus none with the band pool, §7.9; EM:660-680).

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
- **Bounded `L=8` against config A after the packer's repair (stage 6, the packing sub-step).** The replicate deltas are tabulated once, in §7.6 ("L8 against A"). In brief (n=64 / n=128): L2 −12% / −13%, M2 −20% / −25%, proxy values (link census) −24% / −36%, anchors −39% / −29%, `deliverable.bin` (the body of `packing/deliverable.tdp4`) size +19% / −1%. Linkable froth grows by +0.7k / +3.3k values, almost all of them seen exactly twice. The slightly different figures in `STAGE2_LINKABILITY.md:132-135` come from the earlier finals, not the replicates.
- **`CHAIN_ROTATE=1`**, which gives each `L`-bracket segment its own atomic unit (EM:2258-2309), was dropped: no delivered gain, +48–73% size.

**Interaction with carrier migration.** **[stale]** The `FireBox` doc says band writes must be restored by the next unit boundary (`preprocessing_shuffling.rs:17-18`). Chains and drift violate this literally. Safety comes from the in-box written-partner exclusion (§2c.5).

### 2b.8 Band refresh: rerand slots with hot-targeting

**Plan** (EM:1294-1317):

- `straddle_slots = rerand_level` if set, else `m/(4k) = m/8`: 990 slots at n=128 and 354 at n=64. **[stale]** The "~875" in the comments at EM:80 and EM:192 (EM:80 and EM:245 at HEAD).
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

**Hot-value manifest** (`hot_intervals`, EM:500-554). It records the intervals during which a data wire has no open pair, i.e. its value is affine in plaintext and band. It is computed on the pre-migration stream. With a correct build only the I/O fringes appear; anything else is a defect. It is written only when `EMBEDDED_MASKING_HOT_MANIFEST=path` is set on a standalone `gen_sandwich_gadget` run (gen:287, 318-334 at `ea2e516f`; gen:395 and `:427` at HEAD) and must never ship with a deliverable (EM:512-513); the managed driver refuses the variable (registry class diagnostic, `scripts/tdp_env_registry.tsv`) and never writes it.

**`EMBEDDED_MASKING_DIAG`** (EM:2623-2663 at `ea2e516f`; the census is `fire_census` at EM:1906 at HEAD; standalone gen only, the managed driver refuses the variable, registry class diagnostic) prints:

- read counts, bare reads and `|ρ|` (= 2q);
- slot counts;
- replacement, fire-cover and min-open counts;
- `qq_blinded` and `relaxed`;
- the 8-class fire census.

The "straddled=m unplaced=0" field is hard-coded, and "read-cover LGIs" is always 0 (EM:1401). In gen.log, "fire gates" is the entire pre-migration compute stream (`preprocessing_shuffling.rs:481, 503`), not only fire units.

### 2b.10 Correctness evidence

- **Exhaustive unit test** (`tests/unit/stages/preprocessing/embedded_masking.rs:22-103`): n=6 with r=6, 24 random g57 gates, 6 seeds, all 64 data inputs × 8 band states. It covers quad/legacy reads, balanced on/off, `encoded_io`, K=2/3/4 and repair slots.
- **Not covered by unit tests:** PREMASK, chains and seed avoidance; these have only end-to-end checks and the byte-identity test of the typed-knob refactor (`STAGE2_LINKABILITY.md:221-226`; `tests/integration/stage2_typed_params.rs`). At HEAD the library takes every knob as a typed field (`EmbeddedMaskingParams`, EM:174-305); only gen's adapter `stage2_knobs_from_env` (gen:103-183) reads the `EMBEDDED_MASKING_*` names, so unit tests no longer depend on the shell environment, and the managed driver additionally refuses inherited `EMBEDDED_MASKING_*` variables (registry).
- **Validation runs:** all eight validation builds (vA/vL × 64/128 × a/b) pass forward and reverse verification on 256 bit-sliced samples (gen.log).

### 2b.11 Known defects and open questions

1. **Inverted 0-operand fire (latent).** `product_fire` emits `X` when `comp` is true (EM:621-625). A 0-control XGate fires when `comp=0` (`xgate.rs:51-58`), and the legacy path does this correctly (EM:403-407). The branch is unreachable for generated sources: C draws 3 distinct pins (`randomize.rs:22-34`) and D distinct wires (`sandwich/construct.rs:160-170`). It is reachable through `TDP_SOURCE_C`, i.e. `source.path` in the managed config (`src/tdp/config/validate.rs` checks only that the G57 circuit is nonempty and fits `source.wires`, so the gap applies to the managed config as well). A supplied degenerate g57 `[a,x,x]` passes `prepare_source` (`sandwich/construct.rs:64-90`, which checks only emptiness, width and count) and the g57 decoder (`src/circuit/formats/g57.rs:27-44, 127ff`, which checks only 3 wires per gate). `from_g57` then turns it into `x_gate(a)` (`xgate.rs:67-70`, `sandwich/construct.rs:330`), and `product_fire` emits nothing: a wrong circuit that only gen's payload verification would catch. Fix the branch to `if !comp` and/or reject `x == y` in `prepare_source`.
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
    - the `LGI_DEBUG` trace (a typed field at HEAD, EM:292, its env name mapped in gen's adapter) with hard-coded wires 7/29 (EM:1106 at `ea2e516f`; EM:1185 at HEAD);
    - the retirement, `ANC_RESERVE`, `CHAIN_ROTATE` and JIT code (still present: typed params EM:188-205 at HEAD, JIT in gen:78-95 and `sh.rs:214-436`; marked "drop", `STAGE2_LINKABILITY.md:239-242`);
    - the **[stale]** comments listed above, all of which survive at HEAD `83e44ec5` (HEAD lines in brackets): EM:22-23, 37-42 and 107-109 (quad-fire top-up/undo) [EM:22-23, 37-42, 108-109]; EM:80 and 192 ("~875" slots; the actual count is 990) [EM:80, 245]; EM:99 (`active_wires` "seed the band") [EM:99]; EM:101-104 (`extra_lgis`) [EM:101-104]; EM:125-126 ("+92%") [EM:126]; EM:245 ("3m+n") [EM:322]; EM:852-853 and EM:1152 ("the compute only READS the band") [EM:931]; EM:1419 ("r = n") [EM:1317, 1494]; the safety-drain "emits no rerand" [EM:2649]; EM:1154-1157 (carried layout returned in `final_layout`) [EM:1233]; `construct.rs:79-80` ("compute (which only reads the band)") [unchanged]; gen `mod.rs:20-22` (RETURN_HOME "is required"; `parse_shuffling_controls` accepts 0, gen:61-69, and the recommended configuration uses it) [unchanged]. The vacuous `assert!(true)` of old EM:1173 is gone;
    - plus `docs/gadgetization.md:60-63`, which describes the thin seed rather than the default rich seed (`SEED_NL=4`), and `RESULTS.md:1466-1467` (PREMASK "drained by the end-of-compute close").


## Stage 2c: carrier migration (preprocessing shuffling)

**Carrier migration** (called "preprocessing shuffling" in the code) rewrites the masked compute stream (§2b) after it has been emitted. While a data role is being written, the pass moves it off its physical wire. In the recommended mode the role keeps moving for the rest of the computation. The emitter works in *logical roles*:

- data roles `0..np`, where `np = 2n` is the sandwich width;
- band roles `np..np+r`, with `r = np` by default (EM:1178; EM:1257 at HEAD), so `total = np + r = 4n` wires (EM:1204): 4n state wires, of which stage 6 accepts at most 1,024, hence `source.wires ≤ 256` when stage 6 runs (`src/tdp/config/validate.rs:377-381`).

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

| env var (`gen_sandwich_gadget`) | field of `EmbeddedMaskingParams` | default | recommended | managed config (`configs/tdp.toml`) | meaning |
|---|---|---|---|---|---|
| `EMBEDDED_MASKING_SHUFFLING` | `shuffling_segments` | `0` (off) | `8` | `preprocessing.shuffling_segments` (custom only; 0 or ≥ 8; presets L8/A = 8) | dose: maximum transfers per fire box |
| `EMBEDDED_MASKING_SHUFFLING_RETURN_HOME` | `shuffling_return_home` | `true` | `0` | `preprocessing.shuffling_return_home` (custom only; presets = `false`) | `false` selects persistent migration |
| `EMBEDDED_MASKING_JIT` | `shuffling_jit_gap` | `0` (`0`/`off`) | unset | none (refused when inherited, like every other `EMBEDDED_MASKING_*` variable) | just-in-time drains; measured negative and not used |

`gen_sandwich_gadget` parses these knobs only in `parse_shuffling_controls` (gen:45-96 at HEAD; `:42-93` at `ea2e516f`). It calls the parser before writing any artifact (gen:279-287 at HEAD). The parser applies these rules (HEAD lines):

- Any of the three is rejected unless the mode is embedded masking (`:51-55`).
- `SHUFFLING` accepts any `usize` (`:56-58`).
- `RETURN_HOME` accepts `0/1/false/true` (`:65-73`).
- `JIT` accepts `0`, `off` or a gap (`:78-84`). It requires `SHUFFLING > 0` and `RETURN_HOME = 0` (`:87-95`).

Two other paths read or set the same variables:

- The managed driver sets them from the recipe: preset `L8`/`A` exports exactly `EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0` (plus `PREMASK`, `SEED_AVOID_C1` and, for L8, `CHAIN`/`CHAIN_LEN`; `scripts/tdp_gen.sh:392-403`); under `preset = "custom"` a knob is exported only when it differs from gen's base. The stage binaries start from an empty environment and an inherited `EMBEDDED_MASKING_*` value is fatal (`src/tdp/runner.rs:767-793`, `configure_environment`; `scripts/tdp_env_registry.tsv`). (The pre-port runner exported both from TOML, `src/tdp/runner.rs:368-379` at `ea2e516f`.)
- The standalone security binary has its own parser, which still uses the old rules (`security_tests/preprocessing/embedded_masking.rs:55-76`).

The values are placed over the `managed_tdp(gadget_seed, n)` base (gen:248-253; gen:356 at HEAD). The library defaults are off (`production()`, EM:196-217; EM:249 at HEAD, `managed_tdp` at EM:236), and the parameters are documented at EM:155-176 (EM:174-305 at HEAD).

The reference invocation is `scripts/windowed_delivery/waist/val_chain.sh:15-16` (= `W/val_chain.sh`, now a parity tool). It sets `SHUFFLING=8 RETURN_HOME=0` together with `PREMASK=1 SEED_AVOID_C1=1`. For config L8 it also sets `CHAIN=2 CHAIN_LEN=8` (`:13`). The managed equivalent is `preprocessing.preset = "L8"` (default) or `"A"`.

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

The log line is printed at gen:305-317 (the `[gen] embedded-masking:` print, gen:407 ff. at HEAD), where `drain/other = added − 2·transfers − 3·ports`.

### 2c.10 Why it exists

**Value/wire separation.** Without migration, or with per-box return home, each data role occupies one wire for the whole compute. With return home at doses 8 and 32, the stage-2 wire classes stayed perfectly separable (touch AUC 0.000/1.000). That mode "preserves data-wire identity BY DESIGN" (`RESULTS.md:645-652`). RC's directive was "no top/bottom or data/band distinction in the mixed circuit; data bits keep moving across wires; do it in stage 2" (`RESULTS.md:665-666`).

Persistent migration landed in the same change set as read-pressure partners and the rich band seed (`RESULTS.md:663-680`). The n=64 `W/s12_check/` scans partly separate the three changes.

Stage-2 data-vs-band touch AUC (§7.9; line 14 of each `RD/wire_balance.py` report in `W/s12_check/`; 0.5 = indistinguishable, 0.000 or 1.000 = perfectly separable):

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

**Intra-box partial sums (the dose).** During a fire the target accumulates partial sums of monomials. A higher dose spreads those partial sums over more wires and adds more escorts. Dose sweep at n=64, G=128 pack (`RESULTS.md:896-919`). The exposure column is the packer's RESULT count ("exposed (packer)", §6a.1), given as a percentage of the vocabulary, i.e. of all distinct stage-2 values:

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
- **Residual anchors.** Values seen 100+ times: ~200 (n=64) and ~700 (n=128) with L8, against ~320 / ~990 under config A (`RESULTS.md:1743-1748`; replicate values in §7.6). `STAGE2_LINKABILITY.md:214-216` attributes this residual class to "migration hops, which no bracket design reaches".
- **Unimplemented fixes.** Two fixes are proposed but not built: value-aware partner filtering / rotate-before-swap (`RESULTS.md:888`), and shuffler-side band rotation at mask-support-free cuts (`RESULTS.md:1279-1286`). The latter understates its own safety condition: `mask_support` holds only the box target's masks, so a value write on a band role is safe only if every open LGI on *any* data wire that references it is closed or re-keyed, as threshold retirement does (EM:1791-1830). Retirement itself (`EMBEDDED_MASKING_RETIRE=K`) was swept and dropped because it raises L2 and creates interior C leaks (`STAGE2_LINKABILITY.md:141`, `:240`).

**JIT drains (present, unused).** `JitState` (`sh.rs:214-436`) keeps escorts alive past box end. It drains them just before a read of the masked role or a write of the holder, and re-masks idle bare roles. At n=64 with gap 24 the result was negative: +31% stage-2 gates, exposed (packer, % of carried) 16.0% → 43.8%, celebrity class unchanged (`RESULTS.md:1109-1149`). The verdict is "drop" (`STAGE2_LINKABILITY.md:242`).

### 2c.12 Inconsistent validators and stale text (the code is authoritative)

- **Managed TDP config.** Persistent migration is the managed default: `preprocessing.preset = "L8"` (and `"A"`) exports `SHUFFLING=8 RETURN_HOME=0` (`src/tdp/config/mod.rs:129-140`, `Stage2Knobs::A`/`L8`; `scripts/tdp_gen.sh:392-403`). Under `preset = "custom"` the dose must be 0 or at least 8 (`src/tdp/config/validate.rs:168-177`; `scripts/tdp_gen.sh:417-418`; `tests/shell/tdp_flags.bash:51-54`, and its `custom-l8` case at `:139` runs segments 8 with return-home 0) and `shuffling_return_home` may take either value; there is no return-home rule anywhere in `src/tdp`. The floor of 8 is the only remaining disagreement with gen, which accepts any dose (gen:56-58). (The pre-port config rejected `shuffling_return_home = false` whenever `segments > 0`, `validate.rs:110-125` at `ea2e516f`, which made the recommended configuration unreachable; §8.10.) Persistent migration is also reachable in three other ways:
  - direct `gen_sandwich_gadget`;
  - the `preprocess_sandwich` API, which has a unit test for it (`construct_test.rs:209-238`);
  - the Gauntlet `--shuffling-carry-layout` arm, which enforces dose ≥8 (`security_tests/gauntlet/gauntlet_gen.rs:272-275`) and sets `shuffling_return_home: !args.shuffling_carry_layout` (`:280`).
- **Stale docs and comments:**
  - **[stale]** `docs/gadgetization.md:227` ("at least eight"), `:232` ("least-written eligible band destination"), `:244-258` (TDP requires return home; "the full sandwich constructor rejects this mode");
  - **[stale]** gen:20-22 (module doc: return home "is required when enabled here");
  - **[stale]** EM:1154-1157 (EM:1233-1236 at HEAD; "Carried-layout shuffling instead returns their physical locations in `final_layout`", contradicted by `sh.rs:749-755` and `construct.rs:42-48`);
  - **[stale]** `sh.rs:17-18` (FireBox: band writes restored "before the next unit boundary"; false under drift and chains);
  - **[stale]** `security_tests/gauntlet/gauntlet_gen.rs:24-29` (module doc: "carried output roles are decoded through final_layout").
- **Stale tests:** `security_tests/preprocessing/embedded_masking.rs:55-76` is a separate parser that still requires `≥8` (`:62-65`) and return home (`:73-76`). `security_tests/gauntlet/gauntlet_gen.rs:273-276` still requires 8.
- **Done since `ea2e516f`:** the vacuous `assert!(true, …)` (old EM:1173-1176), the `validate.rs` error text "carried layouts are supported only by the embedded-masking API and Gauntlet", and the `docs/tdp_pipeline.md` passages that made return home mandatory are gone; `docs/gadgetization.md:227/232/244-258`, gen `mod.rs:20-22`, EM:1233, `sh.rs:17-18`, `gauntlet_gen.rs:24-29` and the two security-test parsers are the live inconsistencies.

### 2c.13 Open questions

1. Resolved 2026-10-09: the persistent configuration is the managed default (`preprocessing.preset = "L8"`, alternative `"A"`). Still open: under `preset = "custom"` the managed validators keep a floor of 8 (`validate.rs:168-177`, `tdp_gen.sh:417-418`, the security tests and Gauntlet) while gen accepts any dose; which is canonical.
2. RC to confirm the wording for the `mask_support` and `last_read_end` exclusions. Their security rationale above is inferred.
3. The 3-CNOT port-restoration block (up to `3·np` CNOTs on bare outputs) is a recognisable stage-2 structure. No measurement shows whether stage-3 mixing, the split and crossing stages (4, 5) or stage-6 delivery (packing and resynthesis) hides it.
4. Zero skipped cuts and 8 transfers per box were observed in all eight replicates at n=64 and n=128 with `r = np`. Neither is guaranteed. Small `n`, a smaller band, or the reserved ancilla pool could produce skips, and this has not been characterised.
5. The Gauntlet `--shuffling-carry-layout` arm (`security_tests/gauntlet/gauntlet_gen.rs:24-29`; the manual no longer mentions it) no longer carries a layout. It sets `return_home = false` (`gauntlet_gen.rs:280`), which is now persistent migration with an identity data layout, because `apply_shuffling` always restores data ports. It also runs with `encoded_io: true` (`gauntlet_gen.rs:281`), which skips `PREMASK` (EM:1927 at `ea2e516f`). Its meaning needs to be re-stated, or the arm retired.
6. No unit test covers the `written_in_box` exclusion, or a ledger in which a cut separates a chained bracket's body from its deferred restore. Chains, PREMASK and seed avoidance have only end-to-end verification and the typed-knob byte-identity test (`STAGE2_LINKABILITY.md:221-226`).


## Stage 3: database mixing

Stage 3 rewrites the stage-2 circuit G2 into an equivalent circuit G3 about twice as large, almost all of it spelled in g57 gates. It does this by repeating one move:

1. Take a few gates that are adjacent once commuting gates are moved aside.
2. Compute a canonical key for the function they compute together.
3. Look that key up in the regular frozen store of g57 circuits.
4. Splice in a randomly chosen stored circuit that computes the same function.

A size controller decides, round by round, whether a splice may grow the circuit (**MIX**) or may only shrink it or keep its length (**COMP**). A rare second move, the **g57 twist**, swaps the labels of two wires across a long window.

The design documents give stage 3 the role of "un-invertible respelling, dilution, DB-form digestion" (`docs/ANF_WINDOW_DELIVERY.md:374`, `docs/windowed_delivery_design.md:171-172`). It is not a semantic-hiding stage. Stage 2 fixes which values ever appear in the clear, stages 4 and 5 (split, crossing walk) respell and spread the material further, and stage 6 (delivery) fixes where anything can be observed. The curated store and the former stage 6 (`fcompress`) are historic and play no part here; the slot-3 thermostat walk is not used by stage 3 (`--p-db 1.0`) but is the engine of stage 5 (§5). In the six-stage chain G3 feeds stage 4 (`split.mpmct1`), then stage 5 (`crossing.mpmct1`), then stage 6; each stage reads the output of the last stage that ran, and the managed artifact names are `db_mixing.mpmct1`, `split.mpmct1` and `crossing.mpmct1`.

The bare names `scheduling.rs`, `sampling.rs`, `transport.rs`, `transformations.rs`, `state.rs`, `params.rs`, `reporting.rs` and `provenance.rs` are in `src/engine/mixer/`, and in this section `mod.rs` and `cli.rs` are in `src/programs/circuit_mixer/`. Two files share the name `replacement.rs`: **`db/replacement.rs`** is `src/stages/db_mixing/replacement.rs` (lookup and choice), and **`mixer/replacement.rs`** is `src/engine/mixer/replacement.rs` (attempt, descent and splice). Line numbers in this section are as of commit `ea2e516f`; the curated-store removal and the port moved most of them (the anchor table under "Conventions": at HEAD the `--tdp` block is `mod.rs:90-100`, the COMP-override guard `:159-200`, the curated warning `:203-221`, `canon caps` `:808`, the stop text `:812-820`, the final float `:830-835`; `DbLayer::tdp()` `cli.rs:735`, `db_mixing()` `:752`; `db/replacement.rs` shrank from 1,685 to 1,315 lines (503 deleted, 133 added) between `ea2e516f` and HEAD; `scheduling.rs`, `sampling.rs`, `transport.rs`, `provenance.rs`, `canonicalization/xgate.rs`, `keys.rs` and `rules.rs` are unchanged).

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

**Command.** The recommended invocation is the managed driver's (`scripts/tdp_gen.sh:899-904`): `val_chain.sh`'s argv in its order, with the output named `db_mixing.mpmct1` and the seed off argv:

```
circuit_mixer --input tdp.mpmct1 --output db_mixing.mpmct1 --tdp --db-mixing \
    --profile 2,2,2,2,2 --moves $(python3 -c "print(int($NG2*2*2*1.3))") --p-db 1.0 \
    --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
    --no-local-verify --seed-file /dev/fd/3
```

under an empty environment plus `FROZEN_DB_DIR`, `FROZEN_FILTER` (`database.lookup_miss_filter = "auto"` resolves to 1 on a store without `filters.bin`), `FROZEN_REGULAR_VALUE_CONVENTION=native`, the output-neutral caches `LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256 XPOLY_CANON_CACHE_MB=1024`, `MIXER_STOP_FLAG=$RUN/stage3.stop` and `MIXER_DUMP_FLAG=$RUN/stage3.dump` (`scripts/tdp_gen.sh:807, 866-904`). The reference driver (`W/val_chain.sh:17-23`, the parity tool) runs:

```
NG2=$(head -1 tdp.mpmct1 | awk '{print $3}')                 # |G2|
MOVES=$(python3 -c "print(int($NG2*2*2*1.3))")               # ⌊N2·R1·1.3·|G2|⌋ = ⌊5.2·|G2|⌋
FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  circuit_mixer --input tdp.mpmct1 --output s3.mpmct1 --tdp --db-mixing \
    --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
    --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
    --no-local-verify --seed $MSEED
```

- **Store.** This build mixes with the regular frozen store only (`circuit_mixer/mod.rs:203-221` at HEAD warns on and ignores `FROZEN_CURATED_*`; the managed driver refuses them when inherited). Earlier stores: `docs/local_mixing_history.md`.
- **Seed.** `MSEED` is a fresh 40-bit draw from Python's `secrets` module (`val_chain.sh:12`). The managed driver draws `mix_seed` (63-bit, OS CSPRNG) into `$RUN/SEEDS` (`seeds_format=3`) and passes it as `--seed-file /dev/fd/3`; the banner prints `(file)`.
- **Driver extras.** The validation runs' `val_chain.sh` passed `--p-comp 1.0 --p-any 0.1`, the CLI defaults (`cli.rs:164-170`); the tracked copy dropped them on 2026-10-08 and the managed driver never passes them to stage 3 (stages 4 and 5 pass `--p-comp 0 --p-any 0`, §4-§5; see 3.2).
- **Move budget.** The general rule is `moves = ⌊N2·R1·1.3·|G2|⌋` for a profile `N0,N1,N2,R1,R2`. For `2,2,2,2,2` this is ⌊5.2·|G2|⌋ (`W/val_chain.sh:17-18`). The pre-port managed driver used `round(N2·R1·|G2|·1.3)` (`tdp_gen.sh:456-462` at `ea2e516f`), a difference of at most one move; the managed driver now computes `int(G*2*2*1.3)` with Python, exactly as `val_chain.sh` does (`stage3_moves_rule=int(G*2*2*1.3)` in the manifest). Since eff (§3.4) counts passes over a circuit that grows to about R1·|G2|, the budget buys about N2·1.3 passes over the target-size circuit: in practice eff ≈ 3.03-3.05. That is past the ramp (eff 2) but before the eff-3.125 controller update that would enter phase 4. So the run ends on the budget in phase 3, at 2.04-2.14× (§3.4). The pre-port managed driver's comment called the formula a "move ceiling: effs x peak size x margin" (`tdp_gen.sh:461` at `ea2e516f`), i.e. a safety cap rather than the intended stop. With N0 = N1 = N2 = 2, however, the one-transition-per-update rule pushes phase 4 to eff 3.125, beyond the cap. Whether stopping in phase 3 is intended, or the budget should let `ProfileDone` fire, is open (§3.10). Anyone who changes `--profile` should recompute the budget with the general rule.

**Output.** G3 is `db_mixing.mpmct1` in a managed run (`s3.mpmct1` under val_chain.sh), with the same wire count (`mod.rs:955-962`); the mixer writes `db_mixing.mpmct1.tmp`, fsyncs it and renames it. The log goes to stdout/stderr: `stage3.log` in a managed run, where `stage3.settings` records `frozen_filter` and `stop_reason` and `stage3.recipe` is the keyed completion marker a resume checks.

- **No sidecars.** The reference driver leaves `--state-out`, `--origins-out`, `--gens-out` and `--anc-out` unset.
- **Consumer.** Stage 4 (splitting) reads G3; with stages 4 and 5 skipped, the packer (stage 6) reads G3 as `--stage3` and G2 as `--stage2`, its value reference (`scripts/tdp_gen.sh:1117`; `val_chain.sh:29`).
- **Composition.** The vL64a output has 1,593,054 gates: 1,409,244 g57 (88.5%), 159,971 width-2 conjunctions, 23,428 width-1 gates and 411 NOTs. Every gate still has at most two controls.

**Environment.**

| variable | reference value | effect |
|---|---|---|
| `FROZEN_DB_DIR` | `frozen_m1_m11` | Required; the mixer panics if it is absent (`src/database/environment.rs:25-30`). |
| `FROZEN_FILTER` | `1` | Loads `<dir>/filters.bin` if present. `frozen_m1_m11` on this Mac has no such file, so the log prints "running unfiltered" and the flag does nothing (`src/database/frozen.rs:661-696`). Managed: `database.lookup_miss_filter` auto\|on\|off (`scripts/tdp_gen.sh:870-892`); `auto` resolves to 1 without `filters.bin`. |
| `FROZEN_CURATED_*` | unset | Ignored with a WARNING at HEAD (`mod.rs:203-221`); the managed driver refuses an inherited value (registry class forbidden). |
| `FROZEN_REGULAR_VALUE_CONVENTION` | `native` (default) | `src/database/environment.rs:11-19` |
| `MIN_DIR_LOOKUP` | `Min` (default) | See "Lookup" in 3.5; `src/database/environment.rs:42-49`. Refused by the managed driver when inherited (forbidden). |
| `LOOKUP_CACHE_MB`, `CANON_CACHE_MB`, `XPOLY_CANON_CACHE_MB` | 512, 256, 1024 (defaults); managed run: 2048, 256, 1024 (`scripts/tdp_gen.sh:807`) | Exact caches, so their size affects only speed (output-neutral, `docs/STAGE4_RUST_DESIGN.md` D10b; `src/database/environment.rs:31-40`; `src/canonicalization/environment.rs:52-61`); the managed driver ignores inherited values with a note (registry class neutral). |
| `CANON_RULE_L_BRANCH_CAP` | unset (no cap); the managed driver refuses it (registry class forbidden, "unbounded") | `src/canonicalization/environment.rs:6-14` (`:9` at HEAD); applied in `canonicalize.rs:1225`. |
| `MIXER_DB_LEN_BAND` | unset (no band) | `lo,hi` drops candidates outside that length range before any mode rule (`src/stages/db_mixing/environment.rs:4-11`; `db/replacement.rs:1419-1456`). Refused by the managed driver when inherited (forbidden). |
| `TWIST_G57_NO_SLIDE`, `TWIST_G57_NO_RETRY` | unset (slides and redraws on) | Turn off twist slides and twist redraws (`src/engine/mixer/environment.rs:4-12`). Refused when inherited (forbidden). |
| `MIXER_STOP_AT_PHASE`, `MIXER_STOP_AT_SPLICES`, `MIXER_STOP_FLAG` | unset | Extra stop rules: a profile phase, a splice count, or a stop-file path (`src/engine/mixer/environment.rs:28-47`; `mod.rs:826-831`; at HEAD `MIXER_STOP_FLAG` is no longer read in `environment.rs`). Managed run: `MIXER_STOP_AT_*` are refused when inherited (forbidden); the driver sets `MIXER_STOP_FLAG=$RUN/stage3.stop` (`touch` it to stop; the run then has no deliverable) and `MIXER_DUMP_FLAG=$RUN/stage3.dump` (`scripts/tdp_gen.sh:866-867`). |
| `CANON_MONOMIAL_CAP`, `CANON_CACHE_MB` | unset; managed run: `CANON_CACHE_MB=256` (neutral), `CANON_MONOMIAL_CAP` refused | Read in `src/canonicalization/environment.rs:19,29` at HEAD (at `ea2e516f` on the g57 window path, `src/canonicalization/window.rs:202-268,353-423`); the monomial cap acts only on the g57 window path, not on the XGate path that stage 3 uses. |

The pre-port TDP driver pinned different values: a Rule-L cap of 512 and `LOOKUP_CACHE_MB=2048` (`src/tdp/runner.rs:392-396` at `ea2e516f`), and it set or removed `FROZEN_CURATED_DIR` explicitly (`runner.rs:381-386` at `ea2e516f`). The managed driver today leaves the canonicalization caps unbounded, as the reference runs did, and pins only the exact caches `LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256 XPOLY_CANON_CACHE_MB=1024` (`scripts/tdp_gen.sh:807`; output-neutral, `docs/STAGE4_RUST_DESIGN.md` D10b); every stage binary starts from an empty environment, so the calling shell cannot change these (`src/tdp/runner.rs:767-793`, registry). The reference driver inherits these variables, so "the measured runs used the defaults" holds only if the calling shell was clean. The logs cannot confirm this for the Rule-L cap: a cap failure on this path is not counted (see "Canonical key" in 3.5), so `rule_l_skips=0` in all eight logs proves nothing about it. The cache sizes cannot change results. A Rule-L cap can, because a window that hits the cap silently becomes a miss. The setting is settled for managed runs (unbounded, the reference runs' value); the counting gap remains.

### 3.2 Parameter resolution

Options resolve in three layers: an explicitly typed flag, then the preset (`--tdp` above `--db-mixing`), then the shipped defaults (`mod.rs:111-165`, with the DB stack at `mod.rs:140-152`; `cli.rs:713-813`). A flag counts as typed only if it appears on the command line (`mod.rs:19-20`).

- **`--db-mixing`** sets `twist_g57 = true`, `p_twist = 0.0005`, `db_advance = true` and `mix_pay_random = true` (`mod.rs:73-92`).
- **`--tdp`** sets `db_advance = true` and deliberately leaves `p_mix` unset (`mod.rs:90-100` at HEAD). It supplies the DB preset layer `DbLayer::tdp()` (`cli.rs:735` at HEAD; `:789-802` at `ea2e516f`), which is the source of every value in the per-mode table below. That layer outranks the `p_mingen` 0.6 of `DbLayer::db_mixing()` (`cli.rs:752` at HEAD; `:806-812` at `ea2e516f`) and the shipped defaults `s_db` 9, `p_convex` 0.4, `p_mingen` 0.8, `s_db_comp` 12 and `p_convex_comp` 0.9 (`cli.rs:775-785`).
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

- `--profile` is the only size authority. The mixer refuses `--target-size`, `--size-hi`, `--size-lo`, `--p-mix` and `--split` alongside it (`mod.rs:251-254,300-307`; `mod.rs:226-227, 275-280` at HEAD); these are exactly the flags of stages 4 (`--split`) and 5 (`--target-size`), which run as separate invocations.
- `--db-max-degree` must be at most 11 (`mod.rs:518-523`).
- With `p_db > 0` and `p_mix` unset, any typed COMP-mode override is a hard error (exit 2). The affected flags are `--s-db-comp`, `--s-db-comp-ctg`, `--p-convex-comp`, `--p-mingen-comp` and `--db-prefixes-comp` (`mod.rs:171-207`; `mod.rs:159-200` at HEAD; `db_mode` defaults to `mix`, `cli.rs:174-175`). The guard tests only `p_mix < 0`, so it also fires under `--profile`, where COMP rounds do run (`scheduling.rs:181-190`). With `--db-mode comp`, the MIX overrides are refused instead. In stage 3, therefore, the COMP knobs come only from `DbLayer::tdp()` and cannot be changed from the CLI (see 3.10).

**Inert switches.**

- `p_mingen` acts only when `gen_target > 0` (`sampling.rs:242-251`).
- `w_window` and `w_pool` never bind, because no gate has more than two controls.
- `--db-degree-probes` is vestigial, because the degree guard is now exact (`db/replacement.rs:1072-1077`).
- `--p-comp`, `--p-any`, `temp`, `target_size` and `dir_p` are read only by the slot-3 walk (`scheduling.rs:545-578`; `transformations.rs:484-490`), which stage 3 never reaches because `p_db = 1.0`. `p_comp > 0` also gates opening the store (`state.rs:685-694`), and `p_comp`/`p_any` gate the `DB ON` banner (`mod.rs:426`; `:383` at HEAD). Both are already implied by `p_db = 1.0`. They are live again as the engine of stage 5 (the crossing walk, §5) and are passed as 0 by stages 4 and 5; nothing is scheduled for removal.

### 3.3 Round structure

The circuit lives in an index-linked arena. The original circuit is kept for the periodic equivalence check (`state.rs:796,813`).

- **Per-gate metadata** (`state.rs:733-745`). Each input gate gets an origin (its index), generation 0, a litter id, and a direction (L or R) drawn uniformly.
- **RNG.** The main RNG is `StdRng::seed_from_u64(seed)` (`state.rs:731-732`).
- **ANF budget.** At most 2^20 terms per multiply, 1,024 per output wire and 2,048 in total (`state.rs:783-792`).

Each loop iteration is one move (`scheduling.rs:425-651`):

1. **Slot 0: profile tick.** Add 1/size to `eff`, the measure of effective work (`scheduling.rs:162-168`). If a controller update is due, run it. Then draw the round's mode: MIX with probability `pmix`, otherwise COMP (`scheduling.rs:182-190`). If the controller has reached phase 4, report and return `ProfileDone` (`scheduling.rs:474-479`).
2. **Slot 1: twist.** With probability 0.0005, run one g57 twist (`scheduling.rs:351-373,486-492`).
3. **Slot 2: DB move.** In every round in which no twist ran, make one DB attempt in the round's mode (`scheduling.rs:526-544`). Because `p_db = 1.0`, the coin short-circuits and draws no randomness. A failed attempt still uses up the round. In vL64a, the 4,051,814 moves were 4,049,743 DB rounds plus 2,071 twists.

The global shuffle and bridge fusion are off and draw no randomness. Slot 3, the crossing walk (the engine of stage 5, §5), runs only in rounds where no other slot ran. With `p_db = 1.0` it is unreachable, so stage 3's trajectory does not depend on it (`docs/notes/cleanup_inventory_20261007.md`, key fact 1).

`global_check` runs every 10,000 moves, and a report is printed every 50,000 (`scheduling.rs:612-647`). The run ends in one of two ways:

- **`MovesBudget`**: the move budget is spent, and a final `global_check` runs (`scheduling.rs:649-650`).
- **`ProfileDone`**: the profile completes. This exit returns without a `global_check` and relies on the post-float check (`mod.rs:901-904`).

Two other exits exist: `StopFlag` (a stop file or `MIXER_STOP_AT_*`) and `CircuitEmpty`. Both run a `global_check` first (`scheduling.rs:438-441,595-622`; `state.rs:586-612`). In a managed run the stop file is `$RUN/stage3.stop`; a `StopFlag` ending is refused as under-dosed (the output is renamed `db_mixing.stopped.mpmct1`, exit 2, no deliverable, re-run on resume) and only `moves budget spent` / `profile complete (size schedule finished)` complete the stage, recorded as `stop_reason` in `stage3.settings` (`scripts/tdp_gen.sh:906-921`); `MIXER_STOP_AT_*` are refused when inherited.

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
- It moves material without widening gates (the crossing walk widens them, §5), and width is what makes store lookups miss.

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
- `global_check` runs every 10,000 moves and at the end. "At the end" means the `MovesBudget` exit or the post-float check. A `ProfileDone` exit relies on the post-float check alone, so with `--skip-final-float` it gets no final check. The managed driver never passes `--skip-final-float`, so a managed run always gets the post-float check.

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

**More work buys little.** In the lab notebook's long-hold runs (profile `2,30,30,2,2`, under an earlier store configuration), heatmap ridge depth (§7.9; a null ridge is ≈ 0) fell only from about 0.314 to an asymptote near 0.290 between eff 2 and 30. The expansion factor mattered far more (`RESULTS.md:89-117`).

**Stage-3 artifacts handled downstream.**

- **Duplicate pairs.** About 7% of delivered gates are pairs of identical gates that commute next to each other; the packer's order repair (stage 6) cancels them (measured on the four-stage chain, where the packer follows stage 3 directly) (`STAGE2_LINKABILITY.md:103-106`; `RESULTS.md:1728-1741`). Which mechanism produces them has not been established.
- **C-view leaks.** On one artifact (rs64, K=64 retirement), strided C-view scans placed these leaks after stage 2. No C value was visible at every 16th stage-2 cut, while every 32nd cut of the mixed circuit showed them (`RESULTS.md:1371-1376`; §7.6). This was not repeated on the recommended configuration. Without retirement, which the recommended configuration does not use, the remaining fringe leaks come from raw input copies and band-seed products on C's first-layer control pairs. Mixer or packer reordering of same-target writes then exposes them. The notebook calls this mechanism plausible but not verified gate by gate (`RESULTS.md:1424-1452`). The leaks are mitigated by stage 2's `PREMASK` and `SEED_AVOID_C1` settings (`RESULTS.md:1462-1475`) and by the packer's repair (stage 6). On the stage-5 output the C-view is clean at every cut, but the merged-window resynthesis adversary still recovers 244/936 interior values at W=2000 (`docs/notes/stage45_revival_20261011.md`; §7.6).

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

After it come the `dminh`, `reorder-idsk` (cumulative reorder skips), `geo`, `splice sizes out->in`, `profile:`, per-length and `twist-g57` lines (`reporting.rs:384-474`). At the end of the run come the `canon caps:` line (g57-path counters only, so always zero on this path), `chain done` and `final float` (`mod.rs:872-911`; `:808-835` at HEAD). The managed driver reads the `chain done … : <reason>` line into `stage3.settings` (`stop_reason=`; `scripts/tdp_gen.sh:806`) and accepts the stage only for `moves budget spent` or `profile complete (size schedule finished)`.

Counter caveats:

- **`geo`.** Hit counts include only MIX hits, but they are divided by MIX plus COMP attempts.
- **Per-length table.** `attempts` and `hits` are keyed by the sampled window length, while `removed` and `added` are keyed by the length of the rung that spliced.
- **`dmin` and `choice`.** These can count identity candidates that were never drawn.
- **`G = 0`.** Every run ends with `G = 0`, meaning more than 5% of gates still have generation 0. With `gen_target = 0`, nothing steers seeds toward material that has never been rewritten, so stage 3 guarantees no full coverage. **[stale]** The statement in `docs/local_mixing_documentation.md` that MIX samples from a minimum-generation pool half the time does not hold for this configuration.

### 3.10 Open points

- **Stopping rule.** Runs stop either on the budget at eff ≈ 3 or on `ProfileDone`, and the `ProfileDone` exit skips the in-loop final `global_check`. Whether the ⌊N2·R1·1.3·|G2|⌋ budget is meant to stop the run in phase 3 is undecided (§3.1). The managed driver now accepts only the two normal stop reasons and never skips the final float (so a managed run always ends with a `global_check`); whether the budget should let `ProfileDone` fire, and the in-mixer gap for `ProfileDone` + `--skip-final-float`, remain open.
- **COMP knobs are not settable from the CLI** (defect, `mod.rs:171-207`; `:159-200` at HEAD). The inert-flag guard ignores `--profile`, so under the profile controller every typed COMP override exits with code 2, although COMP rounds run. The port should make the guard aware of `--profile`.
- **Direction pivot.** The inclusive `i ≤ pivot` is intentional (`mixer/replacement.rs:758-762`) and biases new gates toward heading left; a single gate always heads left. The effect on positional drift has not been measured.
- **Rule-L cap failures are not counted on this path.** (The cap itself is settled: unbounded in managed runs, as in the reference runs, since an inherited `CANON_RULE_L_BRANCH_CAP` is fatal; the caches are 2048/256/1024 MB, output-neutral.)
- **Generation coverage.** Either the port enables generation targeting, or the documentation should state that stage 3 has no coverage guarantee.
- **Duplicate pairs.** Their source is unknown, and it is open whether stage 3 should cancel them itself.
- **Store coverage.** The code documents the caps as the store's census maxima from `frozen_degree_scan` (`params.rs:305-318`; `cli.rs:322-327`), but the census itself is not recorded in `docs/`. What `frozen_m1_m11` covers is stated inconsistently: the archived write-up `CURATED_DB_V3_SUMMARY.md:14` (removed from the tree in `8070f5a0`; branch `ran/archive/curated-and-postmix`, see `docs/CURATED_AND_POSTMIX_ARCHIVE.md`) calls it "the complete g57 replacement corpus through 11 gates", while `docs/frozen_database.md:84-86` warns that the directory name does not establish exhaustive coverage (§8.1). The degree and span caps never fired (`dsk = ssk = 0`). Term-budget overflows are not counted, so whether the 1,024/2,048 caps bind is unknown.
- **`FROZEN_FILTER`.** `database.lookup_miss_filter = "auto"` sets 1 when the store has no `filters.bin` (so it does nothing on this machine); `"on"` requires the file, `"off"` never loads it; a filtered run records `pinned_recipe = none`. §8.1 gives the build route.
- **Cleanup status.** Of the removals listed in `docs/notes/cleanup_inventory_20261007.md:1-28,149-153`, slot 3 and `--p-comp`/`--p-any`/`temp`/`target_size`/`dir_p` are kept as the engine of stages 4 and 5 (revived 2026-10-11); `--db-degree-probes` (vestigial, `mod.rs:522` at HEAD) is the remaining candidate.


## Stage 4: splitting

Stage 4 takes the stage-3 circuit, almost all of it spelled in the complemented two-control shape $a \mathrel{\oplus{=}} b \vee \neg c$ (g57: 88.5% of the vL64a stage-3 output), and rewrites every such gate into plain conjunctions, joining a fraction of the splits to distant brackets on the same wire through absorbed NOT twists. It is the first of the two fragmentation stages revived from the archive on 2026-10-11 (commit `9499f42a`, from `ea2e516f`); it runs as a mode of the mixer (`circuit_mixer --split --split-stop`) and hands the crossing walk both its circuit and its live mixer state. The stage is optional (`run.skip_stages`); with it skipped, stage 5 starts from the previous circuit with its own seed, and the validated four-stage chain is `skip_stages = [4, 5]` with `delivery.anf_only = true`. Line numbers in this section refer to the worktree at commit `83e44ec5`.

### 4.1 What the stage does

The exact split. A complemented gate $(t;\,\neg(m_1 \wedge \dots \wedge m_r))$ fires iff some literal fails. We case-split on the *first* failing literal in a random order: piece$_j = (t;\, m_1 \wedge \dots \wedge m_{j-1} \wedge \neg m_j)$. Exactly one piece fires whenever the parent fires and none otherwise, and the pieces share a target they never read, so they commute pairwise (`rules::presplit`). For a g57 this is $\{a \mathrel{\oplus{=}} b,\; a \mathrel{\oplus{=}} \neg b \wedge \neg c\}$: a one-control and a two-control conjunction, nothing wider than r57. The literal shuffle is the move's random bit.

One move (`split_twist_move`), while the stage is live the only move the mixer makes (every other round slot is withheld):

1. Draw a g57 uniformly from the complemented-gate pool `comp_ids`. An empty pool is exit A ("g57 pool exhausted").
2. Draw its twist direction with probability proportional to the circuit length remaining on each side (ranks restamped every 8,192 moves or on 25% growth), so a short side is picked exactly as rarely as it is short.
3. Split it in place: $g_1$ = the one-control piece, $g_2$ = the two-control piece. The pieces stay where the parent was and take opposite travel directions (the sibling convention the crossing walk inherits).
4. The join coin: with probability $1 - p_{\text{join}}$ the move ends after the bare split.
5. The bracket draw on the target wire $w$: candidates are the bracket-eligible gates targeting $w$ (complemented, or a non-complemented gate with exactly one control) on $g$'s own side by rank; among $k$ = `split_reach_k` uniform samples the farthest wins ($k = 1$ uniform, $k = 2$ about $2/3$ of the available run). No candidate is a failure; `split_fail_limit` consecutive failures are exit B ("failure limit"). $k = 0$ keeps the original other-half-first cascade as an A/B arm.
6. If the bracket $h$ is itself a g57 it is split the same way and its one-control piece $h_1$ is used.
7. The absorbed pure-NOT twist. Both brackets flip their control polarity (CNOT $\leftrightarrow$ NCNOT: a gate targeting $w$ composes with $X_w$ into one gate), and the open segment $S$ between them is conjugated by $X_w$: every gate reading $w$ flips its $w$ literal, a g57 reading $w$ is force-split first and its pieces' $w$ pins flipped, gates targeting $w$ are invariant. The identity is $g_1' \cdot S' \cdot h_1' = g_1 \cdot X_w \cdot S' \cdot X_w \cdot h_1 = g_1 \cdot S \cdot h_1$, at zero synthetic gates; with `--local-verify` every sub-rewrite is checked exhaustively.
8. One ordinary crossing shot from $g_2$ (`cross_move_on`, the stage-5 move under `--k-max 12`), whether or not a bracket was found. This is why a split output already carries a few gates wider than two controls.

Instrumentation that does not touch the walk's RNG: 256 wire canaries planted at stage start from the metrics RNG count the twist flips that pass over them, reported by original-position decile at the boundary with the span histogram (5% buckets of the circuit) and the counters `prims/hspl/segs/joins/xmid/fails`. Under `--split-stop` the boundary ends the run (`MixStop::SplitDone`, "split stage complete (stopped at the stage boundary)"), the state is saved and the output written; without it the mixer would continue into a part 2 under the rest of its command line, which the driver never asks for (and the piecewise driver refuses).

### 4.2 Parameters

Stage 4 has no settings in `configs/tdp.toml` or `--run-*` flags: the driver pins the archive-era flags verbatim and the manifest records `stage4_split_args=archive(ea2e516f)`; changing them means editing `SPLIT_ARGS` (`scripts/tdp_gen.sh:959-960`) and that tag. The mixer's own defaults equal the pinned values (`params.rs:685-690`).

| flag (pinned) | value | meaning and range |
| --- | --- | --- |
| `--split` | on | Arms the stage: the split twist is the only move until exit A or B. |
| `--split-stop` | on | End the run at the stage boundary (required: the driver reads the boundary as the stage's end; `MixStop::SplitDone`). |
| `--p-join` | 0.8 | Probability a split carries the twist and the cross shot; clamped to $[0, 1]$ in code. |
| `--split-reach-k` | 2 | Length bias of the bracket draw: farthest of $k$ uniform candidates on the own side; 1 = uniform, 0 = the v1 other-half cascade. |
| `--split-fail-limit` | 100 | Consecutive bracket failures that end the stage (exit B); `u32`. |
| `--split-canaries` | 256 | Wire canaries planted at stage start (0 = off); metrics only, drawn from `metrics_rng`. |
| `--k-max` | 12 | Width cap on the residues of the per-move cross shot (step 8). |
| `--moves` | $G + 10^6$ | Absolute ceiling, $G$ = input gates: every move retires at least one g57, so exhaustion comes within $G$ moves; the margin covers exit B. "moves budget spent" is also accepted and recorded as `stop_reason=`. |
| `--p-db 0 --p-comp 0 --p-any 0 --p-twist 0` | 0 | Inert while the stage is live (the stage owns the round); pinned so the recipe is closed. |
| `--seed-file /dev/fd/3`, `--report-every 1000000`, `--state-out split.state` | | The split seed (SEEDS format 3, never on argv), the log cadence, the stage-5 resume point. |

Driver-level knobs that touch the stage: `run.skip_stages` / `--run-skip-stages` (stages 2..6; `validate.rs:356-374`, `config/mod.rs:479-481`), `--run-stop-after-stage 4`, `--run-rerun-from-stage 4`, and the operator stop flag `$RUN/stage4.stop`.

### 4.3 Outputs, markers and resume

| file | content |
| --- | --- |
| `split.mpmct1` | The split circuit, same wire count, handed to stage 5 (or to stage 6 when stage 5 is skipped). |
| `split.state` | Mixer checkpoint v6 (the pre-cleanup v3 layout; `checkpoint.rs:260-267`): RNG, move counter, canaries, journal. Stage 5 resumes it, continuing the split stage's RNG and move counter, which is why the stage-5 budget is *added* to this counter (`--moves` is absolute on a resume). |
| `stage4.log` | `split mv= size= comp= prims= hspl= segs= joins= xmid= fails= streak= tapf= span=` every $10^6$ moves, the `split stage ENDED at move …` line, the span histogram, the canary deciles, the `chain done` line. |
| `stage4.recipe` | Completion marker: `tdp_stage4_recipe=1`, `circuit_mixer_sha256`, `stage4_args`, keyed tags of the split seed, the input and the output, `stop_reason=`. A resume re-runs the stage when the marker does not match (interrupted output, changed input, flags or binary) and forces stages $\ge 5$. |
| `split.stopped.mpmct1` | A run ended by `stage4.stop`: kept, never handed on, fatal; a resume re-runs stage 4. |

### 4.4 Measured behaviour

| run | input gates | output gates | ratio | wall |
| --- | --- | --- | --- | --- |
| n=16, `port_runs/seven_n16` | 92,624 | 177,082 | 1.91 | 2 s |
| n=64 L8, managed `port_runs/seven_n64` | 1,583,698 | 2,987,961 | 1.89 | 1 min |
| n=64 L8 base arm, revival study | 1,603,492 | 3,026,369 | 1.89 | 70 s |
| n=128 L8 e2e, revival study | 4,593,662 | 8,657,194 | 1.88 | 8.5 min |

The ratio is what the construction predicts (*derived*): the g57 share of the stage-3 output (88.5% in vL64a) becomes two pieces each, plus bracket splits, forced segment splits and the cross shot's residues. Per gate, the split output is the sparsest ANF representation of any stage at every window width: at n=64 the median monomials of a $W = 1{,}000$ window go 714 (stage 3) $\to$ 580 (stage 4) and at $W = 4{,}000$ 2,564 $\to$ 1,742; the stage doubles the gate count without adding monomials (`docs/notes/stage45_revival_20261011.md` §2). In the stage-wise exposure study of 2026-10-03 split was a visibility-neutral $t$-raiser: it moved 4–8 points of $t^* = 1$ mass into small parities, erased nothing, and *added* carriers on the TDP (105.7k $\to$ 107.9k carried values; `docs/ANF_WINDOW_DELIVERY.md` §3.1).

### 4.5 Why

**Why it exists.** Staying in r57 leaves one repeated complemented gate shape, which is what the rainbow tables and the algebraic attacks lean on. The split produces plain conjunctions of both polarities, and the twist is the one move in the pipeline whose explanation is non-local: the local split, the distant bracket and the flipped reads of $w$ compensate for one another across the whole segment, at no synthetic gates (`docs/local_mixing_history.md` §Fragmentation). Left beside one another the two pieces would still spell their parent; the join is what separates the pair's accounting.

**Why before crossing (the archived, validated order; RC 2026-10-11).** The crossing rules shoot conjunctions only: `rules::cross` asserts the shot gate is not complemented, and R2 against a complemented collider returns `PresplitColliding` for an in-place presplit. The walk therefore starts from the fragments and their sibling directions and never splits r57 gates itself; running stage 4 to exhaustion first means the walk spends no shots on presplits, starts from the sparsest per-gate representation, and inherits the long-range twists already landed. Stage 5 resumes `split.state` so the two stages are one RNG trajectory.

**What it does not buy.** Split is not an eraser (above), and the pair 4+5 does not close the resynthesis leak: the crossing output has no $C$ value on any wire at any cut (0 interior, 0 early outputs raw, both sizes), but per-window resynthesis at $W = 2{,}000$ brings back 244 of 936 interior values of $C$ at n=64 and 711 of 2,872 at n=128, with 45/64 and 94/128 outputs appearing early, by the same mask-removal factoring as on the stage-3 deliverable. At equal gate width those counts are about 55% of the same run's stage-3 proxies (244 vs 434; 711 vs 1,270), which is the window-fraction effect of crossing's 3.4$\times$ stretch, and about 80% against the delivered stage-3 lattice (244 vs 297) (`docs/notes/stage45_revival_20261011.md` §3-4). Crossing also welds the dependency structure the packer's reorder relies on (median DAG interval 6 $\to$ 26 gates), the reason both stages were retired on 2026-10-07; their defaults are the archive-era calibration and their promotion status was never resolved (`docs/tdp_pipeline.md`, "Splitting and crossing").

### 4.6 Code pointers

- Move and sub-rewrites: `src/stages/post-processing/splitting.rs:41-146` (`split_twist_move`), `:149-190` (`split_g57`), `:194-245` (`pick_bracket`), `:247-287` (`pick_bracket_cascade`, the $k = 0$ arm), `:289-389` (`apply_not_twist`, 5a force-split inside), `:391-413` (`absorb_flip`), `:415-424` (`restamp_ranks`, `RANK_EVERY` at `:12`), `:433-466` (`plant_taps`), `:525-560` (`end_split_stage`, `announce_split_end`).
- The exact presplit and the crossing entry: `src/engine/moves/rules.rs:57-74` (`presplit`), `:76-99` (`cross`: conjunction assert, `PresplitColliding`).
- Mixer wiring: `src/engine/mixer/scheduling.rs:450-467` (the stage owns the round; `SplitDone` under `split_stop`), `state.rs:507-540` (`split_on`, `split_done`, `comp_ids`, `wt_buckets`, canaries), `state.rs:593`, `indices.rs:57-72` (bracket eligibility at `:63`), `params.rs:489-512` and `:685-690`, `checkpoint.rs:260-267` (state v6).
- CLI and stop reasons: `src/programs/circuit_mixer/cli.rs:188-216`, `mod.rs:820` (the accepted stop line), `mod.rs:880-881` (`--state-out`), `mod.rs:635-636` (piecewise refusal).
- Drivers: `scripts/tdp_gen.sh:954-988` (stage block; `SPLIT_ARGS` `:959-960`, marker `:961-967`, run `:977-980`), `:937-952` (`run_mixer_stage`, accepted reasons, stop-flag handling), `:806` (`stop_reason_of`); `src/tdp/runner.rs:155-159` (the managed driver hands the stage to the bash driver), `:866-868` (banner), `src/tdp/manifest.rs:365` (`stage4_split_args`).
- Tests: `tests/unit/engine/mixer/splitting.rs:9` (exhaustion, function preserved), `:44` (sibling directions), `serial_baselines.rs:155` (`GOLDEN_SPLIT_STOP`), `checkpoints.rs:37` (split state round trip).


## Stage 5: the crossing walk

Stage 5 takes the split circuit G4 (or, when stage 4 is skipped, whatever circuit precedes it) and runs the mixer's directional crossing walk under a size thermostat for a fixed number of moves. The walk shoots gates at their first true collider and rewrites each collision by an exact case split so that the two gates cross; the pieces are plain conjunctions of up to twelve controls, which makes stage 5 the one stage that leaves the two-control vocabulary of stages 2 and 3. A thermostat holds the gate count near $T = \mathrm{round}(f \cdot |G_{\mathrm{in}}|)$ with two exact contraction channels. Stage 5 is optional (`run.skip_stages`); the validated four-stage chain skips it together with stage 4 (`--run-skip-stages 4,5 --delivery-anf-only`). It was retired on 2026-10-07 and revived from the archive on 2026-10-11 (commit `9499f42a`, the reverse of cleanup step 4) at RC's request, to measure what crossing buys against the merged-window resynthesis adversary (§5.6).

In this section `crossing.rs` is `src/stages/post-processing/crossing.rs`, `rules.rs` is `src/engine/moves/rules.rs`, the bare names `scheduling.rs`, `transformations.rs`, `transport.rs` and `provenance.rs` are in `src/engine/mixer/`, `cli.rs` and `mixer mod.rs` are in `src/programs/circuit_mixer/`, and `tdp_gen.sh` is `scripts/tdp_gen.sh`; line numbers refer to the worktree at commit `83e44ec5`. Run statistics come from `port_runs/seven_n64` (n = 64, L8, budget $2T$), `port_runs/six_n16b` (n = 16, $2T$) and the revival study's `s45_runs/base64` and `s45_runs/e2e128` (budget $6T$); seeds are in the runs' private `SEEDS` files.

### 5.1 Inputs, command and outputs

**Input and seed.** When stage 4 ran, stage 5 resumes `split.state` (`--resume`): the walk continues the split stage's RNG stream and move counter, no seed is consumed, and the `cross_seed` of `SEEDS` (format 3: gen, mix, split, cross, unpack) stays unused. When stage 4 is skipped it starts from the previous circuit with `--input <prev> --seed-file /dev/fd/3`, the cross seed arriving on fd 3 (`tdp_gen.sh:1006-1017`). A stage 4 that ran without leaving `split.state` is fatal (`--run-rerun-from-stage 4`).

**Command.** The driver derives three numbers from the input size $G$ (`tdp_gen.sh:994-1002`): the target $T = \mathrm{round}(G \cdot f)$, the temperature $\tau = \max(64, \mathrm{round}(T/D))$ and the budget $M = \mathrm{round}(F \cdot T)$ (or the absolute `move_attempts`), and runs

```
circuit_mixer --resume split.state \                     # or: --input <prev> --seed-file /dev/fd/3
    --target-size T --temp τ --split-base B --split-damp d \
    --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 --report-every 500000 \
    --moves state_moves+M --output crossing.mpmct1        # input mode: --moves M
```

- **`--moves` is absolute on a resume**, so the driver reads the `moves` line of `split.state` and adds the budget (`tdp_gen.sh:1008-1011`; `seven_n64` resumed at move 4,773 and ran to 11,956,617).
- **Every other slot is off.** `--p-db 0 --p-comp 0 --p-any 0 --p-twist 0` leave the thermostat, slot 3 of `Mixer::run`, as the only move in every round (`scheduling.rs:546-577`); the store is never opened. Stage 3 passes `--p-db 1.0`, so the same slot never runs there and stage-3 output is unchanged by its presence.
- **`--k-max 12`** is the hard width cap; `--split-base`/`--split-damp` are the width *bias* (§5.2).

**Outputs.** `crossing.mpmct1` (same wire count; the archive mixer's resumed-output header bug, `mpmct1 1 N`, was fixed in `9499f42a`), `stage5.log` (a counter line every 500k moves, the `chain done … moves budget spent` line, the final float) and the completion marker `stage5.recipe` (`tdp_gen.sh:1020-1027`: the binary's hash, the full argument list including `--moves`, `stage5_source=resume split.state|input <file>`, keyed tags of `split.state` or of the cross seed, of the input and of the output, and `stop_reason`). No state file is written: nothing downstream resumes stage 5 (the revival script passed `--state-out crossing.state`; the driver does not). An operator stop (`touch $RUN/stage5.stop`) finishes cleanly; a mismatched marker reruns the stage. The mixer ends with its ordinary final float, verified against the input (`mixer mod.rs:835`; 4.69M gates moved in 17 s at n = 64).

### 5.2 The walk: shots, collisions and the three rules

Every expansion round is one `cross_move` (`transformations.rs:477` → `crossing.rs:12,68`):

1. **Shot.** A uniformly random live gate $g$ floats in its stored direction (L or R) to its first true collider $h$ (`transport.rs:17`). Reaching the boundary (`bnd`) retreats the shot.
2. **Pre-split.** A complemented shot $a \oplus\mathrel{=} \neg(m_1 \cdots m_r)$ (the g57 form) is first rewritten into the exclusive first-failing-literal ladder $\{a \oplus\mathrel{=} m_1 \cdots m_{j-1}\, \neg m_j\}_j$ (`rules.rs:63`); the pieces share a target they never read, so they commute pairwise, and siblings take alternating directions. The few complemented gates that survive stage 4 are consumed in the first 500k moves (`pre=222` at n = 64, 66 at n = 16, flat afterwards).
3. **Rule.** `rules::cross(g, h, k_max)` (`rules.rs:76`) picks the case split by who writes whose control. **R1** ($h$ writes a control of $g$): $g$ splits into the ladder over $h$'s monomial and crosses, $h$ stays intact (`rules.rs:105`). **R2** ($g$ writes a control of $h$): $g$ passes intact and $h$ splits to the other side, a g57 collider being pre-split in place first (`rules.rs:160`). **R3** (mutual): $h$ stays, $g$ splits into a stay-behind residue on $h$'s sensitivity set (`Role::Core`) and crossed residues on its complement (`rules.rs:216`). Leftward crossings use the involution trick $[h,g] = \mathrm{reverse}(\mathrm{rewrite}[g,h])$. A residue wider than `k_max` (`WidthCap`) or a facing-CNOT core (`Deadlock`) leaves the circuit untouched; every rewrite is checked exhaustively under `--local-verify` (`rules.rs:278`).
4. **Width bias.** Before anything is emitted, the splitting gate's width $c$ passes `split_allowed` (`crossing.rs:764`): accepted with probability $B^{-(c-d)}$ for $c > d$, i.e. $1/3$, $1/9$, $1/27$ for widths 2, 3, 4 at the defaults. This, not `k_max`, shapes the width profile: 78% of all rounds at n = 64 end in a declined split (`declined=9.35M` of 11.95M), and the output has median 3 controls, 1.6% at seven or more, a handful at 12 (the $6T$ run's histogram, `stage45_revival_20261011.md` §1).
5. **Dispersal and journal.** New pieces inherit the shot's direction with probability `dir_p` = 0.75, advance $\lfloor 0.85 \cdot \mathrm{slack} \rfloor$ in it at birth (`dir_q`), and a failed cross retreats the shot $\lfloor 0.15 \cdot \mathrm{way} \rfloor$ (`crossing.rs:736,746`). Each successful cross is journaled for exact reversal: the pre-cross pair, its pivot, the piece ids with their arena stamps (`crossing.rs:292-340`; capacity `journal_len` 262,144, saturated after 1M moves at n = 64).

At n = 64 and budget $2T$ the walk made 2.05M crossings (R1 1.06M, R2 0.92M, R3 66k) against 9.35M declines, 207k deadlocks, 486 width-cap blocks and 531 boundary hits.

### 5.3 The thermostat and its contraction channels

Each round computes the excess $e = |G| - T$ and a contraction probability $p = \mathrm{clamp}(\sigma(e/\tau), 0.02, 0.98)$, $\sigma$ the logistic function (`scheduling.rs:548-554`; the ceiling is `contract_ceiling`, and its 2% expansion floor keeps crossings running at target). With probability $p$ a contraction channel runs, otherwise `expand_move`; with `undo_frac` = 0.5 the two channels are tried in either order, the second as the fallback of the first (`scheduling.rs:558-577`):

- **Undo** (`crossing.rs:349,388`): sample up to eight journal entries; drop dead ones (a piece touched since, `dead`), skip those whose split event is younger than `tabu_moves` = 2,000 moves (`provenance.rs:806`), gather the live pieces around the pivot by floating within `merge_reach` = 4,096 gates, verify the block against the original pair and splice $[g, h]$ back. A gather miss is retried at most three times.
- **Merge** (`crossing.rs:503,532`; catalogue `transformations.rs:223,255`): sample up to eight initiators until one has index partners under the same-target key or a one-wire-reduced key, take the nearest partner not wall-blocked, float the two adjacent and apply the exact pair identity: **Cancel** ($gg = \mathrm{id}$), **DropLit** ($xR \oplus \neg x R = R$), **Subsume** ($R \oplus \ell R = \neg\ell R$), **XFuse** (same controls, opposite complement, a NOT) and **Absorb** (a bare NOT into a complemented gate). Siblings of a tabu event are refused. With no complemented gates left, XFuse and Absorb never fire (`x=0 a=0` in every log); at n = 64 the channel made 15.9k cancels, 30.0k drop-lits and 25.3k subsumes against 250.6k undos.

Both channels are exact inverses of recorded or local rewrites, so correctness never depends on them, only size does. There is no profile in stage 5: `target_size` is the driver's $T$ throughout, and the run ends only on `MovesBudget` (`scheduling.rs:651`; `mixer mod.rs:815` prints `moves budget spent`, the stop reason the driver's `run_mixer_stage` requires).

### 5.4 Parameters

`[crossing]` in `configs/tdp.toml:64-74`; defaults `CrossingConfig::DEFAULTS` (`src/tdp/config/mod.rs:371-378`), ranges `src/tdp/config/validate.rs:397-427`, key aliases `src/tdp/config/parse.rs:288-320`, script flags and short forms `tdp_gen.sh:111-116,187,207-212,448-455`, manifest keys `crossing_<key>` (`src/tdp/manifest.rs:366-368`). The Rust driver (`local_mixing_bin tdp_gen --config`) translates its config into the script's flags and runs `tdp_gen.sh` (`src/tdp/runner.rs:56,619-624,690-696`), printing the resolved set (`runner.rs:869-878`).

| key (alias) | script flag | default | range | effect |
|---|---|---|---|---|
| `target_size_factor` (`target_factor`) | `--crossing-target-size-factor`, `--xr` | 2 | $[1, 16]$ | $T = \mathrm{round}(f \cdot G)$ |
| `width_penalty_base` (`width_base`) | `--crossing-width-penalty-base`, `--xb` → `--split-base` | 3 | $[1, 10^6]$ | $B$; 1 turns the bias off |
| `width_penalty_threshold` (`width_threshold`) | `--crossing-width-penalty-threshold`, `--xc` → `--split-damp` | 1 | $[0, 10^9]$ | $d$, the width below which the bias is off |
| `size_tolerance_divisor` (`temperature_divisor`) | `--crossing-size-tolerance-divisor`, `--xtdiv` | 25 | $[1, 10^9]$ | $\tau = \max(64, \mathrm{round}(T/D))$ |
| `move_budget_factor` (`move_factor`) | `--crossing-move-budget-factor`, `--xmf` | 3 | $(0, 1000]$ | $M = \mathrm{round}(F \cdot T)$; RC 2026-10-12 (`83e44ec5`); 2 on 2026-10-11 (`ff83daf6`); 6 in the archive-era driver |
| `move_attempts` (`moves`) | `--crossing-move-attempts`, `--xmoves` | unset | $[1, 10^{12}]$ | absolute $M$; exclusive with the factor (config error / exit 2) |

Fixed by the driver or left at the mixer's defaults (`cli.rs:22-62,450`): `k_max` 12, `dir_p` 0.75, `dir_q` 0.85, `merge_reach` 4,096, `journal_len` 262,144, `undo_frac` 0.5, `tabu_moves` 2,000, `contract_ceiling` 0.98, `report_every` 500,000. The mixer's own defaults for the three values the driver overrides differ (`--temp` $\max(T/100, 64)$, `--split-damp` 2, `--split-base` 2.0); the driver's values are the archive-era calibration, whose promotion status was never resolved (`docs/tdp_pipeline.md`). Stage 4 has no settings.

### 5.5 Measured behaviour

| run | budget | input → output gates | $|G_5| / T$ | wall |
|---|---|---|---|---|
| n = 16, `port_runs/seven_n16` ($T$ = 354,164, $\tau$ = 14,167) | $6T$ = 2.13M | 177,082 → 322,373 | 0.910 | 18 s |
| n = 16, `port_runs/six_n16b` ($T$ = 384,978, $\tau$ = 15,399) | $2T$ = 769,956 | 192,489 → 347,900 | 0.904 | 3.5 s |
| n = 64 L8, `port_runs/seven_n64` ($T$ = 5,975,922, $\tau$ = 239,037) | $2T$ = 11.95M | 2,987,961 → 5,380,721 | 0.900 | 7 min (407 s + 17 s float) |
| n = 64 L8, `s45_runs/base64` ($T$ = 6,052,738) | $6T$ = 36.3M | 3,026,369 → 5,494,528 | 0.908 | 46 min |
| n = 128 L8, `s45_runs/e2e128` ($T$ = 17.31M) | $6T$ | 8,657,194 → 15,712,838 | 0.907 | 6.3 h (shared CPU; several-fold slower under contention) |

**The walk settles at $0.90$–$0.91\,T$, not at $T$.** The $6T$ trajectory at n = 64 reads 4.81M gates at $1T$ moves, 5.45M ($0.900\,T$) at $2T$, 5.51M at $2.5T$, 5.52M ($0.913\,T$) at $3T$, flat to $4T$, then a slow erosion to 5.49M at $6T$. *Derived:* at $0.9\,T$ the excess is $-2.5\tau$, so $p \approx 0.08$–$0.10$ and the thermostat is almost pure expansion; the plateau is where the gross growth of the accepted crossings (about one expansion attempt in five) balances the undo and merge channels, well below the regime where $\sigma$ bites. At $f = 2$ the stage therefore delivers about $1.8\times$ its input, at every n we ran.

**ANF density and resynthesis** (`docs/notes/stage45_revival_20261011.md` §2–3, `seven_stage_20261011.md` §3–4). Uniform windows of the crossing output hold 1.4–1.9× the monomials of stage-3 windows at the median and 3–7× at p90 (degree median 7–11 against 4–8), and composition under $2^{22}$ monomials / 120 s fails for more than 20% of windows from $W \approx 4{,}000$, where stage 3 still composes 90–99% at $W = 8{,}000$; per function covered the picture is milder, since the 3.4× stretch makes a 2,000-gate stage-5 window cover what a ~590-gate stage-3 window covers. Resynthesizing the delivered windows of the crossing output (the stage-6 unpacking sub-step) gives a gate ratio of 0.925 (5,153,273 → 4,766,644 at n = 64, 157 of 10,366 windows not converging in 30 s, which is what the lattice refinement of stage 6 exists for) against 0.354 on the four-stage deliverable. The packer takes 2.2 h instead of 23 s on it, exposes 10.8% of carried values at window boundaries instead of 6.9%, and still reaches the C-view floor (64 visible / 0 interior; 128 / 0 at n = 128).

### 5.6 Why the stage is there, what the thermostat protects, and what it does not buy

- **Why a crossing walk.** It is the only transformation in the chain that leaves the two-control vocabulary: every piece is an exact case split of a pair, re-encoded jointly (ancestry treats a cross as a splice over $\{g, h\}$), with the width profile of §5.2. In the stage-wise exposure study of 2026-10-03 crossing was the only post-stage-3 eraser, $-3$ to $-5$ points of visibility in every arm and 105.7k → 89.8k carried values (`docs/ANF_WINDOW_DELIVERY.md` §3.1), and its raw output carries no C value at any cut (0 interior, 0 early outputs at n = 64 and 128). RC brought stages 4 and 5 back on 2026-10-10 to measure exactly this against the resynthesis adversary.
- **What the thermostat protects.** Size. R1 and R3 emit a ladder of up to $|m|$ pieces per crossing, so an unbraked walk grows without bound. The thermostat's channels are exact, so they cost nothing in correctness; the 2% expansion floor keeps crossings going at target, and the 2,000-move tabu gives a fresh split time to disperse before it may be undone or sibling-merged.
- **Why the budget is a factor of the target, and why 3.** The walk's clock is moves and its state is a function of moves$/T$ (§5.5): size is on the rising flank at $2T$ and on the plateau from about $3T$. RC made the budget a parameter on 2026-10-11 with default 2 (`ff83daf6`), noting that the archive's $6T$ "also held the fragments under the thermostat for a while, eroding them" at seven times the wall-clock (46 min against 7), and raised it to 3 on 2026-10-12 (`83e44ec5`) so the stage ends on the plateau rather than at $0.90\,T$. No delivered run at $3T$ exists yet; the expectation ($\approx 0.91\,T$, about 10 min at n = 64) is read off the $6T$ trajectory. `move_attempts` is the calibration override.
- **What it does not buy: the merged-window adversary.** Resynthesizing uniform $W = 2{,}000$ windows of the crossing output recovers 244 of 936 interior C values at n = 64 and 711 of 2,872 at n = 128 (repeated 88 / 56 and 255 / 163; outputs early 45 / 64 and 94 / 128), by the same mask-removal factoring as on stage 3 (CNOT and 2-control re-creations, none with C's operands as controls). That is ~55% of the same run's stage-3 output at equal gate width (434; 1,270) and ~80% of its delivered lattice (297; 417), the window-fraction effect of the 3.4× stretch. Ten times the time budget on the deadline windows changes 244 to 241, canonical order by at most 1.7% (`docs/notes/stubborn_windows_20261012.md` §2). Crossing halves the adversary's composable width in gates, but the leak needs only the width of one fire. On the same ground the windowing program had already chosen the stage-3 output as its input: crossing welds the dependency structure (median DAG interval 6 → 26 gates; hideable ceiling at $W = 20$ 69.6% against 46.9%; `ANF_WINDOW_DELIVERY.md` §3.2). Stages 4 and 5 therefore do not close the leak, and the validated chain skips them.

### 5.7 Code pointers and stale text

| what | where |
|---|---|
| walk: shot, pre-split, rules, journal | `crossing.rs:12` (`cross_move`), `:24` (`pick_cross_shot`; `p_mincross` off), `:68` (`cross_move_on`), `:292-340` (journal) |
| contraction: undo, merge | `crossing.rs:349` (`undo_move`), `:388` (`try_undo`), `:503` (`merge_candidates`), `:532` (`merge_move`), `:726` (`bring_adjacent`) |
| direction law, retreat, width bias, splice | `crossing.rs:736,746,764,776` |
| rules R1–R3, pre-split, dispatch, verifier | `rules.rs:105,160,216,63,76,278` |
| slot 3, $p$, stop; `expand_move` | `scheduling.rs:426` (`run`), `:546-577`, `:651`; `transformations.rs:477` |
| merge catalogue and keys | `transformations.rs:223,255,348,359` |
| float to collision; tabu | `transport.rs:17`; `provenance.rs:806` |
| mixer defaults, banner, stop text, final float | `cli.rs:22-62,450`; `mixer mod.rs:302-315,815,835` |
| driver stage block, usage, defaults, aliases, validation | `tdp_gen.sh:990-1047`, `:14-17,111-116`, `:187`, `:207-212`, `:448-455` |
| config, validation, manifest, Rust driver, config file | `src/tdp/config/mod.rs:355-404`, `parse.rs:288-320`, `validate.rs:397-427`, `manifest.rs:366-368`, `runner.rs:56,619-624,690-696,869-878`; `configs/tdp.toml:64-74` |
| manual and notes | `docs/tdp_pipeline.md` "Stage 5" and its `[crossing]` table; `docs/notes/stage45_revival_20261011.md`, `seven_stage_20261011.md`, `stubborn_windows_20261012.md`; `docs/CURATED_AND_POSTMIX_ARCHIVE.md` ("Old stage 5") |

**[fixed 2026-10-12]** The `CrossingConfig` doc comment (`src/tdp/config/mod.rs:357-358` in the working tree; `:355-358` at `83e44ec5`, where it still said "move budget 2 * target") and the budget comment in the driver's Python snippet (`tdp_gen.sh:1002-1005` in the working tree; `:999-1001` at `83e44ec5`, "the default budget is 2*target") now state the 3×target default (RC 2026-10-12; 2×target on 2026-10-11; 6×target in the archive-era driver), as do `DEFAULTS` (`src/tdp/config/mod.rs:376`), `XMF=3` (`tdp_gen.sh:187`; `:190` in the working tree), the usage text (`tdp_gen.sh:17`) and `configs/tdp.toml`. The n = 64 timing quoted in the revival note ("3.03M gates to 5.49M in 46 min") is the $6T$ revival run, not the default budget; the `seven_n16` row of `seven_stage_20261011.md` §3 likewise ran at $6T$ (2.13M moves).


## Stage 6: delivery

Stage 6 turns the circuit of the last stage that ran (`crossing.mpmct1` in the six-stage default; `split.mpmct1` with stage 5 skipped; `db_mixing.mpmct1` with stages 4 and 5 skipped) into the published circuit, in two sub-steps. **Packing** (`tdp_packing run`, §6a) emits a sequence of windows, each represented as the canonical vector ANF of its permutation and nothing else (`packing/deliverable.tdp4`, container `LMTDPANF` v1; `docs/ANF_WINDOW_DELIVERY.md:147-152`): the internal windowed form of the circuit. **Unpacking with lattice refinement** (`tdp_unpacking run --refine`, §6b) resynthesizes every window from its ANF alone; a window that does not resynthesize is a packing defect: it is cut in two at an exposure-aware cut, both halves are re-emitted into the deliverable and unpacked separately, recursively. The outputs are the published circuit `unpacking/unpacked.mpmct1`, the canonical-order concatenation of the resynthesized windows, and the internal refined deliverable `unpacking/deliverable.tdp4`, kept for auditing and parity (RC, 2026-10-12 and 2026-10-13; `docs/DELIVERY_STAGE.md`). `delivery.anf_only = true` ends the stage after the packer's deliverable, which is then the published object and is unpacked by the recipient (with `run.skip_stages = [4, 5]` the validated four-stage chain). Both sub-steps read C (§6a.8). On the stage-5 output the packer needs about 4× the windows (10,464 at n=64, 29,969 at n=128) and hours (2.2 h / 7.7 h instead of 23 s / 2 min), with boundary exposure 10.8% of carried values against 6.9% on the stage-3 input, and still reaches the C-view floor (`docs/notes/stage45_revival_20261011.md` §1).

### 6a. Stage 6, first sub-step: packing

Packing has three steps, and the stage continues with a fourth:

1. A joint reorder-and-cut pass, `window_pack3.py` (§6a.4).
2. A function-preserving repair pass that brings the C-view down to its floor, `repair.sh` (§6a.5).
3. Serialization, `anf_dump_eval.py` (§6a.6).
4. Unpacking with lattice refinement, `tdp_unpacking run --refine` (§6b; Python reference `scripts/windowed_delivery/waist/deliver.py`).

*Superseded (2026-10-09 / 2026-10-12): the packing sub-step of stage 6 runs in Rust as `tdp_packing run` (`src/stages/packing/`, `docs/STAGE4_RUST_DESIGN.md`, written in the four-stage numbering), bit-exact against the Python prototypes, which remain its canonical oracle and are never edited (`scripts/windowed_delivery/README.md`). The snapshot was refreshed from the run tree on 2026-10-08 (`waist/order_repair.py`, `waist/val_chain.sh`, the latter without `--p-comp/--p-any`), so every file is current; `waist/deliver.py` (2026-10-12) is the Python reference of the delivery loop (`docs/DELIVERY_STAGE.md` §3). The rest of this paragraph describes the prototypes as they stood before the port.* Before the port, packing existed only as Python/bash prototypes. The committed snapshot `scripts/windowed_delivery/` (commit `7163e457`) is described as "the reference oracle for the Rust port" (`scripts/windowed_delivery/README.md:1-9`). Script paths follow the document-wide convention (Overview, "Conventions"): they mean the tracked snapshot `SW/`, which is authoritative.

#### 6a.1 Threat model and what windows buy

In the default mode the adversary holds the published circuit `unpacking/unpacked.mpmct1` and nothing else: it can cut it anywhere, compute the function of any segment, compose segments and resynthesize them into circuits of its own, but it is never handed the window lattice or the ANFs and does not see how a window was spelled before resynthesis (the packer's spelling is gone; the synthesizer's spelling is what it holds). Under `delivery.anf_only` it holds the packer's `deliverable.tdp4` instead, the lattice of windows and the ANF of each window's permutation: from the ANFs it can derive the wire state at every window boundary on inputs of its choice, compose adjacent windows and resynthesize any window into a circuit of its own, but it cannot see how the function was spelled inside a window. The packing metrics below are evaluated on the windowed representation, at every boundary of its lattice, with no cut sampling (`STAGE2_LINKABILITY.md` §2); for the refined lattice (`unpacking/deliverable.tdp4`) every added cut is scanned at choice time, and since a per-cut C-view result does not depend on the lattice, the packer's final scan of the old cuts plus the choice-time scans equal `tdp_packing check-final` on the refined lattice (`docs/DELIVERY_STAGE.md` §4; §6b). They are the acceptance metrics of the packing sub-step and, under `anf_only`, the metrics of the published object; the published circuit of the default mode is measured by re-cutting it (the all-cuts census and the uniform W=2000 re-cut-and-resynthesize experiment: "Merged-window resynthesis" below, §6b.5, §7.6). Equality and affine membership, however, are tested on random samples, not symbolically:

- the C-view uses 96×64 = 6,144 honest-slice samples with seed 12345 (`HA:112-117`, `HA:367-375`; §7.3);
- the packer, link census and `verify_equiv.py` use 1,024 all-wire samples (`RD/window_pack3.py:22-26,165-166`; `W/link_census.py:25-26`; `W/verify_equiv.py:5-9`).

**What windows buy.** On the windowed representation, and so for the `anf_only` artifact, every exposure metric quantifies over the cuts of the lattice, and a window has no interior cuts: a value is exposed only if it is alive across a boundary (`docs/ANF_WINDOW_DELIVERY.md:225-234`). This has three effects:

- Read values become hideable. The earlier "reader law" assumed a cut between every pair of gates.
- The closure law is sidestepped, because cuts are deleted rather than rewritten.
- Templates, k-grams, gate provenance and the gate count all disappear into a canonical polynomial.

Four things remain observable in that artifact: values alive at boundaries, the equality-linkage graph between boundary values, the outputs, and the window lattice itself, which is new visible structure. In the default mode the resynthesized windows do have interior cuts again, the synthesizer's gates, and the lattice does not ship. What windowing buys there is the separation guarantee (§6b): the packer's and the mixer's spelling never leave the stage, and every window of the published circuit is re-spelled from its function alone by a synthesizer that sees no gate and no window position. The residue of that re-spelling is measured, not argued: the full unpacked n=64 crossing output materialises 5 of 936 interior C values at its own cuts (the raw packed circuit 0), and uniform W=2000 re-cutting and resynthesis of the crossing output recovers 244 of 936 (`docs/notes/stubborn_windows_20261012.md`, `docs/notes/stage45_revival_20261011.md`, `docs/DELIVERY_STAGE.md`; §6b.5, §7.6). The so-called recurrence law is an empirical observation, not a theorem. At migration dose 8, on one n=64 run and one n=128 run, every value with a stage-2 birth multiplicity of 5 or more was boundary-exposed; the largest multiplicity among non-exposed values was 4 (`docs/ANF_WINDOW_DELIVERY.md:290-294`; `RESULTS.md:1064-1066`).

The packing sub-step of stage 6 is judged by three metrics (tooling in §7). The unpacking sub-step adds its own acceptance checks: every converged window exact against its delivered ANF; every cut of the refined lattice clean (`check.new_cuts`) and every window within the 65,536-monomial budget; `unpacked.mpmct1` ≡ deliverable on 4 × 1,024 random states and zero-slice against C (4,096 samples); counters `splits`, `unclean_cuts`, `over_budget`, `verbatim` and `status=ok|verbatim_used` in `unpacking.report` (§6b).

| metric | definition (as coded) | target | tool |
|---|---|---|---|
| C-view | for each value born by a gate of C: t\* = the fewest boundary wires whose XOR (plus a constant) equals it on the samples, minimized over the scanned cuts. t\* is exact when the nullity is ≤ 14 and an upper bound otherwise (`HA:597-616`) | exactly the floor: the n outputs (64 / 128) | `hmap_affine --exposure --g-cuts` (§7.3); Rust: the in-process C-view of `tdp_packing run` and `tdp_packing check-final` |
| stage-2 exposure: **exposed (packer)** | distinct stage-2 values with a span crossing a boundary, or alive at the end, on the **pre-repair** pack | minimize (informational) | `window_pack3.py:488-502` |
| stage-2 exposure: **proxy values (link census)** | distinct stage-2 values seen in at least one run at the input cut or an interior boundary of the **delivered** lattice | minimize (informational) | `W/link_census.py:47-74,86` |
| linkability | per wire, a run is a maximal stretch of consecutive cuts in {input cut} ∪ interior boundaries where the wire holds the same function up to complement. The output cut is excluded: gates after the last boundary are never applied. L2 = proxy values (equal to some stage-2 gate output) with ≥2 runs; M2 = total runs over those values; linkable froth = non-proxy values with ≥2 runs. Froth includes input values that G2 never re-births, so froth is not strictly "mixer-made" | minimize (informational; reference ranges in §8.7) | `W/link_census.py:34-74` (§7.5); Rust: `stages::packing::census`, reproduced as `packing/link_rep512.txt` |

The documented definitions are at `STAGE2_LINKABILITY.md:30-51`. Where they differ, the code is the authority.

**"Exposed" has several senses; this document labels every figure with one of them.**

- **Vocabulary.** The set of distinct nonconstant stage-2 values, with v and ¬v counted once, as fingerprinted on 1,024 all-wire samples. In the packer this is `index`; its RESULT line prints it as "% of all distinct A" (`RD/window_pack3.py:63-71,498-502`). It is about 623k values (config A) or 712k (L8) at n=64, and 1.78-1.79M / 2.04M at n=128 (`W/v*/mult_s2.txt`, `distinct`).
- **Exposed (packer).** The RESULT count: carried stage-2 values with some boundary p, b+1 ≤ p ≤ e, inside one of their spans (birth b, next write e), plus every value forced at the output (`RD/window_pack3.py:488-497`). It is reported as "% of carried" or "% of vocabulary". It describes the pack **before repair**. Repair cancellation crosses boundaries and changes boundary states (§6a.5), so this figure is not the delivered exposure.
- **Proxy values (link census).** The delivered stage-2 exposure: the number of distinct stage-2 values that occupy at least one run on `rep512` (`W/link_census.py:73,86`). It differs from the packer count by definition as well as by repair. It **includes** the input cut, where an input word that some stage-2 gate re-creates counts. It **excludes** the output cut, so values forced at the output count only if they are also seen at an interior boundary. `STAGE2_LINKABILITY.md`'s "exposed" column is this count.

Example, vL64a: 11,539 exposed (packer, pre-repair; `W/vL64a/pack_G512.txt`) against 11,743 proxy values (`W/vL64a/link_rep512.txt`). The two counts have not been reconciled value by value. A re-run of the packer's RESULT computation on `rep512` would separate the effect of repair from the effect of the definitions.

Scope limits:

- Only exact affine predictors and exact equality are measured. Low-degree nonlinear predictors have not been tested on final artifacts (`STAGE2_LINKABILITY.md:178-180`).
- Small windows are canonically re-synthesizable (`docs/windowed_delivery_design.md:208-210`).
- **Open modeling point.** The two kinds of measurement use different evaluation domains. The C-view evaluates the scanned circuit with x on wires 0..n and **zeros elsewhere** (`HA:3-8`, `HA:367-375`). Packer fingerprints, the link census, `verify_equiv.py` and `anf_dump_eval.py` use random values on **all** wires (`RD/window_pack3.py:165-166`; `W/verify_equiv.py:5-9`). An adversary holding the ANF can evaluate any full state. Whether the C-view should also be measured off the honest slice has not been decided (§7.1).
- **Merged-window resynthesis (the default-mode adversary).** Re-cutting the circuit uniformly into windows of width W and resynthesizing them, with no use of the delivered lattice, recovers interior C values that no cut of the packed circuit exposes: 244/936 at W=2000 on the n=64 stage-5 output (241 with 10× time; ≤ 2% change under canonical order), against 0 on the refined bisection assembly at its own lattice and 5 of 936 for the full unpacked output at its own cuts (`docs/notes/stubborn_windows_20261012.md` §2, `docs/notes/stage45_revival_20261011.md`; §6b, §7.6). This experiment and the all-cuts census of the resynthesized circuit are the measurements that model the adversary who holds only the published circuit.

#### 6a.2 Inputs, outputs and formats

Inputs (`tdp_packing run --stage2 … --stage3 … --source-c …`, `scripts/tdp_gen.sh:1117`; `W/val_chain.sh:15-29`). All three are secret (§6a.8).

| file | content | used by |
|---|---|---|
| `tdp.mpmct1` | G2 (unmixed; the sandwich itself when stage 2 is skipped) | packer value reference (`--stage2`; `RD/window_pack3.py:162,168`), link census (`W/link_census.py:20-21`) |
| `crossing.mpmct1` / `split.mpmct1` / `db_mixing.mpmct1` (`s3.mpmct1` under val_chain.sh) | the output of the last stage that ran (G5 / G4 / G3) | packer input (`--stage3`) |
| `tdp.mpmct1.source_c.g57` | source C, written by `gen_sandwich_gadget` | C-view in repair (`W/repair.sh:18-19`; `W/anf_split2.py:26-29`) and the stage-2 span check (`W/val_chain.sh:26-27`) |

**`mpmct1`** is the format of §1.3; its action is `st[t] ^= comp XOR AND_i (p_i ? st[c_i] : NOT st[c_i])` (`src/circuit/xgate.rs:159-166`). The Python loaders do not agree on n:

- `window_pack3.py`, `anf_compose.py` (and so `anf_split2.py` and `anf_dump_eval.py`, which import it) and `boundary_cluster.py` (used by `link_census.py`) take n = max(header n, max wire + 1) (`RD/window_pack3.py:47`; `RD/anf_compose.py:47`).
- `order_repair.py` takes the header n and writes it back unchanged. `verify_equiv.py` imports that loader (`W/order_repair.py:25-44`; `W/verify_equiv.py:4`).

**`.bounds` format.** One integer p per line, with 0 < p < m. A boundary at p sits between emitted gates p-1 and p, and cuts 0 and m are implicit (`RD/window_pack3.py:305-322, 461`). The windows are `[cut_i, cut_{i+1})` over `[0] + bounds + [m]` (`RD/anf_compose.py:129-131`). A value born at gate b and overwritten at gate e is therefore exposed iff some boundary satisfies b+1 ≤ p ≤ e (`RD/window_pack3.py:493-495`). Values alive at the end are added at lines 496-497.

Outputs:

- `packing/packed_G512.mpmct1` and `packing/packed_G512.mpmct1.bounds`, the pack. The packer writes its lattice to `<out>.bounds` (`RD/window_pack3.py:483`).
- `packing/rep512.mpmct1` and `packing/rep512.bounds` (not `rep512.mpmct1.bounds`), the repaired circuit and its lattice (`W/repair.sh:11-12`): the lattice the unpacking sub-step starts from.
- `packing/deliverable.tdp4` (container `LMTDPANF` v1; `deliverable.bin` is its raw body), the packer's windowed representation, and, unless `anf_only`, the unpacking sub-step's `unpacking/unpacked.mpmct1` (resynthesized, canonical order: **the published circuit**) with the internal `unpacking/deliverable.tdp4` (the refined lattice), `bounds.final`, `windows.tsv`, `rounds.tsv`. Only the published circuit leaves the run directory (under `anf_only`, `packing/deliverable.tdp4`); the two ANF deliverables and the reports are internal records. (val_chain.sh: `deliverable.bin` + `.gz`/`.xz`, `W/val_chain.sh:31-33`.)
- Logs and markers: `packing/{pack_G512.txt,repair512.log,anf_rep512.txt,link_rep512.txt,packing.report}`, `unpacking/{unpacking.log,unpacking.report}`, `stage6.log`; `delivery.recipe` (keyed completion marker), `delivery.failed` (`exit=<code>`, `step=packing|unpacking`), `NOT_RECOMMENDED` (status `floor_miss`, `over_budget` or `verbatim_used`). val_chain.sh writes `deliver.txt` instead.

The scratch files `rep512.{wit,allcuts,exposure}.tsv`, `rep512.cview.log` and `split_cands_<pid>.{txt,allcuts.tsv,exposure.tsv}` are derived from C. The full never-ship list is in §6a.8 (it includes the unpacking sub-step's private files).

#### 6a.3 Reference driver and parameters

`scripts/windowed_delivery/waist/val_chain.sh TAG N MC S SLICE CONFIG` (= `W/val_chain.sh`, now a parity tool) runs the four-stage validation chain: stages 1-3 and the packing sub-step. CONFIG is `A` or `L8` (`W/val_chain.sh:2-5`). Seeds are drawn fresh from `secrets` (lines 11-12). The packing part is three lines, run from `RD/` (`W/val_chain.sh:28-31`):

```
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1
bash $W/repair.sh $TAG 512 $N
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG
```

The validation runs' driver passed `--p-comp 1.0 --p-any 0.1` to stage 3 (the defaults); the tracked copy dropped them on 2026-10-08 (§3.1), and the flags stay: stages 4 and 5 pass `--p-comp 0 --p-any 0` (§3.2).

| parameter | CLI pos. | code default | production | managed key (`[packing]`; bash `--packing-*`) | role |
|---|---|---|---|---|---|
| `W_cap` | 3 | 64 | 512 | `interval_cap` (512; ≤ 10^6) | max interval size, in gates (b and e included) |
| `G` | 4 | 64 | 512 | `window_gates` (512; ≤ 10^6) | soft gate cap per window (hard 2G inside a burst) |
| `B_mono` | 5 | 3072 | 24,576 (= 48·G) | `monomial_budget` (24576; ≤ 2^18 = 262,144, a memory bound) | soft monomial budget (hard 2·B_mono inside a burst) |
| `span_cap` | 6 | 3000 | 3000 | `span_cap` (3000; ≤ 10^9) | max spelled span length of a winnable value |
| `label` | 7 | `pack3-G<G>` | `$TAG-G512` | — | log label only: prefixes the first log line (`RD/window_pack3.py:158,171`) |
| `out_path` | 8 | none | `$R/packed_G512.mpmct1` | — (the port writes `packing/packed_G<G>.mpmct1`) | output circuit. The lattice is written to `out_path + ".bounds"`. **Without this argument nothing is written** (`RD/window_pack3.py:159,475-484`) |
| `MUL_CAP` | derived | 8·B_mono | 196,608 | — | product-size abort in `compose` |
| `PAD`, `CMAX` | const | 4, 24 | — | — | pre-emptive-close slack; contagion threshold |
| `LANES` | const | 16 | — | — | 1,024 fingerprint samples |
| `SEED` | const | 20261004 | — | — | fingerprint RNG (fixed, not secret) |
| split/scoring budget | `repair.sh` | — | 65,536 monomials per window | `split_budget` (65536; ≤ 262,144); `fix_iterations`, `fallback_iterations` (4, 4; ≤ 1000); `anf_cap_semantics` (`cpython39`); `allow_floor_miss`, `allow_over_budget` (false) | enforced only by `anf_split2.py` and reported by `anf_compose.py` (`W/repair.sh:31,36`). Not checked by `anf_dump_eval.py`, which serializes with budget `1<<30` (`W/anf_dump_eval.py:20`); the port emits at the 2^18 ceiling and exits 6 above it. |

Code defaults are at `RD/window_pack3.py:22-28, 153-159`. The B_mono = 48·G convention is from `W/score_arm.sh:10`. The production defaults are `PackingConfig::default` in `src/tdp/config/mod.rs:225-240` (window_gates 512, interval_cap 512, monomial_budget 24576, span_cap 3000, split_budget 65536, fix_iterations 4, fallback_iterations 4, anf_cap_semantics cpython39), mirrored by `scripts/tdp_gen.sh` (`PACK_*`); any non-default is recorded as `packing_params=custom`; the validation ranges are in `src/tdp/config/validate.rs` (`[packing]` block).

**File-name contract between packer and repair.** `repair.sh TAG G N` takes no input paths. It rebuilds `R=$W/$TAG` and reads `$R/packed_G$G.mpmct1` and `$R/packed_G$G.mpmct1.bounds` (`W/repair.sh:9-14`). Its G is therefore only a file-name key: it must equal the suffix in the packer's argument 8, spelled exactly `$R/packed_G<G>.mpmct1`. It sets no window parameter. The split and scoring budget of 65,536 is a literal in `repair.sh` (`:31,36`) and does not depend on G. Outputs go to `$R/rep<G>.mpmct1` and `$R/rep<G>.bounds` (`:11-12`). The port keeps the names: `tdp_packing run` writes `packing/rep<G>.mpmct1` and `packing/rep<G>.bounds` (G = `packing.window_gates`), and the driver hands exactly these to `tdp_unpacking run --refine` as `--packed`/`--bounds`, the lattice the refinement starts from (`scripts/tdp_gen.sh:1058-1060, 1140-1142`).

**Packer failure modes.** The scheduler can raise `RuntimeError("burst stuck")` (`RD/window_pack3.py:395`) or `RuntimeError("scheduler stuck: no ready gates")` (`:456`). Either aborts before any output is written, and neither appeared in the eight validation packs. Under `val_chain.sh`, which has no `set -e`, `repair.sh` would then run on a missing or stale `packed_G512.mpmct1`. Check `pack_G512.txt` for a traceback (§8.7). (Oracle. In the port a stuck scheduler is a non-zero exit of `tdp_packing` with nothing written; the managed driver records `exit=<code>`, `step=packing` in `delivery.failed` and stops.)

**Why these values.**

- **G = W_cap = 512.** In the G sweep (packer v3, n=64, link-scored, `RESULTS.md:1520-1529`), each doubling of G from 128 to 512 cost "+4-8% size" and cut M2 by about 25% and linkable froth by about 60%. At n=128, a K=64-retirement arm cut L2 from 43,507 to 30,609 between G=128 and G=512, for +29% size (`RESULTS.md:1590-1592`). The earlier hidden-fraction law was about +14 points of carried values hidden per doubling of G (`RESULTS.md:537-542`). The lever saturates at G=1024: L2/M2 match greedy link-breaking deletions at G=512, and 7 windows exceed 65k monomials (max 386k) (`RESULTS.md:1543-1546`). W_cap = G lets an interval fill one window.
- **B_mono = 48·G = 24,576.** This keeps the monomial budget proportional to the gate cap (`W/score_arm.sh:10`). The notebook records that the ANF budget barely binds at the smaller caps: no overflow at G=128 and one in 3,961 windows at G=256 (`RESULTS.md:472`, `:522-524`). No sweep of the 48 factor is recorded.
- **Split and scoring budget 65,536.** No derivation is recorded. The G=128-era emission budget was 16,384 (`docs/ANF_WINDOW_DELIVERY.md:118`). By the G=512 sweep, 65,536 was in use, and 2-3 packed windows per G=512 run exceeded it (`RESULTS.md:1524-1527,1576`). *Inferred:* it is kept separate from B_mono because B_mono is only a soft scheduling budget that single gates overshoot (§6a.4, "Caps can be overshot").
- **span_cap = 3000.** This is the packer's code default (`RD/window_pack3.py:156`). No sweep or rationale is recorded. At G=512 it disqualifies 7.1-7.9k values at n=64 and 24.8-27.9k at n=128 as "long-span" (line 2 of `W/v*/pack_G512.txt`).

#### 6a.4 Packing: `window_pack3.py`

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
- `sibling` (line 301): the value lost another span. The module docstring calls this `sibling_kill` (line 14); the port reproduces the oracle's log lines byte for byte (`src/stages/packing/scheduler.rs`), so its death-cause names are the oracle's.
- `final` (line 460) and `resumed_later` (lines 372-375; `docs/ANF_WINDOW_DELIVERY.md:84-87`) are still counted in code, but neither appears in the eight validation logs. `resumed_later` appears unreachable in v3.

**Verification and statistics.** After emission, `proxy_spans(A, newB)` (G2 against the reordered G3) is re-run and the final states are compared. This prints `VERIFY: … OK/FAIL` but does not abort, and the outputs are already written by then (`RD/window_pack3.py:470-487`). `RESULT:` reports "exposed (packer)" values against this ground truth, for the emitted pre-repair order (§6a.1). Repair later changes boundary states (§6a.5).

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

#### 6a.5 Repair: `W/repair.sh TAG G N`

The packer works only on stage-2 values. The residual C leaks appear only after reordering, in stage 3 and/or the packer. On one artifact (rs64, which used K=64 retirement) they were localized to the mixer. This was not re-checked on the recommended configuration (§3.8, §7.6; `RESULTS.md:1373-1376`). In the 29-variant study, the span check was zero at every scanned stage-2 cut: every 16th cut at n=64 and every 128th at n=128 (`W/span_jobs.txt`; `W/span_all.sh:6`; `STAGE2_LINKABILITY.md:60-67`; §7.6). The replicate driver repeats the check with strides 64/256 (`W/val_chain.sh:25-27`).

Repair removes the residue. It is organized as a sequence of function-preserving rewrites of the packed order, in which boundaries are markers that stay between the same surviving gates (`W/order_repair.py:1-17`). Two gates commute iff neither's target is a control of the other (lines 66-67). "The same surviving gates" does not mean the same boundary **states**. Cancellation and slides move or delete gates across markers, so the wire state at a boundary can change (step 1).

The script uses `set -uo pipefail` without `-e` and hard-codes `ROOT` and `W` (`W/repair.sh:5-7`). Its steps:

1. **Pair cancellation** (`order_repair.py … cancel`, line 14). For each gate g at position i, `blk` is the first later gate that reads g's target or writes one of g's controls (`W/order_repair.py:88-90`). The next identical gate j with i < j < blk is cancelled together with g, since g is self-inverse and everything between them commutes with it. Passes repeat to a fixed point (lines 101-112). This removes about 7% of delivered gates. For example, vL128a went from 4,596,808 to 4,259,448 gates (168,680 pairs) (`W/vL128a/repair512.log`).
   - **Cancellation crosses boundaries.** `cancel_pass` runs on the bare gate list. `run_cancel` strips the markers before calling it and only afterwards drops the dead gates from the marker sequence (`W/order_repair.py:69-99,101-112`). A pair is therefore cancelled even when boundaries lie between i and j. At every such boundary the state changes: g's increment, applied at i and undone at j, is no longer present. The overall function is unchanged.
   - **Consequence.** The packer's "fully hidden by construction" and RESULT "exposed (packer)" figures (§6a.4) describe the **pre-repair** lattice, not the repaired one the deliverable is built on. The delivered stage-2 exposure is the link census's `proxy values` on `rep512` (§6a.1, §7.5, §7.6). Re-running the packer's RESULT computation on `rep512` would give a like-for-like post-repair count, but this has not been done.
2. **C-view scan.** `cview()` (lines 17-21) runs `hmap_affine --exposure --c-step 1 --g-cuts rep$G.bounds` with `EXPOSURE_WITNESS` and `EXPOSURE_ALLCUTS` set (§7.3). That binary requires `--features security-tools` (`Cargo.toml:308-311`; `:249-251` at HEAD). It uses 6,144 samples, seed 12345 (`HA:112-117`). Per cut, it deduplicates the scanned circuit's wire columns under complement and builds a GF(2) basis; a C value is visible iff it reduces to zero (`HA:512-590`). t\* is minimized over the nullspace coset by Gray code when the nullity is ≤ 14; above that, the value is an upper bound and is reported as "coset-minimizations skipped" (vL128a v1 skipped 1; `HA:597-616`). Only the best cut per value is kept as witness, with ties going to the earliest cut (`HA:619-633`). `cview()` returns the number parsed from line 2 of the log.
3. **Fix loop**, at most 4 iterations (lines 22-26). It stops when `vis ≤ N`. Otherwise it runs `order_repair.py … fix wit.tsv exposure.tsv | tail -1`. The `tail -1` keeps only the gate/boundary count line and drops the `moved N gates` line (`W/order_repair.py:185,189`), so the logs cannot show whether a stalled pass moved anything.
4. **Synchronous equivalence check** (line 27; see below).
5. **Fallback loop**, at most 4 iterations (lines 28-33). If `vis > N`, run `merge_bounds2.py`; then always run `anf_split2.py` with budget 65,536. Stop when `vis ≤ N` and zero splits were added. The `fallback iter` lines of `repair512.log` are the ones to check.
   - **`anf_split2.py` interface.** The call is `anf_split2.py TAG G N BUDGET bounds_in bounds_out` (`W/anf_split2.py:5,9`), but two required inputs come from the environment, not the CLI. `SPLIT_P` is the circuit whose windows are split; it must be the **repaired** circuit `rep<G>.mpmct1`. `SPLIT_EXP` is the C-view `exposure.tsv` from which the C outputs (the last C gate per wire) are taken; it must be the **current** `rep<G>.exposure.tsv`. `repair.sh` sets both (`W/repair.sh:31`).
   - **Dangerous defaults.** Without these variables, the script falls back to `$R/packed_G<G>.mpmct1`, the unrepaired pack (`W/anf_split2.py:12`), and to `$R/fin.exposure.tsv` (`:18`), the output map of the old `finalize3.sh` flow, which is stale or missing in a new run directory. Called outside `repair.sh`, or ported from its CLI alone, it would split windows of the wrong circuit against the wrong output map. `R` is rebuilt from TAG as `$W/$TAG` (`:11`), and scratch files `split_cands_<pid>.*` are written there (`:23-28`).
6. **Scoring.** `anf_compose.py … 65536 … 500` and `link_census.py` (lines 36-37).

`verify_equiv.py` checks the packed circuit against the repaired one on 1,024 random full states. It runs twice: in the background right after cancellation, and in the foreground after the fix loop (lines 15-16, 27). The background check is best-effort only. It reads `$P` while the fix loop may be rewriting `$P` in place, and `save` truncates the file on open (`W/order_repair.py:40-44`). It can therefore see a fix-modified or partly written file. The fallback changes only bounds, so it needs no equivalence check.

**Completer/breaker slides (fix mode).** Each witness row gives a C value, its best cut (as a gate count), and a predictor support S (`W/order_repair.py:148-152`). The relation holds from the **completer** (the last write to S before the cut) to the **breaker** (the first write to S after it). The fix slides the completer forward and the breaker backward as far as commutation allows (`W/order_repair.py:114-137`).

`slide` stops at the first non-commuting gate. It crosses boundary markers only when it also passes at least one commuting gate. A gate whose blocker sits right after a marker is not moved across that marker (`dest == pos`, lines 127-128), even though crossing it would break the relation at the cut. This is a candidate cause of fix-loop stalls. The port reproduces this behaviour bit for bit; letting `slide` cross markers when only markers separate a gate from its blocker is a recorded future option behind a flag, never the default without re-validation (§6a.9 status).

Example: an n=128 bounded-chain fringe leak (five wires XOR to the C value) had its completer at gate 8,507 and its breaker at gate 11,083. The leak needed a window of about 2,600 gates and was resolved by slides alone (`RESULTS.md:1737-1741`).

Output gates of C are excluded by identity: `outs` is the last C gate per wire, taken from `exposure.tsv` (lines 140-146).

There are two versions of fix mode:

- **Snapshot.** `scripts/windowed_delivery/waist/order_repair.py:155-170` re-finds the breaker in `gpos[cut:]` after the completer move (lines 163-170). When the completer has crossed the cut, that gate-count index is stale by one, so the completer itself can be re-selected as "breaker" and slid back.
- **Live.** The run-tree copy `W/order_repair.py` (last modified 2026-10-08 08:24) keeps the pre-move breaker index `brk_i`, which stays valid because the pop and insert shifts cancel past the completer's landing point, and drops it if the completer passed the breaker (`W/order_repair.py:158-171`). `val_chain.sh` calls this copy; the vL128a rerun was the first run made with it, and the eight archived replicates were then re-repaired with it (§6a.7).

Both versions share three further limits:

- **Cross-row cut staleness.** `cut` is a gate count read from the witness file of the preceding scan (line 152). Earlier rows in the same pass can slide gates across a later row's marker, so `gpos[:cut]`/`gpos[cut:]` (lines 156-159) can be off.
- **One cut per value per pass.** Fix acts only on each value's single witness cut, even when `allcuts.tsv` shows the value at several cuts.
- **Lost move count.** The moved count is discarded by `tail -1`, as noted in step 3.

**The ported version is the patched `W/order_repair.py`** (the copy `W/repair.sh:14,25` calls; the snapshot's breaker re-find is a known bug). It is `stages::packing::repair`, which runs in every `tdp_packing run`; the eight archived replicates, re-repaired with it and re-serialized on 2026-10-08, are its parity targets (`tdp_packing run` reproduces every current `rep512.*` and `deliverable.bin` byte for byte: `tests/manual/stage4/run_parity.sh`, `STAGE4_RUST_DESIGN.md` Appendix B; the earlier repairs are in `W/<TAG>/rep512_v1/`, §6a.7). Stalls occur under both versions: the snapshot stalled at 65 (vL64b), 129 (vA128b) and 131 (vL128a v1); the live version stalled at 130 (vL128a rerun).

**Merge and exposure-aware split.**

- `merge_bounds2.py` deletes every boundary at which a non-output C value is exactly affine (`W/merge_bounds2.py:6-19`). Removing cuts is a lattice subset, so nothing can become newly visible (`RESULTS.md:1400-1406`); the cost is larger windows.
- `anf_split2.py` splits each window whose ANF exceeds the budget (product > 8B, running total > 4B, or final > B). Windows of ≤ 2 gates are never split. It takes 32 evenly spaced candidates `lo+(hi−lo)·i//33`, runs one C-view on them, and picks the exposure-free candidate nearest the midpoint, then recurses (`W/anf_split2.py:22-55`).
- If no candidate is clean, it prints `WARNING: no exposure-free split` and splits anyway (`:44-46`).
- The earlier midpoint splitter (`anf_split.py`) re-created exposing cuts and oscillated (`RESULTS.md:1593-1600`).

**Failure modes.**

- **Count-based stop test.** The test is `vis ≤ N` (lines 24, 32), and the scans cover only interior bounds (`RD/window_pack3.py:461`; `HA:331-340`).
  - `vis ≤ N` certifies the floor only if all N outputs are visible at some interior boundary. Otherwise a non-output leak can hide under the count: for example, N−1 visible outputs plus 1 leak still passes.
  - An identity-based test, "no visible row whose gate is not in `outs`", is already computed by `merge_bounds2.py:6-15` and `anf_split2.py:17-21,34`. It should replace the count. The port does this: `tdp_packing` ends with the identity test and exits 5 on a visible non-output row (`--allow-floor-miss` ships it with `status=floor_miss` and `NOT_RECOMMENDED`); the refinement extends the test to every added cut (`check.new_cuts`).
  - In all eight final scans, the visible set was exactly the N last-writer outputs, all with t\*=1. vL128a v1's last scan had 128 outputs plus 1 interior value.
- **No final rescan when the fallback is exhausted** (oracle; the port rescans after the last fallback iteration: identity test, exit 5, and the split's budget test, exit 6). On a normal break (vis ≤ N and 0 splits) the last scan covers the final lattice. If the 4 fallback iterations run out, the final merge/split is never rescanned and the script still scores and emits. The "129" that `repair.sh:39` prints for vL128a v1 is from the scan before the last merge and split.
- **Silent scan failure** (oracle; in the port the C-view engine is in-process, `src/stages/packing/cview.rs`, and a failed check is exit 4). Without `-e`, a failed `hmap_affine` leaves `vis` empty. The `[ "$vis" -le "$N" ]` test then errors and evaluates false, so the loops go on to fix, merge or split using stale TSVs.
- **Failed original run.** In the original vL128a run (`W/vL128a/rep512_v1/repair512.log`):
  - the fix loop went 132, 131, 131, 131;
  - fallback iteration 1 (131 visible) merged 3 boundaries and added 3 splits, one of them forced (`WARNING: no exposure-free split in [7349,8962)`); iterations 2-4 each scored 129, merged 1 boundary and re-added a forced split in the same range;
  - the last scan showed 129 with 1 coset skip, still above the floor, and the final lattice was never scanned;
  - a `deliverable.bin` was produced anyway.
- **Rerun.** The rerun with the live `order_repair.py` finished at 08:44 (`W/vL128a/repair512.log`):
  - cancellation was identical, and the fix loop went 132, 130, 130, 130;
  - fallback iteration 1 merged 1 boundary and added 1 clean split, and iteration 2 scanned 128 with 0 splits, so the floor was reached via merge;
  - ANF size 98.2 MB; L2 16,106; M2 187,248; froth L2 5,579.
- **Stale deliverable** (superseded). `W/vL128a/deliverable.bin` (01:35) came from the v1 lattice until it was re-serialized on 2026-10-08 (§6a.7); the v1 artifacts are archived in `W/vL128a/rep512_v1/`.
- **Concurrent runs.** Two repairs of the same TAG/G collide on the `rep$G.*` scratch names.

#### 6a.6 Serialization and evaluation

**Window ANF.** `window_anf(gates, lo, hi, budget)` (`RD/anf_compose.py:66-91`) composes each touched wire's new value as a polynomial over the **window-input** values. It aborts on a product larger than 8·budget, a running total larger than 4·budget, or a final total larger than budget.

`changed` keeps only wires whose polynomial differs from {x_w} (`RD/anf_compose.py:87`), so a wire that is written and then restored inside the window is not emitted. Pairwise toggling in `mul` is essential: the older `symmetric_difference_update(generator)` deduplicated instead of cancelling (`RESULTS.md:416-419`).

Worked example: a window with gates `2 0 2 0 1 1 1` then `3 0 1 2 0`.

- The first gate sets wire 2 to x2 + x0·x1.
- The second gate has a negative control on wire 2. Its product is 1 + x2 + x0·x1, so wire 3 becomes x3 + 1 + x2 + x0·x1.
- Result: changed = {2, 3}, support = {0, 1, 2, 3}.

**`deliverable.bin` format** (`W/anf_dump_eval.py:14-50`, little-endian). The file is a concatenation of `[u32 blob_len][blob]`, one per window, with no file header: no n, no window count and no format version. This is the raw body, kept as `packing/deliverable.bin`. The stage's container is `deliverable.tdp4` (`LMTDPANF` version 1): a 48-byte header with magic, version, source and state wires, layout, window count, body length and CRC-32 of body and header, followed by this body unchanged (`STAGE4_RUST_DESIGN.md` §6); the refined `unpacking/deliverable.tdp4` has the same format. The container is the published object only under `anf_only`; otherwise it is the internal representation from which the unpacking sub-step resynthesizes the published circuit. Each blob contains, in order:

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

**Evaluating `deliverable.bin` (consumer side).** Decoders exist: `tdp_packing decode <file> (--format raw --n N | --format container) [--stats]` and `tdp_packing eval` (`src/programs/tdp_packing/mod.rs:23-24`), `stages::packing::serialize` (read back on every run), and `tdp_unpacking`, which decodes the container to resynthesize every window (`docs/STAGE7_UNPACKING.md` §2). The procedure below is the raw-body layout, *derived* from the writer (`W/anf_dump_eval.py:14-50`) and the in-memory evaluator (`:92-108`); a consumer validates the container header first. The port runs decode = window ANF and `decode(D) ≡ rep<G>.mpmct1` on every run.

1. **Inputs needed out of band.** n. The file has no header, so the width 4n cannot be read from it. The largest wire id that appears need not be 4n−1, because a wire no window touches never appears. Allocate a state of 4n bits; wires that never appear stay 0 and are never read.
2. **Slice setup.** Put x on wires `0..n`, and set wires `n..4n` (the answer register y and the band) to 0.
3. **Decode loop, in file order** (little-endian throughout). Repeat until end of file:
   1. Read `u32 L` and then the L-byte blob. Inside the blob, read `u16 f` and `u16 c`, then f × `u16` support ids `s_0 < … < s_{f−1}`.
   2. Snapshot `v_i = state[s_i]` for all i.
   3. For each of the c changed wires: read `u16 w` and `u32 k`, then k masks of ⌈f/8⌉ bytes each. Bit i of a mask selects `v_i`, and the all-zero mask is the constant 1. Set `state[w]` to the XOR over the k masks of the AND of the selected `v_i`.
   4. Every wire outside the c changed wires keeps its value. A blob with f = c = 0 is an identity window, and empty windows only arise from the duplicate-boundary quirk (§6a.4).
4. **Readout.** Wires `n..2n` hold C(x). Wires `0..n` and `2n..4n` are junk.

All three exist since the port: the container header (`STAGE4_RUST_DESIGN.md` §6), a validating decoder with read-back, and the per-run checks `decode(D) ≡ rep<G>.mpmct1` (1,024 states) and zero slice against C (4,096 inputs) in `tdp_packing`, repeated on the refined deliverable and on `unpacked.mpmct1` by `tdp_unpacking --refine` (`check.final_deliverable`, `check.zero_slice`; §6a.9, §6b, §7.7, §8.12).

**Checks and size model.**

- `anf_dump_eval.py` compares the gate-form and ANF-form final states on 1,024 random full states (`match=`, line 110). It does this on the in-memory windows, not on a decode of `deliverable.bin` (oracle; the port decodes the written container and checks it, see above and the §6a.9 status). Lines 69-90 run a full placeholder ANF evaluation pass whose result and timing (`ta_wrong`) are discarded (`st2` is reset at line 92).
- `anf_compose.py` spot-checks only windows with index ≡ 0 mod 500 (line 159), on 256 samples. A failure prints but does not abort.
- Its size model is `8 + 8f + 24·|changed| + total·f` bits per window (line 155). It underestimates the real bytes for two reasons. It counts masks in bits rather than whole bytes. It also assumes 8-bit ids, 8+16-bit per-changed-wire fields, and no length prefix; at n=128 (512 wires) an 8-bit id could not even address the wires. Mask padding is the largest term.
  - For vA64a (decode of `W/vA64a/deliverable.bin`): the model gives 24.60 MB; the real file is 26.04 MB = masks 24.38 + per-changed-wire fields 0.94 + support ids 0.69 + length prefixes and headers 0.02.
  - The 1.44 MB gap ≈ mask padding 0.60 + per-changed fields 0.47 + support ids 0.34 + headers 0.02.

#### 6a.7 Reference replicates (G=512)

The eight archived G=512 replicates (configs A and L8, n=64 and n=128, `W/<TAG>/`) are the parity targets of the Rust port: `tdp_packing run` reproduces every `rep512.*` and `deliverable.bin` byte for byte from the archived `tdp.mpmct1` and `s3.mpmct1` (`STAGE4_RUST_DESIGN.md` Appendix B; `tests/manual/stage4/run_parity.sh`), and `gen_sandwich_gadget` and the mixer reproduce the archived inputs from their seeds. How these runs were made and validated in October 2026 is recorded in `docs/local_mixing_history.md`.

#### 6a.8 Trust boundary

Every step of stage 6 (both sub-steps) reads secret material:

- the packer and the link census read G2 = `tdp.mpmct1`, the unmixed stage-2 circuit that stage 3 exists to hide (`RD/window_pack3.py:153,162,168`; `W/link_census.py:20-21`);
- repair and split evaluate C's interior values and test their exact affine membership against boundary states (`W/repair.sh:18-19`; `W/anf_split2.py:26-29`);
- every intermediate circuit is a gate list of G3 (or of G5 in the six-stage chain), which exposes all interior cuts;
- the unpacking sub-step reads C (the exposure-aware cut choice runs the C-view, `CView::leaky`, and the zero-slice check) and the packed gates `rep<G>.mpmct1` (pending set, exact check, verbatim emission of a window that never resynthesizes).

*Inference:* all of stage 6 must run in the secret-holding environment. What may leave is the published circuit `unpacking/unpacked.mpmct1` (`packing/deliverable.tdp4` under `delivery.anf_only`). Everything else is secret or a secret intermediate: both ANF deliverables (in the default mode), the per-round containers and bounds, the packed and repaired circuits, the logs, and the reports `packing.report` and `unpacking.report` (counts and parameters only, but internal records, not part of the TDP). A circuit with `status=verbatim_used` carries packed windows verbatim, the packer's spelling, which is why that outcome is NOT recommended. The run directory holds at least the following never-ship files (classified in §8.8; the oracle's names, with the managed names after the list):

- `tdp.mpmct1` (G2);
- `tdp.mpmct1.source_c.g57` and `tdp.mpmct1.sandwich.mpmct1`, written beside the output (`src/programs/gen_sandwich_gadget/mod.rs:27,212,230`);
- `seed.txt`, `gen.log` and `s3.log`. These record the gen seed and the mix seed (`W/val_chain.sh:14`). `val_chain.sh:16` passes the one gen seed as seed, sandwich seed and gadget seed, and that seed regenerates C and the whole gadget bit for bit (gen `mod.rs:209-211`);
- `s3.mpmct1`, `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1` and `rep512.bounds`;
- `span_s2.tsv` and `span_s2.txt` (from `hmap_affine` on C, `W/val_chain.sh:26-27`);
- `rep512.{wit,allcuts,exposure}.tsv`, `rep512.cview.log` and `split_cands_<pid>.*`.

In a managed run: `SEEDS` (five seeds + `run_key`, mode 0600; the only copy of the seeds, since the banners in `stage12.log`, `stage3.log`, `stage4.log`, `stage5.log` and `stage6.log` print `(file)`), `tdp.mpmct1`, `db_mixing.mpmct1`, `split.mpmct1`, `split.state`, `crossing.mpmct1`, the stage logs, `stage2_span.{tsv,txt}`, everything under `packing/` (`deliverable.tdp4` included, except under `anf_only`, where it is the published object), and under `unpacking/` everything but `unpacked.mpmct1`: `deliverable.tdp4`, the per-round `bounds.<r>` / `deliverable.<r>.tdp4`, `bounds.final`, `windows.tsv`, `rounds.tsv`, `unpacking.report` and `unpacking.log`; the markers `stage3.recipe`..`stage5.recipe` and `delivery.recipe` hold only HMAC-SHA256 tags keyed by `run_key`.

#### 6a.9 Open issues for the Rust port

**Status (2026-10-09): ported; since 2026-10-12 the first sub-step of stage 6 (delivery).** Packing is `tdp_packing run`
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
never the default without re-validation. The same stage then runs
`tdp_unpacking run --refine` (resynthesis with lattice refinement, §6b;
`docs/DELIVERY_STAGE.md`). The text below is the analysis as it
stood before the port.

The documented plan (`docs/windowed_delivery_design.md:178-192`) has four items:

1. `window_pack`;
2. `anf_emit`, described as a port of `anf_compose.py` that includes the binary format;
3. band-refresh/migration as a stage 3.5, now superseded by stage-2 migration at dose 8 (`W/val_chain.sh:4`);
4. metrics in CI.

The plan has two gaps:

- **It omits repair entirely:** pair cancellation and fix slides (`order_repair.py`), `merge_bounds2.py`, `anf_split2.py`, and the C-view scan in the loop. Repair is a required step of the recommended stage 4 (packing, today §6a) and the only step that needs C. It must be ported, or kept as an oracle-backed step, inside the secret-holding environment. The split step's full interface is `TAG G N BUDGET bounds_in bounds_out` on the CLI, plus two environment inputs: `SPLIT_P` (the repaired circuit) and `SPLIT_EXP` (the current `exposure.tsv`). Their defaults point at the unrepaired pack and at `fin.exposure.tsv` (`W/anf_split2.py:12,18`; §6a.5). The port should make both explicit arguments.
- **The binary format to port is `anf_dump_eval.py`'s** (`W/anf_dump_eval.py:14-50`). `anf_compose.py` has only the size model (`RD/anf_compose.py:155`).

The port can reuse `XGate::apply_lanes`, the `formats.rs` I/O and `hmap_affine`. The existing `anf1`/`esop1` format is a per-target format, not this multi-output window format.

Before the port, policy and engineering decisions were open on (resolutions: the status paragraph above and `STAGE4_RUST_DESIGN.md` Appendix C):

- hard failure on a VERIFY mismatch, a failed spot check, a failed or missing C-view scan, or a C-view above the floor;
- an identity-based floor test in place of `vis ≤ N`;
- what to do when no exposure-free split exists;
- a final rescan after the last merge/split;
- verifying against a snapshot copy, not a file the fix loop is rewriting;
- fix mode:
  - validate the canonical (live) breaker handling by re-repairing the seven snapshot-repaired replicates (§6a.5) [done 2026-10-08];
  - let `slide` cross markers when only markers separate a gate from its blocker;
  - track cuts by marker identity rather than gate count;
  - consider all exposing cuts, not just the witness;
  - log the moved count;
- boundary deduplication;
- death-cause naming (`sibling` vs `sibling_kill`; `final`; pre-emptive closes booked as `drain_close`);
- wire-class thresholds;
- a round-trip decode test, a reference decoder (procedure in §6a.6), and a versioned container header (n, window count) [done: container v1, validating decoder];
- the pre-repair packer exposure versus the delivered proxy count (§6a.1, §6a.5): report the delivered figure;
- reporting real bytes instead of the size model;
- location-independent paths: `W/val_chain.sh:7,19`, `W/repair.sh:6-7`, `W/score_arm.sh:6`, `W/anf_split2.py:7,10`, `W/anf_dump_eval.py:5`, `W/verify_equiv.py:3`, `W/link_census.py:17`.

Bit-exact parity with the Python oracle requires replicating its dict-insertion order and heap tie-breaks, including the tail-flush order. Achieved: `pyset.rs` (CPython 3.9 set emulator), `pyrand.rs`, `anf_cap_semantics = cpython39`; parity on all eight archived runs and the CI differential job `stage4-oracle`.


### 6b. Stage 6, second sub-step: unpacking with lattice refinement

*Added 2026-10-12. Stages 6 (packing) and 7 (unpacking) of the seven-stage driver became one delivery stage on 2026-10-12 (RC; `docs/DELIVERY_STAGE.md` §1). The packing sub-step is §6a above; this sub-section describes what follows it. Line numbers refer to commit `83e44ec5` (on `main`).*

Unpacking turns the packer's windowed ANF representation back into a circuit, inside the pipeline in the default mode (its result is the published circuit) and by the recipient under `anf_only`; the refinement makes sure that it succeeds for every window. The sub-step takes the packer's repaired circuit `rep512.mpmct1`, its lattice `rep512.bounds` and its container `packing/deliverable.tdp4`, resynthesizes every window from its delivered ANF alone, and treats a window that does not resynthesize as a packing defect: the window is cut in two at an exposure-aware cut, both halves are re-emitted into the deliverable and unpacked in the next round, recursively, until every window unpacks or `max_rounds` is reached. The size of a resynthesized window is irrelevant ("windows that unpack badly are completely fine", RC). Its two outputs are the resynthesized circuit in canonical commuting order, `unpacking/unpacked.mpmct1`, which is **the published circuit**, and the deliverable on the refined lattice, `unpacking/deliverable.tdp4`, the internal representation it was resynthesized from, kept for auditing and parity (`docs/tdp_pipeline.md`, "Stage 6"). With `delivery.anf_only = true` the stage ends after the packer's deliverable and this sub-step does not run; together with `run.skip_stages = [4, 5]` that is the validated four-stage chain.

#### 6b.1 What it does

**Per window: resynthesis from the ANF alone** (`src/stages/unpacking/run.rs:1-20`; the kernel of `docs/STAGE7_UNPACKING.md` §2). The decoded window (support wires, changed wires, monomials as masks over the support) is mapped onto a local variable universe: its support and changed wires in ascending global order plus the lowest 32 other state wires, which the search may borrow as dirty ancillas (`spec.rs:27`); the monomial width (1, 2, 4, 8 or 16 words) follows the universe size. The synthesizer (`synth.rs:1960`) receives this spec, the run seed and the window index, and nothing else: it imports no circuit type and sees no gate, no file and no window position, the separation guarantee of `anf_synth` (`synth.rs:6-19`). Its state is the residual specification $P_w$, one polynomial per wire; a forward gate $g=(t;L)$ substitutes $x_t \leftarrow x_t + L(x)$ in every polynomial, a backward gate is appended on the output side ($P_t \leftarrow P_t + L(P_a,P_b)$), and the window is done when every $P_w = \{x_w\}$. Per step it collects forward substitution and backward peel candidates with free estimates, rescores the best `exact_k` = 48 forward and `back_k` = 32 backward ones exactly against the degree-weighted distance $D=\sum_w |P_w \oplus \{x_w\}|$ (a monomial weighs $4+(\deg-2)\cdot 4$ above degree 2, a constant 1, a linear term 2), takes the greedy best after a 2-step lookahead over the top `beam` = 6 of each family (backward wins ties), finishes affine residuals by Gauss–Jordan, builds Barenco helper macros for degree $\geq 3$ monomials in amortised batches of up to 32, and perturbs with a tabu move after `stall_limit` = 60 non-improving steps; 16 restarts cycle the degree-weight schedule $(d_w, d_w+2, 0, d_w-2)$ under one 30 s wall-clock deadline per window, and the shortest converged restart is kept. The pairwise peephole (`postopt.rs:100`) then merges or cancels same-target gates brought adjacent by commutation; it is kept only if it re-verifies. The gates, mapped back to global wires, pass two checks before they count: sampled (`verify.rs:103`: on 1,024 random full states every changed wire equals the window ANF evaluated from the decoded masks and every other wire, borrowed ancillas included, is unchanged) and exact (`verify.rs:54`: the gates composed by the packer's audited `window_anf` must encode to the delivered blob byte for byte, at budget $2^{18}$ and once more at $2^{20}$ on overflow, `run.rs:35`). A window is *converged* when its status is `converged` and both checks pass; otherwise it is *failed*, with status `timeout`, `stalled`, `stepcap` (cap $20\cdot$monomials$+64$ gates), `verify-fail` or `exact-overflow` (`run.rs:85-87`, `synth.rs:243-249`). Ties are broken by a splitmix64 stream seeded from (seed, window index, restart) only, so a run reproduces bit for bit unless the deadline truncates a window (`deadline_hit` in `windows.tsv`; the result is still verified).

**The loop** (`refine.rs:397-606`; `docs/DELIVERY_STAGE.md` §2). Round $r$ has a lattice $L_r$ of windows $[lo,hi)$ over the packed circuit and a deliverable $D_r$, the container over their blobs (`refine.rs:148`); $L_0$ and $D_0$ are the packer's. Each round synthesizes the *pending* windows (round 0: all of them; round $r>0$: the halves created in round $r-1$), each with its index in the **current** lattice, which keys the tie-break PRNG and the verify seed (`refine.rs:469-482`). For every failed window of $\geq 2$ gates a cut is chosen as in the packer's budget split (`refine.rs:252-279`): the 32 candidates $lo + \lfloor (hi-lo)\,i/33 \rfloor$, $i=1..32$ (`packing/repair/split.rs:17`), minus those at which some interior value of $C$ is exactly affine in the boundary state, found by **one** C-view scan over the union of all failed windows' candidates (`packing/repair/mod.rs:154`; the C-view is built once on $C$ with the packer's sampling); among the clean candidates the one nearest the midpoint, the smaller on a tie (`split.rs:30`); if none is clean, the midpoint, counted in `unclean_cuts`. The two halves' blobs are emitted by the packer's own `window_anf` emission with CPython 3.9 cap semantics, so a blob is a pure function of `gates[lo:hi]` (`refine.rs:213`), and $D_{r+1}$ is $D_r$ with the split windows' blobs replaced by their halves' and the header (window count, body bytes, CRCs) recomputed; nothing else changes. A half whose ANF exceeds the packer's 65,536-monomial split budget is emitted so that the container stays complete, counted in `over_budget`, not synthesized, and split again in the next round (`refine.rs:182-206,522-538`). A failed window of one gate is emitted verbatim from the packed circuit (`refine.rs:540-544`). Round $r$ may split iff $r+1 \le$ `max_rounds` (`refine.rs:502`), so a run has at most `max_rounds`$+1$ synthesis rounds; whatever is still failed after the last one is emitted verbatim too (`refine.rs:550-554`) and the run's status becomes `verbatim_used`. The loop ends when nothing is pending.

**Assembly, canonical order and checks** (`src/programs/tdp_unpacking/mod.rs:1063-1128`). The pieces' gates are concatenated in final-lattice order, checked against the final deliverable's evaluator on $4\times1024$ random all-wire states, put into the canonical commuting order by `xgate_canonicalize` (`src/canonicalization/gate_order.rs:64`: every gate is pulled as early as commutation allows when that lowers its lexicographic key, so no window boundary survives in the gate order) and checked again; the zero slice is checked against $C$ ($x$ on wires $0..n$, zeros elsewhere, $C(x)$ on $n..2n$, 4,096 samples). The final deliverable is decoded, every window of it is checked blob-equal to `window_anf` of the packed gates and equal to them on 1,024 states, the whole is checked against the packed circuit and against $C$ on the zero slice (`check.final_deliverable`), every final window must fit the budget, and `check.new_cuts` records that every added cut was scanned clean or counted unclean. A per-cut C-view verdict does not depend on the lattice, so the packer's final scan of the old cuts plus the choice-time scan of the new ones is the final scan of the refined lattice, without a second full pass (`docs/DELIVERY_STAGE.md` §4). Any failure is exit 4 and nothing is renamed: every output is written to `*.tmp` and committed only after all checks pass, the report last.

#### 6b.2 The two implementations

| | Python reference | Rust |
|---|---|---|
| entry | `scripts/windowed_delivery/waist/deliver.py --rep rep512.mpmct1 --bounds rep512.bounds --body deliverable.bin --source-c C --n N --out DIR` | `tdp_unpacking run --deliverable packing/deliverable.tdp4 --packed rep512.mpmct1 --bounds rep512.bounds --source-c C --n N --out-dir unpacking --seed-file /dev/fd/3 --refine --max-rounds 16` |
| kernels | orchestration only: `tdp_unpacking run --windows i,j,…` per round (`deliver.py:207-232`), `hmap_affine --exposure --g-cuts` for the cut scan (`:236-265`), `tdp_unpacking canon` or a Python canonicalizer (`:396-409`), `anf_compose.window_anf` for the halves' blobs (`:364-368`) | in process: `stages::unpacking::refine`, the library C-view (`packing::repair::CView`), `packing::serialize::encode_window` |
| report | `delivery.report` | `unpacking.report` (the plain run's keys plus the `refine` keys) |
| over-budget half | synthesized and delivered; `check.final_budget` then fails on it (`deliver.py:364-368,432`) | not synthesized, split again; fatal if still over budget at `max_rounds` (`refine.rs:27-31`; `mod.rs:1113-1119`) |

The two are byte-identical on the n8_L8 fixture in `bounds.final`, `deliverable.tdp4`, `unpacked.mpmct1` and every per-round `bounds.<r>` / `deliverable.<r>.tdp4` (`tests/manual/delivery_parity.sh`, run with `RESTARTS=2` so that window 1 of the fixture stalls and is split: 73 windows become 74). The driver (`scripts/tdp_gen.sh:1140-1146`; `src/tdp` in parity) runs the Rust binary right after the packer, always with `--refine --max-rounds N` and the synthesizer knobs only when they are not at their defaults (`tdp_gen.sh:463-469`), the delivery seed (`unpack_seed`, `SEEDS` format 3) on `/dev/fd/3`, the output appended to `stage6.log` after a separator line. The integration test exercises the split, the re-emission and the next round on n8_L8 with the test-only `--force-fail-rounds 1` (`tests/integration/tdp_unpacking.rs:430-585`).

#### 6b.3 Parameters

| key (`[delivery]`) | flag | default | range | role |
|---|---|---|---|---|
| `anf_only` | `--delivery-anf-only` | false | | end stage 6 after the packer's deliverable: no resynthesis, no refinement, no `unpacking/` |
| `time_limit` | `--delivery-time-limit S` | 30 | $(0, 86400]$ s | synthesis wall limit per window per round, shared by its restarts; a window that hits it keeps its best converged restart or fails the round |
| `restarts` | `--delivery-restarts R` | 16 | 1..1000 | restarts per window ("less optimized" = fewer, e.g. 2) |
| `beam` | `--delivery-beam B` | 6 | 1..1000 | lookahead width |
| `max_rounds` | `--delivery-max-rounds N` | 16 | 1..1000 | rounds that may split; a 1,024-gate window reaches 1-gate pieces in 10 |
| `run.threads` | `--threads` | all cores | 1..1024 | the rayon pool; it affects the output only through the deadline |

Defaults and ranges: `src/tdp/config/mod.rs:299-321`, `validate.rs:428-447`, `tdp_gen.sh:189,458-469`. The inner knobs are the `anf_synth` production defaults and are not exposed by the driver (`synth.rs:162-183`, `types.rs:30-34`; the report prints them as `param.*`): `exact_k` 48, `back_k` 32, `back_look` 8, `back_prefer` true, `step_factor` 20, `stall_limit` 60, `deg_weight` 4, `w0` 1, `w1` 2, `macro_batch` 32, macros on, peephole on, `ancilla_pool` 32, exact budgets $2^{18}, 2^{20}$, `verify_seed` 20261011 (public), `equiv_batches` 4, zero-slice samples 4,096; the refinement's budget 65,536 = the packer's split budget, its 32 candidates = the packer's (`refine.rs:53`, `split.rs:14`), its C-view sampling = the packer's defaults. The seven-stage driver's `unpacking.allow_fallback` has no counterpart: the refinement replaces the verbatim fallback.

#### 6b.4 Outputs (`unpacking/`, created 0700; stale outputs of an earlier run are removed first)

| file | content |
|---|---|
| `deliverable.tdp4` | $D_{final}$, the container (`LMTDPANF` v1, `docs/STAGE4_RUST_DESIGN.md` §6) on the refined lattice: the internal windowed representation that `unpacked.mpmct1` was resynthesized from, kept for auditing and parity; not published |
| `bounds.final` | its cuts, one per line, $0<b<m$ |
| `unpacked.mpmct1` | **the published circuit (the TDP)**: every window's resynthesized gates in final-lattice order, then canonical order; header = the state width |
| `windows.tsv` | one row per final window: `idx lo hi round orig_gates syn_gates status synth_status seconds monomials changed_wires input_vars max_degree`, then the plain run's `syn_offset pre_postopt_gates steps note tag_* universe width exact_budget restarts_used deadline_hit`; a verbatim row has round $-1$, status `verbatim`, the reason as `synth_status` (`refine.rs:644`) |
| `rounds.tsv` | `round pending converged failed split seconds` (`refine.rs:686`); the per-round unclean count is in the log line and the report (`unclean_cuts`), not a column |
| `unpacking.report` | `key=value`, an internal record (counts and parameters only: no seed value, no hash of a private input; not part of the TDP): the plain run's keys (`windows synthesized fallback deadline_hits verify_fail exact_overflow gates_in gates_out gates_canonical ratio gates_by_move.* seconds.* restarts_used.*`) plus `refine=1 rounds windows_initial windows_final splits unclean_cuts over_budget verbatim param.max_rounds param.budget`, every `check.*`, `time.*`, `status=ok\|verbatim_used` (`mod.rs:457-600`) |
| `bounds.<r>`, `deliverable.<r>.tdp4` | each round's lattice and container (round 0 = the inputs, re-encoded) |
| `unpacking.log` | every 50th window, every failure, the round lines, the summary line |

The driver adds `stage6.log`, the keyed marker `delivery.recipe` (both binaries' hashes, the packing and delivery parameters, the tag of the delivery seed, HMAC tags of the inputs, of both deliverables and of `unpacked.mpmct1`; a changed parameter, binary or output reruns the whole stage, both sub-steps), `delivery.failed` (`exit=`, `step=packing|unpacking`) on a failure, and `NOT_RECOMMENDED` next to the final output on `verbatim_used`, which the driver reports as an allowed failure, not an error. *Inferred:* with `verbatim = 0`, `unpacked.mpmct1` is a function of the deliverable, the seed and (through the deadline) timing: every gate in it is the synthesizer's and none the packer's; with `verbatim > 0` it contains packed gates, the packer's spelling, which is what makes `verbatim_used` NOT recommended. `rep512.*`, `bounds.*`, the deliverables and the intermediate containers never leave the secret-holding environment.

#### 6b.5 Why

**Why resynthesize, and why the published circuit carries no lattice.** The packer's windows are judged at the cuts of their lattice (§6a.1), but the published object is a circuit, and the only honest way to show that the ANF windows are a circuit's worth of functions at all is to rebuild a circuit from them and from nothing else. That rebuild is also the separation guarantee: the packer's and the mixer's spelling are gone, and the synthesizer's spelling, made from the window functions alone, is what ships. The canonical commuting order removes the window boundaries from the gate order; the measured effect of that reorder on the merged-window leak is $\le 1.7\%$ either way (244 → 242 and 241 → 239 on the crossing output at $W=2000$, 434 → 434 and 297 → 302 on the stage-3 output; `docs/notes/stubborn_windows_20261012.md` §2), as expected: a leaked value is written by one gate, and commuting reorders change only same-target partial sums. The order is a convenience, not a security lever.

**Why a failed window is split rather than emitted verbatim.** Until 2026-10-12 a window that did not converge fell back to the packed gates. That broke the unpacking contract (those gates are the packer's spelling, not computable from the window's ANF) and it scaled with the input: 0 fallbacks of 2,591 windows on the four-stage (stage-3) deliverable at n=64, but 18 of 639 at n=16 and 157 of 10,366 (1.5%) at n=64 on the crossing output (`docs/notes/seven_stage_20261011.md` §3-4). Cutting the window instead keeps every delivered window a window that unpacks, and it is cheap: on the 157, halves converged 302 of 314, quarters 21 of 24, eighths 5 of 6, sixteenths 1 of 2, thirty-seconds 2 of 2 (15- and 16-gate pieces in 0.02 and 0.2 s), about 90% per round, so the recursion terminates by itself and no verbatim floor is needed (`stubborn_windows_20261012.md` §4). The bisection-to-the-end assembly (10,540 windows = 10,209 packed + 302 + 21 + 5 + 1 + 2; 5,201,064 gates) is equivalent to `rep512.mpmct1` on 8,192 random full states and shows 64 of 1,000 $C$ values visible (the outputs), 0 interior, 0 early outputs. Round 1 took 4 min on 8 threads.

**Why not more time instead.** Ten times the budget rescues little: on the 157, 30 s converges 55 and 300 s converges 70 (15 of the 102 that failed at 30 s); the 45 stalls stay stalls because `stall_limit` ends each restart regardless of the deadline, and 160 restarts at 300 s did not rescue a 62-gate stalled piece; a 600 s rerun of the 164 deadline windows at $W=2000$ rescued 17 (median 527 s) and moved the leak from 244 to 241 interior values, the rescued windows adding 0 new ones (`stubborn_windows_20261012.md` §2-3). Time buys gates, not safety, and the stubborn set is seed- and timing-specific anyway (`anf_synth` converges 55 of the same 157 at the same 30 s).

**Why the size of the result does not matter.** No exposure metric reads the synthesized gate count: the C-view, the proxy census and linkability quantify over the cuts of the window lattice, and the measurements of the published circuit (the all-cuts census at its own cuts, the uniform W=2000 re-cut) count values, not gates; under `anf_only` the recipient may run any synthesizer it likes. We therefore accept any converged size: on today's n=16 run a 141-gate piece converged to 3,217 gates in its first restart, at the deadline, and was taken (`port_runs/six_n16b/unpacking/windows.tsv`, window 30). The gate ratio reflects the input, not the policy: 0.354 on the stage-3 deliverable, 0.925 on the n=64 crossing output, 0.683 on today's n=16 crossing output (the peephole removes 2.6%).

**What the stubborn windows are.** An ANF-density class, not a gate-structure or position class (`stubborn_windows_20261012.md` §3; 157 fallbacks against 10,209 synthesized windows, n=64):

| feature | ordinary median | stubborn median | AUC |
|---|---|---|---|
| mean monomial degree | 2.56 | 6.21 | 0.91 |
| monomials of degree $>3$ | 25% | 92% | 0.91 |
| monomials | 597 | 9,134 | 0.91 |
| packed gates / mean controls / max controls | 492 / 2.88 / 7 | 482 / 2.90 / 8 | 0.36 / 0.57 / 0.56 |
| support / changed wires / position | 111 / 47 / 0.50 | 106 / 44 / 0.46 | 0.38 / 0.39 / 0.48 |

Every window nests multi-control gates deeply (syntactic degree bound about 300), so the normal case is massive cancellation inside the window, and stubborn means that the cancellation failed: quarters 1-3 of a stubborn window hold about 230 monomials like ordinary quarters, quarter 4 a median of 809 (p90 3,829), the density sits in the tail in 74% of them, and the whole-window count far exceeds the sum of the quarters. The picture is an uncancelled half of a crossing-stage identity at the end of the window, which is why a cut near the middle separates an easy head from a dense tail that the next round cuts again.

**Why the cut is exposure-aware.** The raw crossing circuit holds no $C$ value at any cut (`docs/notes/stage45_revival_20261011.md` §3), so a midpoint cut keeps the verbatim floor, but the packer's affine C-view is a different test and must be re-run at every new cut; we reuse the packer's split machinery unchanged (32 candidates, one scan per round, the clean candidate nearest the midpoint) so that the refined lattice satisfies exactly the packer's final-scan condition. Unclean cuts are expected to be 0 (18 of 18 clean at n=16), and the census of the stubborn windows gives no reason to fear them more than ordinary ones: four by-the-book factorings of their ANFs (one multi-controlled gate per monomial into a clean ancilla, in canonical and in random order; positive-Davio decomposition in two variable orders) materialise 0 interior $C$ values, as for 60 ordinary windows, and degree-2-reachable values are rarer per stubborn window (0.34) than per ordinary one (0.62) (`stubborn_windows_20261012.md` §1).

**The open caveat.** Every zero-proxy figure above is measured at the delivered lattice, where the full unpacked output leaks only 5 of 936 interior values, so it has no power to rank stubborn against ordinary windows, and the leaky regime was never tested with bisection: merged-window resynthesis at $W=2000$ exposes 244 of 936 interior $C$ values at n=64 and 711 of 2,872 at n=128 on the crossing output, which stages 4+5 reduce but do not close (`stage45_revival_20261011.md` §4), and that leak is carried by the easy windows (first-creations per window 0.112 under 10 s against 0.055 above 60 s and 0 for fallbacks). Only the verbatim census was applied to synthesized circuits; affine and linkable exposure exist for the packed circuit only. Equivalence is sampled (8,192 or 4,096 states, the project convention). The 60 s baseline of the time study ran under heavier CPU contention than the 600 s rerun, a confound in the rerun's favour that still did not reduce the leak.

#### 6b.6 Measured runs

| run | input | windows | rounds | splits (unclean) | deadline hits | verbatim | gates in → out (ratio) | wall |
|---|---|---|---|---|---|---|---|---|
| `port_runs/six_n16b`, 2026-10-12, six stages, 8 threads | crossing output, n=16, $\|C\|=32$ | 678 → 696 | 3 (16, 2, 0 splits) | 18 (0) | 24 | 0 | 333,588 → 227,981 (0.683) | unpacking 374 s (synthesis 334, checks 40); packing 387 s; stages 1-5 2.5 min; 15 min from `SEEDS` to `delivery.recipe`, 18 min end to end |
| n8_L8 fixture, `RESTARTS=2` (parity test) | stage-3 output, n=8 | 73 → 74 | 2 (*derived*) | 1 | 0 | 0 | 34,825 packed gates | Python and Rust byte-identical |
| `port_runs/seven_n64`, 2026-10-11, pre-refinement fallback policy | crossing output, n=64 | 10,366 | — | — | 371 | 157 fallbacks | 5,153,273 → 4,766,644 (0.925) | 68 min, 8 threads |
| the 157 replayed by bisection to the end (`anf_synth`, 30 s per piece; `unpacking_sel157/final`) | same | 10,366 → 10,540 | 6 | 174 | — | 0 | → 5,201,064 | round 1: 4 min, 8 threads |
| `dressed_runs/base`, four-stage deliverable | stage-3 output, n=64 | 2,591 | — | — | 1 | 0 | 1,487,674 → 526,874 (0.354) | 255 s, 8 threads |

Every check passed on every run (per-window exact and sampled, final deliverable, equivalence before and after canonical order, zero slice). Sources: `six_n16b/unpacking/{unpacking.report,rounds.tsv}` and `six_n16b/stage6.log`; `docs/STAGE7_UNPACKING.md` §7; `docs/notes/seven_stage_20261011.md` §3-4; `docs/notes/stubborn_windows_20261012.md` §4. Both measured crossing outputs were made at `crossing.move_budget_factor = 2`, not the current default 3 (§5.4).

#### 6b.7 Code pointers

| what | where |
|---|---|
| the loop, cut choice, blob emission, container, TSVs; defaults | `src/stages/unpacking/refine.rs:397-606, 241-279, 213-232, 148-162, 644-696; :53-55` |
| per-window pipeline, `unpack_selected`, statuses, plain `windows.tsv` | `src/stages/unpacking/run.rs:1-20, 287, 54-66, 444` |
| synthesizer, parameters and defaults, statuses, Barenco macros | `src/stages/unpacking/synth.rs:1960, 124-183, 243-249, 1237` |
| local universe and spec | `src/stages/unpacking/spec.rs:27, 94, 132` |
| peephole | `src/stages/unpacking/postopt.rs:100` |
| checks | `src/stages/unpacking/verify.rs:54, 103, 139, 174` |
| binary: `--refine` argument rules, `run_refine`, report keys, staged writes | `src/programs/tdp_unpacking/mod.rs:671-693, 963-1200, 457-600, 242-312` |
| split candidates and pick, C-view scan | `src/stages/packing/repair/split.rs:14-36`; `src/stages/packing/repair/mod.rs:89, 123, 154` |
| canonical order | `src/canonicalization/gate_order.rs:64` |
| driver | `scripts/tdp_gen.sh:189, 458-469, 1140-1160`; `src/tdp/config/mod.rs:299-351, 485`; `src/tdp/config/validate.rs:428-447` |
| Python reference and parity | `scripts/windowed_delivery/waist/deliver.py` (loop `:326-380`, cuts `:236-265`, checks `:411-449`); `tests/manual/delivery_parity.sh`; `tests/integration/tdp_unpacking.rs:430-620` |
| documents | `docs/DELIVERY_STAGE.md` (the contract), `docs/STAGE7_UNPACKING.md` (the synthesizer, CLI, exit codes; §10 the loop), `docs/tdp_pipeline.md` "Stage 6", `docs/notes/stubborn_windows_20261012.md`, `docs/notes/stage45_revival_20261011.md`, `docs/notes/seven_stage_20261011.md` |


## 7. Verification, metrics and attack tooling

This section (§7) covers three families of tools:

1. **Functional checks.** Each stage must preserve the function of the circuit it was given. Every runtime check is *relative*: it compares a stage with its own input (G2 vs sandwich, G3 vs G2, packed vs G2, repaired vs packed, ANF vs gates). No step of the reference driver compares the deliverable with C(x); the managed stage 6 does: `tdp_packing` checks the zero slice of the decoded deliverable against C (4,096 inputs) and `tdp_unpacking --refine` checks the refined deliverable and `unpacked.mpmct1` against C (`check.zero_slice=ok(4096)`).
2. **The C-view.** It asks whether any value computed by a gate of the source circuit C is an exact affine function of a delivered boundary state.
3. **Linkability censuses.** They count values that can be recognised at more than one window boundary.

The checks for stages 2-3 are Rust hard assertions: a failure panics, and the binary exits non-zero. Stage 1 (sandwich vs C) has **no runtime functional check**. `construct_seeded_sandwich` asserts only parameters (`src/stages/sandwich/construct.rs:161,311-317`). Its correctness rests on unit tests such as `sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice` (`tests/unit/stages/sandwich/construct.rs:56,168`; §1.5). The packing checks are hard errors in `tdp_packing` (exit 4; floor exit 5; budget exit 6); the link census is ported (`stages::packing::census`, `link_rep512.txt`, L2/M2 in `packing.report`); the C-view engine lives in the library (`src/stages/packing/{gf2,cview}.rs`, shared by `hmap_affine`, `tdp_packing` and `tdp_unpacking --refine`). The Python prototypes remain the oracle. Stages 4 and 5 are mixer modes and inherit the mixer's checks; the unpacking sub-step's checks (per-window exact and sampled, final deliverable, new cuts, equivalence before and after canonical order, zero slice; any failure exit 4) are in §6b and `docs/STAGE7_UNPACKING.md` §5 and §10.

The tracked snapshot `scripts/windowed_delivery/` is byte-identical to the run-tree copies (refreshed in `5619b867`; it is the oracle and is never edited; comparisons run it in a path-rewritten sandbox, `tests/fixtures/stage4/oracle_sandbox.sh`, because the scripts `sys.path.insert` the hard-coded run-tree directory, e.g. `scripts/windowed_delivery/waist/link_census.py:17`, `waist/verify_equiv.py:3-4`). `waist/deliver.py`, the Python reference of the delivery loop, exists only in the snapshot. Script paths in §7 follow the document-wide convention (Overview, "Conventions"): `waist/x.py`, `W/x.py` and bare script names all mean the snapshot files, and every cited line number holds for them.

### 7.1 Shared conventions

**Gate model.** Every tool implements the same gate, `st[t] ^= (AND_i lit(c_i, p_i)) XOR comp`, with `p=1` the positive literal and `p=0` the negated one (Rust: `src/circuit/xgate.rs:159-166`; Python: for example `waist/verify_equiv.py:10-18` and `boundary_cluster.py:67-73`). An empty control list makes the AND all-ones, so with `comp=0` the gate is a NOT; with `comp=1` it would be the identity. The canonical NOT, `XGate::x_gate`, has `comp: false` (`src/circuit/xgate.rs:51-57`). Every gate is an involution, so the reversed gate list is the inverse circuit (`src/stages/preprocessing/verify.rs:66-70`). The g57 semantics and the mpmct1 format are in §1.3; the `.bounds` format and the loaders' disagreement on n are in §6a.2. `order_repair.load` uses the header width only, and `verify_equiv.py` imports that loader, so the repair equivalence check also uses header widths (`waist/order_repair.py:25-38`; `waist/verify_equiv.py:3-4`).

**Three input models.** The tools draw random inputs in three different ways. The difference matters when reading the metrics.

| model | inputs | used by |
|---|---|---|
| honest slice | random `x` on wires 0..n, every other wire 0 | `hmap_affine` (C-view and span check), `verify_zero_slice`, gen reverse check (random `a` on 0..n, run on the reversed circuits); `tdp_packing` and `tdp_unpacking` `check_zero_slice` (4,096 inputs); the refinement's C-view (`CView::leaky`, packer defaults) |
| sandwich-wide | random on sandwich wires 0..2n, band wires 2n..4n pinned to 0 | gen forward check |
| all-wire | an independent random word on every wire, band and auxiliary wires included | mixer global check, packer checks, `verify_equiv`, ANF checks, `link_census`, `boundary_cluster`, `mult_census`; `tdp_packing` `check_deliverable` (1,024 states); `tdp_unpacking` per-window sampled check (1,024) and `check.equiv_concat` / `check.equiv_canonical` (4 × 1,024, fixed public seed `param.verify_seed=20261011`) |

The all-wire model checks the whole permutation. That is stronger than delivery needs, since only the slice (x, 0ⁿ, 0²ⁿ) ↦ C(x) on wires n..2n is contractual (`docs/tdp_pipeline.md`, "Stage 2: preprocessing": on input (x, 0ⁿ, 0²ⁿ) wires n..2n−1 contain C(x)). For linkability, however, the all-wire model means that two values which agree only on the honest slice are *not* linked. Whether the all-wire model is the intended linkability adversary is an open question.

### 7.2 Correctness chain

| check | location | samples | compares | on failure |
|---|---|---|---|---|
| stage 1 | none at run time | — | (unit tests only) | — |
| gen forward | `src/stages/preprocessing/verify.rs:50-65` | 4×64 = 256, RNG `seed ^ 0xA11CE` (`:30`), sandwich-wide | G2 vs sandwich, payload half | `assert_eq!` panic |
| gen reverse | `verify.rs:71-92` | 256, honest slice | reversed G2 vs reversed sandwich, wires n..2n | panic |
| mixer per-splice | `src/engine/mixer/replacement.rs:614-679` | exhaustive (≤24 wires) or ANF proof (≤64) | DB window vs replacement | panic (regular store) |
| mixer global | `src/engine/mixer/reporting.rs:7-48` | 256, all wires | current arena vs mixer input | panic `FUNCTIONALITY BROKEN` |
| packer input / emitted (oracle) | `window_pack3.py:63-90,168-173,485-487` | 1,024, all wires, seed 20261004 | final states of G2 vs G3, then G2 vs reordered G3 | input: prints the count only; emitted: prints `OK`/`FAIL`; neither stops the script |
| repair equivalence (oracle) | `waist/verify_equiv.py:1-21` | 1,024, all wires, seed 77 | unrepaired vs repaired packed circuit | prints `MISMATCH` only |
| ANF spot checks (oracle) | `anf_compose.py:93-121,159-163` | 256 (`LANES=4`, `:25`), every 500th window | window ANF vs gate evaluation | prints `CHECK FAILED` only |
| ANF full evaluation (oracle) | `waist/anf_dump_eval.py:91-124` | 1,024, seed 1 | all-window ANF vs gate evaluation of the whole circuit | prints `match=False` only |
| packing (port) | `src/stages/packing/{certify,verify}.rs` | exact certificates; 1,024 all-wire states; exhaustive up to 16 support wires, sampled above; zero slice 4,096 | linear-extension certificate, the oracle's VERIFY, exact cancellation replay + repair certificate, both sampled equivalences, decode = window ANF, window ANF = gates, `anf_dump_eval.py`'s evaluation check, zero slice vs C | exit 4 (floor miss 5, over budget 6); no deliverable written |
| stage 4 split / stage 5 crossing | mixer `global_check` every `verify_every` moves and at the stop | 256, all wires | current arena vs mixer input | panic; the driver accepts only the stop reasons `split stage complete (stopped at the stage boundary)` / `moves budget spent` (stage 4) and `moves budget spent` (stage 5); a stop-flag stop is kept as `*.stopped.mpmct1` and never handed on |
| unpacking (port) | `src/stages/unpacking/verify.rs`; `src/programs/tdp_unpacking/mod.rs` | per window exact (recomposed ANF == blob) + sampled 1,024; `check.final_deliverable`; `check.new_cuts`; `check.equiv_concat` / `check.equiv_canonical` (4 × 1,024); `check.zero_slice` (4,096) | synthesized gates vs the delivered ANF; the refined deliverable vs the packed circuit and C; `unpacked.mpmct1` vs the deliverable before and after canonical order | exit 4, nothing renamed |
| end-to-end (built into stage 6: `tdp_packing` and `tdp_unpacking`, 4,096 inputs, exit 4; the standalone `verify_zero_slice` is manual and was run on the twelve validation runs, 512 samples) | `tests/manual/tdp/verify_zero_slice.rs:1-13,64-76`; `src/stages/packing/verify.rs` | default 200, seed 1, honest slice | C(x) vs final mpmct1 on wires n..2n | prints `FAIL`, exits 1 |

**The oracle drivers ignore failures.** `val_chain.sh` and `repair.sh` both run with `set -uo pipefail` and without `-e` (`val_chain.sh:6`, `repair.sh:5`). Neither checks exit status. If gen or the mixer panics, the later stages run on missing or stale files (`val_chain.sh:15-23`). The managed driver (`scripts/tdp_gen.sh`) runs `set -euo pipefail`, checks every stage's exit status and stop reason, and records a stage-6 failure in `delivery.failed` (`exit=<code>`, `step=packing|unpacking`).

**Stages 1-2: `verify_payload`** (§2a.7). gen calls it once, after preprocessing (`gen_sandwich_gadget/mod.rs:356-368`). It compares G2 with the sandwich and never evaluates C. The forward check randomizes both sandwich halves and pins the band to 0; the reverse check (classic guarded only) is the gadget-level mirror of `A⁻¹(a,0) = (junk, D⁻¹(a))`. Auxiliary outputs are "intentionally unconstrained" (`verify.rs:7-9`). All eight validation replicates log both lines, e.g. `[gen] verify PASSED (256 bit-sliced samples, payload wires 64..128)` and `… reverse verify PASSED … D^-1 emerges under backward evaluation` (`W/vA64a/gen.log`).

**End-to-end check (standalone binary; its check is built into stage 6).** The manual binary `verify_zero_slice <C.g57> <final.mpmct1> <n> [samples=200] [seed=1]` (`Cargo.toml:257-259`; `:208-209` at HEAD) puts x on wires 0..n with every other wire 0. It compares C(x) with wires n..2n of the final circuit and requires at most 1,024 wires (`verify_zero_slice.rs:11-13,37-43`). Its header says it closes what is "otherwise only a transitive argument" (`:6-9`), and `verify.rs:36` points to it. No driver calls the binary; `tdp_packing` (`stages::packing::verify::check_zero_slice`, 4,096 inputs) and `tdp_unpacking --refine` (`check.zero_slice`) perform the same check on every run.

**Stage 3: mixer** (§3.5, §3.7).

- **Per-splice DB verification** is on by default. `MixParams::default` sets `db_verify: true` (`src/engine/mixer/params.rs:665`), and circuit_mixer sets `db_verify: !args.no_db_verify` (`src/programs/circuit_mixer/mod.rs:576`). Only `--no-db-verify` turns it off (`src/programs/circuit_mixer/cli.rs:307-311`), and the recommended command does not pass that flag. The check works as follows (`replacement.rs:614-679`):
  1. The support is the union of all targets and controls of the outgoing window and of the replacement. Both sides are the identity outside the support, so equivalence on the support is equivalence of permutations.
  2. If the support has at most 24 wires, `rules::verify_rewrite` evaluates both sides on all 2^k assignments (`src/engine/moves/rules.rs:278-318`).
  3. If the support is wider, `polys_equivalent` composes both sides to canonical ANF and compares them (`src/stages/db_mixing/replacement.rs:107-145`). This is a proof, not a sample.
  4. When `polys_equivalent` is undecided (support above 64 wires, or the term budget is hit), the splice is declined and counted (`db_wide_skip`).
  5. A regular-store replacement that fails panics with "DB replacement verification failed".
- **Regular store only.** The run uses the regular store only (§3.1).
- **Global check.** It first asserts that the merge index matches the arena size. It then draws 4 batches × 64 lanes on *every* wire from the mixer RNG, evaluates the original input and the current arena, and panics on any difference. It runs every `verify_every` moves (default 10,000; `cli.rs:616-618`; `scheduling.rs:612-614`), at the stop-flag, eff-budget, canary, dose and moves-budget exits (`scheduling.rs:603-649`), and after the final float unless `--skip-final-float` is set (`circuit_mixer/mod.rs:901-904`). The ProfileDone exit (`scheduling.rs:476-478`) has no check of its own. The replicates ended on the moves budget (`vA64a/s3.log:1234`).
- **`--no-local-verify`** (`cli.rs:622-624`) disables only the per-move exhaustive checks inside transformations. In the classic twist these are packet invertibility (`src/engine/mixer/transformations.rs:652-659`) and the per-gate `P^-1·g·P == g'` (`:689-699`). On the recommended path (`twist-g57=true`, `vA64a/s3.log:1`) they are the g57-twist conjugation check (`:1146-1153`) and the seam checks (`:1168-1186`). Correctness of the mixer output therefore rests on the per-splice proof and the sampled global check.

**Stage 6, packing sub-step (oracle): packer.** `proxy_spans` simulates G2 and G3 on the same 1,024 all-wire inputs and returns `agree`, the number of wires `w < min(na,nb)` whose final words match (`window_pack3.py:63-90`). The log prints `input function check X/Y` (`:171-173`) with no OK/FAIL marker; Y = Y shows G3 ≡ G2 on the full permutation. After scheduling, the packer re-runs `proxy_spans` on the emitted order and prints `VERIFY: emitted function check X/Y wires (OK|FAIL)` (`:485-487`). Neither check stops the script. In the port neither check is print-only: a mismatch is exit 4.

**Stage 6, packing sub-step (oracle): repair and emission** (§6a.5, §6a.6). `verify_equiv` compares every final wire on 1,024 all-wire states (`verify_equiv.py:5-21`), once in the background and once after the fix loop (`repair.sh:15-16,27`); the background run races the fix loop's in-place rewrites (§6a.5). `anf_compose.check_window` runs on every window whose index is divisible by 500: it draws 256 fresh all-wire samples, evaluates the gate window, compares every changed wire against its ANF and requires every unchanged wire to be untouched (`anf_compose.py:93-121`). At n=64 (about 2,600-2,800 windows) this re-evaluates 6 windows; at n=128 (about 7,000-7,400 windows) it re-evaluates 15. `anf_dump_eval.py` then evaluates every window over 1,024 lanes and prints `match=True/False` (`:91-110,124`), but on the in-memory windows, never on a decode of `deliverable.bin` (§6a.6).

### 7.3 C-view: `hmap_affine --exposure`

The binary is `security_tests/heatmaps/hmap_affine.rs` (`HA`). Building it requires the `security-tools` feature (`Cargo.toml:308-311`; `:249-251` at HEAD). In this section **P** is the circuit passed as `--g` (G2, the repaired pack, or a split candidate) and |P| its gate count. The binary that produced the replicate numbers was reproduced by a rebuild at `492564f0` (§8.9). The exposure engine now lives in the library, `src/stages/packing/{gf2,cview}.rs`, byte-identical to it; `hmap_affine` keeps the CLI, `tdp_packing` uses it in-process, and `tdp_unpacking --refine` uses `CView::leaky` to choose refinement cuts.

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
- the port tests by identity: a visible non-output row in the final scan is exit 5 (`packing.allow_floor_miss` ships it as NOT recommended).
- `merge_bounds2.py:6-10`, `anf_split2.py:17-21` and `order_repair.py:140-146` instead identify outputs as "the last writer gate per wire" and act on non-output rows.

**Span check: `SPAN_INTERSECT` / `SPAN_OUTPUTS_TRIVIAL`** (`HA:392-500`). This mode runs inside `--exposure` and returns before the t\* scan. It needs snapshot 0 among the rows (`HA:427-430`). It works with three spaces:

- I = span{1, x_0..x_{n-1}}, extended with C's outputs when `SPAN_OUTPUTS_TRIVIAL ≠ 0` (`HA:437-454`);
- U = I + span{all C born values};
- W = span{1, all wires of P} at each cut.

The reported excess is `rank(W mod I) − rank(W mod U)` = dim(U∩W) − dim(I∩W) (`HA:482-483`). It counts the independent affine combinations of C values that equal affine combinations of the cut's wires, beyond what the inputs (and, if enabled, the outputs) explain. `val_chain.sh:26-27` runs it as `SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=…` on every stage-2 output with `--g-step 64` (n=64) or `256` (n=128). Stride mode includes cuts 0 and |P|, so the outputs-trivial setting is what makes the final cut score 0 rather than 64 / 128.

### 7.4 How the C-view drives the delivery stage

The repair loop is described in §6a.5. In summary, the C-view scan is run with `EXPOSURE_WITNESS` and `EXPOSURE_ALLCUTS` up to 4 times in the fix loop, where each non-output witness drives a completer/breaker slide (the **completer** is the last gate before the witness cut that writes a wire of the predictor support S; the **breaker** is the first such gate at or after it). It then runs up to 4 more times in the fallback loop, where `merge_bounds2.py` deletes exposing boundaries (a cut-lattice subset, so nothing can become newly visible; `RESULTS.md:1400-1406`) and `anf_split2.py` spends one C-view run per split decision to choose an exposure-free cut.

Two properties of the loop matter for interpreting results:

- "Before c" means the first c gates of the *current* sequence (`order_repair.py:153,156-159`), with c taken from the witness file of the pre-pass scan. Slides for earlier witnesses in the same pass move gates across markers (`:114-137`), so for later witnesses gate rank c can drift from the boundary it named.
- If the fallback loop exhausts its 4 iterations, the bounds after the last merge and split are never re-scored, and the figure printed at `repair.sh:39` predates them. This is what happened to vL128a v1 (§6a.5). (Oracle; the port rescans: identity test, exit 5.)

The 2026-10-08 patch to the run-tree `order_repair.py` (old and new breaker handling, and its effect on vL128a) is described in §6a.5.

In the unpacking sub-step the C-view also chooses every refinement cut (§6b): for each failed window [lo, hi) the packer's 32 candidates `lo + (hi−lo)·i/33`, one `CView::leaky` scan per round over the candidates of every failed window (the C-view built once on C with the packer's defaults), the clean candidate nearest the midpoint (the smaller on a tie), the midpoint when none is clean (counted in `unclean_cuts`). `check.new_cuts` certifies the refined lattice without a second full scan: a per-cut result does not depend on the lattice, so the packer's final scan of the old cuts plus the choice-time scan of the new ones equals `tdp_packing check-final` on the refined lattice (`docs/DELIVERY_STAGE.md` §4).

### 7.5 Linkability metrics

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

- `values`, the number of distinct values with at least one run. For proxy, this is the delivered stage-2 exposure ("proxy values (link census)", §6a.1). `STAGE2_LINKABILITY.md`'s "exposed" column is this count, not the packer's;
- `L2`, the number of values with cluster size ≥ 2 (**linkable values**);
- `M2`, the total runs over those values (**sightings**);
- a size histogram in buckets 1, 2, 3-4, 5-9, 10-19, 20-49, 50-99, 100+. **Anchors** are the proxy 100+ bucket.

It also breaks linkable proxy values down by stage-2 birth multiplicity, both by count and by mass (`:75-96`). It does *not* append the output cut m (`:23`), so gates after the last boundary are never simulated. `boundary_cluster.py:54` does append it.

**Other censuses.**

- **`boundary_cluster.py G2 packed bounds`** is the earlier equality-cluster attack. It reports median/p99/max cluster size and the proxy precision `p/(p+f)` at thresholds 2, 3, 5, 6, 10 and 20 (`:105-121`). On born values, "cluster size ≥ 6 is pure proxy (froth never exceeds 5)" (`RESULTS.md:302-317`).
- **`waist/mult_census.py G2 label`** is the stage-2 rebirth census. It prints `born`, `distinct`, `repeats = (born−distinct)/born`, the top multiplicity, the count and cover of values with multiplicity ≥ 65, and the count with multiplicity ≥ 257 (`:1-25`).
- **The packer's own `RESULT` line** (`window_pack3.py:488-503`) counts carried G2 values with a boundary in (birth, next write], plus values forced at the output. It classifies wires with thresholds hard-coded to 64/128 (`:49-50`).
- **Leak-investigation diagnostics** in `waist/`: `trace_witness.py`, `leak_mult.py`, `anf_at_cut.py`, `seed_coincidence.py`, `singleton_runs.py`, `liveness.py` and `link_merge.py`. `link_merge.py` deletes boundaries in a subset-safe way: deleting a boundary never increases any value's run count.

### 7.6 Current measurements

**Threat model.** In the default mode the adversary holds the published circuit `unpacking/unpacked.mpmct1` and nothing else: it can cut it anywhere, compute the function of any segment, compose segments and resynthesize them into circuits of its own; it is never handed the window lattice or the ANFs and does not see how a window was spelled before resynthesis. The measurements that model it are the all-cuts census of the resynthesized circuit against C and the uniform re-cut-and-resynthesize experiments ("Measurements on the six-stage chain" below): merge consecutive windows into a width W, resynthesize in place, and census the materialised values against C; they re-cut the circuit uniformly and never use the delivered lattice. The packing metrics (C-view, proxy census, linkability; `STAGE2_LINKABILITY.md` §2) are evaluated on the windowed representation at the cuts of its lattice, used exhaustively, with function equality sampled (1,024 inputs for linkability and 6,144 for the C-view); the lattice of the refined deliverable is `unpacking/bounds.final` (the packer's lattice plus one cut per window that failed to resynthesize, recursively). Under `delivery.anf_only` that windowed representation is the published object and the older framing applies: the adversary holds the window ANFs, hence the wire state at every window boundary on inputs of its choice, can compose adjacent windows and resynthesize any window into a circuit of its own, and cannot see how the function was spelled inside a window; a window has no interior cuts.

**No affine relation to C beyond the outputs was found in stage 2, at the cuts scanned.** The span check was run on 29 stage-2 variants, with outputs *not* counted trivial. At n=64 it scanned every 16th cut (about 41-60k cuts); at n=128 every 128th cut (about 15-17k cuts).

- The excess was 0 at every scanned interior cut; cuts between the stride points were not tested. It became nonzero only in the final ~3.3k / ~6.6k gates, ramping to exactly 64 / 128 as the outputs emerged.
- A dense tail re-scan with outputs counted trivial covered a subset: A, B, chain and chain+K64 at n=64; A, B and chain at n=128. It found 0 at all 4,001 / 8,001 tail cuts.
- dim U ≈ 1,050 / 3,100, which is far below 6,144 samples, so the result is not sampling-limited.
- Positive control: C against itself.

Sources: `RESULTS.md:1665-1681`. The mixer's role was localized on one artifact, rs64 (K=64 retirement, §2a.5, a lever since dropped for creating interior C leaks, `STAGE2_LINKABILITY.md:141`). There, stage 2 scanned at every 16th cut showed 0 visible values. The mixed circuit scanned at every 32nd prefix cut showed 11 interior values, including 7 of the 8 delivered leaks with identical predictor supports. Those C leaks were therefore made by the mixer (`RESULTS.md:1373-1376`). This localization was not repeated on the recommended configuration.

**Validation replicates.** These are the outputs on disk in `W/v{A,L}{64,128}{a,b}/`, recorded in `docs/STAGE2_LINKABILITY.md` §5. The following passed in all eight runs:

- gen forward and reverse verify;
- the packer's input and emitted checks (256/256 or 512/512);
- `verify_equiv` EQUIVALENT, twice;
- 0 windows over 65,536 monomials;
- 0 spot-check failures;
- span excess 0 at every scanned cut, outputs counted trivial: stride 64 / 256 including cuts 0 and |P|, 12,145-12,176 cuts at n=64 and 8,681-8,733 at n=128; dim U = 1,041-1,051 at n=64 and 3,104-3,113 at n=128. Each `span_s2.txt` is one line, for example vL64a: `[span] dim span(C values,1) = 1049 over 6144 samples; cuts by intersection dim: {0: 12176}; nonzero from cut None`. The span scans are strided like the study's; no dense tail scan was run on the replicates.

`verify_zero_slice` was run afterwards on all twelve validation runs (512 random honest-slice inputs, fresh seeds; `docs/STAGE2_LINKABILITY.md` §5), and the zero-slice check is built into every `tdp_packing run` and `tdp_unpacking run --refine` since (4,096 inputs). Cancellation removed about 6.6-7.3% of gates; for example vA128a went from 4,757,723 to 4,437,893. Pack, repair and serialization timings are in §8.9; serialized sizes per configuration in `docs/STAGE2_LINKABILITY.md` §5.

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

vL128a's first repair (v1, last scan at 129) gave L2 / M2 / froth L2 = 16,107 / 187,247 / 5,578. Its `deliverable.bin`, `deliver.txt` and timings were written at 01:35-01:36, before the patched rerun (08:24-08:44), describe the v1 artifact, and were regenerated on 2026-10-08 (§8.9).

The stage-2 rebirth census (`mult_s2.txt`, §2b.7) is consistent across seeds: config A has repeats 19.8-19.9% and 329-335 (n=64) / 968-973 (n=128) values with multiplicity ≥ 257; config L8 has repeats 8.6-8.7%, none at ≥ 257, and a top multiplicity of 186-214.

**L8 against A.** This is the single reference table of L8-versus-A deltas. Each cell is the mean of the two replicates per configuration; sizes are the per-run `deliverable.bin` sizes (ranges in `docs/STAGE2_LINKABILITY.md` §5).

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

The n=128 L8 row mixes the patched vL128a rerun with vL128b, and its `.bin` mean uses the size of vL128a's pre-rerun (v1) file, 103.2 MB (§8.9); the re-serialized file's size has not been folded in. The doc-level summary in `STAGE2_LINKABILITY.md:120-135` predates the replicates. It uses the earlier finals (`RESULTS.md:1742-1751`) and gives L2 −13%, M2 −21% / −26%, "exposed" (= proxy values) −24% / −37%, anchors −39% / −28% and size +18% / 0% (n=64 / n=128). Those are earlier-final figures and should not be quoted as replicate results.

**Measurements on the six-stage chain (the resynthesis adversary).** The raw crossing output (stage 5, n=64 L8, `s45_runs/base64`; n=128, `s45_runs/e2e128`) holds no C value at any cut: 0 interior, 0 early outputs at both sizes. Merged windows resynthesized in place expose them anyway (`docs/notes/stage45_revival_20261011.md` §3-4; `docs/notes/stubborn_windows_20261012.md` §1-2):

| circuit, lattice | interior C values materialised | repeated | outputs early |
|---|---|---|---|
| stage-3 output, delivered lattice (~590 gates per window), n=64 | 297 | 118 / 92 | 56 |
| stage-3 output, uniform W=2,000, n=64 | 434 | 225 / 198 | 61 |
| **stage-5 output, uniform W=2,000, n=64** | **244** (of 936) | 88 / 56 | 45 of 64 |
| stage-5 output, uniform W=2,000, n=64, 600 s on the 164 deadline windows | 241 (the 17 rescued windows add 0 new values) | | 45 of 64 |
| stage-3 output, delivered lattice (~960 gates), interleaving control arm, n=128 | 417 | | 60 |
| stage-3 output, uniform W=2,000, n=128 | 1,270 | 635 / 583 | 124 |
| **stage-5 output, uniform W=2,000, n=128** | **711** (of 2,872) | 255 / 163 | 94 of 128 |
| the full unpacked n=64 crossing output at its own (delivered) lattice | 5 | | |
| the four by-the-book factorings of the 157 stubborn and 60 ordinary windows | 0 | | |

Canonical order changes these counts by at most 1.7% (244 → 242, 241 → 239, 434 → 434, 297 → 302). Stages 4 and 5 therefore do not close the leak: at equal gate width the stage-5 proxies are ~55% of the same run's stage-3 proxies, and ~80% against the delivered stage-3 lattice, the window-fraction effect of crossing's 3.4× stretch. The policy decided on 2026-10-12 is to accept any synthesized size and bisect the windows that do not resynthesize (`docs/DELIVERY_STAGE.md`; §6b). Caveats of the verification pass: the zero-leak results are measured at the delivered lattice only; only the verbatim census was applied to synthesized circuits; equivalence is sampled (8,192 states).

### 7.7 Limitations and requirements for the Rust port

**Status (2026-10-09).** The port (`docs/STAGE4_RUST_DESIGN.md`) addresses the
end-to-end check (zero slice of the decoded deliverable against C on every
run), hard failures, re-scoring (a final scan, identity test), serialization
(decode round trip, container header) and memory (a streamed C-view with
checkpoints: 0.4-0.5 GB at n=128). Attack scope, witness cut drift, duplicate
wires in witnesses and the input models are unchanged (the port reproduces
the oracle's repair exactly).

- **Attack scope.** Only exact affine predictors and exact value equality are measured. Low-degree nonlinear predictors were not run on the final artifacts (`STAGE2_LINKABILITY.md:178-180`). The C-view is also strictly per-cut: it does not combine wires from different boundaries.
- **End-to-end correctness.** Add an honest-slice check of the deliverable against C: input (x, 0, 0, 0), compare wires n..2n with C(x). Today stage 1 has no runtime check and every other check is relative. At minimum the driver should run `verify_zero_slice` on `s3.mpmct1` and `rep512.mpmct1`. The port should run the equivalent on the ANF-decoded deliverable. Done: the port checks the decoded deliverable's zero slice against C on every run (4,096 inputs), and the unpacking sub-step repeats it on the refined deliverable and on `unpacked.mpmct1`.
- **Hard failures.** Make every Python check a hard error. `FAIL`, `MISMATCH`, `CHECK FAILED` and `match=False` only print today, and neither `val_chain.sh` nor `repair.sh` greps for them. The drivers also ignore exit status: there is no `set -e` and no status checks, so a panicking gen or mixer does not halt `val_chain.sh`. If `hmap_affine` fails inside `repair.sh` (e.g. OOM), `vis` is empty and `[ "$vis" -le "$N" ]` errors, which counts as false. `order_repair fix` and `merge_bounds2` then run on the stale witness and all-cuts files of the previous scan (`repair.sh:17-25,29-30`). Abort on any non-zero exit, and on a missing or non-numeric C-view result.
- **Re-scoring.** Re-score the C-view after the final fallback step. Test floor attainment by identity (no non-output row with t\* ≥ 0) rather than by count.
- **Witness cut drift.** `run_fix` locates the completer and breaker by gate rank `cut` from the pre-pass scan (`order_repair.py:153,156-159`). Locate each boundary by its marker in the current sequence, or recompute cuts after each slide.
- **Duplicate wires in witnesses.** Witness supports list representative wires only, so the completer/breaker search ignores duplicate copies of a value. Whether a relation can survive the slides through a duplicate copy is untested.
- **Serialization.** Add a decode round-trip test for `deliverable.bin`, and decide on a global header (n, window count). Done: container v1 header (`LMTDPANF`, source/state wires, window count, body length, CRCs) and a validating decoder with read-back on every run.
- **Input models.** Unify the input models (honest slice, sandwich-wide, all-wire), and decide whether `link_census` should count the output state.
- **Memory.** The C-view stores `batches × cuts × wires` u64 words. At n=128 with 7,058 cuts the snapshots of P alone are ≈ 2.8 GB. The port's streamed C-view with checkpoints peaks at 0.4-0.5 GB at n=128; `tdp_packing run` peaks at 0.9-1.1 GB (n=64) and 1.9-2.1 GB (n=128) on the stage-3 input.

### 7.8 Stale documentation (the code is the authority)

- **[stale] packer sample counts.** `docs/ANF_WINDOW_DELIVERY.md:44` gives 256 samples for packer re-verification, and `:48` and `:224` give 4,096 fingerprint inputs/lanes (the `ANF_WINDOW_DELIVERY.md` and `STAGE2_LINKABILITY.md` lines in this list are as of the 2026-10-12 working tree, after both documents gained their status notes). The code uses 1,024 for both (`window_pack3.py:22-23`).
- **[stale] "every window re-verified".** `ANF_WINDOW_DELIVERY.md:114-118`, `:127` and `:443` say every window is re-verified before serialization (its §1.2 carries a bracketed 2026-10-09 correction at `:133-138`). `anf_compose` spot-checks every 500th window with 256 samples; only `anf_dump_eval`'s full-state evaluation covers every window.
- **[stale] ANF budget.** The 16,384-monomial budget in the same doc (`:131,223`) belongs to the earlier G=128 pipeline; `repair.sh` uses 65,536. Its parameter section §1.4 (`:201-239`) is from the G=128 era and is marked superseded (`:203-206`).
- **[stale] "exactly (no sampling)".** `STAGE2_LINKABILITY.md:38-39` says the metrics are computed "exactly (no sampling)". The cut lattice is exact, but value equality is sampled.
- **[stale, fixed in this revision] `README.md` "Outputs and continuation"** listed "4. Packing" as the last step with `packing/deliverable.tdp4` as the shipped file, `stage4.log`, `run.stop_after_stage` default `4`, "the historic stages 5-6 are rejected", "`[packing]`: stage-4 parameters" and "the three stage executables"; the driver has six stages, `stage4.log`..`stage6.log`, default stop 6 and four binaries.
- **[superseded numbering] `docs/STAGE4_RUST_DESIGN.md`** keeps the four-stage numbers; its header maps config v2 / recipe v10 / SEEDS format 2 / `stage4.failed` / `packing.recipe` to config v4 / recipe v12 / SEEDS format 3 / `delivery.failed` (`step=packing`) / `delivery.recipe`.
- **[superseded as a stage] `docs/STAGE7_UNPACKING.md`**: its verbatim fallback (§2.5, `status=fallback_used`, `--no-fallback`) is replaced by the refinement loop; the synthesizer, checks and exit codes it describes are current.
- **[superseded] `docs/notes/seven_stage_20261011.md`**: config v3, recipe v11, `--run-skip-stages 4,5,7`, `packing.recipe`/`unpacking.recipe`, `packing.failed`/`unpacking.failed` are the 2026-10-11 driver's; on 2026-10-12 stages 6 and 7 became one delivery stage.
- **[corrected 2026-10-12] `docs/DELIVERY_STAGE.md` §2** listed an "unclean cuts" column in `rounds.tsv` that the Rust writer does not emit (`src/stages/unpacking/refine.rs:685-687`: `round pending converged failed split seconds`) and called `bounds.final` `rep<G>.bounds.final`; the contract document now matches the writer.
- **[fixed 2026-10-12] `src/tdp/config/mod.rs:357-358` and `scripts/tdp_gen.sh:1002-1005`** (working tree; `:355-358` and `:999-1001` at `83e44ec5`) said the stage-5 move budget is 2 × target; both now state the default 3 (§5.7).
- **[fixed 2026-10-12] `scripts/tdp_gen.sh:112-113`** (usage text, working tree; `:110` at `83e44ec5`) said `--threads T` pins `RAYON_NUM_THREADS` and `tdp_packing --threads`; it now names the `--threads` of both `tdp_packing` and `tdp_unpacking`, which is where the flag goes (`THREADFLAGS`, `:1113`, `:1118`, `:1143`; `run.threads` in §6b.3).
- **[fixed 2026-10-12] `src/programs/tdp_unpacking/mod.rs:1-3`** (module doc, working tree) called the binary "the stage-7 production binary"; it now names it the production binary of the unpacking sub-step of stage 6, run by the driver as `run --refine` (§6b, `docs/DELIVERY_STAGE.md`), and documents both the plain run and `--refine`.

### 7.9 Auxiliary statistical tests (cited in design rationale)

The stage-2 and stage-3 design justifications cite several statistical tests that are not part of the pass criteria and were not run on the validation replicates. Each is defined below with its tool. A "safe" value is what a construction with no detectable leak would show.

| test | definition | tool | safe value |
|---|---|---|---|
| **φ** (phi) | Pearson correlation of two binary sample vectors a, b over n samples: `(n·#(a∧b) − #a·#b) / √(#a(n−#a)·#b(n−#b))`, clamped to [−1, 1]. It is undefined (skipped) when either vector lacks a minimum number of 0s and 1s | `phi`, `src/stages/db_mixing/leakage_repair/detect.rs:681-696` | ≈ 0 (\|φ\| near the 1/√n noise floor). The EM comments' φ 0.13-0.26 (§2b.3, §2b.5, §2b.8) are leaks |
| **gauntlet w1 / w2 / w3** | Attack A4 of the gadget gauntlet, a bounded correlation scan of each target bit against trace features: w1 = every single feature; w2 = pairs on a capped, strided feature subset, combined by xor/and/or/andnot; w3 = triples (xor3/and3/or3/(a∧b)⊕c/(a∨b)⊕c). A target is flagged when its best ±1 covariance exceeds max(the NULL noise target's best, 6σ), with σ = 1/√samples. The other attacks are A1 (direct wire match), A2 (state-affine) and A3 (trace-affine) | `security_tests/gauntlet/gauntlet_audit.rs:1-21,1045-1075`, driven by `security_tests/gauntlet/gauntlet.py`; `--features security-tools` (`Cargo.toml:323-326`; `:264-266` at HEAD) | 0 flags on every arm except the unprotected `none` positive control, which must flag (`gauntlet.py:42-51`) |
| **fire_corr** | Fire-block correlation for the blinded-V5 construction, cited in EM's comment on balanced masks (EM:118-128) | Not in this tree. `docs/DRIP_RESHUFFLE_ARCHIVE.md:79,113` lists it as missing, under `red_team_tests/bin/leakage/` on another branch | cannot be re-run here; treat the EM statement as historical |
| **qc_scan "hot"** | A segment of the scanned circuit is hot if it nearly determines a sampled reference feature: \|φ\| ≥ 0.98 in both sample partitions, or an exact affine train fit validating at 100% on held-out samples. The tool prints hot counts per circuit against a fixed reference | `db_gen/analysis/qc_scan.rs:1-19` (plain release build, `Cargo.toml:123-125`; `:137-138` at HEAD); threshold `min_abs_correlation: 0.98` (`detect.rs:34-35,54`) | 0 hot. Reference against itself: 64 hot; independent stage-2 instance: 0 of 4,096 (`RESULTS.md:161-167`) |
| **heatmap ridge depth, null ridge** | From `hmap_affine` heatmap mode (no `--exposure`): H(i,j) = held-out GF(2)-affine reconstruction error of C's prefix-i state from the scanned circuit's prefix-j wires, where 0 = recoverable and 0.5 = hidden (`HA:1-12`; outputs `<out>.bin` and `<out>.meta.json`, `HA:915-917`). With R = 0.5 − H, **depth** = mean over rows of R(ridge cell) − median_j R(i,j). **rho** = Spearman correlation of row index and ridge column. **perm z** = z-score of rho against shuffled ridge columns. A **null ridge** has depth ≈ 0 and perm z ≈ 0 | `security_tests/heatmaps/plot_hmap_ridge.py:1-33,80-82` | depth ≈ 0, perm z ≈ 0. The independence check gave depth 0.0061 and perm z 0.0 (`RESULTS.md:161-167`); mixed circuits sat at 0.29-0.31 (§3.8) |
| **touch AUC** | Per-wire features (touch count, mean position, target fraction) are compared between two wire classes (top vs bottom half; data `0..n` vs band) by the rank statistic AUC = P(x > y) + ½P(x = y). The "touch AUC" is the `touches=` entry, and "line 14" of a report is `AUC data vs band` | `RD/wire_balance.py:1-17,41-55,154-163` (`wire_balance.py circuit.mpmct1 [label] [bins]`); reports `W/s12_check/wb_*.txt` | 0.5 (indistinguishable). Both 0.000 and 1.000 mean perfectly separable |

None of these tests has been run on the recommended configuration's delivered artifact.


## 8. Running the pipeline end to end

This section (§8) explains how to go from nothing to a published circuit: what to install, which seeds to draw and where they go, the exact command for each stage, how to check that a run passed, every file a run leaves behind, and what the eight validation replicates measured. In this section, "gen `mod.rs`" is `src/programs/gen_sandwich_gadget/mod.rs` and "mixer `mod.rs`" / `cli.rs` are in `src/programs/circuit_mixer/`.

A run turns a fresh secret circuit C into the published circuit `unpacking/unpacked.mpmct1` (resynthesized from the internal refined deliverable `unpacking/deliverable.tdp4`) in four programs over six stages; every stage after 1 is optional (`run.skip_stages`), and `delivery.anf_only` ends stage 6 after `packing/deliverable.tdp4`, which is then the published object:

- **Stages 1 (sandwich) and 2 (preprocessing)** run as one Rust process, `gen_sandwich_gadget`.
- **Stage 3 (mixing)** runs as one Rust process, `circuit_mixer --tdp --db-mixing`, against the regular frozen store.
- **Stage 4 (splitting)** is `circuit_mixer --split --split-stop` (§4), and **stage 5 (the crossing walk)** is `circuit_mixer --resume split.state --target-size …` (§5); both are mixer modes.
- **Stage 6 (delivery), packing sub-step** *(superseded 2026-10-09: ported to Rust as `tdp_packing run`, which the managed driver runs; the prototypes below are its oracle)* existed only as Python/bash prototypes: `window_pack3.py`, then `repair.sh`, then `anf_dump_eval.py`. The snapshot in `scripts/windowed_delivery/` (README: the reference oracle for the Rust port, `scripts/windowed_delivery/README.md:1-8`) was first taken at commit `7163e457` (2026-10-07 23:21), before the leak-slide fix to `W/order_repair.py:158-171` (2026-10-08 08:24), and without the reference driver `val_chain.sh`. Both gaps are closed: commit `5619b867` refreshed it, and on `main` (2026-10-12) `waist/order_repair.py`, `repair.sh`, `merge_bounds2.py`, `anf_split2.py`, `verify_equiv.py`, `anf_dump_eval.py`, `link_census.py`, `window_pack3.py` and `anf_compose.py` are byte-identical to the live copies in `W/` and `RD/` (checked with `cmp`), `waist/order_repair.py` has the patched slide (§6a.5), and `waist/val_chain.sh` is tracked (it differs from the run-tree copy only by dropping the inert `--p-comp 1.0 --p-any 0.1` on line 21). The tracked snapshot, which CI runs against `tdp_packing` (`tests/stage4/differential.sh`), is the oracle.
- **Stage 6, unpacking sub-step**: `tdp_unpacking run --refine` (Rust, 2026-10-11/12; Python reference `scripts/windowed_delivery/waist/deliver.py`), §6b.

The reference end-to-end driver is `W/val_chain.sh` (`val_chain.sh:1-35`; a self-contained copy of its body is in the Overview). Every validated number in this section comes from it. *Superseded (2026-10-09):* the managed driver, `scripts/tdp_gen.sh` together with `src/tdp/`, used to implement the earlier six-step pipeline and could not express the recommended configuration (the managed-driver gap analysis, closed 2026-10-09; `docs/local_mixing_history.md`, October 8, 2026). It now runs six stages (config v4, recipe v12, SEEDS format 3) and defaults to config L8; `--run-skip-stages 4,5 --delivery-anf-only` (`run.skip_stages = [4, 5]`, `delivery.anf_only = true`) reproduces `val_chain.sh` from the same seeds, and its full default chain ends with the published circuit `unpacking/unpacked.mpmct1` (beside the internal `unpacking/deliverable.tdp4`) (§8.10; `docs/tdp_pipeline.md`).

### 8.1 Prerequisites

**Toolchain.** The repository pins its Rust toolchain in `rust-toolchain.toml`: channel `1.97.1`, profile `minimal`, components `rustfmt` and `clippy`. `cargo` picks this up automatically through rustup. The Python scripts use the standard library only.

**Binaries.**

- `gen_sandwich_gadget`, `circuit_mixer`, `tdp_packing` and `tdp_unpacking` come from a plain `cargo build --release --locked` (`Cargo.toml:83-101` at HEAD); the managed driver builds and fingerprints all four (`run.build_binaries`); a standalone `tdp_gen.sh` needs them in `TDP_BIN_DIR` (`tdp_unpacking` not under `--delivery-anf-only`).
- Only the oracle's C-view scans need `target/release/hmap_affine`; the port's C-view is in-process. That binary is declared with `required-features = ["security-tools"]` (`Cargo.toml:308-311` at `ea2e516f`; `:249-251` at HEAD), so a plain release build does not rebuild it. Build it explicitly with `--features security-tools --bin hmap_affine`. A stale `hmap_affine` has been seen to silently omit the `EXPOSURE_WITNESS` output that the oracle's repair step depends on. The managed driver builds `hmap_affine` for the report-only stage-2 span check (`stage2_span.{tsv,txt}`); a standalone `tdp_gen.sh` skips that check when `$TDP_BIN_DIR/hmap_affine` is absent.

**Frozen store.**

- `FROZEN_DB_DIR` must point at `frozen_db/frozen_m1_m11`. That directory holds 256 `shard_XX.frz` files plus `tables.bin` and `probes.bin`, about 298 GiB in total. The environment contract is in `src/database/frozen.rs:4-8`, and `src/database/environment.rs:25-29` requires the variable to be set.

**Obtaining the store.** Stage 3 cannot be reproduced without it.

- **Download.** The local copy was fetched by `frozen_db/fetch_frozen_db.sh`. That script is untracked, because the repository's root `.gitignore` rule `/*` excludes `frozen_db/` (`.gitignore:3`). It downloads the 258 store files (`tables.bin`, `probes.bin`, `shard_00.frz` … `shard_ff.frz`) from the project's object store and resumes on restart. It checks each file against the server's MD5 ETag and records the verified digest in `<file>.ok`. It does not fetch `filters.bin`.
- **Identity check.** A correct copy has 258 `*.ok` files, and every file's `md5 -q` equals its `.ok`. In particular, `tables.bin` is 1,915,737 bytes with MD5 `ba4b7ace3027c59c9f3ef2d0b754325b`, and `probes.bin` is 8,380,071 bytes with MD5 `30e525ec0a8e59b0eaef024e08b31b6b` (`frozen_db/frozen_m1_m11/*.ok`). The reader itself checks only the shard count and total entry count, which is not a content digest (`docs/frozen_database.md:194-196`).
- **Rebuilding from scratch.** See `docs/frozen_database.md:223-260`. Build the offline tools with `cargo build --release --locked --features db-tools` (`:228`). Generate regular tables with `db build-regular -m N`, which extends the preceding gate count, or with `db combine-regular` (`:232-239`). Merge with `merge_rocks_parallel`, run `db export-lmdb`, then run `frozen_from_lmdb tables|write|validate` (`:241-249`).
- **Filters.** To make `FROZEN_FILTER=1` meaningful, build `filters.bin` with `frozen_filters_build from-frozen DIR` as part of a fresh build (`:250`). For an existing store, run `bash db_gen/maintenance/build_filters.sh FROZEN_DIR`, which refuses to overwrite and validates key membership, and check it with `check_filters.sh FROZEN_DIR [EXPECTED_SHA256]` (`:203-213`). Filters cost about 25.5 GB of RAM per process (`frozen.rs:7`).
- **Coverage (m range).** *Inferred* from the name and the build command: the store holds regular tables built for gate counts m = 1..11. The construction bounds are not recorded with the store. These are `--min_n`/`--max_n`, which restrict the wire counts used (`:236-238`), and whether each length was generated completely. the archived write-up `CURATED_DB_V3_SUMMARY.md:14` (removed from the tree in `8070f5a0`; branch `ran/archive/curated-and-postmix`, see `docs/CURATED_AND_POSTMIX_ARCHIVE.md`) calls it "the complete g57 replacement corpus through 11 gates", while `docs/frozen_database.md:84-86` warns that the directory name does not establish exhaustive coverage. Treat coverage as undocumented (§3.10).
- The local copy has no `filters.bin`. So `FROZEN_FILTER=1`, which would otherwise load about 25.5 GB of filters (`frozen.rs:662-690`), does nothing on this machine. Every `s3.log` prints `FROZEN_FILTER=1 but no filters.bin ...; running unfiltered`.
- `FROZEN_CURATED_*` are ignored with a WARNING at HEAD (mixer `mod.rs:203-221`), and the managed driver refuses the variables when inherited. (At `ea2e516f` `FROZEN_CURATED_DIR` had to stay unset, §3.1: `--tdp` turned curated lookups on unless `--curated` was given, mixer `mod.rs:94-109`; with the variable unset the mixer fell back to the regular store with a WARNING, `mod.rs:217-229`, the log header still showed `TDP profile ON: curated=true` and the later lines `curated=false` and `value conventions: regular=native, curated=-`; passing `--curated` explicitly turned the missing store into a hard error, `mod.rs:218-222`.)

**Python.** The oracle chain and `waist/deliver.py` need `python3` with the standard library only (the oracle must run under an interpreter that passes the CPython 3.9 set-order self-test, `tests/fixtures/stage4/pyset_selftest.py`). The managed driver uses `python3 -I` for seed drawing, number parsing and the HMAC markers; the stage binaries need no Python.

**Memory and concurrency.** No peak RSS was recorded for any stage. The figures below are *derived* from the code.

- **Mixer.** It keeps the 256 shard offset tables resident, (2^20+1) × 5 B ≈ 5.2 MB per shard, ≈ 1.3 GB per process (`frozen.rs:373-379,636-660`). On top of that come the exact caches, by default `LOOKUP_CACHE_MB` 512 + `XPOLY_CANON_CACHE_MB` 1024 (§3.1), and the circuit arena of up to ~4.8M gates. The store is not memory-mapped: buckets are read with `pread` through the OS page cache, which concurrent mixers share. `filters.bin` would add about 25.5 GB per process.
- **`hmap_affine`.** It stores `batches × cuts × wires` u64 words (§7.7). That is ≈ 2.8 GB for an n=128 repair scan (7,058 cuts), ≈ 3.4 GB for the n=128 stage-2 span check (8,733 cuts × 512 wires) and ≈ 2.4 GB at n=64 (12,176 cuts × 256 wires). In `val_chain.sh` the span check runs in the background while the packer and repair run, so two scans can overlap. The port streams the C-view with checkpoints: 0.4-0.5 GB at n=128; `tdp_packing run` peaks at 0.9-1.1 GB (n=64) / 1.9-2.1 GB (n=128) on the stage-3 input; `tdp_unpacking` on the n=64 four-stage deliverable used 271 MB RSS (`docs/STAGE7_UNPACKING.md` §7).
- **Validated concurrency.** All eight replicates (4 at n=64, 4 at n=128) ran together on an 18-core, 128 GiB host (§8.9). Higher concurrency is untested. Treat 8 jobs per 128 GiB as the recommended maximum. The six-stage chain adds: stage 4 70 s (n=64) / 8.5 min (n=128); stage 5 7 min at the 2× budget (n=64; 46 min at the archive's 6×) / 6.3 h at 6× on shared CPU (n=128); packing of the stage-5 output 2.2 h (n=64, ~4× the windows) / 7.7 h (n=128); unpacking 68 min at n=64 (8 threads, 8.9 CPU-hours). Concurrency of six-stage runs is untested.
- **No `set -e` in the oracle drivers.** Neither `val_chain.sh` nor `repair.sh` uses it, so an out-of-memory kill does not stop the run. Later steps then run on missing or stale files, which is why the §8.7 checks must all be read. The managed driver runs `set -euo pipefail` and stops at the first failed stage (stage 6: `delivery.failed`).

**Hard-coded locations.** The packing scripts assume the run directory is `W/<TAG>` under `/Users/rancanetti/code/local_mixing`. The absolute paths are fixed in `repair.sh:6-9`, `val_chain.sh:7-8`, `anf_split2.py:7,10-12`, `verify_equiv.py:3`, `anf_dump_eval.py:5`, `link_census.py:17` and `mult_census.py:3` (run by `val_chain.sh:24`). Two more scripts are run by a path relative to a working directory that the caller sets with `cd`: `repair.sh:35-36` runs `anf_compose.py` after `cd $ROOT/runs/ncdb_ab_20261001`, and `val_chain.sh:28-29` runs `window_pack3.py` the same way. The oracle is never edited; every comparison runs it in a path-rewritten sandbox (`tests/fixtures/stage4/oracle_sandbox.sh`; `tests/manual/stage4/reproduce_val_chain.sh` for `val_chain.sh`).

### 8.2 Seeds: always random, always recorded, always secret

**Why the seed is secret.** `prepare_source` calls `fastrand::seed(seed)` and generates C from it (`src/stages/sandwich/construct.rs:64-70`; gen `mod.rs:196`). So `(n, m_C, seed)` regenerates C. With all three generator seeds and the logged parameters (gen.log line 1 records n, |C|, |D|, s and slice_gates; `seed.txt` records the `EMBEDDED_MASKING_*` configuration), the whole gadget is reproduced bit for bit (gen `mod.rs:209-211`). In `val_chain.sh` all three seeds are the same value, so the one recorded gen seed is enough. The streams derived from the three seeds are listed in §1.6 and §2a.2. The positional defaults are `seed=1` and `gadget_seed = sandwich_seed = seed` (gen `mod.rs:113-115`). Never rely on them.

**What the reference driver does.**

- `val_chain.sh:11` draws `secrets.randbelow(2**31)` and passes that one value as all three generator seeds (`:16`).
- `val_chain.sh:12` draws a separate mixer seed, `secrets.randbelow(2**40)`.
- Both seeds and the configuration line are written to `R/seed.txt` (`:14`). Redacted, the format is three lines:
  - `<TAG> gen seed: <n>`
  - `mix seed: <n>`
  - `config L8: PREMASK SEED_AVOID_C1 SHUFFLING=8 RETURN_HOME=0 EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8`
- `circuit_mixer --seed` defaults to 0 (`cli.rs:632-633`), so the mixer seed must always be passed.

**The managed driver** draws five independent seeds (`gen_seed`, `mix_seed`, `split_seed`, `cross_seed`, `unpack_seed` = the delivery seed) uniformly from [0, 2^63−16] with the OS CSPRNG, plus a 256-bit `run_key`, into the private `$RUN/SEEDS` (format 3, mode 0600) before anything else is written. `gen_seed` is all three gen seeds; `cross_seed` is used only when stage 5 starts from a file (a resume of `split.state` continues the split RNG); `unpack_seed` feeds `tdp_unpacking --refine`. A format-2 file (gen, mix) is upgraded in place on a standalone resume. Explicit seeds (`calibration.seed_file`: `gen_seed=`, `mix_seed=`, optional `split_seed=`, `cross_seed=`, `unpack_seed=`) are for calibration only. See §8.12 and `docs/tdp_pipeline.md`, "Seeds and calibration".

**Gaps to fix before production** (also §1.8, issues 1-2).

1. **31 bits is too few.** In `val_chain.sh`, C is a deterministic function of a 31-bit value. A wider draw still cannot give C more than 64 bits of entropy, because fastrand's state is 64 bits. [Open; the manifest records `seed_bits=63` as a stopgap; the production route is `source.path` with a CSPRNG-generated C until the 256-bit decision (design §13.1).]
2. **File modes are world-readable.** `seed.txt` and `tdp.mpmct1.source_c.g57` (C itself) are written with mode 0644. [Managed: `SEEDS` is 0600 under `umask 077`; the mode of `tdp.mpmct1.source_c.g57` in a managed run has not been confirmed for this revision.]
3. **The logs record the seeds.** `gen.log` line 1 records all three generator seeds (gen `mod.rs:185`), and `s3.log` line 4 records the mixer seed. Both logs are therefore as secret as `seed.txt`. [Managed: fixed; gen, the mixer and `tdp_unpacking` print `(file)` in their banners; no seed value appears in `tdp_gen.log`, the stage logs, the manifest or stdout.]
4. **Seeds are visible in `ps`.** Both drivers passed seeds on command lines. The generator seeds, which are the secret that regenerates C, go to `gen_sandwich_gadget` as positional arguments (`val_chain.sh:16`; the pre-port `tdp_gen.sh:434-435`; parsed at gen `mod.rs:113-115`) and are visible for the 1–2 s that stages 1+2 run. The mixer seed is on the mixer's command line for the whole of stage 3 (`val_chain.sh:23`; the pre-port `tdp_gen.sh:504`). Seeds should be passed in by file or environment instead. [Managed: fixed; seeds travel on inherited descriptors (gen `@/dev/fd/3..5`, the mixer and `tdp_unpacking` `--seed-file /dev/fd/3`) fed by bash here-strings (bash ≥ 5.1 keeps them off disk). Running gen, the mixer or `val_chain.sh` by hand with a numeric seed still exposes it.]

**The packing sub-step adds no secret randomness; stages 4, 5 and the unpacking sub-step do.** Stage 4 uses `split_seed`; stage 5 continues the split RNG (or `cross_seed` when it starts from a file); `tdp_unpacking --refine` uses `unpack_seed` (`--seed-file /dev/fd/3`): the synthesizer's tie-break PRNG is splitmix64 seeded from (seed, window index, restart), so the same seed reproduces the output unless a window hits its deadline. Its sampled checks use the fixed public `param.verify_seed=20261011`. The packer's own randomness:

- `window_pack3.py` fingerprints values with the fixed seed `SEED = 20261004` (`window_pack3.py:26,165-166`). `link_census.py` fingerprints with the same value, imported from `boundary_cluster.py:26` (`link_census.py:18,25`). `mult_census.py` also uses `Random(20261004)` (`:7`).
- `anf_compose.py` uses the same `SEED` (`:28`) only to draw its spot-check inputs (`:93-94,132`).
- `anf_dump_eval.py` uses `Random(1)` for its timing and match inputs (`:54`).
- `verify_equiv.py` uses seed 77 (`:8`).
- `hmap_affine --seed` defaults to 12345.

Packing and repair are therefore deterministic given G2, G3 and C. Whether window boundaries should be randomized is an open design question (`docs/windowed_delivery_design.md` §8). The refined lattice is not deterministic: which windows fail to resynthesize depends on `unpack_seed` and, through the 30 s per-window deadline, on machine load (`deadline_hits` in `unpacking.report`), so `bounds.final`, `unpacking/deliverable.tdp4` and the published circuit can differ between two runs on the same packed circuit. Raise `delivery.time_limit` for a bit-reproducible refinement.

### 8.3 Scale and launches

All validated runs use the research scale: |C| = |D| = 1000 at n = 64 and 3000 at n = 128. This is **not** the library convention `round(n·log2(n)²)`, which gives 2304 and 6272 (`src/stages/sandwich/construct.rs:56-60`). The pre-port `tdp_gen.sh` used that formula; the managed driver now defaults to the validated sizes (1000 / 3000, `source_gates_rule=validated`) and keeps the formula only as the opt-in `source.gates_rule = "formula"` (§8.10, §8.12). The recommended configuration has not been run at the library size. Small n: n=16 needs an explicit `source.gates` (the six-stage runs used |C| = |D| = 32, s = 64, `slice_gates` = 320, a 208-gate sandwich; `source_gates_rule=explicit`), and stage 6 requires n ≤ 256.

Two other sizes follow from n:

- `s = max(n, round(n·log2 n))` = 384 / 896 (`construct.rs:51-54`).
- `slice_gates = 10·2n` = 1280 / 2560 (gen `mod.rs:109-112`).

The validated launches were:

```
bash W/val_chain.sh <TAG> 64 1000 384 1280 L8      # or A
bash W/val_chain.sh <TAG> 128 3000 896 2560 L8
```

### 8.4 Stages 1 and 2: `gen_sandwich_gadget`

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

### 8.5 Stage 3: `circuit_mixer`

```
NG2=$(head -1 $R/tdp.mpmct1 | awk '{print $3}')     # |G2|, from the mpmct1 header
MOVES=$(python3 -c "print(int($NG2*2*2*1.3))")      # ⌊N2·R1·1.3·|G2|⌋ = ⌊5.2·|G2|⌋ (§3.1)
FROZEN_DB_DIR=.../frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  target/release/circuit_mixer --input $R/tdp.mpmct1 --output $R/s3.mpmct1 --tdp --db-mixing \
  --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
  --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
  --no-local-verify --seed $MSEED > $R/s3.log 2>&1
```

This is `val_chain.sh:17-23`; `val_chain.sh` names the gate-count variable `G`. The validation runs' `val_chain.sh` also passed `--p-comp 1.0 --p-any 0.1`. Those values equal the defaults and have no effect here, because they act only in slot 3 (`scheduling.rs:545-575`), which is never reached when `p_db = 1.0` (§3.2); the tracked copy dropped them, and they are stage 5's live flags. The managed driver runs the same argv with `--output db_mixing.mpmct1` (written as `.tmp`, fsynced, renamed), `--seed-file /dev/fd/3` in place of `--seed`, log `stage3.log`, the completion marker `stage3.recipe`, and the output-neutral caches `LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256 XPOLY_CANON_CACHE_MB=1024`; `FROZEN_FILTER=1` is set by `lookup_miss_filter = "auto"` on a store without `filters.bin`.

| flag | value | default | role |
|---|---|---|---|
| `--tdp` | on | off | Selects the `DbLayer::tdp()` block (`cli.rs:788-802`; `:735` at HEAD). It also turns on `db_advance` (mixer `mod.rs:94-109`; `:90-100` at HEAD; the `curated`/`curated_in_comp` switches it set at `ea2e516f` went with the curated store). |
| `--db-mixing` | on | off | Preset: g57 twists at `p_twist=0.0005`, `db_advance`, `mix_pay_random` (mixer `mod.rs:73-93`). |
| `--profile` | `2,2,2,2,2` | off | N0,N1,N2,R1,R2. The size controller is the sole size authority (§3.4). |
| `--moves` | ⌊5.2·\|G2\|⌋ | 1,000,000 | Binding stop (`cli.rs:28-30`). General rule ⌊N2·R1·1.3·\|G2\|⌋ (§3.1). |
| `--p-db` | 1.0 | 0.0 | Every round that is not a twist is a slot-2 DB move (`scheduling.rs:522-544`). |
| `--db-max-degree` | 9 | off | Skips windows whose ANF degree exceeds the store's maximum degree. |
| `--db-max-span` | 30 | off | A window touching more than 30 wires counts as a miss, without canonicalization. |
| `--db-wire-terms` / `--db-total-terms` | 1024 / 2048 | 2¹⁸ / 2²⁰ | Polynomial term caps (`cli.rs:341-350`). |
| `--no-local-verify` | on | off | Disables only the per-move exhaustive local checks (`cli.rs:622-624`; §7.2). |

**What the run does** (§3.4). The target ramps linearly to 2·|G2| by eff = 2 and then holds. Size peaks near 2.2-2.25× at the 2→3 transition (vA64a: 1,748,654 gates; log lines `phase 1 -> 2 at eff=2.13` and `2 -> 3 at eff=2.63`). The controller then saturates at `pmix=0.000`, so every DB round runs in COMP mode while twist rounds still run, and the size compresses toward S\*. The run stops on `moves budget spent` at eff ≈ 3.03–3.05, before the profile completes, so the final ratio is 2.04–2.14×.

**Verification during the run.** The per-splice DB equivalence check (`verify=true` in the banner) and a global sampled check every 10,000 moves (`cli.rs:616-618`) stay on. At the end, the log reports `final float: … (verified)` and `wrote N gates`.

**Canonicalization caps.** `val_chain.sh` does not set `CANON_RULE_L_BRANCH_CAP` or `CANON_MONOMIAL_CAP`; the readers return `None` when the variables are unset (`src/canonicalization/environment.rs:6-24`), so canonicalization ran uncapped provided the calling shell did not export them (§3.1). `CANON_MONOMIAL_CAP` is read only on the g57 window path and is inert here. The `canon caps: rule_l_skips=0 monomial_skips=0` line counts only that g57 path (`src/canonicalization/window.rs:222-241`), so it is zero by construction and proves nothing about the XGate path. The managed driver no longer sets `CANON_RULE_L_BRANCH_CAP` / `CANON_MONOMIAL_CAP` (an inherited value is refused by the environment registry), so canonicalization runs uncapped, as in the reference runs.

### 8.6 Background stage-2 checks, then stages 4 and 5, then stage 6 (delivery): pack, repair, serialize, unpack with refinement

**Background stage-2 checks** (`val_chain.sh:24-27`). These start from `$ROOT` right after stage 3 and run in the background while the packer proceeds. The driver's final `wait` (`:34`) collects them. The managed driver runs them in the background after stage 2: the rebirth census as `tdp_packing mult-census tdp.mpmct1 tdp > stage2_mult.txt` (the Rust port of `mult_census.py`) and the span check as `hmap_affine … --out stage2_span` into `stage2_span.{tsv,txt}` (report-only; a standalone `tdp_gen.sh` skips it without `hmap_affine`; `scripts/tdp_gen.sh:741-755`).

```
python3 $W/mult_census.py $R/tdp.mpmct1 $TAG > $R/mult_s2.txt 2>&1 &
STEP=$([ $N = 64 ] && echo 64 || echo 256)
SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=$R/span_s2.tsv target/release/hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 \
  --g $R/tdp.mpmct1 --n $N --exposure --c-step 1 --g-step $STEP --out $R/span_s2 2>&1 | grep "\[span\]" > $R/span_s2.txt &
```

- **Rebirth census** (`mult_s2.txt`, §7.5). This prints one line: `born`, `distinct`, `repeats`, `top`, `vals>=65` (with cover) and `>=257` (`W/mult_census.py:23-25`). It is informational; §8.7 gives the reference values.
- **Stage-2 span check** (`span_s2.txt`, §7.3). This is the only runtime evidence in a run that stage 2 itself exposes no affine C relation at the scanned cuts. It scans G2 against C at every 64th (n=64) or 256th (n=128) cut, including cuts 0 and |G2|, with the outputs counted trivial. The binary writes `span_s2.tsv` and the driver keeps only the `[span]` summary line. A pass reads `cuts by intersection dim: {0: <number of scanned cuts>}; nonzero from cut None`. Any nonzero key, or a cut number in place of `None`, means some scanned stage-2 cut carries an affine combination of C values beyond the inputs and outputs. Cuts between the stride points are not tested.

**Stages 4 and 5 (managed driver only; `val_chain.sh` has none).** Stage 4 (§4): `circuit_mixer --input <prev> --split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 --moves G+1000000 --seed-file /dev/fd/3 --report-every 1000000 --state-out split.state --output split.mpmct1` (log `stage4.log`, marker `stage4.recipe`; a stop-flag stop is kept as `split.stopped.mpmct1`). Stage 5 (§5): `circuit_mixer --resume split.state --target-size round(G_in·2) --temp max(64, round(target/25)) --split-base 3 --split-damp 1 --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 --moves state_moves+round(3·target) --report-every 500000 --output crossing.mpmct1` (with stage 4 skipped: `--input <prev> --seed-file /dev/fd/3 --moves round(3·target)`; log `stage5.log`, marker `stage5.recipe` recording `resume split.state` or `input <file>`).

**Stage 6 (oracle chain).** These commands run from the working directory `RD/` (`val_chain.sh:28-33`):

```
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1 > $R/pack_G512.txt 2>&1
bash $W/repair.sh $TAG 512 $N > $R/repair512.log 2>&1
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG > $R/deliver.txt 2>&1
gzip -9 -kf $R/deliverable.bin; xz -9 -kf $R/deliverable.bin
ls -la $R/deliverable.bin* | awk '{print $5, $9}' >> $R/deliver.txt
wait
```

The packer's argument 8 must be exactly `$R/packed_G<G>.mpmct1`, and `repair.sh`'s second argument must be the same `<G>` (§6a.3, "File-name contract").

**Stage 6 (managed).** `tdp_packing run --stage2 tdp.mpmct1 --stage3 <last circuit> --source-c tdp.mpmct1.source_c.g57 --n N --out-dir packing [--packing-* only when not default] [--threads T]` (`--stage3` is the previous stage's output, `crossing.mpmct1` by default, `db_mixing.mpmct1` under `--run-skip-stages 4,5`; `--stage2` is `tdp.mpmct1`, or the sandwich when stage 2 is skipped), then, unless `--delivery-anf-only`, `tdp_unpacking run --deliverable packing/deliverable.tdp4 --packed packing/rep512.mpmct1 --bounds packing/rep512.bounds --source-c tdp.mpmct1.source_c.g57 --n N --out-dir unpacking --seed-file /dev/fd/3 --refine --max-rounds 16 [--time-limit S --restarts R --beam B] [--threads T]`; both append to `stage6.log` (a separator line between them); the marker is `delivery.recipe`, the failure marker `delivery.failed` (`scripts/tdp_gen.sh:1115-1144`).

- **Pack** (§6a.4; arguments in §6a.3). `window_pack3.py` reorders G3 by commutation (same gates, new order) and chooses window boundaries so that stage-2 values stay inside windows. Its monomial accounting does not enforce the delivery budget: vA64a's packed windows reached 77,252 monomials. The 65,536 budget is enforced later, by the split step in repair.
- **Repair** (§6a.5). `repair.sh TAG G N` cancels commuting duplicate pairs (typically about 7% of gates; vL128a went from 4,596,808 to 4,259,448) and checks equivalence in the background. It then runs up to 4 fix iterations, each a C-view scan (`hmap_affine --c source_c.g57 --c-format g57 --g P --n N --exposure --c-step 1 --g-cuts B`, 6,144 honest-slice samples, §7.3) followed, if `visible > N`, by completer/breaker slides. It re-checks equivalence synchronously, runs up to 4 fallback iterations of merge (if `visible > N`) and exposure-aware split at 65,536 monomials, and finally scores with `anf_compose.py P B 65536 TAG-rep512 500` and `link_census.py`. A split with no exposure-free candidate prints `WARNING: no exposure-free split in [lo,hi)`; this is what kept vL128a's first repair at 129.
- **Serialize** (§6a.6). `anf_dump_eval.py` writes `deliverable.bin`, then times evaluation of the gate form against the ANF form.
- **Unpack with refinement** (§6b; managed only). `tdp_unpacking run --refine` resynthesizes every window of `packing/deliverable.tdp4` from its ANF alone (30 s per window per round, 16 restarts, beam 6); every failed window of ≥ 2 gates is cut in two at an exposure-aware cut, the container re-emitted with the halves' blobs, and the next round unpacks the halves, up to 16 rounds; a window still failing after the last round, or a failed 1-gate window, is emitted verbatim (`status=verbatim_used`, NOT recommended, expected 0). Outputs: `unpacked.mpmct1` (canonical order; the published circuit) and the internal `unpacking/deliverable.tdp4` (the refined lattice), `bounds.final`, `windows.tsv`, `rounds.tsv`, `unpacking.report`.

### 8.7 Pass criteria per run

| file | required line |
|---|---|
| `gen.log` | `verify PASSED (256 bit-sliced samples, …)` and `reverse verify PASSED` |
| `s3.log` | `chain done in …: moves budget spent` **or** `chain done in …: profile complete (size schedule finished)` (mixer `mod.rs:882-899`), **together with** `final float: … (verified)` (`:901-911`; never pass `--skip-final-float`, because the `ProfileDone` exit relies on this check, §3.3) and `wrote N gates`; `curated=false` on the `DB ON` line |
| `span_s2.txt` | `cuts by intersection dim: {0: <all scanned cuts>}` and `nonzero from cut None` (§8.6) |
| `pack_G512.txt` | `input function check 4n/4n` and `VERIFY: emitted function check 4n/4n wires (OK)`; no Python traceback (`burst stuck` / `scheduler stuck`, §6a.3) |
| `repair512.log` | `EQUIVALENT (1024 random states)` (twice); a last fallback iteration showing `C-view visible n` followed by `0 splits added`; no `WARNING: no exposure-free split` line |
| `rep512.cview.log` line 2 | `n visible`. `0 coset-minimizations skipped` is advisory: a skip only makes one t\* an upper bound and does not affect visibility (§7.3). It is required only when t\* values are quoted |
| `rep512.exposure.tsv` (recommended identity test) | no visible row (t\* ≥ 0) whose gate is not the last C gate on its wire (command below) |
| `anf_rep512.txt` | `overflow 0 (0.00%)` and `spot checks: k windows re-evaluated, 0 failures (OK)` |
| `deliver.txt` | `EVAL ANF-form … (match=True)` |
| `link_rep512.txt` lines 2-3 and `mult_s2.txt` | **informational, no gate.** Linkability has no acceptance threshold. Reference ranges from the two replicates per cell are in the table below. A value far outside them is a signal to investigate, not a failure |

**Managed run.** The same criteria under the managed names, plus stages 4-6:

| file | required line |
|---|---|
| `stage12.log` | as `gen.log` above (the banner shows `seed=(file) gadget_seed=(file) sandwich_seed=(file)`) |
| `stage3.log` | as `s3.log` above; `stage3.recipe` present (the keyed completion marker) and `stage3.settings` with `stop_reason=moves budget spent` or `profile complete (size schedule finished)` |
| `stage4.log` | ends with `chain done …: split stage complete (stopped at the stage boundary)` or `moves budget spent`; `stage4.recipe` present |
| `stage5.log` | ends with `chain done …: moves budget spent` and `final float: … (verified)`; `stage5.recipe` present |
| `packing/packing.report` | `status=ok` (`repair.final_nonoutput_visible=0`); the oracle-format reports `pack_G512.txt`, `repair512.log`, `anf_rep512.txt`, `link_rep512.txt` are in `packing/` with the lines above |
| `unpacking/unpacking.report` | `status=ok`, `verbatim=0`, `over_budget=0`, every `check.*=ok` (`unclean_cuts` is informational, 0 expected) |
| `delivery.recipe` | `status=ok`; no `delivery.failed`, no `NOT_RECOMMENDED` |
| `tdp_gen.log` | ends `PIPELINE COMPLETE (stages 1-6): <run>/unpacking/unpacked.mpmct1` (`PIPELINE FINISHED WITH AN ALLOWED FAILURE` means `floor_miss`, `over_budget` or `verbatim_used`) |

**Identity-based floor test** (*derived* for the oracle, whose drivers do not run it; `tdp_packing` runs it on every run, exit 5, and `tdp_unpacking --refine` extends it to every added cut, `check.new_cuts`). The count test `vis ≤ N` can hide a leak behind a missing output (§6a.5). This command counts visible non-output rows, and it must print 0:

```
awk -F'\t' '!/^#/{g[NR]=$1; w[NR]=$2; t[NR]=$3; if(!($2 in L) || $1+0>L[$2]+0) L[$2]=$1}
  END{b=0; for(i in t) if(t[i]>=0 && g[i]!=L[w[i]]) b++; print b}' $R/rep512.exposure.tsv
```

It prints 0 on all eight replicates and 1 on vL128a v1 (`W/vL128a/rep512_v1/`). `exposure.tsv` reflects the last scan, so the test is valid only when the fallback loop ended on `0 splits added` (§6a.5).

Linkability and rebirth reference ranges (`W/v*/link_rep512.txt`, `W/v*/mult_s2.txt`):

| config, n | proxy values | L2 | M2 | anchors (100+) | froth L2 | stage-2 repeats | values born ≥ 257× |
|---|---|---|---|---|---|---|---|
| A, 64 | 15,454-15,455 | 6,007-6,039 | 73,178-73,428 | 315-321 | 373-409 | 19.9% | 329-335 |
| L8, 64 | 11,735-11,743 | 5,274-5,366 | 58,598-58,895 | 185-200 | 1,110-1,126 | 8.6-8.7% | 0 |
| A, 128 | 54,345-54,715 | 18,521-18,608 | 250,150-250,448 | 992-995 | 2,219-2,275 | 19.8% | 968-973 |
| L8, 128 | 34,751-34,958 | 16,062-16,106 | 187,248-187,496 | 701-718 | 5,544-5,579 | 8.6% | 0 |

**The chain of equality checks** (§7.2). No single check is end to end. Each link covers one hop, and every link is sampled rather than proved:

1. **Generator.** Forward and reverse verify compare G2 with the sandwich payload on 256 samples (gen `mod.rs:356-372`).
2. **Mixer.** A per-splice DB proof, a global sampled check every 10,000 moves, and `final float … (verified)`.
3. **Packer.** The input function check compares G3 with G2, and the emitted function check compares the packed order with G2. Both cover all 4n wires on 1,024 samples (`window_pack3.py:85-90,168-172,485-487`).
4. **Repair.** `verify_equiv` compares `rep512` with `packed_G512` on all wires, 1,024 states, seed 77 (`verify_equiv.py:1-21`; `repair.sh:15,27`).
5. **Spot checks.** `anf_compose` re-evaluates every 500th window against its gates.
6. **Serialization.** `match=True` compares the ANF form, built in memory from the `rep512` gates, with the gate form of `rep512` on 1,024 inputs (`anf_dump_eval.py:59-110`).

The serialized bytes of `deliverable.bin` are never read back or verified, and C itself is never compared with the deliverable (§7.7). (Oracle. The port reads the container back, decode = window ANF and `check_deliverable` on 1,024 states, and compares the decoded deliverable, the refined deliverable and `unpacked.mpmct1` with C on the zero slice, 4,096 inputs.)

### 8.8 Artifacts and formats

Each oracle run lives in `W/<TAG>` and takes about 132–137 MiB at n = 64 and 427–446 MiB at n = 128 (`du`; four-stage runs. A six-stage run adds `split.mpmct1`, `split.state` and `crossing.mpmct1`, 2-3.5× the stage-3 gates, and `unpacking/` with one container per round). vL128a is about 499 MiB because it also holds the `rep512_v1/` archive. **Treat every file except the published circuit as confidential.** The published object is `unpacking/unpacked.mpmct1` (under `delivery.anf_only`, `packing/deliverable.tdp4`); the two `deliverable.tdp4` are the internal windowed representation (`deliverable.bin` is the raw body kept beside the container), and the reports are internal records. Most files are secret-equivalent (oracle names; the managed run directory follows):

| class | files |
|---|---|
| **Secret: seed-equivalent** | `seed.txt`; `tdp.mpmct1.source_c.g57` (C: 4,000 B / ~15.2 KB); `tdp.mpmct1.sandwich.mpmct1` (contains C); `gen.log` and `s3.log` (both record seeds) |
| **Secret: gate-level circuits** | `tdp.mpmct1` (~13.9 / 41 MB; G2, the input before obfuscation); `s3.mpmct1` (~30 / 88–92 MB); `packed_G512.mpmct1` + `.bounds`; `rep512.{mpmct1,bounds}`. These gate lists carry the mixer's and the packer's spelling, which the ANF representation and the resynthesis exist to replace. |
| **Secret: C-keyed leak maps** | `rep512.{exposure,wit,allcuts}.tsv` and `rep512.cview.log` (per C gate: its wire, t\*, the cut and the predictor support; `exposure.tsv` alone lists C's full per-gate wire sequence); `split_cands_<pid>.*`; `span_s2.{tsv,txt}` (C-versus-G2 span scan) |
| Confidential reports | `pack_G512.txt`; `repair512.log`; `anf_rep512.txt`; `link_rep512.txt`; `mult_s2.txt`; `deliver.txt` |
| **Published (the TDP)** | managed: `unpacking/unpacked.mpmct1` (the resynthesized circuit in canonical order); under `anf_only`, `packing/deliverable.tdp4` (container v1 on the packer's lattice). The oracle chain is ANF-only by construction: `deliverable.bin` (the `.gz` / `.xz` copies exist only to measure compressibility) |
| **Internal windowed representation** (managed) | `packing/deliverable.tdp4` (and `deliverable.bin`), `unpacking/deliverable.tdp4`, the per-round `deliverable.<r>.tdp4`, `bounds.final` and `bounds.<r>`: what the unpacking sub-step resynthesizes from, kept for auditing and parity; never published in the default mode |
| **Reports** (managed; internal records) | `packing/packing.report`, `unpacking/unpacking.report` (counts and parameters only; no seed, no hash of a private input; not part of the TDP) |

The managed run directory, by stage:

| stage | files |
|---|---|
| 1 | `tdp.mpmct1.source_c.g57`, `tdp.mpmct1.sandwich.mpmct1` |
| 2 | `tdp.mpmct1`, `stage12.log`, `stage12.recipe`; report-only `stage2_mult.txt` and `stage2_span.{tsv,txt}` |
| 3 | `db_mixing.mpmct1` (written as `.tmp`, fsynced, renamed), `stage3.log`, `stage3.settings`, `stage3.recipe`; a stop-flag ending leaves `db_mixing.stopped.mpmct1` |
| 4 | `split.mpmct1`, `split.state`, `stage4.log`, `stage4.recipe`; a stop-flag ending leaves `split.stopped.mpmct1` |
| 5 | `crossing.mpmct1`, `stage5.log`, `stage5.recipe` |
| 6 | `packing/`: `deliverable.tdp4`, `deliverable.bin`, `packed_G512.mpmct1(.bounds)`, `rep512.mpmct1`, `rep512.bounds`, `rep512.{exposure,allcuts,wit}.tsv`, `rep512.cview.log`, `pack_G512.txt`, `repair512.log`, `anf_rep512.txt`, `link_rep512.txt`, `packing.report`; `unpacking/` (absent under `anf_only`): `unpacked.mpmct1` (the published circuit), `deliverable.tdp4`, `bounds.final`, `windows.tsv`, `rounds.tsv`, `unpacking.report`, `unpacking.log`, per-round `bounds.<r>` / `deliverable.<r>.tdp4`; `stage6.log`; `delivery.recipe`; on failure `delivery.failed`; `NOT_RECOMMENDED` after an allowed failure |
| run | `tdp_command.conf` (recipe manifest), `tdp_gen.log`, the private `SEEDS` |

**Formats.** mpmct1 is in §1.3 (for example, the line `61 1 2 191 0 222 1` means `st[61] ^= ¬(¬st[191] ∧ st[222])`); `.bounds` is in §6a.2; `deliverable.bin` is in §6a.6, with a worked example. The raw body `deliverable.bin` has no header; the container `deliverable.tdp4` (`LMTDPANF` v1, 48-byte header: version, source and state wires, layout, window count, body length, CRC-32 of body and header) carries them, and `tdp_packing decode --format container` validates it. The "size: ANF X MB" line in `anf_rep512.txt` is a model estimate (`anf_compose.py:154-155`); the real file is slightly larger (vA64a: 24.6 MB estimated, 26.0 MB on disk; §6a.6).

### 8.9 Reference measurements (validation replicates, 2026-10-07/08)

Eight runs were launched together on an 18-core, 128 GiB Apple M5 Max. Each mixer ran single-threaded. They are recorded in `STAGE2_LINKABILITY.md` §5 (which also carries the serialized sizes and evaluation speed per configuration); linkability figures are in §7.6, and the archived runs themselves are the port's parity targets (§6a.7).

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
- `deliverable.bin` was re-serialized on 2026-10-08 (§6a.7); the first repair's file was 103.2 MB (9.52 MB xz).

All eight runs were re-repaired with the patched `order_repair.py` (the ported version, §6a.5) and re-serialized on 2026-10-08; the resulting `rep512.*` and `deliverable.bin` are the port's parity targets (§6a.7).

**Phase timings** (measured with 8 jobs running concurrently; single-job timings were not recorded):

| phase | n = 64 | n = 128 |
|---|---|---|
| stages 1+2 | < 1 s | ~2 s |
| stage 3 | 82–88% of wall time (A 87–88%, L8 82%) | 74–85% of wall time (A 80–85%, L8 74–83%) |
| pack | 55–87 s | 159–205 s |
| repair | 162–276 s | 741–1,705 s |
| serialize | 31–53 s | 70–113 s |

**Evaluation speed.** In Python, batched over 1,024 inputs, the ANF form costs 0.77–1.00 ms per input at n = 64 and 1.77–2.04 ms at n = 128. The gate form costs 0.63–0.75 and 1.52–1.73 ms. No repaired window exceeded the 65,536-monomial budget; the median window holds 419–491 monomials.

**L8 against A.** See the single reference table in §7.6. In brief: L2 −12% / −13%, M2 −20% / −25%, `deliverable.bin` size +19% / −1% (n=64 / n=128), and froth L2 ×2.9 / ×2.5.

**Stage-3 share of wall time** is stage-3 wall time divided by total wall time, per run: 87.4, 88.1, 82.2, 82.5% at n=64 (vA64a, vA64b, vL64a, vL64b) and 84.7, 80.3, 74.0, 82.6% at n=128 (vA128a, vA128b, vL128a, vL128b). vL128a's total includes its first, failed repair.

**Build provenance.** The archived C-view TSVs (`rep512.{exposure,allcuts,wit}.tsv`, line 2 of `rep512.cview.log` and the split-candidate TSVs of all eight tags) are reproduced by `hmap_affine` built from `492564f0` (`STAGE4_RUST_DESIGN.md` §12, P2), so the numbers below stand; gen and the mixer at `492564f0` reproduce the archived stage-2 artifacts of all eight tags and `s3.mpmct1` of vL64a and vL128a (the archived anchor runs). No commit hash was recorded for the binaries that ran; their file times on this machine:

| binary | built | newer sources? |
|---|---|---|
| `target/release/gen_sandwich_gadget` | 2026-10-07 19:18:36 | none: no file under `src/`, `Cargo.toml` or `Cargo.lock` is newer |
| `target/release/circuit_mixer` | 2026-10-07 19:18:35 | none |
| `target/release/hmap_affine` | 2026-10-07 18:54:28 | **yes**: `security_tests/heatmaps/hmap_affine.rs` was last modified at 19:17:46 and committed in `7163e457` (23:21) |

The replicates were launched at 23:23 (`W/v*/seed.txt`), so gen and the mixer match the tree as of 19:18, while every C-view, span-check and repair number (§6a.5, §7.6, §8.6) came from an `hmap_affine` binary older than its committed source; the rebuild from `492564f0` above is what ties those numbers to committed code.

**Six-stage reference runs.** n=64 L8 (`port_runs/seven_n64`, 2026-10-11, budget 2×target, 8 threads): stages 1+2 775,332 gates (1 min); 3: 1,583,698 (29 min, 4,031,726 moves); 4: 2,987,961 (1 min); 5: target 5,975,922, temperature 239,037, budget 11.95M moves → 5,380,721 (7 min); packing 10,150 windows → repair 5,153,273 gates / 10,365 bounds, floor 64/0, 88 min; unpacking before the refinement policy: 10,366 windows, 10,209 synthesized, 157 verbatim (1.5%), 371 deadline hits, 4,766,644 gates (0.925), 68 min (8.9 CPU-hours). n=128 stages 4/5 (the revival study, 6× budget): 4.59M → 8.66M (8.5 min) → 15.7M (6.3 h); packing of the stage-5 output 29,969 windows, 7.7 h, floor 128/0. n=16 six-stage run (`port_runs/six_n16b`, 2026-10-12, |C|=|D|=32, budget factor 2 at the time): 48,447 → 100,826 → 192,489 → 347,900 gates (target 384,978, temperature 15,399); delivery 678 windows, 3 refinement rounds (16 + 2 splits, 0 unclean cuts, 0 over budget, 0 verbatim, 24 deadline hits), 696 final windows, 333,588 → 227,981 gates (0.683), all checks ok, 18 min end to end (`docs/notes/seven_stage_20261011.md` §3-4; `docs/notes/stage45_revival_20261011.md` §1; `port_runs/six_n16b/{tdp_gen.log,unpacking/unpacking.report,unpacking/rounds.tsv}`).

### 8.10 The managed driver today

The managed driver (`scripts/tdp_gen.sh` with `src/tdp/`, run through `tdp_gen --config`) runs all six stages (every stage after 1 optional: `run.skip_stages` / `--run-skip-stages 2..6`; `run.stop_after_stage` default 6), config v4, recipe v12, SEEDS format 3 (gen, mix, split, cross, unpack). Its defaults: preset `L8` (`A` selectable), schedule `validated` (profile `2,2,2,2,2`, moves `int(G·2·2·1.3)`, serial), the validated source sizes (1000 at n=64, 3000 at n=128), stage 4 with the archive-era flags (`--split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 --moves G+10^6`), stage 5 with `target_size_factor` 2, `width_penalty_base` 3, `width_penalty_threshold` 1, `size_tolerance_divisor` 25 (temperature max(64, target/25)), `move_budget_factor` 3 (RC 2026-10-12; 2 on 2026-10-11; 6 in the archive-era driver; `move_attempts` = absolute override), packing at G=512 with split budget 65536, 4+4 iterations and `cpython39` caps, delivery `anf_only=false`, `time_limit` 30 s per window per round, `restarts` 16, `beam` 6, `max_rounds` 16. `--run-skip-stages 4,5 --delivery-anf-only` is the validated four-stage chain and the only chain for which `pinned_recipe` is recorded. Markers: `stage12.recipe`, `stage3.recipe`, `stage4.recipe`, `stage5.recipe`, `delivery.recipe`; failure marker `delivery.failed` (`exit=`, `step=packing|unpacking`); logs `stage12.log`, `stage3.log`, `stage4.log`, `stage5.log`, `stage6.log`. Every gap of the managed-driver gap analysis is closed (`docs/STAGE4_RUST_DESIGN.md` §2.2, D1-D18; `docs/local_mixing_history.md`, October 8, 2026). Use it; `W/val_chain.sh` is a parity tool.

Resume semantics: a stage is skipped only when its keyed marker matches its inputs, parameters, binary and output; `run.rerun_from_stage` recomputes a stage and every later one; rerunning any of stages 2-5, or changing any packing or delivery parameter (`anf_only` included), reruns the whole stage 6 (both sub-steps); a `*.stopped.mpmct1` output is under-dosed and never handed on; a pre-marker run's `db_mixing.mpmct1` is adopted when its `delivery.recipe` (or the seven-stage driver's `packing.recipe`) carries the tags of exactly these bytes. Failure outcomes: `delivery.failed` and a non-zero exit; an allowed failure (`floor_miss`, `over_budget`, `verbatim_used`) writes `NOT_RECOMMENDED` next to the final output and ends with `PIPELINE FINISHED WITH AN ALLOWED FAILURE`.

Driver history, so that old run directories and notes can be read: the four-stage driver (2026-10-09; config v2, recipe v10, SEEDS format 2 with gen and mix, `stage4.failed`, `packing.recipe`), the seven-stage driver (2026-10-11; config v3, recipe v11, `--run-skip-stages 4,5,7`, `packing.recipe`/`unpacking.recipe`, `packing.failed`/`unpacking.failed`, `[unpacking]` with `allow_fallback`), and the six-stage driver with one delivery stage (2026-10-12; config v4, recipe v12, `delivery.recipe`/`delivery.failed`, `[unpacking]` → `[delivery]` plus `max_rounds`, `allow_fallback` removed). A v1-v3 config, an `[unpacking]` table, a `7` in `run.skip_stages` or a `--unpacking-*` flag is refused with a migration pointer; v9-v11 runs are resumed with the checkout that created them.

### 8.11 Pitfalls

- **File-name contract.** `repair.sh TAG G N` finds its input only as `W/TAG/packed_G<G>.mpmct1` (+ `.bounds`), so the packer's output argument must use that exact name and the same `<G>` (§6a.3). `anf_split2.py` run outside `repair.sh` silently defaults to the unrepaired pack and to `fin.exposure.tsv` unless `SPLIT_P` and `SPLIT_EXP` are set (§6a.5).
- **Concurrency.** Never run two repairs in the same TAG directory: they share the `rep<G>.*` scratch files (`repair.sh:10-19`). (Oracle. `tdp_packing run` and `tdp_unpacking run` into a reused `--out-dir` first remove the earlier run's outputs; the managed driver removes `packing/` and `unpacking/` before re-running stage 6.) The earlier finalizer `W/finalize3.sh` (not used by `val_chain.sh`) likewise shares `fin.*` and `final_G<G>.bounds` (`finalize3.sh:10-18`). Never edit a bash driver while instances are running it, because bash reads scripts by byte offset.
- **No `set -e`.** `repair.sh` and `val_chain.sh` use `set -uo pipefail` without `-e`. If `hmap_affine` is missing or fails, `cview` returns an empty string, the `-le` test errors, and the script keeps going. `cview` also depends on line 2 of the hmap log (`repair.sh:20`).
- **The fallback loop can exit unchecked** (oracle; the port rescans and exits 5/6). After its fourth iteration it exits without rescanning, so the final printed C-view can predate the last merge or split (`repair.sh:28-33,39`). Always confirm the `0 splits added` line and the absence of `WARNING: no exposure-free split`.
- **Pack report labels are wrong at n = 128.** `window_pack3.py:49-50` hard-codes the wire-class thresholds at 64/128, so the per-class exposure lines in `pack_G512.txt` are mislabeled for n = 128 runs.
- **Stale documentation.** `docs/ANF_WINDOW_DELIVERY.md` describes 4,096 fingerprint lanes, per-window verification and a G = 128-era parameter table (`:183-187`; §7.8). The code uses 1,024-sample (16 × 64-bit) fingerprints, checks every 500th window and runs at G = 512. *(Superseded 2026-10-09 and 2026-10-12: `docs/tdp_pipeline.md` once described the historic six stages (old 4-6: split, crossing, fcompress) with profile `3,30,30,2,2`; on 2026-10-09 the four stages; since 2026-10-12 the current six stages, 1-3, 4 splitting, 5 crossing, 6 delivery, with the `validated` schedule.)*
- **`--moves` is absolute on a resume**: the stage-5 budget is added to the split counter (§5.1).
- **`delivery.time_limit` makes the refined lattice timing-dependent** (§8.2).
- **`status=verbatim_used` exits 0** with `NOT_RECOMMENDED`: read `delivery.recipe`'s `status=`.
- **Stage 6 needs n ≤ 256** (4n ≤ 1,024 state wires); nonlinear291 widths are refused with stage 6 on.
- **A `config_version = 3` file, an `[unpacking]` table or a `7` in `run.skip_stages` is refused** with a pointer.
- **On the stage-5 output the packer needs ~4× the windows and hours** (2.2 h at n=64, 7.7 h at n=128).

### 8.12 Open decisions

Decided by the port (`docs/STAGE4_RUST_DESIGN.md`): the default |C| is the validated table (1000 / 3000; other n need an explicit size or the opt-in formula); seeds are drawn by the driver into a private `SEEDS` file, which `tdp_gen.sh` reads, and never travel on any argv: gen and the mixer read them from inherited descriptors (`@/dev/fd/3..5` for gen, `--seed-file /dev/fd/3` for the mixer; their banners print `(file)`), unlike `val_chain.sh`, which passes them on argv (the here-strings feeding those descriptors stay off disk because the driver requires bash >= 5.1); L8 is the default; the container is `deliverable.tdp4` (version 1) with a validating decoder. Since then: five seeds in SEEDS format 3 (gen, mix, split, cross, unpack); the split and crossing mixer runs and `tdp_unpacking --refine` read theirs from `--seed-file /dev/fd/3` as well; the published object is the circuit `unpacking/unpacked.mpmct1`, resynthesized from the refined container `unpacking/deliverable.tdp4`, which stays internal (RC, 2026-10-13; under `anf_only` the published object is `packing/deliverable.tdp4`). Still open:

- **Seed entropy:** C is generated from at most 64 bits whatever the seed width; route (a) a 256-bit ChaCha-derived C (changes bytes, needs re-validation) or (b) a CSPRNG-generated C through `source.path` (design §13.1).
- **Boundaries:** whether to randomize window boundaries (the refinement already makes the refined lattice, and with it the published circuit, seed- and timing-dependent, §8.2).
- **Resynthesis adversary:** merged-window resynthesis of the crossing output at W=2000 exposes 244 of 936 interior C values; 10× the time budget on the deadline windows changes it to 241 (rescued windows add 0 new values), canonical order changes it by ≤ 2%, and stages 4+5 do not close the leak (the crossing output has 0 C values at every cut, the stage-3 output under the same test 434); the delivered-lattice resynthesis leaks 5 of 936 (seed- and timing-specific). Policy decided 2026-10-12: accept any synthesized size, bisect windows that do not resynthesize (`docs/DELIVERY_STAGE.md`; §6b, §7.6).
- **Stage-5 parameters:** target 2×, penalty base 3 / threshold 1 and temperature target/25 are the archive-era calibration whose promotion status was never resolved; the budget factor went 6 → 2 (2026-10-11; the walk stopped at 0.90 of target at n=64) → 3 (2026-10-12), and no run at 3× is recorded (§5).
- **`verbatim_used`:** reported as NOT recommended, not fatal (expected 0); whether to make it fatal is open.
- **Timings:** solo stage-3 timings and validation of `--parallel-pieces` with this configuration. (The managed driver no longer sets the 512/200000 canonicalization caps: the reference runs had them unbounded.)


## Glossary

| term | meaning | see |
|---|---|---|
| A | The stage-1 sliced sandwich on 2n wires: `[C ⧢ S1] ; N ; [D ⧢ S2]` | §1.4 |
| anchor | A proxy value with 100+ runs (sightings) in the link census. The residual class under L8 is carrier-migration hops | §7.5, §2c.11 |
| ANF | Algebraic normal form: a GF(2) polynomial written as an XOR of monomials (ANDs of variables). Inside the delivery stage each window is represented only as the canonical vector ANF of its permutation, from which it is resynthesized; under `anf_only` that ANF is what ships | §6a.6, §6b |
| atom | Stage-2 mask-budget unit, `Σ_w (u_w + 1 + extra_lgis)` over data wires (u_w = writes plus control reads of w) | §2b.1, §2b.3 |
| band | The 2n auxiliary wires `2n..4n` (roles `np..np+r`). 0 on the public slice, filled from the input by the band seed, keyed into every mask, junk at both ports | §2a.1 |
| band seed / band reseed | Modules 2 and 4 of stage 2: `seed_band_rich` fills each band wire with a pivot, 1-7 linear terms and 4 products. The reseed plays the seed's role for backward evaluation | §2a.3, §2a.4 |
| birth, born value | A value written by a gate. A born value of C is the value a C gate leaves on its target; these are the C-view rows | §7.3 |
| birth advance | Stage-3 post-splice slide of each new gate by ⌊0.85·slack⌋ in its direction | §3.5 |
| boundary | An interior cut of the delivered window lattice, listed in `rep<G>.bounds` (the packer's lattice) and, after refinement, in `unpacking/bounds.final` (the refined lattice). Internal in the default mode (the published circuit carries no lattice); under `anf_only` the adversary holds the ANFs and can derive the wire state at every boundary | §6a.1, §6a.2, §6b |
| bracket | A dirty-ancilla gadget that adds a product term to the fire target using only ≤2-control gates: the 4-gate `and3` (adds p·x·y) or the 8-gate Gray walk (adds P1·P2). Its final gate restores the ancilla | §2b.5 |
| breaker / completer | For a C-view witness at cut c with predictor support S: the completer is the last gate before c that writes S, the breaker the first at or after c. Fix mode slides them apart | §6a.5 |
| burst (rerand burst), slot | A 16-gate refresh of one band wire, `b ^= lit·lit`; one burst per slot, m/8 slots per build. Not to be confused with the packer's atomic burst of an interval | §2b.8, §6a.4 |
| carrier | The physical wire currently holding a data role | §2c |
| carrier migration | The stage-2 transform (code: preprocessing shuffling) that moves the written data role onto band roles' wires with two-CNOT transfers inside each fire box. In persistent mode, roles stay off home until the tail swaps | §2c |
| celebrity value | A value reborn very many times (multiplicity ≥ 257), typically an ancilla's prior value restored by brackets or a migration hop | §2b.6, §2c.11 |
| chain (chained brackets) | `CHAIN=k`: a fire's brackets share `max(k,2)` fire-safe band ancillas and defer their restores. `CHAIN_LEN=L` flushes an ancilla's queue every L restores | §2b.7 |
| config A / config L8 | The recommended stage-2 environment without / with `CHAIN=2 CHAIN_LEN=8`. L8 is recommended; A remains supported | §2b.2 |
| C-view | For each value born by a C gate, t\* = the fewest boundary wires whose XOR plus a constant equals it on 6,144 honest-slice samples. A value is visible if some boundary admits such a predictor | §7.3 |
| COMP / MIX | Stage-3 round modes. MIX may grow the window's spelling ("free, else pay at random"); COMP only shrinks or respells at equal length, by prefix descent | §3.3, §3.5 |
| cut | A position between two gates, i.e. the state after the first j gates. Boundaries are the cuts kept in the delivered lattice | §7.1 |
| D | The decoy: a random g57 circuit with C's design and size. Reversal on the slice yields D⁻¹, not C⁻¹ | §1.4, §1.5 |
| DAG interval | The gates that must lie between a value's birth and its overwrite in every commuting reorder; the unit the packer keeps inside one window | §6a.4 |
| dose | The maximum number of carrier transfers per fire box (`EMBEDDED_MASKING_SHUFFLING`; recommended 8) | §2c.3 |
| drift (restore-drop) | Dropping a bracket's final restore so the ancilla drifts by p·x. Default on, inert under L8 and nearly inert under A | §2b.6 |
| eff | Stage-3 effective work: the sum of 1/size over moves, roughly the number of passes over the circuit | §3.4 |
| escort | A partner's value riding XORed on the fire target from its transfer to the box drain | §2c.4 |
| exposed (packer) | The packer's RESULT count of stage-2 values alive across a boundary of the **pre-repair** pack, or forced at the output | §6a.1, §6a.4 |
| fire | The masked evaluation of one sandwich gate: read cover, product units, fire cover, straddle | §2b.5 |
| fire box | The stretch of the stage-2 stream holding one fire; the unit inside which carrier migration places transfers | §2c.2 |
| fire cover | A temporary mask opened on the fire target for the duration of the fire, sampled away from the operand masks | §2b.5 |
| first-layer gate, fringe leak | A C gate whose target and controls still hold raw inputs. Its value `1 ⊕ a ⊕ y ⊕ x·y` can become affine-visible after reordering (the fringe leak); PREMASK and SEED_AVOID_C1 close it | §2a.5, §2b.3 |
| floor | The n output values of C, trivially visible on the answer register; the C-view target | §7.3 |
| froth | Values with ≥2 boundary runs that no stage-2 gate outputs: mixer-made values plus input words never reborn in G2. "Linkable froth" is froth L2 | §7.5 |
| G (window cap), 𝒢 | G is the packing window gate cap of stage 6 (`packing.window_gates`, 512); 𝒢 is the published permutation | Overview, §6a.3 |
| G2, G3, G4, G5 | The stage-2 gadget (`tdp.mpmct1`), the stage-3 mixed circuit (`db_mixing.mpmct1` in a managed run; `s3.mpmct1` in the oracle tree), the split circuit (`split.mpmct1`) and the crossing-walk circuit (`crossing.mpmct1`), all on 4n wires. \|G2\| is G2's gate count. In the six-stage chain the packer's `--stage3` input is the last circuit produced, `crossing.mpmct1` by default (`split.mpmct1` with stage 5 skipped, `db_mixing.mpmct1` with 4 and 5 skipped) | Overview |
| g57 gate | `[a,b,c]`: `a ^= b ∨ ¬c` (= `1 ⊕ c ⊕ bc`), the XGate with comp = 1 and monomial `¬b ∧ c` | §1.2 |
| g57 twist | A rare stage-3 move that swaps two wire labels across a long window, closing each end with an all-g57 seam word | §3.6 |
| hidden firing, straddle | The persistent mask opened on the target mid-fire, so that no block containing the fire shows the bare gate increment | §2b.5 |
| hot-targeting | Choosing each rerand slot's band wire among those with the most bracket borrows since their last burst | §2b.8 |
| identity guard | Stage-3 rule that skips a store candidate equal to the window or to a commuting reorder of it | §3.5 |
| LGI | "Locally geodesic identity": the open/close pair of one mask, `g57(w,x,y); CNOT(w,z)`, which XORs `M = 1 ⊕ y ⊕ x·y ⊕ z` onto w and removes it when re-emitted | §2b.3 |
| linkable value (L2) | A proxy value seen in ≥ 2 separate runs at boundaries | §7.5 |
| M2, sightings | A sighting is one run of a value; M2 is the total number of runs over all linkable proxy values | §7.5 |
| mask | A balanced quadratic band term held on a data wire; each data wire carries 2-3 open masks from PREMASK to the final drain | §2b.3 |
| mpmct1 | The gate-list text format of every stage: header `mpmct1 <wires> <gates>`, one `t comp k (wire pol)×k` line per gate | §1.3 |
| multiplicity | The number of times a value is born in G2 (`mult_census.py`) | §7.5 |
| N column, N-float | The n CNOTs `y_i ^= x_i` that copy C(x) into the answer register, floated left or right by commutation | §1.4 |
| partner | The band role whose wire receives the fire target at a transfer; chosen by read pressure among non-excluded roles | §2c.5 |
| payload half | The sandwich half that carries the answer on the slice: wires n..2n for classic | §2a.7 |
| φ (phi) | Pearson correlation of two binary sample vectors; the statistic behind several design-rationale measurements | §7.9 |
| PREMASK | Opening `min_open` masks on every data wire before the first fire, so no raw input copy survives into the compute | §2b.3 |
| proxy value | A boundary value equal to the output of some stage-2 gate. The count of distinct proxy values on `rep512` is the delivered stage-2 exposure ("proxy values (link census)") | §6a.1, §7.5 |
| public slice, zero slice | The inputs with y = 0 and band = 0, on which G2 (and the published circuit) computes C(x) forward and D⁻¹ backward | §1.5, §2a.1 |
| read pressure | Reads since the last write on a physical wire; transfer partners are chosen among the maximal ones | §2c.5 |
| regular frozen store | `frozen_m1_m11`, the g57 store keyed by canonical function key; the only store stage 3 uses | §3.1, §3.5 |
| role | A logical data or band identity, as opposed to a physical wire | §2c |
| run | A maximal stretch of consecutive cuts (input cut and interior boundaries) at which a wire holds the same value up to complement | §7.5 |
| rung | One nested sub-window tried in a COMP prefix descent | §3.5 |
| SEED_AVOID_C1 | Band-seed rule that redraws any product on the control pair of a first-layer sandwich gate | §2a.5 |
| slice blocks S1, S2 | The comp = 0 gates that read one y wire and are the identity on y = 0, merged into C and D | §1.4 |
| slice guard | Modules 1 and 5 of stage 2: identity exactly when the band is 0, junk-half targets only, ≤ 2 controls (accumulate-and-inject) | §2a.6 |
| span check (excess) | `dim(U∩W) − dim(I∩W)`: affine combinations of C values available at a cut beyond inputs (and outputs) | §7.3 |
| t\* | The C-view predictor weight of a born value; 1 means a single boundary wire holds it | §7.3 |
| touch AUC | Rank-statistic separability of two wire classes by per-wire touch count; 0.5 = indistinguishable | §7.9 |
| transfer | The two-CNOT role swap `cnot(q,p); cnot(p,q)` that moves the fire target onto a partner's wire | §2c.4 |
| vocabulary | The set of distinct nonconstant stage-2 values (up to complement), the denominator of "% of vocabulary" | §6a.1 |
| window | The gates between two consecutive boundaries; inside the delivery stage it is represented only as the canonical ANF of its permutation, and every window is resynthesized from that ANF alone; the published circuit is the canonical-order concatenation of the resynthesized windows (under `anf_only` the ANF itself ships). A window that does not resynthesize is cut in two at an exposure-aware cut and both halves re-emitted and unpacked (refinement) | §6a, §6b |
| split stage, complemented-gate pool | Stage 4: every complemented g57 gate (`comp_ids`) is split into plain conjunctions, joined to a distant bracket by an absorbed NOT twist with probability 0.8, to exhaustion | §4 |
| crossing walk, thermostat, target size, temperature, width penalty | Stage 5: fragments shot to their first collider and crossed by exact case splits; a size thermostat holds the gate count near target = round(2 × input) with temperature max(64, target/25), contracting by exact undo and merge; a split of width c above threshold d is accepted with probability 3^−(c−d) | §5 |
| delivery stage | Stage 6: the packing sub-step (`tdp_packing run`) followed by the unpacking sub-step with lattice refinement (`tdp_unpacking run --refine`) | §6a, §6b |
| refinement round; pending / converged / failed | One pass of the delivery loop over the windows not yet unpacked (pending); a window ends converged (exact + sampled checks pass) or failed (timeout, stalled, stepcap, verify-fail, exact-overflow) | §6b |
| stubborn window | A packed window that does not resynthesize within the delivery budget; an ANF-density class, cut in two by the refinement | §6b |
| exposure-aware cut, unclean cut | The refinement's cut of a failed window: among 32 candidates, the one nearest the midpoint at which no interior C value is affine; the midpoint when none is clean (an unclean cut, counted in `unclean_cuts`) | §6b, §7.4 |
| over-budget half | A refinement half whose ANF exceeds the 65,536-monomial budget: emitted but not synthesized, split again next round | §6b |
| verbatim (`status=verbatim_used`) | A window emitted as packed gates because it still failed after `max_rounds` or is a failed 1-gate window; reported as NOT recommended | §6b |
| delivery seed (`unpack_seed`) | The fifth seed of `SEEDS` (format 3), read by `tdp_unpacking run --refine` | §6b, §8.2 |
| `deliverable.tdp4` / `deliverable.bin` | The `LMTDPANF` v1 container (48-byte header, then the window blobs) / its raw body: the internal windowed representation (`packing/`: the packer's lattice; `unpacking/`: the refined lattice, the "refined deliverable"), from which the published circuit is resynthesized; published only under `anf_only` | §6a.6, §6b |
| `bounds.final` | The cuts of the refined lattice, the refined deliverable's boundaries (internal) | §6b |
| `unpacked.mpmct1`, published circuit, TDP | The published object of the pipeline: the canonical-order concatenation of the resynthesized windows, a plain `mpmct1` gate list with no lattice and no ANF (under `anf_only` the published object is `packing/deliverable.tdp4` instead) | Overview, §6b |
| canonical order (`xgate_canonicalize`) | The commuting gate order the published circuit `unpacked.mpmct1` is written in: every gate pulled as early as commutation allows when that lowers its key, so no window boundary survives | §6b |
| `anf_only` | `delivery.anf_only`: end stage 6 after the packer's deliverable (the validated four-stage chain with `skip_stages = [4, 5]`) | §6b, §8.10 |
| `NOT_RECOMMENDED`, `delivery.recipe`, `delivery.failed` | The allowed-failure marker next to the final output; stage 6's keyed completion marker; stage 6's failure marker (`exit=`, `step=packing\|unpacking`) | §6b, §8.10 |
| winnable | A stage-2 value that is not output-forced and whose spans and DAG intervals fit `span_cap` and `W_cap`, so the packer can try to hide it | §6a.4 |
