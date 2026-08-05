# GSS-MIX — the end-to-end mixing pipeline for GSS circuits

`scripts/gss_mix.sh` packages the whole mixing pipeline for the special case
of **GSS inputs** (gadgetized sliced sandwich): one command from "a wire
count" to a compressed, mixed circuit, with every stage's artifact, state
file and log left in the run directory. Companion docs:
`SLICED_SANDWICH.*` (stage 1), `NEW_GADGETIZE.md` (stage 2),
`POSTMIX_MANUAL.md` §2.1.2–2.1.3 (stage 3), `FMIX_SPLIT_TWIST.md` +
`SPLIT_TWIST_REPORT.pdf` (stage 4), `POSTMIX_MANUAL.md` §3 (stage 6).

```
n ──1─▶ sliced sandwich S ──2─▶ GSS gadget ──3─▶ phase A ──4─▶ split stage ──5─▶ crossing walk ──6─▶ fcompress
        (2n wires)              (4n wires)      (DB mixing)   (g57 → pairs)     (pure growth)      (final)
```

## Quick start

```bash
cargo build --release
export FROZEN_DB_DIR=...        # the frozen replacement store (stage 3)
export FROZEN_CURATED_DIR=...   # recommended: curated-first cascade
scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_s1 -s 1
```

Artifacts land in the run dir: `gss.mpmct1` (+ `.sandwich.mpmct1`,
`.source_c.g57`), `phaseA.mpmct1`(+`.state`), `splitB.mpmct1`(+`.state`),
`crossB.mpmct1`(+`.state`), `final.mpmct1`, per-stage logs, and
`gss_mix.log` (the pinned-parameter narrative). A stage whose artifact
already exists is skipped, so a killed pipeline re-invoked with the same
command continues; `--force-from K` rebuilds from stage K; `--stop-after K`
ends early. Each fmix stage arms its own `stageN.stop` / `stageN.dump`
flag files (touch to stop cleanly / snapshot).

## The stages and their parameters

**1+2 — generate + gadgetize** (`gen_sandwich_gadget`). The only free
parameter is **n**, the wires of the source computation C. Everything else
is the library convention, computed and logged by the driver:
`|C| = |D| = round(n·(log₂n)²)`, slicing budget `s = round(n·log₂n)`,
`slice_gates = 20n`, `rg_freq = 1`. The gadgetization runs the **production
preset** (mask plan [2,2,2,3] + Gray fold, single-carrier decode, nonlinear
band fill, band roll, retire-refill epochs) — it is the tool's default; no
`PROD_*` variables are set. The sandwich S (2n wires) and the source C are
dumped beside the gadget (4n wires) for later reconstruction checks. The
tool sample-verifies gadget-low ≡ S on the zero slice (200 samples).
Seeds: C and the sandwich use the master seed, the gadgetization
`seed+1` — vary `-s` for a fully fresh pipeline.

**3 — fmix phase A** (`--gss --phase-a --profile`). The DB-mixing stage,
using the GSS DB profile (curated-first, per-mode s_db, g57-preserving) and
the phase-A block (twist-g57, db-advance, pay-random) with the layer-2 size
profile as the single size authority. The two knobs exposed, per the design:

- `--expand R` — the **max expansion factor** R1 (default 2);
- `--hold E` — the **stable-stage duration** in effs (default 30).

The profile is then `N0,N1,N2,R1,R2 = 3, 3+E, 3+E+20, R, 1+(R−1)/2`: the
expansion leg is fixed at 3 effs, the compression leg at 20 effs, and the
final factor is halfway back from the peak — the defaults give
`--profile 3,33,53,2,1.5`, i.e. 53 effs overall ending at 1.5× the GSS
size. The move budget is a ceiling (`N2 × R1 × gates × 1.3`); the finished
profile ends the run (`ProfileDone`). Needs `FROZEN_DB_DIR` (hard error
without it; `GSS_MIX_ALLOW_EMPTY_STORE=1` bypasses for plumbing tests
only — a null store means zero re-encoding). DB guards are pinned:
degree 9, span 30, terms 1024/2048; CANON caps exported.

**4 — the split stage** (phase B part 1, `--split --split-stop`), at the
current shipped defaults: `p_join 0.8`, `split-reach-k 2` (bracket side ∝
remaining length, farthest of 2), fail-limit 100, 256 canaries, `k_max 12`,
no DB, no swap-family twists. Runs to g57 exhaustion; expect growth ≈
1 + comp-fraction (≈2× on a typical phase-A output, i.e. ≈3× the GSS
size), zero comp gates out, and the stage summary + canary deciles echoed
into `gss_mix.log`.

**5 — the crossing walk** (phase B part 2). Resumes `splitB.state` — the
split form's per-gate directions and litters carry over — and runs the pure
crossing economy (no twists, no DB) under the thermostat. Parameters, all
**PROVISIONAL until the X-panel pins them** (they are the panel's axes):

| knob | flag | provisional default |
|---|---|---|
| target factor over the split size | `--xr` | 2 |
| width-damper base B | `--xb` | 1.5 |
| width-damper threshold c | `--xc` | 3 |
| temperature (target/D) | `--xtdiv` | 25 |
| move budget | `--xmoves` | 12 × target |

`--moves` is ABSOLUTE on a resume; the driver reads the state's move
counter and adds the budget — do not pass raw fmix moves yourself.

**6 — fcompress**, whole-function (`--live-wires all`), the
attacker-computable greedy compressor as the final pass and honesty check.
The driver logs the residual (final/pre-compress gates); healthy mixed
material historically lands ≳ 90%.

## Sizing expectations (defaults, n = 128)

| stage | size |
|---|---|
| sandwich S | ~2·|C| + slicing ≈ 13–15k gates, 256 wires |
| GSS gadget | ~600–700k gates, 512 wires (production preset) |
| phase A out | 1.5× GSS ≈ 0.9–1.1M |
| split out | ≈ 2× phase A ≈ 1.8–2.2M (zero comp) |
| crossing out | `--xr` × split ≈ 3.6–4.4M at the provisional 2 |
| final | ≳ 90% of crossing out |

Runtimes are dominated by stages 3 and 5 (tens of millions of moves);
run production sizes on the server, exporting the store paths in the
launch environment (they are read at startup, not from any rc file).

## Notes and contracts

- **Determinism**: one master seed derives every stage's seed; stages 1+2
  regenerate bit-identically from `(n, seed)`, so `source_c.g57` recovers a
  past run's C. fmix stages are deterministic per (input, flags, seed) at
  fixed binary.
- **State v2**: stages 4–5 write resume states; stage 5's resume of the
  ended split stage never re-arms it (the tri-state contract).
- **Provenance**: a pipeline output promoted into `circuits/` or the
  mixing challenge must get a `CIRCUIT_GENERATION_INFO` entry — record the
  run dir, n, seed, binary commit, and the stage-5 knobs.
- **Do not** run stage 3 with an empty store outside plumbing tests, and
  do not port the driver to zsh (word-splitting silently changes flag
  passing).
- Stage-5 defaults will be updated from the X-panel results
  (`~/tds/xpanel_20260805` on the second server); until then treat any
  stage-5 output as calibration material, not a deliverable.
- **Stage-5 calibration objective** (2026-08-05): minimize the expansion
  factor `--xr` subject to decent ABSOLUTE spread — descendants per input
  gate and farthest-descendant distance in gates, not circuit fractions
  (`xpanel_spread.py` computes both from an arm's state file via the origin
  labels; merges attribute conservatively, the July convention). Runtime is
  expendable: if longer walks (or thermostat breathing) at a smaller target
  buy the same spread, prefer the smaller circuit — the linger-extension
  arms test exactly that trade.
