# `ssg` — Parameters & Environment Variables

Reference for the `ssg` (shuffle-shoot-shuffle **gen**erate-mix) subcommand of
`local_mixing_bin`: every CLI flag and environment variable, with its effect and
default.

```
local_mixing_bin ssg -n <wires> -m <gates> -x <x> \
    -s <source.txt> -d <dest.txt> -r <rounds> [flags...]
```

## What it does (pipeline)

1. **Gadgetize / feistalize** the source circuit (optional, expands the wire
   count — see `--feistalize`/`--gadgetize`). The expanded circuit is written to
   `--gadget_path`.
2. For each **round** (`-r`): run the shooting game (Stage B/C — collision
   shooting + SAMF hiding) to inflate/obfuscate, then **compress** back down.
   Each round's circuit is written to `<dest>round<k>.txt`.
3. After the last round, the final circuit is written to `-d`.

Functional equivalence to the original is checked at the end (and per round with
`--equality_check`); a mismatch panics with `The functionality has changed`.

---

## Required parameters

| Flag | Effect |
|------|--------|
| `-n, --n <usize>` | Number of wires in the **source** circuit. |
| `-m, --m <usize>` | Number of plain SAMF gates inserted per shooting round (0 disables plain-SAMF insertion). |
| `-x, --x <usize>` | Shooting/expansion intensity (gates shot per pass family). |
| `-s, --source <path>` | Source circuit file. |
| `-r, --rounds <usize>` | Number of shuffle→compress rounds. |
| `-d, --destination <path>` | Output (final mixed) circuit. Round files are `<dest>round<k>.txt`. |

## Start-of-run transform (pick at most one)

| Flag | Effect |
|------|--------|
| `--feistalize` | Feistalize at start: **input becomes 3n wires** (middle n-wire block holds `y ⊕ C(x)`). This is the usual mode; it's why circuits balloon (e.g. n=128 → round-1 ≈ 13M gates). |
| `--gadgetize` | Gadgetize at start: **input becomes 2n wires**. |
| `--slice-zero` | Before feistalization, insert M preserving the `(y,z)=0` slice and randomizing x off it. |
| `--slice-zero-random` | Same, but preserves a **public random** `(y,z)` slice. |
| `--slice-zero-hardcoded` | Hardcoded-M variant preserving the `(y,z)=0` slice. |
| `--slice-zero-random-gates <usize>` | # random M gates for `--slice-zero-random` (default `32n`). |
| `--slice-zero-hardcoded-rounds <usize>` | # hardcoded-M rounds (default `1`). |
| `--gadget_path <path>` | Where the gadgetized/feistalized circuit is written (default `./gadgetized/{source filename}`). |

## Shooting / SAMF tuning

| Flag | Default | Effect |
|------|---------|--------|
| `--type_attempts <usize>` | `1` | Distinct SAMF gate types to try per collision before giving up. Only **4** negation types exist, so `>4` is a no-op; to raise the hidden-SAMF *rate*, use the `SAMF_HIDE_PAIRS` env var (below) instead. |
| `--shooting_times <usize>` | `1` | Shooting rounds; each = one collision game + one plain-SAMF insertion before the final unsamf. |
| `--rg-frequency <usize>` | `2` | # SG gadgets between each RG gadget (2 = two SGs then one RG). |
| `--gates_ahead_expand <usize>` | `2` | Gates per curated **expansion** window (2 = pair; >2 shrinks by 1 on a curated-DB miss down to the pair). |
| `--gates_ahead_samf <usize>` | `3` | Context gates prepended to the 3 SAMF gates when hiding a SAMF. |
| `--full-shuffle` | off | Insert n SAMFs between every gate after each round's shooting insertion, before compression. |
| `--full-shuffle-early` | off | Insert n SAMFs between every gate once, right after gadgetization, before the main loop. |
| `--egg` | off | Use the expansion game (expand_loop 2×) instead of the shuffled shooting game. |
| `--single-end` | off | Accumulate SAMFs/NOTs across **all** rounds (functionality broken between rounds) and undo in one pass after the last round, before its compression. |
| `--interleave` | off | Use interleaving. |

## Gen-mode (Stage B) shaping — `ssg`-specific

| Flag | Default | Effect |
|------|---------|--------|
| `--min-gen <usize>` | `1` | Stage B keeps shooting until every gate's *generation* ≥ this. |
| `--min-gen-fraction <f>` | `0.99` | Stop once this fraction of gates reach `--min-gen`. |
| `--pass-length <usize>` | `0` | Max successful replacements per shooting pass (0 = unbounded). |
| `--max-passes <usize>` | `100000` | Safety cap on shooting passes per round. |
| `--max-fanout <usize>` | `50` | Hard cap on per-gate fanout. |
| `--min-median-leeway <usize>` | `10` | Raise low-leeway gates when median leeway < this. |
| `--samf-target <usize>` | `0` | If a round hides ≥ this many SAMFs, skip plain-SAMF insertion (m→0) for later rounds (0 = disabled). |
| `--outgoing-rank-script <path>` | — | Rhai script providing `rank(cands)` for outgoing window selection (#11). |
| `--incoming-rank-script <path>` | — | Rhai script providing `rank(cands)` for incoming replacement selection (#9). |

## Compression / output

| Flag | Default | Effect |
|------|---------|--------|
| `--light-compression` | off | Between rounds, stop compressing once the circuit is ≤ half its max (post-shooting) size. |
| `--equality_check` | off | Run probabilistic equality/functionality checks after each round and at the end. **Recommended while debugging equivalence.** |
| `--record` (`record_replacements`) | off | Record replacements applied. |
| `--track-survivors` | off | Track "survivor" gates across passes. |

---

## Environment variables

### Debugging / equivalence (most relevant right now)

| Var | Default | Effect |
|-----|---------|--------|
| **`STAGEC_CHECK`** | unset | If set (any value), enables Stage-C equivalence instrumentation: materializes the accumulated state after every pass/flush and verifies it still equals the input. On a break it logs the exact pass/flush (`[STAGEC GLOBAL BREAK]`, `[RIGHT-PASS BREAK]`, `[LEFT-PASS BREAK]`, `[STAGEC OUTPUT BREAK]`). **Use on validation runs** to confirm the >64-wire fix closed the hole. Adds overhead. |
| **`ABSORB_NOTS`** | unset | If set, absorbs pending NOTs into the curated lookup (#10/Stage F). **Has a known latent correctness bug — keep UNSET** while chasing equivalence failures. |
| `SURVIVOR_LOG_EVERY` | — | Log survivor stats every N passes (needs `--track-survivors`). |
| `COMPRESSION_TRACE` | unset | If set, trace compression steps. |
| `COMPRESSION_TRACE_MS` | — | Minimum duration (ms) for a compression step to be traced. |
| `RUST_BACKTRACE` | `0` | Set `1` to get a backtrace on any panic (e.g. `The functionality has changed`, the LMDB `NotFound` panic). |

> **Related code-side counter:** `OVERSIZED_CANON_SKIPS` is printed at end of run
> ("Oversized-canon (>64-wire) lookups skipped: N"). N>0 means the >64-wire
> overflow path was exercised and is now being safely skipped by the fix.

### Pass shaping

| Var | Default | Effect |
|-----|---------|--------|
| `CENTRALIZE` | `0` | `=p`: prioritize passes starting at gates in the central p% index band with generation `min_gen+1`; those are shot **bidirectionally**. 0 = off. |
| `STABLE_MAX` | `6` | Compression convergence window: stop when total reduction over the last `STABLE_MAX` iterations is < 50 gates. |

### Hidden-SAMF rate & shooting parallelism (new)

> Requires a binary built from current HEAD. `SAMF_HIDE_PAIRS` shipped in the
> Jul-8 build; `SHOOT_PARALLEL`/`SHOOT_PROFILE` need a rebuild of the checkout.

| Var | Default | Effect |
|-----|---------|--------|
| **`SAMF_HIDE_PAIRS`** | `1` | `=k`: try up to `k` random swap-**pairs** per curated expansion when hiding a SAMF (first success wins). This — not `--type_attempts` — is the real lever on the hidden-SAMF insertion **rate** (`--type_attempts` caps at ~14% because only 4 negation types exist). Calibrated on gadgetized cdcnot: `k=2` → 22.5%, **`k=3` → ~30%**. First-success, no SAT dependency. |
| **`SHOOT_PARALLEL`** | `1` | `=k`: batch up to `k` non-overlapping min-generation RIGHT-pass windows (spaced ≥ `WINDOW_CAP`=8192 apart), shoot them concurrently, then merge serially (right-to-left). **Equivalence-preserving** — verified via `STAGEC_CHECK` (clean) + standalone `equal -n`. Speedup ~**2.5×** at `k=32` today (only the else-branch is parallelized; `CENTRALIZE` bidirectional passes stay serial). ⚠️ **Mixing-quality A/B not yet done — keep OUT of delivered runs until validated; fine for benchmark/throwaway runs.** |
| **`SHOOT_PROFILE`** | unset | Diagnostic only (no behavior change). At each stage-B end prints the shooting phase split `[shoot-profile] core / reconcile / merge / flush` and, when `SHOOT_PARALLEL>1`, the achieved parallel width `[shoot-parallel] avg_anchors_per_batch`. |

### Gadget symmetry

| Var | Default | Effect |
|-----|---------|--------|
| `SymmetricCD` | unset | If set, use symmetric control/data gadget construction. |
| `SymmetricG` | unset | If set (and `SymmetricCD` set), additionally symmetrize G. |

### SAT-hardening family (advanced; off unless enabled)

Master switch **`SAT_HARDEN`** turns on cone-awareness, BCP, and compress-protect
together. Individual switches: `SAT_CONE_AWARE`, `SAT_BCP`, `SAT_COMPRESS_PROTECT`,
`SAT_PROBE`. Sub-knobs (only matter when the corresponding feature is enabled):

| Var | Default | Effect |
|-----|---------|--------|
| `SAT_CONE_MIN_FRACTION` | `0.0` | Minimum cone fraction threshold. |
| `SAT_HIDDEN_SAMF_CANDIDATES` | `16` if cone-aware else `1` | # hidden-SAMF candidates considered. |
| `SAT_BCP_MIN_RESISTANCE` | `0.0` | Min BCP resistance to keep a replacement. |
| `SAT_COMPRESS_PRESERVE_DELTA` | `0.0` | Compression-protect delta. |
| `SAT_EXPAND_MIN_DELTA` | `0.0` | Min expansion delta. |
| `SAT_PROBE` / `SAT_PROBE_SOLVER` | off / `kissat` | Enable SAT probing; choose solver. |
| `SAT_PROBE_FREQUENCY` | `1` | Probe every N. |
| `SAT_PROBE_WINDOW_GATES` | `20000` | Probe window size (gates). |
| `SAT_SCORE_SLACK` | `4` | Scoring slack. |
| `SAT_SCORE_SEED` | `0x5a7ad00dcafef00d` | RNG seed for scoring. |

---

## Actual current run (n=128, m=900) — exact parameters

```
./target/release/local_mixing_bin ssg -n 128 -m 900 -x 10 -r 3 \
    -s work/ssg_n128m900/random_n128_m900.txt \
    -d work/ssg_n128m900/mixed_n128_m900.txt \
    --min-gen 3 --min-gen-fraction 0.95 --pass-length 100
```

Note: this run uses **no `--feistalize`/`--gadgetize`** and **no special env
vars** — the circuit inflation comes entirely from the gen-mode shooting
(`--min-gen 3`, `--min-gen-fraction 0.95`, `--pass-length 100`). Everything else
is default (`--type_attempts 1`, `--shooting_times 1`, `--rg-frequency 2`, …).

## Recommended launch (survives SSH disconnects)

```bash
cd ~/local_mixing
setsid nohup ./target/release/local_mixing_bin ssg \
    -n 128 -m 900 -x 10 -r 3 \
    -s work/ssg_n128m900/random_n128_m900.txt \
    -d work/ssg_n128m900/mixed_n128_m900.txt \
    --min-gen 3 --min-gen-fraction 0.95 --pass-length 100 \
    --equality_check \
    > work/ssg_n128m900/ssg_run.log 2>&1 < /dev/null & disown
```

- **No `timeout` wrapper**, output to a logfile (per-phase counts are tailable).
- For a **validation run** of the equivalence fix, prepend `STAGEC_CHECK=1` (and
  keep `ABSORB_NOTS` unset). Add `--equality_check` to verify per round.

## The >64-wire equivalence fix (branch `ssg-gen-mix`)

`Monomial = u64`, so a replacement window touching **>64 distinct wires** used to
overflow (`1u64 << i`, i≥64 aliases `x_0`), producing a wrong canonical key that
could spuriously match a *non-equivalent* curated-DB entry → occasional
equivalence break. `canonicalize_polys_single[_neg]` now returns empty polys for
such windows; every DB-lookup caller treats empty as a clean miss (window left
unchanged), preserving equivalence. Regression test:
`canonicalize_skips_window_over_64_distinct_wires`.
