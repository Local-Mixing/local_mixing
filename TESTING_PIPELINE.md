# Testing Pipeline — the Gadget Gauntlet

The gauntlet answers one question per gadget construction: **does an
adversary who sees the full execution of the gadgetized circuit learn
anything about the original circuit's intermediate values?** The security
properties being checked are the checklist in `GADGETIZATION.md`; this
document describes the machinery, how to run it, and the current results.

```
gauntlet_build.py     Python gadget builders (gg/big), ideal or band-fed randomness
        │  chain.bin (pre-gadget source) + gadget.bin + init.bin + buildmeta
        ▼
gauntlet_gen.rs       --gadget {none,ss,semi,file,...} [--mix] →
        │  init.cols (chain inputs + aux)   cols.bin (trace values)   meta
        ▼
gauntlet_audit.rs     six attacks, witnesses → hits.jsonl, human log → audit.log
        ▼
gauntlet_heatmap.py   six PNGs per cell (flagged cells show witnesses;
                      clean attacks render an empty grid titled "0 witnesses")
        ▼
gauntlet.py           orchestrator: all | report | maps | clean
```

## 1. Cells and orchestration

```
python3 gauntlet.py all --jobs 4        # build → audit → maps per cell
python3 gauntlet.py report              # regenerate reports/mx/REPORT.md
python3 gauntlet.py maps                # heatmaps only
python3 gauntlet.py clean               # delete reports/mx
```

A **cell** is `(chain length k, gadget, aux mode, mixing)`. Chains are
MPMCT g57 circuits on 8 wires (k ∈ {1, 2, 16} gates). `aux` is the
co-input distribution (`zero` / `random`); mixing (`mix` / `nomix`)
applies the in-process trace-circuit mixer after gadgetization.

Cell naming: `<gadget>_<aux>_<mix>` where gadget names embed the measured
full per-gate cost — `none`, `secretshare14`, `bandproduct92`,
`nonlinear193`, `nonlinear291` (193 mask-first decomposed to fan-in ≤ 2,
`gate_gadget_w2.py`), `behemoth939`, `behemoth1415` (939 over the weight-2
decomposition), plus `_{ideal,band0,band16}`-style pool arms for the four
Python gadgets (§5). Rust gadgetizers are addressed by `rust_gadget` in
the arm table; the Python builders (gg = 193, gg2 = 291, big = 939,
big2 = 1415) are canonical.

## 2. The six attacks

Targets: per original (pre-gadgetization) gate, the five logical values
`a, b, c_old, f, c_new`; plus a `NULL` column (a same-class trace value
that carries no secret) as the noise reference.

| attack | question | pass condition |
|---|---|---|
| `a1` | does any trace value equal a target bit exactly? | 0 hits |
| `xrows` | is a target an affine function of any single wire state `G_j`? (Gaussian elimination per row) | 0 hits |
| `xtrace` | is a target an affine function of the *entire trace* (const-1 + init wires + every gate's flip/newval)? Fit on ⌈(F+256)/64⌉ samples, verified on 2048 held-out | 0 hits |
| `w1` | any significant correlation with a single trace value? | 0 flagged |
| `w2` | …with any xor/and/or pair of trace values? | 0 flagged |
| `w3` | …with any xor/and/or triple? | reported (known residual) |

Correlation attacks are **NULL-referenced**: a target is flagged iff its
max |cov| exceeds both `NULL`'s max and the 6σ threshold for the sample
count. Correlation sample tails: 16384 (k=1), 8192 (k=2), 4096 (k=16),
16384 for `nonlinear193` at k=16.

Exact attacks assume the adversary knows the full *structure* of G (all
trace values) but not the input; the sampled-inputs limitation means a
chain with only 2^n distinct contexts is tested per-context — see §5 for
why the band pool is therefore keyed by extra "neighbor" columns.

## 3. Current results (full matrix, `reports/mx/`)

Verdict columns are per-attack hit/flag counts; `clean` = the property
list holds (a1/xrows/xtrace = 0, w1/w2 unflagged).

| gadget | k | nomix (a1/xrows/xtrace/w1/w2/w3flag) | mix |
|---|---|---|---|
| `none` | all | **broken**: xrows/xtrace find every stored value; a1 matches when the chain touches all wires | same (mixing cannot add sharing) |
| `secretshare14` | 1, 2, 16 | **broken**: a1 + xrows + xtrace + w1 + w2 + w3 — values live on single wires between refreshes; mixing does not rescue | same |
| `bandproduct92` | 1, 2, 16 | **broken**: xrows + xtrace everywhere; a1 + w1 + w2 at k=16 — the product-share band is a mask source, not a wire-state hider | same |
| `nonlinear193` | 1, 2 | **clean** (w1=w2=0; w3 residual = 4 flagged) | mostly clean; w2flag ≤ 5, w1flag ≤ 2 at k=2 |
| `nonlinear193` | 16 | clean except w3flag = 76 (the mask-cascade leak: three mask flips XOR to `b·B(s)`, B 3/4-biased) | w3 demoted by the mixer: w2flag = 62, w3flag = 79 |
| `nonlinear939` | all | **fully clean everywhere** | fully clean everywhere |

Headline: the full-cascade 939-gate gadget passes the entire property
list at every scale, with or without mixing; the 193-gate folded gadget
passes properties 1–5 exactly, with a tracked weight-3 residual that
mixing can demote to weight 2 (mechanism in `README_gate_gadget_v2.md`
§7). Every non-gadget control fails as expected — the battery has teeth.

Heatmaps: each cell directory has six PNGs (`heatmaps/{a1,xrows,xtrace,w1,w2,w3}.png`).
Y axis = targets (per gate: a, b, cold, f, cnew; NULL on top), X axis =
trace values (init wires | per-gate flip/newval), dashed line at the
init/gate boundary, gray ticks at gadget-block boundaries. A clean attack
renders an empty grid titled "0 witnesses".

## 4. Costs and scaling

| gadget | gates per source gate | fixed infrastructure |
|---|---|---|
| none | 1 | — |
| secretshare14 | 14.48 expected (randomized: SG menu 36/7 + 2× RG draw 14/3) | ~220 (Z-bookends + 7-CNOT share setups) |
| bandproduct92 | 92.4 measured (randomized: fold menu, gathers, rolls, top-ups) | ~370 (band fill + injection) |
| nonlinear193 | 193 exact | 0 |
| nonlinear291 | 291 exact (weight-2 decomposition: 2 dec + 24 cache + 1 temp wires) | 0 |
| behemoth939 | 939 exact | 0 |
| behemoth1415 | 1415 exact (291+288+291+291+127+127; weight-2 with exact per-sub-gadget mask classification) | 0 |

At k=16: `none` F=41 trace features; `secretshare14` F=914;
`bandproduct92` F=3890; `nonlinear193` F=6450 (32126 mixed);
`behemoth939` F=30378 (63730 mixed); `nonlinear291` F=9553;
`behemoth1415` F=46971. Runtimes are dominated by
`xrows`/`xtrace` Gaussian elimination (none ~4 s → big ~49 s at k=16,
single-threaded); mixing inflates `bandproduct92` k=16 to ~1200 s.

## 5. Pool arms — borrow wires and band-fed randomness

Production gadgetizers never call a raw RNG inside the gadget: all
randomness arrives on dedicated **borrow wires** whose values come from
the band's avalanche pool. The pool arms test whether that deployment
mode is as safe as ideal randomness.

**The contract** (enforced at build time):

1. **Strict borrows** (chaff wires; the big gadget's gA mask share) are
   consumed *only* inside their own gadget context — a structural check
   asserts no strict borrow is ever read outside it.
2. **Transitory borrows** (the 5-wire re-share block R) legitimately
   become output shares (`share1' = R`) consumed downstream as *data* —
   that is data flow, not a leak.
3. Mask-vs-operand independence is statistical, and is what the battery
   tests.

**Feed modes.** `--pool ideal` = each borrow wire gets an independent
uniform bit per input (the theoretical assumption). `--pool band` = the
faithful port of the production band fill (`emit_band_fill_nl_blind`,
fill_nl=2, blind layers numeric): pivot-uniform marginals
(max |bias| ≈ 0.005), all columns built by the same process.

**The 2^n pitfall.** If the band is keyed only by the 8 chain inputs
there are just 256 contexts: pool columns become *functions of the
input*, exact attacks trivialize vacuously, and the NULL calibrates to
the 0.2557 saturation floor — a run that "passes" has tested nothing.
The pool is therefore keyed by the **full pipeline input** (8 data + 120
fresh "neighbor" columns, `--pool-keys`), giving a 2^128 context space —
exactly the production sandwich situation (n=128, the chain reads 8
wires).

**NULL follows the pool.** In band mode the builder ships one extra band
column that no gate consumes as the NULL reference — a fresh-random NULL
would sit √(contexts) below the input-keyed noise floor and flag
everything.

**Verdict (measured):** band-fed randomness is statistically
indistinguishable from ideal across the whole battery — identical
flag/hit counts at k=1 and k=2 (including the two w1 flags at k=2, which
are mixer artifacts), slightly *fewer* flags at k=16. Blind layers make
no measurable difference for these gadgets. The w1/w2 demotion under
mixing is a **mixer artifact** (merge rules don't respect cascade
structure), not a pool artifact; the behemoths are immune to the w3 leak
itself (the cascade is recursively masked there).

The weight-2 arms (`nonlinear291`, `behemoth1415`, measured k=1/2/16,
mix on/off, ideal and band16 pools) behave exactly as their parents:
`nonlinear291` reproduces `nonlinear193`'s profile (nomix: w3-only
cascade flags with the same 0.75-signature witnesses; mix: demotion to
w2/w1, reaching w1=30/80 targets at k=16 — the decomposition emits more
mask-carrying coordinates for the mixer to merge, so the demotion goes
further) with **no new leak class**: the mask-first decomposition never
exposes a bare operand monomial (w1=w2=0 in every nomix cell).
`behemoth1415` matches `behemoth939`: clean under mixing at every k
(one noise-edge w2 excursion at k=2, cov 0.070 vs the 6σ = 0.066
threshold); nomix shows only noise-edge w3 excursions at the 6σ
threshold (max cov ≈ 0.095 vs the 0.19–0.75 range of real leaks),
not replicating across seeds (a seed-555 rerun of the k=16 nomix cell
flags 0). Results: `reports/mxw2/REPORT.md` (24 cells: the two new
gadgets x k=1,2,16 x mix on/off x ideal/band16).

## 6. Implementation notes

- Attack engine in Rust (`src/bin/gauntlet_gen.rs`,
  `src/bin/gauntlet_audit.rs`); orchestration in Python
  (`gauntlet.py`, `gauntlet_build.py`, `gauntlet_heatmap.py`).
- Build: `cargo build --release --bin gauntlet_gen --bin gauntlet_audit`.
- Heatmaps need matplotlib: `nix-shell shell-gauntlet.nix --run "python3 gauntlet.py maps"`.
- The builder's `--samples` must equal gen's computed count
  (fit + 2048 verify + corr tail) or gen panics on a column-size check —
  the orchestrator keeps them in sync.
- Reports regenerate from `audit.log` files (robust to cell renames):
  `python3 gauntlet.py report`.
