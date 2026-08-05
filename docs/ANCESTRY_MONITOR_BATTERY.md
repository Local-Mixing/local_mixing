# The Ancestry Recording Monitor:

*2026-07-30*

> Markdown rendering of `docs/ANCESTRY_MONITOR_BATTERY.tex`. The PDF is authoritative for figures, diagrams and display math.

## Abstract
The ancestry monitor is `fmix`'s instrument for *material
transport*: it tracks, for every gate of the evolving circuit, the set of
original input gates its material descends from. This note documents the
monitor itself — the recording rules and the definition and rationale of
every reported measure — and then gives a complete account of the
experiment battery run against it on 2026-07-29/30: the A/B/C
× twist-rate campaign, the bracket-placer factorial, the
recording-semantics isolation, the conditional census that audited the
instrument, the functional cross-checks that bounded its meaning, and the
slice-4 sanity battery. Conclusions: the monitor is sound (its headline
number was already immune to the suspected dilution artifact); its readings
are genuinely about transport, not about feasible-adversary legibility —
a 100× spread in `anc` across twist arms has no counterpart in
any functional channel measured; and its cross-run comparisons are valid at
matched effective work within one input universe, with saturation the main
caveat across universes.

## The monitor

### Recording layer

**Litters.** Every gate carries a *litter* tag: the replacement
event that created it. Input gates and born-random material are singleton
litters; splits and merges *propagate* the parent's id (a litter
fragments under churn rather than being reassigned); a DB splice stamps all
its products with one fresh id.

**Ancestor sets.** Per litter, a bitset over the m input-gate
indices: the union of the input gates whose material fed that litter.
Recording rules:

- input gate i starts as the implicit singleton {i} (litter ids
< m read as singletons unless an explicit entry overrides; fresh ids
start at m, so no aliasing);
- a DB splice's product litter gets the *union* of the consumed
window's litters' sets; catalogue merges likewise union their two parents;
- born-random material (`ORIGIN_SYNTH`: copy-pair inserts, twist
brackets) starts *empty* — it carries no input material;
- since placer v2 (commit `83ae0d11`), a g57 twist seam that
consumes context gates unions the consumed litters into the bracket word's
litter, exactly like a DB splice. (v1 dropped them; the isolation experiment
in § measures what that was worth.)

The union can only grow: unlike the scalar `origin` label — which a
mixed-lineage splice destroys (the `osyn` fraction measures exactly
that erosion) — an ancestor set never loses information. That is the
monitor's reason to exist: it keeps answering "what is this gate made of,
and how far has input material travelled" long after per-gate origin labels
have become meaningless.

**Cost envelope.** Exact mode (`–ancestors`) stores m bits per
litter and refuses to arm above 20,000 input gates. Sampled mode
(`–anc-samples K`, from the parallel monitoring session, commit
`8b73a98c`) tracks only K randomly chosen input gates ("tracers"),
K bits per litter, which removes the size ceiling; the tracer set is a
pure function of (m, K, `anc_sample_seed`) and survives resumes.
Calibration on an identical trajectory: the sampled estimate of `anc`
agrees with exact within 3% at K=256 and 8% at K=64.

### The measures

**`anc`** — mean ancestor-set cardinality over live gates,
*conditional on a non-empty set* (gates whose set is empty are skipped
— established by reading `anc_stats` and confirmed by the census of
§). Rationale: "what is a mixed gate made of" — the
width of the input window a gate's material has been assembled from.

**`ancspan`** — mean over the same gates of
( - )/(m-1) of the set's input indices: how far apart in the
*original* circuit the meeting material originated. The report line
carries the normalised form; the ancestry report prints the raw mean
("span (input gates)") plus log-bucketed histograms of per-gate
cardinality, span, and per-input fanout. Rationale: `anc` can grow by
merging neighbours; `span` only grows when *distant* input
material actually meets — it is the transport-distance reading.

**`fanout/input`** — per original gate, how many live gates
descend from it (the transpose view; its histogram exposes extinct inputs
vs. super-spreaders).

**Sampled-mode readouts** (`tracer_report`): `desc` = exact
per-tracer fanout (unbiased for mean fanout since each input has inclusion
probability K/m); from it `est anc` $= `desc`· m /
{size}recovers the exact-mode `anc` without storingm$ bits
anywhere. `cov` = fraction of 64 equal position-buckets of the
*current* circuit containing at least one descendant of a tracer;
`ent` = normalised entropy of the descendant positions over those
buckets. `cov`/`ent` are the security-facing pair — they ask
directly whether an adversary can *localise* where an original gate
went — and they are natively samplable, unlike `span`, which a
sample can only underestimate. In sampled mode the `anc=`/`ancspan=`
fields are deliberately left at zero rather than silently changing meaning.

## The experiment battery

Common setup unless noted: the 20,000-gate, 64-wire sample
(`g16_20k.mpmct1`, the first ≈3/7 of a Gray-fold gadgetized
sliced sandwich, n=16), seed 101, 2M moves, C schedule
(`–p-mix 0.1`), exact ancestry on, server directory
` /tds/menucal_20260728`. "Effective work" (eff) is
the trapezoid integral of moves per gate — the dose axis that removes
size-inflation bias. Legacy = shipped 3-CNOT swap-family brackets; g57 v1 =
anchor-first all-g57 brackets; v2 = v1 + edge slide + joint acceptance +
ancestry union.

### E1: A/B/C × twist rate, legacy vs. g57 v1

Fifteen runs: three schedules (A phased, B 20% MIX, C 10% MIX) ×
{no twist, legacy, g57 v1} × rates {0.002, 0.01}. C-arm finals:

| arm | size | anc | span | polf |
|---|---|---|---|---|
| no twist | 48,493 | 7,941 | 8,626 | 0.000 |
| legacy 0.002 | 133,665 | 425 | 1,111 | 0.422 |
| legacy 0.01 | 262,115 | 62 | 369 | 0.487 |
| g57v1 0.002 | 86,696 | 2,604 | 3,374 | 0.000 |
| g57v1 0.01 | 216,281 | 637 | 1,341 | 0.000 |

At matched eff ≈32: no-twist `anc` 3,961, g57 2,170
(-45%), legacy 425 (-89%). **Result:** both twist kinds suppress
transport at equal dose; the g57 brackets suppress 5–6× less while
also ending smaller. A and B schedules reproduce the ordering.

### E2: placer factorial

C arm, four placer configurations via the env kill-switches
(v1 / slide-only / retry-only / both), both rates. At 0.01: mean net gates
per seam 4.51 → 4.38 → 3.93 → 3.72; size $216{k} →
168{k}; `anc`637 → 921(v1→$ both). **Result:**
joint acceptance (window redraw unless both seams land) is the larger
transport lever; the slide composes.

### E3: v2 full trio

Six runs, all schedules × both rates, full v2. C-arm finals:
size 78,233 / `anc` 3,038 / span 3,895 at 0.002; 168,192 /
921 / 1,625 at 0.01. **Result:** v2 vs. legacy at 0.002 = 7×
the transport in a 41% smaller circuit.

### E4: recording-semantics isolation

Six runs with v2's placement disabled (both env switches) so only the
*ancestry-union* recording change differs from E1's g57 v1 arms.
Sizes match v1 within 1% (clean isolation); `anc` moves by only
+1.5–10% (C 0.002: 2,604 → 2,716; C 0.01: 637 → 698).
**Result:** of the v1→v2 `anc` gain, 10% is the
fairer recording, 90% is placement. The fix matters more as
consumption rates rise (v1 consumed only 0.8 gates/twist).

### E5: the conditional census (instrument audit)

Conjecture under test: the `anc` drop under twists is an artifact —
bracket gates carry no ancestry, so means are deflated. A state-file census
(litters + ancestor sets parsed directly) recomputed, for all fifteen E1
states: the unconditional mean, the conditional mean over non-empty sets,
and the empty-set fraction. Extract:

| state | anc_all | anc_cond | f_empty | f_synth |
|---|---|---|---|---|
| expC (no twist) | 7941.4 | 7941.4 | 0.000 | 0.993 |
| C_tw002 (legacy) | 378.6 | 424.6 | 0.108 | 0.998 |
| A_tw01 (legacy) | 37.8 | 69.4 | 0.455 | 0.997 |
| C_g57tw002 | 2527.4 | 2604.1 | 0.029 | 0.996 |
| C_g57tw01 | 505.6 | 636.6 | 0.206 | 0.998 |

**Result:** the reported `anc` *already equals the
conditional mean* (the reporter skips empty sets), so the simple dilution
form of the conjecture is refuted; conditioning changes no ordering. The
empty-lineage *population* is nonetheless real at high rates (up to
46% legacy at 0.01) — brackets do dilute the material, just not the
reported statistic. (f_synth≈0.99 everywhere is the old
`osyn` story — scalar origins die under ordinary churn — and is
what the set-based monitor exists to survive.)

### E6: functional cross-checks (what `anc` does not measure)

Affine-reconstruction plates of the input's prefix progress against each
mixed circuit, plus the stress battery, on the seven C-family arms:
degree-1 plates are identical across all arms (ρ = 1.000,
z ≈ 7.1) — and are provably *blind* to twists (every twist
type is an affine conjugation; the measure is affine-invariant). Degree-2
plates (not blind): at ceiling in every arm (ρ = 1.000,
z ≈ 4.5), depth differences within noise. Battery statistical
probes: prominence 0.26–0.31 in every arm, progress monotonicity
z = 7–9 everywhere. **Result:** a 100× spread in
`anc` across arms has *no counterpart* in any measured
feasible-adversary channel. `anc` is a transport gauge, not a
legibility gauge; twist-arm `anc` gaps must not be read as security
deltas. (The one twist-sensitive channel found — the syntactic
write-census gap — reversed direction on slice 4, E7, so it is not robust
either.)

### E7: slice-4 sanity battery

The same seven arms on the *middle seventh* of the source gadget
(6,640 gates containing the gadgetized N-band / C|D seam), at matched
moves per gate. Transport results (finals; anc as fraction of the input
universe in parentheses):

| arm | size (×input) | anc | span frac |
|---|---|---|---|
| no twist | 2.23 | 6,464 (0.97) | 0.99 |
| legacy 0.002 | 6.66 | 697 (0.105) | 0.229 |
| legacy 0.01 | 13.1 | 81 (0.012) | 0.064 |
| g57v1 0.002 | 4.13 | 2,927 (0.44) | 0.528 |
| g57v2 0.002 | 3.59 | 3,669 (0.55) | 0.642 |
| g57v2 0.01 | 7.95 | 1,140 (0.18) | 0.253 |

**Result:** size-growth ratios are slice-independent to a few percent;
the arm ordering replicates exactly; but all anc *fractions* run
1.4–2× higher than on the prefix sample because the
6,640-gate ancestor universe saturates (the no-twist arm reaches 97% of
it, span 99%). Legibility instruments, by contrast, do see the slice
difference: every arm is more affine-recoverable (deg-1 mean H
0.31–0.34 vs. 0.38–0.41; battery parity prominence doubled) —
the N-band's linear material survives as a softer region regardless of
twist treatment.

## Conclusions

1. **The monitor is sound.** Its headline statistic was already
conditional (immune to the empty-set dilution suspected of it), and its one
genuine recording unfairness — twist seams dropping consumed lineage —
is fixed and measured at only +1.5–10%. Readings replicate across
schedules and slices.
1. **What it measures is transport, and only transport.** Twists
genuinely suppress input-material transport per unit work (-45% g57,
-89% legacy at matched eff) — that is a true statement about material
flow. But no functional instrument (degree-1/2 affine, statistical probe
battery) registers any corresponding legibility change. Until an adversary
class is exhibited that tracks `anc`, twist-arm `anc` gaps are
economics, not security.
1. **Reading rules.** Compare `anc`/`span` only at
matched effective work; within one input universe, absolute values are
fine; across universes, use fractions and beware saturation (slice 4);
remember both are conditional means and check the empty fraction
(f_empty, or `carriers` in sampled mode) when twist or
insert rates are high; for localisation questions use the sampled
`cov`/`ent`, which have no exact-mode analogue.
1. **Instrument portfolio.** The battery only worked because the
monitor was cross-examined: the state-file census audited the reporter, the
affine plates exposed their own blindness (twists are affine), and the
degree-2/battery probes bounded the functional meaning. That
cross-examination pattern — instrument, audit, ground-truth — is the
reusable method here.
1. **Production.** Exact mode stays a small-input instrument;
`–anc-samples 256` is the production-scale monitor (3% calibration),
with `est anc` recovering this note's headline quantity and
`cov`/`ent` adding the localisation reading.

![E1/E3 combined: anc, span, size, dmin vs. effective work; A/B/C
colours, twist variant by line style.](../reports/ancestry_20260728/abc_twistrate_g57v2_20260730.png)

*E1/E3 combined: anc, span, size, dmin vs. effective work; A/B/C
colours, twist variant by line style.*
