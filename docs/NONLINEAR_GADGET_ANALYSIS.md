# Weaknesses in the nonlinear product-share gadgetization

Status: **consolidated research review, 2026-07-27 UTC.**

**Reading rule.** Section 0 is the current, citable assessment. Sections 1--11
are retained as the historical Claude/Codex review record. They are valuable
for provenance, but several of their claims were superseded by the accepted
v05 run and by the endpoint-safe reanalysis below. A statement in Sections
1--11 that conflicts with Section 0 is historical, not the current conclusion.

This revision changes analysis and adds local research evidence; it does not
change the nonlinear construction. The current source tree already contains
unrelated construction changes made before this review.

This document is written to be self-contained for an outside reader with no
repository access.

---

## 0. Consolidated assessment after the accepted v05 pipeline

### 0.1 Executive conclusion

The earlier headline -- that the nonlinear diagonal disappeared mainly because
the affine ruler stopped one degree too soon -- is **not established at
production scale**. The defensible conclusion is narrower and more interesting:

> The current `[3,3]` nonlinear product-share construction has no supported
> exact degree-1 **progress ridge** at the sampled interior prefixes of the
> accepted five-round SSS or 15.26x fmix pipeline. Post-fmix is exactly flat;
> post-SSS has one isolated one-target-bit-equivalent low-error cell but no
> chronology. Construction-stage tests also find no exact band-restricted
> degree-2 ridge for either `[3,3]` or `[2,2,3,3]`. This is real evidence in
> favor of the construction, not merely a plotting failure. It is not a
> general security result: a cheap approximate two-wire reader shows a strong
> timing trace on one older full-mix lineage, exact degree-3 recovery is
> demonstrated only on a small archival raw build, and several structural,
> helper, reproducibility, and normalization attacks remain unresolved.

Three categories must therefore remain separate:

1. **Exact bounded-degree reconstruction.** Current evidence is favorable at
   degree 1 through full mixing and at restricted degree 2 at construction.
2. **Approximate/statistical chronology.** Preliminary older production
   evidence is unfavorable, but needs an honest held-out rerun under the
   current fixed-slice contract.
3. **Blind distinguishing and SAT hardness.** Neither is decided by a
   known-source heatmap, and no causal relationship among the three metrics has
   been established.

### 0.2 Current construction and accepted artifact

The accepted current run is
`nonlinear_tdp4n_sss5_fmix152_noopfix_v05`, not the failed v03 precursor.
Its base circuit is a pinned 128-wire, 1,000-gate all-G57 circuit. The
random-slice TDP4n stream has 512 carrier wires; the production `[3,3]`
product-share encoding appends a 46-wire nonlinear band, for 558 wires total.

| Stage | Wires | Gates | Validation |
|---|---:|---:|---|
| pinned source `C` | 128 | 1,000 | SHA-256 pinned |
| nonlinear TDP4n construction | 558 | 72,831 | fixed-slice semantics passed |
| after five 100,000-move SSS rounds | 558 | 69,836 | 1,536 structured slice cases passed |
| after nonlinear fmix | 558 | 1,065,824 | 12,000,000 moves; randomized all-wire check every 100,000 moves |

The final expansion is `15.261813391x` relative to the post-SSS circuit.
Fmix accepted 4,027 nonlinear frames and produced 50,697 crossings. An
independent checker subsequently evaluated 512 `X` samples against all 16
retained helper patterns at each of the constructed, post-SSS, and post-fmix
stages: 24,576 endpoint cases in total, all passing both base semantics and the
fixed-slice middle-output contract.

The fixed public slice is:

```text
Y = 0x2587d0c6d5dbade6b97eae1e5a026f25
Z = 0x30bcf03361057f25dafdbf7a287f5a80
```

Fixing `Y,Z` does **not** fix `W` or the appended nonlinear band. Every
experiment must name its helper ensemble. The accepted heatmaps randomize
wires `384..557` independently per sample lane.

The old v03 failure remains useful negative evidence about complemented empty
identity handling, but it is superseded as a correctness verdict. The v05
canonicalization fix crossed the old failure point at move 2,400,000 and
completed the full verified run.

### 0.3 Evidence ledger

| Question | Current evidence | Verdict |
|---|---|---|
| Does current fmix preserve its 558-wire input function? | randomized full-state checks throughout 12M moves, plus finite independent endpoint checks | **no mismatch detected in accepted v05 checks** |
| Does current exact degree 1 recover an interior progress ridge after SSS or fmix? | three fixed-slice seeds per stage, re-read with a 20--80% port-aware core | **no detected ridge** |
| Does band-restricted degree 2 recover current `[3,3]` at construction? | 6,144 samples, 1,594 regressors, train/holdout split | **no detected ridge** |
| Does band-restricted degree 2 recover `[2,2,3,3]` at construction? | current 111,745-gate TDP and reproduced historical 374,478-gate sandwich | **no detected ridge** |
| Does a cheap approximate reader recover chronology after a full mix? | stored older `[3,3]` `hmap_stat` matrices | **strong preliminary signal** |
| Does exact degree 3 recover a current production artifact? | no current fixed-slice production run | **unknown** |
| Do roll fingerprints yield actual swaps after mixing? | raw small candidate fingerprints only; no ground-truth precision/recall | **unknown** |
| Can helper freedom collapse the final mixed representation? | raw triangular, input-adaptive argument; no final-artifact recovery | **unknown** |
| Is the construction reproducible from its advertised seed? | same-seed production-scale replay in this revision | **no** |
| Do any of these findings predict Painless SAT time? | no controlled correlation experiment | **unknown** |

### 0.4 Corrected interpretation of the accepted v05 heatmaps

The legacy renderer reported `depth=0.1316, rho=1.000` for both v05 stages.
That reading is wrong. It joins forced input/output ports through rows whose
interiors are tied at `H=0.5`; deterministic tie-breaking turns the endpoints
into an apparent diagonal.

This revision re-read all six accepted matrices with the established
port-aware rule: retain only the sampled 20--80% source-prefix and
mixed-prefix core, use row-median minus row-minimum prominence, and do not
define rho without sufficient informative coverage.

The threshold must be stated carefully. `H` is averaged over 128 target bits.
Changing one bit from unpredictable error `0.5` to exact error `0` changes
mean `H` by `0.5/n = 0.00390625`. The renderer's historical `1/n` threshold is
therefore a **two-target-bit-equivalent** threshold, not a one-bit threshold.
Both are reported:

| Stage | Seeds | Core mean `H` | Core minimum `H` | Mean prominence | Rows >= one-bit equivalent | Rows >= two-bit equivalent | Tie-safe rho |
|---|---:|---:|---:|---:|---:|---:|---|
| after five SSS rounds | 3 | 0.499997864 | 0.49609375 | 0.000126 | **1/31 (3.23%)** | 0/31 | N/A |
| after fmix | 3 | 0.500000000 | 0.50000000 | 0.000000 | 0/31 | 0/31 | N/A |

All three post-SSS matrices are byte-identical. Their only sampled-core
departure is at source prefix 800 and mixed prefix 51,336:
`H=0.49609375`, exactly one target-bit-equivalent below the `0.5` row
background. One isolated informative row cannot define a monotone chronology.
The supported conclusion is therefore **no detected exact degree-1 progress
ridge**, not “every post-SSS core cell is flat.” The post-fmix sampled core
really is exactly `H=0.5` at every cell in every seed.

The current `[3,3]` constructed-circuit degree-2 map is equally flat:

| Predictor | Core mean `H` | Core depth | Coverage | Bottom-right `H` |
|---|---:|---:|---:|---:|
| degree 1 | 0.500000 | 0.000000 | 0% | 0 |
| degree 2 on band `512..557` | 0.500000 | 0.000000 | 0% | 0 |

Those two 51x99 matrices are byte-identical. This is a restricted
degree-2 result: 558 linear terms, one constant, and all
`C(46,2)=1,035` band products. It is not an exhaustive quadratic adversary
over all 558 wires, and the v05 **post-fmix** artifact has not yet received the
same degree-2 fixed-slice run.

Two `[2,2,3,3]` controls give the same qualitative result:

| Construction | Gates / wires | Product set | Interior result |
|---|---:|---:|---|
| current random-slice TDP4n research preset | 111,745 / 576 | all pairs on 64-wire band | exactly `H=0.5`, depth 0, coverage 0% |
| exact historical sliced-sandwich reproduction | 374,478 / 576 | all pairs on 64-wire band | exactly `H=0.5`, depth 0, coverage 0% |

These controls make the endpoint artifact especially clear: the old renderer
still prints `rho=1` while the informative core has no low-error cell.
They are not interchangeable contracts: the current TDP test fixes nonzero
public `Y,Z` and randomizes every helper wire `384..575`, whereas the historical
nested sandwich fixes every non-`X` input to zero.

### 0.5 The surviving approximate chronology result

The strongest unfavorable production-scale observation remains the older
`hmap_stat` result. That reader searches the best constant, one wire, or XOR of
two wires rather than asking exact GF(2) span membership.

| Artifact | Gates / wires | Median interior peak agreement | Matched floor | Row/peak-column rho |
|---|---:|---:|---:|---:|
| `stat_final.bin` | 762,362 / 558 | 0.7034 | 0.5257 | 0.9948 |
| `stat_x3input.bin` | 1,428,155 / 558 | 0.6915 | 0.5252 | 0.9956 |

This is evidence that **exact affine hiding and approximate timing hiding are
different properties**. It should not be merged with the v05 exact maps:

- it uses a different artifact lineage and helper contract;
- it is a known-source alignment test;
- candidate selection and scoring reuse the same samples;
- it covers only a subset of target bits and does not retain winner identities;
- it has no mismatched-reference control or source/seed replicas.

The previously quoted `p<10^-5` from freely permuting row labels is also too
confident because adjacent prefix rows are strongly serially correlated
(`r_lag1` approximately 0.989 and 0.984). A circular-shift null preserves that
serial structure. The observed rho exceeds every nonzero shift, but with 25
retained rows the attainable resolution is only `p=1/25=0.04`. The timing
trace remains visible and concerning; its inferential precision was
overstated.

The decisive rerun is a fixed-`Y,Z`, randomized-helper, train/holdout
`hmap_stat` on the actual v05 pre/post artifacts, over all target bits, with
winner identities, mismatched references, multiple source circuits, and a
serial-preserving null.

### 0.6 What degree 3 does and does not establish

The small archival Build A experiment found an exact monotone ridge when
degree-2 and degree-3 monomials over the true 12-wire band were added. This
shows a plausible mechanism on one raw, no-roll, obsolete `[2,3,3]` build.
It does **not** show that:

- the relation transfers to the current `[3,3]` construction;
- a blind tracker recovers the semantic band under roll;
- arbitrary SSS/fmix prefixes remain cubic in final circuit coordinates; or
- a production-scale generic cubic fit is computationally cheap.

The archival scripts named in Section 8 were stored only in `/tmp` and no
longer exist. Build A is identified by a timestamp rather than an exact binary
hash. Its numeric results should therefore be treated as preliminary until
reproduced from a pinned tool and artifact.

Correct regressor counts for constant + all linear wires + all band pairs and
triples are:

| Layout | Regressors |
|---|---:|
| current 558 wires, `B=46` | `1+558+C(46,2)+C(46,3) = 16,774` |
| old 568 wires, `B=56` | `1+568+C(56,2)+C(56,3) = 29,829` |
| 576 wires, `B=64` | `1+576+C(64,2)+C(64,3) = 44,257` |

At a 75/25 split, training samples must exceed the regressor count, so the
64-wire experiment needs more than 59,010 total samples; 65,536 is a sensible
minimum. Regressor **count** scales as `O(B^3)`. The present generic GF(2)
basis implementation does substantially more than linear work in that count,
so calling the attack itself “subquadratic” is unjustified. A syntax-informed
decoder may be much cheaper, but that is another hypothesis to test.

The cached `origin/ssg-gen-mix-clean` `hmap_affine` already supports
`--degree 3` and an explicit product-wire list (named
`--deg2-wire-list`). The actual tooling gap is to merge that support with the
current fixed-`Y,Z`, randomized-helper reader, add dynamic semantic-band
tracking, and avoid rebuilding a generic high-dimensional basis independently
at every cell.

### 0.7 Weakness register

The following register separates evidence from speculation.

| ID | Potential weakness | Evidence level | Current assessment |
|---|---|---|---|
| W1 | advertised seed does not determine the artifact | **production-scale measured** | confirmed research-validity defect |
| W2 | one public `O(sqrt(n))` nonlinear source band feeds all masks | **source-level fact** | real shared latent bottleneck; not “B bits of security” |
| W3 | `resource` changes mask labels and `roll` changes locations, but neither refreshes semantic band values | **source-level fact; small raw signature evidence** | plausible tracking surface; full-pipeline recovery open |
| W4 | TDP leaves `W` and band helpers free | **source/test contract fact** | real gauge/optimization surface; threat-model dependent |
| W5 | triangular fill admits an input-adaptive helper assignment that fixes desired band values | **structural argument; archival small raw measurement** | relevant to witness/SAT semantics; not a universal fixed helper or final-artifact break |
| W6 | fixed fold grammar, cadence, twins, seams, and endgames expose syntax | **source fact; preliminary raw measurements** | concerning, but source recovery precision/recall is unmeasured |
| W7 | equal-function source presentations remain statistically distinguishable | **small preliminary experiment** | potentially closest to an indistinguishability failure; needs durable, held-out full-pipeline replicas |
| W8 | public-slice liveness and compression remove substantial raw structure | **measured** | useful attacker normalization; not by itself reconstruction or SAT failure |
| W9 | exact-span heatmaps miss approximate correlation | **demonstrated by metric contrast** | confirmed measurement weakness |
| W10 | construction-stage bounded degree survives every later prefix | **unsupported** | do not assume; measure minimum recovery degree by stage |
| W11 | diagnostic sidecars/manifests reveal internals | **toolchain fact** | release threat depends on packaging; research artifacts are not automatically adversary inputs |
| W12 | heatmap leakage predicts SAT hardness | **no evidence** | keep claims separate |

#### W1: measured same-seed nondeterminism

This revision ran the current production constructor twice with:

- the same 1,000-gate source hash;
- the same `local_mixing_bin` hash;
- `SSS_CNOT_SEED=2026072501`;
- zero SSS rounds, isolating construction;
- identical TDP4n, `[3,3]`, random-slice, CNOT, and `M_length=4096` flags.

Both runs produced the same public `Y,Z` and the same 558-wire layout, but:

| Replay | Constructed gates | `cg_fragments` | Circuit SHA-256 |
|---|---:|---:|---|
| A | 72,827 | 32,169 | `7eb020f1820d1e2275bb9cd3bca9876efad11eae7f36496d13795f7765216aca` |
| B | 72,939 | 32,242 | `2e4a7f83dbf29ed3c3681b9e1794b20da6335552be072643666a41efd7493816` |

This directly confirms the source audit: `shoot_random_gate` opens
`rand::rng()` instead of consuming the advertised seeded stream. The problem
changes both placement and gate count. Until every RNG stream is explicit, a
seeded A/B comparison is not exactly replayable.

#### W2/W3: small shared nonlinear state, not a secret key

All product terms are functions of the same public band. Under the automatic
policy, `B` grows approximately as `sqrt(n)`; current TDP production has
`B=46`. Conditioning on an identified band assignment makes the decode
affine and gives a conceptual `2^B` enumeration ceiling. That does **not** make
`B` a measured security-bit count: the band is neither secret nor necessarily
independently distributed, and locating semantic band variables after mixing
is part of the attack.

Likewise, “doubling B requires 4x the physical wires” is true only if the
automatic `B≈sqrt(n)` formula is held fixed. The research branch exposes an
explicit band knob.

Re-sourcing replaces one bounded-degree product with another over the same
latent variables. Rolling moves a band value between physical wires. These
operations may frustrate static-coordinate readers, but they do not refresh
the underlying content. A full-pipeline semantic tracker is still required
before calling this exploitable.

#### W4/W5: helper freedom, carefully scoped

For fixed `X`, triangular fill has the form

```text
band_out[i] = helper[i] XOR F_i(X, band_out[0..i))
```

so an input-dependent forward solve can choose `helper[i]` to obtain any
desired filled-band vector. This matters in an existential witness model:
SAT helper variables may co-vary with `X` inside a witness.

It does **not** imply that one constant helper vector collapses the band for
all `X` samples. Therefore it does not, by itself, invalidate fixed-helper or
random-helper heatmaps. After fmix, whole-function equivalence preserves the
existence of witnesses, but an attacker still has to recover a useful decoded
observable or formulate the optimization on the final artifact. The earlier
phrase “the helper gauge collapses the entire nonlinear layer” is too broad
unless qualified as an `X`-adaptive raw semantic statement.

The next experiment must explicitly distinguish:

1. a single fixed helper vector;
2. helpers sampled independently of `X`;
3. an `X`-adaptive helper function;
4. solver-selected existential helpers on the final CNF.

#### W6/W7: grammar and source presentation

Polarity-twin candidates and recurring fold families are genuine syntactic
fingerprints. The raw study found 83 candidate twins and 105 candidate wire
pairs, but did not log true rolls, group three transvections into a swap, or
measure precision/recall. “Every roll becomes a known swap” is therefore not
an established result.

The preliminary equal-function presentation experiment is more directly
relevant: target-wire entropy separated ten repeated-identity artifacts from
ten varied-identity artifacts, despite equal function and size. However, its
durable artifacts are incomplete, it used only a small raw setting, and the
one short mixed replica is insufficient. It should be replicated with
cross-validation, label permutation, multiple source pairs, and complete
five-round/fmix stages.

#### W8: current contract-aware compression

For this revision, `fcompress` was applied twice to the accepted current
72,831-gate constructed artifact. An all-wire run isolates structural
compression; a live-output run additionally permits removal of gates irrelevant
to the advertised middle output wires `128..255`.

| Live contract | Output gates | Retained | Gates removed | Verification |
|---|---:|---:|---:|---|
| all 558 outputs | 65,338 | 89.7% | 7,493 | no mismatch in 32 x 64 randomized lanes |
| middle outputs `128..255` | 59,626 | 81.9% | 13,205 | no mismatch in 32 x 64 randomized lanes |

Both dry runs reached their ten-iteration cap rather than a fixed point. The
10.3% all-wire reduction confirms compressible structure independent of
liveness; the 18.1% live-output reduction adds contract pruning. The runs were
configured to preserve their declared live outputs and kept local rewrite
verification enabled; their final global checks found no mismatch in 2,048
randomized lanes. That is strong sampled evidence, not an exhaustive proof over
all 558-bit inputs. Neither run is yet the stronger fixed-`Y,Z` partial
evaluation. Gate-count reduction alone does not establish logical
reconstruction or easier SAT.

### 0.8 Corrections to structural claims

Several older implementation readings need explicit correction:

- “4--12 input bits” describes immediate band-fill fan-in, not transitive
  support. A fill gate may read an earlier band variable, recursively expanding
  both support and algebraic degree.
- A positive two-control idealization gives a 4x4 Cartesian fold, but
  complemented controls, constant atoms, and ledger constants can change the
  fragment count. “Every G57 becomes exactly 16 fragments” is not universal.
- Whole-circuit equivalence does not preserve the algebraic degree of
  intermediate prefixes in final circuit coordinates. SSS/fmix may raise,
  lower, or redistribute that degree.
- `14/64=0.21875` is the error of an idealized XOR of two independent,
  unbiased cubic monomials. Actual terms share correlated band variables.
  It is a diagnostic model, not a measured best-affine error.
- Globally distinct mask slots are a structural constraint, but no concrete
  pruning gain has been measured.
- Clean endpoints mainly explain port artifacts. They become an attack only
  when specialization, liveness, or source-boundary recovery demonstrates an
  interior advantage.

### 0.9 Threat models and what each experiment can claim

The project should report at least four claims separately:

| Claim | Proper test |
|---|---|
| known-source progress hiding | port-aware exact and held-out approximate heatmaps |
| blind equivalent-circuit indistinguishability | classifier on equal-size/equal-function presentations without access to reference `C` |
| solver hardness | controlled SAT distributions, fixed resources, model verification |
| implementation correctness/reproducibility | deterministic replay, full-state rewrite checks, decoded endpoint checks, hashes |

Equivalent-circuit indistinguishability is a recommended stronger goal, not a
project requirement established by this document. A distinguisher can violate
that goal without yielding recovery or a SAT shortcut. Conversely, a flat
known-source heatmap does not prove indistinguishability.

### 0.10 Agreement, disagreement, and uncertainty

| Earlier proposition | Current position | Reason |
|---|---|---|
| the nonlinear product shares remove the supported exact affine progress ridge | **agree** | post-SSS has one isolated one-bit-scale cell; post-fmix is flat; neither yields chronology |
| current restricted degree 2 also fails to reconstruct progress | **agree, narrowly** | shown at construction; post-fmix degree 2 remains unrun |
| the diagonal is simply one degree above the ruler | **uncertain** | only an archival tiny raw degree-3 result supports it |
| approximate chronology survives a full mix | **agree, preliminary** | strong stored trace, but no holdout/current-contract replicas |
| roll carries nearly all the security | **disagree** | confounded A/B; roll only defeated a static-coordinate reader |
| every roll is syntactically recoverable | **disagree as stated** | fingerprints exist; swap recovery was never validated |
| resource/roll refresh labels rather than semantic band content | **agree** | direct source-level fact |
| band functions have only 4--12 input support | **disagree** | direct fan-in was confused with recursive support |
| SSS/fmix preserve cubic prefix degree | **disagree** | final equivalence does not constrain snapshot degree |
| best affine error is 0.21875 | **uncertain/model only** | assumes independent unbiased terms |
| the band supplies `B` security bits | **disagree** | it is a public latent dimension, not a measured secret |
| helper freedom completely defeats final fmix | **uncertain/overstated** | raw `X`-adaptive manifold exists; final recovery is untested |
| syntax, specialization, and compression are legitimate attack surfaces | **agree** | source audit and current compression support this |
| old v03 invalidates postmix evidence | **disagree; stale** | accepted v05 crossed the failure point with extensive sampled validation |
| advertised seeds reproduce construction | **disagree** | production-scale same-seed replay failed |
| heatmap leakage implies SAT weakness | **disagree** | no relationship has been measured |

### 0.11 Highest-value remaining experiments

Ordered by expected information gain:

1. **Held-out approximate reader on v05.** Add fixed `Y,Z`, randomized helper
   support to `hmap_stat`; freeze candidate and polarity on training samples;
   score on disjoint holdout samples; cover all target bits; retain winning
   wire identities; add mismatched-reference, random-label, circular-shift,
   source, and seed controls.
2. **Deterministic roll/fill ablation.** First route all randomness through
   explicit seeds. Hold plan and band fixed and cross
   `{roll=0,1} x {fill_nl=0,2}` over many small exhaustive replicas. Emit true
   semantic band locations and roll events, then report tracker and twin
   precision/recall.
3. **Small exhaustive full-pipeline survival.** At `n=8` or `n=12`, evaluate
   every `X` through five SSS rounds and 15.2x fmix. Measure whether semantic
   band signatures, cubic decoders, and source syntax survive each stage.
4. **Current fixed-slice degree curve.** Merge degree 3 and explicit product
   sets into the fixed-slice reader. Start with a coarse current `B=46` map or
   a syntax-informed decoder; report minimum successful degree by SSS/fmix
   checkpoint. Do not begin with a prohibitively repeated generic 44k-feature
   basis.
5. **Public-slice effective-size audit.** Partially evaluate fixed `Y,Z`,
   leave `X,W,band` free, prune to outputs `128..255`, compress, and verify with
   the view-aware checker at constructed/post-SSS/post-fmix stages.
6. **Helper threat separation.** Compare fixed, independent-random,
   `X`-adaptive, and SAT-selected helper policies; report residual mask rank,
   live cone, heatmap, and SAT effects separately.
7. **Blind presentation test.** Use multiple equal-function/equal-size source
   pairs, 20--30 seeds per class, cross-validation, and full-pipeline
   checkpoints.
8. **Only then correlate with SAT.** Apply the winning structural attack or
   normalization before/after CNF generation and compare verified Painless
   distributions. Do not infer SAT behavior from heatmap depth.

### 0.12 Reproducibility and local evidence

New evidence created for this revision is under:

```text
experiments/nonlinear_gadget_analysis_20260727/
```

It contains:

- `run_reproducibility_probe.sh` and two complete same-seed construction
  replays with logs, timing, sidecars, and hashes;
- `run_portaware_v05_audit.sh`, all six endpoint-safe v05 renderings, metrics,
  and PNG hashes;
- `run_current_v05_compression_probe.sh`, the sampled-verified current-artifact
  live-output compression transcript, timing, and input/tool hashes;
- `run_current_v05_allwire_compression_probe.sh`, the matched all-wire
  structural-compression control;
- `RESULTS.md`, a compact record of commands, outputs, limitations, and hashes.

The accepted pipeline evidence remains under:

```text
1_affine_tests/nonlinear_tdp4n_sss5_fmix152_noopfix_v05/
```

The `[2,2,3,3]` controls remain under:

```text
1_affine_tests/nonlinear_tdp4n_2233_degree2_20260726/
1_affine_tests/sliced_sandwich_2233_degree2_20260726/
```

The `/tmp/nlprobe` commands in historical Section 8 are **not currently
reproducible** because those scripts and the exact Build A binary were not
retained. They are preserved below as a record of what was run, not as a
working recipe.

### 0.13 What this revision changed

This revision:

1. promoted accepted v05 correctness and size evidence over superseded v03;
2. corrected the legacy endpoint/tie heatmap interpretation;
3. added current `[3,3]` and both `[2,2,3,3]` degree-2 results;
4. demoted production degree-3 recovery from implied result to open question;
5. separated exact reconstruction, approximate chronology, blind
   distinguishing, SAT hardness, and implementation validity;
6. corrected the feature counts, tool capabilities, support, fragment-count,
   roll, prefix-degree, affine-error, and `B`-security claims;
7. narrowed helper collapse to an input-adaptive raw semantic argument and
   stated the missing final-artifact test;
8. added a production-scale same-seed reproducibility falsification;
9. added sampled-verified current-v05 all-wire and live-output compression
   measurements;
10. added an explicit agree/disagree/uncertain table and a prioritized
    falsification plan; and
11. marked the missing `/tmp` probes and timestamp-only Build A as archival
    rather than reproducible evidence.

### 0.14 Implementation follow-up

After this review, the unconditional tooling recommendations were implemented
without changing the nonlinear construction:

- seeded gadget, Feistal, and TDP constructors now route gate-order shooting
  through their caller-provided RNG; direct same-seed replay tests are
  byte-identical, while different seeds still diversify the artifact; a
  seedless full-round run is also byte-identical when replayed with its
  advertised seed;
- fixed-slice sidecars distinguish the total artifact gate count from the
  slice-preblock count, cover intermediate rounds, and record the construction
  seed, random-gate frequency, source fingerprint, and nonlinear preset;
- the fixed-slice exact reader accepts explicit product-wire sets, supports
  degrees 1 through 3, reuses each mixed-prefix basis across source rows, and
  can dump finite-sample monomial support for selected cells; production-scale
  snapshot, basis, and solve-work estimates are recorded and guarded by
  default;
- `hmap_stat` is restored with fixed `Y,Z`, independently randomized helpers,
  training-only candidate/polarity selection, disjoint holdout scoring, all
  target bits by default, honest null matrices, retained winner identities,
  input/executable provenance, and a serial-preserving null that remains
  undefined without one-bit-equivalent informative coverage; and
  and
- the active port-aware renderer now separates one-bit and two-bit-equivalent
  coverage, averages tied minima, preserves serial structure with a
  circular-shift null, and leaves rho undefined without adequate informative
  coverage.

W1's production replay remains valid evidence about the binary measured in
this document; the hidden RNG cause is repaired in the subsequent source
state. No Tier-2 construction change from historical Section 9 was applied.

---

## Historical review record (Sections 1--11)

The material below is retained verbatim where possible so a later reviewer can
see how the interpretation evolved. It contains provisional conclusions that
Section 0 corrects.

## 1. Why this document exists

The project measures obfuscation quality with a *reconstruction heatmap*. For a
logical circuit `C` on `n` wires and a mixed/gadgetized circuit `G` on `N >= n`
wires that computes `C` in encoded form, we fix `G`'s non-input wires, inject
the same `x` into both, and ask for each cell `(i, j)`:

> is the state of `C` after `i` gates recoverable from the state of `G` after
> `j` gates, by a GF(2) function of bounded algebraic degree?

`H(i,j)` is the mean per-bit reconstruction error. Low `H` = recoverable
(leak); `H ~ 0.5` = hidden. The headline statistic is the **ridge**: a valley of
low `H` whose location advances monotonically with `i`, meaning `C`'s
computational progress is still legible inside `G`. It is read as

- **depth** — mean per-row prominence of the ridge cell above its row background
- **rho** — Spearman(row index, ridge column); `rho ~ 1` is a clean diagonal

**The observation that motivated this work:** every affine-encoded pipeline
retains a perfect diagonal. Production 4n two-share TDP runs at n=128 logical /
512 physical report `rho = 1.000`, `depth ~ 0.171`, `perm z ~ 6.6` — after full
SSS expansion *and* 15x nonlinear-frame fmix. The compact-300 plates are the
same story: base self-control `depth 0.4465 / rho 1.000`, nonlinear SSS raw
`0.2823 / 1.000`, nonlinear final fmix `0.2489 / 1.000`. Depth erodes; the
diagonal never breaks.

This is expected and understood: wire permutations, NOTs, CNOT transvections
and SAMFs are all *affine* changes of basis, and a degree-1 predictor is
invariant to that entire class, so it looks straight through them.

**Then the nonlinear product-share gadgetization was introduced and the
degree-1 diagonal disappeared.** The question this document answers is whether
that means the construction got stronger, or whether it means the ruler stopped
reaching.

**Historical provisional conclusion:** primarily the latter. Section 0 now
supersedes this wording: current exact degree-1 maps have no supported
interior progress ridge, the post-fmix sampled core is exactly flat, and
production-scale degree 3 remains untested.

---

## 2. The construction under analysis

Implemented in `src/replace/gadgets.rs`, entry point
`gadgetize_xgates_with_prod`. Gate type throughout is `XGate`:

```
fires(x) = comp XOR AND_i lit_i(x)      lit = wire, or NOT wire
effect:   x[target] ^= fires(x)
```

### 2.1 Wire layout

```
0 .. n          data / carriers
n .. 2n         aux  / carriers          carrier_total = 2n
2n .. 2n+B      band ("home" positions)  N = 2n + B
```

`B = band_size(n) = max( ceil(sqrt(4 * n * k_total)), 6, max_degree + 3 )`.

**`B` is `O(sqrt(n))`.** This matters in Section 5.5.

### 2.2 The encoding

Each logical value `v` is represented as an ordinary two-share XOR pair plus a
small number of permanent multiplicative mask terms over band variables:

```
v  =  c0 XOR c1 XOR P_1 XOR ... XOR P_k XOR const
```

where each `P_t` is a *degree-`d` conjunction of band literals*, e.g.
`P = (b3 XOR a3) AND (b7 XOR a7) AND (b11 XOR a11)`.

The production preset (`ProdConfig::production()`) is:

| field | value | meaning |
|---|---:|---|
| `k`, `deg` | 0, 2 | no degree-2 slots |
| `k_hi`, `deg_hi` | **2, 3** | two independent degree-3 mask terms per value |
| `band` | 0 | automatic `B` |
| `rsrc` | 1 | one mask term replaced per source-gate gap |
| `max_width` | 0 | emit wide conjunctions natively |
| `fill_nl` | 2 | two cascaded nonlinear terms per band wire at fill |
| `roll` | **1** | one band-variable *relocation* per source-gate gap |

So `plan = [3, 3]`. An older config seen in the 568-wire tdp4n run logged
`plan = [2, 3, 3]` (`k=1, deg=2`).

### 2.3 The band

`emit_band_fill_nl` computes each band wire once, at the start, as

```
b_i = pivot XOR (1..7 further data wires) XOR (2 quadratic g57-form terms)
```

where the quadratic sources are data wires or *earlier band wires*. Each band
wire therefore depends on roughly **4–12 input bits** and has small constant
algebraic degree.

**The band is never recomputed.** Between the initial fill and the final
mirror-fill, band *values* are frozen. Only their physical *locations* change,
via `roll`.

### 2.4 Applying a logical gate: `fold_cg`

No operand is ever reconstructed. Each control literal is expanded into its
full decode atom list, and the Cartesian product across controls is emitted as
individual conjunction fragments onto a random carrier of the target value.

For a 2-control source gate (all G57 gates are 2-control), each control has
atoms `{c0}, {c1}, T_1, T_2` of sizes `1, 1, 3, 3` (plus an optional empty
constant atom). The product is 16 fragments with literal-set sizes
`2 (x4), 4 (x8), 6 (x4)`.

### 2.5 `roll`

Picks one band variable and one arbitrary wire, exchanges them, and emits three
transvections to move the values. Each transvection is emitted by
`emit_transvection_mixed`, which with probability 1/2 emits a plain
`CNOT(target, source)` and otherwise emits **two width-2 conjunctions that
differ in exactly one literal's polarity**:

```
conj(target, [(source,true), (helper,true )])
conj(target, [(source,true), (helper,false)])
```

These two gates compose to exactly `target ^= source`. The stated intent is to
stop rolled band locations from being identifiable by width-1 writes alone.

### 2.6 Bookends and endgame

`rand_z_xgates` bookends front and back; `inject_all` installs every mask at
the start; `strip_all` removes every mask at the end, followed by the inverse
`W` decode back onto wires `0..n`, a trailing mirror band-fill over all
non-output wires, and a `commuting_shuffle` (3 insertion passes redrawing a
random linear extension of the dependency DAG).

---

## 3. Experimental setup

Deliberately small so that *exact* linear algebra is possible with samples far
exceeding regressors.

- `n = 12` logical, `N = 34` or `36` physical, gadget ~2.5k gates
- source: 60 random G57 gates
- `S = 4096` samples, 3072 train / 1024 holdout
- grid: 21 `C`-prefix rows x 41 `G`-prefix columns
- predictor: **identical** to `hmap_affine` — GF(2) span membership on the
  training lanes; inconsistent ⇒ error 0.5; consistent ⇒ measured holdout error
- degree-1 regressors: constant + all `N` wire columns
- augmented regressors: the above **plus** all degree-2 and degree-3 monomials
  over a chosen `B`-wire subset (`C(12,2)+C(12,3) = 286` extra, or
  `C(10,2)+C(10,3) = 165`)

Two builds were compared because the working tree changed mid-investigation:

- **Build A** — binary as of 2026-07-25 06:13: `plan=[2,3,3]`, `B=12`, **no roll**
- **Build B** — current working tree rebuilt: `plan=[3,3]`, `B=10`, **`roll=1`**

Scripts: `/tmp/nlprobe/{nlprobe.py, nlprobe2.py, struct_probe.py, twins.py}`.

---

## 4. Findings

### 4.1 The diagonal is not gone — it is one degree above the ruler

Build A, raw gadget (2619 gates, 36 wires):

| predictor | extra regressors | meanH | ridge depth | rho |
|---|---:|---:|---:|---:|
| degree 1 (affine) | 0 | 0.4752 | 0.1756 | 0.866 |
| + deg 2,3 over the 12 band wires | 286 | **0.4045** | **0.2637** | **0.997** |

The degree-1 argmin trace is degenerate — column `0` for rows 0–10, then column
`37` for rows 11–20. That is two plateaus, not a diagonal; the `rho = 0.866` is
inflated by the tie structure.

The augmented argmin trace is:

```
0, 0, 7, 7, 9, 11, 11, 15, 17, 17, 20, 20, 23, 25, 25, 29, 29, 31, 33, 33, 37
```

A textbook monotone diagonal. Total runtime: 2.6 seconds.

**Selecting the band by the free statistic in 4.2 gave numerically identical
results to using the true band-home wires.** The attack does not require
knowing the layout.

### 4.2 Band identification was free (Build A)

Two independent statistics each recovered **12/12** band wires with *no overlap
between the two distributions*:

| statistic | band wires | carrier wires |
|---|---|---|
| read / write ratio | 17.61 – 34.42 | **max 2.95** |
| value changes over 41 grid steps | 5 – 8 | 15 – 37 |

Band wires are read ~4x more often and written ~6x less often than carriers.
This is purely a consequence of the design: band wires are read by every mask
literal in every fragment, and written only by fill and roll.

The band-literals-per-gate histogram reads the *plan* off the gate list:

```
{0: 1113, 1: 21, 2: 364, 3: 722, 4: 101, 5: 201, 6: 97}
```

Mass at 2/3 and 5/6, near-zero at 1 — exactly `deg=2` and `deg_hi=3` slots
combined pairwise across 2-control source gates.

Gate-width histogram: `{1:440, 2:818, 3:482, 4:581, 5:201, 6:97}`.

### 4.3 Partial survival through SSS

Build A gadget + 2 native SSS rounds (2881 gates, 37 wires). SSS added an
independent helper wire, which contaminated the top-B selection (11/12):

| predictor | meanH | depth | rho |
|---|---:|---:|---:|
| degree 1 | 0.4770 | 0.1774 | 0.905 |
| + deg 2,3 over 12 fewest-change wires | 0.4604 | 0.1806 | 0.946 |
| + deg 2,3 over true band-home wires | 0.4419 | **0.2118** | **0.966** |

Attenuated but clearly present. This is 2 rounds; production runs many rounds
plus 15x fmix. **Survival through the full production pipeline is untested and
is the single most important open question.**

### 4.4 `roll` is carrying nearly all the security

Build B, raw gadget (2472 gates, 34 wires):

| predictor / band tracker | meanH | depth | rho |
|---|---:|---:|---:|
| degree 1 | 0.4695 | 0.1699 | 0.910 |
| + deg 2,3 over static band-home wires | 0.4643 | 0.1647 | 0.910 |
| + deg 2,3 over frozen-run top-10 | 0.4585 | 0.2165 | 0.838 |
| + deg 2,3 over **windowed read/write top-10** | 0.4512 | **0.2349** | 0.911 |

Static identification collapses to 3–5 of 10. Cubic monomials over the *static*
band-home wires buy nothing (0.1647 vs 0.1699 baseline) — because under rolling
the band variables are not on those wires at mid-circuit prefixes.

A windowed read/write tracker recovers a substantial part: depth `+38%`, and
the argmin trace becomes visibly monotone over the first half
(`0,2,2,0,2,12,14,14,17,14,18,18,38,...`) instead of a flat plateau.

> **Caveat, stated plainly.** The A/B changed three variables at once
> (`B` 12→10, `plan` `[2,3,3]`→`[3,3]`, `roll` 0→1). The attribution to `roll`
> is strongly suggested by the mechanism — static trackers fail exactly where
> relocation was introduced — but is **not isolated**. It cannot currently be
> isolated: `ProdConfig` is reachable only from tests, so `roll` has no CLI
> knob and cannot be ablated on a production build. That is itself a gap.

### 4.5 The roll schedule is syntactically recoverable

`emit_transvection_mixed`'s camouflage (Section 2.5) leaves two gates identical
except for one literal's polarity. In the Build B raw gadget:

- **83 polarity-twin pairs found** (≈88 expected from 59 rolls x 3 transvections x 1/2)
- gate-index gap between twins: min 1, median 3, max 253
- 22 adjacent, 52 within 5 gates
- 105 wire pairs carry >= 3 transvections (swap candidates)

`emit_g57_form` leaves an analogous residue: a g57 immediately followed by a
CNOT on the same target reading one of the g57's own controls.

**Consequence.** Collapse the twins with a peephole rule and every roll becomes
a plain 3-CNOT swap on a known wire pair. The band permutation becomes
trackable at every prefix, and the Section 4.1 attack applies throughout. The
one mechanism holding the construction up is defeated by a syntactic rewrite
that costs nothing.

(Measured on the **raw** gadget. Whether SSS + fmix destroys twin adjacency is
untested — but note the twins do not need to be *adjacent*, only *matchable*,
and they are identified by shape, not position.)

### 4.6 The cost asymmetry favours the attacker

Because `B = O(sqrt(n))`, the band-restricted cubic regressor count is
`O(B^3) = O(n^1.5)` — **sub-quadratic**, and therefore asymptotically cheaper
than the degree-2 predictor over all `N ~ 2n` wires, which is `O(n^2)`.

Concretely, at the 568-wire tdp4n layout (`n = 256` logical, `B = 56`):

| predictor | regressors | project's own assessment |
|---|---:|---|
| degree 1 | 569 | current production measurement |
| **band-restricted degree 3** | **29,829** | *not measured by this historical run* |
| full degree 2 | 161,597 | documented as intractable |

The **feature count** is about 5.4x smaller than full degree 2 and needs roughly
40k total samples for a 75/25 split. This comparison does not establish that
the current generic solver's runtime is 5.4x smaller; its basis work is
superlinear in the number of regressors.

### 4.7 The measurement tool cannot express the attack

**Historical tool snapshot.** This describes the frozen fixed-slice executable
used for these runs, not the project as a whole. The cached
`origin/ssg-gen-mix-clean` `hmap_affine` now supports degree 3 and an explicit
product-wire list. It still lacks the merged current fixed-`Y,Z`,
random-helper and dynamic-band workflow described in Section 0.6.

`hmap_affine_fixed_slice` has two structural limits:

- `--degree` is handled only for 1 and 2 (`if args.degree >= 2`)
- `--deg2-wires W` forms products among wires `0..W` — a **prefix**

The band lives at the **end** of the wire order (`2n .. 2n+B`) and **moves**.
There is no argument that selects it. So the flat degree-1 plate is not
evidence of hiding; it is the only plate the tool is able to draw.

---

## 5. Weaknesses read from the implementation, not yet measured

1. **The construction-stage encoding uses bounded degree 3, independent of
   circuit size.** It is not established that arbitrary prefixes after SSS and
   fmix remain degree 3 in the final coordinates. Whole-circuit equivalence
   preserves endpoint functionality, not intermediate-prefix algebraic degree.

2. **The mask may be sparse and biased.** For two independent unbiased cubic
   terms, `P_1 XOR P_2` is 1 with probability
   `2/8 - 2/64 = 0.21875`. Actual terms share correlated band variables, so
   this is an idealized model, not a measured best-affine error. Exact span
   membership would nevertheless miss any nonzero-error approximation; the
   held-out statistical reader proposed in Section 0.11 is needed.

3. **Two carriers only.** Every value remains exactly `c0 XOR c1 XOR P`. The
   affine part is completely transparent; only `P` hides anything.

4. **Band fill has small immediate fan-in but unknown transitive support.**
   Because a fill gate can read an earlier band variable, recursive support and
   algebraic degree can cascade. Pairwise correlation and learnability must be
   measured; they do not follow from the 4--12 direct fan-in count.

5. **Clean encode/decode endgames.** `inject_all` and `strip_all` + `W^-1` are
   long, fully unmasked prefix and suffix regions. An algebraic-degree-vs-prefix
   profile should show plateau boundaries locating them exactly.

6. **Slice-inert padding is not work.** With `y, z` public — as they are in the
   fixed-slice heatmap runs, where they appear in `profile.env` — the
   slice-zero preblock is the identity and constant-folds away. Any gate count
   contributed by slice-inert structure inflates the size ratio without
   inflating hardness. The live-gate fraction under the known slice has not
   been measured.

7. **Mask slots are globally distinct.** `ProdLedger.used` is a `HashSet`, so
   all live slots across all values are drawn without replacement — a small but
   real structural constraint an attacker can exploit to prune candidates.

8. **No runtime secret randomness.** Every mask is a deterministic circuit
   function rather than information-theoretic masking. A blind distinguisher
   would violate an equivalent-circuit indistinguishability goal, but it would
   not automatically yield reconstruction or a SAT shortcut.

---

## 6. What this document does *not* establish

Stated explicitly so these are not over-read:

- The measurements in historical Sections 3--4 are at `n = 12`, ~2.5k gates.
  Section 0 adds production-scale results and must be read before generalizing.
- The strongest results (4.1, 4.2, 4.5) are on the **raw gadget**, before SSS
  and fmix. The post-SSS result (4.3) is 2 rounds only.
- The attribution of Build A vs Build B differences to `roll` specifically is
  **not isolated** — see the caveat in 4.4.
- No claim is made about SAT hardness. The heatmap ridge is a reconstruction
  measure, not a solver-time measure, and the two have not been shown to track
  each other here.
- No claim is made that a degree-3 band-restricted predictor is the *best*
  attack; it is merely a cheap one that works.

---

## 7. Open questions

1. Does the band-restricted degree-3 ridge survive full production SSS + 15x
   fmix? This is the question that decides whether any of this matters.
2. Under a peephole-collapsed twin analysis, can the roll permutation actually
   be tracked end-to-end on a production circuit, or does mixing destroy the
   matching?
3. Is the `~0.22` best-affine-approximation leak efficiently findable, or is it
   genuinely LPN-hard at production width? (The noise is *structured* — a fixed
   degree-3 function of 6 specific wires — which is the reason to doubt the
   hardness intuition.)
4. Is there any principled reason for `B = O(sqrt(n))` other than slot-space
   accounting? Growing `B` faster directly attacks 4.6.
5. Can the hiding degree be made to *grow* with the circuit rather than sit at
   a fixed `[3,3]`?

---

## 8. Reproduction

**Archival warning.** The `/tmp/nlprobe` scripts and exact Build A binary were
not retained and are no longer present. The commands below document the
historical procedure, but they are not a working reproducibility package.
Section 0.12 points to the durable evidence that does exist.

```bash
# 1. small source circuit
./target/release/local_mixing_bin genran -d /tmp/nlprobe/src12.txt -n 12 -m 60

# 2. nonlinear gadgetization only (0 SSS rounds)
./target/release/local_mixing_bin sss -n 12 -m 1 -x 100 \
  -s /tmp/nlprobe/src12.txt -r 0 -d /tmp/nlprobe/out12.txt \
  --nonlinear_gadgetize --gadget_path /tmp/nlprobe/gad/nl12.mpmct1

# 3. degree-1 vs band-restricted degree-3 heatmap; static band trackers
python3 /tmp/nlprobe/nlprobe.py  /tmp/nlprobe/src12.txt \
  /tmp/nlprobe/gad/nl12.mpmct1 12 /tmp/nlprobe/res.json 4096

# 4. prefix-adaptive band trackers (needed once roll=1)
#    args: C G n samples B window
python3 /tmp/nlprobe/nlprobe2.py /tmp/nlprobe/src12.txt \
  /tmp/nlprobe/gad/nl12.mpmct1 12 4096 10 150

# 5. syntactic fingerprints: read/write ratio, band-literal histogram
python3 /tmp/nlprobe/struct_probe.py /tmp/nlprobe/gad/nl12.mpmct1 12

# 6. polarity-twin / swap-idiom detection
python3 /tmp/nlprobe/twins.py /tmp/nlprobe/gad/nl12.mpmct1 12
```

The probe scripts reimplement the `hmap_affine` predictor (bit-sliced GF(2)
span membership with a train/holdout split) in ~150 lines of numpy, so they are
independent of the production tool and can express degree 3 over an arbitrary
wire subset.

---

## 9. Proposed next steps

**Tier 1 — measurement only, no construction changes.** These decide whether
there is a problem.

1. Extend `hmap_affine_fixed_slice`: `--degree 3`, an explicit
   `--product-wires <list>` (not a prefix), and a `--track-band` mode that
   selects the per-column wire set by windowed read/write ratio. Re-run the
   compact-300 and tdp4n plates.
2. Add an approximate-affine reader (greedy / correlation) alongside span
   membership, so a 0.22-error leak reads as 0.22 rather than 0.5.
3. Run the roll-tracking attack end-to-end — peephole-collapse polarity twins,
   recover swaps, track the permutation, then cubic regression — reported at
   raw gadget / post-SSS / post-fmix. **This is the number that matters.**
4. Expose `ProdConfig` (`roll`, `k`, `deg`, `band`, `fill_nl`) on the CLI so
   the mechanisms can actually be ablated.
5. Measure the live-gate fraction of a production circuit under the known
   public slice.

**Tier 2 — construction changes, only if Tier 1 confirms.**

6. Remove the peephole tells in `emit_transvection_mixed` / `emit_g57_form`, or
   stop counting them as camouflage.
7. Make the encoding degree grow with the circuit instead of a fixed `[3,3]`.
8. Refresh band *values* mid-circuit, not only their locations.
9. Grow `B` faster than `sqrt(n)` so `O(B^3)` stops being cheaper than `O(N^2)`.

---

## 10. Codex peer-review addendum

Status: **proposed corrections and additional read-only findings,
2026-07-25.** This section is intended as the Codex side of the review with
Claude. It changes no construction code.

### 10.1 Points of agreement

The central diagnosis is correct: disappearance of the exact degree-1 ridge is
not evidence that progress information is gone. The small-instance,
train/holdout experiment is a useful demonstration that the old, raw,
no-roll `[2,3,3]` build moves the exact relation from degree 1 into a
band-restricted degree-3 feature space.

The following narrower claims are also supported:

- the product-share encoding has a public, fixed, bounded-degree grammar;
- the automatic band is only `O(sqrt(n))`, making band-restricted features
  much cheaper than dense features over every physical wire;
- static band-home coordinates stop working once `roll=1`;
- raw gates retain width, polarity-pair, fold, resource, and roll fingerprints;
- the decisive missing experiment is attacker recovery after the full
  production pipeline, not another degree-1 plate.

### 10.2 Corrections needed before treating Sections 1--9 as conclusions

1. **Narrow the headline.** “The diagonal is one degree above the ruler” is
   established cleanly only for the tiny raw Build A. Current Build B recovers
   only a partial ridge with the proposed window tracker. Full-pipeline
   degree-3 recovery has not been established. Likewise, “roll is carrying
   nearly all the security” should read “roll supplies most of the resistance
   to this particular static-band reader.”

2. **Twin detection is a fingerprint, not yet roll recovery.** The probe finds
   83 possible polarity-twin pairs and 105 swap candidates for about 59 true
   rolls. It does not report precision/recall against ground truth, uniquely
   group the three transvections belonging to each roll, or show survival
   after mixing. “Every roll becomes a known swap” is therefore a conjecture
   to test, not a measured consequence.

3. **Do not assert that SSS/fmix preserves cubic degree at every prefix.**
   The raw decode relation is cubic in the semantic band variables. Nonlinear
   coordinate changes and equivalent local rewrites can raise the degree of
   that relation in the wires visible at a later mixed snapshot. Whole-circuit
   equivalence does not imply preservation of intermediate-prefix degree.

4. **Direct fan-in is not transitive support.** `emit_band_fill_nl` can read
   earlier band functions and unions their supports. Thus “4--12 input bits,”
   “tiny support,” and “small constant algebraic degree” do not follow from
   the number of immediate controls. Those quantities should be measured.

5. **The `0.219` error is an idealized calculation.** It assumes two unbiased,
   independent cubic terms. The actual terms share deterministic, cascaded
   band variables. Treat `14/64 = 0.21875` as a diagnostic model, not the
   established best-affine error.

6. **Correct the feature count.** For 568 wires and `B=56`, constant plus all
   linear wires plus band degree-2/3 monomials is
   `569 + C(56,2) + C(56,3) = 29,829`, not 29,885. For current production
   (558 wires, `B=46`) it is 16,774. The asymptotic point remains valid.

7. **Make the metric difference explicit.** The `/tmp/nlprobe` depth is row
   mean minus row minimum; the canonical ridge reader uses a row-median
   background. The values should not be compared directly. Endpoint plateaus
   must also be trimmed before interpreting rho.

8. **Fix the strength and provenance language.** The small experiment
   demonstrates leakage; it is not by itself an obfuscation break or a SAT
   result. The Build A binary is identified only by a timestamp, and the
   scripts/artifacts live under `/tmp`; the reproduction commands now produce
   only Build B. Durable hashes, configs, seeds, artifacts, and a randomized
   control are needed before calling the experiment reproducible.

9. **The Section 1 cross-reference should point to Section 4.4, not 5.3.**

### 10.3 Existing production-scale result that should lead the document

An approximate-affine reader already exists on commit `2a807d28` as
`hmap_stat`; Section 9.2 is therefore partly obsolete. It searches the best
constant, single wire, or XOR of two wires for each target bit. Re-reading the
committed final `[3,3]` matrices (`stat_final` at `2a807d28`, corrected x3
artifact at `7cd77cac`) reveals the result that the existing report missed:

| artifact | gates / wires | median interior row peak | null floor | row vs peak-column rho |
|---|---:|---:|---:|---:|
| `stat_final.bin` | 762,362 / 558 | 0.7034 | 0.5257 | **0.9948** |
| `stat_x3input.bin` | 1,428,155 / 558 | 0.6915 | 0.5252 | **0.9956** |

The calculation trims the first/last 10% of rows and columns and uses
tie-aware Spearman ranks. In 100,000 row-label permutations, none matched
either observed rho (`p < 10^-5`). In the x3 artifact, peak columns advance
almost monotonically from 7 to 29 across the retained source rows.

This is currently the strongest evidence in the repository: **the exact
affine diagonal is flat, but a cheap two-wire approximate timing diagonal
survives the full production `[3,3]` mix.** It requires the reference circuit
`C`, so it is a known-source alignment result rather than a standalone blind
distinguisher. Candidate selection and scoring also use the same samples;
the matched random-target floor controls the max-search bias, but a proper
train/holdout rerun is still required.

### 10.4 Additional measured weaknesses

#### A. Semantic tracking can follow values through raw rolling

On two independent current-production `n=8`, `B=8`, `roll=1` raw gadgets, a
chosen-input trace was evaluated at every prefix. Physical wire locations were
discarded; each current wire value was fingerprinted by its truth column over
all 256 inputs.

- Seed 1: the top eight nontrivial signatures occur in
  `684--910 / 1232` snapshots; the ninth occurs in only 255.
- Seed 2: the top eight occur in `798--882 / 1255`; the ninth occurs in 230.
- The sharp count is exactly the band width, and all eight signatures are
  balanced.

This is strong preliminary evidence for the implementation-level concern:
`roll` moves locations but does not refresh semantic band values. A
value-signature tracker can therefore follow them without recognizing the
three-transvection syntax. The experiment is raw and small, and did not
instrument true band identities, so production/full-mix precision must still
be measured.

#### B. Same-function, same-size source syntax remains distinguishable

Two 8-wire, 40-gate identity circuits were constructed: one repeats the same
self-inverse gate 40 times; the other uses 20 varied adjacent self-inverse
pairs. Across ten fresh gadgetization seeds per class, target-wire Shannon
entropy (in bits) separated every raw artifact:

| source presentation | mean entropy | observed range | mean target-count stddev |
|---|---:|---:|---:|
| repeated identity | 4.309 | 4.058--4.437 | 47.7 |
| varied identity | 4.492 | 4.450--4.530 | 24.8 |

One 10,000-move SSS/fmix round retained the ordering (4.363 versus 4.526), but
that is only one seed. This is a different and more direct security test than
a heatmap: it distinguishes equal-size, equal-function sources without
reconstructing a logical intermediate state. It suggests the fixed Cartesian
fold/cadence leaks source presentation.

#### C. Contract specialization and compression remove substantial structure

On the current 558-wire, 72,596-gate nonlinear TDP raw artifact:

- the advertised 4,096-gate public-slice preblock is exactly
  `R ; B ; reverse(R)`. At the advertised Y/Z slice every `B` term is dead, so
  all 4,096 gates cancel after specialization;
- exact backward liveness from the 128 middle output wires keeps
  66,563 gates and drops 6,033;
- all-wire `fcompress` (no liveness pruning) reduces 72,596 to 65,009 gates
  in ten iterations, a 10.5% reduction. Under the same settings, the affine
  raw control falls only from 28,250 to 27,156 (3.9%).

The trailing mirror fill is dead for the advertised output contract, while
mask stripping is followed by a mostly affine decoder. A contract-aware
attacker should therefore specialize constants, prune the live cone, compress,
then locate and eliminate the maximal affine tail before any heatmap or SAT
comparison.

#### D. The TDP helper space is an attacker-controlled gauge

The current TDP sidecar fixes Y and Z but leaves W plus the band arbitrary:
`384..558`, or **174 helper bits**. The nonlinear band fill is triangular:
for fixed X and desired filled-band values, each initial band bit can be solved
forward because its target is XOR-updated using only data and earlier filled
band wires. Thus the initial band helper can realize any desired post-fill
band state.

This does not prove an easy attack, but it creates a large exact
witness-preserving gauge. A solver or preprocessing adversary should try
zero/one/sparse/random and optimized helper assignments and retain the one
with the smallest residual mask rank, live cone, or SAT core. Measuring only
one random-helper distribution is insufficient.

#### E. Grammar, seams, and operational leakage

The current construction uses a fixed `[3,3]` plan, deterministic band-size
formula, one resource/roll/RG cadence per source-gate gap, Cartesian-product
folds, recognizable strip/decoder networks, and recurring width-3 cube
families. A static hypergraph/template attack should be scored for source-gate
boundary, target, and control recovery. Wire count and cadence also leak
parameters such as source gate count.

Separately, the toolchain writes the unmixed transformed circuit, per-round
artifacts, layout/slice sidecars, and sometimes lineage/origin files. Those
are useful research evidence but should not be confused with the release
artifact. A final packaging allowlist is part of the threat model.

### 10.5 Correctness and experiment-validity blockers

This subsection originally described the terminal v03 failure. That history is
retained, but its “current campaign” conclusion is superseded:

- v03 was a 568-wire `[2,3,3]` precursor. It passed 1,536 fixed-slice cases,
  then failed full-state fmix equivalence at move 2,400,000 and produced no
  accepted postmix artifact;
- the defect was traced to noncanonical complemented empty identity gates at
  SSS/compression/fmix boundaries;
- accepted v05 uses the current 558-wire `[3,3]` build, canonicalizes those
  identities, crossed the old failure point, completed 12,000,000 fmix moves,
  and passed periodic randomized all-wire and finite independent fixed-slice
  checks.

The specific observed v03 correctness blocker is therefore resolved in the
accepted v05 evidence; this is extensive sampled validation, not an exhaustive
proof over all 558-bit inputs.
The reproducibility blocker is not: `shoot_random_gate` still creates an
independent `rand::rng()`. Section 0.7 reports a new production-scale replay in
which identical advertised seeds produced 72,827 versus 72,939 gates.

Before comparing security metrics, freeze the exact commit plus dirty patch,
config, binary hash, source hash, every RNG stream, and deterministic
correctness transcript.

### 10.6 Proposed joint investigation order

No construction change should be chosen from the current evidence. The
measurement sequence should be:

0. **Correctness and provenance:** isolate the fmix-invalid move family,
   restore deterministic seeds, pin the current `[3,3]` artifact, and verify
   both TDP output lanes across adversarial helper patterns.
1. **Honest attacker baseline:** public-slice partial evaluation, palindrome
   cancellation, exact live-cone pruning, all-wire `fcompress`, and affine-tail
   elimination.
2. **Approximate chronology:** rerun `hmap_stat` with train/holdout separation,
   all target bits, multiple C/source/seed replicas, and permutation ridge
   metrics. Emit the winning wire pair so it can be audited.
3. **Band attacks:** compare syntactic twin grouping, dynamic
   value-signature/HMM tracking, and a randomized control; then run targeted
   degree-3 reconstruction only on the recovered band candidates.
4. **Gauge attacks:** sweep fixed and solver-selected W/band helpers and measure
   best-of-K preprocessing, heatmap, and SAT residuals.
5. **Blind distinguishers:** use paired equal-size/equal-function source
   presentations and report held-out classifier AUC at raw, each SSS round,
   and final fmix.
6. **Template/CNF attacks:** factor fold bicliques/cube dictionaries, recover
   stage boundaries and carrier pairs, and compare standardized CNF
   preprocessing/solver trajectories.
7. **Only after the winning attack is identified:** ablate `roll`, band size,
   term count/degree, value refresh, and cross-fold interleaving one variable
   at a time before considering a construction change.

Suggested acceptance conditions are: zero semantic failures under
deterministic replay; held-out approximate ridge rho indistinguishable from
row permutations; no recoverable `B`-sized band signature set; source
boundary/presentation classification at chance; and no systematic best-of-K
helper-gauge or attacker-compression advantage.

### 10.7 Questions for Claude

1. Do you agree to narrow the raw degree-3 and roll claims as in 10.2?
2. Do you agree that the already-committed full-mix two-wire statistical ridge
   in 10.3 should become the lead finding?
3. Do you agree that correctness/provenance and attacker-side normalization
   must precede any new construction change?
4. Which threat-model claim is intended: resistance to the listed empirical
   attacks, indistinguishability of equivalent circuits, SAT hardness, or a
   stronger obfuscation notion? The acceptance tests differ materially.

---

## 11. Claude response to the addendum

Status: **2026-07-25.** Corrections accepted, two refinements, one measured
escalation of 10.4.D, and four further weaknesses. No construction code
changed.

### 11.1 Direct answers to 10.7

**Q1 — narrow the claims: yes.** Every correction in 10.2 is accepted. Two are
outright errors on my side and I want them recorded as such:

- **10.2.6 is right and I double-counted.** The correct figure is
  `1 + 568 + C(56,2) + C(56,3) = 29,829`; I added the 56 band wires a second
  time on top of the 568 linear terms. Current production (558 wires, `B=46`)
  is `1 + 558 + 1035 + 15180 = 16,774`.
- **10.2.9 is right** — the Section 1 pointer should be 4.4.

**10.2.4 is also my error, and worth being precise about what it does and does
not touch.** `emit_band_fill_nl` unions `supports[earlier]` when a quadratic
term reads an earlier band wire; I read that line and still wrote "4–12 input
bits," which is direct fan-in, not transitive support. Transitive support can
compound geometrically in the band index. This invalidates weakness 5.4 as
written ("band functions are cheap to learn as functions of `x`"). It does
**not** touch the main attack, which regresses on the band wires as *columns*
and never needs their functional form. Both quantities should be measured.

**10.2.7 is right and it strengthens my own point.** My degree is row-mean minus
row-min, not the canonical row-median background, so the numbers are not
comparable to the production plates. More importantly, untrimmed endpoint
plateaus inflate rho — which is precisely why my degree-1 `rho = 0.866 / 0.910`
should be read as *no diagonal at all*, not as a weak one. The argmin traces are
literally pinned to columns 0 and 37 (the endpoints). Trimming makes the
degree-1 result cleaner, not weaker.

**One refinement to 10.2.1.** The narrowing is correct for *transfer*, but I'd
keep the mechanism claim intact and separate the two:

- established: on a raw, no-roll `[2,3,3]` build, the exact relation is
  present, and it lives in a band-restricted degree-3 feature space — 286 extra
  regressors, `rho = 0.997`, monotone trace.
- **not** established: that this transfers to Build B, to production scale, or
  through SSS + fmix.

"Roll is carrying nearly all the security" should read as 10.2 proposes. I'd add
the operational consequence: because Build A is identified only by a timestamp
and is no longer reproducible from the tree, **the fix is to stop relying on
Build A entirely** and re-derive the no-roll condition from Build B through the
`ProdConfig` CLI knob (Section 9, Tier 1 item 4). That makes 4.1/4.2/4.4
reproducible *and* isolates the roll confound in one move.

**One refinement to 10.2.3, which I think is the strongest technical point in
the addendum.** Agreed without reservation: whole-circuit equivalence does not
imply preservation of intermediate-prefix degree, and I should not have assumed
it. Two things to add. First, there is a cheap empirical handle — measure the
minimum predictor degree that recovers the decode relation *as a function of
mixing depth*, per SSS round and at fmix checkpoints. That converts an
assumption into a curve. Second, my own 4.3 is weak evidence against fast degree
growth: after 2 SSS rounds a cubic predictor over the true band still gave
`depth 0.2118 / rho 0.966` versus `0.1774 / 0.905` at degree 1. Two rounds is
not fifteen-x fmix, and I am not resting anything on it, but the degree did not
visibly explode.

**10.2.2 is fair and I accept it as stated.** Twin detection is a fingerprint.
I reported 83 candidate pairs against ~88 expected from 59 rolls; I did not
measure precision/recall against ground truth, did not group triples into
rolls, and did not test survival through mixing. It is cheap to close — the
emitter can log ground-truth roll events behind a debug flag — and until it is,
"every roll becomes a known swap" is a conjecture.

**10.2.5 accepted.** `14/64 = 0.21875` assumes two unbiased independent cubic
terms; the actual terms share cascaded band variables. Diagnostic model only.

**10.2.8 accepted**, and it is the most actionable item in 10.2.

**Q2 — should 10.3 lead: yes.** A production-scale result on a committed
artifact outranks an `n=12` demonstration, and Section 9 item 2 is partly
obsolete — I proposed building a reader that already exists at `2a807d28`. I
verified the commit and the `stat_final.bin` / `stat_final.meta.json` artifacts
are present there.

Two conditions on promoting it. First, the caveats must travel *in* the lead,
not in a footnote: it is a known-source alignment result requiring the reference
`C`, not a blind distinguisher, and candidate selection and scoring share
samples. Second, the two statistics need separating, because they are not
equally exposed to that flaw. The **rho** result is in good shape — permuting
row labels is exactly the right null for "does the peak column advance
monotonically," and `p < 10^-5` over 100,000 permutations with peak columns
advancing 7 → 29 is a real finding. The **peak height** (0.7034 against a 0.5257
floor) is the number that needs the train/holdout rerun, because the max over
`1 + 558 + C(558,2) ≈ 156,000` candidates is selected on the scoring samples.
The matched random-target floor controls much of that bias, but not all of it.

**Q3 — correctness and provenance first: yes, unconditionally.** 10.5 is
independently confirmed: `shoot_random_gate` in `src/random/random_data.rs:873`
opens `rand::rng()` — the thread RNG — rather than taking the seeded stream, and
it is on the path of `nonlinear_gadgetize_cnot_with_config`, `gadgetize_cnot`
and `tdp2n_xgates`. Gadgetization is therefore **not reproducible from its
advertised seed**, which undercuts every A/B in this document including my own.
Fixing that is prerequisite to any comparison, not a cleanup task.

I would add one item to "attacker-side normalization," and Section 11.3 is why:
**helper normalization**. Every measurement in the repository fixes the helper
block at zero or samples it randomly. Both are non-adversarial choices for a
construction whose own tests certify correctness under *arbitrary* helper input.

**Q4 — which threat model.** My position: the intended claim should be
**indistinguishability of equivalent circuits**, with the heatmap demoted to a
diagnostic.

The reason is structural. Every heatmap variant — degree-1, degree-3,
`hmap_stat`, all of mine — requires the reference circuit `C` as the row axis.
That makes them *known-source alignment* measures. They can show a leak, and
they are useful for that, but they cannot serve as the definition, because an
obfuscation claim cannot be conditioned on the adversary already holding the
plaintext circuit. A distinguisher that needs no reference is strictly stronger
evidence in both directions.

Of the tests in either list, **10.4.B is the only one that measures the actual
definition** — paired equal-size, equal-function sources, classified without
any reference — and it is the one that failed, with disjoint ranges across ten
seeds per class. That should be the primary metric. SAT hardness is a separate,
weaker, and much noisier claim and should not be load-bearing.

### 11.2 Escalation of 10.4.D: the helper gauge is not just a gauge, it is a collapse

**Current qualification.** The forward solve below chooses the helper as a
function of `x`; it is not one constant helper vector that collapses all
samples. It proves a raw witness-preserving adaptive helper manifold, not an
automatic post-fmix recovery attack. Section 0.7 (W4/W5) is the current scoped
statement.

This is the one place I think the addendum understates its own finding. 10.4.D
proposes sweeping "zero/one/sparse/random and optimized helper assignments" and
keeping the best. There is a specific assignment, computable in `B` circuit
evaluations, that removes the nonlinear layer entirely.

**The argument.** `emit_band_fill_nl` is strictly triangular: band wire `i` is
updated only by gates reading data wires and band wires `j < i`, and wire `i`'s
own initial value enters purely additively, since it is the target and never a
control of its own update. So

```
band_final[i] = h[i] XOR F_i(x, band_final[<i])
```

and the forward solve `h[i] := F_i(x, 0)` drives every band variable to zero.
Then every mask slot

```
P = AND_t (b_t XOR a_t)   with all b_t = 0
  = AND_t (a_t)           a constant
```

so the decode `v = c0 XOR c1 XOR P_1 XOR P_2 XOR const` is **purely affine**.
The construction degenerates to the affine two-share gadget — the one this
project's own production runs measure at `rho = 1.000`. `roll` does not save it:
rolling exchanges wire *contents*, so a band variable whose value is zero stays
zero wherever it is moved.

**Measured**, on a current-tree `n=8` TDP4n nonlinear artifact (3,014 gates, 44
wires = 32 TDP + 12 band, public `Y=0x38 Z=0x74`, 1,024 samples):

| check | result |
|---|---|
| greedy triangular solve, 12 evaluations | all 12 band-home popcounts **0** at the post-fill prefix |
| band stays zero under `roll=1` (87 rolls) | 11 / 12 / 12 of 44 wires identically zero across samples at 30% / 50% / 70% depth |
| output contract, helper = 0 | holds |
| output contract, helper = random | holds |
| output contract, helper = **zeroing** | **holds** |

The gauge is witness-preserving: the advertised middle-block contract
`Y XOR C(x)` is satisfied under the collapsing helper. At production
(`B = 46`) the solve costs 46 evaluations.

**This is confirmed by the construction's own test suite.**
`nonlinear_tdp_random_slice_fixes_yz_and_leaves_w_and_band_free` asserts
correctness for randomly chosen `band` across 32 draws per input. The freedom
the test certifies as a robustness property is exactly the freedom the attack
consumes.

**Scope, honestly.** Verified on the raw gadget only. Post-SSS/fmix the solve
needs an observable to solve against, and reading "band-home wires at the
post-fill prefix" is no longer available — so it reduces to the same band
identification problem, but now as an **adaptive `O(B)`-probe oracle attack**
rather than a regression, which is a materially easier problem than anything in
Section 4. Whether a greedy bit-by-bit search on a downstream statistic (ridge
depth, residual mask rank, live-cone size) converges through full mixing is
open, and is the version worth testing.

**Note the internal inconsistency this exposes.** The non-TDP entry point
`nonlinear_gadgetize_with_slice_zero_cnot` closes this hole — its
`slice_zero_ccnot_preblock` is the identity *only* when aux and band are both
zero, so a nonzero band corrupts the data. The TDP production path deliberately
opens it. The two entry points are operating under different threat models, and
the production one has the weaker of the two.

**Implied fix, cheaper than anything in Tier 2:** remove the helper freedom.
Derive the band from `X` alone, or extend the slice contract to cover the band
as the non-TDP path already does. Note that value-refresh (Tier 2 item 8) does
*not* fix this on its own — if the initial values are attacker-chosen and the
refresh is deterministic, the attacker solves the whole chain.

### 11.3 Further weaknesses

**A. Both refresh mechanisms permute labels, never content — and the attack only
reads content.** This unifies several findings that have been treated
separately.

- `resource` swaps mask slot `P_old` for `P_new`, both degree-3 monomials over
  the same `B` variables. The feature space `span{all C(B,3) monomials}` is
  **unchanged**. A regression over that span is exactly invariant to
  re-sourcing. `rsrc=1` costs two slot emissions per source-gate gap and buys
  nothing against the attack that matters — it defends only against an
  adversary who guesses *specific* monomials.
- `roll` changes which physical wire holds a band variable. The regression needs
  only the *set* of `B` columns, and 10.4.A's value-signature tracker recovers
  that set by content regardless of location.
- **Neither mechanism ever changes a band variable's value as a function of
  `x`.** Nothing in the construction does, between the initial fill and the
  final mirror fill.

So the construction spends gates refreshing labels while the feature space that
the attack lives in is invariant to both. 10.4.A is the empirical face of this;
10.2.1's narrowing of my roll claim is another. Stated once: **the design
refreshes names, and every working attack reads values.**

**B. Historical shorthand: a `B`-variable nonlinear bottleneck.** Calling this
“`B` bits of security” was too strong. Every one of the `n·k` mask terms is a
function of the same `B`
band variables. Conditioned on those `B` bits, the entire encoding is affine.
This is a latent-dimension/enumeration ceiling, not a demonstrated secret-key
strength. At production `B = 46`; the automatic policy grows approximately as
`sqrt(n)`, but an explicit band-size knob exists, so “doubling B requires 4x
the physical wires” applies only when that automatic formula is held fixed.

**C. The measurement is taken at the helper value an attacker would never
choose.** `hmap_affine_fixed_slice` fixes non-input wires at zero, or samples
them independently per batch with `--random-helper-start`. For a construction
whose tests certify correctness under arbitrary helper input, both are
non-adversarial. Given 11.2, helper choice is not a nuisance parameter — it is
an attack surface, and the plates were all drawn at its most favorable point.
Any revised reader should take helper assignment as an explicit argument and
report best-of-K over adversarial choices, not just zero.

**D. Two mechanisms are aimed at adversaries that the evidence says do not
exist.** `emit_transvection_mixed` exists to stop rolled band locations being
identifiable "by width-1 writes alone" — but 4.5 finds its camouflage leaves an
83-instance polarity-twin signature, and 10.4.A bypasses location tracking
entirely by fingerprinting values. Similarly, `ProdLedger.used` enforces global
slot distinctness, which defends against slot collisions between values while
the actual attack regresses over the whole span. Both should either be justified
against a stated adversary or dropped; carrying them costs gates and, in the
`emit_transvection_mixed` case, adds a fingerprint that would not otherwise
exist.

### 11.4 Revised ordering

I accept 10.6 as the sequence, with two changes:

- **Step 0 gains an item.** Add the `shoot_random_gate` thread-RNG fix
  (`src/random/random_data.rs:873`) explicitly — without it no A/B in this
  document, mine or the addendum's, is replayable.
- **The helper-gauge test moves from step 4 to step 1.** It is `O(B)`
  evaluations, it is the cheapest attack anyone has proposed, and if the greedy
  variant converges through full mixing it moots most of steps 2, 3 and 6 —
  there is no point measuring degree-3 recovery of a nonlinear layer that the
  attacker can switch off. It belongs in the honest attacker baseline alongside
  public-slice partial evaluation and affine-tail elimination, because it is the
  same kind of move: normalizing away structure that is not doing work.

Everything else in 10.6 stands, including the acceptance conditions, to which I
would add: **no helper assignment materially reduces the residual mask rank.**

### 11.5 Where I think this leaves the construction

Not as a verdict — as the hypothesis the measurements should be designed to
falsify.

Three independent lines now point the same way. The band is a fixed
`O(sqrt(n))`-bit object (11.3.B); nothing ever refreshes its values (11.3.A);
and in the production TDP layout its initial value is attacker-chosen (11.2). If
all three hold up at scale, the nonlinear layer is not adding a hardness
parameter — it is adding a constant-size secret that the pipeline then spends
several million gates permuting the labels of. That is consistent with 10.4.C's
inversion, where the nonlinear raw artifact compressed 10.5% against the affine
control's 3.9%: more structure to remove, not less.

The counter-hypothesis worth taking seriously is 10.2.3 — that mixing raises the
degree of the decode relation at intermediate prefixes, so the attacks measured
on raw artifacts do not transfer. That is a real possibility, it is measurable
as a curve, and it is the single experiment that would most change my reading.

---

## 12. Claude response to the Section 0 consolidation

Status: **2026-07-27.** Accepts Section 0's corrections, adds production-scale
measurements on the accepted v05 artifacts, reports two errors of my own found
while making them, and reviews the new sliced-sandwich code. No construction
code changed.

### 12.1 Corrections accepted

Section 0 corrects several of my Section 4/11 claims and it is right in every
case. Recorded explicitly:

- **The endpoint/tie artifact (0.4).** Accepted, and it is the same artifact I
  identified in my own data in 11.1 — my degree-1 `rho = 0.866 / 0.910` came
  from argmin traces pinned to columns 0 and 37. Section 0 generalizes the
  point correctly: a legacy `rho = 1.000` on a core that is flat at `H = 0.5`
  is a tie-breaking artifact, not a diagonal.
- **"B security bits" (W2).** Withdrawn. The defensible statement is an upper
  bound on what the nonlinear layer can contribute — conditioning on the band
  makes the decode affine — not a measured secret-bit count. The band is
  public and not shown to be independently distributed.
- **"Subquadratic" (0.6).** Withdrawn. I compared regressor *counts*; the GF(2)
  basis solve is superlinear in that count, so the count comparison in 4.6 does
  not license a claim about attack *cost*.
- **Helper collapse scope (W5, 0.10).** The narrowing is right and my section
  title in 11.2 was too broad. What survives is precise and worth keeping in
  that narrow form: for each `X` there is an `X`-adaptive helper computable in
  `B` evaluations that zeroes the band, and I measured that it is
  witness-preserving. That is a statement about the raw semantic construction
  and about existential/SAT witness models. It is **not** a claim that one
  fixed helper vector collapses the band for all `X`, and therefore it does not
  invalidate fixed-helper or random-helper heatmaps. Section 0's four-way split
  of helper policies is the right frame.
- **Roll, twins, transitive support, prefix degree, 0.21875, fragment counts.**
  All accepted; most were already accepted in 11.1.

### 12.2 Two errors of my own, found while measuring

Reported because they affect how much weight the numbers below should carry.

1. **The empty conjunction is `1`, not `0`.** My first liveness pass evaluated
   a zero-width gate as never firing, which silently converts every X gate
   (`comp=0`, no controls) into a no-op. The v05 post-fmix artifact has 14,764
   such gates. The uncorrected run reported 15.86% never-firing; the corrected
   figure is 8.64%. All numbers in 12.3 are post-fix. This is the same
   distinction the new `XGate::is_noop` encodes — `comp=1` with no controls is
   the no-op, `comp=0` is the X gate.

2. **A syntactic "contradicts the public Y/Z" test that I had to discard.** It
   reported 71.72% of the post-fmix circuit as provably dead in one pass, which
   contradicted the 8.64% simulation result. The simulation was right: `y` and
   `z` wires are *written* during the circuit (14,730 and 16,927 times in the
   constructed artifact; from gate 0 and gate 1 in the post-fmix artifact), so
   a literal on those wires is not pinned at execution time. The syntactic
   argument is valid only on the prefix before the first write to `y`/`z` —
   in the constructed artifact that is exactly the 4,096-gate preblock Section
   0.8 already identifies as dead, and post-fmix it is empty.

### 12.3 New measurement: contract-aware effective size across all three v05 stages

Section 0.8 measures `fcompress` on the constructed artifact only. This is a
different and cheaper question — how much can an attacker delete with **no
rewriting at all**, using only (a) gates that never fire on the advertised input
ensemble and (b) gates outside the backward live cone of the advertised output
block `128..256`?

Ensemble per the artifact's own sidecar: `x` free, `Y`/`Z` pinned, helper
`384..558` free. 16,384 samples.

| Stage | Gates | No-op | Never fires | Outside live cone | Surviving | Removable |
|---|---:|---:|---:|---:|---:|---:|
| constructed | 72,831 | 0 | 1,889 (2.59%) | 5,837 (8.01%) | 65,105 | **10.61%** |
| after 5 SSS rounds | 69,836 | 0 | 2,723 (3.90%) | 3,181 (4.55%) | 63,932 | **8.45%** |
| after fmix | 1,065,824 | 0 | 92,085 (8.64%) | 587 (0.06%) | 973,152 | **8.69%** |

The never-fire counts are not an artifact of sampling wide conjunctions. At
16,384 samples the expected miss rate is `0.000` for every width up to 11 and
`0.018` at width 12, and the inert gates are spread proportionally across all
width classes rather than concentrated in the wide ones.

Three readings, one of which cuts against my own prior framing:

1. **The expansion ratio is robust to this normalization.** Advertised
   `1,065,824 / 69,836 = 15.2618x`; normalized to surviving gates,
   `973,152 / 63,932 = 15.2217x`. A **0.26%** change. Because the removable
   fraction is nearly the same at both ends, it does not inflate the ratio. My
   earlier instinct in Section 5.6 — that slice-inert padding overstates the
   work — is **not** supported here and should be dropped as stated.

2. **fmix eliminates the dead cone, and this is favorable evidence.** The
   fraction outside the output live cone falls `8.01% → 4.55% → 0.06%`. Mixing
   genuinely entangles the circuit into the advertised output. Any attack
   premised on pruning a large irrelevant sub-circuit does not apply to the
   post-fmix artifact.

3. **The inert fraction grows 3.3x through fmix** (`2.59% → 8.64%`, 92,085
   gates). This is the one number worth following up: it is stable, cheap to
   detect, and it is structure that mixing is adding rather than removing. It
   is not by itself a reconstruction result.

### 12.4 A refinement to Section 0.4: the isolated post-SSS cell is seed-invariant

Section 0.4 reports one post-SSS core cell at `H = 0.49609375` — exactly one
target-bit-equivalent below the `0.5` background — and correctly concludes that
one cell cannot define a chronology. I would not leave it there.

The three post-SSS matrices are byte-identical
(`9f454314bc6a699163aaa1e91cfd1609ffd7e4c047f52abc68bfc1d50991c023` for seeds
12345, 23456 and 34567), while the three post-fmix matrices have **three
distinct hashes**. So the analysis seed does reach the sampling; the post-SSS
map is genuinely seed-invariant, and so is the departing cell.

That is what an **exact algebraic relation** looks like — span membership either
holds or it does not, independent of the sample draw — and it is not what
sampling noise looks like. The cheap follow-up is to re-run that single cell
`(source prefix 800, mixed prefix 51,336)`, dump which target bit is recovered
and the recovered affine support, and check whether it is a genuine leak on one
logical bit or a boundary/port effect. Conversely, the post-fmix flatness is
strengthened by the same observation: three different sample draws all give
exactly `0.5`.

### 12.5 Review of the new sliced-sandwich code

`sliced_sandwich_with_d` builds `[C || S1] ; N ; [D || S2]` on `2n` wires with
the contract `G(x, 0…0)[n..2n] = C(x)`.

**The raw sandwich decomposes completely by syntax.** Measured on an `n=16`
artifact (544 gates, `C=200`, `D=200`, `s=64`), using three predicates and no
simulation:

| Predicate | Recovered | Expected |
|---|---:|---|
| targets a wire `>= n` | 16 | exactly the `n` N gates |
| targets `< n`, reads `>= n` | 128 | `S1 + S2 = 2s` |
| touches only wires `< n` | 400 | `C + D` |

All 16 N gates match the exact form `CNOT(n+w, w)`, and after floating, the N
column spans gate indices 239–317 — a 78-gate window, **14.3%** of the circuit.
So the C/D boundary, which is the chronology the heatmap exists to measure, is
localized to a seventh of the raw circuit by inspection alone. The float step
changes N-gate *positions* but not the fact that they are the only gates
targeting the upper half.

**`float_extremal` is a weaker randomization than the one used elsewhere.** It
floats each N gate all the way to its commutation extreme in a randomly chosen
direction. `commuting_insertion_pass`, used by `commuting_shuffle`, instead
draws `rng.random_range(0..=span)` — a uniform position within the commuting
window. Extremal placement is a distinguished, lower-entropy choice: a gate at
its extreme sits immediately adjacent to the first gate it collides with. If
the intent is to dissolve the central column, uniform placement in the window
is strictly better and costs nothing.

**Liveness: the gadget hides the sandwich's deadness, which is a point in its
favor.** At the logical level the sandwich is mostly irrelevant to its own
advertised contract — `S1` is inert (64 of 64 gates never fire, since the upper
half is zero before N) and `D + S2` are outside the live cone of `[n, 2n)` (264
gates), leaving 216 of 544 surviving: **60.3% removable**. After `[2,2,3,3]`
gadgetization, the same analysis on the 24,966-gate artifact finds only **3.6%
removable**. Shared carriers and the shared band entangle everything, exactly as
in 12.3's post-fmix result.

**But this exposes a threat-model gap that should be resolved explicitly.** The
advertised contract names only `[n, 2n)` as output. If that is really the
contract, `D` — the block whose entire purpose is one-wayness — is dead for it,
and 60% of the logical sandwich is padding. If instead the adversary sees all
wires (which it does; it holds the circuit), then `D` is live and the contract
understates what is exposed. The construction cannot rely on both readings.
Section 0.9's claim table should say which output view each experiment assumes.

### 12.6 Where I now stand

I agree with Section 0.1's executive conclusion, including the parts that
retract my earlier framing. Adding 12.3: the accepted v05 artifact also
withstands the cheapest attacker normalization — its expansion ratio survives
liveness/inertness pruning essentially intact (15.26x → 15.22x), and fmix
closes the dead cone to 0.06%. That is a second independent line of favorable
evidence alongside the flat exact heatmaps.

The unresolved items I would still weight highest are unchanged from Section
0.11: the held-out approximate reader (0.5 remains the strongest unfavorable
signal), the minimum-recovery-degree curve by stage (10.2.3), and the blind
presentation test (W7), which is the only test measuring the definition I think
the project should adopt.

---

## 13. Handoff notes

For the other session. Written so work is not repeated or silently
contradicted.

### 13.1 State of the record

- Section 0 is the citable assessment. Sections 1–9 are historical; **Section 11
  and Section 12 are my responses and should be read as amendments to 10 and 0
  respectively**, not as independent conclusions.
- Claims withdrawn by me and not to be re-cited: `B` as "security bits"; the
  band-restricted degree-3 attack as "subquadratic"; "the helper gauge collapses
  the entire nonlinear layer" (narrowed — see 12.1); "slice-inert padding
  inflates the expansion ratio" (measured false — see 12.3).
- Claims of mine that stand: the `[2,3,3]` raw Build A degree-3 result as a
  *mechanism* demonstration only; the `X`-adaptive witness-preserving helper
  solve (12.1, narrow form); "resource and roll refresh labels, never band
  values."

### 13.2 New numbers added in Section 12

All from the committed v05 artifacts under
`1_affine_tests/nonlinear_tdp4n_sss5_fmix152_noopfix_v05/`, at 16,384 samples,
`Y = 0x2587d0c6d5dbade6b97eae1e5a026f25`,
`Z = 0x30bcf03361057f25dafdbf7a287f5a80`, output block `[128, 256)`:

- constructed 72,831 gates → 65,105 surviving (10.61% removable)
- post-SSS 69,836 → 63,932 (8.45% removable)
- post-fmix 1,065,824 → 973,152 (8.69% removable)
- normalized expansion 15.2217x versus advertised 15.2618x
- zero `comp=1`-with-no-controls no-ops at every stage
- post-SSS map hashes identical across all three seeds; post-fmix hashes all
  distinct

### 13.3 Scripts

The Section 12 probes are scratch and were **not** committed. They are small
(each under 120 lines of numpy) and should be rewritten into
`experiments/nonlinear_gadget_analysis_20260727/` if the numbers are to be
cited. Behaviour to reproduce:

- `liveness_v05.py <circuit> <y_hex> <z_hex> <samples>` — no-op / never-fires /
  backward-live-cone census plus a per-width expected-miss-rate column.
- `sandwich_split.py <sandwich.mpmct1> <n>` — the three-predicate C/S1/N/D/S2
  decomposition.
- `liveness.py <circuit> <n> <lo> <hi> <samples>` — the zero-slice variant used
  for the sandwich.

**If you rewrite them, carry the fix in 12.2.1:** the empty conjunction
evaluates to `1`, so a zero-width gate fires unless `comp` is set. Getting this
backwards silently turns every X gate into a no-op and inflates inert counts by
roughly a factor of two on post-fmix artifacts.

### 13.4 Open questions I would like answered

1. **The isolated post-SSS cell (12.4).** Which target bit, what affine support,
   and is it a boundary effect? Seed-invariance says it is an exact relation,
   not noise. Cheapest item on the list.
2. **The 8.64% post-fmix inert fraction (12.3).** Is it concentrated in
   particular fmix move families? If a specific move manufactures gates that
   cannot fire on the advertised ensemble, that is a fixable inefficiency.
3. **Sandwich output view (12.5).** Is the contract `[n, 2n)` only, or all
   wires? This decides whether `D` is doing work.
4. **`float_extremal` (12.5).** Was extremal placement deliberate, or should it
   match `commuting_insertion_pass`'s uniform-in-window draw?

### 13.5 Standing disagreements

None outstanding. Every point of disagreement I raised in Section 11 was either
accepted by Section 0 or withdrawn by me in 12.1. The remaining differences are
about *priority*, not correctness — I weight the blind presentation test (W7)
higher than its current position, because it is the only experiment that tests
equivalent-circuit indistinguishability rather than known-source alignment.

---

## 14. Retraction of 12.3 post-fmix, and what replaced it

Status: **2026-07-27, second pass.** The Section 13 review raised three caveats
against Section 12.3. All three were correct. Acting on two of them
**falsified my own post-fmix result**, which is retracted below. The scripts are
now committed. A new and better-supported finding replaces the retracted one.

### 14.1 Retraction

> **Section 12.3's post-fmix figures are withdrawn.** "8.69% removable" and the
> normalized expansion ratio "15.2217x" are wrong. Do not cite them.

The review objected that "never fires" means "not observed firing in 16,384
samples," not formal inertness, and that my per-width miss-rate column assumed
independent unbiased controls — an assumption correlated circuit wires violate.
That is exactly right, and my width table was the weak part of 12.3. I replaced
the assumption with two direct measurements, and they disagree with me:

**Deletion check — post-fmix FAILS.** Rebuilding the circuit without the
candidate gates and comparing against the original on 8,192 fresh lanes from an
independent seed:

| Stage | Advertised block `[128,256)` after deletion |
|---|---|
| after 5 SSS rounds | **identical** — 0 of 128 wires differ |
| after fmix | **differs** — 128 of 128 wires differ |

Gates unfired on the census lanes fire on other inputs. Post-fmix, the
candidate set is not safe to delete, and the effective-size claim collapses.

**Convergence — post-fmix does not converge.** Unfired count against census
lanes:

| Lanes | post-SSS | post-fmix |
|---:|---:|---:|
| 2,048 | 2,748 | 136,111 |
| 8,192 | 2,725 (−23) | 96,881 (−39,230) |
| 16,384 | 2,723 (−2) | 92,085 (−4,796) |
| 65,536 | 2,723 (**+0**) | 91,367 (−718) |

Post-SSS is exactly flat across a 32x increase. Post-fmix is still falling at
65,536 lanes. The two stages behave completely differently and I treated them
as one phenomenon.

**Firing-count histogram — the review's "continuum" objection is confirmed.**
There is no gap at either stage; counts run down to 1. But the tail sizes are
not comparable:

| Gates firing on ≤ 8 of 16,384 lanes | post-SSS | post-fmix |
|---|---:|---:|
| count | 23 | 57,848 |
| of total gates | 0.03% | 5.43% |

Post-SSS, the observable tail is far too sparse to account for 2,723 unfired
gates, which is why that population is stable and its deletion verifies.
Post-fmix, the unfired set is plainly the continuation of an enormous tail.

### 14.2 What survives

| Claim | Status |
|---|---|
| post-SSS 8.45% removable, deletion-verified on fresh lanes | **stands** |
| post-fmix 8.69% removable | **retracted** |
| normalized expansion ratio 15.2217x | **retracted** |
| dead cone falls 8.01% → 4.55% → 0.06% | **stands** (structural, not sampled) |
| "fmix adds inert structure" | **retracted** — see 14.3 |
| zero `comp=1`-no-control no-ops at every stage | **stands** (syntactic) |

The one structurally safe post-fmix removal is the backward-dead cone: **587
gates, 0.06%**. That is sound without any sampling argument, since those gates
provably cannot reach `[128,256)`.

So the corrected reading is **more favorable to the construction than my
original one**, by a different route than I claimed. The post-fmix artifact
gives up essentially nothing (0.06%) to this normalization, while its post-SSS
input gives up 8.45%. fmix's expansion is denser than its own baseline by this
measure, not padded. I reached a roughly similar conclusion in 12.3 through
reasoning that was wrong.

### 14.3 The replacement finding: fmix shifts the firing-rate distribution

The rarely-firing gates are not inert padding, but the population is real and it
is a sharp, cheap, purely observational difference between the two stages:

| Gates firing on ≤ 64 of 16,384 lanes (p ≤ 0.4%) | count | share |
|---|---:|---:|
| after 5 SSS rounds | 503 | **0.72%** |
| after fmix | 250,528 | **23.5%** |

A ~33x shift. Roughly a quarter of the post-fmix circuit fires on under half a
percent of inputs. This is a measurable structural property of fmix output that
its input does not have, and it is obtained from one simulation pass with no
reference circuit `C` — so unlike the heatmap it is a **blind** observable.

Whether it is a weakness is genuinely open and I am not claiming it is. Two
directions worth separating:

1. **Blind stage/presentation distinguishing (W7).** A firing-rate histogram is
   a cheap feature vector. If it separates equal-function, equal-size sources
   the way target-wire entropy did in 10.4.B, that is relevant to the
   indistinguishability goal.
2. **Solver behaviour.** Gates that fire on <0.4% of inputs generate clauses
   that are satisfied trivially on almost all assignments. Whether that helps
   or hinders a CNF solver is exactly the kind of question Section 0.11 item 8
   says not to guess at.

### 14.4 Reproducibility — closed

The review was right that Section 12's scripts were scratch. They are now
committed under `experiments/nonlinear_gadget_analysis_20260727/`:

- `v05_effective_size.py` — the fixed-`Y,Z`/random-helper census, now carrying
  the firing-count histogram, the convergence ladder, and the deletion check.
  Usage: `v05_effective_size.py <circuit> <y_hex> <z_hex> [lanes] [verify_lanes]`
- `zero_slice_effective_size.py` — the all-other-inputs-zero variant used for
  the sandwich artifacts.
- `sandwich_split.py` — the three-predicate C/S1/N/D/S2 decomposition.
- `sandwich_probe/` — a regenerated durable `n=16` sandwich artifact plus its
  source and sidecar, from
  `gen_sliced_sandwich_2233 --n 16 --c-gates 200 --d-gates 200`.

All Section 12.5 sandwich numbers reproduce exactly from the committed scripts
and artifact: 16/16 N gates in the exact form `CNOT(n+w, w)`, N-column span
78/544 (14.3%), raw logical 60.29% removable, gadgetized 3.64% removable.

Every script carries the empty-conjunction warning from 12.2.1 in its header.

### 14.5 What would close the inertness question properly

Sampling cannot prove inertness; it can only fail to find firing. The rigorous
version is a reachability check: for a candidate gate, is its control cube
satisfiable at that point under the advertised contract? That is a SAT query
per gate. At 91,367 post-fmix candidates it is not free, but it does not need
to be exhaustive — a few hundred randomly sampled candidates gives a confidence
interval on the true inert fraction, and the post-fmix deletion failure already
suggests the answer is "most of them are not inert."

For post-SSS the question is closed well enough: stable count across 32x lanes
plus a verified deletion on independent samples.

### 14.6 On the four follow-ups

Agreed as listed, with one status change:

1. **Post-SSS cell (800, 51,336)** — unchanged, still the cheapest item.
2. **Reproduce and verify the three-stage census** — the scripts are now
   committed (14.4) and the verification was run (14.1). What remains is an
   *independent* re-run, not a reconstruction.
3. **Which fmix move families create the population** — this becomes more
   interesting under 14.3, not less: the question is now "which moves generate
   very-low-activity gates," which is a property worth understanding whether or
   not it is a weakness.
4. **Sandwich output contract** — unchanged, and still a question only the
   author can answer.

I have no disagreement with the review. It was right on all three points, and
acting on two of them overturned my own result.
