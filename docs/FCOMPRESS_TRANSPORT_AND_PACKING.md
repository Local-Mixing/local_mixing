# fcompress: transport gathering and canonical packing

*2026-09-05/06. Code: `src/postprocessing/compress.rs`, `src/postprocessing/downhill.rs`,
`src/postprocessing/bin/fcompress.rs`, `src/engine/format.rs` (anf1). Manual: `POSTMIX_MANUAL.md` §3.*

## 1. Summary

fcompress is the attacker-computable final pass of the GSS pipeline and the honest
effective-size evaluator of a mixed circuit. Three new gathering rules make it remove far
more than before, and a packing pass now spells the result in a canonical form.

| circuit | gates in | old pass (2026-08-22) | new pass | literals |
|---|---|---|---|---|
| A1 fmix benchmark (256 wires) | 607,365 | 575,396 (94.7%) | 420,279 (69.2%) | 55.3% |
| K2 hold-10 delivered final (512 wires) | 3,748,210 | fixed point (99.9%) | 2,165,945 (57.8%) | 47.7% |
| balanced K2 2-eff final | 3,067,544 | fixed point | 1,614,661 (52.6%) | 43.2% |
| K2 hold-30 2-eff final | 3,053,636 | fixed point | 1,607,266 (52.6%) | 43.3% |

Every number was checked by fcompress's own sampled global check (64 rounds × 64 lanes on all
wires) and by an independent evaluator sharing no code with fcompress, on 2,048 fully random
inputs over all wires. Consequences: effective sizes and "residual ≳ 90%" figures quoted before
2026-09-05 are about 1.7× too high; the crossing stage's ×1.8 expansion is mostly ladders that
slide back; hiding is unchanged (the pass is attacker-computable), only the honest size shrank.

Packing (default output since this change) turns the 1.6M-cube 2-eff finals into ~350k
generalized gates in algebraic normal form (ANF), at ~2.4× the term count, ~1.2× the raw bytes
and ~1.3× the xz-compressed bytes. The gain is not space but the removal of
representation information: the ANF is the unique spelling of each activation function.

## 2. Background: what the old pass could not see

The old pass gathers same-target gates that can float to a common point, XOR-reduces each
group (pairwise catalogue, then exact/ANF re-expression when it wins), and interleaves an
adjacent conjugation-descent ("downhill") pass. A group closed on any read of its target and
on any write to any wire it reads.

A census of the removable identical-gate pairs left in a delivered final
(`reports/collision_identities_20260905/`, section "Anatomy") showed that 99% of them are
*conjugated copies*: a control `u` of gate `g` is flipped in between by a gate whose condition
is implied by `g`'s other literals, and the second copy reads `u` with the opposite polarity.
Example from the K2 final, gates 837–839:

```
w398 ^= AND(!w146  w356 w442)
w356 ^= AND(!w146       w442)
w398 ^= AND(!w146 !w356 w442)
```

Both toggles equal `!w146 & w356_old & w442`; deleting the pair changes nothing. This is the
three-gate Toffoli relation `g · X = X · g'`. The old gatherer never formed the group: the
flip writes one of `g`'s controls, which closed it.

## 3. The new compression rules

All three are on by default and each has a `--no-*` flag; with all three off the pass
reproduces the 2026-08-22 output byte for byte.

### 3.1 Transport (Toffoli sliding at any distance)

When a gate `h` writes a wire `u` that an open group `T` (target `t`) reads, and `h` does not
read `t`, the group is no longer closed: its ESOP is conjugated across `h` by the substitution
`u ← u ⊕ fire(h)` (the downhill algebra, exact because every gate is an involution):

```
T · h  =  h · T'        T' = T[u ← u ⊕ fire(h)]
```

For a cube `L·u` this gives `L·u ⊕ L·fire(h)`; when `fire(h)`'s cube is implied by `L` the
second term collapses to `L`, so `L·u` becomes `L·¬u` and the conjugated copy of the pair above
meets it and cancels. The transport is accepted when the catalogue-reduced result does not
grow in (gates, literals); otherwise the group closes as before. `--transport-slack N` allows
N extra cubes speculatively (measured worse; default 0).

**Frames.** A transported group's cubes are written in the frame after `h`, so the group
records a dependency on `h`'s group and must be emitted after it. Closing a group closes its
dependencies first; a close set is emitted in dependency order (ascending last-member order
among independent groups). Two open groups with no dependency path between them commute,
which makes any such order legal; the dependency graph is kept acyclic by construction (a
transport that would close a cycle closes the group instead). A subtle ordering matters: all
candidates for a write are decided first, the refused ones are closed, and only then is the
writer's slot fixed and the accepted transports applied — a refusal-close can cascade into the
writer's own group or into an already-transported candidate, which would leave a stale frame.

### 3.2 Separation-aware reads

A reader of `t` that shares a wire with every member of the group at the opposite polarity
fires on disjoint inputs from each of them, so it commutes with the whole group and the group
floats past it instead of closing (`XGate::collides` is false).

### 3.3 Reversed-list gather

Each iteration also gathers the reversed gate list. Reversal is exact (every gate is an
involution, so the reversed list computes the inverse function), and a forward gather of it
is a leftward float of the circuit. The crossing walk floats gates both ways, so a case-split
ladder left by a leftward crossing is only reachable from the left; the adjacent downhill pass
sees only immediate neighbours.

### 3.4 Guard, determinism, verification

Gathering with transport can in principle end an iteration larger (a transported group may
reduce worse than its parts would have separately), so an iteration that grows in (gates,
literals) is discarded and the previous circuit kept; this fired only in the slack
experiment. The output is deterministic and independent of the seed (which feeds only the
verification samplers). Verification: per-group exhaustive/sampled checks, an 8×64-lane check
of every transport span, the global check of the written artifact against the input, and,
for the results above, the independent evaluator.

### 3.5 Results and ablations

A1 benchmark, 607,365 gates, each rule removed in turn:

| variant | gates out | % |
|---|---|---|
| all rules (default) | 420,279 | 69.2 |
| 20 iterations instead of 10 | 419,411 | 69.1 |
| no downhill | 428,342 | 70.5 |
| no separated reads | 446,480 | 73.5 |
| no reverse gather | 491,572 | 80.9 |
| transport slack 1 | 519,727 | 85.6 (regressed at iteration 4) |
| no transport | 563,413 | 92.8 |
| old pass | 575,396 | 94.7 |

Transport is the lever; the other two rules amplify it. Runtime roughly doubles (A1: 296 s →
615 s; the 3.7M-gate K2 final: 13 min). On the K2 final the first iteration accepted 803k
transports, found 1.93M no-op ones (the group commutes with the writer), refused 773k on cost
and 354 for acyclicity, and passed 574k separated readers; over ten iterations 2.9M transports
were accepted and 4.7M refused. The removable-pair census on that circuit drops from 8,972
pairs (8,716 semantic-only) to 3,867 (2,144), mostly pairs the last iteration re-created.

Open leads: run on the server's post-crossing intermediate directly; the new output holds
2,192 gates that never fire on 8,192 random inputs (old: 17), a value-level dead-gate or
wire-equality lead; the refusal policy.

## 4. Packing

### 4.1 Method

At a fixed point of the gather, every maximal run of consecutive same-target gates is one
gathered group: a group floats to one close point and is emitted there, and runs that could
still float together would have been merged. Packing spells each run as one generalized gate
`t ^= f(controls)` with `f` in algebraic normal form, the XOR of positive monomials over the
run's support, obtained by expanding every cube (a negative literal is `1 ⊕ w`, so a cube with
`k` negative literals expands to `2^k` monomials) and cancelling duplicates; a `comp` bit is
the constant monomial. Any support size is handled. The count of packed gates equals the run
count; the packed file is exact (verified as part of fcompress's global check) and, like the
rest of the pass, attacker-computable.

### 4.2 The anf1 format

```
anf1 <wires> <gates>
<target> <n_monomials> [<degree> <w_1> ... <w_degree>]*      (one line per packed gate)
```

Wires ascend inside a monomial, monomials are sorted by (degree, wires), degree 0 is the
constant 1. Example: `168 5 0 2 104 261 3 104 261 262 3 104 261 397 4 104 261 262 397` is
`w168 ^= 1 ⊕ w104·w261 ⊕ w104·w261·w262 ⊕ w104·w261·w397 ⊕ w104·w261·w262·w397`.
`format::read_mpmct` recognises the header and loads an anf1 file as its monomial expansion
(one positive cube per monomial), so every existing consumer accepts packed circuits;
`format::read_anf1` gives the packed gates themselves. `fcompress` writes anf1 by default,
`--no-pack` writes the mpmct1 cube circuit, `--pack-census` prints the statistics below.

### 4.3 Rationale

The cube list the reducer emits is the catalogue-reduced descendant of whatever cubes the
mixer left, so the same activation function comes out as different cube sets depending on the
path that produced it: the spelling carries history that is not needed for evaluation and
depends on the original circuit. A function-determined representation removes that channel.
Among the practical choices, the ANF is the unique one that needs no algorithm to specify; a
deterministic ESOP derived from it would be equally leak-free and smaller, at the cost of
committing to a heuristic.

What packing does not canonicalize: which gates end up in one group, where a group is
emitted, the order among commuting groups and the frame a transported group is written in
are all decided by the gather from the input order, which the mixer's history determined. A
fully canonical circuit would need rules for those too and is out of reach; the pipeline's
randomization is relied on for them, and packing removes the per-gate spelling only.

### 4.4 Measurements (compressed K2 2-eff finals)

| | hold-30 2-eff | balanced 2-eff |
|---|---|---|
| cubes (gates before packing) | 1,607,266 | 1,614,661 |
| packed gates | 347,055 (21.6%) | 351,546 (21.8%) |
| ANF monomials | 3,820,336 (2.38× cubes) | 3,840,880 (2.38×) |
| wire occurrences, cubes → ANF | 3.85M → 8.20M (2.13×) | 3.86M → 8.20M (2.12×) |
| mean cube width / mean monomial degree | 2.40 / 2.15 | 2.39 / 2.14 |
| monomials per packed gate | median 4, p90 31, p99 71, max 508 | max 350 |
| multi-cube runs | 57.6% of runs, 90.8% of the gate mass | 57.1%, 90.7% |
| file bytes, mpmct1 → anf1 | 34.4 MB → 41.2 MB (1.20×) | 35.1 MB → 41.6 MB (1.19×) |
| xz -9 | 6.51 MB → 8.58 MB (1.32×) | 6.56 MB → 8.66 MB (1.32×) |
| gzip -9 / bzip2 -9 | 8.25 → 10.6 MB / 6.80 → 9.16 MB | 8.38 → 10.8 MB / 6.86 → 9.21 MB |

Under xz both forms sit close to a naive entropy count (about 32 bits per cube gate, about 18
bits per monomial), so neither hides much structure from a generic compressor, and the
canonical form costs about a third more information. On the delivered K2 hold-10 final the
same packing gives 449,431 packed gates for 2,165,945 cubes (5.20M monomials, 2.40×); a
deterministic ESOP re-derived from the ANF would be within 5% of the cube count.

**Evaluation cost.** Evaluating the ANF literally costs one AND per wire occurrence, one XOR
per monomial and one per gate: 12.4M word operations against 5.5M for the cube form (2.27×).
The compiled 64-lane evaluator runs the 1.6M-cube hold-30 circuit in about 9.5 ms per batch of
64 inputs, so the literal ANF evaluation is about 22 ms. The canonical form fixes the
description, not the evaluation strategy: an evaluator may re-derive any equivalent form once
(the deterministic ESOP at ~1.05× the cube cost, or truth tables over small supports).

### 4.5 Deliverables

`circuits/bv5gss128_k2h30_2eff_final_fc2.anf1` (347,055 packed gates) and
`circuits/bv5gss128_bal_k2_2eff_final_fc2.anf1` (351,546), the packed forms of the two
re-compressed 2-eff finals; each was sanitized (header plus gate lines only, monomials
checked sorted and in range) and verified equal to its cube-form source by the independent
evaluator on 2,048 random inputs. The pipeline (`scripts/gss_mix.sh`) now writes
`final.anf1` as the stage-6 deliverable and logs both the cube residual and the packed count.
