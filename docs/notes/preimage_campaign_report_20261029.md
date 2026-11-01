# Preimage cryptanalysis campaign — full report

*2026-10-29. Supersedes the interim summary of 2026-10-26.*

## 1. Objective and setup

The target is the declared mixed output of the baseline / as-is run of the TDP pipeline:
a reversible circuit **F** on 256 wires, **666,745 gates**, computing a 64-bit
permutation **C** on the honest input slice (`x` on wires 0–63, zeros elsewhere; the
payload `C(x)` on wires 64–127 at the output).

**Goal.** Find `x` with `C(x) = 0^64`, using only F and the attacker's own rewrites of
it — never the secret C, and never the census's C-derived labels. C appears only in
clearly separated post-hoc measurements of attack success. Every candidate `x` is
verified against the real circuit.

Three equivalent forms of the same C were used as targets, all verified equivalent:

| circuit | gates | origin |
|---|---|---|
| F | 666,745 | declared baseline/as-is mixed output |
| plateau | 355,990 | C-blind ANF resynthesis of F (30 rounds) |
| short-279k | 279,263 | fcompress shortening sweep (`fc4_c5_n64`) |

## 2. The construction (public)

The Classic sliced sandwich on 128 wires (before the stage-2+ pipeline expands it):

```
A = [ C interleaved with S1 ] ; N ( y ^= x ) ; [ D interleaved with S2 ]
```

C is the 1000-gate source on the low half; S1 is a slice block that is dead on the
honest slice (it reads the zero high half), so block 1 computes pure C. N copies the low
half (C(x)) to the high half (the payload). D is a fresh 1000-gate random circuit on the
low half, interleaved with the live slice S2. So the internal values split into: **C
internals** (block 1, functions of x), **D internals** (block 2, functions of C(x)), and
**stage-2 randomness** (the S1/S2 slices plus the preprocessing masks, all low degree).
The pipeline (preprocessing, mixing, split, crossing, packing, ANF resynthesis) then
reshuffles and re-expresses all of this into F.

## 3. Attacks and outcomes

### 3.1 Direct SAT inversion
CNF encoding (one variable per wire and gate birth; Toffoli as AND-aux + native XOR
clause), non-input wires fixed to 0 at entry, payload fixed to 0 at exit, CryptoMiniSat.
Run on F and on the plateau. **No solution** after 30+ hours each.

### 3.2 Structural reconstruction (vocabulary walks)
Build the vocabulary of every function born on any wire of the derived circuits, then
walk g57 gates from the inputs and from the payload, accepting a gate when the implied
value is in the vocabulary. **No chain.** The vocabulary covers 684/1000 (7 circuits) to
735/1000 (65 circuits) of C's births, but membership does not identify a genuine C gate:
the resyntheses and remixes contain many non-C values of exactly the tested form. Forward
walks were spurious from the first gate; backward walks stalled after 20–40 gates.

### 3.3 The replication landmark
The raw sandwich copies C(x) at the midpoint, so an unmixed circuit has an interior cut
carrying a clean C(x) bundle to constrain. Tested C-blind at three levels:

- **Verbatim / up to NOT — absent.** Over every cut of F and its rewrites, the count of
  functions carried by ≥2 wires peaks at 15–21 and only at the very start (the
  input-copy step); it is 0 through the interior. F's 64 output functions are never
  co-present at an interior cut (all 64 only at the last gate).
- **Affine (degree-1) — weak.** F peaks at 10/64 output bits in the affine span of the
  live wires at depth 0.485; the remixes at 5. Insufficient.
- **Degree-2 — the finding (§4).**

### 3.4 Exploiting the degree-2 structure
All three are sound (added constraints are implied by `payload = 0`) and self-verifying:

- **Two-circuit linked SAT** — plateau and F over a shared input, both payloads 0, plus
  clauses equating internal variables that coincide on samples. The two equivalent
  circuits share **42,771** identical internal functions. Variants with 6k and 30k links.
- **Hint-augmented SAT** — full plateau + the sparse degree-2 relations imposed at one
  cut (50 hints) and two cuts (94 hints, 59 of 64 output bits).
- **Prefix-only SAT** — encode only the prefix up to the degree-2 cut, impose the
  relations, enumerate the prefix's satisfying inputs against the full circuit. Plateau:
  59 constraints pin the input to ~2^5 candidates over 178k of 356k gates. short-279k:
  earlier midpoint gives a 112k-gate prefix but only 40 constraints.

**No candidate or preimage from any variant.** The prefix solve — the smallest,
best-shaped instance — produced zero satisfying inputs after 16+ h (279k) and 24+ h
(plateau).

### 3.5 Sub-280k divert
A fcompress shortening sweep produced an equivalent circuit at 279,263 gates (≈21%
smaller than the plateau, 58% smaller than F), verified equivalent. Its degree-2 midpoint
exposure peaks at **40/64 at depth 0.40** (sparser combinations, median 6 terms), below
the plateau's 50 but with an earlier, shorter prefix. The prefix-only solve on it ran
≈30 h with **no candidate** — the shorter prefix did not cross the tractability
threshold. The incidental trend: ANF resynthesis *increases* degree-2 exposure (666k→36,
356k→50), while fcompress shortening does not, so the low-degree leak comes from
resynthesis, not compression in general.

### 3.6 C-blind reconstruction of C from repeated values
The last attempt, fully C-blind: take F's repeated internal values (functions on ≥2 wire
segments), order them by first appearance, drop the low-degree (stage-2) ones, and
forward-chain g57 gates from the inputs in appearance order — whatever connects back to
the inputs is C, since D is computed only from the payload. Result on F:

- 52,139 repeated values; 233 affine (dropped as stage-2); 51,906 high-degree candidates.
- Forward chaining reconstructed **60 gates** into **37 connected fragments**, largest 6.

Post-hoc score against the real C (measurement only, not part of the attack): **of the
51,906 high-degree repeated values, only 177 are genuine C internals**, and **only 1 of
the 60 reconstructed gates is genuine C.** The fragments are spurious g57 coincidences.
C-blindly there is no signature (degree, repetition, reachability) that isolates C's
~177 surviving internals from the ~52k non-C repeated values.

## 4. Key finding: the degree-2 midpoint replica

The sandwich's mid-circuit replication of C survives the full pipeline **not verbatim and
not affinely, but up to degree 2**. At each cut, counting the 64 output functions that are
GF(2) degree-≤2 polynomials in the live wires (over 36,864 samples, so the ~33k degree-2
monomials stay below the sample count; false-positive ≈ 2^-4000):

| circuit | peak deg-2 bits | depth | profile 0.40 / 0.45 / 0.48 / 0.50 |
|---|---|---|---|
| plateau (356k) | 50 of 64 | 0.45 | 13 / 50 / 30 / 44 |
| F (666k) | 36 of 64 | 0.48 | 3 / 20 / 36 / 19 |
| short-279k | 40 of 64 | 0.40 | 40 / 37 / 26 / 29 |
| remix (1.38M) | 20 of 64 | 0.48 | 2 / 12 / 20 / 12 |
| remix2 (1.11M) | 16 of 64 | 0.50 | 3 / 8 / 8 / 16 |

Properties: **sparse** (plateau depth 0.45: median 10 monomials per relation, max 50;
four output bits equal a single interior wire verbatim); **nested and growing** (by
minimal replica on the plateau: 4 direct, 5 affine, 50 degree-2, 14 none — the
affine→quadratic jump is 45 new bits); **complementary across circuits** (plateau and F
overlap on only 29; two plateau cuts cover 59/64, adding F's cut reaches **62/64**). So
almost every output bit is a compact degree-2 function of interior wires at some midpoint
cut.

## 5. Conclusions

1. **No preimage was found.** Across three equivalent circuit sizes and five sound SAT
   formulations plus the structural/vocabulary and C-blind reconstruction routes, nothing
   produced `x` with `C(x) = 0^64`.
2. **The structural leak is real and nearly complete.** C's mid-circuit computation
   survives at degree 2 — 62 of 64 output bits are sparse degree-2 functions of interior
   wires across two circuits. This is a genuine, quantified representational weakness that
   the verbatim and affine scans missed.
3. **It does not translate into a feasible attack.** Two walls, both the same in essence:
   *distinguishing* (C-blindly, C's ~177 surviving internals are indistinguishable from
   ~52k non-C repeated values, so the reconstruction latches onto noise — 1/60 genuine),
   and *solving* (inverting a 100k–180k-gate reversible circuit is intractable for
   CryptoMiniSat even with the degree-2 relations pinning the input and the uncompute half
   removed; shrinking 666k→279k did not cross the threshold).
4. **Interpretation.** The degree-2 replica is a property of the *representation* —
   resynthesis writes C's output as low-degree functions of its own intermediates — not a
   shortcut to `x`. Exploiting it still requires either a C-blind way to pick C's values
   out of the surrounding D/stage-2/mixing noise, or an engine that can solve the
   resulting system. Standard SAT is not enough; a dedicated reversible-circuit inverter
   or an algebraic (Gröbner-style) solve over the sparse quadratic system would be the
   next thing to try.

## 6. Resolution

The six solvers (two baselines, two linked, two hint, two prefix) ran 2–3 days with no
result and were stopped on 2026-10-29 to return the shared machine to the active
campaigns, the attack being concluded with a clear negative. Other projects' processes
on the box were left untouched.

## 7. Tools and artifacts

Server `~/threemask_cmp/py6/`; results under `~/threemask_cmp/run/preimage/`:
`sat_invert.py` (direct SAT); `vocab.py`, `blind_walk3.py`, `vocab_diag*.py` (vocabulary
walks); `payload_cut.py`, `replication_scan.py`, `affine_replica_scan.py`,
`degree2_replica_scan.py` (replication scans); `combo_extract.py`, `classify_replicas.py`
(sparse degree-2 combinations); `combined_link_sat.py`, `plateau_hint_sat.py`,
`plateau_hint_sat2.py`, `prefix_quad_sat.py` (exploits); `c_reconstruct_blind.py` (C-blind
reconstruction). The full chronological log is the "Preimage campaign" section of
`docs/notes/stage2_stage6_campaign_20261021.md`; the earlier compiled summary is
`docs/notes/preimage_summary_20261026.{md,tex,pdf}`.
