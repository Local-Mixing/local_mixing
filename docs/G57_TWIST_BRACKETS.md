# G57 Twist Brackets: Adaptive All-g57 Bracket Insertion for `fmix`,

*2026-07-30*

> Markdown rendering of `docs/G57_TWIST_BRACKETS.tex`. The PDF is authoritative for figures, diagrams and display math.

## Abstract
The `fmix` conjugation twist used to bracket its window with two 3-CNOT
swap packets — six `comp=0` single-control gates per twist, a distinctive
syntactic fingerprint that the frozen store cannot digest. This note describes
the replacement (`–twist-g57`, commits `4e9dbf40` and
`83ae0d11`): each bracket is spelled as a shortest *all-g57 word*
solved online per seam, sited adaptively so it absorbs neighborhood gates —
the hidden-SAMF mechanism of `ssg`, made XGate-native. We describe the
word engine, the two-generation placer, and the measurement campaign:
A/B/C-schedule mixing runs at two twist rates against no-twist and legacy-twist
baselines, a placement factorial, an ancestry-instrument audit, and a
functional-legibility audit (degree-1/degree-2 affine plates and the stress
battery). Headline: at rate 0.002 the new brackets give 40% smaller
circuits and 5–7× the ancestry transport of the legacy packets; and
across *all* twist variants and rates, no functional instrument in the
battery detects either harm or benefit from twisting at these doses — the
one measurable twist effect is syntactic (write-census smearing).

## The bracket problem

A twist rewrites W ↦ P (P W P) P W: it conjugates a window's
interior in place and pays for two bracket packets P. With
P = swap(a,b) realized as 3 CNOTs per side, every twist deposits six
`comp=0` single-control gates. Three problems: (i) the packets are a
syntactic fingerprint (nothing else in the mixed bulk looks like them);
(ii) the frozen store's match rate collapses on CNOT-bearing windows (the
m=6 cliff), so the packets are indigestible and accumulate; (iii) they are
pure cost — placed uniformly at random, nothing ever cancels.

The fix follows the `ssg` hidden-SAMF idea: spell the bracket in the bulk
material's own alphabet (g57 gates, `comp=1` width-2), and *site* it
where the neighborhood pays part of the bill.

## The swap-word engine

The bracket seam only needs the shortest all-g57 word for a
*permutation*: R ctx· S at the opening seam,
S· ctx at the closing seam, where ctx is a few real
neighborhood gates and S = swap(a,b). This is answerable online,
with no precomputed replacement database: on 4 abstract wires a 16-state
permutation packs into one `u64`; the radius-4 BFS ball over the 24 g57
gates (165,443 permutations) builds once in 10 ms, and a
meet-in-the-middle scan answers any target up to word length 7 in
200–350 μs. The tables are target-independent: one engine serves
every context shape, any consumption depth, 3-wire supports, non-g57
neighbors, and (later) the negation twist variants. An earlier design —
a SQLite enumeration of 218k syntactic contexts requiring offline synthesis
— was retired in favor of this after its ≤ k+3 replacement contract
proved unachievable at k=2 in its own scope (every such context sits at
distance 6).

Ground facts (exhaustive; Appendix A for the full tables):
dist(S) = 6 over the g57 alphabet on 3 *and* 4 wires —
the fourth wire buys nothing; there are exactly **64** minimal words
(32 per choice of helper wire, none genuinely 4-wire). Consuming k
neighborhood gates nets 6-2k when they cancel: k=1 always possible in
the abstract (8 of 24 gates are attachable for fixed S), k=2 net +2,
k=3 net 0 (48% of same-support triples), and with *non-g57*
context (worth several g57s each) a seam can net *negative* — a twist
that shrinks the circuit.

## The placer

**v1 (anchor-first).** Candidate wire pairs (a,b) are seeded from the
boundary gates' own pins (a uniformly random b almost never lands inside
the 4-wire seam support), plus one uniform pair to keep fresh-wire routing on
the menu. Both seams are solved for every candidate; cheapest total net wins.
Each seam may consume up to 3 context gates of any shape. Every splice is
verified against the reference 3-CNOT packet under local verification.

**v1's failure mode.** The per-seam net histogram showed almost exactly
*one* bare (+6) seam per twist: the closing bracket inherits (a,b)
from the opening seam, and its fixed neighbor almost never pins both wires.

**v2 (slide + joint acceptance).** Two composable fixes, each with an
env kill-switch for A/Bs (`TWIST_G57_NO_SLIDE`,
`TWIST_G57_NO_RETRY`; both default on):

- *Slide*: a seam that stays bare walks its bracket outward (up to
512 gates, extending the conjugated window over what it passes — window
length was a random draw, and a relabel is far cheaper than the word the
slide saves) to the next g57 pinning both twist wires.
- *Joint acceptance*: both ends are solved before committing; a
window whose best plan nets worse than +8 in total is redrawn (up to 4
tries), so one side stays bare only when its partner's match pays for it
(net ≤ +2) or every redraw failed.

**Always-advance.** Every inserted gate takes the ballistic
birth-advance unconditionally (the `–db-advance` treatment, aimed
outward per bracket) — bracket material rides away from the seam instead of
sitting on the window edge. This is independent of the `–db-advance`
flag itself, which still governs DB splice products only.

**Ancestry.** Consumed context unions its litters' ancestor sets into
the replacement litter, exactly as a DB splice would (v1 dropped them —
an instrument bug worth +1.5–10% of `anc`, isolated by rerunning
v1-placement arms with only the recording changed).

## Mixing experiments

All runs: the same 20,000-gate, 64-wire sample (the first 3/7 of a Gray-fold
gadgetized sliced sandwich, n=16), seed 101, 2M moves, ancestry recording
on, C schedule (`–p-mix 0.1`) unless noted; "legacy" is the shipped
swap-family 3-CNOT packet twist, rates are `–p-twist`.

| C-schedule arm | size | anc | span | polf |
|---|---|---|---|---|
| no twist | 48,493 | 7,941 | 8,626 | 0.000 |
| legacy 0.002 | 133,665 | 425 | 1,111 | 0.422 |
| legacy 0.01 | 262,115 | 62 | 369 | 0.487 |
| g57 v1 0.002 | 86,696 | 2,604 | 3,374 | 0.000 |
| g57 v1 0.01 | 216,281 | 637 | 1,341 | 0.000 |
| g57 v2 0.002 | **78,233** | **3,038** | **3,895** | 0.000 |
| g57 v2 0.01 | **168,192** | **921** | **1,625** | 0.000 |

*Final size and ancestry measures, C arm. The A and B schedules give
the same ordering. `polf`=0 because the g57 path is pure swap — the
negation arms (the polarity-mixing capability) still use the legacy packet;
their word tables are the open extension.*

At matched *effective work* (cumulative moves per gate), the ancestry cost
of twisting is: legacy -89% vs. no-twist, g57 -45% — the new
brackets keep 5–6× more transport at equal dose, *and* end
smaller, despite emitting more gates per twist (8–9 net vs. 6): the
brackets are store-digestible, so COMP chews them back down instead of
accumulating CNOT bulk.

**Placement factorial** (C arm, rate 0.01): mean net gates per seam
4.51 (v1) → 4.38 (slide only) → 3.93 (retry only) $→
{3.72}(both); bare seams50%→35%; size216{k}→
168{k}; anc637→921$. The joint-acceptance retry is the larger
lever; the slide composes. Cost: 2.6× the MITM solves and 1.6
extra window draws per twist — microseconds per move at production rates.

![A/B/C × twist rate: legacy packets (dashed/dotted) vs. g57 v2
brackets (dash-dot styles), no-twist solid. Panels: anc, span, size, dmin vs.\
effective work.](../reports/ancestry_20260728/abc_twistrate_g57v2_20260730.png)

*A/B/C × twist rate: legacy packets (dashed/dotted) vs. g57 v2
brackets (dash-dot styles), no-twist solid. Panels: anc, span, size, dmin vs.\
effective work.*

## Is the ancestry drop a mixing loss? The instrument audit

Twists still cost `anc`/`span` relative to no-twist even under v2.
Two audits asked whether that reading is real.

**(1) Recording audit.** A conditional census over all run states
(parsing litters and ancestor sets) shows the reported `anc` was
*already* conditional on non-empty sets, and the empty-ancestry gate
fraction is moderate (3–21% for g57, up to 46% for legacy at 0.01) — so
the drop is not mean-deflation by ancestry-less brackets; conditioning
changes nothing about the ordering. The union-recording fix (above) moves
`anc` by only +1.5–10%.

**(2) Functional audit.** Affine-reconstruction plates
(`hmap_affine`) of the *input circuit's* prefix progress against
each mixed circuit, read by the ridge measure:

- Degree 1: identical for all seven arms (ρ = 1.000, perm
z ≈ 7.1, depth 0.22–0.26). *Caveat discovered in the
process:* every current twist type is an affine conjugation and the degree-1
measure is affine-invariant — it is **blind to twist effects by
construction**, in both directions.
- Degree 2 (2,081 regressors — not affine-blind): the diagonal is at
ceiling in every arm (ρ = 1.000, z ≈ 4.5); depth 0.305
(no twist) vs. 0.26–0.30 (twisted) — at most a marginal hint of
benefit, within noise.
- Stress battery (bias-sensitive statistical probes; single-wire /
XOR-pair predictors on state bits, parities, and trajectory bits): no
benefit — prominence 0.26–0.31 above floor in *every* arm
including no-twist, with progress monotonicity ρ = 0.85–0.99,
z = 7–9 everywhere. The progress clock is readable by near-trivial
predictors in all arms; no twist dose or variant dents it.
- The one measurable twist effect is *syntactic*: the per-wire
write-count bimodality gap falls from 0.283 (no twist) to 0.09–0.22
under twists.

**Verdict.** At these doses, twisting costs real ancestry transport and
size while buying nothing measurable on functional channels; its measurable
value is syntactic-census smearing. The ancestry gap *between twist
variants* is therefore not a mixing-quality gap — but the case for twisting
at all now rests on adversary classes not yet in the battery. Practical
recommendation: keep `–p-twist` at 0.002 or below, use
`–twist-g57` so the dose costs 5–6× less, and treat any
proposal to raise twist rates as requiring a demonstrated adversary it
defeats.

![Degree-2 plates, six of the seven C-family arms: the input-progress
diagonal survives identically everywhere.](../reports/ancestry_20260728/ridge2_g57_20260730.png)

*Degree-2 plates, six of the seven C-family arms: the input-progress
diagonal survives identically everywhere.*

## Usage

- `–p-twist R –twist-g57` — the full v2 path: all-g57 words,
anchor-first seams, context consumption, slide, joint acceptance,
unconditional birth-advance, ancestry union. Pure swap only
(`–twist-neg-p` is ignored on this path).
- `TWIST_G57_NO_SLIDE=1` / `TWIST_G57_NO_RETRY=1` —
env kill-switches for the two v2 placement features (A/B use).
- Omit `–twist-g57` for the legacy 3-CNOT swap-family packets
(`–twist-neg-p` active there).
- `–db-advance` — separate flag; birth-advance for *DB
splice* products. The g57 bracket path advances its insertions regardless.
- Report line: `twist-g57: consumed= emitted= net/seam[hist]
solves= avg_us= slides= retries=`. Note: this path emits
`ORIGIN_SYNTH` `comp=1` material, so `comp=`/`shaped=`
read population form under it (same caveat as `p_db>0`).
- Tuning constants (compile-time, `mix.rs`):
`TG_SLIDE_CAP`=512, `TG_SLIDE_TRIES`=3, `TG_RETRIES`=4,
`TG_ACCEPT_NET`=+8.

## The replacement table

The complete machine-generated table lives in
`docs/G57_TWIST_REPLACEMENTS.txt`: **T1**, all 64 minimal
(length-6) all-g57 words realizing S=swap(a,b) (32 per helper
wire; gate [t,n,p] means t _= n ∨ ¬ p);
**T2**, the 12 ordered pair-consumption identities on {a,b,c}
(h_1 h_2 S R with |R|=4, net +2); **T3**, the 8
single-gate attachment identities (h S R with |R|=5, net +4).
The engine does not read this table — it re-derives everything from the
BFS ball at startup — but the table is the exhaustive human-checkable
inventory of what the seams can do at k≤2 on a 3-wire support.
Representative entries:

| kind | identity |
|---|---|
| bare word | S [a,b,c] [a,c,b] [b,a,c] [b,c,a] [a,b,c] [a,c,b] |
| pair, net +2 | [a,b,c] [a,c,b] S [b,a,c] [b,c,a] [a,b,c] [a,c,b] |
| attach, net +4 | [a,b,c] S [a,c,b] [b,a,c] [b,c,a] [a,b,c] [a,c,b] |
