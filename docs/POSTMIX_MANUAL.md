# `fmix` & `fcompress` — The Post-Mix Toolkit

Reference for the `postmix` binaries: `fmix` (the randomized size-thermostatted
mixing chain), `fcompress` (the deterministic final compression pass), and
their companion analyzer `fmix_stats`. Every CLI flag, environment variable,
and log field, with defaults and practical guidance.

```
fmix      --input A.txt [--input-format g57] --target-size 3000000 \
          --moves 50000000 --output mixed.txt --origins-out mixed.origins.txt
fcompress --input mixed.txt --output mixed.fc.txt [--live-wires upper-half]
fmix_stats --input mixed.txt [--origins mixed.origins.txt]
```

All three share the **XGate calculus** and the **mpmct1 file format** (§6).

---

## 1. The XGate calculus

An `XGate` is `{target, comp, ctrls}` with at most `K` control literals
`(wire, polarity)`, sorted by wire. Semantics:

```
target ^= comp XOR AND_i ( ctrls[i].wire == ctrls[i].polarity )
```

i.e. a mixed-polarity multi-controlled Toffoli, optionally complemented. Every
XGate is an involution. A g57 gate `a ^= b OR !c` is the special case
`comp=1, ctrls={(b,neg),(c,pos)}` — so g57 circuits embed directly, and the
`comp=1` population is a **fossil count** of surviving original g57 material
(see the comp guard below).

**Commutation (`XGate::collides`).** Two gates collide iff one writes a wire
the other reads or writes — with the **separation exemption**: two pure
conjunctions (`comp=0`) that share a control wire with *opposite* polarities
always commute (their firing subcubes are disjoint on a wire neither writes).
Consequences: for conjunctions, more controls means *less* blocking;
`comp=1` gates (g57s) fire on the complement of a subcube, never separate,
and are maximal roadblocks — the reason pure-g57 circuits mix poorly.

**Pairwise merge catalogue (`merge_result`).** For same-target gates g,h the
XOR `f_g ⊕ f_h` is again a (possibly complemented) monomial in exactly four
cases:

| Merge | Condition | Result |
|---|---|---|
| `Cancel` | identical gates | both vanish |
| `XFuse` | same controls, opposite `comp` | a NOT gate on the target |
| `DropLit` | same wires, exactly one polarity flipped, equal `comp` | drop that wire (`xR ⊕ !xR = R`) |
| `Subsume` | wire sets differ by one literal, shared literals equal, equal `comp` | flip that literal (`R ⊕ lR = !lR`) |

**The comp guard:** a fusion whose result would be complemented — which is
precisely the rejoin of a g57's two presplit pieces — is *refused*. This makes
the fossil count monotone non-increasing **through the merge catalogue**: a
merge never creates a `comp=1` gate, so crossing-driven erosion is irreversible
by that route.

⚠️ It does **not** make the reported `comp=` field monotone, and in the shipped
configuration it is not. DB splices insert `comp=1` g57 material with no comp
filter at the insertion site, so with `p_db > 0` the count grows with the
circuit. The four `"fossil count increased"` tests pass only because
`MixParams::default()` has `p_db: 0.0`; they do not cover production.

---

## 2. `fmix` — what it does

A random walk over functionally-equivalent circuits. The objective is **not
size** — it is to churn the circuit as far from its original description as
possible while a thermostat holds the gate count near `--target-size`:

```
p(contraction move) = sigmoid((size − target) / temp),  clamped to [0.02, 0.98]
```

Run it directly on a g57 circuit (`--input-format g57`) with a target well
above the input size: the growth phase (thermostat pinned at 98% expansion) is
itself the strongest mixing — it both transports material and erodes fossils.

**The directional walk.** Every gate carries a persistent **direction**
(left/right), drawn uniformly for each input fossil at load. Transport is
directional rather than diffusive: a crossing floats its gate along the gate's
*own* direction; every fragment born in a collision inherits the shot gate's
direction with probability `--dir-p` (else the opposite), and then advances
`floor(dir_q · slack)` gates in its own direction at birth (`--dir-q`), where
`slack` is its free run to the first collider. This directional birth-advance
**replaces the old uniform scatter** — material moves ballistically along
aligned directions instead of diffusing symmetrically. A crossing that is
declined (width-damped, capped, or hitting the boundary) does not leave its
gate parked at the collision; it **retreats** `floor((1−dir_q) · way)` of the
distance it floated in.

**Expansion moves** (chosen by relative weight; each produced piece with `c`
controls is width-damped — allowed outright if `c ≤ split_damp`, else with
probability `B^-(c−split_damp)` where `B = --split-base`):

| Move | Weight flag | What it does |
|---|---|---|
| crossing | `--w-cross` (0.70) | float a gate along **its own direction** to its collision point and split it past the collider by one R-rule (R1/R2/R3, a Hurwitz-style conjugation step); a shot g57 pre-splits into its two conjunction pieces. Fragments inherit direction (`dir_p`) and birth-advance (`dir_q`); a failed shot retreats. Recorded in the undo journal. |
| insert | `--w-insert` (0.05) | insert an adjacent identity pair of a **fresh random conjunction** — width uniform in `[1, K]`, random distinct wires, random polarities. The two copies get **opposite** directions and each is immediately **shot once** (so one insert embeds two crossings), separating the pair directionally. |
| unsubsume | `--w-unsub` (0.10) | inverse of `Subsume`: `!lR → R, lR`. |
| negation twist | `--w-twist-neg` (0) | conjugate a window by a wire negation `N` (`+2` gates): every interior gate reading that wire has its literal polarity flipped, brackets `N…N` restore the function. A state-frame rotation — see the twists below. |
| swap twist | `--w-twist-swap` (0) | conjugate a window by a wire swap realized as 3 CNOTs/side (`+6` gates): routes the window's material through a fresh physical wire. |
| transvection twist | `--w-twist-cnot` (0) | conjugate a window by `x_a ^= x_b` (one CNOT/side, `+2` gates): the affine, **non-isometric** rung — the one that breaks Hamming-distance self-gauges neg/swap preserve. Interior `a`-readers case-split on `b` (count ×2, width +1, K-capped); `b` must be unwritten in the window, which caps these windows at the mid scale. |
| fresh split | `--w-fresh` (0) | case split `R → xR, !xR` on a uniformly random uninvolved wire. **Suspended by default** (its wire-coupling entropy is covered by the twists' interior case-splitting); set `> 0` to re-enable. |

**Conjugation twists** (`--w-twist-*`, off by default). A twist picks a window
`W` and an involution `P` and rewrites `P·(P W P)·P ≡ W`: it conjugates every
interior gate in place and brackets the window with one `P`-packet per side.
The function and everything outside `W` are unchanged, but every interior
*state* becomes its image under `P` — so a twist is the one move that rotates
the intermediate-state trajectory, collapsing the prefix-progress diagonal that
no support-local move can touch. Window lengths are log-uniform over
`[--twist-min-len, |circuit|]`, and the virtual window start is drawn
symmetrically (a window may hang off either end and truncate against it) so
head and tail accumulate twist coverage equally. Keep weights small (`~1e-3`):
one twist rewrites `O(window)` gates. Neg/swap are Hamming isometries (invisible
to avalanche/difference gauges); the transvection twist is the affine rung that
is not.

**g57-word brackets** (`--twist-g57`, off by default; needs `--p-twist > 0`).
Replaces the twist bracket packets with **adaptive all-g57 words** — the ssg
hidden-SAMF mechanism, XGate-native. Pure swap only (`--twist-neg-p` is
ignored on this path). Each bracket seam asks a built-in BFS+MITM engine
(`swap_words.rs`: 16-state perms packed in `u64` over the 24 g57 gates on 4
abstract wires; ~10 ms one-time build) for the shortest all-g57 word
realizing `ctx · S` (opening) / `S · ctx` (closing), consuming up to 3
neighborhood gates of any shape into the bracket — non-g57 context is worth
several g57s, so a seam can even net negative (a twist that shrinks the
circuit). Placement is anchor-first (candidate wire pairs from the boundary
gates' own pins, plus one uniform pair for fresh-wire routing), a bare seam
may **slide** its bracket outward (≤ 512 gates, extending the conjugated
window) to a g57 pinning both twist wires, and the two ends are accepted
**jointly** — a window whose best plan nets worse than +8 total is redrawn
(≤ 4 tries). Every inserted gate takes the ballistic birth-advance
**unconditionally** (the `--db-advance` treatment, aimed outward; independent
of that flag, which still governs DB splice products only), and consumed
context unions its litters' ancestor sets into the replacement litter,
DB-splice style. Every splice verifies against the reference 3-CNOT packet
under local verify. Env kill-switches for A/Bs: `TWIST_G57_NO_SLIDE=1`,
`TWIST_G57_NO_RETRY=1` (both features default ON). Report line:
`twist-g57: consumed= emitted= net/seam[hist] solves= avg_us= slides=
retries=`. Measured (20k n=64 sample, C schedule, rate 0.002): ~40% smaller
circuits and 5–7× the ancestry transport vs the legacy 3-CNOT swap brackets;
this path emits `ORIGIN_SYNTH` comp=1 material, so `comp=`/`shaped=` read
population form under it (same caveat as `p_db > 0`). Spec and measurements:
`docs/FMIX_MENU.md` §3.3.1, `docs/G57_TWIST_BRACKETS.pdf`.

**Contraction moves.** With probability `--undo-frac` first try a **journal
undo**: exactly reverse a recorded crossing while all its pieces are alive
(arena-stamp-validated). Crossings are the one expansion the pairwise
catalogue cannot invert (R-ladder rungs are pairwise unmergeable), so without
the journal size creeps up at the crossing rate regardless of the thermostat —
and dead journal entries are permanently unmergeable material, so **size creep
above target tracks irreversible mixing**. (Undo only ever reverses *sterile*
crossings: the stamp check kills an entry the moment any piece feeds a later
move, so nothing that mattered is taken back — but the raw `r1/r2/r3` crossing
counters overcount net work by roughly 2×, since about half of all crossings
are eventually reversed. Read net crossings as `r1+r2+r3 − undos`.) Otherwise a
**catalogue merge**: pick a random gate, find the nearest reachable partner
through the global (target, wire-set) hash index within `--merge-reach`, float
the pair adjacent (incremental wall check), apply the catalogue.

**Tabu.** A split event may not be undone or sibling-merged until
`--tabu-moves` moves have passed — freshly split pairs cannot instantly
rejoin.

**Provenance.** Every gate carries `(origin, event)`: the input-gate index its
material descends from, and the split event that created it. Splits pass the
parent's origin to both pieces; merges keep a survivor's. `--origins-out`
writes the origin of each output gate (one per line, final order;
`4294967295` = synthetic material with no input ancestor).

**Final step.** Every gate floats to a uniform random position in its
two-sided collision box (skip with `--skip-final-float`) — the one place a
uniform (non-directional) float still runs, to decorrelate final positions —
then the output is verified and written.

**Verification.** Three layers, all on by default: (1) every move is
exhaustively verified on its support before commit (disable at your own risk
with `--no-local-verify`); (2) a sampled 64-lane global equality check against
the input every `--verify-every` moves; (3) dumps, the final float, and the
final write are each verified. A failure panics immediately.

### 2.1 `fmix` CLI reference

| Flag | Default | Effect |
|---|---|---|
| `--input` | (required) | input circuit file |
| `--input-format` | `mpmct1` | `mpmct1` or `g57` (base-83 CircuitSeq) |
| `--output` | none | output file (mpmct1). Omit for a churn-only experiment |
| `--target-size` | input size | thermostat target gate count |
| `--temp` | max(target/100, 64) | thermostat softness in gates |
| `--moves` | 1,000,000 | total move attempts |
| `--k-max` | 12 | max controls per gate (**K**). Wider = bigger float boxes (separation exemption) but wider gates and larger files |
| `--split-damp` | 2 | width damping offset **D** (fsplit convention): a produced piece with `c` controls passes w.p. `min(B^-(c−D),1)`. Dominates growth speed far more than K |
| `--split-base` | 2.0 | damper base **B** (was hardcoded 2). Larger = steeper width penalty (narrower gates, slower growth); `B ≤ 1` disables damping |
| `--dir-p` | 0.75 | probability a collision fragment inherits the shot gate's direction (else the opposite) |
| `--dir-q` | 0.85 | directional-transport fraction: a fresh piece advances `floor(q·slack)` in its direction at birth; a failed shot retreats `floor((1−q)·way)` |
| `--merge-reach` | 4096 | max distance (gates) to a merge partner; bounds the locating scan and wall check |
| `--journal-len` | 262,144 | undo journal capacity |
| `--undo-frac` | 0.5 | fraction of contraction moves that try a journal undo first (`0` skips journal recording entirely — pure catalogue contraction, leaks unmergeable crossing ladders) |
| `--tabu-moves` | 2000 | refractory age (moves) before an event may be undone/sibling-merged |
| `--w-cross/-unsub/-insert` | .70/.10/.05 | expansion move weights |
| `--w-fresh` | 0 | fresh-wire split weight — **suspended by default** (covered by the twists); set `>0` to re-enable |
| `--w-twist-neg/-swap/-cnot` | 0/0/0 | conjugation/transvection twist weights (state-frame mixing). `0` = trajectory-identical to the pre-twist chain; keep small (`~1e-3`) when on |
| `--twist-min-len` | 64 | minimum twist window length (max is the current circuit size); set near the local-churn/leeway scale (`~256` in production) |
| `--verify-every` | 10,000 | sampled global equality check cadence (moves) |
| `--report-every` | 50,000 | report line + stop/dump flag check cadence (moves) |
| `--no-local-verify` | off | disable the per-move exhaustive local check |
| `--skip-final-float` | off | skip the final uniform float pass |
| `--origins-out` | none | write the per-gate origin sidecar |
| `--gen-target` | 0 | generation targeting (needs the DB move on): drive every ctrl-cap-eligible gate through at least this many DB re-encodings. `0` = off, trajectory-identical to the untargeted chain |
| `--gen-bias` | 0.9 | probability a DB seed is drawn from the below-target (laggard) gates instead of uniformly |
| `--gen-rescan` | 10,000 | laggard-list rebuild cadence in moves (O(size) scan; stale entries are pruned at draw time) |
| `--p-db-ingest` | 0 | ingest-then-pay CHEAP channel: probability a round is a Compressing-mode DB attempt (non-growing replacements only — zero growth risk, safe to run hot) seeded on a cheap-tier laggard |
| `--p-db-hard` | 0 | ingest-then-pay PAID channel: probability a round is a MinGrow-mode attempt (uniform among the SHORTEST equivalents, growing allowed) seeded on a hard-tier gate. The only channel that spends growth on the generation goal; cost ledgered in `paid=` |
| `--gen-miss-budget` | 6 | seed misses before a laggard graduates cheap → hard tier |
| `--gen-giveup` | 0 | seed misses before a gate is written off as unreachable (excluded from targeting, reported `u=`; still counted by the all-gates dose criterion). 0 = never |
| `--gen-split-inherit` | off | split children INHERIT the parent generation unchanged (only DB replacements raise generations — isolates DB re-encoding depth). Default = ratchet semantics: split children get parent + 1 |
| `--gen-median-low` | off | DB stamp uses the LOWER median of the outgoing window (rounded down on even windows — on 2-gate windows this is the min). Default = upper median (rounded up) |
| `--gen-stop-frac` | −1 | dose stop: end the run at the first report point where the eligible laggard fraction is ≤ this and `--twist-cov-stop` is met. Negative = off; the move budget becomes a ceiling when on |
| `--twist-cov-stop` | 0 | twist-coverage requirement for the dose stop: cumulative twisted span / current size (saturation target ~600). `0` = no requirement |
| `--gens-out` | none | write the per-gate generation sidecar (`4294967295` = born-random material) |
| `--seed` | 0 | chain RNG seed. Metrics are sampled from a **separate** RNG (`seed ^ 0x5EED517A75`), so report cadence never perturbs the trajectory |

Production convention: `--report-every 1000000 --verify-every 100000`.

### 2.1.1 The curated store (`--curated`, bounded DB contract)

**Shipped DB defaults (2026-08-03).** The DB move now defaults to the
curated-first cascade in BOTH modes:

| knob | MIX-DB | COMP-DB | off switch |
|---|---|---|---|
| size-reduction cascade (`--db-prefixes`) | on | on | `--no-db-prefixes` |
| descent start | `--s-db 9` | `--s-db-comp 12` | — |
| store routing | curated over the whole cascade first, then regular (`--curated --curated-exhaust`) | same (`--curated-in-comp`) | `--no-curated`, `--no-curated-exhaust`, `--no-curated-in-comp` |
| sampler | contiguous 60% / convex 40% (`--p-convex 0.4`) | convex 90% / contiguous 10% (`--p-convex-comp 0.9`) | — |

The COMP sampler shipped inverted (0.1) from the 2026-08-03 defaults commit
until 2026-08-04. The 32-arm COMP factorial measured the two geometries
head-to-head on `pre2_100k` at 500k moves: convex removes **16×** more gates
(8,748 vs 552 mean), transports **7×** more ancestry (est_anc 43.9 vs 6.0),
and does it in **1/31** the wall time (156s vs 4,805s). Contiguous's only
edge, per-splice selection entropy (0.766 vs 0.355 bits), does not survive
aggregation — convex delivers more *total* entropy in all 8 matched pairs
because it lands 2.4× more successful splices. The cost asymmetry is window
width, not the span cap: contiguous skips only 0.5–3.6% of attempts at the
cap but pays ~30× per canonicalization on the wide windows that pass it.
Runs that took the COMP default before 2026-08-04 spent ~90% of their COMP
DB budget on the losing geometry.

If `FROZEN_CURATED_DIR` is unset, the curated default degrades to
regular-only **with a startup warning**; passing `--curated` explicitly
makes the missing store a hard error instead (measurement runs should do
that). A **resume** builds its params from the command line, so a resumed
pre-2026-08-03 run picks up these defaults unless the old flags are
repeated explicitly.

### 2.1.1a How a DB knob gets its value

Every DB knob exists at up to three levels of specificity — base
(`--s-db`), mode (`--s-db-comp`), and mode+geometry (`--s-db-comp-ctg`) —
and is supplied by up to three sources. The rule, highest first:

1. **explicit CLI**, most specific level first
2. **preset** (`--gss`, then `--phase-a`), most specific level first
3. **shipped default**, most specific level first

with one exception that is the whole point of the scheme:

> **A shipped mode-level default is withheld when you set the base knob
> explicitly.** `--db-mode comp --s-db 20` gives COMP 20, not 12. A shipped
> default is not a statement about *your* run, so it must not outrank one.
>
> Presets are deliberately *not* withheld this way: `--gss --s-db 15` moves
> MIX to 15 and leaves COMP at the profile's 12/6. A named profile is a
> coherent unit whose mode-level choices are intentional — to move COMP as
> well, say `--s-db-comp`.

Two consequences worth knowing:

*`0` and `false` mean themselves.* Overrides are `Option`, so
`--p-mingen-comp 0` is a real request. Until 2026-08-05 they were
sentinel-encoded (`0` for `usize`, `< 0` for `f64`) and a legitimate zero —
exactly what GSS wants — was indistinguishable from "unset".

*A knob that cannot fire is an error, not a no-op.* Passing a COMP override
with `--db-mode mix` and no `--p-mix` overlay exits with a message naming
the flags, because COMP rounds never happen. Silently-inert flags are how
the shadowing bug survived two days.

The resolution lives in exactly one function, `MixParams::db_knobs`, which
the mixer and the startup banner both call — so the `DB effective per mode`
line is by construction what the run will do. The previous banner re-derived
the rules itself and could drift.

> **Historical note.** Before 2026-08-05, `s_db_comp` (12) and
> `p_convex_comp` (0.9) shipped as `default_value_t` and therefore always
> outranked an explicit `--s-db` / `--p-convex` in COMP rounds. Runs launched
> with `--db-mode comp --s-db 20 --p-convex 1.0` silently executed at
> `s_db 12, p_convex 0.9`. If you are comparing against measurements taken
> before that date, check the `DB effective per mode` line (added the same
> day) rather than the command line.

### 2.1.2 Per-geometry window length and per-mode descent

Two axes were split on 2026-08-05 so the GSS profile below could be
expressed at all.

**Geometry is now drawn once per round, before the length.** It used to be
drawn inside `sample_window`, after the length was already fixed, which made
a geometry-conditional length impossible and let the
best-of-`litter_samples` selection compare windows drawn under *different*
geometries. Now `db_attempt_inner` flips the `p_convex` coin first, resolves
the length from it, and passes the geometry down. `DbSample::Mixed` and
`DbSample::parse` were removed in the same change — nothing had constructed
or called them since the sampler knobs were split per mode.

| flag | meaning | fall-through |
|---|---|---|
| `--s-db-ctg N` | MIX window length when the round drew CONTIGUOUS | `0` = use `--s-db` |
| `--s-db-comp-ctg N` | COMP window length when the round drew CONTIGUOUS | `0` = use `--s-db-comp` |
| `--db-prefixes-mix <bool>` | prefix descent in MIX rounds only | unset = use `--db-prefixes` |
| `--db-prefixes-comp <bool>` | prefix descent in COMP rounds only | unset = use `--db-prefixes` |

Resolution is most-specific-first: mode+geometry → mode → base. A `0` (or an
unset `Option`) falls through; it never clamps the window to nothing.

Per-geometry length exists because the two samplers have very different cost
curves. Measured on `pre2_100k`, MIX mode, matched final size, the
contiguous/convex wall-clock ratio is 1.1× at s_db 3, 3.6× at 5, 12.6× at 7,
29.9× at 9, **47.8× at 12**, then back to 14.7× at 20 — the fall at the top
is the span cap rejecting ~70% of wide contiguous windows before
canonicalization. So a contiguous probe is cheap only while it stays narrow.

Per-mode descent exists because the `--p-mix` overlay runs both modes in one
process and they want opposite settings: COMP descends (worth ~600× on
ancestry transport in the 32-arm factorial), MIX does not (its expansion band
is only lengths 1..~5, so descent there re-probes lengths that cannot expand).

### 2.1.3 The GSS profile (`--gss`)

The DB settings for running fmix on a **gadgetized sliced sandwich** input.
Explicit flags always win, and it composes with `--phase-a` (which supplies
the twist / db-advance / pay-random block).

| | COMP-DB | MIX-DB |
|---|---|---|
| curated | on (`--curated --curated-in-comp`) | on |
| descent | **on** | **off** |
| `p_mingen` | 0 | 0.5 |
| geometry | convex 95% / contiguous 5% | convex 50% / contiguous 50% |
| `s_db`, convex | 12 | 6 |
| `s_db`, contiguous | 6 | 6 |

**`--gss` deliberately does not set `--p-mix`.** The MIX/COMP balance is the
layer-2 controller's lever; this profile is meant to be the right per-mode
setting at *every* p_mix.

Two things to keep in mind when reading it:

*The two `s_db` numbers are not on the same scale.* COMP's 12 is a descent
**start** — every round walks 12, 11, … 1. MIX's 6 is the top of a **uniform
draw** — one length per round. COMP is not "looking twice as wide"; it
touches every length ≤12 per round while MIX touches exactly one ≤6.

*The profile is g57-preserving by design, and this is intentional.* Every DB
splice re-spells a g57 word as another g57 word, so `polf` stays exactly
0.000 — measured across every DB-only run to date (MIX and COMP, convex and
contiguous, curated and not, fresh and near-minimal material), with
`comp = g57 = shaped` holding exactly. Breaking g57 form is a separate
concern from this profile's job; it needs the twist family, not the store.

With `--curated` (and `FROZEN_CURATED_DIR` set), DB **expansion** probes the
curated store FIRST, forward key only, and applies the mode's own size rule
within the curated answer — Mix: random among no-larger spellings, else
random among the minimal ones. Only a complete curated miss falls back to
the regular store (both keys, same rule). Under `--curated-exhaust` (the
default) "first" means the ENTIRE prefix descent runs curated-only before
the regular store sees any length, so curated material at any length beats
regular material at a longer one. **Compression follows the same
curated-first routing while `--curated-in-comp` is on (the default); its
size rule keeps only the spellings strictly shorter than the window. With
`--no-curated-in-comp` it reverts to regular-only, the pre-2026-08-03
contract.** The bounded curated DB (2026-07-30: ≤20 candidates/key, ≤512
decoded value bytes) requires the per-store value conventions in the
environment — a fresh process must set:

```
FROZEN_DB_DIR=…            FROZEN_CURATED_DIR=…
FROZEN_REGULAR_VALUE_CONVENTION=native
FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls
```

and its startup log must show `[frozen] curated=… opened … (filter on|off)`
and `[frozen] value conventions: regular=native,
curated=legacy-swapped-controls`. A warn-once tripwire fires if a curated
value exceeds the bounded contract (wrong DB / stale data / bad parser).
Readouts: `cur=hits/rejected` (rejected must stay 0 under the correct
convention) and the `splice sizes (curated) out->in:` histogram line
(curated-only; total minus curated = regular). Measured comparison vs the
regular-only regime: `docs/CURATED_DB_COMPARISON.pdf` — smaller circuits,
more transport, at every scale tested.

**`--db-advance` should be ON in every run unless an A/B explicitly needs it
off** (directive 2026-07-30): without it, DB splice products carry a
direction nothing ever reads; measured at the curated operating point it
adds ~+50–63% ancestry transport and widens the curated advantage. The CLI
default is still off — set it in every launch script.

### 2.1.2 LAYER 2: the phase-A preset and the size profile

**`--phase-a`** sets the phase-A default block: `--twist-g57` with
`--p-twist 0.0005`, `--db-advance`, `--mix-pay-random`, `--p-mingen 0.6` for
MIX with `--p-mingen-comp 0` for COMP. (It no longer sets `--curated` or
`--p-convex` — the 2026-08-03 shipped defaults, curated ON and `p_convex`
0.4, already cover them; see §2.1.1.) A knob you pass **explicitly on the
command line** always wins — the preset keys on whether the flag was given,
not on whether its value differs from the default, so `--phase-a
--p-twist 0` really does mean no twists (it did not, before 2026-07-31:
0 is also p_twist's default, and an intended no-twist arm silently ran at
0.0005). The startup banner prints the **resolved** values, so a run states
what it actually is.

**`--mix-pay-random`**: when a MIX window's store answer contains only
*larger* spellings, take a uniformly random one rather than a minimal one.
More growth and more spelling diversity per paid splice — and a stronger
up-lever for the controller below.

**`--profile N0,N1,N2,R1,R2`** — a three-phase, best-effort size schedule
in effective-work units (moves per gate, the same clock the analysis uses):

1. **expand** to `R1 x` the input size, by effective work `N0`;
2. **hold** near `R1 x` until `N1`;
3. **compress** toward `R2 x`, ending on arrival or at `N2` — whichever
   comes first.

`N2` may be given as an absolute mark (`N2 >= N1`) or as the compression
leg's *budget*; a value below `N1` can only mean the latter, and the run
logs the absolute end mark it derived. A completed schedule ends the run
(`profile complete`) rather than burning the remaining budget outside any
setpoint.

**How it steers.** Every `--prof-cadence-eff` of effective work the
controller reads the live counters and identifies the plant in gates per
move: `ghat` (drift with the lever at 1), `shat` (removal rate at 0), and
`dhat` — the **disturbance**, i.e. the residual between observed drift and
what the DB move accounts for. Twists live in `dhat`: the controller cannot
steer the twist rate, so it measures its effect instead and solves
`p*ghat - (1-p)*shat + dhat = v*` for the lever, plus a small integral
term on the tracking error. Guards against over-steering: a relative
deadband (`--prof-deadband`), a per-update rate limit (`--prof-dp-max`,
lifted only when more than four deadbands from the setpoint), EWMA
smoothing of the estimates (`--prof-ewma`), and a cadence measured in
effective work so control frequency scales with the circuit.

**Single size authority.** While a profile is active the controller *owns*
`target_size` (the thermostat is conscripted to the moving setpoint) and
the static size brake is inert. Passing `--target-size`, `--size-hi`,
`--size-lo` or `--p-mix` alongside `--profile` is refused outright.

**Best effort, and what that means in practice.** Not every profile is
reachable. The compression leg is the usual limit: `shat` decays as the
circuit approaches local minimality, so an aggressive `R2` may simply not
arrive within `N2`. The controller then pins the lever, logs
`profile: SATURATED`, and carries on. When the disturbance exceeds the
maximum available removal (`dhat > shat` — e.g. a high twist rate), phase 3
says so explicitly: the circuit grows with the lever at 0 no matter what,
and the fix is a lower twist rate or a relaxed `R2`.

**The twist ceiling is real and it binds early.** Measured on the 100k
512-wire slice with `5,50,20,2,1.2`: at `p_twist 0.0005` the controller
holds the setpoint to within 3%, but at `0.005` it cannot even *hold* —
the lever sits pinned at 0 and the circuit still runs 82% above target
(twist growth ≈ +0.037 gates/move against a maximum COMP removal of
≈ +0.021). So a size contract and a high twist rate are mutually
exclusive; choose the twist rate first, then ask for a profile the plant
can deliver. The same experiment at 20k reproduced the effect exactly.

**Tracking, measured.** Same slice, four `R1` values (1.5 / 2 / 2.5 / 3)
with `N = 5, 50, +20`: expansion ramps land and the hold tracks within
−3% … +3% of setpoint across the sweep, and a short-hold variant
(`5,20,+20`) tracked its compression ramp to +1%.

Report line: `[fmix] profile: phase= eff= size= S*= pmix= ghat= shat=
dhat= integ= sat=`.

⚠️ A profile is a whole-run construct: its effective-work clock is not
serialised, so `--profile` on a `--resume` restarts the schedule at eff 0
(warned at startup).

### 2.2 Environment flags: pause-free control

Checked at every report point (so responsiveness = `--report-every`):

| Variable | Effect |
|---|---|
| `FMIX_STOP_FLAG=<path>` | `touch <path>` → graceful finish: final float, verify, write, exit |
| `FMIX_DUMP_FLAG=<path>` | `touch <path>` → **verified snapshot of the current circuit, then the run continues.** Written to `<FMIX_DUMP_OUT>.mv<moves>` plus `.origins` sidecar; the flag file is removed (re-armed) — repeated touches build a move-stamped snapshot series |
| `FMIX_DUMP_OUT=<path>` | snapshot basename (default `<output>.snapshot.txt`) |

### 2.3 Reading the report line

```
[fmix] mv= size= target= comp= | merges c= x= d= s= sib= xorig= tabu= nopart= wall= far= noadj=
| undo ok= dead= tabu= miss= live= | expand r1= r2= r3= pre= fresh= unsub= ins=
  twn= tws= twc= twrel= twsplit= twspan= twskip=
| declined= blockw= dl= bnd= | floats=N/steps scat=N/steps
| disp= owin= fan0= leew= odiff= oadj= width[...]
```

| Field | Meaning |
|---|---|
| `mv, size, target` | moves attempted; current gate count; thermostat target |
| `comp` | every `comp=1` gate, any width. **Not monotone once `p_db > 0`** — DB splices insert `comp=1` material, so this grows with the circuit (measured: pinned at 0.996 of size across a 70× phase-A inhale). Monotone only in the no-DB configuration the comp guard was written for. |
| `g57` | of those, exactly two controls of **opposite** polarity — a true g57 |
| `shaped` | of those, exactly two controls, polarity ignored: the shape the store emits, so `1 − shaped/size` is the material the DB did not produce. **This is the DB-effectiveness reading.** |
| `polf` | `(shaped − g57) / shaped`: a **twist odometer**, not erosion. A negation twist conjugates a window by NOT on one wire and flips one control's polarity, moving a gate out of g57 form without changing its shape, width or `comp`. Random-walks toward 0.5 under twist pressure; swap twists leave it at 0. |
| `merges c/x/d/s` | successful Cancel / XFuse / DropLit / Subsume merges |
| `sib / xorig` | merges between siblings (same split event) vs cross-origin — sibling share ≈ how much contraction merely undoes splits |
| `tabu, nopart, wall, far, noadj` | merges blocked: partner too recent / no partner in index / wall between / beyond `merge-reach` / could not be floated adjacent |
| `undo ok/dead/tabu/miss` | journal undos done / entries invalidated by dead pieces / too recent / gather failed. `dead` ≈ permanently unmergeable material |
| `live` | current journal length |
| `expand r1/r2/r3/pre` | crossings by rule; `pre` = g57 presplits (each costs +1 gate, erodes one fossil). Net crossings = `r1+r2+r3 − undos` (see undo above) |
| `fresh/unsub/ins` | fresh-wire splits (0 unless `--w-fresh>0`) / unsubsumes / inserts. Each insert also fires two crossings (counted under `r1/r2/r3`) |
| `twn/tws/twc` | negation / swap / transvection twists performed |
| `twrel/twsplit/twspan` | interior gates relabeled by twists / of those, transvection case-splits (each +1 gate) / total window gates covered — the twist state-transport gauge |
| `twskip` | twists abandoned (no eligible wire, or transvection `b` pool empty) |
| `declined, blockw, dl, bnd` | width-damping declines; width-cap blocks; rule deadlocks; circuit-boundary hits |
| `floats, scat` | float events (incl. crossing floats + failed-shot retreats) / directional birth-advances, each with total steps — mean steps per event is the mobility gauge; if it decays with width creep, lower K or raise damping |
| `disp` | mean normalized displacement of material from its origin position: 0 = unmixed, 1/3 = independent |
| `owin` | distinct origins per 32-gate window (→ 32 = locally saturated interleave) |
| `fan0` | fraction of writes never read before their wire is overwritten |
| `leew` | mean float-box size (gates) — mobility |
| `odiff` | origin diffusion: piece-weighted std of each origin family's positions; 0 = clumped, 0.2887 (=1/√12) = uniformly dispersed |
| `oadj` | Pearson autocorrelation of adjacent gates' origins: 1 = conveyor belt of the original order, 0 = ancestry-independent neighbors |
| `width[i:n ...]` | **cumulative** histogram of created-piece widths over the run (bucket 15 = ≥15). *Not* the current circuit's width profile — use `fmix_stats` for that |
| `gen tgt= G= Gall= tgtbl= alag=x/y lag=a/b c= h= u= wlag= min=` | generation targeting: target; current CIRCUIT generation (5th-percentile over the **targetable** gates); the legacy all-gates percentile; size of the targetable population; all gates below target / total; **still-targetable below-target gates / targetable total — this is the dose-stop fraction**; cheap tier / hard tier / written-off; below-target gates too wide for the DB channel; minimum generation over all gates (`F` = everything is fresh) |
| `cov` | cumulative per-position twist coverage (`twspan / size`) — the phase-A twist dose meter (saturation target ~600) |
| `ing= hard= paid=` | ingest-then-pay: cheap-round hits/rounds; paid-round hits/rounds; total gates the paid channel added (the growth actually spent on the generation goal) |

**Generation semantics** (benchmark definition, 2026-07-21). Every gate
carries a rewrite generation: input gates start at 0; a DB splice stamps its
products with the outgoing window's **upper-median generation + 1** (median
rounded up on even window sizes); every **split** (presplit, cross piece,
fresh-split, unsubsume, twist case-split) stamps children with **parent + 1**;
merges take the min of their parents; fresh insert pairs and twist bracket
packets are born-random MAXGEN (higher than every real generation). The
**circuit generation** `G=` is the largest G such that at least 95% of the
**targetable** gates have generation ≥ G (the 5th-percentile over that
population), and the dose stop fires when `lag/targetable <=
--gen-stop-frac`. ⚠️ Both were measured over ALL gates before `512ce31c`,
which made them useless on material the DB channel cannot re-encode: a
product-share gadget is ~60% width ≥ 3, so those gates never leave the laggard
count, the all-gates fraction floors around 0.38, and the dose stop could never
fire — the run burned its whole move budget after the dose was long complete.
The legacy figure is still printed as `Gall=`; on narrow material the two
agree. The census is the phase-A dose meter — the churn-phase
analogue of ssg's generation mechanism — and `--gen-target`/`--gen-stop-frac`
turn phase A from "run N moves and hope" into "run until the circuit reaches
generation G, then stop," which is also the minimal-growth schedule: targeted
seeds remove the coupon-collector tail, and the dose stop spends no moves
(hence no incidental growth) past the requirement.

**Ingest-then-pay.** Reaching the generation target cheaply first, and paying
growth only where unavoidable: `--p-db-ingest` rounds try to re-encode
cheap-tier laggards with NON-GROWING replacements (Compressing mode — free in
size, so the rate can be high); a seed that keeps failing proves the gate has
no non-growing spelling and graduates at `--gen-miss-budget` to the hard
tier, where `--p-db-hard` rounds pay for it with the SHORTEST existing
spelling (MinGrow mode) — minimal growth per hard core, spent only on gates
the cheap channel demonstrably cannot ingest, ledgered in `paid=`. Gates the
store has nothing for at all retire at `--gen-giveup` (reported `u=`, never
holding the dose stop open) — their count is the true store-unreachability
census of the material. Splice products restart at miss 0: rewritten
neighborhoods may have become cheaply ingestable.

**Known dynamics** (measured): the chain equilibrates at a content-dependent
floor above target under extreme churn; positional transport is the slow mode —
the fixed-size steady state churns composition fast but barely moves material,
while the *growth phase* is what transports (a 44× growth run reached
odiff 0.20 / oadj 0.34 where fixed-size runs sat at 0.01 / 0.96). Fossil
erosion needs no targeting: uniform selection fully eroded 67k g57s within
~37M moves at 44× growth. The four axes are independent and need separate
dosing: **compositional** (saturates in a few moves/gate), **positional/material**
(set by the growth ratio; audited by the fcompress residual), **state/progress**
(the twists — audited by the prefix-distance heatmap; a doubled-rate twist run
drove the plate to `σ ≈ 1.15` bits, mean 128/256, i.e. isotropic coin-flip
distance, and saturated by ~7M moves), and **CNF-inversion hardness** (a fourth
axis, anti-correlated with state mixing). Temperature is not just a size knob —
higher temp means looser size regulation, larger breathing, and measurably
faster fossil erosion at equal moves.

---

### The split stage (`--split`) — phase B part 1

Full spec: `docs/FMIX_SPLIT_TWIST.md`; measurements:
`reports/split_trials_20260805/` + `docs/SPLIT_TWIST_REPORT.pdf`. One move
owns the whole round while the stage is live: split a random g57 by the
randomized presplit; with `--p-join` wrap an **absorbed pure-NOT twist** on
the g57's target wire — the bracket side is drawn ∝ the circuit length
remaining on each side, the bracket is the farthest of `--split-reach-k`
samples there (`0` = the retired other-half-first cascade, kept as an A/B
arm), both brackets absorb the X by flipping their control polarity (zero
synthetic gates), the segment's w-reading pins flip, and segment g57s
reading w are force-split — then one cross shot from the 2-control piece.
The stage ends on g57 exhaustion (the exit that fires in practice) or
`--split-fail-limit` consecutive bracket failures, then the round belongs to
the rest of the command line (`--split-stop` ends the run at the boundary
instead). Growth ≈ 1 + comp-fraction (measured ×1.7–2.0); the output is
CNOT/NCNOT ≈ 50/50 plus 2-control ANDs at 1:2:1 polarity — no comp gate
survives. Instruments: `--split-canaries` wire canaries (flips by original
position), per-move `span=`, span histogram + canary deciles at the
boundary. State files are v2 (tri-state stage phase; v1 loads, old binaries
refuse v2). NOTE: since 2026-08-05 ALL presplit siblings take alternating
directions from a fair draw — pre-existing walks do not replay under the new
binary.

## 3. `fcompress` — what it does

The deterministic final compression pass, and the honest **effective-size
evaluator**: it is attacker-computable, so running it never weakens hiding,
and its output size is the right number to report for any mixed artifact.
Applied to move-stamped `fmix` snapshots it is also a mixing clock — the
greedy-recoverable fraction shrinks monotonically with churn (measured: fsplit
output → 83%, mid-growth fmix → 86%, fmix final → 90%, long-churned → 94%).

Algorithm, iterated to a fixed point (≤ `--max-iters`):

1. **Gather** — one forward sweep keeping an open group per target wire. Two
   closure rules: any *read* of wire `t` closes `t`'s group (a reader pins the
   accumulated value), and any *write* to a wire in a group's
   union-of-member-controls closes that group (member control values may not
   change). These two rules make it unconditionally legal for members to float
   right to the close point. Closures cascade; groups emit in ascending
   last-member order.
2. **Reduce** — each gathered group is `t ^= f1 ⊕ … ⊕ fk`, an ESOP. Apply the
   pairwise catalogue (§1) to fixed point; if the group's support fits in
   `--anf-support-cap` bits, also try the exact ANF rewrite (canonical —
   duplicate monomials annihilate) and keep it when it wins.
3. **Re-emit** the survivors as consecutive XGates — output stays mpmct1, all
   downstream tooling keeps working. The pass may legitimately create `comp=1`
   gates; the fossil metric is frozen at fmix-end, so record it before
   compressing.

**Optional dead-cone pruning** (`--live-wires`), for gadgetized circuits where
equality is only required on designated output wires: one exact backward pass
in the XOR-accumulate model — a gate is deletable iff its target is dead at
its position; a kept gate makes its controls live, and its target *stays* live
(XOR never overwrites). Default is `all` (full correctness on every wire);
keep it that way unless the artifact's contract really is partial.

**Verification:** per-group checks during the pass (exhaustive on small
supports), plus `--verify-rounds` × 64 lanes of sampled global equality
against the untouched input on the live wires. Omit `--output` for a **dry
run** — verify and report, discard the result (the evaluator mode).

### 3.1 `fcompress` CLI reference

| Flag | Default | Effect |
|---|---|---|
| `--input`, `--input-format` | —, `mpmct1` | as in fmix |
| `--output` | none | output file; omit = dry run (verify + report only) |
| `--live-wires` | `all` | `all` \| `upper-half` \| `lower-half` \| explicit list `"0-255,300,510"` |
| `--max-iters` | 10 | gather/reduce cycles (each reduction opens new float paths) |
| `--group-cap` | 64 | close a group proactively at this many members |
| `--anf-support-cap` | 24 | try the exact ANF rewrite only when the group support fits |
| `--verify-rounds` | 64 | rounds of the final 64-lane global check |
| `--no-local-verify` | off | disable per-group verification |
| `--seed` | 0 | seeds the verification sampling only — the pass itself is deterministic |

Log lines: per iteration `gates in -> out | groups= multi= max= catalogue=
anf_wins= live_dropped=` (groups formed / with ≥2 members / largest;
catalogue-merge and ANF wins; gates pruned by liveness), then a `done` summary
with gate and literal percentages. Runtime is single-threaded and scales with
gates × leeway: **budget ~2¼ h per 3M gates.**

---

## 4. `fmix_stats` — the read-only analyzer

Computes the full stationarity-signature suite on any circuit (no
modification): gate/width profile, fanout, leeway, wire and wire-pair
co-occurrence entropy, window wire-span, and — given `--origins` — the
positional suite (`disp`, `diffusion`, `adj_autocorr`, `owin32`) plus
per-origin **spread quantiles** (`single_frac`, p5–p95 family spans,
`frac_lt_ref`). One grep-able `[fstats]` line per metric family.

| Flag | Default | Effect |
|---|---|---|
| `--input`, `--input-format` | —, `mpmct1` | circuit to analyze |
| `--origins` | none | fmix `--origins-out` sidecar → enables positional metrics |
| `--leeway-samples` / `--leeway-cap` | 20,000 / 65,536 | leeway distribution sampling / per-direction scan cap |
| `--span-windows` / `--span-samples` | `32,256` / 2,000 | window wire-span sizes / samples per size |
| `--spread-ref` | 4·w·log₂(w) | reference scale (gates) for `frac_lt_ref` — the random-circuit PRP length (18,432 at 512 wires) |
| `--seed` | 0 | sampling RNG |

Convergence-curve recipe: touch the dump flag periodically during a run, then
`fmix_stats --input <snap>.mv<N> --origins <snap>.mv<N>.origins` per snapshot.

---

## 4b. Reconstruction readouts — `hmap_affine` and `hmap_stat`

Two instruments, asking different questions about the same pair (original `C`,
gadgetized/mixed `G`). Both snapshot `C` and `G` on a shared random input and
compare prefix against prefix; both write `<out>.bin` + `<out>.meta.json`.

`hmap_affine` — **exact** reconstruction. Per cell it fits a GF(2) map from
`G_j`'s wires (degree 1, or products up to `--degree 3` over `--deg2-wire-list`)
to `C_i`'s bit by span membership, and scores `H = 0.5` whenever no exact
relation exists. Low `H` = leak. Read the plates with
`reports/plot_hmap_ridge.py` — by the **ridge** (depth / depthMed / rho), never
by the mean, which saturates at 0.5.

`hmap_stat` — **approximate** reconstruction, the companion that closes the
exact measure's blind spot. Per cell it takes the best agreement over every
single wire and every XOR of two wires (the family containing a value's carrier
pair), optionally plus one AND term over `--and-wires`, against a null floor
from the same search on a random target. High agreement = leak.

| Flag | Default | Effect |
|---|---|---|
| `--c`, `--g`, `--n` | — | original, gadget, logical width |
| `--c-step` / `--g-step` | 200 / 20000 | prefix strides — this measure is O(W²) per cell, so keep the grid coarse |
| `--samples` | 4096 | rounded up to a multiple of 64 |
| `--target-bits` | 16 | random subset of `C`'s state bits per cell (0 = all) |
| `--wire-list` | all | restrict the predictor's wires |
| `--and-wires` | none | allow one `(w_p^a)&(w_q^b)` over this set — the degree-2-capable adversary |

⚠️ Do **not** read an `hmap_stat` plate with `plot_hmap_ridge.py`: it assumes
the inverted convention and will trace the anti-ridge. Use
`reports/band_hardening_20260725/stat_readout.py`, which trims the port rows
**and** columns — cell (0,0) is `C`'s input state against `G`'s input wires and
reads 1.0 in every build, encoded or not.

---

## 5. Recipes

```bash
# Build the production gadget first: sliced sandwich -> product-share
# gadgetization. THE DEFAULTS ARE NOW THE PRODUCTION SETTING -- mask plan
# [2,2,2,3] with the Gray-code fold, single-carrier decode, nonlinear cascaded
# band fill, one band roll per gap, retire-refill epochs. Pass NO --prod flags:
gen_sandwich_gadget gadget.mpmct1 128 3000 3000
# -> 693k gates / 512 wires at n=128; writes gadget.mpmct1.source_c.g57 too,
#    which is what the heatmaps reconstruct against.
#
#   sss --cnot --gadgetize --slice-zero-ccnot        # same, on the CLI path
#
# WARNING: do NOT assemble --prod flags in a shell variable. zsh does not
# word-split on expansion, so "$FLAGS" arrives as ONE argument and the encoding
# silently reverts to defaults. Pass them literally, or use the env-var form.

# WHY THIS PLAN -- docs/CORRELATING_TWO_COMPUTATIONS. Measured at n=128, same C
# and same sandwich, every arm verified. eps = the piling-up product
# PROD_r (1 - 2^(1-d_r)), and the statistical leak is LINEAR in it (R^2 = 0.996
# across five plans spanning 4x in eps):
#   plan           eps     gates  reachable  F1 raw  battery verdict F1/F2
#   [2,3,3] (old)  0.281   809k    95.5%     0.0817  ALIGNED-LEAK / ALIGNED-LEAK
#   [2,2,2,3] NOW  0.094   693k    97.5%     0.0318  flat / flat
#   [2,2,2,3,3]    0.070   924k    96.6%     0.0258  flat / flat
# A degree-2 atom is the stronger STATISTICAL masker (factor 0.5 against a
# degree-3's 0.75) and the weaker ALGEBRAIC one (it sits inside a degree-2 exact
# adversary's span), so a plan trades one against the other. All of these read
# dead on hmap_affine at degree 1 AND 2 -- what differs is only the statistical
# margin. The new default is cheaper, more digestible AND lower leak than the
# plan it replaces; that is the Gray fold's doing, since it made mask-plan cost
# additive rather than multiplicative.

# STEP UP with --prod-k-hi 2 (= [2,2,2,3,3]) when redundancy matters more than
# size. The default keeps ONE degree-3 atom, and that atom is the only thing
# holding the value out of degree-2 exact range; this restores the second, at
# the lowest leak measured, for +14% gates against the old default.

# Laddering stays OFF. The Gray fold reaches 97.5% without it, and the residual
# wide gates are emit_slot mask emissions rather than fold material.
# --prod-ladder-cap 3 would take reachability to ~99.9% for roughly +40% gates.
# Check reachability, never match rate -- they move in opposite directions:
#   blocker_census --g gadget.mpmct1 --min-window 2 --max-window 5 \
#                  --ctrl-cap 2 --db-max-degree 9   # needs FROZEN_DB_DIR

# Check it before spending mixing time on it (both should be run):
hmap_affine --c gadget.mpmct1.source_c.g57 --g gadget.mpmct1 --n 128 \
            --degree 1 --c-step 30 --g-step 1600 --out ridge
python3 reports/plot_hmap_ridge.py --out ridge.png ridge     # want depthMed 0, rho ~ 0
hmap_stat  --c gadget.mpmct1.source_c.g57 --g gadget.mpmct1 --n 128 \
           --c-step 100 --g-step 3800 --samples 8192 --target-bits 8 --out stat
# want the interior median well under the plain gadget's ~0.91; [3,3] reads ~0.70
# Read the ridge by INTERIOR ROWS, not rho: both axes have unencoded ports, so
# rho is two plateaus ranked against noise in a dead middle (two equally-dead
# artifacts once scored 0.448 and 0.019). Want depthMed 0 and zero interior rows.

# Mixing a product-share gadget needs --no-local-verify: the per-rewrite check
# caps support at 24 wires and panics on two wide gates. Read G=/lag=/tgtbl=,
# not Gall=, which is structurally pinned at 0 on this material.
# (With --prod-gray-fold the fold emits no wide gates at all, and adding
# --prod-ladder-cap 3 leaves only ~0.08% of the circuit above 2 controls, so
# this flag may no longer be needed -- UNTESTED, try without it first.)

# Production mixing run, direct on a gadgetized g57 circuit, twists ON.
# Twists collapse the prefix-progress diagonal (the state axis); the growth
# phase does the material transport and fossil erosion. Keep twist weights
# small and set --twist-min-len near the leeway scale.
FMIX_STOP_FLAG=~/tds/run.stop FMIX_DUMP_FLAG=~/tds/run.dump \
FMIX_DUMP_OUT=~/tds/run.snapshot \
fmix --input A_gadgetized.txt --input-format g57 \
     --target-size 1000000 --moves 10000000 --k-max 20 --split-damp 2 --seed 1 \
     --w-twist-neg 1e-3 --w-twist-swap 1e-3 --twist-min-len 256 \
     --output run.txt --origins-out run.origins.txt \
     --report-every 1000000 --verify-every 100000 > run.log 2>&1

touch ~/tds/run.dump      # verified snapshot run.snapshot.mv<N> (+.origins); run continues
touch ~/tds/run.stop      # graceful finish

# Final compression + effective size (full correctness)
fcompress --input run.txt --output run.fc.txt

# Effective size only (dry run), exploiting gadgetized output don't-cares
fcompress --input run.txt --live-wires upper-half
```

For a faster run at the cost of the development-time per-move proof, add
`--no-local-verify` (the sampled `--verify-every` global check still bounds the
blast radius). Twists carry the largest verification cost — each relabel of a
wide gate is checked with a truth table exponential in its support.

---

## 6. File formats

**mpmct1** (plain text, one gate per line):

```
mpmct1 <num_wires> <num_gates>
<target> <comp:0|1> <k> <wire> <pol:0|1> ...   (k wire/polarity pairs)
```

**g57 input**: the existing base-83 `CircuitSeq` format (ssg outputs, ~12
B/gate). mpmct1 is deliberately human-readable (~6 B per literal); `gzip`
recovers ~5× if transfer size matters — representation, not structure.

**origins sidecar**: one decimal origin index per line, aligned with the gate
order of the circuit it accompanies; `4294967295` marks synthetic material.
