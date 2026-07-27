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
the fossil count monotone non-increasing under `fmix`: g57 erosion is
irreversible.

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
| `comp` | surviving g57 fossils (monotone ↓) |
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
# gadgetization. [3,3] = two degree-3 mask terms per value (two is the measured
# floor; one revives the progress diagonal), nonlinear cascaded band fill, one
# band roll per gap so the band is not a body-static wire set. Band auto = 46.
PROD_K=0 PROD_K_HI=2 PROD_DEG_HI=3 PROD_FILL_NL=2 PROD_ROLL=1 \
gen_sandwich_gadget gadget.mpmct1 128 3000 3000
# -> 247k gates / 558 wires at n=128; writes gadget.mpmct1.source_c.g57 too,
#    which is what the heatmaps reconstruct against.

# Same settings on the CLI path:
#   sss --cnot --gadgetize --slice-zero-ccnot \
#       --prod-k 0 --prod-k-hi 2 --prod-deg-hi 3 --prod-fill-nl 2 --prod-roll 1

# STRONGLY RECOMMENDED: add the Gray-code fold. Without it the fold emits its
# cartesian product as gates of width up to arity*max_deg, and everything above
# 2 controls is material fmix can NEVER re-encode -- 56% of the gadget, and the
# reason a mixing run's reach is capped before it starts. See docs/GRAY_FOLD_CG.
PROD_GRAY_FOLD=1 PROD_LADDER_CAP=3 \
gen_sandwich_gadget gadget.mpmct1 128 3000 3000
#   ... or --prod-gray-fold 1 --prod-ladder-cap 3 on the sss path.
#
# What it costs and buys at n=128 (same sandwich, same C, all verify PASSED):
#                        gates    fold fossils   store-reachable
#   wide fold (today)    340k     153,421         31.6%
#   + Gray fold          809k     0               95.5%
#   + gray + ladder 3   1021k     0               99.9%
# Gray fold alone leaves 4.3% wide gates -- NOT from the fold (fossils=0) but
# from emit_slot, where a degree-3 mask is a 3-control gate every time it is
# injected/re-sourced/stripped. --prod-ladder-cap 3 clears those. This does not
# contradict "laddering peaks at cap 4 and declines after": the Gray fold
# REMOVES the wide fold fragments rather than spelling them out, so cap 3 is
# left doing only single-rung width-3 slot emissions, which is what it is good
# at. Check reachability, never match rate -- they move in opposite directions:
#   blocker_census --g gadget.mpmct1 --min-window 2 --max-window 5 \
#                  --ctrl-cap 2 --db-max-degree 9   # needs FROZEN_DB_DIR

# Check it before spending mixing time on it (both should be run):
hmap_affine --c gadget.mpmct1.source_c.g57 --g gadget.mpmct1 --n 128 \
            --degree 1 --c-step 30 --g-step 1600 --out ridge
python3 reports/plot_hmap_ridge.py --out ridge.png ridge     # want depthMed 0, rho ~ 0
hmap_stat  --c gadget.mpmct1.source_c.g57 --g gadget.mpmct1 --n 128 \
           --c-step 100 --g-step 3800 --samples 8192 --target-bits 8 --out stat
# want the interior median well under the plain gadget's ~0.91; [3,3] reads ~0.70

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
