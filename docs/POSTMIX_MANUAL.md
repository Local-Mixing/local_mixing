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

**Expansion moves** (chosen by relative weight; each produced piece with `c`
controls is width-damped — allowed outright if `c ≤ split_damp`, else with
probability `2^-(c−split_damp)`):

| Move | Weight flag | What it does |
|---|---|---|
| crossing | `--w-cross` (0.70) | float a random gate to its collision point and split it past the collider by one R-rule (R1/R2/R3, a Hurwitz-style conjugation step); a shot g57 pre-splits into its two conjunction pieces. Recorded in the undo journal. |
| fresh split | `--w-fresh` (0.15) | case split `R → xR, !xR` on a **uniformly random uninvolved wire** — the entropy move that couples arbitrary wires into a gate's support. |
| unsubsume | `--w-unsub` (0.10) | inverse of `Subsume`: `!lR → R, lR`. |
| copy-pair insert | `--w-insert` (0.05) | insert an identical adjacent pair of an existing gate (an identity) at a random position. |

**Contraction moves.** With probability `--undo-frac` first try a **journal
undo**: exactly reverse a recorded crossing while all its pieces are alive
(arena-stamp-validated). Crossings are the one expansion the pairwise
catalogue cannot invert (R-ladder rungs are pairwise unmergeable), so without
the journal size creeps up at the crossing rate regardless of the thermostat —
and dead journal entries are permanently unmergeable material, so **size creep
above target tracks irreversible mixing**. Otherwise a **catalogue merge**:
pick a random gate, find the nearest reachable partner through the global
(target, wire-set) hash index within `--merge-reach`, float the pair adjacent
(incremental wall check), apply the catalogue.

**Tabu.** A split event may not be undone or sibling-merged until
`--tabu-moves` moves have passed — freshly split pairs cannot instantly
rejoin.

**Provenance.** Every gate carries `(origin, event)`: the input-gate index its
material descends from, and the split event that created it. Splits pass the
parent's origin to both pieces; merges keep a survivor's. `--origins-out`
writes the origin of each output gate (one per line, final order;
`4294967295` = synthetic material with no input ancestor).

**Final step.** Every gate floats to a uniform random position in its
two-sided collision box (skip with `--skip-final-float`), then the output is
verified and written.

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
| `--split-damp` | 2 | width damping **c** (fsplit convention): a produced piece with `c` controls passes w.p. `min(2^-(c−D),1)`. Dominates growth speed far more than K |
| `--merge-reach` | 4096 | max distance (gates) to a merge partner; bounds the locating scan and wall check |
| `--journal-len` | 262,144 | undo journal capacity |
| `--undo-frac` | 0.5 | fraction of contraction moves that try a journal undo first |
| `--tabu-moves` | 2000 | refractory age (moves) before an event may be undone/sibling-merged |
| `--w-cross/-fresh/-unsub/-insert` | .70/.15/.10/.05 | expansion move weights (set all to 0 for pure-drain mode: recovers recyclable slack, no new mixing) |
| `--verify-every` | 10,000 | sampled global equality check cadence (moves) |
| `--report-every` | 50,000 | report line + stop/dump flag check cadence (moves) |
| `--no-local-verify` | off | disable the per-move exhaustive local check |
| `--skip-final-float` | off | skip the final uniform float pass |
| `--origins-out` | none | write the per-gate origin sidecar |
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
| `expand r1/r2/r3/pre` | crossings by rule; `pre` = g57 presplits (each costs +1 gate, erodes one fossil) |
| `fresh/unsub/ins` | fresh-wire splits / unsubsumes / copy-pair inserts |
| `declined, blockw, dl, bnd` | width-damping declines; width-cap blocks; rule deadlocks; circuit-boundary hits |
| `floats, scat` | float and scatter events / total steps — mean steps per event is the mobility gauge; if it decays with width creep, lower K or raise damping |
| `disp` | mean normalized displacement of material from its origin position: 0 = unmixed, 1/3 = independent |
| `owin` | distinct origins per 32-gate window (→ 32 = locally saturated interleave) |
| `fan0` | fraction of writes never read before their wire is overwritten |
| `leew` | mean float-box size (gates) — mobility |
| `odiff` | origin diffusion: piece-weighted std of each origin family's positions; 0 = clumped, 0.2887 (=1/√12) = uniformly dispersed |
| `oadj` | Pearson autocorrelation of adjacent gates' origins: 1 = conveyor belt of the original order, 0 = ancestry-independent neighbors |
| `width[i:n ...]` | **cumulative** histogram of created-piece widths over the run (bucket 15 = ≥15). *Not* the current circuit's width profile — use `fmix_stats` for that |

**Known dynamics** (measured): the chain equilibrates at a content-dependent
floor above target under extreme churn; positional transport is the slow mode —
the fixed-size steady state churns composition fast but barely moves material,
while the *growth phase* is what transports (a 44× growth run reached
odiff 0.20 / oadj 0.34 where fixed-size runs sat at 0.01 / 0.96). Fossil
erosion needs no targeting: uniform selection fully eroded 67k g57s within
~37M moves at 44× growth.

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

## 5. Recipes

```bash
# Production mixing run, direct on a gadgetized g57 circuit (fsplit-replacement mode)
FMIX_STOP_FLAG=~/tds/run.stop FMIX_DUMP_FLAG=~/tds/run.dump \
FMIX_DUMP_OUT=~/tds/run.snapshot \
fmix --input A_gadgetized.txt --input-format g57 \
     --target-size 3000000 --moves 50000000 --k-max 20 --split-damp 2 --seed 1 \
     --output run.txt --origins-out run.origins.txt \
     --report-every 1000000 --verify-every 100000 > run.log 2>&1

touch ~/tds/run.dump      # verified snapshot run.snapshot.mv<N> (+.origins); run continues
touch ~/tds/run.stop      # graceful finish

# Final compression + effective size (full correctness)
fcompress --input run.txt --output run.fc.txt

# Effective size only (dry run), exploiting gadgetized output don't-cares
fcompress --input run.txt --live-wires upper-half
```

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
