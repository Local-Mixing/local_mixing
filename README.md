# Local Mixing — Circuit Obfuscation

Local Mixing is a construction pipeline aimed at **indistinguishability
obfuscation of general-purpose programs**. The program is reduced to a
boolean circuit, the boolean circuit to a reversible circuit, and the
reversible circuit to a circuit over a single universal gate type — then
two obfuscation stages are applied: **gadgetization** (every intermediate
value is cryptographically shared across many wires) and **mixing**
(the circuit text is re-randomized by a long sequence of semantics-
preserving local rewrites).

## The phases

```
program ──▶ boolean circuit ──▶ reversible circuit ──▶ gadgetized circuit ──▶ mixed circuit
                                (eca57 / MPMCT)          (4n wires, shares)     (fmix + fcompress)
```

1. **Reversible compilation.** Everything is expressed in the *eca57*
   gate (`g57`): `x_a ← x_a ⊕ (x_b ∨ ¬x_c)` — one gate type, universal for
   reversible computation, self-inverse. Two adjacent identical gates
   cancel. See `docs/Mixing_Pieces_Documentation.md` §2.
2. **Sliced sandwich.** The secret circuit C is embedded on the "zero
   slice" of a larger random reversible circuit A on 2n wires with
   `A(x, 0) = (junk, C(x))`; every other slice computes C of an affinely
   disturbed input. See `docs/SLICED_SANDWICH.md`.
3. **Gadgetization.** The sandwich is expanded to 4n wires; every logical
   value becomes a *secret share* spread over several wires, and every
   gate becomes a *gadget* that manipulates shares without ever
   materializing the underlying value on any small set of wires.
   Theory, security properties, and the per-gadget explainers:
   **`GADGETIZATION.md`**.
4. **Mixing.** A move-based random walk re-encodes the circuit
   (expansion/contraction against a frozen replacement database,
   structure-breaking split-and-cross walks), then `fcompress` measures
   how much an attacker-computable compressor can still recover.
   **`MIXING.md`**.

The production packaging of phases 1–4 (every knob and its rationale) is
`docs/PIPELINE_OVERVIEW.md`, runnable via `scripts/gss_mix.sh`.

## Testing

Every gadget construction is run through a six-attack audit battery —
direct wire match, exact linear algebra against single wire states and
against the full execution trace, and statistical correlation scans at
weights 1/2/3 — across chain lengths k ∈ {1, 2, 16}, with and without
mixing, with ideal or band-fed randomness. How it works, how to run it,
and the current results matrix: **`TESTING_PIPELINE.md`**.
Reports: `reports/mx/REPORT.md`.

## Repo map

| path | what |
|---|---|
| `GADGETIZATION.md` | gadgetization theory + security properties + gadget ladder |
| `MIXING.md` | how mixing works (links into `docs/`) |
| `TESTING_PIPELINE.md` | the gauntlet: attacks, cells, thresholds, results |
| `docs/PIPELINE_OVERVIEW.md` | production pipeline: every parameter and why |
| `docs/Mixing_Pieces_Documentation.md` | reference for all mixing pieces/moves |
| `docs/NONLINEAR_GADGETIZATION.md` | why linear sharing leaks; product-share design |
| `README_gate_gadget_v2.md` | the folded 193-gate gadget and the 939-gate six-gadget cascade |
| `SSG_README.md` | the paper's paired secret-share gadgetizer |
| `docs/PRODUCT_SHARE_ENCODING.md` | the band/product-share encoding |
| `src/` | Rust engine (gadgetizers, mixer, attack binaries) |
| `gate_gadget.py`, `gate_gadget_v2.py`, `big_gate_gadget.py` | Python gadget builders + their own test suites |
| `gate_gadget_w2.py` | weight-2 (fan-in ≤ 2) decomposition of the folded gadget → nonlinear291 / behemoth1415 |
| `gauntlet.py`, `gauntlet_build.py`, `gauntlet_heatmap.py` | testing orchestration |
