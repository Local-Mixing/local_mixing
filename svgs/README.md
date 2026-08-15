# svgs/ — gadget wiring specs and diagrams

One standardized readable text form (`.gtx`) per gadget, plus the rendered
SVG. A `.gtx` file is a flat, diff-able description of *every gate the
gadgetizer emits for one source gate*, with `#` comment lines marking
construction phases:

```
@title  ...                       diagram title
@subtitle ...                     one-line explanation
@equation ...                     semantic summary (drawn at the bottom)
@wire <name> [group]              one wire lane per line, top-to-bottom
@note <wire> <text>               annotation (legend area)
# <label>                         section break → dashed box in the SVG
t <--- a & !b                     conjunction gate:   t ^= a ∧ ¬b
t <--- a | !b                     native r57 gate:    t ^= a ∨ ¬b
t <--- a                          CNOT
t <--- 1                          NOT
```

Pin legend (same style as `sg_homomorphic_r57_gadget.svg` / `nonlinear_189.svg`):
**●** positive control · **○** negative control · **◎** target
(a tick on the target marks a complemented-constant gate).

## What counts as "the gadget"

Only the **replicated per-gate body** — the thing emitted once per source
gate, including deep inside a circuit. The one-time translations
cleartext → encoded input (share setup, mask injection) and
encoded output → cleartext (unshare, strip/route-home) are fixed
infrastructure and are **not** part of the gadget or its gate count. This is
the same convention as the names and the gauntlet's marginal-cost
measurements (fixed infra excluded).

For the deterministic Python gadgets the count is exact (193, 939). For the
Rust gadgetizers the per-gate cost is a *seeded draw* (menu variants, RG
choices, mask sourcing): the name carries the expectation (E = 14.48 → 14;
≈92.4 → 92), and the published diagrams use a seed whose instance draws
exactly the rounded average — most honest to show an average example.

## The four gadgets

| file | gadget | what is shown |
|---|---|---|
| `secretshare14.gtx/.svg` | secret sharing (`--gadget ss`) | SG variant 5 (6) + RG1 (6) + RG3 (2) = **14 gates**, seed 14 (average draw) |
| `bandproduct92.gtx/.svg` | band-product (`--gadget semi`) | Gray fold `V = C ⊕ M(B) ⊕ κ` + 2 value relocations + mask re-source + band roll = **92 gates** at the gauntlet's chain width n=8, seed 10 (average draw); mask injection run but not shown |
| `nonlinear193.gtx/.svg` | folded gadget (`gate_gadget_v2.py`) | both masked groups: build B / build scr2 / A_i + run → o_i / unbuild = **193 gates** exactly |
| `nonlinear939.gtx/.svg` | behemoth (`big_gate_gadget.py`) | six max-size sub-gadgets G1–G6 (r57/nab/and/and/copy/copy), each a full 193-style section sequence = **939 gates** exactly |

## Regenerating

The generators (`src/bin/gadget_spec.rs`, `gadget_spec.py`, `gadget_svg.py`)
are currently local-only tooling, not part of this repository; the committed
`.gtx` files below are the canonical text form. With the generators present:

```sh
# Rust gadgetizers — seed is an explicit argument (menu/RG/mask draws)
./target/release/gadget_spec --gadget ss   --seed 14        > svgs/secretshare14.gtx
./target/release/gadget_spec --gadget semi --seed 10 --n 8  > svgs/bandproduct92.gtx

# Python gadget builders — wiring is structurally deterministic; --seed only
# re-seeds the dummy wire values used while recording the wiring
python3 gadget_spec.py --gadget nonlinear193 [--seed N]    > svgs/nonlinear193.gtx
python3 gadget_spec.py --gadget nonlinear939 [--seed N]    > svgs/nonlinear939.gtx

# text → SVG
python3 gadget_svg.py svgs/secretshare14.gtx -o svgs/secretshare14.svg
```

Notes:

- `gadget_spec` (Rust, `src/bin/gadget_spec.rs`) emits the per-gate body
  using the exact production functions (`emit_cg_menu` / `emit_nonlinear_rg` /
  `ProdLedger::fold_cg` / `emit_value_relocation` / `resource` / `roll`) with
  the same RNG discipline as the gauntlet. For `semi` the one-time mask
  injection is *run* (so the fold draws against a stocked ledger) but not
  emitted. `--warmup W` skips W full gate cycles first; `--n` sets the
  ambient value count (the fold's mask slot space depends on it — the
  gauntlet runs chains at n=8, where one gate's body averages ~92).
- `gadget_spec.py` drives `gate_gadget_v2.gadget_gate` /
  `big_gate_gadget.sg_gate` over dummy wires and dumps `gate_log` plus the
  builders' section marks (`Circuit.mark`).
- `gadget_svg.py` is a standalone parser/renderer for the `.gtx` grammar
  (no project imports), so specs can also be hand-written.
