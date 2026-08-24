# Nonlinear gadgetization references

This package contains the two canonical nonlinear r57 reference gadgets imported from
the gadget gauntlet:

- `nonlinear193`: the folded generalized-Toffoli construction. With two-share controls
  and clean scratch, one r57 gate emits exactly 193 physical gates with maximum fan-in 4.
- `nonlinear291`: the mask-first fan-in-two decomposition of the same construction. It
  emits exactly 291 physical gates with maximum fan-in 2 and restores its decomposition
  ancillas and prefix cache.

Both use the five-wire encoding
`E(x) = x0 ^ x1 ^ maj(x2, x3, x4)` and represent mutable logical wires with two encoded
shares. The default r57 operation is
`c_out = c_in ^ (a | ~b) = c_in ^ 1 ^ b ^ (a & b)`.

## Stable API

```python
from gadgetization import nonlinear193, nonlinear291

circuit, info = nonlinear193.run_gate(samples=4096, seed=1)
w2_circuit, w2_info = nonlinear291.run_gate(samples=4096, seed=1)

chain193 = nonlinear193.build_chain(samples=4096, seed=1)
chain291 = nonlinear291.build_chain(samples=4096, seed=1)
```

`run_gate` returns `(circuit, info)`. The circuit exposes:

- `s`: final wire bit-vectors;
- `init`: initial wire bit-vectors;
- `flips` and `newvals`: full per-gate trace vectors;
- `gate_log`: `(target, comp, controls)` for every emitted physical gate;
- `marks`: zero-cost section labels.

The `info` mapping contains sampled clear values (`a`, `b`, `c_in`, `gate_ab`, `c_out`),
the decoded `c_out_actual`, `correct`, restoration status, `n_gates`, `max_fanin`, and a
`layout` mapping of semantic names to wire indices. Restoration fields have literal,
mode-aware meanings:

- `scratch_restored` is true only when every wire in `layout["scratch"]` is zero. It is
  normally false in dirty mode because dirty `scratch2` wires deliberately retain masked
  garbage.
- `required_ancillas_restored` is true when every ancilla that the selected clean/dirty
  mode promises to restore is zero. Dirty `scratch2` wires are excluded from that promise.
- The 291 variant also reports `decomposition_ancillas_restored` for its fallback, cache,
  and temporary wires.

The 291 variant additionally reports `max_physical_fanin` and `max_requested_fanin`;
these are 2 and 4 respectively for the default r57 construction.

`build_chain` returns a mapping containing `circ`, named clear `targets`, `written`,
`correct`, restoration status, and fan-in metadata. Its second r57 gate consumes the first
gate's output as its `a` operand. The package root also exposes convenience aliases
`run_nonlinear193`, `run_nonlinear291`, `build_nonlinear193_chain`, and
`build_nonlinear291_chain`.

For tooling that validates construction identity, each module exports
`CANONICAL_R57_GATE_COUNT` and `CANONICAL_MAX_PHYSICAL_FANIN`. The 291 module also exports
`CANONICAL_MAX_REQUESTED_FANIN`. The package root aliases the counts as
`NONLINEAR193_R57_GATE_COUNT` and `NONLINEAR291_R57_GATE_COUNT`.

For integration into an existing encoded circuit, call each module's `gadget_gate` with
explicit wire blocks. In the 291 module the wrapper reclassifies masks and flushes cached
prefix products before returning, so it can be used repeatedly in a chain. See the
function docstrings for the clean/fresh wire contract. Clean mode accepts either one
integer `scratch2` wire shared by all B groups or a sequence with one wire per group.
Dirty mode requires a sequence of distinct integer wires, exactly one per group. General
sequence types such as tuples, lists, and ranges are accepted.

## Native topology templates

`templates/` contains deterministic `mpmct1` topologies for the `r57`, `nab`, `and`,
and `copy` operations in both variants. They use clean shared-B construction and two
five-wire shares for every logical operand; in particular, `copy` consumes two B blocks.
The files use fixed local wire layouts so a native integrator can relabel their wires.
The Rust GSS adapter in `src/preprocessing/nonlinear_gss.rs` consumes these files at
compile time and validates their wire count, gate count, and physical fan-in before use.

Regenerate or verify the checked-in files from any working directory with:

```bash
python -m gadgetization.export_templates
python -m gadgetization.export_templates --check
```

The exporter validates the canonical gate counts and physical fan-in bounds before it
writes anything. Its `build_template` and `render_all_templates` functions provide the
same topology and serialized text entirely in memory for tests and integration tooling.

## Security scope

The constructions were selected and evaluated with exact-affine and low-weight
correlation tests over sampled traces. Those results are empirical evidence under the
gauntlet's threat model, not a cryptographic proof. New compositions, permutation pairs,
gate variants, or scratch disciplines should be run through the gauntlet again.
