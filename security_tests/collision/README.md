# 2n→n collisions on random reversible circuits

Construct a compressing hash from a reversible circuit $C$ on $\mathrm{pad}+\mathrm{in}$
wires:

$$
H(x) = C(0^{\mathrm{pad}} \Vert x)_{\mathrm{out}}
$$

Wire 0 is the LSB. The default experiment uses a **96-wire, 1024-gate**
random G57 circuit with $\mathrm{pad}=32$, $\mathrm{in}=64$, $\mathrm{out}=32$,
i.e. a 64→32 hash. A collision is a pair $x_1 \neq x_2$ with $H(x_1)=H(x_2)$.

## Why birthday beats SAT here

For a 32-bit digest the birthday bound is about $2^{16}$ evaluations. Each
evaluation of a 1024-gate / 96-wire G57 circuit is cheap, so a hash-table
search finishes in well under a second. A direct SAT encoding duplicates the
circuit (two evaluations), adds equality on the digest bits and a
differ-constraint on the inputs, and yields a formula with a few thousand
variables — solvable, but unnecessary for finding *any* collision at this
size.

Prefer SAT when the digest is large enough that $\sim 2^{\mathrm{out}/2}$
evaluations are impractical, or when you want a constrained
preimage/collision (e.g. sparse inputs). The encoder below is kept for that
regime and for cross-checks on toy widths.

## Generate the circuit

From the repository root:

```bash
cargo build --release --features security-tools \
  --bin gen_collision_circuit --bin birthday_collision

target/release/gen_collision_circuit \
  security_tests/collision/fixtures/c96_g1024 96 1024 20260330
```

This writes `c96_g1024.g57`, `c96_g1024.mpmct1`, and `c96_g1024.meta.json`.
The checked-in fixture was generated with seed `20260330`.

A recorded birthday witness for that fixture (search seed `1`) is
`c96_g1024.birthday.json`:

| | |
| --- | --- |
| $x_1$ | `0xf64d34a740cbd971` |
| $x_2$ | `0x741f0b81c8aaf22a` |
| $H(x_1)=H(x_2)$ | `0xd4fc8d7a` |
| samples | 55 618 (~138 ms) |

## Birthday attack

```bash
mkdir -p target/security-demo/collision
target/release/birthday_collision \
  security_tests/collision/fixtures/c96_g1024.g57 \
  --pad 32 --in-bits 64 --out-bits 32 \
  --samples 300000 --seed 1 \
  --out target/security-demo/collision/birthday.json
```

Re-check a reported pair with the ordinary evaluator (high 32 wires stay 0
when the 64-bit message is passed as the full state):

```bash
cargo run --release --locked -- circuit evaluate -n 96 \
  -s security_tests/collision/fixtures/c96_g1024.g57 \
  --input 0x<x1>
```

Compare the low 32 bits (8 hex digits) of the two outputs.

## SAT attack (optional)

```bash
mkdir -p target/security-demo/collision
g++ -std=c++17 -O3 security_tests/collision/collision_to_cnf.cpp \
  -o target/security-demo/collision/collision_to_cnf

target/security-demo/collision/collision_to_cnf \
  security_tests/collision/fixtures/c96_g1024.mpmct1 \
  target/security-demo/collision/collision.cnf \
  --in-bits 64 --pad 32 --out-bits 32

# requires an external solver on PATH, e.g. kissat
kissat --time=60 target/security-demo/collision/collision.cnf \
  > target/security-demo/collision/solver.log || true

python security_tests/collision/decode_collision_model.py \
  --circuit security_tests/collision/fixtures/c96_g1024.mpmct1 \
  --solver-output target/security-demo/collision/solver.log \
  --in-bits 64 --pad 32 --out-bits 32 \
  --out target/security-demo/collision/sat-verified.json
```

On toy sizes (`--in-bits 8 --pad 4 --out-bits 4`) the same encoder is small
enough for a quick solver smoke test.

## Tests

```bash
python -m unittest security_tests.collision.test_collision -v
```
