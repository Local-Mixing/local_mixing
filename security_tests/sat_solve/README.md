# Zero-slice SAT preimages

This workflow finds an input whose selected output bits equal a target.
The circuit is an expanded `mpmct1` tape. Input wires `[0,n)` are free and
every other input wire is fixed to zero. The constrained output is `n` bits
starting at `--output-start`, which defaults to `n`. All other outputs are
unconstrained. The tape may have any width that contains those ranges.

Wire zero is the least significant bit. For example, target `0x5` means bits
`1,0,1` on output wires `output_start`, `output_start+1`, `output_start+2`.
For a different TDP port layout, set the output start explicitly. Packed
`esop1` files must first be expanded to the equivalent `mpmct1` representation.

## Complete small example

Run from the repository root. Compile with any C++17 compiler:

```bash
mkdir -p target/security-demo
g++ -std=c++17 -O3 security_tests/sat_solve/zero_slice_to_cnf.cpp \
  -o target/security-demo/zero_slice_to_cnf

target/security-demo/zero_slice_to_cnf \
  security_tests/sat_solve/fixtures/zero_slice.mpmct1 \
  target/security-demo/preimage.cnf 3 0x5 --output-start 3
```

The encoder validates the complete tape before writing. It assigns the first
`width` DIMACS variables to input wires and one new target variable to each
gate. A gate with `k` controls requires `2k+2` clauses; extra unit clauses
constrain padding inputs and requested outputs. `--analyze-only` reports
formula size without writing the CNF. `--help` describes the interface.

Install an external Kissat executable and place it on `PATH`. Then:

```bash
status=0
kissat --time=60 target/security-demo/preimage.cnf \
  > target/security-demo/solver.log || status=$?
```

Kissat uses exit code `10` for SAT, `20` for UNSAT, and `0` for an inconclusive
run, including a time limit. Other exit codes indicate errors. Only continue
to the decoder for a SAT result with a printed model (`s SATISFIABLE` and
`v ...` lines):

```bash
python security_tests/sat_solve/decode_model.py \
  --circuit security_tests/sat_solve/fixtures/zero_slice.mpmct1 \
  --solver-output target/security-demo/solver.log \
  --n 3 --output-start 3 --target-hex 0x5 \
  --out target/security-demo/verified-input.json
```

The decoder checks that every input variable is present and consistent,
re-executes the Boolean circuit, verifies zero padding and the requested
target, and records the input and verification result in a new JSON file.
It returns zero only when verification succeeds and refuses to overwrite an
existing report. Initial variable assignments suffice because the independent
execution does not trust the solver's intermediate variables.

## Input-cube variation

For larger instances, try partial assignments of the free input variables.
The wrapper writes a separate CNF and solver log for every sampled cube:

```bash
python security_tests/sat_solve/solve_input_cubes.py \
  --base-cnf target/security-demo/preimage.cnf \
  --out-dir target/security-demo/cubes \
  --kissat kissat --var-count 3 --fixed-bits 1 \
  --cubes 8 --seconds 30 --seed 1
```

`--var-start` defaults to one, matching the encoder's first input variable.
`--var-count` is the logical input width, not the whole CNF variable count.
`--fixed-bits` chooses how many of those inputs to fix per trial. By default
their values are random; `--center-hex` takes values from a supplied input.
`--seed` controls cube selection and the sequence of solver seeds.

The output directory must be new. The wrapper returns `0` when a cube is SAT,
`1` when no sampled cube produces a witness, and `2` for a solver error.
Exhausting sampled cubes is **inconclusive for the original formula**, even
if every selected cube is UNSAT: they need not cover the input space. A SAT
cube log can be passed to the same decoder with the original tape and target.

## Test without an installed solver

```bash
python -m unittest security_tests.sat_solve.test_workflow -v
```

The tests compile the encoder, exhaustively check tiny gate/CNF truth tables,
exercise arbitrary circuit widths and output ranges, independently verify a
known input, reject incomplete/contradictory models, and check cube status
handling with a simulated solver response.
