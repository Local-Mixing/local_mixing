# Stage-4 fixtures

## Pinned reference binaries (docs/STAGE4_RUST_DESIGN.md §10.1)

- **Baseline `hmap_affine`**: built from `492564f0` with
  `cargo build --locked --release --features security-tools --bin hmap_affine`
  from a clean export of that commit (`git archive 492564f0`), into
  `port_runs/baseline_target`. The accepted sha256 values are listed in
  `cview_n8/baseline_sha256.txt` (one per platform build); fixture
  regeneration and `tests/manual/stage4/cview_parity.sh` refuse any other
  binary, so expected outputs never come from a binary that links the
  library under test.
- **Archive-era `hmap_affine`** (produced the Oct 8 re-repairs): sha256
  `94757413406e58c9fe40694e392213da015b198a0cfb22a83f4b9b79ca8a84f3`, copied
  read-only to `port_runs/archive_hmap_affine`.

## `cview_n8/`: C-view exposure engine (P2)

Inputs:

- `c.g57`, `g2.mpmct1`: a real n=8 stage-1+2 output (preset L8, 16-gate C,
  18,238-gate G2 on 32 wires; a small-n feasibility run, not a deliverable).
- `cuts.txt`: 241 distinct cuts with blank lines, padded lines, duplicates and
  one cut beyond the circuit (exercises the cut-file rules).
- `rich_c.mpmct1`, `rich_p.mpmct1`, `rich_cuts.txt`: synthetic pair from
  `gen_rich.py` (deterministic). P shares values with C through linear
  traffic, giving t* of 1 to 3, invisible rows, nullity above 14 and a
  `skip_settled`-dependent approximate count, which the real G2 does not.

`cases.tsv` lists the runs (name, EXPOSURE_ALLCUTS on/off, EXPOSURE_WITNESS
on/off, arguments; `@/` = this directory). `expected/` holds the baseline's
stdout and TSVs for every case; `regen.sh` (with `HMAP_BASELINE=`) rebuilds
it. The CI test `--features security-tools --test hmap_affine_exposure`
runs the refactored binary on every case and compares bytes.

## Oracle runtime (§10.1)

Every expected output below comes from the UNMODIFIED Python oracle
(`scripts/windowed_delivery/`), copied into a path-rewritten sandbox by
`oracle_sandbox.sh` (each rewrite asserted to apply exactly as often as
expected; only the scripts the stage-4 chain runs are copied) and run under
an interpreter that passes `pyset_selftest.py` (CPython 3.9 set iteration
order). Recorded with `/usr/bin/python3` = CPython 3.9.6 (Command Line Tools,
macOS 27), self-test: 1469 small orders and 12 large checkpoints, 0
mismatches. The baseline `hmap_affine` is the pinned one above.

- `pyset_selftest.py`, `pyset39_traces.txt` (small traces: keys up to 300
  bits, iteration order after each step), `pyset39_large.txt` (generator
  seeds + FNV-1a digests at checkpoints; tables up to 2^19 slots, the
  `used > 50000` growth branch). The Rust emulator replays both
  (`tests/unit/stages/packing/pyset.rs`).

## `n8_L8/`: real chain at n=8 (ANF / deliverable phase)

CALIBRATION seeds, never deliverables: gen seed 11 (all three gen seeds),
mix seed 1013. `regen.sh n8_L8` runs `gen_sandwich_gadget 8 16 16 24 1 160`
with the L8 environment, `circuit_mixer` with the val_chain.sh stage-3 argv
on the local frozen store (`FROZEN_FILTER=1`), then the oracle stage 4
exactly as `val_chain.sh`: `window_pack3.py 512 512 24576 3000`, `repair.sh`
(3 cancel passes, 1327 pairs; fix iter 1 at 6 visible <= floor 8; one
exposure-aware split of a 133,935-monomial window), `anf_dump_eval.py`, and
`anf_compose.py` at budgets 65536/500, 64/3 and 256/3.

Committed: `c.g57` (C, the same calibration C as `cview_n8`), `rep.mpmct1`,
`rep.bounds` (the repaired circuit and lattice), `deliverable.bin` (DE body),
`deliver_line1.txt` (DE line 1, time normalized), `deliver_match.txt`,
`anf_rep.txt`, `anf_b64.txt`, `anf_b256.txt` (AC stdout), `provenance.txt`,
`container_header.bin` (the 48-byte header from `container_header.py`).
The intermediate G2/G3/packed files are not committed (the packer and repair
phases will add what they need).

## `anf/`: crafted ANF / serialization cases

`anf_cases.py` (deterministic, seed 20261008) writes the circuits and bounds:
state widths 16, 20, 32, 64 (wires 40..63), 128, 160, 256, 300, 512 and 1024
(monomial widths W = 1, 2, 4, 8, 16; wire ids 61, 63, 64, 127, 128, 255, 256,
299, 511, 1023); k=0, complemented and negative-polarity gates; bounds with
0, m, > m, negative, duplicate and unsorted entries; empty windows (`cancel`:
29 of 48 windows have f = c = 0); a single window; a zero-gate circuit; and
`orderdep`, whose products abort under CPython 3.9 order but fit when
completed. `cases.txt` lists each case with its budgets. Per case:

- `<name>.bin`, `<name>.deliver.txt`, `<name>.match`: `anf_dump_eval.py`
  body, line 1 (time normalized) and its evaluation check;
- `<name>.anf_b<B>.txt`: `anf_compose.py <C> <B> <B> <name>-b<B> 2` stdout
  (spot checks every 2nd window);
- `<name>.win_b<B>.txt`: `anf_windows.py` (the oracle's `window_anf`, as is
  and with a product-completing `mul`): per window `ok:<total>:<digest>` or
  `overflow` under cpython39 and canonical semantics. The fixtures exercise
  all three overflow kinds, 14 order-dependent product aborts and 6 window
  decisions that differ between the two semantics;
- `<name>.tdp4hdr`: container header from `container_header.py`
  (source width 1, state width = header n).

`tests/integration/tdp_packing.rs` compares `tdp_packing serialize`,
`anf-report` and `window-anf` with all of these byte for byte.
`tests/manual/stage4/anf_parity.sh` does the same on the 8 archived
validation runs (Appendix B sha256s).

## `regen.sh`

    HMAP_BASELINE=... ORACLE_PY=/usr/bin/python3 CARGO_TARGET_DIR=... \
    FROZEN_DB_DIR=~/code/local_mixing/frozen_db/frozen_m1_m11 \
    WORK=<durable scratch> bash regen.sh [n8_L8] [anf] [container]
