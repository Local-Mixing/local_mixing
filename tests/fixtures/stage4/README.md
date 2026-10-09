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
