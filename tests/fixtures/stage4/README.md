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

Committed: `c.g57` (C, the same calibration C as `cview_n8`), `g3.mpmct1`
(the stage-3 output; its G2 is `cview_n8/g2.mpmct1`, asserted equal by
`regen.sh`), `rep.mpmct1`,
`rep.bounds` (the repaired circuit and lattice), `deliverable.bin` (DE body),
`deliver_line1.txt` (DE line 1, time normalized), `deliver_match.txt`,
`anf_rep.txt`, `anf_b64.txt`, `anf_b256.txt` (AC stdout), `provenance.txt`,
`container_header.bin` (the 48-byte header from `container_header.py`).
The packed circuit is not committed (the packer fixtures store its digest).

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

## `pack/`: the packer `window_pack3.py` (P3)

`pack/cases.txt` lists `name A B argv...` (`OUT` = the output circuit; paths
relative to this directory). Inputs:

- the real n=8 pair: `cview_n8/g2.mpmct1` (G2) and `n8_L8/g3.mpmct1` (the
  stage-3 G3 of the same calibration run, 37,479 gates on 32 wires), at the
  production argv `512 512 24576 3000` and at five crafted ones (WP's own
  defaults with no output file; small caps; a 40-gate span cap; an 8-gate
  interval cap; a 16-gate window cap);
- synthetic pairs from `pack_cases.py` (deterministic, seeds fixed): `wide300`
  (300 wires, W = 8, data/mid/band classes), `wide128` (A on 128 wires, B on
  160 with ancilla copy/uncopy pairs, so B is wider than A), `dense16` (16
  wires, 2-4 controls: exploding window ANFs, so moderate budgets force
  compose aborts) and `w1024` (1024 wires, W = 16).

Expected (`pack/expected/`), all from the unmodified oracle in the sandbox:
`<name>.txt` (stdout), `<name>.bounds`, and `<name>.digest`: blake2b-128 of
the packed circuit and of the emitted order (WP's `PACK_DUMP_ORDER` file).
`regen.sh pack` asserts coverage from the oracle logs and the Rust counters
(`pack/coverage.txt`, from `tdp_packing pack --stats`): compose aborts (7 in
all, 4 order-dependent: the CPython 3.9 table order decides them), a
duplicate boundary and an empty first window (`dense16_dup`), next-next-writer
rescues, long-span and big-interval disqualifications, the mid and band
classes, all three death causes and a case with none, B wider than A, and a
W = 16 state. `tests/integration/tdp_packing.rs` (`pack_matches_oracle`)
compares stdout, bounds and both digests for every case.
`tests/manual/stage4/pack_parity.sh` compares the 8 archived real packs
against Appendix B.

## `repair/`: the repair `repair.sh` (P4)

`repair/cases.txt` lists `name packed bounds C n split_budget fix fallback`
(paths relative to this directory). Inputs:

- `n8_L8` (`@n8`): the real n=8 pack, i.e. `window_pack3.py` at the
  production argv on `cview_n8/g2.mpmct1` + `n8_L8/g3.mpmct1` (in the tests
  it is made by `tdp_packing pack`, whose output is the oracle's), with the
  calibration C `n8_L8/c.g57`: 3 cancel passes (1,327 pairs), floor met at
  fix iteration 1, one clean exposure-aware split. Its expected circuit is
  `n8_L8/rep.mpmct1` (asserted equal by `regen.sh`).
- synthetic cases from `repair_cases.py` (deterministic, seeds fixed): a
  random g57 C on 8 wires run verbatim on the data wires of a 32-wire P,
  with noise gates on the ancillas (some reading data wires, which blocks
  slides) and inserted identical pairs, some nested; boundaries every few
  gates. `leaky` (split budget 64), `leaky_it23` (the same inputs, budget 256,
  2 fix and 3 fallback iterations), `dense` (64), `sparse` (256): fix moves
  in all four iterations, the fix loop exhausted, merges, forced splits
  (`WARNING`), clean splits, and runs that end with visible non-output rows
  (the oracle's count test `visible <= n` passes on `sparse` while the
  identity test does not: exit 5 in Rust). `orderdep` (budget 104): the
  `anf_cases.py` product windows written so the cancel pass keeps them; the
  split decisions depend on the CPython 3.9 set order (6 splits; none under
  `canonical`).

Expected (`repair/expected/<name>.*`), all from the unmodified oracle, each
case in its own sandbox with `SPLIT_BUDGET`/`FIX_ITERS`/`FALLBACK_ITERS`
rewritten exactly once: `rep.mpmct1` (synthetic cases), `bounds`,
`exposure.tsv`, `allcuts.tsv`, `wit.tsv`, `cview.log` and `log` (the
`repair.sh` stdout up to its report part, `== <name> rep512 final`), with
the sandbox directory normalized to `<DIR>`. `coverage.txt` holds the Rust
`repair --stats` counters per case plus whether `--anf-cap-semantics
canonical` changes the bounds; `regen.sh repair` asserts the coverage.
`tests/integration/tdp_packing.rs` (`repair_matches_oracle`) compares every
file byte for byte (logs after the `<DIR>` normalization; the oracle's first
`verify_equiv` runs in the background, so the two verify lines are compared
as a set and the other lines in order). The lattice without an interior cut
(the oracle's `hmap_affine` panics on it) is a Rust-only test
(`repair_single_window_case`). `tests/manual/stage4/repair_parity.sh`
compares the 8 archived real repairs against Appendix B.

## `regen.sh`

    HMAP_BASELINE=... ORACLE_PY=/usr/bin/python3 CARGO_TARGET_DIR=... \
    FROZEN_DB_DIR=~/code/local_mixing/frozen_db/frozen_m1_m11 \
    WORK=<durable scratch> bash regen.sh [n8_L8] [anf] [container] [pack] [repair]

A full rerun of `regen.sh n8_L8 anf container` (2026-10-08) reproduced every
committed file byte for byte (gen, mixer and the oracle chain are
deterministic for fixed seeds); only sandbox paths, now normalized to
`<SANDBOX>` in `provenance.txt`, differed.
