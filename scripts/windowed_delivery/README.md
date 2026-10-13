# Windowed ANF delivery: the Python/bash ORACLE of the packer (snapshot 2026-10-07)

The packer is the packing sub-step of stage 6 (delivery); it was stage 4 in the
four-stage numbering, which the directory `tests/fixtures/stage4` and the test
names keep.

Python/bash prototypes behind docs/ANF_WINDOW_DELIVERY.md and the
stage-2 linkability program (lab notebook: docs/notes/ncdb_ab_20261001_RESULTS.md).
Snapshot of `runs/ncdb_ab_20261001/` (top level) and its `waist/` folder.
`waist/order_repair.py` and `waist/val_chain.sh` were refreshed from the run
tree on 2026-10-08 (the repair slide now locates the breaker before moving
the completer; `val_chain.sh` drops the inert `--p-comp/--p-any`).

**Role.** The packing sub-step of stage 6 (delivery) runs in Rust (`tdp_packing
run`, `src/stages/packing/`; docs/STAGE4_RUST_DESIGN.md). These scripts are its
**canonical oracle**: the port is bit-exact wherever they are deterministic.
They are **never edited** (the one exception is `waist/deliver.py`, below).

**The sandbox rule.** The scripts hard-code `/Users/rancanetti/code/local_mixing`
and the run tree `runs/ncdb_ab_20261001/`. Every run of the oracle (fixtures,
CI, parity, reproduction) uses a path-rewritten sandbox copy made by
`tests/fixtures/stage4/oracle_sandbox.sh`, which asserts that each rewrite
applies exactly as often as expected (and can rewrite the split budget and
the iteration counts of `repair.sh` for small cases); the reproduction test
does the same for `waist/val_chain.sh`.

**The interpreter.** The ANF caps abort on a prefix of a set's iteration
order, so the oracle must run under an interpreter with CPython 3.9 set
semantics: `tests/fixtures/stage4/pyset_selftest.py` must pass
(`/usr/bin/python3` 3.9.6 on macOS; `actions/setup-python` 3.9 in CI).

**hmap_affine.** `repair.sh` and `anf_split2.py` call `hmap_affine --exposure`.
The oracle side always uses the baseline built from `492564f0`
(sha256 pinned in `tests/fixtures/stage4/cview_n8/baseline_sha256.txt`), never
a binary built from the tree under test.

How to run the packer: `docs/tdp_pipeline.md`, "Stage 6: delivery", "Packing
(the first sub-step)".

Delivery path (post stage 3), with the Rust counterpart:
- `window_pack3.py`  joint window packer (ANF-budget-adaptive) -> `tdp_packing pack`
- `anf_compose.py`   per-window sparse ANF + spot verification -> `tdp_packing anf-report`
- `waist/order_repair.py`  commutation-aware repair (pair cancel, leak slides; part of every packing run) -> `stages::packing::repair`
- `waist/merge_bounds2.py`, `waist/anf_split2.py`  merge / exposure-aware split fallback
- `waist/repair.sh`  repair driver -> `tdp_packing repair`; `waist/finalize3.sh` merge-only finalize
- `waist/verify_equiv.py`  functional equivalence check -> `tdp_packing verify-equiv`
- `waist/anf_dump_eval.py`  the deliverable body -> `tdp_packing serialize`
- `waist/link_census.py`, `waist/mult_census.py` -> `tdp_packing census`, `tdp_packing mult-census`
- `waist/deliver.py` (2026-10-12, not part of the 2026-10-07 snapshot): the
  reference of the delivery loop, packing + unpacking with lattice refinement
  -> `tdp_unpacking run --refine`; the oracle of `tests/manual/delivery_parity.sh`
  (`docs/DELIVERY_STAGE.md` §3). It is the one script here that is maintained,
  not frozen.

Censuses / attacks: `boundary_cluster.py`, `waist/link_census.py` (linkable
values, sightings, froth), `waist/mult_census.py`, `wire_balance.py`, plus
the diagnostic scripts in `waist/` (leak tracing, C structure, births).
Drivers: `waist/val_chain.sh` (the reference driver of the four-stage chain,
today `--run-skip-stages 4,5 --delivery-anf-only` in the managed driver:
gen -> stage-3 mix -> pack G=512 + `repair.sh` -> serialized deliverable; now
a parity tool, see tests/manual/stage4/reproduce_val_chain.sh),
`waist/run_ret3.sh` (full chain: gen -> mix -> pack -> scans),
`waist/score_arm.sh`, `waist/auto_score2.sh`. The drivers call
`circuit_mixer` with the stage-3 flags (`--p-db 1.0`, regular frozen store);
the inert `--p-comp`/`--p-any` flags were dropped from the stage-3 command on
2026-10-08 (they act only in the crossing walk, stage 5; stage 3's `--p-db 1.0`
never runs it).
