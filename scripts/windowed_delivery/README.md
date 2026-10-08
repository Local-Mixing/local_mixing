# Windowed ANF delivery: research prototypes (snapshot 2026-10-07)

Python/bash prototypes behind docs/ANF_WINDOW_DELIVERY.md and the
stage-2 linkability program (lab notebook: docs/notes/ncdb_ab_20261001_RESULTS.md).
Snapshot of `runs/ncdb_ab_20261001/` (top level) and its `waist/` folder.
They are the **reference oracle for the Rust port**; paths inside are
hard-coded to `/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001`
(run them from that layout; the Rust port will remove the hard-coding).
`waist/order_repair.py` and `waist/val_chain.sh` were refreshed from the run
tree on 2026-10-08 (the repair slide now locates the breaker before moving
the completer). How to run stage 4: `docs/tdp_pipeline.md`, "Stage 4: packing".

Delivery path (post stage 3):
- `window_pack3.py`  joint window packer (ANF-budget-adaptive)
- `anf_compose.py`   per-window sparse ANF + spot verification
- `waist/order_repair.py`  commutation-aware repair (pair cancel, leak slides)
- `waist/merge_bounds2.py`, `waist/anf_split2.py`  merge / exposure-aware split fallback
- `waist/repair.sh`  repair driver; `waist/finalize3.sh` merge-only finalize
- `waist/verify_equiv.py`  functional equivalence check

Censuses / attacks: `boundary_cluster.py`, `waist/link_census.py` (linkable
values, sightings, froth), `waist/mult_census.py`, `wire_balance.py`, plus
the diagnostic scripts in `waist/` (leak tracing, C structure, births).
Drivers: `waist/val_chain.sh` (validation run of the four-stage pipeline:
gen -> stage-3 mix -> pack G=512 + `repair.sh` -> serialized deliverable),
`waist/run_ret3.sh` (full chain: gen -> mix -> pack -> scans),
`waist/score_arm.sh`, `waist/auto_score2.sh`. The drivers call
`circuit_mixer` with the stage-3 flags (`--p-db 1.0`, regular frozen store);
the retired `--p-comp`/`--p-any` flags are gone (they acted only in the
historic crossing walk, branch `ran/archive/curated-and-postmix`).
