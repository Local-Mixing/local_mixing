# Windowed ANF delivery: research prototypes (snapshot 2026-10-07)

Python/bash prototypes behind docs/ANF_WINDOW_DELIVERY.md and the
stage-2 linkability program (lab notebook: docs/notes/ncdb_ab_20261001_RESULTS.md).
Snapshot of `runs/ncdb_ab_20261001/` (top level) and its `waist/` folder.
They are the **reference oracle for the Rust port**; paths inside are
hard-coded to `/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001`
and will be made location-independent in the cleanup.

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
Drivers: `run_stage3.sh`, `waist/run_ret3.sh` (full chain: gen -> mix ->
pack -> scans), `waist/score_arm.sh`, `waist/auto_score2.sh`.
