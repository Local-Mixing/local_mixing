Scripts behind the report (copied 2026-10-28; the session scratchpad that held the local ones was wiped by the Mac's temp cleanup).

- `local/expo_model.py`: exposure vs read structure on RC's audited per-birth data (sections 3-4 of the report).
- `local/design_sim.py`: candidate C generators, model predictions, avalanche/degree proxies (section 5); writes g57 triples with `write_g57`.
- `server/py/pipeline.py`: stage 2-5 driver for the C-design arms (`~/cdesign` on machine1); `patch_gen.py` adds `--source-file` to the three-mask generator.
- `server/py/census_by_fanout.py`, `locations.py`, `fire_windows.py`, `narrow_fanout.py`, `narrow_gates.py`, `resil_summary.py`, `report_cdesign.py`: the measurements of sections 6-7.
- `server/py6/cdesign_launch.sh`, `cdesign_loops.sh`, `early_narrow.py`: launchers (stage 6, attacks, resilience) and the early narrow portfolio.
- `server/sources/bal.g57.txt`, `burst67.g57.txt`: the balanced and burst C's as g57 triples (a x y: a ^= x OR NOT y).
