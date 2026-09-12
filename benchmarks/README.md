# Benchmarks and probes

Build registered benchmark executables with `--features benchmark-tools` and
select a target with `--bin NAME`. These tools are outside the ordinary GSS
build. They retain their measurement algorithms and existing artifact paths.

- `canonicalization/`: `canon_probe`, `bench_canon4`, `bench_polycanon`, and
  their shared input-generation helper.
- `circuit/`: `bench_eval` circuit-evaluation measurements.
- `mixing/`: `fmix_stats`, commuting/float/mobility probes, the retained DMR
  scratch probe and `bench_stats_20260715.py` report aggregation.
- `db_mixing/`: DB-hit, candidate, degree, window-span and far-pair probes.

`benchmarks/bench_pipeline.sh` and `benchmarks/canonicalization/compare.py` drive the
registered canonicalization/evaluation measurements. `benchmark-tools` retains
optional comparison support and the SHA implementation required by the graph
canonicalizer. Cargo's explicit binary table is the executable registry; a
scratch source is not automatically a supported command.
