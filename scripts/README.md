# Repository scripts

- `gss_mix.sh`: low-level GSS driver. Normal use goes through the configuration-aware `gss` command. New recipes use quadratic masking or nonlinear291.
- `compat/gss_mix_v3.sh` through `compat/gss_mix_v6.sh`: immutable drivers for existing managed runs. Their hashes, recorded binaries and manifests remain unchanged. New runs use recipe v7.
- `golden_check.sh`: optional correctness baselines; historical arms use `legacy_mixing`.
- `bench_pipeline.sh`, `compare_canon_bench.py`: compatibility launchers delegating to `benchmarks/`.
- `tex2md.py`: convert design documents from TeX to Markdown.

Database maintenance lives in `db_gen/maintenance/`; security campaigns and their launchers live in `security_tests/`. Archived deployments and report/config artifacts keep their paths.
