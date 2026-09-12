#!/usr/bin/env bash
# Compatibility launcher; maintained implementation lives with benchmarks.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/benchmarks/bench_pipeline.sh" "$@"
