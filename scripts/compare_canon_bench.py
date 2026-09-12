#!/usr/bin/env python3
"""Compatibility launcher for the canonicalization benchmark report."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/canonicalization/compare.py"), run_name="__main__")
