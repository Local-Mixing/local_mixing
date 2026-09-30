#!/usr/bin/env python3
"""Smoke tests for the collision birthday hash and a tiny SAT round-trip."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=ROOT, **kwargs
    )


class BirthdayCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Ensure release binaries exist; build only what we need.
        run(
            [
                "cargo",
                "build",
                "--release",
                "--features",
                "security-tools",
                "--bin",
                "gen_collision_circuit",
                "--bin",
                "birthday_collision",
                "--bin",
                "rho_collision",
            ]
        )

    def test_finds_collision_on_small_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "c"
            run(
                [
                    str(BIN / "gen_collision_circuit"),
                    str(prefix),
                    "12",
                    "40",
                    "7",
                ]
            )
            # 8-bit message, 4-bit digest → birthday ~ 2^2
            result = run(
                [
                    str(BIN / "birthday_collision"),
                    f"{prefix}.g57",
                    "--pad",
                    "4",
                    "--in-bits",
                    "8",
                    "--out-bits",
                    "4",
                    "--samples",
                    "5000",
                    "--seed",
                    "3",
                    "--out",
                    str(Path(tmp) / "report.json"),
                ]
            )
            report = json.loads(result.stdout)
            self.assertNotEqual(report["x1_hex"], report["x2_hex"])
            self.assertTrue(report["digest_hex"].startswith("0x"))

    def test_rho_finds_collision_on_small_digest(self) -> None:
        # Use the checked-in 192-wire / 1024-gate fixture with a truncated
        # digest so the search finishes quickly; a 64-gate toy circuit does
        # not mix enough for distinguished-point search to be reliable.
        fixture = (
            ROOT
            / "security_tests"
            / "collision"
            / "fixtures"
            / "c192_g1024.g57"
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run(
                [
                    str(BIN / "rho_collision"),
                    str(fixture),
                    "--pad",
                    "64",
                    "--out-bits",
                    "28",
                    "--dp-bits",
                    "10",
                    "--seed",
                    "5",
                    "--max-evals",
                    "20000000",
                    "--self-check",
                    "--out",
                    str(Path(tmp) / "rho.json"),
                ]
            )
            report = json.loads(result.stdout)
            self.assertNotEqual(report["x1_hex"], report["x2_hex"])
            self.assertEqual(report["out_bits"], 28)


class CollisionCnfTests(unittest.TestCase):
    def test_encoder_analyze_only(self) -> None:
        encoder = ROOT / "target" / "security-demo" / "collision_to_cnf"
        encoder.parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / "security_tests" / "collision" / "collision_to_cnf.cpp"
        subprocess.run(
            ["g++", "-std=c++17", "-O3", str(src), "-o", str(encoder)],
            check=True,
            cwd=ROOT,
        )
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "c"
            run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--features",
                    "security-tools",
                    "--bin",
                    "gen_collision_circuit",
                    "--",
                    str(prefix),
                    "12",
                    "20",
                    "1",
                ]
            )
            # analyze-only should succeed
            proc = subprocess.run(
                [
                    str(encoder),
                    f"{prefix}.mpmct1",
                    str(Path(tmp) / "unused.cnf"),
                    "--in-bits",
                    "8",
                    "--pad",
                    "4",
                    "--out-bits",
                    "4",
                    "--analyze-only",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("vars=", proc.stderr)


if __name__ == "__main__":
    unittest.main()
