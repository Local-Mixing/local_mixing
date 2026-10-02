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


def evaluate_digest(circuit: Path, wires: int, message_hex: str, out_bits: int) -> int:
    """Independently re-evaluate C via `circuit evaluate` and take low out_bits."""
    result = run(
        [
            str(BIN / "local_mixing_bin"),
            "circuit",
            "evaluate",
            "-n",
            str(wires),
            "-s",
            str(circuit),
            "--input",
            message_hex,
        ]
    )
    # Output line ends with "(0x....)" covering all wires, little-endian hex.
    out_line = next(
        line for line in result.stdout.splitlines() if line.startswith("Output:")
    )
    hex_part = out_line[out_line.rfind("0x") :].strip().rstrip(")")
    value = int(hex_part, 16)
    return value & ((1 << out_bits) - 1)


def assert_valid_collision(
    test: unittest.TestCase,
    circuit: Path,
    wires: int,
    x1_hex: str,
    x2_hex: str,
    digest_hex: str,
    out_bits: int,
) -> None:
    test.assertNotEqual(x1_hex.lower(), x2_hex.lower())
    d1 = evaluate_digest(circuit, wires, x1_hex, out_bits)
    d2 = evaluate_digest(circuit, wires, x2_hex, out_bits)
    expected = int(digest_hex, 16) & ((1 << out_bits) - 1)
    test.assertEqual(d1, d2, f"digests differ: {d1:#x} vs {d2:#x}")
    test.assertEqual(d1, expected, f"digest {d1:#x} != reported {expected:#x}")


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
        # Main CLI for independent evaluate checks.
        run(["cargo", "build", "--release", "--locked"])

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
            assert_valid_collision(
                self,
                Path(f"{prefix}.g57"),
                wires=12,
                x1_hex=report["x1_hex"],
                x2_hex=report["x2_hex"],
                digest_hex=report["digest_hex"],
                out_bits=4,
            )

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
            self.assertEqual(report["out_bits"], 28)
            assert_valid_collision(
                self,
                fixture,
                wires=192,
                x1_hex=report["x1_hex"],
                x2_hex=report["x2_hex"],
                digest_hex=report["digest_hex"],
                out_bits=28,
            )

    def test_fixture_witnesses_are_valid(self) -> None:
        fixtures = ROOT / "security_tests" / "collision" / "fixtures"
        birthday = json.loads((fixtures / "c96_g1024.birthday.json").read_text())
        assert_valid_collision(
            self,
            fixtures / "c96_g1024.g57",
            wires=96,
            x1_hex=birthday["x1_hex"],
            x2_hex=birthday["x2_hex"],
            digest_hex=birthday["digest_hex"],
            out_bits=32,
        )
        rho = json.loads((fixtures / "c192_g1024.rho.json").read_text())
        assert_valid_collision(
            self,
            fixtures / "c192_g1024.g57",
            wires=192,
            x1_hex=rho["x1_hex"],
            x2_hex=rho["x2_hex"],
            digest_hex=rho["digest_hex"],
            out_bits=64,
        )


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
