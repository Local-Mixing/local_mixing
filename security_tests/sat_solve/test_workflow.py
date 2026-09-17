"""Small circuit/CNF/model checks; run with python -m unittest security_tests.sat_solve.test_workflow."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from security_tests.sat_solve import decode_model, solve_input_cubes


HERE = Path(__file__).resolve().parent


def read_cnf(path: Path) -> tuple[int, list[list[int]]]:
    variables, expected = 0, 0
    clauses = []
    for line in path.read_text().splitlines():
        if line.startswith("p "):
            _, _, variables, expected = line.split()
            variables, expected = int(variables), int(expected)
        elif line and not line.startswith("c "):
            values = [int(word) for word in line.split()]
            if values[-1] != 0:
                raise ValueError("clause missing terminator")
            clauses.append(values[:-1])
    if len(clauses) != expected:
        raise ValueError("incorrect clause count")
    return variables, clauses


def satisfies(clauses: list[list[int]], assignment: list[bool]) -> bool:
    return all(
        any(assignment[abs(literal) - 1] == (literal > 0) for literal in clause)
        for clause in clauses
    )


class SatWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for the CNF encoder")
        cls.build = tempfile.TemporaryDirectory(prefix="tdp-sat-build-")
        cls.encoder = Path(cls.build.name) / "zero_slice_to_cnf"
        subprocess.run(
            [compiler, "-std=c++17", "-O1", str(HERE / "zero_slice_to_cnf.cpp"), "-o", str(cls.encoder)],
            check=True, capture_output=True, text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tdp-sat-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def encode(self, circuit: Path, target: int, n: int = 3, output_start: int = 3):
        cnf = self.directory / "problem.cnf"
        subprocess.run(
            [str(self.encoder), str(circuit), str(cnf), str(n), hex(target), "--output-start", str(output_start)],
            check=True, capture_output=True, text=True,
        )
        return read_cnf(cnf)

    def test_gate_cnf_matches_exhaustive_truth_tables(self):
        # Every polarity/complement combination, including empty products.
        # Exhausting all CNF assignments also catches unwanted extra models.
        circuit = self.directory / "single_gate.mpmct1"
        for width in range(3):
            for complemented in range(2):
                for polarities in itertools.product((0, 1), repeat=width):
                    controls = " ".join(f"{wire} {polarity}" for wire, polarity in enumerate(polarities, 1))
                    circuit.write_text(f"mpmct1 3 1\n0 {complemented} {width} {controls}\n")
                    for target in range(8):
                        variables, clauses = self.encode(circuit, target, output_start=0)
                        self.assertEqual(variables, 4)
                        for bits in itertools.product((False, True), repeat=4):
                            input_value = sum(int(bit) << index for index, bit in enumerate(bits[:3]))
                            output, _, _ = decode_model.evaluate_mpmct1(circuit, input_value)
                            expected = output == target and bits[3] == bool(output & 1)
                            self.assertEqual(satisfies(clauses, list(bits)), expected)

    def test_arbitrary_width_and_model_verification(self):
        # Nine wires exercises a width that differs from the 4*n layout.
        circuit = self.directory / "wider.mpmct1"
        circuit.write_text((HERE / "fixtures/zero_slice.mpmct1").read_text().replace("mpmct1 12", "mpmct1 9", 1))
        full_output, _, _ = decode_model.evaluate_mpmct1(circuit, 5)
        target = (full_output >> 3) & 7
        variables, _ = self.encode(circuit, target)
        self.assertEqual(variables, 15)
        model = self.directory / "solver.log"
        initial = [str(wire + 1 if (5 >> wire) & 1 else -(wire + 1)) for wire in range(9)]
        model.write_text("s SATISFIABLE\nv " + " ".join(initial) + " 0\n")
        report = self.directory / "verified.json"
        result = subprocess.run(
            [sys.executable, str(HERE / "decode_model.py"), "--circuit", str(circuit), "--solver-output", str(model),
             "--n", "3", "--output-start", "3", "--target-hex", hex(target), "--out", str(report)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(report.read_text())["verified"])
        self.assertEqual(json.loads(report.read_text())["logical_input_hex"], "0x5")

    def test_invalid_output_range_is_rejected_without_cnf(self):
        cnf = self.directory / "invalid.cnf"
        result = subprocess.run(
            [str(self.encoder), str(HERE / "fixtures/zero_slice.mpmct1"), str(cnf), "3", "0x1", "--output-start", "10"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("range exceeds", result.stderr)
        self.assertFalse(cnf.exists())

    def test_decoder_rejects_missing_or_contradictory_initial_bits(self):
        model = self.directory / "bad.log"
        model.write_text("s SATISFIABLE\nv 1 -1 2 0\n")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            decode_model.read_initial_model(model, 2)
        model.write_text("s SATISFIABLE\nv 1 0\n")
        with self.assertRaisesRegex(ValueError, "omits"):
            decode_model.read_initial_model(model, 2)

    def test_solver_status_checks_both_status_and_exit_code(self):
        self.assertEqual(solve_input_cubes.solver_status("s SATISFIABLE\n", 10), "SAT")
        self.assertEqual(solve_input_cubes.solver_status("s UNSATISFIABLE\n", 20), "UNSAT")
        self.assertEqual(solve_input_cubes.solver_status("s UNKNOWN\n", 0), "UNKNOWN")
        self.assertEqual(solve_input_cubes.solver_status("s SATISFIABLE\n", 1), "ERROR")

    def test_cube_exhaustion_is_inconclusive(self):
        base = self.directory / "base.cnf"
        base.write_text("p cnf 3 1\n1 2 3 0\n")
        output = self.directory / "cubes"

        def unsat(command, *, stdout, stderr):
            stdout.write("s UNSATISFIABLE\n")
            return subprocess.CompletedProcess(command, 20)

        transcript = io.StringIO()
        with patch.object(solve_input_cubes.shutil, "which", return_value="/external/kissat"), \
             patch.object(solve_input_cubes.subprocess, "run", side_effect=unsat), \
             contextlib.redirect_stdout(transcript):
            result = solve_input_cubes.main([
                "--base-cnf", str(base), "--out-dir", str(output), "--var-count", "3",
                "--fixed-bits", "1", "--cubes", "2",
            ])
        self.assertEqual(result, 1)
        self.assertIn("INCONCLUSIVE", transcript.getvalue())
        self.assertEqual(len(list(output.glob("*.cnf"))), 2)
        for cube in output.glob("*.cnf"):
            _, clauses = read_cnf(cube)
            self.assertEqual(len(clauses), 2)
            self.assertEqual(len(clauses[1]), 1)


if __name__ == "__main__":
    unittest.main()
