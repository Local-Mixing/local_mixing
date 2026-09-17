"""Check full-width G57 counts, malformed input, and the plot command."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from security_tests.heatmaps import wireplot


class WireplotTest(unittest.TestCase):
    def test_counts_at_alphabet_overflow_and_u16_boundaries(self):
        # 1023 = 12*83 + 27 ('r'); 65535 = 789*83 + 48 ('M').
        wire1023 = "~" * 12 + "r"
        wire65535 = "~" * 789 + "M"
        circuit = f"#?~0;~0{wire1023}{wire65535};{wire65535}#{wire1023};"
        total, targets = wireplot.count_wire_usage(circuit, 65536)
        self.assertEqual(int(total.sum()), 9)
        self.assertEqual(int(targets.sum()), 3)
        for wire, count, target in ((64, 2, 1), (82, 1, 0), (83, 2, 1),
                                    (1023, 2, 0), (65535, 2, 1)):
            with self.subTest(wire=wire):
                self.assertEqual(total[wire], count)
                self.assertEqual(targets[wire], target)
        self.assertEqual(int(np.count_nonzero(total)), 5)

    def test_every_base83_digit_matches_canonical_alphabet(self):
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
        circuit = "".join(f"{digit}01;" for digit in alphabet)
        total, targets = wireplot.count_wire_usage(circuit, 83)
        np.testing.assert_array_equal(targets, np.ones(83, dtype=int))
        expected = np.ones(83, dtype=int)
        expected[:2] += 83
        np.testing.assert_array_equal(total, expected)

    def test_canonical_empty_segments_surrounding_whitespace_and_final_gate(self):
        total, targets = wireplot.count_wire_usage(" \t\n\r\f;;012;;210\r\n ", 3)
        np.testing.assert_array_equal(total, [2, 2, 2])
        np.testing.assert_array_equal(targets, [1, 0, 1])
        total, targets = wireplot.count_wire_usage(" \t;;;\n", 1)
        np.testing.assert_array_equal(total, [0])
        np.testing.assert_array_equal(targets, [0])

    def test_malformed_input_is_not_silently_skipped_or_truncated(self):
        for circuit in ("0;", "01;", "0123;", "01~;", "012~", "~;",
                        "01/;", "01é;", "01 2;", "012;\n210;", "~ 012;",
                        "g57 3 1\n012;", "mpmct1 3 1\n0 0 2 1 1 2 1"):
            with self.subTest(circuit=circuit), self.assertRaises(ValueError):
                wireplot.count_wire_usage(circuit, 65536)

    def test_declared_and_physical_wire_limits(self):
        for count in (0, -1, 65537):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "wire count"):
                wireplot.count_wire_usage("012;", count)
        for circuit, count in (("012;", 2), ("~001;", 83),
                               ("~" * 789 + "N01;", 65536),
                               ("~" * 790 + "001;", 65536)):
            with self.subTest(circuit=circuit, count=count), self.assertRaises(ValueError):
                wireplot.count_wire_usage(circuit, count)

    def test_bad_cli_width_fails_before_reading_input(self):
        for count in ("0", "-1", "65537", "3.5"):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    wireplot.main(["--c", "missing.g57", "--n", count, "--x", "bad"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("wire count", stderr.getvalue())
                self.assertNotIn("No such file", stderr.getvalue())

    def test_cli_renders_highest_wire_to_requested_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            circuit = directory / "wide.g57"
            circuit.write_text("~" * 789 + "M01;", encoding="ascii")
            output = directory / "wide.png"
            environment = dict(os.environ, MPLCONFIGDIR=str(directory / "mpl"))
            result = subprocess.run(
                [sys.executable, "-B", str(Path(wireplot.__file__).resolve()),
                 "--c", str(circuit), "--n", "65536", "--x", "wide", "--out", str(output)],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertFalse((directory / "wire_scatter.png").exists())


if __name__ == "__main__":
    unittest.main()
