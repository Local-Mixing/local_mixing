"""The standalone container v1 writer (tests/manual/stage4/container_header.py)
against the committed stage-4 fixtures, and the CPython 3.9 set-order
self-test when the interpreter is a 3.9 (docs/STAGE4_RUST_DESIGN.md §6, §10.1)."""

import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[3]
FX = ROOT / "tests" / "fixtures" / "stage4"


def load_writer():
    path = ROOT / "tests" / "manual" / "stage4" / "container_header.py"
    spec = importlib.util.spec_from_file_location("container_header", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ContainerHeaderTest(unittest.TestCase):
    def test_fields_and_crcs(self):
        ch = load_writer()
        body = (FX / "n8_L8" / "deliverable.bin").read_bytes()
        h = ch.header(body, 8, 32)
        self.assertEqual(len(h), 48)
        self.assertEqual(h[:8], b"LMTDPANF")
        version, size, flags, n, state, layout, windows, nbytes, crc = struct.unpack_from(
            "<HHIIIIIQI", h, 8
        )
        self.assertEqual((version, size, flags, n, state, layout), (1, 48, 0, 8, 32, 1))
        self.assertEqual(windows, 73)
        self.assertEqual(nbytes, len(body))
        self.assertEqual(crc, zlib.crc32(body))
        self.assertEqual(struct.unpack_from("<I", h, 44)[0], zlib.crc32(h[:44]))
        self.assertEqual(h, (FX / "n8_L8" / "container_header.bin").read_bytes())

    def test_crafted_fixture_headers(self):
        ch = load_writer()
        for line in (FX / "anf" / "cases.txt").read_text().splitlines():
            name = line.split()[0]
            state = int((FX / "anf" / f"{name}.mpmct1").read_text().split()[1])
            body = (FX / "anf" / f"{name}.bin").read_bytes()
            self.assertEqual(
                ch.header(body, 1, state), (FX / "anf" / f"{name}.tdp4hdr").read_bytes(), name
            )

    def test_truncated_body_is_refused(self):
        ch = load_writer()
        body = (FX / "anf" / "cancel.bin").read_bytes()
        with self.assertRaises(SystemExit):
            ch.window_count(body[:-1])


@unittest.skipUnless(sys.version_info[:2] == (3, 9), "the set-order traces are CPython 3.9")
class SetOrderSelfTest(unittest.TestCase):
    def test_traces_replay(self):
        r = subprocess.run(
            [sys.executable, "-I", str(FX / "pyset_selftest.py"),
             str(FX / "pyset39_traces.txt"), str(FX / "pyset39_large.txt")],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
