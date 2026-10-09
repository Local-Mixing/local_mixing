"""Standalone container v1 writer (docs/STAGE4_RUST_DESIGN.md §6).

Wraps an UNCHANGED anf_dump_eval.py body (deliverable.bin) in the 48-byte
`LMTDPANF` header, independently of the Rust implementation, so parity runs
can rebuild deliverable.tdp4 from the oracle's body and `cmp` it with
`tdp_packing serialize --container`. The oracle itself is never edited.

usage: python3 -I container_header.py BODY OUT SOURCE_WIRES STATE_WIRES
"""
import struct
import sys
import zlib

MAGIC = b"LMTDPANF"
FORMAT_VERSION = 1
HEADER_BYTES = 48
IO_LAYOUT_CLASSIC_ZERO_SLICE = 1


def window_count(body):
    pos = n = 0
    while pos < len(body):
        (length,) = struct.unpack_from("<I", body, pos)
        pos += 4 + length
        n += 1
    if pos != len(body):
        raise SystemExit("body: last blob runs past the end")
    return n


def header(body, source_wires, state_wires):
    h = MAGIC + struct.pack(
        "<HHIIIIIQI",
        FORMAT_VERSION,
        HEADER_BYTES,
        0,  # flags
        source_wires,
        state_wires,
        IO_LAYOUT_CLASSIC_ZERO_SLICE,
        window_count(body),
        len(body),
        zlib.crc32(body) & 0xFFFFFFFF,
    )
    return h + struct.pack("<I", zlib.crc32(h) & 0xFFFFFFFF)


def main():
    body_path, out_path, n, state = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    body = open(body_path, "rb").read()
    h = header(body, n, state)
    assert len(h) == HEADER_BYTES
    with open(out_path, "wb") as f:
        f.write(h + body)


if __name__ == "__main__":
    main()
