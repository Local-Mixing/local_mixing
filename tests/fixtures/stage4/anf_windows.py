"""Per-window decisions of the ORACLE's anf_compose.window_anf (unmodified,
imported from a sandbox copy) under both cap semantics, for the stage-4
fixtures (docs/STAGE4_RUST_DESIGN.md §7.2, §10.3).

usage: ORACLE_PY anf_windows.py RD circuit.mpmct1 bounds budget

One line per window: `wi lo hi <cpython39> <canonical>`, each column
`overflow` or `ok:<total>:<fnv64 digest of the sorted ANF>`. The cpython39
column is window_anf as is. The canonical column runs the same function with
`anf_compose.mul` replaced by one that completes the product and refuses it
only if the finished product exceeds the cap (the Rust `Canonical` mode).
Compared line for line with `tdp_packing window-anf`.
"""
import sys

RD, c_path, b_path, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, RD)
import anf_compose as ac  # noqa: E402

MASK64 = (1 << 64) - 1
FNV_OFF = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

n, gates = ac.load(c_path)
bounds = [int(x) for x in open(b_path) if x.strip()]
m = len(gates)
cuts = [0] + sorted(b for b in bounds if 0 < b < m) + [m]
L = max(1, (n + 63) // 64)
orig_mul = ac.mul


def canonical_mul(Ap, Bp, cap):
    res = orig_mul(Ap, Bp, 1 << 62)
    return None if len(res) > cap else res


def col(changed):
    if changed is None:
        return "overflow"
    h = FNV_OFF
    total = 0
    for w in sorted(changed):
        p = sorted(changed[w])
        total += len(p)
        for x in [w, len(p)] + [(mono >> (64 * i)) & MASK64 for mono in p for i in range(L)]:
            h = ((h ^ x) * FNV_PRIME) & MASK64
    return f"ok:{total}:{h:016x}"


for wi in range(len(cuts) - 1):
    lo, hi = cuts[wi], cuts[wi + 1]
    ac.mul = orig_mul
    a, _ = ac.window_anf(gates, lo, hi, budget)
    ac.mul = canonical_mul
    b, _ = ac.window_anf(gates, lo, hi, budget)
    ac.mul = orig_mul
    print(f"{wi} {lo} {hi} {col(a)} {col(b)}")
