"""Serialize the delivered ANF to the binary encoding the size estimate
assumes, measure compressibility, and time evaluation (gate form vs ANF
form, batched 1024 lanes and single input)."""
import sys, time, random, struct
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from anf_compose import load, window_anf

packed, bounds_path, out_bin, label = sys.argv[1:5]
n, gates = load(packed)
total = int(open(packed).readline().split()[2])
bounds = [int(x) for x in open(bounds_path) if x.strip()]
cuts = [0] + sorted(b for b in bounds if 0 < b < total) + [total]

# --- serialize ---
t0 = time.time()
blobs = []
windows = []
for wi in range(len(cuts) - 1):
    lo, hi = cuts[wi], cuts[wi + 1]
    changed, tot = window_anf(gates, lo, hi, 1 << 30)
    wires = set()
    for p in changed.values():
        for m in p:
            mm = m
            while mm:
                low = mm & -mm
                wires.add(low.bit_length() - 1)
                mm ^= low
    wl = sorted(wires)
    idx = {w: i for i, w in enumerate(wl)}
    f = max(len(wl), 1)
    fb = (f + 7) // 8
    b = bytearray()
    b += struct.pack('<HH', len(wl), len(changed))
    for w in wl:
        b += struct.pack('<H', w)
    for w, poly in sorted(changed.items()):
        b += struct.pack('<HI', w, len(poly))
        for m in sorted(poly):
            local = 0
            mm = m
            while mm:
                low = mm & -mm
                local |= 1 << idx[low.bit_length() - 1]
                mm ^= low
            b += local.to_bytes(fb, 'little')
    blobs.append(bytes(b))
    windows.append((wl, {w: sorted(p) for w, p in changed.items()}))
body = b''.join(struct.pack('<I', len(x)) + x for x in blobs)
open(out_bin, 'wb').write(body)
print(f"{label}: serialized {len(body)/1e6:.1f} MB in {time.time()-t0:.0f}s, {len(blobs)} windows")

# --- evaluation timing ---
rng = random.Random(1)
L = 1024
lanes = [rng.getrandbits(L) for _ in range(n)]
M = (1 << L) - 1
# gate form, batched
st = list(lanes)
t0 = time.time()
for t, comp, cps, pol in gates:
    acc = M
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ M
    if comp:
        acc ^= M
    st[t] ^= acc
tg = time.time() - t0
# ANF form, batched
st2 = list(lanes)
t0 = time.time()
for wl, changed in windows:
    cur = {w: st2[w] for w in wl}
    for w, poly in changed.items():
        acc = 0
        for m in poly:
            term = M
            mm = m
            while mm:
                low = mm & -mm
                term &= cur[low.bit_length() - 1]
                if not term:
                    break
                mm ^= low
            acc ^= term
        st2[w] ^= acc ^ cur[w] ^ cur[w]  # placeholder; real update below
    # correct update: new value = polynomial over INPUT state of window
    for w, poly in changed.items():
        pass
ta_wrong = time.time() - t0
# redo ANF eval correctly (window outputs computed from window-input snapshot)
st2 = list(lanes)
t0 = time.time()
for wl, changed in windows:
    snap = {w: st2[w] for w in wl}
    for w, poly in changed.items():
        acc = 0
        for m in poly:
            term = M
            mm = m
            while mm:
                low = mm & -mm
                term &= snap[low.bit_length() - 1]
                if not term:
                    break
                mm ^= low
            acc ^= term
        st2[w] = acc
ta = time.time() - t0
ok = st == st2
# single input, gate form (ints of 1 bit ~ use 64-bit single word)
x = [rng.getrandbits(1) for _ in range(n)]
st3 = list(x)
t0 = time.time()
for t, comp, cps, pol in gates:
    acc = 1
    for c, p in zip(cps, pol):
        acc &= st3[c] if p else st3[c] ^ 1
    if comp:
        acc ^= 1
    st3[t] ^= acc
ts = time.time() - t0
print(f"EVAL gate-form batched({L} lanes): {tg:.1f}s = {tg/L*1e3:.2f} ms/input")
print(f"EVAL ANF-form  batched({L} lanes): {ta:.1f}s = {ta/L*1e3:.2f} ms/input  (match={ok})")
print(f"EVAL gate-form single input (python): {ts*1e3:.0f} ms")
