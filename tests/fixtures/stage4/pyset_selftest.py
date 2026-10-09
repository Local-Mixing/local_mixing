"""CPython 3.9 set iteration-order traces (docs/STAGE4_RUST_DESIGN.md §4.4, §10.1).

The stage-4 oracle's ANF caps abort on a PREFIX of a set's iteration order,
which is CPython-version dependent. This file both RECORDS the traces the
Rust emulator (`stages::packing::pyset`) replays and SELF-TESTS an oracle
interpreter against them:

  python3 -I pyset_selftest.py record pyset39_traces.txt pyset39_large.txt
  python3 -I pyset_selftest.py pyset39_traces.txt [pyset39_large.txt]   # exit 0 iff all orders match

Small-trace format, one op per line (keys in hex, sets numbered 0..3):
  N s         s = set()
  L s k       s = {k}
  G s k       toggle: discard k if present else add k
  A s k       s.add(k)
  D s k       s.discard(k)
  C d s       d = set(s)
  X d a b     d = a ^ b
  I a b       a ^= b
  O s k...    expected iteration order of s
  T name      start of a trace (all sets reset to empty)

Large traces are not stored: each line `large seed target checkpoints...`
names a generator below (only `getrandbits(32)` of random.Random(seed)) and
the FNV-1a-64 digests of the iteration order at its checkpoints.
"""
import random
import sys

MASK64 = (1 << 64) - 1
FNV_OFF = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def digest(s, limbs):
    h = FNV_OFF
    for k in s:
        for i in range(limbs):
            h = ((h ^ ((k >> (64 * i)) & MASK64)) * FNV_PRIME) & MASK64
    return h


def large_key(idx):
    # low bits: the index; one high bit in 64..463 (keys >= 2^61, >= 2^128, >= 2^256)
    return idx | (1 << (64 + idx % 400))


def large_trace(seed, target):
    """Toggle loop over a universe of 2*target keys, 3*target steps; then a
    copy and an xor with a literal at each checkpoint. Yields digests."""
    r = random.Random(seed)
    universe = 2 * target
    steps = 3 * target
    every = steps // 4
    s = set()
    out = []
    for step in range(1, steps + 1):
        k = large_key(r.getrandbits(32) % universe)
        if k in s:
            s.discard(k)
        else:
            s.add(k)
        if step % every == 0:
            c = set(s)
            x = c ^ {large_key(step)}
            out.append((len(s), digest(s, 8), digest(c, 8), digest(x, 8)))
    return out


LARGE = [(101, 70000), (102, 140000), (103, 220000)]


def pool_item(r):
    c = r.random()
    if c < 0.3:
        return 1 << r.randrange(300)
    if c < 0.35:
        return 0
    if c < 0.6:
        return r.getrandbits(r.choice([8, 61, 62, 64, 128, 200, 300]))
    k = (1 << r.randrange(300)) | (1 << r.randrange(300))
    if r.random() < 0.3:
        k |= r.getrandbits(64)
    return k


def record_small(path, seed=20261008, trials=50):
    r = random.Random(seed)
    out = []
    for trial in range(trials):
        out.append(f"T t{trial}")
        sets = [set() for _ in range(4)]
        pool = [pool_item(r) for _ in range(r.choice([3, 6, 20, 60, 150]))]
        nsteps = r.choice([12, 40, 150, 400])
        for step in range(nsteps):
            op = r.random()
            a, b, c = r.randrange(4), r.randrange(4), r.randrange(4)
            if op < 0.45:
                k = r.choice(pool)
                if k in sets[a]:
                    sets[a].discard(k)
                else:
                    sets[a].add(k)
                out.append(f"G {a} {k:x}")
            elif op < 0.55:
                k = r.choice(pool)
                sets[a].add(k)
                out.append(f"A {a} {k:x}")
            elif op < 0.63:
                k = r.choice(pool)
                sets[a].discard(k)
                out.append(f"D {a} {k:x}")
            elif op < 0.71:
                sets[a] = set(sets[b])
                out.append(f"C {a} {b}")
            elif op < 0.79:
                sets[a] = sets[b] ^ sets[c]
                out.append(f"X {a} {b} {c}")
            elif op < 0.87:
                if a != b:
                    sets[a] ^= sets[b]
                    out.append(f"I {a} {b}")
            elif op < 0.95:
                k = r.choice(pool)
                sets[a] = {k}
                out.append(f"L {a} {k:x}")
            else:
                sets[a] = set()
                out.append(f"N {a}")
            if step < 24 or step % 13 == 0:
                out.append(("O %d %s" % (a, " ".join(f"{k:x}" for k in sets[a]))).rstrip())
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def replay_small(path):
    sets = [set() for _ in range(4)]
    bad = checked = 0
    for line in open(path):
        q = line.split()
        if not q:
            continue
        op = q[0]
        if op == "T":
            sets = [set() for _ in range(4)]
        elif op == "N":
            sets[int(q[1])] = set()
        elif op == "L":
            sets[int(q[1])] = {int(q[2], 16)}
        elif op == "G":
            s, k = sets[int(q[1])], int(q[2], 16)
            if k in s:
                s.discard(k)
            else:
                s.add(k)
        elif op == "A":
            sets[int(q[1])].add(int(q[2], 16))
        elif op == "D":
            sets[int(q[1])].discard(int(q[2], 16))
        elif op == "C":
            sets[int(q[1])] = set(sets[int(q[2])])
        elif op == "X":
            sets[int(q[1])] = sets[int(q[2])] ^ sets[int(q[3])]
        elif op == "I":
            sets[int(q[1])] ^= sets[int(q[2])]
        elif op == "O":
            checked += 1
            if [f"{k:x}" for k in sets[int(q[1])]] != q[2:]:
                bad += 1
        else:
            raise SystemExit(f"bad trace op {op!r}")
    return checked, bad


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "record":
        record_small(sys.argv[2])
        with open(sys.argv[3], "w") as f:
            for seed, target in LARGE:
                cps = large_trace(seed, target)
                f.write(f"large {seed} {target} "
                        + " ".join(f"{n}:{a:016x}:{b:016x}:{c:016x}" for n, a, b, c in cps) + "\n")
        return
    checked, bad = replay_small(sys.argv[1])
    lbad = lchecked = 0
    if len(sys.argv) > 2:
        for line in open(sys.argv[2]):
            q = line.split()
            if not q:
                continue
            seed, target = int(q[1]), int(q[2])
            got = [f"{n}:{a:016x}:{b:016x}:{c:016x}" for n, a, b, c in large_trace(seed, target)]
            lchecked += len(got)
            lbad += sum(1 for x, y in zip(got, q[3:]) if x != y) + abs(len(got) - len(q) + 3)
    print(f"{sys.version.split()[0]}: {checked} small orders, {bad} mismatches; "
          f"{lchecked} large checkpoints, {lbad} mismatches")
    sys.exit(1 if bad or lbad else 0)


if __name__ == "__main__":
    main()
