"""Structural profile of C values, leaked vs not.

usage: c_structure.py TAG LEAKED_GATES... [--upto N]
Per C gate g (value V_g on wire a after gate g; g57 gate a ^= x OR NOT y):
  supp   input bits V_g depends on
  ones   fraction of inputs where V_g = 1
  life   gates until wire a is rewritten (or to the end)
  reads  later gates reading wire a within that lifetime
  dupC   other C gate values equal to V_g (up to complement), anywhere in C
  live   of those, how many are simultaneously live during V_g's lifetime
  inp    V_g equals an input bit (up to complement)
  aff    V_g is an XOR of <=2 OTHER simultaneously live C values (+const)
Exhaustive over inputs restricted to the union of supports is not possible
at 64 bits, so functions are compared on 4096 random inputs (exact up to
negligible collision probability).
"""
import sys, random
ALPH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
D = {c: i for i, c in enumerate(ALPH)}
args = sys.argv[1:]
upto = 100
if '--upto' in args:
    i = args.index('--upto'); upto = int(args[i + 1]); args = args[:i] + args[i + 2:]
tag, leaked = args[0], {int(x) for x in args[1:]}
body = open(f'{tag}/tdp.mpmct1.source_c.g57').read().strip().strip(';')
C = [tuple(D[ch] for ch in tok) for tok in body.split(';') if tok]
n, S = 64, 4096
MASK = (1 << S) - 1
rng = random.Random(3)
inp = [rng.getrandbits(S) for _ in range(n)]
st = list(inp)
vals = []
for a, x, y in C:
    st[a] ^= (st[x] | (st[y] ^ MASK))
    vals.append(st[a])
def canon(v): return v ^ MASK if v & 1 else v
# lifetimes and reads
life, reads = [], []
for g, (a, _, _) in enumerate(C):
    nxt = next((h for h in range(g + 1, len(C)) if C[h][0] == a), len(C))
    life.append(nxt - g)
    reads.append(sum(1 for h in range(g + 1, nxt) if a in (C[h][1], C[h][2])))
# support via single-bit flips on a few base points
supp = [set() for _ in C]
for _ in range(24):
    x0 = [rng.getrandbits(1) for _ in range(n)]
    def run(x):
        s = list(x); out = []
        for a, xx, y in C:
            s[a] ^= (s[xx] | (1 - s[y])); out.append(s[a])
        return out
    b0 = run(x0)
    for i in range(n):
        x1 = list(x0); x1[i] ^= 1
        for g, (u, v) in enumerate(zip(b0, run(x1))):
            if u != v: supp[g].add(i)
by_fn = {}
for g, v in enumerate(vals):
    by_fn.setdefault(canon(v), []).append(g)
inputs_fn = {canon(v): i for i, v in enumerate(inp)}
def live_at(h, g):  # is V_h live somewhere within V_g's lifetime?
    return h < g + life[g] and h + life[h] > g
print(f'{tag}: {len(C)} C gates; profiling first {upto}; leaked = {sorted(leaked)}')
print(' gate  L  wire supp  ones  life reads dupC live  inp  aff')
for g in range(min(upto, len(C))):
    v = vals[g]; cv = canon(v)
    dups = [h for h in by_fn[cv] if h != g]
    ldup = [h for h in dups if live_at(h, g)]
    ones = bin(v).count('1') / S
    ii = inputs_fn.get(cv, '')
    # XOR of <=2 other simultaneously live C values or inputs
    pool = [canon(vals[h]) for h in range(len(C)) if h != g and live_at(h, g)]
    pool += [canon(u) for u in inp]
    aff = ''
    ps = set(pool)
    if cv in ps:
        aff = '1'
    else:
        for p in pool:
            if canon(cv ^ p) in ps:
                aff = '2'; break
    flag = '*' if g in leaked else ' '
    if g in leaked or g < upto:
        print(f'{g:5d}  {flag} {C[g][0]:4d} {len(supp[g]):4d} {ones:5.2f} {life[g]:5d} {reads[g]:5d} '
              f'{len(dups):4d} {len(ldup):4d} {str(ii):>4} {aff:>4}')
