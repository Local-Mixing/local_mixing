"""Exploratory measurements for the w>=6 g57 unit synthesizer decision.
Run with PYTHONPATH=<tools/pr_identity>.
"""
import itertools, time, sys, random
import numpy as np
import g57

def gates(m):
    return list(itertools.permutations(range(m), 3))  # (a,x,y) all distinct

def perm_parity(T):
    """parity of permutation table T (0 even, 1 odd)."""
    n = len(T); seen = np.zeros(n, bool); par = 0
    for i in range(n):
        if seen[i]: continue
        j = i; L = 0
        while not seen[j]:
            seen[j] = True; j = T[j]; L += 1
        par ^= (L - 1) & 1
    return par

def check_algebra():
    print("=== algebra checks (w=4 canonical) ===")
    m = 4
    a,x,y,z = 0,1,2,3
    def tbl(word): return g57.word_table(word, m)
    ID = np.arange(1<<m)
    # g57(a,x,y) fires iff x OR NOT y  == NOT(NOT x AND y)
    g = g57.gate_table((a,x,y), m)
    st = np.arange(1<<m)
    fire = ((st>>x)&1) | (1-((st>>y)&1))
    fire2 = 1 ^ (((1-((st>>x)&1)) & ((st>>y)&1)))  # 1 xor (xbar y)
    print("  g57 = NOT(a).Toff(a;xbar,y):", np.array_equal(fire, fire2))
    # double same (a,x): a ^= xbar(y xor z)
    d = tbl([(a,x,y),(a,x,z)])
    st = np.arange(1<<m)
    xb = 1-((st>>x)&1); yz = ((st>>y)&1)^((st>>z)&1)
    exp = st ^ ((xb & yz) << a)
    print("  g57(a,x,y).g57(a,x,z) = a^=xbar(y^z):", np.array_equal(d, exp))
    # swap controls: a ^= x xor y  (linear!)
    l = tbl([(a,x,y),(a,y,x)])
    st = np.arange(1<<m); exp = st ^ ((((st>>x)&1)^((st>>y)&1))<<a)
    print("  g57(a,x,y).g57(a,y,x) = a^=x^y (LINEAR):", np.array_equal(l, exp))
    # parity of a single g57 gate at various w
    for w in (3,4,5,6):
        T = g57.gate_table((0,1,2), w)
        print(f"  parity(single g57) w={w}: {perm_parity(T)} (0=even)")

def group_at_w3():
    print("=== full group enumeration w=3 ===")
    m=3; ID=np.arange(1<<m,dtype=np.int64)
    gs=[g57.gate_table(g,m) for g in gates(m)]
    seen={ID.tobytes()}; frontier=[ID]
    while frontier:
        nf=[]
        for P in frontier:
            for tb in gs:
                Q=tb[P]; k=Q.tobytes()
                if k not in seen:
                    seen.add(k); nf.append(Q)
        frontier=nf
    order=len(seen)
    import math
    print(f"  |<g57>| on 2^3=8 pts: {order}  (|S8|={math.factorial(8)}, |A8|={math.factorial(8)//2})")

def ball_growth(m, maxr=5, cap_seconds=90):
    import hashlib
    print(f"=== ball growth w={m} (blake2b-16 keys) ===", flush=True)
    ID=np.arange(1<<m,dtype=np.int64)
    gs=[g57.gate_table(g,m) for g in gates(m)]
    print(f"  #gates={len(gs)}", flush=True)
    def key(P): return hashlib.blake2b(P.tobytes(), digest_size=16).digest()
    seen={key(ID)}; frontier=[ID]; total=1; prev_new=1
    t0=time.time()
    for r in range(1,maxr+1):
        nf=[]
        for P in frontier:
            for tb in gs:
                Q=tb[P]; k=key(Q)
                if k not in seen:
                    seen.add(k); nf.append(Q)
        frontier=nf; total=len(seen)
        dt=time.time()-t0
        br=len(nf)/max(1,prev_new)
        print(f"  r={r}: ball={total:,}  new={len(nf):,}  branch(new/prevnew)={br:.2f}  cum_time={dt:.1f}s", flush=True)
        prev_new=len(nf)
        if dt>cap_seconds or not frontier:
            print("  (stopped: time cap or exhausted)"); break

if __name__=="__main__":
    check_algebra()
    group_at_w3()
    ball_growth(6, maxr=5, cap_seconds=60)
