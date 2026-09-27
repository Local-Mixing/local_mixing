"""Geodesic-length distribution of real minimal-mask w-wire units inv(M).g.M'
via bidirectional BFS with a per-side node cap.  Decides MITM feasibility.
"""
import itertools, time, random, hashlib
import numpy as np
import g57

def gates(m): return list(itertools.permutations(range(m), 3))

def piece(block, lam, rng):
    tgts=list(block)
    while len(tgts)<lam: tgts.append(rng.choice(block))
    tgts=tgts[:lam]; rng.shuffle(tgts)
    return [(a,)+tuple(rng.sample([u for u in block if u!=a],2)) for a in tgts]

def make_unit(m, lam, rng):
    block=list(range(m))
    mask=piece(block,lam,rng)
    a,x,y=rng.sample(block,3); g=(a,x,y)
    pc=piece(block,lam,rng)
    unit=list(reversed(mask))+[g]+pc          # inv(mask).g.piece
    return unit

def key(P): return P.tobytes()   # exact key; memory ok for capped balls

def geodesic(T, m, gate_tables, node_cap=1_500_000, max_r=9):
    """Exact geodesic length of perm table T via bidirectional BFS; returns
    (dist or None if >found-within-cap, met_flag)."""
    ID=np.arange(1<<m,dtype=np.int64)
    kID=key(ID); kT=key(T)
    if kID==kT: return 0
    # forward from ID, backward from T (gates are involutions -> same set)
    Fseen={kID:0}; Ffront=[ID]
    Bseen={kT:0}; Bfront=[T]
    for r in range(1,max_r+1):
        # expand smaller frontier side
        if len(Ffront)<=len(Bfront):
            nf=[]
            for P in Ffront:
                for tb in gate_tables:
                    Q=tb[P]; k=key(Q)
                    if k not in Fseen:
                        Fseen[k]=r; nf.append(Q)
                        if k in Bseen: return r+Bseen[k]
                if len(Fseen)>node_cap: return None
            Ffront=nf
        else:
            nf=[]
            for P in Bfront:
                for tb in gate_tables:
                    Q=tb[P]; k=key(Q)
                    if k not in Bseen:
                        Bseen[k]=r; nf.append(Q)
                        if k in Fseen: return r+Fseen[k]
                if len(Bseen)>node_cap: return None
            Bfront=nf
        if not Ffront and not Bfront: return None
    return None

def run(m=6, lam=6, N=30, seed=1, node_cap=1_500_000):
    rng=random.Random(seed)
    gt=[g57.gate_table(g,m) for g in gates(m)]
    dist={}; times=[]
    print(f"=== geodesic distribution w={m} lam={lam} N={N} node_cap={node_cap:,} ===",flush=True)
    for i in range(N):
        unit=make_unit(m,lam,rng)
        T=g57.word_table(unit,m)
        t0=time.time(); d=geodesic(T,m,gt,node_cap=node_cap); dt=time.time()-t0
        times.append(dt)
        lab=d if d is not None else f">cap"
        dist[lab]=dist.get(lab,0)+1
        print(f"  unit {i:3d}: raw_len={len(unit)} geodesic={lab}  ({dt:.2f}s)",flush=True)
    print("  --- summary ---")
    for k in sorted(dist,key=lambda z:(isinstance(z,str),z)):
        print(f"    geodesic {k}: {dist[k]}")
    print(f"    mean time/unit: {sum(times)/len(times):.2f}s")

if __name__=="__main__":
    import sys
    N=int(sys.argv[1]) if len(sys.argv)>1 else 30
    run(N=N)
