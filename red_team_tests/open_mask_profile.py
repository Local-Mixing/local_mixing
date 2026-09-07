#!/usr/bin/env python3
"""Open-mask profile of a blinded-V5 gadget (mpmct1): per data wire, track the
number of open g57 mask PAIRS (one per LGI; read-time top-ups and the
fire-cover bracket count while they are open) and report, gate-weighted over
the wire's covered window (first open .. last close), the distribution of the
open count, plus every stretch at exactly ONE open mask (start, length, half).
Usage: open_mask_profile.py <gadget.mpmct1> [NP=256] [BAND=NP]
Interior = stretches starting in the middle 70% of the gate list."""
import sys, collections
path=sys.argv[1]; NP=int(sys.argv[2]) if len(sys.argv)>2 else 256; BAND=int(sys.argv[3]) if len(sys.argv)>3 else NP
L=open(path).read().split("\n"); h=L[0].split(); L=L[1:]
G=[]
for l in L:
    if not l: continue
    t=l.split(); tgt=int(t[0]); comp=int(t[1]); k=int(t[2]); lits=[(int(t[3+2*i]),int(t[4+2*i])) for i in range(k)]
    G.append((tgt,comp,lits))
m=len(G)
def pairshape(comp,lits):
    if comp==1 and len(lits)==2 and lits[0][0]>=BAND and lits[1][0]>=BAND and lits[0][1]!=lits[1][1]:
        return (lits[0][0],lits[1][0])
    return None
openp=[set() for _ in range(NP)]
first=[None]*NP; last=[None]*NP
cnt_since=[0]*NP          # gate index of the last count change
weighted=collections.Counter()   # (count) -> gates, over covered windows
weighted_low=collections.Counter()
ones=[]                   # (wire, start, end)
cur_one=[None]*NP
for i,(tgt,comp,lits) in enumerate(G):
    if tgt>=NP: continue
    pr=pairshape(comp,lits)
    if not pr: continue
    before=len(openp[tgt])
    if pr in openp[tgt]: openp[tgt].remove(pr)
    else: openp[tgt].add(pr)
    after=len(openp[tgt])
    if first[tgt] is None: first[tgt]=i
    if before>0:
        weighted[before]+=i-cnt_since[tgt]
        if tgt<NP//2: weighted_low[before]+=i-cnt_since[tgt]
    cnt_since[tgt]=i
    if after>0: last[tgt]=i
    if before==1 and after!=1 and cur_one[tgt] is not None:
        ones.append((tgt,cur_one[tgt],i)); cur_one[tgt]=None
    if after==1 and before!=1: cur_one[tgt]=i
tot=sum(weighted.values()); totl=sum(weighted_low.values())
print(path.split("/")[-1],"gates",m)
print(" gate-weighted open-mask count over covered windows (all data wires):",
      " ".join(f"{c}:{100*weighted[c]/tot:.1f}%" for c in sorted(weighted)))
print(" same, low (payload) half only:",
      " ".join(f"{c}:{100*weighted_low[c]/totl:.1f}%" for c in sorted(weighted_low)))
inter=[o for o in ones if 0.15*m<=o[1]<=0.85*m]
low=[o for o in inter if o[0]<NP//2]
import statistics
def stats(xs):
    ls=sorted(e-s for _,s,e in xs)
    return f"n={len(ls)} median={ls[len(ls)//2] if ls else 0} p90={ls[int(.9*len(ls))] if ls else 0} max={ls[-1] if ls else 0} total_gates={sum(ls)}"
print(" ONE-mask stretches, all:",stats(ones))
print(" ONE-mask stretches, interior:",stats(inter))
print(" ONE-mask stretches, interior low half:",stats(low))
long=[o for o in inter if o[2]-o[1]>=1000]
print(" interior one-mask stretches >= 1000 gates:",len(long),"(low half:",sum(1 for o in long if o[0]<NP//2),")")
