#!/usr/bin/env python3
"""Idle-bare census of a blinded-V5 gadget (mpmct1): per data wire, track the
parity of every g57 mask pair written to it and report the intervals during
which the wire has NO open pair (= holds its plaintext), with the trigger
(rerand-burst straddle close vs other). Data wires are 0..NP, band wires NP..
Usage: bare_census.py <gadget.mpmct1>   (NP=256, BAND=256 for the n=128 sandwich)
Note: with encoded I/O the pre-opened masks are closed on-trace, which this
parity census reads as opens; use it on production (plain-I/O) gadgets."""
import sys, collections

path=sys.argv[1]; NP=256; BAND=256
L=open(path).read().split("\n"); h=L[0].split(); nw=int(h[1]); L=L[1:]
G=[]
for l in L:
    if not l: continue
    t=l.split(); tgt=int(t[0]); comp=int(t[1]); k=int(t[2]); lits=[(int(t[3+2*i]),int(t[4+2*i])) for i in range(k)]
    G.append((tgt,comp,lits))
m=len(G)
openp=[set() for _ in range(NP)]
bare_start=[None]*NP   # gate index where the wire became bare (None if masked)
ever=[False]*NP        # has the wire ever had an open pair (input fringe)
segs=[]  # (wire, start, end, trigger)
def pairshape(comp,lits):
    if comp==1 and len(lits)==2 and lits[0][0]>=BAND and lits[1][0]>=BAND and lits[0][1]!=lits[1][1]:
        return (lits[0][0],lits[1][0])
    return None
for i,(tgt,comp,lits) in enumerate(G):
    if tgt>=NP: continue
    pr=pairshape(comp,lits)
    was_empty = len(openp[tgt])==0
    if pr:
        if pr in openp[tgt]: openp[tgt].remove(pr)
        else: openp[tgt].add(pr)
    now_empty = len(openp[tgt])==0
    if pr and not now_empty: ever[tgt]=True
    if not was_empty and now_empty and ever[tgt]:
        # became bare: classify by the following gates: burst (band target within next 12 gates and pair read that band wire)?
        nxt=[G[j][0] for j in range(i+1,min(m,i+3000))]
        trig="?"
        if any(t>=BAND for t in nxt):
            # find the burst wire
            bw=[t for t in nxt if t>=BAND][0]
            trig="STRADDLE(burst on %d, pair reads it=%s)"%(bw, bw in pr)
        else:
            # next write on tgt within 2 gates that is a pair open?
            j=i+1; 
            while j<m and G[j][0]!=tgt: j+=1
            if j<m and pairshape(G[j][1],G[j][2]) and j-i<=4: trig="CLOSE-THEN-OPEN(gap %d)"%(j-i)
            else: trig="OTHER"
        bare_start[tgt]=(i,trig)
    if was_empty and not now_empty and bare_start[tgt] is not None:
        s,trig=bare_start[tgt]; segs.append((tgt,s,i,trig.split("(")[0])); bare_start[tgt]=None
from collections import Counter
inter=[s for s in segs if 0.15*m<=s[1]<=0.85*m]
print(path.split("/")[-1], "gates",m)
print(" bare intervals (masked before & after):",len(segs)," interior(15-85%):",len(inter))
print(" by trigger (all):",Counter(s[3] for s in segs))
print(" by trigger (interior):",Counter(s[3] for s in inter))
lens=sorted(s[2]-s[1] for s in inter)
if lens: print(" interior lengths: min",lens[0],"median",lens[len(lens)//2],"max",lens[-1])
# how many interior bare intervals contain a READ of the wire (fire) -> exposure of an operand
reads=[0]*len(inter)
# index reads per wire
from bisect import bisect_left
rd=[[] for _ in range(NP)]
for i,(tgt,comp,lits) in enumerate(G):
    for w,_ in lits:
        if w<NP: rd[w].append(i)
cnt=0
for (w,s,e,_) in inter:
    a=bisect_left(rd[w],s); b=bisect_left(rd[w],e)
    if b>a: cnt+=1
print(" interior bare intervals containing a read of the wire:",cnt)
