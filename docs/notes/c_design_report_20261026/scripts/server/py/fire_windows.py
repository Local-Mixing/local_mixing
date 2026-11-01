"""Map the factor_w250 observer's finds to stage-5 windows and compare with the scaled positions of the fires
that read the value. usage: fire_windows.py <stage2_dir> <stage5.mpmct> <attack_dir> <ledger_dir> <compute_start>"""
import sys,csv,bisect,re,statistics
from collections import defaultdict
d2,s5,att,led,start=sys.argv[1:6];start=int(start)
def gates(path):
    with open(path) as f:
        h=next(f).split();return int(h[1]),[list(map(int,l.split())) for l in f]
n,src=gates(f'{d2}/source.mpmct');_,sw=gates(f'{d2}/sandwich.mpmct')
live=[-w-1 for w in range(n)];cons=defaultdict(list)
for i,g in enumerate(src):
    cons[live[g[3]]].append(i);cons[live[g[5]]].append(i);live[g[0]]=i
# anchors: first logical gate with target t and controls {x,y} of sandwich gate j
_,log=gates(f'{d2}/pilot_logical_compute.mpmct')
key=lambda g:(g[0],frozenset(g[3::2])) if len(g)>3 else None
want={}
for j,g in enumerate(sw):
    k=key(g)
    if k is not None:want.setdefault(k,j)
print('sandwich gates without controls:',sum(1 for g in sw if len(g)<=3))
anchor={}
for p,g in enumerate(log):
    k=key(g)
    if k is not None and k in want and want[k] not in anchor:anchor[want[k]]=p
order=sorted(anchor,key=anchor.get);rank={j:k for k,j in enumerate(order)}
b=[int(l) for l in open(f'{d2}/pilot_fire_boundaries.txt') if l.strip()]
print(f'anchors {len(anchor)}/{len(sw)} sandwich gates; fires {len(b)-1}; fire length median {statistics.median(y-x for x,y in zip(b,b[1:]))} stage-2 gates')
_,w2,m2=open(f'{d2}/pilot_stage2.mpmct').readline().split();m2=int(m2);_,w5,m5=open(s5).readline().split();m5=int(m5);scale=m5/m2
print(f'stage 2 {m2} -> stage 5 {m5} gates, ratio {scale:.2f}; fire footprint ~{round(scale*statistics.median(y-x for x,y in zip(b,b[1:])))} stage-5 gates vs 250-gate windows')
wins=[];off=0
with open(f'{att}/attempts.tsv') as f:
    for r in csv.DictReader(f,delimiter='\t'):
        if int(r['depth'])>0 and r['status']!='ok':continue
        og=int(r['output_gates']);wins.append((off,off+og,int(r['lo']),int(r['hi'])));off+=og
starts=[a for a,_,_,_ in wins];wg=int(open(f'{att}/generic.mpmct').readline().split()[1])
found=defaultdict(set)
with open(f'{led}/labels/values.csv') as f:
    for r in csv.DictReader(f):
        if r['is_output']!='false':continue
        for fld in('first_representation','second_representation'):
            if r[fld]:
                for s in r[fld].split(':')[1].split('+'):
                    s=int(s)
                    if s>=wg:found[int(r['source_gate'])].add(s-wg)
res=[]
for g in sorted(found):
    exp=[]
    for j in cons[g]:
        k=rank.get(j)
        if k is not None:exp.append((j,round(scale*(start+b[k])),round(scale*(start+b[k+1]))))
    locs=sorted({wins[bisect.bisect_right(starts,p)-1][2:] for p in found[g]})
    near=[]
    for lo,hi in locs:
        j,a,c=min(exp,key=lambda e:abs((lo+hi)//2-(e[1]+e[2])//2)) if exp else (None,0,0);near.append(((lo+hi)//2-(a+c)//2));res.append((lo+hi)//2-(a+c)//2)
    print(f'birth {g} (fanout {len(cons[g])}): consumer fires -> expected stage-5 ranges {[(j,a,c) for j,a,c in exp]} | found in {len(locs)} windows at {[lo for lo,hi in locs]} | offset from nearest expected fire {near}')
if res:print('offset found-expected: median',statistics.median(res),'IQR',sorted(res)[len(res)//4],sorted(res)[3*len(res)//4],'n',len(res))
