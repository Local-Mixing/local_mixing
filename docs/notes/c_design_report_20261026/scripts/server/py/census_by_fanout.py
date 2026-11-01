"""Exposure census (proxy_intervals output) stratified by source control fanout + gate-level metrics.
usage: census_by_fanout.py <source.mpmct> <census_dir>... (uses *_validated_intervals.tsv, internal births only)"""
import sys,csv
from collections import defaultdict
from pathlib import Path
def load(path):
    with open(path) as f:
        _,w,c=next(f).split();g=[]
        for l in f:
            v=list(map(int,l.split()));g.append((v[0],v[3],v[5]))
    return int(w),g
def structure(g,n):
    live=[-w-1 for w in range(n)];reads=defaultdict(int);ctrl=[];prev=[]
    for i,(t,x,y) in enumerate(g):
        cb=[live[x],live[y]];ctrl.append(cb);prev.append(live[t])
        for b in cb:reads[b]+=1
        live[t]=i
    outputs=set(live);internal=[i for i in range(len(g)) if i not in outputs]
    return reads,ctrl,prev,outputs,internal
n,g=load(sys.argv[1]);reads,ctrl,prev,outputs,internal=structure(g,n)
print('fanout dist:',dict(sorted(((k,v) for k,v in __import__('collections').Counter(min(reads[i],8) for i in internal).items()))))
for cdir in sys.argv[2:]:
    for fam in ('direct','affine'):
        p=Path(cdir)/f'{fam}_validated_intervals.tsv'
        if not p.exists():p=Path(cdir)/f'{fam}_intervals.tsv'
        ex=set()
        with p.open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                if r['is_output']!='true':ex.add(int(r['source_gate']))
        ex&=set(internal)
        byk=defaultdict(lambda:[0,0])
        for i in internal:k=min(reads[i],8);byk[k][0]+=1;byk[k][1]+=(i in ex)
        P=lambda b:b<0 or b in ex
        both=sum(1 for i in range(len(g)) if P(ctrl[i][0]) and P(ctrl[i][1]));inc=sum(1 for i in internal if P(i) and P(prev[i]))
        rw=sum(reads[i] for i in ex)/max(1,sum(reads[i] for i in internal))
        print(f'{cdir} {fam:6s} exposed {len(ex)}/{len(internal)} | per-fanout '+' '.join(f'{k}:{byk[k][1]}/{byk[k][0]}' for k in sorted(byk))+f' | reads-weighted {rw:.3f} | gates-both {both} | increments {inc}')
