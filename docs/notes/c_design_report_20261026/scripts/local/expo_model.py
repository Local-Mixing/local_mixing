"""Exposure vs source-read structure on RC's audited data (old C and balanced C).
Reads the per-birth labels of ~/Codex/baseline64_fanout2_pipeline_20261019/source_topology_followup_20261021 and prints,
per run and family: exposed count, per-fanout split, reads-weighted exposure, gates with both controls exposed, increments."""
import json
from collections import defaultdict
R='/Users/rancanetti/Codex/baseline64_fanout2_pipeline_20261019/source_topology_followup_20261021/'
def load_src(path):
    with open(path) as f:
        _,w,c=next(f).split();gates=[list(map(int,l.split())) for l in f]
    return int(w),gates
def structure(gates,width):
    live=[-w-1 for w in range(width)];reads=defaultdict(int);ctrl=[];prev=[]
    for i,g in enumerate(gates):
        t=g[0];cb=[live[w] for w in g[3::2]];ctrl.append(cb);prev.append(live[t])
        for b in cb:reads[b]+=1
        live[t]=i
    return reads,ctrl,prev,set(live)
for name,feat in [('old','old_features.json'),('balanced','balanced_features.json')]:
    F=json.load(open(R+feat));w,gates=load_src(F['source']);reads,ctrl,prev,outputs=structure(gates,w);ids=F['internal_ids']
    print('===',name)
    for run,lab in F['labels'].items():
        for fam in ('direct','affine'):
            ex=set(lab[fam]);byk=defaultdict(lambda:[0,0])
            for i in ids:k=min(reads[i],5);byk[k][0]+=1;byk[k][1]+=(i in ex)
            rw=sum(reads[i] for i in ex);tot=sum(reads[i] for i in ids)
            expd=lambda b:b<0 or b in ex
            both=sum(1 for i in range(len(gates)) if all(expd(b) for b in ctrl[i]))
            inc=sum(1 for i in range(len(gates)) if i not in outputs and expd(i) and expd(prev[i]))
            print(f'{run:28s} {fam:6s} exposed={len(ex):4d}/936 per-fanout:',' '.join(f'{k}:{byk[k][1]}/{byk[k][0]}' for k in sorted(byk)),f'| reads-weighted {rw/tot:.3f} | gates both {both} | increments {inc}')
