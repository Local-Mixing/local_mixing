"""Direct C-value copies in a mixed circuit: physical versions per exposed value, simultaneous copies, control reads.
usage: locations.py <label> <source.mpmct> <circuit.mpmct1> <census_dir>"""
import sys,csv,bisect
from collections import defaultdict,Counter
label,src,circ,cdir=sys.argv[1:5]
def structure(path):
    with open(path) as f:
        _,n,c=next(f).split();n=int(n);live=[-w-1 for w in range(n)];reads=defaultdict(int);outs=None
        for i,l in enumerate(f):
            v=list(map(int,l.split()));t=v[0];x,y=v[3],v[5]
            reads[live[x]]+=1;reads[live[y]]+=1;live[t]=i
    outputs=set(live);return reads,{i for i in range(c and int(c)) if i not in outputs}
reads,internal=structure(src)
with open(circ) as f:
    _,nw,m=next(f).split();nw=int(nw);m=int(m);target=[];readers=defaultdict(list)
    for i,l in enumerate(f):
        v=l.split();target.append(int(v[0]))
        for j in range(3,len(v),2):readers[int(v[j])].append(i)
rows=[]
with open(f'{cdir}/direct_validated_intervals.tsv') as f:
    for r in csv.DictReader(f,delimiter='\t'):
        g=int(r['source_gate'])
        if r['is_output']=='true' or g not in internal:continue
        s,e,sup=int(r['start_cut']),int(r['end_cut']),int(r['support'])
        if sup>=nw:
            gi=sup-nw;w=target[gi];assert s==gi+1,(s,gi)
        else:w=sup
        e=min(e,m);rl=readers[w];nreads=bisect.bisect_left(rl,e)-bisect.bisect_left(rl,s)
        rows.append((g,s,e,w,nreads))
per=defaultdict(list)
for g,s,e,w,nr in rows:per[g].append((s,e,w,nr))
multi={g for g,v in per.items() if len(v)>=2}
simul=set()
for g,v in per.items():
    ev=sorted([(s,1) for s,e,w,nr in v]+[(e,-1) for s,e,w,nr in v]);c=0
    for t,d in ev:
        c+=d
        if c>=2:simul.add(g);break
dist=Counter(min(len(v),5) for v in per.values())
tot_reads=sum(nr for g,s,e,w,nr in rows);maxreads=max((nr for g,s,e,w,nr in rows),default=0)
byk=defaultdict(lambda:[0,0,0])
for g,v in per.items():k=min(reads[g],5);byk[k][0]+=1;byk[k][1]+=(g in multi);byk[k][2]+=len(v)
print(f"{label}: direct-exposed {len(per)} | >=2 locations {len(multi)} ({100*len(multi)/max(1,len(per)):.0f}%) | simultaneous copies {len(simul)} | versions {len(rows)} (per value 1:{dist[1]} 2:{dist[2]} 3:{dist[3]} 4:{dist[4]} 5+:{dist[5]}, max {max((len(v) for v in per.values()),default=0)}) | control reads of copies {tot_reads} (max per copy {maxreads}) | by fanout exposed/multi/versions: "+' '.join(f"{k}:{byk[k][0]}/{byk[k][1]}/{byk[k][2]}" for k in sorted(byk)))
