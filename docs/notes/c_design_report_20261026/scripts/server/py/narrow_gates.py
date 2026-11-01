"""Gate-level metrics of the deeper values found by the narrow portfolio. usage: narrow_gates.py <run_dir> <arm> <ledger_name>"""
import json,hashlib,sys
from pathlib import Path
from collections import defaultdict
R=Path(sys.argv[1]);arm=sys.argv[2];name=sys.argv[3]
src=R/"stages35"/arm/"source.mpmct"
with open(src) as f:
    _,n,c=next(f).split();live=[-w-1 for w in range(int(n))];reads=defaultdict(int);ctrl=[];prev=[]
    for i,l in enumerate(f):
        v=list(map(int,l.split()));cb=[live[v[3]],live[v[5]]];ctrl.append(cb);prev.append(live[v[0]])
        for b in cb:reads[b]+=1
        live[v[0]]=i
outputs=set(live);internal=[i for i in range(len(ctrl)) if i not in outputs]
h=hashlib.sha256(src.read_bytes()).hexdigest();bnd=json.load(open(R/"boundary"/(h+".json")))
boundary={r["source_gate"] for r in bnd["rows"] if r["category"]=="public_quadratic_boundary" and not r["is_output"]}
found=set()
for p in (R/"ledgers"/name).glob("*/ledger.json"):
    for cc in json.load(open(p))["representations"]:
        cc=int(cc)
        if cc in internal:found.add(cc)
ex=found|boundary  # boundary births are public anyway
P=lambda b:b<0 or b in ex
rw=sum(reads[i] for i in found if i not in boundary)/sum(reads[i] for i in internal)
both=sum(1 for i in range(len(ctrl)) if P(ctrl[i][0]) and P(ctrl[i][1]));inc=sum(1 for i in internal if P(i) and P(prev[i]))
print(f"{name}: deeper found {len(found-boundary)}/{len(set(internal)-boundary)} | reads-weighted (deeper found) {rw:.3f} | gates both-controls known {both}/1000 | increments known {inc}")
