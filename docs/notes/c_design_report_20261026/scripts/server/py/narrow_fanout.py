"""Per-case R/O and source-fanout distribution of the deeper values found by the narrow portfolio.
usage: narrow_fanout.py <run_dir> <arm> <ledger_name>..."""
import json,hashlib,sys
from pathlib import Path
from collections import defaultdict,Counter
R=Path(sys.argv[1]);arm=sys.argv[2]
src=R/"stages35"/arm/"source.mpmct"
with open(src) as f:
    _,n,c=next(f).split();live=[-w-1 for w in range(int(n))];reads=defaultdict(int)
    for i,l in enumerate(f):
        v=list(map(int,l.split()));reads[live[v[3]]]+=1;reads[live[v[5]]]+=1;live[v[0]]=i
h=hashlib.sha256(src.read_bytes()).hexdigest();bnd=json.load(open(R/"boundary"/(h+".json")))
boundary={r["source_gate"] for r in bnd["rows"] if r["category"]=="public_quadratic_boundary" and not r["is_output"]};internal={r["source_gate"] for r in bnd["rows"] if not r["is_output"]}
for name in sys.argv[3:]:
    reps={};per=[]
    for p in sorted((R/"ledgers"/name).glob("*/ledger.json")):
        j=json.load(open(p));r={int(c):m for c,m in j["representations"].items() if int(c) in internal and int(c) not in boundary}
        per.append(f"{p.parent.name}:{sum(m>=2 for m in r.values())}/{sum(m==1 for m in r.values())}")
        for c,m in r.items():reps[c]=max(reps.get(c,0),m)
    fan=Counter(min(reads[c],5) for c in reps);fanR=Counter(min(reads[c],5) for c,m in reps.items() if m>=2)
    print(f"{name}: deeper found {len(reps)} (R {sum(m>=2 for m in reps.values())}) | per case {' '.join(per)} | by fanout found/R: "+" ".join(f"{k}:{fan[k]}/{fanR[k]}" for k in sorted(fan))+f" | max locations {max(reps.values(),default=0)}")
