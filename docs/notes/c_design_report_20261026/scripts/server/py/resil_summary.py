"""Resilience campaign summary. usage: resil_summary.py <run_dir> <name>..."""
import json,hashlib,sys
from pathlib import Path
R=Path(sys.argv[1])
def rounds(p):
    return [(x["round"],x["input_gates"],x["output_gates"],x["deeper_direct"],x["deeper_affine"]) for x in json.load(open(p))["rounds"]]
for name in sys.argv[2:]:
    arm=name.rsplit("_",1)[0];area=R/"resilience"/name
    if not (area/"mixed_output.json").exists():print(name,"not started");continue
    F=json.load(open(area/"mixed_output.json"))
    src=R/"stages35"/arm/"source.mpmct";h=hashlib.sha256(src.read_bytes()).hexdigest();bnd=json.load(open(R/"boundary"/(h+".json")))
    boundary={r["source_gate"] for r in bnd["rows"] if r["category"]=="public_quadratic_boundary" and not r["is_output"]};internal={r["source_gate"] for r in bnd["rows"] if not r["is_output"]}
    def narrow(led):
        reps={};cases=list(led.glob("full_*/ledger.json"))
        for p in cases:
            for c,n in json.load(open(p))["representations"].items():
                c=int(c)
                if c in internal:reps[c]=max(reps.get(c,0),n)
        deep={c:n for c,n in reps.items() if c not in boundary}
        return f"{len(cases)} cases: deeper R/O {sum(n>=2 for n in deep.values())}/{sum(n==1 for n in deep.values())}, boundary {sum(1 for c in reps if c in boundary)}"
    print(f"{name}: F={F['gates']} gates | narrow portfolio on F: {narrow(R/'ledgers'/name)}")
    for step in ("direct","remix"):
        pj=area/step/arm/"attacker"/"progress.json";fj=area/step/arm/"attacker"/"final.json"
        if pj.exists():
            ex=rounds(pj);print(f"   {step} C-blind: "+"; ".join(f"r{a}:{b}->{c} exp {d}/{e}" for a,b,c,d,e in ex)+(" PLATEAU" if fj.exists() and json.load(open(fj)).get("plateau") else ""))
    rm=area/"remix"/"DONE.json"
    if rm.exists():print(f"   remix circuit {json.load(open(rm))['gates']} gates")
    for extra in ("_resynth","_remix","_remix_resynth"):
        led=R/"ledgers"/(name+extra)/"full_raw"/"ledger.json"
        if led.exists():
            j=json.load(open(led));reps={int(c):n for c,n in j["representations"].items() if int(c) in internal};deep={c:n for c,n in reps.items() if c not in boundary}
            print(f"   raw ledger on {name+extra}: deeper R/O {sum(n>=2 for n in deep.values())}/{sum(n==1 for n in deep.values())}")
