"""C-design arms: sizes per stage, raw census and nine-arm narrow portfolio (R/O deeper/boundary), per arm with its own source."""
import json,sys,zlib
from pathlib import Path
from pipeline import ROOT,ARMS,save,size,digest
def gz(path):
 z=zlib.compressobj(6,zlib.DEFLATED,31);n=0
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):n+=len(z.compress(b))
 return n+len(z.flush())
def main():
 meta=json.loads((ROOT/'stage2/inputs.json').read_text());out=dict(arms={})
 for arm in ARMS:
  d=ROOT/'stages35'/arm;src=d/'source.mpmct'
  if not src.exists():continue
  bf=ROOT/'boundary'/(digest(src)+'.json')
  if bf.exists():
   bnd=json.loads(bf.read_text());boundary={r['source_gate'] for r in bnd['rows'] if r['category']=='public_quadratic_boundary' and not r['is_output']};internal={r['source_gate'] for r in bnd['rows'] if not r['is_output']}
  else:boundary=set();internal=None
  row=dict(stage2_gates=meta['designs'][arm]['stage2_gates'],denominators=None if internal is None else dict(internal=len(internal),deeper=len(internal-boundary),boundary=len(boundary)))
  for s in('stage3','stage4','stage5'):
   m=d/(s+'.DONE.json')
   if m.exists():j=json.loads(m.read_text());row[s+'_gates']=j['gates'];row[s+'_seconds']=round(j['seconds'])
  if (d/'stage5.mpmct').exists():row['stage5_gzip6_bytes']=gz(d/'stage5.mpmct')
  arms={}
  for p in sorted((ROOT/'ledgers'/arm).glob('*/ledger.json')) if (ROOT/'ledgers'/arm).exists() else []:
   j=json.loads(p.read_text());reps={int(c):n for c,n in j['representations'].items()}
   found={c for c in reps if internal is None or c in internal};rep={c for c in found if reps[c]>=2};once=found-rep
   arms[p.parent.name]=dict(deeper_repeated=len(rep-boundary),deeper_once=len(once-boundary),boundary_repeated=len(rep&boundary),boundary_once=len(once&boundary),supports=j['distinct_supports'],locations=j['physical_locations'],audits_passed=j['fresh_passed'] and j['structured_passed'],repeated_ids=sorted(rep),once_ids=sorted(once))
  found=set();rep=set()
  for a in arms.values():found|=set(a['repeated_ids'])|set(a['once_ids']);rep|=set(a['repeated_ids'])
  once=found-rep
  row['arms']=arms;row['portfolio']=dict(completed_arms=len(arms),deeper_repeated=len(rep-boundary),deeper_once=len(once-boundary),boundary_repeated=len(rep&boundary),boundary_once=len(once&boundary))
  out['arms'][arm]=row
 save(ROOT/'cdesign_comparison.json',out)
 print('arm | denominators internal/deeper/boundary | stage2 | stage3 | stage4 | stage5 | gzip5 | raw deeper R/O | narrow deeper R/O (arms) | narrow boundary R/O')
 for arm,r in out['arms'].items():
  raw=r['arms'].get('full_raw',{});pf=r['portfolio'];dn=r['denominators']
  print(f"{arm} | {dn and (str(dn['internal'])+'/'+str(dn['deeper'])+'/'+str(dn['boundary']))} | {r.get('stage2_gates')} | {r.get('stage3_gates')} | {r.get('stage4_gates')} | {r.get('stage5_gates')} | {r.get('stage5_gzip6_bytes')} | {raw.get('deeper_repeated')}/{raw.get('deeper_once')} | {pf['deeper_repeated']}/{pf['deeper_once']} ({pf['completed_arms']}) | {pf['boundary_repeated']}/{pf['boundary_once']}")
if __name__=='__main__':main()
