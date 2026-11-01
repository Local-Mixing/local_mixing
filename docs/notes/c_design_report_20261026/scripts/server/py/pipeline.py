"""C-design arms (2026-10-21): same SEED, sandwich recipe, stage-2 flags and stage 3-5 seeds as ~/threemask_cmp; only the
source C differs (supplied g57 file). Arms: <c>_base = pinned managed_tdp stage 2, <c>_v2 = three-mask modified
(--random-helpers --unique-cover). Protocol otherwise = threemask_cmp/py/pipeline.py (reduced profile 2,2,2,2,2)."""
import hashlib,json,os,shutil,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HOME=Path.home();CMP=HOME/'cdesign';ROOT=CMP/'run';ROOT.mkdir(exist_ok=True)
GEN=CMP/'generator/target/release';MIXER=CMP/'upstream/target/release/circuit_mixer';CHECK=CMP/'anf_synth_768/target/release/circuit_check'
SEED=int((CMP/'SEED').read_text().strip())
SOURCES={'bal':CMP/'sources/bal.g57.txt','burst67':CMP/'sources/burst67.g57.txt'}
ARMS={f'{c}_{k}':(c,f) for c in SOURCES for k,f in (('base','baseline_stage2.mpmct'),('v2','pilot_stage2.mpmct'))}
BASE=['--n','64','--source-gates','1000','--band-wires','128','--banks','8','--retire-budget','2','--max-retiring','0','--rotation','1','--maturity','2','--visit-sweeps','2','--growth','8','--warmup','4','--segments','6','--local-masks','--prefer-fresh','--compact-masks','--forest-masks','--cubic-mask','--cubic-chain','--temporal-masks','--cross-bank-growth','--discard-fire-helpers','--mask-count','3']
V2=['--random-helpers','--unique-cover']
def save(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(o,indent=2)+'\n');t.replace(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def size(p):
 with Path(p).open()as f:return int(f.readline().split()[2])
def run(cmd,log,env=None):
 log.parent.mkdir(parents=True,exist_ok=True);cmd=list(map(str,cmd));meta=dict(command=[c.replace(str(SEED),'<SEED>') for c in cmd],start=time.time())
 print('START',log,flush=True)
 with log.open('w')as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 meta.update(exit_code=rc,end=time.time());save(log.with_suffix(log.suffix+'.process.json'),meta)
 if rc:raise RuntimeError(f'failed ({rc}): {log}')
 print('DONE',log,flush=True)
def generate():
 g=ROOT/'stage2';meta=g/'inputs.json'
 if meta.exists():return json.loads(meta.read_text())
 for c,src in SOURCES.items():
  out=g/c
  if not (out/'stats.json').exists():
   if out.exists():shutil.rmtree(out)
   run([GEN/'stage2_compute_pilot']+BASE+['--seed',SEED,'--out',out,'--source-file',src]+V2,g/(c+'.gen.log'))
   run([GEN/'payload_check',out/'sandwich.mpmct',out/'pilot_stage2.mpmct',8192],out/'payload_check_v2.txt')
   run([GEN/'payload_check',out/'sandwich.mpmct',out/'baseline_stage2.mpmct',8192],out/'payload_check_base.txt')
 m=dict(sources={c:dict(g57=str(p),source=str(g/c/'source.mpmct'),source_sha256=digest(g/c/'source.mpmct'),sandwich_sha256=digest(g/c/'sandwich.mpmct'))for c,p in SOURCES.items()},
  designs={a:dict(c=c,stage2=str(g/c/f),stage2_sha256=digest(g/c/f),stage2_gates=size(g/c/f))for a,(c,f)in ARMS.items()},
  generator_sha256=digest(GEN/'stage2_compute_pilot'),mixer_sha256=digest(MIXER),recipe='threemask_cmp protocol (reduced profile 2,2,2,2,2); SEED copied from ~/threemask_cmp so D, sandwich and stage 3-5 seeds match the original-C arms')
 save(meta,m);return m
def stages(arm):
 meta=generate();c=meta['designs'][arm]['c'];f=Path(meta['designs'][arm]['stage2']);d=ROOT/f'stages35/{arm}';d.mkdir(parents=True,exist_ok=True)
 if (d/'DONE.json').exists():return d
 if not (d/'source.mpmct').exists():shutil.copyfile(meta['sources'][c]['source'],d/'source.mpmct')
 env=os.environ.copy();env.update(FROZEN_DB_DIR=str(HOME/'frozen_db/frozen_m1_m11'),FROZEN_FILTER='0',LOOKUP_CACHE_MB='512',CANON_RULE_L_BRANCH_CAP='512',CANON_MONOMIAL_CAP='200000',MIXER_STOP_FLAG=str(d/'STOP'),MIXER_DUMP_FLAG=str(d/'DUMP'))
 def execute(stage,source,args):
  target=d/(stage+'.mpmct');marker=d/(stage+'.DONE.json')
  if not marker.exists():
   if target.exists():raise RuntimeError(f'unverified partial output: {target}')
   cmd=[MIXER]+args+['--state-out',d/(stage+'.state'),'--output',target]
   save(d/(stage+'.protocol.json'),dict(command=[str(x).replace(str(SEED+2),'<SEED+2>').replace(str(SEED+3),'<SEED+3>') for x in cmd],recipe='reduced profile 2,2,2,2,2',binary_sha256=digest(MIXER),input_sha256=digest(source),source_sha256=meta['sources'][c]['source_sha256'],design=arm))
   t0=time.monotonic();run(cmd,d/(stage+'.log'),env);save(marker,dict(seconds=time.monotonic()-t0,gates=size(target),sha256=digest(target)))
  if not (d/(stage+'.equivalence.txt')).exists():run([CHECK,f,target,8192],d/(stage+'.equivalence.txt'))
  return target
 s3=execute('stage3',f,['--input',f,'--tdp','--db-mixing','--profile','2,2,2,2,2','--moves',round(5.2*size(f)),'--seed',SEED+2,'--p-db',1,'--p-comp',1,'--p-any',0.1,'--db-max-degree',9,'--db-max-span',30,'--db-wire-terms',1024,'--db-total-terms',2048,'--no-local-verify','--verify-every',2000000,'--report-every',100000])
 s4=execute('stage4',s3,['--input',s3,'--split','--split-stop','--p-join',0.8,'--split-reach-k',2,'--split-fail-limit',100,'--split-canaries',256,'--p-db',0,'--p-comp',0,'--p-any',0,'--p-twist',0,'--k-max',12,'--moves',size(s3)+1000000,'--seed',SEED+3,'--report-every',1000000])
 with (d/'stage4.state').open()as sf:moves=next(int(l.split()[1])for l in sf if l.startswith('moves '))
 target=2*size(s4)
 execute('stage5',s4,['--resume',d/'stage4.state','--target-size',target,'--temp',max(64,round(target/25)),'--split-base',3,'--split-damp',1,'--p-twist',0,'--p-db',0,'--p-comp',0,'--p-any',0,'--k-max',12,'--moves',moves+3*target,'--report-every',500000])
 norm=d/'stage5_two_control.mpmct'
 if not norm.exists():run([GEN/'normalize_gates',d/'stage5.mpmct',norm],d/'normalize.log')
 if not (d/'normalize_check.txt').exists():run([CHECK,d/'stage5.mpmct',norm,8192],d/'normalize_check.txt')
 save(d/'DONE.json',dict(passed=True,all_stages_checked=True,arm=arm));return d
if __name__=='__main__':
 arms=sys.argv[1:]or list(ARMS);generate();fails=[]
 with ThreadPoolExecutor(max_workers=4)as pool:
  for arm,fut in [(a,pool.submit(stages,a))for a in arms]:
   try:fut.result();print('ARM DONE',arm,flush=True)
   except Exception as e:fails.append(dict(arm=arm,error=repr(e)));print('FAILED',arm,repr(e),flush=True)
 save(ROOT/'mix_status.json',dict(complete=not fails,failures=fails,updated=time.time()))
