"""Candidate C designs: read distributions, model-predicted exposure (fanout rates from RC's adaptive pass 1),
pseudorandomness proxies (bit-sliced big ints). g57 gate [a,x,y]: a ^= x OR NOT y."""
import json,random,statistics
from collections import Counter,defaultdict
N=64;M=1000
R='/Users/rancanetti/Codex/baseline64_fanout2_pipeline_20261019/'
F=json.load(open(R+'source_topology_followup_20261021/old_features.json'))
def load_mpmct_g57(path):
    with open(path) as f:
        _,w,c=next(f).split();g=[]
        for l in f:
            v=list(map(int,l.split()));t,comp,nc=v[:3];(w1,p1),(w2,p2)=(v[3],v[4]),(v[5],v[6])
            assert comp==1 and nc==2
            x,y=(w1,w2) if (p1,p2)==(0,1) else (w2,w1)
            g.append((t,x,y))
    return int(w),g
def structure(g):
    live=[-w-1 for w in range(N)];reads=defaultdict(int);ctrl=[];prev=[]
    for i,(t,x,y) in enumerate(g):
        cb=[live[x],live[y]];ctrl.append(cb);prev.append(live[t])
        for b in cb:reads[b]+=1
        live[t]=i
    outputs=set(live);return reads,ctrl,prev,outputs,[i for i in range(len(g)) if i not in outputs]
_,old=load_mpmct_g57(F['source']);reads,ctrl,prev,outputs,internal=structure(old)
rates={}
for fam in ('direct','affine'):
    ex=set(F['labels']['adaptive/baseline/round1'][fam]);byk=defaultdict(lambda:[0,0])
    for i in internal:k=min(reads[i],8);byk[k][0]+=1;byk[k][1]+=(i in ex)
    rates[fam]={k:byk[k][1]/byk[k][0] for k in byk}
def f_of(k,r):return r[min(k,max(r))]
def predict(g):
    reads,ctrl,prev,outputs,internal=structure(g);out={'fanout_dist':dict(sorted(Counter(min(reads[i],8) for i in internal).items()))}
    for fam,r in rates.items():
        p={i:f_of(reads[i],r) for i in internal};P=lambda b:1.0 if b<0 else p.get(b,0.0)
        out[fam]=dict(count=round(sum(p.values()),1),reads_weighted=round(sum(reads[i]*p[i] for i in internal)/sum(reads[i] for i in internal),3),
            gates_both=round(sum(P(a)*P(b) for a,b in ctrl),1),increments=round(sum(P(i)*P(prev[i]) for i in internal),1))
    return out
def gen_random(seed):
    rng=random.Random(seed);g=[]
    while len(g)<M:
        a,x,y=rng.sample(range(N),3)
        if not (g and g[-1]==(a,x,y)):g.append((a,x,y))
    return g
def gen_burst(seed,q=0.67):
    """sticky targets: with prob q the next gate writes the same wire again; its intermediate versions are never read."""
    rng=random.Random(seed);g=[];cur=None
    while len(g)<M:
        a=cur if (cur is not None and rng.random()<q) else rng.choice([w for w in range(N) if w!=cur])
        x,y=rng.sample([w for w in range(N) if w!=a],2)
        if not (g and g[-1]==(a,x,y)):g.append((a,x,y));cur=a
    return g
def gen_phased(seed,hot=16,phase=100):
    rng=random.Random(seed);g=[];H=rng.sample(range(N),hot);written=set()
    while len(g)<M:
        cold=[w for w in range(N) if w not in H]
        for _ in range(phase):
            if len(g)>=M:break
            a=rng.choice(cold);x,y=rng.sample(H,2);g.append((a,x,y));written.add(a)
        pool=[w for w in cold if w in written] or cold;H=rng.sample(pool,min(hot,len(pool)));written=set()
    return g
def gen_pref(seed,w=4.0):
    """random targets; controls drawn with prob ~ 1+w*reads(current version)."""
    rng=random.Random(seed);g=[];r=[0]*N
    while len(g)<M:
        a=rng.randrange(N);wts=[0 if i==a else 1+w*r[i] for i in range(N)]
        x=rng.choices(range(N),wts)[0];wts[x]=0;y=rng.choices(range(N),wts)[0]
        g.append((a,x,y));r[x]+=1;r[y]+=1;r[a]=0
    return g
def evaluate(g,st,S):
    mask=(1<<S)-1;st=list(st)
    for a,x,y in g:st[a]^=st[x]|(~st[y]&mask)
    return st
def sac(g,S=4096,seed=1):
    rng=random.Random(seed);X=[rng.getrandbits(S) for _ in range(N)];Y=evaluate(g,X,S);mx=0;mn=1;mask=(1<<S)-1
    for i in range(N):
        Xi=list(X);Xi[i]^=mask;Yi=evaluate(g,Xi,S)
        for j in range(N):p=bin(Yi[j]^Y[j]).count('1')/S;mx=max(mx,p);mn=min(mn,p)
    return dict(flip_min=round(mn,3),flip_max=round(mx,3),max_bias=round(max(mx-0.5,0.5-mn),3))
def degree_test(g,k=12,trials=40,seed=2):
    rng=random.Random(seed);S=1<<k;mask=(1<<S)-1;axis=[sum(1<<idx for idx in range(S) if (idx>>b)&1) for b in range(k)];full=tot=0
    for _ in range(trials):
        sub=rng.sample(range(N),k);base=[(-rng.getrandbits(1))&mask for _ in range(N)]
        for b,w in enumerate(sub):base[w]=axis[b]
        full+=sum(bin(y).count('1')&1 for y in evaluate(g,base,S));tot+=N
    return round(full/tot,3)
def write_g57(path,g):
    with open(path,'w') as f:
        f.write(f'# g57 triples a x y (a ^= x OR NOT y), n={N} m={len(g)}\n');f.writelines(f'{a} {x} {y}\n' for a,x,y in g)
if __name__=='__main__':
    _,bal=load_mpmct_g57(R+'balanced_c/source.mpmct')
    designs={'old_rc':old,'balanced_rc':bal,'random_s7':gen_random(7),'burst_q50':gen_burst(7,0.5),'burst_q67':gen_burst(7,0.67),'phased_h16_p100':gen_phased(7,16,100),'pref_w4':gen_pref(7,4.0)}
    for name,g in designs.items():
        p=predict(g);print(f"## {name}: fanout dist {p['fanout_dist']}")
        for fam in('direct','affine'):print(f"   {fam}: predicted count {p[fam]['count']}, reads-weighted {p[fam]['reads_weighted']}, gates-both {p[fam]['gates_both']}, increments {p[fam]['increments']}")
        print(f"   SAC {sac(g)}  full-degree fraction(k=12) {degree_test(g)}")
    for m in (650,800,1000,1500):
        M=m;g=gen_random(11);print(f'random {m} gates: SAC {sac(g)} read-values {sum(1 for i in structure(g)[4] if structure(g)[0][i]>0)}')
