"""Parametric minter for repeat-free multi-target locally-geodesic g57 identities.

Derived construction (reverse-engineered from the DB census, tools/pr_identity/
identities_m1m11_le12.csv):

  * The DB contains exactly 23 repeat-free multi-target minimal identities
    (lengths 11-12); all 23 are pairwise-inequivalent under wire-relabel+reverse,
    all are non-factoring (connected commutation graph), reduced and geodesic.
  * Each is a HUB + COUPLED-CLASS-O-SPOKE motif: one dominant "hub" target carries
    a self-cancelling class-O block (both orientations of the pairs it reads);
    each remaining target carries a class-O pair (both orientations of one pair)
    that READS the hub. Because the hub wire is written between the spoke's two
    reads, the blocks do not commute -> the atom is genuinely coupled (connected),
    not a product of independent single-target identities.

Parametric axes that mint MORE distinct verified atoms:
  1. STRUCTURE  : any of the 23 DB motifs (+ their reversals, +extra SA-mined ones).
  2. ORIENTATION: a word's reverse is its inverse, hence still the identity.
  3. WIRES      : any injective relabel of the motif's support onto n wires
                  (isomorphism -> preserves identity/repeat-free/geodesic/nonfactor).

Every emitted atom is re-verified from scratch by EXACT 2^support simulation for
all five properties before it is accepted.
"""
import sys, random, itertools, json
sys.path.insert(0,'/Users/rancanetti/Documents/local_mixing/tools/pr_identity')
import g57

def commute(g,h): return g[0] not in {h[0],h[1],h[2]} and h[0] not in {g[0],g[1],g[2]}
def connected(word):
    n=len(word);adj={i:set() for i in range(n)}
    for i in range(n):
        for j in range(i+1,n):
            if not commute(word[i],word[j]):adj[i].add(j);adj[j].add(i)
    seen=set();st=[0]
    while st:
        u=st.pop()
        if u in seen:continue
        seen.add(u);st.extend(adj[u]-seen)
    return len(seen)==n
def reduced(word):
    n=len(word)
    for i in range(n):
        for j in range(i+1,n):
            if word[i]==word[j] and all(commute(word[i],word[k]) for k in range(i+1,j)):return False
    return True
def geodesic(word):
    n=len(word)
    for i in range(n):
        for j in range(i+1,n+1):
            if i==0 and j==n:continue
            sub=word[i:j]
            if not sub:continue
            s=max(max(g) for g in sub)+1
            if g57.is_identity_exact(sub,s):return False
    return True
def is_valid_atom(w):
    """Full exact check of all 5 target properties."""
    supp=max(max(g) for g in w)+1
    if supp>16: return False
    if not g57.is_identity_exact(w,supp): return False           # 1 identity
    if len(set(w))!=len(w): return False                          # 2 repeat-free
    if len(set(g[0] for g in w))<3: return False                 # 3 multi-target
    if not reduced(w): return False                              # 4a reduced
    if not geodesic(w): return False                            # 4b locally-geodesic
    return True                                                  # (connectivity checked separately)

def canon(word):
    wires=sorted({w for g in word for w in g});best=None
    for perm in itertools.permutations(range(len(wires))):
        mp={wires[i]:perm[i] for i in range(len(wires))}
        for wd in (word,list(reversed(word))):
            s=g57.fmt_word([(mp[a],mp[x],mp[y]) for(a,x,y)in wd])
            if best is None or s<best:best=s
    return best

# ---- load base structures ----
bases=[]
found=json.load(open('/private/tmp/claude-501/-Users-rancanetti-Documents-local-mixing/5caea900-4e51-4b23-acfc-17d56b1b33c6/scratchpad/found.json'))
for L,w in found: bases.append([tuple(g) for g in w])
# add SA-mined extra classes if present
try:
    extra=json.load(open('/private/tmp/claude-501/-Users-rancanetti-Documents-local-mixing/5caea900-4e51-4b23-acfc-17d56b1b33c6/scratchpad/harvest.json'))
    for w in extra: bases.append([tuple(g) for g in w])
except Exception: pass

# canonicalize base supports to 0..s-1 and dedup by class
seen_class=set(); norm_bases=[]
for b in bases:
    cw,_=g57.canonical_support(b)
    c=canon(cw)
    if c in seen_class: continue
    seen_class.add(c); norm_bases.append(cw)
print("distinct base structures:",len(norm_bases))

# sanity: all bases valid + non-factoring
for b in norm_bases:
    assert is_valid_atom(b) and connected(b), "bad base"

# ---- mint by relabeling onto a canvas ----
CANVAS=10   # wire index pool 0..9  (keeps support small for exact sim)
rng=random.Random(20260828)
atoms=set()
nonfac=0
per_base_target=60
for b in norm_bases:
    s=g57.support_size(b)
    variants=[b, list(reversed(b))]
    got=0; tries=0
    # enumerate a spread of injective relabelings
    while got<per_base_target and tries<4000:
        tries+=1
        img=rng.sample(range(CANVAS), s)
        mp={i:img[i] for i in range(s)}
        base=variants[rng.randrange(2)]
        w=[(mp[a],mp[x],mp[y]) for (a,x,y) in base]
        key=g57.fmt_word(w)
        if key in atoms: continue
        if is_valid_atom(w) and connected(w):
            atoms.add(key); got+=1
            nonfac+=1   # connected verified

print("total distinct verified atoms:",len(atoms))
print("non-factoring:",nonfac)
lengths=[len(k.split(';')) for k in atoms]
print("min length:",min(lengths),"max length:",max(lengths))

# ---- write output ----
import os
outdir='/Users/rancanetti/Documents/local_mixing/tools/pr_identity/repeatfree_construction'
os.makedirs(outdir,exist_ok=True)
outfile=os.path.join(outdir,'analyze-db.csv')
with open(outfile,'w') as o:
    for k in sorted(atoms):
        o.write(k.replace(';','#').replace(',','~'))  # placeholder, fixed below
# rewrite in required format t,c1,c2;...
with open(outfile,'w') as o:
    for k in sorted(atoms):
        gates=[tuple(int(z) for z in tok.split(',')) if ',' in tok else (int(tok[0]),int(tok[1]),int(tok[2])) for tok in k.split(';')]
        o.write(";".join(f"{a},{x},{y}" for (a,x,y) in gates)+"\n")
print("wrote",outfile)
# final re-verify from file
nchk=0
with open(outfile) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        w=[tuple(int(z) for z in tok.split(',')) for tok in line.split(';')]
        assert is_valid_atom(w) and connected(w), "file atom invalid!"
        nchk+=1
print("re-verified from file:",nchk)
