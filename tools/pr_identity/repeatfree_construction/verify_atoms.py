"""Independent re-verification of search-mint.csv atoms.

Reads every line "t,c1,c2;t,c1,c2;..." and checks all 5 target properties with
EXACT g57 simulation over all 2^support states, independently of the search code:
  1. identity (exact), 2. repeat-free, 3. >=3 distinct targets,
  4. locally-geodesic (no proper contiguous subword identity) + reduced,
  5. reports non-factoring (connected gate-commutation graph).
Prints a summary and exits nonzero if ANY atom fails 1-4.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import g57


def parse(line):
    out = []
    for tok in line.strip().split(";"):
        tok = tok.strip()
        if not tok:
            continue
        a, x, y = (int(z) for z in tok.split(","))
        out.append((a, x, y))
    return out


def commutes(g, h):
    return (g[0] not in (h[0], h[1], h[2])) and (h[0] not in (g[0], g[1], g[2]))


def reduced(word):
    L = len(word)
    for i in range(L):
        for j in range(i + 1, L):
            if word[i] == word[j] and all(commutes(word[i], word[k])
                                          for k in range(i + 1, j)):
                return False
    return True


def geodesic(word, n):
    L = len(word)
    for i in range(L):
        for j in range(i + 2, L + 1):
            if i == 0 and j == L:
                continue
            if g57.is_identity_exact(word[i:j], n):
                return False
    return True


def connected(word):
    L = len(word)
    adj = [[] for _ in range(L)]
    for i in range(L):
        for j in range(i + 1, L):
            if not commutes(word[i], word[j]):
                adj[i].append(j); adj[j].append(i)
    seen = [False] * L; st = [0]; seen[0] = True; c = 1
    while st:
        u = st.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; c += 1; st.append(v)
    return c == L


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "search-mint.csv")
    ok = True
    n_total = n_nf = 0
    from collections import Counter, defaultdict
    lenhist = Counter(); tgthist = Counter()
    canon = set(); dups = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            w = parse(line)
            n = max(z for g in w for z in g) + 1
            reasons = []
            if not g57.is_identity_exact(w, n): reasons.append("not-identity")
            if len(set(w)) != len(w): reasons.append("repeat")
            if len({g[0] for g in w}) < 3: reasons.append("<3targets")
            if not reduced(w): reasons.append("not-reduced")
            if not geodesic(w, n): reasons.append("not-geodesic")
            if reasons:
                ok = False
                print(f"FAIL line {ln}: {reasons}: {line.strip()}")
                continue
            n_total += 1
            nf = connected(w)
            n_nf += 1 if nf else 0
            lenhist[len(w)] += 1
            tgthist[len({g[0] for g in w})] += 1
            cw, _ = g57.canonical_support(w)
            k = tuple(cw)
            if k in canon: dups += 1
            canon.add(k)
    print(f"verified atoms: {n_total}   non-factoring: {n_nf}   "
          f"canonical-duplicates: {dups}")
    print(f"length histogram:  {dict(sorted(lenhist.items()))}")
    print(f"targets histogram: {dict(sorted(tgthist.items()))}")
    print("ALL VERIFIED" if ok else "VERIFICATION FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
