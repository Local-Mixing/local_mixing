#!/usr/bin/env python3
"""Monte Carlo: growth of a perturbation conjugated through a random g57 interior.

Mechanism under test (nonlocal pair replacement for local_mixing):
  segment  A, g_{i+1}, ..., g_{j-1}, B   with interior M
  replace  A -> [A, P]   and   B -> [Ptilde, B]   where Ptilde = M o P^{-1} o M^{-1}
  Overall function unchanged; every prefix strictly inside (i, j) is off by a
  conjugate of P; interior gates are byte-identical.

Measures |Ptilde| (generalized gates / monomials / degree / support) vs interior
length d and width n. Exact symbolic conjugation over positive-ANF gates
(canonical representation), self-verified against brute-force evaluation.

Gate model: g57 [a,x,y] flips a when (NOT x) AND y; x==y encodes an X gate.
Generalized gate: (target t, fire F); F = XOR of AND-monomials over positive
wire literals (monomial = frozenset of wires; empty monomial = constant 1).
Gates never read their own target => involutions.

Conjugation c_g(u) = g o u o g, u=(a,F), g=(t,G), t not in vars(G), a not in vars(F):
  - t in vars(F), a not in vars(G):  [(a, F[t <- t xor G])]           (1 gate)
  - a in vars(G), t not in vars(F):  [(a,F), (t, F * G1)]             (2 gates)
        where G = G0 xor x_a*G1 (G1 a-free); correction fire reads neither a nor t.
  - both (mutual):                   [g, u, g]                        (sandwich, 3)
  - neither:                         [u]                              (commutes)
Conjugation is an automorphism, so lists conjugate elementwise.
"""
import random

# ---------- positive-ANF fires ----------
# monomial: frozenset of wires; fire: frozenset of monomials

def fire_vars(F):
    s = set()
    for m in F:
        s |= m
    return s

def anf_xor(F1, F2):
    return F1 ^ F2

def anf_mul(F1, F2):
    from collections import Counter
    out = Counter()
    for m1 in F1:
        for m2 in F2:
            out[m1 | m2] += 1
    return frozenset(m for m, c in out.items() if c % 2 == 1)

def substitute(F, t, G):
    """F[x_t <- x_t xor G] in positive ANF."""
    from collections import Counter
    out = Counter()
    for m in F:
        if t not in m:
            out[m] += 1
            continue
        rest = m - {t}
        out[m] += 1  # rest * x_t
        for gm in G:  # rest * G
            out[rest | gm] += 1
    return frozenset(m for m, c in out.items() if c % 2 == 1)

def g57(a, x, y):
    """Code convention (circuit/circuit.rs, circuit/xgate.rs): flip a when x=1 OR y=0,
    i.e. fire = 1 xor (NOT x AND y) = 1 xor y xor x*y. x==y => X gate."""
    if x == y:
        return (a, frozenset([frozenset()]))  # X gate
    return (a, frozenset([frozenset(), frozenset([y]), frozenset([x, y])]))

def conj_gate(u, g):
    (a, F) = u
    (t, G) = g
    t_hits = t in fire_vars(F)
    a_hits = a in fire_vars(G)
    if t_hits and a_hits:
        return [g, u, g]  # mutual: exact sandwich fallback
    if t_hits:
        Fp = substitute(F, t, G)
        return [(a, Fp)] if Fp else []
    if a_hits:
        G1 = frozenset(m - {a} for m in G if a in m)
        corr = anf_mul(F, G1)
        out = [(a, F)]
        if corr:
            out.append((t, corr))
        return out
    return [u]

def conj_list(gates, g):
    out = []
    for u in gates:
        out.extend(conj_gate(u, g))
    return simplify(out)

def simplify(gates):
    """Drop zero fires; cancel/merge adjacent same-target gates whose fires do
    not read the other's target. Iterate to fixpoint."""
    gates = [g for g in gates if g[1]]
    changed = True
    while changed:
        changed = False
        out = []
        for g in gates:
            if out:
                (t1, F1) = out[-1]
                (t2, F2) = g
                if t1 == t2 and t1 not in fire_vars(F1) | fire_vars(F2):
                    nf = anf_xor(F1, F2)
                    out.pop()
                    if nf:
                        out.append((t1, nf))
                    changed = True
                    continue
            out.append(g)
        gates = out
    return gates

# ---------- evaluation (for self-verification) ----------
def eval_gate(u, s):
    (t, F) = u
    v = 0
    for m in F:
        prod = 1
        for w in m:
            if not (s >> w) & 1:
                prod = 0
                break
        v ^= prod
    return s ^ (v << t)

def eval_list(gates, s):
    for u in gates:
        s = eval_gate(u, s)
    return s

def verify_conj(u, g, n, trials=48):
    lst = conj_gate(u, g)
    for _ in range(trials):
        s = random.getrandbits(n)
        want = eval_gate(g, eval_gate(u, eval_gate(g, s)))
        got = eval_list(lst, s)
        assert want == got, (u, g, lst)

# ---------- experiment ----------
def rand_g57(n, rng, xgate_rate=0.05):
    a = rng.randrange(n)
    if rng.random() < xgate_rate:
        return g57(a, 0, 0)
    x = rng.randrange(n)
    while x == a:
        x = rng.randrange(n)
    y = rng.randrange(n)
    while y == a or y == x:
        y = rng.randrange(n)
    return g57(a, x, y)

def gate_support(u):
    return {u[0]} | fire_vars(u[1])

def run(n, d, trials, rng, p_mode="random", self_check=False):
    from collections import Counter
    stats = Counter()
    sizes = []
    for _ in range(trials):
        interior = [rand_g57(n, rng) for _ in range(d)]
        if p_mode == "random":
            P = [rand_g57(n, rng)]
        elif p_mode == "overlap":
            g0 = rng.choice(interior)
            w = rng.choice(sorted(gate_support(g0)))
            a = rng.randrange(n)
            while a == w:
                a = rng.randrange(n)
            y = rng.randrange(n)
            while y in (a, w):
                y = rng.randrange(n)
            P = [g57(a, w, y)]
        elif p_mode == "pair":
            a = rng.randrange(n)
            ws = rng.sample([w for w in range(n) if w != a], 3)
            P = [g57(a, ws[0], ws[1]), g57(a, ws[0], ws[2])]
        cur = [u for u in reversed(P)]  # P^{-1} (involutions)
        interactions = 0
        for g in interior:
            # true collision (Gate::collides_index): g writes a wire some element
            # reads, or an element writes a wire g reads
            (tg, G) = g
            gv = fire_vars(G)
            collides = any(tg in fire_vars(F) or t in gv for (t, F) in cur)
            if collides:
                interactions += 1
            if self_check:
                for u in cur:
                    verify_conj(u, g, n)
            cur = conj_list(cur, g)
        ngates = len(cur)
        nmono = sum(len(F) for (_t, F) in cur)
        deg = max((max((len(m) for m in F), default=0) for (_t, F) in cur), default=0)
        sup = set()
        for u in cur:
            sup |= gate_support(u)
        sizes.append((ngates, nmono, deg, len(sup), interactions))
        stats["trials"] += 1
        if interactions >= 1:
            stats["interacted"] += 1
        for b in (1, 2, 3, 4, 6, 8):
            if ngates <= b:
                stats[f"g<={b}"] += 1
            if ngates <= b and interactions >= 1:
                stats[f"g<={b}&hit"] += 1
    return stats, sizes

def pct(stats, key):
    return 100.0 * stats[key] / stats["trials"]

if __name__ == "__main__":
    rng = random.Random(20260816)
    print("self-check conjugation on n=8, d=6, 300 trials ...", flush=True)
    run(8, 6, 300, rng, p_mode="random", self_check=True)
    print("self-check replacement identity Ptilde o M o P == M ...", flush=True)
    for _ in range(300):
        n, d = 12, 8
        interior = [rand_g57(n, rng) for _ in range(d)]
        P = [rand_g57(n, rng)]
        cur = [u for u in reversed(P)]
        for g in interior:
            cur = conj_list(cur, g)
        lhs = P + interior + cur  # circuit order: P, interior, Ptilde
        for _t in range(32):
            s = random.getrandbits(n)
            assert eval_list(lhs, s) == eval_list(interior, s), "identity violated"
    print("self-checks passed.\n", flush=True)

    hdr = (f"{'n':>4} {'d':>4} {'mode':>8} {'hit%':>6} {'g<=2':>6} {'g<=4':>6} "
           f"{'g<=8':>6} {'g<=4&hit':>9} {'med_g':>6} {'p90_g':>6} "
           f"{'med_mono':>9} {'p90_deg':>8} {'med_sup':>8}")
    print(hdr)
    for n in (64, 128, 256):
        for d in (4, 8, 16, 32, 64):
            for mode in ("random", "overlap", "pair"):
                stats, sizes = run(n, d, 400, rng, p_mode=mode)
                gs = sorted(s[0] for s in sizes)
                ms = sorted(s[1] for s in sizes)
                dg = sorted(s[2] for s in sizes)
                sp = sorted(s[3] for s in sizes)
                print(f"{n:>4} {d:>4} {mode:>8} {pct(stats,'interacted'):>5.1f}% "
                      f"{pct(stats,'g<=2'):>5.1f}% {pct(stats,'g<=4'):>5.1f}% "
                      f"{pct(stats,'g<=8'):>5.1f}% {pct(stats,'g<=4&hit'):>8.1f}% "
                      f"{gs[len(gs)//2]:>6} {gs[int(len(gs)*0.9)]:>6} "
                      f"{ms[len(ms)//2]:>9} {dg[int(len(dg)*0.9)]:>8} "
                      f"{sp[len(sp)//2]:>8}", flush=True)
