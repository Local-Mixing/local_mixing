"""Bundle-interleave construction of repeat-free multi-target locally-geodesic
g57 identities ("atoms").

STRATEGY
  Take k independent single-target repeat-free doubled-cycle identities (as in
  atoms.mint_doubled_cycle) on k DISTINCT target wires.  Interleave their gates
  (keeping each edge's two orientation-gates adjacent for geodesic safety, but
  round-robining the pairs across cycles) so no long contiguous run belongs to
  one target.  The result is a multi-target repeat-free identity.

  Two flavours:
    (F) FACTORING bundle -- cycles use only PURE control wires (never a target),
        possibly shared.  Each cycle is independent of the others, so ANY
        interleaving is an exact identity.  The gate-commutation graph splits
        into one clique per target => factoring.  (Reported honestly.)
    (C) COUPLED bundle -- some cycle reads another cycle's TARGET wire as a
        control.  This creates cross-target NON-commuting edges (=> the conflict
        graph can be connected => NON-FACTORING), but it only stays an identity
        when the coupled control is at its ORIGINAL value at the instant each
        reading gate fires.  We schedule the reads before/after the owner block
        and EXACTLY verify every candidate -- couplings that break cancellation
        are simply rejected.

Every accepted atom is EXACTLY re-verified (all 5 properties) before it is
written.  Nothing is claimed without exact simulation.
"""
import itertools
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import g57


# ---------------------------------------------------------------------------
# exact property checks (same semantics as mint_search.py)
# ---------------------------------------------------------------------------
def repeat_free(word):
    return len(set(word)) == len(word)


def targets(word):
    return {g[0] for g in word}


def commutes(g, h):
    return (g[0] not in (h[0], h[1], h[2])) and (h[0] not in (g[0], g[1], g[2]))


def is_reduced(word):
    L = len(word)
    for i in range(L):
        for j in range(i + 1, L):
            if word[i] == word[j]:
                if all(commutes(word[i], word[k]) for k in range(i + 1, j)):
                    return False
    return True


def locally_geodesic(word, n):
    L = len(word)
    for i in range(L):
        for j in range(i + 1, L + 1):
            if i == 0 and j == L:
                continue
            if j - i < 2:
                continue
            if g57.is_identity_exact(word[i:j], n):
                return False
    return True


def non_factoring(word):
    L = len(word)
    if L == 0:
        return False
    adj = [[] for _ in range(L)]
    for i in range(L):
        for j in range(i + 1, L):
            if not commutes(word[i], word[j]):
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * L
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                stack.append(v)
    return cnt == L


def full_verify(word):
    """(ok, non_factor). ok iff all 5 core properties hold, exact."""
    if not word:
        return False, False
    n = max(w for g in word for w in g) + 1
    if n > 18:
        return False, False  # keep exact 2^n check cheap
    if not repeat_free(word):
        return False, False
    if len(targets(word)) < 3:
        return False, False
    if not g57.is_identity_exact(word, n):
        return False, False
    if not is_reduced(word):
        return False, False
    if not locally_geodesic(word, n):
        return False, False
    return True, non_factoring(word)


# ---------------------------------------------------------------------------
# doubled-cycle building blocks (edge = an adjacent orientation-pair of gates)
# ---------------------------------------------------------------------------
def cycle_edge_pairs(target, verts):
    """Return list of edge-pairs; each is [(target,u,v),(target,v,u)] for a
    consecutive edge of the cycle verts[0]-verts[1]-...-verts[-1]-verts[0]."""
    k = len(verts)
    pairs = []
    for i in range(k):
        u, v = verts[i], verts[(i + 1) % k]
        pairs.append([(target, u, v), (target, v, u)])
    return pairs


def interleave_pairs(pair_lists, rng, keep_pair_adjacent=True):
    """Round-robin-ish random interleave of several lists of edge-pairs into a
    flat gate word.  Each element of pair_lists is a list of edge-pairs (each a
    2-gate list).  If keep_pair_adjacent, the 2 gates of a pair stay contiguous
    (with a random internal orientation order); otherwise pairs may be split."""
    # flatten to a list of "units" tagged by owner, then shuffle owners fairly
    units = []  # (owner, [gates])
    for owner, plist in enumerate(pair_lists):
        for pr in plist:
            pr = pr[:]
            rng.shuffle(pr)  # orientation order within the pair
            if keep_pair_adjacent:
                units.append((owner, pr))
            else:
                units.append((owner, [pr[0]]))
                units.append((owner, [pr[1]]))
    # random interleave that avoids two consecutive units of the same owner
    # when possible (spreads each target out)
    remaining = units[:]
    rng.shuffle(remaining)
    out_units = []
    last_owner = None
    while remaining:
        cand = [u for u in remaining if u[0] != last_owner] or remaining
        pick = rng.choice(cand)
        remaining.remove(pick)
        out_units.append(pick)
        last_owner = pick[0]
    word = []
    for _, gates in out_units:
        word.extend(gates)
    return word


# ---------------------------------------------------------------------------
# (F) factoring bundle: disjoint targets, PURE (possibly shared) controls
# ---------------------------------------------------------------------------
def build_factoring(rng, n_targets, ksizes, ctrl_pool, keep_adjacent=True):
    tg = list(range(n_targets))
    ctrls = list(range(n_targets, n_targets + ctrl_pool))
    pair_lists = []
    for i in range(n_targets):
        k = ksizes[i]
        if k > len(ctrls):
            return None
        verts = rng.sample(ctrls, k)  # pure controls, may be shared across cycles
        pair_lists.append(cycle_edge_pairs(tg[i], verts))
    return interleave_pairs(pair_lists, rng, keep_adjacent)


# ---------------------------------------------------------------------------
# (C) coupled bundle: a cycle reads another cycle's target wire
# ---------------------------------------------------------------------------
def _emit_units(units, rng, keep_adjacent):
    """Round-robin interleave of (owner, [gates]) units avoiding same-owner runs."""
    remaining = units[:]
    out = []
    last = None
    while remaining:
        cand = [u for u in remaining if u[0] != last] or remaining
        pick = rng.choice(cand)
        remaining.remove(pick)
        out.append(pick)
        last = pick[0]
    w = []
    for (_i, gates) in out:
        w.extend(gates)
    return w


def _random_topo(units, preds, rng):
    """Random linear extension of a DAG. units: list of ids; preds[u] = set of
    ids that must come before u.  Returns an order (list of ids)."""
    n = len(units)
    remaining_pred = {u: set(preds[u]) for u in units}
    done = set()
    order = []
    ready = [u for u in units if not remaining_pred[u]]
    # successors index
    succ = {u: [] for u in units}
    for u in units:
        for p in preds[u]:
            succ[p].append(u)
    while ready:
        u = rng.choice(ready)
        ready.remove(u)
        order.append(u)
        done.add(u)
        for v in succ[u]:
            remaining_pred[v].discard(u)
            if not remaining_pred[v] and v not in done and v not in ready:
                ready.append(v)
    if len(order) != n:
        return None  # (shouldn't happen for a DAG)
    return order


def build_coupled(rng, n_targets, k, ctrl_pool, keep_adjacent=True):
    """ACYCLIC chain coupling t0 -> t1 -> ... -> t_{m-1}.

    Cycle i (i < m-1) uses target t_{i+1} as ONE cycle-vertex, so its "reading"
    edge-pairs read t_{i+1}; the remaining vertices are pure controls.  Cycle
    m-1 is pure.  Every gate belongs to a doubled-cycle edge-pair, so the word
    is an exact identity for ANY schedule respecting the safety partial order:

        (safety)  every READING pair of cycle i must precede every gate of
                  cycle i+1 (so t_{i+1} is still original when cycle i reads it,
                  and t_i is still original when cycle i-1 reads it).

    We emit a RANDOM linear extension of that partial order (so pure pairs of
    later cycles can float forward and break contiguous-identity prefixes),
    keeping each edge-pair's two gates adjacent.  Every candidate is then
    EXACTLY re-verified -- any schedule that isn't a true identity, or that
    leaves a proper identity window, is rejected downstream.
    """
    m = n_targets
    tg = list(range(m))
    ctrls = list(range(m, m + ctrl_pool))
    if k > len(ctrls):
        return None
    # build edge-pair units
    units = []            # id -> {owner, gates, reading(bool)}
    reading_ids = {i: [] for i in range(m)}
    owner_ids = {i: [] for i in range(m)}
    uid = 0
    for i in range(m):
        if i < m - 1:
            coupled = tg[i + 1]
            pverts = rng.sample(ctrls, k - 1)
            verts = [coupled] + pverts
            rng.shuffle(verts)
            pairs = cycle_edge_pairs(tg[i], verts)
            for pr in pairs:
                pr = pr[:]
                rng.shuffle(pr)
                is_read = any(coupled in (g[1], g[2]) for g in pr)
                units.append({"owner": i, "gates": pr, "reading": is_read})
                owner_ids[i].append(uid)
                if is_read:
                    reading_ids[i].append(uid)
                uid += 1
        else:
            pverts = rng.sample(ctrls, k)
            for pr in cycle_edge_pairs(tg[i], pverts):
                pr = pr[:]
                rng.shuffle(pr)
                units.append({"owner": i, "gates": pr, "reading": False})
                owner_ids[i].append(uid)
                uid += 1
    # precedence: reading pair of cycle i  <  every unit of cycle i+1
    ids = list(range(len(units)))
    preds = {u: set() for u in ids}
    for i in range(m - 1):
        for v in owner_ids[i + 1]:
            for r in reading_ids[i]:
                preds[v].add(r)
    order = _random_topo(ids, preds, rng)
    if order is None:
        return None
    word = []
    for uid in order:
        word.extend(units[uid]["gates"])
    return word


def fmt_atom(word):
    return ";".join(f"{a},{x},{y}" for (a, x, y) in word)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(1 << 30)
    rng = random.Random(seed)
    outdir = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(outdir, "bundle-interleave.csv")

    import time
    t0 = time.time()
    budget = float(os.environ.get("BUNDLE_BUDGET_S", "180"))

    seen = set()
    rows = []            # (word, non_factoring)
    stats = {"F_try": 0, "F_ok": 0, "C_try": 0, "C_ok": 0, "C_nf": 0}

    def consider(word, kind):
        if word is None or len(word) > 24 or not repeat_free(word):
            return
        ok, nf = full_verify(word)
        stats[f"{kind}_try"] += 1
        if not ok:
            return
        cword, _ = g57.canonical_support(word)
        key = tuple(cword)
        if key in seen:
            return
        seen.add(key)
        rows.append((word, nf))
        stats[f"{kind}_ok"] += 1
        if nf:
            stats["C_nf"] += 1

    # ---- factoring bundles (fast, high-yield) ------------------------------
    F_ATTEMPTS = int(os.environ.get("BUNDLE_F", "6000"))
    for _ in range(F_ATTEMPTS):
        nt = rng.choice([3, 3, 3, 4, 4, 5])
        ksizes = [rng.choice([3, 3, 4]) for _ in range(nt)]
        ctrl_pool = rng.choice([3, 4, 4, 5, 6])
        keep_adj = rng.random() < 0.85
        consider(build_factoring(rng, nt, ksizes, ctrl_pool, keep_adj), "F")
        if time.time() - t0 > budget * 0.35:
            break

    # ---- coupled bundles (the non-factoring workhorse) ---------------------
    while time.time() - t0 < budget:
        nt = rng.choice([3, 3, 3, 4])
        k = rng.choice([3, 3, 4])
        ctrl_pool = rng.choice([3, 4, 5, 6])
        keep_adj = rng.random() < 0.9
        consider(build_coupled(rng, nt, k, ctrl_pool, keep_adj), "C")

    with open(outfile, "w") as f:
        for (word, nf) in rows:
            f.write(fmt_atom(word) + "\n")

    total = len(rows)
    total_nf = sum(1 for (_, nf) in rows if nf)
    lens = [len(w) for (w, _) in rows] or [0]
    tgts = [len(targets(w)) for (w, _) in rows] or [0]
    print("=" * 64, file=sys.stderr)
    print(f"SEED={seed}", file=sys.stderr)
    print(f"stats={stats}", file=sys.stderr)
    print(f"TOTAL unique verified atoms: {total}", file=sys.stderr)
    print(f"  non-factoring (connected conflict graph): {total_nf}", file=sys.stderr)
    print(f"  factoring: {total - total_nf}", file=sys.stderr)
    print(f"  length range: {min(lens)}..{max(lens)}", file=sys.stderr)
    print(f"  max targets: {max(tgts)}", file=sys.stderr)
    print(f"written to {outfile}", file=sys.stderr)
    # emit a machine-readable summary line
    print(f"SUMMARY total={total} nonfac={total_nf} minlen={min(lens)} "
          f"maxtgt={max(tgts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
