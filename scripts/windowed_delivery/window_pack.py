#!/usr/bin/env python3
"""Joint window packer v1: commuting reorder + convex windows that swallow
proxy-carrier DAG intervals, then ground-truth verification.

Pipeline:
  1. fingerprint pass (A ref, B candidate) -> proxy spans (birth b, next
     t-writer o, a_id), as in waist_census.py;
  2. for spans <= span_cap: bidirectional taint -> interval gate SET
     I(b,o) (abort if > W);
  3. epoch-DAG list scheduling (sound for the MPMCT commutation relation:
     same-target writers commute, readers pin against ALL later writers
     of a read wire -> reader ready iff writes_done[c] == its epoch;
     writer ready iff readers_done[t] >= #readers of earlier epochs),
     with priorities: finish committed intervals > start smallest
     fitting interval > filler (fewest interval memberships, earliest
     original position); windows close at W gates (or when idle with
     load >= W/2);
  4. emit reordered circuit + window boundaries; re-evaluate from
     scratch: function check, re-derive proxy spans in the EMITTED
     order, count distinct A values alive across any boundary.

usage: window_pack.py A.mpmct1 B.mpmct1 W [span_cap] [label] [out.mpmct1]
"""
import sys, random, hashlib, heapq
from collections import defaultdict
from array import array

LANES = 16
NBITS = LANES * 64
MASK = (1 << NBITS) - 1
NBYTES = NBITS // 8
SEED = 20261004

def load(path):
    with open(path) as f:
        hdr = f.readline().split()
        n_hdr = int(hdr[1])
        gates = []
        maxw = 0
        for line in f:
            q = line.split()
            if not q:
                continue
            t, comp, k = int(q[0]), int(q[1]), int(q[2])
            cps = tuple(int(q[3 + 2*i]) for i in range(k))
            pol = tuple(int(q[4 + 2*i]) for i in range(k))
            m = max((t,) + cps)
            if m > maxw:
                maxw = m
            gates.append((t, comp, cps, pol))
    return max(n_hdr, maxw + 1), gates

def wire_class(w):
    return "data" if w < 64 else ("mid" if w < 128 else "band")

def eval_circuit(gates, st, on_born):
    for i, (t, comp, cps, pol) in enumerate(gates):
        acc = MASK
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ MASK
        if comp:
            acc ^= MASK
        v = st[t] ^ acc
        st[t] = v
        on_born(i, t, v ^ MASK if v & 1 else v)

def proxy_spans(A, B, n, na, nb, inputs):
    index, a_wire = {}, {}
    def on_a(i, t, fp):
        if fp == 0:
            return
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        if d not in index:
            index[d] = i
            a_wire[i] = t
    st = list(inputs[:na]) + [0] * (n - na)
    eval_circuit(A, st, on_a)
    final_a = list(st)
    pending, spans = {}, []
    def on_b(i, t, fp):
        prev = pending.pop(t, None)
        if prev is not None:
            spans.append((prev[0], i, prev[1]))
        if fp == 0:
            return
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        a_id = index.get(d)
        if a_id is not None:
            pending[t] = (i, a_id)
    st = list(inputs[:nb]) + [0] * (n - nb)
    eval_circuit(B, st, on_b)
    forced = set(a for _, a in pending.values())
    agree = sum(1 for w in range(min(na, nb)) if st[w] == final_a[w])
    return index, a_wire, spans, forced, agree

def interval_set(gates, b, e, n, cap):
    TW = bytearray(n); TR = bytearray(n)
    tb, _, cb, _ = gates[b]
    TW[tb] = 1
    for c in cb: TR[c] = 1
    fwd = bytearray(e - b - 1)
    for idx in range(b + 1, e):
        t, _, cps, _ = gates[idx]
        tainted = TR[t] != 0
        if not tainted:
            for c in cps:
                if TW[c]:
                    tainted = True
                    break
        if tainted:
            fwd[idx - b - 1] = 1
            TW[t] = 1
            for c in cps: TR[c] = 1
    TWb = bytearray(n); TRb = bytearray(n)
    to, _, co, _ = gates[e]
    TRb[to] = 1
    for c in co: TWb[c] = 1
    members = []
    for idx in range(e - 1, b, -1):
        if not fwd[idx - b - 1]:
            continue
        t, _, cps, _ = gates[idx]
        tainted = TWb[t] != 0
        if not tainted:
            for c in cps:
                if TRb[c]:
                    tainted = True
                    break
        if tainted:
            members.append(idx)
            if len(members) + 2 > cap:
                return None
            TRb[t] = 1
            for c in cps: TWb[c] = 1
    members.append(b)
    members.append(e)
    return members

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    W = int(sys.argv[3])
    span_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    label = sys.argv[5] if len(sys.argv) > 5 else f"pack-W{W}"
    out_path = sys.argv[6] if len(sys.argv) > 6 else None

    na, A = load(a_path)
    nb, B = load(b_path)
    n = max(na, nb)
    rng = random.Random(SEED)
    inputs = [rng.getrandbits(NBITS) for _ in range(n)]

    index, a_wire, spans, forced, agree = proxy_spans(A, B, n, na, nb, inputs)
    m = len(B)
    carried = set(a for _, _, a in spans) | forced
    print(f"{label}: B={m} gates; {len(spans)} interior spans, {len(carried)} carried values "
          f"({len(forced)} output-forced); input function check {agree}/{min(na,nb)}")

    # --- intervals -----------------------------------------------------
    intervals = []          # (members list, a_id, b, e)
    skipped_long = aborted = 0
    for b, e, a in spans:
        if e - b + 1 > span_cap:
            skipped_long += 1
            continue
        mem = interval_set(B, b, e, n, W)
        if mem is None:
            aborted += 1
            continue
        intervals.append((mem, a, b, e))
    print(f"intervals <= W={W}: {len(intervals)} "
          f"(span>{span_cap}: {skipped_long}, interval>W: {aborted})")
    member_of = defaultdict(list)       # gate -> interval ids
    for iid, (mem, a, b, e) in enumerate(intervals):
        for g in mem:
            member_of[g].append(iid)

    # --- epoch bookkeeping for scheduling -------------------------------
    # per gate: read conditions (wire, epoch), write condition (wire, threshold)
    writes_seen = [0] * n               # epochs so far per wire
    readers_before = [0] * n            # readers of epochs < current, per wire
    read_cond = [None] * m              # list of (wire, epoch)
    write_cond = [None] * m             # (wire, reader-threshold)
    read_bucket = defaultdict(list)     # (wire, epoch) -> gates waiting
    write_bucket = defaultdict(list)    # (wire, threshold) -> gates waiting
    unmet = array("i", bytes(4 * m))
    for i, (t, comp, cps, pol) in enumerate(B):
        rc = [(c, writes_seen[c]) for c in cps]
        wc = (t, readers_before[t])
        read_cond[i] = rc
        write_cond[i] = wc
        cnt = 0
        for c, ep in rc:
            if ep > 0:                  # epoch 0 is satisfied at start
                read_bucket[(c, ep)].append(i)
                cnt += 1
        if wc[1] > 0:
            write_bucket[wc].append(i)
            cnt += 1
        unmet[i] = cnt
        for c in cps:
            readers_before[c] += 1
        writes_seen[t] += 1
    nspans_total = len(spans)

    # --- scheduler: drain + burst, intervals in o-position order ----------
    writes_done = [0] * n
    readers_done = [0] * n
    scheduled = bytearray(m)
    isready = bytearray(m)
    order = []
    boundaries = []
    win_load = 0
    win_id = 0
    committed = {}                      # iid -> remaining members (current window)
    dead = bytearray(len(intervals))
    started = bytearray(len(intervals))
    hidden_iids = []
    isize = [len(mem) for mem, _, _, _ in intervals]
    imem = [sorted(mem) for mem, _, _, _ in intervals]
    prot = array("i", bytes(4 * m))     # living un-started intervals containing g
    for mem in imem:
        for g in mem:
            prot[g] += 1
    filler = []                         # unprotected ready gates, by position
    anyready = []                       # all ready gates, by position

    def on_ready(g):
        isready[g] = 1
        heapq.heappush(anyready, g)
        if prot[g] == 0:
            heapq.heappush(filler, g)

    def satisfy(g):
        unmet[g] -= 1
        if unmet[g] == 0:
            on_ready(g)

    for i in range(m):
        if unmet[i] == 0:
            on_ready(i)

    def close_window():
        nonlocal win_load, win_id, committed
        for iid, rem in committed.items():
            if rem > 0:
                dead[iid] = 1
        committed = {}
        boundaries.append(len(order))
        win_load = 0
        win_id += 1

    def schedule(g):
        nonlocal win_load
        scheduled[g] = 1
        order.append(g)
        win_load += 1
        for iid in member_of.get(g, ()):
            if dead[iid]:
                continue
            if not started[iid]:
                started[iid] = 1
                committed[iid] = isize[iid]
                for h in imem[iid]:
                    prot[h] -= 1
                    if prot[h] == 0 and isready[h] and not scheduled[h]:
                        heapq.heappush(filler, h)
            if iid in committed:
                committed[iid] -= 1
                if committed[iid] == 0:
                    hidden_iids.append(iid)
                    del committed[iid]
            else:
                dead[iid] = 1
        t, _, cps, _ = B[g]
        writes_done[t] += 1
        for h in read_bucket.pop((t, writes_done[t]), ()):
            satisfy(h)
        for c in cps:
            readers_done[c] += 1
            for h in write_bucket.pop((c, readers_done[c]), ()):
                satisfy(h)
        if win_load >= W:
            close_window()

    by_end = sorted(range(len(intervals)), key=lambda i: intervals[i][3])
    progress_report = max(1, len(by_end) // 10)
    for nproc, iid in enumerate(by_end):
        if dead[iid] or started[iid]:
            continue
        e_x = intervals[iid][3]
        # drain: everything unprotected below e_x
        while filler:
            g = filler[0]
            if scheduled[g] or prot[g] > 0:
                heapq.heappop(filler)
                continue
            if g > e_x:
                break
            heapq.heappop(filler)
            schedule(g)
        if dead[iid] or started[iid]:
            continue                    # a drain gate burned it (shared member)
        if W - win_load < isize[iid]:
            close_window()
        for g in imem[iid]:
            while not isready[g]:
                # padding: lowest ready gate (may burn another interval)
                while anyready and (scheduled[anyready[0]] or anyready[0] == g):
                    heapq.heappop(anyready)
                if not anyready:
                    raise RuntimeError("burst stuck")
                schedule(heapq.heappop(anyready))
            if not scheduled[g]:
                schedule(g)
        if (nproc + 1) % progress_report == 0:
            print(f"  processed {nproc+1}/{len(by_end)} intervals, "
                  f"{len(order)}/{m} gates, windows={win_id}, "
                  f"hidden={len(hidden_iids)}", flush=True)
    # tail: drain the rest
    while len(order) < m:
        while anyready and scheduled[anyready[0]]:
            heapq.heappop(anyready)
        if not anyready:
            for i in range(m):
                if not scheduled[i] and unmet[i] == 0 and i not in anyready:
                    heapq.heappush(anyready, i)
            if not anyready:
                raise RuntimeError("scheduler stuck: no ready gates")
            continue
        schedule(heapq.heappop(anyready))
    if win_load:
        close_window()
    boundaries = [b for b in boundaries if 0 < b < m]
    print(f"scheduled: {win_id} windows, {len(hidden_iids)}/{len(intervals)} "
          f"interval-spans hidden by construction ({nspans_total} spans total)")

    # --- emit + verify ---------------------------------------------------
    newB = [B[g] for g in order]
    if out_path:
        with open(out_path, "w") as f:
            f.write(f"mpmct1 {nb} {m}\n")
            for t, comp, cps, pol in newB:
                parts = [str(t), str(comp), str(len(cps))]
                for c, p in zip(cps, pol):
                    parts.append(str(c)); parts.append(str(p))
                f.write(" ".join(parts) + "\n")
        with open(out_path + ".bounds", "w") as f:
            f.write("\n".join(str(b) for b in boundaries) + "\n")
    _, _, spans2, forced2, agree2 = proxy_spans(A, newB, n, na, nb, inputs)
    print(f"VERIFY: emitted function check {agree2}/{min(na,nb)} wires "
          f"({'OK' if agree2 == min(na,nb) else 'FAIL'})")
    bset = sorted(boundaries)
    import bisect
    exposed = set(forced2)
    per_cls_car = defaultdict(set)
    for b, e, a in spans2:
        per_cls_car[wire_class(a_wire[a])].add(a)
        i = bisect.bisect_left(bset, b + 1)
        if i < len(bset) and bset[i] <= e:
            exposed.add(a)
    for a in forced2:
        per_cls_car[wire_class(a_wire[a])].add(a)
    carried2 = set(a for _, _, a in spans2) | forced2
    print(f"RESULT: carried in emitted order {len(carried2)}; "
          f"distinct A values exposed at window boundaries: {len(exposed)} "
          f"({100*len(exposed)/max(len(carried2),1):.1f}% of carried, "
          f"{100*len(exposed)/max(len(index),1):.1f}% of all distinct A)")
    for cls in ("data", "mid", "band"):
        car = per_cls_car[cls]
        ex = sum(1 for a in exposed if wire_class(a_wire[a]) == cls)
        if car:
            print(f"  {cls}: exposed {ex}/{len(car)} ({100*ex/len(car):.1f}%)")
    # window footprints (ANF feasibility proxy)
    fps, sups = [], []
    lo = 0
    for cut in bset + [m]:
        ws = set()
        sup = {}                        # wire -> bitmask over window-input wires
        for t, _, cps, _ in newB[lo:cut]:
            ws.add(t); ws.update(cps)
            acc = sup.get(t, 0)
            if acc == 0:
                acc = 1 << t            # own current value counts as an input
            for c in cps:
                acc |= sup.get(c, 1 << c)
            sup[t] = acc
        fps.append(len(ws))
        sups.append(max((bin(v).count("1") for v in sup.values()), default=0))
        lo = cut
    fps.sort(); sups.sort()
    k = len(fps)
    print(f"window footprint (wires): med={fps[k//2]} p90={fps[9*k//10]} max={fps[-1]}; "
          f"max output support/window: med={sups[k//2]} p90={sups[9*k//10]} max={sups[-1]}; "
          f"windows={k}")

if __name__ == "__main__":
    main()
