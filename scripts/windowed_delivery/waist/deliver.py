#!/usr/bin/env python3
"""deliver.py: the delivery stage (packing + unpacking with lattice refinement), Python reference.

Takes the packer's repaired circuit and bounds, unpacks every window by resynthesis, and treats a
window that does not resynthesize as a packing defect: it is cut in two at an exposure-aware cut,
both halves are re-emitted into the deliverable and unpacked separately, recursively, until every
window unpacks (docs/DELIVERY_STAGE.md). Orchestration only: the synthesizer (`tdp_unpacking run
--windows` or the standalone `anf_synth`) and the C-view scan (`hmap_affine`) are the existing
binaries, exactly as the packer chain already calls hmap_affine.

usage: deliver.py --rep rep.mpmct1 --bounds rep.bounds --source-c C.g57 --n N --out DIR
                  [--body deliverable.bin] [--anf-only] [--time-limit 30] [--restarts 16] [--beam 6]
                  [--max-rounds 16] [--threads T] [--seed 1]
                  [--synth tdp_unpacking|anf_synth] [--synth-bin PATH] [--hmap PATH] [--canon-bin PATH]
                  [--budget 65536] [--label LABEL]
Exit: 0 ok (status=ok or verbatim_used), 2 usage, 3 input, 4 a check failed, 7 I/O.
"""
import argparse, bisect, os, random, shutil, struct, subprocess, sys, time, zlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # scripts/windowed_delivery/anf_compose.py
from anf_compose import window_anf                 # noqa: E402

LANES = 1024
M = (1 << LANES) - 1
MAGIC = b"LMTDPANF"
ALPH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
DD = {c: i for i, c in enumerate(ALPH)}


# ---------------------------------------------------------------- circuits
def load_mpmct1(path):
    """-> (header_wires, gates); gate = (t, comp, ctrls) with ctrls a tuple of (wire, polarity)."""
    with open(path) as f:
        hdr = f.readline().split()
        if not hdr or hdr[0] != "mpmct1":
            raise SystemExit(f"input: {path}: not an mpmct1 file")
        n = int(hdr[1]); gates = []
        for line in f:
            q = line.split()
            if not q:
                continue
            k = int(q[2])
            gates.append((int(q[0]), int(q[1]), tuple((int(q[3 + 2 * i]), int(q[4 + 2 * i])) for i in range(k))))
    return n, gates


def gate_line(g):
    t, comp, ctrls = g
    return f"{t} {comp} {len(ctrls)}" + "".join(f" {c} {p}" for c, p in ctrls) + "\n"


def to_compose(gates):
    """anf_compose's gate tuple (t, comp, ctrl_tuple, pol_tuple)."""
    return [(t, comp, tuple(c for c, _ in ctrls), tuple(p for _, p in ctrls)) for t, comp, ctrls in gates]


def load_g57(path):
    body = open(path).read().strip().strip(';'); out = []
    for tok in body.split(';'):
        if not tok:
            continue
        ws = []; tl = 0
        for ch in tok:
            if ch == '~':
                tl += 1
            else:
                ws.append(83 * tl + DD[ch]); tl = 0
        if len(ws) != 3:
            raise SystemExit(f"input: {path}: bad g57 token {tok!r}")
        out.append(tuple(ws))
    return out


def load_bounds(path, m):
    return sorted({int(x) for x in open(path).read().split() if 0 < int(x) < m})


def sim(gates, st):
    """Bit-sliced simulation in place; st[w] holds LANES states of wire w."""
    for t, comp, ctrls in gates:
        acc = M
        for c, p in ctrls:
            acc &= st[c] if p else st[c] ^ M
        if comp:
            acc ^= M
        st[t] ^= acc


def sim_c(C, x):
    st = list(x)
    for a, u, v in C:
        st[a] ^= st[u] | (st[v] ^ M)
    return st


# ---------------------------------------------------------------- canonical order (gate_order.rs)
def order_key(g):
    t, comp, ctrls = g
    if comp and len(ctrls) == 2:
        (w0, p0), (w1, p1) = ctrls
        if p0 != p1:
            x, y = (w0, w1) if not p0 else (w1, w0)
            return (0, t, 0, ((x, 0), (y, 0)))
    return (1, t, comp, ctrls)


def collides(a, b):
    ta, _, ca = a; tb, _, cb = b
    if not (any(c == tb for c, _ in ca) or any(c == ta for c, _ in cb)):
        return False
    if a[1] or b[1]:
        return True
    i = j = 0
    while i < len(ca) and j < len(cb):
        (wa, pa), (wb, pb) = ca[i], cb[j]
        if wa == wb:
            if pa != pb:
                return False
            i += 1; j += 1
        elif wa < wb:
            i += 1
        else:
            j += 1
    return True


def canonicalize(gates):
    g = list(gates); keys = [order_key(x) for x in g]
    for i in range(1, len(g)):
        pos = None; j = i
        while j > 0:
            j -= 1
            if collides(g[i], g[j]):
                break
            elif keys[j] > keys[i]:
                pos = j
        if pos is not None:
            g[pos:i + 1] = [g[i]] + g[pos:i]
            keys[pos:i + 1] = [keys[i]] + keys[pos:i]
    return g


# ---------------------------------------------------------------- deliverable blobs (anf_dump_eval.py layout)
def encode_blob(changed):
    wires = set()
    for p in changed.values():
        for m in p:
            mm = m
            while mm:
                low = mm & -mm; wires.add(low.bit_length() - 1); mm ^= low
    wl = sorted(wires); idx = {w: i for i, w in enumerate(wl)}
    fb = (max(len(wl), 1) + 7) // 8
    b = bytearray(struct.pack('<HH', len(wl), len(changed)))
    for w in wl:
        b += struct.pack('<H', w)
    for w, poly in sorted(changed.items()):
        b += struct.pack('<HI', w, len(poly))
        for m in sorted(poly):
            local = 0; mm = m
            while mm:
                low = mm & -mm; local |= 1 << idx[low.bit_length() - 1]; mm ^= low
            b += local.to_bytes(fb, 'little')
    return bytes(b)


def split_body(body):
    blobs = []; pos = 0
    while pos < len(body):
        (length,) = struct.unpack_from('<I', body, pos); pos += 4
        blobs.append(body[pos:pos + length]); pos += length
    if pos != len(body):
        raise SystemExit("input: deliverable body: last blob runs past the end")
    return blobs


def container(blobs, source_wires, state_wires):
    body = b''.join(struct.pack('<I', len(x)) + x for x in blobs)
    h = MAGIC + struct.pack('<HHIIIIIQI', 1, 48, 0, source_wires, state_wires, 1, len(blobs), len(body), zlib.crc32(body) & 0xffffffff)
    h += struct.pack('<I', zlib.crc32(h) & 0xffffffff)
    return h + body


def eval_blob(blob, st):
    """Window map on bit-sliced state `st` (in place), from the blob alone."""
    f, c = struct.unpack_from('<HH', blob, 0); pos = 4
    sup = list(struct.unpack_from(f'<{f}H', blob, pos)); pos += 2 * f
    fb = (max(f, 1) + 7) // 8
    snap = [st[w] for w in sup]
    new = {}
    for _ in range(c):
        w, cnt = struct.unpack_from('<HI', blob, pos); pos += 6
        v = 0
        for _ in range(cnt):
            mask = int.from_bytes(blob[pos:pos + fb], 'little'); pos += fb
            acc = M; mm = mask
            while mm:
                low = mm & -mm; acc &= snap[low.bit_length() - 1]; mm ^= low
            v ^= acc
        new[w] = v
    for w, v in new.items():
        st[w] = v


# ---------------------------------------------------------------- synthesizer backends
def run_synth(a, round_dir, deliverable, bounds_path, sel):
    """Synthesize the windows `sel` of the current lattice; -> (rows by idx, gate lines of the output)."""
    os.makedirs(round_dir, exist_ok=True)
    wl = ",".join(map(str, sel))
    if a.synth == "anf_synth":
        cmd = [a.synth_bin, "--packed", a.rep, "--bounds", bounds_path, "--out", round_dir, "--threads", str(a.threads),
               "--time-limit", str(a.time_limit), "--restarts", str(a.restarts), "--beam", str(a.beam), "--windows", wl]
        out_circ = os.path.join(round_dir, "synth.mpmct1")
    else:
        cmd = [a.synth_bin, "run", "--deliverable", deliverable, "--packed", a.rep, "--bounds", bounds_path,
               "--out-dir", round_dir, "--windows", wl, "--threads", str(a.threads), "--time-limit", str(a.time_limit),
               "--restarts", str(a.restarts), "--beam", str(a.beam), "--seed", str(a.seed), "--label", a.label]
        out_circ = os.path.join(round_dir, "unpacked.mpmct1")
    with open(os.path.join(round_dir, "synth.log"), "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"input: synthesizer exit {r.returncode} in {round_dir} (see synth.log)")
    rows = {}
    with open(os.path.join(round_dir, "windows.tsv")) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            if line.strip():
                r_ = dict(zip(hdr, line.rstrip("\n").split("\t"))); rows[int(r_['idx'])] = r_
    with open(out_circ) as f:
        f.readline(); lines = [l for l in f if l.strip()]
    return rows, lines


# ---------------------------------------------------------------- cut choice (anf_split2.py)
def choose_cuts(a, failed, outs, round_dir):
    """failed: list of (lo, hi). -> dict (lo,hi) -> (cut, clean)."""
    cands = {}
    for lo, hi in failed:
        k = 32
        cands[(lo, hi)] = sorted({lo + (hi - lo) * i // (k + 1) for i in range(1, k + 1)} - {lo, hi})
    allc = sorted({c for v in cands.values() for c in v})
    bad = set()
    if allc:
        cf = os.path.join(round_dir, "cut_cands.txt"); open(cf, "w").write("\n".join(map(str, allc)) + "\n")
        env = dict(os.environ, EXPOSURE_ALLCUTS=os.path.join(round_dir, "cut_cands.allcuts.tsv"))
        with open(os.path.join(round_dir, "cut_cands.log"), "w") as log:
            r = subprocess.run([a.hmap, "--c", a.source_c, "--c-format", "g57", "--g", a.rep, "--n", str(a.n), "--exposure",
                                "--c-step", "1", "--g-cuts", cf, "--out", os.path.join(round_dir, "cut_cands")], env=env, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise SystemExit(f"input: hmap_affine exit {r.returncode} in {round_dir}")
        for l in open(env["EXPOSURE_ALLCUTS"]):
            if l.startswith('#'):
                continue
            g, w, c, wt = map(int, l.split())
            if g not in outs:
                bad.add(c)
    out = {}
    for (lo, hi), cs in cands.items():
        good = [c for c in cs if c not in bad]; mid = (lo + hi) // 2
        if good:
            out[(lo, hi)] = (min(good, key=lambda x: (abs(x - mid), x)), True)
        else:
            out[(lo, hi)] = (mid, False)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True); ap.add_argument("--bounds", required=True)
    ap.add_argument("--source-c", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--body")
    ap.add_argument("--anf-only", action="store_true")
    ap.add_argument("--time-limit", type=float, default=30.0); ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--beam", type=int, default=6); ap.add_argument("--max-rounds", type=int, default=16)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--synth", choices=["tdp_unpacking", "anf_synth"], default="tdp_unpacking")
    ap.add_argument("--synth-bin"); ap.add_argument("--hmap", default=os.path.expanduser("~/code/local_mixing/target/release/hmap_affine"))
    ap.add_argument("--canon-bin", help="tdp_unpacking binary for `canon IN OUT`; without it the Python canonicalizer runs")
    ap.add_argument("--budget", type=int, default=65536); ap.add_argument("--label", default="tdp")
    a = ap.parse_args()
    if a.synth_bin is None:
        a.synth_bin = a.synth
    if a.n <= 0 or a.time_limit < 0 or a.restarts < 1 or a.max_rounds < 0:
        raise SystemExit("usage: bad numeric parameter")
    t0 = time.time()
    nw, gates = load_mpmct1(a.rep); m = len(gates)
    if 2 * a.n > nw:
        raise SystemExit(f"usage: --n {a.n} does not fit a {nw}-wire state")
    C = load_g57(a.source_c)
    outs = set({a_: g for g, (a_, _, _) in enumerate(C)}.values())
    cg = to_compose(gates)
    bounds = load_bounds(a.bounds, m)
    os.makedirs(a.out, exist_ok=True)

    # round-0 blobs: the packer's body if given, else composed here
    cuts = [0] + bounds + [m]
    if a.body:
        blobs = split_body(open(a.body, "rb").read())
        if len(blobs) != len(cuts) - 1:
            raise SystemExit(f"input: body has {len(blobs)} windows, bounds give {len(cuts) - 1}")
    else:
        blobs = []
        for lo, hi in zip(cuts, cuts[1:]):
            changed, _ = window_anf(cg, lo, hi, 1 << 30); blobs.append(encode_blob(changed))
    wins = list(zip(cuts, cuts[1:]))          # lattice as (lo, hi) in order; blobs parallel

    def write_deliverable(path):
        with open(path + ".tmp", "wb") as f:
            f.write(container(blobs, a.n, nw))
        os.replace(path + ".tmp", path)

    def write_bounds(path):
        with open(path, "w") as f:
            f.write("".join(f"{lo}\n" for lo, _ in wins[1:]))

    write_bounds(os.path.join(a.out, "bounds.0")); write_deliverable(os.path.join(a.out, "deliverable.0.tdp4"))
    if a.anf_only:
        shutil.copyfile(os.path.join(a.out, "deliverable.0.tdp4"), os.path.join(a.out, "deliverable.tdp4"))
        shutil.copyfile(os.path.join(a.out, "bounds.0"), os.path.join(a.out, "bounds.final"))
        with open(os.path.join(a.out, "delivery.report"), "w") as f:
            f.write(f"delivery_report=1\nstatus=anf_only\nwindows_initial={len(wins)}\nwindows_final={len(wins)}\nrounds=0\ngates_packed={m}\n")
        print(f"anf-only: {len(wins)} windows, {time.time() - t0:.1f}s"); return

    # ---- the loop
    pieces = {}          # (lo, hi) -> (round, row, gate lines)   converged
    verbatim = {}        # (lo, hi) -> reason                      1-gate failures / max_rounds
    pending = list(range(len(wins)))
    rounds = []; splits = 0; unclean = 0; over_budget = 0; rnd = 0
    while pending:
        rd = os.path.join(a.out, f"round_{rnd}")
        bpath = os.path.join(a.out, f"bounds.{rnd}"); dpath = os.path.join(a.out, f"deliverable.{rnd}.tdp4")
        if rnd > 0:
            write_bounds(bpath); write_deliverable(dpath)
        ts = time.time()
        rows, lines = run_synth(a, rd, dpath, bpath, pending)
        failed = []
        for i in pending:
            r = rows.get(i)
            if r is None:
                raise SystemExit(f"input: synthesizer wrote no row for window {i} in round {rnd}")
            lo, hi = wins[i]
            if int(r['orig_lo']) != lo or int(r['orig_hi']) != hi:
                raise SystemExit(f"internal: window {i} is [{lo},{hi}) but the synthesizer reports [{r['orig_lo']},{r['orig_hi']})")
            o, g = int(r['syn_offset']), int(r['syn_gates'])
            if r['status'] == 'ok':
                pieces[(lo, hi)] = (rnd, r, lines[o:o + g])
            else:
                failed.append(i)
        nconv = len(pending) - len(failed)
        # ---- refine
        new_pending = []
        if failed and rnd + 1 <= a.max_rounds:
            splittable = [wins[i] for i in failed if wins[i][1] - wins[i][0] >= 2]
            choice = choose_cuts(a, splittable, outs, rd) if splittable else {}
            new_wins = []; new_blobs = []; new_idx = {}
            for i, (lo, hi) in enumerate(wins):
                if i in failed and (lo, hi) in choice:
                    c, clean = choice[(lo, hi)]; splits += 1; unclean += 0 if clean else 1
                    for (x, y) in ((lo, c), (c, hi)):
                        changed, tot = window_anf(cg, x, y, 1 << 30)
                        if tot > a.budget:
                            over_budget += 1
                        new_idx[(x, y)] = len(new_wins); new_wins.append((x, y)); new_blobs.append(encode_blob(changed))
                else:
                    if i in failed:
                        verbatim[(lo, hi)] = "1-gate window"
                    new_wins.append((lo, hi)); new_blobs.append(blobs[i])
            wins, blobs = new_wins, new_blobs
            new_pending = sorted(new_idx.values())
        elif failed:
            for i in failed:
                verbatim[wins[i]] = f"max_rounds {a.max_rounds}"
        rounds.append((rnd, len(pending), nconv, len(failed), len(new_pending) // 2, time.time() - ts))
        print(f"round {rnd}: pending {len(pending)}, converged {nconv}, failed {len(failed)}, split {len(new_pending) // 2}, {time.time() - ts:.1f}s", flush=True)
        pending = new_pending; rnd += 1

    # ---- final lattice, deliverable, circuit
    write_bounds(os.path.join(a.out, "bounds.final")); write_deliverable(os.path.join(a.out, "deliverable.tdp4"))
    out_lines = []; tsv = []; nb = []; off = 0; nverb = 0
    for i, (lo, hi) in enumerate(wins):
        if (lo, hi) in pieces:
            r0, r, chunk = pieces[(lo, hi)]
            st = (r0, r['status'], r['synth_status'], r['seconds'], r['monomials'], r['changed_wires'], r['input_vars'], r['max_degree'])
        else:
            chunk = [gate_line(g) for g in gates[lo:hi]]; nverb += 1
            st = (-1, 'verbatim', verbatim.get((lo, hi), 'verbatim'), '0', '0', '0', '0', '0')
        if off > 0:
            nb.append(off)
        tsv.append(f"{i}\t{lo}\t{hi}\t{st[0]}\t{hi - lo}\t{len(chunk)}\t{st[1]}\t{st[2]}\t{st[3]}\t{st[4]}\t{st[5]}\t{st[6]}\t{st[7]}\n")
        out_lines.extend(chunk); off += len(chunk)
    concat = [tuple_of(l) for l in out_lines]
    # canonical order
    if a.canon_bin:
        tmp_in = os.path.join(a.out, "concat.mpmct1"); tmp_out = os.path.join(a.out, "unpacked.mpmct1.tmp")
        with open(tmp_in, "w") as f:
            f.write(f"mpmct1 {nw} {len(out_lines)}\n"); f.writelines(out_lines)
        r = subprocess.run([a.canon_bin, "canon", tmp_in, tmp_out])
        if r.returncode != 0:
            raise SystemExit(f"input: canon exit {r.returncode}")
        _, canon = load_mpmct1(tmp_out); os.remove(tmp_in)
    else:
        if len(concat) > 300_000:
            print(f"warning: canonicalizing {len(concat)} gates in Python; pass --canon-bin for speed", file=sys.stderr)
        canon = canonicalize(concat)
        with open(os.path.join(a.out, "unpacked.mpmct1.tmp"), "w") as f:
            f.write(f"mpmct1 {nw} {len(canon)}\n"); f.writelines(gate_line(g) for g in canon)

    # ---- checks
    checks = {}
    rng = random.Random(20261012)
    st0 = [rng.getrandbits(LANES) for _ in range(nw)]
    ref = list(st0); sim(gates, ref)
    s1 = list(st0); sim(concat, s1); checks['equiv_concat'] = (s1 == ref)
    s2 = list(st0); sim(canon, s2); checks['equiv_canonical'] = (s2 == ref)
    # new windows' blobs evaluate like the packed gates (the packer checked the round-0 blobs)
    new_ok = True
    orig_wins = set(zip(cuts, cuts[1:]))
    for (lo, hi), blob in zip(wins, blobs):
        if (lo, hi) in orig_wins:
            continue
        sb = list(st0); eval_blob(blob, sb); sg = list(st0); sim(gates[lo:hi], sg)
        if sb != sg:
            new_ok = False; break
    checks['new_windows_blobs'] = new_ok
    # zero slice: x on 0..n, zeros elsewhere -> wires n..2n hold C(x)
    xs = [rng.getrandbits(LANES) for _ in range(a.n)]
    z = xs + [0] * (nw - a.n); sim(canon, z); cz = sim_c(C, xs)
    checks['zero_slice'] = all(z[a.n + i] == cz[i] for i in range(a.n))
    ok = all(checks.values())
    status = "ok" if ok and nverb == 0 else ("verbatim_used" if ok else "failed")
    with open(os.path.join(a.out, "windows.tsv"), "w") as f:
        f.write("idx\tlo\thi\tround\torig_gates\tsyn_gates\tstatus\tsynth_status\tseconds\tmonomials\tchanged_wires\tinput_vars\tmax_degree\n"); f.writelines(tsv)
    with open(os.path.join(a.out, "rounds.tsv"), "w") as f:
        f.write("round\tpending\tconverged\tfailed\tsplit\tseconds\n")
        for r in rounds:
            f.write("\t".join(str(x) for x in r[:5]) + f"\t{r[5]:.1f}\n")
    with open(os.path.join(a.out, "delivery.report"), "w") as f:
        f.write(f"delivery_report=1\nstatus={status}\nrounds={len(rounds)}\nwindows_initial={len(cuts) - 1}\nwindows_final={len(wins)}\n")
        f.write(f"splits={splits}\nunclean_cuts={unclean}\nover_budget={over_budget}\nverbatim={nverb}\ngates_packed={m}\ngates_out={len(canon)}\nratio={len(canon) / m:.4f}\n")
        f.write(f"param.synth={a.synth}\nparam.time_limit={a.time_limit}\nparam.restarts={a.restarts}\nparam.beam={a.beam}\nparam.max_rounds={a.max_rounds}\nparam.budget={a.budget}\n")
        for k, v in checks.items():
            f.write(f"check.{k}={'ok' if v else 'FAIL'}\n")
        f.write(f"time.total_s={time.time() - t0:.1f}\n")
    if not ok:
        print("delivery: a check FAILED: " + ", ".join(k for k, v in checks.items() if not v), file=sys.stderr); sys.exit(4)
    os.replace(os.path.join(a.out, "unpacked.mpmct1.tmp"), os.path.join(a.out, "unpacked.mpmct1"))
    print(f"delivery: status={status}, {len(rounds)} rounds, windows {len(cuts) - 1} -> {len(wins)}, splits {splits} (unclean {unclean}), verbatim {nverb}, gates {m} -> {len(canon)} ({len(canon) / m:.3f}), {time.time() - t0:.0f}s")


def tuple_of(line):
    q = line.split(); k = int(q[2])
    return (int(q[0]), int(q[1]), tuple((int(q[3 + 2 * i]), int(q[4 + 2 * i])) for i in range(k)))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        msg = str(e)
        if msg.startswith("usage:"):
            print(msg, file=sys.stderr); sys.exit(2)
        if msg.startswith("input:") or msg.startswith("internal:"):
            print(msg, file=sys.stderr); sys.exit(3 if msg.startswith("input:") else 8)
        raise
