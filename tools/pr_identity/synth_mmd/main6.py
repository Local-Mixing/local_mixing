"""Main deliverable measurement: constructive MMD g57 synth on w=6 units.
Reports success, ms/unit, mean length, bare-state-hit (vs raw control),
per-wire exposure rate, and the comeback (unmask) degree as a mask property."""
import random, time, sys
import numpy as np
from collections import Counter
import g57
import audit
from synth import synth_unit

def bare_fast(word, w, n_anc, mask, g, pool=256, seed=0):
    """Faster incremental bare audit: returns (full_hit, perwire_exposed, cuts)."""
    rng = np.random.default_rng(seed)
    low = (1 << w) - 1
    v = rng.integers(0, 1 << w, size=pool, dtype=np.int64)
    anc = (rng.integers(0, 1 << n_anc, size=pool, dtype=np.int64) if n_anc
           else np.zeros(pool, np.int64))
    Mperm = g57.word_table(mask, w)
    cur = (anc << w) | Mperm[v]
    vp = g57.apply_word_batch([g], v)
    vbits = [((v >> s) & 1) for s in range(w)]
    vpbits = [((vp >> s) & 1) for s in range(w)]
    full_hit = False; perwire = 0; L = len(word)
    for t in range(1, L):
        a, x, y = word[t - 1]
        fire = ((cur >> x) & 1) | (1 - ((cur >> y) & 1))
        cur = cur ^ (fire << a)
        src = cur & low
        if np.array_equal(src, v) or np.array_equal(src, vp):
            full_hit = True
        for s in range(w):
            col = (src >> s) & 1
            if (np.array_equal(col, vbits[s]) or np.array_equal(col, 1 - vbits[s]) or
                np.array_equal(col, vpbits[s]) or np.array_equal(col, 1 - vpbits[s])):
                perwire += 1
    return full_hit, perwire, L - 1

def run(w=6, lam=6, N=120, seed=11):
    rng = random.Random(seed)
    n_anc = max(0, w - 3)
    oks = 0; glen = []; tsyn = []
    mmd_full = 0; raw_full = 0
    mmd_pw = []; raw_pw = []
    cb = Counter()
    print(f"=== MMD constructive synth: w={w} lam={lam} n_anc={n_anc} N={N} ===", flush=True)
    for i in range(N):
        U = audit.make_unit(w, lam, rng)
        t0 = time.time()
        word, stt = synth_unit(U["unit"], w, n_anc=n_anc, verify=True)
        tsyn.append((time.time() - t0) * 1000)
        oks += stt["ok"]; glen.append(stt["g57"])
        # bare audits
        mf, mpw, mc = bare_fast(word, w, n_anc, U["mask"], U["g"], seed=1000 + i)
        rf, rpw, rc = bare_fast(U["unit"], w, 0, U["mask"], U["g"], seed=1000 + i)
        mmd_full += mf; raw_full += rf
        mmd_pw.append(mpw / max(1, mc * w)); raw_pw.append(rpw / max(1, rc * w))
        # comeback (unmask) degree — mask property
        d = audit.invmask_degrees(U["mask"], U["g"], w)
        cb[d["max_g_wire_deg"]] += 1
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{N}  verified so far {oks}", flush=True)
    print("  --- results ---")
    print(f"  synthesis success (table-verified): {oks}/{N}")
    print(f"  ms/unit: mean {np.mean(tsyn):.0f}  median {np.median(tsyn):.0f}")
    print(f"  g57 length/unit: mean {np.mean(glen):.0f}  min {min(glen)}  max {max(glen)}")
    print(f"  BARE-STATE-HIT (full 6-wire source state materialized at an interior cut):")
    print(f"     raw factorization : {raw_full}/{N} = {raw_full/N*100:.0f}%   (control)")
    print(f"     MMD synth         : {mmd_full}/{N} = {mmd_full/N*100:.1f}%   (target: << 52% seen at w=4)")
    print(f"  per-wire exposure rate (exposed (cut,wire) / total):")
    print(f"     raw : {np.mean(raw_pw)*100:.2f}%    MMD : {np.mean(mmd_pw)*100:.3f}%")
    print(f"  comeback/unmask degree histogram (inv(M) g-wire max-deg; MASK property): "
          f"{dict(sorted(cb.items()))}")

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    lam = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run(N=N, lam=lam)
