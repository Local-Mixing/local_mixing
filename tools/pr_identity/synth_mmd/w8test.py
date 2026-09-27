"""w=8 feasibility: does the constructive synth verify, and at what cost?"""
import random, time
import numpy as np
from collections import Counter
import g57, audit
from synth import synth_unit
from main6 import bare_fast

def run(N=10, seed=5):
    w = 8; lam = 8; n_anc = w - 3   # =5
    rng = random.Random(seed)
    oks = 0; glen = []; ts = []; full = 0; pw = []; cb = Counter()
    print(f"=== w=8 feasibility  n_anc={n_anc}  N={N} ===", flush=True)
    for i in range(N):
        U = audit.make_unit(w, lam, rng)
        t0 = time.time()
        word, stt = synth_unit(U["unit"], w, n_anc=n_anc, verify=True)
        ts.append(time.time() - t0)
        oks += stt["ok"]; glen.append(stt["g57"])
        mf, mpw, mc = bare_fast(word, w, n_anc, U["mask"], U["g"], pool=128, seed=99 + i)
        full += mf; pw.append(mpw / max(1, mc * w))
        cb[audit.invmask_degrees(U["mask"], U["g"], w)["max_g_wire_deg"]] += 1
        print(f"  unit {i}: verified={stt['ok']} g57={stt['g57']} maxk={stt['maxk']} "
              f"{ts[-1]:.2f}s  bare_full={mf}", flush=True)
    print("  --- w=8 summary ---")
    print(f"  verified {oks}/{N}   mean g57 {np.mean(glen):.0f}   mean {np.mean(ts):.2f}s/unit")
    print(f"  bare-state-hit(full): {full}/{N}   per-wire expo rate {np.mean(pw)*100:.3f}%")
    print(f"  comeback/unmask deg (lam=8): {dict(sorted(cb.items()))}")

if __name__ == "__main__":
    run()
