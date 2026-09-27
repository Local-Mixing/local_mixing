#!/usr/bin/env python3
"""Tables and figures from collision_blocks outputs.  usage: analyze_collisions.py <indir> <outdir> [label=prefix ...]"""
import json, sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

indir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
DEFAULT = [("K2 gadget (unmixed)", "k2_gadget"), ("K2 2-eff", "k2_2eff"), ("K2 h10 pre-split", "k2_h10_presplit"), ("K2 final", "k2_h10_final"),
           ("K2 h20 pre-split (snapshot)", "k2_h20_presplit"), ("K2 h30 pre-split (snapshot)", "k2_h30_presplit"), ("K2 h30 2-eff pre-split", "k2_h30_2eff_presplit"),
           ("K4 gadget (unmixed)", "k4_gadget"), ("K4 2-eff", "k4_2eff"), ("K4 h10 pre-split (snapshot)", "k4_presplit"),
           ("balanced K2 gadget", "bal_k2_gadget"), ("balanced K2 pre-split (=2-eff)", "bal_k2_presplit")]
runs = [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[3:]] or DEFAULT
runs = [(l, p) for l, p in runs if os.path.exists(os.path.join(indir, p + ".json"))]

def log2_q(hist, q):
    h = np.array(hist, dtype=float)
    if h.sum() == 0: return "n/a"
    c = np.cumsum(h) / h.sum(); b = int(np.searchsorted(c, q))
    return "0" if b == 0 else ("1" if b == 1 else f"{2**(b-1)}-{2**b-1}")

data, primes = {}, {}
rows = []
for label, pref in runs:
    d = json.load(open(os.path.join(indir, pref + ".json"))); data[label] = d
    P = np.loadtxt(os.path.join(indir, pref + ".primes.csv"), delimiter=",", skiprows=1, dtype=np.int64).reshape(-1, 4)
    primes[label] = P
    c, b, r = d["collisions"], d["blocks"], d["restore"]; m = d["gates"]
    L = P[:, 2] if len(P) else np.array([1])
    W = P[:, 3] if len(P) else np.array([1])
    ge3 = P[L >= 3] if len(P) else P
    rows.append({
        "label": label, "gates": m,
        "gates in a collision class": c["gates_in_collision"], "frac gates in collision": c["frac_gates_in_collision"],
        "collision pairs": c["pairs"], "pairs same wire": c["pairs_same_wire"], "pairs cross wire": c["pairs_cross_wire"],
        "zero-valued gates (one class)": c["zero_class_size"], "pairs inside the zero class": c["pairs_in_zero_class"],
        "largest classes": "/".join(str(x) for x in c["top_class_sizes"][:4]),
        "dead gates (never fire)": c["never_fire"], "restore pairs (same wire, non-adjacent)": c["restore_pairs_same_wire"],
        "restore gap median / q0.9": f'{log2_q(c["restore_gap_log2_hist"], 0.5)} / {log2_q(c["restore_gap_log2_hist"], 0.9)}',
        "starts with an identity block (<=L)": b["starts_with_an_identity"], "frac starts with identity": b["frac_starts_with_an_identity"],
        "identity (start,end) pairs <=L": b["identity_start_end_pairs_up_to_L"],
        "gates covered by identity blocks": b["gates_covered_by_identity_blocks"], "frac covered": b["frac_gates_covered"],
        "prime blocks": b["prime_blocks"], "prime len 1 (dead gate)": b["prime_len1"], "prime len 2": b["prime_len2"], "prime len >= 3": int((L >= 3).sum()) if len(P) else 0,
        "prime len>=3: median / q0.9 / max": f"{int(np.median(ge3[:,2]))} / {int(np.quantile(ge3[:,2],0.9))} / {int(ge3[:,2].max())}" if len(ge3) else "n/a",
        "prime len>=3: wires written median / max": f"{int(np.median(ge3[:,3]))} / {int(ge3[:,3].max())}" if len(ge3) else "n/a",
        "gates inside primes len>=3": int(ge3[:, 2].sum()) if len(ge3) else 0, "frac gates inside primes len>=3": (ge3[:, 2].sum() / m) if len(ge3) else 0.0,
        "starts hitting the cutoff": b["starts_hitting_cutoff"],
        "restore pairs (value returns to an earlier value of the wire)": r["pairs"], "restore pairs per gate": r["pairs"] / m,
        "restore: exactly 1 write in between": r["inner1"], "  of which the two writes are firing twins": r["inner1_fire_twin"], "  of which they are the identical gate": r["inner1_identical_gate"],
        "  frac twins / identical (of inner=1)": f'{r["inner1_fire_twin"]/max(r["inner1"],1):.3f} / {r["inner1_identical_gate"]/max(r["inner1"],1):.3f}',
        "restore: >=2 writes in between": r["inner_ge2"], "  of which toggles pair up into identical pairs": r["inner_ge2_pairwise_cancelling"],
        "  frac pairwise (of inner>=2)": f'{r["inner_ge2_pairwise_cancelling"]/max(r["inner_ge2"],1):.3f}',
        "inner writes median / q0.9": f'{log2_q(r["inner_writes_log2_hist"], 0.5)} / {log2_q(r["inner_writes_log2_hist"], 0.9)}',
        "twin bracket length (gates) median / q0.9": f'{log2_q(r["inner1_gap_log2_hist"], 0.5)} / {log2_q(r["inner1_gap_log2_hist"], 0.9)}',
        "twin brackets: masked wire read 0 / 1 / 2-3 / 4+ times inside": " / ".join(f"{x/max(sum(r['twin_masked_reads_log2_hist']),1):.3f}" for x in (r["twin_masked_reads_log2_hist"][0], r["twin_masked_reads_log2_hist"][1], r["twin_masked_reads_log2_hist"][2], sum(r["twin_masked_reads_log2_hist"][3:]))),
        "gates inside >=1 twin bracket": r["twin_cover_gates"], "frac inside twin brackets": r["twin_cover_frac"], "twin bracket nesting depth mean / max": f'{r["twin_depth_mean"]:.2f} / {r["twin_depth_max"]}',
        "gates inside >=1 restore bracket": r["restore_cover_gates"], "frac inside restore brackets": r["restore_cover_frac"], "restore nesting depth mean / max": f'{r["restore_depth_mean"]:.2f} / {r["restore_depth_max"]}',
    })
cols = [k for k in rows[0] if k != "label"]
def fmt(x):
    if isinstance(x, str): return x
    if isinstance(x, (int, np.integer)): return f"{x:,}"
    if isinstance(x, float): return f"{x:.4f}" if abs(x) < 10 else f"{x:,.0f}"
    return str(x)
with open(os.path.join(outdir, "summary.md"), "w") as f:
    f.write("| statistic | " + " | ".join(r["label"] for r in rows) + " |\n|---|" + "---|" * len(rows) + "\n")
    for cn in cols: f.write(f"| {cn} | " + " | ".join(fmt(r[cn]) for r in rows) + " |\n")
with open(os.path.join(outdir, "summary.csv"), "w") as f:
    w = csv.writer(f); w.writerow(["statistic"] + [r["label"] for r in rows])
    for cn in cols: w.writerow([cn] + [r[cn] for r in rows])

colors = plt.cm.tab20(np.linspace(0, 1, len(runs)))
# fig 1: prime block length + wires distributions, restore gaps
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for (label, _), col in zip(runs, colors):
    d = data[label]; b = d["blocks"]; c = d["collisions"]
    h = np.array(b["prime_len_log2_hist"], dtype=float); nz = np.nonzero(h)[0]
    axes[0].plot(nz, h[nz], "o-", color=col, ms=3, label=label)
    h = np.array(b["prime_wires_log2_hist"], dtype=float); nz = np.nonzero(h)[0]
    axes[1].plot(nz, h[nz], "o-", color=col, ms=3)
    h = np.array(c["restore_gap_log2_hist"], dtype=float); nz = np.nonzero(h)[0]
    axes[2].plot(nz, h[nz], "o-", color=col, ms=3)
axes[0].set_yscale("log"); axes[0].set_xlabel("prime identity block length [log2 bin: 1, 2, 3-4, 5-8, ...]"); axes[0].set_ylabel("blocks"); axes[0].legend(fontsize=7)
axes[1].set_yscale("log"); axes[1].set_xlabel("wires written inside a prime block [log2 bin]"); axes[1].set_ylabel("blocks")
axes[2].set_yscale("log"); axes[2].set_xlabel("gap of a same-wire restore collision (gates) [log2 bin]"); axes[2].set_ylabel("collisions")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "prime_blocks.png"), dpi=130); plt.close(fig)

# fig 2: coverage along the gate list
fig, axes = plt.subplots(len(runs), 1, figsize=(14, 1.7 * len(runs)), sharex=False)
axes = np.atleast_1d(axes)
for ax, (label, pref), col in zip(axes, runs, colors):
    C = np.loadtxt(os.path.join(indir, pref + ".cover.csv"), delimiter=",", skiprows=1).reshape(-1, 2)
    x = C[:, 0] / data[label]["gates"]; y = C[:, 1] / 1024.0
    ax.fill_between(x, 0, y, color=col, alpha=0.8)
    ax.set_ylim(0, 1); ax.set_xlim(0, 1); ax.set_ylabel("cover", fontsize=8)
    ax.set_title(f"{label}: fraction of gates inside a contiguous identity block, per 1024-gate bin (overall {data[label]['blocks']['frac_gates_covered']:.3f})", fontsize=9)
axes[-1].set_xlabel("position in the gate list (fraction)")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "identity_cover.png"), dpi=130); plt.close(fig)

# fig 3: prime blocks length vs wires (len>=3), per circuit
n = len(runs); ncol = 4; nrow = (n + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
for k, ((label, _), col) in enumerate(zip(runs, colors)):
    ax = axes[k // ncol][k % ncol]; P = primes[label]
    if len(P):
        Q = P[P[:, 2] >= 3]
        if len(Q):
            ax.scatter(Q[:, 2], Q[:, 3], s=4, color=col, alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(label, fontsize=9); ax.set_xlabel("prime block length (gates)", fontsize=8); ax.set_ylabel("wires written", fontsize=8)
for k in range(n, nrow * ncol): axes[k // ncol][k % ncol].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "prime_scatter.png"), dpi=130); plt.close(fig)
# fig 4: bracket map of restore collisions: position vs bracket length
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
for k, ((label, pref), col) in enumerate(zip(runs, colors)):
    ax = axes[k // ncol][k % ncol]
    R = np.loadtxt(os.path.join(indir, pref + ".restores.csv"), delimiter=",", skiprows=1, dtype=np.int64).reshape(-1, 6)
    if len(R):
        if len(R) > 40000:
            R = R[np.random.default_rng(1).choice(len(R), 40000, replace=False)]
        a = np.where(R[:, 0] < 0, -1, R[:, 0]); gap = R[:, 1] - a
        tw = (R[:, 3] == 1) & (R[:, 4] == 1)
        ax.scatter(R[~tw, 1] / data[label]["gates"], gap[~tw], s=2, color="lightgray", alpha=0.5, label="other restore")
        ax.scatter(R[tw, 1] / data[label]["gates"], gap[tw], s=2, color=col, alpha=0.5, label="twin bracket (1 inner write, same toggles)")
    ax.set_yscale("log"); ax.set_title(label, fontsize=9); ax.set_xlabel("position of the restoring write (fraction of gate list)", fontsize=8); ax.set_ylabel("bracket length (gates)", fontsize=8)
    if k == 0: ax.legend(fontsize=6, loc="lower right")
for k in range(n, nrow * ncol): axes[k // ncol][k % ncol].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "bracket_map.png"), dpi=130); plt.close(fig)
print(open(os.path.join(outdir, "summary.md")).read())
