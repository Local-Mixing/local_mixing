#!/usr/bin/env python3
"""Tables, figures and clusterability from segment_stats outputs.
usage: analyze_segstats.py <indir> <outdir> [label=prefix ...]"""
import json, sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

indir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
DEFAULT = [("K4 2-eff", "k4_2eff"), ("K2 2-eff", "k2_2eff"), ("K2 h10 pre-split", "k2_h10_presplit"),
           ("K2 final", "k2_h10_final"), ("balanced K2 2-eff", "bal_k2_2eff")]
runs = [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[3:]] or DEFAULT
runs = [(l, p) for l, p in runs if os.path.exists(os.path.join(indir, p + ".json"))]
rng = np.random.default_rng(12345)

def load_bits(prefix, kind):
    meta = json.load(open(prefix + ".sample.meta.json"))
    raw = np.fromfile(prefix + f".sample.{kind}.bin", dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little").reshape(meta["count"], meta["words"] * 64)
    return bits, meta

def row_null(X):
    """Independent segments with the SAME per-segment firing rates (kills all dependence)."""
    p = X.mean(1)
    return (rng.random(X.shape, dtype=np.float32) < p[:, None]).astype(np.uint8)

def hamming_to_set(P, X):
    """Hamming distances from rows of P to rows of X via dot products (binary)."""
    Pf = P.astype(np.float32); Xf = X.astype(np.float32)
    pp = Pf.sum(1)[:, None]; xx = Xf.sum(1)[None, :]
    return pp + xx - 2.0 * (Pf @ Xf.T)

def hopkins(X, n_probe=1000):
    n = X.shape[0]
    probe = rng.choice(n, size=min(n_probe, n), replace=False)
    D = hamming_to_set(X[probe], X)
    D[np.arange(len(probe)), probe] = np.inf
    w = D.min(1)
    # synthetic points: independent bits at a firing rate drawn from the data's rate distribution
    pr = X.mean(1)[rng.choice(n, len(probe))]
    Y = (rng.random((len(probe), X.shape[1]), dtype=np.float32) < pr[:, None]).astype(np.uint8)
    u = hamming_to_set(Y, X).min(1)
    return float(u.sum() / (u.sum() + w.sum()))

def kmeans(Xf, k, iters=25, restarts=3):
    best = None
    n = Xf.shape[0]
    for _ in range(restarts):
        C = Xf[rng.choice(n, k, replace=False)].copy()
        for _ in range(iters):
            D = (Xf * Xf).sum(1)[:, None] - 2 * Xf @ C.T + (C * C).sum(1)[None, :]
            lab = D.argmin(1)
            for j in range(k):
                m = lab == j
                if m.any():
                    C[j] = Xf[m].mean(0)
                else:
                    C[j] = Xf[rng.integers(n)]
        inertia = D[np.arange(n), lab].sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy())
    return best[1]

def silhouette(D, lab):
    n = len(lab); ks = np.unique(lab)
    if len(ks) < 2:
        return float("nan")
    s = np.zeros(n)
    means = np.stack([D[:, lab == j].mean(1) if (lab == j).sum() > 0 else np.full(n, np.inf) for j in ks], 1)
    sizes = np.array([(lab == j).sum() for j in ks])
    own = np.searchsorted(ks, lab)
    a = means[np.arange(n), own] * sizes[own] / np.maximum(sizes[own] - 1, 1)
    means[np.arange(n), own] = np.inf
    b = means.min(1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-9)
    s[sizes[own] == 1] = 0.0
    return float(s.mean())

def pair_similarities(X):
    Xf = X.astype(np.float32)
    G = Xf @ Xf.T
    p = Xf.sum(1)
    iu = np.triu_indices(len(p), 1)
    inter = G[iu]; union = p[iu[0]] + p[iu[1]] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    N = X.shape[1]
    exp = p[iu[0]] * p[iu[1]] / N
    lift = np.where(exp > 0, inter / np.maximum(exp, 1e-9), np.nan)
    return jac, lift, inter, iu

def log2_quantile(hist, q):
    """quantile of a log2-binned histogram, reported as the bin's range label."""
    h = np.array(hist, dtype=float)
    if h.sum() == 0:
        return "n/a"
    c = np.cumsum(h) / h.sum()
    b = int(np.searchsorted(c, q))
    return "0" if b == 0 else f"{2**(b-1)}-{2**b-1}"

summary = {}
clus = {}
data = {}
for label, pref in runs:
    d = json.load(open(os.path.join(indir, pref + ".json")))
    data[label] = d
    Xf_bits, meta = load_bits(os.path.join(indir, pref), "fire")
    Xv_bits, _ = load_bits(os.path.join(indir, pref), "value")
    res = {}
    for kind, X in (("fire", Xf_bits), ("value", Xv_bits)):
        Xn = row_null(X)
        r = {}
        r["hopkins"] = hopkins(X); r["hopkins_null"] = hopkins(Xn)
        jac, lift, inter, iu = pair_similarities(X)
        jacn, _, _, _ = pair_similarities(Xn)
        r["jaccard_mean"] = float(jac.mean()); r["jaccard_mean_null"] = float(jacn.mean())
        r["jaccard_hist"] = np.histogram(jac, bins=50, range=(0, 1))[0]
        r["jaccard_hist_null"] = np.histogram(jacn, bins=50, range=(0, 1))[0]
        for q in (0.9, 0.99, 0.999):
            r[f"jaccard_q{q}"] = float(np.quantile(jac, q)); r[f"jaccard_q{q}_null"] = float(np.quantile(jacn, q))
        r["frac_identical_pairs"] = float((jac >= 1.0).mean()); r["frac_identical_pairs_null"] = float((jacn >= 1.0).mean())
        tw = np.array(meta["target_wire"])
        same = tw[iu[0]] == tw[iu[1]]
        r["jaccard_same_wire"] = float(jac[same].mean()) if same.any() else float("nan")
        r["jaccard_cross_wire"] = float(jac[~same].mean())
        # within the modal firing-rate stratum (rate bin of width 1/32): dependence beyond rate
        pr = X.mean(1); b = np.floor(pr * 32).astype(int); mode = np.bincount(b).argmax()
        sel = np.nonzero(b == mode)[0]
        r["stratum_rate"] = float(mode / 32); r["stratum_frac"] = float(len(sel) / len(pr))
        if len(sel) >= 200:
            Xst = X[sel]; Xstn = Xn[sel]
            js, _, _, ius = pair_similarities(Xst); jsn, _, _, _ = pair_similarities(Xstn)
            r["stratum_jaccard_mean"] = float(js.mean()); r["stratum_jaccard_mean_null"] = float(jsn.mean())
            r["stratum_jaccard_q0.99"] = float(np.quantile(js, 0.99)); r["stratum_jaccard_q0.99_null"] = float(np.quantile(jsn, 0.99))
            r["stratum_frac_identical"] = float((js >= 1.0).mean()); r["stratum_frac_identical_null"] = float((jsn >= 1.0).mean())
            r["stratum_hopkins"] = hopkins(Xst); r["stratum_hopkins_null"] = hopkins(Xstn)
            subs = rng.choice(len(sel), size=min(2000, len(sel)), replace=False)
            Dst = hamming_to_set(Xst[subs], Xst[subs]); Dstn = hamming_to_set(Xstn[subs], Xstn[subs])
            r["stratum_silhouette_k2"] = silhouette(Dst, kmeans(Xst[subs].astype(np.float32), 2))
            r["stratum_silhouette_k2_null"] = silhouette(Dstn, kmeans(Xstn[subs].astype(np.float32), 2))
        else:
            for kk in ("stratum_jaccard_mean","stratum_jaccard_mean_null","stratum_jaccard_q0.99","stratum_jaccard_q0.99_null","stratum_frac_identical","stratum_frac_identical_null","stratum_hopkins","stratum_hopkins_null","stratum_silhouette_k2","stratum_silhouette_k2_null"):
                r[kk] = float("nan")
        # k-means silhouette on a subsample, Hamming distance, real vs null
        sub = rng.choice(X.shape[0], size=min(2500, X.shape[0]), replace=False)
        Xs = X[sub].astype(np.float32); Xsn = Xn[sub].astype(np.float32)
        Ds = hamming_to_set(X[sub], X[sub]); Dsn = hamming_to_set(Xn[sub], Xn[sub])
        sil, siln = {}, {}
        for k in (2, 3, 4, 6, 8, 12, 16):
            sil[k] = silhouette(Ds, kmeans(Xs, k)); siln[k] = silhouette(Dsn, kmeans(Xsn, k))
        r["silhouette"] = sil; r["silhouette_null"] = siln
        # natural partition: by target wire
        labw = tw[sub]
        r["silhouette_by_wire"] = silhouette(Ds, labw)
        kk = np.array(meta["k"])[sub]
        r["silhouette_by_k"] = silhouette(Ds, kk)
        res[kind] = r
    clus[label] = res
    print(f"[{label}] clusterability done", flush=True)

# ---------------- summary table ----------------
def g(d, *ks):
    x = d
    for k in ks:
        x = x[k]
    return x
rows = []
cols = ["gates", "segments", "touches/wire mean", "touches/wire std", "touches/wire var", "touches min", "touches max",
        "writes/wire mean", "writes/wire std", "reads/wire mean",
        "fanout pooled median", "fanout pooled mean", "fanout median of wire medians", "fanout mean of wire means",
        "frac segments fanout 0",
        "box pooled median", "box pooled mean", "box median of wire medians (touch)", "box median of wire medians (target)",
        "consec writes pooled mean", "consec writes pooled median", "consec writes median of wire means", "consec writes median of wire medians", "runs total",
        "mean fire rate", "never fire", "always fire", "per-input pair frac (fire)", "per-input pair frac (value)",
        "cofire_any@8192 (fire)", "cofire_any@1024 (fire)", "cofire_any@64 (fire)", "identical (fire)", "complement (fire)", "disjoint|both nonzero (fire)",
        "mean pair rate (fire)", "mean expected rate (fire)", "rms dev from indep (fire)",
        "cofire_any@8192 (value)", "identical (value)",
        "frac comp=1 gates", "frac k=2 gates", "frac k=1 gates",
        "distinct fire signatures", "frac in classes>=2 (fire)", "frac in class size 2 (fire)", "frac in class size 3-4 (fire)", "frac in class size 5-8 (fire)", "frac in class size 9+ (fire)", "top-3 class sizes (fire)", "signature entropy bits (fire)", "log2 segments",
        "twin classes (size 2-1000, fire)", "twin class members (fire)", "twin classes: same target wire", "twin classes: same control literals", "twin classes: same control wires", "twin classes: identical gates",
        "twin class span median (gates)", "twin class span q0.9", "twin neighbour gap median", "twin neighbour gap q0.9", "twin classes adjacent pairs",
        "distinct value signatures", "frac in classes>=2 (value)", "top-3 class sizes (value)", "value twin classes: same target wire", "value twin span median",
        "Hopkins (fire)", "Hopkins null (fire)", "Jaccard mean (fire)", "Jaccard mean null (fire)", "Jaccard q0.99 (fire)", "Jaccard q0.99 null (fire)",
        "Jaccard same-wire (fire)", "Jaccard cross-wire (fire)", "identical pairs in sample (fire)", "identical pairs null (fire)",
        "best silhouette k-means (fire)", "best k (fire)", "best silhouette null (fire)", "silhouette k=2 (fire)", "silhouette k=2 null (fire)", "silhouette by wire (fire)", "silhouette by k (fire)",
        "modal rate stratum (fire)", "stratum frac of segments", "stratum Jaccard mean", "stratum Jaccard mean null", "stratum Jaccard q0.99", "stratum Jaccard q0.99 null",
        "stratum identical pairs", "stratum identical pairs null", "stratum Hopkins", "stratum Hopkins null", "stratum silhouette k=2", "stratum silhouette k=2 null",
        "Hopkins (value)", "Hopkins null (value)", "Jaccard mean (value)", "Jaccard mean null (value)", "best silhouette (value)", "best silhouette null (value)"]
table = {}
for label, pref in runs:
    d = data[label]; c = clus[label]
    pf, pv, cf, cv = d["pairs_fire"], d["pairs_value"], d["classes_fire"], d["classes_value"]
    fan_hist = d["fanout"]["log2_hist"]
    curve = pf["cofire_any_cum_by_64_inputs"]
    def at(nin):
        i = min(nin // 64, len(curve)) - 1
        return curve[i] if i >= 0 else float("nan")
    r = {
        "gates": d["gates"], "segments": d["segments"],
        "touches/wire mean": d["touches"]["mean"], "touches/wire std": d["touches"]["std"], "touches/wire var": d["touches"]["var"],
        "touches min": d["touches"]["min"], "touches max": d["touches"]["max"],
        "writes/wire mean": d["writes"]["mean"], "writes/wire std": d["writes"]["std"], "reads/wire mean": d["reads"]["mean"],
        "fanout pooled median": d["fanout"]["pooled_median"], "fanout pooled mean": d["fanout"]["pooled_mean"],
        "fanout median of wire medians": d["fanout"]["median_of_wire_medians"], "fanout mean of wire means": d["fanout"]["mean_of_wire_means"],
        "frac segments fanout 0": fan_hist[0] / d["segments"],
        "box pooled median": d["box"]["pooled_median"], "box pooled mean": d["box"]["pooled_mean"],
        "box median of wire medians (touch)": d["box"]["median_of_wire_medians_touch"], "box median of wire medians (target)": d["box"]["median_of_wire_medians_target"],
        "consec writes pooled mean": d["runs"]["pooled_mean"], "consec writes pooled median": d["runs"]["pooled_median"],
        "consec writes median of wire means": d["runs"]["median_of_wire_means"], "consec writes median of wire medians": d["runs"]["median_of_wire_medians"], "runs total": d["runs"]["total_runs"],
        "mean fire rate": d["firing"]["mean_fire_rate"], "never fire": d["firing"]["never_fire"], "always fire": d["firing"]["always_fire"],
        "per-input pair frac (fire)": d["firing"]["per_input_pair_frac"], "per-input pair frac (value)": d["firing"]["per_input_value_pair_frac"],
        "cofire_any@8192 (fire)": pf["frac_cofire_any"], "cofire_any@1024 (fire)": at(1024), "cofire_any@64 (fire)": at(64),
        "identical (fire)": pf["frac_identical"], "complement (fire)": pf["frac_complement"], "disjoint|both nonzero (fire)": pf["frac_disjoint_given_both_nonzero"],
        "mean pair rate (fire)": pf["mean_rate"], "mean expected rate (fire)": pf["mean_expected_rate"], "rms dev from indep (fire)": pf["rms_dev_from_indep"],
        "cofire_any@8192 (value)": pv["frac_cofire_any"], "identical (value)": pv["frac_identical"],
        "frac comp=1 gates": d["comp_gates"] / d["gates"], "frac k=2 gates": d["k_hist"][2] / d["gates"], "frac k=1 gates": d["k_hist"][1] / d["gates"],
        "distinct fire signatures": cf["distinct_signatures"], "frac in classes>=2 (fire)": cf["frac_in_classes_ge2"],
        "frac in class size 2 (fire)": cf["segments_by_log2_class_size"][2] / d["gates"], "frac in class size 3-4 (fire)": cf["segments_by_log2_class_size"][3] / d["gates"],
        "frac in class size 5-8 (fire)": cf["segments_by_log2_class_size"][4] / d["gates"], "frac in class size 9+ (fire)": sum(cf["segments_by_log2_class_size"][5:]) / d["gates"],
        "top-3 class sizes (fire)": "/".join(str(x) for x in cf["top_class_sizes"][:3]), "signature entropy bits (fire)": cf["signature_entropy_bits"], "log2 segments": np.log2(d["gates"]),
        "distinct value signatures": cv["distinct_signatures"], "frac in classes>=2 (value)": cv["frac_in_classes_ge2"], "top-3 class sizes (value)": "/".join(str(x) for x in cv["top_class_sizes"][:3]),
        "twin classes (size 2-1000, fire)": cf["census"]["classes_2_to_1000"], "twin class members (fire)": cf["census"]["members"],
        "twin classes: same target wire": cf["census"]["frac_same_target"], "twin classes: same control literals": cf["census"]["frac_same_ctrls"],
        "twin classes: same control wires": cf["census"]["frac_same_ctrl_wires"], "twin classes: identical gates": cf["census"]["frac_same_gate"],
        "twin class span median (gates)": log2_quantile(cf["census"]["span_log2_hist"], 0.5), "twin class span q0.9": log2_quantile(cf["census"]["span_log2_hist"], 0.9),
        "twin neighbour gap median": log2_quantile(cf["census"]["neighbour_gap_log2_hist"], 0.5), "twin neighbour gap q0.9": log2_quantile(cf["census"]["neighbour_gap_log2_hist"], 0.9),
        "twin classes adjacent pairs": cf["census"]["adjacent_pairs"],
        "value twin classes: same target wire": cv["census"]["frac_same_target"], "value twin span median": log2_quantile(cv["census"]["span_log2_hist"], 0.5),
        "Hopkins (fire)": c["fire"]["hopkins"], "Hopkins null (fire)": c["fire"]["hopkins_null"],
        "Jaccard mean (fire)": c["fire"]["jaccard_mean"], "Jaccard mean null (fire)": c["fire"]["jaccard_mean_null"],
        "Jaccard q0.99 (fire)": c["fire"]["jaccard_q0.99"], "Jaccard q0.99 null (fire)": c["fire"]["jaccard_q0.99_null"],
        "Jaccard same-wire (fire)": c["fire"]["jaccard_same_wire"], "Jaccard cross-wire (fire)": c["fire"]["jaccard_cross_wire"],
        "identical pairs in sample (fire)": c["fire"]["frac_identical_pairs"], "identical pairs null (fire)": c["fire"]["frac_identical_pairs_null"],
        "best silhouette k-means (fire)": max(c["fire"]["silhouette"].values()), "best k (fire)": max(c["fire"]["silhouette"], key=c["fire"]["silhouette"].get), "best silhouette null (fire)": max(c["fire"]["silhouette_null"].values()),
        "silhouette k=2 (fire)": c["fire"]["silhouette"][2], "silhouette k=2 null (fire)": c["fire"]["silhouette_null"][2],
        "modal rate stratum (fire)": c["fire"]["stratum_rate"], "stratum frac of segments": c["fire"]["stratum_frac"],
        "stratum Jaccard mean": c["fire"]["stratum_jaccard_mean"], "stratum Jaccard mean null": c["fire"]["stratum_jaccard_mean_null"],
        "stratum Jaccard q0.99": c["fire"]["stratum_jaccard_q0.99"], "stratum Jaccard q0.99 null": c["fire"]["stratum_jaccard_q0.99_null"],
        "stratum identical pairs": c["fire"]["stratum_frac_identical"], "stratum identical pairs null": c["fire"]["stratum_frac_identical_null"],
        "stratum Hopkins": c["fire"]["stratum_hopkins"], "stratum Hopkins null": c["fire"]["stratum_hopkins_null"],
        "stratum silhouette k=2": c["fire"]["stratum_silhouette_k2"], "stratum silhouette k=2 null": c["fire"]["stratum_silhouette_k2_null"],
        "silhouette by wire (fire)": c["fire"]["silhouette_by_wire"], "silhouette by k (fire)": c["fire"]["silhouette_by_k"],
        "Hopkins (value)": c["value"]["hopkins"], "Hopkins null (value)": c["value"]["hopkins_null"],
        "Jaccard mean (value)": c["value"]["jaccard_mean"], "Jaccard mean null (value)": c["value"]["jaccard_mean_null"],
        "best silhouette (value)": max(c["value"]["silhouette"].values()), "best silhouette null (value)": max(c["value"]["silhouette_null"].values()),
    }
    table[label] = r

def fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.4f}"

labels = [l for l, _ in runs]
with open(os.path.join(outdir, "summary.md"), "w") as f:
    f.write("| statistic | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n")
    for cname in cols:
        f.write(f"| {cname} | " + " | ".join(fmt(table[l][cname]) for l in labels) + " |\n")
with open(os.path.join(outdir, "summary.csv"), "w") as f:
    w = csv.writer(f); w.writerow(["statistic"] + labels)
    for cname in cols:
        w.writerow([cname] + [table[l][cname] for l in labels])
with open(os.path.join(outdir, "clusterability.json"), "w") as f:
    json.dump({l: {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()} for k, v in c.items()} for l, c in clus.items()}, f, indent=1)

# ---------------- figures ----------------
colors = plt.cm.tab10(np.arange(len(runs)))
# 1. gates per wire
fig, axes = plt.subplots(len(runs), 1, figsize=(14, 2.6 * len(runs)), sharex=True)
axes = np.atleast_1d(axes)
for ax, (label, _), col in zip(axes, runs, colors):
    d = data[label]
    t = np.array(d["touches"]["per_wire"]); wr = np.array(d["writes"]["per_wire"])
    ax.bar(np.arange(len(t)), t, width=1.0, color=col, alpha=0.8, label="gates touching the wire")
    ax.bar(np.arange(len(wr)), wr, width=1.0, color="k", alpha=0.6, label="gates writing the wire")
    ax.set_title(f"{label}: {d['gates']:,} gates; touches/wire mean {d['touches']['mean']:,.0f}, std {d['touches']['std']:,.0f} (var {d['touches']['var']:.3g}); writes/wire mean {d['writes']['mean']:,.0f}, std {d['writes']['std']:,.0f}", fontsize=9)
    ax.set_ylabel("gates")
    ax.axvline(127.5, color="gray", lw=0.6, ls="--"); ax.axvline(255.5, color="gray", lw=0.6, ls="--")
axes[0].legend(fontsize=8, loc="upper right"); axes[-1].set_xlabel("wire index (0-127 input x, 128-255 y=0 half, 256-511 gadget ancillas)")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "gates_per_wire.png"), dpi=130); plt.close(fig)

# 1b. distributions of gates per wire
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for (label, _), col in zip(runs, colors):
    d = data[label]
    t = np.sort(np.array(d["touches"]["per_wire"]))[::-1]
    axes[0].plot(t / t.mean(), color=col, label=label)
    axes[1].hist(np.array(d["touches"]["per_wire"]) / d["touches"]["mean"], bins=40, histtype="step", color=col, label=label)
axes[0].set_xlabel("wire rank"); axes[0].set_ylabel("touches / mean touches"); axes[0].set_title("gates per wire, sorted, normalised by the mean")
axes[1].set_xlabel("touches / mean"); axes[1].set_ylabel("wires"); axes[1].set_title("distribution of gates per wire (normalised)"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "gates_per_wire_dist.png"), dpi=130); plt.close(fig)

# 2. per-wire medians
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
for (label, _), col in zip(runs, colors):
    d = data[label]
    axes[0].plot(d["fanout"]["per_wire_mean"], color=col, lw=0.8, label=label)
    axes[1].plot(d["box"]["per_wire_median_touch"], color=col, lw=0.8)
    axes[2].plot(d["runs"]["per_wire_mean"], color=col, lw=0.8)
    axes[3].plot(d["runs"]["per_wire_median"], color=col, lw=0.8)
axes[0].set_ylabel("mean fanout per segment"); axes[0].set_title("per-wire: mean fanout (median fanout is 0 on every wire — most segments are never read)"); axes[0].legend(fontsize=8, ncol=5)
axes[1].set_ylabel("median commutation box"); axes[1].set_yscale("log"); axes[1].set_title("per-wire: median commutation-box size over gates touching the wire")
axes[2].set_ylabel("mean run of consecutive writes"); axes[2].set_title("per-wire: mean length of consecutive-write runs")
axes[3].set_ylabel("median run"); axes[3].set_title("per-wire: median length of consecutive-write runs"); axes[3].set_xlabel("wire index")
for ax in axes:
    ax.axvline(127.5, color="gray", lw=0.6, ls="--"); ax.axvline(255.5, color="gray", lw=0.6, ls="--")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "per_wire_medians.png"), dpi=130); plt.close(fig)

# 2b. pooled histograms (fanout, box, runs)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for (label, _), col in zip(runs, colors):
    d = data[label]
    for ax, key, name in ((axes[0], "fanout", "fanout per segment"), (axes[1], "box", "commutation-box size per gate"), (axes[2], "runs", "consecutive-write run length")):
        h = np.array(d[key]["log2_hist"], dtype=float); h /= h.sum()
        nz = np.nonzero(h)[0]
        ax.plot(nz, h[nz], "o-", color=col, ms=3, label=label); ax.set_xlabel(name + "  [log2 bin: 0, 1, 2-3, 4-7, ...]"); ax.set_ylabel("fraction"); ax.set_yscale("log")
axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, "pooled_hists.png"), dpi=130); plt.close(fig)

# 3. co-firing
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for (label, _), col in zip(runs, colors):
    d = data[label]; pf = d["pairs_fire"]
    rh = np.array(pf["rate_hist"], dtype=float); rh /= rh.sum()
    x = np.arange(65) / 64.0
    axes[0, 0].plot(x, rh, color=col, label=label)
    lh = np.array(pf["lift_hist_log2_half_steps_m8_to_8"], dtype=float); lh /= max(lh.sum(), 1)
    axes[0, 1].plot(np.arange(-8, 8.01, 0.5), lh, color=col, label=label)
    curve = pf["cofire_any_cum_by_64_inputs"]
    axes[1, 0].plot(64 * np.arange(1, len(curve) + 1), curve, color=col, label=label)
    fh = np.array(d["firing"]["fire_rate_hist64"], dtype=float); fh /= fh.sum()
    axes[1, 1].plot(np.arange(65) / 64.0, fh, color=col, label=label)
axes[0, 0].set_yscale("log"); axes[0, 0].set_xlabel("co-firing rate of a random segment pair (fraction of inputs where both fire)"); axes[0, 0].set_ylabel("fraction of pairs (bin 0 = never co-fire)"); axes[0, 0].legend(fontsize=8)
axes[0, 1].set_yscale("log"); axes[0, 1].set_xlabel("log2 lift = log2( P[both fire] / (P[a] P[b]) )"); axes[0, 1].set_ylabel("fraction of pairs"); axes[0, 1].set_title("dependence between firing of random segment pairs (0 = independent)")
axes[1, 0].set_xscale("log", base=2); axes[1, 0].set_xlabel("number of sampled legal inputs"); axes[1, 0].set_ylabel("fraction of pairs that fired together on >=1 input"); axes[1, 0].legend(fontsize=8)
axes[1, 1].set_yscale("log"); axes[1, 1].set_xlabel("firing probability of a segment (over legal inputs)"); axes[1, 1].set_ylabel("fraction of segments")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "cofire.png"), dpi=130); plt.close(fig)

# 3b. twin-class census
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for (label, _), col in zip(runs, colors):
    cf = data[label]["classes_fire"]; ce = cf["census"]
    sh = np.array(cf["segments_by_log2_class_size"], dtype=float) / data[label]["gates"]; nz = np.nonzero(sh)[0]
    axes[0].plot(nz, sh[nz], "o-", color=col, ms=3, label=label)
    sp = np.array(ce["span_log2_hist"], dtype=float); sp /= max(sp.sum(), 1); nz = np.nonzero(sp)[0]
    axes[1].plot(nz, sp[nz], "o-", color=col, ms=3, label=label)
    gp = np.array(ce["neighbour_gap_log2_hist"], dtype=float); gp /= max(gp.sum(), 1); nz = np.nonzero(gp)[0]
    axes[2].plot(nz, gp[nz], "o-", color=col, ms=3, label=label)
axes[0].set_xlabel("size of the exact co-firing class a segment belongs to [log2 bin: 1, 2, 3-4, 5-8, ...]"); axes[0].set_ylabel("fraction of segments"); axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("span of a twin class in the gate list [log2 bin]"); axes[1].set_ylabel("fraction of classes (size 2-1000)"); axes[1].set_yscale("log")
axes[2].set_xlabel("gap between neighbouring twins in the gate list [log2 bin]"); axes[2].set_ylabel("fraction of neighbour pairs"); axes[2].set_yscale("log")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "twin_census.png"), dpi=130); plt.close(fig)

# 4. clusterability
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for (label, _), col in zip(runs, colors):
    c = clus[label]["fire"]; d = data[label]
    h = c["jaccard_hist"] / c["jaccard_hist"].sum(); hn = c["jaccard_hist_null"] / c["jaccard_hist_null"].sum()
    xs = (np.arange(50) + 0.5) / 50
    axes[0, 0].plot(xs, h, color=col, label=label); axes[0, 0].plot(xs, hn, color=col, ls=":", lw=0.8)
    ks = sorted(c["silhouette"]); axes[0, 1].plot(ks, [c["silhouette"][k] for k in ks], "o-", color=col, label=label); axes[0, 1].plot(ks, [c["silhouette_null"][k] for k in ks], "x:", color=col)
    sh = np.array(d["classes_fire"]["segments_by_log2_class_size"], dtype=float); sh /= sh.sum(); nz = np.nonzero(sh)[0]
    axes[0, 2].plot(nz, sh[nz], "o-", color=col, ms=3, label=label)
    cv = clus[label]["value"]
    hv = cv["jaccard_hist"] / cv["jaccard_hist"].sum(); hvn = cv["jaccard_hist_null"] / cv["jaccard_hist_null"].sum()
    axes[1, 0].plot(xs, hv, color=col, label=label); axes[1, 0].plot(xs, hvn, color=col, ls=":", lw=0.8)
    axes[1, 1].plot(ks, [cv["silhouette"][k] for k in ks], "o-", color=col); axes[1, 1].plot(ks, [cv["silhouette_null"][k] for k in ks], "x:", color=col)
    axes[1, 2].bar(np.arange(len(runs))[[l for l, _ in runs].index(label)] + np.array([-0.2, 0.2]), [c["hopkins"], c["hopkins_null"]], width=0.4, color=[col, "lightgray"])
axes[0, 0].set_yscale("log"); axes[0, 0].set_xlabel("Jaccard co-firing similarity of segment pairs (sample of 8192 segments)"); axes[0, 0].set_ylabel("fraction of pairs"); axes[0, 0].set_title("FIRE: real (solid) vs independent-segments-same-rates null (dotted)"); axes[0, 0].legend(fontsize=8)
axes[0, 1].set_xlabel("k (k-means, Hamming silhouette)"); axes[0, 1].set_ylabel("silhouette"); axes[0, 1].set_title("FIRE: k-means silhouette, real (o) vs null (x)"); axes[0, 1].set_xscale("log", base=2)
axes[0, 2].set_yscale("log"); axes[0, 2].set_xlabel("exact-signature class size [log2 bin]"); axes[0, 2].set_ylabel("fraction of segments"); axes[0, 2].set_title("FIRE: segments in classes of identical firing vectors")
axes[1, 0].set_yscale("log"); axes[1, 0].set_xlabel("Jaccard similarity of segment VALUE vectors"); axes[1, 0].set_title("VALUE (wire = 1): real vs null")
axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("silhouette"); axes[1, 1].set_title("VALUE: k-means silhouette real (o) vs null (x)"); axes[1, 1].set_xscale("log", base=2)
axes[1, 2].set_xticks(np.arange(len(runs))); axes[1, 2].set_xticklabels([l for l, _ in runs], rotation=20, fontsize=8); axes[1, 2].set_ylabel("Hopkins statistic (fire)"); axes[1, 2].axhline(0.5, color="gray", ls="--", lw=0.8); axes[1, 2].set_title("Hopkins (fire): real (colour) vs same-rates null (gray); 0.5 = no cluster tendency")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "clusterability.png"), dpi=130); plt.close(fig)
print(open(os.path.join(outdir, "summary.md")).read())
