#!/usr/bin/env python3
"""Compare collision-attack CPU cost vs λ for 2λ→λ hashes on 3λ-wire circuits.

Measures where practical and extrapolates trends:

* Birthday — hash-table search (1 core, scalar eval)
* Rho/DP — van Oorschot–Wiener (multi-core, bit-sliced)
* SAT — dual-copy collision CNF + Glucose3
* BHT (quantum, theoretical) — ~π·2^{λ/3}/2 evaluations as a reference

CPU-seconds means total core-time: wall_seconds × threads.
Expected classical work uses ~1.25·2^{λ/2} hash evaluations (birthday bound);
rho uses the same leading term with an empirical overhead factor from measured
runs. Gate count is fixed at 1024 to match the checked-in fixtures.

Usage (from repo root):
  python3 security_tests/collision/compare_attack_costs.py \\
    --out-dir target/security-demo/collision/cost_compare
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release"
ENCODER = ROOT / "target" / "security-demo" / "collision" / "collision_to_cnf"
GATES = 1024
SEED = 20260330

# Birthday / rho expected evaluations ≈ sqrt(π/2) · 2^{λ/2}
BIRTHDAY_FACTOR = math.sqrt(math.pi / 2.0)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=ROOT, text=True, capture_output=True, check=False, **kwargs
    )


def ensure_tools() -> None:
    r = run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "security-tools",
            "--bin",
            "gen_collision_circuit",
            "--bin",
            "birthday_collision",
            "--bin",
            "rho_collision",
        ]
    )
    if r.returncode != 0:
        raise SystemExit(r.stderr or r.stdout)
    ENCODER.parent.mkdir(parents=True, exist_ok=True)
    if not ENCODER.exists():
        src = ROOT / "security_tests" / "collision" / "collision_to_cnf.cpp"
        r = run(["g++", "-std=c++17", "-O3", str(src), "-o", str(ENCODER)])
        if r.returncode != 0:
            raise SystemExit(r.stderr or r.stdout)


def gen_circuit(workdir: Path, lam: int) -> Path:
    prefix = workdir / f"c{3 * lam}"
    g57 = Path(f"{prefix}.g57")
    if g57.exists():
        return g57
    r = run(
        [
            str(BIN / "gen_collision_circuit"),
            str(prefix),
            str(3 * lam),
            str(GATES),
            str(SEED + lam),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return g57


def bench_scalar(circuit: Path, lam: int, evals: int = 2_000_000) -> float:
    r = run(
        [
            str(BIN / "birthday_collision"),
            str(circuit),
            "--pad",
            str(lam),
            "--in-bits",
            str(min(64, 2 * lam)),
            "--out-bits",
            str(lam),
            "--bench-evals",
            str(evals),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return float(json.loads(r.stdout)["meval_per_sec"])


def bench_lanes(circuit: Path, lam: int, evals: int = 5_000_000) -> float:
    """Meval/s for one bit-sliced 64-lane walker (1 core)."""
    r = run(
        [
            str(BIN / "rho_collision"),
            str(circuit),
            "--pad",
            str(lam),
            "--out-bits",
            str(lam),
            "--dp-bits",
            str(max(1, min(4, lam // 2))),
            "--bench-evals",
            str(evals),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return float(json.loads(r.stdout)["meval_per_sec"])


def measure_birthday(circuit: Path, lam: int, seed: int = 1) -> dict:
    # Budget a few birthday-bound multiples.
    budget = int(min(50_000_000, max(100_000, 20 * (2 ** (lam / 2)))))
    t0 = time.perf_counter()
    r = run(
        [
            str(BIN / "birthday_collision"),
            str(circuit),
            "--pad",
            str(lam),
            "--in-bits",
            str(min(64, 2 * lam)),
            "--out-bits",
            str(lam),
            "--samples",
            str(budget),
            "--seed",
            str(seed),
        ]
    )
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        return {
            "ok": False,
            "wall_secs": wall,
            "cpu_hours": wall / 3600.0,
            "threads": 1,
            "stderr": (r.stderr or "")[-500:],
        }
    report = json.loads(r.stdout)
    wall = float(report["elapsed_secs"])
    return {
        "ok": True,
        "wall_secs": wall,
        "cpu_hours": wall / 3600.0,
        "threads": 1,
        "evals": int(report["samples_evaluated"]),
        "report": report,
    }


def measure_rho(circuit: Path, lam: int, workers: int, seed: int = 1) -> dict:
    dp = max(4, min(lam // 3, 18))
    # Cap so the sweep finishes; larger λ rely on the model + the λ=64 fixture.
    max_evals = {
        16: 20_000_000,
        20: 50_000_000,
        24: 100_000_000,
        28: 200_000_000,
        32: 500_000_000,
        40: 2_000_000_000,
        48: 8_000_000_000,
    }.get(lam, 50_000_000)
    t0 = time.perf_counter()
    r = run(
        [
            str(BIN / "rho_collision"),
            str(circuit),
            "--pad",
            str(lam),
            "--out-bits",
            str(lam),
            "--dp-bits",
            str(dp),
            "--workers",
            str(workers),
            "--seed",
            str(seed),
            "--max-evals",
            str(max_evals),
        ]
    )
    wall = time.perf_counter() - t0
    m = re.search(r"collision after (\d+) evals", r.stderr or "")
    if r.returncode != 0 or not m:
        return {
            "ok": False,
            "wall_secs": wall,
            "cpu_hours": wall * workers / 3600.0,
            "threads": workers,
            "max_evals": max_evals,
            "stderr": (r.stderr or "")[-500:],
        }
    evals = int(m.group(1))
    # Prefer JSON elapsed when present.
    try:
        report = json.loads(r.stdout)
        wall = float(report["elapsed_secs"])
        evals = int(report["samples_evaluated"])
    except json.JSONDecodeError:
        report = None
    return {
        "ok": True,
        "wall_secs": wall,
        "cpu_hours": wall * workers / 3600.0,
        "threads": workers,
        "evals": evals,
        "report": report,
    }


def measure_sat(circuit_mpmct: Path, lam: int, timeout_s: float) -> dict:
    from pysat.solvers import Glucose3

    with tempfile.TemporaryDirectory() as tmp:
        cnf = Path(tmp) / "c.cnf"
        t_enc0 = time.perf_counter()
        r = run(
            [
                str(ENCODER),
                str(circuit_mpmct),
                str(cnf),
                "--in-bits",
                str(2 * lam),
                "--pad",
                str(lam),
                "--out-bits",
                str(lam),
            ]
        )
        enc_s = time.perf_counter() - t_enc0
        if r.returncode != 0:
            return {"ok": False, "phase": "encode", "stderr": r.stderr}

        clauses: list[list[int]] = []
        with cnf.open() as f:
            for line in f:
                if line.startswith("c") or not line.strip():
                    continue
                if line.startswith("p"):
                    continue
                lits = [int(x) for x in line.split() if x != "0"]
                if lits:
                    clauses.append(lits)

        t0 = time.perf_counter()
        with Glucose3(bootstrap_with=clauses) as g:
            # pysat Glucose has no built-in wall timeout on all builds; poll.
            # Use a worker-style soft timeout via solving with assumptions empty
            # and checking elapsed in a loop is not supported — call solve()
            # and rely on process-level timeout for hard cases.
            sat = g.solve()
        wall = time.perf_counter() - t0
        if wall > timeout_s and not sat:
            return {
                "ok": False,
                "phase": "timeout",
                "wall_secs": wall,
                "cpu_hours": wall / 3600.0,
                "threads": 1,
                "encode_secs": enc_s,
                "clauses": len(clauses),
            }
        return {
            "ok": bool(sat),
            "wall_secs": wall,
            "cpu_hours": (enc_s + wall) / 3600.0,
            "threads": 1,
            "encode_secs": enc_s,
            "clauses": len(clauses),
            "sat": bool(sat),
        }


def measure_sat_with_timeout(circuit_mpmct: Path, lam: int, timeout_s: float) -> dict:
    """Run SAT in a subprocess so we can hard-timeout."""
    helper = ROOT / "security_tests" / "collision" / "_sat_once.py"
    # Inline helper via python -c for portability.
    code = r"""
import json, sys, time
from pathlib import Path
from pysat.solvers import Glucose3
circuit, cnf_tool, lam, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
import subprocess, tempfile, os
lam=lam
with tempfile.TemporaryDirectory() as tmp:
    cnf = Path(tmp)/'c.cnf'
    t0=time.perf_counter()
    r=subprocess.run([cnf_tool, circuit, str(cnf), '--in-bits', str(2*lam), '--pad', str(lam), '--out-bits', str(lam)], capture_output=True, text=True)
    enc=time.perf_counter()-t0
    if r.returncode!=0:
        print(json.dumps({'ok':False,'phase':'encode','stderr':r.stderr[-400:]})); sys.exit(0)
    clauses=[]
    for line in Path(cnf).read_text().splitlines():
        if line.startswith('c') or line.startswith('p') or not line.strip():
            continue
        lits=[int(x) for x in line.split() if x!='0']
        if lits: clauses.append(lits)
    t1=time.perf_counter()
    # Soft budget: Glucose3 solve; parent kills us on hard timeout.
    with Glucose3(bootstrap_with=clauses) as g:
        sat=g.solve()
    wall=time.perf_counter()-t1
    print(json.dumps({'ok':bool(sat),'wall_secs':wall,'cpu_hours':(enc+wall)/3600.0,'threads':1,'encode_secs':enc,'clauses':len(clauses),'sat':bool(sat)}))
"""
    try:
        r = subprocess.run(
            [
                "python3",
                "-c",
                code,
                str(circuit_mpmct),
                str(ENCODER),
                str(lam),
                str(timeout_s),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s + 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "phase": "timeout",
            "wall_secs": timeout_s,
            "cpu_hours": timeout_s / 3600.0,
            "threads": 1,
        }
    if r.returncode != 0 or not r.stdout.strip():
        return {
            "ok": False,
            "phase": "error",
            "stderr": (r.stderr or r.stdout)[-500:],
            "cpu_hours": timeout_s / 3600.0,
            "threads": 1,
        }
    return json.loads(r.stdout.strip().splitlines()[-1])


def expected_evals(lam: int) -> float:
    return BIRTHDAY_FACTOR * (2.0 ** (lam / 2.0))


def plot(results: dict, out_png: Path, out_pdf: Path) -> None:
    lambdas = sorted(results["throughput"].keys())
    workers = results["meta"]["workers"]

    # Model curves over a dense λ grid.
    grid = np.arange(8, 81, 1)
    scalar = np.array(
        [results["throughput"][l]["scalar_meval"] for l in lambdas], dtype=float
    )
    lane = np.array(
        [results["throughput"][l]["lane_meval"] for l in lambdas], dtype=float
    )
    # Extrapolate throughput with log-linear fit vs λ (mild width dependence).
    coef_s = np.polyfit(lambdas, np.log(np.maximum(scalar, 1e-9)), 1)
    coef_l = np.polyfit(lambdas, np.log(np.maximum(lane, 1e-9)), 1)
    scalar_fit = np.exp(np.polyval(coef_s, grid))
    lane_fit = np.exp(np.polyval(coef_l, grid))

    # Rho overhead vs ideal birthday-bound evals, from successful measured runs.
    rho_overheads = []
    for lam, m in results["measured"].get("rho", {}).items():
        if m.get("ok") and m.get("evals"):
            rho_overheads.append(m["evals"] / expected_evals(int(lam)))
    # Include the published λ=64 fixture if present.
    if results.get("fixture_rho64"):
        fx = results["fixture_rho64"]
        rho_overheads.append(fx["evals"] / expected_evals(64))
    rho_overhead = float(np.median(rho_overheads)) if rho_overheads else 4.0

    # Plot in CPU-seconds (wall × threads).
    birthday_cpu = expected_evals(grid) / (scalar_fit * 1e6)  # 1 thread
    # Rho: aggregate rate ≈ lane_fit * workers (each worker runs a 64-lane walk).
    rho_wall = (rho_overhead * expected_evals(grid)) / (lane_fit * workers * 1e6)
    rho_cpu = rho_wall * workers

    # SAT model: fit log(cpu_secs) ~ a + b·λ on successful points.
    sat_pts = [
        (int(lam), m["cpu_hours"] * 3600.0)
        for lam, m in results["measured"].get("sat", {}).items()
        if m.get("ok") and m.get("cpu_hours", 0) > 0
    ]
    sat_curve = None
    if len(sat_pts) >= 2:
        xs = np.array([p[0] for p in sat_pts], dtype=float)
        ys = np.log(np.array([p[1] for p in sat_pts], dtype=float))
        coef = np.polyfit(xs, ys, 1)
        # Only draw near the fitted range, dashed beyond.
        sat_grid = np.arange(int(xs.min()), min(48, int(xs.max()) + 12) + 1)
        sat_curve = (sat_grid, np.exp(np.polyval(coef, sat_grid)))

    # BHT quantum collision: ~π/2 · 2^{λ/3} oracle queries; plot as if each
    # query cost equaled one classical scalar eval (order-of-magnitude hint).
    bht_cpu = (math.pi / 2.0) * (2.0 ** (grid / 3.0)) / (scalar_fit * 1e6)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.semilogy(grid, birthday_cpu, color="#1f77b4", lw=2, label="Birthday (model)")
    ax.semilogy(grid, rho_cpu, color="#ff7f0e", lw=2, label=f"Rho/DP (model, ×{rho_overhead:.1f} work, {workers} thr)")
    ax.semilogy(
        grid,
        bht_cpu,
        color="#2ca02c",
        lw=1.5,
        ls="--",
        label="BHT quantum (theoretical)",
    )
    if sat_curve is not None:
        ax.semilogy(
            sat_curve[0],
            sat_curve[1],
            color="#d62728",
            lw=2,
            label="SAT (fit to solved points)",
        )
    ax.axvspan(16, 80, color="#d62728", alpha=0.06, label="SAT impractical (timeouts)")

    # Measured markers
    for lam, m in results["measured"].get("birthday", {}).items():
        if m.get("ok"):
            ax.scatter(
                [int(lam)],
                [m["cpu_hours"] * 3600.0],
                color="#1f77b4",
                s=40,
                zorder=5,
                edgecolors="k",
                linewidths=0.4,
            )
    for lam, m in results["measured"].get("rho", {}).items():
        if m.get("ok"):
            ax.scatter(
                [int(lam)],
                [m["cpu_hours"] * 3600.0],
                color="#ff7f0e",
                s=40,
                zorder=5,
                edgecolors="k",
                linewidths=0.4,
            )
    if results.get("fixture_rho64"):
        ax.scatter(
            [64],
            [results["fixture_rho64"]["cpu_hours"] * 3600.0],
            color="#ff7f0e",
            s=90,
            marker="*",
            zorder=6,
            edgecolors="k",
            linewidths=0.4,
            label="Rho λ=64 fixture",
        )
    sat_timeout_labeled = False
    for lam, m in results["measured"].get("sat", {}).items():
        if m.get("ok"):
            ax.scatter(
                [int(lam)],
                [m["cpu_hours"] * 3600.0],
                color="#d62728",
                s=40,
                zorder=5,
                edgecolors="k",
                linewidths=0.4,
            )
        elif m.get("phase") == "timeout":
            ax.scatter(
                [int(lam)],
                [m["cpu_hours"] * 3600.0],
                color="#d62728",
                s=55,
                marker="x",
                zorder=5,
                label="SAT timeout (≥ bound)" if not sat_timeout_labeled else None,
            )
            sat_timeout_labeled = True

    ax.set_xlabel(r"$\lambda$ (digest bits; circuit width $3\lambda$, 1024 gates)")
    ax.set_ylabel("CPU-seconds (wall × threads)")
    ax.set_title(r"Collision-finding cost vs $\lambda$ for $H(x)=C(0^\lambda\|x)_\lambda$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(8, 80)
    ax.set_ylim(1e-5, 1e9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "target" / "security-demo" / "collision" / "cost_compare",
    )
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument(
        "--lambdas-throughput",
        default="8,12,16,20,24,28,32,40,48,56,64",
    )
    ap.add_argument("--lambdas-birthday", default="8,12,16,20,24,28,32")
    ap.add_argument("--lambdas-rho", default="16,20,24,28,32,40")
    ap.add_argument("--lambdas-sat", default="4,6,8,10,12,14")
    ap.add_argument("--sat-timeout", type=float, default=60.0)
    args = ap.parse_args()

    ensure_tools()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "circuits"
    work.mkdir(exist_ok=True)

    def parse_lams(s: str) -> list[int]:
        return [int(x) for x in s.split(",") if x.strip()]

    results: dict = {
        "meta": {
            "gates": GATES,
            "workers": args.workers,
            "birthday_factor": BIRTHDAY_FACTOR,
            "note": "Plot uses CPU-seconds = wall_seconds × threads; JSON keeps cpu_hours too",
        },
        "throughput": {},
        "measured": {"birthday": {}, "rho": {}, "sat": {}},
    }

    print("== throughput ==")
    for lam in parse_lams(args.lambdas_throughput):
        c = gen_circuit(work, lam)
        s = bench_scalar(c, lam)
        l = bench_lanes(c, lam)
        results["throughput"][lam] = {"scalar_meval": s, "lane_meval": l}
        print(f"  λ={lam:2d}  scalar={s:7.2f} Meval/s  lanes={l:7.2f} Meval/s")

    print("== birthday (measured) ==")
    for lam in parse_lams(args.lambdas_birthday):
        c = gen_circuit(work, lam)
        m = measure_birthday(c, lam)
        results["measured"]["birthday"][lam] = m
        status = "ok" if m.get("ok") else "fail"
        print(
            f"  λ={lam:2d}  {status:4s}  cpu_hours={m.get('cpu_hours', float('nan')):.3e}  "
            f"evals={m.get('evals', '-')}"
        )

    print("== rho/DP (measured) ==")
    for lam in parse_lams(args.lambdas_rho):
        c = gen_circuit(work, lam)
        m = measure_rho(c, lam, workers=args.workers)
        results["measured"]["rho"][lam] = m
        status = "ok" if m.get("ok") else "fail"
        print(
            f"  λ={lam:2d}  {status:4s}  cpu_hours={m.get('cpu_hours', float('nan')):.3e}  "
            f"evals={m.get('evals', '-')}"
        )

    # Fold in the checked-in λ=64 witness (4 workers, ~693s wall).
    fixture = ROOT / "security_tests" / "collision" / "fixtures" / "c192_g1024.rho.json"
    if fixture.exists():
        fx = json.loads(fixture.read_text())
        wall = float(fx["elapsed_secs"])
        results["fixture_rho64"] = {
            "evals": int(fx["samples_evaluated"]),
            "wall_secs": wall,
            "threads": args.workers,
            "cpu_hours": wall * args.workers / 3600.0,
        }
        print(
            f"  λ=64  fixture  cpu_hours={results['fixture_rho64']['cpu_hours']:.3e}  "
            f"evals={results['fixture_rho64']['evals']}"
        )

    print("== SAT (measured) ==")
    for lam in parse_lams(args.lambdas_sat):
        prefix = work / f"c{3 * lam}"
        gen_circuit(work, lam)
        mpmct = Path(f"{prefix}.mpmct1")
        m = measure_sat_with_timeout(mpmct, lam, timeout_s=args.sat_timeout)
        results["measured"]["sat"][lam] = m
        status = "ok" if m.get("ok") else m.get("phase", "fail")
        print(
            f"  λ={lam:2d}  {status:8s}  cpu_hours={m.get('cpu_hours', float('nan')):.3e}  "
            f"clauses={m.get('clauses', '-')}"
        )

    # Model summary table
    summary_rows = []
    for lam in sorted(results["throughput"]):
        s = results["throughput"][lam]["scalar_meval"]
        l = results["throughput"][lam]["lane_meval"]
        exp = expected_evals(lam)
        b_cpu = exp / (s * 1e6) / 3600.0
        # provisional overhead 4 until plot recomputes median
        r_cpu = (4.0 * exp) / (l * args.workers * 1e6) * args.workers / 3600.0
        summary_rows.append(
            {
                "lambda": lam,
                "scalar_meval": s,
                "lane_meval": l,
                "expected_evals": exp,
                "birthday_model_cpu_hours": b_cpu,
                "rho_model_cpu_hours_overhead4": r_cpu,
            }
        )
    results["model_table"] = summary_rows

    json_path = args.out_dir / "cost_compare.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    png = args.out_dir / "cost_compare.png"
    pdf = args.out_dir / "cost_compare.pdf"
    plot(results, png, pdf)

    # Also copy into fixtures for the PR / docs.
    fixtures = ROOT / "security_tests" / "collision" / "fixtures"
    fixtures.mkdir(exist_ok=True)
    (fixtures / "cost_compare.png").write_bytes(png.read_bytes())
    (fixtures / "cost_compare.json").write_text(json_path.read_text())

    print(f"\nWrote {json_path}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Copied plot/data to {fixtures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
