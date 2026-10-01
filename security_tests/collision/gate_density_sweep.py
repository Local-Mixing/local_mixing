#!/usr/bin/env python3
"""Gate-density sweep: collision attacks vs circuit size at fixed λ.

For each λ ∈ {16,32,64}, n=3λ, and five geometrically spaced gate counts
from (1/2)·n·log₂(n) to n², generate a seeded circuit and run birthday,
rho/DP, and SAT. Circuits + seeds are persisted for reproducibility.

Usage (repo root):
  python3 security_tests/collision/gate_density_sweep.py \\
    --out-dir target/security-demo/collision/gate_density
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release"
ENCODER = ROOT / "target" / "security-demo" / "collision" / "collision_to_cnf"
FIXTURES = ROOT / "security_tests" / "collision" / "fixtures" / "gate_density"

# Master seed; per-circuit seed = BASE_SEED + λ*1_000_000 + gates
BASE_SEED = 20261001
LAMBDAS = (16, 32, 64)
N_STEPS = 5
WORKERS = 4


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


def gate_schedule(lam: int, steps: int = N_STEPS) -> list[int]:
    n = 3 * lam
    lo = 0.5 * n * math.log2(n)
    hi = float(n * n)
    out: list[int] = []
    for k in range(steps):
        g = int(round(lo * (hi / lo) ** (k / (steps - 1))))
        if out and g <= out[-1]:
            g = out[-1] + 1
        out.append(g)
    out[-1] = n * n  # exact upper endpoint
    return out


def circuit_seed(lam: int, gates: int) -> int:
    return BASE_SEED + lam * 1_000_000 + gates


def circuit_prefix(circuits_dir: Path, lam: int, gates: int) -> Path:
    return circuits_dir / f"c{3 * lam}_g{gates}"


def gen_circuit(circuits_dir: Path, lam: int, gates: int) -> dict:
    circuits_dir.mkdir(parents=True, exist_ok=True)
    prefix = circuit_prefix(circuits_dir, lam, gates)
    meta_path = Path(f"{prefix}.meta.json")
    seed = circuit_seed(lam, gates)
    if meta_path.exists() and Path(f"{prefix}.g57").exists() and Path(f"{prefix}.mpmct1").exists():
        meta = json.loads(meta_path.read_text())
        return meta
    r = run(
        [
            str(BIN / "gen_collision_circuit"),
            str(prefix),
            str(3 * lam),
            str(gates),
            str(seed),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"gen λ={lam} g={gates}: {r.stderr or r.stdout}")
    meta = json.loads(meta_path.read_text())
    # Mirror into fixtures for reproducibility check-in.
    fix_dir = FIXTURES / f"lambda_{lam}"
    fix_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".g57", ".mpmct1", ".meta.json"):
        src = Path(f"{prefix}{ext}")
        dst = fix_dir / src.name
        dst.write_bytes(src.read_bytes())
    return meta


def measure_birthday(circuit: Path, lam: int, seed: int) -> dict:
    # Generous multiples of the birthday bound; hard cap for memory/time.
    budget = int(min(400_000_000, max(200_000, 60 * (2 ** (lam / 2)))))
    # λ=64 needs ~2^{32} samples; allow up to ~3× bound (still large).
    if lam >= 64:
        budget = int(min(6_000_000_000, 3 * (2 ** (lam / 2))))
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
            "cpu_seconds": wall,
            "threads": 1,
            "budget": budget,
            "stderr": (r.stderr or "")[-500:],
        }
    report = json.loads(r.stdout)
    wall = float(report["elapsed_secs"])
    return {
        "ok": True,
        "wall_secs": wall,
        "cpu_seconds": wall,
        "threads": 1,
        "budget": budget,
        "evals": int(report["samples_evaluated"]),
        "report": report,
    }


def measure_rho(circuit: Path, lam: int, workers: int, seed: int) -> dict:
    dp = max(4, min(lam // 3, 18))
    max_evals = {
        16: 50_000_000,
        32: 2_000_000_000,
        64: 200_000_000_000,  # headroom above ~3.6e10 fixture
    }.get(lam, int(100 * (2 ** (lam / 2))))
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
    if r.returncode != 0:
        return {
            "ok": False,
            "wall_secs": wall,
            "cpu_seconds": wall * workers,
            "threads": workers,
            "max_evals": max_evals,
            "dp_bits": dp,
            "stderr": (r.stderr or "")[-500:],
        }
    try:
        report = json.loads(r.stdout)
        wall = float(report["elapsed_secs"])
        evals = int(report["samples_evaluated"])
    except json.JSONDecodeError:
        m = re.search(r"collision after (\d+) evals", r.stderr or "")
        if not m:
            return {
                "ok": False,
                "wall_secs": wall,
                "cpu_seconds": wall * workers,
                "threads": workers,
                "max_evals": max_evals,
                "dp_bits": dp,
                "stderr": (r.stderr or "")[-500:],
            }
        report = None
        evals = int(m.group(1))
    return {
        "ok": True,
        "wall_secs": wall,
        "cpu_seconds": wall * workers,
        "threads": workers,
        "max_evals": max_evals,
        "dp_bits": dp,
        "evals": evals,
        "report": report,
    }


def sat_timeout_for(lam: int, gates: int) -> float:
    """Reasonable wall budgets: more for small λ / fewer gates."""
    if lam <= 16:
        # Scale gently with gates; base 180s at ~n log n.
        return min(900.0, 120.0 + 0.15 * gates)
    if lam <= 32:
        return min(600.0, 90.0 + 0.05 * gates)
    # λ=64: already hopeless in prior sweep; short confirmatory timeout.
    return min(300.0, 60.0 + 0.01 * gates)


def measure_sat(mpmct: Path, lam: int, timeout_s: float) -> dict:
    helper = ROOT / "security_tests" / "collision" / "sat_once.py"
    try:
        r = subprocess.run(
            [
                "python3",
                str(helper),
                str(mpmct),
                str(ENCODER),
                str(lam),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "phase": "timeout",
            "wall_secs": timeout_s,
            "cpu_seconds": timeout_s,
            "threads": 1,
            "timeout_s": timeout_s,
            "note": f"killed after {timeout_s:.0f}s",
        }
    if r.returncode != 0 or not r.stdout.strip():
        return {
            "ok": False,
            "phase": "error",
            "timeout_s": timeout_s,
            "stderr": (r.stderr or r.stdout)[-500:],
            "cpu_seconds": timeout_s,
            "threads": 1,
        }
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out["timeout_s"] = timeout_s
    if "cpu_hours" in out and "cpu_seconds" not in out:
        out["cpu_seconds"] = float(out["cpu_hours"]) * 3600.0
    return out


def cell_key(lam: int, gates: int) -> str:
    return f"λ{lam}_g{gates}"


def already_done(cell: dict, attack: str) -> bool:
    m = cell.get(attack)
    if not m:
        return False
    if m.get("ok"):
        return True
    # Keep hard timeouts / skips / exhausted budgets as final.
    if m.get("phase") in ("timeout", "skipped"):
        return True
    if m.get("ok") is False and attack == "birthday" and m.get("budget"):
        return True
    if m.get("ok") is False and attack == "rho" and m.get("max_evals"):
        return True
    return False


def fmt_secs(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.3g}s"
    if s < 3600:
        return f"{s/60:.3g}m"
    if s < 86400:
        return f"{s/3600:.3g}h"
    return f"{s/86400:.3g}d"


def write_table(results: dict, path: Path) -> None:
    lines = [
        "| λ | n | m (gates) | m/(n log₂ n) | birthday | rho/DP | SAT |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    schedules = results.get("schedules", {})
    lambdas = results.get("meta", {}).get("lambdas") or [
        int(k) for k in schedules.keys()
    ]
    for lam in lambdas:
        sched = schedules.get(str(lam)) or schedules.get(lam) or []
        for gates in sched:
            cell = results["cells"].get(cell_key(lam, gates), {})
            n = 3 * lam
            dens = gates / (n * math.log2(n))
            parts = []
            for attack in ("birthday", "rho", "sat"):
                m = cell.get(attack)
                if not m:
                    parts.append("pending")
                    continue
                if m.get("ok"):
                    cpu = m.get("cpu_seconds")
                    ev = m.get("evals")
                    extra = f", {ev:.3g} evals" if ev is not None else ""
                    if attack == "sat" and m.get("clauses") is not None:
                        extra = f", {m['clauses']} clauses"
                    parts.append(f"ok {fmt_secs(cpu)}{extra}")
                else:
                    phase = m.get("phase") or "fail"
                    cpu = m.get("cpu_seconds") or m.get("wall_secs")
                    note = m.get("note") or ""
                    if phase == "skipped" and note:
                        parts.append(f"skipped")
                    else:
                        parts.append(f"{phase} {fmt_secs(cpu)}")
            lines.append(
                f"| {lam} | {n} | {gates} | {dens:.3f} | {parts[0]} | {parts[1]} | {parts[2]} |"
            )
    path.write_text("\n".join(lines) + "\n")


def save(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_density_results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_table(results, out_dir / "gate_density_table.md")
    # Also mirror table + JSON into fixtures.
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "gate_density_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    write_table(results, FIXTURES / "gate_density_table.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "target" / "security-demo" / "collision" / "gate_density",
    )
    ap.add_argument("--lambdas", default="16,32,64")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument(
        "--attacks",
        default="birthday,rho,sat",
        help="Comma list: birthday,rho,sat",
    )
    ap.add_argument(
        "--skip-birthday-lambda-ge",
        type=int,
        default=0,
        help="Skip birthday for λ ≥ this (0 = never skip). Use 64 to defer huge birthday runs.",
    )
    args = ap.parse_args()

    ensure_tools()
    lambdas = [int(x) for x in args.lambdas.split(",") if x.strip()]
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    out_dir = args.out_dir
    circuits_dir = out_dir / "circuits"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "gate_density_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"[resume] loaded {results_path}")
    else:
        results = {
            "meta": {
                "base_seed": BASE_SEED,
                "lambdas": lambdas,
                "steps": args.steps,
                "workers": args.workers,
                "gate_range": "geometric from (1/2)·n·log2(n) to n²",
                "hash": "H(x)=C(0^λ||x)_λ on width n=3λ",
            },
            "schedules": {},
            "cells": {},
        }

    results["meta"]["lambdas"] = lambdas
    results["meta"]["workers"] = args.workers
    results["meta"]["steps"] = args.steps
    # Precompute full schedules so partial saves can render the table.
    for lam in lambdas:
        results["schedules"][str(lam)] = gate_schedule(lam, args.steps)

    for lam in lambdas:
        sched = results["schedules"][str(lam)]
        print(f"== λ={lam} n={3*lam} gates={sched} ==")
        for gates in sched:
            key = cell_key(lam, gates)
            cell = results["cells"].setdefault(
                key,
                {
                    "lambda": lam,
                    "n": 3 * lam,
                    "gates": gates,
                    "density_n_log_n": gates / (3 * lam * math.log2(3 * lam)),
                    "seed": circuit_seed(lam, gates),
                },
            )
            print(f"-- {key} seed={cell['seed']} density={cell['density_n_log_n']:.3f}")
            meta = gen_circuit(circuits_dir, lam, gates)
            cell["circuit"] = {
                "g57": meta["g57"],
                "mpmct1": meta["mpmct1"],
                "meta": str(circuit_prefix(circuits_dir, lam, gates)) + ".meta.json",
                "seed": meta["seed"],
                "fixtures_dir": str(FIXTURES / f"lambda_{lam}"),
            }
            g57 = Path(meta["g57"])
            mpmct = Path(meta["mpmct1"])
            save(results, out_dir)

            if "birthday" in attacks and not already_done(cell, "birthday"):
                if args.skip_birthday_lambda_ge and lam >= args.skip_birthday_lambda_ge:
                    cell["birthday"] = {
                        "ok": False,
                        "phase": "skipped",
                        "note": (
                            f"hash-table birthday impractical at λ>={args.skip_birthday_lambda_ge} "
                            "(~2^{λ/2} table slots); use rho/DP"
                        ),
                    }
                    print("  birthday: skipped (hash-table impractical)")
                else:
                    print("  birthday: running...")
                    cell["birthday"] = measure_birthday(g57, lam, seed=cell["seed"])
                    b = cell["birthday"]
                    print(
                        f"  birthday: ok={b.get('ok')} cpu_s={b.get('cpu_seconds')} "
                        f"evals={b.get('evals')}"
                    )
                save(results, out_dir)

            if "rho" in attacks and not already_done(cell, "rho"):
                print("  rho: running...")
                cell["rho"] = measure_rho(
                    g57, lam, workers=args.workers, seed=cell["seed"]
                )
                r = cell["rho"]
                print(
                    f"  rho: ok={r.get('ok')} cpu_s={r.get('cpu_seconds')} "
                    f"evals={r.get('evals')}"
                )
                save(results, out_dir)

            if "sat" in attacks and not already_done(cell, "sat"):
                timeout = sat_timeout_for(lam, gates)
                print(f"  sat: running (timeout={timeout:.0f}s)...")
                cell["sat"] = measure_sat(mpmct, lam, timeout_s=timeout)
                s = cell["sat"]
                print(
                    f"  sat: ok={s.get('ok')} phase={s.get('phase')} "
                    f"cpu_s={s.get('cpu_seconds')} clauses={s.get('clauses')}"
                )
                save(results, out_dir)

    save(results, out_dir)
    print("\n== table ==")
    print((out_dir / "gate_density_table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
