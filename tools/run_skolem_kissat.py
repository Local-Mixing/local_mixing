#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from pathlib import Path


CONFIGS = {
    "conservative": [
        "--congruence=false",
        "--eliminatebound=8192",
        "--eliminateocclim=100000",
        "--eliminaterounds=10000",
        "--lucky=false",
        "--preprocessrounds=2",
        "--restartint=50",
        "--statistics",
        "--substitute=false",
        "--sweep=false",
        "--target=2",
    ],
    "aggressive": [
        "--sat",
        "--statistics",
        "--lucky=false",
        "--eliminatebound=8192",
        "--eliminateocclim=100000",
        "--eliminaterounds=10000",
        "--preprocessrounds=3",
    ],
    "aggressive_more": [
        "--sat",
        "--statistics",
        "--lucky=false",
        "--eliminatebound=8192",
        "--eliminateocclim=100000",
        "--eliminaterounds=10000",
        "--preprocessrounds=5",
        "--substituterounds=100",
        "--sweepcomplete=true",
        "--sweepeffort=10000",
        "--proberounds=10",
        "--factoreffort=1000000",
        "--factorstructural=true",
    ],
}


def parse_summary(path):
    if not path.exists():
        return {}
    data = {"path": str(path)}
    for line in path.read_text().splitlines():
        parts = line.split()
        if line.startswith("c result ") and len(parts) >= 3:
            data["result"] = parts[2]
        elif line.startswith("p vars "):
            for i in range(1, len(parts) - 1, 2):
                data[parts[i]] = int(parts[i + 1])
        elif line.startswith("s active "):
            for i in range(1, len(parts) - 1, 2):
                data[parts[i]] = int(parts[i + 1])
        elif line.startswith("r frozen_first "):
            for i in range(1, len(parts) - 1, 2):
                data["frozen_" + parts[i]] = int(parts[i + 1])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kissat", default="work/sss_challenge/kissat_src_verbose/build/kissat")
    parser.add_argument("--cnf", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--ignore-models", type=int, default=0)
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()

    base = Path("work/sss_challenge") / f"{args.name}_skolem_{args.config}"
    if args.ignore_models:
        base = base.with_name(base.name + f"_ignore{args.ignore_models}")
    if args.map:
        base = base.with_name(base.name + "_map")
    if args.extend:
        base = base.with_name(base.name + "_full")

    summary = base.with_suffix(".summary")
    log = base.with_suffix(".log")

    env = os.environ.copy()
    env["KISSAT_FREEZE_EXTERNAL_RANGE"] = args.freeze
    env["KISSAT_DUMP_SKOLEM"] = str(summary)
    if args.ignore_models:
        env["KISSAT_IGNORE_MODELS"] = str(args.ignore_models)
    if args.map:
        env["KISSAT_DUMP_SKOLEM_MAP"] = "1"
    if args.extend:
        env["KISSAT_DUMP_SKOLEM_EXTEND"] = "1"

    cmd = [args.kissat, f"--time={args.seconds}", *CONFIGS[args.config], args.cnf]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    elapsed = time.time() - t0
    log.write_text(proc.stdout)

    summary_data = parse_summary(summary)
    bits = [
        f"name={args.name}",
        f"config={args.config}",
        f"ignore={args.ignore_models}",
        f"returncode={proc.returncode}",
        f"elapsed={elapsed:.3f}",
    ]
    for key in ["result", "live", "live_original", "live_extension", "fixed", "eliminated_stack", "extend", "frozen_live", "frozen_eliminated"]:
        if key in summary_data:
            bits.append(f"{key}={summary_data[key]}")
    print(" ".join(bits))


if __name__ == "__main__":
    main()
