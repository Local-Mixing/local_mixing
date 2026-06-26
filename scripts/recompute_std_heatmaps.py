#!/usr/bin/env python3
"""Recompute parameter-test heatmaps in standard-deviation mode."""

from __future__ import annotations

import argparse
import csv
import shutil
import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sgc" / "parameter_test"
X0 = 8_675_309

SERVERS = {
    "n64test": "cc@129.114.109.63",
    "testpieces": "cc@129.114.109.6",
    "nho": "cc@129.114.108.170",
    "localtest": "cc@129.114.109.32",
    "llmtest": "cc@129.114.108.159",
}

STD_DIRS = {
    "pngs": OUT / "std_pngs",
    "incremental": OUT / "std_pngs_incremental",
    "very_enhanced": OUT / "std_png_very_enhanced",
    "original": OUT / "std_original",
    "random": OUT / "std_random",
    "r2": OUT / "std_r2",
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


def ssh(target: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        ["ssh", "-n", target, f"bash -lc {shlex.quote(command)}"],
        check=check,
    )


def rsync_from(target: str, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", f"{target}:~/local_mixing/{remote}", str(local)], check=False)


def heatmap_command(
    c1: str,
    c2: str,
    output: str,
    *,
    xlabel: str,
    ylabel: str,
    incremental: bool,
) -> str:
    command = [
        "python3",
        "./heatmap/heatmap.py",
        "--n",
        "128",
        "--i",
        "100",
        "--x",
        xlabel,
        "--y",
        ylabel,
        "--c1",
        c1,
        "--c2",
        c2,
        "--path",
        output,
        "--std",
    ]
    if incremental:
        command.extend(["--incremental", "--x0", str(X0)])
    return " ".join(shlex.quote(part) for part in command)


def computed_jobs(alias: str) -> list[str]:
    summary = OUT / f"{alias}_worker_summary.csv"
    with summary.open(newline="", encoding="utf-8") as source:
        return [
            row["job_id"]
            for row in csv.DictReader(source)
            if row.get("heatmap_status") == "computed"
        ]


def prepare_remote(target: str) -> None:
    ssh(
        target,
        "cd ~/local_mixing && mkdir -p ./sgc/parameter_test/std_work "
        "&& source ./.venv/bin/activate && maturin develop "
        ">> ./sgc/parameter_test/std_work/maturin.log 2>&1",
    )


def main_worker(alias: str) -> int:
    target = SERVERS[alias]
    for directory in (
        STD_DIRS["pngs"],
        STD_DIRS["incremental"],
        STD_DIRS["very_enhanced"],
    ):
        directory.mkdir(parents=True, exist_ok=True)

    prepare_remote(target)
    for job_id in computed_jobs(alias):
        c1 = f"./gadgetized/parameter_test_n64m1000_{job_id}.txt"
        c2 = f"./sgc/parameter_test/{job_id}.txt"

        normal_name = f"{job_id}_mixing_heatmap.png"
        normal_local = STD_DIRS["pngs"] / normal_name
        very_name = f"{job_id}_mixing_heatmap_very_enhanced.png"
        very_local = STD_DIRS["very_enhanced"] / very_name
        remote_normal = f"./sgc/parameter_test/std_work/{job_id}_std.png"
        if not normal_local.exists():
            command = heatmap_command(
                c1,
                c2,
                remote_normal,
                xlabel="Gadgetized source",
                ylabel="Final circuit",
                incremental=False,
            )
            log = f"./sgc/parameter_test/std_work/{job_id}_std.log"
            ssh(
                target,
                f"cd ~/local_mixing && source ./.venv/bin/activate && "
                f"{command} > {shlex.quote(log)} 2>&1",
                check=False,
            )
            rsync_from(target, remote_normal.removeprefix("./"), normal_local)
        if normal_local.exists() and not very_local.exists():
            shutil.copy2(normal_local, very_local)

        incremental_name = f"{job_id}_mixing_heatmap_incremental.png"
        incremental_local = STD_DIRS["incremental"] / incremental_name
        remote_incremental = (
            f"./sgc/parameter_test/std_work/{job_id}_std_incremental.png"
        )
        if not incremental_local.exists():
            command = heatmap_command(
                c1,
                c2,
                remote_incremental,
                xlabel="Gadgetized source",
                ylabel="Final circuit",
                incremental=True,
            )
            log = f"./sgc/parameter_test/std_work/{job_id}_std_incremental.log"
            ssh(
                target,
                f"cd ~/local_mixing && source ./.venv/bin/activate && "
                f"{command} > {shlex.quote(log)} 2>&1",
                check=False,
            )
            rsync_from(
                target,
                remote_incremental.removeprefix("./"),
                incremental_local,
            )
    return 0


def create_and_copy(
    target: str,
    *,
    c1: str,
    c2: str,
    destination: Path,
    aliases: list[Path],
    xlabel: str,
    ylabel: str,
    incremental: bool,
) -> None:
    if not destination.exists():
        remote = f"./sgc/parameter_test/std_work/{destination.name}"
        command = heatmap_command(
            c1,
            c2,
            remote,
            xlabel=xlabel,
            ylabel=ylabel,
            incremental=incremental,
        )
        log = f"{remote}.log"
        ssh(
            target,
            f"cd ~/local_mixing && source ./.venv/bin/activate && "
            f"{command} > {shlex.quote(log)} 2>&1",
            check=False,
        )
        rsync_from(target, remote.removeprefix("./"), destination)
    if destination.exists():
        for alias in aliases:
            alias.parent.mkdir(parents=True, exist_ok=True)
            if not alias.exists():
                shutil.copy2(destination, alias)


def nho_extras() -> int:
    target = SERVERS["nho"]
    STD_DIRS["original"].mkdir(parents=True, exist_ok=True)
    STD_DIRS["random"].mkdir(parents=True, exist_ok=True)
    prepare_remote(target)

    original_base = "sss_sr2_ta3_r1_m1_x20_original_c1"
    original_c1 = "./circuits/parameter_test_n64m1000.txt"
    final_c2 = "./sgc/parameter_test/sss_sr2_ta3_r1_m1_x20.txt"
    create_and_copy(
        target,
        c1=original_c1,
        c2=final_c2,
        destination=STD_DIRS["original"] / f"{original_base}_heatmap.png",
        aliases=[
            STD_DIRS["original"] / f"{original_base}_enhance_0.01.png",
            STD_DIRS["original"] / f"{original_base}_enhance_0.05.png",
        ],
        xlabel="Original",
        ylabel="Final circuit",
        incremental=False,
    )
    create_and_copy(
        target,
        c1=original_c1,
        c2=final_c2,
        destination=(
            STD_DIRS["original"] / f"{original_base}_enhance_0.01_incremental.png"
        ),
        aliases=[
            STD_DIRS["original"]
            / f"{original_base}_enhance_0.05_incremental.png"
        ],
        xlabel="Original",
        ylabel="Final circuit",
        incremental=True,
    )

    for gates in (10_844, 1_000):
        stem = f"sss_sr2_ta3_r1_m1_x20_random_n128_m{gates}_c1_enhance_0.01"
        random_c1 = f"./sgc/parameter_test/random/random_n128_m{gates}.txt"
        create_and_copy(
            target,
            c1=random_c1,
            c2=final_c2,
            destination=STD_DIRS["random"] / f"{stem}.png",
            aliases=[],
            xlabel=f"Random 128-wire {gates}-gate circuit",
            ylabel="Final circuit",
            incremental=False,
        )
        create_and_copy(
            target,
            c1=random_c1,
            c2=final_c2,
            destination=STD_DIRS["random"] / f"{stem}_incremental.png",
            aliases=[],
            xlabel=f"Random 128-wire {gates}-gate circuit",
            ylabel="Final circuit",
            incremental=True,
        )
    return 0


def r2_extra() -> int:
    target = SERVERS["testpieces"]
    STD_DIRS["r2"].mkdir(parents=True, exist_ok=True)
    prepare_remote(target)
    base = "sss_sr1_ta4_r2_m1_x20"
    create_and_copy(
        target,
        c1=f"./gadgetized/parameter_test_n64m1000_{base}.txt",
        c2=f"./sgc/parameter_test/r2/{base}.txt",
        destination=STD_DIRS["r2"] / f"{base}_mixing_heatmap_enhance_0.01.png",
        aliases=[
            STD_DIRS["r2"] / f"{base}_mixing_heatmap_very_enhanced.png"
        ],
        xlabel="Gadgetized source",
        ylabel="Final r2 circuit",
        incremental=False,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=sorted(SERVERS))
    parser.add_argument("--nho-extras", action="store_true")
    parser.add_argument("--r2-extra", action="store_true")
    args = parser.parse_args()
    if args.nho_extras:
        return nho_extras()
    if args.r2_extra:
        return r2_extra()
    if not args.server:
        parser.error("--server, --nho-extras, or --r2-extra is required")
    return main_worker(args.server)


if __name__ == "__main__":
    raise SystemExit(main())
