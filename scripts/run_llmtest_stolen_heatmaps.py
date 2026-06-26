#!/usr/bin/env python3
"""Run selected very-enhanced heatmaps on idle llmtest."""

from __future__ import annotations

import subprocess
from pathlib import Path

from recompute_very_enhanced_heatmaps import OUT, PNG_OUT, SERVERS, remote_script, ssh


JOBS = [
    "sss_sr2_ta1_r1_m1_x20",
    "sss_sr2_ta2_r1_m1_x10",
]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True)


def main() -> int:
    PNG_OUT.mkdir(parents=True, exist_ok=True)
    target = SERVERS["llmtest"]

    for job_id in JOBS:
        filename = f"{job_id}_mixing_heatmap_very_enhanced.png"
        local_png = PNG_OUT / filename
        if local_png.exists():
            continue

        result = ssh(target, remote_script(job_id), check=False)
        run(
            [
                "rsync",
                "-az",
                f"{target}:~/local_mixing/sgc/parameter_test/{filename}",
                str(PNG_OUT) + "/",
            ]
        )

        log_name = f"{job_id}_very_enhanced_heatmap.log"
        run(
            [
                "rsync",
                "-az",
                f"{target}:~/local_mixing/sgc/parameter_test/{log_name}",
                str(OUT) + "/",
            ]
        )

        status = "complete" if local_png.exists() else f"failed ({result.returncode})"
        print(f"{job_id}: {status}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
