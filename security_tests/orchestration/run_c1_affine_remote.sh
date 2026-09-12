#!/usr/bin/env bash
set -uo pipefail

umask 077
source_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="${SECURITY_REPO_ROOT:-$(cd -- "$source_dir/../.." && pwd)}"
artifact_root="${SECURITY_ARTIFACT_ROOT:-$repo_root/red_team_tests}"
# Optional arguments: existing run directory, existing tools directory.
root_dir="${1:-${SECURITY_RUN_DIR:-$artifact_root/_orchestration}}"
tools_dir="${2:-${SECURITY_TOOLS_DIR:-$root_dir}}"
status_file="$root_dir/status.txt"
log_file="$root_dir/hmap_affine.log"

write_status() {
    local next_status="$1"
    printf '%s\n' "$next_status" >"$status_file.tmp"
    chmod 600 "$status_file.tmp"
    mv -f -- "$status_file.tmp" "$status_file"
}

write_status RUNNING
"$tools_dir/hmap_affine" \
    --c "$root_dir/source_c.g57" \
    --g "$root_dir/c1_final.txt" \
    --n 128 \
    --degree 1 \
    --c-step 210 \
    --g-step 74690 \
    --batches 96 \
    --train-batches 72 \
    --seed 12345 \
    --out "$root_dir/c1_final_d1" \
    >"$log_file" 2>&1
run_rc=$?

if [[ $run_rc -eq 0 ]]; then
    write_status COMPLETE
else
    printf '%s\n' "$run_rc" >"$root_dir/exit_code.txt"
    chmod 600 "$root_dir/exit_code.txt"
    write_status FAILED
fi
exit "$run_rc"
