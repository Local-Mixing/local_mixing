#!/bin/bash
# Quick idle/disk/store census for one fleet box (run remotely via ssh 'bash -s').
hostname
uptime
df -h / | tail -1
free -g | head -2
echo "--- curated_full_20260813:"
ls -la ~/curated_full_20260813/ 2>/dev/null || echo "  (absent)"
echo "--- frozen stores:"
ls -d ~/frozen* 2>/dev/null
echo "--- busy processes:"
ps -eo pcpu,comm --sort=-pcpu | head -6
echo "--- repo:"
# Keep bash -s use independent of sibling source files on the remote host.
DB_GEN_REPO_ROOT=${DB_GEN_REPO_ROOT:-"$HOME/local_mixing"}
ls -d "$DB_GEN_REPO_ROOT" 2>/dev/null && cd -- "$DB_GEN_REPO_ROOT" && git log --oneline -1 2>/dev/null
