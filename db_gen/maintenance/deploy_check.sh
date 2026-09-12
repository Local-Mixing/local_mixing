#!/bin/bash
# Per-target pre-deployment census.
hostname
df -h / | tail -1
ls -d ~/frozen_curated* 2>/dev/null
du -sh ~/frozen_curated_m1_m11_native 2>/dev/null
grep -hs 'FROZEN_CURATED_DIR' ~/*.sh 2>/dev/null | sort -u | head -3
