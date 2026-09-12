#!/bin/bash
hostname
du -sh ~/frozen_m1_m11 2>/dev/null || echo "NO regular store"
ls ~/frozen_m1_m11/filters.bin > /dev/null 2>&1 && echo "regular filters.bin ok"
du -sh ~/frozen_curated_m1_m11_native ~/frozen_curated_DEFAULT_native 2>/dev/null
df -h / | tail -1
