#!/bin/bash
hostname
ls -la ~/frozen_curated_m1_m11_native_v1/filters.bin 2>/dev/null || echo "old curated native: no filters.bin"
ls -la ~/frozen_curated_m1_m11/filters.bin 2>/dev/null || echo "legacy curated: no filters.bin"
ls -la ~/frozen_m1_m11/filters.bin 2>/dev/null || echo "regular: no filters.bin"
grep -hs 'VALUE_CONVENTION\|FROZEN_FILTER' ~/*.sh 2>/dev/null | sort -u | head -5
