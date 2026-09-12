#!/bin/bash
hostname
ls ~/frozen_curated_m1_m11_native | wc -l
sha256sum ~/frozen_curated_m1_m11_native/filters.bin | cut -c1-16
ls -d ~/frozen_curated_m1_m11 ~/frozen_curated_m1_m11_native_v1 2>/dev/null || echo "only v2 present"
