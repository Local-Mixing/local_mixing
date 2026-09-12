#!/bin/bash
hostname
du -sh ~/frozen_curated_m1_m11_native
ls ~/frozen_curated_m1_m11_native | wc -l
ls -d ~/frozen_curated_m1_m11_native_v1 >/dev/null 2>&1 && echo "rollback _v1 present"
ls ~/frozen_curated_m1_m11_native/tables.bin >/dev/null && echo "tables.bin ok"
