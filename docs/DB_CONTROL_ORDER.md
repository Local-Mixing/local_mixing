# Deployed database control order

Read-only check on 2026-09-09 of the four current servers supplied by the owner.
The normal GSS setting is `database.curated_control_order = "native"`, which is
also the default. No legacy control swap was needed in the sampled current data.

| Server | Public address | Curated native matches | Regular native matches | Swapped matches |
| --- | --- | --- | --- | --- |
| sattesting nh | 129.114.109.41 | 355/355 | 128/128 | 0 |
| llmtest | 129.114.108.159 | 355/355 | 128/128 | 0 |
| n64tests | 129.114.109.63 | 355/355 | 128/128 | 0 |
| testpieces | 129.114.109.6 | 355/355 | 128/128 | 0 |

All four have `/home/cc/frozen_curated_m1_m11_native` and
`/home/cc/frozen_m1_m11`, each with 256 shards. The historical top-level
`frozen_curated_m1_m11` and `frozen_curated_m1_m11_native_v1` directories
were absent. This inventory does not claim that no archive contains older data.

The check read the deployed `tables.bin` and sampled 32 nonempty buckets from
each of shards 00, 55, aa and ff in each store. Bucket positions were dispersed
deterministically; buckets larger than 64 KiB were excluded. It decoded up to
eight candidates from the first entry of each bucket. Each circuit was
recanonicalized with the current native polynomial implementation and its
76 stored key bits compared with the sampled entry's shard/bucket/tail.
Repeating the comparison after swapping the two controls produced no matches.
There were no native mismatches or skipped canonicalizations. Samples came
from the servers' files; no local database was substituted.

These are bounded convention checks, not an exhaustive integrity audit of
every entry. The fleet's older `native_verify.out` records 11,858,820 checks
with zero mismatches, but predates the currently deployed rebuild and was not
used as proof for these current files. No database files were modified.

The reader retains `legacy-swapped-controls` only for historical data and saved
recipes. The current template leaves this compatibility field commented out;
normal new runs use native ordering automatically. The legacy campaign in
[`CURATED_DB_COMPARISON.md`](history/research/CURATED_DB_COMPARISON.md) describes an older store.
