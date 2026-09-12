# No-Gray Phase-A experiment (2026-08-12)

## Question

Can the aggregate Gray fold be removed without giving up most of the frozen
database's usefulness in Phase A?

The reason to remove it is a space-time identity. If a Gray accumulator enters
with unknown value `u0` and the gather leaves `u0 + M_i + delta`, two snapshots
of that physical wire reveal the complete aggregate mask `M_i` (up to the known
residual). Together with the entry carrier this recovers the logical operand.
Splitting the mask into Gray tiles does not remove the identity: XORing every
tile delta reconstructs `M_i`.

## Arms

All arms used the same n=16 sliced sandwich, source seed, gadget seed, and mask
plan `[2,2,2,3]` on 64 physical wires.

| Arm | Gates | <=2 controls | Regular-store reachable |
|---|---:|---:|---:|
| Aggregate Gray | 31,691 | 97.59% | 97.37% |
| No Gray, selective ladder cap 4 | 78,730 | 97.26% | 92.22% |
| No Gray, full width-2 ladder | 107,894 | 99.90% | 92.37% |
| Selective cap 4 then `fcompress` | 65,870 | 96.73% | 77.96% |
| Full ladder then `fcompress` | 92,421 | 99.88% | 73.06% |

`blocker_census` used the frozen regular `m1..m11` store, contiguous windows
2..5, control cap 2, degree cap 9, four degree probes, 1,024 per-wire terms,
2,048 total terms, and seed 7707. This is the historical apples-to-apples
reachability audit; it is not a complete simulation of the current GSS sampler.

The current GSS geometry was therefore measured separately with 20,000 dry
rounds per mode, regular plus curated stores, MIX length 6 (50% convex), and
COMP convex length 12 / contiguous length 6 with prefix descent:

| Arm | MIX replacement rounds | COMP prefix hits / probes |
|---|---:|---:|
| Aggregate Gray | 12,240 / 20,000 = 61.20% | 42,539 / 467,724 = 9.09% |
| No Gray, selective cap 4 | 9,603 / 19,989 = 48.04% | 19,484 / 468,396 = 4.16% |
| No Gray, full ladder | 8,718 / 20,000 = 43.59% | 14,035 / 468,312 = 3.00% |

The dry-run COMP hit counter counts matching prefixes, so it can exceed the
round count. The denominator above is `hits + misses` from fmix's DB counters.

## Conclusions

1. Plain `fcompress` is not a replacement for Gray. It cannot reduce the
   algebraic degree of the wide fold material without helper wires.
2. The existing Phase-B splitter is also not a replacement. It presplits
   complemented g57 gates; the no-Gray fold fossils are ordinary wide
   conjunctions. In the paired n=8 test, split then compress left 51.10% of
   gates wider than the Phase-A seed cap.
3. Pre-Phase-A compression is actively harmful. It removes easy narrow bulk
   and disrupts the g57-friendly local spelling that the frozen store matches.
4. Full narrowing buys only 0.15 percentage points of blocker reach over cap 4,
   while adding 37% more gates and performing worse in both current samplers.
5. The best tested no-Gray point is therefore selective construction-time
   narrowing through width four, with no pre-Phase-A `fcompress`.

Run it with:

```bash
export PROD_PRESET=no-gray-phase-a
scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_safe_s1 -s 1
```

This preset changes only `gray_fold=0` and `ladder_cap=4` relative to the
production preset. It is not promoted to the default: it costs about 2.48x as
many stage-2 gates and retains about 78% of Gray's MIX hit rate. It does,
however, preserve most frozen-store reach without ever gathering the complete
operand mask onto one accumulator.

## Construction hardening included with the experiment

For a non-Gray fold, the operand decode constant is absorbed into the carrier
literal's polarity (`C + 1 = !C`) instead of becoming an empty atom. Thus every
arity-2 product fragment contains a nonempty atom from both operands; there is
no constant-times-mask fragment that needs a one-operand dirty ladder. The full
narrow path also retains selected atom boundaries when calling the dirty-ladder
emitter, so its existing whole-atom pivot guard is no longer silently disabled.

The checked-in Gray witness test constructs identical masks in its Gray and
expanded arms before changing the fold strategy. This fixes the earlier test's
RNG-control error, where different pre-injection configs produced different
masks and invalidated the A/B assertion.

This addresses the compact Gray-specific `M_i` disclosure. It is not a proof
that the full gadget is secure against arbitrary whole-trace algebra; the wider
additive-sharing construction still requires a separate end-to-end trace model.
