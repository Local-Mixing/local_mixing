# Punchlist

Things to fix. Add new items at the bottom with a date; strike them through or
move them to Done when resolved.

## Open

8. **Kill the celebrity-value rebirths in the fire brackets (drift
   investigation parked; safe alternative identified).** (2026-10-06)

   Mechanism FOUND: the dirty-ancilla brackets in `product_fire`
   (embedded_masking.rs) re-materialise each ancilla's base value
   verbatim at every final restore -- 129k gap-2 restore events on the
   n=64 artifact, ~430 band values reborn 65..1,700x, 22% of segment
   mass, and recurrence >= 4 implies boundary exposure deterministically
   (RESULTS.md 2026-10-06). The restores are NOT bloat (99.98% of
   intermediates are read); the problem is only the exact-restore.

   ATTEMPTED FIX (code present, env-gated OFF: EMBEDDED_MASKING_DRIFT=1):
   drop the final restore when the band ancilla is outside the fire's
   stability set, letting it drift by the bracket product. Constraint
   map established and enforced: (a) open-mask support incl. open_lin,
   (b) fire cover + straddle (straddle now drawn BEFORE unit emission),
   (c) ancilla unique to its unit (units are shuffled and stop
   commuting if a drifted ancilla is shared), (d) shuffling partners
   exclude all in-box-written band roles (preprocessing_shuffling.rs,
   kept -- conservative hardening), (e) data ancillas always restore.
   RESIDUAL BUG: with balanced=true and DIRTY band fixtures, data
   outputs still pick up initial-band dependence even at a single
   drifted ancilla in fire 0 (test
   preprocessing_shuffling_preserves_exhaustive_data_outputs...,
   np=6/r=24 fixture; bisect via EMBEDDED_MASKING_DRIFT_ONLY=lo:hi;
   gate dumps via EMBEDDED_MASKING_DUMP_GATES=path, append-mode with
   headers; drift decisions via EMBEDDED_MASKING_DRIFT_DEBUG). One
   band-stability consumer is still unidentified.

   STATUS 2026-10-07 overnight: BOTH legs implemented and default-ON,
   suite green (333/334). (a) The drift leg is CORRECT -- the last
   stability consumer was the mid-fire ensure_min_open on the fire
   target (its cycles are now pre-drawn before the restore-drop pass;
   stability set computed live at drop time) -- but it is nearly INERT
   at production parameters: with 128 data wires x 2-3 open masks the
   live support covers almost the whole band, so few restores are
   eligible to drop. (b) Burst hot-targeting (rerand slots pick the
   band wire with most bracket borrows since its last write): top
   multiplicity 1,219 -> 453. RESIDUAL: a NEW celebrity class -- single
   nonlinear values born ~450x across 50-77 MIXED data/band wires,
   still gap-2 -- partially characterised (suspect: bracket pair-product
   deltas with shared bases, or transfer-drain values); diagnosis notes
   and tools in RESULTS.md. Remaining work: identify and kill the
   multi-wire class; consider raising slot rate for hot wires.
   UPDATE (same night): the multi-wire class is IDENTIFIED -- long-lived
   ROLE values re-materialised by the migration drain (carrier ^=
   partner at every box end restores the role bare on its current
   wire; arithmetic matches ~450 births over ~70 wires for inert mid
   roles). Fix direction: track the transfer mask in the ledger
   (open_lin) instead of draining per box -- see RESULTS.md addendum.

   SAFE ALTERNATIVE (likely the better fix): refresh hot band values
   through the EXISTING rerand-burst path, which already carries the
   open-mask repair machinery (is_repair slots close/reopen every
   referencing mask around the write). Add read-count/recurrence-
   triggered slot priority targeting the ~430 celebrity wires; zero
   new invariants. The 8-bracket ancilla-order randomisation (shipped,
   tests green) already halves the 8-gate brackets' rebirth sites.


## Done

1. **Canonicalize the window's gate order at the start of every DB lookup
   (regular and curated).** (2026-09-27)

   The circuits stored in the DB are already in canonicalized gate order, so
   the sampled window should be put in the same canonical order before lookup.
   Today the window is looked up as sampled: the function key is
   order-invariant (it hashes the canonical polynomials), but the post-lookup
   guards compare the decoded candidate against the window's *sampled* gate
   sequence — so a candidate that is the window up to reordering of commuting
   gates passes the exact-equality identity guard and is only caught by the
   gate-multiset reorder ban (`ban_reorder`), which is armed solely for
   2-gate pair rounds and bridge endpoint windows. With the window's gate
   order canonicalized to match the store's convention, such trivial
   re-spellings become detectable by direct comparison at any window length.

   Where: `db_replace_with_value_options` in
   `src/stages/db_mixing/replacement.rs` (lookup entry, identity guard at the
   `gates == window` check, reorder ban at `is_reorder`); window
   canonicalization currently only canonicalizes wires/polynomials via
   `canonicalize_xgates_single` in `src/canonicalization/xgate.rs`.

2. **A curated lookup gives no way to tell which returned candidates are
   "meaningfully different" from the looked-up window `w`.** (2026-09-27)

   HISTORIC on this branch (2026-10-08): the curated DB and every file named
   below were removed by the four-stage cleanup; they exist only on the
   archive branch `ran/archive/curated-and-postmix`.

   The store's difference-by-construction argument (see the routing comment at
   `curated_armed_for` in `src/stages/db_mixing/replacement.rs`) guarantees
   that a candidate `Q⁻¹` is meaningfully different from *its own complement*
   `P` within the identity it was split from — not from `w`, the particular
   spelling sampled from the circuit. Provenance (source identity, complement,
   split point) is discarded at build time: composite records are
   `[function key][circuit bytes]` with empty values, and the frozen value is
   a flat undifferentiated candidate chain. So candidates from unrelated
   identities pool under `w`'s key, some possibly trivially related to `w`,
   and the selector draws among them blind. That candidates are pairwise
   different from their *own* complements is nice but not enough. The only
   lookup-time structural checks today are exact sequence equality (identity
   guard) and, on pair/bridge rounds only, gate-multiset equality
   (`ban_reorder`).

   Wanted: a way to know — or enforce — that a selected candidate is
   meaningfully different from `w` itself. Possible ingredients already in the
   tree: the sieve's relabeled-shingle similarity machinery (`relabel_word` /
   shared 6-gate subword rejection in `db_gen/curated_full.rs`, and
   `db_gen/analysis/compare_similarity.sh`) could be applied at selection time
   between `w` and the drawn candidate; alternatively, retain enough build
   provenance to know each candidate's complement relation.

3. **Recover or rebuild the curated frozen store
   (`frozen_curated_m1_m11_native`) — the canonical copy is likely lost.**
   (2026-09-27)

   HISTORIC on this branch (2026-10-08): the curated DB and every file named
   below were removed by the four-stage cleanup; they exist only on the
   archive branch `ran/archive/curated-and-postmix`.

   The store (3.4 GiB, 256 shards, ~32M entries, native convention) lived at
   `/home/cc/frozen_curated_m1_m11_native` on Chameleon bare-metal leases
   ("ran2" at .242, "rtests" at .89) that TERMINATED 2026-09-20/21, wiping
   their disks. It was never uploaded to the public `frozen-db-container`
   (only the regular store is there). Known surviving artifacts, both
   unverified: `fuller-db-container/rocks_curated_db/` in Chameleon object
   storage (~1.6 GB RocksDB, lineage unknown, not frozen format — if it is
   the legacy append-value curated RocksDB, `build_curated_full
   import-legacy-rocks` + `frozen_from_lmdb --curated --composite` converts
   it; if composite, it freezes directly), and possibly a copy on the still
   -live `db_temp_storage` instance (129.114.108.11, unchecked). The stale
   `/home/cc/curated_native.tgz` (Aug-04, 11.8M entries) died with the lease
   anyway. Fallback: rebuild from the local regular frozen store via
   `build_curated_full from-frozen-identities[-v2]
   frozen_db/frozen_m1_m11 OUT` (docs/frozen_database.md), cost unmeasured on
   this machine. Whatever recovery is chosen, upload the result next to the
   regular store in object storage so this cannot recur, and record its
   construction provenance per docs/frozen_database.md.

   Update 2026-09-28: `rocks_curated_db` was recovered from object storage
   (1.6 GB, MD5-verified) and imported to
   `frozen_db/curated_composite_legacy_import` (audit PASS): 25,901,280 keys,
   81,970,096 exact candidates, max 30 candidates / 512 bytes per key — it is
   the OLD BOUNDED curated build (July, pre-2ed0222a, so requires
   legacy-swapped-controls convention), not the lost uncapped store (38.5M
   entries, max 207,307 candidates/key, native). Candidates truncated by the
   512-byte cap are unrecoverable from it. Remaining routes to a full store:
   v2 rebuild from the local regular frozen store (measured 75 s/shard,
   0.5 GB RSS → ~5.4 h serial for 256 shards on this Mac), or a possible copy
   on the unchecked `db_temp_storage` instance (129.114.108.11).

4. **Preprocessing: data variables should switch carrier wires at each
   write.** (2026-10-01)

   Today the TDP pipeline pins every data role to its physical wire for the
   whole gadget: `tdp_gen.sh` defaults `shuffling_segments=0`, and even when
   preprocessing shuffling is enabled (`EMBEDDED_MASKING_SHUFFLING>=8`) it
   moves only the write-only target role, a few segment cuts per fire box,
   with mandatory return-home after every original gate
   (`src/stages/preprocessing/preprocessing_shuffling.rs`). Wanted: a carrier
   transfer at EVERY write of a data variable, so no wire identifies a
   variable across writes. The output-port contract still requires routing
   roles home at the gadget boundary (or an encoded-I/O variant).

   RESOLVED (RC sign-off 2026-10-05): 1 = canonical-gate-order identity
   guard (working tree, from the canon_ab session; still uncommitted);
   2 = v3 true-friend curated stores (circuit-keyed, chord-tight /
   leeway-with-floor; runs/ncdb_ab_20261001, docs/CURATED_DB_V3_SUMMARY);
   3 = legacy curated store recovered + v3 pools/stores rebuilt locally
   (frozen_db/, build logs there); 4 = persistent carrier migration
   (preprocessing_shuffling return_home=false with stream-tail port
   restoration, 2026-10-05, working tree; segments-per-fire-box, not yet
   literally every write).

   HISTORIC (2026-10-08, four-stage TDP): the curated DB (items 2 and 3,
   including the v3 stores and `docs/CURATED_DB_V3_SUMMARY`) was removed from
   the pipeline; its code, builders and write-up remain on the branch
   `ran/archive/curated-and-postmix`. Stage 3 uses the regular store only.

6. **Preprocessing: avoid wire segments with too many consecutive reads —
   they are hard to mix later.** (2026-10-01)

   2026-10-05 verification: NOT resolved by persistent carrier migration —
   read-run census on the same n=64 recipe: baseline max 285 / 789 runs
   >=100; persistent-8: 348/572; persistent-32: 267/307. Dose helps but
   long runs persist: the heavy readers are band wires sitting in OPEN
   MASK SUPPORT, which are excluded as transfer partners exactly while
   they are being read. The original fix direction stands: read-count-
   triggered rerand priority (bump a band wire once reads-since-write
   passes a threshold).

   Measured on `runs/canon_ab_20260928/stage2_tdp.mpmct1` (702,872 gates):
   read-runs between writes reach 389 consecutive reads on one band wire over
   a ~10K-gate span (32 runs >= 200, 739 >= 100, essentially all on band
   wires). Cause: band wires are written only by count-scheduled rerand slots
   (random band wire per slot, median gap ~530 gates, tail ~10K) while a wire
   sitting in open masks of busy operands is read by every fire polynomial
   (~90 fires in the top run). Nothing bounds reads-since-last-write. Fix
   direction: read-count-triggered refresh — bump a band wire's rerand
   priority once its reads since its last write pass a threshold, capping run
   length at a few extra slots' cost.

   RESOLVED 2026-10-05: transfer-partner selection in
   preprocessing_shuffling now prefers the band wire with the most READS
   SINCE ITS LAST PHYSICAL WRITE (replacing least-write-load, which
   favored cold wires); every carrier transfer then doubles as a
   read-run cap. Measured on the n=64 recipe (persistent migration):
   baseline max run 285 / 17 runs >=200 / 789 >=100; pressure-8:
   245/1/196; pressure-32: 141/0/62. Residual ~100-140 runs occur
   during open-mask-support exclusion windows; if ever needed, the
   remaining headroom is a min-open-aware relaxation of that exclusion
   (an open-mask partner is a legitimate mask rotation whenever the
   open count stays above min_open).

5. **Preprocessing: the initial band seeding is too thin — make it at least
   as substantial as the slicing.** (2026-10-01)

   Module 2 (`seed_band_mode`, src/stages/preprocessing/embedded_masking.rs)
   is 2 linear CNOTs per band wire — 256 gates on the n=64 TDP (`b = x_i ^
   x_j`), vs 1,280 gates per slice guard. The band's whole security story
   rests on its fill; the seed (and the module-4 reseed, same generator)
   should be at least slice-guard-sized and not purely linear. Prior art: the
   production fill `emit_band_fill_nl_pivots_src` (per-band-wire pivot + 1..7
   linear source CNOTs + nonlinear product terms) was built for exactly this
   and measured leakage-neutral; embedded masking never adopted it.

   RESOLVED 2026-10-05: the pre-reorganization production fill
   (emit_band_fill_nl_pivots_src + emit_g57_form, recovered from
   5f386381:src/preprocessing/gadgets.rs -- it was dropped in the
   September restructure, not redesigned away) is revived as
   seed_band_rich in embedded_masking.rs, wired through a new
   PreprocessingParams::EmbeddedMasking.seed_fill_nl knob
   (EMBEDDED_MASKING_SEED_NL env; default 4, 0 = legacy thin seed).
   Measured on the n=64 recipe: 1,445 band-seed + 1,426 band-reseed
   gates per side (vs 256 thin; slice guard = 1,280), forward+reverse
   verify PASS, stage-2 data-vs-band AUC flat (0.51 touches / 0.50
   target-frac) combined with persistent migration. Golden preset
   hashes re-pinned for the new default.

7. **Preprocessing: minimize 3-control gates — also hard to mix later.**
   (2026-10-01)

   The same TDP artifact carries 854 width-3 MPMCT gates, all from the two
   slice guards (`try_slice_zero_block_dims`,
   src/stages/preprocessing/slice_guards.rs — the CCNOT shape with two data
   controls plus a band slice control), 427 per guard. A width-3 gate's
   function needs a long g57 friend (or a multi-gate window) to dissolve in
   db_mixing. Re-express the guard blocks in width-<=2 gates — e.g. g57/CCNOT
   forms with one data + one band control — keeping the zero-slice deadness
   and junk-half targeting contract.

   RESOLVED 2026-10-06: accumulate-and-inject slice guard
   (try_slice_zero_feistel_dims, slice_guards.rs; default, with
   SLICE_GUARD_LEGACY=1 escape): prologue of width-<=2 slice shapes
   (balanced band coverage keeps single-hot sensitivity), band-targeted
   accumulators t ^= d_i&d_j sourced from the never-written payload
   half (uncompute always cancels), band-gated injections d ^= t&s
   with clean-trigger discipline. Same zero-slice contract (exhaustive
   small-dim check), same gate count. Whole-artifact control-arity
   census: {0,1,2} ONLY -- first TDP artifact fully inside the
   db_mixing vocabulary. End-to-end: digestibility UP (g57-form end
   fraction 82.5 -> 83.2%), exposure unchanged (7.1% of carried),
   forward+reverse verify PASS, preset hashes re-pinned.

**2026-10-06 — boundary-redraw exploration (RC's 4th option): ruled out, measured.**
Redrawing window boundaries post-hoc cannot hide the celebrities: their spans are too
long (median 494, p95 2,505 gates; 0/551 protectable @W=256, 452/551 still stuck
@W=4096), too busy (median 6 carrier reads per long span -> no pull-forward collapse),
and too dense (liveness ~110 at every cut position; 232k spans cover the line
~110-deep). Redraw + targeted gap-refresh works functionally but costs >=~35-60%
gates (62.5% measured under current bounds). Full numbers in
runs/ncdb_ab_20261001/RESULTS.md. Remaining viable paths are the three stage-2 drain
fixes; just-in-time drains now clearly the front-runner (attacks all 232k spans at
their ~551 sources for ~0 gates).

**2026-10-06 — #8 fix IMPLEMENTED: just-in-time drains (uncommitted).**
`preprocessing_shuffling.rs`: transfer masks now outlive their fire box and
drain at gate-granularity triggers (right before a read of the masked data
role, or a write of the holder band role — pure XOR algebra makes any trigger
placement correct); bare data roles whose next touch is > gap source gates
away are immediately re-masked with the longest-lived eligible band holder
(same-target reuse barred via holder ledger; cross-target holding allowed).
Knobs: EMBEDDED_MASKING_JIT=<gap> (requires SHUFFLING>0 + RETURN_HOME=0),
params field `shuffling_jit_gap` (default 0 = legacy eager drains, goldens
unchanged). Gen-level verify PASSED both scales; unit tests
jit_drains_preserve_function_and_conserve_the_mask_ledger (+ ledger law:
added = 3*transfers + 2*remasks + 3*ports). At n=64/gap=24: 22,656 transfers
(full dose), 116k remasks = +30% stage-2 gates. Full-chain A/B
(waist/jit64, waist/jit128 vs final64/final128) running — exposure,
celebrities, clusters, sizes.

**2026-10-06 — #8 JIT drains MEASURED: NEGATIVE, mechanism understood.**
Full-chain A/B n=64 gap=24 (waist/jit64 vs final64): celebrity class
UNCHANGED (330 values >=257, top 453 vs 473) — reads pin the bare-rebirth
multiplicity (~450 read clusters re-bear V regardless of drain timing);
meanwhile masked-transit vocabulary (+31% born) + packer saturation
(big-interval disq 5.8k->38.3k; stage 3 scatters drain/read adjacency)
regress delivered exposure 16.0% -> 43.8% of carried (s2-view 27.3 ->
46.4%); C-view stays ~floor (6.6%). Code kept (default OFF, goldens
stable) as an A/B arm. VERDICT: no drain-timing policy can fix #8.
Next: (1) emitter-side dressed carriers — keep the carrier's cross-box
mask as a live open_lin entry so reads consume the dressed value and the
fire algebra compensates; the bare V never recurs (the only multiplicity-
reducing mechanism); (2) scheduler v4 value-unit planning if short-span
modes return; (3) remask key diversity (hygiene only).

**2026-10-07 — #8 SOLVED IN STAGE 2 (pending full-chain confirmation):
threshold retirement.** Census chain: celebrities are 83% bracket restores;
RC's interleave+compensate prices at +132% (keep-3) and drop-all at +125%
(straddle census: avg ~10 live ledger entries per borrow). Winning fix:
EMBEDDED_MASKING_RETIRE=K — mini-slot retirement (early-close + fresh write,
ledger-native, zero compensation) of any band wire with band_uses >= K at
fire-box boundaries. n=64: top mult 473 -> 18-39, >=65 class 452 -> ZERO at
every K in {8,16,32}; K=32 is net -12% gates, K=16 +3.2%. Sub-5 probe (K=2,4)
+ full-chain exposure A/B pending; then flip default on with RC.

**2026-10-07 — linkable-exposure optimization (RC: values seen twice are thread ends).**
New metric waist/link_census.py (L2 linkable values / M2 sightings). Laws:
retirement and the ancilla pool only REDISTRIBUTE rebirths (few heavy ->
many light threads); window size is the strong cheap lever; migration dose 8
keeps ANF compact. Recommended config A = no retirement, dose 8,
EMBEDDED_MASKING_PREMASK=1, EMBEDDED_MASKING_SEED_AVOID_C1=1, pack G=512,
finalize3.sh (merge repair + exposure-aware ANF split): exact C-view floor at
both scales, L2 -22%/-18%, M2 -43%/-39%, froth -84%/-75% vs baseline at
+0%/+20% size (n=64/128). Alternative B (+K=64) has no celebrities but +31%
size and more threads. Open: celebrity rebirths (bracket restores) and
singleton excursions (packer should keep do/undo pairs inside one window).
All new knobs default OFF; nothing committed.

**2026-10-07 late — bounded chains + commutation-aware repair.**
EMBEDDED_MASKING_CHAIN=2 + CHAIN_LEN=8 (bounded chained brackets) vs A,
both finished with waist/repair.sh (identical-pair cancellation, leak
slides, merge/split fallback): linkable -13%, sightings -21/-26%, exposed
-24/-37%, anchors -39/-28%, +0.7k/+3.4k size-2 froth, size +18% (n=64) /
equal (n=128); exact C-view floor both. Full chains (no bound) and rotating
segments: dropped (ANF x2-2.5 / no gain). Cancellation alone = -7% gates.
Open: value-level excursion cancellation; chain + fringe exemption moot
(repair handles the fringe leak).

