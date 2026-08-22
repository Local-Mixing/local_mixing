# ssg compression speedup — what was changed and why

*(branch `ssg-gen-mix-clean`, commits `e4d7e33a` and `ddc5f584`, 2026-07-03/04.
Both changes are env-gated and OFF by default — with the env vars unset, behavior
is byte-identical to the previous code.)*

## The problem, as measured

Large runs (512-wire gadgetized circuits, 0.5–3M gates) were spending nearly all
their wall time in the compression phase, and the machine was idle while they did:

- Instantaneous sampling (`top`, 2s deltas) showed the ssg process at **exactly
  100% CPU — one core — with 60+ cores idle**, zero iowait, no memory pressure.
- Two runs sat **12+ hours frozen inside a single compression sweep** with no
  log output (one at 1.94M gates, one at 767k). They were not hung — they were
  compute-bound on one thread (and ignored SIGTERM; only SIGKILL took them down).
- Lifetime CPU averages told the same story: ~400–600% for a process that peaks
  at 2,000–3,000% early in each sweep.

Two independent causes compound:

### 1. The stop rule never lets go ("compress to the bone")

`compress_loop` stops only when **fewer than 50 gates** were shaved over the last
`stable_max` (6) sweeps. The threshold is absolute, so on a 500k-gate circuit it
means grinding at 0.01% yield per window — hours of sweeps that reduce nothing,
because the loop is chasing an incompressibility floor it can't reach. Observed:
a stage compressing 986k → target 394k spent 5+ hours stuck around 460–509k,
shaving tens of gates per sweep.

### 2. Straggler chunks serialize each sweep

Each sweep splits the circuit into `k = min(ceil(n/1500), 4×threads)` random
contiguous chunks and runs `compress_big_ancillas` (100 trials) on each via
rayon. **The sweep only finishes when its slowest chunk finishes.** Per-chunk
cost is heavily skewed: a chunk that lands on a hard region (wide windows, slow
`compress_lmdb` lookups — the ones `[compress-trace]` was built to spot) runs
its 100 trials at minutes each. So a sweep starts at 100+ cores and collapses to
1–2 threads for most of its wall time. The 12-hour freezes were exactly this:
one chunk × ~100 slow trials.

## The changes

Both in `src/db_mixing/replace.rs`, both opt-in via environment variable, both
announce themselves in the log so a run's recipe is auditable afterwards.

### A. Relative stall rule — `COMPRESS_STALL_FRAC` / `COMPRESS_STALL_WINDOW` (commit `e4d7e33a`)

In `compress_loop`, the stop check becomes: **stop when the total reduction over
the last `COMPRESS_STALL_WINDOW` sweeps (default 2) is below
`COMPRESS_STALL_FRAC × current_size`** (floored at the legacy 50 gates). Unset ⇒
the legacy `<50 gates / 6 sweeps` rule, unchanged. The early-stop log line now
prints the threshold it fired against.

Rationale: the marginal value of the last few percent of compression is low, and
its marginal cost is hours. A relative threshold scales with the circuit and can
fire after just two sweeps instead of seven.

Log banner: `[compress] stall rule: stop when < 5.0% of current size reduced over last 2 sweeps`

### B. Per-chunk wall-clock budget — `COMPRESS_CHUNK_BUDGET_MS` (commit `ddc5f584`)

In `compress_big_ancillas`, the 100-trial loop checks elapsed wall time before
starting each trial; past the budget it breaks, keeping the trials already done
(the trailing dedup still runs, the chunk returns partially compressed). Nothing
is lost: the next sweep re-randomizes chunk boundaries, so hard regions get
revisited from different angles instead of being ground down in one sitting.

This bounds a chunk at `budget + one trial`. Note the check is **between**
trials — a single pathological `compress_lmdb` call is not interrupted. If a
single trial ever proves capable of running for minutes on its own, the next
step would be threading a deadline into `compress_lmdb`'s inner 10-trial loop.

Log banner: `[compress] chunk budget: 60000 ms per chunk per sweep`

Both env reads are cached (`OnceLock`), read once per process.

## Production values and measured effect

Values in use: `COMPRESS_STALL_FRAC=0.05`, `COMPRESS_STALL_WINDOW=2`,
`COMPRESS_CHUNK_BUDGET_MS=60000`.

- **Stall rule:** first production stage compressed 205,220 → 104,245 and
  stopped the moment two sweeps yielded 4,967 < 5,212 gates (5%), instead of
  grinding toward the 82,088 target. Stages consistently settle at ratio ~0.45
  of peak (vs the 0.40 target) — i.e. the rule trades the last ~5 points of
  compression for stage times of minutes instead of hours.
- **Chunk budget:** a full stage — shoot 616k → 1.85M, compress 1.85M → 760,832
  (ratio 0.41, same compression quality as legacy) — completed in **~45
  minutes**, where the identical workload on the old code had been frozen for
  **12+ hours without finishing**. On a 2.9M-gate circuit, sweeps now complete
  every ~20–25 minutes and progress monotonically.

## Trade-offs to be aware of

- With the stall rule, each stage keeps a larger baseline, so under the
  `--grow-threshold` cadence absolute sizes grow faster per stage; `--min-gen`
  still governs termination, so finals come out bigger (and sooner).
- With the chunk budget, a budget-hit chunk contributes less reduction to that
  sweep; empirically the per-stage compression ratio was unaffected (0.41).

## Related correctness fix (separate, commit `9ab7a4ed`)

While debugging these runs we also found `probably_equal` chose its arithmetic
width from `num_wires` (the input/compare contract) rather than the circuits'
actual max wire index. Checking a 512-wire gadgetized circuit against its
256-wire source therefore ran u256 arithmetic on wires ≥ 256 — and
`primitive_types` shifts ≥ width silently return 0 — so every aux-wire access
was garbage and correct circuits were reported non-equivalent (three production
runs were discarded by this false alarm before it was found). Fix: evaluation
width = max wire index either circuit touches; the `num_wires` mask/contract is
unchanged. Any check where the circuit is wider than the compare contract was
affected; ≤256-wire runs never were.
