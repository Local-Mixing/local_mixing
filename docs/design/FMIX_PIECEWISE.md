# FMIX_PIECEWISE — piecewise-parallel rounds for stages 3 and 4

Implemented in `src/engine/mix/piecewise.rs`, driven by
`fmix --parallel-pieces P` or `fmix --parallel-target-piece-gates B`.
Normal `gss` uses the `[parallel]` section of `configs/gss.toml`.
Explicit `pieces = 1` or omitting both sizing settings uses the serial path
(golden tests in `mix_tests::golden_*` pin three serial trajectories).

Choose one sizing option in the recipe:

```toml
[parallel]
# Fixed count: P = 4 throughout stages 3 and 4.
pieces = 4
# threads = 5  # optional; defaults to P+1
```

```toml
[parallel]
# Automatic count: P = max(1, floor(current gate count / 10000)).
target_piece_gates = 10000
# threads = 8  # optional; defaults to available CPU parallelism
```

The automatic count is recalculated from the current circuit at the start of
each round, separately in each stage. The size is **only a divisor to obtain
P**, not a hard minimum: subsequent mixing and shifted end blocks work just
as in fixed-P mode, and the end blocks can be smaller than the supplied size.
For example, 100,000 gates and size 10,000 give P=10: nominally 10 pieces on
an ordinary round and 11 on a shifted round. Integer division rounds down;
a circuit smaller than twice the divisor uses one block. Automatic one-block
rounds still use the round driver, allowing P to rise if the circuit grows.
Empty circuits stop with an explicit `CircuitEmpty` reason; no move budget
is falsely reported as spent. Splitting already stops on g57 exhaustion.
Setting both sizing options, including explicit `pieces = 1`, is an error.

## 1. The idea

The fragmentation stages spend their time on local moves — a DB window of at
most a few dozen gates, a split of one g57 and a bracket drawn on one wire, a
cross shot that floats one gate to its first collider. Two such moves at
distant positions commute. So:

1. cut the current circuit `C` (m gates, circuit order) into `p` contiguous
   pieces `C_1 … C_p` of about `L = m/p` gates;
2. run the Mixer on every piece **in parallel**; each run is function-
   preserving on its piece, so the concatenation `C_1' … C_p'` computes what
   `C` computed;
3. concatenate ("merge") in order;
4. **shift the cuts by half a slice** and repeat: the next round cuts at the
   midpoints between the previous round's actual seams, so every old boundary
   lands in the middle of a piece and the material a window could not reach
   across a cut is mixed in the following round.

Because the pieces grow unevenly (db_mixing grows toward `R1·s_in`, the split
stage grows by `1 + comp-fraction`), the shift is tracked exactly: round `r`
cuts at the midpoints of `[0] ++ seams_{r-1} ++ [m]`. Odd rounds therefore have
`p` cuts and `p + 1` pieces (two half-slice end pieces); even rounds have
`p - 1` cuts and `p` pieces. A jitter of `±12.5 %` of the local interval
(`--piece-jitter`, `< 0.25`) decorrelates the two lattices over many rounds
while keeping every old seam at least `37.5 %` of its interval away from every
new cut (`cut_points`, unit-tested under simulated uneven growth).

In automatic mode, this exact seam tracking applies while P stays unchanged.
When growth or contraction changes P, the next round rebalances the cuts for
that new count and preserves the ordinary/shifted phase. The previous seam
spacing guarantee does not apply to that rebalance round.

## 2. Who owns what

A **whole-circuit mixer W** (the ordinary `Mixer` fmix builds from the input)
holds the truth for the run: the stage input as `original`, the counters, the
move clock `moves_done`, the id counters `next_event`/`next_litter`, the
profile controller `ProfState`, the split tri-state, and the wire canaries.
Every round:

- `W.export_slices(&cuts)` hands each piece its gates **with their `Meta`
  verbatim** (origin, event, dir, dgen, litter, litter_size), the canaries
  anchored inside the slice, the undo-journal entries lying wholly inside it,
  and W's tabu ring (by position; `PieceParts`).
- `Mixer::from_parts` builds a piece mixer over that slice (the
  `resume_state` install sequence in memory): `original` = its own slice, so
  the piece's `global_check` verifies the piece; `moves_done` starts at W's
  clock; event and litter ids are minted from the disjoint band
  `W.next_* + (i << 32)`; the piece is `quiet` and never plants or reports
  canaries.
- The pieces run on a dedicated rayon pool (`--parallel-threads`, default
  `p + 1` for fixed counts, available CPU parallelism for automatic counts),
  all sharing W's `Arc<FrozenDb>` (the store is read-only: pread +
  thread-local buffers, a process-wide lookup cache with byte-identical
  results; opening it per piece would cost ~1.3 GB of offset tables and, with
  filters, ~25 GB per copy).
- `W.rebuild_from_parts(outs)` concatenates in index order and folds:
  counters summed field-wise (`impl AddAssign<&MixCounters>` — a full
  destructuring, so a new counter is a compile error until classified),
  `moves_done += Σ piece moves`, `next_* = max`, tabu merged with ages
  translated onto the new clock (bands keep it sorted by event, which
  `is_tabu`'s binary search needs), journal entries re-positioned and
  truncated to `journal_len`, canaries re-anchored by position. The
  whole-circuit state (original, RNGs, `prof`, split flags, flag paths,
  `db_record`, snapshot base) is carried over.
- `W.global_check()` verifies the merged circuit against the **true stage
  input** after every round; fmix's post-run tail (report, `final_float`,
  `save_state`, `write_mpmct`, sidecars) runs on W unchanged, so `db_mixing.state`
  and `split.state` are ordinary v2 states and stage 5 needs nothing new.

## 3. Steering the pieces

Pieces run with the controller off and are steered from outside per round.

**Profile policy (stage 3, `--profile`).** W's controller is initialised once
(`prof_init`, eff 0). A round's length is `d_eff = next_eff − eff` — the
controller's own cadence (0.125 eff for the first update, then
`prof_cadence_eff` = 0.5), or `--piece-round-eff` when given. Each piece gets
`p_mix = W.prof.pmix` (the `apply_mode_overlay` coin — the same MIX/COMP coin
`prof_tick` draws), `target_size = round(W.target_size · len_i/S)`,
`temp = W.temp · len_i/S` (floor 16), `size_hi = size_lo = 0` (brake inert),
and `eff_budget = d_eff`: the piece stops (`MixStop::RoundDone`) once its own
`Σ 1/size` per move reaches `d_eff`, exactly `prof_tick`'s clock. A moves
ceiling (`3·d_eff·len_i`, capped by the piece's share of the remaining
`--moves`) guards the A_MOVES contract. After the round W advances
`eff += Σ (len_i/S)·eff_local_i` (= Σ moves_i / S, the serial quantity) and,
when `eff ≥ next_eff`, runs the **unchanged** `prof_update`, which reads only
`arena.len()`, the counter deltas since its snapshot and `moves_done` — all of
which W holds after the fold. Phase 4 ends the run with `ProfileDone`, exactly
as in `run()`. With the default round length the controller therefore updates
at the serial cadence with the serial plant estimates.

**Split policy (stage 4, `--split --split-stop`).** `W.plant_taps()` once
(global `orig_permille`). Each piece runs `split = true, split_stop = true,
split_canaries = 0` with the imported canaries, a moves ceiling of
`comp_i + len_i`, and the global frame for statistics (`span_norm = S`,
`rank_base`, `rank_total`, so `split spans` stays frac-of-circuit and `xmid`
uses the whole circuit's midpoint; the direction coin stays piece-relative).
Rounds repeat until `W.remaining_g57() == 0` (g57 count is monotone: no move
mints a comp gate), or `--split-rounds-max` rounds (default 6, reason
"round cap"), or a round with no split at all ("failure limit"). W then ends
the stage as `run()` does under `split_stop`: tri-state to ended,
`announce_split_end` prints the ENDED / spans / deciles triple once from the
folded counters, one boundary move, `global_check`, `report`, `SplitDone`.

**Moves policy (no profile, no split).** Fixed rounds of `--piece-round-eff`
(default 0.5) with proportional setpoints until `--moves`. Used by tests; the
door to a piecewise stage 5.

## 4. Determinism

Output is a function of `(input, flags, --seed)` only. Piece seeds are
`seed_of(seed, round, i)` (splitmix64), the cut jitter comes from a driver RNG
seeded from the run seed, no RNG is shared between pieces, and the pieces are
re-assembled by index. `mix_tests::piecewise_result_is_independent_of_thread_count`
runs sequential, 2-thread and 5-thread pools and compares circuits, Meta,
counters and clocks.

## 5. Flags, guards, logs

`fmix (--parallel-pieces P | --parallel-target-piece-gates B) [--piece-jitter J]
[--piece-round-eff E] [--parallel-threads T] [--split-rounds-max K]
[--piece-verbose]`. `--piece-min-len N` is a fixed-count-only startup guard;
it conflicts with `--parallel-target-piece-gates`. With fixed `P > 1` or automatic sizing,
fmix refuses `--resume` (stage 5 stays serial), ancestry
(`--ancestors/--anc-samples/--anc-in/--anc-out`), `--db-record`, mid-run
snapshots, `--split` without `--split-stop`, jitter outside `[0, 0.25)`, and an
fixed-count input whose pieces would fall below `--piece-min-len` (default
1024) — a hard error, never a silent change of a manually chosen `P`.
Automatic mode uses the supplied divisor without imposing that extra floor.
Both `parallel.pieces` and `parallel.target_piece_gates` are locked recipe values.

Pieces print nothing (`--piece-verbose` re-enables their lines). W prints one
`[fmix] pieces: round r pieces=k len=[..] size a -> b moves+=.. eff+=.. stops=..`
line and one ordinary `[fmix] mv=` report per round, and the stage summaries
once. Stop/dump flags are honoured between rounds (dump = whole-circuit
snapshot); stop latency is one round.

`scripts/gss_mix.sh (--parallel-pieces P | --parallel-target-piece-gates B) [--parallel-threads T]`
passes piece flags only to stages 3 and 4. `local_mixing_bin gss` reads
mutually exclusive `parallel.pieces` and `parallel.target_piece_gates` from
TOML, plus `parallel.threads` (a lifecycle setting). New manifests use
`gss_command_recipe=6`. Their internal `pieces=auto` / `min_block_size` records
retain the earlier format; changing the divisor or switching modes changes
the recipe. Old config names and CLI spellings remain compatibility aliases.

## 6. Tests

- `mix_tests::golden_*` — three serial trajectories pinned before the feature.
- `piecewise::tests` — cut-point properties under uneven growth, seed
  distinctness, counter fold, automatic count growth/contraction, transitions
  to/from one block, and equivalence to fixed-P cuts while P is unchanged.
- `mix_tests::piecewise_auto_*` — automatic/fixed trajectory equivalence at
  a stable count and automatic one-block profile/split completion.
- fmix CLI and GSS config tests — mode validation, conflicts, forwarding,
  and locked manifest values; `tests/gss_block_size.bash` checks stages 3–4
  forwarding and rejection before run-directory side effects.
- `mix_tests::piece_transport_round_trip_is_an_identity` — slice + rebuild
  with zero moves is an identity on gates, Meta, canaries, live tabu, index.
- `mix_tests::piecewise_profile_null_plant_reaches_phase_4`,
  `piecewise_split_stage_exhausts_g57s_and_resumes` (merged state resumes
  into a thermostat walk), `piecewise_result_is_independent_of_thread_count`.
- `tests/fmix_db_move.rs` section (3): three pieces on a 4-thread pool share
  one synthetic frozen store; the compressing move fires inside pieces and the
  concatenation is exhaustively equivalent to the input.
- Plumbing: `GSS_MIX_ALLOW_EMPTY_STORE=1 bash scripts/gss_mix.sh --source-wires 8 --run-directory
  /tmp/pw --calibration-seed 1 --parallel-pieces 4 --run-stop-after-stage 5` (dev box; use `--piece-min-len`
  through a direct fmix call for small fixtures).

## 7. Known differences from the serial walk

- No window, twist or float crosses a cut within a round; the shifted next
  round covers the seam. Measure seam under-mixing with the per-original-
  position displacement histogram before adopting a `P` in production.
- Stage 4: NOT-twist brackets are drawn inside a piece, so spans are capped at
  the piece length, and a g57 exhausted inside a piece cannot later be split
  across the whole circuit. The reported statistics use the global frame.
- db_mixing twist windows (rare) are log-uniform up to the piece length.
- Undo-journal entries straddling a cut are dropped at the cut; tabu ages are
  translated exactly.
