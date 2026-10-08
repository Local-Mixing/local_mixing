# Golden re-pin check against ea2e516f (2026-10-08)

The four-stage cleanup re-pinned the serial mixer goldens in
`tests/unit/engine/mixer/serial_baselines.rs`, because the retired curated,
split and crossing fields left the state that `golden_hash` covers. To show
that the *trajectory* did not change, each golden also has a version-neutral
`trajectory_hash`: the gates and per-gate meta in circuit order, the move
clock, the kept stage-3 counters and the next draw of the main RNG. None of
these change layout in the cleanup.

| Constant | Test | Reproduced at ea2e516f |
| --- | --- | --- |
| `TRAJECTORY_PROFILE_NULL_PLANT = 0xbaa42bedc74dc44e` | `golden_profile_null_plant_is_stable` (empty store, profile schedule) | yes, with that test's ea2e516f parameters (incl. `temp 20`, `p_comp 1.0`, `p_any 0.1`) |
| `TRAJECTORY_STAGE3_FIXTURE = 0xec5a286a7b0a742f` | `golden_stage3_fixture_walk_is_stable` (DB splices on the g57 fixture store, db_advance, twist-g57) | yes, with the stage-3 flags of that build (incl. the inert `p_comp 1.0`, `p_any 0.1`) |

ea2e516f has no in-memory store, so the check needs a small backport:
`golden_repin_ea2e516f.patch` (next to this note) adds the test-only
`FrozenDb::from_memory` and the lookup-cache bypass exactly as the cleanup
branch has them, copies the g57 fixture store, `db_walk_params` and
`trajectory_hash` verbatim from the cleanup tree, and adds the two tests
`repin_check_profile_null_plant` and `repin_check_stage3_fixture`, which
assert the constants above. No mixer code is touched.

To re-run (any scratch directory; nothing is written to the repository):

```
P="$PWD/docs/notes/golden_repin_ea2e516f.patch"
D=$(mktemp -d)
git archive ea2e516f | tar -x -C "$D"
(cd "$D" && git apply "$P" &&
  CARGO_TARGET_DIR="$D/target" cargo test --locked --release --lib -- repin_check)
```

Run it from the repository root. Expected: `2 passed`. Run on 2026-10-08
against the cleanup tree at `ran/cleanup-four-stage`, both passed.

`golden_hash` constants (`GOLDEN_PROFILE_NULL_PLANT`, `GOLDEN_STAGE3_FIXTURE`)
include the full counters line, whose layout the cleanup changed, so they are
pinned on the cleanup branch only. The step-4 re-pin of
`GOLDEN_PROFILE_NULL_PLANT` (old `0xc277215de595bf62`) was additionally
checked by re-inserting the retired counter fields as zeros; that ad-hoc
check was not kept, and the trajectory check above supersedes it.
