# `fmix` grow-then-churn schedule

`fmix` has two mutually exclusive scheduling modes.

## Legacy one-phase mode

```text
fmix --input in.mpmct1 --target-size 4000000 --moves 24000000
```

This remains the default. If `--moves` is omitted, the budget is 1,000,000
attempts. With no nonlinear-frame weight and no schedule flags, the historical
single-call mixer trajectory is preserved for a fixed seed.

## Explicit grow-then-churn mode

```text
fmix --input in.mpmct1 --target-size 4000000 \
  --grow-moves 4000000 --churn-moves 20000000 \
  --churn-merge-reach 16384 \
  --churn-undo-frac 0.10 \
  --tabu-moves 100000 --churn-tabu-moves 100000 \
  --churn-w-cross 0.45 \
  --churn-w-fresh 0.20 \
  --churn-w-unsub 0.20 \
  --churn-w-insert 0.15
```

`--grow-moves` and `--churn-moves` must be supplied together and conflict with
`--moves`. The budgets are interpreted on one cumulative move clock. In the
example, growth runs over moves `[0, 4,000,000)`, and churn continues over
`[4,000,000, 24,000,000)`.

The phase transition does **not** construct a new mixer. It retains:

- the circuit arena and gate identities;
- the same seeded RNG stream;
- per-gate origin and split-event provenance;
- the crossing undo journal;
- currently retained tabu entries; and
- cumulative counters and reporting/verification cadence.

There is no float or serialization between phases. Growth ends with the
ordinary sampled functional check, emits an explicit boundary report, applies
the requested churn overrides, and continues. That boundary check consumes
deterministic samples from the same seeded RNG, so an explicit two-phase run is
reproducible but is intentionally not trajectory-identical to a one-phase run
with the same total number of moves. If a stop flag terminates growth, churn is
not started.

All `--churn-*` values are optional and inherit their growth-phase value. When
`--churn-target-size` changes and `--churn-temp` is absent, the standard
`max(target/100, 64)` temperature is recomputed. A changed tabu duration applies
to retained and future events; entries already evicted during growth cannot be
resurrected. Set the growth `--tabu-moves` to the desired long duration when
cross-boundary protection matters.

## Nonlinear frames

The nonlinear-frame controls exposed by the driver are:

```text
--w-nl-frame
--nl-frame-min-width       # control count; default 2
--nl-frame-max-width       # control count; default 3
--nl-frame-packet-gates    # gates per reversible packet; default 16
--nl-frame-shots           # transport attempts; default 64
--nl-frame-tenure          # protected move age; default 100000
```

Each has a churn-phase override. Width two is the first genuinely nonlinear
(Toffoli-class) setting. In every phase where the frame weight is nonzero, the
maximum is required not to exceed `--k-max`, and the circuit must have at least
`max_width + 1` wires. Dormant geometry is ignored, so legacy low-`k-max`
commands remain valid. The nonlinear move remains
disabled at `--w-nl-frame 0`, which is the default. Its weight is relative to
the other expansion weights; weights do not need to sum to one.

For an affine-heatmap experiment, enable nonlinear frames primarily during the
churn phase so the first phase can efficiently reach the size target and the
second phase spends its budget re-encoding existing material. Treat any
specific nonzero weight as an experimental parameter until coverage, build
cost, degree-1/degree-2 heatmaps, and SAT behavior have been measured together.
