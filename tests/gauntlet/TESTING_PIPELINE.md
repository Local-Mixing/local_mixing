# Testing pipeline

The gauntlet asks whether an observer of every wire state, gate flip, and new
target value in a gadgetized execution can recover or correlate with the
source circuit's intermediate values. It is a reproducible empirical battery,
not a proof of security.

## Data flow

```text
source r57 chain
    |
    |  gauntlet_build.py for nonlinear193/nonlinear291
    |  (native controls are gadgetized in Rust)
    v
gadget circuit + initial columns + build metadata
    |
    |  gauntlet_gen [optional local mixing]
    v
trace.bin + targets.bin + meta + traced circuit
    |
    |  gauntlet_audit
    v
audit.log + hits.jsonl
    |
    |  gauntlet_heatmap.py
    v
six witness PNGs + REPORT.md
```

A cell is `(source chain length, arm, auxiliary-input policy, mixing mode)`.
Source chains contain r57 gates over eight logical wires. Mixing applies the
repository's local-mixing walk after gadgetization and before tracing.

## Arms

| arm | implementation | purpose |
|---|---|---|
| `none` | native Rust, no gadgetization | positive leakage control |
| `secretshare14` | native Rust paired secret sharing | comparison control |
| `bandproduct92` | native Rust product-share band | comparison control |
| `nonlinear193` | `gadgetization.nonlinear193` | 193 emitted gates per source gate |
| `nonlinear291` | `gadgetization.nonlinear291` | fan-in-at-most-two decomposition, 291 emitted gates per source gate |
| `*_band0` | same Python gadget, production-style band-fed borrows | pool integration check |
| `*_band16` | band-fed borrows after 16 U0 blind layers | blinded-pool integration check |

The Python builder supports only `nonlinear193` and `nonlinear291`; there are
no behemoth/939/1415 arms in this port.

The band pool is keyed by the eight visible chain inputs plus 120 neighbor
input columns by default. Without the extra keys there are only 256 visible
contexts and an input-keyed pool degenerates into a small truth table. In band
mode, an unused column built by the same pool process serves as the NULL
correlation baseline.

## Recorded bundle

For prefix `bundle`, generation produces:

- `bundle.meta`: tab-separated trace dimensions, target names, seeds, mixing
  state, and behavioral-check status.
- `bundle.trace.bin`: little-endian, bit-packed columns consisting of initial
  gadget wires followed by each gate's flip and new target value.
- `bundle.targets.bin`: five source values per source gate (`a`, `b`, `cold`,
  `f`, `cnew`) plus a NULL column.
- `bundle.g.mpmct1`: the actual traced circuit.
- For file-mode arms, `bundle.mpmct1`, `bundle.init.bin`, and
  `bundle.buildmeta` are the Python builder's inputs to the Rust tracer.

The orchestrator treats `behavioral_ok=false` as a generation failure. The
builder also checks each gadget locally, the complete decoded source state,
the exact gate count, restored nonlinear291 ancillas, and strict borrow-wire
isolation before writing a successful bundle.

## Attacks and coverage

| name | check | coverage |
|---|---|---|
| `a1` | direct equality or complement with a trace feature | all targets and features |
| `xrows` | affine recovery from any single prefix wire state | every prefix, with held-out verification |
| `xtrace` | affine recovery from initial wires plus all gate flips | full selected trace when under the configured feature limit; otherwise explicitly skipped |
| `w1` | covariance with one feature | every recorded feature |
| `w2` | covariance under XOR, AND, OR, and both directions of AND-NOT | deterministic strided feature subset, default cap 64 |
| `w3` | covariance under five three-input operation templates | deterministic strided feature subset, default cap 16 |

The `w1`/`w2`/`w3` correlation battery is bounded. In particular, `w2` and
`w3` do not enumerate all trace features when the trace exceeds their caps,
and the listed operation families are not all Boolean functions. Reports must
therefore describe these as capped scans, never as exhaustive correlation
testing. Correlations are NULL-referenced and must exceed both the NULL
maximum and the configured statistical threshold to be flagged.

Pair work grows as `C(w2_cap, 2)` and triple work as `C(w3_cap, 3)` times the
operator, target, and correlation-sample dimensions. Raising either cap can
therefore make a matrix run substantially more expensive.

Exact-affine fitting uses an overdetermined fit region followed by 2,048
held-out samples. Correlation samples occupy a separate tail. Defaults are
16,384 samples for `k=1`, 8,192 for `k=2`, and 4,096 for longer controls;
the nonlinear arms use 16,384 at `k>=16`. `--corr-samples` overrides this and
must be a multiple of 64.

`xtrace` has a configurable size guard (`--xtrace-max-features`, default
40,000). A skipped global test is not a zero-hit result: the auditor records a
separate status and the report preserves it.

## Resume and concurrency

Every cell has its own directory and its own `.probe` prefix for mixed
file-mode size discovery. The probe uses `gauntlet_gen --size-only`, so it
does not read sample columns or create a deliberately failed trace bundle. No
shared `/tmp/mxprobe` path is used, so `--jobs` does not race between cells.

`cell-config.json` stores independent generation, audit, and map stage
records. Generation provenance covers the arm settings, seeds, source-chain
recipe, generator binary/source, builder, and gadget modules. Audit records
include caps and auditor provenance; map records include the renderer source.
Artifacts are reused only when the stage configuration digest and each
artifact's recorded size and SHA-256 digest match.

The output ownership marker records the exact resolved directory. Recursive
cleanup refuses repository source paths even with `--force`; unmarked forced
resets are permitted only below `target/` or the system temporary directory.

The default layout is:

```text
target/gauntlet/
  .gauntlet-output.json
  _inputs/chain_k1.mpmct1
  k1/nonlinear193_nomix/
    cell-config.json
    bundle.*
    build.log
    gen.log
    audit.log
    heatmaps/{a1,xrows,xtrace,w1,w2,w3}.png
  index.json
  REPORT.md
```
