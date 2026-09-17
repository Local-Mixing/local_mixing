# Security analysis tools

This directory contains reusable circuit experiments: ordinary and affine
heatmaps, the gadget gauntlet, a SAT preimage workflow, and two demixing probes.
Each experiment takes explicit inputs and writes its results to a directory you
choose. Run the commands below from the repository root. Keep generated results
under `target/` or outside the checkout.

| Directory | Contents |
|---|---|
| [heatmaps/](heatmaps/) | Prefix distance, affine reconstruction, statistical prediction, cumulative trace analysis, and their renderers. |
| [gauntlet/](gauntlet/) | Generate gadget comparisons, collect traces, run six attack families, and render witnesses. |
| [sat_solve/](sat_solve/) | Encode a zero-slice preimage problem as DIMACS, run an external solver, and independently verify its model. |
| [demixing/](demixing/) | Output-cone pruning and inverse crossing reductions using only the supplied circuit. |
| [preprocessing/](preprocessing/) | Standalone embedded-masking generation for controlled experiments. |
| [gadgetization/](gadgetization/) | Python nonlinear gadget references and template exporters shared with the gauntlet and correctness tests. |
| [python/](python/) | Rust implementation of the optional `local_mixing` Python heatmap extension. |
| [fixtures/](fixtures/) | Small circuit generators, evaluators, and equivalence-preserving comparison builders. |

The plotters live beside their heatmap generators. Each workflow below provides
commands for generating, inspecting, and rendering its results.

## Build and Python setup

The ordinary Rust build does not enable analysis executables. Build the tools
you want explicitly:

```bash
cargo build --release --features security-tools \
  --bin hmap --bin hmap_affine --bin hmap_stat --bin hmap_trace_affine \
  --bin gauntlet_gen --bin gauntlet_audit \
  --bin output_cone --bin crossing_downhill
```

Use Python 3.10 or later. NumPy and Matplotlib are needed for plotting and the
gauntlet; Pandas is used by the compression histogram scripts. Installing this
repository also builds its optional Rust heatmap extension and installs those
dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
mkdir -p target/security-demo
```

The C++ SAT encoder requires a C++17 compiler. SAT solving additionally requires
an external Kissat executable; it is not bundled. The encoder, decoder, and
their tests work without Kissat.

## Ordinary prefix-distance heatmaps

`hmap` compares two circuits with corresponding wire meanings. Each matrix cell
is the number of differing wire bits between a prefix of the reference circuit
and a prefix of the comparison circuit, averaged over sampled inputs. Values
range from zero to the circuit width. Small values mean their states
are close on the selected input distribution. Equivalence at the endpoints is
a separate property; a heatmap does not establish it.

```bash
target/release/hmap \
  --c reference.mpmct1 --c-format mpmct1 \
  --d mixed.mpmct1 --d-format mpmct1 \
  --c-step 10 --d-step 100 --batches 8 --seed 1 \
  --out target/security-demo/distance
python security_tests/heatmaps/hmap_pixels.py target/security-demo/distance \
  --out target/security-demo/distance.png --vmin 0
```

Every Rust heatmap writes `<out>.bin`, a row-major float32 matrix, and
`<out>.meta.json`, containing its dimensions, sampled prefix positions, and
metric metadata. Strides set the grid resolution; `--batches` sets 64 input
samples per batch. Useful `hmap` variations are:

- `--d-start`, `--d-end`, and `--d-from-end`: inspect a limited part of the mixed circuit.
- `--input-dist uniform`: uniformly random wire bits.
- `--input-dist bern:0.1`: sparse independent input bits.
- `--input-dist fix:0:8`: hold wires `[0,8)` at one randomly chosen constant while varying the rest.
- `--seed-pos K`: draw the chosen distribution at reference prefix `K`, invert that prefix, and use the resulting inputs for both circuits.

The extension-backed [heatmap.py](heatmaps/heatmap.py) directly generates PNGs
from G57 circuits and normalizes distances by the selected wire count. It adds
corner windows, chunked grids, incremental inputs,
Hamming-weight differences, half-wire projections, and alternate color scales:

```bash
python security_tests/heatmaps/heatmap.py \
  --n 64 --i 256 --c1 reference.g57 --c2 mixed.g57 \
  --x reference --y mixed --corner --canonless \
  --path target/security-demo/python-distance.png
```

Run `python security_tests/heatmaps/heatmap.py --help` for the full variation
list. The G57-only `mean_evo.py` plots chunk averages; `gate_plot.py` draws the G57 gates.
`wireplot.py` plots wire-use counts using the full G57 alphabet and `~` overflow
prefixes, supporting up to **65,536 wires** (IDs 0–65,535). Set `--n` to the
circuit's wire count; malformed encodings and wires outside that range are
rejected. Use `--out` to choose the image path:

```bash
python security_tests/heatmaps/wireplot.py \
  --c circuit.g57 --n 4096 --x circuit --out target/security-demo/wires.png
```

`compression_heatmap.py` and `compression_hist.py` consume compression CSVs.

## Affine and statistical heatmaps

`hmap_affine` compares an original logical circuit with a wider implementation.
It fits each reference-state bit as a GF(2) affine function of **all wires at one
comparison prefix**, then measures errors on held-out samples. Logical inputs
occupy wires `[0,n)` and other input wires are zero. This initialization must
match the implementation being studied.

```bash
target/release/hmap_affine \
  --c reference.mpmct1 --c-format mpmct1 \
  --g gadget.mpmct1 --g-format mpmct1 --n 8 \
  --c-step 1 --g-step 100 --batches 96 --train-batches 72 \
  --out target/security-demo/affine
python security_tests/heatmaps/plot_hmap_ridge.py \
  target/security-demo/affine --out target/security-demo/affine.png
```

Here zero means affine recoverability and approximately `0.5` means the tested
affine predictor does not recover the target bit. Endpoints can be trivially
recoverable; examine the interior and the ridge of recoverability rather than
only the whole-matrix mean.

`--degree 2` additionally offers pairwise wire products. Restrict their wires
with `--deg2-wires` or `--deg2-wire-list`; the feature count grows quadratically,
and training samples should substantially exceed the feature count. Use
`--c-from/--c-to`, `--g-from/--g-to`, and `--dump-best` to localize and inspect a
candidate leak.

`hmap_stat` searches single-wire and two-wire-XOR predictors and reports their
best agreement, with a random-target noise floor. Its scale runs in the other
direction: **higher agreement means stronger prediction**. It also supports
one additional product term with `--and-wires`.

```bash
target/release/hmap_stat \
  --c reference.mpmct1 --c-format mpmct1 --g gadget.mpmct1 --n 8 \
  --c-step 2 --g-step 500 --samples 4096 --target-bits 8 \
  --out target/security-demo/statistical
python security_tests/heatmaps/stat_readout.py --trim 0.1 \
  target/security-demo/statistical
```

Use `stat_readout.py` for these agreement plates. Affine ridge scores assume an
error metric and should not be applied to agreement values.

## Cumulative trace variation

`hmap_trace_affine` accumulates features across the comparison circuit instead
of using one isolated state. `checkpoint-state` contributes every wire at the
selected checkpoints. `gate-delta` contributes the initial wires and gate
firing bits; with stride one, these span every intermediate wire value.

```bash
target/release/hmap_trace_affine \
  --c reference.mpmct1 --c-format mpmct1 --g gadget.mpmct1 --n 8 \
  --trace-mode gate-delta --delta-stride 8 --g-checkpoints 12 \
  --fit-batches 112 --validation-batches 16 --test-batches 32 \
  --out target/security-demo/trace
python security_tests/heatmaps/plot_hmap_trace.py target/security-demo/trace \
  --out target/security-demo/trace.png
```

Fit, validation, and final test samples are separate. A stride greater than one
is a trace sketch, so unselected deltas remain outside the tested feature
family. `--include-checkpoint-states` adds whole-state anchors. Inspect the
feature/sample budget before opting into `--allow-underdetermined`; a clean
result is always relative to the stated feature family and sampling policy.

## Gadget gauntlet

The gauntlet creates its own small source chains, generates gadget arms,
optionally mixes them, audits traces, and writes witness heatmaps and a report:

```bash
python security_tests/gauntlet/gauntlet.py all \
  --ks 1,2 --mix both --jobs 2 \
  --arms none,embedded_masking_balanced,embedded_masking_balanced_wideband,nonlinear291 \
  --outdir target/security-demo/gauntlet
```

The arms include an unprotected positive control, balanced embedded masking,
balanced embedded masking with 256 band wires, and nonlinear291. Keep `none`
in comparative runs: detectors should succeed on the positive control.

Two additional arms exercise internal wire shuffling: `embedded_masking_shuffled`
returns data roles to their physical ports at the end of each fire box, while
`embedded_masking_shuffled_carried` carries the layout forward and decodes through
its final permutation. Both use balanced masks and eight target-write segments
per box by default. `--shuffling-segments N` changes that count for both arms
(minimum eight); transfers can be skipped when no eligible band role exists.
The production generator keeps shuffling off unless explicitly configured, and
only supports the return-home variant.

Compare the baseline and both variants with paired mixer seeds:

```bash
python security_tests/gauntlet/gauntlet.py all \
  --ks 1,2 --mix both --mix-seeds 777,778 --jobs 2 \
  --arms none,embedded_masking_balanced,embedded_masking_shuffled,embedded_masking_shuffled_carried \
  --outdir target/security-demo/shuffling
```

`--mix-seeds` accepts a comma-separated list of unsigned 64-bit integers; its
default remains `777`. Each mixed cell has its seed in the directory name,
manifest, bundle metadata, and report index. The same source chain, construction
seed, and input-sampling seed are used for a given chain length across mixer
seeds. Unmixed controls run once, and report rows keep both shuffling variants
and every mixer seed separate. Use the same options when resuming `gen`, `audit`,
`maps`, or `report`; duplicate axes are rejected to prevent concurrent jobs from
writing the same cell. Mixed results use directory suffixes such as
`_mix_seed777` to identify the mixer seed.

The six attack families are direct wire matches (`a1`), affine reconstruction
from all wires at one prefix (`xrows`), affine reconstruction from initial
wires and the complete gate-flip trace (`xtrace`), and correlations using one,
two, or three selected features (`w1`, `w2`, `w3`). A feature-cap skip is reported
as a skip rather than a successful defense.

Adjust `--ks` for source-chain lengths, `--n-wires` for logical width (minimum six),
`--mix-moves` for walk length,
`--corr-samples` for correlation samples, `--w2-cap/--w3-cap` for the selected
pair/triple feature sets, and `--xtrace-max-features` for the full-trace guard.
`--witnesses` limits saved witnesses; `--jobs` changes concurrent jobs.
`gen`, `audit`, `maps`, and `report` rerun individual phases. `--force` allows
replacement of stale artifacts; use a new output directory to retain a run.

The gauntlet's embedded-masking arm encodes and decodes logical I/O off trace
and starts the band randomly. It is a controlled gadget experiment with an
explicit observation model; the full TDP generator has its own slice contract.
Bundle metadata records the final role-to-wire permutation, return-home policy,
and shuffling gate overhead, transfer count, and skipped cuts. These are
construction diagnostics, not a security score. Comparisons across mixer seeds
measure variation in the walk for fixed constructions, not variation across
independently generated gadgets. A smoke run establishes workflow correctness;
it does not establish resistance to the attack families above.

## SAT preimage solving

The [SAT workflow guide](sat_solve/README.md) includes a complete example using
the small checked-in fixture, the adjustable input/output layout, solver exit
codes, independent witness verification, and a random-input-cube variation.
Use it with an `mpmct1` tape whose port convention you know. If the final
pipeline output is packed ESOP, supply the equivalent expanded `mpmct1` tape.

## Demixing probes

These tools use the circuit alone, without its generator seed or provenance.

```bash
target/release/crossing_downhill --input mixed.mpmct1 --top 20
target/release/crossing_downhill --input mixed.mpmct1 --passes 3 \
  --output target/security-demo/reduced.mpmct1 --verify-rounds 16

target/release/output_cone --input mixed.mpmct1 --live-start 8 \
  --max-moves 100000 --output target/security-demo/cone.mpmct1
```

`crossing_downhill` searches inverse crossing substitutions that shrink
same-target ESOP groups. Its scan-only mode reports opportunities; `--passes`
applies them. Rewrites preserve the full circuit function and applied runs
perform sampled all-wire checks.

`output_cone` moves gates affecting discarded outputs toward the end and
prunes them using backward liveness. `--live-start 8` preserves output wires
`[8,width)` and allows outputs `[0,8)` to change. Select that range according to
the observation model; its output is a projection-preserving circuit.

## Fixtures and checks

`gen_random_mpmct OUT N GATES SEED` creates a reproducible random circuit.
`shuffle_mpmct INPUT OUTPUT SEED` changes commuting gate order.
`eval_point CIRCUIT mpmct1 HEX_INPUT` prints a concrete output value.
`id_rewrite --help` describes a frozen-database-only equivalence-preserving
diversifier for G57 source pairs. `check_output` is a specialized zero/random
auxiliary-input comparison for the four-block sandwich layout; its source
header gives the positional interface.

The `gadgetization/` reference code and template assets support the current
nonlinear291 implementation; nonlinear193 remains a dependency of that
reference implementation. Check templates and the retained workflows with:

```bash
python -m security_tests.gadgetization.export_templates --check
python -m unittest discover -s tests/python
python -m unittest security_tests.sat_solve.test_workflow -v
python security_tests/heatmaps/plot_hmap_trace.py --self-test
```
