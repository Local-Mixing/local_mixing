# Documentation

Start with the current operator and architecture guides:

- [GSS usage](GSS_MIX.md) and [all configuration fields and flags](GSS_FLAGS.md).
- [Complete code walkthrough](CODE_WALKTHROUGH.md): architecture, important functions, algorithms, optimization techniques and correctness contracts.
- [Code layout](CODE_LAYOUT.md), [refactoring record](REORGANIZATION_PLAN.md), and [cleanup audit](CLEANUP_AUDIT.md).
- [Database control order](DB_CONTROL_ORDER.md) and [runtime leakage repair](DB_QUALITY_CONTROL.md).
- [Saved-run and checkpoint compatibility](formats/checkpoints.md).

`design/` groups the retained algorithm documentation with its TeX, PDFs and figures. Start with [quadratic masking](design/QUADRATIC_MASKING.md), [the sliced sandwich](design/SLICED_SANDWICH.md), [piecewise mixing](design/FMIX_PIECEWISE.md) and [compression](design/FCOMPRESS_TRANSPORT_AND_PACKING.md). The design papers retain research terminology where that describes the construction; the operator guide defines current names and defaults.

`history/` contains past plans and `history/research/` holds experimental designs, dated measurements and retired constructions. They are references for retained comparisons, not instructions for a fresh GSS run. Generated circuit runs and report artifacts remain outside this source documentation.
