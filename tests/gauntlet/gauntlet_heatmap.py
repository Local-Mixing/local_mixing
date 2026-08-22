#!/usr/bin/env python3
"""Render one witness heatmap per gauntlet attack class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402


ATTACK_ORDER = ("a1", "xrows", "xtrace", "w1", "w2", "w3")
ATTACK_TITLES = {
    "a1": "direct wire match",
    "xrows": "exact affine recovery from one prefix state",
    "xtrace": "exact affine recovery from the global trace",
    "w1": "single-feature correlation",
    "w2": "capped weight-2 correlation scan (xor/and/or/a&!b/b&!a)",
    "w3": "capped weight-3 correlation scan",
}
GADGET_PERIODS = {
    "nonlinear193": 193,
    "nonlinear291": 291,
    # Read old bundles without advertising the old internal names.
    "gg": 193,
    "gg2": 291,
}


def load_metadata(prefix: Path) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    names: dict[int, str] = {}
    with Path(f"{prefix}.meta").open(encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if parts[0].startswith("target["):
                index = int(parts[0][7:-1])
                if len(parts) < 2:
                    raise ValueError(f"target metadata {index} has no name")
                names[index] = parts[1]
            elif len(parts) >= 2:
                metadata[parts[0]] = parts[1]
    expected = list(range(len(names)))
    if sorted(names) != expected:
        raise ValueError("target metadata indices are not contiguous")
    return metadata, [names[index] for index in expected]


def builder_gadget(prefix: Path, metadata: dict[str, str]) -> str:
    value = metadata.get("builder_gadget", "")
    if value:
        return value
    buildmeta = Path(f"{prefix}.buildmeta")
    if buildmeta.exists():
        with buildmeta.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("gadget\t"):
                    return line.split("\t", 1)[1].strip()
    return metadata.get("gadget", "")


def load_audit_status(prefix: Path) -> dict[str, str]:
    """Read the final machine-readable audit line when it is available."""

    result: str | None = None
    audit_log = prefix.parent / "audit.log"
    if audit_log.exists():
        with audit_log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("RESULT"):
                    result = line.strip()
    fields: dict[str, str] = {}
    for token in (result or "").split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def render(
    prefix: Path,
    attack: str,
    hits: Sequence[dict[str, Any]],
    metadata: dict[str, str],
    names: Sequence[str],
    outdir: Path,
    audit_status: dict[str, str],
) -> Path:
    n_wires = int(metadata["n_wires"])
    n_features = int(metadata["n_features"])
    n_targets = len(names)
    source_gates = int(metadata.get("k", "0"))
    name_to_index = {name: index for index, name in enumerate(names)}

    grid = np.full((n_targets, n_features), np.nan)
    for hit in hits:
        target = hit.get("target")
        if target not in name_to_index:
            continue
        target_index = name_to_index[target]
        strength = float(hit.get("strength", 1.0))
        for feature_value in hit.get("features", ()):
            feature = int(feature_value)
            if not 0 <= feature < n_features:
                continue
            previous = grid[target_index, feature]
            grid[target_index, feature] = max(
                0.0 if np.isnan(previous) else float(previous), strength
            )

    colormap = LinearSegmentedColormap.from_list(
        "gauntlet-hit", ("#ffffcc", "#fd8d3c", "#bd0026")
    )
    colormap.set_bad("#f7f7f7")
    figure_width = min(26, max(12, n_features / 1_200))
    figure_height = max(2.2, min(6.5, n_targets * 0.085 + 1.0))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(
        np.ma.masked_invalid(grid),
        interpolation="nearest",
        cmap=colormap,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_yticks(range(n_targets))
    axis.set_yticklabels(names, fontsize=4.5 if n_targets > 20 else 7)
    axis.tick_params(axis="y", length=0, pad=1)
    axis.axvline(n_wires - 0.5, color="#08519c", linewidth=1.2, linestyle="--")
    text_y = n_targets * 0.55
    axis.text(
        n_wires + n_features * 0.004,
        text_y,
        " gate flips/new values →",
        fontsize=6,
        color="#08519c",
        va="center",
    )
    axis.text(
        n_wires - n_features * 0.004,
        text_y,
        "initial wires ",
        fontsize=6,
        color="#08519c",
        va="center",
        ha="right",
    )

    period = GADGET_PERIODS.get(builder_gadget(prefix, metadata), 0)
    if period and source_gates > 1 and metadata.get("mixed") != "true":
        for gate_index in range(1, source_gates):
            axis.axvline(
                n_wires + 2 * period * gate_index - 0.5,
                color="#999999",
                linewidth=0.4,
                alpha=0.6,
            )

    axis.set_xlabel("trace feature (initial wires, then each flip/new value)", fontsize=8)
    axis.set_ylabel("source-circuit target", fontsize=8)
    gadget = builder_gadget(prefix, metadata) or metadata.get("gadget", "?")
    mixed = metadata.get("mixed") == "true"
    witness_label = f"{len(hits)} witness{'es' if len(hits) != 1 else ''}"
    if attack == "xtrace" and audit_status.get("xtrace_status") == "skipped":
        witness_label = "SKIPPED by feature limit"
    axis.set_title(
        f"{prefix.parent.name} — {gadget}{' + mix' if mixed else ''} — "
        f"{attack}: {ATTACK_TITLES.get(attack, attack)} "
        f"({witness_label})",
        fontsize=9,
    )
    figure.colorbar(image, ax=axis, label="strength (1 = exact; otherwise |covariance|)", shrink=0.6)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{attack}.png"
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args(argv)

    prefix = args.prefix.resolve()
    outdir = args.outdir.resolve() if args.outdir else Path(f"{prefix}.heatmaps")
    metadata, names = load_metadata(prefix)
    audit_status = load_audit_status(prefix)
    hits: list[dict[str, Any]] = []
    hit_path = Path(f"{prefix}.hits.jsonl")
    if hit_path.exists():
        with hit_path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    hits.append(json.loads(raw))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{hit_path}:{line_number}: {error}") from error

    outputs = []
    for attack in ATTACK_ORDER:
        outputs.append(
            render(
                prefix,
                attack,
                [hit for hit in hits if hit.get("attack") == attack],
                metadata,
                names,
                outdir,
                audit_status,
            )
        )
    print(f"[heatmap] {prefix}: {len(outputs)} maps -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
