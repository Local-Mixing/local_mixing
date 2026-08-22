#!/usr/bin/env python3
"""Run the nonlinear gadget security gauntlet.

The pipeline builds source chains, gadgetizes them, optionally mixes them,
records bit-sliced traces, runs exact/correlation audits, and renders witness
heatmaps.  Run ``python tests/gauntlet/gauntlet.py --help`` from any directory
for the command-line interface.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BUILD_SCRIPT = SCRIPT_DIR / "gauntlet_build.py"
HEATMAP_SCRIPT = SCRIPT_DIR / "gauntlet_heatmap.py"
DEFAULT_BIN_DIR = REPO_ROOT / "target" / "release"
DEFAULT_OUTDIR = REPO_ROOT / "target" / "gauntlet"

PIPELINE_SCHEMA = 3
N_WIRES = 8
DEFAULT_MIX_MOVES = 20_000
DEFAULT_W2_CAP = 64
DEFAULT_W3_CAP = 16
DEFAULT_XTRACE_MAX_FEATURES = 40_000

# name -> construction policy.  The three native arms are controls.  Every
# file-mode arm uses one of the two requested gadgetization modules.
ARMS: dict[str, dict[str, Any]] = {
    "none": {
        "kind": "native",
        "aux": ("zero", "random"),
        "rust_gadget": "none",
    },
    "secretshare14": {
        "kind": "native",
        "aux": ("zero", "random"),
        "rust_gadget": "ss",
    },
    "bandproduct92": {
        "kind": "native",
        "aux": ("zero", "random"),
        "rust_gadget": "semi",
    },
    "nonlinear193": {
        "kind": "file",
        "aux": ("builder",),
        "builder_gadget": "nonlinear193",
        "pool": "ideal",
        "blind_layers": 0,
    },
    "nonlinear291": {
        "kind": "file",
        "aux": ("builder",),
        "builder_gadget": "nonlinear291",
        "pool": "ideal",
        "blind_layers": 0,
    },
    "nonlinear193_band0": {
        "kind": "file",
        "aux": ("builder",),
        "builder_gadget": "nonlinear193",
        "pool": "band",
        "blind_layers": 0,
    },
    "nonlinear193_band16": {
        "kind": "file",
        "aux": ("builder",),
        "builder_gadget": "nonlinear193",
        "pool": "band",
        "blind_layers": 16,
    },
    "nonlinear291_band0": {
        "kind": "file",
        "aux": ("builder",),
        "builder_gadget": "nonlinear291",
        "pool": "band",
        "blind_layers": 0,
    },
    "nonlinear291_band16": {
        "kind": "file",
        "aux": ("builder",),
        "builder_gadget": "nonlinear291",
        "pool": "band",
        "blind_layers": 16,
    },
}


def root_relative(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    signature: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }
    return signature


def binary_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            key, separator, value = raw.rstrip("\n").partition("\t")
            if separator:
                values[key] = value
    return values


def run(
    command: Sequence[str | os.PathLike[str]],
    log_path: Path,
    *,
    env_extra: dict[str, str] | None = None,
    append: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one stage from the repository root and tee its output to a log."""

    argv = [os.fspath(part) for part in command]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] $ {shlex.join(argv)}\n")
        log.flush()
        result = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[exit {result.returncode}]\n")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")
    return result


def last_prefixed_line(path: Path, prefix: str) -> str | None:
    found: str | None = None
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(prefix):
                found = line.strip()
    return found


def corr_samples(arm: str, k: int, override: int | None) -> int:
    if override is not None:
        value = override
    elif arm.startswith(("nonlinear193", "nonlinear291")) and k >= 16:
        value = 16_384
    elif k <= 1:
        value = 16_384
    elif k <= 2:
        value = 8_192
    else:
        value = 4_096
    if value <= 0 or value % 64:
        raise ValueError("correlation samples must be a positive multiple of 64")
    return value


def source_chain(path: Path, k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"mpmct1 {N_WIRES} {k}"]
    for gate_index in range(k):
        lines.append(
            f"{gate_index % N_WIRES} 1 2 "
            f"{(gate_index + 3) % N_WIRES} 0 "
            f"{(gate_index + 5) % N_WIRES} 1"
        )
    content = "\n".join(lines) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(
            f"generated chain {path} has unexpected contents; use clean or another --outdir"
        )
    path.write_text(content, encoding="utf-8")


def analytic_file_layout(gadget: str, k: int) -> tuple[int, int]:
    extra = 27 if gadget == "nonlinear291" else 0
    return 10 * N_WIRES + 12 * k + 2 + extra, EXPECTED_PERIOD[gadget] * k


EXPECTED_PERIOD = {"nonlinear193": 193, "nonlinear291": 291}


def required_samples(n_features: int, correlation_samples: int) -> int:
    fit = ((n_features + 256 + 63) // 64) * 64
    return fit + 2_048 + correlation_samples


def build_command(
    policy: dict[str, Any],
    chain: Path,
    prefix: Path,
    *,
    seed: int,
    samples: int,
    pool_keys: int,
) -> list[str]:
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--gadget",
        policy["builder_gadget"],
        "--c-in",
        str(chain),
        "--out-prefix",
        str(prefix),
        "--n",
        str(N_WIRES),
        "--seed",
        str(seed),
        "--samples",
        str(samples),
        "--pool",
        policy.get("pool", "ideal"),
        "--blind-layers",
        str(policy.get("blind_layers", 0)),
        "--pool-keys",
        str(pool_keys),
    ]
    return command


def generation_artifacts(prefix: Path, kind: str) -> list[Path]:
    artifacts = [
        Path(f"{prefix}.meta"),
        Path(f"{prefix}.trace.bin"),
        Path(f"{prefix}.targets.bin"),
        Path(f"{prefix}.g.mpmct1"),
    ]
    if kind == "file":
        artifacts.extend(
            [
                Path(f"{prefix}.mpmct1"),
                Path(f"{prefix}.init.bin"),
                Path(f"{prefix}.buildmeta"),
            ]
        )
    return artifacts


def generation_config(
    *,
    arm: str,
    policy: dict[str, Any],
    aux: str,
    k: int,
    mix_on: bool,
    correlation_samples: int,
    mix_moves: int,
    pool_keys: int,
    gen_binary: Path,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema": PIPELINE_SCHEMA,
        "arm": arm,
        "policy": policy,
        "aux": aux,
        "k": k,
        "n": N_WIRES,
        "mix": mix_on,
        "mix_moves": mix_moves if mix_on else 0,
        "mix_seed": 777,
        "seed": 100 + k,
        "gadget_seed": 7_000 + k,
        "correlation_samples": correlation_samples,
        "pool_keys": pool_keys,
        "source_chain_recipe": "serial-r57-v1",
        "generator_binary": binary_signature(gen_binary),
        "generator_source": source_signature(SCRIPT_DIR / "gauntlet_gen.rs"),
    }
    if policy["kind"] == "file":
        config["builder_source"] = source_signature(BUILD_SCRIPT)
        config["nonlinear193_source"] = source_signature(
            REPO_ROOT / "gadgetization" / "nonlinear193.py"
        )
        config["nonlinear291_source"] = source_signature(
            REPO_ROOT / "gadgetization" / "nonlinear291.py"
        )
    return config


def audit_config(
    *,
    generation_digest: str,
    audit_binary: Path,
    w2_cap: int,
    w3_cap: int,
    xtrace_max_features: int,
    witnesses: int,
) -> dict[str, Any]:
    return {
        "schema": PIPELINE_SCHEMA,
        "generation_digest": generation_digest,
        "auditor_binary": binary_signature(audit_binary),
        "auditor_source": source_signature(SCRIPT_DIR / "gauntlet_audit.rs"),
        "w2_cap": w2_cap,
        "w3_cap": w3_cap,
        "xtrace_max_features": xtrace_max_features,
        "witnesses": witnesses,
    }


def maps_config(*, audit_digest: str) -> dict[str, Any]:
    return {
        "schema": PIPELINE_SCHEMA,
        "audit_digest": audit_digest,
        "heatmap_source": source_signature(HEATMAP_SCRIPT),
    }


def cell_name(arm: str, aux: str, mix_on: bool) -> str:
    auxiliary = f"_{aux}" if aux != "builder" else ""
    return f"{arm}{auxiliary}_{'mix' if mix_on else 'nomix'}"


def iter_cells(
    arms: Sequence[str], ks: Sequence[int], mix_modes: Sequence[bool]
) -> list[tuple[int, str, str, bool]]:
    return [
        (k, arm, aux, mix_on)
        for k in ks
        for arm in arms
        for aux in ARMS[arm]["aux"]
        for mix_on in mix_modes
    ]


def reset_cell(cdir: Path) -> None:
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)


def artifact_signatures(artifacts: Sequence[Path]) -> list[dict[str, Any]]:
    """Return content-bound records for a stage's expected regular files."""

    signatures: list[dict[str, Any]] = []
    for artifact in artifacts:
        resolved = artifact.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"stage artifact is not a regular file: {artifact}")
        stat = resolved.stat()
        signatures.append(
            {
                "path": str(resolved),
                "size": stat.st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return signatures


def save_stage(
    manifest_path: Path,
    stage: str,
    config: dict[str, Any],
    artifacts: Sequence[Path],
) -> None:
    manifest = read_json(manifest_path, {"schema": PIPELINE_SCHEMA, "stages": {}})
    manifest["schema"] = PIPELINE_SCHEMA
    manifest.setdefault("stages", {})[stage] = {
        "config": config,
        "digest": digest_json(config),
        "artifacts": artifact_signatures(artifacts),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # Downstream stages are invalidated when an upstream stage is replaced.
    if stage == "generation":
        manifest["stages"].pop("audit", None)
        manifest["stages"].pop("maps", None)
    elif stage == "audit":
        manifest["stages"].pop("maps", None)
    write_json_atomic(manifest_path, manifest)


def stage_is_current(
    manifest: dict[str, Any], stage: str, config: dict[str, Any], artifacts: Sequence[Path]
) -> bool:
    saved = manifest.get("stages", {}).get(stage, {})
    if saved.get("digest") != digest_json(config):
        return False
    return saved_stage_artifacts_are_current(saved, artifacts)


def saved_stage_artifacts_are_current(
    saved_stage: dict[str, Any], artifacts: Sequence[Path]
) -> bool:
    """Validate artifacts without requiring the stage's original configuration."""

    try:
        current_artifacts = artifact_signatures(artifacts)
    except (FileNotFoundError, OSError):
        return False
    return saved_stage.get("artifacts") == current_artifacts


def generate_cell(
    *,
    cdir: Path,
    arm: str,
    aux: str,
    k: int,
    mix_on: bool,
    chain: Path,
    gen_binary: Path,
    correlation_samples: int,
    mix_moves: int,
    pool_keys: int,
    rayon_threads: int,
) -> None:
    policy = ARMS[arm]
    prefix = cdir / "bundle"
    seed, gadget_seed = 100 + k, 7_000 + k
    env = {"RAYON_NUM_THREADS": str(rayon_threads)}

    if policy["kind"] == "file":
        builder_gadget = policy["builder_gadget"]
        n_wires, n_gates = analytic_file_layout(builder_gadget, k)
        if mix_on:
            # The mixer may change the gate count.  Probe with a cell-local
            # prefix; this avoids the shared /tmp/mxprobe race in the PR script.
            run(
                build_command(
                    policy,
                    chain,
                    prefix,
                    seed=seed,
                    samples=64,
                    pool_keys=pool_keys,
                ),
                cdir / "build.log",
                append=False,
            )
            probe_dir = cdir / ".probe"
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_log = probe_dir / "probe.log"
            probe_prefix = probe_dir / "bundle"
            probe_command = [
                str(gen_binary),
                "--gadget",
                "file",
                "--g-in",
                f"{prefix}.mpmct1",
                "--init-in",
                f"{prefix}.init.bin",
                "--c-in",
                str(chain),
                "--out-prefix",
                str(probe_prefix),
                "--n",
                str(N_WIRES),
                "--seed",
                str(seed),
                "--gadget-seed",
                str(gadget_seed),
                "--corr-samples",
                "64",
                "--aux",
                "zero",
                "--mix",
                str(mix_moves),
                "--size-only",
            ]
            run(
                probe_command,
                probe_log,
                env_extra=env,
                append=False,
            )
            size_line = last_prefixed_line(probe_log, "[size]")
            match = re.search(r"\bgates=(\d+)\b", size_line or "")
            if not match:
                raise RuntimeError(f"mix-size probe did not report a gate count; see {probe_log}")
            n_gates = int(match.group(1))

        samples = required_samples(n_wires + 2 * n_gates, correlation_samples)
        run(
            build_command(
                policy,
                chain,
                prefix,
                seed=seed,
                samples=samples,
                pool_keys=pool_keys,
            ),
            cdir / "build.log",
            append=mix_on,
        )
        buildmeta = read_kv(Path(f"{prefix}.buildmeta"))
        if buildmeta.get("builder_checked") != "true":
            raise RuntimeError(f"builder did not certify its output: {prefix}.buildmeta")
        if int(buildmeta["n_wires"]) != n_wires:
            raise RuntimeError("builder wire layout disagrees with orchestrator")

        command = [
            str(gen_binary),
            "--gadget",
            "file",
            "--g-in",
            f"{prefix}.mpmct1",
            "--init-in",
            f"{prefix}.init.bin",
            "--c-in",
            str(chain),
            "--out-prefix",
            str(prefix),
            "--n",
            str(N_WIRES),
            "--seed",
            str(seed),
            "--gadget-seed",
            str(gadget_seed),
            "--corr-samples",
            str(correlation_samples),
            "--aux",
            "zero",
        ]
    else:
        command = [
            str(gen_binary),
            "--gadget",
            policy["rust_gadget"],
            "--c-in",
            str(chain),
            "--out-prefix",
            str(prefix),
            "--n",
            str(N_WIRES),
            "--seed",
            str(seed),
            "--gadget-seed",
            str(gadget_seed),
            "--corr-samples",
            str(correlation_samples),
            "--aux",
            aux,
        ]
    if mix_on:
        command.extend(("--mix", str(mix_moves)))
    run(command, cdir / "gen.log", env_extra=env, append=False)

    metadata = read_kv(Path(f"{prefix}.meta"))
    if metadata.get("behavioral_ok") != "true":
        raise RuntimeError(
            f"generated circuit failed its behavioral check; see {prefix}.meta and {cdir / 'gen.log'}"
        )


def audit_cell(
    *,
    cdir: Path,
    audit_binary: Path,
    w2_cap: int,
    w3_cap: int,
    xtrace_max_features: int,
    witnesses: int,
    rayon_threads: int,
) -> str:
    prefix = cdir / "bundle"
    command = [
        str(audit_binary),
        "--prefix",
        str(prefix),
        "--w2-cap",
        str(w2_cap),
        "--w3-cap",
        str(w3_cap),
        "--a3-max-f",
        str(xtrace_max_features),
        "--witnesses",
        str(witnesses),
    ]
    log_path = cdir / "audit.log"
    run(
        command,
        log_path,
        env_extra={"RAYON_NUM_THREADS": str(rayon_threads)},
        append=False,
    )
    result = last_prefixed_line(log_path, "RESULT")
    if result is None:
        raise RuntimeError(f"auditor emitted no RESULT line; see {log_path}")
    return result


def maps_cell(cdir: Path) -> None:
    prefix = cdir / "bundle"
    run(
        [
            sys.executable,
            str(HEATMAP_SCRIPT),
            "--prefix",
            str(prefix),
            "--outdir",
            str(cdir / "heatmaps"),
        ],
        cdir / "maps.log",
        append=False,
    )


def parse_result(result: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not result:
        return parsed
    for token in result.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            parsed[key] = value
    return parsed


def build_report(
    outdir: Path,
    *,
    arms: Sequence[str],
    ks: Sequence[int],
    mix_modes: Sequence[bool],
) -> None:
    lines = [
        "# Gadget gauntlet report",
        "",
        "The correlation battery is a bounded empirical scan, not an exhaustive "
        "security proof. `w1` covers every recorded feature; `w2` and `w3` use "
        "deterministic strided feature subsets. Each row records its actual sampled "
        "feature counts, so a report remains accurate when regenerated with different "
        "CLI defaults. See `tests/gauntlet/README.md` for the exact attack families.",
        "",
    ]
    index: list[dict[str, Any]] = []
    for k in ks:
        lines.extend(
            [
                f"## k = {k}",
                "",
                "| cell | a1 nontrivial | xrows nontrivial | xtrace | w1 flags | w2 flags | w3 flags | scan/status |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for arm in arms:
            for aux in ARMS[arm]["aux"]:
                for mix_on in mix_modes:
                    name = cell_name(arm, aux, mix_on)
                    cdir = outdir / f"k{k}" / name
                    audit_log = cdir / "audit.log"
                    audit_artifacts = [
                        Path(f"{cdir / 'bundle'}.hits.jsonl"),
                        audit_log,
                    ]
                    manifest = read_json(cdir / "cell-config.json", {})
                    saved_audit = manifest.get("stages", {}).get("audit")
                    if saved_audit is None and not audit_log.exists():
                        continue
                    artifacts_current = isinstance(
                        saved_audit, dict
                    ) and saved_stage_artifacts_are_current(saved_audit, audit_artifacts)
                    if not artifacts_current:
                        raise RuntimeError(
                            f"{cdir}: audit artifacts failed manifest integrity; "
                            "rerun the audit stage"
                        )
                    result = last_prefixed_line(audit_log, "RESULT")
                    values = parse_result(result)
                    if not values:
                        continue
                    xtrace = values.get("xtrace_nt", "?")
                    xtrace_status = values.get("xtrace_status")
                    if xtrace_status and xtrace_status not in ("run", "completed"):
                        xtrace = f"{xtrace} ({xtrace_status})"
                    coverage = ", ".join(
                        f"{key}={values[key]}"
                        for key in (
                            "w1_scanned",
                            "w2_sampled",
                            "w3_sampled",
                            "n_features",
                            "corr_selection",
                            "xtrace_status",
                        )
                        if key in values
                    ) or "see audit.log"
                    heatmaps = cdir / "heatmaps"
                    try:
                        heatmap_link = heatmaps.relative_to(outdir).as_posix()
                    except ValueError:
                        heatmap_link = str(heatmaps)
                    lines.append(
                        f"| [{name}]({heatmap_link}) | {values.get('a1_nt', '?')} | "
                        f"{values.get('xrows_nt', '?')} | {xtrace} | "
                        f"{values.get('w1flag', '?')} | {values.get('w2flag', '?')} | "
                        f"{values.get('w3flag', '?')} | {coverage} |"
                    )
                    index.append(
                        {"cell": [k, name], "result": result, "fields": values}
                    )
        lines.append("")
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json_atomic(outdir / "index.json", index)
    print(f"[report] wrote {outdir / 'REPORT.md'}")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def clean_output(outdir: Path, *, force: bool) -> None:
    resolved = outdir.resolve()
    repo = REPO_ROOT.resolve()
    target_root = (repo / "target").resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    # Source paths are never output-owned. Inside the repository, only a
    # descendant of target/ can be removed, and never target/ itself.
    unsafe_repo_path = is_within(resolved, repo) and not (
        resolved != target_root and is_within(resolved, target_root)
    )
    if unsafe_repo_path or resolved in repo.parents:
        raise RuntimeError(f"refusing to clean unsafe path: {resolved}")
    if not resolved.exists():
        print(f"[clean] nothing to remove at {resolved}")
        return
    marker = resolved / ".gauntlet-output.json"
    if not valid_output_marker(marker, resolved):
        if not force:
            raise RuntimeError(
                f"{resolved} has no valid path-bound gauntlet marker"
            )
        safe_force_root = any(
            resolved != root and is_within(resolved, root)
            for root in (target_root, temp_root)
        )
        if not safe_force_root:
            raise RuntimeError(
                "forced cleanup of an unmarked directory is limited to a child of "
                f"{target_root} or {temp_root}: {resolved}"
            )
    shutil.rmtree(resolved)
    print(f"[clean] removed {resolved}")


def valid_output_marker(path: Path, outdir: Path) -> bool:
    try:
        marker = read_json(path, {})
    except (OSError, ValueError, TypeError):
        return False
    marker_repo = marker.get("repo")
    marker_outdir = marker.get("outdir")
    return (
        marker.get("schema") == PIPELINE_SCHEMA
        and marker.get("kind") == "local-mixing-gauntlet-output"
        and isinstance(marker_repo, str)
        and Path(marker_repo).resolve() == REPO_ROOT.resolve()
        and isinstance(marker_outdir, str)
        and Path(marker_outdir).resolve() == outdir.resolve()
    )


def prepare_output(outdir: Path, *, force: bool) -> None:
    """Create or validate a gauntlet-owned output directory.

    An existing nonempty directory is never silently adopted: doing so would
    make the marker authorize a later recursive clean of unrelated files.
    Within ``target/`` or the system temporary directory, ``--force`` can
    explicitly reset such a directory before ownership is marked.
    """

    resolved = outdir.resolve()
    repo = REPO_ROOT.resolve()
    target_root = (repo / "target").resolve()
    unsafe_repo_path = is_within(resolved, repo) and not (
        resolved != target_root and is_within(resolved, target_root)
    )
    if unsafe_repo_path or resolved in repo.parents:
        raise RuntimeError(f"refusing unsafe gauntlet output path: {resolved}")

    marker = outdir / ".gauntlet-output.json"
    if outdir.exists() and not valid_output_marker(marker, outdir) and any(outdir.iterdir()):
        if not force:
            raise RuntimeError(
                f"refusing to adopt nonempty or invalidly marked --outdir {outdir}; "
                "choose an empty directory, or use --force under target/ or the "
                "system temporary directory"
            )
        clean_output(outdir, force=True)
    outdir.mkdir(parents=True, exist_ok=True)
    if not valid_output_marker(marker, outdir):
        write_json_atomic(
            marker,
            {
                "schema": PIPELINE_SCHEMA,
                "kind": "local-mixing-gauntlet-output",
                "repo": str(REPO_ROOT),
                "outdir": str(outdir.resolve()),
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd", choices=("all", "gen", "audit", "maps", "report", "clean")
    )
    parser.add_argument("--ks", default="1,2,16", help="comma-separated chain lengths")
    parser.add_argument("--mix", choices=("both", "on", "off"), default="both")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--mix-moves", type=int, default=DEFAULT_MIX_MOVES)
    parser.add_argument(
        "--corr-samples",
        type=int,
        default=None,
        help="override every arm's correlation-tail samples (multiple of 64)",
    )
    parser.add_argument("--pool-keys", type=int, default=120)
    parser.add_argument(
        "--w2-cap",
        type=int,
        default=DEFAULT_W2_CAP,
        help="weight-2 feature-subset cap (minimum 2; pair work is combinatorial)",
    )
    parser.add_argument(
        "--w3-cap",
        type=int,
        default=DEFAULT_W3_CAP,
        help="weight-3 feature-subset cap (minimum 3; triple work is combinatorial)",
    )
    parser.add_argument(
        "--xtrace-max-features", type=int, default=DEFAULT_XTRACE_MAX_FEATURES
    )
    parser.add_argument("--witnesses", type=int, default=10)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "replace stale generation artifacts; unmarked directory resets are "
            "limited to children of target/ or the system temporary directory"
        ),
    )
    args = parser.parse_args(argv)

    outdir = root_relative(args.outdir).resolve()
    bin_dir = root_relative(args.bin_dir).resolve()
    if args.cmd == "clean":
        clean_output(outdir, force=args.force)
        return 0

    try:
        ks = [int(value) for value in args.ks.split(",") if value]
    except ValueError as error:
        parser.error(f"invalid --ks: {error}")
    arms = [value for value in args.arms.split(",") if value]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {', '.join(unknown)}")
    if not ks or any(k <= 0 for k in ks):
        parser.error("--ks must contain positive integers")
    if not arms:
        parser.error("--arms cannot be empty")
    if args.jobs <= 0 or args.mix_moves <= 0:
        parser.error("--jobs and --mix-moves must be positive")
    if min(args.pool_keys, args.xtrace_max_features, args.witnesses) <= 0:
        parser.error("pool keys, feature limit, and witnesses must be positive")
    if args.w2_cap < 2:
        parser.error("--w2-cap must be at least 2")
    if args.w3_cap < 3:
        parser.error("--w3-cap must be at least 3")
    if args.corr_samples is not None and (
        args.corr_samples <= 0 or args.corr_samples % 64
    ):
        parser.error("--corr-samples must be a positive multiple of 64")

    mix_modes = {"both": (False, True), "on": (True,), "off": (False,)}[args.mix]
    gen_binary = bin_dir / "gauntlet_gen"
    audit_binary = bin_dir / "gauntlet_audit"

    # Report is intentionally usable without built binaries or Python plotting
    # dependencies; it re-reads existing audit logs.
    if args.cmd == "report":
        prepare_output(outdir, force=args.force)
        build_report(
            outdir,
            arms=arms,
            ks=ks,
            mix_modes=mix_modes,
        )
        return 0

    if args.cmd in ("all", "gen") and not gen_binary.is_file():
        parser.error(
            f"missing {gen_binary}; run `cargo build --release --bin gauntlet_gen --bin gauntlet_audit`"
        )
    if args.cmd in ("all", "audit") and not audit_binary.is_file():
        parser.error(
            f"missing {audit_binary}; run `cargo build --release --bin gauntlet_gen --bin gauntlet_audit`"
        )

    prepare_output(outdir, force=args.force)

    cells = iter_cells(arms, ks, mix_modes)
    # Create shared deterministic inputs before worker threads start so no two
    # cells can observe a partially written chain.
    for k in ks:
        source_chain(outdir / "_inputs" / f"chain_k{k}.mpmct1", k)
    rayon_threads = max(1, (os.cpu_count() or 1) // args.jobs)

    def do_cell(cell: tuple[int, str, str, bool]) -> None:
        k, arm, aux, mix_on = cell
        name = cell_name(arm, aux, mix_on)
        cdir = outdir / f"k{k}" / name
        cdir.mkdir(parents=True, exist_ok=True)
        prefix = cdir / "bundle"
        manifest_path = cdir / "cell-config.json"
        chain = outdir / "_inputs" / f"chain_k{k}.mpmct1"
        correlation_samples = corr_samples(arm, k, args.corr_samples)
        gen_config = generation_config(
            arm=arm,
            policy=ARMS[arm],
            aux=aux,
            k=k,
            mix_on=mix_on,
            correlation_samples=correlation_samples,
            mix_moves=args.mix_moves,
            pool_keys=args.pool_keys,
            gen_binary=gen_binary,
        )
        gen_digest = digest_json(gen_config)
        gen_artifacts = generation_artifacts(prefix, ARMS[arm]["kind"])
        manifest = read_json(manifest_path, {})
        gen_current = stage_is_current(
            manifest, "generation", gen_config, gen_artifacts
        )

        if args.cmd in ("all", "gen") and not gen_current:
            has_complete_old_generation = all(path.exists() for path in gen_artifacts)
            saved_generation = manifest.get("stages", {}).get("generation", {})
            config_changed = bool(saved_generation) and saved_generation.get("digest") != gen_digest
            if has_complete_old_generation and config_changed and not args.force:
                raise RuntimeError(
                    f"{cdir}: generation provenance changed; rerun with --force or use clean"
                )
            # A matching configuration with mismatched artifact hashes is a
            # corrupt/stale owned cell, so rebuild it from an empty directory.
            # A complete bundle whose configuration changed still requires
            # the explicit authorization checked above.
            reset_cell(cdir)
            print(f"[cell] gen {name} k={k}", flush=True)
            generate_cell(
                cdir=cdir,
                arm=arm,
                aux=aux,
                k=k,
                mix_on=mix_on,
                chain=chain,
                gen_binary=gen_binary,
                correlation_samples=correlation_samples,
                mix_moves=args.mix_moves,
                pool_keys=args.pool_keys,
                rayon_threads=rayon_threads,
            )
            save_stage(manifest_path, "generation", gen_config, gen_artifacts)
            manifest = read_json(manifest_path, {})
            gen_current = True

        if args.cmd in ("audit", "maps") and not gen_current:
            raise RuntimeError(
                f"{cdir}: no current generated bundle; run the gen command with the same options"
            )

        audit_cfg = audit_config(
            generation_digest=gen_digest,
            audit_binary=audit_binary,
            w2_cap=args.w2_cap,
            w3_cap=args.w3_cap,
            xtrace_max_features=args.xtrace_max_features,
            witnesses=args.witnesses,
        )
        audit_artifacts = [Path(f"{prefix}.hits.jsonl"), cdir / "audit.log"]
        audit_current = stage_is_current(
            manifest, "audit", audit_cfg, audit_artifacts
        )
        if args.cmd in ("all", "audit") and not audit_current:
            if not gen_current:
                raise RuntimeError(f"{cdir}: audit requires a current generated bundle")
            print(f"[cell] audit {name} k={k}", flush=True)
            audit_cell(
                cdir=cdir,
                audit_binary=audit_binary,
                w2_cap=args.w2_cap,
                w3_cap=args.w3_cap,
                xtrace_max_features=args.xtrace_max_features,
                witnesses=args.witnesses,
                rayon_threads=rayon_threads,
            )
            save_stage(manifest_path, "audit", audit_cfg, audit_artifacts)
            manifest = read_json(manifest_path, {})
            audit_current = True

        if args.cmd == "maps" and not audit_current:
            raise RuntimeError(
                f"{cdir}: no current audit results; run audit with the same options"
            )

        map_cfg = maps_config(audit_digest=digest_json(audit_cfg))
        map_artifacts = [
            cdir / "heatmaps" / f"{attack}.png"
            for attack in ("a1", "xrows", "xtrace", "w1", "w2", "w3")
        ]
        maps_current = stage_is_current(manifest, "maps", map_cfg, map_artifacts)
        if args.cmd in ("all", "maps") and not maps_current:
            if not audit_current:
                raise RuntimeError(f"{cdir}: maps require current audit results")
            print(f"[cell] maps {name} k={k}", flush=True)
            maps_cell(cdir)
            save_stage(manifest_path, "maps", map_cfg, map_artifacts)

    if args.jobs == 1:
        for cell in cells:
            do_cell(cell)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(do_cell, cell) for cell in cells]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if args.cmd in ("all", "audit"):
        build_report(
            outdir,
            arms=arms,
            ks=ks,
            mix_modes=mix_modes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
