#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


HEADER_RE = re.compile(r"parsed 'p cnf (\d+) (\d+)' header")

FIELDS = [
    "orig_vars",
    "orig_clauses",
    "progress_seconds",
    "conflicts",
    "vars",
    "clauses",
    "remaining_pct",
    "binary_clauses",
    "redundant_clauses",
    "conflicts_per_sec_window",
    "vars_delta_window",
    "clauses_delta_window",
    "redundant_delta_window",
    "clause_density",
    "binary_ratio",
    "motion",
    "progress_line",
]

EXTRA_FIELDS = [
    "orig_vars",
    "orig_clauses",
    "progress_seconds",
    "binary_clauses",
    "redundant_clauses",
    "conflicts_per_sec_window",
    "vars_delta_window",
    "clauses_delta_window",
    "redundant_delta_window",
    "clause_density",
    "binary_ratio",
    "motion",
    "progress_line",
]


def read_text(path: Path, max_bytes: int | None = None) -> str:
    if not path.exists():
        return ""
    if max_bytes is None:
        return path.read_text(errors="replace")
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", "replace")


def read_head(path: Path, max_bytes: int = 256_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        return handle.read(max_bytes).decode("utf-8", "replace")


def parse_progress_line(line: str) -> dict[str, str] | None:
    parts = line.split()
    if len(parts) < 21 or parts[0] != "c" or len(parts[1]) != 1:
        return None
    if not parts[20].endswith("%"):
        return None
    try:
        float(parts[2])
        int(parts[9])
        int(parts[10])
        int(parts[17])
        int(parts[18])
        int(parts[19])
    except ValueError:
        return None
    return {
        "type": parts[1],
        "progress_seconds": parts[2],
        "conflicts": parts[9],
        "redundant_clauses": parts[10],
        "binary_clauses": parts[17],
        "clauses": parts[18],
        "vars": parts[19],
        "remaining_pct": parts[20],
        "progress_line": line.rstrip(),
    }


def parse_header(text: str) -> tuple[str, str]:
    for line in text.splitlines():
        match = HEADER_RE.search(line)
        if match:
            return match.group(1), match.group(2)
    return "", ""


def progress_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        parsed = parse_progress_line(line)
        if parsed:
            rows.append(parsed)
    return rows


def _int(row: dict[str, str], key: str) -> int:
    return int(row.get(key, "") or "0")


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "") or "0")


def _fmt_float(value: float, digits: int = 2) -> str:
    if value != value:
        return ""
    return f"{value:.{digits}f}"


def motion_label(cps: float, d_vars: int, d_clauses: int, d_redundant: int) -> str:
    if d_vars < 0 or d_clauses < 0:
        return "shrinking"
    if cps >= 5000:
        return "fast churn"
    if cps >= 1000:
        return "grinding"
    if cps > 0:
        return "slow/stalled"
    if d_redundant > 0:
        return "quiet learning"
    return "quiet"


def metrics_from_text(text: str, header_text: str = "", window_rows: int = 10) -> dict[str, str]:
    out = {field: "" for field in FIELDS}
    orig_vars, orig_clauses = parse_header(header_text or text)
    out["orig_vars"] = orig_vars
    out["orig_clauses"] = orig_clauses

    rows = progress_rows(text)
    if not rows:
        return out

    last = rows[-1]
    out.update({key: last.get(key, "") for key in out if key in last})

    base = rows[max(0, len(rows) - window_rows - 1)]
    dt = max(1e-9, _float(last, "progress_seconds") - _float(base, "progress_seconds"))
    dc = _int(last, "conflicts") - _int(base, "conflicts")
    d_vars = _int(last, "vars") - _int(base, "vars")
    d_clauses = _int(last, "clauses") - _int(base, "clauses")
    d_redundant = _int(last, "redundant_clauses") - _int(base, "redundant_clauses")
    cps = dc / dt

    vars_now = _int(last, "vars")
    clauses_now = _int(last, "clauses")
    binary_now = _int(last, "binary_clauses")
    density = clauses_now / vars_now if vars_now else 0.0
    binary_ratio = binary_now / (binary_now + clauses_now) if binary_now + clauses_now else 0.0

    out["conflicts_per_sec_window"] = _fmt_float(cps, 0)
    out["vars_delta_window"] = str(d_vars)
    out["clauses_delta_window"] = str(d_clauses)
    out["redundant_delta_window"] = str(d_redundant)
    out["clause_density"] = _fmt_float(density, 2)
    out["binary_ratio"] = _fmt_float(binary_ratio, 4)
    out["motion"] = motion_label(cps, d_vars, d_clauses, d_redundant)
    return out


def metrics_from_file(path: Path, tail_bytes: int | None = None) -> dict[str, str]:
    if tail_bytes is None:
        text = read_text(path)
        return metrics_from_text(text)
    head = read_head(path)
    tail = read_text(path, tail_bytes)
    return metrics_from_text(tail, header_text=head)


def short_metric_summary(metrics: dict[str, str]) -> str:
    bits = []
    if metrics.get("vars"):
        bits.append(f"vars={metrics['vars']}")
    if metrics.get("clauses"):
        bits.append(f"irr={metrics['clauses']}")
    if metrics.get("conflicts_per_sec_window"):
        bits.append(f"c/s={metrics['conflicts_per_sec_window']}")
    if metrics.get("clause_density"):
        bits.append(f"dens={metrics['clause_density']}")
    if metrics.get("motion"):
        bits.append(metrics["motion"])
    return " ".join(bits)
