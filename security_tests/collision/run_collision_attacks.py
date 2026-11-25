#!/usr/bin/env python3
"""Run birthday, rho, and SAT collisions on one 3λ-wire hex circuit.

The circuit file is the hexadecimal gate list written by `CircuitSeq::to_hex`
(one `target pos_ctrl neg_ctrl` line per gate). Wire 0 is the LSB. The hash is

    H(x) = C(0^λ || x)_λ

with λ = wires/3, a 2λ-bit message, and a λ-bit digest. Reported outputs are
the full wire state, so the bits above the digest are included.

CPU time is `/usr/bin/time -p` user + sys, summed across threads. SAT progress
is appended to `<out-dir>/sat-progress.log`. Rho progress is written to
`<out-dir>/rho-progress.log` about every `--progress-secs`. Both are mirrored
on stderr.

Usage (from the repo root):

    python3 security_tests/collision/run_collision_attacks.py \\
        ckt-out/c96.9216.txt \\
        --out-dir target/security-demo/collision/c96
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIME = "/usr/bin/time"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=ROOT, text=True, capture_output=True, check=False, **kwargs
    )


def bin_dir() -> Path:
    base = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    return base / "release"


def ensure_tools(encoder: Path) -> Path:
    built = bin_dir()
    result = run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "security-tools",
            "--bin",
            "birthday_collision",
            "--bin",
            "rho_collision",
            "--bin",
            "local_mixing_bin",
        ]
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    encoder.parent.mkdir(parents=True, exist_ok=True)
    if not encoder.exists():
        source = ROOT / "security_tests" / "collision" / "collision_to_cnf.cpp"
        compiled = run(["g++", "-std=c++17", "-O3", str(source), "-o", str(encoder)])
        if compiled.returncode != 0:
            raise SystemExit(compiled.stderr or compiled.stdout)
    for name in ("birthday_collision", "rho_collision", "local_mixing_bin"):
        path = built / name
        if not path.exists():
            raise SystemExit(f"missing binary {path}")
    return built


def load_hex_gates(path: Path) -> tuple[list[tuple[int, int, int]], int]:
    gates: list[tuple[int, int, int]] = []
    max_wire = 0
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3 or not all(all(c in "0123456789abcdefABCDEF" for c in f) for f in fields):
            raise SystemExit(f"{path}:{line_number} is not a 3-field hex gate")
        gate = (int(fields[0], 16), int(fields[1], 16), int(fields[2], 16))
        gates.append(gate)
        max_wire = max(max_wire, *gate)
    if not gates:
        raise SystemExit(f"{path} has no gates")
    return gates, max_wire + 1


def write_mpmct(path: Path, gates: list[tuple[int, int, int]], wires: int) -> None:
    """G57 [a, x, y] means a ^= x OR NOT y, i.e. comp=1 and monomial NOT x AND y."""
    lines = [f"mpmct1 {wires} {len(gates)}"]
    for target, pos, neg in gates:
        if pos == neg:
            lines.append(f"{target} 0 0")
        elif pos < neg:
            lines.append(f"{target} 1 2 {pos} 0 {neg} 1")
        else:
            lines.append(f"{target} 1 2 {neg} 1 {pos} 0")
    path.write_text("\n".join(lines) + "\n")


def parse_time(stderr: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for line in stderr.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"real", "user", "sys"}:
            found[parts[0]] = float(parts[1])
    missing = {"real", "user", "sys"} - found.keys()
    if missing:
        raise SystemExit(f"/usr/bin/time output missing {sorted(missing)}:\n{stderr[-800:]}")
    found["cpu_seconds"] = found["user"] + found["sys"]
    return found


def run_timed(cmd: list[str]) -> tuple[str, str, dict[str, float], int]:
    """Run cmd under /usr/bin/time -p. Child stderr is echoed as it arrives."""
    proc = subprocess.Popen(
        [TIME, "-p", *cmd],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_lines: list[str] = []

    def pump() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            print(line, end="", flush=True)

    thread = threading.Thread(target=pump)
    thread.start()
    stdout = proc.stdout.read() if proc.stdout is not None else ""
    returncode = proc.wait()
    thread.join()
    stderr = "".join(stderr_lines)
    return stdout, stderr, parse_time(stderr), returncode


def tool_json(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise SystemExit("attack produced no JSON on stdout")
    return json.loads(text)


def full_output(evaluate: Path, circuit: Path, wires: int, message: str) -> str:
    result = run(
        [
            str(evaluate),
            "circuit",
            "evaluate",
            "-n",
            str(wires),
            "-s",
            str(circuit),
            "-x",
            message,
        ]
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    for line in result.stdout.splitlines():
        if line.startswith("Output:"):
            return line[line.rfind("0x") :].rstrip(")")
    raise SystemExit(f"evaluate produced no Output line:\n{result.stdout}")


def birthday_table_bytes(out_bits: int) -> tuple[float, int]:
    """Peak hashbrown size for a full birthday search on a `out_bits` digest.

    The map holds about sqrt(π/2)·2^{out_bits/2} digests. hashbrown then rounds
    the slot count up to a power of two at a 7/8 load factor. Each slot is one
    (u64 digest, u64 preimage) pair plus a control byte.
    """
    expected = math.sqrt(math.pi / 2.0) * 2 ** (out_bits / 2.0)
    need = int(expected / 0.875) + 1
    capacity = 1
    while capacity < need and capacity < (1 << 40):
        capacity *= 2
    return expected, capacity * 17


def digest_of(output_hex: str, out_bits: int) -> str:
    value = int(output_hex, 16) & ((1 << out_bits) - 1)
    return "0x" + format(value, f"0{(out_bits + 3) // 4}x")


def attack_record(
    name: str,
    timing: dict[str, float],
    threads: int,
    report: dict,
    circuit: Path,
    wires: int,
    out_bits: int,
    evaluate: Path,
) -> dict:
    x1 = report["x1_hex"]
    x2 = report["x2_hex"]
    output1 = full_output(evaluate, circuit, wires, x1)
    output2 = full_output(evaluate, circuit, wires, x2)
    digest1 = digest_of(output1, out_bits)
    digest2 = digest_of(output2, out_bits)
    if digest1 != digest2:
        raise SystemExit(f"{name} outputs disagree on the low {out_bits} bits: {digest1} vs {digest2}")
    if int(x1, 16) == int(x2, 16):
        raise SystemExit(f"{name} returned equal inputs")
    return {
        "cpu_seconds": timing["cpu_seconds"],
        "time": {"real": timing["real"], "user": timing["user"], "sys": timing["sys"]},
        "threads": threads,
        "x1_hex": x1,
        "x2_hex": x2,
        "output1_hex": output1,
        "output2_hex": output2,
        "digest_hex": digest1,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path, help="Hexadecimal G57 gate list")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the mpmct twin, progress log, and report.json",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--attacks",
        default="birthday,rho,sat",
        help="Comma-separated subset of birthday,rho,sat",
    )
    parser.add_argument("--progress-conflicts", type=int, default=20_000)
    parser.add_argument(
        "--progress-secs",
        type=float,
        default=10.0,
        help="Seconds between rho progress lines (0 disables them)",
    )
    args = parser.parse_args()

    attacks = [part.strip() for part in args.attacks.split(",") if part.strip()]
    unknown = set(attacks) - {"birthday", "rho", "sat"}
    if unknown or not attacks:
        raise SystemExit(f"unknown attacks: {sorted(unknown)}")

    circuit = args.circuit if args.circuit.is_absolute() else ROOT / args.circuit
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gates, wires = load_hex_gates(circuit)
    if wires % 3 != 0:
        raise SystemExit(f"{wires} wires is not divisible by 3; this script expects a 3λ circuit")
    lam = wires // 3
    if lam < 1:
        raise SystemExit(f"λ={lam} is not positive")
    in_bits = 2 * lam
    # Birthday and rho hash into a u64, so a digest wider than 64 bits is
    # SAT-only. Rho searches min(64, 2λ) message bits when it does run.
    # Birthday's tool can only accept a message of at most 64 bits, and a
    # 64-bit digest still needs a 2^{λ/2}-entry table.
    skipped_wide: list[str] = []
    if lam > 64:
        for name in ("birthday", "rho"):
            if name in attacks:
                attacks.remove(name)
                skipped_wide.append(name)
        if not attacks:
            raise SystemExit(
                f"λ={lam} is outside the birthday and rho tools' 1..=64 output range"
            )

    encoder = out_dir / "collision_to_cnf"
    bins = ensure_tools(encoder)
    mpmct = out_dir / f"c{wires}.mpmct1"
    write_mpmct(mpmct, gates, wires)
    print(
        f"[run] circuit={circuit} gates={len(gates)} wires={wires} λ={lam} "
        f"message={in_bits} digest={lam}",
        flush=True,
    )

    samples = int(min(250_000_000, max(100_000, 40 * (2 ** (lam / 2.0)))))
    dp_bits = max(4, min(lam // 3, lam - 1))
    report: dict = {
        "circuit": str(circuit),
        "wires": wires,
        "gates": len(gates),
        "lambda": lam,
        "in_bits": in_bits,
        "out_bits": lam,
        "pad_bits": lam,
        "seed": args.seed,
        "cpu_time": "/usr/bin/time -p user+sys, summed across threads",
        "attacks": {},
    }
    for name in skipped_wide:
        print(
            f"[run] {name} skipped: digest is {lam} bits and that tool accepts 1..=64",
            flush=True,
        )
        report["attacks"][name] = {
            "skipped": True,
            "reason": "tool accepts digests of 1..=64 bits only",
        }

    if "birthday" in attacks:
        expected, table_bytes = birthday_table_bytes(lam)
        table_gib = table_bytes / 2**30
        # A full λ=64 table is ~150 GiB. Even the 250M-sample cap is ~9 GiB,
        # which is too much on a 32 GiB machine that is already compressing.
        if table_gib > 4 or in_bits > 64 and table_gib > 1:
            print(
                f"[run] birthday skipped: ~{expected:.3e} samples, hash table ~{table_gib:.0f} GiB",
                flush=True,
            )
            report["attacks"]["birthday"] = {
                "skipped": True,
                "reason": "hash table would exceed available memory",
                "expected_samples": expected,
                "table_bytes": table_bytes,
            }
        else:
            search_in_bits = min(64, in_bits)
            stdout, stderr, timing, code = run_timed(
                [
                    str(bins / "birthday_collision"),
                    str(circuit),
                    "--pad",
                    str(lam),
                    "--in-bits",
                    str(search_in_bits),
                    "--out-bits",
                    str(lam),
                    "--samples",
                    str(samples),
                    "--seed",
                    str(args.seed),
                    "--out",
                    str(out_dir / "birthday.json"),
                ]
            )
            (out_dir / "birthday.stderr").write_text(stderr)
            if code != 0:
                raise SystemExit(f"birthday failed ({code})\n{stderr[-800:]}")
            found = tool_json(stdout)
            report["attacks"]["birthday"] = attack_record(
                "birthday", timing, 1, found, circuit, wires, lam, bins / "local_mixing_bin"
            )
            print(
                f"[run] birthday cpu={timing['cpu_seconds']:.6f}s "
                f"x1={found['x1_hex']} x2={found['x2_hex']}",
                flush=True,
            )

    if "rho" in attacks:
        stdout, stderr, timing, code = run_timed(
            [
                str(bins / "rho_collision"),
                str(circuit),
                "--pad",
                str(lam),
                "--out-bits",
                str(lam),
                "--dp-bits",
                str(dp_bits),
                "--workers",
                str(args.workers),
                "--seed",
                str(args.seed),
                "--progress-secs",
                str(args.progress_secs),
                "--progress-log",
                str(out_dir / "rho-progress.log"),
                "--out",
                str(out_dir / "rho.json"),
            ]
        )
        (out_dir / "rho.stderr").write_text(stderr)
        if code != 0:
            raise SystemExit(f"rho failed ({code})\n{stderr[-800:]}")
        found = tool_json(stdout)
        report["attacks"]["rho"] = attack_record(
            "rho", timing, args.workers, found, circuit, wires, lam, bins / "local_mixing_bin"
        )
        print(
            f"[run] rho cpu={timing['cpu_seconds']:.6f}s threads={args.workers} "
            f"x1={found['x1_hex']} x2={found['x2_hex']}",
            flush=True,
        )

    if "sat" in attacks:
        progress = out_dir / "sat-progress.log"
        stdout, stderr, timing, code = run_timed(
            [
                sys.executable,
                str(ROOT / "security_tests" / "collision" / "sat_once.py"),
                str(mpmct),
                str(encoder),
                str(lam),
                "--progress-conflicts",
                str(args.progress_conflicts),
                "--progress-log",
                str(progress),
            ]
        )
        (out_dir / "sat.stderr").write_text(stderr)
        (out_dir / "sat.stdout").write_text(stdout)
        if code != 0:
            raise SystemExit(f"sat failed ({code})\n{stderr[-800:]}\n{stdout[-800:]}")
        found = tool_json(stdout)
        if not found.get("ok"):
            raise SystemExit(f"sat did not verify: {found}")
        witness = found["report"]
        # Prefer the rust evaluator's full state, same as birthday and rho.
        found_for_eval = {
            "x1_hex": witness["x1_hex"],
            "x2_hex": witness["x2_hex"],
            "digest_hex": witness["digest_hex"],
            "sat": found,
        }
        record = attack_record(
            "sat", timing, 1, found_for_eval, circuit, wires, lam, bins / "local_mixing_bin"
        )
        report["attacks"]["sat"] = record
        print(
            f"[run] sat cpu={timing['cpu_seconds']:.6f}s "
            f"x1={witness['x1_hex']} x2={witness['x2_hex']}",
            flush=True,
        )

    destination = out_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[run] wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
