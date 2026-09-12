#!/usr/bin/env python3
"""Rebuild public and private red-team challenge indexes without logging secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(os.environ.get("SECURITY_REPO_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("SECURITY_ARTIFACT_ROOT", REPO_ROOT / "red_team_tests"))


def atomic_write(path: Path, text: str, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(text)
        os.chmod(path, mode)
    except BaseException:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    root = parser.parse_args().artifact_root.resolve()
    circuits = (
        ("circuit_01", "public_c1", "c1_final.txt"),
        ("circuit_03", "public_c2", "c2_final.txt"),
        ("circuit_02", "public_c3", "final.txt"),
    )
    public_lines = [
        "# Safe-to-share GSS preimage challenges",
        "# circuit_id\tcircuit_file\tchallenge_file\ttarget_logical_output_hex",
    ]
    private_lines = [
        "# PRIVATE: answers undermine the public challenges",
        "# circuit_id\tlogical_input_hex\tlogical_output_hex\tanswer_record",
    ]
    for circuit_id, public_id, final_name in circuits:
        challenge_path = root / "circuits" / "challenges" / public_id / "challenge.json"
        answer_path = root / "circuits" / circuit_id / "private" / "challenge_answer.json"
        challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        if challenge["circuit_id"] != circuit_id or answer["circuit_id"] != circuit_id:
            raise ValueError(f"circuit-id mismatch for {circuit_id}")
        if challenge["target_logical_output_hex"] != answer["logical_output_hex"]:
            raise ValueError(f"challenge/answer target mismatch for {circuit_id}")
        public_lines.append(
            "\t".join(
                (
                    circuit_id,
                    f"circuits/challenges/{public_id}/{final_name}",
                    f"circuits/challenges/{public_id}/challenge.json",
                    challenge["target_logical_output_hex"],
                )
            )
        )
        private_lines.append(
            "\t".join(
                (
                    circuit_id,
                    answer["logical_input_hex"],
                    answer["logical_output_hex"],
                    f"circuits/{circuit_id}/private/challenge_answer.json",
                )
            )
        )
    atomic_write(root / "red_team_challenges.txt", "\n".join(public_lines) + "\n", 0o644)
    atomic_write(root / "red_team_answers.txt", "\n".join(private_lines) + "\n", 0o600)
    print("rebuilt 3 challenge indexes; private values were not logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
