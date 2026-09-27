"""Remote DB backend for chain.compile_mono(backend='db').

`db_synth_batch(pending, rng, host=None, w=None)` is the batch synth hook that
compile_mono calls once, with every deferred unit (chain._Deferred: .raw = the
g57 word to match on .block, .M = the accreted mask that was inverted, .block =
the w block wires).  It:

  1. writes each unit's .raw word (as-is, on its actual .block wire ids -- the
     remote tool canonicalizes internally) to a temp file, one unit per line;
  2. ONE scp up to $H:~/db_synth_src/batch_in.txt;
  3. ONE ssh run of the enhanced db_unit_synth with --candidates K, which emits
     up to K distinct short block-local realizations per unit
     ("idx \t cand1 | cand2 | ...");
  4. ONE scp back of batch_out.txt;
  5. per unit, PICK the first returned candidate that is chain._bornbare_free
     against (.M, .block); else the shortest candidate; else (tool solved
     nothing) fall back to .raw.

Returns the chosen words in pending order.  compile_mono re-verifies each chosen
word equals .raw as a permutation on .block, so a wrong word asserts loudly.

Host is sourced from the local memory note (never printed); see _default_host.
"""
import os
import re
import subprocess
import tempfile

import chain

_MEM = os.path.expanduser(
    "~/.claude/projects/-Users-rancanetti-Documents-local-mixing/memory/server-242.md"
)
_REMOTE_DIR = "~/db_synth_src"
_FROZEN = "$HOME/frozen_m1_m11"
_BIN = "./target/release/db_unit_synth"


def _default_host():
    """user@ip of .242, read from the local memory note.  Never logged."""
    with open(_MEM) as f:
        m = re.search(r"[a-z]+@[0-9.]+", f.read())
    if not m:
        raise RuntimeError("could not source .242 host from memory note")
    return m.group(0)


def _fmt_word_csv(word):
    """g57 word -> 'a,x,y a,x,y ...' on its actual wire ids (tool canonicalizes)."""
    return " ".join(f"{a},{x},{y}" for (a, x, y) in word)


def _parse_cand(tok):
    """'a,x,y a,x,y ...' -> [(a,x,y), ...]."""
    out = []
    for g in tok.split():
        p = g.split(",")
        if len(p) != 3:
            continue
        out.append((int(p[0]), int(p[1]), int(p[2])))
    return out


def db_synth_batch(pending, rng, host=None, w=None, rs=2, k=32,
                   scratch=None, keep=False):
    """Two-pass DB synth hook.  `w` is accepted for signature parity (block
    width is read from each unit's .block) and unused.  `rs`/`k` map to the
    tool's --rs / --candidates.

    rs default is 2 (cheap+exact): with acc>0 accretion, refresh units carry
    long accreted masks whose 3-wire permutation geodesic reaches 8, past rs=1's
    reach (rs=1 solves only ~75%, leaving ~35% of units with 0 block-local
    candidates -> raw fallback).  rs=2 solves ~95% with mean ~7 candidates/unit;
    rs=3 reaches ~100% (mean ~24 cands/unit) at ~4.5x cpu but is no longer
    ball-exact."""
    host = host or _default_host()
    scratch = scratch or (
        "/private/tmp/claude-501/-Users-rancanetti-Documents-local-mixing/"
        "6914ffc5-7369-49a7-a4db-5d39a30deb9d/scratchpad"
    )
    os.makedirs(scratch, exist_ok=True)
    in_path = os.path.join(scratch, "db_batch_in.txt")
    out_path = os.path.join(scratch, "db_batch_out.txt")

    # pass 1: write every unit's raw word, one per line, in pending order
    with open(in_path, "w") as f:
        for s in pending:
            f.write(_fmt_word_csv(s.raw) + "\n")

    # ONE scp up, ONE ssh run, ONE scp down
    subprocess.run(["scp", "-q", in_path, f"{host}:{_REMOTE_DIR}/batch_in.txt"],
                   check=True)
    remote_cmd = (
        f"cd {_REMOTE_DIR} && FROZEN_DB_DIR={_FROZEN} {_BIN} batch_in.txt "
        f"--rs {rs} --candidates {k} --out batch_out.txt"
    )
    subprocess.run(["ssh", host, remote_cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["scp", "-q", f"{host}:{_REMOTE_DIR}/batch_out.txt", out_path],
                   check=True)

    # parse: idx -> [candidate words] (sorted shortest-first by the tool)
    cands = {}
    with open(out_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            idx = int(parts[0])
            rest = parts[1] if len(parts) > 1 else ""
            words = []
            if rest.strip():
                for tok in rest.split(" | "):
                    tok = tok.strip()
                    if tok:
                        words.append(_parse_cand(tok))
            cands[idx] = words

    # pass 2: pick the first born-bare-free candidate per unit, in pending order
    out = []
    fallback_raw = 0
    fallback_shortest = 0
    bb_hits = 0
    for i, s in enumerate(pending):
        block = set(s.block)
        # defensive: only block-local candidates are usable (the tool already
        # filters ancilla-using realizations, but guard anyway)
        cs = [c for c in cands.get(i, []) if {w for g in c for w in g} <= block]
        if not cs:
            out.append(list(s.raw))          # tool solved nothing -> raw
            fallback_raw += 1
            continue
        chosen = None
        for c in cs:
            if chain._bornbare_free(c, s.M, s.block):
                chosen = c
                bb_hits += 1
                break
        if chosen is None:
            chosen = cs[0]                    # shortest (candidates are sorted)
            fallback_shortest += 1
        out.append(chosen)

    db_synth_batch.last_stats = {
        "units": len(pending),
        "bornbare_free_pick": bb_hits,
        "fallback_shortest": fallback_shortest,
        "fallback_raw": fallback_raw,
        "cand_counts": [len(cands.get(i, [])) for i in range(len(pending))],
        "raw_lens": [len(s.raw) for s in pending],
        "chosen_lens": [len(c) for c in out],
        "pending": list(pending),   # local handle for downstream measurement
    }
    if not keep:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
    return out
