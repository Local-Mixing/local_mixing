# `local_mixing_bin` commands

DB-backed (`./db` required at runtime): `sss`, `compress`, `equal`.
DB-free: `genran`, `shuffle`, `shoot`.

## `sss` — Shuffle-shoot-shuffle obfuscation + compression game

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--n` | `-n` | usize | yes | | Number of wires |
| `--m` | `-m` | usize | yes | | SAMFs per insertion |
| `--x` | `-x` | usize | yes | | Insert SAMFs every `x` gates |
| `--source` | `-s` | string | yes | | Source circuit file |
| `--rounds` | `-r` | usize | yes | | Number of rounds |
| `--destination` | `-d` | string | yes | | Output circuit file |
| `--intermediate` | `-i` | string | yes | | Intermediate circuit file |
| `--interleave` | | flag | no | | Use interleaving |
| `--gadgetize` | | flag | no | | Gadgetize at start (wires → 2n) |
| `--full-shuffle` | | flag | no | | Insert n SAMFs between every gate once before the loop |
| `--gates_ahead` | | usize | no | 2 | Replacement window size (2 = pair; >2 uses curated shard lookup) |
| `--type_attempts` | | usize | no | 1 | Distinct SAMF negation types to try per collision (max useful = 6) |
| `--rg-frequency` | | usize | no | 2 | SG gadgets between each RG gadget |
| `--egg` | | flag | no | | Use expansion game instead of simple shooting |
| `--shuffled` | | flag | no | | Use shuffled shooting game (SAMF-assisted curated DB compression) |
| `--single-end` | | flag | no | | Shuffled only: accumulate SAMFs across all rounds, single unsamf after the last round |

## `compress` — Run compression trials on a circuit file

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--source` | `-s` | string | yes | | Starting circuit file |
| `--destination` | `-d` | string | yes | | Output circuit file |
| `--wires` | `-n` | usize | yes | | Number of wires |
| `--seq` | | flag | no | | Enable seq mode |

## `genran` — Generate a random circuit

| Flag | Short | Type | Required | Description |
|------|-------|------|----------|-------------|
| `--destination` | `-d` | string | yes | Output circuit file |
| `--wires` | `-n` | usize | yes | Number of wires |
| `--gates` | `-m` | usize | yes | Number of gates |

## `shuffle` — Shuffle a circuit (insert SAMFs)

| Flag | Short | Type | Required | Description |
|------|-------|------|----------|-------------|
| `--n` | `-n` | usize | yes | Number of wires |
| `--source` | `-s` | string | yes | Source circuit file |
| `--iterations` | `-i` | usize | yes | Number of iterations |
| `--destination` | `-d` | string | yes | Output circuit file |
| `--knuth` | | flag | no | Use Knuth shuffle instead of simple |

## `shoot` — Shoot random gates through a circuit

| Flag | Short | Type | Required | Description |
|------|-------|------|----------|-------------|
| `--iterations` | `-i` | usize | yes | Number of iterations |
| `--source` | `-s` | string | yes | Source circuit file |
| `--destination` | `-d` | string | yes | Output circuit file |

## `equal` — Check if two circuits are functionally equivalent

| Flag | Short | Type | Required | Description |
|------|-------|------|----------|-------------|
| `--wires` | `-n` | usize | yes | Number of wires |
| `--iterations` | `-i` | usize | yes | Number of test iterations |
| `--circuit-a` | `-a` | string | yes | First circuit file |
| `--circuit-b` | `-b` | string | yes | Second circuit file |
