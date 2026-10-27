import subprocess
import sys

import numpy as np

WIRES = 128
SEQ_LEN = 1 << 23
PAR = 150

TEST_PREFIX = f"cargo run --quiet --release --bin block_cipher -- -r {PAR} -l {SEQ_LEN}"

test_lens = np.geomspace(128, 4096, 16).astype(int).tolist()

print(f"{test_lens = }")

def run_test(mode, ckt, m, n, out):
    cmd = TEST_PREFIX + f" -o {mode} -c {ckt} -m {m} -n {n} -f {out}"
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(
            f"ERROR: {mode} {ckt} n={n} m={m} failed with exit code {result.returncode}",
            file=sys.stderr,
        )

subprocess.run("mkdir -p out", shell=True)

ckt_types = ['fanout-two', 'balanced', 'random', 'aes']
mode = 'ofb'

# for mode in ('ctr', 'ofb'):
    # for ckt in ('balanced', 'random'):

n = 128
for m in test_lens:
    for ckt in ckt_types:
        print(mode, ckt, m)
        run_test(mode, ckt, m, n, f"out/{mode}.{ckt}.n{n}.txt")

# Exact CMR lengths, matched across the other gate-count modes. Each width's
# step is round(log2(n)) * log3(n) * n / 3. The middle point is the lattice
# size nearest the geometric mean of the endpoints.
# 27 wires, step 135: 270, 945, 4050
# 81 wires, step 648: 648, 1296, 3888
# 243 wires, step 3240: 3240
cmr_sizes = {
    27: [270, 945, 4050],
    81: [648, 1296, 3888],
    243: [3240],
}
cmr_modes = ['fanout-two', 'balanced', 'random', 'cmr22']

for n, lengths in cmr_sizes.items():
    for m in lengths:
        for ckt in cmr_modes:
            print("[cmr-cmp]", mode, ckt, n, m)
            run_test(mode, ckt, m, n, f"out/{mode}.{ckt}.n{n}-cmr.txt")