import subprocess
import numpy as np

WIRES = 128
SEQ_LEN = 1 << 23
PAR = 150

TEST_PREFIX = f"cargo run --quiet --release --bin block_cipher -- -r {PAR} -l {SEQ_LEN} -f"

test_lens = sorted(set((np.round(np.geomspace(128, 4096, 16) / 128) * 128).astype(int).tolist()))

print(f"{test_lens = }")

subprocess.run("mkdir -p out", shell=True)

ckt_types = ['fanout-two', 'balanced', 'random', 'aes']
mode = 'ofb'

# cmr22 lengths with density = 5
#
# 27 => 405
# 81 => 2160
# 243 => 10125

# for mode in ('ctr', 'ofb'):
    # for ckt in ('balanced', 'random'):

n = 128
for m in test_lens:
    for ckt in ckt_types:
        print(mode, ckt, m)
        subprocess.run(TEST_PREFIX + f" -o {mode} -c {ckt} -m {m} -n {n} >> out/{mode}.{ckt}.n{n}.txt", shell = True)

# density 5
cmr_lens = {
    27: 405,
    81: 2160,
    243: 10125,
}

for n, m in cmr_lens.items():
    for ckt in ckt_types + ['cmr22']:
        print("[cmr-cmp]", mode, ckt, m)
        subprocess.run(TEST_PREFIX + f" -o {mode} -c {ckt} -m {m} -n {n} >> out/{mode}.{ckt}.n{n}-cmr.txt", shell = True)