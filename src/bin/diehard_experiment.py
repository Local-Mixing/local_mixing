import subprocess
import numpy as np

WIRES = 128
SEQ_LEN = 1 << 23
PAR = 150

TEST_PREFIX = f"cargo run --quiet --release --bin block_cipher -- -r {PAR} -l {SEQ_LEN} -f"

test_lens = sorted(set((np.round(np.geomspace(128, 4096, 16) / 128) * 128).astype(int).tolist()))

print(f"{test_lens = }")

subprocess.run("mkdir -p out", shell=True)

ckt = 'fanout-two'
mode = 'ofb'

# for mode in ('ctr', 'ofb'):
    # for ckt in ('balanced', 'random'):

n = 64
# for n in (32, 64, 128, 256):
for m in test_lens:
    print(mode, ckt, m)
    subprocess.run(TEST_PREFIX + f" -o {mode} -c {ckt} -m {m} -n {n} >> out/{mode}.{ckt}.n{n}.txt", shell = True)