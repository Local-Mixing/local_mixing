import subprocess
import numpy as np

WIRES = 128
SEQ_LEN = 1 << 23
PAR = 150

TEST_PREFIX = f"cargo run --quiet --release --bin block_cipher -- -n {WIRES} -r {PAR} -l {SEQ_LEN} -f"

test_lens = np.geomspace(32, 4096, 32).astype(int)

subprocess.run("mkdir -p out", shell=True)

ckt = 'balanced'

for mode in ('ctr', 'ofb'):
    # for ckt in ('balanced', 'random'):
    for m in test_lens:
        print(mode, ckt, m)
        subprocess.run(TEST_PREFIX + f" -o {mode} -c {ckt} -m {m} >> out/{mode}.{ckt}.txt", shell = True)