import subprocess
import numpy as np

WIRES = 32
SEQ_LEN = 1 << 23

TEST_PREFIX = f"cargo run --quiet --release --bin block_cipher -- -n {WIRES} -r 128 -l {SEQ_LEN} -f"

test_lens = np.geomspace(32, 1024, 16).astype(int)

subprocess.run("mkdir -p out", shell=True)

for mode in ('ctr', 'ofb'):
    for ckt in ('balanced', 'random'):
        for m in test_lens:
            print(mode, ckt, m)
            subprocess.run(TEST_PREFIX + f" -o {mode} -c {ckt} -m {m} >> out/{mode}.{ckt}.txt", shell = True)