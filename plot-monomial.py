#!/usr/bin/env python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def line_to_data(line):
    return list(map(int, line.removeprefix("[").removesuffix("]\n").split(", ")))

w = []

pctile = 5

for fn in sys.argv[1:]:
    n = int(fn.removeprefix("monomial-").removesuffix(".txt"))
    w.append(n)
    with open(fn) as f:
        print(f"Loading {n}-wire data")

        dd = []
        p10s = []
        p90s = []

        for line in f:
            d = list(map(lambda x: math.log2(x),  line_to_data(line)))
            dd.append(d)

        max_len = max(len(d) for d in dd)
        for idx in range(max_len):
            values = [lst[idx] for lst in dd if idx < len(lst)]
            if values:
                p10 = np.percentile(values, pctile)
                p10s.append(p10)
                p90 = np.percentile(values, 100 - pctile)
                p90s.append(p90)

        # print(n, p10s, p90s)

        r = plt.fill_between(np.arange(max_len) / n, p10s, p90s, alpha=0.7, label=f"{n} wires")
        plt.hlines(n, 0, 10, color='gray', linestyle=':', zorder=-1, linewidth=1)

w = np.array(w)
z = 2 * np.log(w)
print(z)
plt.scatter(z, w, color='black', s=10, label='$2 \\ln n$')

plt.xlabel("Aspect ratio $m/n$")
plt.ylabel("log # monomials")
plt.xlim(0, 10)
plt.legend(loc='lower right')
plt.title(f"p{pctile}-p{100-pctile} log u.b. # monomials")
plt.savefig("mono-plot-out.png", dpi=600)
