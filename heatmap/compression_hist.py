import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="compression_histogram.csv")
parser.add_argument("--out", default="compression_histogram.png")
args = parser.parse_args()

df = pd.read_csv(args.csv)

before_vals = sorted(df["before"].unique())
after_vals = sorted(df["after"].unique())

pivot = df.pivot_table(index="before", columns="after", values="count", aggfunc="sum", fill_value=0)
pivot = pivot.reindex(index=before_vals, columns=after_vals, fill_value=0)

colors = plt.cm.tab10(np.linspace(0, 1, len(after_vals)))

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(before_vals))
width = 0.8 / len(after_vals)

for i, after in enumerate(after_vals):
    counts = pivot[after].values if after in pivot.columns else np.zeros(len(before_vals))
    ax.bar(x + i * width, counts, width, label=f"→ {after}", color=colors[i])

ax.set_xlabel("Gates before compression")
ax.set_ylabel("Count")
ax.set_title("Compression events by subcircuit size")
ax.set_xticks(x + width * (len(after_vals) - 1) / 2)
ax.set_xticklabels(before_vals)
ax.legend(title="Gates after")
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved to {args.out}")
