import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="compression_histogram.csv")
parser.add_argument("--out", default="compression_heatmap.png")
args = parser.parse_args()

df = pd.read_csv(args.csv)

before_vals = sorted(df["before"].unique())
after_vals = sorted(df["after"].unique())

pivot = df.pivot_table(index="after", columns="before", values="count", aggfunc="sum", fill_value=0)
pivot = pivot.reindex(index=after_vals, columns=before_vals, fill_value=0)

log_data = np.log10(pivot.values.astype(float) + 1)

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(log_data, aspect="auto", origin="lower", cmap="viridis")

ax.set_xticks(range(len(before_vals)))
ax.set_xticklabels(before_vals)
ax.set_yticks(range(len(after_vals)))
ax.set_yticklabels(after_vals)

ax.set_xlabel("Gates before (input to LMDB)")
ax.set_ylabel("Gates after (output from LMDB)")
ax.set_title("LMDB compression heatmap (log10 count)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("log10(count + 1)")

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved to {args.out}")
