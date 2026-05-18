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

counts = pivot.values.astype(float)
log_data = np.where(counts > 0, np.log10(counts), np.nan)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

im = ax.imshow(log_data, aspect="auto", origin="lower", cmap="viridis",
               vmin=0, vmax=np.nanmax(log_data))

for i in range(len(after_vals)):
    for j in range(len(before_vals)):
        c = counts[i, j]
        if c > 0:
            ax.text(j, i, f"{int(c):,}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")

ax.set_xticks(range(len(before_vals)))
ax.set_xticklabels(before_vals, color="white")
ax.set_yticks(range(len(after_vals)))
ax.set_yticklabels(after_vals, color="white")

ax.set_xlabel("gates before", color="white", labelpad=8)
ax.set_ylabel("gates after", color="white", labelpad=8)

ax.tick_params(colors="white", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("log count", color="white", rotation=270, labelpad=15)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
print(f"Saved to {args.out}")
