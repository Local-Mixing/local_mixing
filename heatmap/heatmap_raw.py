import argparse
import numpy as np
import local_mixing as heatmap_rust 
import matplotlib.pyplot as plt
import os

def plot_heatmap(results, save_path, xlabel, ylabel, vmin=0.0, vmax=1.0):
    points = np.array(results, dtype=float)
    x, y, values = points[:, 0], points[:, 1], points[:, 2]

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    x_indices = {val: idx for idx, val in enumerate(x_unique)}
    y_indices = {val: idx for idx, val in enumerate(y_unique)}

    heatmap = np.full((len(y_unique), len(x_unique)), np.nan)
    for xi, yi, v in zip(x, y, values):
        heatmap[y_indices[yi], x_indices[xi]] = v

    plt.imshow(
        heatmap,
        interpolation="nearest",
        cmap="RdYlGn",
        aspect="auto",
        origin="lower",
        extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Average Hamming Distance")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(
        0.98, 0.02,
        f"Mean = {np.nanmean(values):.3f}",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- Call Rust and plot ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate circuit heatmap using Rust backend")
    parser.add_argument("--n", type=int, required=True, help="Number of wires")
    parser.add_argument("--i", type=int, required=True, help="Number of input samples")
    flag = False
    parser.add_argument("--x", type=str, required=True, help="Label for X-axis")
    parser.add_argument("--y", type=str, required=True, help="Label for Y-axis")
    output = "./heatmap.png"

    args = parser.parse_args()

    results = heatmap_rust.heatmap(args.n, args.i, flag)

    plot_heatmap(results, output, xlabel=args.x, ylabel=args.y)
    print(f" Heatmap saved to {output}")