import argparse
import numpy as np
import local_mixing as heatmap_rust
import matplotlib.pyplot as plt
import os
from pathlib import Path

def _downsample(grid, max_dim=4000):
    """Block-mean a 2D grid down so neither dimension exceeds max_dim.

    A multi-hundred-thousand-pixel imshow is both pointless (it gets resampled to
    a few thousand pixels in the PNG) and a memory bomb (matplotlib pushes the full
    array through norm+colormap). Binning first keeps plotting cheap and bounded.
    """
    h, w = grid.shape
    fy = max(1, h // max_dim)
    fx = max(1, w // max_dim)
    if fy == 1 and fx == 1:
        return grid
    h2 = (h // fy) * fy
    w2 = (w // fx) * fx
    blocks = grid[:h2, :w2].reshape(h2 // fy, fy, w2 // fx, fx)
    return np.nanmean(blocks, axis=(1, 3))

def plot_heatmap_raw(results, save_path, xlabel, ylabel, vmin=0.0, vmax=1.0):
    points = np.asarray(results, dtype=float)  # no copy when already float64
    x, y, values = points[:, 0], points[:, 1], points[:, 2]

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)

    # Scatter values into the (y, x) grid without a Python loop.
    if values.size == nx * ny:
        # Backend emits a complete row-major grid (x outer, y inner) -> reshape + transpose.
        heatmap = np.ascontiguousarray(values).reshape(nx, ny).T
    else:
        xi = np.searchsorted(x_unique, x)
        yi = np.searchsorted(y_unique, y)
        heatmap = np.full((ny, nx), np.nan)
        heatmap[yi, xi] = values

    heatmap = _downsample(heatmap)
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

def plot_heatmap_std(results, save_path, xlabel="X-axis", ylabel="Y-axis", vmin=-3, vmax=3):
    plt.clf()
    points = np.asarray(results, dtype=float)
    x, y, values = points[:, 0], points[:, 1], points[:, 2]

    # Compute z-scores
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        std = 1
    z_values = (values - mean) / std

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)

    # Scatter z-scores into the (y, x) grid without a Python loop.
    if z_values.size == nx * ny:
        heatmap = np.ascontiguousarray(z_values).reshape(nx, ny).T
    else:
        xi = np.searchsorted(x_unique, x)
        yi = np.searchsorted(y_unique, y)
        heatmap = np.full((ny, nx), np.nan)
        heatmap[yi, xi] = z_values

    heatmap = _downsample(heatmap)
    plt.imshow(
        heatmap,
        interpolation='nearest',
        cmap="Spectral_r",
        aspect='auto',
        origin='lower',
        extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label='Standard deviations from mean')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.text(
        0.98, 0.02,
        f"Std Dev = {np.std(values):.3f}",
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        fontsize=9,
        color='white',
        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
    )
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def count_semicolons(path):
    """Counts semicolons in a circuit file (for circuit length)."""
    with open(path, "r") as f:
        text = f.read()
    return text.count(";")  # number of gates

# --- Call Rust and plot ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate circuit heatmap using Rust backend")
    parser.add_argument("--n", type=int, required=True, help="Number of wires")
    parser.add_argument("--i", type=int, required=True, help="Number of input samples")
    parser.add_argument("--x", type=str, required=True, help="Label for X-axis")
    parser.add_argument("--y", type=str, required=True, help="Label for Y-axis")
    parser.add_argument("--pieces", action="store_true", help="Break heatmap into pieces if too large")
    parser.add_argument("--c1", type=str, required=False, help="Path to first circuit file")
    parser.add_argument("--c2", type=str, required=False, help="Path to second circuit file")
    parser.add_argument("--chunk", type=int, default=10_000, help="Size of each chunk (default 10000)")
    parser.add_argument("--path", type=str, default="./heatmap.png", help="Path to the heatmap generation")
    parser.add_argument("--corner", action="store_true", help="Only compute the bottom corner (first 5000 gates of both circuits); supports --incremental")
    parser.add_argument("--canonless", action="store_true", help="Don't canonicalize before heatmap")
    parser.add_argument("--small", action="store_true", help="Only check small inputs")
    parser.add_argument("--mini", action="store_true", help="Check with mini chunks inputs")
    parser.add_argument("--fix", type=int, default=0, help="Number of fixed bits in each random input")
    parser.add_argument("--hw", action="store_true", help="Use hamming weight difference mode")
    parser.add_argument("--incremental", action="store_true", help="First input random, then x0+1, x0+2, ... instead of random each iteration")
    parser.add_argument("--x0", type=int, default=None, help="Incremental mode: starting input x0 (default: random)")
    parser.add_argument("--std", action="store_true", help="Use standard deviation for heatmaps")
    parser.add_argument("--enhance", type=float, nargs="?", const=0.05, default=None,
                        help="Tighten color scale around 0.5 by ENHANCE (e.g. --enhance 0.1 → [0.4, 0.6]); default window is 0.05")
    args = parser.parse_args()

    if args.enhance is not None:
        vmin, vmax = 0.5 - args.enhance, 0.5 + args.enhance
    else:
        vmin, vmax = 0.0, 1.0

    flag = False

    if args.pieces or args.mini:
        if not args.c1 or not args.c2:
            raise ValueError("--c1 and --c2 are required when --pieces is used")

        # Determine circuit lengths from files
        c1_len = count_semicolons(args.c1)
        c2_len = count_semicolons(args.c2)
        print(f"Circuit lengths: c1={c1_len}, c2={c2_len}")

        # Compute slices
        chunk = args.chunk
        for x_start in range(0, c1_len, chunk):
            x_end = min(x_start + chunk - 1, c1_len - 1)
            for y_start in range(0, c2_len, chunk):
                y_end = min(y_start + chunk - 1, c2_len - 1)

                print(f"Computing slice x[{x_start}:{x_end}], y[{y_start}:{y_end}]...")
                if args.pieces:
                    results = heatmap_rust.heatmap_slice(
                        args.n, args.i, flag, x_start, x_end, y_start, y_end, args.c1, args.c2, args.fix, args.hw
                    )
                else:
                    results = heatmap_rust.heatmap_mini_slice(
                        args.n, args.i, flag, x_start, x_end, y_start, y_end, args.c1, args.c2, args.fix
                    )

                output_dir = args.path
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, f"heatmap_x{x_start}-{x_end}_y{y_start}-{y_end}.png"
                )

                if args.std:
                    plot_heatmap_std(results, output_path, xlabel=args.x, ylabel=args.y)
                else:
                    plot_heatmap_raw(results, output_path, xlabel=args.x, ylabel=args.y, vmin=vmin, vmax=vmax)
                print(f"Saved {output_path}")

    elif args.corner:
        mode = "incremental " if args.incremental else ""
        print(f"Generating {mode}corner heatmap (first 5000 gates of both circuits)...")
        results = heatmap_rust.heatmap_corner(args.n, args.i, flag, args.c1, args.c2, args.fix, args.hw, args.incremental, args.x0)
        output = args.path
        if args.std:
            plot_heatmap_std(results, output, xlabel=args.x, ylabel=args.y)
        else:
            plot_heatmap_raw(results, output, xlabel=args.x, ylabel=args.y, vmin=vmin, vmax=vmax)
        print(f"Heatmap saved to {output}")

    elif args.small:
        print("Generating full heatmap...")
        results = heatmap_rust.heatmap_small(args.n, flag, args.c1, args.c2, not args.canonless)
        output = args.path
        if args.std:
            plot_heatmap_std(results, output, xlabel=args.x, ylabel=args.y)
        else:
            plot_heatmap_raw(results, output, xlabel=args.x, ylabel=args.y, vmin=vmin, vmax=vmax)
        print(f"Heatmap saved to {output}")

    else:
        print("Generating full heatmap...")
        if args.incremental:
            results = heatmap_rust.heatmap_incremental(args.n, args.i, flag, args.c1, args.c2, not args.canonless, args.fix, args.hw, args.x0)
        else:
            results = heatmap_rust.heatmap(args.n, args.i, flag, args.c1, args.c2, not args.canonless, args.fix, args.hw)
        output = args.path
        if args.std:
            plot_heatmap_std(results, output, xlabel=args.x, ylabel=args.y)
        else:
            plot_heatmap_raw(results, output, xlabel=args.x, ylabel=args.y, vmin=vmin, vmax=vmax)
        print(f"Heatmap saved to {output}")