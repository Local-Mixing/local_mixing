import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap

def plot_heatmap(data, save_path, xlabel="X-axis", ylabel="Y-axis", vmin=-3, vmax=3):
    plt.clf()
    points = np.array(data)
    x, y, values = points[:, 0], points[:, 1], points[:, 2]

    # Compute z-scores
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        std = 1
    z_values = (values - mean) / std

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_indices = {val: idx for idx, val in enumerate(x_unique)}
    y_indices = {val: idx for idx, val in enumerate(y_unique)}

    heatmap = np.full((len(y_unique), len(x_unique)), np.nan)
    for xi, yi, z in zip(x, y, z_values):
        heatmap[y_indices[yi], x_indices[xi]] = z

    plt.imshow(
        heatmap,
        cmap='RdBu_r',  # red = high, blue = low
        aspect='auto',
        origin='lower',
        extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label='Standard deviations from mean')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmap from JSON results.")
    parser.add_argument("--json", default="heatmap.json", help="Path to JSON file (default: heatmap.json)")
    parser.add_argument("--output", default="heatmap.png", help="Output PNG file (default: heatmap.png)")
    parser.add_argument("--xlabel", default="Circuit 1 gate index", help="Label for X axis")
    parser.add_argument("--ylabel", default="Circuit 2 gate index", help="Label for Y axis")
    args = parser.parse_args()

    try:
        with open(args.json, 'r') as f:
            data = json.load(f)
        plot_heatmap(
            data['results'], 
            args.output, 
            xlabel=args.xlabel, 
            ylabel=args.ylabel,
            vmin=-3,
            vmax=3
        )
        print(f"Heatmap saved to {args.output}")
    except FileNotFoundError:
        print(f"Error: Could not find file {args.json}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file {args.json}")
    except KeyError:
        print("Error: JSON file does not contain 'results' key")
