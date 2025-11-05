import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys 

def plot_heatmap(results, save_path, xlabel="X-axis", ylabel="Y-axis", vmin=-0, vmax=1):
    plt.clf()
    points = np.array(results)
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
        interpolation='nearest',
        cmap="RdYlGn",
        aspect='auto',
        origin='lower',
        extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label='Average hamming diff')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.text(
        0.98, 0.02,
        f"Mean = {np.mean(values):.3f}",
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        fontsize=9,
        color='white',
        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
    )
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# New CLI mode to read JSON from stdin
if __name__ == "__main__":
    if not sys.stdin.isatty():  # data piped in
        try:
            data = json.load(sys.stdin)

            xlabel = data.get('xlabel', 'X-axis')
            ylabel = data.get('ylabel', 'Y-axis')

            plot_heatmap(
                data['results'],
                "heatmap.png",
                xlabel=xlabel,
                ylabel=ylabel
            )
            print("Heatmap saved to heatmap.png")
        except Exception as e:
            print("Error:", e)
