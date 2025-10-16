import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_heatmap(data, save_path):
    plt.clf()
    points = np.array(data)
    x, y, values = points[:, 0], points[:, 1], points[:, 2]

    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        std = 1

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_indices = {val: idx for idx, val in enumerate(x_unique)}
    y_indices = {val: idx for idx, val in enumerate(y_unique)}

    heatmap = np.full((len(y_unique), len(x_unique)), np.nan)
    for xi, yi, z in zip(x, y, values):
        heatmap[y_indices[yi], x_indices[xi]] = z

    plt.imshow(
        heatmap,
        cmap='viridis',
        aspect='auto',
        origin='lower',
        extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
    )
    plt.colorbar(label='Absolute overlap')
    plt.xlabel('Circuit 1 gate index')
    plt.ylabel('Circuit 2 gate index')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    save_path = "heatmap.png"  # output file
    json_path = "heatmap.json"  # always read this file

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        plot_heatmap(data['results'], save_path)
        print(f"Heatmap saved to {save_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file {json_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file {json_path}")
    except KeyError:
        print("Error: JSON file does not contain 'results' key")
