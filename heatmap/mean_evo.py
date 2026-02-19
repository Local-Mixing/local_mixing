import argparse
import numpy as np
import local_mixing as heatmap_rust
import matplotlib.pyplot as plt
import os

def count_semicolons(path):
    with open(path, "r") as f:
        return f.read().count(";")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute chunk means and plot evolution")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--i", type=int, required=True)
    parser.add_argument("--c1", type=str, required=True)
    parser.add_argument("--c2", type=str, required=True)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--path", type=str, default="./means.png")
    parser.add_argument("--pieces", action="store_true", help="Break heatmap into pieces if too large")
    args = parser.parse_args()

    flag = False

    c1_len = count_semicolons(args.c1)
    c2_len = count_semicolons(args.c2)
    print(f"Circuit lengths: c1={c1_len}, c2={c2_len}")

    chunk = args.chunk
    x_start = 0
    x_end = c1_len - 1

    means = []
    chunk_ids = []

    chunk_id = 0
    fn = heatmap_rust.heatmap_mini_slice if args.pieces else heatmap_rust.heatmap_slice

    for y_start in range(0, c2_len, chunk):
        y_end = min(y_start + chunk - 1, c2_len - 1)

        print(f"Computing slice x[{x_start}:{x_end}], y[{y_start}:{y_end}]...")

        results = fn(
            args.n, args.i, flag,
            x_start, x_end,
            y_start, y_end,
            args.c1, args.c2
        )

        arr = np.array(results)

        if arr.size == 0:
            mean_val = np.nan
        else:
            mean_val = np.nanmean(arr[:, 2])

        means.append(mean_val)
        chunk_ids.append(chunk_id)

        print(f"Chunk {chunk_id} mean = {mean_val:.6f}")
        chunk_id += 1


    # ---- final dot graph ----
    plt.figure()
    plt.scatter(chunk_ids, means)
    plt.plot(chunk_ids, means)  # optional connecting line
    plt.xlabel("Chunk index")
    plt.ylabel("Mean value")
    plt.title("Mean per Chunk Evolution")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.path) or ".", exist_ok=True)
    plt.savefig(args.path, dpi=300)
    print(f"Saved plot → {args.path}")

