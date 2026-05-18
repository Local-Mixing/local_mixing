import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# data = {
#     (4, 2): 12,
#     (3, 2): 4,
#     (7, 3): 50,
#     (8, 3): 3,
#     (4, 6): 70937,
#     (6, 2): 1,
#     (6, 3): 229,
#     (4, 3): 185,
#     (6, 6): 5273448,
#     (10, 6): 175710,
#     (9, 5): 4905,
#     (7, 4): 16549,
#     (13, 6): 60,
#     (12, 6): 1160,
#     (4, 4): 3430,
#     (4, 5): 24056,
#     (3, 6): 2,
#     (9, 6): 1072950,
#     (5, 5): 193447,
#     (7, 5): 181410,
#     (8, 5): 43676,
#     (10, 4): 14,
#     (9, 4): 301,
#     (5, 4): 20680,
#     (7, 6): 6615084,
#     (11, 5): 7,
#     (5, 2): 6,
#     (5, 6): 1427495,
#     (5, 3): 390,
#     (6, 4): 31183,
#     (8, 4): 3524,
#     (10, 5): 269,
#     (8, 6): 3694106,
#     (3, 5): 33,
#     (6, 5): 322455,
#     (11, 6): 17553,
#     (3, 3): 12,
#     (3, 4): 50,
# }

data = {
    ((10, 4), (10, 6)): 8,
    ((10, 4), (11, 6)): 6,
    ((10, 5), (10, 5)): 32,
    ((10, 5), (10, 6)): 205,
    ((10, 6), (10, 6)): 88691,
    ((10, 6), (11, 6)): 84,
    ((11, 5), (11, 6)): 7,
    ((11, 6), (11, 6)): 8828,
    ((12, 6), (12, 6)): 580,
    ((13, 6), (13, 6)): 30,
    ((3, 2), (4, 4)): 3,
    ((3, 2), (4, 5)): 1,
    ((3, 2), (5, 6)): 5,
    ((3, 3), (4, 5)): 11,
    ((3, 3), (4, 6)): 4,
    ((3, 3), (5, 5)): 1,
    ((3, 3), (5, 6)): 2,
    ((3, 4), (4, 5)): 2,
    ((3, 4), (4, 6)): 44,
    ((3, 4), (5, 6)): 11,
    ((3, 5), (3, 5)): 2,
    ((3, 5), (4, 5)): 3,
    ((3, 5), (4, 6)): 31,
    ((3, 5), (5, 6)): 7,
    ((3, 6), (3, 6)): 1,
    ((4, 2), (4, 4)): 7,
    ((4, 2), (4, 5)): 9,
    ((4, 2), (5, 4)): 4,
    ((4, 2), (5, 5)): 1,
    ((4, 2), (6, 6)): 14,
    ((4, 3), (4, 3)): 1,
    ((4, 3), (4, 4)): 24,
    ((4, 3), (4, 5)): 129,
    ((4, 3), (4, 6)): 153,
    ((4, 3), (5, 5)): 59,
    ((4, 3), (5, 6)): 17,
    ((4, 3), (6, 5)): 1,
    ((4, 3), (6, 6)): 14,
    ((4, 4), (4, 4)): 80,
    ((4, 4), (4, 5)): 880,
    ((4, 4), (4, 6)): 1346,
    ((4, 4), (5, 4)): 11,
    ((4, 4), (5, 5)): 18,
    ((4, 4), (5, 6)): 2079,
    ((4, 4), (6, 6)): 139,
    ((4, 4), (7, 6)): 8,
    ((4, 5), (4, 5)): 615,
    ((4, 5), (4, 6)): 23470,
    ((4, 5), (5, 4)): 11,
    ((4, 5), (5, 5)): 45,
    ((4, 5), (5, 6)): 1040,
    ((4, 5), (6, 6)): 176,
    ((4, 5), (7, 6)): 5,
    ((4, 6), (4, 6)): 24589,
    ((4, 6), (5, 5)): 173,
    ((4, 6), (5, 6)): 2068,
    ((4, 6), (6, 5)): 1,
    ((5, 2), (5, 5)): 3,
    ((5, 2), (6, 6)): 2,
    ((5, 2), (7, 6)): 5,
    ((5, 3), (5, 4)): 7,
    ((5, 3), (5, 5)): 286,
    ((5, 3), (5, 6)): 360,
    ((5, 3), (6, 5)): 84,
    ((5, 3), (6, 6)): 25,
    ((5, 3), (7, 6)): 6,
    ((5, 4), (5, 4)): 186,
    ((5, 4), (5, 5)): 2153,
    ((5, 4), (5, 6)): 11122,
    ((5, 4), (6, 4)): 5,
    ((5, 4), (6, 5)): 14,
    ((5, 4), (6, 6)): 10094,
    ((5, 4), (7, 6)): 341,
    ((5, 4), (8, 6)): 5,
    ((5, 5), (5, 5)): 4870,
    ((5, 5), (5, 6)): 190732,
    ((5, 5), (6, 4)): 3,
    ((5, 5), (6, 5)): 44,
    ((5, 5), (6, 6)): 3132,
    ((5, 5), (7, 6)): 309,
    ((5, 5), (8, 6)): 4,
    ((5, 6), (5, 6)): 635420,
    ((5, 6), (6, 5)): 365,
    ((5, 6), (6, 6)): 20208,
    ((6, 2), (8, 6)): 1,
    ((6, 3), (6, 5)): 137,
    ((6, 3), (6, 6)): 227,
    ((6, 3), (7, 5)): 18,
    ((6, 3), (8, 6)): 1,
    ((6, 4), (6, 4)): 96,
    ((6, 4), (6, 5)): 1060,
    ((6, 4), (6, 6)): 18858,
    ((6, 4), (7, 4)): 3,
    ((6, 4), (7, 6)): 13148,
    ((6, 4), (8, 6)): 150,
    ((6, 4), (9, 6)): 1,
    ((6, 5), (6, 5)): 12467,
    ((6, 5), (6, 6)): 309235,
    ((6, 5), (7, 5)): 146,
    ((6, 5), (7, 6)): 2226,
    ((6, 5), (8, 6)): 124,
    ((6, 6), (6, 6)): 2523036,
    ((6, 6), (7, 5)): 150,
    ((6, 6), (7, 6)): 30414,
    ((7, 3), (7, 5)): 11,
    ((7, 3), (7, 6)): 51,
    ((7, 3), (8, 5)): 3,
    ((7, 4), (7, 4)): 17,
    ((7, 4), (7, 5)): 156,
    ((7, 4), (7, 6)): 10801,
    ((7, 4), (8, 6)): 6150,
    ((7, 4), (9, 6)): 19,
    ((7, 5), (7, 5)): 10369,
    ((7, 5), (7, 6)): 165873,
    ((7, 5), (8, 5)): 43,
    ((7, 5), (8, 6)): 498,
    ((7, 5), (9, 6)): 8,
    ((7, 6), (7, 6)): 3259010,
    ((7, 6), (8, 5)): 33,
    ((7, 6), (8, 6)): 17289,
    ((8, 3), (8, 6)): 3,
    ((8, 4), (10, 6)): 1,
    ((8, 4), (8, 5)): 7,
    ((8, 4), (8, 6)): 2490,
    ((8, 4), (9, 6)): 1200,
    ((8, 5), (8, 5)): 3507,
    ((8, 5), (8, 6)): 37516,
    ((8, 5), (9, 5)): 5,
    ((8, 5), (9, 6)): 25,
    ((8, 6), (8, 6)): 1850530,
    ((8, 6), (9, 6)): 6993,
    ((9, 4), (10, 6)): 105,
    ((9, 4), (9, 6)): 202,
    ((9, 5), (9, 5)): 500,
    ((9, 5), (9, 6)): 3965,
    ((9, 6), (10, 6)): 1152,
    ((9, 6), (9, 6)): 541129,
    (None, (10, 4)): 69,
    (None, (10, 5)): 152254,
    (None, (10, 6)): 167891212,
    (None, (11, 4)): 6,
    (None, (11, 5)): 16486,
    (None, (11, 6)): 33004295,
    (None, (12, 4)): 1,
    (None, (12, 5)): 1221,
    (None, (12, 6)): 4165169,
    (None, (13, 5)): 83,
    (None, (13, 6)): 356492,
    (None, (14, 5)): 6,
    (None, (14, 6)): 22750,
    (None, (15, 5)): 1,
    (None, (15, 6)): 1295,
    (None, (16, 6)): 83,
    (None, (17, 6)): 6,
    (None, (18, 6)): 1,
    (None, (3, 1)): 1,
    (None, (3, 5)): 150,
    (None, (3, 6)): 558,
    (None, (4, 3)): 1,
    (None, (4, 4)): 6,
    (None, (4, 5)): 33385,
    (None, (4, 6)): 892976,
    (None, (5, 3)): 6,
    (None, (5, 4)): 434,
    (None, (5, 5)): 794402,
    (None, (5, 6)): 42986775,
    (None, (6, 3)): 6,
    (None, (6, 4)): 2714,
    (None, (6, 5)): 3286015,
    (None, (6, 6)): 333547673,
    (None, (7, 3)): 9,
    (None, (7, 4)): 4177,
    (None, (7, 5)): 4531066,
    (None, (7, 6)): 836284991,
    (None, (8, 3)): 3,
    (None, (8, 4)): 2337,
    (None, (8, 5)): 2758138,
    (None, (8, 6)): 918705398,
    (None, (9, 3)): 1,
    (None, (9, 4)): 578,
    (None, (9, 5)): 862348,
    (None, (9, 6)): 520400366,
}

# Set to True to normalize counts per column (values become fractions ≤ 1).
# When enabled the heatmap shows log10(column_fraction) (negative up to 0).
NORMALIZE_PER_COLUMN = False

# Build the full set of valid tuples for axes.
# For each gate-count `m` (1..6), `n` ranges from 3 up to 3*m (inclusive).
# Vertical axis includes an extra `None` row; horizontal axis is only tuples.
tuples = [(n, m) for m in range(1, 7) for n in range(3, 3 * m + 1)]
# Sort tuples by product (area), then by n then m
sorted_tuples = sorted(tuples) #, key=lambda t: (t[0], t[0], t[1]))
xs = sorted_tuples[:]  # horizontal (no None)
ys = [None] + sorted_tuples  # vertical (include None row)

# Create grid and populate counts where keys fall into our axis sets.
grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
for (x, y), count in data.items():
    # Interpret the first element of the key as the vertical (row) value
    # and the second element as the horizontal (column) value so that
    # `None` entries (which appear as the first element in the data)
    # are placed in the vertical "None" row.
    if x not in ys or y not in xs:
        # skip entries outside the requested axis ranges
        continue

    yi = ys.index(x)
    xi = xs.index(y)
    grid[yi, xi] = count

    if yi > 0:
        grid[xi + 1, yi - 1] = count

# Colorscale: either raw log10(count) or log10(column fraction) if requested.
if NORMALIZE_PER_COLUMN:
    # Compute column sums (over vertical axis), avoid division by zero
    col_sums = np.nansum(grid, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_cols = np.where(col_sums > 0, grid / col_sums, np.nan)
    log_grid = np.where(norm_cols > 0, np.log10(norm_cols), np.nan)
    cbar_label = "log10(column fraction)"
else:
    log_grid = np.where(grid > 0, np.log10(grid), np.nan)
    cbar_label = "log10(count)"

# Trim empty rows at the top: find the last row that contains any data
row_has_data = ~np.all(np.isnan(grid), axis=1)
if np.any(row_has_data):
    last_row = int(np.max(np.where(row_has_data)[0]))
    # keep rows up to last_row (inclusive)
    ys = ys[: last_row + 1]
    grid = grid[: last_row + 1, :]
    log_grid = log_grid[: last_row + 1, :]

fig, ax = plt.subplots(figsize=(24, 18))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

cmap = plt.colormaps["plasma"]
norm = mcolors.Normalize()

# Draw empty cells
for yi in range(len(ys)):
    for xi in range(len(xs)):
        if np.isnan(log_grid[yi, xi]):
            ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, color="#2a2a3e"))

img = ax.imshow(
    log_grid,
    norm=norm,
    cmap=cmap,
    aspect="auto",
    origin="lower",
    extent=[-0.5, len(xs) - 0.5, -0.5, len(ys) - 0.5],
)

# Extract minimum gate count per wire count from the data
n_to_min_m = {}
for (x, y) in data.keys():
    if x is not None:
        x_n, x_m = x
        if x_n not in n_to_min_m:
            n_to_min_m[x_n] = x_m
        else:
            n_to_min_m[x_n] = min(n_to_min_m[x_n], x_m)
    if y is not None:
        y_n, y_m = y
        if y_n not in n_to_min_m:
            n_to_min_m[y_n] = y_m
        else:
            n_to_min_m[y_n] = min(n_to_min_m[y_n], y_m)

# Draw reference lines: vertical and horizontal from (n, m_min)
for n, m_min in sorted(n_to_min_m.items()):
    ref_tuple = (n, m_min)
    if ref_tuple not in xs or ref_tuple not in ys:
        continue
    xi = xs.index(ref_tuple)
    yi = ys.index(ref_tuple)
    # Vertical line from bottom to this point
    ax.plot([xi, xi], [-0.5, yi], color="#fff", linewidth=1.5, alpha=0.5)
    # Horizontal line from left to this point
    ax.plot([-0.5, xi], [yi, yi], color="#fff", linewidth=1.5, alpha=0.5)

# (Deliberately do not draw cell count labels — colors show log(count)).

ax.set_xticks(range(len(xs)))
ax.set_xticklabels([f"{t[0]},{t[1]}" for t in xs], color="#fff", fontsize=12)
ax.set_yticks(range(len(ys)))
ax.set_yticklabels(["FRIENDLESS"] + [f"{t[0]},{t[1]}" for t in sorted_tuples[:len(ys)-1]], color="#fff", fontsize=12)
ax.tick_params(axis="x", rotation=90)
ax.tick_params(colors="#ddd", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(img, ax=ax, pad=0.02)
cbar.set_label("log10(num friends)", color="#aaa", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="#555")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaa", fontsize=9)
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig("heatmap.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved heatmap.png")
