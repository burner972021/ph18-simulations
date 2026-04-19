import numpy as np
import matplotlib.pyplot as plt


def get_edges(vals):
    d = vals[1] - vals[0]
    edges = np.concatenate((
        [vals[0] - d / 2],
        vals[:-1] + d / 2,
        [vals[-1] + d / 2]
    ))
    return edges


# Load previously generated data
data = np.load("heatmap_data.npz")
heatmap3 = data["heatmap3"]
heatmap4 = data["heatmap4"]
pflip_vals = data["pflip_vals"]
log10L_vals = data["log10L_vals"]

# Shared color scale
vmin = min(np.min(heatmap3), np.min(heatmap4))
vmax = max(np.max(heatmap3), np.max(heatmap4))

# Cell edges
x_edges = get_edges(log10L_vals)
y_edges = get_edges(pflip_vals)

dx = log10L_vals[1] - log10L_vals[0]
dy = pflip_vals[1] - pflip_vals[0]

# Figure sizing for vertical layout
ncols = len(log10L_vals)
nrows = len(pflip_vals)
cell_size = 0.42
fig_width = ncols * cell_size + 2.2
fig_height = 2 * (nrows * cell_size) + 2.8

fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), constrained_layout=True)

# Top: 3-basis
mesh1 = axes[0].pcolormesh(
    x_edges, y_edges, heatmap3,
    cmap="viridis",
    shading="flat",
    vmin=vmin, vmax=vmax
)
axes[0].invert_yaxis()
axes[0].set_aspect(dx / dy)
axes[0].set_title("3-basis", fontsize=13)
axes[0].set_xlabel(r'$\log_{10} L$', fontsize=12)
axes[0].set_ylabel(r'$p_{\mathrm{flip}}$', fontsize=12)
axes[0].set_xticks(log10L_vals)
axes[0].set_xticklabels([f"{x:.1f}" for x in log10L_vals], rotation=60)
axes[0].set_yticks(pflip_vals)
axes[0].set_yticklabels([f"{y:.2f}" for y in pflip_vals])

# Bottom: 4-basis
mesh2 = axes[1].pcolormesh(
    x_edges, y_edges, heatmap4,
    cmap="viridis",
    shading="flat",
    vmin=vmin, vmax=vmax
)
axes[1].invert_yaxis()
axes[1].set_aspect(dx / dy)
axes[1].set_title("4-basis", fontsize=13)
axes[1].set_xlabel(r'$\log_{10} L$', fontsize=12)
axes[1].set_ylabel(r'$p_{\mathrm{flip}}$', fontsize=12)
axes[1].set_xticks(log10L_vals)
axes[1].set_xticklabels([f"{x:.1f}" for x in log10L_vals], rotation=60)
axes[1].set_yticks(pflip_vals)
axes[1].set_yticklabels([f"{y:.2f}" for y in pflip_vals])

# One shared colorbar
cbar = fig.colorbar(mesh2, ax=axes, shrink=0.95)
cbar.set_label("secret key rate (bit/s)", fontsize=12)

plt.show()