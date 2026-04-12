import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulations import three, four

def compute_heatmap(
    sim_fn, pflip_vals, log10L_vals, p_dark=0.0,
    det_eff=0.9, eta_channel=0.9, eta_degrees=45, f_ec=1.05, n_trials=1, label='progress'):

    L_vals = np.round(10 ** log10L_vals).astype(int)
    heatmap = np.zeros((len(pflip_vals), len(log10L_vals)), dtype=float)

    total_steps = len(pflip_vals) * len(log10L_vals)

    with tqdm(total=total_steps, desc=label) as pbar:
        for i, pflip in enumerate(pflip_vals):
            for j, L in enumerate(L_vals):
                vals = [
                    sim_fn(
                        n=L, p_dark=p_dark, det_eff=det_eff, p_flip=pflip, 
                        eta_channel=eta_channel, eta_degrees=eta_degrees, f_ec=f_ec
                    )
                    for _ in range(n_trials)
                ]
                heatmap[i, j] = np.mean(vals)
                pbar.update(1)

    return heatmap


def get_edges(vals):
    d = vals[1] - vals[0]
    edges = np.concatenate((
        [vals[0] - d / 2],
        vals[:-1] + d / 2,
        [vals[-1] + d / 2]
    ))
    return edges


if __name__ == "__main__":
    # Vertical axis: p_flip from 0.00 to 0.10
    pflip_vals = np.arange(0.00, 0.101, 0.01)   # 11 rows

    # Horizontal axis: ~10 squares
    log10L_vals = np.arange(3.0, 7.1, 0.4)     # 10 columns

    # Compute heatmaps
    heatmap3 = compute_heatmap(
        sim_fn=three, pflip_vals=pflip_vals, log10L_vals=log10L_vals, p_dark=0.0,
        det_eff=0.9, eta_channel=0.9, eta_degrees=45, f_ec=1.05, n_trials=3
    )

    heatmap4 = compute_heatmap(
        sim_fn=four, pflip_vals=pflip_vals, log10L_vals=log10L_vals, p_dark=0.0,
        det_eff=0.9, eta_channel=0.9, eta_degrees=45, f_ec=1.05, n_trials=3
    )

    # Shared color scale
    vmin = min(np.min(heatmap3), np.min(heatmap4))
    vmax = max(np.max(heatmap3), np.max(heatmap4))

    # Cell edges
    x_edges = get_edges(log10L_vals)
    y_edges = get_edges(pflip_vals)

    dx = log10L_vals[1] - log10L_vals[0]
    dy = pflip_vals[1] - pflip_vals[0]

    # Make each subplot roughly square-celled
    ncols = len(log10L_vals)
    nrows = len(pflip_vals)
    cell_size = 0.42
    fig_width = 2 * (ncols * cell_size) + 2.8
    fig_height = nrows * cell_size + 1.8

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)

    # Left: 3-basis
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

    # Right: 4-basis
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

    np.savez(
        "heatmap_data.npz",
        heatmap3=heatmap3,
        heatmap4=heatmap4,
        pflip_vals=pflip_vals,
        log10L_vals=log10L_vals
    )
    
    plt.show()