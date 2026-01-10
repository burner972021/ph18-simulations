import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six
from e91lib import finite_key_rate

block_sizes = np.logspace(3, 5, 3, dtype=int) 
detector_errors = np.linspace(0.0, 0.20, 10) 
# eta_degrees = 45  
f_ec = 1.05 

bases = [three, four, five, six]
titles = ['3 angles', '4 angles', '5 angles', '6 angles']
heatmaps = []

for func in bases:
    KR = np.zeros((len(detector_errors), len(block_sizes)))  # rows=y, cols=x
    for i, e in enumerate(detector_errors):
        for j, n in enumerate(block_sizes):
            KR[i, j] = func(e, e, n)
    heatmaps.append(KR)

fig, axs = plt.subplots(2, 2, figsize=(9, 6))
axs = axs.flatten()

for ax, KR, title in zip(axs, heatmaps, titles):
    im = ax.imshow(KR, origin='lower', aspect='auto',
                   extent=[block_sizes[0], block_sizes[-1], detector_errors[0], detector_errors[-1]],
                   cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("L (block size)")
    ax.set_ylabel("symmetric detector error rate")

    barely_secure = KR < 1e-3
    y_idx, x_idx = np.where(barely_secure)
    ax.scatter(block_sizes[x_idx], detector_errors[y_idx], color='red', marker='x', label='Barely secure')

cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.03, pad=0.1)
cbar.set_label("Secret Key Rate")
plt.tight_layout()
plt.show()
