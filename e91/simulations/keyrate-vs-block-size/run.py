import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

p_flip = 0.03
p_dark = 0.03
det_eff = 0.9

round_sizes = np.logspace(4, 6, 10).astype(int)
results3 = []
results4 = []
results5 = []
results6 = []

for size in round_sizes:
    results3.append(three(size, p_dark, det_eff, p_flip))
    results4.append(four(size, p_dark, det_eff, p_flip))
    results5.append(five(size, p_dark, det_eff, p_flip))
    results6.append(six(size, p_dark, det_eff, p_flip))

plt.figure(figsize=(5, 3.75))

plt.plot(round_sizes, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(round_sizes, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(round_sizes, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(round_sizes, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

plt.xscale('log')
plt.xlabel('L (block size)')
plt.ylabel('secret key rate (bits/signal)')

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc=2)
plt.tight_layout()
plt.show()
