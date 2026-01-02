import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

error = 0.03

round_sizes = np.logspace(3, 6, 20).astype(int)
results3 = []
results4 = []
results5 = []
results6 = []

for size in round_sizes:
    kr3 = three(error, error, size)
    kr4 = four(error, error, size)
    kr5 = five(error, error, size)
    kr6 = six(error, error, size)

    results3.append(kr3)
    results4.append(kr4)
    results5.append(kr5)
    results6.append(kr6)

plt.figure(figsize=(5, 3.75))

plt.plot(round_sizes, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(round_sizes, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(round_sizes, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(round_sizes, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

plt.xscale('log')
plt.xlabel('L (block size)')
plt.ylabel('secret key rate (bits/signal)')
plt.ylim(0.0, 0.20)
plt.xlim(1e3, 1e6)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc=2)
plt.tight_layout()
plt.show()
