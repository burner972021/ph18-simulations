import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 100000
det_eff = 0.9
p_flip = 0.0

p_darks = np.linspace(0.0, 0.15, 10)
results3 = []
results4 = []
results5 = []
results6 = []

for p_dark in p_darks:
    results3.append(three(n, p_dark, det_eff, p_flip))
    results4.append(four(n, p_dark, det_eff, p_flip))
    results5.append(five(n, p_dark, det_eff, p_flip))
    results6.append(six(n, p_dark, det_eff, p_flip))

plt.figure(figsize=(5, 3.75))

plt.plot(p_darks, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(p_darks, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(p_darks, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(p_darks, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

# plt.xlim(0.0, 0.1)
# plt.ylim(0.0, 0.3)
plt.xticks(np.arange(0.0, 0.151, 0.05))
plt.yticks(np.arange(0.0, 0.151, 0.05))

plt.xlabel(r'dark count error rate ($p_{dark}$)')
plt.ylabel('secret key rate (bits/signal)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()