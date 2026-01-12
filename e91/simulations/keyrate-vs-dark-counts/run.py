import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 100000
det_eff = 0.9
p_flip = 0.0
trials = 5

p_darks = np.linspace(0.0, 0.15, 10)

def run(func, p_darks, trials):
    means, errs = [], []
    for p_dark in p_darks:
        vals = np.array([func(n, p_dark, det_eff, p_flip) for _ in range(trials)], dtype=float)
        means.append(vals.mean())
        errs.append(vals.std(ddof=1) / np.sqrt(trials))
    return np.array(means), np.array(errs)

m3, e3 = run(three, p_darks, trials)
m4, e4 = run(four, p_darks, trials)
m5, e5 = run(five, p_darks, trials)
m6, e6 = run(six, p_darks, trials)

plt.figure(figsize=(5, 3.75))

plt.errorbar(p_darks, m3, yerr=e3, marker='x', markersize=4, capsize=3, color='darkorange', label="3 angles")
plt.errorbar(p_darks, m4, yerr=e4, marker='o', markersize=4, capsize=3, color='khaki', label="4 angles")
plt.errorbar(p_darks, m5, yerr=e5, marker='^', markersize=4, capsize=3, color='forestgreen', label="5 angles")
plt.errorbar(p_darks, m6, yerr=e6, marker='P', markersize=4, capsize=3, color='royalblue', label="6 angles")

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