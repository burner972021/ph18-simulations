import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

p_flip = 0.03
p_dark = 0.03
det_eff = 0.9
sem = True
trials = 5

round_sizes = np.logspace(4, 7, 10).astype(int)

def run(func, round_sizes, trials):
    means, errs = [], []
    for size in round_sizes:
        vals = np.array([func(size, p_dark, det_eff, p_flip) for _ in range(trials)], dtype=float)
        means.append(vals.mean())
        if sem: 
            errs.append(vals.std(ddof=1) / np.sqrt(trials))
        else:
            errs.append(vals.std(ddof=1))
    return np.array(means), np.array(errs)

m3, e3 = run(three, round_sizes, trials)
m4, e4 = run(four, round_sizes, trials)
m5, e5 = run(five, round_sizes, trials)
m6, e6 = run(six, round_sizes, trials)

plt.figure(figsize=(5, 3.75))

plt.errorbar(round_sizes, m3, yerr=e3, marker='x', markersize=4, capsize=3, color='darkorange', label="3 angles")
plt.errorbar(round_sizes, m4, yerr=e4, marker='o', markersize=4, capsize=3, color='khaki', label="4 angles")
plt.errorbar(round_sizes, m5, yerr=e5, marker='^', markersize=4, capsize=3, color='forestgreen', label="5 angles")
plt.errorbar(round_sizes, m6, yerr=e6, marker='P', markersize=4, capsize=3, color='royalblue', label="6 angles")

plt.xscale('log')
plt.xlabel('N (number of rounds)')
plt.ylabel('secret key rate (bits/signal)')

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc=2)
plt.tight_layout()
plt.show()
