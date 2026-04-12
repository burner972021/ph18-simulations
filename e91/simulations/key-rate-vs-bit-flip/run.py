import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
plt.rcParams['ps.fonttype'] = 42

n = 100000
p_dark = 0.0
det_eff = 0.9
p_flips = np.linspace(0.0, 0.15, 10)

trials = 1
use_sem = True  

def run(func, p_flip_values, trials):
    means, errs = [], []
    for p in p_flip_values:
        vals = np.array([func(n, p_dark, det_eff, p) for _ in range(trials)], dtype=float)
        means.append(vals.mean())
        if use_sem: 
            errs.append(vals.std(ddof=1) / np.sqrt(trials))
        else:
            errs.append(vals.std(ddof=1))
    return np.array(means), np.array(errs)

m3, e3 = run(three, p_flips, trials)
m4, e4 = run(four, p_flips, trials)
m5, e5 = run(five, p_flips, trials)
m6, e6 = run(six, p_flips, trials)

plt.figure(figsize=(5, 3.75))

plt.errorbar(p_flips, m3, yerr=e3, marker='x', markersize=4, capsize=3, color='darkorange', label="3 angles")
plt.errorbar(p_flips, m4, yerr=e4, marker='o', markersize=4, capsize=3, color='khaki', label="4 angles")
plt.errorbar(p_flips, m5, yerr=e5, marker='^', markersize=4, capsize=3, color='forestgreen', label="5 angles")
plt.errorbar(p_flips, m6, yerr=e6, marker='P', markersize=4, capsize=3, color='royalblue', label="6 angles")

plt.xticks(np.arange(0.0, 0.151, 0.05))
plt.yticks(np.arange(0.0, 0.151, 0.05))
plt.xlabel(r'bit flip error rate ($p_{flip}$)')
plt.ylabel('secret key rate (bits/signal)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('../../results/keyrate-vs-bit-flip.png')
# plt.savefig('../../results/keyrate-vs-bit-flip.pdf')