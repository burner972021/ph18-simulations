import numpy as np
import matplotlib.pyplot as plt
from simulations import three_distance, four_distance, five_distance, six_distance

plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
plt.rcParams['ps.fonttype'] = 42

n = 100000
det_eff = 0.9
p_flip = 0.05
trials = 5

distances = np.linspace(0, 60, 10)

def run(func, distances, trials):
    means, errs = [], []
    for distance in distances:
        vals = np.array([func(n, p_flip, det_eff, distance) for _ in range(trials)], dtype=float)
        means.append(vals.mean())
        errs.append(vals.std(ddof=1) / np.sqrt(trials))
    return np.array(means), np.array(errs)

m3, e3 = run(three_distance, distances, trials)
m4, e4 = run(four_distance, distances, trials)
m5, e5 = run(five_distance, distances, trials)
m6, e6 = run(six_distance, distances, trials)

plt.figure(figsize=(5, 3.75))

plt.errorbar(distances, m3, yerr=e3, marker='x', markersize=4, capsize=3, color='darkorange', label="3 angles")
plt.errorbar(distances, m4, yerr=e4, marker='o', markersize=4, capsize=3, color='khaki', label="4 angles")
plt.errorbar(distances, m5, yerr=e5, marker='^', markersize=4, capsize=3, color='forestgreen', label="5 angles")
plt.errorbar(distances, m6, yerr=e6, marker='P', markersize=4, capsize=3, color='royalblue', label="6 angles")


plt.ylim(1e-6, 1e-1)
plt.yscale("log")
plt.xlabel("distance (km)")
plt.ylabel("secret key rate (bits/signal)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()