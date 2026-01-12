import numpy as np
import matplotlib.pyplot as plt
from simulations import three_distance, four_distance, five_distance, six_distance

n = 100000
det_eff = 0.9
p_flip = 0.05

distances = np.linspace(0, 60, 10)

results3, results4, results5, results6 = [], [], [], []

for d in distances:
    results3.append(three_distance(n, p_flip, det_eff, d))
    results4.append(four_distance(n, p_flip, det_eff, d))
    results5.append(five_distance(n, p_flip, det_eff, d))
    results6.append(six_distance(n, p_flip, det_eff, d))

plt.figure(figsize=(5, 3.75))
plt.plot(distances, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(distances, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(distances, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(distances, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

plt.ylim(1e-6, 1e-1)
plt.yscale("log")
plt.xlabel("distance (km)")
plt.ylabel("secret key rate (bits/signal)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()