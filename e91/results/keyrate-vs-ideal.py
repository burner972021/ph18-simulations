import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 100000
p_dark = 0.0
det_eff = 0.9

p_flips = np.linspace(0.0, 0.15, 10)
results3 = []
results4 = []
results5 = []
results6 = []

for p_flip in p_flips:
    results3.append(three(n, p_dark, det_eff, p_flip))
    results4.append(four(n, p_dark, det_eff, p_flip))
    results5.append(five(n, p_dark, det_eff, p_flip))
    results6.append(six(n, p_dark, det_eff, p_flip))

plt.figure(figsize=(5, 3.75))

plt.plot(p_flips, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(p_flips, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(p_flips, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(p_flips, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

three_ideal = 2/9
four_ideal = 1/4
five_ideal = 1/5
six_ideal = 1/6

plt.xlabel(r'bit flip error rate ($p_{flip}$)')
plt.ylabel('secret key rate (bits/signal)')
plt.axhline(three_ideal, ls='--', color='darkorange', alpha=0.65)
plt.axhline(four_ideal, ls='--', color='khaki', alpha=0.85)
plt.axhline(five_ideal, ls='--', color='forestgreen', alpha=0.65)
plt.axhline(six_ideal, ls='--', color='royalblue', alpha=0.65)
plt.grid(True, which="both", ls="--", alpha=0.65)
plt.legend()
plt.tight_layout()
plt.show()
