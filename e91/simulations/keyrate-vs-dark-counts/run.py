import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 100000
det_eff = 0.8

errors = np.linspace(0.0, 0.05, 10)
results3 = []
results4 = []
results5 = []
results6 = []

for error in errors:
    results3.append(three(n, error, det_eff))
    results4.append(four(n, error, det_eff))
    results5.append(five(n, error, det_eff))
    results6.append(six(n, error, det_eff))

plt.figure(figsize=(5, 3.75))

plt.plot(errors, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(errors, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(errors, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(errors, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

# plt.xlim(0.0, 0.1)
# plt.ylim(0.0, 0.3)
# plt.xticks(np.arange(0.0, 0.151, 0.05))
# plt.yticks(np.arange(0.0, 0.201, 0.05))

plt.xlabel('detector dark count error rate')
plt.ylabel('secret key rate (bits/signal)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()