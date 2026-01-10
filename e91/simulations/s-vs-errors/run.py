import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 1000

errors = np.linspace(0.0, 0.15, 30)
results3 = []
results4 = []
results5 = []
results6 = []

for error in errors:
    s3, s3eff = three(error, error, n)
    s4, s4eff = five(error, error, n)
    s5, s5eff = four(error, error, n)
    s6, s6eff = six(error, error, n)
    results3.append()
    results4.append()
    results5.append()
    results6.append()

plt.figure(figsize=(5, 3.75))

plt.plot(errors, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(errors, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(errors, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(errors, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

plt.xlabel('symmetric detector error rate')
plt.ylabel('secret key rate (bits/signal)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
