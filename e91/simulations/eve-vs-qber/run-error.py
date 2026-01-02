import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 50000

errors = np.linspace(0.0, 0.30, 15)
results3 = []
results4 = []
results5 = []
results6 = []

for error in errors:
    results3.append(three(error, error, n))
    results4.append(four(error, error, n))
    results5.append(five(error, error, n))
    results6.append(six(error, error, n))

plt.figure(figsize=(5, 3.75))

plt.plot(errors, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(errors, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(errors, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(errors, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

plt.xlabel('symmetric detector error rate')
plt.ylabel('Eve Holevo information')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
