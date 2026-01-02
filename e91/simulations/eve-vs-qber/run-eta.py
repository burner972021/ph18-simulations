import numpy as np
import matplotlib.pyplot as plt
from simulations import three, four, five, six

n = 30000
error = 0.03

etas = np.linspace(0.0, 90.0, 20)
results3 = []
results4 = []
results5 = []
results6 = []

for eta in etas:
    results3.append(three(error, error, n, eta_degrees=eta))
    results4.append(four(error, error, n, eta_degrees=eta))
    results5.append(five(error, error, n, eta_degrees=eta))
    results6.append(six(error, error, n, eta_degrees=eta))

plt.figure(figsize=(5, 3.75))

plt.plot(etas, results3, marker='x', markersize=4, color='darkorange', label="3 angles")
plt.plot(etas, results4, marker='o', markersize=4, color='khaki', label="4 angles")
plt.plot(etas, results5, marker='^', markersize=4, color='forestgreen', label="5 angles")
plt.plot(etas, results6, marker='P', markersize=4, color='royalblue', label="6 angles")

plt.xlabel(r'$\eta$')
plt.ylabel('Eve Holevo information')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
