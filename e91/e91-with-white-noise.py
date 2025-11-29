import numpy as np
import matplotlib.pyplot as plt

runs = 10        
rounds = 10000   
etas = np.linspace(0, 2*np.pi, 100)
ideal_S = 2*np.sqrt(2)

alice_angles_deg = np.array([0.0, 22.5, 45.0]) 
alice_rad = np.radians(alice_angles_deg)

bob_angles_deg = np.array([0.0, 22.5, -22.5]) 
bob_rad = np.radians(bob_angles_deg)

store = {'01': 0, '02': 1, '21': 2, '22': 3}

def compute_S_vectorised(eta):
    a = np.random.randint(0, 3, rounds)
    b = np.random.randint(0, 3, rounds)

    ra = alice_rad[a]
    rb = bob_rad[b]

    pair_keys = [f"{a[i]}{b[i]}" for i in range(rounds)]
    pair_idx = np.array([store.get(key, -1) for key in pair_keys])

    p_cc = (np.cos(eta)*np.cos(ra)*np.cos(rb) + np.sin(eta)*np.sin(ra)*np.sin(rb))**2
    p_cnc = (-np.cos(eta)*np.cos(ra)*np.sin(rb) + np.sin(eta)*np.sin(ra)*np.cos(rb))**2
    p_ncc = (-np.cos(eta)*np.sin(ra)*np.cos(rb) + np.sin(eta)*np.cos(ra)*np.sin(rb))**2
    p_ncnc = (np.cos(eta)*np.sin(ra)*np.sin(rb) + np.sin(eta)*np.cos(ra)*np.cos(rb))**2

    r = np.random.random(rounds)
    outcome = np.zeros(rounds, dtype=int)
    outcome[r >= p_cc] = 1
    outcome[r >= p_cc + p_cnc] = 2
    outcome[r >= p_cc + p_cnc + p_ncc] = 3

    counts = np.zeros((4, 4), dtype=int)
    for idx in range(4):
        mask = pair_idx == idx
        if np.any(mask):
            vals, freq = np.unique(outcome[mask], return_counts=True)
            for v, f in zip(vals, freq):
                counts[idx, v] = f

    with np.errstate(divide='ignore', invalid='ignore'):
        E = (counts[:,0] + counts[:,3] - counts[:,1] - counts[:,2]) / counts.sum(axis=1)

    E[np.isnan(E)] = 0

    e13, e12, e23, e22 = E
    return e13 + e12 + e23 - e22

S_values = np.array([compute_S_vectorised(eta) for eta in etas])

deviations = []
for eta in etas:
    S_many = np.array([compute_S_vectorised(eta) for _ in range(runs)])
    deviations.append(np.abs(S_many.mean() - ideal_S))

deviations = np.array(deviations)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(etas, S_values, marker='o', linestyle='None')
axes[0].plot(etas, np.full_like(etas, ideal_S), label='Ideal S, 2√2', color='red')
axes[0].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
axes[0].set_xticklabels(['0', '½π', 'π', '3/2π', '2π'])
axes[0].set_title("Single-run S value vs η")
axes[0].set_xlabel("η")
axes[0].set_ylabel("S value")
axes[0].legend(loc='lower left')

axes[1].plot(etas, deviations)
axes[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
axes[1].set_xticklabels(['0', '1/2π', 'π', '3/2π', '2π'])
axes[1].set_title("Average deviation from ideal vs η")
axes[1].set_xlabel("η")
axes[1].set_ylabel("$|\\langle S \\rangle - 2\\sqrt{2}|$")

plt.tight_layout()
plt.show()