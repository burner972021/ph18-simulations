import numpy as np
import matplotlib.pyplot as plt
import random

runs_per_round_size = 30         
round_sizes = np.logspace(3, 6, 20).astype(int)   

alice_angles = [0.0, 45.0, 22.5]
bob_angles   = [0.0, -22.5, 22.5]
store = {'02': 0, '01': 1, '12': 2, '11': 3}

ideal_S = 2 * np.sqrt(2)

def calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, arr):
    if r < p_cc:                       arr[store[i]][0] += 1
    elif r < p_cc + p_cnc:             arr[store[i]][1] += 1
    elif r < p_cc + p_cnc + p_ncc:     arr[store[i]][2] += 1
    else:                              arr[store[i]][3] += 1

def new_counts():
    return [[0]*4 for _ in range(4)]

def compute_S(rounds):
    counts = new_counts()
    for _ in range(rounds):
        a = random.randint(0, 2)
        b = random.randint(0, 2)
        ra = alice_angles[a]
        rb = bob_angles[b]

        if ra != rb:
            diff = np.radians(abs(ra - rb))
            p_cc = 0.5*np.cos(diff)**2
            p_cnc = 0.5*np.sin(diff)**2
            p_ncc = 0.5*np.sin(diff)**2
            p_ncnc = 0.5*np.cos(diff)**2

            r = random.random()
            i = str(a) + str(b)
            if i in store:
                calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, counts)

    e13 = (counts[0][0]+counts[0][3]-counts[0][1]-counts[0][2]) / sum(counts[0])
    e12 = (counts[1][0]+counts[1][3]-counts[1][1]-counts[1][2]) / sum(counts[1])
    e23 = (counts[2][0]+counts[2][3]-counts[2][1]-counts[2][2]) / sum(counts[2])
    e22 = (counts[3][0]+counts[3][3]-counts[3][1]-counts[3][2]) / sum(counts[3])

    return e13 + e12 + e23 - e22

avg_devs = []

for N in round_sizes:
    values = [compute_S(N) for _ in range(runs_per_round_size)]
    avg_S  = np.mean(values)
    deviation = abs(avg_S - ideal_S)
    avg_devs.append(deviation)
    print(f"N={N}  avg S={avg_S:.4f}  deviation={deviation:.4f}")

plt.figure(figsize=(8,5))
plt.loglog(round_sizes, avg_devs, marker='o')

plt.xlabel("Number of rounds (dataset size)")
plt.ylabel("Average |S - ideal|")
plt.title("Convergence of CHSH S-value to ideal (2√2)")
plt.grid(True, which="both", ls="--")

plt.show()
