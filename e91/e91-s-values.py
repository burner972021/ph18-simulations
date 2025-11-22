import matplotlib.pyplot as plt
import numpy as np
import random

runs = 100

round_list = [int(1000 * (10 ** (i / (runs/3)))) for i in range(runs)]

alice_angles = [0.0, 45.0, 22.5]
bob_angles = [0.0, -22.5, 22.5]

store = {'02': 0, '01': 1, '12': 2, '11': 3}

def calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, arr):
    if r < p_cc:
        arr[store[i]][0] += 1
    elif r < p_cc + p_cnc:
        arr[store[i]][1] += 1
    elif r < p_cc + p_cnc + p_ncc:
        arr[store[i]][2] += 1
    else:
        arr[store[i]][3] += 1

def new_counts():
    return [[0]*4 for _ in range(4)]

s_values = []

for rounds in round_list:

    counts = new_counts()
    key = ""

    for _ in range(rounds):
        a = random.randint(0, 2)
        b = random.randint(0, 2)
        ra = alice_angles[a]
        rb = bob_angles[b]

        if ra == rb:
            key += str(np.round(random.random()))
        else:
            diff = np.radians(abs(ra - rb))
            p_cc = 0.5 * np.cos(diff)**2
            p_cnc = 0.5 * np.sin(diff)**2
            p_ncc = 0.5 * np.sin(diff)**2
            p_ncnc = 0.5 * np.cos(diff)**2

            r = random.random()
            i = str(a) + str(b)
            if i in store:
                calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, counts)

    e13 = (counts[0][0]+counts[0][3]-counts[0][1]-counts[0][2]) / sum(counts[0])
    e12 = (counts[1][0]+counts[1][3]-counts[1][1]-counts[1][2]) / sum(counts[1])
    e23 = (counts[2][0]+counts[2][3]-counts[2][1]-counts[2][2]) / sum(counts[2])
    e22 = (counts[3][0]+counts[3][3]-counts[3][1]-counts[3][2]) / sum(counts[3])

    S = e13 + e12 + e23 - e22
    s_values.append(S)

x = np.array(round_list)
y = np.array(s_values)

plt.scatter(x, y)

ideal = 2*np.sqrt(2)

plt.ylim(1.0, 3.5)

ticks = [1.0, 1.5, 2.0, 2.5, ideal, 3.0, 3.5]
labels = ["1.0", "1.5", "2.0", "2.5", f"{ideal:.3f} ideal", "3.0", "3.5"]
plt.yticks(ticks, labels)

coeffs = np.polyfit(np.log(x), y, 1)
fit = coeffs[0] * np.log(x) + coeffs[1]
plt.plot(x, fit, color='g')

plt.xscale("log")
plt.axhline(ideal, color='r')
plt.xlabel("Number of rounds (log scale)")
plt.ylabel("S-value")
plt.title("S-value vs number of rounds")
plt.show()
