import numpy as np 
import random

runs = 100
rounds = 10000
eta = np.linspace(0, 2*np.pi, 100)

alice_angles = [0.0, 45.0, 22.5]
bob_angles = [0.0, -22.5, 22.5]

a1b3 = [0] * 4
a1b2 = [0] * 4
a2b3 = [0] * 4
a2b2 = [0] * 4

counts = [a1b3, a1b2, a2b3, a2b2] 
store = {'02': 0, '01': 1, '12': 2, '11': 3}
s_values = []

def calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, arr):
    if r < p_cc:
        arr[store[i]][0] += 1
    elif r < p_cc + p_cnc:
        arr[store[i]][1] += 1
    elif r < p_cc + p_cnc + p_ncc:
        arr[store[i]][2] += 1
    else:
        arr[store[i]][3] += 1
    return arr


for _1 in range(runs):
    new_counts = counts
    key = ''
    for _2 in range(rounds):
        a = random.randint(0, 2)
        b = random.randint(0, 2)
        ra = np.radians(alice_angles[a])
        rb = np.radians(bob_angles[b])

        if ra == rb:
            key += str(np.round(random.random()))[0]
        else:
            diff = np.radians(abs(ra - rb))
            p_cc = ((np.cos(eta) * np.cos(ra) * np.cos(rb)) + (np.sin(eta) * np.sin(ra) * np.cos(rb))) ** 2
            p_cnc = ((-np.cos(eta) * np.cos(ra) * np.sin(rb)) + (np.sin(eta) * np.sin(ra) * np.cos(rb))) ** 2
            p_ncc = ((-np.cos(eta) * np.sin(ra) * np.cos(rb)) + (np.sin(eta) * np.cos(ra) * np.sin(rb))) ** 2
            p_ncnc = ((np.cos(eta) * np.sin(ra) * np.sin(rb)) + (np.sin(eta) * np.cos(ra) * np.cos(rb))) ** 2

            r = random.random()
            i = str(a + b)
            if i in store:
                calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, new_counts)

    e13 = (counts[0][0]+counts[0][3]-counts[0][1]-counts[0][2]) / sum(counts[0])
    e12 = (counts[1][0]+counts[1][3]-counts[1][1]-counts[1][2]) / sum(counts[1])
    e23 = (counts[2][0]+counts[2][3]-counts[2][1]-counts[2][2]) / sum(counts[2])
    e22 = (counts[3][0]+counts[3][3]-counts[3][1]-counts[3][2]) / sum(counts[3])

    S = e13 + e12 + e23 - e22
    s_values.append(S)

x = eta
y1 = np.array(s_values)
fig, axs = plt.subplot('left', 'right')
axs['left'].scatter(x, y2)

ideal = np.array([2*np.sqrt(2)] * 100)
axs['left'].plot(x, ideal, label='ideal', color='red')
plt.legend()

axs['left'].set_xlabel(r'eta ($\eta$)')
axs['left'].set_ylabel('average calculated S value')

plt.show()