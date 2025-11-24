import numpy as np 
import random

alice_angles = [0.0, 45.0, 22.5]
bob_angles = [0.0, -22.5, 22.5]

key = ''
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

for _ in range(10000):
    a = random.randint(0, 2)
    b = random.randint(0, 2)
    ra = alice_angles[a]
    rb = bob_angles[b]  

    if ra == rb:
        key += str(np.round(random.random(), 0))[0]
    else:
        diff = np.radians(abs(ra - rb))

        p_cc = 0.5 * np.cos(diff)**2
        p_cnc = 0.5 * np.sin(diff)**2
        p_ncc = 0.5 * np.sin(diff)**2
        p_ncnc = 0.5 * np.cos(diff)**2

        r = random.random()
        
        i = str(a) + str(b)
        if i in store:
            counts = calc_probabilities(r, i, p_cc, p_cnc, p_ncc, p_ncnc, counts)

e13 = (counts[0][0]+counts[0][3]-counts[0][1]-counts[0][2])/(counts[0][0]+counts[0][3]+counts[0][1]+counts[0][2])
e12 = (counts[1][0]+counts[1][3]-counts[1][1]-counts[1][2])/(counts[1][0]+counts[1][3]+counts[1][1]+counts[1][2])
e23 = (counts[2][0]+counts[2][3]-counts[2][1]-counts[2][2])/(counts[2][0]+counts[2][3]+counts[2][1]+counts[2][2])
e22 = (counts[3][0]+counts[3][3]-counts[3][1]-counts[3][2])/(counts[3][0]+counts[3][3]+counts[3][1]+counts[3][2])
s = e13 + e12 + e23 - e22


print('a1b3:')
print(f'P(cc) = {counts[0][0]}')
print(f'P(cnc) = {counts[0][1]}')
print(f'P(ncc) = {counts[0][2]}')
print(f'P(ncnc) = {counts[0][3]}')

print('a1b2:')
print(f'P(cc) = {counts[1][0]}')
print(f'P(cnc) = {counts[1][1]}')
print(f'P(ncc) = {counts[1][2]}')
print(f'P(ncnc) = {counts[1][3]}')

print('a2b3:')
print(f'P(cc) = {counts[2][0]}')
print(f'P(cnc) = {counts[2][1]}')
print(f'P(ncc) = {counts[2][2]}')
print(f'P(ncnc) = {counts[2][3]}')

print('a2b2:')
print(f'P(cc) = {counts[3][0]}')
print(f'P(cnc) = {counts[3][1]}')
print(f'P(ncc) = {counts[3][2]}')
print(f'P(ncnc) = {counts[3][3]}')

print(f'\nS Value = {s}')

print(f'\nKey = {key}')