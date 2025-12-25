import numpy as np 
import random
from e91lib import calc_sval, output, update_counts

n = 100000
eta_degrees = 45        # default: 0 noise
eta = np.radians(eta_degrees)
alice_error = 0         # alice's detector error rate
bob_error = 0       # bob detector error rate

alice_angles = [0.0, 45.0, 22.5]
bob_angles = [0.0, -22.5, 22.5]
key = ''
matchcount = 0
keylength = 0

a1b3 = [0] * 4
a1b2 = [0] * 4
a2b3 = [0] * 4
a2b2 = [0] * 4

counts = [a1b3, a1b2, a2b3, a2b2] 
store = {'02': 0, '01': 1, '12': 2, '11': 3}
s_values = []

for _ in range(n):
    a = random.randint(0, 2)
    b = random.randint(0, 2)
    ra = alice_angles[a]
    rb = bob_angles[b]  
    r = random.random()

    p_cc = (np.cos(eta)*np.cos(ra)*np.cos(rb) + np.sin(eta)*np.sin(ra)*np.sin(rb))**2
    p_cnc = (-np.cos(eta)*np.cos(ra)*np.sin(rb) + np.sin(eta)*np.sin(ra)*np.cos(rb))**2
    p_ncc = (-np.cos(eta)*np.sin(ra)*np.cos(rb) + np.sin(eta)*np.cos(ra)*np.sin(rb))**2
    p_ncnc = (np.cos(eta)*np.sin(ra)*np.sin(rb) + np.sin(eta)*np.cos(ra)*np.cos(rb))**2

    if (ra == rb):
        key += str(np.round(random.random(), 0))[0]
        matchcount += 1
        
        if (r < p_cc):
            alice_bit = 1
            bob_bit = 1
        elif (p_cc <= r < p_cnc):
            alice_bit = 1
            bob_bit = 0
        elif (p_cc + p_cnc <= r < p_ncc):
            alice_bit = 0
            bob_bit = 1
        else:
            alice_bit = 0
            bob_bit = 0
        
        alice_rand = random.random()
        bob_rand = random.random()
        if (alice_rand < alice_error): alice_bit = 1 - alice_bit
        if (bob_rand < bob_error): bob_bit = 1 - bob_bit
        if (alice_bit == bob_bit): keylength += 1
        
    else:
        r = random.random()
        
        i = str(a) + str(b)
        if i in store:
            counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, counts, store)

s = calc_sval(counts)

print(f'\nS Value = {s}')
output(counts)
print(f'Asymptotic key rate = {matchcount/n}')
print(f'True key rate = {keylength/n}')