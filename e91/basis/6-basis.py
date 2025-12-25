import numpy as np
import random
from e91lib import calc_sval, output, update_counts

n = 100000
eta_degrees = 45        # default: 0 noise
eta = np.radians(eta_degrees)
alice_error = 0         # alice's detector error rate
bob_error = 0       # bob detector error rate

alice_angles = [-22.5, 0.0, 22.5, 45.0, 67.5, 90.0]
bob_angles = [-22.5, 0.0, 22.5, 45.0, 67.5, 90.0]
key = ''
keylength = 0
matchcount = 0

s1_counts = np.zeros((4, 4))
s2_counts = np.zeros((4, 4))
s3_counts = np.zeros((4, 4))
s4_counts = np.zeros((4, 4))

store1 = {'10': 0, '12': 1, '32': 2, '30': 3}
store2 = {'01': 0, '21': 1, '23': 2, '03': 3}
store3 = {'10': 0, '32': 1, '34': 2, '14': 3}
store4 = {'12': 0, '23': 1, '45': 2, '25': 3}

for _ in range(n):
    a = random.randint(0, 5)
    b = random.randint(0, 5)
    ra = np.radians(alice_angles[a])
    rb = np.radians(bob_angles[b])
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
        diff = np.radians(abs(ra - rb))
        i = str(a) + str(b)
        if (i in store1):
            s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s1_counts, store1)
        if (i in store2):
            s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s2_counts, store2)
        if (i in store3):
            s3_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s1_counts, store3)
        if (i in store4):
            s4_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s1_counts, store4)


s1 = calc_sval(s1_counts)
s2 = calc_sval(s2_counts)
s3 = calc_sval(s3_counts)
s4 = calc_sval(s4_counts)

print(f'S_1 = {s1}')
output(s1_counts)
print('------------------')
print(f'S_2 = {s2}')
output(s2_counts)
print('------------------')
print(f'S_3 = {s3}')
output(s3_counts)
print('------------------')
print(f'S_3 = {s4}')
output(s4_counts)
print('------------------')
print(f'Asymptotic key rate = {matchcount/n}') 
print(f'True key rate = {keylength/n}')