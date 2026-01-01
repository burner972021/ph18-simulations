import numpy as np
import random
from e91lib import calc_sval, output, update_counts, s_uncertainty, finite_key_rate, i_eve

n = 100000
eta_degrees = 45        # default: 0 noise
eta = np.radians(eta_degrees)
alice_error = 0.05         # alice's detector error rate
bob_error = 0.05       # bob detector error rate

alice_angles = [-22.5, 0.0, 22.5, 45.0, 67.5]
bob_angles = [-22.5, 0.0, 22.5, 45.0, 67.5]
key = ''
keylength = 0
matchcount = 0

s1_counts = np.zeros((4, 4))
s2_counts = np.zeros((4, 4))
s3_counts = np.zeros((4, 4))

store1 = {'10': 0, '12': 1, '32': 2, '30': 3}
store2 = {'01': 0, '21': 1, '23': 2, '03': 3}
store3 = {'10': 0, '32': 1, '34': 2, '14': 3}

for _ in range(n):
    a = random.randint(0, 4)
    b = random.randint(0, 4)
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


s1 = calc_sval(s1_counts)
s2 = calc_sval(s2_counts)
s3 = calc_sval(s3_counts)
s = (s1 + s2 + s3)/3    # take average s

s_delta = (s_uncertainty(s1_counts) + s_uncertainty(s2_counts) + s_uncertainty(s3_counts))/3    # statistical deviation of S values
s_eff = s - s_delta     # effective S value
qber = 1 - keylength/matchcount     # quantum bit error rate
f_ec = 1.05      # error correction inefficiency factor (using cascade)
leaked = i_eve(s_eff)
key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

print(f'Number of rounds = {n}')
print(f'S Value 1 = {s1}')
print(f'S Value 2 = {s2}')
print(f'S Value 3 = {s3}')
print(f'Average S value = {s}')
print(f'Finite-key S uncertainty = {s_delta}')
print(f'Maximum Holevo information leaked to Eve = {leaked}')
print(f'Effective S value = {s_eff}')
print(f'Asymptotic key rate = {matchcount/n}')
print(f'key rate with probabilistic faulty detectors = {keylength/n}')
print(f'True finite secret key rate (per signal) = {key_rate}')