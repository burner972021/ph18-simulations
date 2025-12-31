import numpy as np
# library of helper functions for PH18 Ekert-91 simulations.
# functions can be used to calculate Bell parameters, expectation values, S-values (for CHSH).
# 31 Dec 2025 updatw: added functions to calculate statistical uncertainty of S, binary entropy, and Eve's information rate.


def update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, arr, store):
    if r < p_cc:
        arr[store[i]][0] += 1
    elif r < p_cc + p_cnc:
        arr[store[i]][1] += 1
    elif r < p_cc + p_cnc + p_ncc:
        arr[store[i]][2] += 1
    else:
        arr[store[i]][3] += 1
    return arr

def calc_sval(counts):
    e13 = (counts[0][0]+counts[0][3]-counts[0][1]-counts[0][2])/(counts[0][0]+counts[0][3]+counts[0][1]+counts[0][2])
    e12 = (counts[1][0]+counts[1][3]-counts[1][1]-counts[1][2])/(counts[1][0]+counts[1][3]+counts[1][1]+counts[1][2])
    e23 = (counts[2][0]+counts[2][3]-counts[2][1]-counts[2][2])/(counts[2][0]+counts[2][3]+counts[2][1]+counts[2][2])
    e22 = (counts[3][0]+counts[3][3]-counts[3][1]-counts[3][2])/(counts[3][0]+counts[3][3]+counts[3][1]+counts[3][2])
    return e13 + e12 + e23 - e22

def output(counts):
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

def s_uncertainty(counts, eps=1e-10):
    delta = 0
    for setting in counts:
        N = sum(setting)
        if N == 0:
            continue
        delta += np.sqrt(np.log(1/eps) / (2 * N))
    return delta

def binary_entropy(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

def i_eve(s_eff):
    if s_eff <= 2:
        return 1.0   # classical S value
    C = np.sqrt((s_eff/2)**2 - 1)
    p = (1 + C) / 2
    return binary_entropy(p)

def finite_key_rate(matchcount, n, s_eff, f_ec, qber):
    rate = 1 - i_eve(s_eff) - f_ec * binary_entropy(qber)
    return (matchcount/n) * max(0, rate)