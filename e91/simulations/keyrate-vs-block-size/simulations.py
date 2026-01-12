import numpy as np 
from e91lib import outcome, rand_outcome, calc_sval, update_counts, s_uncertainty, finite_key_rate

rng = np.random.default_rng()

def three(n, p_dark, det_eff, p_flip, eta_channel=0.9, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([0.0, 45.0, 22.5]))
    bob_angles = np.radians(np.array([0.0, -22.5, 22.5]))

    matchcount = 0
    keylength = 0

    counts = np.zeros((4, 4))
    store = {'02': 0, '01': 1, '12': 2, '11': 3}

    for _ in range(n):
        a = rng.integers(0, 3)
        b = rng.integers(0, 3)
        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2

        photon_a = rng.random() < eta_channel
        photon_b = rng.random() < eta_channel
        a_from_photon = photon_a and (rng.random() < det_eff)   # detector click caused by actual photon
        b_from_photon = photon_b and (rng.random() < det_eff)
        dark_a = rng.random() < p_dark      # detector click caused by dark count
        dark_b = rng.random() < p_dark
        a_click = a_from_photon or dark_a
        b_click = b_from_photon or dark_b

        if ra == rb and a_click and b_click:
            matchcount += 1
            if a_from_photon and b_from_photon: 
                alice_bit, bob_bit = outcome(r, p_cc, p_cnc, p_ncc)
                if rng.random() < p_flip: alice_bit ^= 1
                if rng.random() < p_flip: bob_bit ^= 1
            else: alice_bit, bob_bit = rand_outcome()
            if alice_bit == bob_bit: keylength += 1

        else:
            if not (a_click and b_click): continue
            i = str(a) + str(b)
            if i in store:
                if a_from_photon and b_from_photon: counts = update_counts(r, i, p_cc, p_cnc, p_ncc, counts, store)
                else:
                    ab = rng.integers(0, 4)
                    counts[store[i]][ab] += 1

    if matchcount == 0: return 0.0

    s = calc_sval(counts)
    s_eff = s - s_uncertainty(counts)
    qber = 1 - keylength / matchcount
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate


def four(n, p_dark, det_eff, p_flip, eta_channel=0.9, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0]))
    bob_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0]))

    matchcount = 0
    keylength = 0

    s1_counts = np.zeros((4, 4))
    s2_counts = np.zeros((4, 4))
    store1 = {'10': 0, '12': 1, '32': 2, '30': 3}
    store2 = {'01': 0, '21': 1, '23': 2, '03': 3}

    for _ in range(n):
        a = rng.integers(0, 4)
        b = rng.integers(0, 4)

        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2

        photon_a = rng.random() < eta_channel
        photon_b = rng.random() < eta_channel
        a_from_photon = photon_a and (rng.random() < det_eff)   # detector click caused by actual photon
        b_from_photon = photon_b and (rng.random() < det_eff)
        dark_a = rng.random() < p_dark      # detector click caused by dark count
        dark_b = rng.random() < p_dark
        a_click = a_from_photon or dark_a
        b_click = b_from_photon or dark_b

        if ra == rb and a_click and b_click:
            matchcount += 1
            if a_from_photon and b_from_photon: 
                alice_bit, bob_bit = outcome(r, p_cc, p_cnc, p_ncc)
                if rng.random() < p_flip: alice_bit ^= 1
                if rng.random() < p_flip: bob_bit ^= 1
            else: alice_bit, bob_bit = rand_outcome()
            if alice_bit == bob_bit: keylength += 1

        else:
            if not (a_click and b_click):
                continue
            i = str(a) + str(b)
            if i in store1:
                if a_from_photon and b_from_photon:
                    s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s1_counts, store1)
                else:
                    ab = rng.integers(0, 4)
                    s1_counts[store1[i]][ab] += 1
            if i in store2:
                if a_from_photon and b_from_photon:
                    s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s2_counts, store2)
                else:
                    ab = rng.integers(0, 4)
                    s2_counts[store2[i]][ab] += 1
                
    if matchcount == 0:
        return 0.0

    s = (calc_sval(s1_counts) + calc_sval(s2_counts))/2
    s_eff = s - (s_uncertainty(s1_counts) + s_uncertainty(s2_counts))/2
    qber = 1 - keylength / matchcount if matchcount > 0 else np.nan
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate


def five(n, p_dark, det_eff, p_flip, eta_channel=0.9, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5]))
    bob_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5]))

    matchcount = 0
    keylength = 0

    s1_counts = np.zeros((4, 4))
    s2_counts = np.zeros((4, 4))
    s3_counts = np.zeros((4, 4))
    store1 = {'10': 0, '12': 1, '32': 2, '30': 3}
    store2 = {'01': 0, '21': 1, '23': 2, '03': 3}
    store3 = {'10': 0, '32': 1, '34': 2, '14': 3}

    for _ in range(n):
        a = rng.integers(0, 5)
        b = rng.integers(0, 5)

        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2

        photon_a = rng.random() < eta_channel
        photon_b = rng.random() < eta_channel
        a_from_photon = photon_a and (rng.random() < det_eff)   # detector click caused by actual photon
        b_from_photon = photon_b and (rng.random() < det_eff)
        dark_a = rng.random() < p_dark      # detector click caused by dark count
        dark_b = rng.random() < p_dark
        a_click = a_from_photon or dark_a
        b_click = b_from_photon or dark_b

        if ra == rb and a_click and b_click:
            matchcount += 1
            if a_from_photon and b_from_photon: 
                alice_bit, bob_bit = outcome(r, p_cc, p_cnc, p_ncc)
                if rng.random() < p_flip: alice_bit ^= 1
                if rng.random() < p_flip: bob_bit ^= 1
            else: alice_bit, bob_bit = rand_outcome()
            if alice_bit == bob_bit: keylength += 1

        else:
            if not (a_click and b_click):
                continue
            i = str(a) + str(b)
            if i in store1:
                if a_from_photon and b_from_photon:
                    s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s1_counts, store1)
                else:
                    ab = rng.integers(0, 4)
                    s1_counts[store1[i]][ab] += 1
            if i in store2:
                if a_from_photon and b_from_photon:
                    s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s2_counts, store2)
                else:
                    ab = rng.integers(0, 4)
                    s2_counts[store2[i]][ab] += 1
            if i in store3:
                if a_from_photon and b_from_photon:
                    s3_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s3_counts, store3)
                else:
                    ab = rng.integers(0, 4)
                    s3_counts[store3[i]][ab] += 1  
                
    if matchcount == 0:
        return 0.0

    s = (calc_sval(s1_counts) + calc_sval(s2_counts) + calc_sval(s3_counts))/3
    s_eff = s - ((s_uncertainty(s1_counts) + s_uncertainty(s2_counts) + s_uncertainty(s3_counts))/3)
    qber = 1 - keylength / matchcount if matchcount > 0 else np.nan
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate


def six(n, p_dark, det_eff, p_flip, eta_channel=0.9, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5, 90]))
    bob_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5, 90]))

    matchcount = 0
    keylength = 0

    s1_counts = np.zeros((4, 4))
    s2_counts = np.zeros((4, 4))
    s3_counts = np.zeros((4, 4))
    s4_counts = np.zeros((4, 4))
    store1 = {'10': 0, '12': 1, '32': 2, '30': 3}
    store2 = {'01': 0, '21': 1, '23': 2, '03': 3}
    store3 = {'10': 0, '32': 1, '34': 2, '14': 3}
    store4 = {'12': 0, '23': 1, '45': 2, '25': 3}

    for _ in range(n):
        a = rng.integers(0, 6)
        b = rng.integers(0, 6)

        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2

        photon_a = rng.random() < eta_channel
        photon_b = rng.random() < eta_channel
        a_from_photon = photon_a and (rng.random() < det_eff)   # detector click caused by actual photon
        b_from_photon = photon_b and (rng.random() < det_eff)
        dark_a = rng.random() < p_dark      # detector click caused by dark count
        dark_b = rng.random() < p_dark
        a_click = a_from_photon or dark_a
        b_click = b_from_photon or dark_b

        if ra == rb and a_click and b_click:
            matchcount += 1
            if a_from_photon and b_from_photon: 
                alice_bit, bob_bit = outcome(r, p_cc, p_cnc, p_ncc)
                if rng.random() < p_flip: alice_bit ^= 1
                if rng.random() < p_flip: bob_bit ^= 1
            else: alice_bit, bob_bit = rand_outcome()
            if alice_bit == bob_bit: keylength += 1

        else:
            if not (a_click and b_click):
                continue
            i = str(a) + str(b)
            if i in store1:
                if a_from_photon and b_from_photon:
                    s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s1_counts, store1)
                else:
                    ab = rng.integers(0, 4)
                    s1_counts[store1[i]][ab] += 1
            if i in store2:
                if a_from_photon and b_from_photon:
                    s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s2_counts, store2)
                else:
                    ab = rng.integers(0, 4)
                    s2_counts[store2[i]][ab] += 1
            if i in store3:
                if a_from_photon and b_from_photon:
                    s3_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s3_counts, store3)
                else:
                    ab = rng.integers(0, 4)
                    s3_counts[store3[i]][ab] += 1  
            if i in store4:
                if a_from_photon and b_from_photon:
                    s4_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, s4_counts, store4)
                else:
                    ab = rng.integers(0, 4)
                    s4_counts[store4[i]][ab] += 1  
                
    if matchcount == 0:
        return 0.0

    s = (calc_sval(s1_counts) + calc_sval(s2_counts) + calc_sval(s3_counts) + calc_sval(s4_counts))/4
    s_eff = s - ((s_uncertainty(s1_counts) + s_uncertainty(s2_counts) + s_uncertainty(s3_counts) + s_uncertainty(s4_counts))/4)
    qber = 1 - keylength / matchcount if matchcount > 0 else np.nan
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate