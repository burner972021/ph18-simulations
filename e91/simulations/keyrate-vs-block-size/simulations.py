import numpy as np
from e91lib import calc_sval, update_counts, s_uncertainty, finite_key_rate, i_eve

rng = np.random.default_rng()

def three(alice_error, bob_error, n, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([0.0, 45.0, 22.5]))
    bob_angles   = np.radians(np.array([0.0, -22.5, 22.5]))

    matchcount = 0
    keylength = 0

    a1b3 = [0] * 4
    a1b2 = [0] * 4
    a2b3 = [0] * 4
    a2b2 = [0] * 4

    counts = [a1b3, a1b2, a2b3, a2b2]
    store = {'02': 0, '01': 1, '12': 2, '11': 3}

    for _ in range(n):
        a = rng.integers(0, 3)
        b = rng.integers(0, 3)

        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc   = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc  = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc  = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2
        p_ncnc = (c_eta*np.sin(ra)*np.sin(rb) + s_eta*np.cos(ra)*np.cos(rb))**2

        if ra == rb:
            matchcount += 1

            if r < p_cc:
                alice_bit = 1
                bob_bit = 1
            elif r < p_cc + p_cnc:
                alice_bit = 1
                bob_bit = 0
            elif r < p_cc + p_cnc + p_ncc:
                alice_bit = 0
                bob_bit = 1
            else:
                alice_bit = 0
                bob_bit = 0

            if rng.random() < alice_error:
                alice_bit ^= 1
            if rng.random() < bob_error:
                bob_bit ^= 1

            if alice_bit == bob_bit:
                keylength += 1

        else:
            i = str(a) + str(b)
            if i in store:
                counts = update_counts(
                    r, i,
                    p_cc, p_cnc, p_ncc, p_ncnc,
                    counts, store
                )

    s = calc_sval(counts)
    s_delta = s_uncertainty(counts)
    s_eff = s - s_delta

    qber = 1 - keylength / matchcount
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate

def four(alice_error, bob_error, n, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0]))
    bob_angles   = np.radians(np.array([-22.5, 0.0, 22.5, 45.0]))

    keylength = 0
    matchcount = 0

    s1_counts = np.zeros((4, 4))
    s2_counts = np.zeros((4, 4))

    store2 = {'01': 0, '21': 1, '23': 2, '03': 3}
    store1 = {'10': 0, '12': 1, '32': 2, '30': 3}

    for _ in range(n):
        a = rng.integers(0, 4)
        b = rng.integers(0, 4)

        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc   = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc  = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc  = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2
        p_ncnc = (c_eta*np.sin(ra)*np.sin(rb) + s_eta*np.cos(ra)*np.cos(rb))**2

        if ra == rb:
            matchcount += 1

            if r < p_cc:
                alice_bit = 1; bob_bit = 1
            elif r < p_cc + p_cnc:
                alice_bit = 1; bob_bit = 0
            elif r < p_cc + p_cnc + p_ncc:
                alice_bit = 0; bob_bit = 1
            else:
                alice_bit = 0; bob_bit = 0

            if rng.random() < alice_error:
                alice_bit ^= 1
            if rng.random() < bob_error:
                bob_bit ^= 1

            if alice_bit == bob_bit:
                keylength += 1

        else:
            i = str(a) + str(b)
            if i in store1:
                s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s1_counts, store1)
            if i in store2:
                s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s2_counts, store2)

    s1 = calc_sval(s1_counts)
    s2 = calc_sval(s2_counts)
    s = (s1 + s2) / 2

    s_delta = (s_uncertainty(s1_counts) + s_uncertainty(s2_counts)) / 2
    s_eff = s - s_delta

    qber = 1 - keylength / matchcount
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate

def five(alice_error, bob_error, n, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5]))
    bob_angles   = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5]))

    keylength = 0
    matchcount = 0

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

        p_cc   = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc  = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc  = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2
        p_ncnc = (c_eta*np.sin(ra)*np.sin(rb) + s_eta*np.cos(ra)*np.cos(rb))**2

        if ra == rb:
            matchcount += 1

            if r < p_cc:
                alice_bit = 1; bob_bit = 1
            elif r < p_cc + p_cnc:
                alice_bit = 1; bob_bit = 0
            elif r < p_cc + p_cnc + p_ncc:
                alice_bit = 0; bob_bit = 1
            else:
                alice_bit = 0; bob_bit = 0

            if rng.random() < alice_error:
                alice_bit ^= 1
            if rng.random() < bob_error:
                bob_bit ^= 1

            if alice_bit == bob_bit:
                keylength += 1

        else:
            i = str(a) + str(b)
            if i in store1:
                s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s1_counts, store1)
            if i in store2:
                s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s2_counts, store2)
            if i in store3:
                s3_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s3_counts, store3)

    s1 = calc_sval(s1_counts)
    s2 = calc_sval(s2_counts)
    s3 = calc_sval(s3_counts)
    s = (s1 + s2 + s3) / 3

    s_delta = (
        s_uncertainty(s1_counts)
        + s_uncertainty(s2_counts)
        + s_uncertainty(s3_counts)
    ) / 3

    s_eff = s - s_delta
    qber = 1 - keylength / matchcount
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate

def six(alice_error, bob_error, n, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5, 90.0]))
    bob_angles   = np.radians(np.array([-22.5, 0.0, 22.5, 45.0, 67.5, 90.0]))

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
        a = rng.integers(0, 6)
        b = rng.integers(0, 6)

        ra = alice_angles[a]
        rb = bob_angles[b]
        r = rng.random()

        p_cc   = (c_eta*np.cos(ra)*np.cos(rb) + s_eta*np.sin(ra)*np.sin(rb))**2
        p_cnc  = (-c_eta*np.cos(ra)*np.sin(rb) + s_eta*np.sin(ra)*np.cos(rb))**2
        p_ncc  = (-c_eta*np.sin(ra)*np.cos(rb) + s_eta*np.cos(ra)*np.sin(rb))**2
        p_ncnc = (c_eta*np.sin(ra)*np.sin(rb) + s_eta*np.cos(ra)*np.cos(rb))**2

        if ra == rb:
            matchcount += 1

            if r < p_cc:
                alice_bit = 1; bob_bit = 1
            elif r < p_cc + p_cnc:
                alice_bit = 1; bob_bit = 0
            elif r < p_cc + p_cnc + p_ncc:
                alice_bit = 0; bob_bit = 1
            else:
                alice_bit = 0; bob_bit = 0

            if rng.random() < alice_error:
                alice_bit ^= 1
            if rng.random() < bob_error:
                bob_bit ^= 1

            if alice_bit == bob_bit:
                keylength += 1

        else:
            i = str(a) + str(b)
            if i in store1:
                s1_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s1_counts, store1)
            if i in store2:
                s2_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s2_counts, store2)
            if i in store3:
                s3_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s3_counts, store3)
            if i in store4:
                s4_counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, s4_counts, store4)

    s1 = calc_sval(s1_counts)
    s2 = calc_sval(s2_counts)
    s3 = calc_sval(s3_counts)
    s4 = calc_sval(s4_counts)

    s = (s1 + s2 + s3 + s4) / 4

    s_delta = (
        s_uncertainty(s1_counts)
        + s_uncertainty(s2_counts)
        + s_uncertainty(s3_counts)
        + s_uncertainty(s4_counts)
    ) / 4

    s_eff = s - s_delta
    qber = 1 - keylength / matchcount
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return key_rate
