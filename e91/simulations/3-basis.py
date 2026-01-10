import numpy as np 
from e91lib import outcome, calc_sval, update_counts, s_uncertainty, finite_key_rate, i_eve

rng = np.random.default_rng()

def three(n, p_dark, det_eff, eta_channel=1.0, eta_degrees=45, f_ec=1.05):
    eta = np.radians(eta_degrees)
    c_eta = np.cos(eta)
    s_eta = np.sin(eta)

    alice_angles = np.radians(np.array([0.0, 45.0, 22.5]))
    bob_angles   = np.radians(np.array([0.0, -22.5, 22.5]))

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
        p_ncnc = (c_eta*np.sin(ra)*np.sin(rb) + s_eta*np.cos(ra)*np.cos(rb))**2

        photon_a = rng.random() < eta_channel
        photon_b = rng.random() < eta_channel
        a_from_photon = photon_a and (rng.random() < det_eff)   # detector click caused by actual photon
        b_from_photon = photon_b and (rng.random() < det_eff)
        dark_a = rng.random() < p_dark      # detector click caused by dark count
        dark_b = rng.random() < p_dark
        a_click = a_from_photon or dark_a
        b_click = b_from_photon or dark_b

        if (a_click and b_click) == False: continue

        if ra == rb and (a_from_photon or b_from_photon):
            matchcount += 1

            if a_from_photon and b_from_photon:
                alice_bit, bob_bit = outcome(r, p_cc, p_cnc, p_ncc)
            elif a_from_photon and dark_b:
                alice_bit, _ = outcome(r, p_cc, p_cnc, p_ncc)
                bob_bit = rng.integers(0, 2)
            elif dark_a and b_from_photon:
                bob_bit, _ = outcome(r, p_cc, p_cnc, p_ncc)
                alice_bit = rng.integers(0, 2)
            else:  # dark + dark
                alice_bit = rng.integers(0, 2)
                bob_bit   = rng.integers(0, 2)
                
            if alice_bit == bob_bit: keylength += 1

        else:
            if not (a_from_photon and b_from_photon): continue
            i = str(a) + str(b)
            if i in store: counts = update_counts(r, i, p_cc, p_cnc, p_ncc, p_ncnc, counts, store)

    if matchcount == 0:
        return 0.0

    s = calc_sval(counts)
    s_eff = s - s_uncertainty(counts)
    qber = 1 - keylength / matchcount
    leaked = i_eve(s_eff)
    key_rate = finite_key_rate(matchcount, n, s_eff, f_ec, qber)

    return s, s_eff, qber, leaked, key_rate