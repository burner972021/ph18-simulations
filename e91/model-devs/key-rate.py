import numpy as np
import random
from math import sqrt, log2
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
runs_per_round_size = 40          # independent experiments per N
round_sizes = np.logspace(3, 6, 12).astype(int)   # fewer points for speed; adjust as desired
p_noise = 0.10                    # depolarizing (white) noise strength
alice_angles = [0.0, 45.0, 22.5]
bob_angles   = [0.0, -22.5, 22.5]

# which (a,b) string pairs are used to produce the raw key (sifting)
# format: f"{a}{b}" where a,b in {0,1,2}
# default: use only '02' as key-generating pair (change if your protocol uses different pair(s))
key_pairs = ['02']

# store mapping used for organizing counts (keeps ability to inspect per pair type)
store = {'02': 0, '01': 1, '12': 2, '11': 3}

ideal_S_value = 2.0 * np.sqrt(2)
classical_bound = 2.0

# -------------------------
# HELPERS
# -------------------------
def binary_entropy(x):
    x = float(np.clip(x, 0.0, 1.0))
    if x == 0.0 or x == 1.0: return 0.0
    return -x*log2(x) - (1.0-x)*log2(1.0-x)

def secret_key_rate_from_S_Q(S, Q):
    """K per sifted raw bit using your formula (clamped safely)."""
    s_over_2 = S / 2.0
    arg = s_over_2**2 - 1.0
    if arg <= 0.0:
        second_prob = 0.5
    else:
        second_prob = (1.0 + np.sqrt(arg)) / 2.0
        second_prob = float(np.clip(second_prob, 0.0, 1.0))
    return 1.0 - binary_entropy(np.clip(Q, 0.0, 1.0)) - binary_entropy(second_prob)

# -------------------------
# SIMULATION: per-trial outcome model (with depolarizing noise applied to each produced pair)
# We'll simulate outcomes (00,01,10,11) consistent with the singlet correlations,
# then apply depolarizing noise by replacing the pair with a uniformly random one with prob p.
# -------------------------
def sample_outcome_pair(a, b, p_depolarize=0.0):
    """
    Return outcome as tuple (bit_a, bit_b) each in {0,1}.
    Underlying probabilities for the singlet-derived model:
      - p_same (00 or 11) and p_diff (01 or 10) derived from angle difference.
    The mapping used matches earlier code: outcomes indices:
      0 -> (0,0)  # cc
      1 -> (0,1)  # cnc
      2 -> (1,0)  # ncc
      3 -> (1,1)  # ncnc
    Depolarizing (white) noise: with probability p replace with uniformly random pair (00/01/10/11).
    """
    ra = alice_angles[a]
    rb = bob_angles[b]
    if ra == rb:
        # in the earlier code these cases were not used; return None to indicate discard
        return None

    diff = abs(ra - rb) * np.pi / 180.0
    p_cc = 0.5 * (np.cos(diff)**2)
    p_cnc = 0.5 * (np.sin(diff)**2)
    p_ncc = 0.5 * (np.sin(diff)**2)
    p_ncnc = 0.5 * (np.cos(diff)**2)

    r = random.random()
    if r < p_cc:
        out = (0, 0)
    elif r < p_cc + p_cnc:
        out = (0, 1)
    elif r < p_cc + p_cnc + p_ncc:
        out = (1, 0)
    else:
        out = (1, 1)

    # depolarizing: with prob p replace with uniform random outcome
    if p_depolarize > 0.0 and random.random() < p_depolarize:
        choice = random.randint(0, 3)
        out = [(0,0),(0,1),(1,0),(1,1)][choice]

    return out

# -------------------------
# MAIN: compute for each N:
#   - number of sifted rounds (rounds with pair in key_pairs)
#   - number and fraction of matching bits among sifted rounds
#   - same for noisy case (dep applied)
#   - also compute S, Q, and K per sifted bit (for info)
# -------------------------
results = []

for N in round_sizes:
    # We'll run runs_per_round_size independent experiments for each N to get mean+std
    sifts_ideal = []
    matches_ideal = []
    sifts_noisy = []
    matches_noisy = []
    S_samples = []     # for ideal S computation (per experiment)
    S_noisy_samples = []

    for run in range(runs_per_round_size):
        # counters for this run
        sift_count_ideal = 0
        match_count_ideal = 0

        sift_count_noisy = 0
        match_count_noisy = 0

        # For S computation, we also need the same E-structure as before (counts per store mapping)
        counts_ideal = [[0]*4 for _ in range(4)]
        counts_noisy = [[0]*4 for _ in range(4)]

        for _ in range(N):
            a = random.randint(0, 2)
            b = random.randint(0, 2)
            key = f"{a}{b}"

            # IDEAL outcome (no dep)
            out_ideal = sample_outcome_pair(a, b, p_depolarize=0.0)
            # NOISY outcome (apply depolarizing to outcome)
            out_noisy = sample_outcome_pair(a, b, p_depolarize=p_noise)

            # Fill counts for E calculation if this pair is one of the tracked store keys
            if key in store and out_ideal is not None:
                # map outcome to index 0..3 as before
                if out_ideal == (0,0):
                    counts_ideal[store[key]][0] += 1
                elif out_ideal == (0,1):
                    counts_ideal[store[key]][1] += 1
                elif out_ideal == (1,0):
                    counts_ideal[store[key]][2] += 1
                elif out_ideal == (1,1):
                    counts_ideal[store[key]][3] += 1

            if key in store and out_noisy is not None:
                if out_noisy == (0,0):
                    counts_noisy[store[key]][0] += 1
                elif out_noisy == (0,1):
                    counts_noisy[store[key]][1] += 1
                elif out_noisy == (1,0):
                    counts_noisy[store[key]][2] += 1
                elif out_noisy == (1,1):
                    counts_noisy[store[key]][3] += 1

            # SIFTING: if this pair is in key_pairs, update sift & match counters
            if key in key_pairs and out_ideal is not None:
                sift_count_ideal += 1
                # count as matching if bits equal (00 or 11)
                if out_ideal[0] == out_ideal[1]:
                    match_count_ideal += 1

            if key in key_pairs and out_noisy is not None:
                sift_count_noisy += 1
                if out_noisy[0] == out_noisy[1]:
                    match_count_noisy += 1

        # compute S from counts (safe division)
        def S_from_counts(counts):
            E = []
            for row in counts:
                tot = sum(row)
                if tot == 0:
                    E.append(0.0)
                else:
                    E.append((row[0] + row[3] - row[1] - row[2]) / tot)
            e13, e12, e23, e22 = E
            return e13 + e12 + e23 - e22

        S_run_ideal = S_from_counts(counts_ideal)
        S_run_noisy = S_from_counts(counts_noisy)

        S_samples.append(S_run_ideal)
        S_noisy_samples.append(S_run_noisy)

        sifts_ideal.append(sift_count_ideal)
        matches_ideal.append(match_count_ideal)
        sifts_noisy.append(sift_count_noisy)
        matches_noisy.append(match_count_noisy)

    # aggregate over runs
    sifts_ideal = np.array(sifts_ideal)
    matches_ideal = np.array(matches_ideal)
    sifts_noisy = np.array(sifts_noisy)
    matches_noisy = np.array(matches_noisy)

    S_samples = np.array(S_samples)
    S_noisy_samples = np.array(S_noisy_samples)

    # mean and std over independent runs
    mean_sift_ideal = np.mean(sifts_ideal)
    std_sift_ideal = np.std(sifts_ideal)
    mean_match_ideal = np.mean(matches_ideal)
    std_match_ideal = np.std(matches_ideal)

    mean_sift_noisy = np.mean(sifts_noisy)
    mean_match_noisy = np.mean(matches_noisy)

    # fraction of matching among sifted = matches / sifts (avoid divide by zero)
    match_frac_ideal = (mean_match_ideal / mean_sift_ideal) if mean_sift_ideal > 0 else 0.0
    match_frac_noisy = (mean_match_noisy / mean_sift_noisy) if mean_sift_noisy > 0 else 0.0

    # QBER (alternative): fraction mismatched among sifted
    Q_ideal = 1.0 - match_frac_ideal
    Q_noisy = 1.0 - match_frac_noisy

    # estimate S stats
    mean_S_ideal = float(np.mean(S_samples))
    std_S_ideal = float(np.std(S_samples))
    mean_S_noisy = float(np.mean(S_noisy_samples))
    std_S_noisy = float(np.std(S_noisy_samples))

    # secret key rate per sifted raw bit (K) using your formula (just for reference)
    K_ideal = secret_key_rate_from_S_Q(mean_S_ideal, Q_ideal)
    K_noisy = secret_key_rate_from_S_Q(mean_S_noisy, Q_noisy)

    results.append({
        'N': int(N),
        'mean_sifted_ideal': mean_sift_ideal,
        'mean_matched_ideal': mean_match_ideal,
        'match_frac_ideal': match_frac_ideal,
        'mean_sifted_noisy': mean_sift_noisy,
        'mean_matched_noisy': mean_match_noisy,
        'match_frac_noisy': match_frac_noisy,
        'mean_S_ideal': mean_S_ideal, 'std_S_ideal': std_S_ideal,
        'mean_S_noisy': mean_S_noisy, 'std_S_noisy': std_S_noisy,
        'Q_ideal': Q_ideal, 'Q_noisy': Q_noisy,
        'K_ideal': K_ideal, 'K_noisy': K_noisy
    })

    # quick print for progress
    print(f"N={N:7d} | sifted (ideal) ~ {mean_sift_ideal:.1f} matched ~ {mean_match_ideal:.1f} "
          f"match_frac={match_frac_ideal:.4f} | noisy match_frac={match_frac_noisy:.4f} "
          f"| mean S ideal={mean_S_ideal:.4f} noisy={mean_S_noisy:.4f}")

# -------------------------
# SUMMARY TABLE (print)
# -------------------------
print("\nSummary (per N):")
print(" N    sifted_ideal  matched_ideal  match_frac_ideal   sifted_noisy  matched_noisy  match_frac_noisy   K_ideal  K_noisy")
for r in results:
    print(f"{r['N']:6d}  {r['mean_sifted_ideal']:12.1f}  {r['mean_matched_ideal']:13.1f}  {r['match_frac_ideal']:14.4f}  "
          f"{r['mean_sifted_noisy']:12.1f}  {r['mean_matched_noisy']:13.1f}  {r['match_frac_noisy']:15.4f}  "
          f"{r['K_ideal']:+7.4f}  {r['K_noisy']:+7.4f}")

# -------------------------
# OPTIONAL: plot match fractions vs N
# -------------------------
Ns = [r['N'] for r in results]
match_ideal = [r['match_frac_ideal'] for r in results]
match_noisy = [r['match_frac_noisy'] for r in results]

plt.figure(figsize=(8,4))
plt.semilogx(Ns, match_ideal, 'o-', label='match fraction (ideal)')
plt.semilogx(Ns, match_noisy, 's--', label=f'match fraction (noisy p={p_noise})')
plt.xlabel('Rounds N')
plt.ylabel('Fraction of sifted bits that match (after sifting)')
plt.title(f'Matching fraction among sifted key bits (key_pairs={key_pairs})')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.show()
