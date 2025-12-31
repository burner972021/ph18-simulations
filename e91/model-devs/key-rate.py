import numpy as np
import random
import matplotlib.pyplot as plt
from e91lib import binary_entropy

runs_per_round_size = 40         
round_sizes = np.logspace(3, 6, 12).astype(int)  
p_noise = 0.10                    # white noise strength
alice_angles = [0.0, 45.0, 22.5]
bob_angles   = [0.0, -22.5, 22.5]

key_pairs = ['02']

store = {'02': 0, '01': 1, '12': 2, '11': 3}

ideal_S_value = 2.0 * np.sqrt(2)
classical_bound = 2.0

def secret_key_rate_from_S_Q(S, Q):
    s_over_2 = S / 2.0
    arg = s_over_2**2 - 1.0
    if arg <= 0.0:
        second_prob = 0.5
    else:
        second_prob = (1.0 + np.sqrt(arg)) / 2.0
        second_prob = float(np.clip(second_prob, 0.0, 1.0))
    return 1.0 - binary_entropy(np.clip(Q, 0.0, 1.0)) - binary_entropy(second_prob)


def sample_outcome_pair(a, b, p_depolarize=0.0):
    ra = alice_angles[a]
    rb = bob_angles[b]
    if ra == rb:
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

    if p_depolarize > 0.0 and random.random() < p_depolarize:
        choice = random.randint(0, 3)
        out = [(0,0),(0,1),(1,0),(1,1)][choice]

    return out

results = []

for N in round_sizes:
    sifts_ideal = []
    matches_ideal = []
    sifts_noisy = []
    matches_noisy = []
    S_samples = []  
    S_noisy_samples = []

    for run in range(runs_per_round_size):
        sift_count_ideal = 0
        match_count_ideal = 0

        sift_count_noisy = 0
        match_count_noisy = 0

        counts_ideal = [[0]*4 for _ in range(4)]
        counts_noisy = [[0]*4 for _ in range(4)]

        for _ in range(N):
            a = random.randint(0, 2)
            b = random.randint(0, 2)
            key = f"{a}{b}"

            out_ideal = sample_outcome_pair(a, b, p_depolarize=0.0)
            out_noisy = sample_outcome_pair(a, b, p_depolarize=p_noise)

            if key in store and out_ideal is not None:
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

            if key in key_pairs and out_ideal is not None:
                sift_count_ideal += 1
                if out_ideal[0] == out_ideal[1]:
                    match_count_ideal += 1

            if key in key_pairs and out_noisy is not None:
                sift_count_noisy += 1
                if out_noisy[0] == out_noisy[1]:
                    match_count_noisy += 1

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

    sifts_ideal = np.array(sifts_ideal)
    matches_ideal = np.array(matches_ideal)
    sifts_noisy = np.array(sifts_noisy)
    matches_noisy = np.array(matches_noisy)

    S_samples = np.array(S_samples)
    S_noisy_samples = np.array(S_noisy_samples)

    mean_sift_ideal = np.mean(sifts_ideal)
    std_sift_ideal = np.std(sifts_ideal)
    mean_match_ideal = np.mean(matches_ideal)
    std_match_ideal = np.std(matches_ideal)

    mean_sift_noisy = np.mean(sifts_noisy)
    mean_match_noisy = np.mean(matches_noisy)

    match_frac_ideal = (mean_match_ideal / mean_sift_ideal) if mean_sift_ideal > 0 else 0.0
    match_frac_noisy = (mean_match_noisy / mean_sift_noisy) if mean_sift_noisy > 0 else 0.0

    Q_ideal = 1.0 - match_frac_ideal
    Q_noisy = 1.0 - match_frac_noisy

    # estimate S 
    mean_S_ideal = float(np.mean(S_samples))
    std_S_ideal = float(np.std(S_samples))
    mean_S_noisy = float(np.mean(S_noisy_samples))
    std_S_noisy = float(np.std(S_noisy_samples))

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

    print(f"N={N:7d} | sifted (ideal) ~ {mean_sift_ideal:.1f} matched ~ {mean_match_ideal:.1f} "
          f"match_frac={match_frac_ideal:.4f} | noisy match_frac={match_frac_noisy:.4f} "
          f"| mean S ideal={mean_S_ideal:.4f} noisy={mean_S_noisy:.4f}")


print("\nSummary (per N):")
print(" N    sifted_ideal  matched_ideal  match_frac_ideal   sifted_noisy  matched_noisy  match_frac_noisy   K_ideal  K_noisy")
for r in results:
    print(f"{r['N']:6d}  {r['mean_sifted_ideal']:12.1f}  {r['mean_matched_ideal']:13.1f}  {r['match_frac_ideal']:14.4f}  "
          f"{r['mean_sifted_noisy']:12.1f}  {r['mean_matched_noisy']:13.1f}  {r['match_frac_noisy']:15.4f}  "
          f"{r['K_ideal']:+7.4f}  {r['K_noisy']:+7.4f}")


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
