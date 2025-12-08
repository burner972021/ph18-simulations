import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt, log2

# -------------------------
# PARAMETERS (tweak here)
# -------------------------
runs_per_round_size = 50           # independent experiments per N (increase for smoother stats)
round_sizes = np.logspace(3, 6, 20).astype(int)  # N from 1e3 to 1e6 (20 points)
p_noise = 0.10                     # depolarizing (white) noise strength (0 = ideal, 1 = fully mixed)

# angles (degrees) matching your earlier code (order matters for store mapping)
alice_angles = [0.0, 45.0, 22.5]
bob_angles   = [0.0, -22.5, 22.5]

# mapping of pair keys to CHSH correlator rows (keeps consistency with your earlier code)
store = {'02': 0, '01': 1, '12': 2, '11': 3}

ideal_S_value = 2.0 * np.sqrt(2)   # ideal maximum

# classical/local-realistic bound
classical_bound = 2.0

# sigma thresholds to check (kσ)
k_values = [3, 5, 7, 10]

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def binary_entropy(x):
    """Binary entropy in bits, safe at x=0 or 1."""
    x = float(np.clip(x, 0.0, 1.0))
    if x == 0.0 or x == 1.0:
        return 0.0
    return -x*log2(x) - (1.0-x)*log2(1.0-x)

def compute_S_single_rounds(rounds):
    """Monte Carlo simulation returning the CHSH S value for 'rounds' trials (ideal E91 sampling)."""
    # counts per store entry: 4 rows (pair types) x 4 outcomes (00,01,10,11) stored as 0..3
    counts = [[0]*4 for _ in range(4)]

    for _ in range(rounds):
        a = random.randint(0, 2)
        b = random.randint(0, 2)
        ra = alice_angles[a]
        rb = bob_angles[b]

        # Only consider the relevant measurement pairs (this mirrors your earlier code)
        if ra != rb:
            diff = np.radians(abs(ra - rb))
            # Using standard singlet probabilities for outcome agreement/disagreement:
            # p_cc = 0.5 * cos^2(diff), p_cnc = 0.5 * sin^2(diff), ...
            p_cc = 0.5 * (np.cos(diff)**2)
            p_cnc = 0.5 * (np.sin(diff)**2)
            p_ncc = 0.5 * (np.sin(diff)**2)
            p_ncnc = 0.5 * (np.cos(diff)**2)

            r = random.random()
            key = f"{a}{b}"
            if key in store:
                idx = store[key]
                if r < p_cc:
                    counts[idx][0] += 1
                elif r < p_cc + p_cnc:
                    counts[idx][1] += 1
                elif r < p_cc + p_cnc + p_ncc:
                    counts[idx][2] += 1
                else:
                    counts[idx][3] += 1

    # compute correlators E for each of the 4 pair types (safe division)
    E = []
    for row in counts:
        tot = sum(row)
        if tot == 0:
            E.append(0.0)
        else:
            E.append((row[0] + row[3] - row[1] - row[2]) / tot)

    e13, e12, e23, e22 = E
    S = e13 + e12 + e23 - e22
    return S

def apply_depolarizing_to_S(S_values, p):
    """
    For white (depolarizing) noise model we assume S scales linearly:
    S_noisy = (1 - p) * S_ideal_sample
    This is the standard behavior for mixing with white noise.
    """
    return (1.0 - p) * np.array(S_values)

def S_to_visibility(S):
    """Visibility V from S via S = 2*sqrt(2) * V"""
    return S / (2.0 * np.sqrt(2))

# -------------------------
# RAW KEY (SIFTED) MATCHING RATE
# -------------------------

def simulate_matching_rate(N, p_noise):
    """
    Simulate the fraction of matching key bits for N rounds.
    Sift only (0,0) and (2,2) basis matches.
    Matching probability = cos^2(angle difference).
    White noise: outcomes are random with prob p_noise.
    """
    sifted = 0
    matched = 0

    for _ in range(N):
        a = random.randint(0, 2)
        b = random.randint(0, 2)

        # only keep same-basis: (0,0) and (2,2)
        if not ((a == 0 and b == 0) or (a == 2 and b == 2)):
            continue

        sifted += 1

        ra = alice_angles[a]
        rb = bob_angles[b]
        diff = np.radians(abs(ra - rb))

        # ideal quantum correlation: prob(match) = cos^2(diff)
        p_match = (np.cos(diff) ** 2)

        # depolarizing noise: replace result with random bit with prob p_noise
        if random.random() < p_noise:
            # noisy: 50% chance they match
            if random.random() < 0.5:
                matched += 1
        else:
            # ideal behavior
            if random.random() < p_match:
                matched += 1

    # avoid division by zero
    if sifted == 0:
        return 0.0

    return matched / sifted, sifted


# -------------------------
# MAIN EXPERIMENT LOOP
# -------------------------
# We'll collect per-N stats for both ideal and noisy cases
results = {
    'N': [],
    'ideal_mean_S': [], 'ideal_std_S': [],
    'noisy_mean_S': [], 'noisy_std_S': [],
    # QBERs and keyrates (means computed from mean S)
    'ideal_Q': [], 'noisy_Q': [],
    'ideal_keyrate': [], 'noisy_keyrate': [],
    # sigma bands (store only 1..10 sigma for plotting convenience)
    'ideal_sigmas': {k: [] for k in range(1, 11)},
    'noisy_sigmas': {k: [] for k in range(1, 11)}
}

# track smallest N meeting security criterion mean - k*sigma > 2 for each k, for ideal and noisy
security_passing = {
    'ideal': {k: None for k in k_values},
    'noisy': {k: None for k in k_values}
}

print("Starting experiments: runs_per_round_size =", runs_per_round_size)
for N in round_sizes:
    S_samples = [compute_S_single_rounds(N) for _ in range(runs_per_round_size)]
    S_samples = np.array(S_samples, dtype=float)

    # ideal stats
    mean_S_ideal = np.mean(S_samples)
    std_S_ideal = np.std(S_samples, ddof=0)  # population std

    # noisy: apply depolarizing scaling to each sample
    S_noisy_samples = apply_depolarizing_to_S(S_samples, p_noise)
    mean_S_noisy = float(np.mean(S_noisy_samples))
    std_S_noisy = float(np.std(S_noisy_samples, ddof=0))

    # store
    results['N'].append(N)
    results['ideal_mean_S'].append(mean_S_ideal)
    results['ideal_std_S'].append(std_S_ideal)
    results['noisy_mean_S'].append(mean_S_noisy)
    results['noisy_std_S'].append(std_S_noisy)

    # sigma bands
    for k in range(1, 11):
        results['ideal_sigmas'][k].append(k * std_S_ideal)
        results['noisy_sigmas'][k].append(k * std_S_noisy)

    # compute matching rate AFTER SIFTING
    match_frac_ideal, sifted_ideal = simulate_matching_rate(N, p_noise=0.0)
    match_frac_noisy, sifted_noisy = simulate_matching_rate(N, p_noise=p_noise)

    results['ideal_keyrate'].append(match_frac_ideal)
    results['noisy_keyrate'].append(match_frac_noisy)


    print(f"N={N:7d}  ideal: mean S={mean_S_ideal:.4f}, std={std_S_ideal:.4f}  "
          f"noisy(p={p_noise}): mean S={mean_S_noisy:.4f}, std={std_S_noisy:.4f}")

# -------------------------
# POST PROCESS: find min N for each k (report)
# -------------------------
print("\nMinimum N such that <S> - kσ > 2 (classical bound):")
for k in k_values:
    ide = security_passing['ideal'][k]
    noi = security_passing['noisy'][k]
    print(f"  k={k:2d}: ideal -> {ide if ide is not None else 'not reached in tested N'}, "
          f"noisy(p={p_noise}) -> {noi if noi is not None else 'not reached in tested N'}")

# -------------------------
# PLOTTING
# -------------------------
N_array = np.array(results['N'])

plt.figure(figsize=(14, 10))

# 1) Deviation from ideal S (mean) vs N (ideal and noisy)
plt.subplot(2, 2, 1)
plt.loglog(N_array, np.abs(np.array(results['ideal_mean_S']) - ideal_S_value), 'o-', label='|<S>_ideal - 2√2|')
plt.loglog(N_array, np.abs(np.array(results['noisy_mean_S']) - ideal_S_value), 's--', label=f'|<S>_noisy(p={p_noise}) - 2√2|')
plt.xlabel('Rounds N')
plt.ylabel('|⟨S⟩ − 2√2|')
plt.title('Convergence: deviation from ideal S (log-log)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)

# 2) Std dev and sigma bands (ideal vs noisy)
plt.subplot(2, 2, 2)
plt.loglog(N_array, results['ideal_std_S'], 'o-', label='ideal 1σ')
plt.loglog(N_array, results['noisy_std_S'], 's--', label=f'noisy 1σ (p={p_noise})')
# plot 2σ,5σ and 10σ for visibility
for k in [2, 5, 10]:
    plt.loglog(N_array, results['ideal_sigmas'][k], linestyle=':', label=f'ideal {k}σ')
    plt.loglog(N_array, results['noisy_sigmas'][k], linestyle='--', label=f'noisy {k}σ' if k==2 else None)
plt.xlabel('Rounds N')
plt.ylabel('Standard deviation of S (and kσ bands)')
plt.title('σ bands vs dataset size')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)

# 3) Mean S (ideal vs noisy) with 5σ band shading
plt.subplot(2, 2, 3)
plt.semilogx(N_array, results['ideal_mean_S'], 'o-', label='ideal ⟨S⟩')
plt.fill_between(N_array,
                 np.array(results['ideal_mean_S']) - 5*np.array(results['ideal_std_S']),
                 np.array(results['ideal_mean_S']) + 5*np.array(results['ideal_std_S']),
                 color='C0', alpha=0.15, label='ideal ±5σ')
plt.semilogx(N_array, results['noisy_mean_S'], 's--', label=f'noisy ⟨S⟩ (p={p_noise})')
plt.fill_between(N_array,
                 np.array(results['noisy_mean_S']) - 5*np.array(results['noisy_std_S']),
                 np.array(results['noisy_mean_S']) + 5*np.array(results['noisy_std_S']),
                 color='C1', alpha=0.15, label='noisy ±5σ')
plt.axhline(classical_bound, color='red', linestyle=':', label='classical bound S=2')
plt.xlabel('Rounds N')
plt.ylabel('⟨S⟩')
plt.title('Mean S with ±5σ bands')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)

# 4) Matching rate after sifting (ideal vs noisy)
plt.subplot(2, 2, 4)
plt.semilogx(N_array, results['ideal_keyrate'], 'o-', label='ideal match fraction')
plt.semilogx(N_array, results['noisy_keyrate'], 's--', label=f'noisy match fraction (p={p_noise})')
plt.xlabel('Rounds N')
plt.ylabel('Matching fraction after sifting')
plt.title('Raw key agreement rate (Alice = Bob)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)


plt.tight_layout()
plt.show()

# -------------------------
# FINAL SUMMARY PRINT
# -------------------------
print("\nSummary (selected values):")
for i, N in enumerate(results['N']):
    print(f" N={N:7d} | ideal: ⟨S⟩={results['ideal_mean_S'][i]:.4f}, σ={results['ideal_std_S'][i]:.4f}, "
          f"Q={results['ideal_Q'][i]:.4f}, K={results['ideal_keyrate'][i]:.4f} || "
          f"noisy: ⟨S⟩={results['noisy_mean_S'][i]:.4f}, σ={results['noisy_std_S'][i]:.4f}, "
          f"Q={results['noisy_Q'][i]:.4f}, K={results['noisy_keyrate'][i]:.4f}")
