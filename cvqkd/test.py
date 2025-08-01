import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from qutip import *

# Constants
hbar = 2.0
V_A = 10.0
eta = 0.6
xi = 0.01
N_shots = 5000

# 1. Alice prepares coherent states with Gaussian modulation
def prepare_gaussian_states(N, V_A):
    x = np.random.normal(0, np.sqrt(V_A), N)
    p = np.random.normal(0, np.sqrt(V_A), N)
    return x, p

# 2. Channel adds loss and excess noise
def apply_channel(x, p, eta, xi):
    T = np.sqrt(eta)
    sigma = np.sqrt((1 - eta + eta * xi) * hbar / 2)
    x_channel = T * x + np.random.normal(0, sigma, len(x))
    p_channel = T * p + np.random.normal(0, sigma, len(p))
    return x_channel, p_channel

# 3. Bob's heterodyne measurement
def heterodyne_detection(x, p):
    noise = np.random.normal(0, np.sqrt(hbar / 2), len(x))
    x_bob = x + noise
    p_bob = p + noise
    return x_bob, p_bob

# 4. Raw key extraction using binary slicing
def extract_raw_key(values, threshold=0):
    # Binary decision: 0 if < threshold, 1 if >= threshold
    return (values >= threshold).astype(int)

# 5. Simple error correction: parity check simulation
def simulate_reconciliation(key_alice, key_bob):
    # Find bit mismatches
    mismatches = key_alice != key_bob
    n_errors = np.sum(mismatches)
    corrected_bob = key_bob.copy()

    # Simulate parity-based correction (not real LDPC)
    for i in range(len(key_alice)):
        if key_alice[i] != key_bob[i]:
            corrected_bob[i] = key_alice[i]  # Bob corrects to Alice's value (idealized)

    return corrected_bob, n_errors

# 6. Key rate estimation
def calc_key_rate(V_A, eta, xi, beta=0.95):
    V = V_A + 1
    chi_line = (1 - eta) / eta + xi
    I_AB = 0.5 * np.log2((V + chi_line) / (1 + chi_line))
    S_EB = np.log2(np.e) * ((1 - eta) * V_A / (1 + chi_line))  # Very simplified
    K = beta * I_AB - S_EB
    return max(0, K)

# 7. Run the protocol
def run_simulation():
    # Alice prepares states
    x_a, p_a = prepare_gaussian_states(N_shots, V_A)
    
    # Channel noise
    x_chan, p_chan = apply_channel(x_a, p_a, eta, xi)
    
    # Bob's heterodyne detection
    x_bob, p_bob = heterodyne_detection(x_chan, p_chan)

    # Extract raw keys (use only x quadrature here)
    raw_key_alice = extract_raw_key(x_a)
    raw_key_bob = extract_raw_key(x_bob)

    # Error correction (simplified)
    rec_key_bob, n_errors = simulate_reconciliation(raw_key_alice, raw_key_bob)

    # Post-processing
    V_A_est = np.var(x_a)
    key_rate = calc_key_rate(V_A_est, eta, xi)

    # Print results
    print(f"Raw key length: {len(raw_key_alice)}")
    print(f"Errors before correction: {n_errors}")
    print(f"Estimated key rate (asymptotic): {key_rate:.4f} bits/use")

    # Show part of the key
    print("Alice's key: ", hex(int(''.join(map(str, raw_key_alice)), 2)))
    print("Bob's corrected key: ", hex(int(''.join(map(str, rec_key_bob)), 2)))


run_simulation()
