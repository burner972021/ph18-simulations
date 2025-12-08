"""
CZ gate simulation using the simplified |11> <-> |02> 2x2 avoided-crossing block model
Option B (fast, captures real CZ physics qualitatively)

Produces three panels (similar to the paper's Fig.15):
 (a) Spectrum vs flux showing the |11> - |02> avoided crossing
 (b) Population of |02> (starting from |11>) vs time and flux (swap probability heatmap)
 (c) Conditional phase accumulation at an operating flux and a simple CZ-fidelity proxy

Notes on units:
 - Energies (freqs) are in GHz
 - Time is in ns
 - When evolving U = exp(-i * 2*pi * H(GHz) * t(ns)) the factor 2*pi converts GHz to angular frequency

This script is minimal-dependency (numpy, scipy, matplotlib) and self-contained.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, expm

# ---------------------------
# Physical / model parameters
# ---------------------------
omega1_max = 5.2       # GHz, qubit 1 max frequency (flux-tunable)
omega2 = 4.8           # GHz, qubit 2 fixed frequency
alpha2 = 0 #-0.300        # GHz, anharmonicity of qubit 2 (sets |02> energy)
g = 0.020              # GHz, bare coupling between modes

# coupling between |11> and |02> in the two-excitation manifold is sqrt(2)*g
g_eff = np.sqrt(2.0) * g

# Flux sweep for qubit 1 (in units of Phi0)
phis = np.linspace(0.0, 0.40, 351)

# Flux-to-frequency model for qubit 1 (simple cosine-root model - captures tunability)
def omega1_of_phi(phi):
    # use a typical transmon-like flux dependence (keeps values positive)
    # map phi in [0,0.35] -> cos(pi*phi) near 1; take sqrt(abs(cos)) to mimic junction dependence
    return omega1_max * np.sqrt(np.abs(np.cos(np.pi * phi)))

# ---------------------------
# Helper: 2x2 block Hamiltonian
# basis: [|11>, |02>]
# Energies (bare):
#  E(|11>) = omega1(phi) + omega2
#  E(|02>) = 2*omega2 + alpha2
# Coupling: g_eff
# ---------------------------
def H_block(phi):
    w1 = omega1_of_phi(phi)
    E11 = w1 + omega2
    E02 = 2.0 * omega2 + alpha2
    H = np.array([[E11, g_eff], [g_eff, E02]], dtype=float)
    return H

# ---------------------------
# (a) Spectrum: diagonalize H_block across flux
# ---------------------------
energies = np.zeros((len(phis), 2))
for i, phi in enumerate(phis):
    vals, vecs = eigh(H_block(phi))
    energies[i, :] = np.sort(vals)

# ---------------------------
# (b) Dynamics: population transfer |11> -> |02>
# For each phi and time, start in |11> = [1,0] (block basis) and compute P(|02>)
# ---------------------------
# choose times (ns) up to a few hundred ns depending on gap
t_max = 35.0  # ns
nt = 400
times = np.linspace(0.0, t_max, nt)

pop_02 = np.zeros((nt, len(phis)))

for j, phi in enumerate(phis):
    H = H_block(phi)
    for i, t in enumerate(times):
        U = expm(-1j * 2.0 * np.pi * H * t)   # H in GHz, t in ns -> factor 2pi
        psi0 = np.array([1.0, 0.0], dtype=complex)  # start in |11>
        psi_t = U @ psi0
        pop_02[i, j] = np.abs(psi_t[1])**2


# ---------------------------
# (c) Conditional phase accumulation at a chosen operating flux
# We compute the phase picked up by the amplitude on |11> relative to the bare |11> dynamical phase
# (i.e. remove the phase e^{-i 2pi (omega1+omega2) t} that would occur if there were no coupling)
# Then compute a fidelity proxy: F = (1 + cos(phi_cond - pi))/2  which equals 1 when phi_cond==pi
# ---------------------------
# choose operating flux near the avoided crossing (where the gap is visible)
phi_operating = 0.176
H_op = H_block(phi_operating)

unwrapped_phase = np.zeros_like(times)
fidelity_proxy = np.zeros_like(times)

for i, t in enumerate(times):
    U = expm(-1j * 2.0 * np.pi * H_op * t)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi_t = U @ psi0

    # phase of the |11> component
    amp_11 = psi_t[0]
    raw_phase = np.angle(amp_11)

    # remove the single-particle dynamical phase contribution of the bare |11>
    w1_op = omega1_of_phi(phi_operating)
    bare_phase = -2.0 * np.pi * (w1_op + omega2) * t  # note sign: U included negative sign
    # we want conditional phase coming from interaction only, so combine
    phi_cond = (raw_phase - bare_phase)  # this is an unwrapped-like phase (but modulo 2pi cents)

    # unwrap relative to previous value for smooth growth
    if i == 0:
        unwrapped_phase[i] = phi_cond
    else:
        # unwrap relative to previous sample
        delta = ((phi_cond - unwrapped_phase[i-1] + np.pi) % (2*np.pi)) - np.pi
        unwrapped_phase[i] = unwrapped_phase[i-1] + delta

    # fidelity proxy: 1 when phi_cond == pi
    fidelity_proxy[i] = 0.5 * (1.0 + np.cos(unwrapped_phase[i] - np.pi))

# normalize time axis for plotting to the time where phi reaches pi
# find t_CZ such that unwrapped_phase(t_CZ) ~= pi
idx_close = np.argmin(np.abs(unwrapped_phase - np.pi))
t_CZ = times[idx_close]
print(f"Estimated CZ gate time (phi_cond ~ pi) at operating flux {phi_operating:.3f}: t_CZ = {t_CZ:.2f} ns")

# ---------------------------
# Plotting: produce a 1x3 figure to match the paper layout
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# (a) Spectrum
ax = axes[0]
ax.plot(phis, energies[:, 0], color='#1f77b4', lw=1.5, label='Lower dressed')
ax.plot(phis, energies[:, 1], color='#ff7f0e', lw=1.5, label='Upper dressed')
# overlay bare energies for guidance
bare_E11 = np.array([omega1_of_phi(phi) + omega2 for phi in phis])
bare_E02 = np.full_like(phis, 2.0*omega2 + alpha2)
ax.plot(phis, bare_E11, color='gray', lw=0.8, ls='--', label='Bare |11>')
ax.plot(phis, bare_E02, color='gray', lw=0.8, ls=':', label='Bare |02>')
ax.axvline(phi_operating, color='red', ls='--', label='Operating flux')
ax.set_xlabel('Magnetic flux (qubit 1, Φ/Φ0)')
ax.set_ylabel('Energy / Frequency (GHz)')
ax.set_title('(a) Avoided crossing: |11> ↔ |02>')
ax.legend(loc='best', fontsize=9)
ax.grid(True, ls=':')

# (b) Population heatmap of |02>
ax = axes[1]
im = ax.imshow(pop_02, aspect='auto', origin='lower',
               extent=[phis[0], phis[-1], times[0], times[-1]],
               cmap='viridis', vmin=0.0, vmax=1.0)
ax.set_xlabel('Magnetic flux (qubit 1, Φ/Φ0)')
ax.set_ylabel('Time (ns)')
ax.set_title('(b) Population in |02> starting from |11> (swap probability)')
ax.axvline(phi_operating, color='w', ls='--')
fig.colorbar(im, ax=ax, label='P(|02>)')

# Legends (combine)
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='lower right')

plt.tight_layout()
plt.show()

# ---------------------------
# End of file
# ---------------------------
