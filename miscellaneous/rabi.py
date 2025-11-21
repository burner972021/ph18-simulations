import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (typical experimental choices) ---
T_pi = 20e-9          # target π-pulse duration (20 ns)
T_pi2 = T_pi / 2      # π/2 pulse duration
Omega_pi = np.pi / T_pi          # Rabi frequency (rad/s) for π pulse
f_rabi_pi = Omega_pi / (2 * np.pi)  # Rabi frequency in Hz

# --- Time array for plotting ---
t = np.linspace(0, 2*T_pi, 4000)
# Population of |1> during resonant drive (starting from |0>)
P1 = np.sin(Omega_pi * t / 2)**2

# --- Plot Rabi oscillations ---
plt.figure(figsize=(7,4))
plt.plot(t*1e9, P1)
plt.xlabel('Time (ns)')
plt.ylabel('Population $P(|1\\rangle)$')
plt.title(f'Rabi oscillation (on-resonance): π-pulse = {T_pi*1e9:.0f} ns')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print useful numerical results ---
print(f"Target π-pulse duration: {T_pi*1e9:.1f} ns")
print(f"Target π/2-pulse duration: {T_pi2*1e9:.1f} ns")
print(f"Rabi rate Ω (rad/s) for π: {Omega_pi:.3e} rad/s")
print(f"Rabi frequency (Ω/2π): {f_rabi_pi/1e6:.3f} MHz")
print('Rotation angle θ = ∫ Ω(t) dt ; on resonance for constant Ω, θ = Ω·T')
print('So for an on-resonance square pulse: Ω = θ/T. For π pulse θ=π, for π/2 θ=π/2.')

# --- Example: shorter π/2 pulse ---
T_example = 10e-9
Omega_pi2_example = (np.pi/2) / T_example
f_rabi_pi2_example = Omega_pi2_example / (2*np.pi)
print(f"\nExample: perform π/2 in {T_example*1e9:.1f} ns -> Ω/2π = {f_rabi_pi2_example/1e6:.3f} MHz")
