import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import expm
import cmath

# set constants
h_bar = 1
c = 4.8
omega_2 = 9.6
omega_1_max = 5.2
g_eff = np.sqrt(2) * 0.02
res = 200

phi_min = 0
phi_max = 0.35

t_min = 0
t_max = 35

E_2 = h_bar * omega_2

qb_1 = np.array([0, 1])     # |1> state
qb_2 = np.array([1, 0])     # |0> state
psi_0 = np.kron(qb_1, qb_2)     # combined state |10>


# plot (a)
phi = np.linspace(phi_min, phi_max, res)
E_1 = omega_1_max * np.sqrt(abs(np.cos(np.pi*phi))) + c
E_2 = np.array([omega_2] * res)

# approx.
swap_flux = 0.175
swap_flux_index = np.argmin(np.abs(phi - swap_flux)) 
# plot (b)
times = np.linspace(t_min, t_max, res)
hamiltonians = []
for E in E_1:
    hamiltonians.append(np.array([[1, 0, 0, 0],
                                  [0, omega_2, g_eff, 0],
                                  [0, g_eff, E, 0],
                                  [0, 0, 0, 1]]))

population = np.zeros((res, res))

for k, p in enumerate(phi):
    H = hamiltonians[k]
    for i, t in enumerate(times):
        U = expm(-2j * np.pi * t * H)
        state = U @ psi_0
        population[i, k] = np.abs(state[1])**2


#plot (c)
E1_swap = E_2[swap_flux_index]
hamiltonian = np.array([[0, 0, 0, 0],        
                             [0, omega_2, g_eff, 0],
                             [0, g_eff, E1_swap, 0],
                             [0, 0, 0, 0]])     

p10 = np.zeros(res) 
p01 = np.zeros(res)

for i, t in enumerate(times):
    U = expm(-2j * np.pi * t * hamiltonian)
    state = U @ psi_0
    p01[i] = np.abs(state[1])**2 
    p10[i] = np.abs(state[2])**2

# plot (d)
eigvals_high = [] 
eigvecs_high = [] 
eigvals_low = []
eigvecs_low = []  

for E in E_1:
    x = 0.5 * np.sqrt((omega_2 - E)**2 + 4*(g_eff**2))
    plus = (omega_2 + E)/2 + x
    minus = (omega_2 + E)/2 - x
    eigvals_high.append(plus)
    eigvals_low.append(minus)

    aplus = g_eff
    bplus = plus - omega_2
    norm_plus = np.sqrt(aplus**2 + bplus**2)
    vec = np.array([aplus/norm_plus, bplus/norm_plus])
    eigvecs_high.append(vec)

    aminus = g_eff
    bminus = minus - omega_2
    norm_minus = np.sqrt(aminus**2 + bminus**2)
    vec = np.array([aminus/norm_minus, bminus/norm_minus])
    eigvecs_low.append(vec)



# plotting
fig, axs = plt.subplot_mosaic([['left_top', 'right_top'],
                               ['bottom', 'bottom']])

axs['left_top'].plot(phi, E_1, color='grey', ls='--', lw=1)
axs['left_top'].plot(phi, E_2, color='grey', ls='--', lw=1)
axs['left_top'].set_title('(a) Frequency / Magnetic Flux', fontsize=10)
axs['left_top'].set_xlabel(r'Magnetic flux, $\phi$')
axs['left_top'].set_ylabel('Bare energies (GHz)')
axs['left_top'].plot(phi, eigvals_high, label='|1>', color='orange', lw=1.3)
axs['left_top'].plot(phi, eigvals_low, label='|0>', color='blue', lw=1.3)
axs['left_top'].set_xlim(0.1, 0.25)
axs['left_top'].set_ylim(9.0, 10.0)
axs['left_top'].axvline(swap_flux, color='red', ls='--', lw=1.0, alpha=0.5)
axs['left_top'].legend()

axs['right_top'].set_xlabel(r'Magnetic flux, $\phi$')
axs['right_top'].set_ylabel('time (ns)')
im = axs['right_top'].imshow(population, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
phi_ticks = np.array([0, 0.10, 0.20, 0.30, 0.40])
t_ticks = np.linspace(0, 35, 5)
phi_tick_positions = np.interp(phi_ticks, phi, np.arange(res))
t_tick_positions   = np.interp(t_ticks, times, np.arange(res))
axs['right_top'].set_xticks(phi_tick_positions)
axs['right_top'].set_xticklabels([f"{v:.1f}" for v in phi_ticks])
axs['right_top'].set_yticks(t_tick_positions)
axs['right_top'].set_yticklabels([f"{v:.1f}" for v in t_ticks])
axs['right_top'].axvline(100, label=r'$\phi_{SWAP}', color='w', ls='--', lw='1.0')
axs['right_top'].set_title('(b) Time / Magnetic Flux', fontsize=10)
cbar = fig.colorbar(im, ax=axs['right_top'], orientation='vertical', pad=0.04)
cbar.set_label(r'Probability $P_{01}$', rotation=270, labelpad=15, fontsize=8)

axs['bottom'].plot(times, p01, label=r'$P_{01}$ (Final)', color='k', linestyle='-', linewidth=2)
axs['bottom'].plot(times, p10, label=r'$P_{10}$ (Initial)', color='gray', linestyle='--', linewidth=2)
swap_time_index = np.argmax(p01)
t_SWAP = times[swap_time_index]
axs['bottom'].axvline(t_SWAP, color='gray', linestyle=':', alpha=0.7)
axs['bottom'].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axs['bottom'].set_xlabel('Time (ns)')
axs['bottom'].set_ylabel('Probability')
axs['bottom'].set_ylim(0.0, 1.0)
axs['bottom'].set_xlim(t_min, t_max / 2) 
axs['bottom'].grid(True, linestyle='--', alpha=0.6)
axs['bottom'].legend(loc='lower left')
axs['bottom'].set_title(r'(c) Probability of finding states $|10\rangle$,$|01\rangle$ / time', fontsize=10)

fig.tight_layout()

plt.show()