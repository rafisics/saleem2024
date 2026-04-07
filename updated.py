import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ==========================================
# 1. Physical Parameters & Normalization
# ==========================================
# Mass ratios Q_j = m_O / m_j 
m_O = 16.0
m_H = 1.0
m_e = 1.0 / 1836.0

Q_H = m_O / m_H
Q_O = 1.0
Q_e = m_O / m_e
Q_sp = m_O / m_H
Q_se = m_O / m_e

# Temperatures (normalized to Te)
sigma_H = 0.05
sigma_O = 0.05
sigma_e = 1.0

# ==========================================
# 2. Mathematical Formulation
# ==========================================
def calc_H(xi, Y, params):
    """
    Evaluates the characteristic equation H(xi) = 0. 
    The expansion only starts when this singularity condition is met.
    """
    N_H, V_H, N_O, V_O, N_e, V_e, N_sp, V_sp, N_se, V_se, Phi = Y
    
    N = np.array([N_H, N_O, N_e, N_sp, N_se])
    V = np.array([V_H, V_O, V_e, V_sp, V_se])
    Q = np.array([Q_H, Q_O, Q_e, Q_sp, Q_se])
    sigma = np.array([sigma_H, sigma_O, sigma_e, params['sigma_sp'], params['sigma_se']])
    coeff = np.array([params['alpha'], 1.0, params['beta'], params['gamma'], params['delta']])
    
    # Characteristic denominator for each species
    S = (V - xi)**2 - Q * sigma
    W = Q / S
    
    return np.sum(coeff * N * W)

def dY_dxi(xi, Y, params):
    """
    Explicit ODE system dY/dxi = F(Y, xi).
    Derived by analytically differentiating the quasi-neutrality constraint.
    """
    N_H, V_H, N_O, V_O, N_e, V_e, N_sp, V_sp, N_se, V_se, Phi = Y
    
    N = np.array([N_H, N_O, N_e, N_sp, N_se])
    V = np.array([V_H, V_O, V_e, V_sp, V_se])
    Q = np.array([Q_H, Q_O, Q_e, Q_sp, Q_se])
    sigma = np.array([sigma_H, sigma_O, sigma_e, params['sigma_sp'], params['sigma_se']])
    coeff = np.array([params['alpha'], 1.0, params['beta'], params['gamma'], params['delta']])
    sign_C = np.array([1.0, 1.0, -1.0, 1.0, -1.0])  # + for ions, - for electrons
    
    # Algebraic states
    S = (V - xi)**2 - Q * sigma
    W = Q / S
    D = S / (V - xi)
    
    # Coefficients for N' and V' in terms of Phi'
    C_N = sign_C * N * W
    C_V = -sign_C * Q / D
    P = 2.0 * Q * (V - xi) / (S**2)
    
    # Analytically isolate Phi_prime (dPhi/dxi)
    K = np.sum(coeff * (W * C_N - N * P * C_V))
    Phi_prime = -np.sum(coeff * N * P) / (K + 1e-20)
    
    # Final derivatives
    N_prime = C_N * Phi_prime
    V_prime = C_V * Phi_prime
    
    return [N_prime[0], V_prime[0], N_prime[1], V_prime[1], 
            N_prime[2], V_prime[2], N_prime[3], V_prime[3], 
            N_prime[4], V_prime[4], Phi_prime]

# ==========================================
# 3. Solver Workflow Setup
# ==========================================
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Column Configurations corresponding to Figure 3
configs = [
    {"V_0": [10, 30], "sigma_r": [1.0, 1.0], "gamma": [0.5, 0.5]},
    {"V_0": [10, 10], "sigma_r": [1.0, 1.5], "gamma": [0.5, 0.5]},
    {"V_0": [10, 10], "sigma_r": [1.0, 1.0], "gamma": [0.5, 0.2]}
]

# print("Integrating Multi-fluid DAE system. This may take a moment due to system stiffness...")

for col, config in enumerate(configs):
    for style_idx, style in enumerate(['-', '--']):
        
        # Extract variables for current configuration loop
        V_0 = config["V_0"][style_idx]
        sig_r = config["sigma_r"][style_idx]
        gma = config["gamma"][style_idx]
        
        # Base Densities
        alpha = 0.1
        gamma_val = gma
        # Balance beta and delta strictly to enforce initial quasi-neutrality exactly
        beta = (1.0 + alpha + gamma_val) / 2.0 
        delta = beta
        
        params = {
            'alpha': alpha, 'beta': beta, 'gamma': gamma_val, 'delta': delta,
            'sigma_sp': sig_r, 'sigma_se': sig_r,
            'Q_H': Q_H, 'Q_e': Q_e, 'Q_sp': Q_sp, 'Q_se': Q_se
        }
        
        # Initial Unperturbed State
        Y0 = [1.0, 5.0,     # N_H, V_H
              1.0, 1.0,     # N_O, V_O
              1.0, 10.0,    # N_e, V_e
              1.0, V_0,     # N_sp, V_sp
              1.0, V_0,     # N_se, V_se
              0.0]          # Phi
        
        # Find the singularity point where the expansion begins
        # Searches between xi=0 and the sonic point of Oxygen
        try:
            xi_front = brentq(calc_H, 0.0, 0.77, args=(Y0, params))
        except ValueError:
            xi_front = 0.0  # Fallback if already singular
            
        # Integrate the expansion curve from xi_front
        xi_span = (xi_front, xi_front + 3.0)
        xi_eval = np.linspace(xi_front, xi_front + 3.0, 300)
        
        sol = solve_ivp(
            fun=lambda xi, Y: dY_dxi(xi, Y, params),
            t_span=xi_span,
            y0=Y0,
            method='Radau', # Implicit solver for stiffness handling
            t_eval=xi_eval
        )
        
        # To match the paper's visualization axis exactly, shift xi so the expansion starts at 0
        plot_xi = sol.t - xi_front
        
        # Extract results
        N_H_res, V_H_res = sol.y[0], sol.y[1]
        N_O_res, V_O_res = sol.y[2], sol.y[3]
        Phi_res = sol.y[10]
        
        # Row 1: Densities
        axs[0, col].plot(plot_xi, N_H_res, color='black', linestyle=style, label=r'$N_{H^+}$' if style_idx==0 else "")
        axs[0, col].plot(plot_xi, N_O_res, color='red', linestyle=style, label=r'$N_{O^+}$' if style_idx==0 else "")
        axs[0, col].set_ylabel('N')
        
        # Row 2: Velocities
        axs[1, col].plot(plot_xi, V_H_res, color='black', linestyle=style)
        axs[1, col].plot(plot_xi, V_O_res, color='red', linestyle=style)
        axs[1, col].set_ylabel('V')
        
        # Row 3: Electric Potential
        axs[2, col].plot(plot_xi, Phi_res, color='black', linestyle=style)
        axs[2, col].set_ylabel(r'$\phi$')

# Axes limits and formatting
letters = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)'], ['(g)', '(h)', '(i)']]
for r in range(3):
    for c in range(3):
        axs[r, c].set_xlabel(r'$\xi$')
        axs[r, c].set_xlim(0, 3.0)
        axs[r, c].text(0.5, -0.3, letters[r][c], transform=axs[r, c].transAxes, fontsize=12, ha='center')

axs[0, 0].set_ylim(0, 1.2)
axs[0, 1].set_ylim(0, 1.2)
axs[0, 2].set_ylim(0, 1.2)

axs[1, 0].set_ylim(0, 9.0)
axs[1, 1].set_ylim(0, 9.0)
axs[1, 2].set_ylim(0, 9.0)

axs[0, 0].legend(loc='lower left')
plt.show()
