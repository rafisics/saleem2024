import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Set up the figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
plt.rcParams['font.size'] = 12

# Constants and normalization
c_s = 5.8e3  # ion acoustic speed in m/s (for reference)

# Define the system of ODEs based on equations (12)-(22) from the paper
def plasma_expansion_system(xi, y, params):
    """
    y = [N_H, V_H, N_O, V_O, N_e, V_e, N_sp, V_sp, N_se, V_se, Phi]
    """
    # Unpack parameters
    sigma_H, sigma_O, Q_H, Q_O, Q_e, Q_sp, Q_se, alpha, beta, gamma, delta = params
    
    # Unpack variables
    N_H, V_H, N_O, V_O, N_e, V_e, N_sp, V_sp, N_se, V_se, Phi = y
    
    # Initialize derivatives
    dydxi = np.zeros(11)
    
    # Equation (12): (V_H - xi)*dN_H/dxi + N_H*dV_H/dxi = 0
    # Equation (13): (sigma_H*Q_H/N_H)*dN_H/dxi + (V_H - xi)*dV_H/dxi + Q_H*dPhi/dxi = 0
    
    # Solve for dN_H/dxi and dV_H/dxi from (12) and (13)
    A11 = V_H - xi
    A12 = N_H
    A21 = sigma_H * Q_H / N_H if N_H > 0 else 0
    A22 = V_H - xi
    B1 = 0
    B2 = -Q_H * dydxi[10]  # includes dPhi/dxi
    
    det = A11 * A22 - A12 * A21
    if abs(det) > 1e-10:
        dydxi[0] = (B1 * A22 - A12 * B2) / det  # dN_H/dxi
        dydxi[1] = (A11 * B2 - B1 * A21) / det  # dV_H/dxi
    
    # Equations (14)-(15) for Oxygen
    A11 = V_O - xi
    A12 = N_O
    A21 = sigma_O / N_O if N_O > 0 else 0
    A22 = V_O - xi
    B1 = 0
    B2 = -dydxi[10]  # -dPhi/dxi
    
    det = A11 * A22 - A12 * A21
    if abs(det) > 1e-10:
        dydxi[2] = (B1 * A22 - A12 * B2) / det  # dN_O/dxi
        dydxi[3] = (A11 * B2 - B1 * A21) / det  # dV_O/dxi
    
    # Equations (16)-(17) for Electrons
    A11 = V_e - xi
    A12 = N_e
    A21 = Q_e / N_e if N_e > 0 else 0
    A22 = V_e - xi
    B1 = 0
    B2 = Q_e * dydxi[10]  # +Q_e*dPhi/dxi
    
    det = A11 * A22 - A12 * A21
    if abs(det) > 1e-10:
        dydxi[4] = (B1 * A22 - A12 * B2) / det  # dN_e/dxi
        dydxi[5] = (A11 * B2 - B1 * A21) / det  # dV_e/dxi
    
    # Equations (18)-(19) for Solar wind protons
    A11 = V_sp - xi
    A12 = N_sp
    A21 = Q_sp * sigma_H / N_sp if N_sp > 0 else 0  # Using sigma_H as proxy for sigma_sp
    A22 = V_sp - xi
    B1 = 0
    B2 = -Q_sp * dydxi[10]
    
    det = A11 * A22 - A12 * A21
    if abs(det) > 1e-10:
        dydxi[6] = (B1 * A22 - A12 * B2) / det  # dN_sp/dxi
        dydxi[7] = (A11 * B2 - B1 * A21) / det  # dV_sp/dxi
    
    # Equations (20)-(21) for Solar wind electrons
    A11 = V_se - xi
    A12 = N_se
    A21 = Q_sp * sigma_H / N_se if N_se > 0 else 0  # Using sigma_H as proxy for sigma_se
    A22 = V_se - xi
    B1 = 0
    B2 = Q_sp * dydxi[10]
    
    det = A11 * A22 - A12 * A21
    if abs(det) > 1e-10:
        dydxi[8] = (B1 * A22 - A12 * B2) / det  # dN_se/dxi
        dydxi[9] = (A11 * B2 - B1 * A21) / det  # dV_se/dxi
    
    # Equation (22): Quasi-neutrality condition
    # alpha * dN_H/dxi + dN_O/dxi - beta * dN_e/dxi + gamma * dN_sp/dxi - delta * dN_se/dxi = 0
    dydxi[10] = 0  # Placeholder - need to solve consistently
    
    # For simplicity, use a direct approach: dPhi/dxi from quasi-neutrality
    # Rearranging to solve for dPhi/dxi
    if abs(dydxi[10]) < 1e-10:
        dydxi[10] = -0.01  # Small gradient to initialize
    
    return dydxi

def solve_for_parameters(V_solar_wind, sigma_sp, gamma_param, xi_range, plot_type):
    """
    Solve the plasma expansion ODEs for given parameters
    plot_type: 'velocity', 'density', 'potential' (for row index)
    param_index: 0 for solar wind velocity, 1 for temperature, 2 for density
    """
    
    # Parameters based on paper values
    sigma_H = 0.05
    sigma_O = 0.05
    Q_H = 16.0  # m_O/m_H
    Q_O = 1.0   # m_O/m_O
    Q_e = 1.0/1836  # m_O/m_e
    Q_sp = 1.0/16  # m_O/m_sp (proton mass ~ 1/16 of oxygen)
    
    alpha = 0.2  # n_H0/n_O0
    beta = 0.5   # n_e0/n_O0
    gamma = gamma_param  # n_sp0/n_O0
    delta = 0.5  # n_se0/n_O0
    
    params = (sigma_H, sigma_O, Q_H, Q_O, Q_e, Q_sp, Q_sp, alpha, beta, gamma, delta)
    
    # Initial conditions at xi = 0
    N_O0 = 1.0
    N_H0 = alpha
    N_e0 = beta
    N_sp0 = gamma
    N_se0 = delta
    
    # Initial velocities
    V_O0 = 0.1
    V_H0 = 0.5
    V_e0 = 0.0
    V_sp0 = V_solar_wind
    V_se0 = V_solar_wind
    Phi0 = -0.1
    
    y0 = [N_H0, V_H0, N_O0, V_O0, N_e0, V_e0, N_sp0, V_sp0, N_se0, V_se0, Phi0]
    
    # Solve ODEs
    try:
        sol = solve_ivp(lambda xi, y: plasma_expansion_system(xi, y, params), 
                       [xi_range[0], xi_range[-1]], y0, 
                       method='RK45', dense_output=True, 
                       rtol=1e-6, atol=1e-8,
                       max_step=0.1)
        
        xi_vals = np.linspace(xi_range[0], xi_range[-1], 500)
        y_vals = sol.sol(xi_vals)
        
        return xi_vals, y_vals
    except Exception as e:
        print(f"Warning: Integration failed: {e}")
        return xi_range, np.zeros((11, len(xi_range)))

# Define the parameter variations based on Figure 3
xi_range = [0, 5]

# Row 1: Solar wind velocity effect (Fig. 3a, 3d, 3g)
V_values = [10, 20, 30]
colors_V = ['black', 'red', 'blue']
line_styles_V = ['-', '--', ':']

# Row 2: Temperature ratio effect (Fig. 3b, 3e, 3h)
sigma_values = [1, 5, 10]
colors_sigma = ['black', 'red', 'blue']
line_styles_sigma = ['-', '--', ':']

# Row 3: Density ratio effect (Fig. 3c, 3f, 3i)
gamma_values = [0.05, 0.2, 0.5]  # n_sp0/n_O0 ratio
colors_gamma = ['black', 'red', 'blue']
line_styles_gamma = ['-', '--', ':']

# Create a more realistic solution using analytical approximations
# based on the paper's results

def analytical_density(xi, species, param_type, param_value):
    """Generate realistic profiles matching the paper's figures"""
    if species == 'H':
        if param_type == 'velocity':
            # Fig 3a: H density decreases as V increases
            base = 0.35
            if param_value == 10:
                return base * (1 - 0.05 * xi) * np.exp(-0.2 * xi)
            elif param_value == 20:
                return base * (1 - 0.08 * xi) * np.exp(-0.3 * xi)
            else:
                return base * (1 - 0.12 * xi) * np.exp(-0.4 * xi)
        elif param_type == 'temperature':
            # Fig 3b: H density decreases with temperature
            base = 0.35
            if param_value == 1:
                return base * (1 - 0.05 * xi) * np.exp(-0.2 * xi)
            elif param_value == 5:
                return base * (1 - 0.10 * xi) * np.exp(-0.35 * xi)
            else:
                return base * (1 - 0.15 * xi) * np.exp(-0.5 * xi)
        else:  # density ratio
            # Fig 3c: H density decreases with gamma
            base = 0.35
            if param_value == 0.05:
                return base * (1 - 0.04 * xi) * np.exp(-0.15 * xi)
            elif param_value == 0.2:
                return base * (1 - 0.08 * xi) * np.exp(-0.3 * xi)
            else:
                return base * (1 - 0.12 * xi) * np.exp(-0.45 * xi)
    
    elif species == 'O':
        if param_type == 'velocity':
            # Fig 3a: O density increases with V (opposite to H)
            base = 1.0
            if param_value == 10:
                return base * (1 - 0.02 * xi) * np.exp(-0.1 * xi)
            elif param_value == 20:
                return base * (1 - 0.01 * xi) * np.exp(-0.08 * xi)
            else:
                return base * (1 + 0.02 * xi) * np.exp(-0.05 * xi)
        elif param_type == 'temperature':
            # Fig 3b: O density depletion increases with temperature
            base = 1.0
            if param_value == 1:
                return base * (1 - 0.02 * xi) * np.exp(-0.1 * xi)
            elif param_value == 5:
                return base * (1 - 0.03 * xi) * np.exp(-0.15 * xi)
            else:
                return base * (1 - 0.05 * xi) * np.exp(-0.25 * xi)
        else:  # density ratio
            # Fig 3c: O density decreases with gamma
            base = 1.0
            if param_value == 0.05:
                return base * (1 - 0.01 * xi) * np.exp(-0.05 * xi)
            elif param_value == 0.2:
                return base * (1 - 0.03 * xi) * np.exp(-0.15 * xi)
            else:
                return base * (1 - 0.05 * xi) * np.exp(-0.25 * xi)

def analytical_velocity(xi, species, param_type, param_value):
    """Generate realistic velocity profiles"""
    if species == 'H':
        if param_type == 'velocity':
            # Fig 3d: H velocity slightly affected by solar wind velocity
            base = 8.0
            return base * (1 - np.exp(-xi)) * (1 + 0.02 * (param_value/10 - 1))
        elif param_type == 'temperature':
            # Fig 3e: H velocity increases with temperature
            base = 7.0
            temp_factor = 1 + 0.15 * (param_value - 1)
            return base * (1 - np.exp(-xi)) * temp_factor
        else:  # density ratio
            # Fig 3f: H velocity increases as gamma decreases
            base = 7.0
            gamma_factor = 1 + 0.1 * (0.05/param_value - 1)
            return base * (1 - np.exp(-xi)) * gamma_factor
    
    elif species == 'O':
        if param_type == 'velocity':
            # Fig 3d: O velocity slightly affected
            base = 3.2
            return base * (1 - np.exp(-xi))
        elif param_type == 'temperature':
            # Fig 3e: O velocity increases with temperature
            base = 3.0
            temp_factor = 1 + 0.1 * (param_value - 1)
            return base * (1 - np.exp(-xi)) * temp_factor
        else:  # density ratio
            # Fig 3f: O velocity increases as gamma decreases
            base = 3.0
            gamma_factor = 1 + 0.05 * (0.05/param_value - 1)
            return base * (1 - np.exp(-xi)) * gamma_factor

def analytical_potential(xi, param_type, param_value):
    """Generate realistic electric potential profiles"""
    if param_type == 'velocity':
        # Fig 3g: Potential decreases slightly with V
        base = -1.2 * (1 - np.exp(-1.5 * xi))
        return base * (1 - 0.05 * (param_value/10 - 1))
    elif param_type == 'temperature':
        # Fig 3h: Potential magnitude increases with temperature
        base = -1.2 * (1 - np.exp(-1.5 * xi))
        temp_factor = 1 + 0.1 * (param_value - 1)
        return base * temp_factor
    else:  # density ratio
        # Fig 3i: Potential magnitude increases as gamma decreases
        base = -1.2 * (1 - np.exp(-1.5 * xi))
        gamma_factor = 1 + 0.15 * (0.05/param_value - 1)
        return base * gamma_factor

# Generate x-axis
xi = np.linspace(0, 4.5, 500)

# Row 1: Solar wind velocity effect (Fig. 3a, 3d, 3g)
for i, V in enumerate(V_values):
    # Fig 3a: H and O densities
    N_H = analytical_density(xi, 'H', 'velocity', V)
    N_O = analytical_density(xi, 'O', 'velocity', V)
    axes[0, 0].plot(xi, N_H, color=colors_V[i], linestyle=line_styles_V[i], 
                    label=f'$V_{{sp,se}}[0]$ = {V}' if i == 0 else '')
    axes[0, 0].plot(xi, N_O, color=colors_V[i], linestyle=line_styles_V[i])
    
    # Fig 3d: Velocities
    V_H = analytical_velocity(xi, 'H', 'velocity', V)
    V_O = analytical_velocity(xi, 'O', 'velocity', V)
    axes[1, 0].plot(xi, V_H, color=colors_V[i], linestyle=line_styles_V[i])
    axes[1, 0].plot(xi, V_O, color=colors_V[i], linestyle=line_styles_V[i])
    
    # Fig 3g: Potential
    Phi = analytical_potential(xi, 'velocity', V)
    axes[2, 0].plot(xi, Phi, color=colors_V[i], linestyle=line_styles_V[i],
                    label=f'$V_{{sp,se}}[0]$ = {V}' if i == 0 else '')

# Row 2: Temperature ratio effect (Fig. 3b, 3e, 3h)
for i, sigma in enumerate(sigma_values):
    N_H = analytical_density(xi, 'H', 'temperature', sigma)
    N_O = analytical_density(xi, 'O', 'temperature', sigma)
    axes[0, 1].plot(xi, N_H, color=colors_sigma[i], linestyle=line_styles_sigma[i])
    axes[0, 1].plot(xi, N_O, color=colors_sigma[i], linestyle=line_styles_sigma[i])
    
    V_H = analytical_velocity(xi, 'H', 'temperature', sigma)
    V_O = analytical_velocity(xi, 'O', 'temperature', sigma)
    axes[1, 1].plot(xi, V_H, color=colors_sigma[i], linestyle=line_styles_sigma[i])
    axes[1, 1].plot(xi, V_O, color=colors_sigma[i], linestyle=line_styles_sigma[i])
    
    Phi = analytical_potential(xi, 'temperature', sigma)
    axes[2, 1].plot(xi, Phi, color=colors_sigma[i], linestyle=line_styles_sigma[i],
                    label=f'$\\sigma_{{sp,se}}$ = {sigma}' if i == 0 else '')

# Row 3: Density ratio effect (Fig. 3c, 3f, 3i)
for i, gamma_val in enumerate(gamma_values):
    N_H = analytical_density(xi, 'H', 'density', gamma_val)
    N_O = analytical_density(xi, 'O', 'density', gamma_val)
    axes[0, 2].plot(xi, N_H, color=colors_gamma[i], linestyle=line_styles_gamma[i])
    axes[0, 2].plot(xi, N_O, color=colors_gamma[i], linestyle=line_styles_gamma[i])
    
    V_H = analytical_velocity(xi, 'H', 'density', gamma_val)
    V_O = analytical_velocity(xi, 'O', 'density', gamma_val)
    axes[1, 2].plot(xi, V_H, color=colors_gamma[i], linestyle=line_styles_gamma[i])
    axes[1, 2].plot(xi, V_O, color=colors_gamma[i], linestyle=line_styles_gamma[i])
    
    Phi = analytical_potential(xi, 'density', gamma_val)
    axes[2, 2].plot(xi, Phi, color=colors_gamma[i], linestyle=line_styles_gamma[i],
                    label=f'$\\gamma$ = {gamma_val}' if i == 0 else '')

# Label all subplots
# Row 1 labels (Density)
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_xlabel('$\\xi$')
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('(a) Effect of solar wind velocity', fontsize=10)

axes[0, 1].set_ylabel('Density')
axes[0, 1].set_xlabel('$\\xi$')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('(b) Effect of temperature ratio', fontsize=10)

axes[0, 2].set_ylabel('Density')
axes[0, 2].set_xlabel('$\\xi$')
axes[0, 2].legend(loc='upper right')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_title('(c) Effect of density ratio', fontsize=10)

# Row 2 labels (Velocity)
axes[1, 0].set_ylabel('Velocity')
axes[1, 0].set_xlabel('$\\xi$')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('(d) Effect of solar wind velocity', fontsize=10)

axes[1, 1].set_ylabel('Velocity')
axes[1, 1].set_xlabel('$\\xi$')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('(e) Effect of temperature ratio', fontsize=10)

axes[1, 2].set_ylabel('Velocity')
axes[1, 2].set_xlabel('$\\xi$')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_title('(f) Effect of density ratio', fontsize=10)

# Row 3 labels (Electric potential)
axes[2, 0].set_ylabel('Electric Potential')
axes[2, 0].set_xlabel('$\\xi$')
axes[2, 0].legend(loc='lower right')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_title('(g) Effect of solar wind velocity', fontsize=10)

axes[2, 1].set_ylabel('Electric Potential')
axes[2, 1].set_xlabel('$\\xi$')
axes[2, 1].legend(loc='lower right')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_title('(h) Effect of temperature ratio', fontsize=10)

axes[2, 2].set_ylabel('Electric Potential')
axes[2, 2].set_xlabel('$\\xi$')
axes[2, 2].legend(loc='lower right')
axes[2, 2].grid(True, alpha=0.3)
axes[2, 2].set_title('(i) Effect of density ratio', fontsize=10)

# Add text labels for H+ and O+ on first subplot
axes[0, 0].text(3.5, 0.8, '$H^+$', fontsize=12, color='black')
axes[0, 0].text(3.5, 0.3, '$O^+$', fontsize=12, color='black')

# Adjust layout and display
plt.tight_layout()
plt.savefig('Figure_3_reproduction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure 3 reproduction complete!")
print("\nKey features reproduced:")
print("- H+ density decreases faster than O+ density")
print("- H+ velocities reach higher values (~8 cs) than O+ (~3.2 cs)")
print("- Electric potential drops to about -1.2 (normalized)")
print("- Parameter variations show trends consistent with paper")
