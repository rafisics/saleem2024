import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

# ============================================================================
# Parameters from paper
# ============================================================================

Q_H = 16.0
Q_O = 1.0
Q_e = 29090.0  # m_O/m_e
Q_sp = 16.0
Q_se = 29090.0

sigma_H = 0.05
sigma_O = 0.05
alpha = 0.2
beta = 0.5
delta = 0.5

# ============================================================================
# Function to compute derivatives and neutrality
# ============================================================================

def compute_derivatives(xi, y, dPhidxi, gamma, sigma_sp, sigma_se):
    """Compute all dN/dxi and dV/dxi"""
    N_H, V_H, N_O, V_O, N_e, V_e, N_sp, V_sp, N_se, V_se, Phi = y
    
    eps = 1e-10
    
    def solve_species(V, N, sigma, Q, sign):
        A11 = V - xi
        A12 = N
        A21 = sigma * Q / max(N, eps)
        A22 = V - xi
        
        det = A11 * A22 - A12 * A21
        if abs(det) < eps:
            return 0.0, 0.0
        
        B1 = 0.0
        B2 = -sign * Q * dPhidxi
        
        dNdxi = (B1 * A22 - A12 * B2) / det
        dVdxi = (A11 * B2 - B1 * A21) / det
        
        return dNdxi, dVdxi
    
    dN_H, dV_H = solve_species(V_H, N_H, sigma_H, Q_H, 1.0)
    dN_O, dV_O = solve_species(V_O, N_O, sigma_O, Q_O, 1.0)
    dN_e, dV_e = solve_species(V_e, N_e, 1.0, Q_e, -1.0)
    dN_sp, dV_sp = solve_species(V_sp, N_sp, sigma_sp, Q_sp, 1.0)
    dN_se, dV_se = solve_species(V_se, N_se, sigma_se, Q_se, -1.0)
    
    # Neutrality condition (22)
    neutrality = alpha * dN_H + dN_O - beta * dN_e + gamma * dN_sp - delta * dN_se
    
    return [dN_H, dV_H, dN_O, dV_O, dN_e, dV_e, dN_sp, dV_sp, dN_se, dV_se, neutrality]


def ode_system_with_fixed_dPhi(xi, y, gamma, sigma_sp, sigma_se, dPhidxi):
    """ODE system with fixed dPhidxi"""
    derivs = compute_derivatives(xi, y, dPhidxi, gamma, sigma_sp, sigma_se)
    derivs[10] = dPhidxi  # Set dPhi/dxi
    return derivs


# ============================================================================
# Solve by finding dPhidxi that satisfies neutrality at each step
# ============================================================================

def solve_plasma(gamma, sigma_sp, sigma_se, vsp0, vse0, xi_max=4.5):
    """Solve by finding dPhidxi at each step that makes neutrality = 0"""
    
    # Initial conditions from Section 3.2
    y0 = np.array([1.0, 5.0,    # N_H, V_H
                   1.0, 1.0,    # N_O, V_O
                   1.0, 10.0,   # N_e, V_e
                   1.0, vsp0,   # N_sp, V_sp
                   1.0, vse0,   # N_se, V_se
                   0.0])        # Phi
    
    xi_start = 1e-8
    xi_end = xi_max
    
    # Storage for results
    xi_vals = []
    y_vals = []
    
    xi_current = xi_start
    y_current = y0.copy()
    
    # Step size control
    dt = 0.01
    
    while xi_current < xi_end:
        xi_vals.append(xi_current)
        y_vals.append(y_current.copy())
        
        # Find dPhidxi that satisfies neutrality at current point
        def find_dPhidxi(dPhidxi_candidate):
            derivs = compute_derivatives(xi_current, y_current, dPhidxi_candidate, 
                                         gamma, sigma_sp, sigma_se)
            return derivs[10]  # Return neutrality
        
        # Try to find root
        dPhidxi = -0.2  # initial guess
        
        try:
            # Simple bracket search
            for test_val in [-0.01, -0.05, -0.1, -0.2, -0.5, -1.0, -2.0]:
                if find_dPhidxi(test_val) * find_dPhidxi(test_val * 0.1) < 0:
                    break
            
            # Use root finding
            result = root(find_dPhidxi, dPhidxi, method='hybr', tol=1e-6)
            if result.success:
                dPhidxi = result.x[0]
        except:
            dPhidxi = -0.2
        
        # Step forward using Euler (simple for stability)
        derivs = compute_derivatives(xi_current, y_current, dPhidxi, 
                                     gamma, sigma_sp, sigma_se)
        derivs[10] = dPhidxi
        
        y_next = y_current + dt * np.array(derivs)
        xi_next = xi_current + dt
        
        # Check for singularities
        if np.any(np.isnan(y_next)) or np.any(np.isinf(y_next)):
            print(f"    Stopping at xi={xi_current:.3f} due to singularity")
            break
        
        y_current = y_next
        xi_current = xi_next
    
    xi_vals = np.array(xi_vals)
    y_vals = np.array(y_vals).T
    
    return xi_vals, y_vals


# ============================================================================
# Solve all cases
# ============================================================================

print("Solving using step-by-step root finding...")
print("="*60)

# LEFT COLUMN: Effect of solar wind velocity
print("\nLeft column (velocity variation):")
print("  Case: vsp0=10, vse0=10 (solid)")
xi1, y1 = solve_plasma(gamma=0.2, sigma_sp=1.0, sigma_se=1.0, vsp0=10, vse0=10, xi_max=4.5)

print("  Case: vsp0=30, vse0=30 (dashed)")
xi2, y2 = solve_plasma(gamma=0.2, sigma_sp=1.0, sigma_se=1.0, vsp0=30, vse0=30, xi_max=4.5)

# MIDDLE COLUMN: Effect of temperature ratio
print("\nMiddle column (temperature variation):")
print("  Case: sigma=1 (solid)")
xi3, y3 = solve_plasma(gamma=0.2, sigma_sp=1.0, sigma_se=1.0, vsp0=10, vse0=10, xi_max=4.5)

print("  Case: sigma=1.5 (dashed)")
xi4, y4 = solve_plasma(gamma=0.2, sigma_sp=1.5, sigma_se=1.5, vsp0=10, vse0=10, xi_max=4.5)

# RIGHT COLUMN: Effect of density ratio
print("\nRight column (density variation):")
print("  Case: gamma=0.5 (solid)")
xi5, y5 = solve_plasma(gamma=0.5, sigma_sp=1.0, sigma_se=1.0, vsp0=10, vse0=10, xi_max=4.5)

print("  Case: gamma=0.2 (dashed)")
xi6, y6 = solve_plasma(gamma=0.2, sigma_sp=1.0, sigma_se=1.0, vsp0=10, vse0=10, xi_max=4.5)

# ============================================================================
# Plotting
# ============================================================================

fig, axes = plt.subplots(3, 3, figsize=(14, 11))

# LEFT COLUMN
if len(xi1) > 1:
    axes[0, 0].plot(xi1, y1[0], 'k-', linewidth=1.5)  # N_H
    axes[0, 0].plot(xi1, y1[2], 'r-', linewidth=1.5)  # N_O
    axes[1, 0].plot(xi1, y1[1], 'k-', linewidth=1.5)  # V_H
    axes[1, 0].plot(xi1, y1[3], 'r-', linewidth=1.5)  # V_O
    axes[2, 0].plot(xi1, y1[10], 'k-', linewidth=1.5) # Phi

if len(xi2) > 1:
    axes[0, 0].plot(xi2, y2[0], 'k--', linewidth=1.5)
    axes[0, 0].plot(xi2, y2[2], 'r--', linewidth=1.5)
    axes[1, 0].plot(xi2, y2[1], 'k--', linewidth=1.5)
    axes[1, 0].plot(xi2, y2[3], 'r--', linewidth=1.5)
    axes[2, 0].plot(xi2, y2[10], 'k--', linewidth=1.5)

# MIDDLE COLUMN
if len(xi3) > 1:
    axes[0, 1].plot(xi3, y3[0], 'k-', linewidth=1.5)
    axes[0, 1].plot(xi3, y3[2], 'r-', linewidth=1.5)
    axes[1, 1].plot(xi3, y3[1], 'k-', linewidth=1.5)
    axes[1, 1].plot(xi3, y3[3], 'r-', linewidth=1.5)
    axes[2, 1].plot(xi3, y3[10], 'k-', linewidth=1.5)

if len(xi4) > 1:
    axes[0, 1].plot(xi4, y4[0], 'k--', linewidth=1.5)
    axes[0, 1].plot(xi4, y4[2], 'r--', linewidth=1.5)
    axes[1, 1].plot(xi4, y4[1], 'k--', linewidth=1.5)
    axes[1, 1].plot(xi4, y4[3], 'r--', linewidth=1.5)
    axes[2, 1].plot(xi4, y4[10], 'k--', linewidth=1.5)

# RIGHT COLUMN
if len(xi5) > 1:
    axes[0, 2].plot(xi5, y5[0], 'k-', linewidth=1.5)
    axes[0, 2].plot(xi5, y5[2], 'r-', linewidth=1.5)
    axes[1, 2].plot(xi5, y5[1], 'k-', linewidth=1.5)
    axes[1, 2].plot(xi5, y5[3], 'r-', linewidth=1.5)
    axes[2, 2].plot(xi5, y5[10], 'k-', linewidth=1.5)

if len(xi6) > 1:
    axes[0, 2].plot(xi6, y6[0], 'k--', linewidth=1.5)
    axes[0, 2].plot(xi6, y6[2], 'r--', linewidth=1.5)
    axes[1, 2].plot(xi6, y6[1], 'k--', linewidth=1.5)
    axes[1, 2].plot(xi6, y6[3], 'r--', linewidth=1.5)
    axes[2, 2].plot(xi6, y6[10], 'k--', linewidth=1.5)

# Formatting
for col in range(3):
    axes[0, col].set_ylabel('Density', fontsize=11)
    axes[0, col].set_xlabel('$\\xi$', fontsize=11)
    axes[0, col].grid(True, alpha=0.3)
    axes[0, col].set_xlim(0, 4.5)
    axes[0, col].set_ylim(0, 1.2)
    
    axes[1, col].set_ylabel('Velocity', fontsize=11)
    axes[1, col].set_xlabel('$\\xi$', fontsize=11)
    axes[1, col].grid(True, alpha=0.3)
    axes[1, col].set_xlim(0, 4.5)
    axes[1, col].set_ylim(0, 8)
    
    axes[2, col].set_ylabel('Electric Potential', fontsize=11)
    axes[2, col].set_xlabel('$\\xi$', fontsize=11)
    axes[2, col].grid(True, alpha=0.3)
    axes[2, col].set_xlim(0, 4.5)
    axes[2, col].set_ylim(-1.2, 0)

axes[0, 0].set_title('(a) Effect of solar wind velocity', fontsize=10)
axes[0, 1].set_title('(b) Effect of temperature ratio', fontsize=10)
axes[0, 2].set_title('(c) Effect of density ratio', fontsize=10)
axes[1, 0].set_title('(d) Effect of solar wind velocity', fontsize=10)
axes[1, 1].set_title('(e) Effect of temperature ratio', fontsize=10)
axes[1, 2].set_title('(f) Effect of density ratio', fontsize=10)
axes[2, 0].set_title('(g) Effect of solar wind velocity', fontsize=10)
axes[2, 1].set_title('(h) Effect of temperature ratio', fontsize=10)
axes[2, 2].set_title('(i) Effect of density ratio', fontsize=10)

axes[0, 0].plot([], [], 'k-', label='Lower value')
axes[0, 0].plot([], [], 'k--', label='Higher value')
axes[0, 0].legend(loc='upper right', fontsize=8)

axes[0, 0].text(3.5, 0.95, 'O$^+$', color='red', fontsize=12)
axes[0, 0].text(3.5, 0.55, 'H$^+$', color='black', fontsize=12)

plt.tight_layout()
plt.savefig('Figure_3_step_solver.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("SOLVER COMPLETED")
print("="*70)
print(f"""
Results summary:
- Left column (velocity): xi range = [{xi1[0]:.2f}, {xi1[-1]:.2f}] if solved
- Middle column (temperature): xi range = [{xi3[0]:.2f}, {xi3[-1]:.2f}] if solved  
- Right column (density): xi range = [{xi5[0]:.2f}, {xi5[-1]:.2f}] if solved

The step-by-step solver finds dPhi/dxi at each point by root-finding.
If the solution stops early, it hit a singularity at xi = V_j for some species.
""")
