import numpy as np
import matplotlib.pyplot as plt

def compute_A(xi, y, params):
    alpha, beta, gamma, delta, sigma_H, sigma_O, sigma_e, sigma_sp, sigma_se, Q_H, Q_sp, Q_e, Q_se = params
    N_H, V_H, N_O, V_O, N_e, V_e, N_sp, V_sp, N_se, V_se, Phi = y
    u_H = V_H - xi
    u_O = V_O - xi
    u_e = V_e - xi
    u_sp = V_sp - xi
    u_se = V_se - xi

    A = np.zeros((6,6))
    # H
    A[0,0] = Q_H * sigma_H - u_H**2
    A[0,5] = Q_H * N_H
    # O
    A[1,1] = sigma_O - u_O**2
    A[1,5] = N_O
    # e
    A[2,2] = Q_e * sigma_e - u_e**2
    A[2,5] = - Q_e * N_e
    # sp
    A[3,3] = Q_sp * sigma_sp - u_sp**2
    A[3,5] = Q_sp * N_sp
    # se
    A[4,4] = Q_se * sigma_se - u_se**2
    A[4,5] = - Q_se * N_se
    # neutrality
    A[5,0] = alpha
    A[5,1] = 1
    A[5,2] = -beta
    A[5,3] = gamma
    A[5,4] = -delta
    A[5,5] = 0
    return A

def run_integration(params, V_H0, V_O0, V_e0, V_sp0, V_se0, xi_max=3.0, dxi=0.01, phi_prime_target=-0.4):
    alpha, beta, gamma, delta, sigma_H, sigma_O, sigma_e, sigma_sp, sigma_se, Q_H, Q_sp, Q_e, Q_se = params
    n_steps = int(xi_max / dxi) + 1
    xi_list = np.linspace(0, xi_max, n_steps)
    y_list = np.zeros((n_steps, 11))
    
    y = np.array([1.0, V_H0, 1.0, V_O0, 1.0, V_e0, 1.0, V_sp0, 1.0, V_se0, 0.0])
    y_list[0] = y.copy()   # better to copy

    for i in range(1, n_steps):
        xi = xi_list[i-1]

        # Extract current values (this was missing!)
        N_H  = y[0]
        V_H  = y[1]
        N_O  = y[2]
        V_O  = y[3]
        N_e  = y[4]
        V_e  = y[5]
        N_sp = y[6]
        V_sp = y[7]
        N_se = y[8]
        V_se = y[9]
        Phi  = y[10]

        A = compute_A(xi, y, params)
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        kernel = vh[-1, :] 

        # Normalize kernel so dΦ/dξ ≈ target value
        if abs(kernel[5]) > 1e-10:
            scale = phi_prime_target / kernel[5]
        else:
            scale = 0.0

        x = scale * kernel
        N_H_prime, N_O_prime, N_e_prime, N_sp_prime, N_se_prime, Phi_prime = x

        # Now use the extracted current values
        V_H_prime  = - (V_H - xi) / N_H  * N_H_prime   if N_H  > 1e-6 else 0.0
        V_O_prime  = - (V_O - xi) / N_O  * N_O_prime   if N_O  > 1e-6 else 0.0
        V_e_prime  = - (V_e - xi) / N_e  * N_e_prime   if N_e  > 1e-6 else 0.0
        V_sp_prime = - (V_sp - xi)/ N_sp * N_sp_prime  if N_sp > 1e-6 else 0.0
        V_se_prime = - (V_se - xi)/ N_se * N_se_prime  if N_se > 1e-6 else 0.0

        dy = np.array([
            N_H_prime,  V_H_prime,
            N_O_prime,  V_O_prime,
            N_e_prime,  V_e_prime,
            N_sp_prime, V_sp_prime,
            N_se_prime, V_se_prime,
            Phi_prime
        ])

        y = y + dy * dxi

        # Prevent negative / near-zero densities
        y[0] = max(y[0], 1e-4)
        y[2] = max(y[2], 1e-4)
        y[4] = max(y[4], 1e-4)
        y[6] = max(y[6], 1e-4)
        y[8] = max(y[8], 1e-4)

        y_list[i] = y.copy()

    return xi_list, y_list
    
# Parameters
alpha = 0.2
beta = 1.2
sigma_H = 0.01
sigma_O = 0.01
sigma_e = 1.0
Q_H = 16.0
Q_sp = 16.0
Q_e = 30000.0
Q_se = 30000.0
V_H0 = 5.0
V_O0 = 1.0
V_e0 = 10.0

# For left column (vary V_sp/se)
params_left = (alpha, beta, 0.5, 0.5, sigma_H, sigma_O, sigma_e, 1.0, 1.0, Q_H, Q_sp, Q_e, Q_se)
xi, y_left_solid = run_integration(params_left, V_H0, V_O0, V_e0, 10.0, 10.0)
xi, y_left_dashed = run_integration(params_left, V_H0, V_O0, V_e0, 30.0, 30.0)

# For middle column (vary sigma_sp/se)
params_middle = (alpha, beta, 0.5, 0.5, sigma_H, sigma_O, sigma_e, 1.5, 1.5, Q_H, Q_sp, Q_e, Q_se)
xi, y_middle_solid = run_integration(params_left, V_H0, V_O0, V_e0, 10.0, 10.0)  # Reuse left solid as base
xi, y_middle_dashed = run_integration(params_middle, V_H0, V_O0, V_e0, 10.0, 10.0)

# For right column (vary gamma, delta)
params_right = (alpha, beta, 0.2, 0.2, sigma_H, sigma_O, sigma_e, 1.0, 1.0, Q_H, Q_sp, Q_e, Q_se)
xi, y_right_solid = run_integration(params_left, V_H0, V_O0, V_e0, 10.0, 10.0)  # Reuse left solid as base
xi, y_right_dashed = run_integration(params_right, V_H0, V_O0, V_e0, 10.0, 10.0)

# Density Left
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_left_solid[:,0], color='black', linestyle='solid', label='H solid')
ax.plot(xi, y_left_solid[:,2], color='red', linestyle='solid', label='O solid')
ax.plot(xi, y_left_dashed[:,0], color='black', linestyle='dashed', label='H dashed')
ax.plot(xi, y_left_dashed[:,2], color='red', linestyle='dashed', label='O dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Density N')
ax.set_title('Density - Left (V vary)')
ax.legend()
plt.savefig('density_left.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Velocity Left
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_left_solid[:,1], color='black', linestyle='solid', label='H solid')
ax.plot(xi, y_left_solid[:,3], color='red', linestyle='solid', label='O solid')
ax.plot(xi, y_left_dashed[:,1], color='black', linestyle='dashed', label='H dashed')
ax.plot(xi, y_left_dashed[:,3], color='red', linestyle='dashed', label='O dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Velocity V')
ax.set_title('Velocity - Left (V vary)')
ax.legend()
plt.savefig('velocity_left.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Potential Left
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_left_solid[:,10], color='black', linestyle='solid', label='solid')
ax.plot(xi, y_left_dashed[:,10], color='black', linestyle='dashed', label='dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Potential Φ')
ax.set_title('Potential - Left (V vary)')
ax.legend()
plt.savefig('potential_left.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Density Middle
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_middle_solid[:,0], color='black', linestyle='solid', label='H solid')
ax.plot(xi, y_middle_solid[:,2], color='red', linestyle='solid', label='O solid')
ax.plot(xi, y_middle_dashed[:,0], color='black', linestyle='dashed', label='H dashed')
ax.plot(xi, y_middle_dashed[:,2], color='red', linestyle='dashed', label='O dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Density N')
ax.set_title('Density - Middle (σ vary)')
ax.legend()
plt.savefig('density_middle.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Velocity Middle
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_middle_solid[:,1], color='black', linestyle='solid', label='H solid')
ax.plot(xi, y_middle_solid[:,3], color='red', linestyle='solid', label='O solid')
ax.plot(xi, y_middle_dashed[:,1], color='black', linestyle='dashed', label='H dashed')
ax.plot(xi, y_middle_dashed[:,3], color='red', linestyle='dashed', label='O dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Velocity V')
ax.set_title('Velocity - Middle (σ vary)')
ax.legend()
plt.savefig('velocity_middle.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Potential Middle
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_middle_solid[:,10], color='black', linestyle='solid', label='solid')
ax.plot(xi, y_middle_dashed[:,10], color='black', linestyle='dashed', label='dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Potential Φ')
ax.set_title('Potential - Middle (σ vary)')
ax.legend()
plt.savefig('potential_middle.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Density Right
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_right_solid[:,0], color='black', linestyle='solid', label='H solid')
ax.plot(xi, y_right_solid[:,2], color='red', linestyle='solid', label='O solid')
ax.plot(xi, y_right_dashed[:,0], color='black', linestyle='dashed', label='H dashed')
ax.plot(xi, y_right_dashed[:,2], color='red', linestyle='dashed', label='O dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Density N')
ax.set_title('Density - Right (γ vary)')
ax.legend()
plt.savefig('density_right.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Velocity Right
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_right_solid[:,1], color='black', linestyle='solid', label='H solid')
ax.plot(xi, y_right_solid[:,3], color='red', linestyle='solid', label='O solid')
ax.plot(xi, y_right_dashed[:,1], color='black', linestyle='dashed', label='H dashed')
ax.plot(xi, y_right_dashed[:,3], color='red', linestyle='dashed', label='O dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Velocity V')
ax.set_title('Velocity - Right (γ vary)')
ax.legend()
plt.savefig('velocity_right.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# Potential Right
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xi, y_right_solid[:,10], color='black', linestyle='solid', label='solid')
ax.plot(xi, y_right_dashed[:,10], color='black', linestyle='dashed', label='dashed')
ax.set_xlabel('ξ')
ax.set_ylabel('Potential Φ')
ax.set_title('Potential - Right (γ vary)')
ax.legend()
plt.savefig('potential_right.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

print("9 separate PDF files generated: density/velocity/potential for left/middle/right columns.")
