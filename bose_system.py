import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

# --------------------------------------------------
# Parameters for the near-degenerate Bose system
# --------------------------------------------------
N = 1e5           # Total number of bosons
d = 5             # Number of nearly degenerate levels
delta = 1e-3      # Energy splitting between these low-lying levels

# Define the energies of the discrete near-degenerate levels:
discrete_energies = np.array([i * delta for i in range(d)])

# For the continuum part, we mimic the excited states.
# In our original discrete model, levels for i>=d were given by:
#   E_i = (i - d + 1) + (d-1)*delta,
# i.e. with spacing 1. In the continuum limit the density of states is constant.
E_cut = 1 + (d - 1) * delta   # starting energy for the continuum (matching the first excited level)
E_max = 100.0                 # an upper cutoff (choose sufficiently high so the tail is negligible)

# --------------------------------------------------
# Functions using the DOS approach
# --------------------------------------------------
def particle_number(mu, T):
    """
    Returns the total number of particles computed as the sum over the discrete
    near-degenerate states plus the continuum contribution with DOS g(E)=1.
    """
    # Discrete contribution:
    N_disc = np.sum([1.0 / (np.exp((E - mu) / T) - 1) for E in discrete_energies])
    
    # Continuum contribution:
    def integrand(E):
        return 1.0 / (np.exp((E - mu) / T) - 1)
    
    N_cont, err = quad(integrand, E_cut, E_max, limit=200)
    return N_disc + N_cont

def total_energy(mu, T):
    """
    Returns the total energy of the system.
    """
    # Energy from the discrete near-degenerate levels:
    E_disc = np.sum([E / (np.exp((E - mu) / T) - 1) for E in discrete_energies])
    
    # Energy from the continuum part:
    def integrand(E):
        return E / (np.exp((E - mu) / T) - 1)
    
    E_cont, err = quad(integrand, E_cut, E_max, limit=200)
    return E_disc + E_cont

# --------------------------------------------------
# Set up temperature range and arrays to store results
# --------------------------------------------------
T_min = 0.1
T_max = 5.0
num_T = 100
T_array = np.linspace(T_min, T_max, num_T)

mu_array = np.zeros_like(T_array)
n0_array = np.zeros_like(T_array)       # Ground state occupation (<n0> for E0=0)
E_total_array = np.zeros_like(T_array)    # Total energy of the system

# --------------------------------------------------
# Main loop: solve for mu(T) and compute physical quantities
# --------------------------------------------------
for i, T in enumerate(T_array):
    
    # Define the function whose root (in mu) gives the correct particle number:
    def f(mu):
        return particle_number(mu, T) - N
    
    # For a Bose gas, μ must be less than the lowest energy (here, E0 = 0).
    # We choose a bracket: mu in [mu_min, mu_max] with:
    mu_min = -100 * T    # very negative value gives very low occupation
    mu_max = -1e-12      # slightly below zero
    mu_sol = brentq(f, mu_min, mu_max)
    mu_array[i] = mu_sol
    
    # Ground state occupation (E0 = 0)
    n0_array[i] = 1.0 / (np.exp((0 - mu_sol) / T) - 1)
    
    # Total energy of the system:
    E_total_array[i] = total_energy(mu_sol, T)

# --------------------------------------------------
# Compute numerical derivatives
# --------------------------------------------------
dn0_dT = np.gradient(n0_array, T_array)        # derivative of <n0> with respect to T
Cv_array = np.gradient(E_total_array, T_array)   # Specific heat: Cv = dE_total/dT

# --------------------------------------------------
# Plotting the results
# --------------------------------------------------
plt.figure(figsize=(12, 10))

# 1. Negative chemical potential: -μ vs. Temperature
plt.subplot(3, 2, 1)
plt.plot(T_array, -mu_array, 'b-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('-μ')
plt.title('Negative Chemical Potential vs. T')

# 2. Ground state occupation <n0> vs. Temperature
plt.subplot(3, 2, 2)
plt.plot(T_array, n0_array, 'r-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('<n₀>')
plt.title('Ground State Occupation vs. T')

# 3. Logarithm of ground state occupation, log(<n0>), vs. Temperature
plt.subplot(3, 2, 3)
plt.semilogy(T_array, n0_array, 'm-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('log(<n₀>)')
plt.title('Log of Ground State Occupation vs. T')

# 4. Negative gradient of ground state occupation, -d<n0>/dT, vs. Temperature
plt.subplot(3, 2, 4)
plt.plot(T_array, -dn0_dT, 'g-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('-d<n₀>/dT')
plt.title('Negative Gradient of <n₀> vs. T')

# 5. Specific Heat Cv vs. Temperature
plt.subplot(3, 2, 5)
plt.plot(T_array, Cv_array, 'k-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('C_v')
plt.title('Specific Heat C_v vs. T')

plt.tight_layout()
plt.show()
plt.savefig("nonBEC_bose_system.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gamma, zeta

# =============================================================================
# Custom polylog implementation
# =============================================================================
def polylog(s, z, tol=1e-10, maxiter=100000):
    """
    Compute the polylogarithm function Li_s(z) = sum_{k=1}^∞ z^k / k^s.
    
    For z close to 1 (|z-1| < tol), returns the Riemann zeta function ζ(s).
    Otherwise, sums the series until the next term is smaller than tol.
    """
    if np.abs(z - 1) < 1e-12:
        return zeta(s, 1)  # zeta(s, 1) is the Riemann zeta function ζ(s)
    sum_val = 0.0
    k = 1
    term = z**k / k**s
    while np.abs(term) > tol and k < maxiter:
        sum_val += term
        k += 1
        term = z**k / k**s
    return sum_val

# --------------------------------------------------
# System Parameters and Units
# --------------------------------------------------
N = 1e5   # Total number of bosons

# In a 3D ideal Bose gas (in appropriate units) the excited-state density of states leads to:
#   N_ex(T, mu=0) = T^(3/2) * Gamma(3/2) * ζ(3/2)
Gamma_3_2 = np.sqrt(np.pi) / 2.0         # Gamma(3/2)
zeta_3_2 = polylog(1.5, 1.0)             # ζ(3/2) ≈ 2.612

# Critical temperature defined by N = T_c^(3/2) * Gamma(3/2) * ζ(3/2)
T_c = (N / (Gamma_3_2 * zeta_3_2))**(2/3)
print("Critical Temperature T_c =", T_c)

# --------------------------------------------------
# Temperature Range for the Simulation
# --------------------------------------------------
T_min = 0.1
T_max = 1.5 * T_c   # go a bit above T_c to see the singular behavior
num_T = 200
T_array = np.linspace(T_min, T_max, num_T)

# Arrays to store computed quantities:
mu_array = np.zeros_like(T_array)   # Chemical potential μ(T)
n0_array = np.zeros_like(T_array)   # Ground state occupation n0 = ⟨n₀⟩
E_array  = np.zeros_like(T_array)    # Total energy of the system

# --------------------------------------------------
# Main Loop: Compute μ, n0, and E for each temperature T
# --------------------------------------------------
for i, T in enumerate(T_array):
    if T < T_c:
        # For T below T_c the excited states cannot hold all N particles.
        # The prescription is to fix μ = 0 and let the "excess" particles condense.
        mu = 0.0
        mu_array[i] = mu
        # Number of particles in the excited states:
        N_ex = T**(1.5) * Gamma_3_2 * polylog(1.5, 1.0)  # polylog(3/2,1)=ζ(3/2)
        # The ground state occupation:
        n0_array[i] = N - N_ex
        # Energy (only excited states contribute, ground state energy = 0):
        E_array[i] = T**(2.5) * gamma(2.5) * polylog(2.5, 1.0)
    else:
        # For T above T_c, the chemical potential μ is less than 0.
        # Solve for μ(T) from the equation:
        #   N = [1/(e^{-μ/T}-1)] + T^(3/2)*Gamma(3/2)*Li_{3/2}(e^{μ/T})
        # The first term is the ground state contribution.
        def f(mu):
            return (1.0/(np.exp(-mu/T)-1) +
                    T**(1.5) * Gamma_3_2 * polylog(1.5, np.exp(mu/T))
                   ) - N
        # μ must lie below 0. We bracket the root between a very negative value and ~0.
        mu_solution = brentq(f, -100*T, -1e-12)
        mu_array[i] = mu_solution
        # Ground state occupation (should be very small for T > T_c)
        n0_array[i] = 1.0 / (np.exp(-mu_solution/T) - 1)
        # Energy (only excited states contribute, with energy scaling as E ~ T^(5/2))
        E_array[i] = T**(2.5) * gamma(2.5) * polylog(2.5, np.exp(mu_solution/T))

# --------------------------------------------------
# Compute Numerical Derivatives:
#   - d⟨n₀⟩/dT for the ground state occupation derivative,
#   - Cv = dE/dT for the specific heat.
# --------------------------------------------------
dn0_dT = np.gradient(n0_array, T_array)
Cv_array = np.gradient(E_array, T_array)

# --------------------------------------------------
# Plotting the Results
# --------------------------------------------------
plt.figure(figsize=(12, 12))

# 1. Negative Chemical Potential: -μ vs Temperature
plt.subplot(3, 2, 1)
plt.plot(T_array, -mu_array, 'b-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('-μ')
plt.title('Negative Chemical Potential vs T')

# 2. Ground State Occupation n₀ vs Temperature
plt.subplot(3, 2, 2)
plt.plot(T_array, n0_array, 'r-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('n₀')
plt.title('Ground State Occupation vs T')

# 3. Logarithm of Ground State Occupation, log(n₀), vs Temperature
plt.subplot(3, 2, 3)
plt.semilogy(T_array, n0_array, 'm-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('log(n₀)')
plt.title('Log of Ground State Occupation vs T')

# 4. Negative Gradient of Ground State Occupation: -d⟨n₀⟩/dT vs Temperature
plt.subplot(3, 2, 4)
plt.plot(T_array, -dn0_dT, 'g-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('-d⟨n₀⟩/dT')
plt.title('Negative Gradient of Ground State Occupation vs T')

# 5. Specific Heat Cv vs Temperature
plt.subplot(3, 2, 5)
plt.plot(T_array, Cv_array, 'k-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('C_v')
plt.title('Specific Heat C_v vs T')

plt.tight_layout()
plt.show()
plt.savefig("BEC_bose_system.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------------------------
# System Parameters
# ---------------------------
N = 1e5         # Total number of bosons
d = 5           # Number of nearly degenerate levels
delta = 1e-3    # Energy splitting among the nearly-degenerate states
N_levels = 500  # Total number of energy levels in the simulation

# Construct the energy spectrum:
energies = np.zeros(N_levels)
# The first d levels: nearly degenerate (spread by delta)
for i in range(d):
    energies[i] = i * delta
# For i >= d, choose a “normal” dispersion (here, linear)
for i in range(d, N_levels):
    # The offset ensures a smooth transition from the low-lying levels.
    energies[i] = (i - d + 1) + (d - 1) * delta

# ---------------------------
# Helper Function: Total Number of Particles
# ---------------------------
def total_particles(mu, T, energies):
    """Calculate total number of particles for given chemical potential mu and temperature T."""
    return np.sum(1.0 / (np.exp((energies - mu) / T) - 1))

# ---------------------------
# Temperature Range
# ---------------------------
T_min = 1
T_max = 300
num_T = 10000
T_array = np.linspace(T_min, T_max, num_T)

# Arrays to store computed quantities
mu_array = np.zeros(num_T)
n0_array = np.zeros(num_T)         # Occupation of the lowest level (E0 = 0)
E_total_array = np.zeros(num_T)      # Total energy

# ---------------------------
# Main Loop: Solve for mu(T) and Compute Quantities
# ---------------------------
for idx, T in enumerate(T_array):
    # Define the function whose zero gives the correct chemical potential:
    def f(mu):
        return total_particles(mu, T, energies) - N
    
    # Since the Bose distribution requires mu < E0 (with E0 = 0), we search for mu in a range well below 0.
    mu_min = -100 * T
    mu_max = -1e-12  # just below 0
    
    # Solve for mu using a robust root-finding algorithm:
    mu_sol = brentq(f, mu_min, mu_max)
    mu_array[idx] = mu_sol
    
    # Compute the occupation of the lowest energy state (ground state)
    n0_array[idx] = 1.0 / (np.exp((energies[0] - mu_sol) / T) - 1)
    
    # Compute the total energy of the system
    E_total_array[idx] = np.sum(energies / (np.exp((energies - mu_sol) / T) - 1))

# ---------------------------
# Numerical Derivatives
# ---------------------------
dn0_dT = np.gradient(n0_array, T_array)   # derivative of ground state occupation w.r.t T
Cv_array = np.gradient(E_total_array, T_array)  # Specific heat C_v = dE/dT

# ---------------------------
# Plotting the Results
# ---------------------------
plt.figure(figsize=(12, 10))

# 1. Negative chemical potential: -mu vs. Temperature
plt.subplot(3, 2, 1)
plt.plot(T_array, -mu_array, 'b-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('-μ')
plt.title('Negative Chemical Potential vs. T')

# 2. Ground state occupation <n₀> vs. Temperature
plt.subplot(3, 2, 2)
plt.plot(T_array, n0_array, 'r-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('<n₀>')
plt.title('Ground State Occupation vs. T')

# 3. Logarithm of ground state occupation, log(<n₀>), vs. Temperature
plt.subplot(3, 2, 3)
plt.semilogy(T_array, n0_array, 'm-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('log(<n₀>)')
plt.title('Log of Ground State Occupation vs. T')

# 4. Negative gradient of ground state occupation, -d<n₀>/dT, vs. Temperature
plt.subplot(3, 2, 4)
plt.plot(T_array, -dn0_dT, 'g-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('-d<n₀>/dT')
plt.title('Negative Derivative of <n₀> vs. T')

# 5. Specific Heat, C_v, vs. Temperature
plt.subplot(3, 2, 5)
plt.plot(T_array, Cv_array, 'k-', lw=2)
plt.xlabel('Temperature T')
plt.ylabel('C_v')
plt.title('Specific Heat C_v vs. T')

plt.tight_layout()
plt.show()
plt.savefig("nonBEC_bose_system2.png")