"""
Task 2: Bose Einstein Condensate (BEC)

We consider a system of N indistinguishable bosons in a 2-level system with energies:
    Ground state: 0
    Excited state: ε

-----------------------------------------------------
Part (a): Microstates
-----------------------------------------------------
Each boson can occupy either the ground state (energy 0) or the excited state (energy ε).
A microstate is specified by the occupancy numbers (n0, nε) that satisfy:
    n0 + nε = N
and the total energy is:
    E = nε * ε

In a classical (distinguishable) picture, for a given nε the number of ways (degeneracy) to
choose which bosons are excited is given by the binomial coefficient:
    C(N, nε) = (N choose nε)

-----------------------------------------------------
Part (b): (Classical) Partition Function under the Canonical Ensemble
-----------------------------------------------------
In the classical treatment we sum over all microstates, including their multiplicity.
The classical partition function is given by:

    Z_C = Σₙₑ₌₀ᴺ  [ (N choose nε) * exp(-β * ε * nε) ]

where β = 1/(kB*T) (with kB = 1 for simplicity). Thus, the probability of finding a 
microstate with nε particles in the excited state (i.e. energy E = nε * ε) is:

    P(nε) = [ (N choose nε) * exp(-β * ε * nε) ] / Z_C
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# ---------------------- Parameters ----------------------
N = 50            # Total number of bosons
epsilon = 1.0     # Energy gap between the two levels
kB = 1.0          # Boltzmann constant (set to 1 for simplicity)
T_min = 0.1       # Minimum temperature (avoid T=0 to prevent divergence)
T_max = 5.0       # Maximum temperature
num_T = 200       # Number of temperature points

# Temperature array for classical analysis
T_array = np.linspace(T_min, T_max, num_T)

# Arrays to store the classical average occupation numbers
n0_classical = np.zeros_like(T_array)  # ⟨n0⟩_C : Average ground state occupancy
ne_classical = np.zeros_like(T_array)  # ⟨nε⟩_C : Average excited state occupancy

# ---------------------- Classical Partition Function and Averages (Parts b & c) ----------------------
# For each temperature T, compute the classical partition function:
#   Z_C = Σₙₑ₌₀ᴺ  [ (N choose nε) * exp(-β * ε * nε) ]
# and then calculate the average excited state occupation:
#   ⟨nε⟩_C = (1/Z_C) * Σₙₑ₌₀ᴺ [ nε * (N choose nε) * exp(-β * ε * nε) ]
# Finally, the ground state occupation is:
#   ⟨n0⟩_C = N - ⟨nε⟩_C

for i, T in enumerate(T_array):
    beta = 1.0 / (kB * T)
    Z_classical = 0.0  # Initialize classical partition function at temperature T
    avg_ne = 0.0       # For summing up the weighted contribution of excited state occupancy
    
    # Sum over all possible values of nε (0 to N)
    for n_e in range(N + 1):
        weight = comb(N, n_e) * np.exp(-beta * epsilon * n_e)
        Z_classical += weight
        avg_ne += n_e * weight
    avg_ne /= Z_classical      # Compute ⟨nε⟩_C
    avg_n0 = N - avg_ne        # Compute ⟨n0⟩_C

    ne_classical[i] = avg_ne
    n0_classical[i] = avg_n0

# ---------------------- Plotting Classical Average Occupation Numbers (Part c) ----------------------
plt.figure(figsize=(8, 5))
plt.plot(T_array, n0_classical, label=r'$\langle n_0\rangle_C$ (Ground state)')
plt.plot(T_array, ne_classical, label=r'$\langle n_\epsilon\rangle_C$ (Excited state)')
plt.xlabel("Temperature T")
plt.ylabel("Average Occupation Number")
plt.title("Classical Average Occupation Numbers vs Temperature")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("classical_occupation.png")

"""
-----------------------------------------------------
Part (d): (Quantum) Partition Function under the Canonical Ensemble
-----------------------------------------------------
In the quantum treatment for indistinguishable bosons, the microstates are labeled only
by their occupancy numbers, and the multiplicity factors are not included. The quantum 
partition function is therefore:

    Z = Σₙₑ₌₀ᴺ  exp(-β * ε * nε)

Since this is a finite geometric series, it can be summed in closed form:

    Z = [1 - exp(-β * ε * (N+1))] / [1 - exp(-β * ε)]

and the probability of a state with nε excited bosons is:

    P(nε) = exp(-β * ε * nε) / Z
"""
# ---------------------- (Quantum) Partition Function and Probability Distribution (Part d) ----------------------
# For a fixed temperature T0, we now compute the quantum partition function.
# In the quantum treatment, there is no multiplicity factor because the bosons are indistinguishable.
# The quantum partition function is given by:
#
#    Z = Σₙₑ₌₀ᴺ exp(-β * ε * nε)
#
# This finite geometric series can be summed in closed form as:
#
#    Z = [1 - exp(-β * ε * (N+1))] / [1 - exp(-β * ε)]
#
# The probability of having nε bosons in the excited state is then:
#
#    P(nε) = exp(-β * ε * nε) / Z

# Choose a temperature T0 for the quantum analysis:

T0 = 10.0
beta0 = 1.0 / (kB * T0)

# Compute the quantum partition function Z using the closed-form expression.
if np.abs(np.exp(-beta0 * epsilon) - 1) > 1e-10:
    Z_quantum = (1 - np.exp(-beta0 * epsilon * (N + 1))) / (1 - np.exp(-beta0 * epsilon))
else:
    Z_quantum = N + 1  # In the limit β*ε -> 0, each term contributes ~1

# For comparison, we also compute the classical partition function at T0:
Z_classical_T0 = sum([comb(N, n_e) * np.exp(-beta0 * epsilon * n_e) for n_e in range(N + 1)])

# Print the partition functions:
print("At Temperature T = {:.2f}:".format(T0))
print("Classical Partition Function, Z_C = {:.4e}".format(Z_classical_T0))
print("Quantum Partition Function, Z   = {:.4e}".format(Z_quantum))

# Compute the probability distributions for the number of excited bosons in both treatments:
probs_classical = np.zeros(N + 1)
probs_quantum = np.zeros(N + 1)

for n_e in range(N + 1):
    # Classical probability includes the multiplicity factor
    probs_classical[n_e] = comb(N, n_e) * np.exp(-beta0 * epsilon * n_e) / Z_classical_T0
    # Quantum probability omits the multiplicity factor
    probs_quantum[n_e] = np.exp(-beta0 * epsilon * n_e) / Z_quantum

# ---------------------- Plotting the Probability Distributions ----------------------
plt.figure(figsize=(8, 5))
n_e_vals = np.arange(N + 1)
plt.plot(n_e_vals, probs_classical, 'o-', label="Classical")
plt.plot(n_e_vals, probs_quantum, 's-', label="Quantum")
plt.xlabel(r"Number of particles in excited state, $n_\epsilon$")
plt.ylabel("Probability")
plt.title("Probability Distribution at T = {:.2f}".format(T0))
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("quantum_classical_prob.png")
# ---------------------- Part (e): Quantum Average Occupation Numbers ----------------------
# Here we compute the quantum average number of excited particles as:
#    ⟨nε⟩ = (1/Z) · Σₙₑ₌₀ᴺ [ nε · exp(–β·ε·nε) ]
# and the ground state occupancy as:
#    ⟨n₀⟩ = N – ⟨nε⟩
# We then plot these averages as functions of temperature.

# Create a temperature array for the quantum averages
T_array_quantum = np.linspace(T_min, T_max, num_T)
ne_quantum = np.zeros_like(T_array_quantum)  # ⟨nε⟩ (excited state, quantum)
n0_quantum = np.zeros_like(T_array_quantum)  # ⟨n₀⟩ (ground state, quantum)

for i, T in enumerate(T_array_quantum):
    beta = 1.0 / (kB * T)
    Z_q = 0.0      # Quantum partition function (without multiplicity)
    numerator = 0.0  # For computing the weighted sum for ⟨nε⟩
    for n_e in range(N + 1):
        weight = np.exp(-beta * epsilon * n_e)
        Z_q += weight
        numerator += n_e * weight
    avg_ne_quantum = numerator / Z_q  # ⟨nε⟩ in the quantum case
    avg_n0_quantum = N - avg_ne_quantum  # ⟨n₀⟩ = N - ⟨nε⟩

    ne_quantum[i] = avg_ne_quantum
    n0_quantum[i] = avg_n0_quantum

# Plot the quantum average occupation numbers as a function of temperature.
plt.figure(figsize=(8, 5))
plt.plot(T_array_quantum, n0_quantum, label=r'$\langle n_0\rangle$ (Ground state, Quantum)')
plt.plot(T_array_quantum, ne_quantum, label=r'$\langle n_\epsilon\rangle$ (Excited state, Quantum)')
plt.xlabel("Temperature T")
plt.ylabel("Average Occupation Number")
plt.title("Quantum Average Occupation Numbers vs Temperature")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("quantum_occupation.png")

# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# =============================================================================
# PART (j): (Near)-degenerate Bose System WITHOUT BEC
#
# Goal: Design a Bose system that does not experience BEC so that thermodynamic 
# quantities are smooth functions of temperature. In conventional BEC the unique 
# (or weakly degenerate) ground state becomes macroscopically occupied below a 
# critical temperature. Here we prevent this singular behavior by “engineering” the 
# energy spectrum so that a large number of the lowest states are degenerate (or nearly
# degenerate). With many available low-energy states the density of states near zero 
# energy diverges, so the total particle number can be accommodated with μ remaining 
# strictly less than the lowest energy.
#
# Degeneracy Conditions and Physical Explanation:
#   - In a typical Bose gas (with a unique ground state) the low density of states 
#     at the bottom forces the chemical potential μ to approach the ground state 
#     energy as T is lowered, leading to condensation.
#   - By contrast, if many states share (or nearly share) the same low energy (here, set 
#     to 0), then the ground state “capacity” is very high. In our simulation we set 
#     the first M0 levels to have energy 0. Then, the contribution to the total number 
#     from these levels is 
#
#         ∑_{i=0}^{M0-1} 1/(exp(β*(0 - μ)) - 1) = M0/(exp(-βμ)-1)
#
#     which diverges as μ → 0. Thus, for any finite total particle number N_target, the 
#     solution of N(μ,T)=N_target will have μ < 0 for all T.
#
#   - Physically, this means that as T decreases the particles spread among many 
#     low-energy states rather than “condensing” into one. Therefore, none of the 
#     thermodynamic quantities (μ, ⟨n₀⟩, log⟨n₀⟩, –d⟨n₀⟩/dT, C_v) show singular 
#     behavior.
# =============================================================================

# ---------------------- System Parameters ----------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# =============================================================================
# Part (j): (Near)-degenerate Bose system, WITHOUT BEC
# =============================================================================
# We set up a system where the first M0 levels all have zero energy, creating a
# large degeneracy. Because of this high degeneracy, µ will remain strictly < 0
# for all temperatures, so no macroscopic occupation of a single state occurs.

# ---------------------- System Parameters ----------------------
N_target = int(1e5)   # Target total particle number
M_total = 1000        # Total number of energy levels
M0 = 500              # Number of degenerate ground states (energy = 0)
delta = 1e-3          # Energy spacing for levels above M0

# Create the energy array:
#   levels 0..M0-1 => 0 energy,
#   levels M0..M_total-1 => linear increase
energies = np.zeros(M_total)
energies[M0:] = delta * np.arange(1, M_total - M0 + 1)

# Temperature parameters
T_min = 0.01
T_max = 1.0
num_T = 300
T_array = np.linspace(T_min, T_max, num_T)

kB = 1.0  # Boltzmann constant (taken as 1 for convenience)

# ---------------------- Functions ----------------------
def total_number_nonBEC(mu, T, energies):
    """
    Returns total particle number N(mu,T) = sum_i [1 / (exp(β(ε_i - mu)) - 1)].
    Here, the large degeneracy at ε=0 ensures that we can always solve for mu < 0.
    """
    beta = 1.0 / (kB * T)
    with np.errstate(divide='ignore', invalid='ignore'):
        n_occup = 1.0 / (np.exp(beta * (energies - mu)) - 1.0)
    return np.nansum(n_occup)

def total_energy_nonBEC(mu, T, energies):
    """
    Returns total energy E(mu,T) = sum_i [ε_i / (exp(β(ε_i - mu)) - 1)].
    """
    beta = 1.0 / (kB * T)
    with np.errstate(divide='ignore', invalid='ignore'):
        n_occup = 1.0 / (np.exp(beta * (energies - mu)) - 1.0)
    return np.nansum(energies * n_occup)

def find_mu_nonBEC(T, energies, N_target):
    """
    Solve total_number_nonBEC(mu,T) = N_target for mu in the range [mu_low, mu_high],
    with mu_high < 0. Because of the large degeneracy, the solution is always < 0.
    """
    mu_low = -100.0     # sufficiently negative
    mu_high = -1e-10    # just below 0
    def f(mu):
        return total_number_nonBEC(mu, T, energies) - N_target
    mu_sol = bisect(f, mu_low, mu_high, xtol=1e-12)
    return mu_sol

# ---------------------- Arrays to Store Computed Quantities ----------------------
mu_array = np.zeros(num_T)      # chemical potential μ vs T
n0_array = np.zeros(num_T)      # total occupation in the degenerate manifold
E_array = np.zeros(num_T)       # total energy

# ---------------------- Main Loop ----------------------
for i, T in enumerate(T_array):
    mu_val = find_mu_nonBEC(T, energies, N_target)
    mu_array[i] = mu_val

    # Compute occupations for all levels at this mu, T
    beta = 1.0 / (kB * T)
    with np.errstate(divide='ignore', invalid='ignore'):
        occ_all = 1.0 / (np.exp(beta*(energies - mu_val)) - 1.0)

    # Summation over the M0 degenerate states at ε=0
    n0_array[i] = np.sum(occ_all[:M0])

    # Compute total energy (the degenerate states contribute 0 * n0 to E)
    E_array[i] = np.sum(energies * occ_all)

# ---------------------- Derived Quantities ----------------------
# log(ground-state occupation)
log_n0_array = np.log(n0_array + 1e-12)

# finite-difference derivatives
dT = T_array[1] - T_array[0]
dn0_dT = np.gradient(n0_array, dT)
neg_dn0_dT = -dn0_dT
dE_dT = np.gradient(E_array, dT)
Cv_array = dE_dT  # specific heat

# ---------------------- Plotting ----------------------
plt.figure(figsize=(10,6))
# We plot -mu so that it looks positive and “upward sloping”; you can just plot mu_array if preferred.
plt.plot(T_array, -mu_array, 'b-o')
plt.xlabel("Temperature T")
plt.ylabel("Negative Chemical Potential (−µ)")
plt.title("Non-BEC System: µ < 0 (High Degeneracy)")
plt.grid(True)
plt.show()
plt.savefig("nonBEC_chemical_potential.png")

plt.figure(figsize=(10,6))
plt.plot(T_array, n0_array, 'r-o')
plt.xlabel("Temperature T")
plt.ylabel("Ground State Occupation ⟨n₀⟩")
plt.title("Non-BEC System: Ground State Occupation (No Condensation)")
plt.grid(True)
plt.show()
plt.savefig("nonBEC_ground_state_occupation.png")

plt.figure(figsize=(10,6))
plt.plot(T_array, log_n0_array, 'm-o')
plt.xlabel("Temperature T")
plt.ylabel("log(⟨n₀⟩)")
plt.title("Non-BEC System: log(⟨n₀⟩) vs T")
plt.grid(True)
plt.show()
plt.savefig("nonBEC_log_ground_state_occupation.png")

plt.figure(figsize=(10,6))
plt.plot(T_array, neg_dn0_dT, 'g-o')
plt.xlabel("Temperature T")
plt.ylabel(r"-$\frac{\partial \langle n_0 \rangle}{\partial T}$")
plt.title("Non-BEC System: Negative d⟨n₀⟩/dT")
plt.grid(True)
plt.show()
plt.savefig("nonBEC_negative_gradient.png")

plt.figure(figsize=(10,6))
plt.plot(T_array, Cv_array, 'k-o')
plt.xlabel("Temperature T")
plt.ylabel("Specific Heat $C_v$")
plt.title("Non-BEC System: Specific Heat vs T")
plt.grid(True)
plt.show()
plt.savefig("nonBEC_specific_heat.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# ---------------------- Parameters ----------------------
N_target = int(1e5)       # total number of bosons
M = 1000                  # number of single-particle levels
delta = 1e-2             # small energy spacing for excited states
energies = delta * np.arange(M)  # energies: 0, δ, 2δ, ..., (M-1)*δ

# temperature range
T_min = 0.1
T_max = 2.0   # pick high enough so that excited states can hold all N at high T
num_T = 100
T_array = np.linspace(T_min, T_max, num_T)

kB = 1.0  # Boltzmann constant

# -------------- Functions --------------
def total_number(mu, T, energies):
    """
    Bose-Einstein sum: N(mu,T) = Σ_i [1/(exp(β(ε_i - mu)) - 1)].
    """
    beta = 1/(kB*T)
    with np.errstate(divide='ignore', invalid='ignore'):
        occ = 1.0/(np.exp(beta*(energies - mu)) - 1.0)
    return np.nansum(occ)

def total_energy(mu, T, energies):
    """
    E(mu,T) = Σ_i [ε_i/(exp(β(ε_i - mu)) - 1)].
    """
    beta = 1/(kB*T)
    with np.errstate(divide='ignore', invalid='ignore'):
        occ = 1.0/(np.exp(beta*(energies - mu)) - 1.0)
    return np.nansum(energies*occ)

def find_mu(T, energies, N_target):
    """
    At temperature T, find mu s.t. total_number(mu,T) = N_target.
    If the excited states at mu=0 can hold at least N_target => normal phase => mu < 0.
    Else => condensed phase => mu=0.
    """
    beta = 1/(kB*T)

    # exclude ground state from capacity check:
    if len(energies) > 1:
        excited_energies = energies[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            occ_ex = 1.0/(np.exp(beta*excited_energies) - 1.0)
        N_ex_max = np.nansum(occ_ex)
    else:
        # only 1 level => ground state
        N_ex_max = 0.0

    if N_target <= N_ex_max:
        # can accommodate all N => normal phase => solve mu < 0
        mu_low, mu_high = -100.0, -1e-12
        def f(mu):
            return total_number(mu, T, energies) - N_target
        mu_sol = bisect(f, mu_low, mu_high, xtol=1e-12)
        return mu_sol
    else:
        # condensed phase => mu=0
        return 0.0

# -------------- Arrays to store results --------------
mu_array = np.zeros(num_T)
n0_array = np.zeros(num_T)   # ground-state occupancy
E_array = np.zeros(num_T)

for i, T in enumerate(T_array):
    mu_val = find_mu(T, energies, N_target)
    mu_array[i] = mu_val
    beta = 1.0/(kB*T)

    if mu_val == 0.0:
        # condensed phase
        if len(energies) > 1:
            excited_energies = energies[1:]
            with np.errstate(divide='ignore', invalid='ignore'):
                occ_ex = 1.0/(np.exp(beta*excited_energies) - 1.0)
            N_ex = np.nansum(occ_ex)
            n0 = N_target - N_ex
            E_ex = np.nansum(excited_energies*occ_ex)
        else:
            # if there's only 1 level, everything is ground state
            n0 = N_target
            E_ex = 0.0
        n0_array[i] = n0
        E_array[i] = E_ex
    else:
        # normal phase: mu < 0
        with np.errstate(divide='ignore', invalid='ignore'):
            occ_all = 1.0/(np.exp(beta*(energies - mu_val)) - 1.0)
        n0_array[i] = occ_all[0]
        E_array[i] = np.nansum(energies*occ_all)

# Some derived quantities
log_n0 = np.log(n0_array + 1e-15)
dT = T_array[1] - T_array[0]
dn0_dT = np.gradient(n0_array, dT)
neg_dn0_dT = -dn0_dT
dE_dT = np.gradient(E_array, dT)
Cv = dE_dT

# -------------- Plotting --------------
plt.figure()
plt.plot(T_array, -mu_array, 'g')
plt.xlabel("T")
plt.ylabel("Negative Chemical Potential mu")
plt.title("BEC System: mu(T)")
plt.grid(True)
plt.show()
plt.savefig("BEC_chemical_potential.png")

plt.figure()
plt.plot(T_array, n0_array, 'k')
plt.xlabel("T")
plt.ylabel("Ground State Occupation")
plt.title("BEC System: Ground State Occupation")
plt.grid(True)
plt.show()
plt.savefig("BEC_ground_state_occupation.png")

plt.figure()
plt.semilogy(T_array, n0_array, 'k')
plt.xlabel("T")
plt.ylabel("Ground State Occupation (log scale)")
plt.title("BEC System: log(n0) vs T")
plt.grid(True)
plt.show()
plt.savefig("BEC_log_ground_state_occupation.png")

plt.figure()
plt.plot(T_array, neg_dn0_dT, 'r')
plt.xlabel("T")
plt.ylabel(r"$-\partial n_0/\partial T$")
plt.title("BEC System: Negative d(n0)/dT")
plt.grid(True)
plt.show()
plt.savefig("BEC_negative_gradient.png")

plt.figure()
plt.plot(T_array, Cv, 'b')
plt.xlabel("T")
plt.ylabel("Specific Heat Cv")
plt.title("BEC System: Cv vs T")
plt.grid(True)
plt.show()
plt.savefig("BEC_specific_heat.png")
