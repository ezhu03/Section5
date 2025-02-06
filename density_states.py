import numpy as np
import matplotlib.pyplot as plt

def harmonic_hamiltonian(x, y, px, py, m=1.0, omega=1.0):
    """Return H = p^2/(2m) + 1/2 m omega^2 (x^2 + y^2)."""
    return 0.5/m*(px**2 + py**2) + 0.5*m*(omega**2)*(x**2 + y**2)

# ------------------------------
# USER PARAMETERS
# ------------------------------
m = 1.0
omega = 1.0

# Phase-space grid boundaries
xmax  = 3.0
pxmax = 3.0
N     = 100   # number of points in each dimension

# Energy points where we'll measure Omega(E)
E_max = 10.0
nE    = 100
E_vals = np.linspace(0, E_max, nE)

# ------------------------------
# 1) Construct the phase-space mesh
# ------------------------------
x_array  = np.linspace(-xmax,  xmax,  N)
y_array  = np.linspace(-xmax,  xmax,  N)
px_array = np.linspace(-pxmax, pxmax, N)
py_array = np.linspace(-pxmax, pxmax, N)

dx   = (x_array[-1]  - x_array[0])  / (N-1)
dpx  = (px_array[-1] - px_array[0]) / (N-1)
# same for dy, dpy
dy   = dx
dpy  = dpx

# We'll flatten a 4D grid into a 1D list of energies to simplify counting.
# But be aware this can become big (N^4 points).
# For demonstration, N=100 might be large but still feasible on many machines.
# If it is too big, reduce N.

print("Constructing 4D grid of size N^4 = ", N**4, "points ...")
HH = []
for x in x_array:
    for y in y_array:
        for px in px_array:
            for py in py_array:
                Hval = harmonic_hamiltonian(x, y, px, py, m=m, omega=omega)
                HH.append(Hval)

HH = np.array(HH, dtype=float)
print("Done. Shape of HH flat array:", HH.shape)

# Volume element in 4D phase space:
dVolume = dx * dy * dpx * dpy

# ------------------------------
# 2) Compute Omega(E) at each E
# ------------------------------
Omega_vals = []
HH_sorted = np.sort(HH)

# We'll do a cumulative approach:
# For each E, we can find how many points in HH <= E by binary search
# (np.searchsorted). Then multiply by the per-point volume element.
for E in E_vals:
    count = np.searchsorted(HH_sorted, E, side='right')
    Omega = count * dVolume
    Omega_vals.append(Omega)

Omega_vals = np.array(Omega_vals)

# ------------------------------
# 3) Approximate g(E) by finite difference
#    g(E) = dOmega/dE
# ------------------------------
g_numerical = np.zeros_like(Omega_vals)
dE = E_vals[1] - E_vals[0]
g_numerical[1:] = (Omega_vals[1:] - Omega_vals[:-1]) / dE

# ------------------------------
# 4) Compare with the analytic result:
#    Omega(E)   = 2 pi^2 / omega^2 * E^2  (classical, ignoring 2D factor)
#    g(E)       = d/dE Omega(E) = 4 pi^2 / omega^2 * E
#    But more precisely, dividing by (2 pi)^2 could appear if you
#    are counting quantum cells.  We’ll just compare shapes.
# ------------------------------
# For simplicity, let's do the "classical raw" formula:
Omega_analytic = (2.0 * np.pi**2 / omega**2) * E_vals**2
g_analytic     = (4.0 * np.pi**2 / omega**2) * E_vals

# Plot results
plt.figure(figsize=(7,5))
plt.plot(E_vals, Omega_vals, 'ro', label='Omega(E) numeric')
plt.plot(E_vals, Omega_analytic, 'b-', label='Omega(E) analytic')
plt.xlabel('E')
plt.ylabel('Omega(E)')
plt.legend()
plt.title('Cumulative Phase-Space Volume (2D Harmonic Osc.)')
plt.grid(True)

plt.figure(figsize=(7,5))
plt.plot(E_vals, g_numerical, 'ro', label='g(E) numeric (finite diff)')
plt.plot(E_vals, g_analytic, 'b-', label='g(E) analytic = 4 pi^2/omega^2 * E')
plt.xlabel('E')
plt.ylabel('g(E)')
plt.legend()
plt.title('Density of States (2D Harmonic Osc.)')
plt.grid(True)
plt.show()

# ------------------------------
# 5) Partition function from g(E)
#    Z(beta) = \int_0^∞ g(E) e^{-beta E} dE
# ------------------------------

def partition_function_from_gE(E_vals, g_vals, beta):
    """Numerical trapezoid-rule integration of g(E)*exp(-beta E)."""
    integrand = g_vals * np.exp(-beta*E_vals)
    return np.trapz(integrand, x=E_vals)

beta = 1.0   # inverse temperature
Z_num = partition_function_from_gE(E_vals, g_numerical, beta)

# Analytic result: Z_analytic = 1/(beta^2 * omega^2) (classical 2D oscillator)
Z_analytic = 1.0 / (beta**2 * omega**2)

print("Numeric partition function Z(beta=1):", Z_num)
print("Analytic partition function Z        :", Z_analytic)

import numpy as np
import matplotlib.pyplot as plt

def nonlinear_hamiltonian(x, y, px, py, m=1.0, omega=1.0, lam=0.1):
    """H = p^2/(2m) + 1/2 m omega^2 (r^2) + lambda (r^2)^2."""
    r2 = x*x + y*y
    kinetic = 0.5/m * (px*px + py*py)
    potential = 0.5*m*(omega**2)*r2 + lam*(r2**2)
    return kinetic + potential

# Parameters
m       = 1.0
omega   = 1.0
lam     = 0.1  # lambda
xmax    = 3.0
pxmax   = 3.0
N       = 80
E_max   = 10.0
nE      = 100
E_vals  = np.linspace(0, E_max, nE)

x_array  = np.linspace(-xmax,  xmax,  N)
y_array  = np.linspace(-xmax,  xmax,  N)
px_array = np.linspace(-pxmax, pxmax, N)
py_array = np.linspace(-pxmax, pxmax, N)

dx   = (x_array[-1] -  x_array[0]) / (N-1)
dy   = dx
dpx  = (px_array[-1] - px_array[0]) / (N-1)
dpy  = dpx
dVolume = dx * dy * dpx * dpy

# Build 4D Hamiltonian array
Hvals = []
for x in x_array:
    for y in y_array:
        for px in px_array:
            for py in py_array:
                Hval = nonlinear_hamiltonian(x,y,px,py,m=m,omega=omega,lam=lam)
                Hvals.append(Hval)

Hvals = np.array(Hvals)
Hvals_sorted = np.sort(Hvals)

# Compute Omega(E)
Omega_vals = []
for E in E_vals:
    count = np.searchsorted(Hvals_sorted, E, side='right')
    Omega_vals.append(count * dVolume)

Omega_vals = np.array(Omega_vals)

# Finite-difference for g(E)
g_numerical = np.zeros_like(Omega_vals)
dE = E_vals[1] - E_vals[0]
g_numerical[1:] = (Omega_vals[1:] - Omega_vals[:-1]) / dE

# Compare with the analytic expression
# g(E) = pi^2*m * r_max^2, where r_max solves:
# E = 0.5*m*omega^2*r_max^2 + lambda*r_max^4
# We can solve for r_max^2 via the quadratic in r_max^2:
#     lambda*r^4 + (0.5*m*omega^2)*r^2 - E = 0
# i.e.   lam * (r^2)^2 + (0.5*m*omega^2)*(r^2) - E = 0
# Let R = r^2 => lam R^2 + (0.5 m omega^2) R - E = 0

def r_max_squared(E, m=1.0, omega=1.0, lam=0.1):
    """Return r_max^2 that satisfies
         lam*(r^2)^2 + 0.5*m*omega^2*(r^2) = E.
    If no real solution, return 0.
    """
    a = lam
    b = 0.5*m*(omega**2)
    c = -E
    disc = b*b - 4*a*c
    if disc <= 0:
        return 0.0
    # Quadratic formula for R = (-b + sqrt(b^2 - 4ac)) / (2a) but
    # we want the POSITIVE root.
    R = (-b + np.sqrt(disc)) / (2*a)
    return R if R>0 else 0.0

g_analytic = []
for E in E_vals:
    R = r_max_squared(E, m=m, omega=omega, lam=lam)
    g_analytic.append(2.0* np.pi**2 * m * R )
g_analytic = np.array(g_analytic)

# Plot
plt.figure()
plt.plot(E_vals, g_numerical, 'ro', label='g(E) numeric')
plt.plot(E_vals, g_analytic, 'b-', label='g(E) analytic')
plt.xlabel('E')
plt.ylabel('density of states')
plt.title('2D Nonlinear Oscillator: Density of States')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('density_states.png')

def partition_function_from_gE(E_vals, g_vals, beta):
    integrand = g_vals * np.exp(-beta * E_vals)
    return np.trapz(integrand, x=E_vals)

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) Define the Hamiltonian
# ---------------------------
def nonlinear_hamiltonian(x, y, px, py, m=1.0, omega=1.0, lam=0.1):
    """Nonlinear oscillator H = p^2/(2m) + 1/2 m omega^2 (r^2) + lam*(r^2)^2."""
    r2 = x*x + y*y
    kinetic   = 0.5*(px*px + py*py)/m
    potential = 0.5*m*(omega**2)*r2 + lam*(r2**2)
    return kinetic + potential


# ---------------------------
# 2) Set up parameters
# ---------------------------
m       = 1.0
omega   = 1.0
lam     = 0.1

xmax    = 3.0  # grid boundary for x,y
pxmax   = 3.0  # grid boundary for px,py

N       = 50   # number of points in each dimension (4D total is N^4)
E_max   = 10.0
nE      = 100
E_vals  = np.linspace(0, E_max, nE)

# Build phase-space arrays
x_array  = np.linspace(-xmax,  xmax,  N)
y_array  = np.linspace(-xmax,  xmax,  N)
px_array = np.linspace(-pxmax, pxmax, N)
py_array = np.linspace(-pxmax, pxmax, N)

# 4D volume element
dx   = x_array[1]  - x_array[0]
dy   = y_array[1]  - y_array[0]
dpx  = px_array[1] - px_array[0]
dpy  = py_array[1] - py_array[0]
dVolume = dx * dy * dpx * dpy

# ---------------------------
# 3) Compute H on each grid point
#    (Flatten into a single large array)
# ---------------------------
print("Constructing 4D grid of size", N**4, "points ...")

H_values = []
for x in x_array:
    for y in y_array:
        for px in px_array:
            for py in py_array:
                H = nonlinear_hamiltonian(x, y, px, py, m=m, omega=omega, lam=lam)
                H_values.append(H)

H_values = np.array(H_values, dtype=float)
H_values_sorted = np.sort(H_values)
print("Done. Shape:", H_values.shape)

# ---------------------------
# 4) For each E, count how many grid points have H <= E
#    => Omega(E). Then differentiate to get g(E).
# ---------------------------
Omega_vals = []
for E in E_vals:
    # 'right' means index of first element > E
    count = np.searchsorted(H_values_sorted, E, side='right')
    Omega_vals.append(count * dVolume)

Omega_vals = np.array(Omega_vals)

# Numerical derivative for g(E)
g_numerical = np.zeros_like(Omega_vals)
dE = E_vals[1] - E_vals[0]
g_numerical[1:] = (Omega_vals[1:] - Omega_vals[:-1]) / dE

# ---------------------------
# 5) Compare with the analytic formula
#    g(E) = 2 pi^2 * m * r_max^2
#    where r_max solves 1/2 m omega^2 r^2 + lam r^4 = E
# ---------------------------
def r_max_squared(E, m, omega, lam):
    # Solve lam*(r^2)^2 + (1/2 m omega^2)*(r^2) - E = 0
    # => lam * R^2 + (1/2 m omega^2)*R - E = 0, with R = r^2
    a = lam
    b = 0.5*m*(omega**2)
    c = -E
    disc = b*b - 4*a*c
    if disc <= 0:
        return 0.0
    # Positive root:
    R = (-b + np.sqrt(disc)) / (2*a)
    # If negative, r^2=0. That means E below threshold
    return R if R>0 else 0.0

g_analytic = []
for E in E_vals:
    R = r_max_squared(E, m, omega, lam)
    g_analytic.append( 2.0 * np.pi**2 * m * R )
g_analytic = np.array(g_analytic)

# ---------------------------
# 6) Plot Results
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(E_vals, g_numerical, 'ro', label='g(E) numeric')
plt.plot(E_vals, g_analytic,  'b-', label='g(E) analytic')
plt.xlabel("Energy E")
plt.ylabel("Density of States g(E)")
plt.title("2D Nonlinear Harmonic Oscillator: Density of States")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('nonlinear_density_states.png')