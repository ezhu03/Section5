import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def double_pendulum_ode(t, y, m1, m2, L1, L2, g):
    """
    Returns the time-derivative of the state vector y for the double pendulum.
    y = [theta1, omega1, theta2, omega2].
    """
    theta1, omega1, theta2, omega2 = y

    # --- Elements of the mass matrix M(theta1,theta2) ---
    M11 = (m1 + m2) * L1**2
    M12 = m2 * L1 * L2 * np.cos(theta1 - theta2)
    M21 = M12
    M22 = m2 * L2**2

    # --- The Coriolis/centrifugal + gravity terms C(theta, dot{theta}) ---
    C1 = (m2 * L1 * L2 * np.sin(theta1 - theta2) * omega2**2
          + (m1 + m2) * g * L1 * np.sin(theta1))
    C2 = (- m2 * L1 * L2 * np.sin(theta1 - theta2) * omega1**2
          + m2 * g * L2 * np.sin(theta2))

    # We want M * [ddtheta1, ddtheta2]^T = -C, i.e.  ddtheta = -M^{-1} C
    # First assemble M and C as arrays:
    M = np.array([[M11, M12],
                  [M21, M22]])
    C = np.array([C1, C2])

    # Solve for ddtheta
    # ddtheta = np.linalg.solve(M, -C)
    ddtheta = -np.linalg.inv(M).dot(C)  # small systems => direct inversion is OK
    ddtheta1, ddtheta2 = ddtheta

    # Return [dot{theta1}, ddtheta1, dot{theta2}, ddtheta2]
    return [omega1, ddtheta1, omega2, ddtheta2]


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def M_mat(theta1, theta2, m1, m2, L1, L2):
    """ Mass/inertia matrix M(theta1,theta2). """
    M11 = (m1 + m2) * L1**2
    M12 = m2 * L1 * L2 * np.cos(theta1 - theta2)
    M22 = m2 * L2**2
    return np.array([[M11, M12],
                     [M12, M22]])

def dM_dtheta1(theta1, theta2, m1, m2, L1, L2):
    """
    Partial derivative of M wrt theta1.
    Only the off-diagonal terms depend on (theta1 - theta2),
    d/dtheta1 [cos(theta1 - theta2)] = -sin(...).
    """
    d = m2 * L1 * L2 * (-np.sin(theta1 - theta2))
    return np.array([[0, d],
                     [d, 0]])

def dM_dtheta2(theta1, theta2, m1, m2, L1, L2):
    """
    Partial derivative of M wrt theta2.
    d/dtheta2 [cos(theta1 - theta2)] = +sin(...).
    """
    d = m2 * L1 * L2 * np.sin(theta1 - theta2)
    return np.array([[0, d],
                     [d, 0]])

def V(theta1, theta2, m1, m2, L1, L2, g):
    """ Potential energy. """
    return - (m1 + m2)*g*L1*np.cos(theta1) - m2*g*L2*np.cos(theta2)

def dV_dtheta1(theta1, theta2, m1, m2, L1, L2, g):
    """ dV/dtheta1 """
    return (m1 + m2)*g*L1*np.sin(theta1)

def dV_dtheta2(theta1, theta2, m1, m2, L1, L2, g):
    """ dV/dtheta2 """
    return m2*g*L2*np.sin(theta2)

def double_pendulum_hamilton_ode(t, y, m1, m2, L1, L2, g):
    """
    ODE in Hamiltonian form.
    State y = [theta1, p1, theta2, p2].
    We compute:
      dot{theta_i} = + dH/dp_i,
      dot{p_i}     = - dH/dtheta_i.
    """
    theta1, p1, theta2, p2 = y

    # -- Construct M, M^-1
    M = M_mat(theta1, theta2, m1, m2, L1, L2)
    M_inv = np.linalg.inv(M)

    # Momentum vector p
    p_vec = np.array([p1, p2])

    # 1) dtheta/dt = partial H / partial p = M^-1 p
    dtheta1_dt, dtheta2_dt = M_inv @ p_vec

    # 2) dp/dt = - partial H / partial theta
    #    partial H / partial theta = 0.5 * p^T ( d/dtheta M^-1 ) p + dV/dtheta
    # But d/dtheta (M^-1) = - M^-1 (dM/dtheta) M^-1.
    dM1 = dM_dtheta1(theta1, theta2, m1, m2, L1, L2)
    dM2 = dM_dtheta2(theta1, theta2, m1, m2, L1, L2)

    # partial M^-1 / partial theta1
    dM_inv_theta1 = - M_inv @ dM1 @ M_inv
    # partial M^-1 / partial theta2
    dM_inv_theta2 = - M_inv @ dM2 @ M_inv

    # Now p^T (dM_inv/dtheta1) p:
    p_dM1_p = p_vec @ (dM_inv_theta1 @ p_vec)
    p_dM2_p = p_vec @ (dM_inv_theta2 @ p_vec)

    # So partial H / partial theta1:
    dH_dtheta1 = 0.5*p_dM1_p + dV_dtheta1(theta1, theta2, m1, m2, L1, L2, g)
    # partial H / partial theta2:
    dH_dtheta2 = 0.5*p_dM2_p + dV_dtheta2(theta1, theta2, m1, m2, L1, L2, g)

    # Finally dp_i/dt = -dH/dtheta_i
    dp1_dt = - dH_dtheta1
    dp2_dt = - dH_dtheta2

    return [dtheta1_dt, dp1_dt, dtheta2_dt, dp2_dt]


# ------------------------
# Example usage
# ------------------------

# Parameters
m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
g = 9.81

# Convert an initial condition from "Lagrangian style" to the Hamiltonian p_i
# Suppose initial angles and angular velocities: 
theta1_0 = np.pi/2
theta2_0 = np.pi/2
omega1_0 = 0.0
omega2_0 = 0.0

# To find p_i = partial L / partial dot{theta}_i = M(theta1_0,theta2_0)*[omega1_0,omega2_0].
M0 = M_mat(theta1_0, theta2_0, m1, m2, L1, L2)
p1_0, p2_0 = M0 @ np.array([omega1_0, omega2_0])
ref_ic = np.array([theta1_0, p1_0, theta2_0, p2_0])

# Full initial state in Hamiltonian coordinates
y0 = [theta1_0, p1_0, theta2_0, p2_0]


# Time span
t_span = (0, 10)
fps = 30
t_eval = np.linspace(0, t_span[1], fps*t_span[1])

# Solve
sol = solve_ivp(double_pendulum_hamilton_ode, t_span, y0,
                t_eval=t_eval,
                args=(m1, m2, L1, L2, g),
                rtol=1e-9, atol=1e-9)

theta1_sol = sol.y[0]
p1_sol     = sol.y[1]
theta2_sol = sol.y[2]
p2_sol     = sol.y[3]
time       = sol.t

##############################################################################
# 3) Plot the phase‐space trajectory (theta2, p2)
##############################################################################

plt.figure(figsize=(6,5))
plt.plot(theta2_sol, p2_sol, 'b-')
plt.xlabel(r'$\theta_2$ (rad)')
plt.ylabel(r'$p_2$')
plt.title("Phase space trajectory: (theta2, p2)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('phase_space_trajectory.png')

##############################################################################
# 4) Animate the double pendulum in real (x,y) space
#    using (theta1, theta2) from the Hamiltonian solution.
##############################################################################

# The pivot is at (0,0).
# Mass 1 => (x1, y1) = ( L1 sin(theta1), - L1 cos(theta1) )
# Mass 2 => (x2, y2) = ( x1 + L2 sin(theta2), y1 - L2 cos(theta2) )

x1 = L1 * np.sin(theta1_sol)
y1 = -L1 * np.cos(theta1_sol)

x2 = x1 + L2 * np.sin(theta2_sol)
y2 = y1 - L2 * np.cos(theta2_sol)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_aspect('equal', 'box')
ax.set_xlim(-(L1+L2)-0.2, (L1+L2)+0.2)
ax.set_ylim(-(L1+L2)-0.2, (L1+L2)+0.2)
ax.set_title("Double Pendulum (Hamiltonian) in Real Space")
ax.grid(True)

line, = ax.plot([], [], 'o-', lw=2, markersize=8)

def init():
    line.set_data([], [])
    return (line,)

def update(frame):
    # frame is an index in our time array
    xvals = [0, x1[frame], x2[frame]]
    yvals = [0, y1[frame], y2[frame]]
    line.set_data(xvals, yvals)
    return (line,)

anim = FuncAnimation(fig, update, frames=len(time),
                     init_func=init, interval=30, blit=True)
anim.save("double_pendulum.mp4", writer="ffmpeg", fps=fps)
plt.show()

def graham_scan(data):
    def polar_angle(p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    y_min_ind = np.argmin(data[:, 1])
    y_min = data[y_min_ind]
    polar_dict = {}
    for point in data:
        polar_dict.update({tuple(point): polar_angle(y_min, point)})
    polar_dict = dict(sorted(polar_dict.items(), key=lambda item: item[1]))
    polar_points = [key for key in polar_dict]
    hull = []
    for point in polar_points:
        while len(hull) > 1 and np.cross(
            np.array(hull[-1]) - np.array(hull[-2]),
            np.array(point) - np.array(hull[-1])
        ) <= 0:
            hull.pop()
        hull.append(point)
    
    return np.array(hull)
def polygon_area(points):
    """
    Computes the area of a polygon whose vertices are in `points` (x,y).
    The points should be in order (clockwise or counterclockwise).
    Returns a positive area (absolute value).
    """
    if len(points) < 3:
        return 0.0  # no area if fewer than 3 points
    x = points[:,0]
    y = points[:,1]
    # shoelace sum
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

N = 100  # number of points
np.random.seed(42)

delta_theta2 = 0.5
delta_p2     = 0.5
rand_offsets = 2.0*(np.random.rand(N,2)-0.5)

init_conds = []
for i in range(N):
    dth2 = rand_offsets[i,0]*delta_theta2
    dp2  = rand_offsets[i,1]*delta_p2
    y_i  = np.array([
        ref_ic[0],    # same theta1
        ref_ic[1],    # same p1
        ref_ic[2] + dth2,
        ref_ic[3] + dp2
    ])
    init_conds.append(y_i)
init_conds = np.array(init_conds)

###############################################################################
# 6) Integrate each condition, storing (theta2(t), p2(t)) for each time
###############################################################################
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 200)

all_theta2 = np.zeros((N, len(t_eval)))
all_p2     = np.zeros((N, len(t_eval)))

for i in range(N):
    sol_i = solve_ivp(double_pendulum_hamilton_ode,
                      t_span, init_conds[i],
                      t_eval=t_eval,
                      args=(m1, m2, L1, L2, g),
                      rtol=1e-9, atol=1e-9)
    all_theta2[i,:] = sol_i.y[2]  # index 2 => theta2
    all_p2[i,:]     = sol_i.y[3]  # index 3 => p2

###############################################################################
# 7) For each time, compute the 2D convex hull area in (theta2, p2) using
#    your graham_scan + polygon_area.
###############################################################################
areas = np.zeros(len(t_eval))

for it, t in enumerate(t_eval):
    # Collect the N points in (theta2, p2) at time 'it'
    points_t = np.column_stack((all_theta2[:,it], all_p2[:,it]))
    
    # If we have fewer than 3 unique points, area=0
    # (But let's still call graham_scan to keep consistent.)
    unique_points_t = np.unique(points_t, axis=0)
    if len(unique_points_t) < 3:
        areas[it] = 0.0
        continue
    
    hull_pts = graham_scan(unique_points_t)
    area_t   = polygon_area(hull_pts)
    areas[it] = area_t

###############################################################################
# 8) Plot area vs. time
###############################################################################
plt.figure()
plt.plot(t_eval, areas, 'b-', linewidth=2)
plt.xlabel("time (s)")
plt.ylabel("Hull area in (theta2, p2)")
plt.title("Evolution of partial phase-space area via Graham scan")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('phase_space_area.png')