import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from aero_pf import aero_coeff

# =====================================
# constants
# =====================================

g = 9.81

mass = 25000.0

Iy = 1.0e6

Sref = 10.52

Lref = 46.0

# =====================================
# atmosphere
# =====================================

def atmosphere(h):

    rho0 = 1.225
    H = 8500.0

    rho = rho0*np.exp(-h/H)

    a = 295.0

    return rho, a

# =====================================
# equations of motion
# =====================================

def eom(t, X):


    x, y, V, gamma, theta, q = X

    rho, a = atmosphere(y)

    M = V/a

    alpha = theta - gamma

    alpha_deg = np.degrees(alpha)

    CD, CL, CM = aero_coeff(
        M,
        alpha_deg
    )

    qinf = 0.5*rho*V**2

    D = qinf*Sref*CD

    L = qinf*Sref*CL

    Maero = qinf*Sref*Lref*CM

    T = 0.0

    dx = V*np.cos(gamma)

    dy = V*np.sin(gamma)

    dV = (
        (T*np.cos(alpha)-D)/mass
        - g*np.sin(gamma)
    )

    dgamma = (
        L/(mass*V)
        - g*np.cos(gamma)/V
    )

    dq = Maero / Iy

    dtheta = q


    return [
        dx,
        dy,
        dV,
        dgamma,
        dtheta,
        dq
    ]

# =====================================
# initial condition
# =====================================

x0 = 0.0

y0 = 22600.0

V0 = 330.0

gamma0 = np.radians(-70)

alpha0 = np.radians(5)

theta0 = gamma0 + alpha0

q0 = 0.0

X0 = [
    x0,
    y0,
    V0,
    gamma0,
    theta0,
    q0
]

# =====================================
# stop condition
# =====================================

def stop_event(t, X):

    return X[1]-6800.0

stop_event.terminal = True
stop_event.direction = -1


def alpha_limit_event(t, X):

    gamma = X[3]
    theta = X[4]
    alpha_deg = np.degrees(theta - gamma)

    return 10.0 - abs(alpha_deg)

alpha_limit_event.terminal = True
alpha_limit_event.direction = -1

# =====================================
# initial aerodynamic check
# =====================================

CD_init, CL_init, CM_init = aero_coeff(
    V0/atmosphere(y0)[1],
    np.degrees(alpha0)
)
qinf_init = 0.5*atmosphere(y0)[0]*V0**2
Maero_init = qinf_init*Sref*Lref*CM_init

print()
print("===== INITIAL AERO CHECK =====")
print("Alpha [deg] =", np.degrees(alpha0))
print("CL [-] =", CL_init)
print("CM [-] =", CM_init)
print("Initial aero moment [N m] =", Maero_init)

# =====================================
# integrate
# =====================================

sol = solve_ivp(
    eom,
    [0,100],
    X0,
    events=[stop_event, alpha_limit_event],
    max_step=0.05
)

# =====================================
# result
# =====================================

x = sol.y[0]/1000
y = sol.y[1]/1000

V = sol.y[2]

gamma = np.degrees(sol.y[3])

theta = np.degrees(sol.y[4])

alpha = theta - gamma

print()
print("===== FINAL =====")
print()

print("Altitude [km] =", y[-1])
print("Velocity [m/s] =", V[-1])
print("Gamma [deg] =", gamma[-1])
print("Theta [deg] =", theta[-1])

if sol.t_events[0].size > 0:
    termination_reason = "高度 6.8 km 到達"
elif sol.t_events[1].size > 0:
    termination_reason = "迎角が空力表の有効範囲 |alpha|=10 deg に到達"
else:
    termination_reason = "積分終了時刻に到達"

print("Termination reason =", termination_reason)
print("Alpha [deg] =", alpha[-1])

# -------------------------------------
# Theta
# -------------------------------------

theta_deg = np.degrees(sol.y[4])

plt.figure(figsize=(8,5))

plt.plot(
    sol.t,
    theta_deg,
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("Pitch Angle History")

plt.grid(True)

plt.savefig("theta_history.png")

# -------------------------------------
# Alpha
# -------------------------------------

gamma_deg = np.degrees(sol.y[3])

alpha_deg = theta_deg - gamma_deg

plt.figure(figsize=(8,5))

plt.plot(
    sol.t,
    alpha_deg,
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Alpha [deg]")
plt.title("Angle of Attack History")

plt.grid(True)

plt.savefig("alpha_history.png")

# -------------------------------------
# q
# -------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    sol.t,
    sol.y[5],
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Pitch Rate q [rad/s]")
plt.title("Pitch Rate History")

plt.grid(True)

plt.savefig("q_history.png")

plt.close('all')
