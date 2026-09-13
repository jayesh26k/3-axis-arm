"""
arm_dynamics.py

Symbolic rigid-body dynamics for the 3-DOF arm, via recursive
Newton-Euler (RNEA). Built directly from arm_final_controlled.xml:
masses, COM positions, and inertia tensors are the real values MuJoCo
uses; the fixed inter-link rotation offsets (R_offset) are derived from
the <body euler="..."> attributes.

IMPORTANT: before trusting this for trajectory optimization, run
validate_dynamics.py, which compares rnea() against MuJoCo's own
mj_inverse() at several random configurations. Hand-deriving the euler
rotation convention is the one step here with real risk of a sign/axis
error -- don't skip validation.

Only the 3 moving links (base-rotary, shoulder, elbow) are modeled.
arm_root is fixed to the world and contributes no dynamics.
"""

import casadi as ca
import numpy as np

GRAVITY = 9.81  # m/s^2

# ---------------------------------------------------------------------------
# Fixed geometry, derived from arm_final_controlled.xml
# ---------------------------------------------------------------------------

# R_offset[i]: fixed rotation from link i's frame to link (i-1)'s frame,
# BEFORE the joint's own rotation is applied (i.e. rotation at theta_i = 0).
# Derived from each body's <euler> attribute using MuJoCo's default
# eulerseq="xyz" convention: R = Rx(a) * Ry(b) * Rz(c).
R_OFFSET = [
    np.array([[0.0, 1.0, 0.0],    # link 0: base-rotary, euler="0 0 -1.5708"
               [-1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0]]),
    np.array([[0.0, 0.0, 1.0],    # link 1: shoulder, euler="0 1.5708 0"
               [0.0, 1.0, 0.0],
               [-1.0, 0.0, 0.0]]),
    np.array([[0.0, -1.0, 0.0],   # link 2: elbow -- CORRECTED via extract_offsets.py.
               [-1.0, 0.0, 0.0],   # The original hand-derived Rx(pi)@Rz(-pi/2) had the
               [0.0, 0.0, -1.0]]), # multiplication order backwards for this two-angle
                                   # case (confirmed by the exact-sign-flip pattern in
                                   # the gravity-only isolation test).
]

# p_offset[i]: origin of link i relative to origin of link (i-1),
# expressed in link (i-1)'s frame. Taken directly from <body pos="...">.
P_OFFSET = [
    np.array([0.0, 0.0, 0.0]),        # rotary rel. to arm_root
    np.array([0.027, 0.0, 0.065]),    # shoulder rel. to rotary
    np.array([-0.135, 0.0, -0.004]),  # elbow rel. to shoulder
]

MASS = [0.11473, 0.16216, 0.26414]

# COM position of each link, expressed in that link's own frame
# (<inertial pos="...">).
COM = [
    np.array([0.0016484, -5.4453e-07, 0.050018]),
    np.array([-0.085778, 7.1453e-07, 0.020049]),
    np.array([1.9571e-08, 0.087078, 0.012121]),
]

# Inertia tensor about the COM, expressed in each link's own frame,
# built from <inertial fullinertia="Ixx Iyy Izz Ixy Ixz Iyz">.
INERTIA = [
    np.array([[0.00011646, 2.1954e-10, -2.8336e-06],
              [2.1954e-10, 0.00011303, -7.8031e-11],
              [-2.8336e-06, -7.8031e-11, 7.9819e-05]]),
    np.array([[7.5801e-05, -2.1107e-09, 4.6784e-05],
              [-2.1107e-09, 0.00045993, 1.0257e-09],
              [4.6784e-05, 1.0257e-09, 0.00047777]]),
    np.array([[0.00061693, 1.9089e-10, -1.6288e-10],
              [1.9089e-10, 8.3562e-05, -2.0211e-05],
              [-1.6288e-10, -2.0211e-05, 0.00064911]]),
]

# Joint damping, from <default><joint damping="0.5" .../></default>.
# Applied as a viscous term (b*qd) added on top of RNEA's rigid-body torque.
DAMPING = np.array([0.5, 0.5, 0.5])

# Joint Coulomb friction, from <default><joint frictionloss="0.05" .../>.
# Confirmed via validation: MuJoCo's qfrc_inverse included a constant
# 0.05 N*m offset per joint (matching sign of qd) that DAMPING alone
# didn't account for.
FRICTIONLOSS = np.array([0.05, 0.05, 0.05])
FRICTION_SMOOTH_EPS = 1e-3  # smooths sign(qd) near 0 for solver-friendly gradients

Z_AXIS = np.array([0.0, 0.0, 1.0])  # all 3 joints rotate about local z


def _rotz(theta):
    c, s = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(c, -s, 0),
        ca.horzcat(s, c, 0),
        ca.horzcat(0, 0, 1),
    )


def rnea(q, qd, qdd):
    """
    Recursive Newton-Euler inverse dynamics (no damping included).

    q, qd, qdd: 3x1 CasADi symbolic or numeric vectors (base, shoulder, elbow)
    Returns: 3x1 CasADi expression -- joint torques required to produce
             the given motion, including gravity, EXCLUDING joint damping
             (add DAMPING * qd separately -- see dynamics_residual()).
    """
    n = 3
    omega = [ca.DM.zeros(3)]
    omega_dot = [ca.DM.zeros(3)]
    # Base "acceleration" trick: setting a0 = (0,0,+g) injects gravity
    # into the recursion without a separate gravity term.
    a = [ca.DM(np.array([0.0, 0.0, GRAVITY]))]

    R = []  # R[i]: rotation from link i's frame into link (i-1)'s frame

    # ---- outward (forward) recursion: velocities and accelerations ----
    for i in range(n):
        Ri = ca.DM(R_OFFSET[i]) @ _rotz(q[i])
        R.append(Ri)
        RiT = Ri.T  # rotation from frame i-1 into frame i

        omega_i = RiT @ omega[i] + Z_AXIS * qd[i]
        omega_dot_i = (
            RiT @ omega_dot[i]
            + ca.cross(RiT @ omega[i], Z_AXIS * qd[i])
            + Z_AXIS * qdd[i]
        )

        p_i = ca.DM(P_OFFSET[i])
        a_i = RiT @ (
            a[i]
            + ca.cross(omega_dot[i], p_i)
            + ca.cross(omega[i], ca.cross(omega[i], p_i))
        )

        omega.append(omega_i)
        omega_dot.append(omega_dot_i)
        a.append(a_i)

    # ---- per-link net force/moment about its own COM ----
    F, N = [], []
    for i in range(n):
        c_i = ca.DM(COM[i])
        a_c_i = (
            a[i + 1]
            + ca.cross(omega_dot[i + 1], c_i)
            + ca.cross(omega[i + 1], ca.cross(omega[i + 1], c_i))
        )
        F.append(MASS[i] * a_c_i)

        I_i = ca.DM(INERTIA[i])
        N.append(I_i @ omega_dot[i + 1] + ca.cross(omega[i + 1], I_i @ omega[i + 1]))

    # ---- inward (backward) recursion: joint torques ----
    f_next = ca.DM.zeros(3)  # no payload / force beyond the end-effector
    n_next = ca.DM.zeros(3)
    tau = [None] * n

    for i in reversed(range(n)):
        c_i = ca.DM(COM[i])
        if i < n - 1:
            R_next = R[i + 1]                 # frame i+1 -> frame i
            f_next_in_i = R_next @ f_next
            n_next_in_i = R_next @ n_next
            p_next = ca.DM(P_OFFSET[i + 1])   # origin i+1 rel. to origin i
        else:
            f_next_in_i = ca.DM.zeros(3)
            n_next_in_i = ca.DM.zeros(3)
            p_next = ca.DM.zeros(3)

        f_i = f_next_in_i + F[i]
        n_i = (
            N[i]
            + n_next_in_i
            + ca.cross(c_i, F[i])
            + ca.cross(p_next, f_next_in_i)
        )

        tau[i] = ca.dot(n_i, Z_AXIS)

        f_next, n_next = f_i, n_i

    return ca.vertcat(*tau)


def dynamics_residual(q, qd, qdd, tau):
    """
    Implicit dynamics constraint for direct collocation:
    rnea(q, qd, qdd) + damping*qd + frictionloss*tanh(qd/eps) - tau == 0

    The tanh term is a smooth stand-in for MuJoCo's frictionloss (a
    constant force opposing motion direction, i.e. frictionloss*sign(qd)).
    tanh(qd/eps) with a small eps matches sign(qd) closely for any qd not
    extremely close to zero, while keeping the constraint differentiable
    everywhere -- important for IPOPT's gradient-based solve.
    """
    friction = ca.DM(FRICTIONLOSS) * ca.tanh(qd / FRICTION_SMOOTH_EPS)
    return rnea(q, qd, qdd) + ca.DM(DAMPING) * qd + friction - tau