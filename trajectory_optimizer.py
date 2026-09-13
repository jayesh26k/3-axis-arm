"""
trajectory_optimizer.py

Trajectory optimization for the 3-DOF arm via direct (trapezoidal)
collocation, solved with CasADi + IPOPT. Minimizes control effort
subject to:
  - the arm's real rigid-body dynamics (arm_dynamics.rnea)
  - joint position limits (from arm_final_controlled.xml)
  - joint velocity limits (derived from NEMA17 + gearbox estimates --
    UPDATE these once real motor specs are confirmed)
  - joint torque limits (same caveat)
  - zero velocity at start and goal (rest-to-rest motion)

q_start and q_goal below are JOINT ANGLES (radians: base, shoulder,
elbow) -- NOT Cartesian positions. If you want to target a Cartesian
end-effector position instead, run it through JacobianIKSolver first
to get joint angles, then pass those in here as q_goal.

No obstacle avoidance yet -- this is step 1 (smooth, constrained
motion). Obstacle constraints get added as extra inequality
constraints on end-effector / link positions later.

Run validate_dynamics.py first. Don't trust this until that passes.
"""

import casadi as ca
import numpy as np
from ik_solver import JacobianIKSolver

from arm_dynamics import dynamics_residual


# ---------------------------------------------------------------------------
# Joint limits (from arm_final_controlled.xml <joint range="...">)
# ---------------------------------------------------------------------------
Q_MIN = np.array([-3.14, -2.13, -3.14])
Q_MAX = np.array([3.14, 2.13, 3.14])

# ---------------------------------------------------------------------------
# Velocity / torque limits derived from NEMA17 (~0.45 N*m holding torque)
# + gearboxes (15:1 base, 30:1 shoulder/elbow), with a 60% continuous-duty
# derating and 85% gearbox efficiency. SEE the conversation notes --
# these are engineering estimates, not a confirmed datasheet. Update
# once you have your exact motor's holding torque.
# ---------------------------------------------------------------------------
QD_MAX = np.array([0.838, 0.419, 0.419])       # rad/s
TAU_MAX = np.array([3.44, 6.89, 6.89])         # N*m


def solve_trajectory(q_start, q_goal, total_time=5.0, num_intervals=40,
                      solver_print_level=0):
    """
    Solve for a joint-space trajectory from q_start to q_goal (both rest,
    zero velocity) over total_time seconds, using num_intervals collocation
    intervals.

    Returns a dict with time, q, qd, qdd, tau arrays (each (N+1) or N long),
    or None if the solver did not converge.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    N = num_intervals
    dt = total_time / N

    opti = ca.Opti()

    Q = opti.variable(3, N + 1)
    QD = opti.variable(3, N + 1)
    QDD = opti.variable(3, N + 1)
    TAU = opti.variable(3, N + 1)

    # ---- dynamics + collocation defects ----
    for k in range(N + 1):
        res = dynamics_residual(Q[:, k], QD[:, k], QDD[:, k], TAU[:, k])
        opti.subject_to(res == 0)

    for k in range(N):
        opti.subject_to(
            Q[:, k + 1] == Q[:, k] + dt / 2 * (QD[:, k] + QD[:, k + 1])
        )
        opti.subject_to(
            QD[:, k + 1] == QD[:, k] + dt / 2 * (QDD[:, k] + QDD[:, k + 1])
        )

    # ---- boundary conditions: rest-to-rest ----
    opti.subject_to(Q[:, 0] == q_start)
    opti.subject_to(QD[:, 0] == 0)
    opti.subject_to(Q[:, N] == q_goal)
    opti.subject_to(QD[:, N] == 0)

    # ---- joint / velocity / torque limits ----
    for k in range(N + 1):
        opti.subject_to(opti.bounded(Q_MIN, Q[:, k], Q_MAX))
        opti.subject_to(opti.bounded(-QD_MAX, QD[:, k], QD_MAX))
        opti.subject_to(opti.bounded(-TAU_MAX, TAU[:, k], TAU_MAX))

    # ---- objective: minimize control effort ----
    effort = 0
    for k in range(N + 1):
        effort += ca.sumsqr(TAU[:, k]) * dt
    opti.minimize(effort)

    # ---- initial guess: straight-line interpolation ----
    for k in range(N + 1):
        alpha = k / N
        opti.set_initial(Q[:, k], (1 - alpha) * q_start + alpha * q_goal)
    opti.set_initial(QD, 0)
    opti.set_initial(QDD, 0)
    opti.set_initial(TAU, 0)

    p_opts = {"expand": True}
    s_opts = {"print_level": solver_print_level, "max_iter": 500}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"Solver failed to converge: {e}")
        return None

    return {
        "time": np.linspace(0, total_time, N + 1),
        "q": sol.value(Q),
        "qd": sol.value(QD),
        "qdd": sol.value(QDD),
        "tau": sol.value(TAU),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 trajectory_optimizer.py /path/to/arm_final_controlled.xml")
        sys.exit(1)

    model_path = sys.argv[1]

    # q_start, q_goal are JOINT ANGLES (radians): base, shoulder, elbow.
    # Example: move from a folded start pose to a reaching pose.
    
    
    ik_solver = JacobianIKSolver(model_path)
   
    q_start = np.array([.5, -.3, .8])
    q_goal = np.array(ik_solver.solve_ik(target_pos=np.array([0.15, 0.0, 0.02]), initial_qpos=q_start)[0])

    result = solve_trajectory(q_start, q_goal, total_time=5.0, num_intervals=40)

    if result is not None:
        print("\nSolved trajectory:")
        print(f"  duration: {result['time'][-1]:.2f}s over {len(result['time'])} knots")
        print(f"  peak |tau| per joint: {np.abs(result['tau']).max(axis=1).round(3)} N*m")
        print(f"  peak |qd| per joint:  {np.abs(result['qd']).max(axis=1).round(3)} rad/s")
        np.savez("trajectory_result.npz", **result)
        print("\nSaved to trajectory_result.npz")