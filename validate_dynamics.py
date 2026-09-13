"""
validate_dynamics.py

Run this in your WSL2/Ubuntu environment (needs mujoco + casadi installed)
BEFORE trusting trajectory_optimizer.py. It compares arm_dynamics.rnea()
against MuJoCo's own mj_inverse() at several random joint configurations.

mj_inverse computes exactly the same thing rnea() is meant to compute --
the torque required to produce a given (q, qd, qdd) -- so any mismatch
here means an error in the hand-derived R_OFFSET/P_OFFSET geometry in
arm_dynamics.py, not a fundamental modeling choice.

Usage:
    python3 validate_dynamics.py /path/to/arm_final_controlled.xml
"""

import sys

import casadi as ca
import mujoco
import numpy as np

from arm_dynamics import dynamics_residual, rnea

N_TESTS = 20
TOLERANCE_NM = 0.01  # 10 mN*m -- generous, but catches real errors easily


def build_numeric_rnea():
    q = ca.SX.sym("q", 3)
    qd = ca.SX.sym("qd", 3)
    qdd = ca.SX.sym("qdd", 3)
    tau_expr = rnea(q, qd, qdd)
    return ca.Function("rnea_numeric", [q, qd, qdd], [tau_expr])


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 validate_dynamics.py /path/to/arm_final_controlled.xml")
        sys.exit(1)

    model_path = sys.argv[1]
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    rnea_fn = build_numeric_rnea()

    rng = np.random.default_rng(42)
    joint_limits = [(-3.14, 3.14), (-2.13, 2.13), (-3.14, 3.14)]

    # Diagnostic: disable all contact forces. arm_dynamics.py models ONLY
    # the open 3-DOF chain -- it has no concept of the free-floating
    # target_object block in the scene, or of collisions with it. If the
    # earlier mismatch was caused by the arm contacting that block, this
    # should make the comparison apples-to-apples and errors should drop
    # sharply.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    max_error = 0.0
    print(f"Running {N_TESTS} random configuration checks (contacts disabled)...\n")

    for test_i in range(N_TESTS):
        q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
        qd = rng.uniform(-1.0, 1.0, size=3)
        qdd = rng.uniform(-2.0, 2.0, size=3)

        # --- MuJoCo's ground truth ---
        data.qpos[:3] = q
        data.qvel[:3] = qd
        mujoco.mj_forward(model, data)
        print(f"  (test {test_i}: ncon={data.ncon})", end="  ")
        data.qacc[:3] = qdd
        mujoco.mj_inverse(model, data)
        tau_mujoco = data.qfrc_inverse[:3].copy()

        # --- Our RNEA (rigid-body torque only -- no damping) ---
        tau_ours = np.array(rnea_fn(q, qd, qdd)).flatten()

        # MuJoCo's qfrc_inverse INCLUDES joint damping AND frictionloss
        # (from <default><joint damping="0.5" frictionloss="0.05" .../>),
        # so add both here for an apples-to-apples comparison. Using exact
        # np.sign() here (not the smoothed tanh from arm_dynamics.py) since
        # this is validation against MuJoCo's actual discrete behavior.
        tau_ours_with_damping = tau_ours + 0.5 * qd + 0.05 * np.sign(qd)

        error = np.abs(tau_mujoco - tau_ours_with_damping)
        max_error = max(max_error, error.max())

        status = "OK" if error.max() < TOLERANCE_NM else "MISMATCH"
        print(f"  test {test_i:2d} [{status}]: "
              f"mujoco={tau_mujoco.round(4)}  ours={tau_ours_with_damping.round(4)}  "
              f"max_err={error.max():.5f} N*m")

    print(f"\nWorst-case error (full dynamics): {max_error:.5f} N*m")
    if max_error < TOLERANCE_NM:
        print("PASS -- dynamics model matches MuJoCo closely enough to trust "
              "for trajectory optimization.")
    else:
        print("FAIL on full dynamics -- see gravity-only isolation test below.")

    # -----------------------------------------------------------------
    # Isolation test: pure gravity torque only (qd=0, qdd=0).
    # This removes Coriolis/centrifugal coupling AND friction entirely,
    # leaving tau = G(q). If THIS still fails, the error is in the
    # kinematics (R_OFFSET/P_OFFSET/COM) -- not the velocity terms.
    # If this passes but the full test above doesn't, the error is in
    # the velocity-dependent terms or the friction/damping model instead.
    # -----------------------------------------------------------------
    print(f"\n--- Isolation test: gravity only (qd=0, qdd=0) ---\n")
    max_gravity_error = 0.0
    for test_i in range(N_TESTS):
        q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
        zero = np.zeros(3)

        data.qpos[:3] = q
        data.qvel[:3] = zero
        mujoco.mj_forward(model, data)
        data.qacc[:3] = zero
        mujoco.mj_inverse(model, data)
        tau_mujoco = data.qfrc_inverse[:3].copy()

        tau_ours = np.array(rnea_fn(q, zero, zero)).flatten()  # no damping needed, qd=0

        error = np.abs(tau_mujoco - tau_ours)
        max_gravity_error = max(max_gravity_error, error.max())
        status = "OK" if error.max() < TOLERANCE_NM else "MISMATCH"
        print(f"  gtest {test_i:2d} [{status}]: mujoco={tau_mujoco.round(4)}  "
              f"ours={tau_ours.round(4)}  max_err={error.max():.5f} N*m")

    print(f"\nWorst-case gravity-only error: {max_gravity_error:.5f} N*m")
    if max_gravity_error < TOLERANCE_NM:
        print("Gravity term matches -- error is in velocity-coupling terms "
              "or the friction/damping model, not the link geometry.")
    else:
        print("Gravity term ALSO mismatches -- error is in R_OFFSET/P_OFFSET/"
              "COM (the kinematics), not the velocity terms.")


if __name__ == "__main__":
    main()