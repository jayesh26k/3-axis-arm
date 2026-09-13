"""
ik_solver.py

Numerical inverse kinematics using MuJoCo's Jacobian (position-only),
with a multi-start wrapper for robustness against the twisted link frames
inherited from the CAD export.

This is a direct refactor of jacobian_ik.py into a class so it can be
instantiated cleanly inside a ROS2 node (no module-level globals). The
core algorithm -- pseudoinverse Jacobian IK, multi-start seeding, and
least-folded-elbow ranking -- is unchanged from the original.
"""

import math

import mujoco
import numpy as np


class JacobianIKSolver:
    """
    Wraps a MuJoCo model + numerical Jacobian IK for a 3-DOF arm.

    JOINT_LIMITS and the analytic reference function are kept as
    documented in the original script -- update JOINT_LIMITS once real
    hardware limits are confirmed.
    """

    # Joint limits (radians): base, shoulder, elbow
    # NOTE: update these to match real hardware limits once confirmed.
    JOINT_LIMITS = [(-3.14, 3.14), (-2.14, 2.14), (-3.14, 3.14)]

    def __init__(self, model_path: str, end_effector_site: str = "end_effector"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site
        )

    # -----------------------------------------------------------------
    # Analytic IK (reference / comparison only -- NOT used directly).
    # Unreliable for this arm due to CAD twist offsets between links.
    # -----------------------------------------------------------------
    @staticmethod
    def inverse_kinematics_analytic(x, y, z, a=135, b=135, h=115):
        r = math.sqrt(x**2 + y**2)
        dz = z - h
        c = math.sqrt(r**2 + dz**2)

        if c > (a + b) or c < abs(a - b):
            raise ValueError(
                f"Target unreachable: distance {c:.1f}mm outside range "
                f"[{abs(a - b)}, {a + b}]mm"
            )

        theta1 = math.atan2(y, x)
        theta3 = math.acos((a**2 + b**2 - c**2) / (2 * a * b))

        phi = math.atan2(dz, r)
        alpha = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
        theta2 = phi + alpha

        return theta1, theta2, theta3

    # -----------------------------------------------------------------
    # Numerical IK using MuJoCo's Jacobian (pseudoinverse + joint clamping)
    # -----------------------------------------------------------------
    def solve_ik(self, target_pos, initial_qpos=(0.0, -0.5, -1.0),
                 max_iterations=200, step_size=0.1, tolerance=0.0001,
                 verbose=False):
        """
        Iteratively solve for joint angles that move the end-effector site
        to target_pos (meters, world frame).
        Returns (qpos, error_magnitude, iters).
        """
        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = initial_qpos
        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv))
        jacr = None  # rotational Jacobian not needed for a positioning task

        error_magnitude = float("inf")
        i = 0

        for i in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)

            current_ee_pos = self.data.site_xpos[self.site_id]
            error = target_pos - current_ee_pos
            error_magnitude = np.linalg.norm(error)

            if verbose and i % 20 == 0:
                print(f"  iter {i}: error={error_magnitude * 1000:.2f}mm")

            if error_magnitude < tolerance:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jacp_arm = jacp[:, :3]  # only the 3 real arm joints

            dtheta = np.linalg.pinv(jacp_arm) @ error
            self.data.qpos[:3] += step_size * dtheta

            for j in range(3):
                self.data.qpos[j] = np.clip(
                    self.data.qpos[j], self.JOINT_LIMITS[j][0], self.JOINT_LIMITS[j][1]
                )

        return self.data.qpos[:3].copy(), error_magnitude, i

    # -----------------------------------------------------------------
    # Multi-start wrapper: tries many random starting poses, keeps the
    # best (least-folded-elbow among converged) result.
    # -----------------------------------------------------------------
    def solve_ik_multistart(self, target_pos, num_attempts=30, max_iterations=200,
                             step_size=0.1, tolerance=0.001, seed=0,
                             preferred_qpos=(0.0, 0.0, 0.0),
                             error_tolerance_for_ranking=0.003, verbose=False):
        rng = np.random.default_rng(seed)
        preferred_qpos = np.array(preferred_qpos)

        candidates = []  # (qpos, error) for every attempt that converges well enough

        for attempt in range(num_attempts):
            random_seed_pose = tuple(
                rng.uniform(low, high) for (low, high) in self.JOINT_LIMITS
            )

            qpos, error, _ = self.solve_ik(
                target_pos,
                initial_qpos=random_seed_pose,
                max_iterations=max_iterations,
                step_size=step_size,
                tolerance=tolerance,
                verbose=False,
            )

            if verbose:
                print(f"attempt {attempt}: seed={np.degrees(random_seed_pose).round(1)}, "
                      f"error={error * 1000:.2f}mm")

            if error < error_tolerance_for_ranking:
                candidates.append((qpos.copy(), error))

        if not candidates:
            if verbose:
                print("No attempt converged well enough -- target may be unreachable.")
            return None, float("inf")

        def elbow_bend_magnitude(candidate):
            qpos, _ = candidate
            return abs(qpos[2])  # elbow angle magnitude, radians

        best_qpos, best_error = min(candidates, key=elbow_bend_magnitude)

        if verbose:
            print(f"\n{len(candidates)} candidate(s) converged.")
            print(f"Chosen (least-folded) qpos (deg): {np.degrees(best_qpos)}, "
                  f"error={best_error*1000:.3f}mm")

        return best_qpos, best_error