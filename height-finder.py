import math

import mujoco
import mujoco.viewer
import numpy as np


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

model = mujoco.MjModel.from_xml_path("arm_final_controlled.xml")
data = mujoco.MjData(model)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

# Joint limits (radians): base, shoulder, elbow
JOINT_LIMITS = [(-3.14, 3.14), (-3.14, 1.57), (-3.14, 3.14)]


# ---------------------------------------------------------------------------
# Analytic IK (kept for reference / comparison only -- NOT used directly).
# theta1 and theta2 are unreliable for this arm because the real CAD frames
# include twist offsets between links, so the flat-plane assumption breaks.
# ---------------------------------------------------------------------------

def inverse_kinematics_analytic(x, y, z, a=105, b=105, h=100):
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


# ---------------------------------------------------------------------------
# Numerical IK using MuJoCo's Jacobian (pseudoinverse + joint clamping).
# ---------------------------------------------------------------------------

def solve_ik(target_pos, initial_qpos=(0.0, -0.5, -1.0),
             max_iterations=200, step_size=0.1, tolerance=0.001, verbose=True):
    """
    Iteratively solve for joint angles that move the end-effector site to
    target_pos (meters, world frame). Starts from initial_qpos.
    Returns (qpos, error_magnitude, iters).
    """
    data.qpos[0], data.qpos[1], data.qpos[2] = initial_qpos
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = None  # rotational Jacobian not needed for a positioning task

    error_magnitude = float("inf")
    i = 0

    for i in range(max_iterations):
        mujoco.mj_forward(model, data)

        current_ee_pos = data.site_xpos[site_id]
        error = target_pos - current_ee_pos
        error_magnitude = np.linalg.norm(error)

        if verbose and i % 20 == 0:
            print(f"  iter {i}: error={error_magnitude * 1000:.2f}mm")

        if error_magnitude < tolerance:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jacp_arm = jacp[:, :3]  # only the 3 real arm joints, ignore target's freejoint columns

        dtheta = np.linalg.pinv(jacp_arm) @ error
        data.qpos[:3] += step_size * dtheta

        for j in range(3):
            data.qpos[j] = np.clip(data.qpos[j], JOINT_LIMITS[j][0], JOINT_LIMITS[j][1])

    return data.qpos[:3].copy(), error_magnitude, i


# ---------------------------------------------------------------------------
# Multi-start wrapper: tries many random starting poses instead of guessing
# one seed by hand. Keeps the best (lowest-error) result found.
# ---------------------------------------------------------------------------

def solve_ik_multistart(target_pos, num_attempts=30, max_iterations=200,
                         step_size=0.1, tolerance=0.001, seed=0):
    rng = np.random.default_rng(seed)

    best_qpos = None
    best_error = float("inf")

    for attempt in range(num_attempts):
        random_seed_pose = tuple(
            rng.uniform(low, high) for (low, high) in JOINT_LIMITS
        )

        qpos, error, _ = solve_ik(
            target_pos,
            initial_qpos=random_seed_pose,
            max_iterations=max_iterations,
            step_size=step_size,
            tolerance=tolerance,
            verbose=False,
        )

        print(f"attempt {attempt}: seed={np.degrees(random_seed_pose).round(1)}, "
              f"error={error * 1000:.2f}mm")

        if error < best_error:
            best_error = error
            best_qpos = qpos.copy()

        if best_error < tolerance:
            print(f"Converged on attempt {attempt}, error={best_error * 1000:.3f}mm")
            break

    return best_qpos, best_error


def map_reachable_workspace(num_samples=500, seed=0):
    """
    Samples random VALID joint configurations (within real limits) and
    records where the end effector actually lands for each. This maps out
    the genuine reachable workspace directly, without needing IK at all.
    """
    rng = np.random.default_rng(seed)
    positions = []

    for _ in range(num_samples):
        sample_qpos = tuple(rng.uniform(low, high) for (low, high) in JOINT_LIMITS)
        data.qpos[0], data.qpos[1], data.qpos[2] = sample_qpos
        mujoco.mj_forward(model, data)
        positions.append(data.site_xpos[site_id].copy())

    positions = np.array(positions) * 1000  # convert to mm

    print(f"Sampled {num_samples} valid joint configurations.")
    print(f"X range: {positions[:,0].min():.1f} to {positions[:,0].max():.1f} mm")
    print(f"Y range: {positions[:,1].min():.1f} to {positions[:,1].max():.1f} mm")
    print(f"Z range: {positions[:,2].min():.1f} to {positions[:,2].max():.1f} mm")

    # How close does ANY sample get to ground level (z=0), where your
    # actual pick targets will be?
    near_ground = positions[np.abs(positions[:, 2]) < 20]  # within 20mm of z=0
    print(f"\nSamples within 20mm of ground level (z=0): {len(near_ground)} / {num_samples}")
    if len(near_ground) > 0:
        print("Example ground-level-ish reachable points (mm):")
        for p in near_ground[:10]:
            print(f"  {p.round(1)}")

    return positions


def find_minimum_reachable_height(num_steps=100):
    """
    Systematically sweeps shoulder and elbow through their full ranges
    (base fixed at 0, since it doesn't affect height) to find the true
    minimum z the end effector can reach -- more precise than random sampling.
    """
    shoulder_range = np.linspace(JOINT_LIMITS[1][0], JOINT_LIMITS[1][1], num_steps)
    elbow_range = np.linspace(JOINT_LIMITS[2][0], JOINT_LIMITS[2][1], num_steps)

    min_z = float("inf")
    min_z_pose = None

    for shoulder in shoulder_range:
        for elbow in elbow_range:
            data.qpos[0], data.qpos[1], data.qpos[2] = 0.0, shoulder, elbow
            mujoco.mj_forward(model, data)
            z = data.site_xpos[site_id][2]
            if z < min_z:
                min_z = z
                min_z_pose = (0.0, shoulder, elbow)

    print(f"Absolute minimum reachable height: {min_z * 1000:.1f}mm")
    print(f"Achieved at qpos (deg): {np.degrees(min_z_pose)}")

    data.qpos[0], data.qpos[1], data.qpos[2] = min_z_pose
    mujoco.mj_forward(model, data)
    print(f"Full position at minimum height (mm): {data.site_xpos[site_id] * 1000}")

    return min_z_pose, min_z


if __name__ == "__main__":
    # --- Map the real reachable workspace first ---
    map_reachable_workspace(num_samples=500)

    # --- Find the true minimum reachable height precisely ---
    print()
    find_minimum_reachable_height()

    # --- Then try IK toward a target once we know the real workspace ---
    # target_pos = np.array([0.150, 0.0, 0.0])  # meters
    # best_qpos, best_error = solve_ik_multistart(target_pos, num_attempts=30)
    # data.qpos[0], data.qpos[1], data.qpos[2] = best_qpos
    # mujoco.mj_forward(model, data)
    # print(f"\nBest result: error={best_error * 1000:.2f}mm")
    # print(f"Best qpos (deg): {np.degrees(best_qpos)}")
    # print(f"Final end-effector position (mm): {data.site_xpos[site_id] * 1000}")
    # print(f"Target was (mm): {target_pos * 1000}")

    mujoco.viewer.launch(model, data)