"""Simulate the 3-axis arm in MuJoCo.

Examples
--------
Interactive viewer (needs a display / local machine)::

    python simulate.py

Headless: record an mp4 of the demo motion (no display required)::

    python simulate.py --record out/arm.mp4

Headless: physics only, save a joint/TCP trajectory plot::

    python simulate.py --no-viewer --plot out/trajectory.png

The demo drives each joint with a smooth sinusoidal sweep so you can see the
three axes (base yaw, shoulder pitch, elbow pitch) articulate. Swap in your own
``target_angles`` controller to command real trajectories.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "arm.xml")

# Joint limits (rad), used to clamp commanded targets. Keep in sync with arm.xml.
JOINT_RANGES = np.array([
    [-np.pi, np.pi],       # axis 1: base yaw
    [-np.pi / 2, np.pi / 2],  # axis 2: shoulder pitch
    [-2.618, 2.618],       # axis 3: elbow pitch
])


def target_angles(t: float) -> np.ndarray:
    """Desired joint angles (rad) at time ``t`` for the demo sweep.

    Replace the body of this function to command your own trajectory.
    """
    q = np.array([
        1.5 * np.sin(0.5 * t),
        0.6 * np.sin(0.7 * t + 0.5),
        1.2 * np.sin(0.9 * t + 1.0),
    ])
    return np.clip(q, JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


def run_viewer(duration: float) -> None:
    """Open the interactive MuJoCo viewer and run the demo."""
    import mujoco.viewer

    model, data = load_model()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (duration <= 0 or data.time < duration):
            data.ctrl[:] = target_angles(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()


def run_headless(duration: float, record: str | None, plot: str | None,
                 fps: int = 30) -> dict:
    """Step the simulation without a viewer.

    Optionally render an mp4 (``record``) and/or save a trajectory plot
    (``plot``). Returns a summary dict of the final state.
    """
    model, data = load_model()

    renderer = None
    frames: list[np.ndarray] = []
    if record:
        renderer = mujoco.Renderer(model, height=480, width=640)

    log_t: list[float] = []
    log_q: list[np.ndarray] = []
    log_tcp: list[np.ndarray] = []
    tcp_adr = model.sensor("tcp_pos").adr[0]

    next_frame_t = 0.0
    while data.time < duration:
        data.ctrl[:] = target_angles(data.time)
        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_q.append(data.qpos.copy())
        log_tcp.append(data.sensordata[tcp_adr:tcp_adr + 3].copy())

        if renderer is not None and data.time >= next_frame_t:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render())
            next_frame_t += 1.0 / fps

    if renderer is not None:
        renderer.close()

    if record and frames:
        _save_video(record, frames, fps)
    if plot:
        _save_plot(plot, np.array(log_t), np.array(log_q), np.array(log_tcp))

    return {
        "steps": len(log_t),
        "sim_time": data.time,
        "final_qpos": np.array(log_q[-1]),
        "final_tcp": np.array(log_tcp[-1]),
        "frames": len(frames),
    }


def _save_video(path: str, frames: list[np.ndarray], fps: int) -> None:
    import imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _save_plot(path: str, t: np.ndarray, q: np.ndarray, tcp: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    labels = ["axis 1 (base yaw)", "axis 2 (shoulder)", "axis 3 (elbow)"]
    for i, lbl in enumerate(labels):
        ax1.plot(t, np.degrees(q[:, i]), label=lbl)
    ax1.set_ylabel("joint angle (deg)")
    ax1.set_title("3-axis arm joint trajectories")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    for i, lbl in enumerate(["x", "y", "z"]):
        ax2.plot(t, tcp[:, i], label=f"TCP {lbl}")
    ax2.set_ylabel("TCP position (m)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-axis arm MuJoCo simulation")
    parser.add_argument("--duration", type=float, default=12.0,
                        help="simulation duration in seconds (<=0 = run until viewer closed)")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false",
                        help="run headless instead of opening the interactive viewer")
    parser.add_argument("--record", metavar="PATH", default=None,
                        help="render the demo to an mp4 at PATH (implies headless)")
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="save a joint/TCP trajectory plot to PATH (implies headless)")
    args = parser.parse_args()

    headless = (not args.viewer) or args.record or args.plot
    if headless:
        duration = args.duration if args.duration > 0 else 12.0
        summary = run_headless(duration, args.record, args.plot)
        print("Simulation complete:")
        print(f"  steps      : {summary['steps']}")
        print(f"  sim time   : {summary['sim_time']:.2f} s")
        print(f"  final qpos : {np.round(np.degrees(summary['final_qpos']), 1)} deg")
        print(f"  final TCP  : {np.round(summary['final_tcp'], 3)} m")
        if args.record:
            print(f"  video      : {args.record} ({summary['frames']} frames)")
        if args.plot:
            print(f"  plot       : {args.plot}")
    else:
        run_viewer(args.duration)


if __name__ == "__main__":
    main()
