"""
animate_trajectory.py

Plays back the trajectory saved by trajectory_optimizer.py
(trajectory_result.npz) in the MuJoCo viewer, in real time.

Usage:
    python3 animate_trajectory.py /path/to/arm_final_controlled.xml
    python3 animate_trajectory.py /path/to/arm_final_controlled.xml --save video.mp4
    python3 animate_trajectory.py /path/to/arm_final_controlled.xml --loop
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--trajectory", default="trajectory_result.npz")
    parser.add_argument("--loop", action="store_true", help="replay continuously")
    parser.add_argument("--save", default=None,
                         help="if given, save an MP4 to this path instead of "
                              "(or in addition to) live viewing")
    args = parser.parse_args()

    data_npz = np.load(args.trajectory)
    t, q = data_npz["time"], data_npz["q"]  # q shape: (3, N+1)

    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)

    if args.save:
        record_video(model, data, t, q, args.save)
    else:
        play_live(model, data, t, q, loop=args.loop)


def play_live(model, data, t, q, loop=False):
    """Real-time interactive playback in the MuJoCo viewer window."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start_wall = time.time()

            for k in range(q.shape[1]):
                if not viewer.is_running():
                    return
                data.qpos[:3] = q[:, k]
                mujoco.mj_forward(model, data)
                viewer.sync()

                if k < q.shape[1] - 1:
                    dt = t[k + 1] - t[k]
                    time.sleep(max(0.0, dt))

            if not loop:
                # Hold on the final frame until the window is closed.
                while viewer.is_running():
                    time.sleep(0.1)
                break

            time.sleep(0.5)  # brief pause before looping
            _ = start_wall  # (kept for clarity if you add timing diagnostics)


def record_video(model, data, t, q, out_path, width=640, height=480, fps=30):
    """
    Offscreen-render the trajectory to an MP4. Requires opencv-python:
        pip install opencv-python --break-system-packages

    Default resolution (640x480) matches MuJoCo's default offscreen
    framebuffer size. For higher resolution (e.g. 1280x720), add this to
    arm_final_controlled.xml instead of just raising width/height here:
        <visual>
          <global offwidth="1280" offheight="720"/>
        </visual>
    """
    try:
        import cv2
    except ImportError:
        print("Video export needs opencv-python. Install it with:")
        print("  pip install opencv-python --break-system-packages")
        return

    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    total_time = t[-1]
    n_frames = int(total_time * fps)

    for frame_i in range(n_frames + 1):
        frame_time = frame_i / fps
        # Find nearest trajectory knot for this frame's time.
        k = np.searchsorted(t, frame_time)
        k = min(k, q.shape[1] - 1)

        data.qpos[:3] = q[:, k]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data)
        frame = renderer.render()  # RGB, shape (height, width, 3)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Saved video to {out_path} ({n_frames + 1} frames at {fps} fps)")


if __name__ == "__main__":
    main()