# 3-axis-arm

A **MuJoCo** simulation of a 3-axis robotic arm (base yaw → shoulder pitch →
elbow pitch). The model ships with primitive placeholder geometry so it runs
immediately; drop in your **SolidWorks** STL exports later to simulate the real
arm.

## Layout

```
├── arm.xml            # MJCF model of the 3-axis arm (edit "FILL IN LATER" TODOs)
├── simulate.py        # Run the sim: interactive viewer, mp4 record, or trajectory plot
├── requirements.txt   # Python dependencies
└── assets/            # <- put your SolidWorks base.stl / link1..3.stl here (fill in later)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Interactive viewer (local machine with a display):

```bash
python simulate.py
```

Headless — record the demo motion to an mp4 (no display required):

```bash
python simulate.py --record out/arm.mp4
```

Headless — physics only, save a joint/TCP trajectory plot:

```bash
python simulate.py --no-viewer --plot out/trajectory.png
```

Useful flags: `--duration <seconds>`, `--record <path.mp4>`, `--plot <path.png>`,
`--no-viewer`.

> Headless rendering needs an OpenGL backend. On a headless server set
> `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (software) before running with
> `--record`.

## The model

| Axis | Joint    | Motion | Range         | Actuator |
| ---- | -------- | ------ | ------------- | -------- |
| 1    | `joint1` | Yaw    | ±180°         | `a1`     |
| 2    | `joint2` | Pitch  | ±90°          | `a2`     |
| 3    | `joint3` | Pitch  | ±150°         | `a3`     |

Each joint is driven by a MuJoCo `position` actuator; `data.ctrl[:]` takes the
three target angles in radians. A `tcp` site marks the tool center point and is
logged via a `framepos` sensor.

To command your own motion, edit `target_angles(t)` in `simulate.py`.

## Adding your SolidWorks geometry (fill in later)

1. In SolidWorks, **Save As → STL** for each link: `base`, `link1`, `link2`,
   `link3`. Copy the files into [`assets/`](assets/).
2. In [`arm.xml`](arm.xml), uncomment the `<mesh .../>` lines in `<asset>` and
   switch each link's placeholder `<geom>` to `type="mesh" mesh="<name>"`.
3. Confirm the STL units — `arm.xml` assumes millimetres (`scale="0.001 ..."`).

See [`assets/README.md`](assets/README.md) for details.
