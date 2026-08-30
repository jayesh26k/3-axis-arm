# 3-axis-arm

An interactive **3-axis robotic arm simulator** rendered in real time in the
browser. Pose the arm by commanding each of its three joints and watch the tool
center point (TCP) update live.

<img src="public/arm.svg" alt="3-axis arm logo" width="48" height="48" />

## Features

- Real-time 3D visualization of a 3-axis arm (base yaw, shoulder pitch, elbow pitch).
- Per-axis slider controls with live joint-angle readouts and enforced joint limits.
- Forward-kinematics readout of the tool center point in world coordinates.
- Orbit / zoom camera, `Home` and `Random` pose presets.

## Tech stack

- [Vite](https://vitejs.dev/) + [React](https://react.dev/) + TypeScript
- [three.js](https://threejs.org/) via
  [@react-three/fiber](https://github.com/pmndrs/react-three-fiber) and
  [@react-three/drei](https://github.com/pmndrs/drei)

## Getting started

```bash
npm install      # install dependencies
npm run dev      # start the dev server on http://localhost:5173
```

Other scripts:

```bash
npm run build      # type-check and produce a production build in dist/
npm run preview    # preview the production build
npm run lint       # run ESLint
npm run typecheck  # run the TypeScript compiler without emitting
```

## Project structure

```
├── index.html            # Vite entry HTML
├── src/
│   ├── main.tsx          # React entry point
│   ├── App.tsx           # UI, control panel, and 3D canvas
│   ├── RoboticArm.tsx    # Nested-group 3D arm model
│   ├── kinematics.ts     # Joint definitions, limits, and forward kinematics
│   └── index.css         # Styling
└── .cursor/environment.json  # Cloud Agent dev environment
```

## Kinematics

The arm is modeled as three revolute joints in series:

| Axis | Joint    | Motion | Limits        |
| ---- | -------- | ------ | ------------- |
| 1    | Base     | Yaw    | -180° … 180°  |
| 2    | Shoulder | Pitch  | -90° … 90°    |
| 3    | Elbow    | Pitch  | -150° … 150°  |

`forwardKinematics()` in `src/kinematics.ts` computes the TCP world position
using the same transform chain the renderer applies, so the on-screen readout
always matches the rendered pose.
