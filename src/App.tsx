import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, Environment } from "@react-three/drei";
import { RoboticArm } from "./RoboticArm";
import {
  forwardKinematics,
  HOME_POSE,
  JOINT_LIMITS,
  type JointAngles,
} from "./kinematics";

const AXES: {
  key: keyof JointAngles;
  label: string;
  description: string;
}[] = [
  { key: "base", label: "Axis 1 · Base", description: "Yaw" },
  { key: "shoulder", label: "Axis 2 · Shoulder", description: "Pitch" },
  { key: "elbow", label: "Axis 3 · Elbow", description: "Pitch" },
];

function randomPose(): JointAngles {
  const rand = (key: keyof JointAngles) => {
    const [min, max] = JOINT_LIMITS[key];
    return Math.round(min + Math.random() * (max - min));
  };
  return { base: rand("base"), shoulder: rand("shoulder"), elbow: rand("elbow") };
}

export default function App() {
  const [angles, setAngles] = useState<JointAngles>(HOME_POSE);

  const tcp = useMemo(() => forwardKinematics(angles), [angles]);

  const update = (key: keyof JointAngles, value: number) =>
    setAngles((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="app">
      <div className="overlay-title">
        <h1>
          <span className="accent">3-Axis</span> Arm Simulator
        </h1>
        <p>Drag to orbit · scroll to zoom · use the sliders to pose the arm</p>
      </div>

      <div className="canvas-wrap">
        <Canvas shadows camera={{ position: [4.5, 3.5, 5.5], fov: 45 }}>
          <color attach="background" args={["#0b1120"]} />
          <hemisphereLight intensity={0.6} groundColor="#0b1120" />
          <directionalLight
            position={[5, 8, 5]}
            intensity={1.4}
            castShadow
            shadow-mapSize={[2048, 2048]}
          />
          <Environment preset="city" />

          <RoboticArm angles={angles} />

          <Grid
            position={[0, 0, 0]}
            args={[20, 20]}
            cellSize={0.5}
            cellColor="#1e293b"
            sectionSize={2.5}
            sectionColor="#334155"
            fadeDistance={22}
            infiniteGrid
          />

          <OrbitControls
            enablePan={false}
            minDistance={3}
            maxDistance={16}
            target={[0, 1.6, 0]}
          />
        </Canvas>
      </div>

      <div className="panel">
        <h2>Joint Control</h2>
        <p className="hint">Command each axis independently.</p>

        {AXES.map(({ key, label, description }) => {
          const [min, max] = JOINT_LIMITS[key];
          return (
            <div className="control" key={key}>
              <div className="control-header">
                <label htmlFor={key}>
                  {label} <span style={{ color: "#64748b" }}>· {description}</span>
                </label>
                <span className="value">{angles[key]}°</span>
              </div>
              <input
                id={key}
                type="range"
                min={min}
                max={max}
                step={1}
                value={angles[key]}
                onChange={(e) => update(key, Number(e.target.value))}
              />
            </div>
          );
        })}

        <div className="actions">
          <button onClick={() => setAngles(HOME_POSE)}>Home</button>
          <button className="secondary" onClick={() => setAngles(randomPose())}>
            Random
          </button>
        </div>

        <div className="tcp-readout">
          Tool center point (world)
          <div className="coords">
            <span>
              <b>X</b> {tcp.x.toFixed(2)}
            </span>
            <span>
              <b>Y</b> {tcp.y.toFixed(2)}
            </span>
            <span>
              <b>Z</b> {tcp.z.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
