export interface JointAngles {
  /** Base yaw rotation about the vertical axis, in degrees. */
  base: number;
  /** Shoulder pitch, in degrees. */
  shoulder: number;
  /** Elbow pitch relative to the upper arm, in degrees. */
  elbow: number;
}

/** Physical dimensions of the arm, in scene units. */
export const ARM = {
  baseHeight: 1.1,
  upperArm: 1.6,
  forearm: 1.3,
} as const;

export const JOINT_LIMITS: Record<keyof JointAngles, [number, number]> = {
  base: [-180, 180],
  shoulder: [-90, 90],
  elbow: [-150, 150],
};

export const HOME_POSE: JointAngles = { base: 0, shoulder: -30, elbow: 60 };

const deg2rad = (deg: number) => (deg * Math.PI) / 180;

/**
 * Forward kinematics for the tool center point (TCP).
 * Mirrors the nested-group transforms used by the 3D renderer so the
 * readout always matches what is drawn on screen.
 */
export function forwardKinematics(angles: JointAngles): {
  x: number;
  y: number;
  z: number;
} {
  const b = deg2rad(angles.base);
  const s = deg2rad(angles.shoulder);
  const e = deg2rad(angles.shoulder + angles.elbow);

  const planeX = -ARM.upperArm * Math.sin(s) - ARM.forearm * Math.sin(e);
  const y = ARM.baseHeight + ARM.upperArm * Math.cos(s) + ARM.forearm * Math.cos(e);

  return {
    x: planeX * Math.cos(b),
    y,
    z: -planeX * Math.sin(b),
  };
}
