import { ARM, type JointAngles } from "./kinematics";

const deg2rad = (deg: number) => (deg * Math.PI) / 180;

interface RoboticArmProps {
  angles: JointAngles;
}

/** A cylindrical arm link that grows upward from its group origin along +Y. */
function Link({
  length,
  radius,
  color,
}: {
  length: number;
  radius: number;
  color: string;
}) {
  return (
    <mesh position={[0, length / 2, 0]} castShadow>
      <cylinderGeometry args={[radius, radius, length, 24]} />
      <meshStandardMaterial color={color} metalness={0.4} roughness={0.35} />
    </mesh>
  );
}

function Joint({ radius, color }: { radius: number; color: string }) {
  return (
    <mesh castShadow>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial color={color} metalness={0.6} roughness={0.25} />
    </mesh>
  );
}

/**
 * A 3-axis robotic arm rendered as nested groups:
 *   base yaw -> shoulder pitch -> elbow pitch.
 */
export function RoboticArm({ angles }: RoboticArmProps) {
  return (
    <group>
      {/* Static plinth */}
      <mesh position={[0, 0.1, 0]} receiveShadow castShadow>
        <cylinderGeometry args={[0.75, 0.9, 0.2, 40]} />
        <meshStandardMaterial color="#1e293b" metalness={0.3} roughness={0.6} />
      </mesh>

      {/* Axis 1 - base yaw */}
      <group rotation={[0, deg2rad(angles.base), 0]}>
        <Link length={ARM.baseHeight} radius={0.28} color="#334155" />

        {/* Axis 2 - shoulder pitch */}
        <group position={[0, ARM.baseHeight, 0]}>
          <Joint radius={0.34} color="#38bdf8" />
          <group rotation={[0, 0, deg2rad(angles.shoulder)]}>
            <Link length={ARM.upperArm} radius={0.2} color="#64748b" />

            {/* Axis 3 - elbow pitch */}
            <group position={[0, ARM.upperArm, 0]}>
              <Joint radius={0.26} color="#38bdf8" />
              <group rotation={[0, 0, deg2rad(angles.elbow)]}>
                <Link length={ARM.forearm} radius={0.15} color="#94a3b8" />

                {/* Tool center point */}
                <group position={[0, ARM.forearm, 0]}>
                  <mesh castShadow>
                    <boxGeometry args={[0.22, 0.22, 0.22]} />
                    <meshStandardMaterial
                      color="#f472b6"
                      emissive="#f472b6"
                      emissiveIntensity={0.35}
                      metalness={0.5}
                      roughness={0.3}
                    />
                  </mesh>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}
