# assets/ — SolidWorks mesh drop-in (fill in later)

Export each link of the arm from SolidWorks as an STL and place the files here:

| File        | Link                                   |
| ----------- | -------------------------------------- |
| `base.stl`  | Fixed base                             |
| `link1.stl` | Axis 1 body (base yaw)                 |
| `link2.stl` | Axis 2 body (shoulder / upper arm)     |
| `link3.stl` | Axis 3 body (elbow / forearm)          |

Then, in [`../arm.xml`](../arm.xml):

1. Uncomment the matching `<mesh .../>` lines in the `<asset>` block.
2. Change each link's placeholder `<geom ... type="cylinder|capsule" .../>` to
   `type="mesh" mesh="<name>"` (e.g. `mesh="link2_mesh"`).
3. Check the STL export units. `arm.xml` assumes millimetres and applies
   `scale="0.001 0.001 0.001"` to convert to MuJoCo's metres — adjust if your
   export is already in metres.

SolidWorks STL export tip: use **Save As → STL → Options** and set a fine
resolution so collision/visual geometry is accurate. The origin of each STL
should match the joint frame defined in `arm.xml` (child body positions), so
set the SolidWorks coordinate system per link accordingly, or nudge the `pos`
attributes in `arm.xml` after import.
