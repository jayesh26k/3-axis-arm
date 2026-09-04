"""
find_elbow_tip_offset.py

Finds the actual local-frame extent of elbow-joint.STL (the forearm link),
so we know precisely which axis it extends along and how far -- instead of
guessing the end-effector site offset.

Run this in the same folder as your meshes/ directory.
"""

from stl import mesh
import numpy as np

m = mesh.Mesh.from_file("meshes/elbow-joint.STL")

# All vertex coordinates across all triangles, for each axis
x_coords = m.vectors[:, :, 0]
y_coords = m.vectors[:, :, 1]
z_coords = m.vectors[:, :, 2]

x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()
z_min, z_max = z_coords.min(), z_coords.max()

x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

print("Elbow-joint mesh bounding box (local frame, mm):")
print(f"  X: {x_min:.2f} to {x_max:.2f}  (range {x_range:.2f})")
print(f"  Y: {y_min:.2f} to {y_max:.2f}  (range {y_range:.2f})")
print(f"  Z: {z_min:.2f} to {z_max:.2f}  (range {z_range:.2f})")

ranges = {"X": x_range, "Y": y_range, "Z": z_range}
longest_axis = max(ranges, key=ranges.get)

print(f"\nLongest axis: {longest_axis} (range {ranges[longest_axis]:.2f}mm)")
print("This is very likely the direction the forearm physically extends.")

if longest_axis == "X":
    far_end = x_max if abs(x_max) > abs(x_min) else x_min
elif longest_axis == "Y":
    far_end = y_max if abs(y_max) > abs(y_min) else y_min
else:
    far_end = z_max if abs(z_max) > abs(z_min) else z_min

print(f"\nSuggested end-effector site position (local frame, meters):")
if longest_axis == "X":
    print(f'  pos="{far_end/1000:.4f} 0 0"')
elif longest_axis == "Y":
    print(f'  pos="0 {far_end/1000:.4f} 0"')
else:
    print(f'  pos="0 0 {far_end/1000:.4f}"')

print("\nNote: this assumes the joint pivot (local origin) sits at the NEAR")
print("end of the link, not the middle. If the origin is actually in the")
print("middle of the mesh, the true tip offset would be roughly half this")
print("range instead of the full min/max value shown above -- visually")
print("confirm in the viewer once applied.")
