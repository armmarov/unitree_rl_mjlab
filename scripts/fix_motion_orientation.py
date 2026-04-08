"""Fix root orientation of a motion CSV so the robot faces forward (yaw=0)."""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
  parser = argparse.ArgumentParser(description="Fix motion CSV root orientation to face forward")
  parser.add_argument("input_csv", help="Input CSV motion file")
  parser.add_argument("--output-csv", help="Output CSV file (default: overwrites input)")
  parser.add_argument("--target-yaw", type=float, default=0.0,
                      help="Target yaw in degrees (default: 0 = forward)")
  parser.add_argument("--height-offset", type=float, default=0.0,
                      help="Add offset to root Z height in meters (e.g. 0.25)")
  args = parser.parse_args()

  motion = np.loadtxt(args.input_csv, delimiter=",")
  print(f"Loaded: {motion.shape[0]} frames")

  # Get first frame orientation.
  first_quat_xyzw = motion[0, 3:7]
  first_euler = R.from_quat(first_quat_xyzw).as_euler("xyz", degrees=True)
  print(f"First frame euler (xyz deg): {first_euler}")
  print(f"Current yaw: {first_euler[2]:.1f} deg")

  # Compute yaw correction: rotate all frames so first frame faces target yaw.
  yaw_offset = args.target_yaw - first_euler[2]
  print(f"Applying yaw correction: {yaw_offset:.1f} deg")

  r_correction = R.from_euler("z", yaw_offset, degrees=True)

  # Rotate root position around origin (keep height).
  for i in range(motion.shape[0]):
    # Rotate position (XY only, keep Z).
    pos = motion[i, :3]
    pos_centered = pos - motion[0, :3]  # Center on first frame.
    pos_rotated = r_correction.apply(pos_centered)
    motion[i, :3] = pos_rotated + np.array([0.0, 0.0, motion[0, 2]])

    # Rotate quaternion.
    r_frame = R.from_quat(motion[i, 3:7])
    r_corrected = r_correction * r_frame
    motion[i, 3:7] = r_corrected.as_quat()

  # Apply height offset.
  if args.height_offset != 0.0:
    motion[:, 2] += args.height_offset
    print(f"Applied height offset: +{args.height_offset}m")
    print(f"New height range: {motion[:, 2].min():.3f} - {motion[:, 2].max():.3f}m")

  # Verify.
  new_euler = R.from_quat(motion[0, 3:7]).as_euler("xyz", degrees=True)
  print(f"New first frame euler (xyz deg): {new_euler}")
  print(f"New yaw: {new_euler[2]:.1f} deg")

  output_path = args.output_csv or args.input_csv
  np.savetxt(output_path, motion, delimiter=",")
  print(f"Saved to: {output_path}")


if __name__ == "__main__":
  main()
