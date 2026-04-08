"""Prepend a smooth standing-to-motion transition to a CSV motion file.

Adds frames that interpolate from PM01's home pose to the first frame
of the motion, so the robot transitions smoothly from standing still.
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# PM01 home pose: root position, root quaternion (xyzw), 24 joint angles.
HOME_ROOT_POS = np.array([0.0, 0.0, 0.92])
HOME_ROOT_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])  # Identity
HOME_JOINTS = np.array([
  -0.24, 0.0, 0.0, 0.48, -0.24, 0.0,    # Left leg
  -0.24, 0.0, 0.0, 0.48, -0.24, 0.0,    # Right leg
  0.0,                                     # Waist
  0.35, 0.25, 0.0, -0.7, 0.0,            # Left arm
  0.35, -0.25, 0.0, -0.7, 0.0,           # Right arm
  0.0,                                     # Head
])


def interpolate_frames(start_frame, end_frame, num_frames):
  """Interpolate between two frames with SLERP for quaternions."""
  frames = np.zeros((num_frames, start_frame.shape[0]))

  for i in range(num_frames):
    t = i / max(num_frames - 1, 1)
    # Smooth ease-in-out curve.
    t = t * t * (3 - 2 * t)

    # Linear interpolation for position.
    frames[i, :3] = (1 - t) * start_frame[:3] + t * end_frame[:3]

    # SLERP for quaternion (columns 3:7, xyzw format).
    r_start = R.from_quat(start_frame[3:7])
    r_end = R.from_quat(end_frame[3:7])
    slerp = Slerp([0, 1], R.concatenate([r_start, r_end]))
    frames[i, 3:7] = slerp(t).as_quat()

    # Linear interpolation for joint angles.
    frames[i, 7:] = (1 - t) * start_frame[7:] + t * end_frame[7:]

  return frames


def main():
  parser = argparse.ArgumentParser(
    description="Prepend standing-to-motion transition to CSV motion file"
  )
  parser.add_argument("input_csv", help="Input CSV motion file")
  parser.add_argument("--output-csv", help="Output CSV file (default: input with _with_transition suffix)")
  parser.add_argument("--transition-seconds", type=float, default=1.5,
                      help="Duration of standing-to-motion transition (default: 1.5s)")
  parser.add_argument("--hold-seconds", type=float, default=0.5,
                      help="Duration to hold standing pose before transition (default: 0.5s)")
  parser.add_argument("--fps", type=float, default=30.0,
                      help="Frame rate of the CSV file (default: 30)")
  args = parser.parse_args()

  # Load original motion.
  motion = np.loadtxt(args.input_csv, delimiter=",")
  num_joints = motion.shape[1] - 7
  print(f"Loaded: {motion.shape[0]} frames, {num_joints} joints")

  # Detect and skip bad initial frames (large position jumps).
  skip = 0
  for i in range(1, min(5, motion.shape[0])):
    jump = np.linalg.norm(motion[i, :3] - motion[i - 1, :3])
    if jump > 0.1:
      skip = i
      print(f"Skipping frame 0-{i-1}: position jump of {jump:.3f}m detected at frame {i-1}→{i}")
      break
  if skip > 0:
    motion = motion[skip:]
    print(f"Using {motion.shape[0]} frames after skipping")

  # Build home frame matching CSV format.
  home_frame = np.zeros(motion.shape[1])
  home_frame[:3] = motion[0, :3]  # Use first frame's root XY position, not home's.
  home_frame[2] = HOME_ROOT_POS[2]  # Use home height.
  home_frame[3:7] = HOME_ROOT_QUAT_XYZW
  home_frame[7:7 + len(HOME_JOINTS)] = HOME_JOINTS[:num_joints]

  first_frame = motion[0]

  # Generate hold frames (standing still).
  hold_frames_count = int(args.hold_seconds * args.fps)
  hold_frames = np.tile(home_frame, (hold_frames_count, 1))

  # Generate transition frames.
  transition_frames_count = int(args.transition_seconds * args.fps)
  transition_frames = interpolate_frames(home_frame, first_frame, transition_frames_count)

  # Concatenate: hold + transition + original motion.
  full_motion = np.vstack([hold_frames, transition_frames, motion])
  print(f"Hold: {hold_frames_count} frames ({args.hold_seconds}s)")
  print(f"Transition: {transition_frames_count} frames ({args.transition_seconds}s)")
  print(f"Original: {motion.shape[0]} frames")
  print(f"Total: {full_motion.shape[0]} frames ({full_motion.shape[0] / args.fps:.1f}s)")

  # Save.
  if args.output_csv:
    output_path = args.output_csv
  else:
    output_path = args.input_csv.replace(".csv", "_with_transition.csv")

  np.savetxt(output_path, full_motion, delimiter=",")
  print(f"Saved to: {output_path}")


if __name__ == "__main__":
  main()
