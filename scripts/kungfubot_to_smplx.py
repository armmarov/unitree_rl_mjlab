"""Convert KungfuBot/PBHC motion pkl to SMPL-X npz format for GMR retargeting."""

import argparse
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
  parser = argparse.ArgumentParser(description="Convert KungfuBot pkl to SMPL-X npz for GMR")
  parser.add_argument("input_pkl", help="Input KungfuBot motion pkl file")
  parser.add_argument("--output-npz", help="Output SMPL-X npz file (default: same name with .npz)")
  args = parser.parse_args()

  # Load KungfuBot pkl.
  data = joblib.load(args.input_pkl)
  key = list(data.keys())[0]
  motion = data[key]

  pose_aa = motion["pose_aa"]          # (T, 27, 3) axis-angle per joint
  root_trans = motion["root_trans_offset"]  # (T, 3)
  root_rot = motion["root_rot"]        # (T, 4) quaternion
  fps = motion["fps"]
  num_frames = pose_aa.shape[0]

  print(f"Motion: {key.split('/')[-1]}")
  print(f"Frames: {num_frames}, FPS: {fps}, Duration: {num_frames/fps:.1f}s")

  # Convert root quaternion (wxyz or xyzw?) to axis-angle.
  # KungfuBot uses (w, x, y, z) format based on PHC convention.
  root_rot_scipy = R.from_quat(root_rot[:, [1, 2, 3, 0]])  # wxyz -> xyzw for scipy
  root_orient = root_rot_scipy.as_rotvec().astype(np.float32)  # (T, 3)

  # Build SMPL-X compatible format.
  # pose_aa has 27 joints. SMPL-X expects:
  # - root_orient: (T, 3) - root rotation (already extracted)
  # - pose_body: (T, 63) - 21 body joints x 3
  # - pose_hand: (T, 90) - 30 hand joints x 3 (zeros, no hand data)
  # - pose_jaw: (T, 3) - jaw rotation (zeros)
  # - pose_eye: (T, 6) - eye rotations (zeros)

  # KungfuBot pose_aa: joint 0 is root (skip), joints 1-21 are body, rest are extra.
  # Use first 21 body joints after root.
  body_joints = min(21, pose_aa.shape[1] - 1)
  pose_body = pose_aa[:, 1:1+body_joints, :].reshape(num_frames, -1).astype(np.float32)

  # Pad to 63 if less than 21 joints.
  if pose_body.shape[1] < 63:
    pad = np.zeros((num_frames, 63 - pose_body.shape[1]), dtype=np.float32)
    pose_body = np.concatenate([pose_body, pad], axis=1)

  # Save as SMPL-X npz.
  output_path = args.output_npz or args.input_pkl.replace(".pkl", "_smplx.npz")
  np.savez(
    output_path,
    trans=root_trans.astype(np.float32),
    root_orient=root_orient,
    pose_body=pose_body,
    pose_hand=np.zeros((num_frames, 90), dtype=np.float32),
    pose_jaw=np.zeros((num_frames, 3), dtype=np.float32),
    pose_eye=np.zeros((num_frames, 6), dtype=np.float32),
    betas=np.zeros(16, dtype=np.float32),
    gender="neutral",
    surface_model_type="smplx",
    mocap_frame_rate=float(fps),
    mocap_time_length=float(num_frames / fps),
  )
  print(f"Saved SMPL-X npz to: {output_path}")
  print(f"  root_orient: {root_orient.shape}")
  print(f"  pose_body: {pose_body.shape}")
  print(f"  trans: {root_trans.shape}")


if __name__ == "__main__":
  main()
