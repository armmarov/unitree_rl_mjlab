"""Cap the base yaw rate of a motion NPZ.

Reduces aggressive cornering by clipping yaw rate at a configurable threshold
and reintegrating the motion. World positions, orientations, and velocities are
all updated consistently.
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def axis_angle_from_quat_diff(q_curr_xyzw: np.ndarray, q_next_xyzw: np.ndarray, dt: float) -> np.ndarray:
  """Numerical angular velocity from successive quaternions (T,4) xyzw."""
  r_curr = R.from_quat(q_curr_xyzw)
  r_next = R.from_quat(q_next_xyzw)
  r_rel = r_next * r_curr.inv()
  return r_rel.as_rotvec() / dt


def main():
  parser = argparse.ArgumentParser(description="Cap base yaw rate in motion NPZ")
  parser.add_argument("input_npz")
  parser.add_argument("--output-npz", help="Output path (default: <input>_yawcap.npz)")
  parser.add_argument("--max-yaw-rate-deg", type=float, default=100.0,
                      help="Max base yaw rate in deg/s (default: 100)")
  args = parser.parse_args()

  data = dict(np.load(args.input_npz))
  fps = float(data["fps"][0])
  dt = 1.0 / fps
  T = data["joint_pos"].shape[0]
  num_bodies = data["body_pos_w"].shape[1]

  # Extract base yaw curve.
  base_quat_wxyz = data["body_quat_w"][:, 0, :]
  base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]
  base_euler = R.from_quat(base_quat_xyzw).as_euler("xyz")
  yaw_orig = np.unwrap(base_euler[:, 2])

  # Compute and clip yaw rate.
  yaw_rate = np.gradient(yaw_orig) * fps
  max_rate = np.deg2rad(args.max_yaw_rate_deg)
  yaw_rate_new = np.clip(yaw_rate, -max_rate, max_rate)

  n_clipped = int((np.abs(yaw_rate) > max_rate).sum())
  print(f"Frames clipped: {n_clipped}/{T} ({100*n_clipped/T:.1f}%)")
  print(f"  Original max |yaw_rate|: {np.degrees(np.abs(yaw_rate).max()):.0f} deg/s")
  print(f"  New max |yaw_rate|:      {np.degrees(np.abs(yaw_rate_new).max()):.0f} deg/s")

  # Reintegrate yaw curve from clipped rate (start from original yaw[0]).
  yaw_new = np.zeros_like(yaw_orig)
  yaw_new[0] = yaw_orig[0]
  for t in range(1, T):
    yaw_new[t] = yaw_new[t-1] + yaw_rate_new[t] * dt
  delta_yaw = yaw_new - yaw_orig  # per-frame correction

  # Reintegrate base XY position using base velocity rotated by NEW yaw.
  base_pos_orig = data["body_pos_w"][:, 0, :].copy()
  base_lin_vel_w = data["body_lin_vel_w"][:, 0, :]

  # Convert world velocity to body frame (frame-by-frame), then rotate by new yaw.
  base_lin_vel_body = np.zeros_like(base_lin_vel_w)
  for t in range(T):
    c, s = np.cos(yaw_orig[t]), np.sin(yaw_orig[t])
    Rz_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    base_lin_vel_body[t] = Rz_inv @ base_lin_vel_w[t]

  base_pos_new = np.zeros_like(base_pos_orig)
  base_pos_new[0] = base_pos_orig[0]
  for t in range(1, T):
    c, s = np.cos(yaw_new[t-1]), np.sin(yaw_new[t-1])
    Rz_new = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    base_pos_new[t] = base_pos_new[t-1] + (Rz_new @ base_lin_vel_body[t-1]) * dt

  # Build new body positions and orientations by applying R_delta around base.
  body_pos_w_new = np.zeros_like(data["body_pos_w"])
  body_quat_w_new = np.zeros_like(data["body_quat_w"])

  for t in range(T):
    c, s = np.cos(delta_yaw[t]), np.sin(delta_yaw[t])
    R_delta = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_delta_obj = R.from_euler("z", delta_yaw[t])

    rel = data["body_pos_w"][t] - base_pos_orig[t]      # relative to original base
    rotated = rel @ R_delta.T
    body_pos_w_new[t] = base_pos_new[t] + rotated

    for b in range(num_bodies):
      q_old_xyzw = data["body_quat_w"][t, b, [1, 2, 3, 0]]
      q_new_xyzw = (R_delta_obj * R.from_quat(q_old_xyzw)).as_quat()
      body_quat_w_new[t, b] = q_new_xyzw[[3, 0, 1, 2]]

  # Recompute velocities by numerical differentiation (clean, consistent).
  body_lin_vel_w_new = np.gradient(body_pos_w_new, dt, axis=0)

  body_ang_vel_w_new = np.zeros_like(data["body_ang_vel_w"])
  q_xyzw = body_quat_w_new[:, :, [1, 2, 3, 0]]
  for b in range(num_bodies):
    qb = q_xyzw[:, b, :]  # (T, 4)
    body_ang_vel_w_new[:-2, b] = axis_angle_from_quat_diff(qb[:-2], qb[2:], 2*dt)
    body_ang_vel_w_new[-2, b] = axis_angle_from_quat_diff(qb[-2:-1], qb[-1:], dt)
    body_ang_vel_w_new[-1, b] = body_ang_vel_w_new[-2, b]
    body_ang_vel_w_new[0, b] = axis_angle_from_quat_diff(qb[0:1], qb[1:2], dt)

  data["body_pos_w"] = body_pos_w_new
  data["body_quat_w"] = body_quat_w_new
  data["body_lin_vel_w"] = body_lin_vel_w_new.astype(np.float32)
  data["body_ang_vel_w"] = body_ang_vel_w_new.astype(np.float32)

  output = args.output_npz or args.input_npz.replace(".npz", "_yawcap.npz")
  np.savez(output, **data)
  print(f"Saved: {output}")


if __name__ == "__main__":
  main()
