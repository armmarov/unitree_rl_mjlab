# Motion Tracking Training Guide (PM01)

This guide explains how to train PM01 to perform special movements (dance, gestures, etc.) using motion tracking in mjlab.

## 1. Approaches Overview

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Direct Imitation (mjlab)** | Exponential error rewards tracking reference pose frame-by-frame | Simple, reliable, no discriminator tuning | Requires good motion data |
| **DeepMimic** | Same as above + Random Start Initialization (RSI) | Proven on PM01 dance (EngineAI EXP-09) | Needs RSI sync |
| **AMP (Adversarial)** | Discriminator classifies expert vs policy motion | Style transfer, no per-frame tracking | Cold-start problem when distributions are far apart |

**Recommendation:** Use mjlab's direct imitation approach. EngineAI's experiments confirmed that AMP's discriminator fails when the robot starts far from the reference motion (cold-start problem — discriminator becomes perfect classifier in 2-3 iterations and locks up). Pure DeepMimic achieved reward 74 and 15s episodes on PM01 dance.

## 2. How mjlab Motion Tracking Works

The system is in `src/tasks/tracking/` and works as follows:

### Observation Space
- **Actor**: motion command (joint pos/vel), anchor pose in body frame, base ang/lin vel, joint pos/vel, last action
- **Critic**: all actor obs + body positions/orientations relative to base

### Reward Structure (exponential error)
All rewards use `exp(-error / std^2)` form:

| Reward | Weight | What it tracks |
|---|---|---|
| `motion_global_root_pos` | 0.5 | Root position error (std=0.3m) |
| `motion_global_root_ori` | 0.5 | Root orientation error (std=0.4) |
| `motion_body_pos` | 1.0 | All body position errors (std=0.3m) |
| `motion_body_ori` | 1.0 | All body orientation errors (std=0.4) |
| `motion_body_lin_vel` | 1.0 | Body linear velocity tracking (std=1.0) |
| `motion_body_ang_vel` | 1.0 | Body angular velocity tracking (std=3.14) |
| `action_rate_l2` | -0.1 | Action smoothness |
| `joint_limit` | -10.0 | Joint limit violations |
| `self_collisions` | -10.0 | Self-contact penalty |

### Adaptive Motion Sampling
Instead of always starting from frame 0, the system tracks which motion phases the robot fails at most, and samples those more often. This acts as an automatic curriculum.

### Termination Conditions
- Episode timeout (10s default)
- Root height deviation > 0.25m from reference
- Root orientation deviation > 0.8
- End-effector height deviation > 0.25m

## 3. Motion Data Preparation

### Option A: From CSV (MoCap/Retargeted)

1. Prepare a CSV file with columns: `pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, joint_0, ..., joint_23`
   - Position: root position in meters
   - Quaternion: root orientation in XYZW format
   - Joints: 24 joint angles in radians (matching PM01 joint order j00-j23)

2. Convert to NPZ:
```bash
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/pm01/your_motion.csv \
  --output-name your_motion.npz \
  --input-fps 30 \
  --output-fps 100 \
  --device cuda:0
```

Note: Set `--output-fps 100` to match our 100Hz control frequency.

### Option B: From Blender/Mixamo

EngineAI has a Blender export pipeline at:
```
engineai_rl_workspace_upd/engineai_gym/resources/robots/biped/pm01/mocap_motions/
  export_from_blender.py   # Mixamo FBX → PM01 joint angles (JSON)
  preview_in_blender.py    # Verify exported motion
  data.py                  # Joint mapping definitions
```

Workflow:
1. Download motion from Mixamo as FBX
2. Import into Blender
3. Run `export_from_blender.py` to convert to PM01 joint angles (JSON)
4. Convert JSON to CSV format matching the CSV spec above
5. Run `csv_to_npz.py` to create NPZ

### NPZ Output Format
```python
{
  "fps": [100],                              # Frame rate
  "joint_pos": ndarray(T, 24),               # Joint positions over time
  "joint_vel": ndarray(T, 24),               # Joint velocities
  "body_pos_w": ndarray(T, num_bodies, 3),   # Body positions in world
  "body_quat_w": ndarray(T, num_bodies, 4),  # Body quaternions (WXYZ)
  "body_lin_vel_w": ndarray(T, num_bodies, 3),
  "body_ang_vel_w": ndarray(T, num_bodies, 3),
}
```

## 4. Adding PM01 Tracking Task

Follow G1's pattern. Create these files:

### `src/tasks/tracking/config/pm01/__init__.py`
```python
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import engineai_pm01_flat_tracking_env_cfg
from .rl_cfg import engineai_pm01_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="EngineAI-PM01-Tracking",
  env_cfg=engineai_pm01_flat_tracking_env_cfg(),
  play_env_cfg=engineai_pm01_flat_tracking_env_cfg(play=True),
  rl_cfg=engineai_pm01_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
```

### `src/tasks/tracking/config/pm01/env_cfgs.py`
```python
from src.assets.robots import PM01_ACTION_SCALE, get_pm01_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def engineai_pm01_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}

  # 100Hz control to match velocity training.
  cfg.decimation = 2

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="link_base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="link_base", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM01_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "link_torso_yaw"
  motion_cmd.body_names = (
    "link_base",
    "link_hip_pitch_l", "link_knee_pitch_l", "link_ankle_roll_l",
    "link_hip_pitch_r", "link_knee_pitch_r", "link_ankle_roll_r",
    "link_torso_yaw",
    "link_shoulder_pitch_l", "link_elbow_pitch_l", "link_elbow_yaw_l",
    "link_shoulder_pitch_r", "link_elbow_pitch_r", "link_elbow_yaw_r",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("link_torso_yaw",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "link_ankle_roll_l",
    "link_ankle_roll_r",
    "link_elbow_yaw_l",
    "link_elbow_yaw_r",
  )

  cfg.viewer.body_name = "link_torso_yaw"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
```

### `src/tasks/tracking/config/pm01/rl_cfg.py`
```python
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def engineai_pm01_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pm01_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
  )
```

## 5. Training

```bash
# Prepare motion data first
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/pm01/your_motion.csv \
  --output-name your_motion.npz \
  --input-fps 30 --output-fps 100

# Train
python scripts/train.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/your_motion.npz \
  --env.scene.num-envs 4096

# Play
python scripts/play.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/your_motion.npz \
  --checkpoint-file logs/rsl_rl/pm01_tracking/<run>/model_XXXXX.pt
```

## 6. Expected Training Progression

Based on EngineAI's PM01 dance experiments (EXP-09):

| Stage | Steps | Reward | Episode Length | What's happening |
|---|---|---|---|---|
| Early | 0-500 | 0-5 | 100-150 | Robot falls, learning to balance |
| Mid | 500-2000 | 5-15 | 150-300 | Rough motion tracking, stays alive longer |
| Late | 2000-10000 | 15-50 | 500-1000 | Motion becomes recognizable |
| Converged | 10000+ | 50-75+ | 1000+ | Clean motion, looping through reference |

## 7. Deployment

The `MotionTrackingOnPolicyRunner` automatically exports a bundled ONNX that contains both the policy network AND the motion reference data. The output file is named `{experiment_name}.onnx` and takes two inputs:
- `obs`: observation tensor
- `time_step`: current frame index in the motion

It returns:
- `actions`: joint position targets
- `joint_pos`, `joint_vel`: reference motion joint states
- `body_pos_w`, `body_quat_w`: reference body poses
- `body_lin_vel_w`, `body_ang_vel_w`: reference body velocities

This means **no separate motion file is needed on the robot** — everything is in one ONNX/MNN file.

## 8. Lessons Learned from EngineAI AMP Experiments

1. **AMP cold-start problem**: When robot starts far from reference motion (e.g., standing vs dancing), the discriminator becomes a perfect classifier in 2-3 iterations and locks up permanently (policy_pred = -1.0).

2. **DeepMimic works better**: Direct pose tracking with exponential error rewards is more reliable. EngineAI achieved episode_length 1 → 1495 and reward 0 → 74 using pure DeepMimic.

3. **Remove competing penalties**: Orientation and height penalties fight against dance poses that are naturally tilted/dynamic. Remove them for motion tracking.

4. **RSI sync is critical**: Random Start Initialization must sync the robot's starting frame with the reference time. Misalignment causes tracking reward to collapse.

5. **Scale pose tracking reward high**: EngineAI used `dof_ref_pos_diff = 20.0` (dominant) to make pose matching the primary learning signal.
