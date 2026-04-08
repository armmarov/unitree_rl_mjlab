# BeyondMimic Training Guide

Complete guide to train a robot to perform specific movements using motion imitation (BeyondMimic).

## 1. Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MOTION DATA SOURCE                          │
│  AMASS (mocap dataset)  or  Mixamo (animations)  or  Custom MoCap  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ .npz (SMPL-X) or .bvh
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 1: RETARGETING (GMR)                      │
│  Maps human skeleton motion → robot joint angles                   │
│  Input:  SMPL-X .npz (human motion)                                │
│  Output: .pkl (robot joint angles + root position/rotation)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ .pkl
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STEP 2: PKL → CSV CONVERSION                      │
│  Converts GMR pickle to CSV format for BeyondMimic                 │
│  Input:  .pkl                                                      │
│  Output: .csv (root_pos, root_quat, joint_angles per frame)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ .csv
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STEP 3: CSV → NPZ CONVERSION                      │
│  Runs forward kinematics in MuJoCo to compute body poses           │
│  Input:  .csv                                                      │
│  Output: .npz (joint pos/vel + all body positions/orientations)    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ .npz
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 4: TRAINING (mjlab)                       │
│  PPO policy learns to track reference motion frame-by-frame        │
│  Input:  .npz (motion reference)                                   │
│  Output: model_XXXXX.pt, policy.onnx, policy.mnn                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ .pt / .onnx / .mnn
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 5: PLAY / DEPLOY                          │
│  Visualize in MuJoCo or deploy to real robot                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Git Repositories

| Repository | Purpose | Link |
|---|---|---|
| **GMR** | Retarget human motion to robot joints | https://github.com/YanjieZe/GMR |
| **unitree_rl_mjlab** | Training and simulation (this repo) | https://github.com/armmarov/unitree_rl_mjlab |
| **BeyondMimic (reference)** | Original paper implementation (Isaac Lab) | https://github.com/HybridRobotics/whole_body_tracking |

### GMR Setup

```bash
git clone https://github.com/YanjieZe/GMR
cd GMR
pip install -e .
```

## 3. Datasets to Download

### Required: SMPL-X Body Models (human skeleton definition)

| What | Where | Format | Notes |
|---|---|---|---|
| SMPL-X body models | https://smpl-x.is.tue.mpg.de/ | .pkl | Free, requires registration. Place in `GMR/assets/body_models/smplx/` |

Expected structure after download:
```
GMR/assets/body_models/smplx/
├── SMPLX_MALE.pkl
├── SMPLX_FEMALE.pkl
└── SMPLX_NEUTRAL.pkl
```

### Required: Motion Capture Data (the actual movements)

| Dataset | Where | Best for | Format |
|---|---|---|---|
| **AMASS** | https://amass.is.tue.mpg.de/ | Largest variety — walking, dance, sports, martial arts | SMPL-X .npz |
| **KungfuBot (PBHC)** | https://github.com/TeleHuman/PBHC | Kicks, punches, martial arts, dance | SMPL .pkl |
| **LAFAN1** | https://github.com/ubisoft/ubisoft-laforge-animation-dataset | Locomotion, acrobatics | .bvh |

**AMASS sub-datasets by motion type:**

| Sub-dataset | Good for |
|---|---|
| CMU | General locomotion, dance, sports |
| ACCAD | Dance, martial arts, punching |
| BMLrub | Walking, running, everyday actions |
| HDM05 | Dance, gymnastics |
| KIT | Manipulation, interaction |

**KungfuBot example motions (pre-included):**

| Motion | Frames | Description |
|---|---|---|
| Hooks_punch | 175 | Hook punches |
| Roundhouse_kick | 158 | Spinning roundhouse kick |
| Side_kick | ~180 | Double side kick |
| Bruce_Lee_pose | ? | Wushu stance |
| Charleston_dance | ? | Dance |
| Horse-stance_punch | ? | Horse stance + punch combo |

Download the **SMPL-X G** (gender neutral) version from AMASS.

After download:
```bash
mkdir -p GMR/assets/motions/amass
tar -xjf ACCAD.tar.bz2 -C GMR/assets/motions/amass/
```

Expected structure:
```
GMR/assets/motions/amass/ACCAD/
├── Male2MartialArtsPunches_c3d/
│   ├── E3_-__cross_left_stageii.npz
│   ├── E4_-__jab_right_stageii.npz
│   └── ...
├── Female1Walking_c3d/
│   └── ...
└── ...
```

## 4. Commands

### Full Pipeline Example (PM01 Punch)

```bash
# 1. Retarget human motion to PM01
cd /path/to/GMR
python scripts/smplx_to_robot.py \
  --robot engineai_pm01 \
  --smplx_file assets/motions/amass/ACCAD/Male2MartialArtsPunches_c3d/E3_-__cross_left_stageii.npz \
  --save_path output/pm01_punch.pkl

# 2. Convert pkl to CSV
python scripts/batch_gmr_pkl_to_csv.py --folder output/

# 3. Fix orientation (face forward) and height
cd /path/to/unitree_rl_mjlab
```

#### Using KungfuBot (PBHC) Dataset

KungfuBot motions are in SMPL pkl format and need to be converted to SMPL-X npz first:

```bash
# 0. Convert KungfuBot pkl to SMPL-X npz
cd /path/to/unitree_rl_mjlab
mkdir -p /path/to/GMR/assets/motions/kungfubot
python scripts/kungfubot_to_smplx.py \
  /path/to/dataset/PBHC/example/motion_data/Roundhouse_kick.pkl \
  --output-npz /path/to/GMR/assets/motions/kungfubot/Roundhouse_kick_smplx.npz

# 1. Retarget to PM01 via GMR
cd /path/to/GMR
python scripts/smplx_to_robot.py \
  --robot engineai_pm01 \
  --smplx_file assets/motions/kungfubot/Roundhouse_kick_smplx.npz \
  --save_path output/pm01_roundhouse_kick.pkl

# 2. Convert pkl to CSV
python scripts/batch_gmr_pkl_to_csv.py --folder output/

# 3. Fix orientation (face forward) and height
cd /path/to/unitree_rl_mjlab
python scripts/fix_motion_orientation.py \
  /path/to/GMR/output/csv/pm01_punch.csv \
  --output-csv /path/to/GMR/output/csv/pm01_punch_fixed.csv \
  --height-offset 0.25

# 4. Prepend standing-to-motion transition
python scripts/prepend_standing.py \
  /path/to/GMR/output/csv/pm01_punch_fixed.csv \
  --output-csv /path/to/GMR/output/csv/pm01_punch_final.csv

# 5. Convert to NPZ
python scripts/csv_to_npz.py \
  --robot pm01 \
  --input-file /path/to/GMR/output/csv/pm01_punch_final.csv \
  --output-name pm01_punch.npz \
  --input-fps 30 --output-fps 100

# 6. Visualize (check motion looks correct before training)
python scripts/play.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/pm01_punch.npz \
  --agent zero --no-terminations True

# 7. Train
python scripts/train.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/pm01_punch.npz \
  --env.scene.num-envs 4096

# 8. Play trained policy
python scripts/play.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/pm01_punch.npz \
  --checkpoint-file logs/rsl_rl/pm01_tracking/<run>/model_XXXXX.pt

# 9. Export release
python scripts/export_release.py pm01_tracking <run_name>
```

---

### Detailed Steps

### Step 1: Retarget (GMR)

```bash
cd /path/to/GMR

# From AMASS (SMPL-X format)
python scripts/smplx_to_robot.py \
  --robot engineai_pm01 \
  --smplx_file assets/motions/amass/ACCAD/Male2MartialArtsPunches_c3d/E3_-__cross_left_stageii.npz \
  --save_path output/pm01_punch.pkl

# From LAFAN1 (BVH format)
python scripts/bvh_to_robot.py \
  --robot engineai_pm01 \
  --bvh_file /path/to/motion.bvh \
  --save_path output/pm01_motion.pkl
```

This opens a MuJoCo viewer showing the retargeted motion on the robot. Close the viewer when done.

**Supported robots:** `unitree_g1`, `engineai_pm01`, `unitree_h1`, `unitree_h1_2`, and others.

### Step 2: PKL → CSV

```bash
cd /path/to/GMR
python scripts/batch_gmr_pkl_to_csv.py --folder output/
```

Output: `output/csv/pm01_punch.csv`

### Step 3: Fix Orientation and Height

GMR retargeting may produce motions where the robot faces the wrong direction or the root height doesn't match the robot's standing height. Use `fix_motion_orientation.py` to correct this.

```bash
cd /path/to/unitree_rl_mjlab
python scripts/fix_motion_orientation.py \
  /path/to/GMR/output/csv/pm01_punch.csv \
  --output-csv /path/to/GMR/output/csv/pm01_punch_fixed.csv \
  --target-yaw 0 \
  --height-offset 0.25
```

Options:
- `--target-yaw` — target facing direction in degrees (default: 0 = forward)
- `--height-offset` — add to root Z height in meters (PM01 standing is 0.92m, GMR may output ~0.67m, so use ~0.25)

### Step 4: Prepend Standing Transition

The robot needs a smooth transition from its standing pose to the first frame of the motion. Without this, the robot would jerk into the motion pose instantly.

```bash
python scripts/prepend_standing.py \
  /path/to/GMR/output/csv/pm01_punch_fixed.csv \
  --output-csv /path/to/GMR/output/csv/pm01_punch_final.csv \
  --hold-seconds 0.5 \
  --transition-seconds 1.5
```

Options:
- `--hold-seconds` — how long to hold the standing pose before transitioning (default: 0.5s)
- `--transition-seconds` — duration of smooth interpolation to first motion frame (default: 1.5s)
- `--fps` — frame rate of the CSV (default: 30)

The script automatically detects and skips bad initial frames from GMR (large position jumps).

### Step 5: CSV → NPZ

```bash
cd /path/to/unitree_rl_mjlab

# For PM01 (100Hz control)
python scripts/csv_to_npz.py \
  --robot pm01 \
  --input-file /path/to/GMR/output/csv/pm01_punch.csv \
  --output-name pm01_punch.npz \
  --input-fps 30 \
  --output-fps 100

# For G1 (50Hz control)
python scripts/csv_to_npz.py \
  --robot g1 \
  --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 \
  --output-fps 50
```

Output: `src/assets/motions/pm01/pm01_punch.npz`

### Step 4: Train

```bash
cd /path/to/unitree_rl_mjlab

# PM01
python scripts/train.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/pm01_punch.npz \
  --env.scene.num-envs 4096

# G1
python scripts/train.py Unitree-G1-Tracking \
  --motion-file src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs 4096
```

### Step 5: Play

```bash
# PM01
python scripts/play.py EngineAI-PM01-Tracking \
  --motion-file src/assets/motions/pm01/pm01_punch.npz \
  --checkpoint-file logs/rsl_rl/pm01_tracking/<run>/model_XXXXX.pt

# G1
python scripts/play.py Unitree-G1-Tracking \
  --motion-file src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_XXXXX.pt
```

### Export for Deployment

```bash
python scripts/export_release.py <run_name>
```

## 5. File Formats

### SMPL-X .npz (Input — from AMASS)

Human motion in parametric body model format.

| Key | Shape | Description |
|---|---|---|
| `trans` | (T, 3) | Root translation per frame |
| `root_orient` | (T, 3) | Root orientation (axis-angle) |
| `pose_body` | (T, 63) | Body joint rotations (21 joints x 3 axis-angle) |
| `pose_hand` | (T, 90) | Hand joint rotations |
| `betas` | (16,) | Body shape parameters |
| `gender` | str | "male", "female", or "neutral" |
| `surface_model_type` | str | Must be "smplx" |
| `mocap_frame_rate` | float | Recording frame rate |

### GMR .pkl (Intermediate — from retargeting)

Robot-specific motion after inverse kinematics retargeting.

| Key | Shape | Description |
|---|---|---|
| `root_pos` | (T, 3) | Robot root position (x, y, z) in meters |
| `root_rot` | (T, 4) | Robot root quaternion (x, y, z, w) |
| `dof_pos` | (T, N_joints) | Joint angles in radians (24 for PM01, 29 for G1) |
| `fps` | float | Frame rate of the motion |

### CSV (Intermediate — for csv_to_npz)

Flat text file, one row per frame, comma-separated.

```
pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, joint_0, joint_1, ..., joint_N
```

| Column | Count | Description |
|---|---|---|
| 0-2 | 3 | Root position (x, y, z) in meters |
| 3-6 | 4 | Root quaternion (x, y, z, w) |
| 7+ | N_joints | Joint angles in radians (24 for PM01, 29 for G1) |

Total columns: 7 + N_joints (31 for PM01, 36 for G1)

### NPZ (Final — for training)

Complete motion reference with all body kinematics computed via forward kinematics in MuJoCo.

| Key | Shape | Description |
|---|---|---|
| `fps` | (1,) | Output frame rate (100 for PM01, 50 for G1) |
| `joint_pos` | (T, N_joints) | Joint positions in radians per frame |
| `joint_vel` | (T, N_joints) | Joint velocities in rad/s per frame |
| `body_pos_w` | (T, N_bodies, 3) | World positions of all tracked bodies |
| `body_quat_w` | (T, N_bodies, 4) | World quaternions (w, x, y, z) of all tracked bodies |
| `body_lin_vel_w` | (T, N_bodies, 3) | World linear velocities of all tracked bodies |
| `body_ang_vel_w` | (T, N_bodies, 3) | World angular velocities of all tracked bodies |

This file contains everything needed for the training reward computation — the policy is rewarded for matching these body poses frame-by-frame.

### model_XXXXX.pt (Output — checkpoint)

PyTorch checkpoint containing:
- Actor network weights + observation normalizer stats
- Critic network weights
- Optimizer state
- Training iteration number

### policy.onnx (Output — for deployment)

The `MotionTrackingOnPolicyRunner` exports a bundled ONNX that contains both the policy network AND the motion reference data. No separate motion file needed on the robot.

Inputs: `obs` (observation tensor), `time_step` (current frame index)
Outputs: `actions` (joint targets), `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w` (reference motion state)
