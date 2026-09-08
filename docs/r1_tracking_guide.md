# Unitree R1: Training, Visualization, and Deployment Guide

This guide covers how to train, visualize, and (eventually) deploy RL policies for the Unitree R1 in this repo — both the pre-existing velocity/locomotion tasks and the new BeyondMimic-style motion-tracking tasks.

All R1 tracking work lives on the git branch **`r1-tracking`** (branched off `main`). That branch adds:

- `src/tasks/tracking/config/r1/` (`__init__.py`, `env_cfgs.py`, `rl_cfg.py`) — registers the two R1 tracking tasks
- `scripts/csv_to_npz.py` — modified with an `r1` robot branch
- `src/assets/motions/r1/standing.npz` — a dummy motion file (see the caveat below)

All commands assume you run from the repo root (`/home/armmarov/work/robot/unitree/unitree_rl_mjlab`) using the repo venv's python (`venv/bin/python`, which has CUDA torch installed).

## Available R1 tasks

Registered via `mjlab.tasks.registry`:

| Task | Type | Needs `--motion-file`? |
|------|------|------------------------|
| `Unitree-R1-Flat` | Velocity/locomotion (pre-existing) | No |
| `Unitree-R1-Rough` | Velocity/locomotion (pre-existing) | No |
| `Unitree-R1-Tracking` | Motion tracking (new) | Yes |
| `Unitree-R1-Tracking-No-State-Estimation` | Motion tracking (new) | Yes |

There are no state-estimation variants of the velocity tasks — only the tracking task has a `-No-State-Estimation` variant.

## Motion data: a synthetic placeholder today, real mocap now possible via GMR

The only motion file bundled here, `src/assets/motions/r1/standing.npz`, is a **synthetic single static standing pose** (R1's built-in `HOME_KEYFRAME`) baked through real forward kinematics. Its body/joint trajectories are real and correctly shaped, but the "motion" is just standing still — no locomotion, nothing interesting. It exists purely to validate that the tracking pipeline (env, rewards, PPO loop) runs end-to-end. A policy trained on it will just learn to hold a standing pose.

Real R1 motion capture is now possible: R1 support was added to GMR (the retargeting pipeline that maps human SMPL-X mocap onto a robot skeleton), on GMR's `r1-support` branch (`/home/armmarov/work/robot/GMR`). See `GMR/R1_RETARGETING_GUIDE.md` for the full pipeline — it retargets a human motion clip and lands a ready-to-train NPZ directly in `src/assets/motions/r1/` via `make retarget MJLAB_ROBOT=r1 ...`. Note that pipeline's mjlab-side step depends on `csv_to_npz.py`'s `--robot r1` support, which only exists on **this branch** (`r1-tracking`) — make sure this repo stays on `r1-tracking` when running it.

## Training

### Velocity (works fully today)

This is real, robot-relevant locomotion training:

```bash
venv/bin/python scripts/train.py Unitree-R1-Flat --env.scene.num-envs=4096
```

(Substitute `Unitree-R1-Rough` for rough-terrain training.)

### Tracking: smoke test (what you can do today)

Small and fast, just to confirm the tracking pipeline runs end-to-end:

```bash
venv/bin/python scripts/train.py Unitree-R1-Tracking \
  --motion-file src/assets/motions/r1/standing.npz \
  --env.scene.num-envs=64 \
  --agent.max-iterations=5 \
  --agent.save-interval=5
```

### Tracking: real run (once real motion data exists)

Same command, but drop the `--agent.max-iterations` / `--agent.save-interval` overrides (defaults are `max_iterations=30001` and `save_interval=500`, set in `src/tasks/tracking/config/r1/rl_cfg.py`) and use a realistic env count, e.g. `--env.scene.num-envs=4096`.

### Useful flags

- `--gpu-ids 0 1` — multi-GPU training
- `--agent.resume` — resume from the last checkpoint in the same experiment log dir

### Where outputs land

Logs and checkpoints go to `logs/rsl_rl/<experiment_name>/<timestamp>/`, where `experiment_name` is:

- `r1_velocity` for `Unitree-R1-Flat` / `Unitree-R1-Rough`
- `r1_tracking` for `Unitree-R1-Tracking` / `-No-State-Estimation`

Each run directory contains:

- `model_<iter>.pt` — checkpoints
- `policy.onnx` + `policy.onnx.data` — auto-exported bundled policy. For tracking tasks the ONNX embeds **both the policy and the motion reference**, so no separate motion file is needed on-device.
- `events.out.tfevents...` — TensorBoard event file
- `params/env.yaml`, `params/agent.yaml` — the exact configs used for the run

## Creating your own motion data (`scripts/csv_to_npz.py`)

To convert a motion CSV into a tracking-ready NPZ:

```bash
venv/bin/python scripts/csv_to_npz.py --robot r1 \
  --input-file <path/to/motion.csv> \
  --output-name <name> \
  --input-fps <fps> \
  --output-fps 50
```

Output is saved to `src/assets/motions/r1/<name>.npz`.

The script runs a real MuJoCo forward-kinematics simulation to bake exact body-frame poses/velocities, so the resulting NPZ is guaranteed schema-correct for the R1 tracking task.

**Input CSV format** — one row per frame, at least 2 rows (duration must be > 0), columns in order:

1. Base position xyz (3 columns)
2. Base orientation quaternion in **xyzw** order (4 columns)
3. The 24 R1 joint angles in radians, in this exact order (the `R1_JOINT_NAMES` list baked into `csv_to_npz.py`):

```
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint,
right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint,
waist_roll_joint, waist_yaw_joint,
left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint,
left_elbow_joint, left_wrist_roll_joint,
right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint,
right_elbow_joint, right_wrist_roll_joint
```

This is exactly how `standing.npz` was made: a CSV with 60 identical rows repeating R1's `HOME_KEYFRAME` standing pose (position `[0, 0, 0.76]`, identity quaternion `[0, 0, 0, 1]`, joint angles from `HOME_KEYFRAME` in `src/assets/robots/unitree_r1/r1_constants.py`), at 30 fps input / 50 fps output.

## Play / visualize a trained policy

`scripts/play.py` runs a checkpoint in the MuJoCo viewer.

Velocity:

```bash
venv/bin/python scripts/play.py Unitree-R1-Flat \
  --checkpoint_file=logs/rsl_rl/r1_velocity/<run_name>/model_<iter>.pt
```

Tracking:

```bash
venv/bin/python scripts/play.py Unitree-R1-Tracking \
  --checkpoint_file=logs/rsl_rl/r1_tracking/<run_name>/model_<iter>.pt \
  --motion-file src/assets/motions/r1/standing.npz
```

Useful flags:

- `--no-terminations` — disables episode terminations; good for just watching a motion play out without early resets
- `--video` — record a video instead of / alongside live viewing
- `--viewer native|viser|auto` — choose the viewer backend
- `--agent zero` or `--agent random` — run without a checkpoint at all; useful to sanity-check a motion file or env setup before any training

## Real-robot deployment

The two policy types are in very different states of readiness. Velocity deployment is wired up today; tracking deployment does not exist for R1 yet.

### Velocity policies — ready today

`deploy/robots/r1/` contains a full C++ control program: `CMakeLists.txt`, `main.cpp`, `src/State_RLBase.cpp`, `include/Types.h`, and `config/config.yaml`. The config defines FSM states `Passive` → `FixStand` → `Velocity` (type `RLBase`), with `Velocity.policy_dir: config/policy/velocity`.

Note: the expected policy location `deploy/robots/r1/config/policy/velocity/v0/exported/` does **not** yet contain a `policy.onnx` / `policy.onnx.data` — it's a placeholder waiting for a trained model.

Steps:

1. **Install prerequisites** (if not already done):
   - cyclonedds: https://github.com/eclipse-cyclonedds/cyclonedds.git
   - unitree_sdk2: https://github.com/unitreerobotics/unitree_sdk2.git

2. **Copy the trained policy** — from your run dir, copy `policy.onnx` and `policy.onnx.data` into:

   ```
   deploy/robots/r1/config/policy/velocity/v0/exported/
   ```

3. **Build the controller**:

   ```bash
   cd deploy/robots/r1 && mkdir build && cd build && cmake .. && make
   ```

4. **Prepare the robot**: power on into zero-torque mode, then press `L2 + R2` on the controller to enter debug mode (joint damping enabled).

5. **Network setup**: connect your PC to the robot via Ethernet. Configure your PC's interface as address `192.168.123.222`, netmask `255.255.255.0`. Find the Ethernet interface name with `ifconfig`.

6. **Simulate first (strongly recommended)** — verify safe behavior before touching a real robot. Build `unitree_mujoco`:

   ```bash
   cd simulate && mkdir build && cd build && cmake .. && make -j8
   ```

   Launch it (needs a gamepad connected; select the R1 robot in `simulate/config`):

   ```bash
   ./simulate/build/unitree_mujoco
   ```

   Then in another terminal, run the controller against the simulator over loopback:

   ```bash
   cd deploy/robots/r1/build && ./r1_ctrl --network=lo
   ```

7. **Run on the real robot** — only once verified in sim, using the real interface name from step 5 (e.g. `enp5s0`):

   ```bash
   ./r1_ctrl --network=<your_real_interface>
   ```

### Tracking policies — NOT wired up for R1 yet

Deploying a tracking/motion-imitation policy on a real R1 is **future work, not a today capability**.

G1 has a working example of what this looks like: `deploy/robots/g1/` has an extra FSM state type, `Mimic`, implemented in `src/State_Mimic.cpp` / `include/State_Mimic.h`, with per-motion config entries like:

```yaml
Mimic_Dance1_subject2:
  id: 5
  type: Mimic
...
Mimic_Dance1_subject2:
  transitions:
    Passive: LT + B.on_pressed
    Velocity: RT + A.on_pressed
  motion_file: config/policy/mimic/dance1_subject2/params/dance1_subject2.npz
  policy_dir: config/policy/mimic/dance1_subject2/
  time_start: 0.0
  time_end: 1000.0
```

and a per-motion folder layout:

```
deploy/robots/g1/config/policy/mimic/dance1_subject2/
├── exported/
│   ├── policy.onnx
│   └── policy.onnx.data
└── params/
    ├── deploy.yaml
    └── dance1_subject2.npz
```

R1 has **none** of this — `deploy/robots/r1/` contains no `State_Mimic` files, no `Mimic` FSM state, and no `config/policy/tracking/` folder.

Closing the gap would require:

1. Porting `State_Mimic.cpp` / `State_Mimic.h` from `deploy/robots/g1/{src,include}` into `deploy/robots/r1/{src,include}`, adapting them to R1's joint layout and count (24 DOF vs G1's).
2. Adding a `Mimic_<motion_name>` FSM entry to `deploy/robots/r1/config/config.yaml` following the G1 pattern above.
3. Creating `deploy/robots/r1/config/policy/tracking/<motion_name>/{exported/{policy.onnx,policy.onnx.data}, params/{deploy.yaml,<motion_name>.npz}}`.
4. Rebuilding via cmake/make.

This is real, non-trivial C++ engineering work that has not been started — it is not a copy-paste job. Combined with the missing GMR retargeting support for R1, both the motion-data side and the deployment side of R1 tracking are open items.
