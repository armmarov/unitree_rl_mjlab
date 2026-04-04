# R1 vs PM01 Training Comparison

Both robots train full-body velocity tracking using PPO with 24 actuated DOFs.

## Robot Specs

| Property | Unitree R1 | EngineAI PM01 |
|---|---|---|
| Total actuated DOFs | 24 | 24 |
| Legs | 12 (6 per leg) | 12 (6 per leg) |
| Waist | 1 (yaw) | 1 (yaw) |
| Arms | 10 (5 per arm) | 10 (5 per arm) |
| Head | 0 | 1 (yaw) |
| Wrist | 1 (roll per arm) | 0 |
| Base body | `torso_link` | `link_base` |
| Torso body | `torso_link` | `link_torso_yaw` |
| Foot sites | `left_foot`, `right_foot` | `left_foot`, `right_foot` |
| Foot collision geoms | 14 (7 per foot) | 8 (4 per foot) |
| Standing height | ~0.92m | ~0.92m |

## PPO Hyperparameters

| Parameter | R1 | PM01 | Notes |
|---|---|---|---|
| Network (actor/critic) | 512-256-128 | 512-256-128 | Same |
| Activation | ELU | ELU | Same |
| Obs normalization | Yes | Yes | Same |
| Std type | `scalar` | `log` | PM01 uses log to ensure std > 0 |
| Gamma | **0.99** | **0.97** | PM01 lower to prevent value loss instability |
| Lambda (GAE) | 0.95 | 0.95 | Same |
| Learning rate | 1e-3 | 1e-3 | Same |
| Schedule | adaptive | adaptive | Same |
| Desired KL | 0.01 | 0.01 | Same |
| Clip param | 0.2 | 0.2 | Same |
| Entropy coef | 0.01 | 0.01 | Same |
| Num learning epochs | 5 | 5 | Same |
| Num mini batches | 4 | 4 | Same |
| Max grad norm | 1.0 | 1.0 | Same |
| Num steps per env | 24 | 24 | Same |
| Max iterations | 10,001 | 50,001 | PM01 trains 5x longer |
| Save interval | 100 | 100 | Same |

## Key Differences

### 1. Discount Factor (gamma)
- **R1: 0.99** -> discounted return horizon ~100 steps
- **PM01: 0.97** -> discounted return horizon ~33 steps
- PM01 uses a lower gamma because its per-step reward tends to be higher, which can cause value loss instability with gamma=0.99.

### 2. Distribution Std Type
- **R1: `scalar`** -> fixed scalar std per action dimension
- **PM01: `log`** -> parameterizes log(std), ensuring std is always positive
- This is a stability choice from the PM01 rl_lab config.

### 3. Max Training Iterations
- **R1: 10,001**
- **PM01: 50,001**
- PM01 is configured for longer training runs.

### 4. Foot Collision
- **R1**: 7 collision geoms per foot (more detailed ground contact)
- **PM01**: 4 collision geoms per foot

### 5. Gait Period
- **R1**: 0.6s (default)
- **PM01**: 0.8s (slower gait cycle from rl_lab config)

### 6. Foot Clearance Target
- **R1**: default
- **PM01**: 0.15m (explicitly set)

## Actuator Tuning

### R1
| Group | Stiffness | Effort | Joints |
|---|---|---|---|
| Leg | 100.0 | 60.0 | hip_pitch/roll/yaw, knee |
| Ankle | 40.0 | 50.0 | ankle_pitch/roll |
| Waist | 100.0 | 60.0 | waist |
| Shoulder | 40.0 | 60.0 | shoulder_pitch/roll |
| Arm | 20.0 | 33.0 | shoulder_yaw, elbow, wrist |

### PM01
| Group | Stiffness | Effort | Joints |
|---|---|---|---|
| Hip pitch | 70.0 | 164.0 | hip_pitch L/R |
| Hip roll | 50.0 | 164.0 | hip_roll L/R |
| Hip yaw | 50.0 | 52.0 | hip_yaw L/R |
| Knee | 70.0 | 164.0 | knee_pitch L/R |
| Ankle | 20.0 | 52.0 | ankle_pitch/roll L/R |
| Waist | 200.0 | 52.0 | waist_yaw |
| Shoulder | 40.0 | 52.0 | shoulder_pitch/roll/yaw L/R |
| Elbow | 40.0 | 52.0 | elbow_pitch/yaw L/R |
| Head | 40.0 | 52.0 | head_yaw |

PM01 has much higher effort limits on legs (164 vs 60), meaning stronger leg motors. The action scale (0.25 * effort / stiffness) is therefore larger for PM01 legs, allowing bigger joint position changes per step.
