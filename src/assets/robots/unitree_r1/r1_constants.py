"""Unitree R1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

R1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_r1" / "xmls" / "r1.xml"
)
assert R1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, R1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(R1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

R1_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
    ".*_hip_yaw.*",
    ".*_knee.*",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)
R1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
)
R1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_.*",
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)
R1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)
R1_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=33.0,
  armature=0.01,
)


##
# Delayed actuator wrappers, for deploy-loop latency robustness.
##
# R1_ACTUATOR_* above stay untouched (R1_ACTION_SCALE below asserts
# isinstance(a, BuiltinPositionActuatorCfg) and reads .effort_limit/.stiffness
# directly, which DelayedActuatorCfg does not expose). These _DELAYED variants
# wrap them for use in R1_ARTICULATION instead.
#
# Physics timestep is 0.005s, decimation 4 -> 50Hz control (matches
# deploy/robots/r1/config/policy/velocity/v0/params/deploy.yaml's step_dt=0.02).
# So 1 lag unit = 5ms. delay_min_lag/delay_max_lag below (0-8 steps = 0-40ms)
# are not a measured figure -- we have not instrumented our actual
# r1_ctrl/unitree_mujoco DDS round-trip latency. Widened from an initial 0-4
# (0-20ms) guess per r1-rl's review: RL domain-randomization practice favors
# training for a wider margin than a naive latency point-estimate, since
# deploy-time jitter (thread scheduling, etc.) tends to spike above it.
# delay_update_period=20 (~0.1s) resamples periodically rather than either
# fully-fixed-per-episode or independent per-physics-step white noise.
# Reviewed by r1-rl 2026-09-08; R1_ARTICULATION scoping issue they flagged
# (this used to default-apply to both Velocity and Tracking) is fixed below --
# see R1_ARTICULATION_DELAYED.
R1_ACTUATOR_LEG_DELAYED = DelayedActuatorCfg(
  base_cfg=R1_ACTUATOR_LEG,
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=8,
  delay_hold_prob=0.0,
  delay_update_period=20,
  delay_per_env_phase=True,
)
R1_ACTUATOR_ANKLE_DELAYED = DelayedActuatorCfg(
  base_cfg=R1_ACTUATOR_ANKLE,
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=8,
  delay_hold_prob=0.0,
  delay_update_period=20,
  delay_per_env_phase=True,
)
R1_ACTUATOR_WAIST_DELAYED = DelayedActuatorCfg(
  base_cfg=R1_ACTUATOR_WAIST,
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=8,
  delay_hold_prob=0.0,
  delay_update_period=20,
  delay_per_env_phase=True,
)
R1_ACTUATOR_ARM_DELAYED = DelayedActuatorCfg(
  base_cfg=R1_ACTUATOR_ARM,
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=8,
  delay_hold_prob=0.0,
  delay_update_period=20,
  delay_per_env_phase=True,
)
R1_ACTUATOR_WRIST_DELAYED = DelayedActuatorCfg(
  base_cfg=R1_ACTUATOR_WRIST,
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=8,
  delay_hold_prob=0.0,
  delay_update_period=20,
  delay_per_env_phase=True,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

R1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    R1_ACTUATOR_LEG,
    R1_ACTUATOR_ANKLE,
    R1_ACTUATOR_WAIST,
    R1_ACTUATOR_ARM,
    R1_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Delay-robust variant, opt-in only (get_r1_robot_cfg(delayed=True)). R1_ARTICULATION
# above stays the plain version since get_r1_robot_cfg() is shared by both the
# velocity and tracking tasks (see src/tasks/{velocity,tracking}/config/r1/env_cfgs.py) --
# defaulting it to delayed would have silently changed Tracking's actuator model too.
R1_ARTICULATION_DELAYED = EntityArticulationInfoCfg(
  actuators=(
    R1_ACTUATOR_LEG_DELAYED,
    R1_ACTUATOR_ANKLE_DELAYED,
    R1_ACTUATOR_WAIST_DELAYED,
    R1_ACTUATOR_ARM_DELAYED,
    R1_ACTUATOR_WRIST_DELAYED,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_r1_robot_cfg(delayed: bool = False) -> EntityCfg:
  """Get a fresh R1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    delayed: use R1_ARTICULATION_DELAYED (randomized actuator delay, for
      deploy-loop latency robustness) instead of the plain R1_ARTICULATION.
      Defaults to False so existing callers (e.g. the tracking task) are
      unaffected; pass True explicitly (e.g. from the velocity task) to opt in.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=R1_ARTICULATION_DELAYED if delayed else R1_ARTICULATION,
  )


R1_ACTION_SCALE: dict[str, float] = {}
for a in R1_ARTICULATION.actuators:
  base = a.base_cfg if isinstance(a, DelayedActuatorCfg) else a
  assert isinstance(base, BuiltinPositionActuatorCfg)
  e = base.effort_limit
  s = base.stiffness
  names = base.target_names_expr
  assert e is not None
  for n in names:
    R1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_r1_robot_cfg())

  viewer.launch(robot.spec.compile())
