"""EngineAI PM01 24-DOF constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PM01_XML: Path = (
  SRC_PATH / "assets" / "robots" / "engineai_pm01" / "xmls" / "pm01.xml"
)
assert PM01_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PM01_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PM01_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
# Values from rl_lab PM01 24-DOF config.
##

PM01_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j00_hip_pitch_l",
    "j06_hip_pitch_r",
  ),
  stiffness=70.0,
  damping=7.0,
  effort_limit=164.0,
  armature=0.0453,
)
PM01_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j01_hip_roll_l",
    "j07_hip_roll_r",
  ),
  stiffness=50.0,
  damping=5.0,
  effort_limit=164.0,
  armature=0.0453,
)
PM01_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j02_hip_yaw_l",
    "j08_hip_yaw_r",
  ),
  stiffness=50.0,
  damping=5.0,
  effort_limit=52.0,
  armature=0.0067,
)
PM01_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j03_knee_pitch_l",
    "j09_knee_pitch_r",
  ),
  stiffness=70.0,
  damping=7.0,
  effort_limit=164.0,
  armature=0.0453,
)
PM01_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j04_ankle_pitch_l",
    "j05_ankle_roll_l",
    "j10_ankle_pitch_r",
    "j11_ankle_roll_r",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=52.0,
  armature=0.0067,
)
PM01_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j12_waist_yaw",
  ),
  stiffness=200.0,
  damping=5.0,
  effort_limit=52.0,
  armature=0.0067,
)
PM01_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j13_shoulder_pitch_l",
    "j14_shoulder_roll_l",
    "j15_shoulder_yaw_l",
    "j18_shoulder_pitch_r",
    "j19_shoulder_roll_r",
    "j20_shoulder_yaw_r",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=52.0,
  armature=0.0067,
)
PM01_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j16_elbow_pitch_l",
    "j17_elbow_yaw_l",
    "j21_elbow_pitch_r",
    "j22_elbow_yaw_r",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=52.0,
  armature=0.0067,
)
PM01_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "j23_head_yaw",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=52.0,
  armature=0.0067,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.92),
  joint_pos={
    # Left leg.
    "j00_hip_pitch_l": -0.24,
    "j01_hip_roll_l": 0.0,
    "j02_hip_yaw_l": 0.0,
    "j03_knee_pitch_l": 0.48,
    "j04_ankle_pitch_l": -0.24,
    "j05_ankle_roll_l": 0.0,
    # Right leg.
    "j06_hip_pitch_r": -0.24,
    "j07_hip_roll_r": 0.0,
    "j08_hip_yaw_r": 0.0,
    "j09_knee_pitch_r": 0.48,
    "j10_ankle_pitch_r": -0.24,
    "j11_ankle_roll_r": 0.0,
    # Waist.
    "j12_waist_yaw": 0.0,
    # Left arm.
    "j13_shoulder_pitch_l": 0.35,
    "j14_shoulder_roll_l": 0.25,
    "j15_shoulder_yaw_l": 0.0,
    "j16_elbow_pitch_l": -0.7,
    "j17_elbow_yaw_l": 0.0,
    # Right arm.
    "j18_shoulder_pitch_r": 0.35,
    "j19_shoulder_roll_r": -0.25,
    "j20_shoulder_yaw_r": 0.0,
    "j21_elbow_pitch_r": -0.7,
    "j22_elbow_yaw_r": 0.0,
    # Head.
    "j23_head_yaw": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-4]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-4]_collision$": 1},
  friction={r"^(left|right)_foot[1-4]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-4]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-4]_collision$": 1},
  friction={r"^(left|right)_foot[1-4]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-4]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

PM01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PM01_ACTUATOR_HIP_PITCH,
    PM01_ACTUATOR_HIP_ROLL,
    PM01_ACTUATOR_HIP_YAW,
    PM01_ACTUATOR_KNEE,
    PM01_ACTUATOR_ANKLE,
    PM01_ACTUATOR_WAIST,
    PM01_ACTUATOR_SHOULDER,
    PM01_ACTUATOR_ELBOW,
    PM01_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_pm01_robot_cfg() -> EntityCfg:
  """Get a fresh PM01 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PM01_ARTICULATION,
  )


PM01_ACTION_SCALE: dict[str, float] = {}
for a in PM01_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    PM01_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pm01_robot_cfg())

  viewer.launch(robot.spec.compile())
