"""EngineAI PM01 flat tracking environment configurations."""

from src.assets.robots import PM01_ACTION_SCALE, get_pm01_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def engineai_pm01_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PM01 flat terrain tracking configuration."""
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
    "link_hip_pitch_l",
    "link_knee_pitch_l",
    "link_ankle_roll_l",
    "link_hip_pitch_r",
    "link_knee_pitch_r",
    "link_ankle_roll_r",
    "link_torso_yaw",
    "link_shoulder_pitch_l",
    "link_elbow_pitch_l",
    "link_elbow_yaw_l",
    "link_shoulder_pitch_r",
    "link_elbow_pitch_r",
    "link_elbow_yaw_r",
  )

  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
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
