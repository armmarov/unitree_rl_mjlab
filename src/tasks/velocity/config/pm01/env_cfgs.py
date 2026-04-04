"""EngineAI PM01 velocity environment configurations."""

from src.assets.robots import (
  PM01_ACTION_SCALE,
  get_pm01_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import root_height_below_minimum
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def engineai_pm01_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PM01 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}

  # Set raycast sensor frame to PM01 base link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "link_base"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(link_ankle_roll_l|link_ankle_roll_r)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="link_base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="link_base", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM01_ACTION_SCALE

  cfg.viewer.body_name = "link_torso_yaw"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.05

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("link_torso_yaw",)

  # PM01 pose standard deviations.
  # Legs get the most freedom for natural stride.
  # Hip roll/yaw tighter to prevent lateral sway.
  # Ankle roll very tight for balance.
  # Upper body (waist, arms, head) kept tight since legs-focused locomotion.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r"j0[03]_.*|j0[69]_.*": 0.5,   # hip_pitch, knee
    r"j0[17]_.*": 0.15,             # hip_roll
    r"j0[28]_.*": 0.15,             # hip_yaw
    r"j04_.*|j10_.*": 0.15,         # ankle_pitch
    r"j05_.*|j11_.*": 0.1,          # ankle_roll
    # Waist.
    r"j12_.*": 0.15,
    # Arms.
    r"j1[38]_.*": 0.15,             # shoulder_pitch
    r"j14_.*|j19_.*": 0.1,          # shoulder_roll
    r"j15_.*|j20_.*": 0.1,          # shoulder_yaw
    r"j1[67]_.*|j2[12]_.*": 0.1,    # elbow
    # Head.
    r"j23_.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r"j0[03]_.*|j0[69]_.*": 0.5,   # hip_pitch, knee
    r"j0[17]_.*": 0.25,             # hip_roll
    r"j0[28]_.*": 0.25,             # hip_yaw
    r"j04_.*|j10_.*": 0.25,         # ankle_pitch
    r"j05_.*|j11_.*": 0.1,          # ankle_roll
    # Waist.
    r"j12_.*": 0.25,
    # Arms.
    r"j1[38]_.*": 0.25,             # shoulder_pitch
    r"j14_.*|j19_.*": 0.1,          # shoulder_roll
    r"j15_.*|j20_.*": 0.1,          # shoulder_yaw
    r"j1[67]_.*|j2[12]_.*": 0.1,    # elbow
    # Head.
    r"j23_.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("link_torso_yaw",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("link_torso_yaw",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["target_height"] = 0.15  # PM01 clearance target from rl_lab.
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # PM01 gait period from rl_lab: 0.8s (vs default 0.6s).
  cfg.rewards["foot_gait"].params["period"] = 0.8

  # Terminate if robot crouches too low (standing height is 0.92m).
  cfg.terminations["base_height"] = TerminationTermCfg(
    func=root_height_below_minimum,
    params={"minimum_height": 0.55},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def engineai_pm01_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PM01 flat terrain velocity configuration."""
  cfg = engineai_pm01_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
