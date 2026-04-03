from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  engineai_pm01_flat_env_cfg,
  engineai_pm01_rough_env_cfg,
)
from .rl_cfg import engineai_pm01_ppo_runner_cfg

register_mjlab_task(
  task_id="EngineAI-PM01-Rough",
  env_cfg=engineai_pm01_rough_env_cfg(),
  play_env_cfg=engineai_pm01_rough_env_cfg(play=True),
  rl_cfg=engineai_pm01_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="EngineAI-PM01-Flat",
  env_cfg=engineai_pm01_flat_env_cfg(),
  play_env_cfg=engineai_pm01_flat_env_cfg(play=True),
  rl_cfg=engineai_pm01_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
