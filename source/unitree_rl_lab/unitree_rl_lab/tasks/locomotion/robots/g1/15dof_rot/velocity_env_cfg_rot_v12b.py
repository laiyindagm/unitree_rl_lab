"""15-DOF rotation config V12b -- "Scheduled Sigma + Scaled Action Rate".

Combines V12a (scheduled sigma annealing) with V11b (velocity-scaled action rate)
for maximum dead zone elimination from BOTH reward and penalty sides.

Dead zone math at Phase 3 (σ=0.3, scaled action_rate min_scale=0.3):
  cmd=0.3: marginal = 0.632, cost = 0.15 × 2.73 × 0.3 = 0.123
  → 0.632 >> 0.123 → strong tracking

  cmd=0.15: marginal = 0.221, cost = 0.123
  → 0.221 > 0.123 → even cmd=0.15 breaks dead zone!

  Theoretical dead zone threshold: |cmd| ≈ 0.11 (down from ~0.4)

Even in Phase 1 (σ=0.5, scaled action_rate):
  cmd=0.3: marginal = 0.302, cost = 0.123
  → 0.302 > 0.123 → dead zone broken in Phase 1 already!

Base: V12a (scheduled sigma + V10d bucketed curriculum + per-joint waist)
Changes: replace action_rate_l2 with action_rate_scaled_by_vel (same as V11b)
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v12a import (
    CommandsCfg,
    CurriculumCfg,
    RewardsCfg as V12aRewardsCfg,
    RobotEnvCfg as V12aEnvCfg,
)


@configclass
class RewardsCfg(V12aRewardsCfg):
    """V12a rewards + velocity-scaled action rate (from V11b)."""

    action_rate = RewTerm(
        func=mdp.action_rate_scaled_by_vel,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "min_scale": 0.3,
        },
    )


@configclass
class RobotEnvCfg(V12aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
