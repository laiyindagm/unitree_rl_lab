"""15-DOF rotation config V11a -- "Adaptive Sigma + Bucketed".

Fix the dead zone from the REWARD side. The root cause:
  exp(-error^2/s^2) with fixed s=0.5 gives standing_reward=0.85 at cmd=0.2.
  Marginal tracking gain (0.15) < action_rate cost (0.33) -> agent stays still.

Fix: s = max(s_min, s_scale * |cmd|). At low speed s shrinks -> standing
reward drops -> marginal gain exceeds action cost.

  cmd=0.2: s=0.15 -> standing=0.17, marginal=0.83 >> action cost
  cmd=0.5: s=0.50 -> same as original (no degradation)

Key difference from V8b (which deadlocked): V8b used old lin_vel_cmd_levels
curriculum that became too hard with adaptive s. V11a uses bucketed curriculum
where 50% standing/rotating envs contribute tracking=1.0, preventing deadlock.

Base: V10d (bucketed curriculum + per-joint waist + V6c DR)
Changes: replace both tracking rewards with adaptive-sigma versions
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v10d import (
    CommandsCfg,
    CurriculumCfg,
    RewardsCfg as V10dRewardsCfg,
    RobotEnvCfg as V10dEnvCfg,
)


@configclass
class RewardsCfg(V10dRewardsCfg):
    """V10d rewards with adaptive-sigma tracking."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_adaptive_sigma,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sigma_min": 0.15,
            "sigma_scale": 0.5,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_adaptive_sigma,
        weight=0.5 * 2.6,
        params={
            "command_name": "base_velocity",
            "sigma_min": 0.15,
            "sigma_scale": 0.5,
        },
    )


@configclass
class RobotEnvCfg(V10dEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
