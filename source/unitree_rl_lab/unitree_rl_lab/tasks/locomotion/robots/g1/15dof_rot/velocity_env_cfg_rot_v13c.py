"""15-DOF rotation config V13c — "Full Parameter Recalibration".

Three-factor experiment: reduce penalties + boost tracking signal.
Tests whether strengthening the tracking reward side provides additional
benefit on top of penalty reduction.

Changes from V13b:
  - track_lin_vel_xy weight: 1.0 → 1.5 (boost signal, not noise)
  - track_ang_vel_z weight: 1.0 → 1.5 (match lin_vel boost)
  Everything else from V13b (action_rate = -0.02, feet_slide = -0.1)
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v13b import (
    RewardsCfg as V13bRewardsCfg,
    RobotEnvCfg as V13bEnvCfg,
)


@configclass
class RewardsCfg(V13bRewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class RobotEnvCfg(V13bEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
