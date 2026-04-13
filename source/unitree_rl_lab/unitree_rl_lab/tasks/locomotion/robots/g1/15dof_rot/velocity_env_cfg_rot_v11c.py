"""15-DOF rotation config V11c -- "Adaptive Sigma + Scaled Action Rate".

Fix the dead zone from BOTH sides simultaneously. Combines V11a and V11b:
  1. Adaptive sigma tracking: marginal gain at cmd=0.2 jumps from 0.15 to 0.63
  2. Scaled action rate: penalty at low speed drops from 0.33 to ~0.12

Combined effect -- theoretical dead zone threshold drops to ~0.05:
  cmd=0.1: adaptive marginal=0.36, scaled cost=0.12 -> net +0.24 (responds!)
  cmd=0.2: adaptive marginal=0.63, scaled cost=0.12 -> net +0.51

Maximum dead zone elimination. Risk: two changes at once make failure
attribution harder, but mathematically this is the optimal configuration.

Base: V11a (adaptive sigma + V10d)
Changes: also replace action_rate with scaled version (from V11b)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v11a import (
    RewardsCfg as V11aRewardsCfg,
    RobotEnvCfg as V11aEnvCfg,
)


@configclass
class RewardsCfg(V11aRewardsCfg):
    """V11a rewards (adaptive sigma) + scaled action rate."""

    action_rate = RewTerm(
        func=mdp.action_rate_scaled_by_vel,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "min_scale": 0.3,
        },
    )


@configclass
class RobotEnvCfg(V11aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
