"""15-DOF rotation config V14a — "Baselined Tracking Kernel".

Hypothesis: The exp(-error²/σ²) kernel gives ~0.85 free reward for standing
still at low cmd, creating a local optimum where moving is not worth the
penalty cost. Baselined tracking subtracts this free reward so standing
still = 0, making any actual tracking improvement purely positive.

Tracking reward math (σ²=0.25):
  cmd=0.2: original standing_still=0.852, baselined=0.  Perfect=0.148
  cmd=0.3: original standing_still=0.698, baselined=0.  Perfect=0.302
  cmd=0.5: original standing_still=0.368, baselined=0.  Perfect=0.632

With weight=3.0:
  cmd=0.2 perfect tracking → 3.0 * 0.148 = 0.444 reward (was 0.148 marginal)
  cmd=0.5 perfect tracking → 3.0 * 0.632 = 1.896 reward

Changes from V13a:
  - track_lin_vel_xy: exp kernel → baselined kernel, weight 1.0 → 3.0
  - track_ang_vel_z: rotating_aware → baselined, weight 1.3 → 3.0
  Everything else identical to V13a (action_rate -0.02, V10d base).
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v13a import (
    RewardsCfg as V13aRewardsCfg,
    RobotEnvCfg as V13aEnvCfg,
)


@configclass
class RewardsCfg(V13aRewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_baselined, weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_baselined, weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class RobotEnvCfg(V13aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
