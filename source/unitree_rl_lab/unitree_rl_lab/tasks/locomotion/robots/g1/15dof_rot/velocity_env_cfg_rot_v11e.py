"""15-DOF rotation config V11e -- "V11b + Low-Speed Bonus" (combined attack).

Combines BOTH dead zone fixes:
  1. V11b's scaled action rate (penalty side): reduce action cost at low speed
  2. V11d's tracking bonus (reward side): increase marginal gain at low speed

Dead zone math:
  cmd=0.2 before: marginal=0.15, cost=0.327 -> DEAD ZONE
  cmd=0.2 V11b:   marginal=0.15, cost=0.121 -> marginal wins (margin 0.03)
  cmd=0.2 V11d:   marginal=0.55, cost=0.327 -> marginal wins (margin 0.22)
  cmd=0.2 V11e:   marginal=0.45, cost=0.121 -> marginal wins (margin 0.33!)

V11e has the LARGEST margin. If V11b alone is borderline (margin 0.03 may
not survive noise/terrain), V11e's margin of 0.33 should be robust.

Bonus weights slightly lower than V11d (0.3/0.2 vs 0.4/0.3) since the
scaled action rate already provides significant dead zone reduction.

Base: V11b (scaled action rate from V10d)
Changes: ADD low_speed_tracking_bonus + low_speed_rotation_bonus
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v11b import (
    CommandsCfg,
    CurriculumCfg,
    RewardsCfg as V11bRewardsCfg,
    RobotEnvCfg as V11bEnvCfg,
)


@configclass
class RewardsCfg(V11bRewardsCfg):
    """V11b rewards + low-speed tracking bonuses."""

    low_speed_tracking = RewTerm(
        func=mdp.low_speed_tracking_bonus,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "speed_threshold": 0.5,
        },
    )
    low_speed_rotation = RewTerm(
        func=mdp.low_speed_rotation_bonus,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "ang_threshold": 0.5,
        },
    )


@configclass
class RobotEnvCfg(V11bEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
