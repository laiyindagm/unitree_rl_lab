"""15-DOF rotation config V11d -- "Low-Speed Tracking Bonus".

Fix the dead zone from the REWARD side with a SAFE additive approach.
Unlike V11a (adaptive sigma) which REDUCED early-training "free" rewards
and caused 55% bad_orientation, V11d only ADDS a bonus for accurate
tracking at low speed -- zero reward when standing still, positive reward
only when actually tracking the command.

Dead zone math before fix:
  cmd=0.2: marginal_tracking=0.15, action_cost=0.327 -> DEAD ZONE

After V11d:
  cmd=0.2: marginal_tracking=0.15, bonus_marginal=0.40, total=0.55
  action_cost=0.327 -> 0.55 >> 0.327 -> DEAD ZONE BROKEN

Key safety property: at cmd=0.2 standing still, bonus=0 (accuracy=0),
so early training (learning to stand) is COMPLETELY UNAFFECTED.

Base: V10d (bucketed curriculum + per-joint waist + V6c DR)
Changes: ADD low_speed_tracking_bonus + low_speed_rotation_bonus
"""

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
    """V10d rewards + low-speed tracking bonuses."""

    low_speed_tracking = RewTerm(
        func=mdp.low_speed_tracking_bonus,
        weight=0.4,
        params={
            "command_name": "base_velocity",
            "speed_threshold": 0.5,
        },
    )
    low_speed_rotation = RewTerm(
        func=mdp.low_speed_rotation_bonus,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "ang_threshold": 0.5,
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
