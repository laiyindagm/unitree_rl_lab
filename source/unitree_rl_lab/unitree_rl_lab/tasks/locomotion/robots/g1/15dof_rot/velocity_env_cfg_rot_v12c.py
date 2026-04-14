"""15-DOF rotation config V12c -- "Scheduled Sigma + Scaled Action Rate + Low-Speed Bonus".

Triple attack on the dead zone from all three directions:
  1. Scheduled σ annealing (from V12a): tighten tracking kernel after iter 3000
  2. Scaled action rate (from V11b): reduce penalty at low speed
  3. Low-speed tracking bonus (from V11d): additive reward for accurate tracking

Why V12c matters: In V11d/V11e, low_speed_bonus barely activated (0.055/0.025)
because the dead zone prevented the agent from even ATTEMPTING to track at low
speed → accuracy ≈ 0 → bonus ≈ 0. With V12b breaking the dead zone, the agent
should start trying to track → accuracy > 0 → bonus provides extra gradient.

This tests: "Was the low_speed_bonus useless in V11d because of the dead zone?"

Experimental matrix:
  V12a: σ annealing only        → isolates σ annealing effect
  V12b: σ annealing + scaled AR → tests penalty reduction synergy
  V12c: V12b + low-speed bonus  → tests additive bonus when dead zone is broken

Base: V12b (scheduled sigma + scaled action rate + V10d)
Changes: ADD low_speed_tracking_bonus + low_speed_rotation_bonus (same as V11d)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v12b import (
    CommandsCfg,
    CurriculumCfg,
    RewardsCfg as V12bRewardsCfg,
    RobotEnvCfg as V12bEnvCfg,
)


@configclass
class RewardsCfg(V12bRewardsCfg):
    """V12b rewards + low-speed tracking bonuses (from V11d)."""

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
class RobotEnvCfg(V12bEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
