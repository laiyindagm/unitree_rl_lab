"""15-DOF rotation config V20e — "V20c + Sharp Linear Tracking".

Single-axis A/B over V20c (V19f + token):
  Tighten `track_lin_vel_xy_rotation_skip` std from sqrt(0.25)=0.5 to 0.30.
  All other rewards/commands/curriculum/observations unchanged.

Rationale (vs V20c metrics @ 19.6k iter):
  - error_vel_xy = 0.50 m/s (50% of full ±1 m/s range) but
    track_lin_vel_xy = 0.81/1.0 max — apparent paradox resolved by reward
    kernel std: at error=0.5, exp(-0.25/0.25) = 0.37 of max reward. Wide
    kernel pays substantially even for sloppy tracking.
  - Sharper kernel (std=0.30): error=0.5 -> exp(-0.25/0.09) = 0.063 (-83%).
    Real gradient pressure to reduce tracking error.

NOT changed vs V20c: rotation_skip function (only its `std` param),
all wz tracking terms, all penalties, command bins, curriculum, DR.

Predictions @ 12-15k iter:
  - error_vel_xy < 0.30 m/s; gait visibly smoother in vx/vy commands
  - track_lin_vel_xy reward dips early (~0.5 plateau possible) but
    Mean reward should recover via reduced tracking error
  - bad_orient unchanged
  - Pure-rotation behavior unchanged (rotation_skip path returns 1.0
    independent of std)

Risks:
  - If reward dips sustainedly (Mean reward < 60 by iter 5k), std is too
    sharp; fallback config V20e' would use std=0.40.

Decision tree:
  - error_vel_xy halved AND other metrics held -> sharp kernel validated
  - error_vel_xy unchanged -> sigma is not the bottleneck (curriculum or
    physical capability), abandon this direction
  - Reward collapse -> std too tight, retry with 0.40
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v19f import RewardsCfg as V19fRewardsCfg
from .velocity_env_cfg_rot_v20c import (
    ObservationsCfg as V20cObservationsCfg,
    RobotEnvCfg as V20cEnvCfg,
)


@configclass
class RewardsCfg(V19fRewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_rotation_skip,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.30,
            "lin_threshold": 0.05,
            "yaw_threshold": 0.05,
        },
    )


@configclass
class RobotEnvCfg(V20cEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Play: expose the full curriculum-extended command range.
        # V19f bin counts: 15 vx_pos, 8 vx_neg, 11 vy, 16 wz.
        self.commands.base_velocity.num_active_vx_pos = 15
        self.commands.base_velocity.num_active_vx_neg = 8
        self.commands.base_velocity.num_active_vy = 11
        self.commands.base_velocity.num_active_wz = 16
