"""15-DOF rotation config V21g - linear relative-error tracking everywhere.

Inherits V21f2 (best 19999-iter baseline: actor velocity estimator + decayed
gait/feet_clearance/rotation_single_support shaping in iter 6000-12000).

ONLY change vs V21f2: the two exp-kernel tracking rewards
    track_lin_vel_xy : track_lin_vel_xy_hybrid_low_speed -> track_lin_vel_xy_relative_full
    track_ang_vel_z  : track_ang_vel_z_hybrid_low_speed  -> track_ang_vel_z_relative_full

Reward form (both):
    r = clamp(1 - |err| / max(|cmd|, b_abs), 0, 1),   b_abs = 1.5670 * std
where 1.5670 solves (2v+1)*exp(-v) = 1 -> r <= exp(-|err|^2/std^2) pointwise.

This pushes the relative-error linear form (already used in the low-speed
regime of V21f2's hybrid) to the entire velocity space, including |cmd| = 0
which now uses 1 - |x|/b_abs (the limit of the relative form).

Decayed gait shaping (feet_clearance / gait / rotation_single_support) is
KEPT UNCHANGED from V21f2 to preserve the established early-training prior.

Runner is identical to V21e/f/f2 (G115DofV21eVelocityEstimatorPPORunnerCfg).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v21c import TRACK_STD
from .velocity_env_cfg_rot_v21f2 import (
    RewardsCfg as V21f2RewardsCfg,
    RobotEnvCfg as V21f2EnvCfg,
    RobotPlayEnvCfg as V21f2PlayEnvCfg,
)


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_relative_full,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_relative_full,
        weight=0.5 * 2 * 3,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
        },
    )


@configclass
class RobotEnvCfg(V21f2EnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21f2PlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
