"""15-DOF rotation config V21i - STRICT piecewise tracking per spec.

Re-implements V21g to match the user's literal specification:
    cmd = 0:  r = 1 - |v|     / b
    cmd > 0:  r = 1 - |v-cmd| / |cmd|     (NO max with b)
    where b = 1.5670 * std satisfies 1 - |x|/b <= exp(-x^2/std^2) for x>=0.

Difference vs V21g (`*_relative_full`):
    V21g:  denom = max(|cmd|, b)        (smooth interpolation, NO dead zone)
    V21i:  denom = b if |cmd|<eps else |cmd|  (HARD switch, dead zone for cmd>0)

Decayed gait shaping kept identical to V21f2.
Runner identical to V21e/f/f2.
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
        func=mdp.track_lin_vel_xy_piecewise_strict,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_piecewise_strict,
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
