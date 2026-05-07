"""15-DOF rotation config V21j - V21i strict piecewise + LEAKY negative tail.

Same denominator structure as V21i (literal user spec):
    cmd = 0:  raw = 1 - |v|     / b
    cmd > 0:  raw = 1 - |v-cmd| / |cmd|

V21i clamps `raw` to [0, 1] -> dead zone for cmd>0, err>|cmd|.
V21j keeps the SAME `raw` formula but replaces clamp(0,1) with a leaky map:
    r = raw                if raw >= 0
        slope_neg * raw    if raw <  0   (slope_neg = 0.1)
    capped at <= 1.

Effect:
- raw >= 0: identical to V21i (gradient -1/denom)
- raw < 0:  not dead; gradient -slope_neg/denom (small but nonzero)
- Worst-case stationary penalty per step:
    cmd in [-0.8, 0.8]; if err = 2 * cmd_mag (badly off), raw ~= -1, r ~= -0.1
    times weight (1.0 lin + 3.0 ang) -> per-step ~= -0.4 worst case
    vs alive_reward = +0.15/step -> negative is bounded but not catastrophic.

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

LEAKY_SLOPE_NEG = 0.1


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_piecewise_leaky,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_piecewise_leaky,
        weight=0.5 * 2 * 3,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
        },
    )


@configclass
class RobotEnvCfg(V21f2EnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21f2PlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
