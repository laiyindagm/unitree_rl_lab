"""15-DOF rotation config V21k - constant-denominator leaky linear.

Hypothesis under test (user mathematical intuition):
    A leaky linear kernel with FIXED slope -1/b_abs everywhere
    (b_abs = LINEAR_REL_B_RATIO * std = 0.7834) should asymptotically
    match the exp kernel in V21f2 because:
      - constant gradient -1/0.78 across all error magnitudes
      - stays > 0 for |err| < 0.78 (no premature zero saturation)
      - only mildly negative beyond (slope_neg=0.1) so it never
        encourages termination.

Reward shape (for BOTH lin and ang):
    raw = 1 - |v - v_cmd| / b_abs
    r   = raw                if raw >= 0
        = slope_neg * raw    if raw <  0
    capped at <= 1.

Differs from V21g (per-cmd-scaled denom max(|cmd|, b)) and V21j (piecewise);
this is a SINGLE constant-denominator linear + leaky.

Decayed gait shaping kept identical to V21f2.
Runner identical to V21e/f/f2.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
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
        func=mdp.track_lin_vel_xy_constant_leaky,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_constant_leaky,
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
