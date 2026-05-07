"""15-DOF rotation config V21l - LIRPG (learnable intrinsic tracking reward).

Per-channel MLP r_phi(v, v_cmd) replaces the V21k constant-leaky-linear
tracking reward. The MLP is initialized to mimic V21k and then meta-updated
online by the LIRPG PPO subclass to maximize task return = -|v - v_cmd|.

All other reward terms / curriculum / observations match V21f2/k.
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
MONO_COEF = 1.0          # weight for e*f monotonicity penalty
META_LR = 2e-5
PRIOR_WEIGHT = 1.0
PRIOR_PARAM_L2_COEF = 1e-1


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_intrinsic,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
            "meta_lr": META_LR,
            "prior_weight": PRIOR_WEIGHT,
            "prior_param_l2_coef": PRIOR_PARAM_L2_COEF,
            "mono_coef": MONO_COEF,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_intrinsic,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
            "meta_lr": META_LR,
            "prior_weight": PRIOR_WEIGHT,
            "prior_param_l2_coef": PRIOR_PARAM_L2_COEF,
            "mono_coef": MONO_COEF,
        },
    )


@configclass
class RobotEnvCfg(V21f2EnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21f2PlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
