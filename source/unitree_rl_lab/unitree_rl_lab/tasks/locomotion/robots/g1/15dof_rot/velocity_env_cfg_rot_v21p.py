"""15-DOF rotation config V21p - strict validated command-conditioned LIRPG.

V21p keeps V21l's linear tracking reward form:

    r_phi = leaky(1 - error * f_phi(command, mode))

but removes actual velocity from the learnable slope inputs. Since
f_phi(command, mode) is positive and independent of error, the reward is
monotone in tracking error by construction and does not need V21l's
autograd monotonicity penalty.
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
from .velocity_env_cfg_rot_v21l import (
    LEAKY_SLOPE_NEG,
    META_LR,
    PRIOR_PARAM_L2_COEF,
    PRIOR_WEIGHT,
)


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_intrinsic_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
            "meta_lr": META_LR,
            "prior_weight": PRIOR_WEIGHT,
            "prior_param_l2_coef": PRIOR_PARAM_L2_COEF,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_intrinsic_cmd,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "slope_neg": LEAKY_SLOPE_NEG,
            "meta_lr": META_LR,
            "prior_weight": PRIOR_WEIGHT,
            "prior_param_l2_coef": PRIOR_PARAM_L2_COEF,
        },
    )


@configclass
class RobotEnvCfg(V21f2EnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21f2PlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
