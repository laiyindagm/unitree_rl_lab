"""15-DOF rotation config V21m - Gaussian LIRPG (learnable sigma kernel).

V21l used r_phi = leaky(1 - e * f_phi(v, v_cmd)) with a slope MLP.
V21m replaces this with a Gaussian kernel:

    r = exp(-|v - v_cmd|^2 / sigma(v_cmd)^2)

where sigma(v_cmd) is a small MLP (input: v_cmd only, output: sigma > 0 via
softplus). Default init: sigma_0 = 0.5 (matches V21f2 exp-kernel std).

Motivation:
  - Gaussian is C-infinity, no discontinuous slope at zero (unlike leaky-linear).
  - sigma is command-conditioned only (v_cmd), not state-conditioned, so it
    learns a per-command "acceptance width" that persists across rollouts.
  - Monotone in |err| by construction (no mono_coef needed).

Weights kept at 1.5 : 1.5 (lin : ang), same as V21l modified run.
Runner: same LirpgVelocityEstimatorPPO as V21l.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v21f2 import (
    RewardsCfg as V21f2RewardsCfg,
    RobotEnvCfg as V21f2EnvCfg,
    RobotPlayEnvCfg as V21f2PlayEnvCfg,
)

SIGMA_0 = 0.5            # initial sigma value (matches V21f2 std)
META_LR = 2e-5
PRIOR_WEIGHT = 1.0
PRIOR_PARAM_L2_COEF = 0.0


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_intrinsic_sigma,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sigma_0": SIGMA_0,
            "meta_lr": META_LR,
            "prior_weight": PRIOR_WEIGHT,
            "prior_param_l2_coef": PRIOR_PARAM_L2_COEF,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_intrinsic_sigma,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sigma_0": SIGMA_0,
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
