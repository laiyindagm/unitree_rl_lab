"""15-DOF rotation config V21h - V21f2 hybrid with dead-zone fix.

Single-variable ablation isolating "removing the r_rel dead zone" from
"changing the kernel form" (V21g changed BOTH).

ONLY change vs V21f2: the hybrid r_rel branch's `rel_floor` parameter
   0.05  ->  0.5
which lifts the divisor floor from "tiny epsilon" to half the std-bandwidth.
For yaw cmd in [-0.8, 0.8] (where err often exceeds |cmd| in V21f2), the
floor=0.5 ensures `denom = max(|cmd|, 0.5)`, eliminating the dead zone
(err > |cmd| -> r=0, grad=0) while keeping the EXACT SAME kernel SHAPE as
V21f2 elsewhere. The exp-kernel high-speed branch is untouched.

Decayed gait shaping (feet_clearance / gait / rotation_single_support) is
KEPT UNCHANGED from V21f2.

Runner is identical to V21e/f/f2 (G115DofV21eVelocityEstimatorPPORunnerCfg).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v21c import (
    TRACK_STD,
    LOW_SPEED_THRESHOLD_SCALE,
    LOW_SPEED_TRANSITION_WIDTH,
    LOW_SPEED_CMD_MIN,
)
from .velocity_env_cfg_rot_v21f2 import (
    RewardsCfg as V21f2RewardsCfg,
    RobotEnvCfg as V21f2EnvCfg,
    RobotPlayEnvCfg as V21f2PlayEnvCfg,
)

REL_FLOOR_FIX = 0.5  # was 0.05 in V21f2; raised to kill dead zone


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_hybrid_low_speed,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "threshold_scale": LOW_SPEED_THRESHOLD_SCALE,
            "transition_width": LOW_SPEED_TRANSITION_WIDTH,
            "lin_cmd_min": LOW_SPEED_CMD_MIN,
            "rel_floor": REL_FLOOR_FIX,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_hybrid_low_speed,
        weight=0.5 * 2 * 3,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "threshold_scale": LOW_SPEED_THRESHOLD_SCALE,
            "transition_width": LOW_SPEED_TRANSITION_WIDTH,
            "ang_cmd_min": LOW_SPEED_CMD_MIN,
            "rel_floor": REL_FLOOR_FIX,
        },
    )


@configclass
class RobotEnvCfg(V21f2EnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21f2PlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
