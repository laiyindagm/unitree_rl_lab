"""15-DOF rotation config V19g — "Balanced Rotation".

V19f Results (test):
  - Zero-speed standing: FIXED
  - wz responds from 0.1: WORKING
  - vx/vy 0.1-0.3: NO RESPONSE  (regression vs V19e)
  - Pure rotation: robot drifts BACKWARD  (cheating strategy)

Root cause:
  - track_lin_vel_xy_rotation_skip returns 1.0 for pure rotation
    -> ZERO penalty for any linear motion during rotation.
  - wz_proportional weight=-4.0 dominates -> policy uses asymmetric leg
    push (= backward thrust) as the easiest way to spin.
  - In joint envs (35%), small cmd_wz triggers strong penalty -> policy
    biased to "rotate first, ignore small linear cmd".

V19g Minimal Fixes (3 surgical changes vs V19f):
  1. REVERT track_lin_vel_xy: standard yaw_frame_exp -> linear motion is
     again penalized via exp(-vel^2/sigma^2) when cmd_lin=0.
  2. NEW pure_rotation_lin_drift (w=-1.5): direct, scale-aware penalty
     for linear drift during pure rotation. Targets backward-push cheat.
  3. REDUCE wz_proportional: -4.0 -> -2.0. Still 2x V19e binary version,
     but no longer dominates linear tracking.

Kept from V19f (verified working):
  - track_ang_vel_z = 3.0 (boosted from 2.0)
  - track_ang_vel_z_sharp (sigma=0.20, w=1.0)
  - zero_cmd_body_vel (w=-1.5)
  - 25% pure_wz envs

Distribution: 10% standing, 15% pure_vx, 15% pure_vy, 25% pure_wz, 35% joint.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v16b import (
    RewardsCfg as V16bRewardsCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


@configclass
class CommandsCfg:
    base_velocity = mdp.MarginalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.10,
        rel_rotating_envs=0.25,
        rel_pure_vx_envs=0.15,
        rel_pure_vy_envs=0.15,
        rel_linear_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.MarginalVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3),
        ),
        vx_pos_bins=[
            (0.1, 0.1), (0.2, 0.2), (0.3, 0.3),
            (0.4, 0.4), (0.5, 0.5),
            (0.6, 0.6), (0.7, 0.7),
            (0.8, 0.8), (0.9, 0.9),
            (1.0, 1.0), (1.1, 1.1),
            (1.2, 1.2), (1.3, 1.3),
            (1.4, 1.4), (1.5, 1.5),
        ],
        vx_neg_bins=[
            (-0.1, -0.1), (-0.2, -0.2), (-0.3, -0.3),
            (-0.4, -0.4), (-0.5, -0.5),
            (-0.6, -0.6), (-0.7, -0.7),
            (-0.8, -0.8),
        ],
        vy_bins=[
            (0.0, 0.0),
            (0.1, 0.1), (-0.1, -0.1),
            (0.2, 0.2), (-0.2, -0.2),
            (0.3, 0.3), (-0.3, -0.3),
            (0.4, 0.4), (-0.4, -0.4),
            (0.5, 0.5), (-0.5, -0.5),
        ],
        wz_bins=[
            (0.1, 0.1), (-0.1, -0.1),
            (0.2, 0.2), (-0.2, -0.2),
            (0.3, 0.3), (-0.3, -0.3),
            (0.4, 0.4), (-0.4, -0.4),
            (0.5, 0.5), (-0.5, -0.5),
            (0.6, 0.6), (-0.6, -0.6),
            (0.7, 0.7), (-0.7, -0.7),
            (0.8, 0.8), (-0.8, -0.8),
        ],
        num_active_vx_pos=3,
        num_active_vx_neg=3,
        num_active_vy=7,
        num_active_wz=6,
        ema_alpha=0.1,
        temperature=8.0,
        min_sampling_prob=0.01,
        accuracy_cmd_min=0.05,
        zero_speed_accuracy_threshold=0.10,
        min_response_speed=0.05,
    )


@configclass
class RewardsCfg(V16bRewardsCfg):
    # FIX 1: REVERT to standard linear tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=0.5 * 2 * 3,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_sharp = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.20},
    )
    # FIX 3: REDUCE -4.0 -> -2.0
    wz_proportional = RewTerm(
        func=mdp.wz_proportional_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
        },
    )
    # FIX 2: NEW direct anti-backward-drift penalty
    pure_rotation_drift = RewTerm(
        func=mdp.pure_rotation_lin_drift,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "lin_threshold": 0.05,
            "yaw_threshold": 0.05,
        },
    )
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )
    torso_flat_orient = RewTerm(
        func=mdp.torso_flat_orientation,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    movement_incentive = RewTerm(
        func=mdp.movement_incentive_scheduled,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "cmd_threshold": 0.05,
            "start_step": 48000,
            "end_step": 96000,
        },
    )
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
            "vel_threshold": 0.05,
        },
    )
    zero_cmd_body_vel = RewTerm(
        func=mdp.zero_cmd_body_vel,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )


@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.marginal_vel_curriculum,
        params={
            "range_expand_threshold": 0.3,
            "min_perf_floor": 0.1,
        },
    )


@configclass
class RobotEnvCfg(V15aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096

# play的设置必须加
@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.num_active_vx_pos = None
        self.commands.base_velocity.num_active_vx_neg = None
        self.commands.base_velocity.num_active_vy = None
        self.commands.base_velocity.num_active_wz = None