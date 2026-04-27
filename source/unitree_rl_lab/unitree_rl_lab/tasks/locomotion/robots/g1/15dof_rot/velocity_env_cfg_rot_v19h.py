"""15-DOF rotation config V19h — "Force Standing".

V19g Results (18.7k iter):
  - vx/vy 0.1+ response: WORKING
  - wz 0.1+ response: WORKING
  - Pure rotation backward drift: FIXED (drift reward only -0.06)
  - Zero-cmd standing: STILL FAILS (robot keeps stepping)

Root cause analysis (from log):
  - Walking signals total ~+5.0/episode (track_lin/ang, gait, feet_clearance)
  - Standing signals total ~-0.15/episode (zero_cmd_body_vel, stand_still,
    feet_contact_without_cmd)
  - Signal ratio 33:1 in favor of moving -> "always walking" attractor wins.
  - exp(-vel^2/0.25) is too soft at small vel: 0.2m/s drift gives 0.85 reward.
  - 90% of envs have nonzero cmd -> single "walk policy" learned, no mode switch.
  - zero_cmd_body_vel is cheatable by symmetric stepping (root vel cancels out).

V19h Strategy: AMPLIFY standing signals + add geometric anti-step penalty.

Changes vs V19g:
  1. NEW zero_cmd_foot_height (w=-3.0): direct geometric penalty for foot
     z-height when cmd_norm < 0.1.  Cannot be cheated by symmetric stepping.
  2. zero_cmd_body_vel: -1.5 -> -5.0 (3.3x stronger root-velocity penalty)
  3. stand_still override: -0.3 -> -1.5 (5x stronger joint-deviation penalty
     when cmd~0)
  4. feet_contact_without_cmd override: 0.3 -> 1.0 (3x positive bonus for
     keeping both feet on ground at cmd=0)
  5. rel_standing_envs: 0.10 -> 0.20 (double standing training data)

Kept from V19g (verified working):
  - Standard track_lin_vel_xy (no rotation skip)
  - track_ang_vel_z (w=3.0) + track_ang_vel_z_sharp (sigma=0.20)
  - wz_proportional (w=-2.0)
  - pure_rotation_drift (w=-1.5)

Distribution: 20% standing, 13% pure_vx, 13% pure_vy, 24% pure_wz, 30% joint.

Expected standing signal increase at zero cmd:
  V19g: -0.15 / +0.03 (penalty / bonus) per episode
  V19h: -1.0 / +0.10 (~7x stronger penalty + new geometric term)
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
        rel_standing_envs=0.20,         # 0.10 -> 0.20 (double standing data)
        rel_rotating_envs=0.24,         # slight reduction to make room
        rel_pure_vx_envs=0.13,
        rel_pure_vy_envs=0.13,
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
    # ---------- V19g kept (linear/angular tracking, anti-cheat) ----------
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
    wz_proportional = RewTerm(
        func=mdp.wz_proportional_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
        },
    )
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

    # ---------- V19h: AMPLIFIED standing signals ----------
    # Boost: -1.5 -> -5.0
    zero_cmd_body_vel = RewTerm(
        func=mdp.zero_cmd_body_vel,
        weight=-5.0,
        params={"command_name": "base_velocity"},
    )
    # NEW: geometric anti-stepping (cannot be cheated by symmetric stepping)
    zero_cmd_foot_height = RewTerm(
        func=mdp.zero_cmd_foot_height,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    # Override parent (V5a) -0.3 -> -1.5
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )
    # Override parent (V5a) 0.3 -> 1.0
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
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