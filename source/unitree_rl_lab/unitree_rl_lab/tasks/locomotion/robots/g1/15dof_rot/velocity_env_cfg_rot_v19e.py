"""15-DOF rotation config V19e — "Rotation Gradient Fix".

V19d Remaining Issues:
  1. wz PERF STILL LOW (0.13-0.23): exp kernel gives ~96% free reward at
     cmd_wz=0.1 (sigma=0.5, exp(-0.01/0.25)=0.961).  Cannot modify kernel
     (V8b/V11a/V14a all failed catastrophically).
  2. GRADIENT OPPOSITION: track_lin_vel_xy penalizes parasitic translation
     during rotation -> opposes wz learning in pure_wz envs.
  3. UPPER BODY SWAY: head oscillates during rotation, waist_roll_vel=-0.20
     insufficient.

V19e Fixes (C+A+2+3+4):
  C. ROTATION-SKIP LIN TRACKING: track_lin_vel_xy returns 1.0 when command
     is pure rotation (cmd_lin~0, cmd_wz>threshold).  Removes orthogonal
     gradient opposition.
  A. wz NONRESPONSE PENALTY: explicit penalty when cmd_wz active but robot
     not rotating.  Provides gradient even where exp kernel is saturated.
  2. UPPER BODY ACC: body_lin_acc_l2 on head_link penalizes head oscillation.
  3. WAIST ROLL: -0.20 -> -0.35 to suppress lateral sway during rotation.
  4. STAND STILL: -0.3 (inherited, no change needed).

Distribution: 8% standing, 15% pure_vx, 15% pure_vy, 20% pure_wz, 42% joint.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
    RewardsCfg as RotRewardsCfg,
)
from .velocity_env_cfg_rot_v16b import (
    RewardsCfg as V16bRewardsCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Commands: same as V19d (direction-gated + focused sampling) ------
@configclass
class CommandsCfg:
    base_velocity = mdp.MarginalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.10,
        rel_rotating_envs=0.20,
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
        # --- vx_pos: 15 discrete levels (0.1 to 1.5) ---
        vx_pos_bins=[
            (0.1, 0.1), (0.2, 0.2), (0.3, 0.3),
            (0.4, 0.4), (0.5, 0.5),
            (0.6, 0.6), (0.7, 0.7),
            (0.8, 0.8), (0.9, 0.9),
            (1.0, 1.0), (1.1, 1.1),
            (1.2, 1.2), (1.3, 1.3),
            (1.4, 1.4), (1.5, 1.5),
        ],
        # --- vx_neg: 8 discrete levels (-0.1 to -0.8) ---
        vx_neg_bins=[
            (-0.1, -0.1), (-0.2, -0.2), (-0.3, -0.3),
            (-0.4, -0.4), (-0.5, -0.5),
            (-0.6, -0.6), (-0.7, -0.7),
            (-0.8, -0.8),
        ],
        # --- vy: 11 levels (0, +/-0.1..+/-0.5) ---
        vy_bins=[
            (0.0, 0.0),
            (0.1, 0.1), (-0.1, -0.1),
            (0.2, 0.2), (-0.2, -0.2),
            (0.3, 0.3), (-0.3, -0.3),
            (0.4, 0.4), (-0.4, -0.4),
            (0.5, 0.5), (-0.5, -0.5),
        ],
        # --- wz: 16 levels (+/-0.1..+/-0.8) ---
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
        # Staged curriculum: conservative initial range
        num_active_vx_pos=3,
        num_active_vx_neg=3,
        num_active_vy=7,
        num_active_wz=6,
        # Adaptive parameters
        ema_alpha=0.1,
        temperature=8.0,
        min_sampling_prob=0.01,
        accuracy_cmd_min=0.05,
        zero_speed_accuracy_threshold=0.10,
        min_response_speed=0.05,
    )


# ---------- Rewards: rotation gradient fix ----------
@configclass
class RewardsCfg(V16bRewardsCfg):
    # C: rotation-skip linear tracking (returns 1.0 for pure rotation cmds)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_rotation_skip,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "lin_threshold": 0.05,
            "yaw_threshold": 0.05,
        },
    )
    # Boost angular velocity tracking (kept from V19d)
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=0.5 * 2 * 2,    # 2.0
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # A: wz-specific nonresponse penalty
    wz_nonresponse = RewTerm(
        func=mdp.wz_nonresponse_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
            "vel_threshold": 0.05,
        },
    )
    # 3: stronger waist roll damping (-0.20 -> -0.35)
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )
    # 2: torso flat orientation — keep camera level
    torso_flat_orient = RewTerm(
        func=mdp.torso_flat_orientation,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # Kept from V19d: strong movement incentive
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
    # Kept from V19d: strong general nonresponse penalty
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
            "vel_threshold": 0.05,
        },
    )


# ---------- Curriculum: same as V19d ----------
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


# ---------- Env ----------
@configclass
class RobotEnvCfg(V15aEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()


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
