"""15-DOF rotation config V19f — "Aggressive Rotation".

V19e Results (17k iter):
  - vx perf 0.408-0.651 (low bins ~40% depressed by direction-gate)
  - vy perf 0.300-0.464 (zero-bin = 0.300, standing issue)
  - wz perf 0.143-0.197 (NO improvement over V19d)
  - Zero-speed: robot still walks when cmd=0

V19f Aggressive Changes:
  1. SHARP wz TRACKING: supplementary exp kernel with sigma=0.20 (vs 0.50).
     Provides 5-9x stronger gradient at small cmd_wz. Weight=1.0.
  2. PROPORTIONAL wz PENALTY: replaces binary wz_nonresponse.
     penalty = |cmd_wz| * (1 - signed_ratio). Continuous gradient, no dead
     zone after actual_wz > 0.05. Weight=-4.0.
  3. BOOST ORIGINAL wz TRACKING: weight 2.0 -> 3.0. Stronger large-cmd gradient.
  4. MORE ROTATION ENVS: pure_wz 20% -> 25%.
  5. ZERO-SPEED FIX: zero_cmd_body_vel (weight=-1.5) penalizes root velocity
     when cmd_norm < 0.1.

Expected wz incentive at cmd=0.1 (standing -> perfect):
  V19e: 0.078 (exp) + ~0.1 (binary nonresponse) = ~0.18
  V19f: 0.117 (exp) + 0.221 (sharp) + 0.400 (proportional) = ~0.74  (4x)

Distribution: 10% standing, 15% pure_vx, 15% pure_vy, 25% pure_wz, 35% joint.
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


# ---------- Commands: increased pure_wz fraction ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.MarginalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.10,
        rel_rotating_envs=0.25,       # 0.20 -> 0.25 (more rotation training)
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


# ---------- Rewards: aggressive rotation ----------
@configclass
class RewardsCfg(V16bRewardsCfg):
    # C (from V19e): rotation-skip linear tracking
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
    # Original wz tracking — BOOSTED: 2.0 -> 3.0
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=0.5 * 2 * 3,    # 3.0 (was 2.0 in V19e)
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # NEW: supplementary SHARP wz tracking (sigma=0.20)
    # Provides 5-9x stronger gradient at small cmd_wz
    track_ang_vel_z_sharp = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.20},
    )
    # NEW: proportional wz penalty (replaces binary wz_nonresponse)
    wz_proportional = RewTerm(
        func=mdp.wz_proportional_penalty,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
        },
    )
    # Waist damping (from V19e)
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )
    # Torso flat orientation (from V19e)
    torso_flat_orient = RewTerm(
        func=mdp.torso_flat_orientation,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # Movement incentive (from V19e)
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
    # General linear nonresponse (from V19e)
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
            "vel_threshold": 0.05,
        },
    )
    # NEW: zero-speed standing fix
    zero_cmd_body_vel = RewTerm(
        func=mdp.zero_cmd_body_vel,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )


# ---------- Curriculum: same as V19d/V19e ----------
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
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Reduce scene size for faster rotation learning (same as V19e)
        self.scene = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)
