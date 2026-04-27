"""15-DOF rotation config V19i — "Scheduled Standing".

V19h Catastrophic Failure (20k iter):
  - Mean reward 76.5 -> 34.4 (-55%)
  - Episode length 994 -> 723 (-27%, early termination)
  - bad_orientation 1.6% -> 43.5% (26x more falls!)
  - entropy_loss -0.10 -> +9.02 (entropy INCREASING, action_std 0.34->0.54)
  - track_ang_vel_z 2.14 -> 1.01 (-53%)
  - 15k/20k pt fails sim2sim (10k still mostly OK)

Diagnosis: SUICIDE POLICY
  - V19h piled ~9.5/step penalty in standing envs (20% of all envs).
  - Policy discovered: falling -> early termination -> escape penalty.
  - PPO saw incoherent value function -> entropy bonus exploded action_std.
  - Late checkpoints capture the collapsed policy -> sim2sim fails.

Pattern observed across V19 series (user insight):
  - V19e: 17k stable (baseline)
  - V19f: backward drift bug
  - V19g: 18.7k stable but won't stand
  - V19h: collapsed at ~10k
  - Each round added MORE always-on penalties -> shrinking stable basin.

V19i Strategy: minimal changes from V19g + scheduled standing penalties.

Changes vs V19g (which trained stably for 18.7k):
  1. zero_cmd_body_vel -> zero_cmd_body_vel_scheduled
     weight -1.5 -> -3.0 PEAK, ramp from step 24k to 72k.
     Policy learns walking first; standing constraint introduced gradually.
  2. NEW stand_still_scheduled (peak weight -1.0, same ramp)
     Acts as joint-pose anchor at zero cmd, complementary to body_vel.
  3. KEEP feet_contact_without_cmd inherited from V5a (weight 0.3)
  4. KEEP rel_standing_envs = 0.10 (NOT 0.20)
  5. SKIP zero_cmd_foot_height (didn't help in V19h, 5x of nothing is nothing)

Kept identical to V19g (proven stable):
  - Standard track_lin_vel_xy
  - track_ang_vel_z (3.0) + track_ang_vel_z_sharp (sigma=0.20)
  - wz_proportional (-2.0)
  - pure_rotation_drift (-1.5)
  - Distribution: 10% standing, 15% pure_vx, 15% pure_vy, 25% pure_wz, 35% joint
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
        rel_standing_envs=0.10,         # back to V19g
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
    # ---------- V19g identical (linear/angular tracking, anti-cheat) ----------
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
        params={"command_name": "base_velocity", "cmd_threshold": 0.08},
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

    # ---------- V19i: SCHEDULED standing constraints ----------
    zero_cmd_body_vel = RewTerm(
        func=mdp.zero_cmd_body_vel_scheduled,
        weight=-3.0,                    # peak weight
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.1,
            "start_step": 24000,        # let policy learn walking first
            "end_step": 72000,          # full strength after stable locomotion
        },
    )
    stand_still_scheduled = RewTerm(
        func=mdp.stand_still_scheduled,
        weight=-1.0,                    # peak weight
        params={
            "command_name": "base_velocity",
            "start_step": 24000,
            "end_step": 72000,
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
