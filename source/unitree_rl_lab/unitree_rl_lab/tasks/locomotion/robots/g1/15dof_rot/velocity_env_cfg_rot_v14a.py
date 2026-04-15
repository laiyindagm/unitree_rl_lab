"""15-DOF rotation config V14a -- "Penalty Budget Rebalance".

Root-cause analysis of the low-speed dead zone (V6c→V13) showed that
action_rate was only ~30% of the problem. The remaining ~70% comes from
the total penalty budget being too heavy relative to marginal tracking
gains at low speeds.

Cross-repo comparison (legged_rl_lab G1, AgibotTech X1, legged_gym):
  - joint_deviation_legs punishes the exact joints needed for walking
  - flat_orientation_l2 at -5.0 is 2.5x the standard (-2.0)
  - Rotation-specific penalties fire at all speeds, blocking low-speed rotation
  - No explicit penalty for ignoring commands (standing at nonzero cmd)

Changes from V10d (via V13b baseline: action_rate=-0.02, feet_slide=-0.1):
  Tier 1 (highest impact):
    - joint_deviation_legs:   -1.0 → -0.1  (free up hip joints for walking/rotation)
    - flat_orientation_l2:    -5.0 → -2.0  (align with industry standard)
    - NEW cmd_nonresponse:    -0.5          (penalize standing still at nonzero cmd)
  Tier 2:
    - waist_roll_vel:         -0.20 → -0.05 (allow normal walking waist motion)
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    TRAIN_JOINT_NAMES,
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v6c import (
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)


# ---------- Curriculum: fixed buffer-based bucketed (from V10d) ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    speed_bucketed = CurrTerm(
        func=mdp.speed_bucketed_vel_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "low_speed_threshold": 0.3,
            "mid_speed_threshold": 0.6,
            "unlock_reward_ratio": 0.6,
        },
    )


# ---------- Commands: same as V10d ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.20,
        rel_rotating_envs=0.30,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.3, 0.3),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


# ---------- Rewards: rebalanced penalty budget ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    # --- V13b carry-over: reduced action_rate and feet_slide ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # --- Tier 1: penalty rebalance ---
    # hip_roll/hip_yaw must deviate to walk and rotate; -1.0 directly fights movement
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # -5.0 is 2.5x the industry standard; normal walking causes minor tilt
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # Direct penalty for ignoring nonzero commands -- breaks the "rational laziness" equilibrium
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty, weight=-0.5,
        params={"command_name": "base_velocity", "cmd_threshold": 0.15, "vel_threshold": 0.05},
    )

    # --- Tier 2: waist_roll was too aggressive ---
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )

    # --- V10d per-joint waist (pitch/yaw unchanged) ---
    waist_pitch_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])},
    )
    waist_yaw_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )


# ---------- Env ----------
@configclass
class RobotEnvCfg(V6cEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
