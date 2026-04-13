"""15-DOF rotation config V10d -- "Fixed Bucketed + Per-Joint Waist".

Combines the fixed (buffer-based) bucketed curriculum with per-joint
waist penalty differentiation. The key insight: waist_yaw contributes
to angular momentum during rotation and should be lightly penalized,
while waist_roll (head lateral sway) needs heavy damping.

V10c showed:
  - waist_yaw oscillation 0.06 amplitude during rotation
  - ang_vel > 0.5 needed to start rotation from standstill
  - lin_vel > 0.4 for normal walking, amplitude < 0.03

Per-joint waist split rationale:
  - waist_roll:  -0.20  (strongest -- prevents head lateral sway)
  - waist_pitch: -0.06  (moderate -- limits forward/backward tilt)
  - waist_yaw:   -0.02  (lightest -- allow rotation contribution)

Changes from V10a:
  - Replace uniform waist_joint_vel(-0.08) with 3 per-joint terms
  - Everything else identical: bucketed curriculum, same commands, V6c rewards
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v6c import (
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)


# ---------- Curriculum: fixed buffer-based bucketed ----------
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


# ---------- Commands: narrow initial range (same as V10a/V10c) ----------
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


# ---------- Rewards: V6c + per-joint waist ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    # Per-joint waist penalties: roll >> pitch > yaw
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )
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
