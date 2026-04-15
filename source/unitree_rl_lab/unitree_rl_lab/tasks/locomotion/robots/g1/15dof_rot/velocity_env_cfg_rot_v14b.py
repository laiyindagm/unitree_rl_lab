"""15-DOF rotation config V14b -- "V14a + speed-gated rotation penalties".

V14a rebalances the general penalty budget. V14b additionally addresses
the angular velocity dead zone by gating rotation-specific penalties
behind a minimum yaw command threshold.

The rotation_double_support_slide and rotation_twist_joint penalties
shape *how* the robot rotates (stepping turn vs torso twist). At low
yaw commands the robot hasn't yet learned *whether* to rotate, so
constraining the method too early blocks exploration. Gating these
penalties to |cmd_yaw| > 0.3 lets the robot first learn to respond
to low yaw commands, then refine the rotation style at higher speeds.

Changes from V14a:
  - rotation_double_support_slide → gated version (active only |cmd_yaw|>0.3)
  - rotation_twist_joint → gated version (active only |cmd_yaw|>0.3)
  - Wider initial curriculum range for both lin_vel and ang_vel
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v14a import (
    CurriculumCfg as V14aCurriculumCfg,
    EventCfg,
    RewardsCfg as V14aRewardsCfg,
    RobotEnvCfg as V14aEnvCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg


# ---------- Commands: wider initial ranges ----------
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
            lin_vel_x=(-0.3, 0.5),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


# ---------- Rewards: V14a + gated rotation penalties ----------
@configclass
class RewardsCfg(V14aRewardsCfg):
    # Override rotation penalties with gated versions that only activate at |cmd_yaw|>0.3
    rotation_double_support_slide = RewTerm(
        func=mdp.rotation_double_support_slide_penalty_gated,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_yaw_cmd": 0.3,
        },
    )
    rotation_twist_joint = RewTerm(
        func=mdp.rotation_twist_joint_penalty_gated,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", ".*_hip_yaw_joint"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_yaw_cmd": 0.3,
        },
    )


# ---------- Curriculum: same as V14a ----------
@configclass
class CurriculumCfg(V14aCurriculumCfg):
    pass


# ---------- Env ----------
@configclass
class RobotEnvCfg(V14aEnvCfg):
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
