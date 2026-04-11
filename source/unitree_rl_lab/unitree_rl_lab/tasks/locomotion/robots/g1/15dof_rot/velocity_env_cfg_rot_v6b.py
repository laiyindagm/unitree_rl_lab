"""15-DOF rotation config V6b — "Rotation-Transition Fix": V6a stability + rotation boost.

V5c's biggest gap: cannot transition from standstill to pure rotation, and
standstill-to-movement transitions are aggressive. This version addresses both
by increasing rotation exposure and strengthening rotation incentives, on top
of V6a's stability improvements.

Changes from V6a:
  Command distribution:
    - rel_rotating_envs: 0.20 -> 0.30    (50% more rotation practice)
    - initial ang_vel_z: +-0.2 -> +-0.3  (wider initial range)
    - limit  ang_vel_z: +-0.6 -> +-0.8   (wider limit for curriculum)
    - resampling_time_range: (10, 10) -> (8, 12)  (varied command changes for transition diversity)
  Rotation rewards:
    - track_ang_vel_z: 1.0 -> 1.3        (stronger rotation tracking incentive)
    - rotation_single_support: 1.0 -> 1.5 (stronger single-foot rotation reward)
  Runner: BasePPORunnerV3Cfg
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v5c import EventCfg
from .velocity_env_cfg_rot_v6a import (
    RewardsCfg as V6aRewardsCfg,
    RobotEnvCfg as V6aEnvCfg,
)
from .velocity_env_cfg_rot import CommandsCfg as RotCommandsCfg


@configclass
class CommandsCfg(RotCommandsCfg):
    """Rot commands with wider rotation and varied resampling."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
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


@configclass
class RewardsCfg(V6aRewardsCfg):
    """V6a stability rewards + stronger rotation incentives."""

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=0.5 * 2.6,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    rotation_single_support = RewTerm(
        func=mdp.rotation_single_support_reward,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class RobotEnvCfg(V6aEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
