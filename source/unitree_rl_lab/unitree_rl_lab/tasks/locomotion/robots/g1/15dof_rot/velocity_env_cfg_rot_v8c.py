"""15-DOF rotation config V8c — "Low-Speed Curriculum".

Different angle: fix the dead zone through training distribution, not rewards.

V6c's command distribution: only 5% standing, init lin_vel +-0.1 (quickly
expands via curriculum). The policy barely sees low-speed commands during
training, so it never learns to track them.

Fix: increase standing/low-speed exposure + faster command resampling.
  - rel_standing: 0.05 -> 0.15 (3x more standing practice)
  - rel_rotating: 0.30 -> 0.25 (slightly less, reclaimed for standing)
  - resampling: (8,12) -> (6,10) (more frequent command changes = more
    transitions to practice static→moving)
  - Gentle waist stabilization (-0.04) for head sway (same safe budget as V8a)

This doesn't touch tracking rewards or action penalties at all.
The reward budget change is minimal (+waist -0.04 → ~-0.003/step).

Changes from V6c:
  Command distribution:
    - rel_standing: 0.05 -> 0.15
    - rel_rotating: 0.30 -> 0.25
    - resampling_time_range: (8, 12) -> (6, 10)
  Rewards:
    - waist_joint_vel_penalty: -0.04 (new, gentle)
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v6c import (
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg


@configclass
class CommandsCfg(V6bCommandsCfg):
    """More standing + faster resampling for low-speed practice."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.15,
        rel_rotating_envs=0.25,
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
class RewardsCfg(V6cRewardsCfg):
    """V6c + gentle waist stabilization."""

    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


@configclass
class RobotEnvCfg(V6cEnvCfg):
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
