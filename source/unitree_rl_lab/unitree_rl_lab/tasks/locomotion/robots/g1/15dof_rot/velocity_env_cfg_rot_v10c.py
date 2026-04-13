"""15-DOF rotation config V10c -- "Manual Schedule Fallback".

Most robust curriculum approach: expand velocity ranges at fixed iterations,
completely independent of reward thresholds. Eliminates all curriculum bugs.

Schedule with 10000 iterations training:
  iter 0-1999:    initial narrow range (+/-0.1 lin, +/-0.3 ang) -- learn low-speed
  iter 2000-4999: 30% expansion toward limit -- introduce walking
  iter 5000-7999: 65% expansion -- moderate speed
  iter 8000+:     full limit_ranges -- final polish

Same starting ranges as V9c/V10a to get the same excellent
low-speed/rotation/transition behavior in early training.

Changes from V6c:
  Curriculum:
    - Replace lin/ang_vel_cmd_levels with iteration_based_vel_curriculum
  Commands:
    - Initial lin_vel: +/-0.1, ang_vel: +/-0.3
    - rel_standing: 0.20, rel_rotating: 0.30
  Rewards:
    - waist_joint_vel: -0.08 (stronger)
  Runner: BasePPORunnerV3Cfg
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


# ---------- Curriculum: deterministic iteration-based ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    iter_schedule = CurrTerm(
        func=mdp.iteration_based_vel_curriculum,
        params={
            "expand_iterations": (2000, 5000, 8000),
        },
    )


# ---------- Commands: narrow initial range ----------
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


# ---------- Rewards: V6c + stronger waist ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
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
