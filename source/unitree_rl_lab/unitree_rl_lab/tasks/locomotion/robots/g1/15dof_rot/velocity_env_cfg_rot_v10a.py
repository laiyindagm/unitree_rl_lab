"""15-DOF rotation config V10a -- "Fixed Bucketed + Stronger Waist".

V9c showed that the narrow-range bucketed curriculum produces the BEST
rotation response (>0.1 cmd) and static-to-rotation transitions ever.
But V9c had two bugs:
  1. Curriculum stuck at bucket 0 forever -- running average was dragged
     down by early poor performance (now fixed: snapshot evaluation)
  2. Waist oscillation +/-0.06 (vs V8c +/-0.028) -- waist penalty too weak

Fix:
  - speed_bucketed_vel_curriculum: now uses snapshot evaluation matching
    lin_vel_cmd_levels pattern (current batch only, no accumulation)
  - waist_joint_vel: -0.04 -> -0.08 (2x stronger, targeting +/-0.03)

Everything else same as V9c:
  - Initial lin_vel range: +/-0.1 (curriculum expands)
  - Initial ang_vel range: +/-0.3 (tied to bucket)
  - rel_standing: 0.20, rel_rotating: 0.30
  - Runner: BasePPORunnerV3Cfg
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


# ---------- Curriculum: fixed snapshot-based bucketed ----------
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


# ---------- Commands: narrow initial range (same as V9c) ----------
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
