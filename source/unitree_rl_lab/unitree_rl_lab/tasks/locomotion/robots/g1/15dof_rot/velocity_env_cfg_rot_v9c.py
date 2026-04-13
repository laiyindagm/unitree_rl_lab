"""15-DOF rotation config V9c — "Speed-Bucketed Curriculum" (Plan D).

Problem: Linear cmd_levels curriculum doesn't distinguish speed regimes:
  - Policy can earn high reward at medium/high speed and pass expansion
    threshold while being terrible at low speed — the dead zone emerges
    because low-speed tracking has lower marginal reward gain than penalties.
  - V8b's adaptive-sigma tried to fix per-step reward: CURRICULUM DEADLOCK.

Fix (curriculum layer, no observation/reward change except waist):
  - Replace lin_vel_cmd_levels + ang_vel_cmd_levels with speed_bucketed_vel_curriculum
  - Bucket 0 (cmd<0.3): must achieve 60% of tracking reward weight before unlock
  - Bucket 1 (0.3-0.6): must achieve 60% before expanding to full range
  - Bucket 2 (0.6+): full range (limit_ranges)
  - ang_vel expansion tied to bucket level (unlock rotation when walking established)
  - CommandsCfg: start with small lin_vel range +/-0.1 (first ~500 episodes are
    near-standstill only), rel_standing=0.20 (more standing practice)

Changes from V6c:
  Curriculum:
    - Remove lin_vel_cmd_levels, ang_vel_cmd_levels
    - Add speed_bucketed_vel_curriculum
  Commands:
    - Initial lin_vel_x: [-0.1, 0.1] (very narrow start, curriculum will expand)
    - Initial lin_vel_y: [-0.1, 0.1]
    - Initial ang_vel_z: [-0.3, 0.3] (narrow, tied to bucket level)
    - rel_standing: 0.20 (more standing practice than V6c's 0.10)
    - rel_rotating: 0.30 (restored from V6b)
  Rewards:
    - waist_joint_vel_penalty: -0.04 (from V8c)
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


# ---------- Curriculum: speed-bucketed ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # Replace lin/ang cmd_levels with bucketed curriculum
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


# ---------- Commands: narrow initial range for curriculum ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    """V6b base with narrow initial ranges (curriculum will expand).

    Same limit_ranges as V6b, but initial ranges are very small so the
    bucketed curriculum controls progression.
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.20,  # more standing practice
        rel_rotating_envs=0.30,  # restored from V6b
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),   # narrow start — curriculum will expand
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.3, 0.3),   # narrow — tied to bucket level
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),   # same ceiling as V6b
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


# ---------- Rewards: V6c + waist ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.04,
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
