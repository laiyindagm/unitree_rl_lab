"""15-DOF rotation config V15b — "Adaptive Command Sampling".

Combines V15a's surgical reward adjustments with performance-weighted
adaptive command sampling (001.md Plan B implementation).

Core idea: Instead of uniform sampling within command ranges, track per-speed-bin
tracking performance via EMA, then oversample bins where the policy performs
worst. This automatically discovers that low-speed is harder than mid-speed
(contradicting the old bucketed curriculum's assumption that difficulty
increases with speed).

Key components:
  1. PerformanceWeightedVelocityCommand: 5 speed bins x 3 yaw bins, softmax
     sampling inversely proportional to performance
  2. performance_weighted_vel_curriculum: feeds reward data back to command,
     expands ranges only when ALL bins exceed threshold
  3. V15a reward changes (surgical rebalance)

Wider initial ranges than V10d to give adaptive sampler more to work with:
  lin_vel_x: (-0.3, 0.8) instead of (-0.1, 0.1)
  ang_vel_z: (-0.5, 0.5) instead of (-0.3, 0.3)
The adaptive sampler will automatically focus on problematic sub-ranges.

Changes from V15a:
  - Replace UniformLevelVelocityCommand with PerformanceWeightedVelocityCommand
  - Replace speed_bucketed curriculum with performance_weighted_vel_curriculum
  - Wider initial command ranges
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Commands: Performance-weighted adaptive sampling ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.PerformanceWeightedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.20,
        rel_rotating_envs=0.30,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.PerformanceWeightedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.8),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.PerformanceWeightedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
        # Adaptive sampling hyperparameters
        num_speed_bins=5,
        num_yaw_bins=3,
        ema_alpha=0.1,
        temperature=2.0,
        min_sampling_prob=0.05,
    )


# ---------- Curriculum: performance-weighted ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.performance_weighted_vel_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "range_expand_threshold": 0.5,
        },
    )


# ---------- Env ----------
@configclass
class RobotEnvCfg(V15aEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
