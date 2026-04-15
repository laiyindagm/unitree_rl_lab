"""15-DOF rotation config V15a -- "Performance-Weighted Adaptive Command Sampling".

Replaces the bucketed curriculum (which assumes monotonic difficulty
low→mid→high) with a performance-weighted sampling system that discovers
empirically which speed/yaw regions the policy struggles with and
automatically allocates more training there.

Key insight: low-speed locomotion is often *harder* than medium-speed
for humanoids because the Gaussian tracking kernel gives "free reward"
for standing still, and maintaining balance at low speed requires more
precise control than at medium speed where momentum helps. A fixed
low→high curriculum therefore wastes time on easy medium speeds while
starving the harder low-speed regime.

The new system:
  - Divides speed magnitude into N bins (default 5) and |yaw| into M bins
    (default 3).
  - Tracks per-bin normalised tracking reward via EMA.
  - On each command resample, chooses bins with probability inversely
    proportional to performance (worse → more samples).
  - Expands the active command range toward limit_ranges only when the
    *worst* bin across all current bins exceeds a threshold — ensuring
    mastery across the full active range before expanding.

Built on V14b rewards (penalty rebalance + speed-gated rotation penalties).
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v14b import (
    RewardsCfg as V14bRewardsCfg,
    RobotEnvCfg as V14bEnvCfg,
)


# ---------- Commands: performance-weighted adaptive sampling ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    base_velocity = mdp.PerformanceWeightedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.15,
        rel_rotating_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # Start with full range from the beginning — the adaptive sampler
        # will automatically focus on whatever the policy finds hardest
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
        # Adaptive sampling parameters
        num_speed_bins=5,
        num_yaw_bins=3,
        ema_alpha=0.05,
        min_sampling_prob=0.05,
        temperature=1.0,
    )


# ---------- Rewards: same as V14b ----------
@configclass
class RewardsCfg(V14bRewardsCfg):
    pass


# ---------- Curriculum: performance-weighted (replaces bucketed) ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.performance_weighted_vel_curriculum,
        params={
            "lin_reward_term_name": "track_lin_vel_xy",
            "ang_reward_term_name": "track_ang_vel_z",
            "range_expand_threshold": 0.5,
            "range_expand_delta": 0.1,
        },
    )


# ---------- Env ----------
@configclass
class RobotEnvCfg(V14bEnvCfg):
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
