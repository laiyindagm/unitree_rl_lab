"""15-DOF rotation config V12a -- "Scheduled Sigma Annealing".

Fix the dead zone by globally annealing sigma over training iterations.
Unlike V11a (adaptive sigma per-sample) which poisoned early training by
cutting 33% of the "free" standing reward, scheduled sigma keeps sigma=0.5
during Phase 1 (iter 0-3000) — identical to the proven V6c/V10d baseline.

Phase schedule:
  Phase 1 (iter 0-3000):    σ=0.5  — standard training, learn to stand/walk
  Phase 2 (iter 3000-8000): σ 0.5→0.3 (lin) / 0.5→0.25 (ang) — tighten tracking
  Phase 3 (iter 8000+):     σ=0.3/0.25 fixed — low-speed dead zone eliminated

Dead zone math at Phase 3 (σ=0.3):
  cmd=0.3: standing_reward = exp(-0.09/0.09) = 0.368
  marginal = 1.0 - 0.368 = 0.632 >> action_cost 0.327 → DEAD ZONE BROKEN

Base: V10d (bucketed curriculum + per-joint waist + V6c DR)
Changes: replace track_lin_vel_xy and track_ang_vel_z with scheduled sigma versions
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v10d import (
    CommandsCfg,
    CurriculumCfg,
    RewardsCfg as V10dRewardsCfg,
    RobotEnvCfg as V10dEnvCfg,
)


@configclass
class RewardsCfg(V10dRewardsCfg):
    """V10d rewards with scheduled sigma annealing for tracking."""

    # Replace standard exp tracking with scheduled sigma versions
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_scheduled_sigma,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sigma_start": math.sqrt(0.25),   # 0.5 — same as V6c/V10d baseline
            "sigma_end": 0.3,
            "anneal_start_iter": 3000,
            "anneal_end_iter": 8000,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_scheduled_sigma,
        weight=0.5 * 2.6,  # 1.3 — same as V6b/V10d
        params={
            "command_name": "base_velocity",
            "sigma_start": math.sqrt(0.25),   # 0.5
            "sigma_end": 0.25,                 # tighter for rotation precision
            "anneal_start_iter": 3000,
            "anneal_end_iter": 8000,
        },
    )


@configclass
class RobotEnvCfg(V10dEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
