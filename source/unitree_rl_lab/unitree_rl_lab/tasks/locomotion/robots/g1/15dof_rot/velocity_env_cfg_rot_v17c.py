"""15-DOF rotation config V17c — "Discrete Adaptive Sampling".

Based on V16a's adaptive framework, but with DISCRETE velocity levels
instead of continuous bin ranges. Each command level is its own bin
(lo == hi → zero-width → sampling gives exact value).

Key design:
  1. Discrete levels: vx ∈ {0, ±0.1, ±0.2, ±0.3, ±0.5, ±0.8, 1.0, 1.2, 1.5}
                      wz ∈ {0, ±0.1, ±0.2, ±0.3, ±0.5, ±0.8}
  2. Bins ordered CENTER-OUTWARD: (0,0) first, then ±0.1, ±0.2, etc.
     This enables staged curriculum — start with inner bins, expand outward.
  3. Staged curriculum: initially 5 vx × 5 wz = 25 active cells (speeds ≤ 0.2)
     When all active bins exceed 0.5 accuracy, activate next 2 bins per axis.
     Expansion: {0,±0.1,±0.2} → {+±0.3} → {+±0.5} → {+±0.8} → {+1.0,1.2,1.5}
  4. Zero-speed bin (0,0): accuracy = clamp(1 - vel/0.05, 0, 1)
     Perfectly still = 1.0. Standing/rotating envs route to this bin.
  5. Adaptive sampling: per-cell accuracy EMA, worse cells get more sampling.
  6. Movement incentive from V16b to break standing optimum.

Compared to V16a continuous bins:
  - No dead zone gap [-0.05, 0.05] — replaced by explicit level 0
  - Each level gets concentrated training (no continuous interpolation)
  - Training distribution matches joystick deployment exactly
  - Curriculum adds higher speeds progressively (not widening ranges)

Total cells: 14 vx × 11 wz = 154 cells (25 initially active).
Full expansion covers vx ∈ [-0.8, 1.5], wz ∈ [-0.8, 0.8] — same as V17a.

Hypothesis: discrete levels + adaptive sampling + staged curriculum
→ policy learns each speed level precisely, zero dead zone.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v16b import (
    RewardsCfg as V16bRewardsCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Commands: discrete adaptive bins (center-outward) ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.PerformanceWeightedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.15,
        rel_rotating_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.PerformanceWeightedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2),   # initial range (updated by curriculum)
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
        ),
        limit_ranges=mdp.PerformanceWeightedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
        # Discrete bins: ordered CENTER-OUTWARD for staged curriculum.
        # Each bin is (val, val) → sampling gives exactly val.
        # Full expansion: vx ∈ [-0.8, 1.5], wz ∈ [-0.8, 0.8] (same as V17a)
        vx_bins=[
            (0.0,  0.0),    # 0: standing
            (-0.1, -0.1),   # 1: slow backward
            (0.1,  0.1),    # 2: slow forward
            (-0.2, -0.2),   # 3: medium backward
            (0.2,  0.2),    # 4: medium forward
            # --- initially active up to here (num_active_vx_bins=5) ---
            (-0.3, -0.3),   # 5: tier 2
            (0.3,  0.3),    # 6: tier 2
            (-0.5, -0.5),   # 7: tier 3
            (0.5,  0.5),    # 8: tier 3
            (-0.8, -0.8),   # 9: tier 4
            (0.8,  0.8),    # 10: tier 4
            (0.9,  0.9), 
            (1.0,  1.0),    # 11: tier 5 (fast forward)
            (1.1,  1.1),
            (1.2,  1.2),    # 12: tier 5
            (1.3,  1.3),
            (1.4,  1.4),
            (1.5,  1.5),    # 13: tier 5 (sprint)
        ],
        wz_bins=[
            (0.0,  0.0),    # 0: no rotation
            (-0.1, -0.1),   # 1: slow CW
            (0.1,  0.1),    # 2: slow CCW
            (-0.2, -0.2),   # 3: medium CW
            (0.2,  0.2),    # 4: medium CCW
            # --- initially active up to here (num_active_wz_bins=5) ---
            (-0.3, -0.3),   # 5: tier 2
            (0.3,  0.3),    # 6: tier 2
            (-0.5, -0.5),   # 7: tier 3
            (0.5,  0.5),    # 8: tier 3
            (-0.8, -0.8),   # 9: tier 4
            (0.8,  0.8),    # 10: tier 4
        ],
        ema_alpha=0.1,
        temperature=2.0,
        min_sampling_prob=0.015,
        accuracy_cmd_min=0.05,      # lower threshold for small discrete levels
        zero_speed_accuracy_threshold=0.05,
        # Staged curriculum: start with 5 bins (speeds ≤ 0.2)
        num_active_vx_bins=5,
        num_active_wz_bins=5,
    )


# ---------- Rewards: V16b movement incentive ----------
@configclass
class RewardsCfg(V16bRewardsCfg):
    # Inherit V16b's movement_incentive_scheduled
    # Default schedule: start=72000 (~iter3000), end=120000 (~iter5000)
    pass


# ---------- Curriculum: performance-weighted with staged expansion ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.performance_weighted_vel_curriculum,
        params={
            "range_expand_threshold": 0.5,
        },
    )


# ---------- Env ----------
@configclass
class RobotEnvCfg(V15aEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Evaluation: all bins active, full range
        self.commands.base_velocity.num_active_vx_bins = None
        self.commands.base_velocity.num_active_wz_bins = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
