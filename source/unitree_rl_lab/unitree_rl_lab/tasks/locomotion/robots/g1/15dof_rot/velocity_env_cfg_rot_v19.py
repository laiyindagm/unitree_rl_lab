"""15-DOF rotation config V19a — "Marginal Continuous (Axis-Independent)".

Fundamental architecture change: axis-INDEPENDENT 1D marginal sampling
replaces 2D combined-accuracy cell grid.

V18 Root Cause Analysis:
  Combined accuracy (lin+ang) dilutes axis-specific dead-zone signals.
  vx marginal prob max/min ratio was only 1.13x (nearly uniform!).
  perf spread across vx bins = 0.043 (should be 0.50+).
  The 8×8 grid smears everything → adaptive sampling sees ~uniform difficulty.

V19a Solution (CONTINUOUS variant):
  - vx, vy, wz each have SEPARATE 1D performance EMA and softmax sampling
  - Per-axis accuracy: vx_acc uses |vx_cmd - vx_actual|, independent of vy/wz
  - Per-direction curriculum: vx_pos, vx_neg, vy, wz expand independently
    using mean(perf) > threshold (not min)
  - vy fully included (was uniform in V18 — contributes to dead zone)
  - Continuous bin ranges (width 0.1 for low speeds, wider for high speeds)
  - V19b is the discrete (lo==hi) variant with 0.1 granularity

Expected improvement: axis-specific perf spread ~0.50 (vs V18's 0.043).
This should give real adaptive signal for vx dead zone.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
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


# ---------- Commands: marginal axis-independent continuous bins ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.MarginalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.15,
        rel_rotating_envs=0.25,
        rel_linear_envs=0.20,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.MarginalVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.35, 0.35),   # initial range (fallback)
            lin_vel_y=(-0.35, 0.35),
            ang_vel_z=(-0.35, 0.35),
        ),
        # --- vx_pos: 8 continuous bins (0.05 to 1.5) ---
        vx_pos_bins=[
            (0.05, 0.15), (0.15, 0.25), (0.25, 0.35),  # tier 1 (initial)
            (0.35, 0.50), (0.50, 0.70),                  # tier 2
            (0.70, 0.90), (0.90, 1.20),                  # tier 3
            (1.20, 1.50),                                 # tier 4
        ],
        # --- vx_neg: 5 continuous bins (-0.8 to -0.05) ---
        vx_neg_bins=[
            (-0.15, -0.05), (-0.25, -0.15), (-0.35, -0.25),  # tier 1 (initial)
            (-0.50, -0.35),                                    # tier 2
            (-0.80, -0.50),                                    # tier 3
        ],
        # --- vy: 9 levels (zero region + ±pairs, zero first) ---
        vy_bins=[
            (-0.05, 0.05),                                    # zero region
            (0.05, 0.15), (-0.15, -0.05),                     # ±0.1 (initial)
            (0.15, 0.25), (-0.25, -0.15),                     # ±0.2 (initial)
            (0.25, 0.35), (-0.35, -0.25),                     # ±0.3 (initial)
            (0.35, 0.50), (-0.50, -0.35),                     # tier 2
        ],
        # --- wz: 12 continuous bins (±pairs, by magnitude) ---
        wz_bins=[
            (0.05, 0.15), (-0.15, -0.05),                     # ±0.1 (initial)
            (0.15, 0.25), (-0.25, -0.15),                     # ±0.2 (initial)
            (0.25, 0.35), (-0.35, -0.25),                     # ±0.3 (initial)
            (0.35, 0.50), (-0.50, -0.35),                     # tier 2
            (0.50, 0.65), (-0.65, -0.50),                     # tier 3
            (0.65, 0.80), (-0.80, -0.65),                     # tier 4
        ],
        # Active bins for staged curriculum (same initial coverage as V19b)
        num_active_vx_pos=3,   # up to 0.35
        num_active_vx_neg=3,   # down to -0.35
        num_active_vy=7,       # zero + ±0.15..±0.35
        num_active_wz=6,       # ±0.15..±0.35
        # Adaptive parameters
        ema_alpha=0.1,
        temperature=3.0,
        min_sampling_prob=0.02,
        accuracy_cmd_min=0.05,
        zero_speed_accuracy_threshold=0.10,
    )


# ---------- Rewards: V16b movement incentive ----------
@configclass
class RewardsCfg(V16bRewardsCfg):
    pass


# ---------- Curriculum: marginal per-direction expansion ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.marginal_vel_curriculum,
        params={
            "range_expand_threshold": 0.3,
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
        self.commands.base_velocity.num_active_vx_pos = None
        self.commands.base_velocity.num_active_vx_neg = None
        self.commands.base_velocity.num_active_vy = None
        self.commands.base_velocity.num_active_wz = None
