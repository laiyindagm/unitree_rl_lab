"""15-DOF rotation config V19b — "Marginal Discrete (Axis-Independent)".

Fundamental architecture change: axis-INDEPENDENT 1D marginal sampling
replaces 2D combined-accuracy cell grid.

V18 Root Cause Analysis:
  Combined accuracy (lin+ang) dilutes axis-specific dead-zone signals.
  vx marginal prob max/min ratio was only 1.13x (nearly uniform!).
  perf spread across vx bins = 0.043 (should be 0.50+).
  The 8×8 grid smears everything → adaptive sampling sees ~uniform difficulty.

V19 Solution:
  - vx, vy, wz each have SEPARATE 1D performance EMA and softmax sampling
  - Per-axis accuracy: vx_acc uses |vx_cmd - vx_actual|, independent of vy/wz
  - Per-direction curriculum: vx_pos, vx_neg, vy, wz expand independently
    using mean(perf) > threshold (not min)
  - vy fully included (was uniform in V18 — contributes to dead zone)
  - V19b: Discrete 0.1 granularity (V19a is continuous variant)

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


# ---------- Commands: marginal axis-independent discrete bins ----------
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
            lin_vel_x=(-0.3, 0.3),   # initial range (fallback)
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3),
        ),
        # --- vx_pos: 15 discrete levels (0.1 to 1.5) ---
        vx_pos_bins=[
            (0.1, 0.1), (0.2, 0.2), (0.3, 0.3),    # tier 1 (initial)
            (0.4, 0.4), (0.5, 0.5),                   # tier 2
            (0.6, 0.6), (0.7, 0.7),                   # tier 3
            (0.8, 0.8), (0.9, 0.9),                   # tier 4
            (1.0, 1.0), (1.1, 1.1),                   # tier 5
            (1.2, 1.2), (1.3, 1.3),                   # tier 6
            (1.4, 1.4), (1.5, 1.5),                   # tier 7
        ],
        # --- vx_neg: 8 discrete levels (-0.1 to -0.8) ---
        vx_neg_bins=[
            (-0.1, -0.1), (-0.2, -0.2), (-0.3, -0.3),  # tier 1 (initial)
            (-0.4, -0.4), (-0.5, -0.5),                  # tier 2
            (-0.6, -0.6), (-0.7, -0.7),                  # tier 3
            (-0.8, -0.8),                                 # tier 4
        ],
        # --- vy: 11 levels (0, ±0.1..±0.5, zero first) ---
        vy_bins=[
            (0.0, 0.0),                                  # standing vy=0
            (0.1, 0.1), (-0.1, -0.1),                    # ±0.1 (initial)
            (0.2, 0.2), (-0.2, -0.2),                    # ±0.2 (initial)
            (0.3, 0.3), (-0.3, -0.3),                    # ±0.3 (initial)
            (0.4, 0.4), (-0.4, -0.4),                    # tier 2
            (0.5, 0.5), (-0.5, -0.5),                    # tier 3
        ],
        # --- wz: 16 levels (±0.1..±0.8, pairs) ---
        wz_bins=[
            (0.1, 0.1), (-0.1, -0.1),                    # ±0.1 (initial)
            (0.2, 0.2), (-0.2, -0.2),                    # ±0.2 (initial)
            (0.3, 0.3), (-0.3, -0.3),                    # ±0.3 (initial)
            (0.4, 0.4), (-0.4, -0.4),                    # tier 2
            (0.5, 0.5), (-0.5, -0.5),                    # tier 3
            (0.6, 0.6), (-0.6, -0.6),                    # tier 4
            (0.7, 0.7), (-0.7, -0.7),                    # tier 5
            (0.8, 0.8), (-0.8, -0.8),                    # tier 6
        ],
        # Active bins for staged curriculum
        num_active_vx_pos=3,   # 0.1, 0.2, 0.3
        num_active_vx_neg=3,   # -0.1, -0.2, -0.3
        num_active_vy=7,       # 0, ±0.1, ±0.2, ±0.3
        num_active_wz=6,       # ±0.1, ±0.2, ±0.3
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
