"""15-DOF rotation config V19c — "Pure-Env Accuracy + Stronger Anti-Standing".

Based on V19b (discrete marginal) with critical fixes:

V19a/V19b Root Cause Analysis:
  1. Low-speed vx perf inflated to ~0.43 (should be ~0.05) because:
     - Joint envs (50%+ of vx data): rotation side-effect creates 0.06 m/s
       translational velocity, "free" matching 40-60% of cmd_vx=0.1.
     - Relative accuracy formula: tiny body sway satisfies small commands.
  2. Curriculum expanded too early: inflated perf > 0.3 triggers expansion
     even though robot literally doesn't walk at vx < 0.3.
  3. vy also non-responsive: same inflation mechanism via joint envs.

V19c Fixes:
  1. PURE ENV TYPES: 15% pure_vx (only vx active), 15% pure_vy (only vy active),
     15% pure_wz (rotating). vx_perf ONLY tracks pure_vx envs → true accuracy
     without rotation contamination. Same for vy and wz.
  2. STRONGER anti-standing: cmd_nonresponse weight -2.0, movement_incentive
     weight -3.0 (from -0.5 and -1.0). Direct gradient to break standing.
  3. CURRICULUM FLOOR GATE: expansion blocked if ANY active bin has perf < 0.2.
     Prevents expansion while dead-zone bins are completely non-responsive.
  4. HIGHER EXPANSION THRESHOLD: 0.5 (from 0.3). With pure-env accuracy,
     perfs should correctly reflect true dead-zone difficulty.
  5. Reduced standing allocation: 5% (from 15%). More budget for pure envs.

Distribution: 5% standing, 15% pure_vx, 15% pure_vy, 15% pure_wz, 50% joint.
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


# ---------- Commands: discrete marginal + pure env types ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.MarginalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        rel_rotating_envs=0.15,    # pure_wz
        rel_pure_vx_envs=0.15,
        rel_pure_vy_envs=0.15,
        rel_linear_envs=0.0,       # deprecated, replaced by pure_vx/pure_vy
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.MarginalVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3),
        ),
        # --- vx_pos: 15 discrete levels (0.1 to 1.5) ---
        vx_pos_bins=[
            (0.1, 0.1), (0.2, 0.2), (0.3, 0.3),
            (0.4, 0.4), (0.5, 0.5),
            (0.6, 0.6), (0.7, 0.7),
            (0.8, 0.8), (0.9, 0.9),
            (1.0, 1.0), (1.1, 1.1),
            (1.2, 1.2), (1.3, 1.3),
            (1.4, 1.4), (1.5, 1.5),
        ],
        # --- vx_neg: 8 discrete levels (-0.1 to -0.8) ---
        vx_neg_bins=[
            (-0.1, -0.1), (-0.2, -0.2), (-0.3, -0.3),
            (-0.4, -0.4), (-0.5, -0.5),
            (-0.6, -0.6), (-0.7, -0.7),
            (-0.8, -0.8),
        ],
        # --- vy: 11 levels (0, ±0.1..±0.5) ---
        vy_bins=[
            (0.0, 0.0),
            (0.1, 0.1), (-0.1, -0.1),
            (0.2, 0.2), (-0.2, -0.2),
            (0.3, 0.3), (-0.3, -0.3),
            (0.4, 0.4), (-0.4, -0.4),
            (0.5, 0.5), (-0.5, -0.5),
        ],
        # --- wz: 16 levels (±0.1..±0.8) ---
        wz_bins=[
            (0.1, 0.1), (-0.1, -0.1),
            (0.2, 0.2), (-0.2, -0.2),
            (0.3, 0.3), (-0.3, -0.3),
            (0.4, 0.4), (-0.4, -0.4),
            (0.5, 0.5), (-0.5, -0.5),
            (0.6, 0.6), (-0.6, -0.6),
            (0.7, 0.7), (-0.7, -0.7),
            (0.8, 0.8), (-0.8, -0.8),
        ],
        # Staged curriculum: conservative initial range
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


# ---------- Rewards: stronger anti-standing ----------
@configclass
class RewardsCfg(V16bRewardsCfg):
    # Override movement_incentive with higher weight
    movement_incentive = RewTerm(
        func=mdp.movement_incentive_scheduled,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "cmd_threshold": 0.05,
            "start_step": 48000,    # ~iter 2000 (earlier ramp)
            "end_step": 96000,      # ~iter 4000
        },
    )
    # Stronger direct nonresponse penalty
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,   # lower threshold: respond to smaller cmds
            "vel_threshold": 0.05,
        },
    )


# ---------- Curriculum: higher threshold + floor gate ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.marginal_vel_curriculum,
        params={
            "range_expand_threshold": 0.5,
            "min_perf_floor": 0.2,
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
        # Evaluation: all bins active
        self.commands.base_velocity.num_active_vx_pos = None
        self.commands.base_velocity.num_active_vx_neg = None
        self.commands.base_velocity.num_active_vy = None
        self.commands.base_velocity.num_active_wz = None
