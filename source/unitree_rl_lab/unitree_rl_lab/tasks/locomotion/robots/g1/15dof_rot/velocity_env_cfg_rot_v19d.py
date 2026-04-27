"""15-DOF rotation config V19d — "Direction-Gated Accuracy + wz Boost".

V19c Remaining Issues:
  1. NOISE INFLATION PERSISTS: Pure-env accuracy still inflated for small commands.
     Base velocity noise (~0.03-0.05 m/s) gives relative accuracy ~0.3-0.5 for
     cmd=0.1 even when robot stands still. Result: vx expanded to 24/24 instantly.
  2. SAMPLING NOT FOCUSED: temperature=3.0 with softmax gives only 2x ratio between
     worst/best bins. min_sampling_prob=0.02 with 23 bins locks 46% to uniform.
  3. wz CAN'T LEARN: track_ang_vel_z(weight=1.0) competes against
     joint_deviation_legs(weight=-1.0) on hip_yaw — the very joint needed for rotation.
     Net gradient too weak. In pure_wz envs, track_lin_vel_xy OPPOSES rotation
     (rotation creates parasitic xy velocity → reward drops).

V19d Fixes:
  1. DIRECTION-GATED ACCURACY: accuracy counts only when actual velocity in commanded
     direction exceeds min_response_speed=0.05. Noise doesn't pass gate → perf≈0
     for non-responsive bins. Eliminates inflation entirely.
  2. MIN-BASED EXPANSION: curriculum uses min(bin_perfs) > threshold instead of mean.
     Every bin must genuinely respond before expansion. Threshold=0.3 (safe with
     direction-gated accuracy since perfs are no longer inflated).
  3. FOCUSED SAMPLING: temperature=8.0 (from 3.0, ~10x ratio worst/best) and
     min_sampling_prob=0.01 (from 0.02, 23% floor instead of 46%).
  4. wz BOOST: track_ang_vel_z weight=2.0 (from 1.0), rotating envs=20% (from 15%).
  5. Standing: 8% (from 5%).

Distribution: 8% standing, 15% pure_vx, 15% pure_vy, 20% pure_wz, 42% joint.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
    RewardsCfg as RotRewardsCfg,
)
from .velocity_env_cfg_rot_v16b import (
    RewardsCfg as V16bRewardsCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Commands: direction-gated + focused sampling ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.MarginalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.08,
        rel_rotating_envs=0.20,    # pure_wz (boosted)
        rel_pure_vx_envs=0.15,
        rel_pure_vy_envs=0.15,
        rel_linear_envs=0.0,
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
        # Adaptive parameters — much more focused
        ema_alpha=0.1,
        temperature=8.0,           # 8.0 from 3.0: ~10x ratio worst/best
        min_sampling_prob=0.01,    # 0.01 from 0.02: 23% floor instead of 46%
        accuracy_cmd_min=0.05,
        zero_speed_accuracy_threshold=0.10,
        min_response_speed=0.05,   # Direction gate: noise < 0.05 doesn't count
    )


# ---------- Rewards: wz boost + anti-standing ----------
@configclass
class RewardsCfg(V16bRewardsCfg):
    # Boost angular velocity tracking to overcome hip_yaw deviation penalty
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rotating_aware,
        weight=0.5 * 2 * 2,    # 2.0 (from 1.0): overcome joint_deviation_legs
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Override movement_incentive with higher weight
    movement_incentive = RewTerm(
        func=mdp.movement_incentive_scheduled,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "cmd_threshold": 0.05,
            "start_step": 48000,    # ~iter 2000
            "end_step": 96000,      # ~iter 4000
        },
    )
    # Stronger direct nonresponse penalty
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.08,
            "vel_threshold": 0.05,
        },
    )


# ---------- Curriculum: min-based + lower threshold ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.marginal_vel_curriculum,
        params={
            "range_expand_threshold": 0.3,   # Lower: direction-gated perfs are honest
            "min_perf_floor": 0.1,           # Floor for warning only
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
        # Evaluation: all bins active, no direction gate
        self.commands.base_velocity.num_active_vx_pos = None
        self.commands.base_velocity.num_active_vx_neg = None
        self.commands.base_velocity.num_active_vy = None
        self.commands.base_velocity.num_active_wz = None


# ---------- CLP variants: extend observation history to 10 frames ----------
# CLP encoder (TCN with dilations [1,2,4] or scaled Transformer) needs a richer
# temporal context than the default 5-frame history.  Override only history_length
# so policy/critic obs become 10*54 = 540 dims (matches model._latent_dim=744
# = 540 + 96 + 108).

@configclass
class RobotEnvCfgCLP(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10
        self.observations.critic.history_length = 10


@configclass
class RobotPlayEnvCfgCLP(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10
        self.observations.critic.history_length = 10
