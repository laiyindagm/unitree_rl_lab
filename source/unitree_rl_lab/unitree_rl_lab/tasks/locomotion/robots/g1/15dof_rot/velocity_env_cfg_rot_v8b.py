"""15-DOF rotation config V8b — "Adaptive-Sigma Tracking".

Novel approach: fix the dead zone at its mathematical root.

The dead zone (commands < 0.3 ignored) comes from exp(-error²/σ²) with fixed
σ=0.5. At cmd=0.2: doing_nothing→0.85, perfect→1.0, marginal gain=0.15.
This is less than action penalties (~0.25), so the agent rationally stays still.

Fix: make σ proportional to command magnitude: σ = max(σ_min, scale * |cmd|).
  At cmd=0.2 with σ_min=0.15, scale=0.5: σ = 0.15
    doing_nothing = exp(-0.04/0.0225) = 0.17
    perfect = 1.0, marginal = 0.83  >> action penalties
  At cmd=1.0: σ = 0.5 (same as original → no degradation at high speed)

This REPLACES the tracking rewards, no new penalties needed.
No change to reward budget → zero risk of V7-style catastrophic failure.

Changes from V6c:
  - track_lin_vel_xy: replaced with adaptive-sigma version (same weight 1.0)
  - track_ang_vel_z: replaced with adaptive-sigma version (same weight 1.3)
  - Everything else: V6c UNCHANGED
  Runner: BasePPORunnerV3Cfg
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v6c import (
    CommandsCfg,
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)


@configclass
class RewardsCfg(V6cRewardsCfg):
    """V6c rewards with adaptive-sigma tracking (only change)."""

    # --- Replace fixed-sigma tracking with adaptive-sigma ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_adaptive_sigma,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sigma_min": 0.15,
            "sigma_scale": 0.5,
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_adaptive_sigma,
        weight=0.5 * 2.6,
        params={
            "command_name": "base_velocity",
            "sigma_min": 0.15,
            "sigma_scale": 0.5,
        },
    )


@configclass
class RobotEnvCfg(V6cEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
