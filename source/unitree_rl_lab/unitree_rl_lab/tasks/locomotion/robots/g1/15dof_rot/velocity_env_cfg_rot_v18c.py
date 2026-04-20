"""15-DOF rotation config V18c — "Pure-Linear Allocation".

V18a (fixed adaptive continuous) + 20% pure-linear envs.

Design rationale:
  V17a/V18a have 25% pure-rotation envs (wz only, vx=0) which successfully
  solved rotation dead zone (from 0.1). But lin_vel dead zone persists at 0.4.
  Hypothesis: dedicated linear practice (vx only, wz=0) should help lin_vel
  the same way dedicated rotation practice helped ang_vel.

  Pure-rotation envs: wz is adaptively sampled from marginal P(wz) = Σ_vx P(vx,wz).
  Pure-linear envs: vx is adaptively sampled from marginal P(vx) = Σ_wz P(vx,wz).
  So both dedicated allocations respect the adaptive difficulty weighting.

Allocation:
  rel_standing = 0.15 (same as V18a)
  rel_rotating = 0.25 (same as V18a — pure rotation practice)
  rel_linear   = 0.20 (NEW — pure linear velocity practice)
  Normal (combined vx+wz) ≈ remaining ~40%

  Pure-linear envs only track lin_vel accuracy (has_ang=False since wz=0).
  They are excluded from bin EMA updates (same as rotating).

Hypothesis: dedicated linear practice → more concentrated vx training →
lin_vel dead zone reduced from 0.4 to ≤0.2.
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v18a import (
    RobotEnvCfg as V18aEnvCfg,
    RobotPlayEnvCfg as V18aPlayEnvCfg,
)


@configclass
class RobotEnvCfg(V18aEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_linear_envs = 0.20


@configclass
class RobotPlayEnvCfg(V18aPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_linear_envs = 0.20
