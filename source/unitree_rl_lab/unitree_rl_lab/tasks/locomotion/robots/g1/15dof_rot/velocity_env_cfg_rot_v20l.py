"""15-DOF rotation config V20l — "V20g + 3-mode token".

Next experiment after V20i/V20j, with the user's hard constraint kept:
0-command, pure_w, and other must remain isolated.

Observed facts:
  - V20i (no token) reproduced old V19 behavior: zero-command standing became
    unstable and walking quality did not improve. This falsifies the idea that
    simply removing token isolation solves the problem.
  - V20j (3-mode token on top of V20f) kept {standing, pure_wz, other}, but
    pure_vx/pure_vy still failed and joint walking degraded. That shows the
    problem is not token existence; it is the reward landscape inside pure_xy.

Root-cause hypothesis:
  Pure_vx/pure_vy envs are structurally different from joint envs. Pure-linear
  envs have cmd_wz == 0 by construction, but V20f/V20j still use
  track_ang_vel_z_rotating_aware, which returns a free +1.0 for straight walk
  (cmd_lin_norm > 0.1 and |cmd_wz| < 0.05). With main+sharp yaw trackers, that
  is +4.0 yaw reward with no gradient toward actual_wz -> 0. The policy can use
  yaw/waist wobble as a cheap way to satisfy linear tracking.

  Joint envs often include cmd_wz != 0, so they receive active yaw-tracking
  gradients. This explains the asymmetry: joint succeeds because it has yaw
  supervision; pure_xy fails because yaw is unsupervised while dynamically
  coupled to the legs and waist.

V20l fix:
  Keep V20j's required 3-way isolation ({standing, pure_wz, other}), but inherit
  V20g's reward stack where both yaw trackers use standard track_ang_vel_z_exp.
  Thus cmd_wz == 0 pure-linear samples must learn quiet yaw; cmd_wz != 0 joint
  samples still learn commanded yaw; standing and pure_wz keep dedicated token
  branches.

Single-variable interpretation:
  Compared with V20j, only yaw reward changes (rotating-aware skip -> standard
  exp). Compared with V20g, only token granularity changes (5-mode -> 3-mode).
  This tests whether pure_xy fails because cmd_wz==0 samples lacked yaw
  supervision, not because token isolation exists.

Predictions @ iter 8-12k:
  - waist_roll_vel should move from about -0.18 toward > -0.10.
  - error_vel_yaw should drop below 0.35 before pure_xy improves.
  - standing should improve vs V20i because standing is isolated again.
  - joint walking should recover vs V20j because pure_xy no longer injects a
    yaw-free wobble optimum into the shared "other" branch.

Kill conditions:
  - iter 5k: bad_orientation > 8% or mean reward < 55.
  - iter 8k: waist_roll_vel < -0.15 and error_vel_yaw > 0.45 together.
  - iter 12k: pure_xy sim2sim still fails while joint is good -> add explicit
    no-yaw linear penalty instead of changing token structure.
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v20g import RobotEnvCfg as V20gEnvCfg
from .velocity_env_cfg_rot_v20j import ObservationsCfg


@configclass
class RobotEnvCfg(V20gEnvCfg):
    # Keep selective isolation: {standing, pure_wz, other}. Rewards are inherited
    # from V20g, so cmd_wz==0 pure-linear envs receive real yaw supervision.
    observations: ObservationsCfg = ObservationsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Play: expose the full curriculum-extended command range.
        # V19f bin counts: 15 vx_pos, 8 vx_neg, 11 vy, 16 wz.
        self.commands.base_velocity.num_active_vx_pos = 15
        self.commands.base_velocity.num_active_vx_neg = 8
        self.commands.base_velocity.num_active_vy = 11
        self.commands.base_velocity.num_active_wz = 16
