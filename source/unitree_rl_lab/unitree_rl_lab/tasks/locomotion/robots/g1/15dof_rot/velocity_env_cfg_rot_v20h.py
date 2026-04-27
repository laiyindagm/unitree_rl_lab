"""15-DOF rotation config V20h — "V20f + Boost Pure-Linear Sampling".

Parallel single-axis A/B vs V20f (NOT vs V20g):
  Increase rel_pure_vx_envs and rel_pure_vy_envs from 0.15 to 0.25 each
  (joint envs reduced from 0.35 to 0.15). All rewards, command bins, std,
  weights unchanged from V20f.

Rationale (V20f sim2sim correction from user):
  Tested behavior in token modes:
    pure_wz (cmd=(0,0,wz))    -> walks fine
    joint  (any 2+ axis cmd)  -> walks fine
    pure_vx (cmd=(vx,0,0))    -> FAILS (cannot walk straight, staggers)
    pure_vy (cmd=(0,vy,0))    -> FAILS

  Critical fact: V19f already allocates rel_pure_vx_envs=0.15 +
  rel_pure_vy_envs=0.15 = 30% of training envs. So failure is NOT lack of
  exposure to the distribution. Yet only the pure_vx/pure_vy gait_mode_token
  branches collapsed.

  Why these branches collapsed in V19f reward stack:
    - In pure_vx env (cmd=(vx,0,0)): track_ang_vel_z and _sharp use
      `_rotating_aware` which detects |cmd_lin|>0.1 AND |cmd_yaw|<0.05 -> 
      returns 1.0 (SKIP). Two terms (weight 3.0+1.0=4.0) become FREE reward.
    - Only wz_proportional(-4.0, cmd_threshold=0.08) gives wz pressure, but
      it's weak (linear in |wz|, not exp).
    - Net effect: pure_vx subpolicy receives ~4× weaker wz supervision than
      joint subpolicy -> never learns to walk straight, just maintains
      forward momentum with arbitrary yaw drift.

  Joint token routes (vx, vy, 0) and (vx, 0, wz) etc. to a different
  subpolicy that received strong wz supervision (because most joint envs
  had wz != 0 due to wz_bins lacking a 0 entry). This explains why "any
  2+ axis combo works" while pure_vx and pure_vy fail.

V20h hypothesis (parallel to V20g):
  Even though pure_vx envs were 15%, the per-sample gradient quality was
  poor (weak wz signal). Boost their density to 25% each (50% total
  pure-linear) to compensate via sample count. If V20g's reward fix and
  V20h's sampling boost are both root causes, both should improve;
  V20i would combine them.

NOT changed vs V20f: all rewards (track_lin_vel_xy stays standard
yaw_frame_exp, wz tracking stays _rotating_aware), all weights/std,
command bin definitions, curriculum, DR, observations (token retained).

Predictions @ iter 12-15k:
  - sim2sim pure_vx and pure_vy walking: improved (less stagger)
  - error_vel_xy: comparable to V20f or slightly worse (more low-speed
    samples in distribution)
  - error_vel_yaw: unchanged (wz reward stack not changed)
  - bad_orient: unchanged
  - Joint mode: stable (15% sampling still adequate; was already learned)

Risks:
  - Joint subpolicy could weaken from undersampling -> regression in
    "good" modes. Monitor joint sim2sim qualitatively.
  - wz_proportional alone may still be too weak even with more samples.
    If so V20g (skip removal) is the necessary fix and this is
    insufficient on its own.

Decision tree:
  - V20g works AND V20h works -> reward-side AND data-side both contribute;
    combine in V20i.
  - V20g works AND V20h fails -> V19f wz skip was THE bottleneck; sampling
    is fine; V20g is the answer.
  - V20g fails AND V20h works -> sampling density was THE bottleneck;
    keep skip; V20h is the answer.
  - Both fail -> token routing itself is the pathology; V20j removes
    gait_mode_token and tries pure-curriculum approach.
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v20f import RobotEnvCfg as V20fEnvCfg


@configclass
class RobotEnvCfg(V20fEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Boost pure-linear envs from 30% to 50% total.
        # rel_standing(0.10) + rel_rotating(0.25) + pure_vx(0.25)
        # + pure_vy(0.25) = 0.85; joint = 0.15.
        self.commands.base_velocity.rel_pure_vx_envs = 0.25
        self.commands.base_velocity.rel_pure_vy_envs = 0.25


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
