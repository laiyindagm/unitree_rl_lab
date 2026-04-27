"""15-DOF rotation config V20i — "V20f without gait_mode_token".

Single-axis A/B over V20f testing the deepest hypothesis behind
pure_vx/pure_vy gait failure: the gait_mode_token (5-dim one-hot)
enforces per-mode subpolicy specialization that breaks transfer of
yaw control from joint-mode envs to pure-linear-mode envs.

Mechanism (verified by reading rewards.py + observations.py):
  - track_ang_vel_z_rotating_aware: SKIP triggers on cmd_yaw < 0.05.
    Returns 1.0 (free reward) for ALL envs with cmd_yaw=0, regardless
    of token. Total wz tracking weight 3.0 + 1.0 = 4.0 contributes ZERO
    yaw-control gradient in these envs.
  - wz_proportional_penalty: has_cmd = cmd_wz_abs > 0.08. Returns 0
    when cmd_yaw=0. Also ZERO yaw signal.
  - Therefore: ALL pure_vx (cmd=(vx,0,0)), pure_vy (cmd=(0,vy,0)), AND
    joint envs with cmd_yaw=0 (e.g. (vx,vy,0)) receive ZERO yaw-control
    reward signal.

Why does this matter differently across modes?
  - Without gait_mode_token (V19f baseline): all envs share one policy.
    Some joint envs DO have cmd_wz != 0 (because wz_bins lacks a 0 entry
    -- joint envs sample wz from non-zero bins). These envs train the
    SHARED policy in yaw control. The learned yaw-control skill applies
    (via shared parameters) to pure_vx/pure_vy/joint-with-zero-wz envs
    too. Result: stable gait everywhere.
  - With gait_mode_token (V20c+, V20f): one-hot token routes envs to
    de-facto subpolicies (network output near-discontinuous in token
    dimension). pure_vx subpolicy ONLY receives gradients from envs
    classified as pure_vx -- 100% of which have cmd_yaw=0 -- so its
    reward landscape NEVER includes yaw-control signal. The subpolicy
    cannot learn yaw control. Same for pure_vy.
  - User's empirical observation aligns: "any 2+ axis combo walks fine"
    because joint token routes to a subpolicy that DOES include some
    cmd_wz!=0 envs in its training distribution. "Pure vx/vy fails"
    because those subpolicies only ever see cmd_yaw=0 envs.

User correction (2026-04-27): "data-density boost (V20h) cannot work
because more samples of zero-gradient data don't help." Correct.
The bottleneck is the structural information isolation that
gait_mode_token enforces between modes with asymmetric reward signals.

V20i fix: revert observations to base (V19f) ObservationsCfg --
removes gait_mode_token from both PolicyCfg and CriticCfg. Keeps V20f's
linear-side rotation_skip removal (which still benefits all envs).

NOT changed vs V20f: all rewards (wz tracking still uses
_rotating_aware), command bins, curriculum, DR, std/weight values.

Predictions @ iter 12-15k:
  - sim2sim pure_vx and pure_vy walk smoothly (matches "with-wz" quality)
  - error_vel_xy < 0.40, error_vel_yaw < 0.30
  - bad_orient unchanged
  - Loses theoretical sample-complexity advantage of token, but
    empirical evidence > theory.

Risks:
  - C++ deploy code registered gait_mode_token observation. Must
    deploy V19f-style policies (no token) to match this config. The
    REGISTER_OBSERVATION block remains harmless; it only matters if
    the policy obs vector includes the token slot.

Decision tree:
  - V20i works (pure xy walks) -> token routing was the root cause;
    keep token-free design.
  - V20i fails -> token is not the bottleneck; some other reward
    structure issue. Investigate per-axis curriculum interaction.
  - V20i works AND V20g (independent) also works -> both fixes valid;
    choose simpler (V20i removes a feature, lower risk surface).
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot import ObservationsCfg as BaseObservationsCfg
from .velocity_env_cfg_rot_v20f import RobotEnvCfg as V20fEnvCfg


@configclass
class RobotEnvCfg(V20fEnvCfg):
    # Replace V20c's gait_mode_token-augmented observations with the
    # base (V19f) ObservationsCfg. All other V20f changes are inherited
    # (linear-side rotation_skip removal, curriculum, etc.).
    observations: BaseObservationsCfg = BaseObservationsCfg()


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
