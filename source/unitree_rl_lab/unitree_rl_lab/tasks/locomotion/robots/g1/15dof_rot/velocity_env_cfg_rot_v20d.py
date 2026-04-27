"""15-DOF rotation config V20d — "V20c + Anti-Drift".

Single-axis A/B over V20c (V19f + token):
  Add `pure_rotation_drift` reward term (weight=-1.0) targeting V19f's known
  pure-rotation backward-drift bug. Triggered only when commanded as pure
  rotation (cmd_lin<0.05 & |cmd_wz|>0.05). Does not affect zero-cmd standing,
  pure-linear walking, or joint commands.

Rationale (vs V20c sim2sim observation @ 19.6k iter):
  - V20c kept V19f's `track_lin_vel_xy_rotation_skip` which returns 1.0 during
    pure rotation -> zero penalty for ANY linear motion -> policy learned
    asymmetric leg push (= backward thrust) as cheapest rotation strategy.
  - V19g had `pure_rotation_drift=-1.5` and the doc reported "drift fixed",
    but V19g simultaneously reverted rotation_skip and weakened wz_proportional,
    coupling three changes. V20d isolates JUST the drift penalty (and at -1.0,
    not -1.5, leaving margin for V19f-style aggressive rotation incentive).

NOT changed vs V20c: rotation_skip, wz_proportional, track_ang_vel_z*,
sigma values, action_rate, waist penalties, command bins, curriculum, DR.

Predictions @ 12-15k iter:
  - sim2sim pure-rotation: base xy drift < 0.05 m/s (vs V20c >> 0.1 m/s)
  - bad_orient < 5%, ep_len ~ 1000 (no collapse — minimal change)
  - track_lin_vel_xy reward unchanged in non-rotation envs
  - wz tracking unchanged

Decision tree:
  - Drift fixed + other metrics held -> token+drift validated, proceed to V20f
  - Drift fixed BUT zero-speed lost / wz regressed -> reduce weight to -0.5
  - Full collapse -> drift weight is in tension with V19f reward zoo, abandon
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v19f import RewardsCfg as V19fRewardsCfg
from .velocity_env_cfg_rot_v20c import (
    ObservationsCfg as V20cObservationsCfg,
    RobotEnvCfg as V20cEnvCfg,
)


@configclass
class RewardsCfg(V19fRewardsCfg):
    pure_rotation_drift = RewTerm(
        func=mdp.pure_rotation_lin_drift,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "lin_threshold": 0.05,
            "yaw_threshold": 0.05,
        },
    )


@configclass
class RobotEnvCfg(V20cEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


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
