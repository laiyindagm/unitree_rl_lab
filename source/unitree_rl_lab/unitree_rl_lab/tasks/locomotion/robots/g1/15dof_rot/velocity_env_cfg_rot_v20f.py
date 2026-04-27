"""15-DOF rotation config V20f — "V20c + Eliminate Rotation-Skip Blind Spot".

Single-axis A/B over V20c (V19f + token):
  Replace \`track_lin_vel_xy_rotation_skip\` (which returns 1.0 during pure
  rotation, creating a zero-gradient blind spot for linear drift) with the
  IsaacLab-standard \`track_lin_vel_xy_yaw_frame_exp\` (penalizes ‖v_b - v_cmd‖²
  uniformly, so cmd=(0,0,wz) actively drives v_xy -> 0).

Rationale (vs V20d sim2sim @ ~18k iter):
  - V20d's pure_rotation_drift penalty produced episode reward only -0.0517
    (0.07% of total). At weight=-1.0 with linear ‖v_xy‖ form, gradient is
    20× weaker than the +1.0 free reward V19f's rotation_skip provides
    during the same envs. Result: half-rotation accumulates 1m drift in
    sim2sim despite the patch.
  - Root cause: rotation_skip is a STRUCTURAL blind spot, not a missing
    penalty. Adding more penalties stacks against +1.0 free reward;
    removing the +1.0 free reward and replacing it with standard tracking
    is geometrically correct (during pure base-frame rotation, ground-truth
    v_xy is 0 -- standard exp kernel gives reward=1.0 only when policy
    actually achieves that).
  - User insight: command-tracking success requires verifying both
    "commanded axes met goal" AND "zero-commanded axes are zero".
    V19f's rotation_skip violates the second criterion by construction.

Coupled removal: pure_rotation_drift IS NOT added. Standard yaw_frame_exp
already covers it (v_xy=0.3 m/s -> reward=exp(-0.36)=0.70 vs perfect 1.00,
gap=0.30 -- 6× stronger than V20d's -0.05 patch).

NOT changed vs V20c: wz tracking trio (track_ang_vel_z_rotating_aware,
_sharp, wz_proportional), all penalties (waist, torso, action_rate),
command bins, curriculum, DR, observations (token retained), std=0.5.

Predictions @ iter 12-15k:
  - sim2sim pure-rotation: |v_xy| < 0.05 m/s (drift eliminated)
  - error_vel_xy < 0.35 (off-axis leakage on vx/vy commands also reduced
    because std=0.5 kernel now penalizes any off-axis component everywhere)
  - bad_orient < 5%, ep_len ~ 1000 (no termination-path change)
  - wz tracking unchanged (its skip on the OTHER axis is intact;
    addressing it is V20g's job if needed)

Risks:
  - Standard yaw_frame_exp computes v in yaw-aligned frame, so during
    aggressive wz, v_xy_yaw_frame remains the linear drift component
    (NOT centripetal artifact). Geometrically correct.
  - If wz tracking regresses (track_ang_vel_z drops > 30%), the lin
    penalty is fighting wz incentive. Fallback: add pure_rotation_drift
    back at -0.5 in V20f-r1.

Decision tree:
  - sim2sim drift fixed AND vx/vy stagger reduced -> blind-spot hypothesis
    confirmed; proceed to V20g (same fix on wz side).
  - Drift fixed BUT wz regressed -> tension between lin and wz; fallback.
  - No improvement -> std=0.5 kernel still too wide; combine with V20e
    (std=0.30) in V20h.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v19f import RewardsCfg as V19fRewardsCfg
from .velocity_env_cfg_rot_v20c import RobotEnvCfg as V20cEnvCfg


@configclass
class RewardsCfg(V19fRewardsCfg):
    # Replace V19f's rotation_skip variant with standard yaw-frame exp.
    # Same weight (1.0) and std (sqrt(0.25)=0.5) as V19f -- only the
    # function changes, isolating the blind-spot fix.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
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
