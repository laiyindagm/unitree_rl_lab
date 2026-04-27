"""15-DOF rotation config V20k — "V20i + Symmetric wz Tracking".

User intermediate result (V20i @ 9.8k iter) confirmed predictions:
  - waist_roll_vel = -0.1792 (matches V20j -0.1801) — yaw-wobble cheat
  - error_vel_xy = 0.6074 (worse than V20j 0.5050)
  - error_vel_yaw = 0.4736
  - bad_orientation 2.4% (standing degraded)
  - User: "0 指令不稳定，行走和之前 V19 差不多"

Diagnosis (revised, definitive):
  V20f's root pathology was NEVER about token routing. It was
  reward-structure asymmetry between lin and wz tracking:

  cmd_yaw≈0 envs experience:
    lin (V20f yaw_frame_exp):  STRICT — every component of v_b must match
    wz (V19f rotating_aware):  FREE   — SKIP returns 1.0 when cmd_yaw<0.05

  Optimal policy under this asymmetric landscape: wobble yaw to
  generate v_xy artifact (matches lin reward) without yaw cost
  (covered by SKIP). Fingerprint: waist_roll_vel < -0.15.

  V20f isolated this cheat to pure_xy subpolicy via 5-mode token.
  V20j collapsed it into "other" subpolicy → contaminated joint.
  V20i removed token → cheat became GLOBAL → all modes wobble.

  Standing (cmd_norm<0.1) failure under V20i is downstream of the
  same cheat: the global yaw-wobble policy doesn't cleanly turn off
  when cmd hits zero, so even with zero_cmd_body_vel(-1.5) the
  policy keeps oscillating.

V20k single-variable fix:
  Replace BOTH wz tracking terms (track_ang_vel_z and
  track_ang_vel_z_sharp) with the standard track_ang_vel_z_exp
  (no SKIP). Now cmd_yaw≈0 envs experience SYMMETRIC strict
  tracking on both axes:
    lin: standard yaw_frame_exp (V20f, retained)
    wz:  standard exp — penalizes any actual_wz when cmd_wz=0

  This destroys the yaw-wobble cheat at its source. Standing policy
  should re-stabilize as a side effect, since the dominant walking
  strategy no longer involves yaw oscillation.

NOT changed vs V20i:
  - All other rewards (waist_roll_vel, wz_proportional, gait,
    zero_cmd_body_vel, etc.) — same weights and forms
  - lin tracking (V20f's yaw_frame_exp, std=0.5)
  - wz_proportional (still has has_cmd guard at 0.08 — when
    cmd_yaw=0 the proportional penalty is 0 anyway because the
    formula yields 0; the SKIP removal in tracking does the work)
  - Observations: NO token (V20i style)
  - Curriculum, command bins, scene, DR: V19f base unchanged

Predictions @ iter 5-8k (early signal):
  - waist_roll_vel > -0.05 (cheat killed)
  - track_ang_vel_z drops initially (no more +4.0 free reward) —
    NORMAL, expect 1.5-2.0 range vs V20i's 2.12 free
  - error_vel_yaw < 0.30 by 12k
  - bad_orient < 1% by 8k

Kill criteria:
  - iter 3k: waist_roll_vel < -0.10 → wz proportional too weak,
    fallback V20k-r1 (wz_proportional weight -4 → -8)
  - iter 5k: Mean reward < 50 → over-correction, restore V19f wz
    weights (3.0 → 4.0 to compensate for harder landscape)
  - iter 8k: bad_orient > 5% → need 2-mode standing token; V20k-r2

Decision tree:
  - V20k works (xy walks + standing stable) → root cause confirmed
    as reward asymmetry; V20-token series fully deprecated
  - V20k walking improves but standing still bad → V20k-r2 adds
    2-mode {standing, moving} token (user's earlier "0 vs other"
    intuition retained at minimum granularity)
  - V20k regresses below V19f → reward asymmetry was load-bearing;
    V19f's lin SKIP must be restored (V20k-r3 = V19f + only wz exp)
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v20f import RewardsCfg as V20fRewardsCfg
from .velocity_env_cfg_rot_v20i import RobotEnvCfg as V20iEnvCfg


@configclass
class RewardsCfg(V20fRewardsCfg):
    # Replace V19f's rotating_aware (which SKIPs cmd_yaw<0.05 with
    # +1.0 free reward) with the IsaacLab-standard track_ang_vel_z_exp.
    # Same weights and std as V19f — only the SKIP behavior changes.
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5 * 2 * 3,  # 3.0 (matches V19f)
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_sharp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.20},
    )


@configclass
class RobotEnvCfg(V20iEnvCfg):
    # Inherit V20i's no-token observations + V20f's strict lin tracking,
    # then layer the symmetric wz fix on top.
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
