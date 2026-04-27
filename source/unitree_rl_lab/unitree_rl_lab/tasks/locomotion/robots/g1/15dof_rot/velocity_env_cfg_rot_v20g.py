"""15-DOF rotation config V20g — "V20f + Eliminate wz Skip Blind Spot".

Sequential A/B over V20f (mirror fix on the wz axis):
  Replace \`track_ang_vel_z_rotating_aware\` (returns 1.0 during straight
  walking, creating zero-gradient blind spot for yaw drift when cmd_yaw=0)
  with the IsaacLab-standard \`track_ang_vel_z_exp\` for BOTH the main and
  the sharp wz tracking terms. Same weight, same std, only function changes.

Rationale (V20f sim2sim observation):
  - User reported: pure rotation drift FIXED (V20f worked on lin side).
    Pure vx/vy: stagger persists. CRITICAL: adding even a small wz cmd
    makes vx/vy walking smooth and stable.
  - Diagnosis: V19f's \`_rotating_aware\` skip (|cmd_lin|>0.1 AND |cmd_yaw|<0.05
    -> reward=1.0) is the wz-side mirror of V19f's lin-side rotation_skip.
    During pure vx/vy, BOTH wz tracking terms (weight 3.0+1.0 = 4.0) give
    free reward regardless of actual yaw -> policy free to wobble in yaw
    -> asymmetric leg work -> stagger.
  - Once user adds cmd_yaw>=0.05, the skip releases, wz becomes actively
    monitored, and the policy that DID learn (during rotation training)
    activates -> smooth walk. This is direct empirical proof of the wz
    blind-spot.
  - V20f confirmed the "remove skip" recipe works on lin side
    (drift eliminated, no instability). Same recipe applied here.

Coupling check:
  - V19f added the skip originally to handle "natural drift during straight
    walking" (humans need micro yaw corrections too). But:
    (a) Standard exp(-dyaw²/0.25) at dyaw=0.1 rad/s gives reward=0.96, only
        4% loss -- not a real penalty for natural micro-drift.
    (b) zero_cmd_body_vel(-1.5) already explicitly handles cmd=0 standing.
    (c) wz_proportional(-4.0, cmd_threshold=0.08) already penalizes large
        wz when cmd_wz=0 -- but it stops at the threshold, leaving the
        wobble band un-penalized. Standard exp covers this band.

NOT changed vs V20f: lin tracking (stays standard yaw_frame_exp from V20f),
all penalties, command bins, curriculum, DR, observations (token retained),
all std/weight values for wz.

Predictions @ iter 12-15k:
  - sim2sim pure vx/vy stagger eliminated, walking matches "with-wz" smoothness
  - error_vel_yaw < 0.30 (down from V20f's 0.47)
  - error_vel_xy < 0.40 (down from V20f's 0.57; secondary effect because
    yaw stability -> step symmetry -> lin tracking improves)
  - entropy_loss recovers to < -0.5 (policy converges)
  - Mean reward recovers to ~78 (no more free-reward inflation but real
    tracking improves)
  - bad_orient unchanged

Risks:
  - Standard wz exp will mildly penalize natural yaw micro-drift during
    straight walk (~4% reward loss at dyaw=0.1 rad/s). Acceptable.
  - If wz tracking regresses (track_ang_vel_z drops > 30%) due to lin
    pressure, fallback V20g-r1 would relax std 0.5 -> 0.7.
  - If zero-speed standing destabilizes (V19c-e pathology), zero_cmd_body_vel
    weight might need to grow. Monitor stand_still and zero_cmd_body_vel
    rewards.

Decision tree:
  - Stagger fixed AND error_vel_yaw down -> wz blind-spot confirmed; full
    structural fix complete (V20f+V20g recipe). Move to deployment.
  - Stagger fixed BUT zero-speed regresses -> add zero_cmd_yaw_vel term.
  - No improvement -> wz blind-spot was not the root cause; investigate
    physical capability or curriculum.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v20f import (
    RewardsCfg as V20fRewardsCfg,
    RobotEnvCfg as V20fEnvCfg,
)


@configclass
class RewardsCfg(V20fRewardsCfg):
    # Replace V19f's rotating_aware (skips on straight walk) with standard
    # IsaacLab exp kernel. Same weight (3.0) and std (sqrt(0.25)=0.5) as V19f.
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5 * 2 * 3,  # 3.0
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    # Sharp wz tracker also loses the skip; same weight (1.0) and std (0.20).
    track_ang_vel_z_sharp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.20,
        },
    )


@configclass
class RobotEnvCfg(V20fEnvCfg):
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
