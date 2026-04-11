"""15-DOF rotation config V8a — "Gentle Low-Speed Fix".

V7 failure analysis: new penalty weights were ~10x too large.
  - velocity_mismatch -1.0 → -0.037/step, standstill -0.5 → -0.033/step,
    waist -0.3 → -0.024/step. Total new: -0.094/step.
  - alive reward is only +0.009/step → death stops penalty accumulation →
    policy learned to fall intentionally.

Fix: reduce all new penalty weights ~8x. Target total new penalty ≤ -0.012/step
(~10% of total penalty budget). Keep V6c's action_rate and joint_acc UNCHANGED.

Changes from V6c:
  New rewards (conservative weights):
    - velocity_mismatch_l1: -0.15  (was -1.0 in V7a)
    - standstill_joint_vel: -0.05  (was -0.5 in V7a)
    - waist_joint_vel_penalty: -0.04 (was -0.3 in V7a)
  Everything else: V6c UNCHANGED (including action_rate=-0.12, joint_acc=-8e-7)
  Runner: BasePPORunnerV3Cfg
"""

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


TRAIN_JOINT_NAMES = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_.*",
    "waist_.*_joint",
]


@configclass
class RewardsCfg(V6cRewardsCfg):
    """V6c rewards + gentle low-speed tracking + stability."""

    # --- V6c action_rate and joint_acc are INHERITED UNCHANGED ---

    # --- New: direct velocity error for low-speed following (gentle) ---
    velocity_mismatch = RewTerm(
        func=mdp.velocity_mismatch_l1, weight=-0.15,
        params={"command_name": "base_velocity"},
    )

    # --- New: standstill oscillation damping (gentle) ---
    standstill_joint_vel = RewTerm(
        func=mdp.standstill_joint_vel, weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES),
        },
    )

    # --- New: waist velocity damping for head sway (gentle) ---
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
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
