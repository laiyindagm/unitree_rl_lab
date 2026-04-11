"""15-DOF rotation config V7a — "Low-speed Tracking": fix dead zone + standstill.

Root cause of V6 dead zone (lin/ang < 0.3):
  exp(-error²/σ²) with σ=0.5 gives reward 0.85 for doing nothing at cmd=0.2,
  while action penalties cost ~0.25. The agent rationally stays still.

Solution: add direct L1 velocity error penalty (constant gradient at ALL speeds)
+ reduce action_rate slightly + add standstill joint velocity damping.

Changes from V6c:
  New rewards:
    - velocity_mismatch_l1: -1.0  (direct L1 error, solves dead zone)
    - standstill_joint_vel: -0.5  (damp joint motion when standing)
    - waist_joint_vel_penalty: -0.3 (suppress head lateral sway)
  Reduced penalties (create room for low-speed movement):
    - action_rate: -0.12 -> -0.08
    - joint_acc: -8e-7 -> -5e-7
  Everything else: V6c unchanged (V6a stability + V6b rotation + DR)
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
    """V6c rewards + low-speed tracking + standstill stability."""

    # --- Reduce penalties to create room for low-speed movement ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.08)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
    )

    # --- New: direct velocity error for low-speed following ---
    velocity_mismatch = RewTerm(
        func=mdp.velocity_mismatch_l1, weight=-1.0,
        params={"command_name": "base_velocity"},
    )

    # --- New: standstill oscillation damping ---
    standstill_joint_vel = RewTerm(
        func=mdp.standstill_joint_vel, weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES),
        },
    )

    # --- New: waist velocity damping for head sway ---
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.3,
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
