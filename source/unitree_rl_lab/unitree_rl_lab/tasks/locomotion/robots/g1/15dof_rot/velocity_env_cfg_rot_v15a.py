"""15-DOF rotation config V15a — "Surgical Reward Rebalance".

Informed by 001.md analysis + V14 lessons. Targeted penalty reduction:

001.md Root Causes addressed:
  1. joint_deviation_legs (-1.0 -> -0.3): hip_roll/yaw are walking executors
  2. flat_orientation_l2 (-8.0 -> -3.0): was 2.5x industry standard at -8.0
  3. NEW cmd_nonresponse (-0.5): directly penalize standing when cmd active

V14 lessons applied:
  - Keep waist penalties at V10d FULL strength (V14b halved -> upper body sway)
  - Do NOT modify tracking kernel (V14a baselined -> destroyed early learning)
  - Do NOT zero gait at low speed (V14c speed-gated -> 55.8% bad_orient)

Inheritance: V13a (V10d + action_rate -0.02)
Changes from V13a:
  - joint_deviation_legs: -1.0 -> -0.3  (001.md Tier 1A)
  - flat_orientation_l2: -8.0 -> -3.0   (001.md Tier 1B, was -5 in rot base
    but -8 in V6a inheritance chain)
  - backward_lean: -6.0 -> -3.0         (proportional with flat_orient reduction)
  - NEW cmd_nonresponse_penalty: -0.5    (001.md Tier 1C)
  - ALL waist penalties UNCHANGED at V10d strength
"""


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v13a import (
    RewardsCfg as V13aRewardsCfg,
    RobotEnvCfg as V13aEnvCfg,
)


@configclass
class RewardsCfg(V13aRewardsCfg):
    # --- 001.md Tier 1A: reduce joint_deviation_legs ---
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # --- 001.md Tier 1B: reduce flat_orientation_l2 ---
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    backward_lean = RewTerm(func=mdp.backward_lean_penalty, weight=-3.0)
    # --- 001.md Tier 1C: cmd nonresponse penalty ---
    cmd_nonresponse = RewTerm(
        func=mdp.cmd_nonresponse_penalty, weight=-0.5,
        params={"command_name": "base_velocity", "cmd_threshold": 0.15, "vel_threshold": 0.05},
    )


@configclass
class RobotEnvCfg(V13aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
