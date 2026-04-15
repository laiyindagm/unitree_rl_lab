"""15-DOF rotation config V14b — "Half Penalties".

Hypothesis: The total penalty budget (~28 terms) overwhelms the low-speed
marginal tracking gain. At cmd=0.2, tracking improvement is only +0.148
but transitioning from standing to walking incurs penalties across
joint_deviation, joint_acc, waist_vel, action_rate, etc. that sum > 0.148.

Fix: Halve ALL penalty weights. This uniformly reduces the "startup energy"
needed to begin moving, testing whether the penalty MAGNITUDE (not shape)
is the bottleneck.

V13a penalty episode data (for reference):
  joint_deviation_legs:  -0.1485 (weight -1.0 → -0.5)
  action_rate:           -0.1217 (keep -0.02 from V13a)
  waist_roll_vel:        -0.1124 (weight -0.20 → -0.10)
  base_angular_velocity: -0.1028 (weight -0.15 → -0.075)
  joint_acc:             -0.0797 (weight -8e-7 → -4e-7)
  ... all penalties halved

Changes from V13a:
  - ALL penalty weights × 0.5 (except action_rate which stays at -0.02)
  - ALL positive reward weights UNCHANGED
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v13a import (
    RewardsCfg as V13aRewardsCfg,
    RobotEnvCfg as V13aEnvCfg,
)


@configclass
class RewardsCfg(V13aRewardsCfg):
    # --- Halved stability penalties ---
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.075)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-4e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
    )
    energy = RewTerm(
        func=mdp.energy, weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
    )
    # --- Halved posture/deviation penalties ---
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.78})
    backward_lean = RewTerm(func=mdp.backward_lean_penalty, weight=-3.0)
    # --- Halved behavioral penalties ---
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still, weight=-0.15,
        params={"command_name": "base_velocity"},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near, weight=-0.15,
        params={"threshold": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")},
    )
    body_lin_acc = RewTerm(
        func=mdp.body_lin_acc_l2, weight=-0.0025,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-0.5,
        params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*)(?!.*shoulder.*)(?!.*elbow.*)(?!.*wrist.*)(?!.*rubber_hand).*"])},
    )
    # --- Halved rotation penalties ---
    yaw_rate_l1 = RewTerm(
        func=mdp.yaw_rate_l1, weight=-0.075,
        params={"command_name": "base_velocity"},
    )
    rotation_double_support_slide = RewTerm(
        func=mdp.rotation_double_support_slide_penalty, weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    rotation_twist_joint = RewTerm(
        func=mdp.rotation_twist_joint_penalty, weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", ".*_hip_yaw_joint"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # --- Halved per-joint waist (from V10d) ---
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )
    waist_pitch_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])},
    )
    waist_yaw_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
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
