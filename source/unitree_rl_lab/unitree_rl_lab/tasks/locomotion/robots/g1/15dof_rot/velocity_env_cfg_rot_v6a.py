"""15-DOF rotation config V6a — "Smooth V5c": aggressive stability penalties.

V5c is the best sim2sim platform. Its main weakness is upper-body oscillation
during walking and aggressive motions during command transitions.

Changes from V5c:
  Reward tuning (stability focus):
    - ang_vel_xy_l2 (base_angular_velocity): -0.05 -> -0.15  (3x, suppress roll/pitch oscillation)
    - base_linear_velocity (lin_vel_z_l2): -2.0 -> -4.0      (2x, suppress vertical bouncing)
    - joint_acc: -2.5e-7 -> -8e-7                             (3.2x, smoother joint motion = smoother transitions)
    - backward_lean: -2.0 -> -6.0                             (3x, V5 only contributed -0.004)
    - flat_orientation_l2: -6.0 -> -8.0                       (from V5a)
    - action_rate: -0.10 (keep V5c)
  New reward:
    - body_lin_acc_l2: -0.005  (directly penalize base linear acceleration)
  Everything else: V5c DR, V5a behavioral rewards, rot base unchanged.
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v5c import (
    EventCfg,
    RewardsCfg as V5cRewardsCfg,
    RobotEnvCfg as V5cEnvCfg,
)


@configclass
class RewardsCfg(V5cRewardsCfg):
    """V5c rewards + aggressive stability penalties."""

    # --- Strengthened stability penalties ---
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.0)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-8e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "waist_.*_joint"])},
    )
    backward_lean = RewTerm(func=mdp.backward_lean_penalty, weight=-6.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)

    # --- New: base acceleration penalty ---
    body_lin_acc = RewTerm(
        func=mdp.body_lin_acc_l2, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )


@configclass
class RobotEnvCfg(V5cEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
