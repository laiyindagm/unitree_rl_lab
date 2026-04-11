"""15-DOF rotation config V7b — "Velocity-Adaptive": innovative action scaling.

Novel approach: replace fixed action_rate penalty with velocity-scaled version.
At high speed: full smoothness penalty.
At low speed (~0.2): only 25% penalty, allowing movement initiation.
At standstill: min_scale * penalty for stability.

This directly addresses the dead zone mechanism instead of patching it.

Changes from V7a:
  Replace action_rate_l2 with action_rate_scaled_by_vel:
    - weight=-0.15 (higher raw weight, but scaled down at low speed)
    - min_scale=0.25 (at standstill, effective weight = -0.0375)
  Stronger standstill:
    - standstill_joint_vel: -0.5 -> -0.8
    - stand_still: -0.3 -> -0.5
  Stronger waist:
    - waist_joint_vel: -0.3 -> -0.5
    - joint_deviation_waists: -1.0 -> -1.5
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v7a import (
    TRAIN_JOINT_NAMES,
    RewardsCfg as V7aRewardsCfg,
    RobotEnvCfg as V7aEnvCfg,
)


@configclass
class RewardsCfg(V7aRewardsCfg):
    """V7a + velocity-adaptive action penalty + stronger anti-oscillation."""

    # --- Replace fixed action_rate with velocity-scaled version ---
    action_rate = RewTerm(
        func=mdp.action_rate_scaled_by_vel, weight=-0.15,
        params={"command_name": "base_velocity", "min_scale": 0.25},
    )

    # --- Stronger standstill damping ---
    standstill_joint_vel = RewTerm(
        func=mdp.standstill_joint_vel, weight=-0.8,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still, weight=-0.5,
        params={"command_name": "base_velocity"},
    )

    # --- Stronger waist anti-sway ---
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1, weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


@configclass
class RobotEnvCfg(V7aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
