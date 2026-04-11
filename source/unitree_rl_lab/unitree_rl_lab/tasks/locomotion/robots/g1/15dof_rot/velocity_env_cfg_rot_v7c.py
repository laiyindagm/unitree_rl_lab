"""15-DOF rotation config V7c — "Full Package": all novel approaches combined.

Maximum anti-oscillation + velocity-adaptive action penalty + direct L1 tracking.
Strongest standstill damping and waist stabilization across V7 family.

Changes from V7b:
  Even stronger standstill:
    - standstill_joint_vel: -0.8 -> -1.0
    - stand_still: -0.5 -> -0.6
  Even stronger waist:
    - joint_deviation_waists: -1.5 -> -2.0
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v7a import TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v7b import (
    RewardsCfg as V7bRewardsCfg,
    RobotEnvCfg as V7bEnvCfg,
)


@configclass
class RewardsCfg(V7bRewardsCfg):
    """V7b + maximum anti-oscillation weights."""

    # --- Maximum standstill damping ---
    standstill_joint_vel = RewTerm(
        func=mdp.standstill_joint_vel, weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still, weight=-0.6,
        params={"command_name": "base_velocity"},
    )

    # --- Maximum waist stabilization ---
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


@configclass
class RobotEnvCfg(V7bEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
