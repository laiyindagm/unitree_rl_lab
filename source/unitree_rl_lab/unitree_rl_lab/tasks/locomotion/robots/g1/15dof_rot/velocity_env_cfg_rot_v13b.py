"""15-DOF rotation config V13b — "Action Rate + Feet Slide Reset".

Two-factor experiment: reduce action_rate AND feet_slide to near industry
standard. Tests whether both penalties together are needed.

Cross-repo benchmark:
  IsaacLab G1 official:  action_rate = -0.005, feet_slide = NONE
  This project V6c:      action_rate = -0.12,  feet_slide = -0.3

Changes from V13a:
  - feet_slide: -0.3 → -0.1 (from V6c; IsaacLab G1 has none at all)
  Everything else from V13a (action_rate = -0.02, V10d base)
"""

import math

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
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
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
