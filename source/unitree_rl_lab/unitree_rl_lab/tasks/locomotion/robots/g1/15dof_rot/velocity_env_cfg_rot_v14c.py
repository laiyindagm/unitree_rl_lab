"""15-DOF rotation config V14c — "Speed-Gated Steps".

Hypothesis: The gait (weight=0.5) and feet_clearance (weight=1.0) rewards
force full periodic stepping at ALL command magnitudes. At cmd=0.2, the
policy earns ~1.4 from gait+clearance for "marching in place" but only
~0.15 marginal from tracking. The policy discovers that marching in place
maximizes reward regardless of low-speed commands.

Fix: Scale gait and feet_clearance by command magnitude. At cmd=0, no
stepping reward (policy can stand still). At cmd≥0.3, full stepping reward.
Between 0 and 0.3, linear interpolation.

This allows the policy to learn speed-appropriate movement: shuffle at
low cmd, full steps at high cmd. The "march in place" strategy is no
longer rewarded at low commands.

Changes from V13a:
  - gait: feet_gait → feet_gait_speed_scaled (speed_gate=0.3)
  - feet_clearance: foot_clearance_reward → foot_clearance_speed_scaled (speed_gate=0.3)
  Everything else identical to V13a (action_rate -0.02, V10d base).
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
    gait = RewTerm(
        func=mdp.feet_gait_speed_scaled, weight=0.5,
        params={
            "period": 0.7, "offset": [0.0, 0.5], "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "speed_gate": 0.3,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_speed_scaled, weight=1.0,
        params={
            "std": 0.05, "tanh_mult": 2.0, "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "speed_gate": 0.3,
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
