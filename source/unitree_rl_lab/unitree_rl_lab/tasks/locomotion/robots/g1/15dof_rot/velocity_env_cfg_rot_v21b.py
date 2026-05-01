"""15-DOF rotation config V21b — V21a + velocity-scaled action rate."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v20g import RewardsCfg as V20gRewardsCfg
from .velocity_env_cfg_rot_v21a import RobotEnvCfg as V21aEnvCfg
from .velocity_env_cfg_rot_v21a import RobotPlayEnvCfg as V21aPlayEnvCfg


@configclass
class RewardsCfg(V20gRewardsCfg):
    action_rate = RewTerm(
        func=mdp.action_rate_scaled_by_vel,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "min_scale": 0.3,
        },
    )


@configclass
class RobotEnvCfg(V21aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21aPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
