"""15-DOF rotation config V21d — V21c + decayed gait shaping with full policy obs + velocity prediction runner."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v21c import (
    ObservationsCfg,
    RewardsCfg as V21cRewardsCfg,
    RobotEnvCfg as V21cEnvCfg,
    RobotPlayEnvCfg as V21cPlayEnvCfg,
)


@configclass
class RewardsCfg(V21cRewardsCfg):
    gait = RewTerm(
        func=mdp.feet_gait_speed_scaled_decayed,
        weight=0.5,
        params={
            "period": 0.7,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "speed_gate": 0.3,
            "end_step": 2000,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_speed_scaled_decayed,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "speed_gate": 0.3,
            "end_step": 2000,
        },
    )
    rotation_single_support = RewTerm(
        func=mdp.rotation_single_support_reward_decayed,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "end_step": 2000,
        },
    )


@configclass
class RobotEnvCfg(V21cEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()


@configclass
class RobotPlayEnvCfg(V21cPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
