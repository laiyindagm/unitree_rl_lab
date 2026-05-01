"""15-DOF rotation config V21c — V21a + hybrid low-speed xy/yaw tracking."""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v20g import RewardsCfg as V20gRewardsCfg
from .velocity_env_cfg_rot_v21a import RobotEnvCfg as V21aEnvCfg
from .velocity_env_cfg_rot_v21a import RobotPlayEnvCfg as V21aPlayEnvCfg

TRACK_STD = math.sqrt(0.25)
LOW_SPEED_THRESHOLD_SCALE = 1.567
LOW_SPEED_TRANSITION_WIDTH = 0.05
LOW_SPEED_CMD_MIN = 0.05
LOW_SPEED_REL_FLOOR = 0.05


@configclass
class RewardsCfg(V20gRewardsCfg):
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_hybrid_low_speed,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "threshold_scale": LOW_SPEED_THRESHOLD_SCALE,
            "transition_width": LOW_SPEED_TRANSITION_WIDTH,
            "lin_cmd_min": LOW_SPEED_CMD_MIN,
            "rel_floor": LOW_SPEED_REL_FLOOR,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_hybrid_low_speed,
        weight=0.5 * 2 * 3,
        params={
            "command_name": "base_velocity",
            "std": TRACK_STD,
            "threshold_scale": LOW_SPEED_THRESHOLD_SCALE,
            "transition_width": LOW_SPEED_TRANSITION_WIDTH,
            "ang_cmd_min": LOW_SPEED_CMD_MIN,
            "rel_floor": LOW_SPEED_REL_FLOOR,
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        lin_speed_reward_regime_token = ObsTerm(
            func=mdp.lin_speed_reward_regime_token,
            params={
                "command_name": "base_velocity",
                "std": TRACK_STD,
                "threshold_scale": LOW_SPEED_THRESHOLD_SCALE,
                "transition_width": LOW_SPEED_TRANSITION_WIDTH,
                "lin_cmd_min": LOW_SPEED_CMD_MIN,
            },
        )
        gait_mode_token_3 = ObsTerm(
            func=mdp.gait_mode_token_3,
            params={
                "command_name": "base_velocity",
                "eps_x": 0.1,
                "eps_y": 0.1,
                "eps_w": 0.1,
            },
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        lin_speed_reward_regime_token = ObsTerm(
            func=mdp.lin_speed_reward_regime_token,
            params={
                "command_name": "base_velocity",
                "std": TRACK_STD,
                "threshold_scale": LOW_SPEED_THRESHOLD_SCALE,
                "transition_width": LOW_SPEED_TRANSITION_WIDTH,
                "lin_cmd_min": LOW_SPEED_CMD_MIN,
            },
        )
        gait_mode_token_3 = ObsTerm(
            func=mdp.gait_mode_token_3,
            params={
                "command_name": "base_velocity",
                "eps_x": 0.1,
                "eps_y": 0.1,
                "eps_w": 0.1,
            },
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES)},
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RobotEnvCfg(V21aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.diagnostic_lin_speed_threshold = LOW_SPEED_THRESHOLD_SCALE * TRACK_STD
        self.commands.base_velocity.diagnostic_lin_speed_transition_width = LOW_SPEED_TRANSITION_WIDTH


@configclass
class RobotPlayEnvCfg(V21aPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.diagnostic_lin_speed_threshold = LOW_SPEED_THRESHOLD_SCALE * TRACK_STD
        self.commands.base_velocity.diagnostic_lin_speed_transition_width = LOW_SPEED_TRANSITION_WIDTH
