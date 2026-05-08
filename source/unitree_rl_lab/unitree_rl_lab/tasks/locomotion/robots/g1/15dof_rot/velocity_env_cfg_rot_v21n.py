"""15-DOF rotation config V21n - V21f2 with 5-mode command token.

Derived from V21f2. The policy/critic command-mode token is expanded from
V21f2's 3-way {standing, pure_wz, other} token to the original 5-way partition:
standing / pure_vx / pure_vy / pure_wz / joint (>=2 active command axes).

The yaw-rate tracking reward coefficient is reduced from V21f2's 3.0 to 2.0.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v21c import (
    LOW_SPEED_CMD_MIN,
    LOW_SPEED_REL_FLOOR,
    LOW_SPEED_THRESHOLD_SCALE,
    LOW_SPEED_TRANSITION_WIDTH,
    TRACK_STD,
)
from .velocity_env_cfg_rot_v21f2 import (
    RewardsCfg as V21f2RewardsCfg,
    RobotEnvCfg as V21f2EnvCfg,
    RobotPlayEnvCfg as V21f2PlayEnvCfg,
)


@configclass
class RewardsCfg(V21f2RewardsCfg):
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_hybrid_low_speed,
        weight=2.0,
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
        gait_mode_token = ObsTerm(
            func=mdp.gait_mode_token,
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
        gait_mode_token = ObsTerm(
            func=mdp.gait_mode_token,
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
class RobotEnvCfg(V21f2EnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()


@configclass
class RobotPlayEnvCfg(V21f2PlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
