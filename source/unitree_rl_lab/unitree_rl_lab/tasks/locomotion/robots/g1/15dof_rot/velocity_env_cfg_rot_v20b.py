"""15-DOF rotation config V20b — "Token + V19g (clean baseline)".

V20a (Token + V19i) failed sim2sim at iter 14k with the SAME signature as V19h
suicide-policy collapse: bad_orient=23.5%, action_std=0.46, entropy_loss=+5.7.

Hypothesis isolated by V20b: V19i's scheduled standing penalties (which were
intended to FIX V19h's always-on standing penalties) still create a falling-as-
escape attractor — schedule only delays the collapse. The 5-mode token cannot
help because the bottleneck is reward-landscape, not observation.

V20b reverts the env baseline to **V19g** (the last empirically stable config,
18.7k iter without collapse) and adds ONLY the gait_mode_token observation.
This is a clean A/B test of token efficacy without the V19i reward pathology.

Changes vs V19g:
  1. ObservationsCfg overridden to add gait_mode_token (5-dim one-hot) to BOTH
     PolicyCfg and CriticCfg, appended right after velocity_commands.
  2. Token uses no noise (it's a derived structural feature).
  3. ALL OTHER training settings identical to V19g.

Deploy compatibility: shares the same C++ REGISTER_OBSERVATION(gait_mode_token)
already added for V20a (deploy/robots/g1_15dof_keyboard/src/State_RLBase.cpp).
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    TRAIN_JOINT_NAMES,
)
from .velocity_env_cfg_rot_v19g import (
    CommandsCfg as V19gCommandsCfg,
    CurriculumCfg as V19gCurriculumCfg,
    RewardsCfg as V19gRewardsCfg,
    RobotEnvCfg as V19gEnvCfg,
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
class RobotEnvCfg(V19gEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: V19gRewardsCfg = V19gRewardsCfg()
    commands: V19gCommandsCfg = V19gCommandsCfg()
    curriculum: V19gCurriculumCfg = V19gCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
