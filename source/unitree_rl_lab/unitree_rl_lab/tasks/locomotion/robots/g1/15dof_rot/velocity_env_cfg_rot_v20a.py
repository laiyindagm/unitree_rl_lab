"""15-DOF rotation config V20a — "Mode-Token Locomotion".

Hypothesis (theoretical, see proof in conversation 2026-04-24):
  Adding a discrete 5-mode one-hot token (standing/pure_vx/pure_vy/pure_wz/joint)
  as observation removes the high-Lipschitz requirement on the policy across
  command-mode boundaries. Sample complexity ratio vs V19i baseline:

        N_no_token / N_5mode  >=  Omega( (Delta_x + Delta_y + Delta_w)^2 / eps^2 )

  In V19i parameters (Delta ~ 950, eps = 0.1) this is order 1e7 in the
  worst case; even with conservative Delta = 10/axis it remains > 1e4.

  The 5-mode encoding (vs 3-bit (i,j,k)) is chosen to align with the V19i
  sampling distribution, avoiding the "mode fragmentation" issue where rare
  3-bit cells (e.g. (0,0,0) all-axis-active joint commands) get insufficient
  samples per sub-policy.

Changes vs V19i (which is the most stable scheduled-standing baseline):
  1. ObservationsCfg: add gait_mode_token (5-dim one-hot) to BOTH
     PolicyCfg and CriticCfg, appended right after velocity_commands.
  2. Token uses no noise (it's a derived structural feature).
  3. ALL OTHER training settings identical to V19i — clean A/B test.

Deploy compatibility:
  - The C++ deploy code registers a corresponding gait_mode_token
    observation that derives the one-hot from the keyboard command.
  - See deploy/robots/g1_15dof_keyboard/src/State_RLBase.cpp (V20a addition).
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
from .velocity_env_cfg_rot_v19i import (
    CommandsCfg as V19iCommandsCfg,
    CurriculumCfg as V19iCurriculumCfg,
    RewardsCfg as V19iRewardsCfg,
    RobotEnvCfg as V19iEnvCfg,
)


# ---------------------------------------------------------------------------
# Observations: V19i baseline + 5-mode gait token (5 extra dims per term)
# ---------------------------------------------------------------------------
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
        # V20a: 5-mode one-hot token, appended right after velocity_commands.
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
        # V20a: critic also sees the token so V^pi can decompose by mode.
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


# ---------------------------------------------------------------------------
# Re-use V19i for everything else (rewards, commands, curriculum, scheduling).
# ---------------------------------------------------------------------------
@configclass
class RobotEnvCfg(V19iEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: V19iRewardsCfg = V19iRewardsCfg()
    commands: V19iCommandsCfg = V19iCommandsCfg()
    curriculum: V19iCurriculumCfg = V19iCurriculumCfg()

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
