"""15-DOF rotation config V20j — "V20f + 3-mode token (standing|pure_wz|other)".

Refines V20i's "remove token entirely" approach with user insight:
isolation IS valuable -- but only where modes have qualitatively
distinct objectives. The 5-mode partition over-isolates because
pure_vx/pure_vy subpolicies, by construction, never observe cmd_wz!=0
envs and therefore never receive yaw-control gradient (track_ang_vel_z
SKIP at cmd_yaw<0.05 + wz_proportional has_cmd guard at cmd_yaw<0.08
both produce zero signal there).

V20j collapses {pure_vx, pure_vy, joint} into a single "other" bucket
while keeping {standing, pure_wz} isolated. The "other" subpolicy now
sees joint envs' cmd_wz!=0 samples alongside pure_vx/vy samples -- the
same shared-parameter yaw-control transfer V19f used globally, scoped
to the linear-motion subspace. Standing and pure_wz keep their
specialized subpolicies (in-place stillness vs drift-free rotation
have qualitatively different objectives, so isolation pays off).

Single-axis A/B vs V20f: only the token observation changes (5-dim ->
3-dim, function gait_mode_token -> gait_mode_token_3). All rewards,
bins, curriculum, std/weights, and V20f's lin-side rotation_skip
removal are inherited unchanged.

Predictions @ iter 12-15k:
  - pure_vx, pure_vy walk smoothly (yaw-control transfer restored
    via shared "other" subpolicy)
  - pure_wz drift control matches or beats V20f (dedicated subpolicy)
  - standing remains stable (dedicated subpolicy)
  - error_vel_xy < 0.40, error_vel_yaw < 0.30
  - bad_orient unchanged

Risks:
  - "other" bucket holds 65% of envs (15+15+35) -> majority subpolicy.
    Should still benefit from joint envs' yaw signal, but if pure_vx
    samples dominate the "other" bucket's gradient, yaw transfer may
    be weak. Mitigation: V20f's lin-side rotation_skip removal already
    helps; if needed V20j+ can rebalance bucket weights.
  - Deploy C++ side currently registers a 5-dim gait_mode_token in
    State_RLBase.cpp. To deploy a V20j policy, that observation slot
    must be replaced with a 3-dim version computed from the same
    {standing, pure_wz, other} partition. NOT changed here -- training
    only.

Decision tree:
  - V20j works (pure xy walks) AND V20i (no token) also works -> token
    isolation neutral; prefer simpler V20i.
  - V20j works AND V20i fails -> selective isolation is the answer;
    keep 3-mode design and propagate to deploy.
  - V20j fails AND V20i works -> any token harms; deprecate the
    architecture entirely.
  - Both fail -> token is not the bottleneck; investigate per-axis
    curriculum or wz_proportional structural issues.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v20f import RobotEnvCfg as V20fEnvCfg


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
class RobotEnvCfg(V20fEnvCfg):
    # Replace V20c/V20f's 5-mode token observations with the 3-mode
    # variant ({standing, pure_wz, other}). V20f rewards (lin-side
    # rotation_skip removal) are inherited unchanged.
    observations: ObservationsCfg = ObservationsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Play: expose the full curriculum-extended command range.
        # V19f bin counts: 15 vx_pos, 8 vx_neg, 11 vy, 16 wz.
        self.commands.base_velocity.num_active_vx_pos = 15
        self.commands.base_velocity.num_active_vx_neg = 8
        self.commands.base_velocity.num_active_vy = 11
        self.commands.base_velocity.num_active_wz = 16
