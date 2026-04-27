"""15-DOF rotation config V20c (REVISED 2026-04-25) — "Token + V19f".

Re-evaluation of V19 evolution against ACTUAL training logs:
  - V5c: only 13 aborted runs (max model_999, ~1k iter), NOT a verified base.
  - V19c-e: zero-speed instability (robot walks at cmd=0) — "still walks" per
    V19f docstring observation of V19e.
  - V19f (18.5k iter, completed): "Zero-speed standing: FIXED" + "wz responds
    from 0.1: WORKING". Remaining issues per V19g docstring:
      (i)  vx/vy 0.1-0.3 NO RESPONSE
      (ii) Pure rotation drifts BACKWARD (asymmetric leg push cheat)
  - V19g (18.5k iter): reverted V19f's track_lin_vel_xy_rotation_skip and
    weakened wz_proportional → fixed (i)(ii) but LOST zero-speed standing.
  - V19h/i: piled standing penalty → suicide policy, full collapse.

V19f is therefore the ONLY V19 version simultaneously satisfying:
  (a) trained to completion without collapse,
  (b) zero-speed standing works,
  (c) wz responds from small command.

Its two open issues map naturally to mode-token theoretical gains:
  - (i) vx/vy small-cmd no response: in V19f the policy mixes pure_vx envs
    (15%) with joint envs (35%) where rotation creates parasitic vx; without
    a mode signal the policy averages strategies and ignores small vx. With
    the 5-mode token, pure_vx mode gets a separable feature → policy can
    learn small-vx response without rotation contamination.
  - (ii) Pure-rotation backward drift: V19f's track_lin_vel_xy_rotation_skip
    returns 1.0 for pure rotation (zero penalty for any drift). With a token
    flagging "pure_wz mode", the policy can learn drift-free rotation
    directly conditioned on the mode, without needing a wider reward zoo.

V20c changes vs V19f:
  1. ObservationsCfg overridden: gait_mode_token added to PolicyCfg + CriticCfg
     after velocity_commands, eps_x=eps_y=eps_w=0.1, no noise. history_length=5.
  2. ALL OTHER training settings identical to V19f (rewards, commands,
     curriculum, env). Clean A/B test isolating the token's effect.

Predictions @ iter 10-15k (vs V19f baseline at 18.5k):
  - Zero-speed standing: should remain FIXED (token doesn't change reward).
  - Pure-rotation backward drift: should reduce (token enables mode-specific
    learning for pure_wz quadrant).
  - vx/vy small-cmd response: should improve (pure_vx token disambiguates
    rotation-induced parasitic vx).
  - Stability: should match V19f (no collapse), since reward landscape
    unchanged.

Decision tree:
  - V20c stable + (i)(ii) improved → token validated, proceed to V20d
    (V19f base + token + slightly stronger pure_rotation_drift maybe).
  - V20c stable but (i)(ii) unchanged → token is observation-noise, abandon.
  - V20c collapses despite identical V19f rewards → token itself is harmful
    (e.g. 5-mode discrete jumps confuse value function); abandon V20.

Deploy: shares the same C++ REGISTER_OBSERVATION(gait_mode_token) already
present in deploy/robots/g1_15dof_keyboard/src/State_RLBase.cpp.
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
from .velocity_env_cfg_rot_v19f import (
    RobotEnvCfg as V19fEnvCfg,
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
class RobotEnvCfg(V19fEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

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
        # Play: expose the full curriculum-extended command range.
        # V19f bin counts: 15 vx_pos, 8 vx_neg, 11 vy, 16 wz.
        self.commands.base_velocity.num_active_vx_pos = 15
        self.commands.base_velocity.num_active_vx_neg = 8
        self.commands.base_velocity.num_active_vy = 11
        self.commands.base_velocity.num_active_wz = 16
        # Play: expose the full curriculum-extended command range.
        # V19f bin counts: 15 vx_pos, 8 vx_neg, 11 vy, 16 wz.
        self.commands.base_velocity.num_active_vx_pos = 15
        self.commands.base_velocity.num_active_vx_neg = 8
        self.commands.base_velocity.num_active_vy = 11
        self.commands.base_velocity.num_active_wz = 16
