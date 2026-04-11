"""15-DOF rotation config V5a — Behavioral-fix experiment.

Based directly on the PROVEN rot base with MINIMAL additions:
  1. 4 new behavioral rewards (backward_lean, stand_still, feet_contact_without_cmd, feet_too_near)
  2. action_rate slightly increased (-0.05 -> -0.08) for smoother low-speed motion
  3. flat_orientation strengthened (-5.0 -> -6.0) for better upright posture
  4. NaN safety via clip_actions=100 (BasePPORunnerV3Cfg)
  Everything else IDENTICAL to rot base (commands, obs, events, terrain, curriculum).
  Total: 26 reward terms (22 from rot + 4 new)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    RewardsCfg as RotRewardsCfg,
    RobotEnvCfg as RotEnvCfg,
)


@configclass
class RewardsCfg(RotRewardsCfg):
    """Rot rewards + 4 behavioral fixes, 2 weight tweaks."""

    # --- Override existing weights ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.08)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)

    # --- New behavioral rewards ---
    backward_lean = RewTerm(func=mdp.backward_lean_penalty, weight=-2.0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.3,
        params={"command_name": "base_velocity"},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-0.3,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class RobotEnvCfg(RotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
