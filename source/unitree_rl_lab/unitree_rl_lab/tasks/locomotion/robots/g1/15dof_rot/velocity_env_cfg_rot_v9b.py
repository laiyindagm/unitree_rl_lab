"""15-DOF rotation config V9b — "Action Penalty Reshaping".

Problem: Fixed action_rate(-0.12) punishes small corrections at low speed the
same as at high speed. At low cmd (<0.3), the penalty-to-tracking-gain ratio
makes immobility rational (dead zone). "velocity_mismatch_l1" from V8a also
helps by giving a linear penalty proportional to error.

Fix (reward layer only, zero observation change):
  - Replace action_rate_l2(-0.12) → action_rate_scaled_by_vel(-0.15)
    At standstill: weight ×1.2 → -0.18 (strong stillness)
    At walk cmd=0.5: weight ×0.8 → -0.12 (same as V6c)
    At run cmd=1.0: weight ×0.5 → -0.075 (relaxed for agile moves)
  - Add velocity_mismatch_l1(-0.20): linear penalty on |cmd_xy - actual_xy|
    Gives monotone gradient toward correct speed (unlike exp(-)||²)
  - Keep waist_joint_vel -0.04 (V8c proven)

Changes from V6c:
  Rewards:
    - action_rate_l2(-0.12) → action_rate_scaled_by_vel(-0.15)
    - velocity_mismatch_l1: -0.20 (from V8a, slightly stronger)
    - waist_joint_vel_penalty: -0.04 (from V8c)
  Commands:
    - rel_rotating: 0.30 (restored from V6b)
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v6c import (
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)


# ---------- Rewards: reshape action penalty + mismatch ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    # Override: scaled by velocity (high at standstill, low at run)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_scaled_by_vel,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "min_scale": 0.25,
        },
    )
    # New: linear tracking penalty
    velocity_mismatch_l1 = RewTerm(
        func=mdp.velocity_mismatch_l1,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"},
    )
    # Waist sway suppression
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


# ---------- Commands ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    pass


# ---------- Env ----------
@configclass
class RobotEnvCfg(V6cEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
