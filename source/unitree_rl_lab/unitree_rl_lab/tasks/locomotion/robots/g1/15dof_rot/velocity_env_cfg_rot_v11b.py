"""15-DOF rotation config V11b -- "Scaled Action Rate".

Fix the dead zone from the PENALTY side. The root cause:
  action_rate(-0.12) costs ~0.33/step uniformly, but exp tracking marginal
  gain is only 0.15 at cmd=0.2. Agent rationally stays still.

Fix: replace fixed action_rate_l2 with action_rate_scaled_by_vel.
Penalty scales with command magnitude:
  standstill/low (cmd~0.2): effective = -0.15 * 0.3 = -0.045 (~14% of original)
  mid (cmd=0.5):            effective = -0.15 * 0.5 = -0.075
  high (cmd=1.0):           effective = -0.15 * 1.0 = -0.15 (STRONGER than -0.12)

This breaks the dead zone equation at low speed while IMPROVING high-speed
smoothness. Single parameter replacement, minimal change.

Base: V10d (bucketed curriculum + per-joint waist + V6c DR)
Changes: replace action_rate_l2(-0.12) with action_rate_scaled_by_vel(-0.15)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v10d import (
    CommandsCfg,
    CurriculumCfg,
    RewardsCfg as V10dRewardsCfg,
    RobotEnvCfg as V10dEnvCfg,
)


@configclass
class RewardsCfg(V10dRewardsCfg):
    """V10d rewards with velocity-scaled action rate."""

    action_rate = RewTerm(
        func=mdp.action_rate_scaled_by_vel,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "min_scale": 0.3,
        },
    )


@configclass
class RobotEnvCfg(V10dEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
