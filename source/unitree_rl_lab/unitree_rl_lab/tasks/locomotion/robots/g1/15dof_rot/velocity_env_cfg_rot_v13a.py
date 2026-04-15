"""15-DOF rotation config V13a — "Action Rate Baseline Reset".

Single-factor experiment: reduce action_rate from -0.12 (24x industry standard)
to -0.02 (4x standard, conservative given our strong DR). Everything else
identical to V10d.

Cross-repo benchmark (Apr 2025):
  IsaacLab G1 official:  action_rate = -0.005
  IsaacLab base locomotion: -0.01
  This project V6c:      action_rate = -0.12  (24x G1 official)

Hypothesis: reducing action_rate alone will eliminate the low-speed dead zone
(cmd < 0.4 lin_vel, < 0.5 ang_vel) that motivated V7-V12 complex solutions.

Math at -0.02:
  cmd=0.1: marginal_gain = 0.039, action_cost ≈ 0.02 * 2.7 ≈ 0.054
  cmd=0.2: marginal_gain = 0.148, action_cost ≈ 0.054
  → Dead zone threshold drops from ~0.4 to ~0.15

Changes from V10d:
  - action_rate: -0.12 → -0.02 (ONLY change)
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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


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
