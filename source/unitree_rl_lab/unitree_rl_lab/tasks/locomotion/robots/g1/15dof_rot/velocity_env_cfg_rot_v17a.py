"""15-DOF rotation config V17a — "Combined Baseline + Standing Fix".

Combines V16a (adaptive sampling) + V16b (scheduled movement incentive).
This is structurally similar to V16c but with critical standing fix.

V16 Results:
  V16a: lin_vel from 0.2 (adaptive bins), ang_vel from 0.15, BUT zero-speed
        stepping (rel_standing_envs=0.05 was too low)
  V16b: ang_vel from 0.1 (movement incentive), lin_vel still 0.3,
        zero-speed still (rel_standing_envs=0.20 was fine)

Key changes from V16c:
  1. rel_standing_envs = 0.15 (V16a had 0.05 -> stepping, V16b had 0.20)
     Compromise: enough standing practice for stillness, not too much to
     waste training on easy commands.
  2. rel_rotating_envs = 0.25 (was 0.20 in V16a user-modified config)
     Slightly more rotation practice since ang_vel was harder to improve.

Hypothesis: adaptive sampling (lin_vel down to 0.2) + movement incentive
(ang_vel down to 0.1) -> both axes reach <=0.15 when combined. Standing fix
ensures zero-speed stillness.
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v16a import (
    CommandsCfg as V16aCommandsCfg,
    CurriculumCfg as V16aCurriculumCfg,
)
from .velocity_env_cfg_rot_v16b import (
    RewardsCfg as V16bRewardsCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Env ----------
@configclass
class RobotEnvCfg(V15aEnvCfg):
    commands: V16aCommandsCfg = V16aCommandsCfg()
    curriculum: V16aCurriculumCfg = V16aCurriculumCfg()
    rewards: V16bRewardsCfg = V16bRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Fix V16a's zero-speed stepping: 0.05 was too low, V16b's 0.20 was fine
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_rotating_envs = 0.25


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Expand all bins to limits for evaluation
        self.commands.base_velocity.vx_bins = [
            (-0.8, -0.35),
            (-0.35, -0.25),
            (-0.25, -0.15),
            (-0.15, -0.05),
            (0.05, 0.15),
            (0.15, 0.25),
            (0.25, 0.35),
            (0.35, 1.5),
        ]
        self.commands.base_velocity.wz_bins = [
            (-0.8, -0.35),
            (-0.35, -0.25),
            (-0.25, -0.15),
            (-0.15, -0.05),
            (0.05, 0.15),
            (0.15, 0.25),
            (0.25, 0.35),
            (0.35, 0.8),
        ]
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
