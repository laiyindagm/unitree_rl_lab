"""15-DOF rotation config V16c — "Adaptive Sampling + Movement Incentive".

Combines V16a (fixed adaptive sampling) with V16b (scheduled movement incentive).
This is the full-intervention variant in the V16 2x2 factorial design:

  V16a = adaptive sampling + standard rewards   (distribution fix only)
  V16b = standard sampling + movement incentive  (reward fix only)
  V16c = adaptive sampling + movement incentive   (BOTH fixes)

If V16c >> V16a and V16c >> V16b, the two fixes are synergistic.
If V16c ~= max(V16a, V16b), one fix dominates.
If V16c < V16a or V16b, there's destructive interference.

Reward (from V16b): V15a tracking + scheduled movement incentive
Command (from V16a): PerformanceWeightedVelocityCommand with relative accuracy
Curriculum (from V16a): performance_weighted_vel_curriculum
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v16a import (
    CommandsCfg,
    CurriculumCfg,
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
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: V16bRewardsCfg = V16bRewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
