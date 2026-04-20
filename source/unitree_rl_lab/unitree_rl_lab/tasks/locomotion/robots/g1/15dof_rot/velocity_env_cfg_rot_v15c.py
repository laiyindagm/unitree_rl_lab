"""15-DOF rotation config V15c — "Wide Start + Anti-Stagnation".

Hypothesis: The narrow initial curriculum (-0.1, 0.1) is the primary cause of
the "stand still" prior. In this range, standing still earns tracking ~0.96-1.0,
forming an extremely strong stand-still prior. When curriculum expands to 0.3+,
this prior is nearly impossible to break.

Test: Skip the bucketed curriculum entirely. Start with the FULL command range
from iteration 0. Combined with V15a's reward surgery + cmd_nonresponse, the
policy must learn to respond to the full velocity spectrum from the beginning,
never forming a stand-still prior.

This is the most aggressive approach. Risk: early training instability due to
the full range. Mitigation: V15a's reduced penalties lower the "movement cost"
so even early-stage policies can afford to explore movement.

001.md Root Cause #5: "Curriculum starts from extremely narrow range (-0.1, 0.1)"

Changes from V15a:
  - Remove bucketed curriculum entirely (use standard lin/ang_vel_cmd_levels)
  - Start with FULL command ranges (no curriculum needed)
  - All reward changes from V15a retained (surgical rebalance)
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Commands: Full range from start ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.20,
        rel_rotating_envs=0.30,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # Start at FULL range — no progressive curriculum
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


# ---------- Curriculum: terrain only, no velocity curriculum ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]


# ---------- Env ----------
@configclass
class RobotEnvCfg(V15aEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
