"""15-DOF rotation config V16a — "Adaptive Sampling V2".

Redesigned adaptive command sampling with:
  1. RELATIVE ACCURACY metric (speed-invariant, standing always = 0)
  2. EXPLICIT asymmetric bin layout (not uniform grid)
  3. Dead zone [-0.05, 0.05] excluded from sampling
  4. Outermost bins expand via curriculum
  5. Guaranteed pure-rotation and standing proportions

Bin layout (vx):
  Backward: [-0.55,-0.35]* [-0.35,-0.25] [-0.25,-0.15] [-0.15,-0.05]
  Dead zone: [-0.05, 0.05]  -- NOT sampled, standing only
  Forward:  [0.05,0.15] [0.15,0.25] [0.25,0.35] [0.35,0.80]*
  * = expandable via curriculum to limit_ranges

Bin layout (wz):
  Negative: [-0.50,-0.35]* [-0.35,-0.25] [-0.25,-0.15] [-0.15,-0.05]
  Dead zone: [-0.05, 0.05]  -- NOT sampled
  Positive: [0.05,0.15] [0.15,0.25] [0.25,0.35] [0.35,0.50]*
  * = expandable via curriculum to limit_ranges

Total cells: 8 vx_bins x 8 wz_bins = 64 adaptive cells.
Each cell tracked independently via relative accuracy EMA.

Hypothesis: explicit low-speed bins (0.05-0.15) + adaptive concentration
on difficult cells + dead zone exclusion → solves low-speed tracking.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v15a import (
    RobotEnvCfg as V15aEnvCfg,
)


# ---------- Commands: Performance-weighted adaptive sampling V2 ----------
@configclass
class CommandsCfg:
    base_velocity = mdp.PerformanceWeightedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        rel_rotating_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # ranges/limit_ranges are still needed by base class for vy and for
        # logging, but vx/wz sampling is driven by explicit bin edges.
        ranges=mdp.PerformanceWeightedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.55, 0.80),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.50, 0.50),
        ),
        limit_ranges=mdp.PerformanceWeightedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
        # Explicit bin edges (asymmetric, non-uniform)
        vx_bins=[
            (-0.55, -0.35),   # backward far (expandable to -0.8)
            (-0.35, -0.25),
            (-0.25, -0.15),
            (-0.15, -0.05),   # backward near
            (0.05, 0.15),     # forward near
            (0.15, 0.25),
            (0.25, 0.35),
            (0.35, 0.80),     # forward far (expandable to 1.5)
        ],
        wz_bins=[
            (-0.50, -0.35),   # negative far (expandable to -0.8)
            (-0.35, -0.25),
            (-0.25, -0.15),
            (-0.15, -0.05),   # negative near
            (0.05, 0.15),     # positive near
            (0.15, 0.25),
            (0.25, 0.35),
            (0.35, 0.50),     # positive far (expandable to 0.8)
        ],
        ema_alpha=0.1,
        temperature=2.0,
        min_sampling_prob=0.02,
        accuracy_cmd_min=0.08,
    )


# ---------- Curriculum: performance-weighted ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.performance_weighted_vel_curriculum,
        params={
            "range_expand_threshold": 0.5,
        },
    )


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
        # Expand all bins to their limits for evaluation
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
