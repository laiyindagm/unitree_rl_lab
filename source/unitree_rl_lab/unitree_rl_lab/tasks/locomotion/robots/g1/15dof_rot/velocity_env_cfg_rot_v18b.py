"""15-DOF rotation config V18b — "Fixed Discrete Adaptive".

V17c discrete bins + all framework bug fixes + tuned curriculum parameters.

V17c Issues Fixed:
  1. Zero-speed bin over-sampling: n=304k vs ~34k for others.
     Root cause: standing(15%)+rotating(25%) force-routed to vx=0 bin,
     zero-speed accuracy=0.085 (threshold 0.05 too strict) → appears
     "hardest" → adaptive sampler concentrates there → vicious cycle.
     Fix: exclude standing/rotating from bin EMA + exclude zero-speed
     cell from adaptive softmax (both in framework code).
  2. Curriculum stuck: all active bin perfs 0.08-0.18, threshold=0.5
     unreachable. perf_weighted=0.0968 even at iter 9837.
     Fix: lower threshold to 0.3 (achievable accuracy level).
  3. Zero-speed accuracy threshold 0.05 too strict → artificially low perf.
     Fix: raise to 0.10 (still requires near-still, but less extreme).

Changes from V17c:
  - temperature = 3.0 (from 2.0): better differentiation
  - curriculum threshold = 0.3 (from 0.5): achievable expansion trigger
  - zero_speed_accuracy_threshold = 0.10 (from 0.05): less strict

Hypothesis: with proper adaptive sampling and reachable curriculum,
discrete levels should expand progressively. The interesting gait
variation seen in V17c should improve as curriculum opens higher speeds.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
)
from .velocity_env_cfg_rot_v17c import (
    RobotEnvCfg as V17cEnvCfg,
    RobotPlayEnvCfg as V17cPlayEnvCfg,
)


# ---------- Curriculum: lower expansion threshold ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    perf_weighted = CurrTerm(
        func=mdp.performance_weighted_vel_curriculum,
        params={
            "range_expand_threshold": 0.3,
        },
    )


# ---------- Env ----------
@configclass
class RobotEnvCfg(V17cEnvCfg):
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.temperature = 3.0
        self.commands.base_velocity.zero_speed_accuracy_threshold = 0.10


@configclass
class RobotPlayEnvCfg(V17cPlayEnvCfg):
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.temperature = 3.0
        self.commands.base_velocity.zero_speed_accuracy_threshold = 0.10
