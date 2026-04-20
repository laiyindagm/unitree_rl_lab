"""15-DOF rotation config V18a — "Fixed Adaptive Continuous".

V17a baseline + framework bug fixes + higher temperature.

V17 Results & Lessons:
  V17a: rotation from 0.1 ✓, standing still ✓, BUT lin_vel from 0.4 (regression).
        ROOT CAUSE: min_sampling_prob=0.02 > 1/64=0.0156 → ALL softmax probs
        clamped to 0.02 → renormalized to uniform 0.0156 → adaptive sampling
        completely destroyed. V16a reached lin_vel=0.2 with same (broken) bins,
        but had 75% walking envs vs V17a's 60%.
  V17b: WORSE (ang>0.4, lin>0.4). Tighter incentive + waist damping hurt.
  V17c: rotation from 0.1, interesting gait variation, BUT zero-speed bin
        over-sampled (n=304k vs ~34k), curriculum stuck at 0.5 threshold.

Bug fixes applied (in velocity_command.py):
  1. Auto-scale min_sampling_prob floor: min(cfg, 0.5/N_active) → prevents
     uniform clamping for any grid size.
  2. Exclude standing/rotating envs from bin EMA updates → stops zero-speed
     bin from appearing artificially "hard".
  3. Exclude zero-speed cell from adaptive softmax → standing/rotating
     allocation already handles zero-speed practice.

Changes from V17a:
  - temperature = 3.0 (from 2.0): stronger differentiation in 64-cell softmax.
    With perfs clustered at 0.14-0.27, difficulty spread is only ~0.13.
    temp=2.0 → logit spread=0.26 → minimal softmax differentiation.
    temp=3.0 → logit spread=0.39 → better adaptive signal.

Hypothesis: fixed adaptive + higher temp → more practice on hard low-speed
cells → lin_vel improved from 0.4 back toward 0.2.
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v17a import (
    RobotEnvCfg as V17aEnvCfg,
    RobotPlayEnvCfg as V17aPlayEnvCfg,
)


@configclass
class RobotEnvCfg(V17aEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.temperature = 3.0


@configclass
class RobotPlayEnvCfg(V17aPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.temperature = 3.0
