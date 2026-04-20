"""15-DOF rotation config V16b — "Scheduled Movement Incentive".

Addresses dead zone from the REWARD LANDSCAPE side via a scheduled mechanism.

Problem: Standard exp kernel gives "free" standing reward (0.698 at cmd=0.3).
All previous attempts to reduce this reward killed early training:
  - V8b/V11a adaptive sigma:    standing dropped to 0.018 -> 55% bad_orient
  - V14a baselined tracking:    standing = 0 -> 65% bad_orient
Key insight: the dead zone local optimum forms AFTER balance is learned (~iter
3000, when bad_orient < 5%). Adding an anti-standing penalty AFTER this point
preserves early stability while breaking the standing optimum.

New reward term: movement_incentive_scheduled
  - Tent-shaped: penalty proportional to (1 - vel/sigma) when vel < sigma
  - Scales with command magnitude (larger cmd -> stronger push)
  - Scheduled: 0% during iter 0-3000, ramps to 100% by iter 5000
  - Smooth gradient everywhere (unlike binary cmd_nonresponse)

Why this is different from ALL previous dead-zone attempts:
  V14a baselined tracking:  reduced standing reward from iter 0 -> 65% bad_orient
  V8b/V11a adaptive sigma:  reduced easy reward globally -> training collapse
  V15a cmd_nonresponse:     binary threshold (vel<0.05), gameable, 0.003/step
  Old V16b cmd_nonresponse: -2.0 from iter 0 -> destabilized upper body
  V11d tracking bonus:      chicken-and-egg: policy doesn't explore -> bonus=0
  V13c weight boost:        proportional scaling -> same relative advantage
  This design: SCHEDULED + SMOOTH GRADIENT + PENALTY (not bonus) + TARGETED

Math at full schedule (iter 5000+), cmd=0.3:
  Standing: tracking(0.698) + penalty(-1.0*0.3) = 0.398
  Perfect:  tracking(1.0) + penalty(0) = 1.0
  Marginal: 0.602 >> action costs (~0.21)
  Dead zone threshold: ~0.05 (down from 0.3!)

Orthogonal to V16a: V16a changes WHERE to sample, V16b changes the reward
landscape around the standing equilibrium point. Independent axes.

Changes from V15a:
  1. NEW movement_incentive_scheduled (weight=-1.0, scheduled iter 3000-5000)
  Everything else unchanged (tracking kernels, penalties, curriculum, commands).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v15a import (
    RewardsCfg as V15aRewardsCfg,
    RobotEnvCfg as V15aEnvCfg,
)


@configclass
class RewardsCfg(V15aRewardsCfg):
    """V15a rewards + scheduled movement incentive.

    Only one addition: movement_incentive_scheduled.
    All tracking kernels, penalties, and other rewards inherited unchanged.
    """

    movement_incentive = RewTerm(
        func=mdp.movement_incentive_scheduled,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "cmd_threshold": 0.05,
            "start_step": 72000,   # ~iter 3000 (24 steps/iter)
            "end_step": 120000,    # ~iter 5000
        },
    )


@configclass
class RobotEnvCfg(V15aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
