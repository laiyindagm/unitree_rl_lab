"""15-DOF rotation config V17b — "Tighter Incentive + Waist Damping".

V17a baseline + two targeted improvements:

1. movement_incentive std: 0.25 -> 0.15
   V16b's incentive contributed only -0.0069 because std=0.25 was too
   forgiving — the robot only needs vel>0.25 to fully escape the penalty.
   Tightening to 0.15 means the robot must reach vel>0.15 to escape,
   which directly attacks the 0.1-0.2 dead zone.
   Math: at cmd=0.3, standing penalty = 1.0 * 0.3 = 0.3 (same).
   But at vel=0.15, old penalty = (1-0.15/0.25)*0.3 = 0.12.
   New penalty = (1-0.15/0.15)*0.3 = 0.0 (fully escaped at vel=0.15).
   The gradient is STEEPER near vel=0, pushing harder to start moving.

2. Stronger waist damping for upper body stability:
   waist_roll_vel:  -0.20 -> -0.35  (75% increase, targets rotation wobble)
   waist_pitch_vel: -0.06 -> -0.10  (67% increase)
   V16a waist_roll contribution was -0.1200, V16b was -0.1490 — both
   indicate significant waist oscillation. Stronger damping should
   reduce rotation wobble without affecting gait (V14b lesson: keeping
   waist at full strength is safe; only HALVING caused problems).

Hypothesis: tighter incentive pushes through 0.1-0.15 dead zone +
stronger waist damping eliminates rotation wobble.
Risk: tighter std might create constant penalty even when tracking well
at medium speeds (vel 0.10-0.15 range).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v17a import (
    RobotEnvCfg as V17aEnvCfg,
)
from .velocity_env_cfg_rot_v16b import (
    RewardsCfg as V16bRewardsCfg,
)


@configclass
class RewardsCfg(V16bRewardsCfg):
    """V16b rewards with tighter movement incentive + stronger waist damping."""

    # Tighter movement incentive: std 0.25 -> 0.15
    movement_incentive = RewTerm(
        func=mdp.movement_incentive_scheduled,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.15,           # was 0.25 in V16b
            "cmd_threshold": 0.05,
            "start_step": 72000,   # ~iter 3000
            "end_step": 120000,    # ~iter 5000
        },
    )

    # Stronger waist damping for upper body stability
    waist_roll_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.35,  # was -0.20
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )
    waist_pitch_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.10,  # was -0.06
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])},
    )


@configclass
class RobotEnvCfg(V17aEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
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
