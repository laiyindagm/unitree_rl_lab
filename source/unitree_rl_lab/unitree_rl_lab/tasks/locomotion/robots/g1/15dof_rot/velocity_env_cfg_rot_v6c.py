"""15-DOF rotation config V6c — "Robust Full": V6b content + stronger DR + gait tuning.

Combines all V6a stability + V6b rotation improvements, and adds stronger DR
to improve sim2sim transfer robustness. Also tunes gait parameters to reduce
the aggressive leg motions observed in V5a/V5b.

Changes from V6b:
  DR enhancements (beyond V5c):
    - External force: +-10N -> +-15N, torque +-3Nm -> +-5Nm
    - Push: +-0.8 m/s -> +-1.0 m/s
    - Actuator gains: 0.8-1.2 -> 0.75-1.25 (wider range)
    - Friction range: 0.3-1.0 -> 0.2-1.0 (lower bound reduced)
  Gait tuning:
    - gait period: 0.8 -> 0.7  (slightly faster for better dynamic stability)
    - feet_slide: -0.2 -> -0.3  (stronger slide penalty)
  Smoothness:
    - action_rate: -0.10 -> -0.12  (slightly more for DR robustness)
  Runner: BasePPORunnerV3Cfg
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v5c import EventCfg as V5cEventCfg
from .velocity_env_cfg_rot_v6b import (
    CommandsCfg,
    RewardsCfg as V6bRewardsCfg,
    RobotEnvCfg as V6bEnvCfg,
)


@configclass
class EventCfg(V5cEventCfg):
    """V5c DR with stronger ranges."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-15.0, 15.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg(V6bRewardsCfg):
    """V6b rewards + gait tuning + stronger smoothness."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.12)
    gait = RewTerm(
        func=mdp.feet_gait, weight=0.5,
        params={
            "period": 0.7, "offset": [0.0, 0.5], "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class RobotEnvCfg(V6bEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
