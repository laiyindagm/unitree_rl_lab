"""15-DOF rotation config V5c — V5a + enhanced Domain Randomization.

Based on V5a (rot base + 4 behavioral rewards) with added DR:
  1. Actuator gain randomization (stiffness/damping scale 0.8-1.2)
  2. Center of mass randomization (torso +-5cm x/y, +-2cm z)
  3. External force/torque enabled (+-10N, +-3Nm)
  4. Stronger push robot (+-0.8 m/s, 5-8s interval)
  5. action_rate slightly stronger (-0.08 -> -0.10) for DR robustness
  Runner: BasePPORunnerV3Cfg (clip_actions=100, no symmetry)
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import EventCfg as RotEventCfg, TRAIN_JOINT_NAMES
from .velocity_env_cfg_rot_v5a import (
    RewardsCfg as V5aRewardsCfg,
    RobotEnvCfg as V5aEnvCfg,
)


@configclass
class EventCfg(RotEventCfg):
    """Rot events + DR enhancements."""

    # Override: enable external force/torque
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-3.0, 3.0),
        },
    )
    # Override: stronger push with variable interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
    )
    # New: actuator gain randomization
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=TRAIN_JOINT_NAMES),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    # New: center of mass randomization
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
        },
    )


@configclass
class RewardsCfg(V5aRewardsCfg):
    """V5a rewards with slightly stronger smoothness for DR robustness."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.10)


@configclass
class RobotEnvCfg(V5aEnvCfg):
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
