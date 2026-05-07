"""15-DOF rotation config V21f - V21e env + decayed gait-shaping rewards.

V21e log analysis (run 2026-05-01_15-01-01) showed walking is firmly
established around iter ~6000:
  - bad_orientation termination < 5%
  - mean_episode_length > 950
  - track_lin_vel_xy reward > 0.55
  - track_ang_vel_z   reward > 0.5

After that point the three explicit gait-shape priors become *constraints*
that suppress gait diversity at higher speeds:
  - gait                    (feet_gait_speed_scaled)        : locks period to 0.7s
  - feet_clearance          (foot_clearance_speed_scaled)   : pins swing height to 0.1m
  - rotation_single_support                                 : forces alternating support during yaw

V21f swaps each to its *_decayed variant with a linear schedule
start_step=6000 -> end_step=12000 (6k-iter smooth fade so PPO sees no
sudden gradient cliff). All other rewards inherit from V21c unchanged.

Runner is identical to V21e (G115DofV21eVelocityEstimatorPPORunnerCfg):
the actor-side detached velocity estimator is preserved.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot_v21c import (
    RewardsCfg as V21cRewardsCfg,
)
from .velocity_env_cfg_rot_v21e import (
    RobotEnvCfg as V21eEnvCfg,
    RobotPlayEnvCfg as V21ePlayEnvCfg,
)


# Walking-established iter from V21e log analysis. Smooth fade over 6k iters
# so the gait-shaping gradient signal disappears gradually rather than as a cliff.
_GAIT_DECAY_START = 6000
_GAIT_DECAY_END = 12000


@configclass
class RewardsCfg(V21cRewardsCfg):
    gait = RewTerm(
        func=mdp.feet_gait_speed_scaled_decayed,
        weight=0.5,
        params={
            "period": 0.7,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "speed_gate": 0.3,
            "start_step": _GAIT_DECAY_START,
            "end_step": _GAIT_DECAY_END,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_speed_scaled_decayed,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "speed_gate": 0.3,
            "start_step": _GAIT_DECAY_START,
            "end_step": _GAIT_DECAY_END,
        },
    )
    rotation_single_support = RewTerm(
        func=mdp.rotation_single_support_reward_decayed,
        weight=1.5,  # match V21c weight (the active reward listing showed 1.5)
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "start_step": _GAIT_DECAY_START,
            "end_step": _GAIT_DECAY_END,
        },
    )


@configclass
class RobotEnvCfg(V21eEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21ePlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
