"""15-DOF rotation config V21f2 - corrected V21f.

V21f used start_step=6000, end_step=12000 in the *_decayed reward params,
intending iter-based decay. But _linear_step_decay reads
env.common_step_counter (env steps, NOT iterations). With
num_steps_per_env=24, those values resolved to iter 250..500, so the
gait-shaping rewards collapsed to zero almost immediately. The V21f run
therefore became an unintentional "no gait-shaping" baseline.

V21f2 fixes the unit conversion. Same physical intent as V21f:
    iter 6000  -> start fading        => start_step = 6000  * 24 = 144000
    iter 12000 -> fully faded to 0    => end_step   = 12000 * 24 = 288000

Walking-established iter (~6000) was determined from the V21e log:
bad_orient<5%, ep_len>950, r_track_lin>0.55, r_track_ang>0.5.

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


# Decay window in env-step units (= iter * num_steps_per_env, with num_steps_per_env=24).
_GAIT_DECAY_START = 6000 * 24   # iter 6000  -> 144000
_GAIT_DECAY_END   = 12000 * 24  # iter 12000 -> 288000


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
        weight=1.5,
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
