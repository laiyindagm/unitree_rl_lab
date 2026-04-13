"""15-DOF rotation config V9a — "Gait Signal Fix" (FEAP-inspired).

Problem: V8 showed policy has NO gait clock input (gait_phase was never enabled
in rot configs). Fixed period=0.7 doesn't distinguish low/high speed. No
"stop stepping" signal at standstill → oscillation.

Fix (observation layer only, zero reward change):
  - Enable gait_phase_speed_adaptive: speed-dependent period + standstill decay
  - walk_period=1.0 (slow, deliberate steps at low speed)
  - run_period=0.7 (matches V6c gait reward period)
  - Standstill: phase *= 0.95/step → decays to 0 → "stop stepping" signal
  - Observation: 54D → 56D per frame, 270D → 280D total

  Also:
  - Keep waist_joint_vel -0.04 (V8c proved effective)
  - Restore rel_rotating=0.30 (V8c's 0.25 hurt ang_vel curriculum)

Changes from V6c:
  Observations:
    - Add gait_phase_speed_adaptive (2D: sin/cos)
  Rewards:
    - waist_joint_vel_penalty: -0.04 (from V8c)
  Commands:
    - rel_rotating: 0.30 (restored from V6b, V8c had 0.25)
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    ObservationsCfg as RotObservationsCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v6c import (
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)


# ---------- Observations: add speed-adaptive gait phase ----------
@configclass
class ObservationsCfg(RotObservationsCfg):

    @configclass
    class PolicyCfg(RotObservationsCfg.PolicyCfg):
        """Policy obs + gait phase clock (2D)."""
        gait_phase = ObsTerm(
            func=mdp.gait_phase_speed_adaptive,
            params={
                "walk_period": 1.0,
                "run_period": 0.7,
                "speed_threshold": 0.8,
                "decay_factor": 0.95,
                "standstill_threshold": 0.1,
                "command_name": "base_velocity",
            },
        )

    @configclass
    class CriticCfg(RotObservationsCfg.CriticCfg):
        """Critic obs + gait phase clock (2D)."""
        gait_phase = ObsTerm(
            func=mdp.gait_phase_speed_adaptive,
            params={
                "walk_period": 1.0,
                "run_period": 0.7,
                "speed_threshold": 0.8,
                "decay_factor": 0.95,
                "standstill_threshold": 0.1,
                "command_name": "base_velocity",
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------- Rewards: V6c + waist vel ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


# ---------- Commands: restore rel_rotating=0.30 ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    """V6b commands (rel_rotating=0.30) — unchanged."""
    pass


# ---------- Env ----------
@configclass
class RobotEnvCfg(V6cEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
