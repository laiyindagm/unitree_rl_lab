"""15-DOF rotation config V10b -- "Fixed Bucketed + Aligned Gait Clock".

Combines the two most promising V9 findings:
  - V9c's bucketed curriculum (now fixed with snapshot evaluation)
  - V9a's gait clock (but with ALIGNED period: walk=0.7 matching V6c gait reward)

V9a's problem was period MISMATCH: walk_period=1.0 (obs) vs gait reward period=0.7.
The policy got conflicting timing signals, causing non-periodic gait.

V10b fix: walk_period=0.7 (same as gait reward), run_period=0.5 (faster for running).
This gives the policy a consistent clock that matches the reward's expected contact pattern.

Changes from V6c:
  Curriculum:
    - Replace lin/ang_vel_cmd_levels with speed_bucketed_vel_curriculum (snapshot)
  Observations:
    - Add gait_phase_speed_adaptive (2D: sin/cos) -- walk_period=0.7 (aligned)
    - Observation: 54D -> 56D per frame, 270D -> 280D total
  Commands:
    - Initial lin_vel: +/-0.1, ang_vel: +/-0.3 (curriculum expands)
    - rel_standing: 0.20, rel_rotating: 0.30
  Rewards:
    - waist_joint_vel: -0.08 (stronger)
  Runner: BasePPORunnerV3Cfg
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg_rot import (
    CurriculumCfg as RotCurriculumCfg,
    ObservationsCfg as RotObservationsCfg,
)
from .velocity_env_cfg_rot_v6b import CommandsCfg as V6bCommandsCfg
from .velocity_env_cfg_rot_v6c import (
    EventCfg,
    RewardsCfg as V6cRewardsCfg,
    RobotEnvCfg as V6cEnvCfg,
)


# ---------- Observations: gait phase with ALIGNED period ----------
@configclass
class ObservationsCfg(RotObservationsCfg):

    @configclass
    class PolicyCfg(RotObservationsCfg.PolicyCfg):
        """Policy obs + gait phase clock (2D) -- period aligned to gait reward."""
        gait_phase = ObsTerm(
            func=mdp.gait_phase_speed_adaptive,
            params={
                "walk_period": 0.7,   # ALIGNED with V6c gait reward period
                "run_period": 0.5,    # faster for running
                "speed_threshold": 0.8,
                "decay_factor": 0.95,
                "standstill_threshold": 0.1,
                "command_name": "base_velocity",
            },
        )

    @configclass
    class CriticCfg(RotObservationsCfg.CriticCfg):
        """Critic obs + gait phase clock (2D) -- period aligned to gait reward."""
        gait_phase = ObsTerm(
            func=mdp.gait_phase_speed_adaptive,
            params={
                "walk_period": 0.7,
                "run_period": 0.5,
                "speed_threshold": 0.8,
                "decay_factor": 0.95,
                "standstill_threshold": 0.1,
                "command_name": "base_velocity",
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------- Curriculum: fixed snapshot-based bucketed ----------
@configclass
class CurriculumCfg(RotCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # type: ignore[assignment]
    ang_vel_cmd_levels = None  # type: ignore[assignment]
    speed_bucketed = CurrTerm(
        func=mdp.speed_bucketed_vel_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "low_speed_threshold": 0.3,
            "mid_speed_threshold": 0.6,
            "unlock_reward_ratio": 0.6,
        },
    )


# ---------- Commands: narrow initial range ----------
@configclass
class CommandsCfg(V6bCommandsCfg):
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.20,
        rel_rotating_envs=0.30,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.3, 0.3),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


# ---------- Rewards: V6c + stronger waist ----------
@configclass
class RewardsCfg(V6cRewardsCfg):
    waist_joint_vel = RewTerm(
        func=mdp.waist_joint_vel_penalty, weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


# ---------- Env ----------
@configclass
class RobotEnvCfg(V6cEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
