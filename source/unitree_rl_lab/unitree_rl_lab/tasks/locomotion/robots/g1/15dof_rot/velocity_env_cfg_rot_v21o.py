"""15-DOF rotation config V21o - validated bilevel linear LIRPG.

V21o keeps V21l's learnable linear tracking reward:

    r_phi = leaky(1 - |v - v_cmd| * f_phi(v, v_cmd))

with V21l's prior and monotonicity constraints. The experiment changes only
the runner/algorithm: meta-updates are validated after a differentiable shadow
PPO actor update, instead of using current-rollout reward/advantage correlation.
"""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v21l import (
    RewardsCfg,
    RobotEnvCfg as V21lEnvCfg,
    RobotPlayEnvCfg as V21lPlayEnvCfg,
)


@configclass
class RobotEnvCfg(V21lEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(V21lPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
