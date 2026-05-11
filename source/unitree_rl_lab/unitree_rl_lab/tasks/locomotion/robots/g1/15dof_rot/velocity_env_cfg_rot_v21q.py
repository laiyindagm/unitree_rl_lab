"""15-DOF rotation config V21q.

V21q uses the same command-conditioned learnable tracking reward as V21p.
The experimental change is in the runner: after validated meta-updating the
reward parameters, the updated reward is committed back into the full PPO
rollout update.
"""

from .velocity_env_cfg_rot_v21p import RewardsCfg, RobotEnvCfg, RobotPlayEnvCfg

__all__ = ["RewardsCfg", "RobotEnvCfg", "RobotPlayEnvCfg"]
