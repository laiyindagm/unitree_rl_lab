"""15-DOF rotation config V5b — V5a env + mirror loss runner.

Same environment as V5a (rot base + 4 behavioral rewards).
Uses BasePPORunnerV5bCfg which enables symmetry MIRROR LOSS
(not data augmentation) to encourage left-right policy symmetry.
"""

from .velocity_env_cfg_rot_v5a import RobotEnvCfg, RobotPlayEnvCfg

__all__ = ["RobotEnvCfg", "RobotPlayEnvCfg"]
