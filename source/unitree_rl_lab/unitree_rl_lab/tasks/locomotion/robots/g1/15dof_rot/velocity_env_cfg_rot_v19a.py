"""15-DOF rotation config V19a — Convenience alias.

Imports from velocity_env_cfg_rot_v19.py which contains the V19a continuous
marginal config. This file exists so the registration can reference v19a
explicitly.
"""

from .velocity_env_cfg_rot_v19 import RobotEnvCfg, RobotPlayEnvCfg  # noqa: F401

__all__ = ["RobotEnvCfg", "RobotPlayEnvCfg"]
