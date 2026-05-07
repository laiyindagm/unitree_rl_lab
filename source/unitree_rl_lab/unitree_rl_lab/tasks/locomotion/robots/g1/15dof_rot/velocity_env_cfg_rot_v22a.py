"""V22a env: identical to V21g, repackaged so the V22a task id binds cleanly.

V22a differs from V21g only at the algorithm/actor side (frozen V3 z_gait
injection); the environment, observations, rewards, and curriculum are
unchanged. We simply re-export V21g's env classes under V22a names.
"""
from __future__ import annotations

from .velocity_env_cfg_rot_v21g import (
    RobotEnvCfg as V21gEnvCfg,
    RobotPlayEnvCfg as V21gPlayEnvCfg,
)


class RobotEnvCfg(V21gEnvCfg):
    pass


class RobotPlayEnvCfg(V21gPlayEnvCfg):
    pass
