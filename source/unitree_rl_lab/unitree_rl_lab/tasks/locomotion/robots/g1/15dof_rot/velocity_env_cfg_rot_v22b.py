"""V22b env: identical to V21g/V22a (env-side); algorithm differs (CIC intrinsic)."""
from __future__ import annotations

from .velocity_env_cfg_rot_v21g import (
    RobotEnvCfg as V21gEnvCfg,
    RobotPlayEnvCfg as V21gPlayEnvCfg,
)


class RobotEnvCfg(V21gEnvCfg):
    pass


class RobotPlayEnvCfg(V21gPlayEnvCfg):
    pass
