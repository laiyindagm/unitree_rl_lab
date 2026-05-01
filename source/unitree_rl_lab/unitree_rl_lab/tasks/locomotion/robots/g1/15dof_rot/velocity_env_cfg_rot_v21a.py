"""15-DOF rotation config V21a — V20l + mode-conditioned diagnostics only."""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v20l import RobotEnvCfg as V20lEnvCfg
from .velocity_env_cfg_rot_v20l import RobotPlayEnvCfg as V20lPlayEnvCfg


@configclass
class RobotEnvCfg(V20lEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.enable_diagnostics = True
        self.commands.base_velocity.diagnostic_every_n_iters = 100


@configclass
class RobotPlayEnvCfg(V20lPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.enable_diagnostics = True
        self.commands.base_velocity.diagnostic_every_n_iters = 100
