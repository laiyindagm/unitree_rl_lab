"""15-DOF rotation config V21e — V21c env unchanged + actor-side velocity estimator (runner change only)."""

from isaaclab.utils import configclass

from .velocity_env_cfg_rot_v21c import RobotEnvCfg as V21cEnvCfg
from .velocity_env_cfg_rot_v21c import RobotPlayEnvCfg as V21cPlayEnvCfg


@configclass
class RobotEnvCfg(V21cEnvCfg):
    """Identical to V21c. The V21e change lives entirely in the runner cfg
    (``G115DofV21eVelocityEstimatorPPORunnerCfg``), which adds an actor-side
    detached velocity-estimator MLP supervised by an auxiliary regression loss.
    """


@configclass
class RobotPlayEnvCfg(V21cPlayEnvCfg):
    pass
