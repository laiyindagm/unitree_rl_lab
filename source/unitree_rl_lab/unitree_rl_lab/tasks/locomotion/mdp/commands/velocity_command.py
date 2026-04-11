from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg, UniformVelocityCommand
from isaaclab.utils import configclass

import torch
from collections.abc import Sequence


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    rel_rotating_envs: float = 0.0
    """Probability of environments that should be pure-rotation (lin_vel=0, ang_vel sampled). Defaults to 0.0."""


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Extends the base velocity command with rotating-only environments."""

    cfg: UniformLevelVelocityCommandCfg

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_rotating_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # update rotating envs
        r = torch.empty(len(env_ids), device=self.device)
        self.is_rotating_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_rotating_envs

    def _update_command(self):
        super()._update_command()
        # Enforce pure rotation (zero lin vel) for rotating envs
        rotating_env_ids = self.is_rotating_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[rotating_env_ids, :2] = 0.0


# Patch the cfg to use our custom command class
UniformLevelVelocityCommandCfg.class_type = UniformLevelVelocityCommand
