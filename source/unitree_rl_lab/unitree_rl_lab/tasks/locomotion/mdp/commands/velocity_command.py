from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Performance-weighted adaptive velocity command
# ---------------------------------------------------------------------------


@configclass
class PerformanceWeightedVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Config for performance-weighted velocity command sampling.

    Divides the command space into bins by speed magnitude and tracks
    per-bin tracking performance via EMA.  Bins where the policy performs
    poorly receive *higher* sampling probability (inverse-performance
    weighting).  This is agnostic to which speed is "harder" — it
    discovers difficulty empirically.
    """

    num_speed_bins: int = 5
    """Number of speed bins to divide the linear velocity magnitude range into."""

    num_yaw_bins: int = 3
    """Number of bins for angular velocity magnitude."""

    ema_alpha: float = 0.05
    """EMA smoothing factor for per-bin performance tracking (lower = slower update)."""

    min_sampling_prob: float = 0.05
    """Floor probability for any bin so it never starves completely."""

    temperature: float = 1.0
    """Softmax temperature for converting inverse-performance to probability.
    Higher = more uniform; lower = more focused on weak bins."""


class PerformanceWeightedVelocityCommand(UniformLevelVelocityCommand):
    """Velocity command that biases sampling toward speed bins where
    the policy tracks poorly.

    On each resample the class:
      1. Chooses a *speed bin* for each env according to performance-weighted
         probabilities (worse tracking → higher probability).
      2. Samples a velocity command uniformly within that bin's range.
      3. Independently samples angular velocity from yaw bins with the
         same inverse-performance logic.

    Per-bin performance is updated externally by the companion curriculum
    term ``performance_weighted_vel_curriculum`` which calls
    ``update_bin_performance()`` with episode-level reward data.
    """

    cfg: PerformanceWeightedVelocityCommandCfg

    def __init__(self, cfg: PerformanceWeightedVelocityCommandCfg, env):
        super().__init__(cfg, env)
        n_speed = cfg.num_speed_bins
        n_yaw = cfg.num_yaw_bins

        # Per-bin EMA of normalised tracking reward (init to 0.5 → neutral)
        self._speed_perf = torch.full((n_speed,), 0.5, device=self.device)
        self._yaw_perf = torch.full((n_yaw,), 0.5, device=self.device)

        # Sampling probabilities (init uniform)
        self._speed_probs = torch.full((n_speed,), 1.0 / n_speed, device=self.device)
        self._yaw_probs = torch.full((n_yaw,), 1.0 / n_yaw, device=self.device)

        # Track which bin each env was assigned to (for performance attribution)
        self._env_speed_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_yaw_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # -- public helpers for the curriculum term --

    @property
    def speed_bin_edges(self) -> torch.Tensor:
        """Return (num_speed_bins+1,) tensor of bin edges for linear speed magnitude."""
        ranges = self.cfg.limit_ranges
        lo = 0.0  # speed magnitude always starts at 0
        hi = max(abs(ranges.lin_vel_x[0]), abs(ranges.lin_vel_x[1]),
                 abs(ranges.lin_vel_y[0]), abs(ranges.lin_vel_y[1]))
        return torch.linspace(lo, hi, self.cfg.num_speed_bins + 1, device=self.device)

    @property
    def yaw_bin_edges(self) -> torch.Tensor:
        """Return (num_yaw_bins+1,) tensor of bin edges for |ang_vel_z|."""
        ranges = self.cfg.limit_ranges
        hi = max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1]))
        return torch.linspace(0.0, hi, self.cfg.num_yaw_bins + 1, device=self.device)

    def update_bin_performance(
        self,
        speed_bin_ids: torch.Tensor,
        speed_rewards: torch.Tensor,
        yaw_bin_ids: torch.Tensor,
        yaw_rewards: torch.Tensor,
    ) -> None:
        """Update EMA performance for each bin with new episode data.

        Called by the curriculum term after each evaluation gate.

        Args:
            speed_bin_ids: (N,) bin index for each sample
            speed_rewards: (N,) normalised reward [0, 1] for each sample
            yaw_bin_ids: (M,) bin index for each sample
            yaw_rewards: (M,) normalised reward [0, 1] for each sample
        """
        alpha = self.cfg.ema_alpha
        for b in range(self.cfg.num_speed_bins):
            mask = speed_bin_ids == b
            if mask.sum() > 0:
                mean_r = speed_rewards[mask].mean()
                self._speed_perf[b] = (1.0 - alpha) * self._speed_perf[b] + alpha * mean_r

        for b in range(self.cfg.num_yaw_bins):
            mask = yaw_bin_ids == b
            if mask.sum() > 0:
                mean_r = yaw_rewards[mask].mean()
                self._yaw_perf[b] = (1.0 - alpha) * self._yaw_perf[b] + alpha * mean_r

        # Recompute probabilities: inverse-performance softmax
        self._speed_probs = self._inverse_perf_to_probs(self._speed_perf)
        self._yaw_probs = self._inverse_perf_to_probs(self._yaw_perf)

    def _inverse_perf_to_probs(self, perf: torch.Tensor) -> torch.Tensor:
        """Convert performance vector to sampling probabilities.

        Lower performance → higher probability.  Uses softmax over
        (1 - perf) / temperature, then clamps to min_sampling_prob.
        """
        inv = (1.0 - perf.clamp(0.0, 1.0)) / max(self.cfg.temperature, 1e-6)
        probs = torch.softmax(inv, dim=0)
        # Apply floor
        probs = probs.clamp(min=self.cfg.min_sampling_prob)
        probs = probs / probs.sum()
        return probs

    # -- override resampling --

    def _resample_command(self, env_ids: Sequence[int]):
        # First let the parent do its standard uniform + rotating logic
        super()._resample_command(env_ids)

        if len(env_ids) == 0:
            return

        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # -- Resample linear velocity via performance-weighted bins --
        speed_edges = self.speed_bin_edges
        chosen_speed_bins = torch.multinomial(
            self._speed_probs.unsqueeze(0).expand(n, -1), 1
        ).squeeze(-1)  # (n,)
        self._env_speed_bin[env_ids_t] = chosen_speed_bins

        # For each chosen bin, sample a speed magnitude uniformly within the bin
        lo_speed = speed_edges[chosen_speed_bins]
        hi_speed = speed_edges[chosen_speed_bins + 1]
        sampled_speed = lo_speed + (hi_speed - lo_speed) * torch.rand(n, device=self.device)

        # Random direction angle for lin vel (preserves x/y ratio diversity)
        ranges = self.cfg.ranges
        # Sample angle biased toward forward: use the actual range asymmetry
        # to determine sign/direction
        lx_lo, lx_hi = ranges.lin_vel_x[0], ranges.lin_vel_x[1]
        ly_lo, ly_hi = ranges.lin_vel_y[0], ranges.lin_vel_y[1]

        # Sample angle uniformly in [-pi, pi], then clamp to feasible range
        angle = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        vx = sampled_speed * torch.cos(angle)
        vy = sampled_speed * torch.sin(angle)

        # Clamp to the current curriculum range
        vx = vx.clamp(lx_lo, lx_hi)
        vy = vy.clamp(ly_lo, ly_hi)

        self.vel_command_b[env_ids_t, 0] = vx
        self.vel_command_b[env_ids_t, 1] = vy

        # -- Resample angular velocity via performance-weighted bins --
        yaw_edges = self.yaw_bin_edges
        chosen_yaw_bins = torch.multinomial(
            self._yaw_probs.unsqueeze(0).expand(n, -1), 1
        ).squeeze(-1)  # (n,)
        self._env_yaw_bin[env_ids_t] = chosen_yaw_bins

        lo_yaw = yaw_edges[chosen_yaw_bins]
        hi_yaw = yaw_edges[chosen_yaw_bins + 1]
        sampled_yaw_mag = lo_yaw + (hi_yaw - lo_yaw) * torch.rand(n, device=self.device)

        # Random sign
        sign = torch.where(torch.rand(n, device=self.device) < 0.5,
                           torch.ones(n, device=self.device),
                           -torch.ones(n, device=self.device))
        az_lo, az_hi = ranges.ang_vel_z[0], ranges.ang_vel_z[1]
        sampled_yaw = (sampled_yaw_mag * sign).clamp(az_lo, az_hi)
        self.vel_command_b[env_ids_t, 2] = sampled_yaw

        # Re-apply standing/rotating env overrides (from parent)
        standing_mask = self.is_standing_env[env_ids_t]
        self.vel_command_b[env_ids_t[standing_mask]] = 0.0
        rotating_mask = self.is_rotating_env[env_ids_t]
        self.vel_command_b[env_ids_t[rotating_mask], :2] = 0.0


# Patch cfg → class mapping
PerformanceWeightedVelocityCommandCfg.class_type = PerformanceWeightedVelocityCommand
