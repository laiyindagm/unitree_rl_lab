"""Velocity-estimator PPO.

Subclass of ``TransformerPPO`` that fixes the achieved-velocity target extraction
for the actor-side velocity-prediction head used by V21e.

V21d's ``TransformerPPO._get_achieved_velocity_targets`` assumed that the critic
flat-observation tensor was laid out as ``(history_len, single_frame_dim)`` so
that the last frame could be sliced via ``view(-1, H, D)[:, -1, :]``. IsaacLab's
``ObservationManager``, however, concatenates per-term history buffers
**term-major**: each term contributes ``history_len * term_dim`` contiguous
values, then all terms are concatenated. Therefore the position of the most
recent frame for any single term is ``term_offset + (history_len - 1) * term_dim``.

This subclass lets the runner cfg pass explicit indices into the critic flat
observation for the three regression targets ``(vx, vy, wz)`` plus per-component
inverse scales (e.g. ``1 / 0.2 = 5.0`` for ``base_ang_vel`` which is scaled by
``0.2`` in CriticCfg).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from unitree_rl_lab.utils.transformer_ppo import TransformerPPO


class VelocityEstimatorPPO(TransformerPPO):
    """PPO with corrected achieved-velocity target extraction."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        velocity_target_indices: list[int] | tuple[int, int, int] = (12, 13, 29),
        velocity_target_scales: list[float] | tuple[float, float, float] = (1.0, 1.0, 5.0),
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if len(velocity_target_indices) != 3 or len(velocity_target_scales) != 3:
            raise ValueError("velocity_target_indices/scales must have length 3 (vx, vy, wz).")
        self.velocity_target_indices = [int(i) for i in velocity_target_indices]
        self.velocity_target_scales = [float(s) for s in velocity_target_scales]

    def _get_achieved_velocity_targets(self, obs_t: TensorDict) -> torch.Tensor:
        critic_flat = self._flatten_obs(obs_t, self.critic.obs_groups)
        i0, i1, i2 = self.velocity_target_indices
        s0, s1, s2 = self.velocity_target_scales
        return torch.stack(
            [
                critic_flat[:, i0] * s0,
                critic_flat[:, i1] * s1,
                critic_flat[:, i2] * s2,
            ],
            dim=-1,
        )
