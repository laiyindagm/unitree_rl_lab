"""LIRPG PPO: VelocityEstimatorPPO + meta-gradient on intrinsic reward MLPs.

Adds two hooks to the standard PPO loop:
  (1) After each `process_env_step`, push `dones` to every IntrinsicRewardChannel
      so its meta-update can correctly truncate cross-episode bootstraps.
  (2) Before each PPO `update`, run one meta-gradient step on every channel,
      using the just-completed rollout buffer.

All channels are managed by `intrinsic_reward._REGISTRY` and are created lazily
by the corresponding reward-term functions in `mdp/rewards.py`.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from unitree_rl_lab.utils import intrinsic_reward as ir
from unitree_rl_lab.utils.velocity_estimator_ppo import VelocityEstimatorPPO


class LirpgVelocityEstimatorPPO(VelocityEstimatorPPO):
    """VelocityEstimatorPPO with LIRPG hooks."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        lirpg_meta_gamma: float = 0.99,
        lirpg_meta_lam: float = 0.95,
        lirpg_warmup_iters: int = 6000,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.lirpg_meta_gamma = float(lirpg_meta_gamma)
        self.lirpg_meta_lam = float(lirpg_meta_lam)
        self.lirpg_warmup_iters = int(lirpg_warmup_iters)
        self._lirpg_step = 0
        

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        super().process_env_step(obs, rewards, dones, extras)
        for chan in ir.all_channels().values():
            chan.record_dones(dones)

    def update(self) -> dict[str, float]:
        meta_logs: dict[str, float] = {}
        if self._lirpg_step >= self.lirpg_warmup_iters:
            for name, chan in ir.all_channels().items():
                r = chan.meta_update(gamma=self.lirpg_meta_gamma, lam=self.lirpg_meta_lam)
                for k, v in r.items():
                    meta_logs[f"lirpg_{name}/{k}"] = float(v)
        else:
            for chan in ir.all_channels().values():
                chan.clear_buffer()
            meta_logs["lirpg/warmup_remaining"] = float(
                self.lirpg_warmup_iters - self._lirpg_step
            )
        self._lirpg_step += 1

        out = super().update()
        out.update(meta_logs)
        return out
