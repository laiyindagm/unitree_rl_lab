"""LIRPG: per-channel intrinsic tracking-reward with learnable slope.

Parameterization
----------------
Instead of directly outputting reward, the MLP outputs a **slope** f_phi > 0,
and the reward is:

    r_phi = leaky_{slope_neg}(1 - |v - v_cmd| * f_phi(v, v_cmd))

This is structurally superior to direct-reward output because:
  1. Zero-error → r = 1 always (structure, not learned).
  2. r is monotonically non-increasing in |err| (f_phi > 0 forced via softplus).
  3. Prior = constant slope 1/b_abs (V21k baseline), trust-region constraint is
     in interpretable "slope space".
  4. Output is naturally bounded: r ∈ (slope_neg * (1 - err*f_max), 1].

Meta-objective
--------------
At each PPO rollout end, one gradient step on phi minimising:

    L = -E[adv_norm.detach() * r_phi]  +  lambda_l2 * ||phi - phi_0||^2

where adv_norm is the MC-discounted task return (-|err|) normalised per batch.

Warmup
------
The channel itself is always ready; warmup (no meta-update for N iters) is
controlled by LirpgVelocityEstimatorPPO in lirpg_ppo.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_REGISTRY: dict[str, "IntrinsicRewardChannel"] = {}
_RECORDING_ENABLED = True


def set_recording_enabled(enabled: bool) -> None:
    """Enable/disable rollout-buffer recording inside reward evaluation.

    Validation rollouts for bilevel reward learning still need the environment
    reward to be computed, but must not pollute the just-collected train
    rollout buffers used for the differentiable inner update.
    """

    global _RECORDING_ENABLED
    _RECORDING_ENABLED = bool(enabled)


def get_or_create(name: str, **kwargs) -> "IntrinsicRewardChannel":
    if name not in _REGISTRY:
        _REGISTRY[name] = IntrinsicRewardChannel(name=name, **kwargs)
    return _REGISTRY[name]


def get_or_create_command(name: str, **kwargs) -> "CommandConditionedIntrinsicRewardChannel":
    if name not in _REGISTRY:
        _REGISTRY[name] = CommandConditionedIntrinsicRewardChannel(name=name, **kwargs)
    return _REGISTRY[name]  # type: ignore[return-value]


def get(name: str) -> "IntrinsicRewardChannel | None":
    return _REGISTRY.get(name)


def all_channels() -> dict[str, "IntrinsicRewardChannel"]:
    return _REGISTRY


def reset_registry() -> None:
    _REGISTRY.clear()


def _leaky(raw: torch.Tensor, slope_neg: float) -> torch.Tensor:
    """Leaky activation: identity above 0, slope_neg below 0, capped at 1."""
    pos = torch.clamp(raw, min=0.0)
    neg = slope_neg * torch.clamp(raw, max=0.0)
    return (pos + neg).clamp(max=1.0)


class IntrinsicRewardChannel(nn.Module):
    """One learnable-slope reward channel for one velocity axis.

    MLP: features -> raw_logit -> softplus -> f_phi (positive slope)
    Reward: r = leaky_{slope_neg}(1 - |err| * f_phi)
    Baseline (V21k): f = 1/b_abs = constant
    """

    def __init__(
        self,
        name: str,
        feature_dim: int,           # lin: 4 (vx,vy,cmd_x,cmd_y); ang: 2 (wz,cmd_wz)
        err_dim: int,               # lin: 2; ang: 1
        b: float = 0.7834,         # V21k b_abs = LINEAR_REL_B_RATIO * std
        slope_neg: float = 0.1,
        hidden: tuple[int, ...] = (64, 64),
        meta_lr: float = 2e-5,
        prior_weight: float = 1.0,
        prior_param_l2_coef: float = 1e-1,
        device: str = "cuda",
        init_fit_steps: int = 500,
        init_fit_lr: float = 1e-3,
        init_fit_samples: int = 4096,
        sample_lo: float = -1.5,
        sample_hi: float = 1.5,
        f_phi_max: float = 8.0,     # hard clamp on slope to prevent explosion
        mono_coef: float = 0.0,     # weight for e*f monotonicity penalty (0 = disabled)
    ):
        super().__init__()
        self.name = name
        self.feature_dim = feature_dim
        self.err_dim = err_dim
        self.b = float(b)
        self.slope_neg = float(slope_neg)
        self.prior_weight = float(prior_weight)
        self.prior_param_l2_coef = float(prior_param_l2_coef)
        self._sample_range = (sample_lo, sample_hi)
        self.f_phi_max = float(f_phi_max)
        self.mono_coef = float(mono_coef)

        # Escape inference_mode: reward terms run inside torch.inference_mode
        # during env.step, which forbids autograd. We need grad for prior fit.
        with torch.inference_mode(False), torch.enable_grad():
            layers: list[nn.Module] = []
            prev = feature_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ELU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
            self.to(device)

            self._init_to_prior(init_fit_steps, init_fit_lr, init_fit_samples)

            self.optim = torch.optim.Adam(self.parameters(), lr=meta_lr)
            self._prior_state: dict[str, torch.Tensor] = {
                k: v.detach().clone() for k, v in self.state_dict().items()
            }

        self._features_list: list[torch.Tensor] = []
        self._task_err_list: list[torch.Tensor] = []
        self._reward_list: list[torch.Tensor] = []
        self._dones_list: list[torch.Tensor] = []
        self._prev_meta_state: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }

    # ------------------------------------------------------------------ prior

    def _compute_err(self, features: torch.Tensor) -> torch.Tensor:
        v = features[..., : self.err_dim]
        v_cmd = features[..., self.err_dim :]
        if self.err_dim > 1:
            return (v - v_cmd).norm(dim=-1)
        else:
            return (v - v_cmd).abs().squeeze(-1)

    def _init_to_prior(self, n_steps: int, lr: float, n_samples: int) -> None:
        """Fit net so that softplus(net(feat)) ≈ 1/b_abs (V21k slope)."""
        device = next(self.parameters()).device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        lo, hi = self._sample_range
        target_slope = torch.tensor(1.0 / self.b, device=device)

        for _ in range(n_steps):
            feats = torch.empty(n_samples, self.feature_dim, device=device).uniform_(lo, hi)
            pred_slope = F.softplus(self.net(feats)).squeeze(-1)
            loss = F.mse_loss(pred_slope, target_slope.expand_as(pred_slope))
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            feats = torch.empty(n_samples, self.feature_dim, device=device).uniform_(lo, hi)
            pred_slope = F.softplus(self.net(feats)).squeeze(-1)
            mse = F.mse_loss(pred_slope, target_slope.expand_as(pred_slope)).item()
        print(
            f"[IntrinsicReward:{self.name}] init slope MSE vs V21k (f={1/self.b:.4f}) = {mse:.6f}"
        )

    # ----------------------------------------------------------------- runtime

    @torch.no_grad()
    def evaluate(
        self,
        v: torch.Tensor,
        v_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for env reward term. No grad. Records (feat, err)."""
        if v.dim() == 1:
            v = v.unsqueeze(-1)
        if v_cmd.dim() == 1:
            v_cmd = v_cmd.unsqueeze(-1)
        feats = torch.cat([v, v_cmd], dim=-1)

        if self.err_dim > 1:
            err = (v - v_cmd).norm(dim=-1)
        else:
            err = (v - v_cmd).abs().squeeze(-1)

        f_phi = F.softplus(self.net(feats)).squeeze(-1).clamp(max=self.f_phi_max)
        raw = 1.0 - err * f_phi
        r = _leaky(raw, self.slope_neg)

        # Clone to plain (non-inference) tensors for the meta-update buffer.
        if _RECORDING_ENABLED:
            self._features_list.append(feats.detach().clone())
            self._task_err_list.append(err.detach().clone())
            self._reward_list.append(r.detach().clone())
        return r

    def record_dones(self, dones: torch.Tensor) -> None:
        """Called by the LIRPG PPO subclass after each env step."""
        self._dones_list.append(dones.detach().to(torch.float32).clone())

    def buffer_len(self) -> int:
        return len(self._features_list)

    def clear_buffer(self) -> None:
        self._features_list.clear()
        self._task_err_list.clear()
        self._reward_list.clear()
        self._dones_list.clear()

    def stacked_buffer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return train rollout tensors as (features, err, reward, dones)."""
        T = len(self._features_list)
        if T == 0:
            raise RuntimeError(f"IntrinsicRewardChannel '{self.name}' has an empty rollout buffer.")
        if len(self._task_err_list) < T or len(self._reward_list) < T or len(self._dones_list) < T:
            raise RuntimeError(
                f"IntrinsicRewardChannel '{self.name}' has inconsistent buffers: "
                f"features={len(self._features_list)}, err={len(self._task_err_list)}, "
                f"reward={len(self._reward_list)}, dones={len(self._dones_list)}."
            )
        feats = torch.stack([f.clone() for f in self._features_list[:T]], dim=0)
        task_err = torch.stack([e.clone() for e in self._task_err_list[:T]], dim=0)
        rewards = torch.stack([r.clone() for r in self._reward_list[:T]], dim=0)
        dones = torch.stack([d.clone() for d in self._dones_list[:T]], dim=0)
        return feats, task_err, rewards, dones

    def differentiable_reward_from_buffer(
        self,
        feats: torch.Tensor | None = None,
        task_err: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Recompute V21l linear reward on buffered rollout with gradients."""
        if feats is None or task_err is None:
            feats, task_err, _, _ = self.stacked_buffer()
        T = feats.shape[0]
        flat_feats = feats.view(-1, self.feature_dim)
        flat_err = task_err.view(-1).detach()

        f_phi_flat = F.softplus(self.net(flat_feats)).squeeze(-1).clamp(max=self.f_phi_max)
        raw_flat = 1.0 - flat_err * f_phi_flat
        r_phi_flat = torch.where(raw_flat >= 0, raw_flat, self.slope_neg * raw_flat)
        r_phi_flat = r_phi_flat.clamp(max=1.0)
        logs = {
            "r_phi_mean": r_phi_flat.mean(),
            "f_phi_mean": f_phi_flat.mean(),
            "f_phi_min": f_phi_flat.min(),
            "f_phi_max": f_phi_flat.max(),
        }
        return r_phi_flat.view(T, -1), logs

    def regularization_loss(
        self,
        feats: torch.Tensor | None = None,
        task_err: torch.Tensor | None = None,
        step_param_l2_coef: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return prior, monotonicity, and optional per-step trust-region losses."""
        if feats is None or task_err is None:
            feats, task_err, _, _ = self.stacked_buffer()
        flat_feats = feats.view(-1, self.feature_dim)
        flat_err = task_err.view(-1).detach()
        device = flat_feats.device

        prior_loss = torch.tensor(0.0, device=device)
        if self.prior_weight > 0.0 and self.prior_param_l2_coef > 0.0:
            for k, p in self.named_parameters():
                p0 = self._prior_state[k].to(p.device)
                prior_loss = prior_loss + (p - p0).pow(2).sum()
            prior_loss = self.prior_weight * self.prior_param_l2_coef * prior_loss

        mono_loss = torch.tensor(0.0, device=device)
        if self.mono_coef > 0.0:
            flat_v = flat_feats[:, : self.err_dim].detach().clone().requires_grad_(True)
            flat_vc = flat_feats[:, self.err_dim :].detach()
            f_phi_m = F.softplus(self.net(torch.cat([flat_v, flat_vc], dim=-1))).squeeze(-1).clamp(
                max=self.f_phi_max
            )
            grad_v = torch.autograd.grad(f_phi_m.sum(), flat_v, create_graph=True)[0]
            e_safe = flat_err.clamp(min=1e-6)
            e_hat = (flat_v.detach() - flat_vc) / e_safe.unsqueeze(-1)
            df_de = (grad_v * e_hat).sum(dim=-1)
            violation = F.relu(-(f_phi_m + flat_err * df_de))
            active = (flat_err > 1e-3).float()
            mono_loss = self.mono_coef * (violation.pow(2) * active).mean()

        step_loss = torch.tensor(0.0, device=device)
        if step_param_l2_coef > 0.0:
            for k, p in self.named_parameters():
                p_old = self._prev_meta_state[k].to(p.device)
                step_loss = step_loss + (p - p_old).pow(2).sum()
            step_loss = float(step_param_l2_coef) * step_loss

        total = prior_loss + mono_loss + step_loss
        logs = {
            "prior_loss": prior_loss.detach(),
            "mono_loss": mono_loss.detach(),
            "step_loss": step_loss.detach(),
        }
        return total, logs

    def snapshot_meta_state(self) -> None:
        self._prev_meta_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

    # --------------------------------------------------------------- meta-step

    def meta_update(self, gamma: float = 0.99, lam: float = 0.95) -> dict[str, float]:
        """One meta-gradient step on phi using the just-completed rollout."""
        T = len(self._features_list)
        if T < 2 or len(self._dones_list) < T:
            return {
                "meta_loss": 0.0, "adv_mean": 0.0, "adv_std": 0.0,
                "prior_loss": 0.0, "mono_loss": 0.0, "r_phi_mean": 0.0, "f_phi_mean": 0.0, "T": float(T),
            }

        with torch.inference_mode(False), torch.enable_grad():
            feats = torch.stack([f.clone() for f in self._features_list[:T]], dim=0)    # (T,N,F)
            task_err = torch.stack([e.clone() for e in self._task_err_list[:T]], dim=0) # (T,N)
            dones = torch.stack([d.clone() for d in self._dones_list[:T]], dim=0)       # (T,N)

            # MC discounted task return: adv_t = sum_k gamma^k*lam^k * (-err_{t+k})
            # Task signal: |v - v_cmd| + |v - v_cmd| / |v_cmd|  (absolute + relative error)
            v_cmd_part = feats[..., self.err_dim:]                          # (T, N, err_dim)
            if self.err_dim > 1:
                v_cmd_norm = v_cmd_part.norm(dim=-1)                        # (T, N)
            else:
                v_cmd_norm = v_cmd_part.abs().squeeze(-1)                   # (T, N)
            v_cmd_safe = v_cmd_norm.clamp(min=0.1)
            task_r = -(task_err + task_err / v_cmd_safe)
            adv = torch.zeros_like(task_r)
            running = torch.zeros(task_r.shape[1], device=task_r.device)
            for t in reversed(range(T)):
                running = task_r[t] + gamma * lam * running * (1.0 - dones[t])
                adv[t] = running

            adv_mean = adv.mean()
            adv_std = adv.std() + 1e-6
            adv_norm = (adv - adv_mean) / adv_std  # (T, N)

            # Re-evaluate r_phi WITH grad using the slope parameterisation.
            flat_feats = feats.view(-1, self.feature_dim)        # (T*N, F)
            flat_err = task_err.view(-1).detach()                # (T*N,)  task signal, no grad

            f_phi_flat = F.softplus(self.net(flat_feats)).squeeze(-1).clamp(max=self.f_phi_max)
            raw_flat = 1.0 - flat_err * f_phi_flat
            # Differentiable leaky: gradient flows through raw_flat -> f_phi -> net
            r_phi_flat = torch.where(raw_flat >= 0, raw_flat, self.slope_neg * raw_flat)
            r_phi_flat = r_phi_flat.clamp(max=1.0)
            r_phi = r_phi_flat.view(T, -1)

            meta_loss = -(adv_norm.detach() * r_phi).mean()

            # Trust-region: L2 toward initial V21k-baseline prior (in param space).
            prior_loss = torch.tensor(0.0, device=meta_loss.device)
            if self.prior_weight > 0.0 and self.prior_param_l2_coef > 0.0:
                for k, p in self.named_parameters():
                    p0 = self._prior_state[k].to(p.device)
                    prior_loss = prior_loss + (p - p0).pow(2).sum()
                prior_loss = self.prior_weight * self.prior_param_l2_coef * prior_loss

            # Monotonicity penalty: d(e*f_phi)/de = f_phi + e*(∇_v f_phi · ê) >= 0
            mono_loss = torch.tensor(0.0, device=meta_loss.device)
            if self.mono_coef > 0.0:
                flat_v = flat_feats[:, :self.err_dim].detach().clone().requires_grad_(True)
                flat_vc = flat_feats[:, self.err_dim:].detach()
                f_phi_m = F.softplus(self.net(torch.cat([flat_v, flat_vc], dim=-1))).squeeze(-1).clamp(max=self.f_phi_max)
                grad_v = torch.autograd.grad(f_phi_m.sum(), flat_v, create_graph=True)[0]  # (T*N, err_dim)
                e_safe = flat_err.clamp(min=1e-6)
                e_hat = (flat_v.detach() - flat_vc) / e_safe.unsqueeze(-1)  # unit dir
                df_de = (grad_v * e_hat).sum(dim=-1)  # d f_phi / d e, directional
                violation = F.relu(-(f_phi_m + flat_err * df_de))  # > 0 only when non-monotone
                active = (flat_err > 1e-3).float()
                mono_loss = self.mono_coef * (violation.pow(2) * active).mean()

            total_loss = meta_loss + prior_loss + mono_loss

            self.optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optim.step()

            result = {
                "meta_loss": meta_loss.item(),
                "adv_mean": adv_mean.item(),
                "adv_std": adv_std.item(),
                "prior_loss": prior_loss.item(),
                "mono_loss": mono_loss.item(),
                "r_phi_mean": r_phi.mean().item(),
                "f_phi_mean": f_phi_flat.mean().item(),
                "T": float(T),
            }

        self.clear_buffer()
        return result


class CommandConditionedIntrinsicRewardChannel(IntrinsicRewardChannel):
    """Learnable-slope reward whose slope depends only on command/mode features.

    Reward shape is still V21l's linear form:

        r = leaky(1 - e * f_phi(command, mode))

    Since f_phi is positive and independent of the actual velocity, monotonicity
    in the tracking error e is guaranteed by construction.
    """

    @torch.no_grad()
    def evaluate(
        self,
        v: torch.Tensor,
        v_cmd: torch.Tensor,
        slope_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for env reward term. No grad. Records (condition, err)."""
        if v.dim() == 1:
            v = v.unsqueeze(-1)
        if v_cmd.dim() == 1:
            v_cmd = v_cmd.unsqueeze(-1)
        if slope_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"CommandConditionedIntrinsicRewardChannel '{self.name}' expected "
                f"{self.feature_dim} feature dims, got {slope_features.shape[-1]}."
            )

        if self.err_dim > 1:
            err = (v - v_cmd).norm(dim=-1)
        else:
            err = (v - v_cmd).abs().squeeze(-1)

        f_phi = F.softplus(self.net(slope_features)).squeeze(-1).clamp(max=self.f_phi_max)
        raw = 1.0 - err * f_phi
        r = _leaky(raw, self.slope_neg)

        if _RECORDING_ENABLED:
            self._features_list.append(slope_features.detach().clone())
            self._task_err_list.append(err.detach().clone())
            self._reward_list.append(r.detach().clone())
        return r

    def meta_update(self, gamma: float = 0.99, lam: float = 0.95) -> dict[str, float]:  # noqa: ARG002
        raise RuntimeError(
            "CommandConditionedIntrinsicRewardChannel must be trained through "
            "validated bilevel updates, not current-rollout LIRPG correlation."
        )


def get_or_create_gauss(name: str, **kwargs) -> "GaussianIntrinsicRewardChannel":
    """Registry helper for GaussianIntrinsicRewardChannel (V21m sigma-kernel)."""
    if name not in _REGISTRY:
        _REGISTRY[name] = GaussianIntrinsicRewardChannel(name=name, **kwargs)
    return _REGISTRY[name]  # type: ignore[return-value]


class GaussianIntrinsicRewardChannel(nn.Module):
    """Learnable-sigma Gaussian reward channel for V21m.

    MLP: v_cmd -> softplus -> sigma > 0
    Reward: r = exp(-|v - v_cmd|^2 / sigma(v_cmd)^2)
    Monotone in |err| by construction (sigma > 0).
    Default init: sigma_0 = 0.5.
    """

    def __init__(
        self,
        name: str,
        err_dim: int,               # 2 for lin_xy, 1 for ang_z; also = sigma MLP input dim
        sigma_0: float = 0.5,       # initial sigma value
        hidden: tuple[int, ...] = (128, 64, 32),
        meta_lr: float = 2e-5,
        prior_weight: float = 1.0,
        prior_param_l2_coef: float = 1e-1,
        device: str = "cuda",
        init_fit_steps: int = 500,
        init_fit_lr: float = 1e-3,
        init_fit_samples: int = 4096,
        sample_lo: float = -1.5,
        sample_hi: float = 1.5,
        sigma_min: float = 0.05,
        sigma_max: float = 4.0,
    ):
        super().__init__()
        self.name = name
        self.err_dim = err_dim
        self.sigma_0 = float(sigma_0)
        self.prior_weight = float(prior_weight)
        self.prior_param_l2_coef = float(prior_param_l2_coef)
        self._sample_range = (sample_lo, sample_hi)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        with torch.inference_mode(False), torch.enable_grad():
            layers: list[nn.Module] = []
            prev = err_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ELU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
            self.to(device)

            self._init_to_prior(init_fit_steps, init_fit_lr, init_fit_samples)

            self.optim = torch.optim.Adam(self.parameters(), lr=meta_lr)
            self._prior_state: dict[str, torch.Tensor] = {
                k: v.detach().clone() for k, v in self.state_dict().items()
            }

        self._task_err_list: list[torch.Tensor] = []
        self._v_cmd_list: list[torch.Tensor] = []
        self._dones_list: list[torch.Tensor] = []

    def _init_to_prior(self, n_steps: int, lr: float, n_samples: int) -> None:
        """Fit net so that softplus(net(v_cmd)) ≈ sigma_0 for any v_cmd."""
        device = next(self.parameters()).device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        lo, hi = self._sample_range
        target = torch.tensor(self.sigma_0, device=device)

        for _ in range(n_steps):
            v_cmd = torch.empty(n_samples, self.err_dim, device=device).uniform_(lo, hi)
            pred = F.softplus(self.net(v_cmd)).squeeze(-1)
            loss = F.mse_loss(pred, target.expand_as(pred))
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            v_cmd = torch.empty(n_samples, self.err_dim, device=device).uniform_(lo, hi)
            pred = F.softplus(self.net(v_cmd)).squeeze(-1)
            mse = F.mse_loss(pred, target.expand_as(pred)).item()
        print(
            f"[GaussianIntrinsicReward:{self.name}] init sigma MSE vs default "
            f"(sigma_0={self.sigma_0}) = {mse:.6f}"
        )

    @torch.no_grad()
    def evaluate(
        self,
        v: torch.Tensor,
        v_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """r = exp(-||v - v_cmd||^2 / sigma(v_cmd)^2). Buffers err and v_cmd."""
        if v.dim() == 1:
            v = v.unsqueeze(-1)
        if v_cmd.dim() == 1:
            v_cmd = v_cmd.unsqueeze(-1)

        if self.err_dim > 1:
            err = (v - v_cmd).norm(dim=-1)
        else:
            err = (v - v_cmd).abs().squeeze(-1)

        sigma = F.softplus(self.net(v_cmd)).squeeze(-1).clamp(
            min=self.sigma_min, max=self.sigma_max
        )
        r = torch.exp(-(err ** 2) / (sigma ** 2))

        self._task_err_list.append(err.detach().clone())
        self._v_cmd_list.append(v_cmd.detach().clone())
        return r

    def record_dones(self, dones: torch.Tensor) -> None:
        self._dones_list.append(dones.detach().to(torch.float32).clone())

    def buffer_len(self) -> int:
        return len(self._task_err_list)

    def clear_buffer(self) -> None:
        self._task_err_list.clear()
        self._v_cmd_list.clear()
        self._dones_list.clear()

    def meta_update(self, gamma: float = 0.99, lam: float = 0.95) -> dict[str, float]:
        """One meta-gradient step on sigma-MLP using just-completed rollout."""
        T = len(self._task_err_list)
        if T < 2 or len(self._dones_list) < T:
            return {
                "meta_loss": 0.0, "adv_mean": 0.0, "adv_std": 0.0,
                "prior_loss": 0.0, "r_phi_mean": 0.0, "sigma_mean": 0.0, "T": float(T),
            }

        with torch.inference_mode(False), torch.enable_grad():
            task_err = torch.stack([e.clone() for e in self._task_err_list[:T]], dim=0)  # (T, N)
            v_cmd_stk = torch.stack([c.clone() for c in self._v_cmd_list[:T]], dim=0)    # (T, N, D)
            dones = torch.stack([d.clone() for d in self._dones_list[:T]], dim=0)        # (T, N)

            # Task signal: |err| + |err| / |v_cmd|  (absolute + relative)
            if self.err_dim > 1:
                v_cmd_norm = v_cmd_stk.norm(dim=-1)
            else:
                v_cmd_norm = v_cmd_stk.abs().squeeze(-1)
            v_cmd_safe = v_cmd_norm.clamp(min=1e-2)
            task_r = -(task_err + task_err / v_cmd_safe)

            adv = torch.zeros_like(task_r)
            running = torch.zeros(task_r.shape[1], device=task_r.device)
            for t in reversed(range(T)):
                running = task_r[t] + gamma * lam * running * (1.0 - dones[t])
                adv[t] = running

            adv_mean = adv.mean()
            adv_std = adv.std() + 1e-6
            adv_norm = (adv - adv_mean) / adv_std  # (T, N)

            flat_err = task_err.view(-1).detach()                    # (T*N,)
            flat_v_cmd = v_cmd_stk.view(-1, self.err_dim)            # (T*N, D)

            sigma_flat = F.softplus(self.net(flat_v_cmd)).squeeze(-1).clamp(
                min=self.sigma_min, max=self.sigma_max
            )
            r_flat = torch.exp(-(flat_err ** 2) / (sigma_flat ** 2))
            r_phi = r_flat.view(T, -1)

            meta_loss = -(adv_norm.detach() * r_phi).mean()

            prior_loss = torch.tensor(0.0, device=meta_loss.device)
            if self.prior_weight > 0.0 and self.prior_param_l2_coef > 0.0:
                for k, p in self.named_parameters():
                    p0 = self._prior_state[k].to(p.device)
                    prior_loss = prior_loss + (p - p0).pow(2).sum()
                prior_loss = self.prior_weight * self.prior_param_l2_coef * prior_loss

            total_loss = meta_loss + prior_loss

            self.optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optim.step()

            result = {
                "meta_loss": meta_loss.item(),
                "adv_mean": adv_mean.item(),
                "adv_std": adv_std.item(),
                "prior_loss": prior_loss.item(),
                "r_phi_mean": r_phi.mean().item(),
                "sigma_mean": sigma_flat.mean().item(),
                "T": float(T),
            }

        self.clear_buffer()
        return result
