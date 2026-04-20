"""Contrastive Latent Policy PPO — two-phase update with factored InfoNCE.

Phase A (representation): Re-encode observations, compute factored InfoNCE loss
over product spheres and FiLM-conditioned sequence prediction loss.  Optimised by
a dedicated ``repr_optimizer``.

Phase B (policy): Use rollout-cached representations (z_cat, cmd, a_pred,
o_current) to build latent vectors for the MLP policy head.  Standard PPO
surrogate + value + entropy, optimised by the inherited ``self.optimizer``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from unitree_rl_lab.utils.contrastive_latent_model import ContrastiveLatentModel
from unitree_rl_lab.utils.rsl_rl_custom_ppo import _sanitize_model_cfg


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

_NCE_MAX_SAMPLES = 2048 * 2  # cap to avoid O(B^2) blowup


def factored_infonce(
    projections: list[torch.Tensor],
    labels_per_sphere: list[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """Factored SupCon-style InfoNCE over independent spheres.

    Args:
        projections: list of [B, sphere_dim] L2-normalised projections, one per sphere.
        labels_per_sphere: list of [B] integer labels (quantised command axis), one per sphere.
        temperature: softmax temperature.

    Returns:
        Scalar mean loss across spheres.
    """
    B = projections[0].shape[0]
    # Subsample if batch too large to keep O(B^2) tractable
    if B > _NCE_MAX_SAMPLES:
        idx = torch.randperm(B, device=projections[0].device)[:_NCE_MAX_SAMPLES]
        projections = [p[idx] for p in projections]
        labels_per_sphere = [l[idx] for l in labels_per_sphere]
        B = _NCE_MAX_SAMPLES

    total = torch.tensor(0.0, device=projections[0].device)
    for p, labels in zip(projections, labels_per_sphere):
        sim = torch.mm(p, p.t()) / temperature  # [B, B]
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        mask_self = torch.eye(B, dtype=torch.bool, device=p.device)
        mask_pos = mask_pos & ~mask_self

        # Numerical stability
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max
        exp_logits = torch.exp(logits) * (~mask_self).float()
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        log_prob = logits - log_denom

        num_pos = mask_pos.float().sum(dim=1)
        has_pos = num_pos > 0
        if has_pos.any():
            mean_log_prob = (log_prob * mask_pos.float()).sum(dim=1) / (num_pos + 1e-8)
            total = total - mean_log_prob[has_pos].mean()

    return total / len(projections)


def sequence_prediction_loss(
    a_pred: torch.Tensor,
    future_gt: torch.Tensor,
    future_mask: torch.Tensor,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Time-decayed MSE for future action prediction.

    Args:
        a_pred: [B, K*num_actions] predicted future actions.
        future_gt: [B, K*num_actions] ground-truth future actions.
        future_mask: [B, K] boolean mask (True = valid).
        gamma: temporal discount factor.
    """
    K = future_mask.shape[1]
    num_actions = a_pred.shape[1] // K

    pred_steps = a_pred.view(-1, K, num_actions)
    gt_steps = future_gt.view(-1, K, num_actions)

    mse = (pred_steps - gt_steps).pow(2).mean(dim=-1)  # [B, K]

    weights = torch.tensor(
        [gamma ** k for k in range(K)], device=a_pred.device, dtype=a_pred.dtype
    )

    masked_mse = mse * future_mask.float() * weights.unsqueeze(0)
    valid_count = (future_mask.float() * weights.unsqueeze(0)).sum()

    if valid_count > 0:
        return masked_mse.sum() / valid_count
    return torch.tensor(0.0, device=a_pred.device)


_METRIC_SUBSAMPLE = 512  # cap to avoid O(B^2) blowup on large batches


def compute_uniformity(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Uniformity metric: log E[exp(-t * ||z_i - z_j||^2)].  Subsampled."""
    if z.shape[0] > _METRIC_SUBSAMPLE:
        idx = torch.randperm(z.shape[0], device=z.device)[:_METRIC_SUBSAMPLE]
        z = z[idx]
    sq_dists = torch.cdist(z, z, p=2).pow(2)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean() + 1e-8)


def compute_alignment(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Alignment metric: mean intra-class distance.  Subsampled."""
    if z.shape[0] > _METRIC_SUBSAMPLE:
        idx = torch.randperm(z.shape[0], device=z.device)[:_METRIC_SUBSAMPLE]
        z = z[idx]
        labels = labels[idx]
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_self = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    mask_pos = mask_pos & ~mask_self
    if mask_pos.any():
        dists = torch.cdist(z, z, p=2)
        return dists[mask_pos].mean()
    return torch.tensor(0.0, device=z.device)


def quantize_to_levels(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Quantize continuous values to nearest discrete level indices."""
    dists = (values.unsqueeze(1) - levels.unsqueeze(0)).abs()
    return dists.argmin(dim=1)


# ---------------------------------------------------------------------------
# ContrastivePPO algorithm
# ---------------------------------------------------------------------------

class ContrastivePPO(PPO):
    """Two-phase PPO with factored contrastive representation learning."""

    def __init__(
        self,
        actor: ContrastiveLatentModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        nce_coef: float = 0.1,
        gen_coef: float = 0.5,
        gen_coef_end: float = 0.1,
        gen_decay_iters: int = 10000,
        tau_init: float = 0.5,
        learnable_tau: bool = True,
        repr_lr: float = 1e-4,
        pred_gamma: float = 0.9,
        warmup_iters: int = 0,
        vx_levels: list[float] | None = None,
        vy_levels: list[float] | None = None,
        wz_levels: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)

        self.warmup_iters = warmup_iters
        self.nce_coef = nce_coef
        self.gen_coef_start = gen_coef
        self.gen_coef_end = gen_coef_end
        self.gen_decay_iters = gen_decay_iters
        self.pred_gamma = pred_gamma
        self.counter = 0

        # Learnable temperature
        if learnable_tau:
            self.log_tau = nn.Parameter(
                torch.tensor(math.log(tau_init), device=self.device)
            )
        else:
            self.log_tau = torch.tensor(math.log(tau_init), device=self.device)

        # Quantization levels for InfoNCE labels
        default_vx = [-0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
        default_vy = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        default_wz = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
        self.vx_levels = torch.tensor(vx_levels or default_vx, dtype=torch.float32, device=self.device)
        self.vy_levels = torch.tensor(vy_levels or default_vy, dtype=torch.float32, device=self.device)
        self.wz_levels = torch.tensor(wz_levels or default_wz, dtype=torch.float32, device=self.device)

        # Representation optimizer — separate from policy optimizer
        repr_params = list(actor.encoder.parameters())
        repr_params += list(actor.sphere_proj.parameters())
        repr_params += list(actor.contrast_projs.parameters())
        repr_params += list(actor.generator.parameters())
        if learnable_tau:
            repr_params.append(self.log_tau)
        self.repr_optimizer = torch.optim.Adam(repr_params, lr=repr_lr)

        # Override inherited PPO optimizer: exclude repr params so encoder
        # is only updated by repr_optimizer (prevents conflicting Adam states).
        repr_ids = {id(p) for p in repr_params}
        policy_params = [p for p in actor.parameters() if id(p) not in repr_ids]
        self.optimizer = type(self.optimizer)(
            policy_params + list(critic.parameters()), lr=self.learning_rate
        )

        # Monkey-patch storage with cache buffers
        T = storage.num_transitions_per_env
        N = storage.num_envs
        num_spheres = actor.num_spheres
        sphere_dim = actor.sphere_dim
        pred_horizon = actor.pred_horizon
        num_actions = actor._num_actions
        storage.cached_z_cat = torch.zeros(T, N, num_spheres * sphere_dim, device=self.device)
        storage.cached_cmd = torch.zeros(T, N, actor.cmd_dim, device=self.device)
        storage.cached_a_pred = torch.zeros(T, N, pred_horizon * num_actions, device=self.device)
        storage.cached_o_current = torch.zeros(T, N, actor.history_obs_dim, device=self.device)

    # ------------------------------------------------------------------
    # Coefficient schedules
    # ------------------------------------------------------------------

    def _get_gen_coef(self) -> float:
        if self.gen_decay_iters <= 0:
            return self.gen_coef_end
        progress = min(self.counter / self.gen_decay_iters, 1.0)
        return self.gen_coef_start + progress * (self.gen_coef_end - self.gen_coef_start)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    # ------------------------------------------------------------------
    # Override act() to cache representations
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        z_cat, cmd_cached, a_pred, o_current = self.actor.get_cached_repr()
        step = self.storage.step - 1  # add_transition already incremented step
        self.storage.cached_z_cat[step] = z_cat
        self.storage.cached_cmd[step] = cmd_cached
        self.storage.cached_a_pred[step] = a_pred
        self.storage.cached_o_current[step] = o_current
        return actions

    # ------------------------------------------------------------------
    # Build future action ground truth from storage
    # ------------------------------------------------------------------

    def _build_future_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build future action targets and validity mask from rollout actions.

        Returns:
            future_actions: [T, N, K*num_actions]
            future_mask: [T, N, K] boolean
        """
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs
        K = self.actor.pred_horizon
        num_actions = self.actor._num_actions

        future_actions = torch.zeros(T, N, K * num_actions, device=self.device)
        future_mask = torch.zeros(T, N, K, dtype=torch.bool, device=self.device)

        actions = self.storage.actions  # [T, N, num_actions]
        dones = self.storage.dones.squeeze(-1)  # [T, N] (storage keeps [T, N, 1])
        not_done = ~dones.bool()  # [T, N]

        for k in range(K):
            # future_t = t + k + 1, so valid range: t in [0, T-k-2]
            t_end = T - k - 1
            if t_end <= 0:
                break
            # Copy actions: for step t, future action at offset k is actions[t+k]
            future_actions[:t_end, :, k * num_actions : (k + 1) * num_actions] = actions[k:k + t_end]
            # Mask: no done in [t, t+k+1) → all not_done[t], not_done[t+1], ..., not_done[t+k]
            valid = torch.ones(t_end, N, dtype=torch.bool, device=self.device)
            for j in range(k + 1):
                valid = valid & not_done[j:j + t_end]
            future_mask[:t_end, :, k] = valid

        return future_actions, future_mask

    # ------------------------------------------------------------------
    # Flatten obs helper
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs: TensorDict) -> torch.Tensor:
        parts = [obs[k] for k in self.actor.obs_groups]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Two-phase update: representation learning then cached-PPO."""

        gen_coef = self._get_gen_coef()
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs

        # Build future action ground truth
        future_actions, future_mask = self._build_future_actions()

        # Flatten cached buffers: [T, N, ...] → [T*N, ...]
        flat_cached_z = self.storage.cached_z_cat.flatten(0, 1)
        flat_cached_cmd = self.storage.cached_cmd.flatten(0, 1)
        flat_cached_apred = self.storage.cached_a_pred.flatten(0, 1)
        flat_cached_ocur = self.storage.cached_o_current.flatten(0, 1)
        flat_future_actions = future_actions.flatten(0, 1)
        flat_future_mask = future_mask.flatten(0, 1)

        # Flatten standard storage for PPO
        observations = self.storage.observations.flatten(0, 1)
        actions_stored = self.storage.actions.flatten(0, 1)
        values = self.storage.values.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)
        old_actions_log_prob = self.storage.actions_log_prob.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        old_distribution_params = tuple(
            p.flatten(0, 1) for p in self.storage.distribution_params
        )

        batch_size = T * N
        mini_batch_size = batch_size // self.num_mini_batches

        # Accumulators
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_nce_loss = 0.0
        mean_gen_loss = 0.0
        mean_repr_loss = 0.0
        uniformity_sums = [0.0] * self.actor.num_spheres
        alignment_sums = [0.0] * self.actor.num_spheres

        indices = torch.randperm(
            self.num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        for epoch in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                batch_obs = observations[batch_idx]
                batch_actions = actions_stored[batch_idx]
                batch_values = values[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_prob = old_actions_log_prob[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_dist_params = tuple(p[batch_idx] for p in old_distribution_params)

                batch_z_cat = flat_cached_z[batch_idx]
                batch_cmd = flat_cached_cmd[batch_idx]
                batch_a_pred = flat_cached_apred[batch_idx]
                batch_o_cur = flat_cached_ocur[batch_idx]
                batch_future = flat_future_actions[batch_idx]
                batch_future_mask = flat_future_mask[batch_idx]

                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                            batch_advantages.std() + 1e-8
                        )

                # ====== Phase A: Representation Learning ======
                flat_obs_a = self._flatten_obs(batch_obs)
                z_spheres, cmd, _ = self.actor.encode(flat_obs_a)
                p_spheres = self.actor.project_contrastive(z_spheres)

                labels_per_sphere = [
                    quantize_to_levels(cmd[:, 0], self.vx_levels),
                    quantize_to_levels(cmd[:, 1], self.vy_levels),
                    quantize_to_levels(cmd[:, 2], self.wz_levels),
                ]

                tau_val = self.tau.clamp(min=0.01, max=2.0)
                nce_loss = factored_infonce(p_spheres, labels_per_sphere, tau_val.item())

                z_cat_a = torch.cat(z_spheres, dim=-1)
                a_pred_a = self.actor.generator(z_cat_a, cmd)
                gen_loss = sequence_prediction_loss(
                    a_pred_a, batch_future, batch_future_mask, self.pred_gamma
                )

                repr_loss = self.nce_coef * nce_loss + gen_coef * gen_loss

                self.repr_optimizer.zero_grad()
                repr_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.repr_optimizer.param_groups for p in group["params"]],
                    self.max_grad_norm,
                )
                self.repr_optimizer.step()

                with torch.no_grad():
                    for s_idx, (z_s, lbl) in enumerate(zip(z_spheres, labels_per_sphere)):
                        uniformity_sums[s_idx] += compute_uniformity(z_s.detach()).item()
                        alignment_sums[s_idx] += compute_alignment(z_s.detach(), lbl).item()

                # ====== Phase B: PPO (warmup=live / post-warmup=cached) ======
                if self.counter < self.warmup_iters:
                    # Warmup: re-encode with gradients so PPO can train encoder
                    flat_obs_b = self._flatten_obs(batch_obs)
                    flat_obs_b = self.actor.obs_normalizer(flat_obs_b)
                    h_nc, cmd_b, o_cur_b = self.actor._split_obs(flat_obs_b)
                    h_enc_b = self.actor.encoder(h_nc)
                    z_sph_b = self.actor.sphere_proj(h_enc_b)
                    z_cat_b = torch.cat(z_sph_b, dim=-1)
                    a_pred_b = self.actor.generator(z_cat_b, cmd_b)
                    latent = torch.cat([o_cur_b, cmd_b, z_cat_b, a_pred_b], dim=-1)
                else:
                    latent = ContrastiveLatentModel.get_latent_from_cache(
                        batch_o_cur, batch_cmd, batch_z_cat, batch_a_pred
                    )

                self.actor.evaluate_from_latent(latent, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(batch_actions)
                critic_values = self.critic(batch_obs)

                distribution_params = self.actor.output_distribution_params
                entropy = self.actor.output_entropy

                # Adaptive LR
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(batch_old_dist_params, distribution_params)
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob - torch.squeeze(batch_old_log_prob))
                surrogate = -torch.squeeze(batch_advantages) * ratio
                surrogate_clipped = -torch.squeeze(batch_advantages) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value loss
                if self.use_clipped_value_loss:
                    value_clipped = batch_values + (critic_values - batch_values).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (critic_values - batch_returns).pow(2)
                    value_losses_clipped = (value_clipped - batch_returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch_returns - critic_values).pow(2).mean()

                ppo_loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.mean().item()
                mean_nce_loss += nce_loss.item()
                mean_gen_loss += gen_loss.item()
                mean_repr_loss += repr_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_nce_loss /= num_updates
        mean_gen_loss /= num_updates
        mean_repr_loss /= num_updates
        for s_idx in range(self.actor.num_spheres):
            uniformity_sums[s_idx] /= num_updates
            alignment_sums[s_idx] /= num_updates

        self.storage.clear()
        self.counter += 1

        loss_dict: dict[str, float] = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "nce_loss": mean_nce_loss,
            "gen_loss": mean_gen_loss,
            "repr_loss": mean_repr_loss,
            "gen_coef": gen_coef,
            "tau": self.tau.item(),
            "warmup": float(self.counter < self.warmup_iters),
        }
        sphere_names = ["x", "y", "w"]
        for s_idx in range(min(self.actor.num_spheres, len(sphere_names))):
            loss_dict[f"uniformity_{sphere_names[s_idx]}"] = uniformity_sums[s_idx]
            loss_dict[f"alignment_{sphere_names[s_idx]}"] = alignment_sums[s_idx]

        return loss_dict

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct ContrastivePPO (CLP actor + MLP critic)."""
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_cfg = _sanitize_model_cfg(cfg["actor"], actor_class)
        critic_cfg = _sanitize_model_cfg(cfg["critic"], critic_class)

        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
        ).to(device)
        print(f"Actor Model: {actor}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns

        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **critic_cfg
        ).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        alg: PPO = alg_class(
            actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        return alg
