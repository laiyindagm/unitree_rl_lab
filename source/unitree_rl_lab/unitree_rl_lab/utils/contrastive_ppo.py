"""Contrastive Latent Policy PPO — two-phase update with factored InfoNCE.

Phase A (representation): Re-encode observations, compute factored InfoNCE loss
over product spheres and FiLM-conditioned observation residual prediction loss.  Optimised by
a dedicated ``repr_optimizer``.

Phase B (policy): Use rollout-cached representations (z_cat, o_pred)
combined with flat_obs to build latent vectors for the MLP policy head.  Standard PPO
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
    temperature: float | torch.Tensor,
    max_samples: int | None = None,
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
    sample_cap = _NCE_MAX_SAMPLES if max_samples is None else max_samples
    # Subsample if batch too large to keep O(B^2) tractable
    if sample_cap > 0 and B > sample_cap:
        idx = torch.randperm(B, device=projections[0].device)[:sample_cap]
        projections = [p[idx] for p in projections]
        labels_per_sphere = [l[idx] for l in labels_per_sphere]
        B = sample_cap

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


def factored_cross_infonce(
    source_projections: list[torch.Tensor],
    target_projections: list[torch.Tensor],
    source_labels_per_sphere: list[torch.Tensor],
    target_labels_per_sphere: list[torch.Tensor],
    temperature: float | torch.Tensor,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Factored cross-view InfoNCE between source and target towers."""
    B = source_projections[0].shape[0]
    sample_cap = _NCE_MAX_SAMPLES if max_samples is None else max_samples
    if sample_cap > 0 and B > sample_cap:
        idx = torch.randperm(B, device=source_projections[0].device)[:sample_cap]
        source_projections = [p[idx] for p in source_projections]
        target_projections = [p[idx] for p in target_projections]
        source_labels_per_sphere = [l[idx] for l in source_labels_per_sphere]
        target_labels_per_sphere = [l[idx] for l in target_labels_per_sphere]

    total = torch.tensor(0.0, device=source_projections[0].device)
    for src, tgt, src_labels, tgt_labels in zip(
        source_projections,
        target_projections,
        source_labels_per_sphere,
        target_labels_per_sphere,
    ):
        sim = torch.mm(src, tgt.t()) / temperature
        mask_pos = src_labels.unsqueeze(1) == tgt_labels.unsqueeze(0)

        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max
        exp_logits = torch.exp(logits)
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        log_prob = logits - log_denom

        num_pos = mask_pos.float().sum(dim=1)
        has_pos = num_pos > 0
        if has_pos.any():
            mean_log_prob = (log_prob * mask_pos.float()).sum(dim=1) / (num_pos + 1e-8)
            total = total - mean_log_prob[has_pos].mean()

    return total / len(source_projections)


def time_infonce(
    p_anchors: list[torch.Tensor],
    p_positives: list[torch.Tensor],
    valid_mask: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """SimCLR-style NT-Xent over time-shifted positive pairs.

    Each anchor i has exactly one positive (the same env at t+k_shift).
    All other 2B-2 entries (other anchors and other positives) are negatives.
    Operates per sphere then averages.

    Args:
        p_anchors: list of [B, D] L2-normalised anchor projections, one per sphere.
        p_positives: list of [B, D] L2-normalised positive projections (time-shifted).
        valid_mask: [B] boolean mask — True where the (anchor, positive) pair is valid
            (no episode termination between t and t+k_shift).
        temperature: softmax temperature.

    Returns:
        Scalar mean loss; zero tensor if fewer than 4 valid pairs.
    """
    valid = valid_mask
    if valid.sum().item() < 4:
        return torch.tensor(0.0, device=p_anchors[0].device)

    # Subsample if very large to keep O(B^2) tractable
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() > _NCE_MAX_SAMPLES:
        perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:_NCE_MAX_SAMPLES]
        valid_idx = valid_idx[perm]

    total = torch.tensor(0.0, device=p_anchors[0].device)
    for pa, pp in zip(p_anchors, p_positives):
        a = pa[valid_idx]
        b = pp[valid_idx]
        B = a.shape[0]
        z = torch.cat([a, b], dim=0)                         # [2B, D]
        sim = (z @ z.t()) / temperature                      # [2B, 2B]
        sim.fill_diagonal_(-1e9)
        # positive index: i (in 0..B-1) ↔ i+B (in B..2B-1) and vice versa
        targets = torch.arange(2 * B, device=z.device)
        targets = (targets + B) % (2 * B)
        loss = F.cross_entropy(sim, targets)
        total = total + loss
    return total / len(p_anchors)


def sequence_prediction_loss(
    pred: torch.Tensor,
    future_gt: torch.Tensor,
    future_mask: torch.Tensor,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Time-decayed MSE for future sequence prediction.

    Args:
        pred: [B, K*D] predicted future values (obs deltas or actions).
        future_gt: [B, K*D] ground-truth future values.
        future_mask: [B, K] boolean mask (True = valid).
        gamma: temporal discount factor.
    """
    K = future_mask.shape[1]
    D = pred.shape[1] // K

    pred_steps = pred.view(-1, K, D)
    gt_steps = future_gt.view(-1, K, D)

    mse = (pred_steps - gt_steps).pow(2).mean(dim=-1)  # [B, K]

    weights = torch.tensor(
        [gamma ** k for k in range(K)], device=pred.device, dtype=pred.dtype
    )

    masked_mse = mse * future_mask.float() * weights.unsqueeze(0)
    valid_count = (future_mask.float() * weights.unsqueeze(0)).sum()

    if valid_count > 0:
        return masked_mse.sum() / valid_count
    return torch.tensor(0.0, device=pred.device)


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
        time_nce_coef: float = 0.0,        # B: SimCLR time-positive InfoNCE weight
        time_shift: int = 4,               # B: positive pair offset in env steps
        use_achieved_labels: bool = False, # X: label by achieved velocity (from critic obs) instead of cmd
        achieved_obs_key: str = "critic",  # X: which obs group holds base_lin_vel + base_ang_vel
        achieved_ang_vel_scale: float = 0.2,  # X: critic obs scale on base_ang_vel; we undo it
        gen_coef: float = 0.5,
        gen_coef_end: float = 0.1,
        gen_decay_iters: int = 10000,
        tau_init: float = 0.5,
        learnable_tau: bool = True,
        repr_lr: float = 1e-4,
        pred_gamma: float = 0.9,
        warmup_iters: int = 0,
        gate_open_iters: int = 1000,
        vx_levels: list[float] | None = None,
        vy_levels: list[float] | None = None,
        wz_levels: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)

        self.warmup_iters = warmup_iters
        self.gate_open_iters = gate_open_iters
        self.nce_coef = nce_coef
        self.time_nce_coef = time_nce_coef
        self.time_shift = int(time_shift)
        self.use_achieved_labels = bool(use_achieved_labels)
        self.achieved_obs_key = str(achieved_obs_key)
        self.achieved_ang_vel_scale = float(achieved_ang_vel_scale)
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
        # default_vx = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,1.1, 1.2, 1.3, 1.4, 1.5]
        # default_vy = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # default_wz = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
        self._policy_params = policy_params  # for separate gradient clipping
        self.optimizer = type(self.optimizer)(
            policy_params + list(critic.parameters()), lr=self.learning_rate
        )

        # Monkey-patch storage with cache buffers
        T = storage.num_transitions_per_env
        N = storage.num_envs
        num_spheres = actor.num_spheres
        sphere_dim = actor.sphere_dim
        pred_horizon = actor.pred_horizon
        storage.cached_z_cat = torch.zeros(T, N, num_spheres * sphere_dim, device=self.device)
        storage.cached_o_pred = torch.zeros(T, N, actor._pred_out_dim, device=self.device)

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
        # Update gate schedule before each rollout step
        self.actor.gate_value = min(self.counter / self.gate_open_iters, 1.0)
        actions = super().act(obs)
        z_cat, o_pred = self.actor.get_cached_repr()
        step = self.storage.step  # act() runs BEFORE add_transition
        self.storage.cached_z_cat[step] = z_cat
        self.storage.cached_o_pred[step] = o_pred
        return actions

    # ------------------------------------------------------------------
    # Build future action ground truth from storage
    # ------------------------------------------------------------------

    def _build_future_obs_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build future observation-delta targets and validity mask.

        Computes Δo[t, k] = o_current[t+k+1] - o_current[t] where o_current is
        the last frame's observation without cmd (history_obs_dim=51).

        Returns:
            future_obs: [T, N, K*history_obs_dim]
            future_mask: [T, N, K] boolean
        """
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs
        K = self.actor.pred_horizon
        pred_dim = self.actor.pred_obs_dim  # 36 (ang_vel + gravity + joint_pos + joint_vel)
        cs = self.actor.cmd_start_idx
        cd = self.actor.cmd_dim

        # Extract flat obs for all T steps: [T, N, history_len * single_obs_dim]
        all_flat_obs = torch.cat(
            [self.storage.observations[k] for k in self.actor.obs_groups], dim=-1
        )  # [T, N, history_len*54]

        # Extract o_current (last frame, no cmd) for all steps
        frames = all_flat_obs.view(T, N, self.actor.history_len, self.actor.single_obs_dim)
        last_frame = frames[:, :, -1, :]  # [T, N, 54]
        # history_no_cmd layout: [ang_vel(3), gravity(3), joint_pos(15), joint_vel(15), last_action(15)] = 51
        # Take only the first pred_dim dims (physical state, excludes last_action)
        o_pred_target = torch.cat(
            [last_frame[:, :, :cs], last_frame[:, :, cs + cd:]], dim=-1
        )[:, :, :pred_dim]  # [T, N, pred_dim]

        future_obs = torch.zeros(T, N, K * pred_dim, device=self.device)
        future_mask = torch.zeros(T, N, K, dtype=torch.bool, device=self.device)

        dones = self.storage.dones.squeeze(-1)  # [T, N]
        not_done = ~dones.bool()

        for k in range(K):
            t_end = T - k - 1
            if t_end <= 0:
                break
            # Δo: observation at t+k+1 minus observation at t (physical state only)
            delta = o_pred_target[k + 1 : k + 1 + t_end] - o_pred_target[:t_end]
            future_obs[:t_end, :, k * pred_dim : (k + 1) * pred_dim] = delta
            # Mask: no done in [t, t+k+1)
            valid = torch.ones(t_end, N, dtype=torch.bool, device=self.device)
            for j in range(k + 1):
                valid = valid & not_done[j : j + t_end]
            future_mask[:t_end, :, k] = valid

        return future_obs, future_mask

    # ------------------------------------------------------------------
    # B: Time-shifted positive pair obs
    # ------------------------------------------------------------------

    def _build_time_pair_obs(self, k_shift: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build time-shifted positive observation tensor and validity mask.

        For each (t, n) the positive is the obs at (t+k_shift, n), valid only if
        no episode termination occurred in [t, t+k_shift).

        Returns:
            pair_obs: [T*N, total_obs_dim] flattened
            pair_mask: [T*N] boolean
        """
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs
        all_flat = torch.cat(
            [self.storage.observations[k] for k in self.actor.obs_groups], dim=-1
        )  # [T, N, D]
        D = all_flat.shape[-1]
        pair_obs = torch.zeros_like(all_flat)
        pair_mask = torch.zeros(T, N, dtype=torch.bool, device=self.device)

        if k_shift > 0 and T - k_shift > 0:
            pair_obs[: T - k_shift] = all_flat[k_shift:]
            dones = self.storage.dones.squeeze(-1).bool()        # [T, N]
            not_done = ~dones
            valid = torch.ones(T - k_shift, N, dtype=torch.bool, device=self.device)
            for j in range(k_shift):
                valid = valid & not_done[j : j + (T - k_shift)]
            pair_mask[: T - k_shift] = valid

        return pair_obs.flatten(0, 1), pair_mask.flatten(0, 1)

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

        # Build future observation delta ground truth
        future_obs_deltas, future_mask = self._build_future_obs_deltas()

        # B: build time-shifted positive obs (no-op if time_nce_coef==0)
        if self.time_nce_coef > 0.0:
            flat_pair_obs, flat_pair_mask = self._build_time_pair_obs(self.time_shift)
        else:
            flat_pair_obs = None
            flat_pair_mask = None

        # C: label distribution probe — one-shot per update on full storage cmd
        with torch.no_grad():
            all_obs_flat_full = torch.cat(
                [self.storage.observations[k] for k in self.actor.obs_groups], dim=-1
            )  # [T, N, D]
            frames_full = all_obs_flat_full.view(
                T, N, self.actor.history_len, self.actor.single_obs_dim
            )
            cs_full = self.actor.cmd_start_idx
            cd_full = self.actor.cmd_dim
            cmd_full = frames_full[:, :, -1, cs_full : cs_full + cd_full].reshape(-1, cd_full)
            lbl_full_x = quantize_to_levels(cmd_full[:, 0], self.vx_levels)
            lbl_full_y = quantize_to_levels(cmd_full[:, 1], self.vy_levels)
            lbl_full_w = quantize_to_levels(cmd_full[:, 2], self.wz_levels)

            def _lbl_stats(lbl: torch.Tensor, n_bins: int) -> tuple[float, float]:
                cnt = torch.bincount(lbl, minlength=n_bins).float()
                tot = cnt.sum().clamp(min=1.0)
                uniq = (cnt > 0).sum().item()
                top1 = (cnt.max() / tot).item()
                return float(uniq), float(top1)

            lbl_uniq_x, lbl_top1_x = _lbl_stats(lbl_full_x, len(self.vx_levels))
            lbl_uniq_y, lbl_top1_y = _lbl_stats(lbl_full_y, len(self.vy_levels))
            lbl_uniq_w, lbl_top1_w = _lbl_stats(lbl_full_w, len(self.wz_levels))

        # X: build per-step achieved velocity tensor (label source) from critic obs.
        # achieved = [base_lin_vel_x, base_lin_vel_y, base_ang_vel_z * 1/scale]
        # Critic single-frame layout (V19d, 57 dims): base_lin_vel(3), base_ang_vel(3),
        # grav(3), cmd(3), jpos(15), jvel(15), last_action(15).
        # base_ang_vel is z-scored only by EmpiricalNormalization later; here it's raw
        # times the static cfg scale (0.2). We undo that scale to recover physical rad/s.
        if self.use_achieved_labels:
            critic_flat = self.storage.observations[self.achieved_obs_key]  # [T, N, hl*57]
            T_c, N_c = critic_flat.shape[0], critic_flat.shape[1]
            crit_total = critic_flat.shape[-1]
            # Infer single-frame dim: history_len shared with policy
            crit_single = crit_total // self.actor.history_len
            critic_frames = critic_flat.view(T_c, N_c, self.actor.history_len, crit_single)
            last_critic = critic_frames[:, :, -1, :]  # [T, N, crit_single]
            inv_ang_scale = 1.0 / max(self.achieved_ang_vel_scale, 1e-6)
            achieved_full = torch.stack(
                [
                    last_critic[:, :, 0],                   # base_lin_vel_x  (raw m/s)
                    last_critic[:, :, 1],                   # base_lin_vel_y
                    last_critic[:, :, 5] * inv_ang_scale,   # base_ang_vel_z (un-scaled)
                ],
                dim=-1,
            )  # [T, N, 3]
            flat_achieved = achieved_full.flatten(0, 1)  # [T*N, 3]

            # Probe achieved-label distribution too
            with torch.no_grad():
                ach_lbl_x = quantize_to_levels(flat_achieved[:, 0], self.vx_levels)
                ach_lbl_y = quantize_to_levels(flat_achieved[:, 1], self.vy_levels)
                ach_lbl_w = quantize_to_levels(flat_achieved[:, 2], self.wz_levels)
                ach_uniq_x, ach_top1_x = _lbl_stats(ach_lbl_x, len(self.vx_levels))
                ach_uniq_y, ach_top1_y = _lbl_stats(ach_lbl_y, len(self.vy_levels))
                ach_uniq_w, ach_top1_w = _lbl_stats(ach_lbl_w, len(self.wz_levels))
        else:
            flat_achieved = None
            ach_uniq_x = ach_top1_x = ach_uniq_y = ach_top1_y = ach_uniq_w = ach_top1_w = 0.0

        # Flatten cached buffers: [T, N, ...] → [T*N, ...]
        flat_cached_z = self.storage.cached_z_cat.flatten(0, 1)
        flat_cached_o_pred = self.storage.cached_o_pred.flatten(0, 1)
        flat_future_obs = future_obs_deltas.flatten(0, 1)
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
        mean_time_nce_loss = 0.0
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
                batch_o_pred = flat_cached_o_pred[batch_idx]
                batch_future = flat_future_obs[batch_idx]
                batch_future_mask = flat_future_mask[batch_idx]

                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                            batch_advantages.std() + 1e-8
                        )

                # ====== Phase A: Representation Learning ======
                flat_obs_a = self._flatten_obs(batch_obs)
                z_spheres, cmd = self.actor.encode(flat_obs_a)
                p_spheres = self.actor.project_contrastive(z_spheres)

                # X: prefer achieved velocity over cmd for SupCon labels when enabled.
                # Achieved is what the encoder *can actually see* in history; cmd may be
                # un-realisable when curriculum is stuck (e.g. cmd_x=1.5 but robot
                # only walks at 0.13 m/s) → SupCon target becomes unsatisfiable.
                if flat_achieved is not None:
                    label_src = flat_achieved[batch_idx]
                else:
                    label_src = cmd
                labels_per_sphere = [
                    quantize_to_levels(label_src[:, 0], self.vx_levels),
                    quantize_to_levels(label_src[:, 1], self.vy_levels),
                    quantize_to_levels(label_src[:, 2], self.wz_levels),
                ]

                tau_val = self.tau.clamp(min=0.01, max=2.0)
                nce_loss = factored_infonce(p_spheres, labels_per_sphere, tau_val)

                # B: time-shifted positive InfoNCE (SimCLR style)
                if self.time_nce_coef > 0.0 and flat_pair_obs is not None:
                    batch_pair_obs = flat_pair_obs[batch_idx]
                    batch_pair_mask = flat_pair_mask[batch_idx]
                    if batch_pair_mask.any():
                        z_pos_spheres, _ = self.actor.encode(batch_pair_obs)
                        p_pos_spheres = self.actor.project_contrastive(z_pos_spheres)
                        time_nce_loss = time_infonce(
                            p_spheres, p_pos_spheres, batch_pair_mask, tau_val
                        )
                    else:
                        time_nce_loss = torch.tensor(0.0, device=self.device)
                else:
                    time_nce_loss = torch.tensor(0.0, device=self.device)

                z_cat_a = torch.cat(z_spheres, dim=-1)
                o_pred_a = self.actor.generator(z_cat_a, cmd)
                gen_loss = sequence_prediction_loss(
                    o_pred_a, batch_future, batch_future_mask, self.pred_gamma
                )

                repr_loss = (
                    self.nce_coef * nce_loss
                    + self.time_nce_coef * time_nce_loss
                    + gen_coef * gen_loss
                )

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
                    h_nc, cmd_b, _ = self.actor._split_obs(flat_obs_b)
                    h_enc_b = self.actor.encoder(h_nc)
                    z_sph_b = self.actor.sphere_proj(h_enc_b)
                    z_cat_b = torch.cat(z_sph_b, dim=-1)
                    o_pred_b = self.actor.generator(z_cat_b, cmd_b)
                    gate = min(self.counter / self.gate_open_iters, 1.0)
                    latent = torch.cat([flat_obs_b, gate * z_cat_b, gate * o_pred_b], dim=-1)
                else:
                    batch_flat_obs = self._flatten_obs(batch_obs)
                    batch_flat_obs = self.actor.obs_normalizer(batch_flat_obs)
                    latent = ContrastiveLatentModel.get_latent_from_cache(
                        batch_flat_obs, batch_z_cat, batch_o_pred
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
                # Clip actor and critic separately (matches baseline PPO)
                nn.utils.clip_grad_norm_(self._policy_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.mean().item()
                mean_nce_loss += nce_loss.item()
                mean_time_nce_loss += time_nce_loss.item()
                mean_gen_loss += gen_loss.item()
                mean_repr_loss += repr_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_nce_loss /= num_updates
        mean_time_nce_loss /= num_updates
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
            "time_nce_loss": mean_time_nce_loss,
            "gen_loss": mean_gen_loss,
            "repr_loss": mean_repr_loss,
            "gen_coef": gen_coef,
            "tau": self.tau.item(),
            "warmup": float(self.counter < self.warmup_iters),
            "gate": self.actor.gate_value,
            # C: label distribution probe (per-update, full storage)
            "lbl_uniq_x": lbl_uniq_x,
            "lbl_uniq_y": lbl_uniq_y,
            "lbl_uniq_w": lbl_uniq_w,
            "lbl_top1_x": lbl_top1_x,
            "lbl_top1_y": lbl_top1_y,
            "lbl_top1_w": lbl_top1_w,
            # X: achieved-velocity label distribution (only nonzero when use_achieved_labels=True)
            "ach_uniq_x": ach_uniq_x,
            "ach_uniq_y": ach_uniq_y,
            "ach_uniq_w": ach_uniq_w,
            "ach_top1_x": ach_top1_x,
            "ach_top1_y": ach_top1_y,
            "ach_top1_w": ach_top1_w,
        }
        sphere_names = ["x", "y", "w"]
        for s_idx in range(min(self.actor.num_spheres, len(sphere_names))):
            loss_dict[f"uniformity_{sphere_names[s_idx]}"] = uniformity_sums[s_idx]
            loss_dict[f"alignment_{sphere_names[s_idx]}"] = alignment_sums[s_idx]

        return loss_dict

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self) -> dict:
        saved = super().save()
        saved["repr_optimizer_state_dict"] = self.repr_optimizer.state_dict()
        saved["counter"] = self.counter
        saved["log_tau"] = self.log_tau.data if isinstance(self.log_tau, nn.Parameter) else self.log_tau
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if "repr_optimizer_state_dict" in loaded_dict:
            self.repr_optimizer.load_state_dict(loaded_dict["repr_optimizer_state_dict"])
        if "counter" in loaded_dict:
            self.counter = loaded_dict["counter"]
        if "log_tau" in loaded_dict:
            if isinstance(self.log_tau, nn.Parameter):
                self.log_tau.data.copy_(loaded_dict["log_tau"])
            else:
                self.log_tau = loaded_dict["log_tau"]
        return result

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
