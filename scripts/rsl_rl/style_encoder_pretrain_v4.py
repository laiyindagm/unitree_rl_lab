# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Train the frnc_style_v4 gait-style encoder.

V4 treats z as a continuous, phase-invariant gait-style coordinate:

  * z -> Y0 for phase-invariant style targets.
  * (z, phi) -> Yphi for phase-dependent frame targets.
  * phase-shifted windows from the same parent segment are invariance pairs.
  * Rank-N-Contrast is defined on gait-style target distance, not command
    distance.  A residual variant is restricted to similar-command pairs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from frnc_segment_pretrain import vicreg_cov, vicreg_var
from style_obs_layout import build_mask_indices, describe_layout


PRESETS = {
    "E1_reg_only": dict(l_y0=1.0, l_yphi=1.0, l_inv=0.0, l_rnc_g=0.0, l_rnc_res=0.0, l_var=0.05, l_cov=0.005),
    "E2_reg_inv": dict(l_y0=1.0, l_yphi=1.0, l_inv=1.0, l_rnc_g=0.0, l_rnc_res=0.0, l_var=0.05, l_cov=0.005),
    "E3_reg_inv_rnc": dict(l_y0=1.0, l_yphi=1.0, l_inv=1.0, l_rnc_g=0.5, l_rnc_res=0.0, l_var=0.05, l_cov=0.005),
    "E4_full": dict(l_y0=1.0, l_yphi=1.0, l_inv=1.0, l_rnc_g=0.5, l_rnc_res=0.5, l_var=0.05, l_cov=0.005),
    "E5_mask_m1_full": dict(l_y0=1.0, l_yphi=1.0, l_inv=1.0, l_rnc_g=0.5, l_rnc_res=0.5, l_var=0.05, l_cov=0.005, mask_kind="M1_rich"),
    "E6_mask_m2_full": dict(l_y0=1.0, l_yphi=1.0, l_inv=1.0, l_rnc_g=0.5, l_rnc_res=0.5, l_var=0.05, l_cov=0.005, mask_kind="M2_old_strict"),
}


@dataclass
class TrainConfig:
    preset: str
    mask_kind: str
    in_dim: int
    d_back: int
    d_gait: int
    d_aux: int
    encoder_kind: str
    phi_fourier_k: int
    yphi_hidden_dim: int
    y0_dim: int
    yphi_dim: int
    lr: float
    batch_size: int
    dataset_device: str
    epochs: int
    seed: int
    tau: float
    cond_cmd_delta: float
    rnc_max_pos: int
    l_y0: float
    l_yphi: float
    l_inv: float
    l_rnc_g: float
    l_rnc_res: float
    l_var: float
    l_cov: float
    l_phase_adv: float
    l_phase_corr: float
    phase_adv_mode: str
    phase_adv_steps: int
    phase_adv_hidden: int
    yphi_encoder_grad: str
    metric_y0_groups: str
    sample_strategy: str
    sample_weight_cap: float
    l_cmd_adv: float
    l_mode_adv: float
    cmd_mode_adv_mode: str
    cmd_mode_adv_steps: int
    cmd_mode_adv_hidden: int


class StyleWindowDataset(Dataset):
    FLOAT_KEYS = [
        "obs_a",
        "obs_b",
        "phi_a",
        "phi_b",
        "phi_valid_a",
        "phi_valid_b",
        "y0",
        "y0_valid",
        "yphi_a",
        "yphi_b",
        "yphi_valid_a",
        "yphi_valid_b",
        "cmd",
    ]
    INT_KEYS = ["mode_id", "bucket_id", "source_id", "parent_id"]

    def __init__(self, data_dir: str, max_shards: int | None = None, max_samples: int | None = None):
        paths = sorted(glob.glob(os.path.join(data_dir, "style_shard_*.npz")))
        if max_shards is not None:
            paths = paths[:max_shards]
        if not paths:
            raise FileNotFoundError(f"no style_shard_*.npz under {data_dir}")

        stats_path = os.path.join(data_dir, "feature_stats.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"missing {stats_path}; run style_gait_features.py first")
        with open(stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        keys = [
            "obs_a",
            "obs_b",
            "phi_a",
            "phi_b",
            "phi_valid_a",
            "phi_valid_b",
            "y0",
            "y0_valid",
            "yphi_a",
            "yphi_b",
            "yphi_valid_a",
            "yphi_valid_b",
            "cmd",
            "mode_id",
            "bucket_id",
        ]
        optional_keys = ["source_id", "parent_id"]
        loaded = {k: [] for k in keys}
        loaded_optional = {k: [] for k in optional_keys}
        for path in paths:
            d = np.load(path, allow_pickle=True)
            for k in keys:
                loaded[k].append(d[k])
            n = int(d["cmd"].shape[0])
            for k in optional_keys:
                if k in d:
                    loaded_optional[k].append(d[k])
                else:
                    loaded_optional[k].append(np.zeros((n,), dtype=np.int64))
            d.close()
        for k, vals in loaded.items():
            setattr(self, k, np.concatenate(vals, axis=0))
        for k, vals in loaded_optional.items():
            setattr(self, k, np.concatenate(vals, axis=0))

        if max_samples is not None and max_samples < len(self.cmd):
            idx = np.arange(len(self.cmd))[:max_samples]
            for k in keys:
                setattr(self, k, getattr(self, k)[idx])
            for k in optional_keys:
                setattr(self, k, getattr(self, k)[idx])

        for k in self.FLOAT_KEYS:
            setattr(self, k, np.ascontiguousarray(getattr(self, k), dtype=np.float32))
        for k in self.INT_KEYS:
            setattr(self, k, np.ascontiguousarray(getattr(self, k), dtype=np.int64).reshape(-1))

        self.y0_mean = np.asarray(self.stats["y0_mean"], dtype=np.float32)
        self.y0_std = np.asarray(self.stats["y0_std"], dtype=np.float32)
        self.yphi_mean = np.asarray(self.stats["yphi_mean"], dtype=np.float32)
        self.yphi_std = np.asarray(self.stats["yphi_std"], dtype=np.float32)

        self.y0_norm = np.ascontiguousarray(
            np.nan_to_num((self.y0 - self.y0_mean) / np.maximum(self.y0_std, 1e-6), nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float32,
        )
        self.yphi_a_norm = self._normalize_yphi(self.yphi_a)
        self.yphi_b_norm = self._normalize_yphi(self.yphi_b)
        self.y0_res_norm = self._fit_cmd_mode_residuals()

    def _normalize_yphi(self, yphi: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            np.nan_to_num((yphi - self.yphi_mean) / np.maximum(self.yphi_std, 1e-6), nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float32,
        )

    def _fit_cmd_mode_residuals(self, ridge: float = 1e-3):
        cmd = self.cmd
        mode = self.mode_id.reshape(-1)
        onehot = np.zeros((len(mode), 3), dtype=np.float32)
        onehot[np.arange(len(mode)), np.clip(mode, 0, 2)] = 1.0
        x = np.concatenate([cmd, onehot, np.ones((len(cmd), 1), dtype=np.float32)], axis=1)
        y = self.y0_norm
        valid = self.y0_valid.astype(bool)
        pred = np.zeros_like(y, dtype=np.float32)
        eye = np.eye(x.shape[1], dtype=np.float32)
        for j in range(y.shape[1]):
            m = valid[:, j]
            if int(m.sum()) < x.shape[1] + 2:
                continue
            xtx = x[m].T @ x[m] + ridge * eye
            xty = x[m].T @ y[m, j]
            beta = np.linalg.solve(xtx, xty)
            pred[:, j] = x @ beta
        res = y - pred
        res[~valid] = 0.0
        return res.astype(np.float32)

    def __len__(self):
        return int(self.cmd.shape[0])

    def sampling_weights(self, strategy: str, cap: float = 20.0) -> np.ndarray | None:
        if strategy == "uniform":
            return None
        if strategy == "bucket_balanced":
            group = self.bucket_id.reshape(-1)
        elif strategy == "source_bucket_balanced":
            source = self.source_id.reshape(-1)
            bucket = self.bucket_id.reshape(-1)
            group = source * 100 + bucket
        else:
            raise ValueError(f"unknown sample_strategy={strategy!r}")

        _, inv, counts = np.unique(group, return_inverse=True, return_counts=True)
        group_weight = 1.0 / counts.astype(np.float64)
        if cap is not None and cap > 0:
            group_weight = np.minimum(group_weight, group_weight.min() * float(cap))
        weights = group_weight[inv]
        weights = weights / max(float(weights.mean()), 1e-12)
        return weights.astype(np.float64)

    def estimated_tensor_bytes(self) -> int:
        keys = [
            "obs_a",
            "obs_b",
            "phi_a",
            "phi_b",
            "phi_valid_a",
            "phi_valid_b",
            "y0_norm",
            "y0_res_norm",
            "y0_valid",
            "yphi_a_norm",
            "yphi_b_norm",
            "yphi_valid_a",
            "yphi_valid_b",
            "cmd",
            "mode_id",
        ]
        return int(sum(getattr(self, k).nbytes for k in keys))

    def as_tensor_dict(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "obs_a": torch.as_tensor(self.obs_a, dtype=torch.float32, device=device),
            "obs_b": torch.as_tensor(self.obs_b, dtype=torch.float32, device=device),
            "phi_a": torch.as_tensor(self.phi_a, dtype=torch.float32, device=device),
            "phi_b": torch.as_tensor(self.phi_b, dtype=torch.float32, device=device),
            "phi_valid_a": torch.as_tensor(self.phi_valid_a, dtype=torch.float32, device=device),
            "phi_valid_b": torch.as_tensor(self.phi_valid_b, dtype=torch.float32, device=device),
            "y0": torch.as_tensor(self.y0_norm, dtype=torch.float32, device=device),
            "y0_res": torch.as_tensor(self.y0_res_norm, dtype=torch.float32, device=device),
            "y0_valid": torch.as_tensor(self.y0_valid, dtype=torch.float32, device=device),
            "yphi_a": torch.as_tensor(self.yphi_a_norm, dtype=torch.float32, device=device),
            "yphi_b": torch.as_tensor(self.yphi_b_norm, dtype=torch.float32, device=device),
            "yphi_valid_a": torch.as_tensor(self.yphi_valid_a, dtype=torch.float32, device=device),
            "yphi_valid_b": torch.as_tensor(self.yphi_valid_b, dtype=torch.float32, device=device),
            "cmd": torch.as_tensor(self.cmd, dtype=torch.float32, device=device),
            "mode_id": torch.as_tensor(self.mode_id, dtype=torch.long, device=device),
        }

    def __getitem__(self, idx):
        return {
            "obs_a": torch.from_numpy(self.obs_a[idx]),
            "obs_b": torch.from_numpy(self.obs_b[idx]),
            "phi_a": torch.from_numpy(self.phi_a[idx]),
            "phi_b": torch.from_numpy(self.phi_b[idx]),
            "phi_valid_a": torch.from_numpy(self.phi_valid_a[idx]),
            "phi_valid_b": torch.from_numpy(self.phi_valid_b[idx]),
            "y0": torch.from_numpy(self.y0_norm[idx]),
            "y0_res": torch.from_numpy(self.y0_res_norm[idx]),
            "y0_valid": torch.from_numpy(self.y0_valid[idx]),
            "yphi_a": torch.from_numpy(self.yphi_a_norm[idx]),
            "yphi_b": torch.from_numpy(self.yphi_b_norm[idx]),
            "yphi_valid_a": torch.from_numpy(self.yphi_valid_a[idx]),
            "yphi_valid_b": torch.from_numpy(self.yphi_valid_b[idx]),
            "cmd": torch.from_numpy(self.cmd[idx]),
            "mode_id": torch.tensor(int(self.mode_id[idx]), dtype=torch.long),
            "bucket_id": torch.tensor(int(self.bucket_id[idx]), dtype=torch.long),
            "source_id": torch.tensor(int(self.source_id[idx]), dtype=torch.long),
            "parent_id": torch.tensor(int(self.parent_id[idx]), dtype=torch.long),
        }


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _gpu_epoch_batches(
    tensors: dict[str, torch.Tensor],
    n_samples: int,
    batch_size: int,
    sample_weights: torch.Tensor | None,
    generator: torch.Generator,
) -> Iterable[dict[str, torch.Tensor]]:
    device = tensors["cmd"].device
    if sample_weights is None:
        order = torch.randperm(n_samples, device=device, generator=generator)
    else:
        order = torch.multinomial(sample_weights, num_samples=n_samples, replacement=True, generator=generator)
    for start in range(0, n_samples, batch_size):
        idx = order[start : start + batch_size]
        yield {k: v.index_select(0, idx) for k, v in tensors.items()}


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, scale)


def phase_mean_target(phi: torch.Tensor, phi_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = phi_valid.float()
    denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
    target = (phi * w.unsqueeze(-1)).sum(dim=1) / denom
    valid = (w.sum(dim=1, keepdim=True) >= max(4, phi.shape[1] // 4)).float()
    return target, valid


def masked_phase_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = ((pred - target) ** 2).mean(dim=-1, keepdim=True) * valid
    return loss.sum() / valid.sum().clamp(min=1.0)


def phase_corr_loss(z: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    m = valid.reshape(-1) > 0.5
    if int(m.sum()) < 4:
        return z.new_zeros(())
    x = z[m]
    y = target[m]
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4)
    y = (y - y.mean(dim=0, keepdim=True)) / y.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4)
    corr = x.t() @ y / max(int(m.sum()) - 1, 1)
    return (corr ** 2).mean()


def cmd_adv_loss(
    model: "StyleEncoderV4",
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    cmd_norm: torch.Tensor,
    detach_z: bool = False,
    reverse: bool = False,
) -> torch.Tensor:
    if model.cmd_head is None:
        return z_a.new_zeros(())
    za = z_a.detach() if detach_z else z_a
    zb = z_b.detach() if detach_z else z_b
    if reverse:
        za = grad_reverse(za)
        zb = grad_reverse(zb)
    pred_a = model.cmd_head(za)
    pred_b = model.cmd_head(zb)
    return 0.5 * (F.mse_loss(pred_a, cmd_norm) + F.mse_loss(pred_b, cmd_norm))


def mode_adv_loss(
    model: "StyleEncoderV4",
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    mode: torch.Tensor,
    detach_z: bool = False,
    reverse: bool = False,
) -> torch.Tensor:
    if model.mode_head is None:
        return z_a.new_zeros(())
    target = torch.clamp(mode.long(), 0, 2)
    za = z_a.detach() if detach_z else z_a
    zb = z_b.detach() if detach_z else z_b
    if reverse:
        za = grad_reverse(za)
        zb = grad_reverse(zb)
    logits_a = model.mode_head(za)
    logits_b = model.mode_head(zb)
    return 0.5 * (F.cross_entropy(logits_a, target) + F.cross_entropy(logits_b, target))


def _y0_group_indices(names: list[str], group_spec: str) -> np.ndarray:
    if group_spec == "full":
        return np.arange(len(names), dtype=np.int64)
    idx: list[int] = []
    for i, name in enumerate(names):
        is_action = name.startswith("action_") or "action_" in name
        if group_spec == "no_action":
            if not is_action:
                idx.append(i)
        elif group_spec == "kinematic_only":
            if (
                name.startswith("duty_")
                or name.startswith("switch_rate_")
                or name.startswith("step_freq_")
                or name.startswith("double_support")
                or name.startswith("no_support")
                or name.startswith("single_support")
                or name.startswith("contact_")
                or name.startswith("foot_")
                or name.startswith("root_")
                or name.startswith("gravity_")
                or name.startswith("joint_")
            ):
                idx.append(i)
        else:
            raise ValueError(f"unknown metric_y0_groups={group_spec!r}")
    if not idx:
        raise ValueError(f"metric_y0_groups={group_spec!r} selected no y0 features")
    return np.asarray(idx, dtype=np.int64)


class StyleEncoderV4(nn.Module):
    def __init__(
        self,
        in_dim: int,
        y0_dim: int,
        yphi_dim: int,
        d_back: int = 128,
        d_gait: int = 32,
        d_aux: int = 32,
        encoder_kind: str = "tcn",
        phi_fourier_k: int = 1,
        yphi_hidden_dim: int | None = None,
        phase_adv: bool = False,
        phase_adv_hidden: int | None = None,
        cmd_adv: bool = False,
        mode_adv: bool = False,
        cmd_mode_adv_hidden: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_back = d_back
        self.d_gait = d_gait
        self.d_aux = d_aux
        self.encoder_kind = encoder_kind
        self.y0_dim = y0_dim
        self.yphi_dim = yphi_dim
        self.phi_fourier_k = max(1, int(phi_fourier_k))
        self.yphi_hidden_dim = int(yphi_hidden_dim if yphi_hidden_dim is not None else d_back)
        if encoder_kind not in {"tcn", "stats_tcn", "stats_only", "dual_latent", "stats_tcn_dual"}:
            raise ValueError(f"unknown encoder_kind={encoder_kind!r}")
        self.frame_encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, d_back),
            nn.ELU(),
        )
        self.tcn = nn.Sequential(
            nn.Conv1d(d_back, d_back, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(d_back, d_back, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.stats_encoder = None
        if encoder_kind in {"stats_tcn", "stats_only", "stats_tcn_dual"}:
            self.stats_encoder = nn.Sequential(
                nn.Linear(in_dim * 2, 512),
                nn.ELU(),
                nn.Linear(512, d_back),
                nn.ELU(),
            )
        z_in_dim = d_back * 2
        if encoder_kind in {"stats_tcn", "stats_tcn_dual"}:
            z_in_dim += d_back
        elif encoder_kind == "stats_only":
            z_in_dim = d_back
        self.z_head = nn.Sequential(
            nn.Linear(z_in_dim, d_back),
            nn.ELU(),
            nn.Linear(d_back, d_gait),
        )
        self.aux_head = None
        if encoder_kind in {"dual_latent", "stats_tcn_dual"}:
            self.aux_head = nn.Sequential(
                nn.Linear(z_in_dim, d_back),
                nn.ELU(),
                nn.Linear(d_back, d_aux),
            )
        self.proj_head = nn.Sequential(
            nn.Linear(d_gait, d_gait),
            nn.ELU(),
            nn.Linear(d_gait, d_gait),
        )
        self.y0_head = nn.Sequential(
            nn.Linear(d_gait, d_back),
            nn.ELU(),
            nn.Linear(d_back, y0_dim),
        )
        yphi_in_dim = d_gait + 2 * self.phi_fourier_k
        if self.aux_head is not None:
            yphi_in_dim += d_aux
        self.yphi_decoder = nn.Sequential(
            nn.Linear(yphi_in_dim, self.yphi_hidden_dim),
            nn.ELU(),
            nn.Linear(self.yphi_hidden_dim, self.yphi_hidden_dim),
            nn.ELU(),
            nn.Linear(self.yphi_hidden_dim, yphi_dim),
        )
        self.phase_head = None
        if phase_adv:
            h = int(phase_adv_hidden if phase_adv_hidden is not None else d_back)
            self.phase_head = nn.Sequential(
                nn.Linear(d_gait, h),
                nn.ELU(),
                nn.Linear(h, 2),
            )
        h_adv = int(cmd_mode_adv_hidden if cmd_mode_adv_hidden is not None else d_back)
        self.cmd_head = None
        if cmd_adv:
            self.cmd_head = nn.Sequential(
                nn.Linear(d_gait, h_adv),
                nn.ELU(),
                nn.Linear(h_adv, h_adv),
                nn.ELU(),
                nn.Linear(h_adv, 3),
            )
        self.mode_head = None
        if mode_adv:
            self.mode_head = nn.Sequential(
                nn.Linear(d_gait, h_adv),
                nn.ELU(),
                nn.Linear(h_adv, h_adv),
                nn.ELU(),
                nn.Linear(h_adv, 3),
            )

    def _stats_repr(self, obs: torch.Tensor) -> torch.Tensor:
        mean = obs.mean(dim=1)
        std = torch.sqrt(obs.var(dim=1, unbiased=False) + 1e-4)
        if self.stats_encoder is None:
            raise RuntimeError("stats representation requested without stats_encoder")
        return self.stats_encoder(torch.cat([mean, std], dim=-1))

    def _tcn_repr(self, obs: torch.Tensor) -> torch.Tensor:
        b, t, d = obs.shape
        f = self.frame_encoder(obs.reshape(b * t, d)).reshape(b, t, -1)
        h = self.tcn(f.transpose(1, 2)).transpose(1, 2)
        mean = h.mean(dim=1)
        std = torch.sqrt(h.var(dim=1, unbiased=False) + 1e-4)
        return torch.cat([mean, std], dim=-1)

    def encode_parts(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.encoder_kind == "stats_only":
            rep = self._stats_repr(obs)
        else:
            rep = self._tcn_repr(obs)
            if self.encoder_kind in {"stats_tcn", "stats_tcn_dual"}:
                rep = torch.cat([rep, self._stats_repr(obs)], dim=-1)
        z = self.z_head(rep)
        aux = self.aux_head(rep) if self.aux_head is not None else None
        return z, aux

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode_parts(obs)
        return z

    def decode_yphi(self, z: torch.Tensor, phi: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        b, t = phi.shape[:2]
        zt = z.unsqueeze(1).expand(-1, t, -1)
        phi_feat = self.phase_features(phi)
        parts = [zt, phi_feat]
        if self.aux_head is not None:
            if aux is None:
                raise ValueError("dual_latent decoder requires aux")
            parts.append(aux.unsqueeze(1).expand(-1, t, -1))
        return self.yphi_decoder(torch.cat(parts, dim=-1))

    def forward(self, obs: torch.Tensor, phi: torch.Tensor):
        z, aux = self.encode_parts(obs)
        out = {
            "z": z,
            "aux": aux,
            "proj": self.proj_head(z),
            "y0": self.y0_head(z),
            "yphi": self.decode_yphi(z, phi, aux),
        }
        if self.phase_head is not None:
            out["phase_pred"] = self.phase_head(z)
        return out

    def phase_features(self, phi: torch.Tensor) -> torch.Tensor:
        if self.phi_fourier_k <= 1:
            return phi
        angle = torch.atan2(phi[..., 0:1], phi[..., 1:2])
        feats = []
        for k in range(1, self.phi_fourier_k + 1):
            feats += [torch.sin(k * angle), torch.cos(k * angle)]
        return torch.cat(feats, dim=-1)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.float()
    loss = F.smooth_l1_loss(pred, target, reduction="none") * valid
    return loss.sum() / valid.sum().clamp(min=1.0)


def feature_delta(y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.unsqueeze(1) * valid.unsqueeze(0)
    diff2 = (y.unsqueeze(1) - y.unsqueeze(0)) ** 2 * v
    denom = v.sum(dim=-1).clamp(min=1.0)
    return torch.sqrt(diff2.sum(dim=-1) / denom + 1e-8)


def cmd_delta(cmd: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    cw = cmd / sigma.unsqueeze(0).clamp(min=1e-3)
    return torch.norm(cw.unsqueeze(1) - cw.unsqueeze(0), dim=-1)


def rnc_loss_ranked(
    z: torch.Tensor,
    delta: torch.Tensor,
    tau: float = 0.1,
    pair_mask: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    max_pos: int = 32,
) -> torch.Tensor:
    """Approximate Rank-N-Contrast, vectorized on GPU.

    For each anchor i and selected positive j, the denominator contains all
    allowed candidates k with delta[i, k] >= delta[i, j].  The previous
    implementation looped over every (i, j) pair in Python; that made the RnC
    presets CPU/kernel-launch bound.  B=256 gives a manageable BxBxB mask and
    keeps the expensive logsumexp in one batched GPU operation.
    """
    b = z.shape[0]
    if b < 2:
        return z.new_zeros(())
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.t() / tau
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    if pair_mask is None:
        pair_mask = ~eye
    else:
        pair_mask = pair_mask.bool() & (~eye)
    if anchor_mask is None:
        anchor_mask = torch.ones((b,), dtype=torch.bool, device=z.device)
    base_pos = pair_mask & anchor_mask[:, None]
    if not base_pos.any():
        return z.new_zeros(())

    if max_pos is not None and max_pos > 0 and max_pos < b - 1:
        # Randomly select up to max_pos positives per anchor without leaving
        # Python loops on the critical path.
        scores = torch.rand((b, b), device=z.device)
        scores = scores.masked_fill(~base_pos, -1.0)
        k = min(max_pos, int(base_pos.sum(dim=1).max().detach().cpu().item()))
        if k <= 0:
            return z.new_zeros(())
        chosen_idx = torch.topk(scores, k=k, dim=1).indices
        pos_mask = torch.zeros_like(base_pos)
        pos_mask.scatter_(1, chosen_idx, True)
        pos_mask &= base_pos
    else:
        pos_mask = base_pos

    if not pos_mask.any():
        return z.new_zeros(())

    # denom_mask[i, j, k] = pair_mask[i, k] and delta[i, k] >= delta[i, j].
    denom_mask = (delta.unsqueeze(1) >= delta.unsqueeze(2)) & pair_mask.unsqueeze(1)
    neg_inf = -torch.finfo(sim.dtype).max
    sim_ijk = sim.unsqueeze(1).expand(b, b, b)
    log_z = torch.logsumexp(sim_ijk.masked_fill(~denom_mask, neg_inf), dim=-1)
    loss = -(sim - log_z)
    valid = pos_mask & torch.isfinite(loss)
    if not valid.any():
        return z.new_zeros(())
    return loss[valid].mean()


def _apply_mask(obs: torch.Tensor, mask_idx: torch.Tensor | None) -> torch.Tensor:
    if mask_idx is None or mask_idx.numel() == 0:
        return obs
    out = obs.clone()
    out[..., mask_idx] = 0.0
    return out


def apply_preset(args):
    preset = PRESETS[args.preset].copy()
    if "mask_kind" in preset and args.mask_kind is None:
        args.mask_kind = preset.pop("mask_kind")
    if args.mask_kind is None:
        args.mask_kind = "M0_conservative"
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    for k in [
        "l_y0",
        "l_yphi",
        "l_inv",
        "l_rnc_g",
        "l_rnc_res",
        "l_var",
        "l_cov",
        "l_phase_adv",
        "l_phase_corr",
        "l_cmd_adv",
        "l_mode_adv",
    ]:
        if getattr(args, k) is None:
            setattr(args, k, 0.0)


def train(args):
    apply_preset(args)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    ds = StyleWindowDataset(args.data_dir, max_shards=args.max_shards, max_samples=args.max_samples)
    print(f"[style-v4] samples={len(ds)} y0_dim={len(ds.stats['y0_names'])} yphi_dim={len(ds.stats['yphi_names'])}")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sample_weights = ds.sampling_weights(args.sample_strategy, cap=args.sample_weight_cap)
    sampler = None
    if sample_weights is None:
        print("[style-v4] sample_strategy=uniform")
    else:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(ds),
            replacement=True,
            generator=generator,
        )
        print(
            f"[style-v4] sample_strategy={args.sample_strategy} "
            f"weight_min={float(sample_weights.min()):.4f} "
            f"weight_max={float(sample_weights.max()):.4f}"
        )
    use_gpu_dataset = args.dataset_device == "cuda"
    tensor_data = None
    sample_weights_t = None
    gpu_generator = None
    if use_gpu_dataset:
        if device.type != "cuda":
            raise ValueError("--dataset_device cuda requires --device to be a CUDA device")
        est_bytes = ds.estimated_tensor_bytes()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if est_bytes > int(free_bytes * 0.85):
            raise RuntimeError(
                f"GPU resident dataset needs ~{est_bytes / 2**30:.2f} GiB, "
                f"but only {free_bytes / 2**30:.2f} GiB is free on {device}; rerun with --dataset_device cpu"
            )
        print(
            f"[style-v4] dataset_device=cuda estimated={est_bytes / 2**30:.2f}GiB "
            f"free={free_bytes / 2**30:.2f}GiB total={total_bytes / 2**30:.2f}GiB"
        )
        tensor_data = ds.as_tensor_dict(device)
        if sample_weights is not None:
            sample_weights_t = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
            sample_weights_t = sample_weights_t / sample_weights_t.sum().clamp(min=1e-12)
        gpu_generator = torch.Generator(device=device)
        gpu_generator.manual_seed(args.seed)
        loader = None
    else:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_seed_worker if args.num_workers > 0 else None,
            generator=generator,
        )

    in_dim = int(ds.obs_a.shape[-1])
    y0_dim = int(ds.y0.shape[-1])
    yphi_dim = int(ds.yphi_a.shape[-1])
    if args.yphi_hidden_dim is None:
        args.yphi_hidden_dim = args.d_back
    model = StyleEncoderV4(
        in_dim=in_dim,
        y0_dim=y0_dim,
        yphi_dim=yphi_dim,
        d_back=args.d_back,
        d_gait=args.d_gait,
        d_aux=args.d_aux,
        encoder_kind=args.encoder_kind,
        phi_fourier_k=args.phi_fourier_k,
        yphi_hidden_dim=args.yphi_hidden_dim,
        phase_adv=args.l_phase_adv > 0.0,
        phase_adv_hidden=args.phase_adv_hidden,
        cmd_adv=args.l_cmd_adv > 0.0,
        mode_adv=args.l_mode_adv > 0.0,
        cmd_mode_adv_hidden=args.cmd_mode_adv_hidden,
    )
    model.to(device)
    phase_alt = args.phase_adv_mode == "alternating" and model.phase_head is not None
    cmd_mode_alt = args.cmd_mode_adv_mode == "alternating" and (
        model.cmd_head is not None or model.mode_head is not None
    )
    exclude_prefixes: list[str] = []
    if phase_alt:
        exclude_prefixes.append("phase_head.")
    if cmd_mode_alt:
        exclude_prefixes += ["cmd_head.", "mode_head."]
    if exclude_prefixes:
        main_params = [
            p
            for name, p in model.named_parameters()
            if not any(name.startswith(prefix) for prefix in exclude_prefixes)
        ]
        opt = torch.optim.AdamW(main_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if phase_alt:
        phase_opt = torch.optim.AdamW(model.phase_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        phase_opt = None
    if cmd_mode_alt:
        nuisance_params = []
        if model.cmd_head is not None:
            nuisance_params += list(model.cmd_head.parameters())
        if model.mode_head is not None:
            nuisance_params += list(model.mode_head.parameters())
        cmd_mode_opt = torch.optim.AdamW(nuisance_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        cmd_mode_opt = None

    mask_idx_list = build_mask_indices(args.mask_kind)
    mask_idx = torch.tensor(mask_idx_list, dtype=torch.long, device=device) if mask_idx_list else None
    cmd_mean = torch.from_numpy(np.mean(ds.cmd.astype(np.float32), axis=0)).float().to(device)
    cmd_sigma = torch.from_numpy(np.std(ds.cmd.astype(np.float32), axis=0).clip(min=0.05)).float().to(device)
    y0_metric_idx = torch.tensor(_y0_group_indices(ds.stats["y0_names"], args.metric_y0_groups), dtype=torch.long, device=device)
    print(f"[style-v4] preset={args.preset} mask_kind={args.mask_kind} mask_dims={len(mask_idx_list)} cmd_sigma={cmd_sigma.tolist()}")
    print(f"[style-v4] encoder_kind={args.encoder_kind} yphi_encoder_grad={args.yphi_encoder_grad} metric_y0_groups={args.metric_y0_groups} metric_dims={int(y0_metric_idx.numel())}")

    cfg = TrainConfig(
        preset=args.preset,
        mask_kind=args.mask_kind,
        in_dim=in_dim,
        d_back=args.d_back,
        d_gait=args.d_gait,
        d_aux=args.d_aux,
        encoder_kind=args.encoder_kind,
        phi_fourier_k=args.phi_fourier_k,
        yphi_hidden_dim=args.yphi_hidden_dim,
        y0_dim=y0_dim,
        yphi_dim=yphi_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        dataset_device=args.dataset_device,
        epochs=args.epochs,
        seed=args.seed,
        tau=args.tau,
        cond_cmd_delta=args.cond_cmd_delta,
        rnc_max_pos=args.rnc_max_pos,
        l_y0=args.l_y0,
        l_yphi=args.l_yphi,
        l_inv=args.l_inv,
        l_rnc_g=args.l_rnc_g,
        l_rnc_res=args.l_rnc_res,
        l_var=args.l_var,
        l_cov=args.l_cov,
        l_phase_adv=args.l_phase_adv,
        l_phase_corr=args.l_phase_corr,
        phase_adv_mode=args.phase_adv_mode,
        phase_adv_steps=args.phase_adv_steps,
        phase_adv_hidden=args.phase_adv_hidden,
        yphi_encoder_grad=args.yphi_encoder_grad,
        metric_y0_groups=args.metric_y0_groups,
        sample_strategy=args.sample_strategy,
        sample_weight_cap=args.sample_weight_cap,
        l_cmd_adv=args.l_cmd_adv,
        l_mode_adv=args.l_mode_adv,
        cmd_mode_adv_mode=args.cmd_mode_adv_mode,
        cmd_mode_adv_steps=args.cmd_mode_adv_steps,
        cmd_mode_adv_hidden=args.cmd_mode_adv_hidden,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()

    for epoch in range(args.epochs):
        ep = {
            k: 0.0
            for k in [
                "y0",
                "yphi",
                "inv",
                "rnc_g",
                "rnc_res",
                "var",
                "cov",
                "phase_adv",
                "phase_corr",
                "cmd_adv",
                "mode_adv",
                "total",
            ]
        }
        nb = 0
        if use_gpu_dataset:
            assert tensor_data is not None and gpu_generator is not None
            epoch_iter = _gpu_epoch_batches(tensor_data, len(ds), args.batch_size, sample_weights_t, gpu_generator)
        else:
            assert loader is not None
            epoch_iter = loader
        for batch in epoch_iter:
            obs_a = _apply_mask(batch["obs_a"].to(device, non_blocking=True), mask_idx)
            obs_b = _apply_mask(batch["obs_b"].to(device, non_blocking=True), mask_idx)
            phi_a = batch["phi_a"].to(device, non_blocking=True)
            phi_b = batch["phi_b"].to(device, non_blocking=True)
            y0 = batch["y0"].to(device, non_blocking=True)
            y0_res = batch["y0_res"].to(device, non_blocking=True)
            y0_valid = batch["y0_valid"].to(device, non_blocking=True)
            yphi_a = batch["yphi_a"].to(device, non_blocking=True)
            yphi_b = batch["yphi_b"].to(device, non_blocking=True)
            yphi_valid_a = batch["yphi_valid_a"].to(device, non_blocking=True)
            yphi_valid_b = batch["yphi_valid_b"].to(device, non_blocking=True)
            phi_valid_a = batch["phi_valid_a"].to(device, non_blocking=True)
            phi_valid_b = batch["phi_valid_b"].to(device, non_blocking=True)
            cmd = batch["cmd"].to(device, non_blocking=True)
            mode = batch["mode_id"].to(device, non_blocking=True)
            cmd_norm = (cmd - cmd_mean.unsqueeze(0)) / cmd_sigma.unsqueeze(0).clamp(min=1e-3)

            out_a = model(obs_a, phi_a)
            out_b = model(obs_b, phi_b)
            z_a = out_a["z"]
            z_b = out_b["z"]

            l_y0 = 0.5 * (
                masked_huber(out_a["y0"], y0, y0_valid)
                + masked_huber(out_b["y0"], y0, y0_valid)
            )
            if args.yphi_encoder_grad == "off" or args.l_yphi <= 0.0:
                l_yphi = z_a.new_zeros(())
            elif args.yphi_encoder_grad == "stopgrad":
                aux_a = out_a["aux"].detach() if out_a["aux"] is not None else None
                aux_b = out_b["aux"].detach() if out_b["aux"] is not None else None
                yphi_pred_a = model.decode_yphi(z_a.detach(), phi_a, aux_a)
                yphi_pred_b = model.decode_yphi(z_b.detach(), phi_b, aux_b)
                l_yphi = 0.5 * (
                    masked_huber(yphi_pred_a, yphi_a, yphi_valid_a)
                    + masked_huber(yphi_pred_b, yphi_b, yphi_valid_b)
                )
            else:
                l_yphi = 0.5 * (
                    masked_huber(out_a["yphi"], yphi_a, yphi_valid_a)
                    + masked_huber(out_b["yphi"], yphi_b, yphi_valid_b)
                )
            l_inv = F.mse_loss(z_a, z_b)

            phase_target_a, phase_window_valid_a = phase_mean_target(phi_a, phi_valid_a)
            phase_target_b, phase_window_valid_b = phase_mean_target(phi_b, phi_valid_b)
            if args.l_phase_adv > 0.0:
                if phase_opt is not None and model.phase_head is not None:
                    for _ in range(max(1, args.phase_adv_steps)):
                        phase_opt.zero_grad(set_to_none=True)
                        phase_fit = 0.5 * (
                            masked_phase_mse(model.phase_head(z_a.detach()), phase_target_a, phase_window_valid_a)
                            + masked_phase_mse(model.phase_head(z_b.detach()), phase_target_b, phase_window_valid_b)
                        )
                        phase_fit.backward()
                        phase_opt.step()
                phase_pred_a = model.phase_head(grad_reverse(z_a)) if model.phase_head is not None else z_a.new_zeros((z_a.shape[0], 2))
                phase_pred_b = model.phase_head(grad_reverse(z_b)) if model.phase_head is not None else z_b.new_zeros((z_b.shape[0], 2))
                l_phase_adv = 0.5 * (
                    masked_phase_mse(phase_pred_a, phase_target_a, phase_window_valid_a)
                    + masked_phase_mse(phase_pred_b, phase_target_b, phase_window_valid_b)
                )
            else:
                l_phase_adv = z_a.new_zeros(())
            if args.l_phase_corr > 0.0:
                l_phase_corr = 0.5 * (
                    phase_corr_loss(z_a, phase_target_a, phase_window_valid_a)
                    + phase_corr_loss(z_b, phase_target_b, phase_window_valid_b)
                )
            else:
                l_phase_corr = z_a.new_zeros(())

            if cmd_mode_opt is not None:
                for _ in range(max(1, args.cmd_mode_adv_steps)):
                    cmd_mode_opt.zero_grad(set_to_none=True)
                    fit = z_a.new_zeros(())
                    if args.l_cmd_adv > 0.0:
                        fit = fit + cmd_adv_loss(model, z_a, z_b, cmd_norm, detach_z=True, reverse=False)
                    if args.l_mode_adv > 0.0:
                        fit = fit + mode_adv_loss(model, z_a, z_b, mode, detach_z=True, reverse=False)
                    fit.backward()
                    cmd_mode_opt.step()
            if args.l_cmd_adv > 0.0:
                l_cmd_adv = cmd_adv_loss(model, z_a, z_b, cmd_norm, detach_z=False, reverse=True)
            else:
                l_cmd_adv = z_a.new_zeros(())
            if args.l_mode_adv > 0.0:
                l_mode_adv = mode_adv_loss(model, z_a, z_b, mode, detach_z=False, reverse=True)
            else:
                l_mode_adv = z_a.new_zeros(())

            z_cat = torch.cat([z_a, z_b], dim=0)
            l_var = vicreg_var(z_cat)
            l_cov = vicreg_cov(z_cat)

            valid_style = (mode != 0) & (y0_valid.sum(dim=-1) > 1.0)
            if args.l_rnc_g > 0.0:
                y0_metric = y0.index_select(dim=-1, index=y0_metric_idx)
                y0_valid_metric = y0_valid.index_select(dim=-1, index=y0_metric_idx)
                delta_g = feature_delta(y0_metric, y0_valid_metric)
                pair = valid_style[:, None] & valid_style[None, :]
                l_rnc_g = rnc_loss_ranked(
                    out_a["proj"],
                    delta_g,
                    tau=args.tau,
                    pair_mask=pair,
                    anchor_mask=valid_style,
                    max_pos=args.rnc_max_pos,
                )
            else:
                l_rnc_g = z_a.new_zeros(())

            if args.l_rnc_res > 0.0:
                y0_res_metric = y0_res.index_select(dim=-1, index=y0_metric_idx)
                y0_valid_metric = y0_valid.index_select(dim=-1, index=y0_metric_idx)
                delta_res = feature_delta(y0_res_metric, y0_valid_metric)
                dcmd = cmd_delta(cmd, cmd_sigma)
                same_mode = mode[:, None] == mode[None, :]
                pair = valid_style[:, None] & valid_style[None, :] & same_mode & (dcmd <= args.cond_cmd_delta)
                l_rnc_res = rnc_loss_ranked(
                    out_a["proj"],
                    delta_res,
                    tau=args.tau,
                    pair_mask=pair,
                    anchor_mask=valid_style,
                    max_pos=args.rnc_max_pos,
                )
            else:
                l_rnc_res = z_a.new_zeros(())

            total = (
                args.l_y0 * l_y0
                + args.l_yphi * l_yphi
                + args.l_inv * l_inv
                + args.l_rnc_g * l_rnc_g
                + args.l_rnc_res * l_rnc_res
                + args.l_var * l_var
                + args.l_cov * l_cov
                + args.l_phase_adv * l_phase_adv
                + args.l_phase_corr * l_phase_corr
                + args.l_cmd_adv * l_cmd_adv
                + args.l_mode_adv * l_mode_adv
            )

            opt.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()

            vals = {
                "y0": l_y0,
                "yphi": l_yphi,
                "inv": l_inv,
                "rnc_g": l_rnc_g,
                "rnc_res": l_rnc_res,
                "var": l_var,
                "cov": l_cov,
                "phase_adv": l_phase_adv,
                "phase_corr": l_phase_corr,
                "cmd_adv": l_cmd_adv,
                "mode_adv": l_mode_adv,
                "total": total,
            }
            for k, v in vals.items():
                ep[k] += float(v.detach().cpu())
            nb += 1

        rec = {k: v / max(nb, 1) for k, v in ep.items()}
        rec["epoch"] = epoch
        rec["elapsed_s"] = round(time.time() - t0, 2)
        print("[style-v4] " + json.dumps(rec, sort_keys=True))
        log.write(json.dumps(rec, sort_keys=True) + "\n")
        log.flush()

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "feature_stats": ds.stats,
                "cmd_mean": cmd_mean.detach().cpu().numpy().tolist(),
                "cmd_sigma": cmd_sigma.detach().cpu().numpy().tolist(),
                "mask_indices": mask_idx_list,
                "obs_layout": describe_layout(),
                "encoder_kind": "frnc_style_v4",
            }
            torch.save(ckpt, os.path.join(args.out_dir, "encoder.pt"))

    log.close()
    print(f"[style-v4] wrote {os.path.join(args.out_dir, 'encoder.pt')}")


def main():
    ap = argparse.ArgumentParser(description="Train frnc_style_v4 encoder.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="E4_full")
    ap.add_argument("--mask_kind", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--dataset_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--d_back", type=int, default=128)
    ap.add_argument("--d_gait", type=int, default=32)
    ap.add_argument("--d_aux", type=int, default=32)
    ap.add_argument("--encoder_kind", choices=["tcn", "stats_tcn", "stats_only", "dual_latent", "stats_tcn_dual"], default="tcn")
    ap.add_argument("--phi_fourier_k", type=int, default=1)
    ap.add_argument("--yphi_hidden_dim", type=int, default=None)
    ap.add_argument("--yphi_encoder_grad", choices=["full", "stopgrad", "off"], default="full")
    ap.add_argument("--phase_adv_mode", choices=["grl", "alternating"], default="grl")
    ap.add_argument("--phase_adv_steps", type=int, default=1)
    ap.add_argument("--phase_adv_hidden", type=int, default=128)
    ap.add_argument("--metric_y0_groups", choices=["full", "no_action", "kinematic_only"], default="full")
    ap.add_argument("--sample_strategy", choices=["uniform", "bucket_balanced", "source_bucket_balanced"], default="uniform")
    ap.add_argument("--sample_weight_cap", type=float, default=20.0)
    ap.add_argument("--cmd_mode_adv_mode", choices=["grl", "alternating"], default="grl")
    ap.add_argument("--cmd_mode_adv_steps", type=int, default=1)
    ap.add_argument("--cmd_mode_adv_hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--cond_cmd_delta", type=float, default=0.35)
    ap.add_argument("--rnc_max_pos", type=int, default=32)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=10)
    for key in [
        "l_y0",
        "l_yphi",
        "l_inv",
        "l_rnc_g",
        "l_rnc_res",
        "l_var",
        "l_cov",
        "l_phase_adv",
        "l_phase_corr",
        "l_cmd_adv",
        "l_mode_adv",
    ]:
        ap.add_argument(f"--{key}", type=float, default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
