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
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
    y0_dim: int
    yphi_dim: int
    lr: float
    batch_size: int
    epochs: int
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


class StyleWindowDataset(Dataset):
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
        loaded = {k: [] for k in keys}
        for path in paths:
            d = np.load(path, allow_pickle=True)
            for k in keys:
                loaded[k].append(d[k])
            d.close()
        for k, vals in loaded.items():
            setattr(self, k, np.concatenate(vals, axis=0))

        if max_samples is not None and max_samples < len(self.cmd):
            idx = np.arange(len(self.cmd))[:max_samples]
            for k in keys:
                setattr(self, k, getattr(self, k)[idx])

        self.y0_mean = np.asarray(self.stats["y0_mean"], dtype=np.float32)
        self.y0_std = np.asarray(self.stats["y0_std"], dtype=np.float32)
        self.yphi_mean = np.asarray(self.stats["yphi_mean"], dtype=np.float32)
        self.yphi_std = np.asarray(self.stats["yphi_std"], dtype=np.float32)

        self.y0_norm = (self.y0.astype(np.float32) - self.y0_mean) / np.maximum(self.y0_std, 1e-6)
        self.y0_norm = np.nan_to_num(self.y0_norm, nan=0.0, posinf=0.0, neginf=0.0)
        self.y0_res_norm = self._fit_cmd_mode_residuals()

    def _fit_cmd_mode_residuals(self, ridge: float = 1e-3):
        cmd = self.cmd.astype(np.float32)
        mode = np.asarray(self.mode_id, dtype=np.int64).reshape(-1)
        onehot = np.zeros((len(mode), 3), dtype=np.float32)
        onehot[np.arange(len(mode)), np.clip(mode, 0, 2)] = 1.0
        x = np.concatenate([cmd, onehot, np.ones((len(cmd), 1), dtype=np.float32)], axis=1)
        y = self.y0_norm.astype(np.float32)
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

    def __getitem__(self, idx):
        yphi_a = (self.yphi_a[idx].astype(np.float32) - self.yphi_mean) / np.maximum(self.yphi_std, 1e-6)
        yphi_b = (self.yphi_b[idx].astype(np.float32) - self.yphi_mean) / np.maximum(self.yphi_std, 1e-6)
        yphi_a = np.nan_to_num(yphi_a, nan=0.0, posinf=0.0, neginf=0.0)
        yphi_b = np.nan_to_num(yphi_b, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "obs_a": torch.from_numpy(self.obs_a[idx].astype(np.float32)),
            "obs_b": torch.from_numpy(self.obs_b[idx].astype(np.float32)),
            "phi_a": torch.from_numpy(self.phi_a[idx].astype(np.float32)),
            "phi_b": torch.from_numpy(self.phi_b[idx].astype(np.float32)),
            "phi_valid_a": torch.from_numpy(self.phi_valid_a[idx].astype(np.float32)),
            "phi_valid_b": torch.from_numpy(self.phi_valid_b[idx].astype(np.float32)),
            "y0": torch.from_numpy(self.y0_norm[idx].astype(np.float32)),
            "y0_res": torch.from_numpy(self.y0_res_norm[idx].astype(np.float32)),
            "y0_valid": torch.from_numpy(self.y0_valid[idx].astype(np.float32)),
            "yphi_a": torch.from_numpy(yphi_a.astype(np.float32)),
            "yphi_b": torch.from_numpy(yphi_b.astype(np.float32)),
            "yphi_valid_a": torch.from_numpy(self.yphi_valid_a[idx].astype(np.float32)),
            "yphi_valid_b": torch.from_numpy(self.yphi_valid_b[idx].astype(np.float32)),
            "cmd": torch.from_numpy(self.cmd[idx].astype(np.float32)),
            "mode_id": torch.tensor(int(np.asarray(self.mode_id[idx]).reshape(())), dtype=torch.long),
            "bucket_id": torch.tensor(int(np.asarray(self.bucket_id[idx]).reshape(())), dtype=torch.long),
        }


class StyleEncoderV4(nn.Module):
    def __init__(self, in_dim: int, y0_dim: int, yphi_dim: int, d_back: int = 128, d_gait: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.d_back = d_back
        self.d_gait = d_gait
        self.y0_dim = y0_dim
        self.yphi_dim = yphi_dim
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
        self.z_head = nn.Sequential(
            nn.Linear(d_back * 2, d_back),
            nn.ELU(),
            nn.Linear(d_back, d_gait),
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
        self.yphi_decoder = nn.Sequential(
            nn.Linear(d_gait + 2, d_back),
            nn.ELU(),
            nn.Linear(d_back, d_back),
            nn.ELU(),
            nn.Linear(d_back, yphi_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        b, t, d = obs.shape
        f = self.frame_encoder(obs.reshape(b * t, d)).reshape(b, t, -1)
        h = self.tcn(f.transpose(1, 2)).transpose(1, 2)
        mean = h.mean(dim=1)
        std = torch.sqrt(h.var(dim=1, unbiased=False) + 1e-4)
        return self.z_head(torch.cat([mean, std], dim=-1))

    def forward(self, obs: torch.Tensor, phi: torch.Tensor):
        z = self.encode(obs)
        b, t = obs.shape[:2]
        zt = z.unsqueeze(1).expand(-1, t, -1)
        yphi = self.yphi_decoder(torch.cat([zt, phi], dim=-1))
        return {
            "z": z,
            "proj": self.proj_head(z),
            "y0": self.y0_head(z),
            "yphi": yphi,
        }


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
    """Approximate Rank-N-Contrast without allocating a BxBxB tensor."""
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
    losses = []
    for i in torch.where(anchor_mask)[0]:
        js = torch.where(pair_mask[i])[0]
        if js.numel() == 0:
            continue
        if js.numel() > max_pos:
            js = js[torch.randperm(js.numel(), device=z.device)[:max_pos]]
        for j in js:
            cand = (delta[i] >= delta[i, j]) & pair_mask[i]
            if cand.sum() <= 0:
                continue
            log_z = torch.logsumexp(sim[i, cand], dim=0)
            losses.append(-(sim[i, j] - log_z))
    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


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
    for k in ["l_y0", "l_yphi", "l_inv", "l_rnc_g", "l_rnc_res", "l_var", "l_cov"]:
        if getattr(args, k) is None:
            setattr(args, k, 0.0)


def train(args):
    apply_preset(args)
    device = torch.device(args.device)
    ds = StyleWindowDataset(args.data_dir, max_shards=args.max_shards, max_samples=args.max_samples)
    print(f"[style-v4] samples={len(ds)} y0_dim={len(ds.stats['y0_names'])} yphi_dim={len(ds.stats['yphi_names'])}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    in_dim = int(ds.obs_a.shape[-1])
    y0_dim = int(ds.y0.shape[-1])
    yphi_dim = int(ds.yphi_a.shape[-1])
    model = StyleEncoderV4(in_dim=in_dim, y0_dim=y0_dim, yphi_dim=yphi_dim, d_back=args.d_back, d_gait=args.d_gait)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mask_idx_list = build_mask_indices(args.mask_kind)
    mask_idx = torch.tensor(mask_idx_list, dtype=torch.long, device=device) if mask_idx_list else None
    cmd_sigma = torch.from_numpy(np.std(ds.cmd.astype(np.float32), axis=0).clip(min=0.05)).float().to(device)
    print(f"[style-v4] preset={args.preset} mask_kind={args.mask_kind} mask_dims={len(mask_idx_list)} cmd_sigma={cmd_sigma.tolist()}")

    cfg = TrainConfig(
        preset=args.preset,
        mask_kind=args.mask_kind,
        in_dim=in_dim,
        d_back=args.d_back,
        d_gait=args.d_gait,
        y0_dim=y0_dim,
        yphi_dim=yphi_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
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
    )

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()

    for epoch in range(args.epochs):
        ep = {k: 0.0 for k in ["y0", "yphi", "inv", "rnc_g", "rnc_res", "var", "cov", "total"]}
        nb = 0
        for batch in loader:
            obs_a = _apply_mask(batch["obs_a"].to(device), mask_idx)
            obs_b = _apply_mask(batch["obs_b"].to(device), mask_idx)
            phi_a = batch["phi_a"].to(device)
            phi_b = batch["phi_b"].to(device)
            y0 = batch["y0"].to(device)
            y0_res = batch["y0_res"].to(device)
            y0_valid = batch["y0_valid"].to(device)
            yphi_a = batch["yphi_a"].to(device)
            yphi_b = batch["yphi_b"].to(device)
            yphi_valid_a = batch["yphi_valid_a"].to(device)
            yphi_valid_b = batch["yphi_valid_b"].to(device)
            cmd = batch["cmd"].to(device)
            mode = batch["mode_id"].to(device)

            out_a = model(obs_a, phi_a)
            out_b = model(obs_b, phi_b)
            z_a = out_a["z"]
            z_b = out_b["z"]

            l_y0 = 0.5 * (
                masked_huber(out_a["y0"], y0, y0_valid)
                + masked_huber(out_b["y0"], y0, y0_valid)
            )
            l_yphi = 0.5 * (
                masked_huber(out_a["yphi"], yphi_a, yphi_valid_a)
                + masked_huber(out_b["yphi"], yphi_b, yphi_valid_b)
            )
            l_inv = F.mse_loss(z_a, z_b)

            z_cat = torch.cat([z_a, z_b], dim=0)
            l_var = vicreg_var(z_cat)
            l_cov = vicreg_cov(z_cat)

            valid_style = (mode != 0) & (y0_valid.sum(dim=-1) > 1.0)
            if args.l_rnc_g > 0.0:
                delta_g = feature_delta(y0, y0_valid)
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
                delta_res = feature_delta(y0_res, y0_valid)
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
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--d_back", type=int, default=128)
    ap.add_argument("--d_gait", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--cond_cmd_delta", type=float, default=0.35)
    ap.add_argument("--rnc_max_pos", type=int, default=32)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=10)
    for key in ["l_y0", "l_yphi", "l_inv", "l_rnc_g", "l_rnc_res", "l_var", "l_cov"]:
        ap.add_argument(f"--{key}", type=float, default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
