# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Segment-level encoder pretraining.

Models gait mode `g` as a per-segment SLOW factor that is nearly constant
within a contiguous trajectory segment, and a periodic time variable phi(t)
that carries the fast (within-segment) variation. Decoder psi(g, phi) is
trained to reconstruct (joint relative phase, foot contact); a per-frame
velocity bypass head v(f_t) keeps cmd information out of g unless the
segment-level RnC explicitly injects it.

Outputs:
  * encoder.pt     state dict + config (compatible loader expected by probe)
  * train.log      per-epoch losses
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from frnc_pretrain import _hilbert_global_phase_targets, _parse_index_list


# ---------------------- segment dataset ---------------------- #
@dataclass
class _Segment:
    shard_idx: int
    rows: np.ndarray  # absolute row indices in shard (sorted by gstep)


def _build_segments(shards, episode_step_key="episode_step",
                    global_step_key="global_step",
                    env_id_key="env_id",
                    min_len: int = 32,
                    max_len: int = 64):
    """Walk each shard, group rows by env_id, sort by global_step, split on
    episode_step==0 boundary. Each maximal run >= min_len is chunked into
    pieces of length `max_len` (last short piece dropped if < min_len).
    Returns list of `_Segment`.
    """
    segs = []
    for si, c in enumerate(shards):
        envs = c[env_id_key]
        gst = c[global_step_key]
        est = c[episode_step_key]
        for eid in np.unique(envs):
            mask = envs == eid
            rows = np.where(mask)[0]
            order = rows[np.argsort(gst[rows])]
            es = est[order]
            starts = np.concatenate([[0], np.where(es == 0)[0]])
            starts = np.unique(starts)
            ends = np.concatenate([starts[1:], [len(order)]])
            for s_, e_ in zip(starts, ends):
                run = order[s_:e_]
                L = len(run)
                if L < min_len:
                    continue
                # chunk by max_len
                k = 0
                while k + min_len <= L:
                    end = min(k + max_len, L)
                    segs.append(_Segment(si, run[k:end]))
                    k = end
    return segs


class SegmentDataset(Dataset):
    """Yields per-segment dicts of arrays.

    Each item:
      obs       (T, D)
      cmd       (T, 3)
      foot      (T, F)  uint8
      phase     (T, 2J) float32 with NaN where invalid (anchor relative phases)
      anchor_sc (T, 2)  float32 with NaN: (sin phi_anchor, cos phi_anchor)
    Variable-length T per segment. Use a custom collate to pad+mask.
    """

    def __init__(self, shard_paths, mask_obs_indices=None,
                 phase_joint_slice=(245, 260),
                 phase_anchor_joint_idx: int = 0,
                 segment_min_len: int = 32, segment_max_len: int = 64):
        self.mask = mask_obs_indices
        self.phase_anchor_joint_idx = int(phase_anchor_joint_idx)
        self.shards = []
        for p in shard_paths:
            d = np.load(p)
            self.shards.append({
                "policy_obs": d["policy_obs"].astype(np.float32),
                "cmd": d["cmd"].astype(np.float32),
                "foot": d["foot_contact"].astype(np.uint8),
                "env_id": d["env_id"].astype(np.int64),
                "episode_step": d["episode_step"].astype(np.int64),
                "global_step": d["global_step"].astype(np.int64),
            })
        self.lo, self.hi = phase_joint_slice
        self.segments = _build_segments(self.shards,
                                        min_len=segment_min_len,
                                        max_len=segment_max_len)
        # precompute joint phase per segment (relative to anchor) and anchor sin/cos
        self.phase_cache = []
        self.anchor_cache = []
        from scipy.signal import hilbert
        for seg in self.segments:
            c = self.shards[seg.shard_idx]
            jpos = c["policy_obs"][seg.rows, self.lo:self.hi].astype(np.float32)
            T, J = jpos.shape
            ph = _hilbert_global_phase_targets(jpos, anchor_idx=self.phase_anchor_joint_idx,
                                               min_len=segment_min_len)
            # per-frame anchor sin/cos: detrend anchor and take analytic phase
            x = jpos - jpos.mean(axis=0, keepdims=True)
            asc = np.full((T, 2), np.nan, dtype=np.float32)
            try:
                if float(np.std(x[:, self.phase_anchor_joint_idx])) >= 1e-3:
                    phi = np.angle(hilbert(x[:, self.phase_anchor_joint_idx]))
                    asc[:, 0] = np.sin(phi).astype(np.float32)
                    asc[:, 1] = np.cos(phi).astype(np.float32)
                    edge = max(4, T // 10)
                    asc[:edge] = np.nan
                    asc[-edge:] = np.nan
            except Exception:
                pass
            self.phase_cache.append(ph)
            self.anchor_cache.append(asc)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        c = self.shards[seg.shard_idx]
        obs = c["policy_obs"][seg.rows]
        if self.mask is not None:
            obs = obs.copy()
            obs[:, self.mask] = 0.0
        return {
            "obs": obs,
            "cmd": c["cmd"][seg.rows],
            "foot": c["foot"][seg.rows].astype(np.float32),
            "phase": self.phase_cache[idx],
            "anchor_sc": self.anchor_cache[idx],
        }


def collate_segments(batch):
    """Pad to the max T in batch; return tensors + a (B, T) valid mask."""
    Tmax = max(b["obs"].shape[0] for b in batch)
    B = len(batch)
    D = batch[0]["obs"].shape[1]
    F_ = batch[0]["foot"].shape[1]
    Pd = batch[0]["phase"].shape[1]
    obs = np.zeros((B, Tmax, D), dtype=np.float32)
    cmd = np.zeros((B, Tmax, 3), dtype=np.float32)
    foot = np.zeros((B, Tmax, F_), dtype=np.float32)
    phase = np.full((B, Tmax, Pd), np.nan, dtype=np.float32)
    anc = np.full((B, Tmax, 2), np.nan, dtype=np.float32)
    mask = np.zeros((B, Tmax), dtype=np.float32)
    for i, b in enumerate(batch):
        T = b["obs"].shape[0]
        obs[i, :T] = b["obs"]
        cmd[i, :T] = b["cmd"]
        foot[i, :T] = b["foot"]
        phase[i, :T] = b["phase"]
        anc[i, :T] = b["anchor_sc"]
        mask[i, :T] = 1.0
    return {
        "obs": torch.from_numpy(obs),
        "cmd": torch.from_numpy(cmd),
        "foot": torch.from_numpy(foot),
        "phase": torch.from_numpy(phase),
        "anchor_sc": torch.from_numpy(anc),
        "mask": torch.from_numpy(mask),
    }


# ---------------------- model ---------------------- #
class SegmentEncoder(nn.Module):
    def __init__(self, in_dim: int, d_back: int = 128, d_gait: int = 32,
                 phase_dim: int = 30, foot_dim: int = 2,
                 decoder_hidden: int = 128, hard_seg_mean: bool = True):
        super().__init__()
        self.hard_seg_mean = hard_seg_mean
        self.phase_dim = phase_dim
        self.foot_dim = foot_dim
        self.d_gait = d_gait
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, d_back), nn.ELU(),
        )
        self.gait_proj = nn.Sequential(
            nn.Linear(d_back, d_gait), nn.ELU(),
        )
        # decoder psi(g, sin phi, cos phi) -> (phase 30d, foot 2d)
        self.decoder = nn.Sequential(
            nn.Linear(d_gait + 2, decoder_hidden), nn.ELU(),
            nn.Linear(decoder_hidden, decoder_hidden), nn.ELU(),
        )
        self.dec_phase = nn.Linear(decoder_hidden, phase_dim)
        self.dec_foot = nn.Linear(decoder_hidden, foot_dim)
        # vel head on per-frame f (bypass; cmd should NOT flow through g)
        self.vel_head = nn.Linear(d_back, 3)

    def encode_frames(self, obs):
        """obs: (B, T, D). Returns f (B,T,d_back), e (B,T,d_gait)."""
        B, T, D = obs.shape
        f = self.backbone(obs.reshape(B * T, D)).reshape(B, T, -1)
        e = self.gait_proj(f)
        return f, e

    def segment_mean(self, e, mask):
        """e: (B,T,d), mask: (B,T) -> z_gait (B,d)."""
        m = mask.unsqueeze(-1)
        s = (e * m).sum(dim=1)
        n = m.sum(dim=1).clamp(min=1.0)
        return s / n

    def forward(self, obs, mask, anchor_sc):
        """anchor_sc: (B, T, 2) sin/cos phi (NaN where invalid)."""
        f, e = self.encode_frames(obs)
        z_gait = self.segment_mean(e, mask)  # (B, d_gait)
        # broadcast z_gait to per-frame for decoder input
        B, T = obs.shape[:2]
        if self.hard_seg_mean:
            z_per_frame = z_gait.unsqueeze(1).expand(-1, T, -1)
        else:
            z_per_frame = e  # soft: per-frame e_t still feeds decoder
        anc = torch.nan_to_num(anchor_sc, nan=0.0)
        dec_in = torch.cat([z_per_frame, anc], dim=-1)
        h = self.decoder(dec_in)
        phase_pred = self.dec_phase(h)
        foot_pred = self.dec_foot(h)
        v_pred = self.vel_head(f)  # per-frame
        return {
            "f": f, "e": e, "z_gait": z_gait,
            "phase_pred": phase_pred, "foot_pred": foot_pred,
            "v_pred": v_pred,
        }


# ---------------------- losses ---------------------- #
def rnc_loss_seg(z, delta, tau=0.1):
    """Segment-level RnC. z: (B,d), delta: (B,B). Memory O(B^3); B<=128 ok."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / tau
    B = z.size(0)
    diag = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag, -1e4)
    M = (delta.unsqueeze(1) >= delta.unsqueeze(2)).float()
    M = M * (~diag).unsqueeze(1).float()
    sim_exp = sim.unsqueeze(1).expand(-1, B, -1)
    masked = sim_exp.masked_fill(M == 0, -1e4)
    logZ = torch.logsumexp(masked, dim=-1)
    log_p = sim - logZ
    log_p = log_p.masked_fill(diag, 0.0)
    return -(log_p.sum() / (B * (B - 1)))


def vicreg_var(z, gamma=1.0):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


def vicreg_cov(z):
    """Off-diagonal covariance penalty (decorrelation)."""
    z = z - z.mean(dim=0, keepdim=True)
    n = z.size(0)
    cov = (z.t() @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / z.size(1)


def slow_loss(e, z_gait, mask):
    """||e_t - z_gait||^2 averaged over valid frames."""
    diff = (e - z_gait.unsqueeze(1)) ** 2
    diff = diff.sum(dim=-1) * mask  # (B, T)
    return diff.sum() / mask.sum().clamp(min=1.0)


# ---------------------- training ---------------------- #
def train(args):
    device = torch.device(args.device)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    assert shards, f"no shards in {args.data_dir}"
    print(f"[seg] {len(shards)} shards")
    mask_idx = _parse_index_list(args.mask_obs_indices)
    if mask_idx is not None:
        print(f"[seg] masking {len(mask_idx)} obs dims")
    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))
    ds = SegmentDataset(
        shards, mask_obs_indices=mask_idx,
        phase_joint_slice=phase_slice,
        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
        segment_min_len=args.segment_min_len,
        segment_max_len=args.segment_max_len,
    )
    print(f"[seg] {len(ds)} segments")
    sample = ds[0]
    in_dim = sample["obs"].shape[1]
    phase_dim = sample["phase"].shape[1]
    foot_dim = sample["foot"].shape[1]
    print(f"[seg] in_dim={in_dim} phase_dim={phase_dim} foot_dim={foot_dim}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, drop_last=True, collate_fn=collate_segments,
        pin_memory=True,
    )

    model = SegmentEncoder(
        in_dim=in_dim, d_back=args.d_back, d_gait=args.d_gait,
        phase_dim=phase_dim, foot_dim=foot_dim,
        hard_seg_mean=bool(args.hard_seg_mean),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    W = torch.tensor([1.0, 1.0, args.r_leg], device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log_f = open(log_path, "w")

    t0 = time.time()
    for epoch in range(args.epochs):
        ep = {k: 0.0 for k in [
            "rnc", "recon_phase", "recon_foot", "var", "cov", "slow", "vel", "total",
        ]}
        nb = 0
        for batch in loader:
            obs = batch["obs"].to(device, non_blocking=True)
            cmd = batch["cmd"].to(device, non_blocking=True)
            foot = batch["foot"].to(device, non_blocking=True)
            phase = batch["phase"].to(device, non_blocking=True)
            anc = batch["anchor_sc"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            out = model(obs, mask, anc)

            # segment-level cmd label = mean cmd within segment
            cmd_seg = (cmd * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            with torch.no_grad():
                cmd_w = cmd_seg * W.unsqueeze(0)
                delta = torch.cdist(cmd_w, cmd_w, p=2)

            # 1. segment RnC
            l_rnc = rnc_loss_seg(out["z_gait"], delta, args.tau) if args.l_seg_rnc > 0 else torch.zeros((), device=device)

            # 2. reconstruction: only where anchor & phase are valid
            # phase target shape (B,T,Pd); anchor (B,T,2). Valid frame iff anchor sin/cos not NaN.
            valid_anc = ~torch.isnan(anc).any(dim=-1)  # (B, T)
            valid_phase = ~torch.isnan(phase).any(dim=-1)  # (B, T)
            valid = mask.bool() & valid_anc & valid_phase

            if args.l_recon_phase > 0 and valid.any():
                ph_pred = out["phase_pred"]
                ph_t = torch.nan_to_num(phase, nan=0.0)
                d2 = ((ph_pred - ph_t) ** 2).sum(dim=-1)
                l_recon_phase = (d2 * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
            else:
                l_recon_phase = torch.zeros((), device=device)

            if args.l_recon_foot > 0 and (mask.bool() & valid_anc).any():
                vf = (mask.bool() & valid_anc).float()
                fp = out["foot_pred"]
                bce = F.binary_cross_entropy_with_logits(fp, foot, reduction="none").sum(dim=-1)
                l_recon_foot = (bce * vf).sum() / vf.sum().clamp(min=1.0)
            else:
                l_recon_foot = torch.zeros((), device=device)

            # 3. VICReg on segment-level z_gait
            l_var = vicreg_var(out["z_gait"]) if args.l_var > 0 else torch.zeros((), device=device)
            l_cov = vicreg_cov(out["z_gait"]) if args.l_cov > 0 else torch.zeros((), device=device)

            # 4. slow constraint (only if soft)
            if (not bool(args.hard_seg_mean)) and args.l_slow > 0:
                l_slow = slow_loss(out["e"], out["z_gait"], mask)
            else:
                l_slow = torch.zeros((), device=device)

            # 5. vel bypass (per-frame)
            if args.l_vel > 0:
                d2v = ((out["v_pred"] - cmd) ** 2).sum(dim=-1) * mask
                l_vel = d2v.sum() / mask.sum().clamp(min=1.0)
            else:
                l_vel = torch.zeros((), device=device)

            loss = (args.l_seg_rnc * l_rnc
                    + args.l_recon_phase * l_recon_phase
                    + args.l_recon_foot * l_recon_foot
                    + args.l_var * l_var
                    + args.l_cov * l_cov
                    + args.l_slow * l_slow
                    + args.l_vel * l_vel)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep["rnc"] += float(l_rnc); ep["recon_phase"] += float(l_recon_phase)
            ep["recon_foot"] += float(l_recon_foot); ep["var"] += float(l_var)
            ep["cov"] += float(l_cov); ep["slow"] += float(l_slow)
            ep["vel"] += float(l_vel); ep["total"] += float(loss)
            nb += 1
        for k in ep:
            ep[k] /= max(nb, 1)
        ep["epoch"] = epoch
        ep["elapsed"] = time.time() - t0
        line = json.dumps(ep)
        print(line)
        log_f.write(line + "\n")
        log_f.flush()
    log_f.close()

    save_path = os.path.join(args.out_dir, "encoder.pt")
    torch.save({
        "model": model.state_dict(),
        "config": vars(args),
        "in_dim": in_dim,
        "phase_dim": phase_dim,
        "foot_dim": foot_dim,
        "encoder_kind": "segment",
    }, save_path)
    print(f"[seg] saved {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64,
                   help="segments per batch (RnC is O(B^3) memory).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d_back", type=int, default=128)
    p.add_argument("--d_gait", type=int, default=32)
    p.add_argument("--segment_min_len", type=int, default=32)
    p.add_argument("--segment_max_len", type=int, default=64)
    p.add_argument("--hard_seg_mean", type=int, default=1,
                   help="1: decoder always uses segment-mean z; 0: decoder uses e_t and L_slow penalises drift.")
    p.add_argument("--phase_anchor_joint_idx", type=int, default=0)
    p.add_argument("--phase_joint_slice", type=str, default="245:260")
    p.add_argument("--mask_obs_indices", type=str, default=None)
    p.add_argument("--r_leg", type=float, default=0.3)
    # loss weights
    p.add_argument("--l_seg_rnc", type=float, default=1.0)
    p.add_argument("--l_recon_phase", type=float, default=1.0)
    p.add_argument("--l_recon_foot", type=float, default=0.5)
    p.add_argument("--l_var", type=float, default=1.0)
    p.add_argument("--l_cov", type=float, default=0.1)
    p.add_argument("--l_slow", type=float, default=0.1)
    p.add_argument("--l_vel", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
