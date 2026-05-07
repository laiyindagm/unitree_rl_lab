# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Offline Factorized Rank-N-Contrast + Metric-Calibration encoder pretraining.

Loads ``.npz`` shards produced by ``collect_pretrain_data.py``, trains a
backbone (matches TransformerLatentModel's history encoder) + 4 projection
heads (z^x, z^y, z^omega, z^g) with the FRnC-MC objective.

Outputs:
  * ``encoder.pt``   - state_dict of backbone + heads + calibration scalars
  * ``train.log``    - per-epoch losses + diagnostic metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------- data ---------------------- #
def _hilbert_global_phase_targets(joint_pos_seq: np.ndarray,
                                  anchor_idx: int = 0,
                                  min_len: int = 32,
                                  anchor_min_std: float = 1e-3) -> np.ndarray:
    """Per-joint relative phase (cos dphi, sin dphi) wrt anchor joint.

    phi_global(t) = angle(hilbert(detrended anchor)),
    dphi_j = phi_j - phi_global,
    output[:, 2j]   = cos(dphi_j)
    output[:, 2j+1] = sin(dphi_j)

    Returns NaN where the segment is too short or the anchor is near-stationary
    (e.g. standing): in that case phi_global is undefined.
    """
    from scipy.signal import hilbert
    T, J = joint_pos_seq.shape
    out = np.full((T, 2 * J), np.nan, dtype=np.float32)
    if T < min_len:
        return out
    x = joint_pos_seq - joint_pos_seq.mean(axis=0, keepdims=True)
    if float(np.std(x[:, anchor_idx])) < anchor_min_std:
        return out  # standing / no swing -> phase undefined
    try:
        anchor_phi = np.angle(hilbert(x[:, anchor_idx]))
    except Exception:
        return out
    for j in range(J):
        try:
            phi_j = np.angle(hilbert(x[:, j]))
        except Exception:
            continue
        d = phi_j - anchor_phi
        out[:, 2 * j]     = np.cos(d).astype(np.float32)
        out[:, 2 * j + 1] = np.sin(d).astype(np.float32)
    edge = max(4, T // 10)
    out[:edge] = np.nan
    out[-edge:] = np.nan
    return out
    # detrend per joint (subtract mean) so analytic signal has well-defined phase
    x = joint_pos_seq - joint_pos_seq.mean(axis=0, keepdims=True)
    # apply hilbert column-wise
    for j in range(J):
        try:
            analytic = hilbert(x[:, j])
        except Exception:
            continue
        phi = np.angle(analytic)
        out[:, 2 * j] = np.sin(phi).astype(np.float32)
        out[:, 2 * j + 1] = np.cos(phi).astype(np.float32)
    # drop edges where Hilbert is unreliable (~10% on each side)
    edge = max(4, T // 10)
    out[:edge] = np.nan
    out[-edge:] = np.nan
    return out


class ShardDataset(Dataset):
    def __init__(self, shard_paths, in_memory: bool = True,
                 mask_obs_indices=None,
                 use_phase_aux: bool = False,
                 phase_joint_slice=(245, 260),
                 phase_min_segment: int = 32,
                 phase_anchor_joint_idx: int = 0):
        self.paths = shard_paths
        self.in_memory = in_memory
        self.use_phase_aux = use_phase_aux
        self.mask_obs_indices = mask_obs_indices
        self.phase_anchor_joint_idx = int(phase_anchor_joint_idx)
        self.cache = []
        if not in_memory:
            raise NotImplementedError("only in-memory mode is supported")

        for p in shard_paths:
            d = np.load(p)
            entry = {
                "policy_obs": d["policy_obs"].astype(np.float32),
                "cmd": d["cmd"].astype(np.float32),
                "actual_lin_vel": d["actual_lin_vel"].astype(np.float32),
                "actual_ang_vel": d["actual_ang_vel"].astype(np.float32),
                "env_id": d["env_id"].astype(np.int64),
                "episode_step": d["episode_step"].astype(np.int64),
                "global_step": d["global_step"].astype(np.int64),
            }
            if use_phase_aux:
                lo, hi = phase_joint_slice
                jpos_all = entry["policy_obs"][:, lo:hi].astype(np.float32)
                phase = np.full((entry["policy_obs"].shape[0], 2 * (hi - lo)),
                                np.nan, dtype=np.float32)
                # group by env_id, walk by gstep, segment on episode_step==0 reset
                envs = entry["env_id"]
                gsteps = entry["global_step"]
                esteps = entry["episode_step"]
                for eid in np.unique(envs):
                    mask = envs == eid
                    rows = np.where(mask)[0]
                    order = rows[np.argsort(gsteps[rows])]
                    es = esteps[order]
                    # split segments where episode_step resets to 0
                    starts = np.concatenate([[0], np.where(es == 0)[0]])
                    starts = np.unique(starts)
                    ends = np.concatenate([starts[1:], [len(order)]])
                    for s_, e_ in zip(starts, ends):
                        if e_ - s_ < phase_min_segment:
                            continue
                        seg_rows = order[s_:e_]
                        seg = jpos_all[seg_rows]
                        ph = _hilbert_global_phase_targets(seg, anchor_idx=self.phase_anchor_joint_idx, min_len=phase_min_segment)
                        phase[seg_rows] = ph
                entry["phase"] = phase
            self.cache.append(entry)

        self.lengths = [c["cmd"].shape[0] for c in self.cache]
        self.cumlen = np.cumsum([0] + self.lengths)
        self.total = int(self.cumlen[-1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        s = int(np.searchsorted(self.cumlen, idx, side="right") - 1)
        local = idx - int(self.cumlen[s])
        c = self.cache[s]
        obs = c["policy_obs"][local]
        if self.mask_obs_indices is not None:
            obs = obs.copy()
            obs[self.mask_obs_indices] = 0.0
        item = {
            "obs": obs,
            "cmd": c["cmd"][local],
        }
        if self.use_phase_aux:
            item["phase"] = c["phase"][local]
        return item


# ---------------------- model ---------------------- #
class FRnCEncoder(nn.Module):
    def __init__(self, in_dim: int, d_back: int = 128, d_axis: int = 16,
                 phase_dim: int = 0, d_gait: int = 32,
                 use_hierarchical: bool = True):
        """Hierarchical bottleneck encoder.

        backbone: obs -> f (d_back)
        if use_hierarchical: gait_proj: f -> z_gait (d_gait), and ALL heads
            (x,y,w,g, vel, phase) take z_gait as input. vel_head is a single
            Linear so the only path obs -> cmd is obs -> f -> z_gait -> v.
        else: legacy mode, all heads take f.
        """
        super().__init__()
        self.use_hierarchical = use_hierarchical
        self.d_gait = d_gait
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, d_back), nn.ELU(),
        )
        if use_hierarchical:
            self.gait_proj = nn.Sequential(
                nn.Linear(d_back, d_gait), nn.ELU(),
            )
            head_in = d_gait
        else:
            self.gait_proj = nn.Identity()
            head_in = d_back

        def head():
            return nn.Sequential(
                nn.Linear(head_in, 64), nn.ELU(), nn.Linear(64, d_axis),
            )
        self.head_x = head()
        self.head_y = head()
        self.head_w = head()
        self.head_g = head()
        # vel predictor: single Linear in hierarchical mode (no extra capacity)
        if use_hierarchical:
            self.vel_head = nn.Linear(head_in, 3)
        else:
            self.vel_head = nn.Linear(d_back, 3)
        self.phase_dim = phase_dim
        if phase_dim > 0:
            self.phase_head = nn.Sequential(
                nn.Linear(head_in, 128), nn.ELU(), nn.Linear(128, phase_dim),
            )
        self.beta_raw = nn.Parameter(torch.zeros(3))

    @property
    def beta(self):
        return F.softplus(self.beta_raw) + 1e-3

    def forward(self, obs):
        f = self.backbone(obs)
        z_gait = self.gait_proj(f)
        zx = self.head_x(z_gait)
        zy = self.head_y(z_gait)
        zw = self.head_w(z_gait)
        zg = self.head_g(z_gait)
        v_pred = self.vel_head(z_gait)
        out = {"f": f, "z_gait": z_gait, "zx": zx, "zy": zy, "zw": zw,
               "zg": zg, "v_pred": v_pred}
        if self.phase_dim > 0:
            out["phase_pred"] = self.phase_head(z_gait)
        return out


# ---------------------- losses ---------------------- #
def rnc_loss(z: torch.Tensor, delta: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Rank-N-Contrast (Zha et al. NeurIPS'23).

    z: (B, d) latent for one block. delta: (B, B) pairwise label distance.
    Returns scalar loss.
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / tau                                  # (B, B)
    B = z.size(0)
    diag = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag, -1e4)

    # for each anchor i and each candidate j, denominator is sum over k with
    # delta_ik >= delta_ij (the ones at least as far). Use a vectorized form:
    # sort each row of delta in ascending order; for sorted index r,
    # the denominator is logsumexp over indices with sorted position >= r.
    # Equivalently: build indicator M_{ijk} = 1[delta_ik >= delta_ij].
    # Memory O(B^3); we limit B (batch size) to ~256.
    delta_no_diag = delta.clone()
    delta_no_diag.masked_fill_(diag, -1.0)
    # M: (B, B, B): M[i,j,k] = 1 if delta[i,k] >= delta[i,j]
    M = (delta.unsqueeze(1) >= delta.unsqueeze(2)).float()  # (B, B, B)
    # zero out k=i (excluded as anchor)
    M = M * (~diag).unsqueeze(1).float()
    # logsumexp(sim[i,k]) over allowed k for each (i,j)
    # mask sim with -inf where M=0
    sim_exp = sim.unsqueeze(1).expand(-1, B, -1)             # (B, B, B)
    masked = sim_exp.masked_fill(M == 0, -1e4)
    logZ = torch.logsumexp(masked, dim=-1)                   # (B, B)
    log_p = sim - logZ                                       # (B, B)
    # exclude anchor==j
    log_p = log_p.masked_fill(diag, 0.0)
    # average over (i,j), j != i
    return -(log_p.sum() / (B * (B - 1)))


def calibration_loss(z: torch.Tensor, delta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """L_cal = E[(||z_i-z_j|| - beta * delta_ij)^2]. z is unnormalized."""
    B = z.size(0)
    dz = torch.cdist(z, z, p=2)                              # (B, B)
    diff = (dz - beta * delta) ** 2
    diag = torch.eye(B, dtype=torch.bool, device=z.device)
    diff = diff.masked_fill(diag, 0.0)
    return diff.sum() / (B * (B - 1))


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance regularizer."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


# ---------------------- training ---------------------- #
def _parse_index_list(spec: str):
    """Parse "6:9,64:67,122:125" -> sorted list of indices."""
    if spec is None or spec.strip() == "":
        return None
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            a, b = chunk.split(":")
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(chunk))
    return np.array(sorted(set(out)), dtype=np.int64)


def train(args):
    device = torch.device(args.device)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    assert shards, f"no shards in {args.data_dir}"
    print(f"[frnc] {len(shards)} shards")
    mask_idx = _parse_index_list(args.mask_obs_indices)
    if mask_idx is not None:
        print(f"[frnc] masking {len(mask_idx)} obs dims (cmd shortcut block)")
    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))
    use_phase = args.l_phase > 0
    ds = ShardDataset(
        shards, in_memory=True,
        mask_obs_indices=mask_idx,
        use_phase_aux=use_phase,
        phase_joint_slice=phase_slice,
        phase_min_segment=args.phase_min_segment,
        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
    )
    print(f"[frnc] total samples: {len(ds)} (use_phase_aux={use_phase})")

    sample = ds[0]
    in_dim = sample["obs"].shape[0]
    phase_dim = sample["phase"].shape[0] if use_phase else 0
    print(f"[frnc] obs in_dim = {in_dim}  phase_dim = {phase_dim}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, drop_last=True, pin_memory=True,
    )

    model = FRnCEncoder(in_dim=in_dim, d_back=args.d_back, d_axis=args.d_axis,
                        phase_dim=phase_dim, d_gait=args.d_gait,
                        use_hierarchical=args.use_hierarchical).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    W = torch.tensor([1.0, 1.0, args.r_leg], device=device)  # weight for global metric

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log_f = open(log_path, "w")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_stats = {k: 0.0 for k in [
            "rnc_x", "rnc_y", "rnc_w", "rnc_g", "cal_x", "cal_y", "cal_w",
            "vel", "var", "phase", "total",
        ]}
        n_bat = 0
        for batch in loader:
            obs = batch["obs"].to(device, non_blocking=True)
            cmd = batch["cmd"].to(device, non_blocking=True)
            out = model(obs)
            B = obs.size(0)

            # pairwise label distances
            with torch.no_grad():
                delta_x = (cmd[:, 0:1] - cmd[:, 0:1].t()).abs()
                delta_y = (cmd[:, 1:2] - cmd[:, 1:2].t()).abs()
                delta_w = (cmd[:, 2:3] - cmd[:, 2:3].t()).abs()
                cmd_w = cmd * W.unsqueeze(0)
                delta_g = torch.cdist(cmd_w, cmd_w, p=2)

            l_rnc_x = rnc_loss(out["zx"], delta_x, args.tau)
            l_rnc_y = rnc_loss(out["zy"], delta_y, args.tau)
            l_rnc_w = rnc_loss(out["zw"], delta_w, args.tau)
            l_rnc_g = rnc_loss(out["zg"], delta_g, args.tau)

            beta = model.beta
            l_cal_x = calibration_loss(out["zx"], delta_x, beta[0])
            l_cal_y = calibration_loss(out["zy"], delta_y, beta[1])
            l_cal_w = calibration_loss(out["zw"], delta_w, beta[2])

            l_vel = F.mse_loss(out["v_pred"], cmd)

            l_var = (
                variance_loss(out["zx"]) + variance_loss(out["zy"])
                + variance_loss(out["zw"]) + variance_loss(out["zg"])
            )

            l_phase = torch.zeros((), device=device)
            if use_phase:
                phase_t = batch["phase"].to(device, non_blocking=True)
                valid = ~torch.isnan(phase_t)
                if valid.any():
                    pred = out["phase_pred"]
                    diff = (pred - torch.nan_to_num(phase_t, nan=0.0)) ** 2
                    l_phase = (diff * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

            loss = (
                args.l_rnc * (l_rnc_x + l_rnc_y + l_rnc_w)
                + args.l_g * l_rnc_g
                + args.l_cal * (l_cal_x + l_cal_y + l_cal_w)
                + args.l_vel * l_vel
                + args.l_var * l_var
                + args.l_phase * l_phase
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_stats["rnc_x"] += float(l_rnc_x); ep_stats["rnc_y"] += float(l_rnc_y)
            ep_stats["rnc_w"] += float(l_rnc_w); ep_stats["rnc_g"] += float(l_rnc_g)
            ep_stats["cal_x"] += float(l_cal_x); ep_stats["cal_y"] += float(l_cal_y)
            ep_stats["cal_w"] += float(l_cal_w)
            ep_stats["vel"] += float(l_vel); ep_stats["var"] += float(l_var)
            ep_stats["phase"] += float(l_phase); ep_stats["total"] += float(loss)
            n_bat += 1
            step += 1

        for k in ep_stats:
            ep_stats[k] /= max(n_bat, 1)
        ep_stats["epoch"] = epoch
        ep_stats["beta_x"] = float(model.beta[0])
        ep_stats["beta_y"] = float(model.beta[1])
        ep_stats["beta_w"] = float(model.beta[2])
        ep_stats["elapsed"] = time.time() - t0
        line = json.dumps(ep_stats)
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log_f.close()
    save_path = os.path.join(args.out_dir, "encoder.pt")
    torch.save({
        "model": model.state_dict(),
        "config": vars(args),
        "in_dim": in_dim,
    }, save_path)
    print(f"[frnc] saved {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d_back", type=int, default=128)
    p.add_argument("--d_axis", type=int, default=16)
    p.add_argument("--d_gait", type=int, default=32,
                   help="bottleneck dim for hierarchical mode.")
    p.add_argument("--use_hierarchical", type=int, default=1,
                   help="1=enable hierarchical bottleneck (heads take z_gait); 0=legacy.")
    p.add_argument("--phase_anchor_joint_idx", type=int, default=0,
                   help="index into joint_pos_rel slice used as global phase anchor (verify w/ data).")
    p.add_argument("--r_leg", type=float, default=0.3)
    p.add_argument("--l_rnc", type=float, default=1.0)
    p.add_argument("--l_g", type=float, default=0.5)
    p.add_argument("--l_cal", type=float, default=0.5)
    p.add_argument("--l_vel", type=float, default=1.0)
    p.add_argument("--l_var", type=float, default=0.1)
    p.add_argument("--l_phase", type=float, default=0.0,
                   help=">0 enables Hilbert-phase aux head supervision.")
    p.add_argument("--phase_joint_slice", type=str, default="245:260",
                   help="slice into policy_obs that holds joint_pos_rel (last frame).")
    p.add_argument("--phase_min_segment", type=int, default=32)
    p.add_argument("--mask_obs_indices", type=str, default=None,
                   help="comma-separated indices/ranges to zero out (e.g. for cmd shortcut: '6:9,64:67,122:125,180:183,238:241').")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.use_hierarchical = bool(args.use_hierarchical)
    train(args)


if __name__ == "__main__":
    main()
