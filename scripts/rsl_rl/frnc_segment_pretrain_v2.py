# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""V2 segment-level encoder pretraining.

Adds on top of V1 (frnc_segment_pretrain.py):
  * --mask_kind {none,cmd,cmd_tokens,strict}
        what subset of the per-frame obs to zero out.
  * --rnc_label_kind {cmd,gait_props,both}
        what to use as the segment-level RnC distance label.
  * --l_gait_prop > 0
        regression head: z_gait -> per-segment gait features (step_freq, duty,
        bilat_phase cos/sin, knee amp, lat_sway, waist std).
  * --l_adv_cmd > 0
        adversarial: z_gait -> cmd via a cmd-predictor + gradient-reversal,
        forcing z_gait NOT to encode cmd.

The encoder/decoder shape is identical to V1 to keep results comparable.
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from frnc_pretrain import _parse_index_list
from frnc_segment_pretrain import (
    SegmentDataset, collate_segments, SegmentEncoder,
    rnc_loss_seg, vicreg_var, vicreg_cov, slow_loss,
)
from frnc_gait_features import (
    build_mask_indices, gait_features_segment, gait_features_to_array,
    GAIT_FEATURE_NAMES,
)

# Per-feature supervision weight for prop_head, derived from input-side
# diagnostic ceiling (R2 input->feature on strict mask). step_freq R2=0.22
# is treated as no-signal and dropped; weak features (bilat_sin, waist_*) downweighted.
PROP_WEIGHTS = {
    "step_freq": 0.0,
    "duty_l": 1.0, "duty_r": 1.0,
    "bilat_cos": 1.0, "bilat_sin": 0.5,
    "step_amp_lk": 1.0, "step_amp_rk": 1.0,
    "lat_sway": 1.0,
    "waist_yaw_std": 0.5, "waist_pitch_std": 0.5,
    "ang_act": 0.5,
}
import torch as _torch
_PROP_W = _torch.tensor([PROP_WEIGHTS.get(n, 1.0) for n in GAIT_FEATURE_NAMES], dtype=_torch.float32)


# ---------------------- per-segment labels (compute once) ---------------------- #
def precompute_segment_labels(ds: SegmentDataset, fs: float = 50.0):
    """Returns dict with arrays of shape (N_segments, dim)."""
    N = len(ds)
    feats = np.zeros((N, len(GAIT_FEATURE_NAMES)), dtype=np.float32)
    feat_valid = np.ones((N, len(GAIT_FEATURE_NAMES)), dtype=np.float32)
    for i in range(N):
        seg = ds.segments[i]
        c = ds.shards[seg.shard_idx]
        jpos = c["policy_obs"][seg.rows, ds.lo:ds.hi]
        foot = c["foot"][seg.rows]
        d = gait_features_segment(jpos, foot, fs=fs)
        arr = gait_features_to_array(d)
        nan = np.isnan(arr)
        arr[nan] = 0.0
        feats[i] = arr
        feat_valid[i, nan] = 0.0
    # Standardize features by column std (ignoring zeros from invalid)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    feats_norm = (feats - mu) / sd
    return {
        "feats_raw": feats, "feats_norm": feats_norm.astype(np.float32),
        "valid": feat_valid, "mu": mu, "sd": sd,
    }


# ---------------------- gradient reversal ---------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return _GradReverse.apply(x, lam)


# ---------------------- extended encoder (adds prop_head + adv_head) ---------------------- #
class SegmentEncoderV2(SegmentEncoder):
    def __init__(self, in_dim, d_back=128, d_gait=32, phase_dim=30, foot_dim=2,
                 prop_dim=0, adv_cmd=False, hard_seg_mean=True):
        super().__init__(in_dim=in_dim, d_back=d_back, d_gait=d_gait,
                         phase_dim=phase_dim, foot_dim=foot_dim,
                         hard_seg_mean=hard_seg_mean)
        self.prop_dim = prop_dim
        if prop_dim > 0:
            self.prop_head = nn.Sequential(
                nn.Linear(d_gait, 64), nn.ELU(), nn.Linear(64, prop_dim),
            )
        self.adv_cmd = adv_cmd
        if adv_cmd:
            self.adv_head = nn.Sequential(
                nn.Linear(d_gait, 64), nn.ELU(), nn.Linear(64, 3),
            )

    def forward(self, obs, mask, anchor_sc, adv_lambda: float = 0.0):
        out = super().forward(obs, mask, anchor_sc)
        z = out["z_gait"]
        if self.prop_dim > 0:
            out["prop_pred"] = self.prop_head(z)
        if self.adv_cmd:
            out["adv_cmd_pred"] = self.adv_head(grad_reverse(z, adv_lambda))
        return out


# ---------------------- training ---------------------- #
def train(args):
    device = torch.device(args.device)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    assert shards, f"no shards in {args.data_dir}"

    mask_spec = build_mask_indices(args.mask_kind)
    mask_idx = _parse_index_list(mask_spec) if mask_spec else None
    n_mask = 0 if mask_idx is None else len(mask_idx)
    print(f"[v2] mask_kind={args.mask_kind} masks {n_mask} obs dims")

    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))
    ds = SegmentDataset(
        shards, mask_obs_indices=mask_idx,
        phase_joint_slice=phase_slice,
        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
        segment_min_len=args.segment_min_len,
        segment_max_len=args.segment_max_len,
    )
    print(f"[v2] {len(ds)} segments")
    sample = ds[0]
    in_dim = sample["obs"].shape[1]
    phase_dim = sample["phase"].shape[1]
    foot_dim = sample["foot"].shape[1]

    # gait features per segment
    use_gait_props = args.l_gait_prop > 0 or args.rnc_label_kind in ("gait_props", "both")
    if use_gait_props:
        print("[v2] precomputing per-segment gait features ...")
        seg_lab = precompute_segment_labels(ds)
        feats_norm = torch.from_numpy(seg_lab["feats_norm"])
        feats_valid = torch.from_numpy(seg_lab["valid"])
        prop_dim = feats_norm.shape[1]
        print(f"[v2] gait features: {GAIT_FEATURE_NAMES}")
    else:
        feats_norm, feats_valid, prop_dim = None, None, 0

    model = SegmentEncoderV2(
        in_dim=in_dim, d_back=args.d_back, d_gait=args.d_gait,
        phase_dim=phase_dim, foot_dim=foot_dim,
        prop_dim=prop_dim if args.l_gait_prop > 0 else 0,
        adv_cmd=(args.l_adv_cmd > 0),
        hard_seg_mean=bool(args.hard_seg_mean),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # need segment indices in batch -> use a custom sampler that returns (i, item)
    class _IndexedDS(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            it = self.base[i]; it["seg_idx"] = i; return it
    def _collate_idx(batch):
        idx = torch.tensor([b.pop("seg_idx") for b in batch], dtype=torch.long)
        out = collate_segments(batch); out["seg_idx"] = idx; return out
    loader = DataLoader(_IndexedDS(ds), batch_size=args.batch_size,
                        shuffle=True, num_workers=2, drop_last=True,
                        collate_fn=_collate_idx, pin_memory=True)

    W = torch.tensor([1.0, 1.0, args.r_leg], device=device)
    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, "train.log"), "w")

    t0 = time.time()
    for epoch in range(args.epochs):
        # ramp adv lambda from 0 to args.l_adv_cmd over first 5 epochs
        adv_lam = float(args.l_adv_cmd) * min(1.0, (epoch + 1) / 5.0) if args.l_adv_cmd > 0 else 0.0
        ep = {k: 0.0 for k in [
            "rnc_cmd", "rnc_props", "recon_phase", "recon_foot",
            "var", "cov", "slow", "vel", "prop", "adv_cmd", "total",
        ]}
        nb = 0
        for batch in loader:
            obs = batch["obs"].to(device, non_blocking=True)
            cmd = batch["cmd"].to(device, non_blocking=True)
            foot = batch["foot"].to(device, non_blocking=True)
            phase = batch["phase"].to(device, non_blocking=True)
            anc = batch["anchor_sc"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            seg_idx = batch["seg_idx"]

            out = model(obs, mask, anc, adv_lambda=adv_lam)

            cmd_seg = (cmd * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

            # ---- RnC label sources ---- #
            l_rnc_cmd = torch.zeros((), device=device)
            l_rnc_props = torch.zeros((), device=device)
            if args.l_seg_rnc > 0 and args.rnc_label_kind in ("cmd", "both"):
                with torch.no_grad():
                    cmd_w = cmd_seg * W.unsqueeze(0)
                    delta_c = torch.cdist(cmd_w, cmd_w, p=2)
                l_rnc_cmd = rnc_loss_seg(out["z_gait"], delta_c, args.tau)
            if args.l_seg_rnc > 0 and args.rnc_label_kind in ("gait_props", "both"):
                feats_b = feats_norm[seg_idx].to(device)
                valid_b = feats_valid[seg_idx].to(device)
                # mask invalid columns by zeroing; cdist ignores nothing so we
                # just rely on valid==0 -> column constant -> contributes 0.
                feats_b = feats_b * valid_b
                with torch.no_grad():
                    delta_p = torch.cdist(feats_b, feats_b, p=2)
                l_rnc_props = rnc_loss_seg(out["z_gait"], delta_p, args.tau)

            # ---- reconstruction ---- #
            valid_anc = ~torch.isnan(anc).any(dim=-1)
            valid_phase = ~torch.isnan(phase).any(dim=-1)
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
                bce = F.binary_cross_entropy_with_logits(
                    out["foot_pred"], foot, reduction="none").sum(dim=-1)
                l_recon_foot = (bce * vf).sum() / vf.sum().clamp(min=1.0)
            else:
                l_recon_foot = torch.zeros((), device=device)

            # VICReg
            l_var = vicreg_var(out["z_gait"]) if args.l_var > 0 else torch.zeros((), device=device)
            l_cov = vicreg_cov(out["z_gait"]) if args.l_cov > 0 else torch.zeros((), device=device)

            # slow (if soft)
            if (not bool(args.hard_seg_mean)) and args.l_slow > 0:
                l_slow = slow_loss(out["e"], out["z_gait"], mask)
            else:
                l_slow = torch.zeros((), device=device)

            # vel head
            if args.l_vel > 0:
                d2v = ((out["v_pred"] - cmd) ** 2).sum(dim=-1) * mask
                l_vel = d2v.sum() / mask.sum().clamp(min=1.0)
            else:
                l_vel = torch.zeros((), device=device)

            # gait property regression head (z_gait -> normalized features)
            if args.l_gait_prop > 0 and "prop_pred" in out:
                feats_b = feats_norm[seg_idx].to(device)
                valid_b = feats_valid[seg_idx].to(device) * _PROP_W.to(device)
                d2p = ((out["prop_pred"] - feats_b) ** 2) * valid_b
                l_prop = d2p.sum() / valid_b.sum().clamp(min=1.0)
            else:
                l_prop = torch.zeros((), device=device)

            # adversarial cmd
            if args.l_adv_cmd > 0 and "adv_cmd_pred" in out:
                # MSE on cmd_seg; gradient through GradReverse already negates
                l_adv = ((out["adv_cmd_pred"] - cmd_seg) ** 2).mean()
            else:
                l_adv = torch.zeros((), device=device)

            loss = (args.l_seg_rnc * (l_rnc_cmd + l_rnc_props)
                    + args.l_recon_phase * l_recon_phase
                    + args.l_recon_foot * l_recon_foot
                    + args.l_var * l_var
                    + args.l_cov * l_cov
                    + args.l_slow * l_slow
                    + args.l_vel * l_vel
                    + args.l_gait_prop * l_prop
                    + l_adv)  # adv weight already encoded in adv_lam ramp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep["rnc_cmd"] += float(l_rnc_cmd); ep["rnc_props"] += float(l_rnc_props)
            ep["recon_phase"] += float(l_recon_phase); ep["recon_foot"] += float(l_recon_foot)
            ep["var"] += float(l_var); ep["cov"] += float(l_cov)
            ep["slow"] += float(l_slow); ep["vel"] += float(l_vel)
            ep["prop"] += float(l_prop); ep["adv_cmd"] += float(l_adv)
            ep["total"] += float(loss)
            nb += 1
        for k in ep: ep[k] /= max(nb, 1)
        ep["epoch"] = epoch; ep["adv_lam"] = adv_lam
        ep["elapsed"] = time.time() - t0
        line = json.dumps(ep)
        print(line); log_f.write(line + "\n"); log_f.flush()
    log_f.close()

    save = os.path.join(args.out_dir, "encoder.pt")
    torch.save({
        "model": model.state_dict(),
        "config": vars(args),
        "in_dim": in_dim,
        "phase_dim": phase_dim,
        "foot_dim": foot_dim,
        "prop_dim": prop_dim if args.l_gait_prop > 0 else 0,
        "adv_cmd": bool(args.l_adv_cmd > 0),
        "encoder_kind": "segment_v2",
        "mask_spec": mask_spec,
        "gait_feature_names": GAIT_FEATURE_NAMES,
    }, save)
    print(f"[v2] saved {save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d_back", type=int, default=128)
    p.add_argument("--d_gait", type=int, default=32)
    p.add_argument("--segment_min_len", type=int, default=32)
    p.add_argument("--segment_max_len", type=int, default=64)
    p.add_argument("--hard_seg_mean", type=int, default=1)
    p.add_argument("--phase_anchor_joint_idx", type=int, default=0)
    p.add_argument("--phase_joint_slice", type=str, default="245:260")
    p.add_argument("--mask_kind", type=str, default="cmd",
                   choices=["none", "cmd", "cmd_tokens", "strict"])
    p.add_argument("--rnc_label_kind", type=str, default="cmd",
                   choices=["cmd", "gait_props", "both"])
    p.add_argument("--r_leg", type=float, default=0.3)
    # weights
    p.add_argument("--l_seg_rnc", type=float, default=1.0)
    p.add_argument("--l_recon_phase", type=float, default=1.0)
    p.add_argument("--l_recon_foot", type=float, default=0.5)
    p.add_argument("--l_var", type=float, default=1.0)
    p.add_argument("--l_cov", type=float, default=0.1)
    p.add_argument("--l_slow", type=float, default=0.1)
    p.add_argument("--l_vel", type=float, default=0.5)
    p.add_argument("--l_gait_prop", type=float, default=0.0)
    p.add_argument("--l_adv_cmd", type=float, default=0.0,
                   help="adversarial weight; ramps from 0 over 5 epochs.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
