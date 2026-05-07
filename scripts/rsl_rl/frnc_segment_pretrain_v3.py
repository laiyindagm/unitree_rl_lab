"""V3 segment encoder pretraining.

Replaces V2's categorical bucket-RnC + adv with a metric-learning formulation
that mirrors the math regularization on the latent gait space G:

    (A1) axial additivity:
        g*(v) ≈ rho_x · g_x ⊕ rho_y · g_y ⊕ rho_w · g_w,
        rho_a = v_a / ||v||_W
    (A2) magnitude Lipschitz lower bound:
        d_Z(phi(λ1·v̂), phi(λ2·v̂)) >= L · |λ1 - λ2|

Loss menu (vs V2):
  * L_metric_rnc : Rank-N-Contrast on cmd-space distance (continuous label),
                   replaces V2's categorical SupCon.
  * L_axial      : ||z - sum_a rho_a · E_a||^2 with 3 learnable axial bases E_a.
  * L_lip        : hinge max(0, L · |λ1-λ2| - ||z1-z2||)^2 on same-direction pairs.
  * L_prop       : kept (per-feature direct supervision, weighted by input ceiling).
  * L_recon_*    : kept (phase + foot reconstruction).
  * L_var/cov    : kept (VICReg anti-collapse).
  * adv_cmd      : DROPPED (V2 ablation showed loss→0 yet R²(z→cmd)=0.85).
"""
from __future__ import annotations
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from frnc_segment_pretrain import (
    SegmentDataset, collate_segments, SegmentEncoder,
    rnc_loss_seg, vicreg_var, vicreg_cov, slow_loss,
)
from frnc_gait_features import (
    build_mask_indices, gait_features_segment, gait_features_to_array,
    GAIT_FEATURE_NAMES,
)
from frnc_pretrain import _parse_index_list
from frnc_segment_pretrain_v2 import (
    PROP_WEIGHTS, _PROP_W, precompute_segment_labels,
)

# Copied from V2 train() (defined nested there). Module-level so V3 can use.
class _IndexedDS(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]; item['seg_idx'] = idx; return item

def _collate_idx(batch):
    seg_idx = torch.tensor([b.pop('seg_idx') for b in batch], dtype=torch.long)
    out = collate_segments(batch); out['seg_idx'] = seg_idx; return out


# ---------------------- V3 encoder ---------------------- #
class SegmentEncoderV3(SegmentEncoder):
    """V2 backbone + 3 axial bases (A1) + prop_head."""

    def __init__(self, in_dim, d_back=128, d_gait=32, phase_dim=30, foot_dim=2,
                 prop_dim=0, hard_seg_mean=True):
        super().__init__(in_dim=in_dim, d_back=d_back, d_gait=d_gait,
                         phase_dim=phase_dim, foot_dim=foot_dim,
                         hard_seg_mean=hard_seg_mean)
        self.prop_dim = prop_dim
        if prop_dim > 0:
            self.prop_head = nn.Sequential(
                nn.Linear(d_gait, 64), nn.ELU(), nn.Linear(64, prop_dim),
            )
        # 3 axial bases E_x, E_y, E_omega in z space (A1)
        self.axial_bases = nn.Parameter(torch.zeros(3, d_gait))
        nn.init.normal_(self.axial_bases, std=1.0 / math.sqrt(d_gait))

    def axial_predict(self, cmd, sigma):
        """g*(v) = sum_a rho_a · E_a, rho_a = v_a / ||v||_W with W=diag(1/sigma_a^2).
        cmd: (B, 3); sigma: (3,) float tensor (per-axis std).
        Returns (B, d_gait).
        """
        v_w = cmd / sigma.unsqueeze(0).clamp(min=1e-3)            # (B, 3)
        norm = v_w.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        rho = v_w / norm                                           # (B, 3)
        return rho @ self.axial_bases                              # (B, d_gait)

    def forward(self, obs, mask, anchor_sc):
        out = super().forward(obs, mask, anchor_sc)
        if self.prop_dim > 0:
            out["prop_pred"] = self.prop_head(out["z_gait"])
        return out


# ---------------------- losses ---------------------- #
def cmd_metric_delta(cmd_seg: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Pairwise W-weighted distance ||v_i - v_j||_W in cmd space.
    cmd_seg: (B, 3); sigma: (3,). Returns (B, B)."""
    v_w = cmd_seg / sigma.unsqueeze(0).clamp(min=1e-3)
    diff = v_w.unsqueeze(1) - v_w.unsqueeze(0)                     # (B, B, 3)
    return diff.norm(dim=-1)


def lipschitz_pair_loss(z, cmd, sigma, L=1.0,
                        cos_thresh=0.95, mag_diff_min=0.2,
                        max_pairs=512):
    """Hinge: max(0, L * |λ1 - λ2| - ||z1 - z2||)^2 on same-direction pairs.
    Sample pairs from batch where cosine(v_i, v_j) > cos_thresh (after norming)
    and |λ1 - λ2| > mag_diff_min.
    """
    v_w = cmd / sigma.unsqueeze(0).clamp(min=1e-3)
    lam = v_w.norm(dim=-1)                                         # (B,)
    valid = lam > 1e-3
    if valid.sum() < 2:
        return z.new_zeros(())
    v_dir = v_w / lam.clamp(min=1e-3).unsqueeze(-1)
    cos = v_dir @ v_dir.t()                                        # (B, B)
    dlam = (lam.unsqueeze(0) - lam.unsqueeze(1)).abs()             # (B, B)
    mask = (cos > cos_thresh) & (dlam > mag_diff_min)
    mask = mask & valid.unsqueeze(0) & valid.unsqueeze(1)
    mask.fill_diagonal_(False)
    idx = mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return z.new_zeros(())
    if idx.size(0) > max_pairs:
        idx = idx[torch.randperm(idx.size(0), device=idx.device)[:max_pairs]]
    i, j = idx[:, 0], idx[:, 1]
    z1, z2 = z[i], z[j]
    dz = (z1 - z2).norm(dim=-1)
    margin = L * (lam[i] - lam[j]).abs()
    hinge = F.relu(margin - dz) ** 2
    return hinge.mean()


# ---------------------- training ---------------------- #
def train(args):
    device = torch.device(args.device)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    assert shards, f"no shards in {args.data_dir}"

    mask_spec = build_mask_indices(args.mask_kind)
    mask_idx = _parse_index_list(mask_spec) if mask_spec else None
    n_mask = 0 if mask_idx is None else len(mask_idx)
    print(f"[v3] mask_kind={args.mask_kind} masks {n_mask} obs dims")

    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))
    ds = SegmentDataset(
        shards, mask_obs_indices=mask_idx,
        phase_joint_slice=phase_slice,
        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
        segment_min_len=args.segment_min_len,
        segment_max_len=args.segment_max_len,
    )
    print(f"[v3] segments: {len(ds)}")

    feats_norm_t = feats_valid_t = None
    if args.l_prop > 0:
        labels = precompute_segment_labels(ds)
        feats_norm_t = torch.from_numpy(labels["feats_norm"]).float()
        feats_valid_t = torch.from_numpy(labels["valid"]).float()
        prop_dim = feats_norm_t.shape[1]
        print(f"[v3] gait features ({prop_dim}d): {GAIT_FEATURE_NAMES}")
    else:
        prop_dim = 0

    iod = _IndexedDS(ds)
    loader = DataLoader(iod, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=_collate_idx,
                        pin_memory=True, drop_last=True)

    obs_dim = ds[0]["obs"].shape[-1]
    model = SegmentEncoderV3(in_dim=obs_dim, d_back=args.d_back, d_gait=args.d_gait,
                             phase_dim=args.phase_dim, foot_dim=2,
                             prop_dim=prop_dim, hard_seg_mean=bool(args.hard_seg_mean))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # estimate per-axis cmd std from a sweep over the dataset (use first ~32 batches)
    cmds_for_sigma = []
    with torch.no_grad():
        for k, b in enumerate(loader):
            cmd = b["cmd"].to(device)                          # (B, T, 3)
            mask = b["mask"].to(device)                        # (B, T)
            cmd_seg = (cmd * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            cmds_for_sigma.append(cmd_seg.cpu())
            if k >= 31:
                break
    sigma = torch.stack(cmds_for_sigma).reshape(-1, 3).std(dim=0).clamp(min=0.05).to(device)
    print(f"[v3] per-axis cmd sigma = {sigma.tolist()}")

    log_path = os.path.join(args.out_dir, "train.log")
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(log_path, "w")
    t0 = time.time()

    for epoch in range(args.epochs):
        ep = {k: 0.0 for k in [
            "rnc_metric", "axial", "lip", "prop",
            "recon_phase", "recon_foot", "var", "cov", "slow", "vel", "total",
        ]}
        nb = 0
        for batch in loader:
            seg_idx = batch["seg_idx"]
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            anchor_sc = batch["anchor_sc"].to(device)
            cmd = batch["cmd"].to(device)
            phase_target = batch["phase"].to(device)
            foot_target = batch["foot"].to(device)

            cmd_seg = (cmd * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

            out = model(obs, mask, anchor_sc)
            z = out["z_gait"]

            # ----- L_metric_rnc: continuous-label RnC on cmd metric -----
            if args.l_rnc > 0:
                delta = cmd_metric_delta(cmd_seg, sigma)
                l_rnc = rnc_loss_seg(z, delta, args.tau)
            else:
                l_rnc = z.new_zeros(())

            # ----- L_axial: ||z - axial_predict(cmd)||^2  (A1) -----
            if args.l_axial > 0:
                z_pred = model.axial_predict(cmd_seg, sigma)
                l_axial = ((z - z_pred) ** 2).sum(dim=-1).mean()
            else:
                l_axial = z.new_zeros(())

            # ----- L_lip: same-direction Lipschitz lower bound (A2) -----
            if args.l_lip > 0:
                l_lip = lipschitz_pair_loss(z, cmd_seg, sigma,
                                            L=args.lip_L,
                                            cos_thresh=args.lip_cos_thresh,
                                            mag_diff_min=args.lip_mag_diff)
            else:
                l_lip = z.new_zeros(())

            # ----- L_recon (kept; mask NaNs in phase/anchor) -----
            valid_anc = ~torch.isnan(anchor_sc).any(dim=-1)
            valid_phase = ~torch.isnan(phase_target).any(dim=-1)
            valid_recon = mask.bool() & valid_anc & valid_phase
            if args.l_recon_phase > 0 and valid_recon.any():
                ph_t = torch.nan_to_num(phase_target, nan=0.0)
                d2 = ((out["phase_pred"] - ph_t) ** 2).sum(dim=-1)
                l_recon_phase = (d2 * valid_recon.float()).sum() / valid_recon.float().sum().clamp(min=1.0)
            else:
                l_recon_phase = z.new_zeros(())
            if args.l_recon_foot > 0 and (mask.bool() & valid_anc).any():
                vf = (mask.bool() & valid_anc).float()
                bce = F.binary_cross_entropy_with_logits(out["foot_pred"], foot_target, reduction="none").sum(dim=-1)
                l_recon_foot = (bce * vf).sum() / vf.sum().clamp(min=1.0)
            else:
                l_recon_foot = z.new_zeros(())

            # ----- VICReg anti-collapse -----
            l_var = vicreg_var(z, gamma=1.0) if args.l_var > 0 else z.new_zeros(())
            l_cov = vicreg_cov(z) if args.l_cov > 0 else z.new_zeros(())

            # ----- slow loss (per-frame e -> z_gait closeness) -----
            l_slow = slow_loss(out["e"], z, mask) if args.l_slow > 0 else z.new_zeros(())

            # ----- vel head MSE -----
            if args.l_vel > 0:
                d2v = ((out["v_pred"] - cmd) ** 2).sum(dim=-1) * mask
                l_vel = d2v.sum() / mask.sum().clamp(min=1.0)
            else:
                l_vel = z.new_zeros(())

            # ----- prop head (kept; weighted) -----
            if args.l_prop > 0 and "prop_pred" in out:
                feats_b = feats_norm_t[seg_idx].to(device)
                valid_b = feats_valid_t[seg_idx].to(device) * _PROP_W.to(device)
                d2p = ((out["prop_pred"] - feats_b) ** 2) * valid_b
                l_prop = d2p.sum() / valid_b.sum().clamp(min=1.0)
            else:
                l_prop = z.new_zeros(())

            loss = (args.l_rnc * l_rnc
                    + args.l_axial * l_axial
                    + args.l_lip * l_lip
                    + args.l_recon_phase * l_recon_phase
                    + args.l_recon_foot * l_recon_foot
                    + args.l_var * l_var
                    + args.l_cov * l_cov
                    + args.l_slow * l_slow
                    + args.l_vel * l_vel
                    + args.l_prop * l_prop)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep["rnc_metric"] += float(l_rnc); ep["axial"] += float(l_axial)
            ep["lip"] += float(l_lip); ep["prop"] += float(l_prop)
            ep["recon_phase"] += float(l_recon_phase); ep["recon_foot"] += float(l_recon_foot)
            ep["var"] += float(l_var); ep["cov"] += float(l_cov)
            ep["slow"] += float(l_slow); ep["vel"] += float(l_vel)
            ep["total"] += float(loss); nb += 1

        for k in ep: ep[k] /= max(1, nb)
        ep["epoch"] = epoch; ep["elapsed"] = time.time() - t0
        log.write(json.dumps(ep) + "\n"); log.flush()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[ep {epoch:3d}] total={ep['total']:.4f} rnc={ep['rnc_metric']:.3f} "
                  f"axial={ep['axial']:.3f} lip={ep['lip']:.4f} prop={ep['prop']:.3f} "
                  f"recon_ph={ep['recon_phase']:.3f}")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "encoder_kind": "segment_v3",
            "in_dim": obs_dim, "d_back": args.d_back, "d_gait": args.d_gait,
            "phase_dim": args.phase_dim, "foot_dim": 2,
            "prop_dim": prop_dim, "hard_seg_mean": bool(args.hard_seg_mean),
            "mask_kind": args.mask_kind, "mask_spec": mask_spec or "",
            "phase_joint_slice": args.phase_joint_slice,
            "phase_anchor_joint_idx": args.phase_anchor_joint_idx,
            "sigma_cmd": sigma.cpu().tolist(),
            "gait_feature_names": GAIT_FEATURE_NAMES,
            "lip_L": args.lip_L,
        },
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(args.out_dir, "encoder.pt"))
    log.close()
    print(f"[v3] wrote {args.out_dir}/encoder.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    # encoder
    p.add_argument("--d_back", type=int, default=128)
    p.add_argument("--d_gait", type=int, default=32)
    p.add_argument("--phase_dim", type=int, default=30)
    p.add_argument("--hard_seg_mean", type=int, default=1)
    # data
    p.add_argument("--segment_min_len", type=int, default=32)
    p.add_argument("--segment_max_len", type=int, default=64)
    p.add_argument("--mask_kind", default="strict",
                   choices=["none", "cmd", "cmd_tokens", "strict"])
    p.add_argument("--phase_joint_slice", default="245:260")
    p.add_argument("--phase_anchor_joint_idx", type=int, default=0)
    # losses
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--l_rnc", type=float, default=1.0)
    p.add_argument("--l_axial", type=float, default=1.0)
    p.add_argument("--l_lip", type=float, default=1.0)
    p.add_argument("--l_prop", type=float, default=1.0)
    p.add_argument("--l_recon_phase", type=float, default=1.0)
    p.add_argument("--l_recon_foot", type=float, default=0.5)
    p.add_argument("--l_var", type=float, default=1.0)
    p.add_argument("--l_cov", type=float, default=0.1)
    p.add_argument("--l_slow", type=float, default=0.3)
    p.add_argument("--l_vel", type=float, default=0.5)
    # lip params
    p.add_argument("--lip_L", type=float, default=1.0)
    p.add_argument("--lip_cos_thresh", type=float, default=0.95)
    p.add_argument("--lip_mag_diff", type=float, default=0.2)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
