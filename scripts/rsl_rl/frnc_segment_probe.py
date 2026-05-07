# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Probe for segment-level encoder.

Reports the metrics required to decide if `z_gait_seg` is fit for downstream
"per-bucket motion mode" induction:

  bucket_AUC_pairwise          AUC of pairwise (bucket_i vs bucket_j) logistic
                               regression on z_gait_seg (mean over 9 pairs).
  intra_bucket_var_ratio       mean over buckets of Var(z_gait | bucket) /
                               Var(z_gait_global). 0.3-0.7 is healthy.
  R2_zg_step_freq              z_gait_seg -> dominant gait freq (per segment,
                               estimated from foot contact FFT).
  R2_zg_cmd                    z_gait_seg -> cmd_seg_mean (sanity).
  intra_seg_z_var              mean per-segment Var(e_t) / Var(z_gait_global).
                               Should be << 1 for slow z_gait.
  cond_foot_AUC                linear([z_gait_bcast, sin phi, cos phi]) -> foot.
  uncond_foot_AUC              linear(z_gait_bcast) -> foot. Should be ~ 0.5.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from itertools import combinations

from frnc_segment_pretrain import (SegmentDataset, collate_segments,
                                   SegmentEncoder)
from frnc_pretrain import _parse_index_list


def _load_dataset(args):
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    if args.max_shards:
        shards = shards[:args.max_shards]
    mask_idx = _parse_index_list(args.mask_obs_indices)
    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))
    ds = SegmentDataset(shards, mask_obs_indices=mask_idx,
                        phase_joint_slice=phase_slice,
                        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
                        segment_min_len=args.segment_min_len,
                        segment_max_len=args.segment_max_len)
    return ds


def _encode_all(model, ds, device, batch_size=32):
    """Encode every segment. Returns dicts of per-segment and per-frame arrays."""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False,
                        collate_fn=collate_segments)
    z_segs, cmd_segs = [], []
    e_frames, f_frames, anc_frames, foot_frames, mask_frames = [], [], [], [], []
    cmd_frames = []
    seg_lens = []
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            anc = batch["anchor_sc"].to(device)
            out = model(obs, mask, anc)
            z_segs.append(out["z_gait"].cpu().numpy())
            mfloat = mask.cpu().numpy()
            n_per = mfloat.sum(axis=1)
            cmd = batch["cmd"].numpy()  # (B,T,3)
            cmd_seg = (cmd * mfloat[..., None]).sum(axis=1) / np.maximum(n_per[:, None], 1.0)
            cmd_segs.append(cmd_seg)
            B, T = obs.shape[:2]
            e = out["e"].cpu().numpy()
            f = out["f"].cpu().numpy()
            anc_np = batch["anchor_sc"].numpy()
            foot_np = batch["foot"].numpy()
            for i in range(B):
                L = int(n_per[i])
                seg_lens.append(L)
                e_frames.append(e[i, :L])
                f_frames.append(f[i, :L])
                anc_frames.append(anc_np[i, :L])
                foot_frames.append(foot_np[i, :L])
                cmd_frames.append(cmd[i, :L])
    return {
        "z_seg": np.concatenate(z_segs, axis=0),
        "cmd_seg": np.concatenate(cmd_segs, axis=0),
        "seg_lens": np.array(seg_lens, dtype=np.int64),
        "e_frames": e_frames,    # list of (L_i, d_gait)
        "f_frames": f_frames,
        "anc_frames": anc_frames,
        "foot_frames": foot_frames,
        "cmd_frames": cmd_frames,
    }


def _bucket_label(cmd_seg, vx_edges, wz_edges):
    """3-way bucket: 0=standing (|cmd|<0.05), 1=pure_wz (|vx|<0.1, |vy|<0.1, |wz|>=0.2), 2=other."""
    vx, vy, wz = cmd_seg[:, 0], cmd_seg[:, 1], cmd_seg[:, 2]
    cmd_norm = np.linalg.norm(cmd_seg, axis=-1)
    lab = np.full(len(cmd_seg), 2, dtype=np.int64)
    lab[cmd_norm < 0.05] = 0
    pure_wz = (np.abs(vx) < 0.1) & (np.abs(vy) < 0.1) & (np.abs(wz) >= 0.2) & (lab != 0)
    lab[pure_wz] = 1
    return lab


def bucket_pairwise_auc(z_seg, labels):
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan"), {}
    aucs = []
    detail = {}
    for a, b in combinations(classes, 2):
        m = (labels == a) | (labels == b)
        y = (labels[m] == b).astype(np.int64)
        if len(np.unique(y)) < 2 or m.sum() < 20:
            continue
        idx = np.random.permutation(m.sum())
        n = len(idx) // 2
        tr, te = idx[:n], idx[n:]
        Z = z_seg[m]
        try:
            clf = LogisticRegression(max_iter=200).fit(Z[tr], y[tr])
            p = clf.predict_proba(Z[te])[:, 1]
            auc = roc_auc_score(y[te], p)
        except Exception:
            continue
        detail[f"AUC_b{int(a)}_vs_b{int(b)}"] = float(auc)
        aucs.append(auc)
    return (float(np.mean(aucs)) if aucs else float("nan")), detail


def intra_bucket_var_ratio(z_seg, labels):
    """mean_b Var(z|b) / Var(z_global). VICReg-style: trace of cov."""
    var_g = float(z_seg.var(axis=0).mean())
    if var_g < 1e-8:
        return float("nan")
    ratios = []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 5:
            continue
        v = float(z_seg[m].var(axis=0).mean())
        ratios.append(v / var_g)
    return float(np.mean(ratios)) if ratios else float("nan")


def _segment_step_freq(foot, fs=50.0):
    """Estimate dominant gait freq from foot contact toggles in one segment.

    foot: (L, F) {0,1}. Returns scalar Hz (NaN if no clear oscillation).
    """
    L, F_ = foot.shape
    if L < 16:
        return float("nan")
    # sum of foot indicators detrended; FFT
    sig = foot.sum(axis=-1).astype(np.float32)
    sig = sig - sig.mean()
    if sig.std() < 1e-3:
        return float("nan")
    sp = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    # ignore DC + very-low (<0.3 Hz)
    keep = freqs >= 0.3
    if not keep.any():
        return float("nan")
    sp_k = sp[keep]; fr_k = freqs[keep]
    return float(fr_k[np.argmax(sp_k)])


def r2_z_step_freq(z_seg, foot_frames, fs=50.0):
    freqs = np.array([_segment_step_freq(np.asarray(f), fs) for f in foot_frames])
    valid = ~np.isnan(freqs)
    if valid.sum() < 50:
        return float("nan"), float("nan")
    Z = z_seg[valid]; y = freqs[valid]
    idx = np.random.permutation(len(Z)); n = len(idx) // 2
    tr, te = idx[:n], idx[n:]
    reg = LinearRegression().fit(Z[tr], y[tr])
    return float(r2_score(y[te], reg.predict(Z[te]))), float(np.std(y))


def r2_z_cmd(z_seg, cmd_seg):
    out = {}
    idx = np.random.permutation(len(z_seg)); n = len(idx) // 2
    tr, te = idx[:n], idx[n:]
    for a, name in [(0, "vx"), (1, "vy"), (2, "wz")]:
        reg = LinearRegression().fit(z_seg[tr], cmd_seg[tr, a])
        out[f"R2_zg_seg_{name}"] = float(r2_score(cmd_seg[te, a], reg.predict(z_seg[te])))
    return out


def intra_seg_z_var(e_frames, z_seg):
    """mean over segments of mean_dim Var_t(e_t) / Var_global(z_seg)."""
    var_g = float(z_seg.var(axis=0).mean())
    if var_g < 1e-8:
        return float("nan")
    rs = []
    for e in e_frames:
        if e.shape[0] < 4:
            continue
        rs.append(float(e.var(axis=0).mean()) / var_g)
    return float(np.mean(rs)) if rs else float("nan")


def conditional_foot_probe(z_seg, e_frames, anc_frames, foot_frames):
    """Per-frame probe: linear([z_gait_seg_bcast, sin phi, cos phi]) -> foot.

    Compares to unconditional linear(z_gait_seg_bcast) -> foot.
    """
    Xc, Xu, Y = [], [], []
    for zg, anc, foot in zip(z_seg, anc_frames, foot_frames):
        valid = ~np.isnan(anc).any(axis=-1)
        if valid.sum() < 4:
            continue
        zg_b = np.broadcast_to(zg[None, :], (valid.sum(), zg.shape[0]))
        a = anc[valid]
        Xc.append(np.concatenate([zg_b, a], axis=-1))
        Xu.append(zg_b)
        Y.append(foot[valid])
    if not Xc:
        return {}
    Xc = np.concatenate(Xc, 0); Xu = np.concatenate(Xu, 0); Y = np.concatenate(Y, 0)
    idx = np.random.permutation(len(Y)); n = min(20000, len(idx) // 2)
    tr, te = idx[:n], idx[n:n + n]
    out = {}
    for fi in range(Y.shape[1]):
        y = Y[:, fi].astype(np.int64)
        if len(np.unique(y[tr])) < 2:
            out[f"cond_foot{fi}_AUC"] = float("nan")
            out[f"uncond_foot{fi}_AUC"] = float("nan"); continue
        try:
            cc = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xc[tr], y[tr])
            cu = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xu[tr], y[tr])
            out[f"cond_foot{fi}_AUC"] = float(roc_auc_score(y[te], cc.predict_proba(Xc[te])[:, 1]))
            out[f"uncond_foot{fi}_AUC"] = float(roc_auc_score(y[te], cu.predict_proba(Xu[te])[:, 1]))
        except Exception:
            out[f"cond_foot{fi}_AUC"] = float("nan")
            out[f"uncond_foot{fi}_AUC"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mask_obs_indices", type=str, default=None)
    ap.add_argument("--phase_joint_slice", type=str, default="245:260")
    ap.add_argument("--phase_anchor_joint_idx", type=int, default=0)
    ap.add_argument("--segment_min_len", type=int, default=32)
    ap.add_argument("--segment_max_len", type=int, default=64)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SegmentEncoder(
        in_dim=ckpt["in_dim"],
        d_back=cfg["d_back"],
        d_gait=cfg["d_gait"],
        phase_dim=ckpt["phase_dim"],
        foot_dim=ckpt["foot_dim"],
        hard_seg_mean=bool(cfg.get("hard_seg_mean", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[probe] loaded encoder; in_dim={ckpt['in_dim']} d_gait={cfg['d_gait']} hard={cfg.get('hard_seg_mean', 1)}")

    ds = _load_dataset(args)
    print(f"[probe] {len(ds)} segments")
    enc = _encode_all(model, ds, device)

    z_seg = enc["z_seg"]
    cmd_seg = enc["cmd_seg"]
    labels = _bucket_label(cmd_seg, None, None)
    print(f"[probe] bucket counts: stand={int((labels==0).sum())} pure_wz={int((labels==1).sum())} other={int((labels==2).sum())}")

    results = {}

    auc_mean, auc_detail = bucket_pairwise_auc(z_seg, labels)
    results["bucket_AUC_pairwise_mean"] = auc_mean
    results.update(auc_detail)
    results["intra_bucket_var_ratio"] = intra_bucket_var_ratio(z_seg, labels)

    r2_freq, freq_std = r2_z_step_freq(z_seg, enc["foot_frames"])
    results["R2_zg_step_freq"] = r2_freq
    results["step_freq_std_hz"] = freq_std

    results.update(r2_z_cmd(z_seg, cmd_seg))

    results["intra_seg_z_var_ratio"] = intra_seg_z_var(enc["e_frames"], z_seg)

    results.update(conditional_foot_probe(
        z_seg, enc["e_frames"], enc["anc_frames"], enc["foot_frames"]))

    # bucket size sanity
    results["n_segments"] = int(len(z_seg))
    results["bucket_counts"] = {
        "standing": int((labels == 0).sum()),
        "pure_wz": int((labels == 1).sum()),
        "other": int((labels == 2).sum()),
    }

    print(json.dumps(results, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[probe] wrote {args.out_json}")


if __name__ == "__main__":
    main()
