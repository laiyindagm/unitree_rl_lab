# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Linear-Evaluation-Protocol probe harness for the FRnC encoder.

Loads a trained encoder.pt + the same shards used (or a held-out set) and
reports:
  * per-axis linear-probe R^2 (z^a -> v_a)            [target > 0.85]
  * cross-axis linear-probe R^2 (z^x -> v_y, etc.)    [target < 0.10]
  * Spearman rank correlation between -sim(z^a_i, z^a_j) and |v_a^i - v_a^j|
  * gait-phase probe AUC: z -> foot_contact[k]        [target > 0.85]
  * time-shift consistency: sim(z_t, z_{t+dt}) curve
  * metric-distortion check: linear fit slope dz vs dv, and per-bin slope
    variance (the v21f-style "high-speed flat region" test).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr

from frnc_pretrain import FRnCEncoder


def load_shards(data_dir, max_shards=None):
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if max_shards:
        paths = paths[:max_shards]
    obs, cmd, lin, ang, foot, env_id, gstep = [], [], [], [], [], [], []
    for p in paths:
        d = np.load(p)
        obs.append(d["policy_obs"]); cmd.append(d["cmd"])
        lin.append(d["actual_lin_vel"]); ang.append(d["actual_ang_vel"])
        foot.append(d["foot_contact"]); env_id.append(d["env_id"])
        gstep.append(d["global_step"])
    return {
        "obs": np.concatenate(obs).astype(np.float32),
        "cmd": np.concatenate(cmd).astype(np.float32),
        "lin": np.concatenate(lin).astype(np.float32),
        "ang": np.concatenate(ang).astype(np.float32),
        "foot": np.concatenate(foot).astype(np.uint8),
        "env_id": np.concatenate(env_id).astype(np.int64),
        "gstep": np.concatenate(gstep).astype(np.int64),
    }


def encode_all(model, obs, device, batch=4096):
    out = {k: [] for k in ["zx", "zy", "zw", "zg", "v_pred", "f"]}
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            x = torch.from_numpy(obs[i:i+batch]).to(device)
            o = model(x)
            for k in out:
                out[k].append(o[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def linear_probe(z, y, n_train=20000):
    n = min(n_train, len(z) // 2)
    idx = np.random.permutation(len(z))
    tr, te = idx[:n], idx[n:n+n]
    reg = LinearRegression().fit(z[tr], y[tr])
    return r2_score(y[te], reg.predict(z[te]))


def logistic_probe(z, y, n_train=20000):
    n = min(n_train, len(z) // 2)
    idx = np.random.permutation(len(z))
    tr, te = idx[:n], idx[n:n+n]
    if len(np.unique(y[tr])) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=200).fit(z[tr], y[tr])
    p = clf.predict_proba(z[te])[:, 1]
    return roc_auc_score(y[te], p)


def spearman_pairwise(z, v_axis, n_pairs=20000):
    rng = np.random.default_rng(0)
    idx_i = rng.integers(0, len(z), n_pairs)
    idx_j = rng.integers(0, len(z), n_pairs)
    keep = idx_i != idx_j
    i, j = idx_i[keep], idx_j[keep]
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    sim = np.einsum("nd,nd->n", zn[i], zn[j])
    delta = np.abs(v_axis[i] - v_axis[j])
    rho, _ = spearmanr(-sim, delta)  # higher delta -> higher dissim
    return rho


def metric_distortion(z, v_axis, n_pairs=20000, n_bins=5):
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(z), n_pairs)
    j = rng.integers(0, len(z), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dz = np.linalg.norm(z[i] - z[j], axis=-1)
    dv = np.abs(v_axis[i] - v_axis[j])
    # global linear fit
    if np.std(dv) < 1e-6:
        slope, intercept = float("nan"), float("nan")
    else:
        try:
            slope, intercept = np.polyfit(dv, dz, 1)
        except Exception:
            slope, intercept = float("nan"), float("nan")
    pred = slope * dv + intercept
    r2 = 1 - np.sum((dz - pred) ** 2) / (np.sum((dz - dz.mean()) ** 2) + 1e-8)
    # per-bin slope (split by mean |v_axis| of pair)
    v_mid = 0.5 * (np.abs(v_axis[i]) + np.abs(v_axis[j]))
    edges = np.quantile(v_mid, np.linspace(0, 1, n_bins + 1))
    bin_slopes = []
    for b in range(n_bins):
        m = (v_mid >= edges[b]) & (v_mid <= edges[b + 1])
        if m.sum() < 50 or np.std(dv[m]) < 1e-6:
            bin_slopes.append(float("nan"))
            continue
        try:
            s_, _ = np.polyfit(dv[m], dz[m], 1)
        except Exception:
            bin_slopes.append(float("nan")); continue
        bin_slopes.append(float(s_))
    bs = np.array(bin_slopes, dtype=np.float64)
    valid = ~np.isnan(bs)
    slope_cv = float(np.std(bs[valid]) / (np.mean(bs[valid]) + 1e-8)) if valid.any() else float("nan")
    return {
        "global_slope": float(slope),
        "global_r2_linear": float(r2),
        "bin_slopes": bs.tolist(),
        "slope_cv": slope_cv,
    }


def time_shift_curve(z, env_id, gstep, max_dt=20):
    # group indices by env, sort by gstep
    order = np.lexsort((gstep, env_id))
    z = z[order]; env_id = env_id[order]; gstep = gstep[order]
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    # find boundaries
    boundaries = np.flatnonzero(np.diff(env_id) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(z)]])
    sims = {dt: [] for dt in range(1, max_dt + 1)}
    for s, e in zip(starts, ends):
        seg = zn[s:e]
        L = len(seg)
        for dt in range(1, max_dt + 1):
            if L <= dt:
                break
            sims[dt].append(np.einsum("nd,nd->n", seg[:-dt], seg[dt:]))
    return {dt: float(np.concatenate(v).mean()) if v else float("nan")
            for dt, v in sims.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mask_obs_indices", type=str, default=None,
                    help="same indices used during pretrain to ensure consistent eval.")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    # Recover phase_dim from saved state_dict for backward compatibility.
    sd = ckpt["model"]
    phase_dim = 0
    for k, v in sd.items():
        if k.startswith("phase_head") and k.endswith("weight") and v.ndim == 2:
            # last linear layer's out_features
            if v.shape[0] not in (128,):
                phase_dim = max(phase_dim, v.shape[0])
    cfg = ckpt["config"]
    d_gait = int(cfg.get("d_gait", 32))
    use_hierarchical = bool(cfg.get("use_hierarchical", False))
    model = FRnCEncoder(in_dim=ckpt["in_dim"],
                        d_back=cfg["d_back"],
                        d_axis=cfg["d_axis"],
                        phase_dim=phase_dim,
                        d_gait=d_gait,
                        use_hierarchical=use_hierarchical).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[probe] loaded encoder, in_dim={ckpt['in_dim']}")

    data = load_shards(args.data_dir, args.max_shards)
    print(f"[probe] loaded {len(data['obs'])} samples")
    obs_for_eval = data["obs"]
    if args.mask_obs_indices:
        idx = []
        for chunk in args.mask_obs_indices.split(","):
            chunk = chunk.strip()
            if not chunk: continue
            if ":" in chunk:
                a,b = chunk.split(":"); idx.extend(range(int(a), int(b)))
            else:
                idx.append(int(chunk))
        idx = np.array(sorted(set(idx)), dtype=np.int64)
        obs_for_eval = obs_for_eval.copy()
        obs_for_eval[:, idx] = 0.0
        print(f"[probe] masked {len(idx)} obs dims for eval")
    enc = encode_all(model, obs_for_eval, device)

    cmd = data["cmd"]
    results = {}

    # 1. axis-aligned probes
    for a, k, name in [(0, "zx", "x"), (1, "zy", "y"), (2, "zw", "w")]:
        results[f"R2_axis_{name}->{name}"] = linear_probe(enc[k], cmd[:, a])
    # 2. cross-axis (should be small)
    for src, sk in [("zx", 0), ("zy", 1), ("zw", 2)]:
        for tgt in range(3):
            if tgt == sk:
                continue
            results[f"R2_cross_{src}->v{tgt}"] = linear_probe(enc[src], cmd[:, tgt])

    # 3. global block predicts full v
    full_z = enc["zg"]
    for tgt, name in enumerate(["x", "y", "w"]):
        results[f"R2_global_zg->v{name}"] = linear_probe(full_z, cmd[:, tgt])

    # 4. spearman
    for a, k, name in [(0, "zx", "x"), (1, "zy", "y"), (2, "zw", "w")]:
        results[f"spearman_{name}"] = float(spearman_pairwise(enc[k], cmd[:, a]))

    # 5. metric distortion (KEY for the user's concern)
    for a, k, name in [(0, "zx", "x"), (1, "zy", "y"), (2, "zw", "w")]:
        results[f"metric_{name}"] = metric_distortion(enc[k], cmd[:, a])

    # 6. gait-phase probe per foot
    foot = data["foot"]
    for f in range(foot.shape[1]):
        results[f"AUC_foot{f}_from_zg"] = float(logistic_probe(enc["zg"], foot[:, f]))

    # 7. time-shift consistency
    results["time_shift_zg"] = time_shift_curve(enc["zg"], data["env_id"], data["gstep"])

    # 7b. phase head consistency (only if model has phase head)
    if model.phase_dim > 0:
        with torch.no_grad():
            phase_pred = []
            for i in range(0, len(obs_for_eval), 4096):
                x = torch.from_numpy(obs_for_eval[i:i+4096]).to(device)
                phase_pred.append(model(x)["phase_pred"].cpu().numpy())
            phase_pred = np.concatenate(phase_pred)
        # ||(sin,cos)|| should approach 1 for well-learned phase
        J = phase_pred.shape[1] // 2
        norms = np.sqrt(phase_pred[:, 0::2] ** 2 + phase_pred[:, 1::2] ** 2)
        results["phase_pred_norm_mean"] = float(norms.mean())
        results["phase_pred_norm_std"] = float(norms.std())
        # Per-joint variance of predicted angle (high variance -> phase is being tracked)
        ang = np.arctan2(phase_pred[:, 0::2], phase_pred[:, 1::2])
        results["phase_pred_angle_std_mean"] = float(np.std(ang, axis=0).mean())

    # 8. velocity head MSE
    v_pred = enc["v_pred"]
    results["vel_head_mse_x"] = float(((v_pred[:, 0] - cmd[:, 0]) ** 2).mean())
    results["vel_head_mse_y"] = float(((v_pred[:, 1] - cmd[:, 1]) ** 2).mean())
    results["vel_head_mse_w"] = float(((v_pred[:, 2] - cmd[:, 2]) ** 2).mean())

    print(json.dumps(results, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[probe] wrote {args.out_json}")


if __name__ == "__main__":
    main()
