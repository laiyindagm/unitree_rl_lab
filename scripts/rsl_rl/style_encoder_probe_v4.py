# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Probe metrics for frnc_style_v4 encoders."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from style_encoder_pretrain_v4 import StyleEncoderV4, StyleWindowDataset, _apply_mask, cmd_delta
from style_obs_layout import build_mask_indices, randomize_shortcut_terms


def _onehot_mode(mode: np.ndarray) -> np.ndarray:
    mode = np.asarray(mode, dtype=np.int64).reshape(-1)
    out = np.zeros((len(mode), 3), dtype=np.float32)
    out[np.arange(len(mode)), np.clip(mode, 0, 2)] = 1.0
    return out


def _ridge_r2(X: np.ndarray, Y: np.ndarray, valid: np.ndarray | None = None, seed: int = 0) -> tuple[float, list[float]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if valid is None:
        valid = np.ones_like(Y, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for j in range(Y.shape[1]):
        m = (valid[:, j] > 0.5) & np.isfinite(Y[:, j]) & np.isfinite(X).all(axis=1)
        idx = np.where(m)[0]
        if len(idx) < 40 or float(np.std(Y[idx, j])) < 1e-6:
            scores.append(float("nan"))
            continue
        idx = idx[rng.permutation(len(idx))]
        cut = max(20, int(0.8 * len(idx)))
        if cut >= len(idx):
            cut = len(idx) - 1
        tr, te = idx[:cut], idx[cut:]
        model = Ridge(alpha=1.0)
        model.fit(X[tr], Y[tr, j])
        pred = model.predict(X[te])
        scores.append(float(r2_score(Y[te, j], pred)))
    finite = [s for s in scores if np.isfinite(s)]
    return (float(np.mean(finite)) if finite else float("nan")), scores


def _phase_target(phi: np.ndarray, phi_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = phi_valid.astype(np.float32)
    denom = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    sc = (phi * w[..., None]).sum(axis=1) / denom
    valid = (w.sum(axis=1) >= max(4, phi.shape[1] // 4)).astype(np.float32)
    return sc.astype(np.float32), np.repeat(valid[:, None], 2, axis=1)


def _effective_rank(z: np.ndarray) -> float:
    x = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = (s ** 2) / max(float(np.sum(s ** 2)), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def _shift_ratio(z_a: np.ndarray, z_b: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(z_a)
    if n < 2:
        return float("nan")
    same = np.linalg.norm(z_a - z_b, axis=-1).mean()
    perm = rng.permutation(n)
    diff = np.linalg.norm(z_a - z_a[perm], axis=-1).mean()
    return float(same / max(diff, 1e-8))


def _sampled_style_spearman(
    z: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    mode: np.ndarray,
    cmd: np.ndarray,
    cmd_sigma: np.ndarray,
    cond_delta: float,
    conditional: bool,
    seed: int = 0,
    max_pairs: int = 20000,
) -> tuple[float, int]:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    candidates = np.where((mode.reshape(-1) != 0) & (valid.sum(axis=1) > 1))[0]
    if len(candidates) < 2:
        return float("nan"), 0
    dz_all: list[np.ndarray] = []
    dy_all: list[np.ndarray] = []
    tries = 0
    while sum(len(x) for x in dz_all) < max_pairs and tries < 40:
        tries += 1
        i = rng.choice(candidates, size=max_pairs, replace=True)
        j = rng.choice(candidates, size=max_pairs, replace=True)
        m = i != j
        if conditional:
            dcmd = np.linalg.norm((cmd[i] - cmd[j]) / np.maximum(cmd_sigma[None, :], 1e-3), axis=-1)
            m = m & (mode[i].reshape(-1) == mode[j].reshape(-1)) & (dcmd <= cond_delta)
        i = i[m]
        j = j[m]
        if len(i) == 0:
            continue
        v = valid[i] * valid[j]
        denom = np.maximum(v.sum(axis=1), 1.0)
        dy = np.sqrt((((y[i] - y[j]) ** 2) * v).sum(axis=1) / denom)
        dz = np.linalg.norm(z[i] - z[j], axis=-1)
        ok = np.isfinite(dy) & np.isfinite(dz) & (denom > 1.0)
        if ok.any():
            dz_all.append(dz[ok])
            dy_all.append(dy[ok])
    if not dz_all:
        return float("nan"), 0
    dz = np.concatenate(dz_all)[:max_pairs]
    dy = np.concatenate(dy_all)[:max_pairs]
    if len(dz) < 20:
        return float("nan"), int(len(dz))
    rho = spearmanr(dz, dy).correlation
    return float(rho), int(len(dz))


def _encode_all(model, ds, device, mask_kind: str, batch_size: int, num_workers: int, seed: int):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    mask_idx_list = build_mask_indices(mask_kind)
    mask_idx = torch.tensor(mask_idx_list, dtype=torch.long, device=device) if mask_idx_list else None
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["z_a", "z_b", "z_short", "cmd", "mode", "bucket", "y0", "y0_valid", "phi", "phi_valid", "yphi", "yphi_valid"]}
    with torch.no_grad():
        for batch in loader:
            obs_a_raw = batch["obs_a"]
            obs_b_raw = batch["obs_b"]
            obs_a = _apply_mask(obs_a_raw.to(device), mask_idx)
            obs_b = _apply_mask(obs_b_raw.to(device), mask_idx)
            phi_a = batch["phi_a"].to(device)
            phi_b = batch["phi_b"].to(device)
            z_a = model(obs_a, phi_a)["z"].cpu().numpy()
            z_b = model(obs_b, phi_b)["z"].cpu().numpy()

            obs_short_np = randomize_shortcut_terms(obs_a_raw.numpy(), rng=rng)
            obs_short = _apply_mask(torch.from_numpy(obs_short_np).float().to(device), mask_idx)
            z_short = model(obs_short, phi_a)["z"].cpu().numpy()

            out["z_a"].append(z_a)
            out["z_b"].append(z_b)
            out["z_short"].append(z_short)
            out["cmd"].append(batch["cmd"].numpy())
            out["mode"].append(batch["mode_id"].numpy().reshape(-1))
            out["bucket"].append(batch["bucket_id"].numpy().reshape(-1))
            out["y0"].append(batch["y0"].numpy())
            out["y0_valid"].append(batch["y0_valid"].numpy())
            out["phi"].append(batch["phi_a"].numpy())
            out["phi_valid"].append(batch["phi_valid_a"].numpy())
            out["yphi"].append(batch["yphi_a"].numpy())
            out["yphi_valid"].append(batch["yphi_valid_a"].numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    return obj


def main():
    ap = argparse.ArgumentParser(description="Probe frnc_style_v4 encoder.")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--mask_kind", default=None)
    ap.add_argument("--cond_cmd_delta", type=float, default=None)
    ap.add_argument("--max_frame_samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mask_kind = args.mask_kind or cfg.get("mask_kind", "M0_conservative")
    cond_delta = args.cond_cmd_delta if args.cond_cmd_delta is not None else cfg.get("cond_cmd_delta", 0.35)
    model = StyleEncoderV4(
        in_dim=cfg["in_dim"],
        y0_dim=cfg["y0_dim"],
        yphi_dim=cfg["yphi_dim"],
        d_back=cfg["d_back"],
        d_gait=cfg["d_gait"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = StyleWindowDataset(args.data_dir, max_shards=args.max_shards, max_samples=args.max_samples)
    enc = _encode_all(model, ds, device, mask_kind, args.batch_size, args.num_workers, args.seed)

    z = enc["z_a"].astype(np.float32)
    z_b = enc["z_b"].astype(np.float32)
    z_short = enc["z_short"].astype(np.float32)
    cmd = enc["cmd"].astype(np.float32)
    mode = enc["mode"].astype(np.int64)
    y0 = enc["y0"].astype(np.float32)
    y0_valid = enc["y0_valid"].astype(np.float32)
    cmd_mode = np.concatenate([cmd, _onehot_mode(mode)], axis=1)
    cmd_sigma = np.asarray(ckpt.get("cmd_sigma", np.std(cmd, axis=0).clip(min=0.05)), dtype=np.float32)

    results: dict[str, object] = {
        "n_samples": int(len(z)),
        "mask_kind": mask_kind,
        "cond_cmd_delta": float(cond_delta),
        "d_gait": int(z.shape[1]),
    }

    r2_y0_z, r2_y0_z_each = _ridge_r2(z, y0, y0_valid, seed=args.seed)
    r2_y0_cmd, _ = _ridge_r2(cmd_mode, y0, y0_valid, seed=args.seed)
    r2_y0_z_cmd, _ = _ridge_r2(np.concatenate([z, cmd_mode], axis=1), y0, y0_valid, seed=args.seed)
    results["R2_Y0_from_Z"] = r2_y0_z
    results["R2_Y0_from_cmd_mode"] = r2_y0_cmd
    results["R2_Y0_from_Z_cmd_mode"] = r2_y0_z_cmd
    results["R2_Y0_gain_over_cmd_mode"] = float(r2_y0_z_cmd - r2_y0_cmd) if np.isfinite(r2_y0_z_cmd) and np.isfinite(r2_y0_cmd) else float("nan")
    results["R2_Y0_feature_mean"] = r2_y0_z_each

    r2_cmd_z, r2_cmd_each = _ridge_r2(z, cmd, np.ones_like(cmd), seed=args.seed)
    results["R2_cmd_from_Z"] = r2_cmd_z
    results["R2_cmd_from_Z_each"] = r2_cmd_each

    phase_sc, phase_valid = _phase_target(enc["phi"], enc["phi_valid"])
    r2_phase_z, r2_phase_each = _ridge_r2(z, phase_sc, phase_valid, seed=args.seed)
    results["R2_phase_from_Z"] = r2_phase_z
    results["R2_phase_from_Z_each"] = r2_phase_each

    results["shift_ratio"] = _shift_ratio(z, z_b, seed=args.seed)
    results["effective_rank"] = _effective_rank(z)
    pair_dist = np.linalg.norm(z - z[np.random.default_rng(args.seed).permutation(len(z))], axis=-1).mean()
    results["shortcut_delta_ratio"] = float(np.linalg.norm(z - z_short, axis=-1).mean() / max(pair_dist, 1e-8))

    rho_g, n_pairs_g = _sampled_style_spearman(
        z,
        y0,
        y0_valid,
        mode,
        cmd,
        cmd_sigma,
        cond_delta,
        conditional=False,
        seed=args.seed,
    )
    rho_res, n_pairs_res = _sampled_style_spearman(
        z,
        y0,
        y0_valid,
        mode,
        cmd,
        cmd_sigma,
        cond_delta,
        conditional=True,
        seed=args.seed + 17,
    )
    results["rho_G_spearman"] = rho_g
    results["rho_G_pairs"] = n_pairs_g
    results["rho_G_cond_cmd_spearman"] = rho_res
    results["rho_G_cond_cmd_pairs"] = n_pairs_res

    yphi = enc["yphi"].reshape(-1, enc["yphi"].shape[-1]).astype(np.float32)
    yphi_valid = enc["yphi_valid"].reshape(-1, enc["yphi_valid"].shape[-1]).astype(np.float32)
    phi_frame = enc["phi"].reshape(-1, 2).astype(np.float32)
    z_frame = np.repeat(z, enc["phi"].shape[1], axis=0)
    rng = np.random.default_rng(args.seed)
    if len(yphi) > args.max_frame_samples:
        idx = rng.permutation(len(yphi))[: args.max_frame_samples]
        yphi = yphi[idx]
        yphi_valid = yphi_valid[idx]
        phi_frame = phi_frame[idx]
        z_frame = z_frame[idx]
    r2_yphi_z_phi, _ = _ridge_r2(np.concatenate([z_frame, phi_frame], axis=1), yphi, yphi_valid, seed=args.seed)
    r2_yphi_phi, _ = _ridge_r2(phi_frame, yphi, yphi_valid, seed=args.seed)
    results["R2_Yphi_from_Z_phi"] = r2_yphi_z_phi
    results["R2_Yphi_from_phi_only"] = r2_yphi_phi
    results["R2_Yphi_gain_over_phi"] = float(r2_yphi_z_phi - r2_yphi_phi) if np.isfinite(r2_yphi_z_phi) and np.isfinite(r2_yphi_phi) else float("nan")

    bucket_metrics = {}
    for bucket in sorted(set(int(x) for x in enc["bucket"].reshape(-1))):
        m = enc["bucket"].reshape(-1) == bucket
        if int(m.sum()) < 40:
            continue
        r2_b, _ = _ridge_r2(z[m], y0[m], y0_valid[m], seed=args.seed)
        bucket_metrics[str(bucket)] = {"n": int(m.sum()), "R2_Y0_from_Z": r2_b}
    results["bucket_metrics"] = bucket_metrics

    safe_results = _json_sanitize(results)
    print(json.dumps(safe_results, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(safe_results, f, indent=2)
        print(f"[style-probe-v4] wrote {args.out_json}")


if __name__ == "__main__":
    main()
