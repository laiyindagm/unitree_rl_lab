# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""t-SNE visualization of FRnC encoder latents."""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from frnc_pretrain import FRnCEncoder


def load_shards(data_dir, max_shards):
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if max_shards: paths = paths[:max_shards]
    obs, cmd, foot = [], [], []
    for p in paths:
        d = np.load(p)
        obs.append(d["policy_obs"]); cmd.append(d["cmd"]); foot.append(d["foot_contact"])
    return (np.concatenate(obs).astype(np.float32),
            np.concatenate(cmd).astype(np.float32),
            np.concatenate(foot).astype(np.uint8))


def encode(model, obs, device, batch=4096):
    out = {k: [] for k in ["zg", "z_gait", "f"]}
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            x = torch.from_numpy(obs[i:i+batch]).to(device)
            o = model(x)
            for k in out:
                if k in o: out[k].append(o[k].cpu().numpy())
    return {k: (np.concatenate(v) if v else None) for k, v in out.items()}


def make_plot(ax, emb, color, cmap, title, vmin=None, vmax=None, s=3, alpha=0.55):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, linewidth=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--max_shards", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--mask_obs_indices", type=str, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    sd = ckpt["model"]; cfg = ckpt["config"]
    phase_dim = 0
    for k, v in sd.items():
        if k.startswith("phase_head") and k.endswith("weight") and v.ndim == 2:
            if v.shape[0] != 128: phase_dim = max(phase_dim, v.shape[0])
    model = FRnCEncoder(in_dim=ckpt["in_dim"], d_back=cfg["d_back"], d_axis=cfg["d_axis"],
                        phase_dim=phase_dim, d_gait=int(cfg.get("d_gait", 32)),
                        use_hierarchical=bool(cfg.get("use_hierarchical", False))).to(device)
    model.load_state_dict(sd); model.eval()
    has_z_gait = bool(cfg.get("use_hierarchical", False))

    obs, cmd, foot = load_shards(args.data_dir, args.max_shards)
    if args.mask_obs_indices:
        idx = []
        for c in args.mask_obs_indices.split(","):
            c = c.strip()
            if not c: continue
            if ":" in c:
                a, b = c.split(":"); idx.extend(range(int(a), int(b)))
            else:
                idx.append(int(c))
        idx = np.array(sorted(set(idx)), dtype=np.int64)
        obs = obs.copy(); obs[:, idx] = 0.0

    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(obs), size=min(args.n_samples, len(obs)), replace=False)
    obs_s, cmd_s, foot_s = obs[sub], cmd[sub], foot[sub]

    enc = encode(model, obs_s, device)
    foot_class = (foot_s[:, 0].astype(int) << 1) | foot_s[:, 1].astype(int)
    cmd_norm = np.linalg.norm(cmd_s, axis=1)
    cmd_dir = np.arctan2(cmd_s[:, 1], cmd_s[:, 0])
    cmd_dir_masked = np.where(cmd_norm > 0.2, cmd_dir, np.nan)

    layouts = [("zg", enc["zg"])]
    if has_z_gait and enc["z_gait"] is not None:
        layouts.append(("z_gait", enc["z_gait"]))
    layouts.append(("f", enc["f"]))

    fig, axes = plt.subplots(len(layouts), 4, figsize=(18, 4.2 * len(layouts)))
    if len(layouts) == 1: axes = axes[None, :]

    for r, (name, z) in enumerate(layouts):
        print(f"[tsne] {name} shape={z.shape}, fitting ...", flush=True)
        emb = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                   learning_rate="auto", random_state=args.seed,
                   max_iter=1000).fit_transform(z)
        sc = make_plot(axes[r, 0], emb, foot_class, "tab10",
                       f"{name} | foot_contact (0=air,1=R,2=L,3=both)", vmin=-0.5, vmax=3.5)
        plt.colorbar(sc, ax=axes[r, 0], fraction=0.04)
        sc = make_plot(axes[r, 1], emb, cmd_norm, "viridis",
                       f"{name} | |cmd|", vmin=0, vmax=cmd_norm.max())
        plt.colorbar(sc, ax=axes[r, 1], fraction=0.04)
        m = ~np.isnan(cmd_dir_masked)
        sc = make_plot(axes[r, 2], emb[m], cmd_dir_masked[m], "hsv",
                       f"{name} | cmd direction (|cmd|>0.2)", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(sc, ax=axes[r, 2], fraction=0.04)
        sc = make_plot(axes[r, 3], emb, cmd_s[:, 0], "coolwarm",
                       f"{name} | cmd_x", vmin=-1.5, vmax=1.5)
        plt.colorbar(sc, ax=axes[r, 3], fraction=0.04)

    fig.suptitle(os.path.basename(os.path.dirname(args.encoder)), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=110, bbox_inches="tight")
    print(f"[tsne] saved {args.out_png}")


if __name__ == "__main__":
    main()
