# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""V2 probe: V1 probe metrics + per-gait-feature R^2 from z_gait."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import DataLoader

from frnc_pretrain import _parse_index_list
from frnc_segment_pretrain import SegmentDataset, collate_segments
from frnc_segment_pretrain_v2 import SegmentEncoderV2, precompute_segment_labels
from frnc_segment_probe import (_bucket_label, bucket_pairwise_auc,
                                intra_bucket_var_ratio, _segment_step_freq,
                                r2_z_step_freq, r2_z_cmd, intra_seg_z_var,
                                conditional_foot_probe)
from frnc_gait_features import GAIT_FEATURE_NAMES, build_mask_indices


def _encode_all(model, ds, device, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False, collate_fn=collate_segments)
    z_segs, cmd_segs = [], []
    e_frames, anc_frames, foot_frames = [], [], []
    seg_lens = []
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            anc = batch["anchor_sc"].to(device)
            out = model(obs, mask, anc, adv_lambda=0.0)
            z_segs.append(out["z_gait"].cpu().numpy())
            mfloat = mask.cpu().numpy()
            n_per = mfloat.sum(axis=1)
            cmd = batch["cmd"].numpy()
            cmd_seg = (cmd * mfloat[..., None]).sum(axis=1) / np.maximum(n_per[:, None], 1.0)
            cmd_segs.append(cmd_seg)
            B, T = obs.shape[:2]
            e = out["e"].cpu().numpy()
            anc_np = batch["anchor_sc"].numpy()
            foot_np = batch["foot"].numpy()
            for i in range(B):
                L = int(n_per[i])
                seg_lens.append(L)
                e_frames.append(e[i, :L])
                anc_frames.append(anc_np[i, :L])
                foot_frames.append(foot_np[i, :L])
    return {
        "z_seg": np.concatenate(z_segs, axis=0),
        "cmd_seg": np.concatenate(cmd_segs, axis=0),
        "e_frames": e_frames, "anc_frames": anc_frames, "foot_frames": foot_frames,
        "seg_lens": np.array(seg_lens, dtype=np.int64),
    }


def r2_z_to_features(z, feats, valid):
    """For each gait feature: linear probe z -> feature, masking invalid rows."""
    out = {}
    n_total = len(z)
    for j, name in enumerate(GAIT_FEATURE_NAMES):
        v = valid[:, j].astype(bool)
        if v.sum() < 100:
            out[f"R2_zg_{name}"] = float("nan"); continue
        Z = z[v]; y = feats[v, j]
        if y.std() < 1e-6:
            out[f"R2_zg_{name}"] = float("nan"); continue
        idx = np.random.permutation(len(Z))
        n = len(idx) // 2
        tr, te = idx[:n], idx[n:]
        try:
            reg = LinearRegression().fit(Z[tr], y[tr])
            out[f"R2_zg_{name}"] = float(r2_score(y[te], reg.predict(Z[te])))
        except Exception:
            out[f"R2_zg_{name}"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mask_kind", type=str, default=None,
                    help="override; default: read from ckpt config.mask_kind")
    ap.add_argument("--phase_joint_slice", type=str, default="245:260")
    ap.add_argument("--phase_anchor_joint_idx", type=int, default=0)
    ap.add_argument("--segment_min_len", type=int, default=32)
    ap.add_argument("--segment_max_len", type=int, default=64)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mask_kind = args.mask_kind or cfg.get("mask_kind", "cmd")
    mask_spec = build_mask_indices(mask_kind)
    mask_idx = _parse_index_list(mask_spec) if mask_spec else None
    print(f"[probe-v2] mask_kind={mask_kind} masks {0 if mask_idx is None else len(mask_idx)} dims")

    model = SegmentEncoderV2(
        in_dim=ckpt["in_dim"], d_back=cfg["d_back"], d_gait=cfg["d_gait"],
        phase_dim=ckpt["phase_dim"], foot_dim=ckpt["foot_dim"],
        prop_dim=ckpt.get("prop_dim", 0),
        adv_cmd=ckpt.get("adv_cmd", False),
        hard_seg_mean=bool(cfg.get("hard_seg_mean", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    if args.max_shards: shards = shards[:args.max_shards]
    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))
    ds = SegmentDataset(shards, mask_obs_indices=mask_idx,
                        phase_joint_slice=phase_slice,
                        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
                        segment_min_len=args.segment_min_len,
                        segment_max_len=args.segment_max_len)
    print(f"[probe-v2] {len(ds)} segments")

    enc = _encode_all(model, ds, device)
    z_seg = enc["z_seg"]; cmd_seg = enc["cmd_seg"]
    labels = _bucket_label(cmd_seg, None, None)

    seg_lab = precompute_segment_labels(ds)
    feats_raw = seg_lab["feats_raw"]
    valid = seg_lab["valid"]

    results = {}
    auc_mean, auc_detail = bucket_pairwise_auc(z_seg, labels)
    results["bucket_AUC_pairwise_mean"] = auc_mean; results.update(auc_detail)
    results["intra_bucket_var_ratio"] = intra_bucket_var_ratio(z_seg, labels)
    r2_freq, freq_std = r2_z_step_freq(z_seg, enc["foot_frames"])
    results["R2_zg_step_freq"] = r2_freq; results["step_freq_std_hz"] = freq_std
    results.update(r2_z_cmd(z_seg, cmd_seg))
    results["intra_seg_z_var_ratio"] = intra_seg_z_var(enc["e_frames"], z_seg)
    results.update(conditional_foot_probe(z_seg, enc["e_frames"],
                                          enc["anc_frames"], enc["foot_frames"]))
    results.update(r2_z_to_features(z_seg, feats_raw, valid))
    results["n_segments"] = int(len(z_seg))
    results["bucket_counts"] = {
        "standing": int((labels == 0).sum()),
        "pure_wz": int((labels == 1).sum()),
        "other": int((labels == 2).sum()),
    }
    print(json.dumps(results, indent=2))
    if args.out_json:
        json.dump(results, open(args.out_json, "w"), indent=2)
        print(f"[probe-v2] wrote {args.out_json}")


if __name__ == "__main__":
    main()
