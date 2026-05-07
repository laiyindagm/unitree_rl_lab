"""V3 probe: mirrors V2 probe + adds metric/axial diagnostics."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from frnc_pretrain import _parse_index_list
from frnc_segment_pretrain import SegmentDataset, collate_segments
from frnc_segment_pretrain_v2 import precompute_segment_labels
from frnc_segment_pretrain_v3 import SegmentEncoderV3
from frnc_gait_features import build_mask_indices, GAIT_FEATURE_NAMES
from frnc_segment_probe import (_bucket_label, bucket_pairwise_auc,
                                intra_bucket_var_ratio,
                                r2_z_step_freq, r2_z_cmd, intra_seg_z_var,
                                conditional_foot_probe)
from frnc_segment_probe_v2 import r2_z_to_features


def _encode_all(model, ds, device, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_segments)
    z_segs, cmd_segs = [], []
    e_frames, anc_frames, foot_frames = [], [], []
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            anc = batch["anchor_sc"].to(device)
            out = model(obs, mask, anc)                          # V3 forward (no adv_lambda)
            z_segs.append(out["z_gait"].cpu().numpy())
            mfloat = mask.cpu().numpy(); n_per = mfloat.sum(axis=1)
            cmd = batch["cmd"].numpy()
            cmd_seg = (cmd * mfloat[..., None]).sum(axis=1) / np.maximum(n_per[:, None], 1.0)
            cmd_segs.append(cmd_seg)
            e = out["e"].cpu().numpy()
            anc_np = batch["anchor_sc"].numpy()
            foot_np = batch["foot"].numpy()
            B = obs.shape[0]
            for i in range(B):
                L = int(n_per[i])
                e_frames.append(e[i, :L])
                anc_frames.append(anc_np[i, :L])
                foot_frames.append(foot_np[i, :L])
    return {
        "z_seg": np.concatenate(z_segs, 0),
        "cmd_seg": np.concatenate(cmd_segs, 0),
        "e_frames": e_frames, "anc_frames": anc_frames, "foot_frames": foot_frames,
    }


def axial_r2_np(z, cmd, sigma, axial_bases):
    v_w = cmd / np.clip(sigma[None, :], 1e-3, None)
    norm = np.clip(np.linalg.norm(v_w, axis=-1, keepdims=True), 1e-3, None)
    rho = v_w / norm
    z_pred = rho @ axial_bases
    ss_res = float(((z - z_pred) ** 2).sum())
    ss_tot = float(((z - z.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def magnitude_lipschitz_median(z, cmd, sigma, cos_thr=0.95, mag_min=0.2, max_pairs=2000):
    v_w = cmd / np.clip(sigma[None, :], 1e-3, None)
    lam = np.linalg.norm(v_w, axis=-1)
    mvalid = lam > 1e-3
    if mvalid.sum() < 2: return float("nan")
    v_dir = v_w / np.clip(lam[:, None], 1e-3, None)
    cos = v_dir @ v_dir.T
    dlam = np.abs(lam[None, :] - lam[:, None])
    M = (cos > cos_thr) & (dlam > mag_min) & mvalid[None, :] & mvalid[:, None]
    np.fill_diagonal(M, False)
    pairs = np.argwhere(M)
    if pairs.size == 0: return float("nan")
    if len(pairs) > max_pairs:
        pairs = pairs[np.random.permutation(len(pairs))[:max_pairs]]
    i, j = pairs[:, 0], pairs[:, 1]
    dz = np.linalg.norm(z[i] - z[j], axis=-1)
    dl = np.abs(lam[i] - lam[j])
    return float(np.median(dz / np.clip(dl, 1e-3, None)))


def cmd_distance_spearman(z, cmd, sigma, max_n=1500):
    n = min(len(z), max_n)
    perm = np.random.permutation(len(z))[:n]
    z = z[perm]; cmd = cmd[perm]
    v_w = cmd / np.clip(sigma[None, :], 1e-3, None)
    iu = np.triu_indices(n, k=1)
    dz = np.linalg.norm(z[iu[0]] - z[iu[1]], axis=-1)
    dv = np.linalg.norm(v_w[iu[0]] - v_w[iu[1]], axis=-1)
    return float(np.corrcoef(np.argsort(np.argsort(dz)), np.argsort(np.argsort(dv)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mask_kind", type=str, default=None)
    ap.add_argument("--phase_joint_slice", type=str, default=None)
    ap.add_argument("--phase_anchor_joint_idx", type=int, default=None)
    ap.add_argument("--segment_min_len", type=int, default=32)
    ap.add_argument("--segment_max_len", type=int, default=64)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mask_kind = args.mask_kind or cfg.get("mask_kind", "strict")
    mask_spec = build_mask_indices(mask_kind)
    mask_idx = _parse_index_list(mask_spec) if mask_spec else None

    model = SegmentEncoderV3(
        in_dim=cfg["in_dim"], d_back=cfg["d_back"], d_gait=cfg["d_gait"],
        phase_dim=cfg["phase_dim"], foot_dim=cfg.get("foot_dim", 2),
        prop_dim=cfg.get("prop_dim", 0),
        hard_seg_mean=bool(cfg.get("hard_seg_mean", 1)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    phase_slice_str = args.phase_joint_slice or cfg.get("phase_joint_slice", "245:260")
    phase_slice = tuple(int(x) for x in phase_slice_str.split(":"))
    anchor_idx = args.phase_anchor_joint_idx if args.phase_anchor_joint_idx is not None \
                 else cfg.get("phase_anchor_joint_idx", 0)

    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    if args.max_shards: shards = shards[:args.max_shards]
    ds = SegmentDataset(shards, mask_obs_indices=mask_idx,
                        phase_joint_slice=phase_slice,
                        phase_anchor_joint_idx=anchor_idx,
                        segment_min_len=args.segment_min_len,
                        segment_max_len=args.segment_max_len)
    print(f"[probe-v3] {len(ds)} segments, mask_kind={mask_kind}")

    enc = _encode_all(model, ds, device)
    z_seg = enc["z_seg"]; cmd_seg = enc["cmd_seg"]
    labels = _bucket_label(cmd_seg, None, None)
    seg_lab = precompute_segment_labels(ds)
    feats_raw, valid = seg_lab["feats_raw"], seg_lab["valid"]

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

    # ----- V3-only metric/axial diagnostics -----
    sigma = np.array(cfg["sigma_cmd"], dtype=np.float32)
    bases = model.axial_bases.detach().cpu().numpy()
    results["axial_R2"] = axial_r2_np(z_seg, cmd_seg, sigma, bases)
    results["magnitude_lipschitz_median"] = magnitude_lipschitz_median(z_seg, cmd_seg, sigma)
    results["cmd_distance_spearman"] = cmd_distance_spearman(z_seg, cmd_seg, sigma)
    results["sigma_cmd"] = cfg["sigma_cmd"]

    print(json.dumps(results, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump(results, open(args.out_json, "w"), indent=2)
        print(f"[probe-v3] wrote {args.out_json}")


if __name__ == "__main__":
    main()
