# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Input-side diagnostic: how recoverable is each gait feature from the
(masked) raw obs?

For each `mask_kind` in {none, cmd, cmd_tokens, strict}:
  1. Build the SegmentDataset with that mask.
  2. Compute per-segment input representation = concat(mean_t obs, std_t obs)
     restricted to UNmasked dims (so the linear probe sees what the encoder
     would see).
  3. Linear probe input -> each gait feature, report R^2.

This tells us which "gait properties" are actually present in the input under
each masking regime, and is the upper bound that any encoder could achieve.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from frnc_segment_pretrain import SegmentDataset
from frnc_segment_pretrain_v2 import precompute_segment_labels
from frnc_pretrain import _parse_index_list
from frnc_gait_features import GAIT_FEATURE_NAMES, build_mask_indices


def segment_input_repr(ds: SegmentDataset, mask_idx):
    """Per-segment representation = (mean, std) of the per-frame obs over the
    segment, restricted to UNmasked indices.
    """
    N = len(ds)
    sample = ds.shards[ds.segments[0].shard_idx]["policy_obs"]
    D = sample.shape[1]
    if mask_idx is None:
        keep = np.arange(D)
    else:
        keep = np.setdiff1d(np.arange(D), mask_idx)
    feats = np.zeros((N, 2 * len(keep)), dtype=np.float32)
    for i in range(N):
        seg = ds.segments[i]
        c = ds.shards[seg.shard_idx]
        obs = c["policy_obs"][seg.rows][:, keep]
        feats[i, :len(keep)] = obs.mean(axis=0)
        feats[i, len(keep):] = obs.std(axis=0)
    return feats


def linear_probes(X, feats, valid, names):
    out = {}
    for j, name in enumerate(names):
        v = valid[:, j].astype(bool)
        if v.sum() < 100:
            out[name] = float("nan"); continue
        Xj = X[v]; yj = feats[v, j]
        if yj.std() < 1e-6:
            out[name] = float("nan"); continue
        idx = np.random.permutation(len(Xj))
        n = len(idx) // 2
        tr, te = idx[:n], idx[n:]
        try:
            reg = LinearRegression().fit(Xj[tr], yj[tr])
            out[name] = float(r2_score(yj[te], reg.predict(Xj[te])))
        except Exception:
            out[name] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--phase_joint_slice", type=str, default="245:260")
    ap.add_argument("--phase_anchor_joint_idx", type=int, default=0)
    ap.add_argument("--segment_min_len", type=int, default=32)
    ap.add_argument("--segment_max_len", type=int, default=64)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    if args.max_shards: shards = shards[:args.max_shards]
    phase_slice = tuple(int(x) for x in args.phase_joint_slice.split(":"))

    # Build dataset ONCE (mask not applied here; we manually subset).
    ds = SegmentDataset(shards, mask_obs_indices=None,
                        phase_joint_slice=phase_slice,
                        phase_anchor_joint_idx=args.phase_anchor_joint_idx,
                        segment_min_len=args.segment_min_len,
                        segment_max_len=args.segment_max_len)
    print(f"[diag] {len(ds)} segments")
    seg_lab = precompute_segment_labels(ds)
    feats_raw = seg_lab["feats_raw"]
    valid = seg_lab["valid"]

    # Also compute cmd_seg (segment mean of cmd) for an extra "R^2(input -> cmd)" probe.
    # cmd is at indices 6:9 of last frame; use first 3 cols of cmd buffer instead.
    cmd_segs = np.zeros((len(ds), 3), dtype=np.float32)
    for i in range(len(ds)):
        seg = ds.segments[i]
        c = ds.shards[seg.shard_idx]
        cmd_segs[i] = c["cmd"][seg.rows].mean(axis=0)
    cmd_valid = np.ones((len(ds), 3), dtype=np.float32)

    results = {"by_mask_kind": {}}
    for mk in ["none", "cmd", "cmd_tokens", "strict"]:
        spec = build_mask_indices(mk)
        mask_idx = _parse_index_list(spec) if spec else None
        n_mask = 0 if mask_idx is None else len(mask_idx)
        print(f"[diag] mask_kind={mk} ({n_mask} dims masked)")
        X = segment_input_repr(ds, mask_idx)
        gait = linear_probes(X, feats_raw, valid, GAIT_FEATURE_NAMES)
        cmd = linear_probes(X, cmd_segs, cmd_valid, ["cmd_vx", "cmd_vy", "cmd_wz"])
        results["by_mask_kind"][mk] = {
            "n_dims": int(X.shape[1]),
            "n_mask": int(n_mask),
            **{f"R2_{k}": v for k, v in gait.items()},
            **{f"R2_{k}": v for k, v in cmd.items()},
        }
    results["n_segments"] = int(len(ds))

    print(json.dumps(results, indent=2))
    if args.out_json:
        json.dump(results, open(args.out_json, "w"), indent=2)
        print(f"[diag] wrote {args.out_json}")


if __name__ == "__main__":
    main()
