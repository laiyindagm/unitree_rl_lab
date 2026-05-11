# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Build frnc_style_v4 feature shards from rollout shards.

Input shards come from collect_style_pretrain_data.py.  Output samples use a
parent segment plus two phase-shifted windows:

  * obs_a / obs_b: runtime windows for encoder input.
  * y0: phase-invariant gait-style targets computed over the parent segment.
  * yphi_a / yphi_b with phi_a / phi_b: phase-dependent targets.

No command or mode term is used as a style target.  Commands are saved only for
conditional diagnostics and conditional RnC.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from style_obs_layout import latest_term, mode_id_from_cmd


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


def _load_shard_arrays(path: str) -> dict[str, np.ndarray]:
    needed = {
        "policy_obs",
        "cmd",
        "foot_contact",
        "action",
        "root_pos_w",
        "foot_pos_w",
        "foot_lin_vel_w",
        "env_id",
        "episode_step",
        "global_step",
    }
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files if k in needed}


def _find_shards(data_dir: str) -> list[str]:
    direct = glob.glob(os.path.join(data_dir, "shard_*.npz"))
    recursive = glob.glob(os.path.join(data_dir, "**", "shard_*.npz"), recursive=True)
    return sorted(set(direct + recursive))


def _iter_runs(d, min_len: int) -> Iterable[np.ndarray]:
    envs = d["env_id"].astype(np.int64)
    gst = d["global_step"].astype(np.int64)
    est = d["episode_step"].astype(np.int64)
    for eid in np.unique(envs):
        rows = np.where(envs == eid)[0]
        order = rows[np.argsort(gst[rows])]
        es = est[order]
        starts = np.concatenate([[0], np.where(es == 0)[0]])
        starts = np.unique(starts)
        ends = np.concatenate([starts[1:], [len(order)]])
        for s, e in zip(starts, ends):
            run = order[s:e]
            if len(run) >= min_len:
                yield run


def _finite_mean(x, axis=None):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).any():
        return np.nan
    return np.nanmean(x, axis=axis)


def _safe_std(x, axis=None):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).any():
        return np.nan
    return np.nanstd(x, axis=axis)


def _dominant_freq(x: np.ndarray, dt: float, min_std: float = 1e-3) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 8 or not np.isfinite(x).all() or float(np.std(x)) < min_std:
        return np.nan
    y = x - x.mean()
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=dt)
    if len(spec) <= 1:
        return np.nan
    idx = int(np.argmax(spec[1:]) + 1)
    return float(freqs[idx])


def _contact_lag(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32) - float(np.mean(a))
    b = np.asarray(b, dtype=np.float32) - float(np.mean(b))
    if float(np.std(a)) < 1e-3 or float(np.std(b)) < 1e-3:
        return np.nan
    corr = np.correlate(a, b, mode="full")
    lag = int(np.argmax(corr) - (len(a) - 1))
    return float(lag / max(len(a), 1))


def _phase_from_parent(joint_pos: np.ndarray, contact: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import hilbert
    except Exception:
        hilbert = None

    sig = joint_pos[:, 0].astype(np.float32)
    if float(np.std(sig)) < 1e-3 and contact is not None and contact.shape[1] > 0:
        sig = contact[:, 0].astype(np.float32) * 2.0 - 1.0
    phi_sc = np.zeros((len(sig), 2), dtype=np.float32)
    valid = np.zeros((len(sig),), dtype=np.float32)
    if hilbert is None or len(sig) < 16 or float(np.std(sig)) < 1e-3:
        return phi_sc, valid

    x = sig - sig.mean()
    try:
        phi = np.angle(hilbert(x))
    except Exception:
        return phi_sc, valid
    phi_sc[:, 0] = np.sin(phi).astype(np.float32)
    phi_sc[:, 1] = np.cos(phi).astype(np.float32)
    edge = max(4, len(sig) // 10)
    if edge * 2 < len(sig):
        valid[edge:-edge] = 1.0
    return phi_sc, valid


@dataclass(frozen=True)
class FeatureSchema:
    n_feet: int
    has_foot_pos: bool
    has_foot_vel: bool
    has_root_pos: bool
    action_dim: int

    @property
    def y0_names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.n_feet):
            names += [f"duty_f{i}", f"switch_rate_f{i}", f"step_freq_f{i}"]
        if self.n_feet >= 2:
            names += [
                "double_support_ratio",
                "no_support_ratio",
                "single_support_f0",
                "single_support_f1",
                "contact_balance",
                "contact_phase_lag",
            ]
        if self.has_foot_pos:
            for i in range(self.n_feet):
                names += [
                    f"foot_height_mean_f{i}",
                    f"foot_height_std_f{i}",
                    f"foot_height_amp_f{i}",
                    f"foot_clearance95_f{i}",
                ]
            if self.n_feet >= 2:
                names += ["foot_width_mean", "foot_width_std"]
        if self.has_foot_vel:
            for i in range(self.n_feet):
                names.append(f"foot_slip_contact_f{i}")
        if self.has_root_pos:
            names += ["root_height_mean", "root_height_std"]
        for i in range(3):
            names += [f"gravity_mean_{i}", f"gravity_std_{i}"]
        for j in range(15):
            names += [f"joint_mean_{j}", f"joint_std_{j}", f"joint_rom_{j}"]
        for j in range(self.action_dim):
            names += [f"action_rms_{j}", f"action_delta_rms_{j}"]
        names += ["action_energy_mean", "action_delta_energy_mean"]
        return names

    @property
    def yphi_names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.n_feet):
            names.append(f"contact_f{i}")
        for j in range(15):
            names.append(f"joint_pos_{j}")
        for i in range(3):
            names.append(f"gravity_{i}")
        for j in range(self.action_dim):
            names.append(f"action_{j}")
        if self.has_foot_pos:
            for i in range(self.n_feet):
                names.append(f"foot_height_f{i}")
            for i in range(self.n_feet):
                names += [f"foot_rel_x_f{i}", f"foot_rel_y_f{i}"]
        return names


def infer_schema(paths: list[str]) -> FeatureSchema:
    if not paths:
        raise ValueError("no input shards")
    n_feet = 0
    has_foot_pos = False
    has_foot_vel = False
    has_root_pos = False
    action_dim = 15
    for path in paths:
        d = _load_npz(path)
        if "foot_contact" in d:
            n_feet = max(n_feet, min(2, int(d["foot_contact"].shape[1])))
        has_foot_pos = has_foot_pos or ("foot_pos_w" in d and "root_pos_w" in d)
        has_foot_vel = has_foot_vel or ("foot_lin_vel_w" in d)
        has_root_pos = has_root_pos or ("root_pos_w" in d)
        if "action" in d:
            action_dim = int(d["action"].shape[-1])
        d.close()
    if n_feet == 0:
        raise ValueError("input shards must include foot_contact")
    return FeatureSchema(
        n_feet=n_feet,
        has_foot_pos=has_foot_pos,
        has_foot_vel=has_foot_vel,
        has_root_pos=has_root_pos,
        action_dim=action_dim,
    )


def _take(d, key: str, rows: np.ndarray):
    if key not in d:
        return None
    return d[key][rows].astype(np.float32)


def _actions_for(d, obs: np.ndarray, rows: np.ndarray, action_dim: int) -> np.ndarray:
    if "action" in d:
        return d["action"][rows].astype(np.float32)
    act = latest_term(obs, "last_action").astype(np.float32)
    if act.shape[1] != action_dim:
        return np.zeros((obs.shape[0], action_dim), dtype=np.float32)
    return act


def _foot_rel(d, rows: np.ndarray) -> np.ndarray | None:
    if "foot_pos_w" not in d or "root_pos_w" not in d:
        return None
    foot = d["foot_pos_w"][rows].astype(np.float32)
    root = d["root_pos_w"][rows].astype(np.float32)
    return foot - root[:, None, :]


def _foot_vel(d, rows: np.ndarray) -> np.ndarray | None:
    if "foot_lin_vel_w" not in d:
        return None
    return d["foot_lin_vel_w"][rows].astype(np.float32)


def compute_y0(d, rows: np.ndarray, schema: FeatureSchema, dt: float) -> tuple[np.ndarray, np.ndarray]:
    obs = d["policy_obs"][rows].astype(np.float32)
    contact = d["foot_contact"][rows, : schema.n_feet].astype(np.float32)
    joint = latest_term(obs, "joint_pos_rel").astype(np.float32)
    grav = latest_term(obs, "projected_gravity").astype(np.float32)
    action = _actions_for(d, obs, rows, schema.action_dim)
    foot_rel = _foot_rel(d, rows)
    foot_vel = _foot_vel(d, rows)
    root_pos = _take(d, "root_pos_w", rows)

    vals: list[float] = []
    valid: list[float] = []

    def add(x, ok=True):
        x = float(x) if np.ndim(x) == 0 else float(np.asarray(x).reshape(-1)[0])
        ok = bool(ok) and np.isfinite(x)
        vals.append(x if ok else 0.0)
        valid.append(1.0 if ok else 0.0)

    for i in range(schema.n_feet):
        c = contact[:, i]
        add(np.mean(c))
        add(np.mean(np.abs(np.diff(c))) if len(c) > 1 else np.nan)
        add(_dominant_freq(c, dt))

    if schema.n_feet >= 2:
        c0 = contact[:, 0] > 0.5
        c1 = contact[:, 1] > 0.5
        add(np.mean(c0 & c1))
        add(np.mean((~c0) & (~c1)))
        add(np.mean(c0 & (~c1)))
        add(np.mean((~c0) & c1))
        add(abs(float(np.mean(c0)) - float(np.mean(c1))))
        add(_contact_lag(contact[:, 0], contact[:, 1]))

    if schema.has_foot_pos:
        if foot_rel is None:
            foot_rel = np.full((len(rows), schema.n_feet, 3), np.nan, dtype=np.float32)
        foot_rel = foot_rel[:, : schema.n_feet]
        foot_h = foot_rel[:, :, 2]
        for i in range(schema.n_feet):
            h = foot_h[:, i]
            add(_finite_mean(h))
            add(_safe_std(h))
            add(np.nanpercentile(h, 95) - np.nanpercentile(h, 5), np.isfinite(h).any())
            add(np.nanpercentile(h, 95), np.isfinite(h).any())
        if schema.n_feet >= 2:
            width = np.abs(foot_rel[:, 0, 1] - foot_rel[:, 1, 1])
            add(_finite_mean(width))
            add(_safe_std(width))

    if schema.has_foot_vel:
        if foot_vel is None:
            foot_vel = np.full((len(rows), schema.n_feet, 3), np.nan, dtype=np.float32)
        foot_vel = foot_vel[:, : schema.n_feet]
        for i in range(schema.n_feet):
            speed_xy = np.linalg.norm(foot_vel[:, i, :2], axis=-1)
            m = contact[:, i] > 0.5
            add(np.mean(speed_xy[m]) if np.any(m) else np.nan)

    if schema.has_root_pos:
        if root_pos is None:
            add(np.nan)
            add(np.nan)
        else:
            add(_finite_mean(root_pos[:, 2]))
            add(_safe_std(root_pos[:, 2]))

    for i in range(3):
        add(_finite_mean(grav[:, i]))
        add(_safe_std(grav[:, i]))

    for j in range(15):
        q = joint[:, j]
        add(_finite_mean(q))
        add(_safe_std(q))
        add(np.nanpercentile(q, 95) - np.nanpercentile(q, 5), np.isfinite(q).any())

    if action.shape[1] != schema.action_dim:
        action = np.zeros((len(rows), schema.action_dim), dtype=np.float32)
    da = np.diff(action, axis=0) if len(action) > 1 else np.zeros_like(action[:1])
    for j in range(schema.action_dim):
        add(np.sqrt(np.mean(np.square(action[:, j]))))
        add(np.sqrt(np.mean(np.square(da[:, j]))))
    add(np.mean(np.sum(np.square(action), axis=-1)))
    add(np.mean(np.sum(np.square(da), axis=-1)))

    return np.asarray(vals, dtype=np.float32), np.asarray(valid, dtype=np.float32)


def compute_yphi(d, rows: np.ndarray, schema: FeatureSchema) -> tuple[np.ndarray, np.ndarray]:
    obs = d["policy_obs"][rows].astype(np.float32)
    T = len(rows)
    values: list[np.ndarray] = []
    valid: list[np.ndarray] = []

    contact = d["foot_contact"][rows, : schema.n_feet].astype(np.float32)
    values.append(contact)
    valid.append(np.ones_like(contact, dtype=np.float32))

    joint = latest_term(obs, "joint_pos_rel").astype(np.float32)
    values.append(joint)
    valid.append(np.ones_like(joint, dtype=np.float32))

    grav = latest_term(obs, "projected_gravity").astype(np.float32)
    values.append(grav)
    valid.append(np.ones_like(grav, dtype=np.float32))

    action = _actions_for(d, obs, rows, schema.action_dim)
    if action.shape[1] != schema.action_dim:
        action = np.zeros((T, schema.action_dim), dtype=np.float32)
    values.append(action.astype(np.float32))
    valid.append(np.ones_like(action, dtype=np.float32))

    if schema.has_foot_pos:
        foot_rel = _foot_rel(d, rows)
        if foot_rel is None:
            foot_rel = np.full((T, schema.n_feet, 3), np.nan, dtype=np.float32)
        foot_rel = foot_rel[:, : schema.n_feet]
        foot_h = foot_rel[:, :, 2]
        foot_xy = foot_rel[:, :, :2].reshape(T, schema.n_feet * 2)
        values += [foot_h.astype(np.float32), foot_xy.astype(np.float32)]
        valid += [
            np.isfinite(foot_h).astype(np.float32),
            np.isfinite(foot_xy).astype(np.float32),
        ]

    y = np.concatenate(values, axis=-1).astype(np.float32)
    v = np.concatenate(valid, axis=-1).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, v


class RunningStats:
    def __init__(self, dim: int):
        self.count = np.zeros((dim,), dtype=np.float64)
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sumsq = np.zeros((dim,), dtype=np.float64)

    def update(self, x: np.ndarray, valid: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        valid = np.asarray(valid, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
            valid = valid[None, :]
        self.count += valid.sum(axis=0)
        self.sum += (x * valid).sum(axis=0)
        self.sumsq += ((x ** 2) * valid).sum(axis=0)

    def finalize(self):
        count = np.maximum(self.count, 1.0)
        mean = self.sum / count
        var = np.maximum(self.sumsq / count - mean ** 2, 1e-6)
        std = np.sqrt(var)
        std[self.count < 2] = 1.0
        mean[self.count == 0] = 0.0
        return mean.astype(np.float32), std.astype(np.float32), self.count.astype(np.int64)


def bucket_id_from_cmd(cmd: np.ndarray, eps_xy: float = 0.1, eps_w: float = 0.1) -> int:
    mode = int(mode_id_from_cmd(cmd[None, :], eps_xy=eps_xy, eps_w=eps_w)[0])
    xy = float(np.linalg.norm(cmd[:2]))
    wz = abs(float(cmd[2]))
    if mode == 0:
        return 0
    if mode == 1:
        return 1
    if xy > eps_xy and wz > eps_w:
        return 5
    if xy < 0.35:
        return 2
    if xy < 0.8:
        return 3
    return 4


def main():
    ap = argparse.ArgumentParser(description="Build frnc_style_v4 feature shards.")
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--parent_len", type=int, default=64)
    ap.add_argument("--window_len", type=int, default=32)
    ap.add_argument("--parent_stride", type=int, default=32)
    ap.add_argument("--samples_per_parent", type=int, default=1)
    ap.add_argument("--out_shard_size", type=int, default=2048)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_compression", choices=["compressed", "stored"], default="compressed")
    ap.add_argument("--max_input_shards", type=int, default=None)
    ap.add_argument("--max_parents", type=int, default=None)
    args = ap.parse_args()

    if args.parent_len < args.window_len:
        raise ValueError("parent_len must be >= window_len")

    input_paths: list[str] = []
    for data_dir in args.data_dirs:
        input_paths.extend(_find_shards(data_dir))
    if args.max_input_shards is not None:
        input_paths = input_paths[: args.max_input_shards]
    if not input_paths:
        candidates = []
        for root in ["logs/style_pretrain_data", "logs/pretrain_data"]:
            for p in sorted(glob.glob(os.path.join(root, "*", "shard_*.npz"))):
                candidates.append(os.path.dirname(p))
        candidates = sorted(set(candidates))
        hint = f" Existing shard dirs: {candidates}" if candidates else ""
        raise FileNotFoundError(f"no shard_*.npz under {args.data_dirs}.{hint}")

    rng = np.random.default_rng(args.seed)
    schema = infer_schema(input_paths)
    y0_names = schema.y0_names
    yphi_names = schema.yphi_names
    y0_stats = RunningStats(len(y0_names))
    yphi_stats = RunningStats(len(yphi_names))

    os.makedirs(args.out_dir, exist_ok=True)
    buf: dict[str, list[np.ndarray]] = {
        "obs_a": [],
        "obs_b": [],
        "phi_a": [],
        "phi_b": [],
        "phi_valid_a": [],
        "phi_valid_b": [],
        "y0": [],
        "y0_valid": [],
        "yphi_a": [],
        "yphi_b": [],
        "yphi_valid_a": [],
        "yphi_valid_b": [],
        "cmd": [],
        "mode_id": [],
        "bucket_id": [],
        "source_id": [],
        "parent_id": [],
    }
    shard_idx = 0
    n_samples = 0
    parent_id = 0
    sample_counts_by_bucket: dict[str, int] = defaultdict(int)
    sample_counts_by_mode: dict[str, int] = defaultdict(int)
    sample_counts_by_source: dict[str, int] = defaultdict(int)
    sample_counts_by_source_bucket: dict[str, int] = defaultdict(int)
    parent_counts_by_bucket: dict[str, int] = defaultdict(int)
    parent_counts_by_source: dict[str, int] = defaultdict(int)

    def flush():
        nonlocal shard_idx
        if not buf["cmd"]:
            return
        out = {}
        for k, v in buf.items():
            out[k] = np.stack(v, axis=0)
        path = os.path.join(args.out_dir, f"style_shard_{shard_idx:05d}.npz")
        if args.save_compression == "stored":
            np.savez(path, **out)
        else:
            np.savez_compressed(path, **out)
        print(f"[style-features] wrote {path} ({out['cmd'].shape[0]} samples)", flush=True)
        shard_idx += 1
        for k in buf:
            buf[k] = []

    stop = False
    for source_id, path in enumerate(input_paths):
        d = _load_shard_arrays(path)
        print(f"[style-features] source {source_id + 1}/{len(input_paths)}: {path}", flush=True)
        for run in _iter_runs(d, min_len=args.parent_len):
            max_start = len(run) - args.parent_len
            for start in range(0, max_start + 1, args.parent_stride):
                parent_rows = run[start : start + args.parent_len]
                parent_obs = d["policy_obs"][parent_rows].astype(np.float32)
                parent_cmd = d["cmd"][parent_rows].astype(np.float32)
                parent_contact = d["foot_contact"][parent_rows, : schema.n_feet].astype(np.float32)
                parent_joint = latest_term(parent_obs, "joint_pos_rel").astype(np.float32)
                phi_parent, phi_valid_parent = _phase_from_parent(parent_joint, parent_contact)
                y0, y0_valid = compute_y0(d, parent_rows, schema, dt=args.dt)
                cmd_seg = np.mean(parent_cmd, axis=0).astype(np.float32)
                mode_id = int(mode_id_from_cmd(cmd_seg[None, :])[0])
                bucket_id = bucket_id_from_cmd(cmd_seg)
                parent_counts_by_bucket[str(bucket_id)] += 1
                parent_counts_by_source[str(source_id)] += 1

                y0_stats.update(y0, y0_valid)

                for _ in range(args.samples_per_parent):
                    hi = args.parent_len - args.window_len
                    off_a = int(rng.integers(0, hi + 1))
                    off_b = int(rng.integers(0, hi + 1))
                    if hi > 0 and off_b == off_a:
                        off_b = (off_a + max(1, hi // 2)) % (hi + 1)
                    rows_a = parent_rows[off_a : off_a + args.window_len]
                    rows_b = parent_rows[off_b : off_b + args.window_len]
                    yphi_a, yphi_valid_a = compute_yphi(d, rows_a, schema)
                    yphi_b, yphi_valid_b = compute_yphi(d, rows_b, schema)
                    phi_a = phi_parent[off_a : off_a + args.window_len]
                    phi_b = phi_parent[off_b : off_b + args.window_len]
                    pv_a = phi_valid_parent[off_a : off_a + args.window_len]
                    pv_b = phi_valid_parent[off_b : off_b + args.window_len]

                    yphi_stats.update(yphi_a.reshape(-1, len(yphi_names)), yphi_valid_a.reshape(-1, len(yphi_names)))
                    yphi_stats.update(yphi_b.reshape(-1, len(yphi_names)), yphi_valid_b.reshape(-1, len(yphi_names)))

                    buf["obs_a"].append(d["policy_obs"][rows_a].astype(np.float32))
                    buf["obs_b"].append(d["policy_obs"][rows_b].astype(np.float32))
                    buf["phi_a"].append(phi_a.astype(np.float32))
                    buf["phi_b"].append(phi_b.astype(np.float32))
                    buf["phi_valid_a"].append(pv_a.astype(np.float32))
                    buf["phi_valid_b"].append(pv_b.astype(np.float32))
                    buf["y0"].append(y0)
                    buf["y0_valid"].append(y0_valid)
                    buf["yphi_a"].append(yphi_a)
                    buf["yphi_b"].append(yphi_b)
                    buf["yphi_valid_a"].append(yphi_valid_a * pv_a[:, None])
                    buf["yphi_valid_b"].append(yphi_valid_b * pv_b[:, None])
                    buf["cmd"].append(cmd_seg)
                    buf["mode_id"].append(np.asarray(mode_id, dtype=np.int64))
                    buf["bucket_id"].append(np.asarray(bucket_id, dtype=np.int64))
                    buf["source_id"].append(np.asarray(source_id, dtype=np.int64))
                    buf["parent_id"].append(np.asarray(parent_id, dtype=np.int64))
                    n_samples += 1
                    sample_counts_by_bucket[str(bucket_id)] += 1
                    sample_counts_by_mode[str(mode_id)] += 1
                    sample_counts_by_source[str(source_id)] += 1
                    sample_counts_by_source_bucket[f"{source_id}:{bucket_id}"] += 1
                    if len(buf["cmd"]) >= args.out_shard_size:
                        flush()
                parent_id += 1
                if args.max_parents is not None and parent_id >= args.max_parents:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    flush()
    y0_mean, y0_std, y0_count = y0_stats.finalize()
    yphi_mean, yphi_std, yphi_count = yphi_stats.finalize()
    stats = {
        "schema": {
            "n_feet": schema.n_feet,
            "has_foot_pos": schema.has_foot_pos,
            "has_foot_vel": schema.has_foot_vel,
            "has_root_pos": schema.has_root_pos,
            "action_dim": schema.action_dim,
        },
        "input_paths": input_paths,
        "n_samples": int(n_samples),
        "n_parents": int(parent_id),
        "parent_len": args.parent_len,
        "window_len": args.window_len,
        "parent_stride": args.parent_stride,
        "samples_per_parent": args.samples_per_parent,
        "dt": args.dt,
        "y0_names": y0_names,
        "y0_mean": y0_mean.tolist(),
        "y0_std": y0_std.tolist(),
        "y0_count": y0_count.tolist(),
        "yphi_names": yphi_names,
        "yphi_mean": yphi_mean.tolist(),
        "yphi_std": yphi_std.tolist(),
        "yphi_count": yphi_count.tolist(),
        "bucket_names": ["standing", "pure_wz", "low_xy", "mid_xy", "high_xy", "mixed"],
        "sample_counts_by_bucket": dict(sorted(sample_counts_by_bucket.items())),
        "sample_counts_by_mode": dict(sorted(sample_counts_by_mode.items())),
        "sample_counts_by_source": dict(sorted(sample_counts_by_source.items())),
        "sample_counts_by_source_bucket": dict(sorted(sample_counts_by_source_bucket.items())),
        "parent_counts_by_bucket": dict(sorted(parent_counts_by_bucket.items())),
        "parent_counts_by_source": dict(sorted(parent_counts_by_source.items())),
    }
    with open(os.path.join(args.out_dir, "feature_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[style-features] done: {n_samples} samples, stats={os.path.join(args.out_dir, 'feature_stats.json')}", flush=True)


if __name__ == "__main__":
    main()
