# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Probe metrics for frnc_style_v4 encoders."""

from __future__ import annotations

import argparse
import json
import os
import re
import weakref

import numpy as np
import torch
from torch.utils.data import DataLoader

from style_encoder_pretrain_v4 import StyleEncoderV4, StyleWindowDataset, _apply_mask, cmd_delta
from style_obs_layout import build_mask_indices, randomize_shortcut_terms


_PROBE_BACKEND = "torch"
_PROBE_DEVICE = torch.device("cpu")
_MLP_PROBE_EPOCHS = 80
_MLP_PROBE_LR = 1e-3
_MLP_PROBE_BATCH_SIZE = 0
_TENSOR_CACHE: dict[tuple[int, tuple[int, ...], str, str], tuple[weakref.ReferenceType[np.ndarray], torch.Tensor]] = {}


def _torch_probe_enabled() -> bool:
    return _PROBE_BACKEND == "torch"


def _as_float_tensor(x: np.ndarray) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float32)
    key = (id(arr), tuple(arr.shape), arr.dtype.str, str(_PROBE_DEVICE))
    cached = _TENSOR_CACHE.get(key)
    if cached is not None and cached[0]() is arr:
        return cached[1]
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=_PROBE_DEVICE)
    if arr.nbytes >= 1 << 20:
        try:
            ref = weakref.ref(arr, lambda _ref, cache_key=key: _TENSOR_CACHE.pop(cache_key, None))
            _TENSOR_CACHE[key] = (ref, tensor)
        except TypeError:
            pass
    return tensor


def _torch_multi_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    scores = _torch_multi_r2_each(y_true, y_pred)
    valid = torch.isfinite(scores)
    if not bool(valid.any()):
        return torch.tensor(float("nan"), device=y_true.device)
    return scores[valid].mean()


def _torch_multi_r2_each(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    sse = torch.sum((y_true - y_pred) ** 2, dim=0)
    centered = y_true - y_true.mean(dim=0, keepdim=True)
    sst = torch.sum(centered**2, dim=0)
    scores = torch.full_like(sst, float("nan"))
    valid = sst > 1e-8
    scores[valid] = 1.0 - sse[valid] / sst[valid]
    return scores


def _torch_split_indices(idx: torch.Tensor, seed: int, min_train: int, min_test: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=idx.device)
    gen.manual_seed(int(seed))
    idx = idx[torch.randperm(idx.numel(), device=idx.device, generator=gen)]
    cut = max(min_train, int(0.8 * idx.numel()))
    if cut >= idx.numel():
        cut = idx.numel() - 1
    tr, te = idx[:cut], idx[cut:]
    if tr.numel() < min_train or te.numel() < min_test:
        return idx[:0], idx[:0]
    return tr, te


def _torch_ridge_beta(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)], dim=1)
    xtx = x_aug.T @ x_aug
    reg = torch.eye(xtx.shape[0], dtype=x.dtype, device=x.device) * float(alpha)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(xtx + reg, x_aug.T @ y)


def _torch_ridge_predict(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)], dim=1)
    return x_aug @ beta


def _onehot_mode(mode: np.ndarray) -> np.ndarray:
    mode = np.asarray(mode, dtype=np.int64).reshape(-1)
    out = np.zeros((len(mode), 3), dtype=np.float32)
    out[np.arange(len(mode)), np.clip(mode, 0, 2)] = 1.0
    return out


def _source_group_from_path(path: str) -> str:
    norm = path.replace("\\", "/")
    match = re.search(r"(model_\d+)", norm)
    if match:
        return match.group(1)
    parts = [p for p in norm.split("/") if p]
    for part in reversed(parts[:-1]):
        if "final" in part or part.startswith("v"):
            return part
    return os.path.basename(os.path.dirname(norm)) or os.path.basename(norm) or "source_unknown"


def _source_groups_for_dataset(ds: StyleWindowDataset) -> np.ndarray:
    input_paths = list(ds.stats.get("input_paths", []))
    source_id = np.asarray(getattr(ds, "source_id", np.zeros((len(ds),), dtype=np.int64)), dtype=np.int64).reshape(-1)
    labels = []
    for sid in source_id:
        if 0 <= int(sid) < len(input_paths):
            labels.append(_source_group_from_path(input_paths[int(sid)]))
        else:
            labels.append(f"source_{int(sid)}")
    return np.asarray(labels, dtype=object)


def _feature_groups(names: list[str], target_kind: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        if target_kind == "y0":
            if name.startswith(("duty_", "switch_rate_", "step_freq_", "double_support", "no_support", "single_support", "contact_")):
                key = "contact"
            elif name.startswith("foot_"):
                key = "foot"
            elif name.startswith(("root_", "gravity_")):
                key = "root_gravity"
            elif name.startswith("joint_"):
                key = "joint"
            elif name.startswith("action_") or "action_" in name:
                key = "action"
            else:
                key = "other"
        else:
            if name.startswith("contact_"):
                key = "contact"
            elif name.startswith("joint_"):
                key = "joint"
            elif name.startswith("gravity_"):
                key = "gravity"
            elif name.startswith("action_"):
                key = "action"
            elif name.startswith("foot_"):
                key = "foot"
            else:
                key = "other"
        groups.setdefault(key, []).append(i)
    return groups


def _mean_scores_for_groups(scores: list[float], groups: dict[str, list[int]]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float32)
    out: dict[str, float] = {}
    for name, idx in groups.items():
        vals = arr[np.asarray(idx, dtype=np.int64)]
        vals = vals[np.isfinite(vals)]
        out[name] = float(vals.mean()) if len(vals) else float("nan")
    return out


def _ridge_r2(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray | None = None,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    if _torch_probe_enabled():
        return _ridge_r2_torch(X, Y, valid, seed=seed, train_mask=train_mask, test_mask=test_mask)
    return _ridge_r2_sklearn(X, Y, valid, seed=seed, train_mask=train_mask, test_mask=test_mask)


def _ridge_r2_torch(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray | None = None,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    x = _as_float_tensor(X)
    y = _as_float_tensor(Y)
    if valid is None:
        valid_t = torch.ones_like(y, dtype=torch.bool)
    else:
        valid_t = _as_float_tensor(valid) > 0.5
    finite_x = torch.isfinite(x).all(dim=1)
    if train_mask is not None:
        train_mask_t = torch.as_tensor(np.asarray(train_mask, dtype=bool).reshape(-1), dtype=torch.bool, device=_PROBE_DEVICE)
    else:
        train_mask_t = None
    if test_mask is not None:
        test_mask_t = torch.as_tensor(np.asarray(test_mask, dtype=bool).reshape(-1), dtype=torch.bool, device=_PROBE_DEVICE)
    else:
        test_mask_t = None
    if (train_mask_t is None) != (test_mask_t is None):
        raise ValueError("train_mask and test_mask must be provided together")

    same_valid = y.shape[1] > 0 and bool((valid_t == valid_t[:, :1]).all().detach().cpu())
    if same_valid and train_mask_t is not None and test_mask_t is not None:
        row_mask = valid_t[:, 0] & finite_x & torch.isfinite(y).all(dim=1)
        idx = torch.nonzero(row_mask, as_tuple=False).flatten()
        tr = idx[train_mask_t[idx]]
        te = idx[test_mask_t[idx]]
        if tr.numel() >= 20 and te.numel() >= 20:
            with torch.no_grad():
                y_te = y[te]
                beta = _torch_ridge_beta(x[tr], y[tr], alpha=1.0)
                pred = _torch_ridge_predict(x[te], beta)
                scores_t = _torch_multi_r2_each(y_te, pred)
                std_t = torch.std(y_te, dim=0)
                scores = [
                    float(s.detach().cpu()) if bool(torch.isfinite(s).detach().cpu()) and float(std_t[i].detach().cpu()) >= 1e-6 else float("nan")
                    for i, s in enumerate(scores_t)
                ]
                finite = [s for s in scores if np.isfinite(s)]
                return (float(np.mean(finite)) if finite else float("nan")), scores

    scores: list[float] = []
    with torch.no_grad():
        for j in range(y.shape[1]):
            m = valid_t[:, j] & torch.isfinite(y[:, j]) & finite_x
            idx = torch.nonzero(m, as_tuple=False).flatten()
            if idx.numel() < 40:
                scores.append(float("nan"))
                continue
            if float(torch.std(y[idx, j]).detach().cpu()) < 1e-6:
                scores.append(float("nan"))
                continue
            if train_mask_t is not None and test_mask_t is not None:
                tr = idx[train_mask_t[idx]]
                te = idx[test_mask_t[idx]]
            else:
                tr, te = _torch_split_indices(idx, seed + j * 104729, min_train=20, min_test=20)
            if tr.numel() < 20 or te.numel() < 20:
                scores.append(float("nan"))
                continue
            y_te = y[te, j : j + 1]
            if float(torch.std(y_te).detach().cpu()) < 1e-6:
                scores.append(float("nan"))
                continue
            beta = _torch_ridge_beta(x[tr], y[tr, j : j + 1], alpha=1.0)
            pred = _torch_ridge_predict(x[te], beta)
            r2 = _torch_multi_r2(y_te, pred)
            scores.append(float(r2.detach().cpu()) if bool(torch.isfinite(r2).detach().cpu()) else float("nan"))
    finite = [s for s in scores if np.isfinite(s)]
    return (float(np.mean(finite)) if finite else float("nan")), scores


def _ridge_r2_sklearn(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray | None = None,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if valid is None:
        valid = np.ones_like(Y, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    if train_mask is not None:
        train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool).reshape(-1)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for j in range(Y.shape[1]):
        m = (valid[:, j] > 0.5) & np.isfinite(Y[:, j]) & np.isfinite(X).all(axis=1)
        idx = np.where(m)[0]
        if len(idx) < 40 or float(np.std(Y[idx, j])) < 1e-6:
            scores.append(float("nan"))
            continue
        if train_mask is not None or test_mask is not None:
            if train_mask is None or test_mask is None:
                raise ValueError("train_mask and test_mask must be provided together")
            tr = idx[train_mask[idx]]
            te = idx[test_mask[idx]]
        else:
            idx = idx[rng.permutation(len(idx))]
            cut = max(20, int(0.8 * len(idx)))
            if cut >= len(idx):
                cut = len(idx) - 1
            tr, te = idx[:cut], idx[cut:]
        if len(tr) < 20 or len(te) < 20 or float(np.std(Y[te, j])) < 1e-6:
            scores.append(float("nan"))
            continue
        model = Ridge(alpha=1.0)
        model.fit(X[tr], Y[tr, j])
        pred = model.predict(X[te])
        scores.append(float(r2_score(Y[te, j], pred)))
    finite = [s for s in scores if np.isfinite(s)]
    return (float(np.mean(finite)) if finite else float("nan")), scores


def _mlp_r2(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray | None = None,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
    max_samples: int = 20000,
) -> float:
    if _torch_probe_enabled():
        return _mlp_r2_torch(X, Y, valid, seed=seed, train_mask=train_mask, test_mask=test_mask, max_samples=max_samples)
    return _mlp_r2_sklearn(X, Y, valid, seed=seed, train_mask=train_mask, test_mask=test_mask, max_samples=max_samples)


def _mlp_r2_torch(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray | None = None,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
    max_samples: int = 20000,
) -> float:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if valid is None:
        row_valid = np.ones((len(X),), dtype=bool)
    else:
        row_valid = (np.asarray(valid, dtype=np.float32) > 0.5).all(axis=1)
    row_valid = row_valid & np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if train_mask is not None and test_mask is not None:
        tr = np.where(row_valid & np.asarray(train_mask, dtype=bool).reshape(-1))[0]
        te = np.where(row_valid & np.asarray(test_mask, dtype=bool).reshape(-1))[0]
    else:
        idx = np.where(row_valid)[0]
        if len(idx) < 80:
            return float("nan")
        rng = np.random.default_rng(seed)
        idx = idx[rng.permutation(len(idx))]
        cut = max(40, int(0.8 * len(idx)))
        if cut >= len(idx):
            cut = len(idx) - 1
        tr, te = idx[:cut], idx[cut:]
    if len(tr) < 40 or len(te) < 40:
        return float("nan")

    rng = np.random.default_rng(seed)
    if max_samples is not None and max_samples > 0:
        if len(tr) > max_samples:
            tr = rng.choice(tr, size=max_samples, replace=False)
        if len(te) > max_samples:
            te = rng.choice(te, size=max_samples, replace=False)

    torch.manual_seed(int(seed))
    xtr = _as_float_tensor(X[tr])
    xte = _as_float_tensor(X[te])
    ytr = _as_float_tensor(Y[tr])
    yte = _as_float_tensor(Y[te])
    x_mean = xtr.mean(dim=0, keepdim=True)
    x_std = xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = ytr.mean(dim=0, keepdim=True)
    y_std = ytr.std(dim=0, keepdim=True).clamp_min(1e-6)
    xtr = (xtr - x_mean) / x_std
    xte = (xte - x_mean) / x_std
    ytr = (ytr - y_mean) / y_std
    yte = (yte - y_mean) / y_std

    model = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, ytr.shape[1]),
    ).to(_PROBE_DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=_MLP_PROBE_LR, weight_decay=1e-4)
    batch_size = int(_MLP_PROBE_BATCH_SIZE)
    use_minibatch = batch_size > 0 and batch_size < xtr.shape[0]
    gen = torch.Generator(device=_PROBE_DEVICE)
    gen.manual_seed(int(seed) + 97)
    for _ in range(int(_MLP_PROBE_EPOCHS)):
        if use_minibatch:
            perm = torch.randperm(xtr.shape[0], device=_PROBE_DEVICE, generator=gen)
            for start in range(0, xtr.shape[0], batch_size):
                idx = perm[start : start + batch_size]
                pred = model(xtr[idx])
                loss = torch.nn.functional.mse_loss(pred, ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        else:
            pred = model(xtr)
            loss = torch.nn.functional.mse_loss(pred, ytr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = model(xte)
        r2 = _torch_multi_r2(yte, pred)
    return float(r2.detach().cpu()) if bool(torch.isfinite(r2).detach().cpu()) else float("nan")


def _mlp_r2_sklearn(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray | None = None,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
    max_samples: int = 20000,
) -> float:
    from sklearn.metrics import r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if valid is None:
        row_valid = np.ones((len(X),), dtype=bool)
    else:
        row_valid = (np.asarray(valid, dtype=np.float32) > 0.5).all(axis=1)
    row_valid = row_valid & np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if train_mask is not None and test_mask is not None:
        tr = np.where(row_valid & np.asarray(train_mask, dtype=bool).reshape(-1))[0]
        te = np.where(row_valid & np.asarray(test_mask, dtype=bool).reshape(-1))[0]
    else:
        idx = np.where(row_valid)[0]
        if len(idx) < 80:
            return float("nan")
        rng = np.random.default_rng(seed)
        idx = idx[rng.permutation(len(idx))]
        cut = max(40, int(0.8 * len(idx)))
        if cut >= len(idx):
            cut = len(idx) - 1
        tr, te = idx[:cut], idx[cut:]
    if len(tr) < 40 or len(te) < 40:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(tr) > max_samples:
        tr = rng.choice(tr, size=max_samples, replace=False)
    if len(te) > max_samples:
        te = rng.choice(te, size=max_samples, replace=False)
    sx = StandardScaler()
    sy = StandardScaler()
    xtr = sx.fit_transform(X[tr])
    xte = sx.transform(X[te])
    ytr = sy.fit_transform(Y[tr])
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", alpha=1e-4, max_iter=80, random_state=seed)
    model.fit(xtr, ytr)
    pred = sy.inverse_transform(model.predict(xte).reshape(len(te), -1))
    return float(r2_score(Y[te], pred, multioutput="uniform_average"))


def _phase_target(phi: np.ndarray, phi_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = phi_valid.astype(np.float32)
    denom = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    sc = (phi * w[..., None]).sum(axis=1) / denom
    valid = (w.sum(axis=1) >= max(4, phi.shape[1] // 4)).astype(np.float32)
    return sc.astype(np.float32), np.repeat(valid[:, None], 2, axis=1)


def _effective_rank(z: np.ndarray) -> float:
    if _torch_probe_enabled():
        with torch.no_grad():
            x = _as_float_tensor(z)
            x = x - x.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(x)
            power = s**2
            denom = torch.sum(power).clamp_min(1e-12)
            p = power / denom
            p = p[p > 1e-12]
            if p.numel() == 0:
                return float("nan")
            rank = torch.exp(-(p * torch.log(p)).sum())
            return float(rank.detach().cpu())
    x = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = (s ** 2) / max(float(np.sum(s ** 2)), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def _shift_ratio(z_a: np.ndarray, z_b: np.ndarray, seed: int = 0) -> float:
    if _torch_probe_enabled():
        with torch.no_grad():
            a = _as_float_tensor(z_a)
            b = _as_float_tensor(z_b)
            n = a.shape[0]
            if n < 2:
                return float("nan")
            gen = torch.Generator(device=_PROBE_DEVICE)
            gen.manual_seed(int(seed))
            perm = torch.randperm(n, device=_PROBE_DEVICE, generator=gen)
            same = torch.linalg.norm(a - b, dim=-1).mean()
            diff = torch.linalg.norm(a - a[perm], dim=-1).mean()
            return float((same / diff.clamp_min(1e-8)).detach().cpu())
    rng = np.random.default_rng(seed)
    n = len(z_a)
    if n < 2:
        return float("nan")
    same = np.linalg.norm(z_a - z_b, axis=-1).mean()
    perm = rng.permutation(n)
    diff = np.linalg.norm(z_a - z_a[perm], axis=-1).mean()
    return float(same / max(diff, 1e-8))


def _linear_residualize(
    X: np.ndarray,
    Z: np.ndarray,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if _torch_probe_enabled():
        return _linear_residualize_torch(X, Z, seed=seed, train_mask=train_mask)
    return _linear_residualize_sklearn(X, Z, seed=seed, train_mask=train_mask)


def _linear_residualize_torch(
    X: np.ndarray,
    Z: np.ndarray,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    X_np = np.asarray(X, dtype=np.float32)
    Z_np = np.asarray(Z, dtype=np.float32)
    row_valid = np.isfinite(X_np).all(axis=1) & np.isfinite(Z_np).all(axis=1)
    if train_mask is not None:
        tr_np = np.where(row_valid & np.asarray(train_mask, dtype=bool).reshape(-1))[0]
    else:
        idx = np.where(row_valid)[0]
        rng = np.random.default_rng(seed)
        idx = idx[rng.permutation(len(idx))]
        cut = max(40, int(0.8 * len(idx)))
        if cut >= len(idx):
            cut = len(idx) - 1
        tr_np = idx[:cut]
    if len(tr_np) < 40:
        return Z_np.copy(), {"R2_Z_from_cmd_mode_linear": float("nan")}

    with torch.no_grad():
        x = _as_float_tensor(X_np)
        z = _as_float_tensor(Z_np)
        tr = torch.as_tensor(tr_np, dtype=torch.long, device=_PROBE_DEVICE)
        beta = _torch_ridge_beta(x[tr], z[tr], alpha=1.0)
        pred = _torch_ridge_predict(x, beta)
        r2_train = _torch_multi_r2(z[tr], pred[tr])
        residual = (z - pred).detach().cpu().numpy().astype(np.float32)
    score = float(r2_train.detach().cpu()) if bool(torch.isfinite(r2_train).detach().cpu()) else float("nan")
    return residual, {"R2_Z_from_cmd_mode_linear": score}


def _linear_residualize_sklearn(
    X: np.ndarray,
    Z: np.ndarray,
    seed: int = 0,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    row_valid = np.isfinite(X).all(axis=1) & np.isfinite(Z).all(axis=1)
    if train_mask is not None:
        tr = np.where(row_valid & np.asarray(train_mask, dtype=bool).reshape(-1))[0]
    else:
        idx = np.where(row_valid)[0]
        rng = np.random.default_rng(seed)
        idx = idx[rng.permutation(len(idx))]
        cut = max(40, int(0.8 * len(idx)))
        if cut >= len(idx):
            cut = len(idx) - 1
        tr = idx[:cut]
    if len(tr) < 40:
        return Z.copy(), {"R2_Z_from_cmd_mode_linear": float("nan")}
    model = Ridge(alpha=1.0)
    model.fit(X[tr], Z[tr])
    pred = np.asarray(model.predict(X), dtype=np.float32)
    r2_train = float(r2_score(Z[tr], pred[tr], multioutput="uniform_average"))
    return Z - pred, {"R2_Z_from_cmd_mode_linear": r2_train}


def _phase_shift_metrics(
    z: np.ndarray,
    parent: np.ndarray,
    bucket: np.ndarray,
    mode: np.ndarray,
    seed: int = 0,
    max_pairs: int = 20000,
) -> dict[str, float | int]:
    z = np.asarray(z, dtype=np.float32)
    parent = np.asarray(parent).reshape(-1)
    bucket = np.asarray(bucket).reshape(-1)
    mode = np.asarray(mode).reshape(-1)
    if len(z) < 2 or len(np.unique(parent)) <= 1:
        return {
            "phase_shift_n_parent_groups": 0,
            "phase_shift_n_pairs": 0,
            "phase_shift_same_parent_dist": float("nan"),
            "phase_shift_same_bucket_dist": float("nan"),
            "phase_shift_ratio": float("nan"),
        }

    groups = [np.where(parent == p)[0] for p in np.unique(parent)]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return {
            "phase_shift_n_parent_groups": 0,
            "phase_shift_n_pairs": 0,
            "phase_shift_same_parent_dist": float("nan"),
            "phase_shift_same_bucket_dist": float("nan"),
            "phase_shift_ratio": float("nan"),
        }

    rng = np.random.default_rng(seed)
    n_pairs = min(max_pairs, max(len(groups), 1) * 8)
    same_dists = []
    ref_dists = []
    z_t = _as_float_tensor(z) if _torch_probe_enabled() else None
    finite_z = np.isfinite(z).all(axis=1)
    for _ in range(n_pairs):
        g = groups[int(rng.integers(0, len(groups)))]
        i, j = rng.choice(g, size=2, replace=False)
        if z_t is not None:
            same_dists.append(float(torch.linalg.norm(z_t[int(i)] - z_t[int(j)]).detach().cpu()))
        else:
            same_dists.append(float(np.linalg.norm(z[i] - z[j])))
        m = (bucket == bucket[i]) & (mode == mode[i]) & (parent != parent[i]) & finite_z
        cand = np.where(m)[0]
        if len(cand) > 0:
            k = int(cand[int(rng.integers(0, len(cand)))])
            if z_t is not None:
                ref_dists.append(float(torch.linalg.norm(z_t[int(i)] - z_t[k]).detach().cpu()))
            else:
                ref_dists.append(float(np.linalg.norm(z[i] - z[k])))
    same = float(np.mean(same_dists)) if same_dists else float("nan")
    ref = float(np.mean(ref_dists)) if ref_dists else float("nan")
    return {
        "phase_shift_n_parent_groups": int(len(groups)),
        "phase_shift_n_pairs": int(len(same_dists)),
        "phase_shift_same_parent_dist": same,
        "phase_shift_same_bucket_dist": ref,
        "phase_shift_ratio": float(same / max(ref, 1e-8)) if np.isfinite(same) and np.isfinite(ref) else float("nan"),
    }


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
    if _torch_probe_enabled():
        return _sampled_style_spearman_torch(
            z,
            y,
            valid,
            mode,
            cmd,
            cmd_sigma,
            cond_delta,
            conditional,
            seed=seed,
            max_pairs=max_pairs,
        )
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


def _sampled_style_spearman_torch(
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

    valid_np = np.asarray(valid, dtype=np.float32)
    mode_np = np.asarray(mode, dtype=np.int64).reshape(-1)
    candidates = np.where((mode_np != 0) & (valid_np.sum(axis=1) > 1))[0]
    if len(candidates) < 2:
        return float("nan"), 0

    z_t = _as_float_tensor(z)
    y_t = _as_float_tensor(y)
    valid_t = _as_float_tensor(valid)
    cmd_t = _as_float_tensor(cmd)
    mode_t = torch.as_tensor(mode_np, dtype=torch.long, device=_PROBE_DEVICE)
    cmd_sigma_t = _as_float_tensor(np.asarray(cmd_sigma, dtype=np.float32)).view(1, -1).clamp_min(1e-3)

    rng = np.random.default_rng(seed)
    dz_all: list[np.ndarray] = []
    dy_all: list[np.ndarray] = []
    tries = 0
    while sum(len(x) for x in dz_all) < max_pairs and tries < 40:
        tries += 1
        i_np = rng.choice(candidates, size=max_pairs, replace=True)
        j_np = rng.choice(candidates, size=max_pairs, replace=True)
        i = torch.as_tensor(i_np, dtype=torch.long, device=_PROBE_DEVICE)
        j = torch.as_tensor(j_np, dtype=torch.long, device=_PROBE_DEVICE)
        m = i != j
        if conditional:
            dcmd = torch.linalg.norm((cmd_t[i] - cmd_t[j]) / cmd_sigma_t, dim=-1)
            m = m & (mode_t[i] == mode_t[j]) & (dcmd <= cond_delta)
        if not bool(m.any().detach().cpu()):
            continue
        i = i[m]
        j = j[m]
        v = valid_t[i] * valid_t[j]
        denom = v.sum(dim=1).clamp_min(1.0)
        dy = torch.sqrt((((y_t[i] - y_t[j]) ** 2) * v).sum(dim=1) / denom)
        dz = torch.linalg.norm(z_t[i] - z_t[j], dim=-1)
        ok = torch.isfinite(dy) & torch.isfinite(dz) & (denom > 1.0)
        if bool(ok.any().detach().cpu()):
            dz_all.append(dz[ok].detach().cpu().numpy())
            dy_all.append(dy[ok].detach().cpu().numpy())
    if not dz_all:
        return float("nan"), 0
    dz_np = np.concatenate(dz_all)[:max_pairs]
    dy_np = np.concatenate(dy_all)[:max_pairs]
    if len(dz_np) < 20:
        return float("nan"), int(len(dz_np))
    rho = spearmanr(dz_np, dy_np).correlation
    return float(rho), int(len(dz_np))


def _encode_all(model, ds, device, mask_kind: str, batch_size: int, num_workers: int, seed: int):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    mask_idx_list = build_mask_indices(mask_kind)
    mask_idx = torch.tensor(mask_idx_list, dtype=torch.long, device=device) if mask_idx_list else None
    rng = np.random.default_rng(seed)
    out = {
        k: []
        for k in [
            "z_a",
            "z_b",
            "z_short",
            "cmd",
            "mode",
            "bucket",
            "source",
            "parent",
            "y0",
            "y0_valid",
            "phi",
            "phi_valid",
            "yphi",
            "yphi_valid",
        ]
    }
    with torch.no_grad():
        for batch in loader:
            obs_a_raw = batch["obs_a"]
            obs_b_raw = batch["obs_b"]
            obs_a = _apply_mask(obs_a_raw.to(device, non_blocking=True), mask_idx)
            obs_b = _apply_mask(obs_b_raw.to(device, non_blocking=True), mask_idx)
            phi_a = batch["phi_a"].to(device, non_blocking=True)
            phi_b = batch["phi_b"].to(device, non_blocking=True)
            z_a = model(obs_a, phi_a)["z"].cpu().numpy()
            z_b = model(obs_b, phi_b)["z"].cpu().numpy()

            obs_short_np = randomize_shortcut_terms(obs_a_raw.numpy(), rng=rng)
            obs_short = _apply_mask(torch.from_numpy(obs_short_np).float().to(device, non_blocking=True), mask_idx)
            z_short = model(obs_short, phi_a)["z"].cpu().numpy()

            out["z_a"].append(z_a)
            out["z_b"].append(z_b)
            out["z_short"].append(z_short)
            out["cmd"].append(batch["cmd"].numpy())
            out["mode"].append(batch["mode_id"].numpy().reshape(-1))
            out["bucket"].append(batch["bucket_id"].numpy().reshape(-1))
            out["source"].append(batch["source_id"].numpy().reshape(-1))
            out["parent"].append(batch["parent_id"].numpy().reshape(-1))
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


def _compute_results(
    enc: dict[str, np.ndarray],
    ds: StyleWindowDataset,
    ckpt: dict,
    args,
    mask_kind: str,
    cond_delta: float,
    split_name: str,
    heldout_group: str | None,
    train_mask: np.ndarray | None,
    test_mask: np.ndarray | None,
    eval_mask: np.ndarray,
    source_counts: dict[str, int],
) -> dict[str, object]:
    z = enc["z_a"].astype(np.float32)
    z_b = enc["z_b"].astype(np.float32)
    z_short = enc["z_short"].astype(np.float32)
    cmd = enc["cmd"].astype(np.float32)
    mode = enc["mode"].astype(np.int64)
    y0 = enc["y0"].astype(np.float32)
    y0_valid = enc["y0_valid"].astype(np.float32)
    mode_oh = _onehot_mode(mode)
    cmd_mode = np.concatenate([cmd, mode_oh], axis=1)
    cmd_sigma = np.asarray(ckpt.get("cmd_sigma", np.std(cmd, axis=0).clip(min=0.05)), dtype=np.float32)

    results: dict[str, object] = {
        "n_samples": int(len(z)),
        "n_eval_samples": int(eval_mask.sum()),
        "mask_kind": mask_kind,
        "cond_cmd_delta": float(cond_delta),
        "d_gait": int(z.shape[1]),
        "split": split_name,
        "heldout_source_group": heldout_group,
        "source_group_counts": source_counts,
        "probe_backend": _PROBE_BACKEND,
        "probe_device": str(_PROBE_DEVICE),
    }

    r2_y0_z, r2_y0_z_each = _ridge_r2(z, y0, y0_valid, seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    r2_y0_cmd, _ = _ridge_r2(cmd_mode, y0, y0_valid, seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    r2_y0_z_cmd, _ = _ridge_r2(
        np.concatenate([z, cmd_mode], axis=1),
        y0,
        y0_valid,
        seed=args.seed,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    results["R2_Y0_from_Z"] = r2_y0_z
    results["R2_Y0_from_cmd_mode"] = r2_y0_cmd
    results["R2_Y0_from_Z_cmd_mode"] = r2_y0_z_cmd
    results["R2_Y0_gain_over_cmd_mode"] = float(r2_y0_z_cmd - r2_y0_cmd) if np.isfinite(r2_y0_z_cmd) and np.isfinite(r2_y0_cmd) else float("nan")
    results["R2_Y0_feature_mean"] = r2_y0_z_each
    results["R2_Y0_groups"] = _mean_scores_for_groups(r2_y0_z_each, _feature_groups(ds.stats.get("y0_names", []), "y0"))

    r2_cmd_z, r2_cmd_each = _ridge_r2(z, cmd, np.ones_like(cmd), seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    results["R2_cmd_from_Z"] = r2_cmd_z
    results["R2_cmd_from_Z_each"] = r2_cmd_each
    r2_mode_z, r2_mode_each = _ridge_r2(z, mode_oh, np.ones_like(mode_oh), seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    results["R2_mode_from_Z"] = r2_mode_z
    results["R2_mode_from_Z_each"] = r2_mode_each

    phase_sc, phase_valid = _phase_target(enc["phi"], enc["phi_valid"])
    r2_phase_z, r2_phase_each = _ridge_r2(z, phase_sc, phase_valid, seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    results["R2_phase_from_Z"] = r2_phase_z
    results["R2_phase_from_Z_each"] = r2_phase_each
    results["R2_phase_linear_from_Z"] = r2_phase_z

    z_res, residual_meta = _linear_residualize(cmd_mode, z, seed=args.seed, train_mask=train_mask)
    results.update(residual_meta)
    r2_y0_zres, r2_y0_zres_each = _ridge_r2(z_res, y0, y0_valid, seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    results["R2_Y0_from_Zres"] = r2_y0_zres
    results["R2_Y0res_retention"] = float(r2_y0_zres / max(r2_y0_z, 1e-8)) if np.isfinite(r2_y0_zres) and np.isfinite(r2_y0_z) else float("nan")
    results["R2_Y0res_groups"] = _mean_scores_for_groups(r2_y0_zres_each, _feature_groups(ds.stats.get("y0_names", []), "y0"))
    r2_cmd_zres, _ = _ridge_r2(z_res, cmd, np.ones_like(cmd), seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    r2_mode_zres, _ = _ridge_r2(z_res, mode_oh, np.ones_like(mode_oh), seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    r2_phase_zres, _ = _ridge_r2(z_res, phase_sc, phase_valid, seed=args.seed, train_mask=train_mask, test_mask=test_mask)
    results["R2_cmd_from_Zres"] = r2_cmd_zres
    results["R2_mode_from_Zres"] = r2_mode_zres
    results["R2_phase_from_Zres"] = r2_phase_zres
    results["effective_rank_Zres"] = _effective_rank(z_res[eval_mask]) if int(eval_mask.sum()) >= 2 else float("nan")
    phase_shift = _phase_shift_metrics(
        z[eval_mask],
        enc["parent"][eval_mask],
        enc["bucket"][eval_mask],
        mode[eval_mask],
        seed=args.seed,
        max_pairs=args.phase_shift_max_pairs,
    )
    results.update(phase_shift)

    if not args.skip_mlp_probe:
        results["R2_cmd_mlp_from_Z"] = _mlp_r2(
            z,
            cmd,
            np.ones_like(cmd),
            seed=args.seed,
            train_mask=train_mask,
            test_mask=test_mask,
            max_samples=args.mlp_probe_max_samples,
        )
        results["R2_cmd_mlp_from_Zres"] = _mlp_r2(
            z_res,
            cmd,
            np.ones_like(cmd),
            seed=args.seed + 30,
            train_mask=train_mask,
            test_mask=test_mask,
            max_samples=args.mlp_probe_max_samples,
        )
        results["R2_mode_mlp_from_Zres"] = _mlp_r2(
            z_res,
            mode_oh,
            np.ones_like(mode_oh),
            seed=args.seed + 31,
            train_mask=train_mask,
            test_mask=test_mask,
            max_samples=args.mlp_probe_max_samples,
        )
        results["R2_phase_mlp_from_Zres"] = _mlp_r2(
            z_res,
            phase_sc,
            phase_valid,
            seed=args.seed + 3,
            train_mask=train_mask,
            test_mask=test_mask,
            max_samples=args.mlp_probe_max_samples,
        )
        results["R2_mode_mlp_from_Z"] = _mlp_r2(
            z,
            mode_oh,
            np.ones_like(mode_oh),
            seed=args.seed + 1,
            train_mask=train_mask,
            test_mask=test_mask,
            max_samples=args.mlp_probe_max_samples,
        )
        results["R2_phase_mlp_from_Z"] = _mlp_r2(
            z,
            phase_sc,
            phase_valid,
            seed=args.seed + 2,
            train_mask=train_mask,
            test_mask=test_mask,
            max_samples=args.mlp_probe_max_samples,
        )

    z_eval = z[eval_mask]
    if len(z_eval) >= 2:
        results["shift_ratio"] = _shift_ratio(z_eval, z_b[eval_mask], seed=args.seed)
        results["effective_rank"] = _effective_rank(z_eval)
        pair_dist = np.linalg.norm(
            z_eval - z_eval[np.random.default_rng(args.seed).permutation(len(z_eval))],
            axis=-1,
        ).mean()
        results["shortcut_delta_ratio"] = float(np.linalg.norm(z_eval - z_short[eval_mask], axis=-1).mean() / max(pair_dist, 1e-8))
    else:
        results["shift_ratio"] = float("nan")
        results["effective_rank"] = float("nan")
        results["shortcut_delta_ratio"] = float("nan")

    rho_g, n_pairs_g = _sampled_style_spearman(
        z_eval,
        y0[eval_mask],
        y0_valid[eval_mask],
        mode[eval_mask],
        cmd[eval_mask],
        cmd_sigma,
        cond_delta,
        conditional=False,
        seed=args.seed,
    )
    rho_res, n_pairs_res = _sampled_style_spearman(
        z_eval,
        y0[eval_mask],
        y0_valid[eval_mask],
        mode[eval_mask],
        cmd[eval_mask],
        cmd_sigma,
        cond_delta,
        conditional=True,
        seed=args.seed + 17,
    )
    results["rho_G_spearman"] = rho_g
    results["rho_G_pairs"] = n_pairs_g
    results["rho_G_cond_cmd_spearman"] = rho_res
    results["rho_G_cond_cmd_pairs"] = n_pairs_res
    rho_zres, n_pairs_zres = _sampled_style_spearman(
        z_res[eval_mask],
        y0[eval_mask],
        y0_valid[eval_mask],
        mode[eval_mask],
        cmd[eval_mask],
        cmd_sigma,
        cond_delta,
        conditional=True,
        seed=args.seed + 29,
    )
    results["rho_G_Zres_cond_cmd_spearman"] = rho_zres
    results["rho_G_Zres_cond_cmd_pairs"] = n_pairs_zres

    yphi = enc["yphi"].reshape(-1, enc["yphi"].shape[-1]).astype(np.float32)
    yphi_valid = enc["yphi_valid"].reshape(-1, enc["yphi_valid"].shape[-1]).astype(np.float32)
    phi_frame = enc["phi"].reshape(-1, 2).astype(np.float32)
    frames_per_sample = enc["phi"].shape[1]
    z_frame = np.repeat(z, frames_per_sample, axis=0)
    frame_train_mask = np.repeat(train_mask, frames_per_sample) if train_mask is not None else None
    frame_test_mask = np.repeat(test_mask, frames_per_sample) if test_mask is not None else None
    rng = np.random.default_rng(args.seed)
    if len(yphi) > args.max_frame_samples:
        idx = rng.permutation(len(yphi))[: args.max_frame_samples]
        yphi = yphi[idx]
        yphi_valid = yphi_valid[idx]
        phi_frame = phi_frame[idx]
        z_frame = z_frame[idx]
        if frame_train_mask is not None:
            frame_train_mask = frame_train_mask[idx]
            frame_test_mask = frame_test_mask[idx]
    r2_yphi_z_phi, r2_yphi_each = _ridge_r2(
        np.concatenate([z_frame, phi_frame], axis=1),
        yphi,
        yphi_valid,
        seed=args.seed,
        train_mask=frame_train_mask,
        test_mask=frame_test_mask,
    )
    r2_yphi_phi, _ = _ridge_r2(
        phi_frame,
        yphi,
        yphi_valid,
        seed=args.seed,
        train_mask=frame_train_mask,
        test_mask=frame_test_mask,
    )
    results["R2_Yphi_from_Z_phi"] = r2_yphi_z_phi
    results["R2_Yphi_from_phi_only"] = r2_yphi_phi
    results["R2_Yphi_gain_over_phi"] = float(r2_yphi_z_phi - r2_yphi_phi) if np.isfinite(r2_yphi_z_phi) and np.isfinite(r2_yphi_phi) else float("nan")
    results["R2_Yphi_groups"] = _mean_scores_for_groups(r2_yphi_each, _feature_groups(ds.stats.get("yphi_names", []), "yphi"))

    bucket_metrics = {}
    for bucket in sorted(set(int(x) for x in enc["bucket"].reshape(-1))):
        m = (enc["bucket"].reshape(-1) == bucket) & eval_mask
        if int(m.sum()) < 40:
            continue
        r2_b, _ = _ridge_r2(z[m], y0[m], y0_valid[m], seed=args.seed)
        bucket_metrics[str(bucket)] = {"n": int(m.sum()), "R2_Y0_from_Z": r2_b}
    results["bucket_metrics"] = bucket_metrics
    return results


def _mean_metric(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _ood_drop(random_results: dict[str, object], heldout_mean: dict[str, float], key: str) -> float:
    base = random_results.get(key)
    val = heldout_mean.get(key)
    if base is None or val is None or not np.isfinite(float(base)) or not np.isfinite(float(val)) or abs(float(base)) < 1e-8:
        return float("nan")
    return float(1.0 - float(val) / float(base))


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
    ap.add_argument("--split", choices=["random", "source_heldout", "all_source_heldout"], default="random")
    ap.add_argument("--heldout_source_group", default=None)
    ap.add_argument("--max_frame_samples", type=int, default=50000)
    ap.add_argument("--mlp_probe_max_samples", type=int, default=20000)
    ap.add_argument("--mlp_probe_epochs", type=int, default=80)
    ap.add_argument("--mlp_probe_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_probe_batch_size", type=int, default=0)
    ap.add_argument("--phase_shift_max_pairs", type=int, default=20000)
    ap.add_argument("--probe_backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--skip_mlp_probe", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    global _PROBE_BACKEND, _PROBE_DEVICE, _MLP_PROBE_EPOCHS, _MLP_PROBE_LR, _MLP_PROBE_BATCH_SIZE
    _PROBE_BACKEND = args.probe_backend
    _PROBE_DEVICE = device if args.probe_backend == "torch" else torch.device("cpu")
    _MLP_PROBE_EPOCHS = int(args.mlp_probe_epochs)
    _MLP_PROBE_LR = float(args.mlp_probe_lr)
    _MLP_PROBE_BATCH_SIZE = int(args.mlp_probe_batch_size)
    if args.probe_backend == "torch":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[style-probe-v4] probe_backend={args.probe_backend} probe_device={_PROBE_DEVICE}", flush=True)

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
        d_aux=cfg.get("d_aux", 32),
        encoder_kind=cfg.get("encoder_kind", "tcn"),
        phi_fourier_k=cfg.get("phi_fourier_k", 1),
        yphi_hidden_dim=cfg.get("yphi_hidden_dim", cfg["d_back"]),
        phase_adv=cfg.get("l_phase_adv", 0.0) > 0.0,
        phase_adv_hidden=cfg.get("phase_adv_hidden", cfg["d_back"]),
        cmd_adv=cfg.get("l_cmd_adv", 0.0) > 0.0,
        mode_adv=cfg.get("l_mode_adv", 0.0) > 0.0,
        cmd_mode_adv_hidden=cfg.get("cmd_mode_adv_hidden", cfg["d_back"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = StyleWindowDataset(args.data_dir, max_shards=args.max_shards, max_samples=args.max_samples)
    enc = _encode_all(model, ds, device, mask_kind, args.batch_size, args.num_workers, args.seed)

    source_group = _source_groups_for_dataset(ds)
    source_counts = {str(g): int((source_group == g).sum()) for g in sorted(set(source_group.tolist()))}

    if args.split == "all_source_heldout":
        random_eval = np.ones((len(source_group),), dtype=bool)
        random_results = _compute_results(
            enc,
            ds,
            ckpt,
            args,
            mask_kind,
            cond_delta,
            "random",
            None,
            None,
            None,
            random_eval,
            source_counts,
        )
        per_source = {}
        heldout_rows = []
        for group in sorted(source_counts):
            test_mask = source_group == group
            train_mask = ~test_mask
            if int(train_mask.sum()) < 40 or int(test_mask.sum()) < 40:
                continue
            row = _compute_results(
                enc,
                ds,
                ckpt,
                args,
                mask_kind,
                cond_delta,
                "source_heldout",
                group,
                train_mask,
                test_mask,
                test_mask.copy(),
                source_counts,
            )
            per_source[group] = row
            heldout_rows.append(row)
        mean_keys = [
            "R2_Y0_from_Z",
            "R2_Y0_gain_over_cmd_mode",
            "R2_cmd_from_Z",
            "R2_mode_from_Z",
            "R2_phase_from_Z",
            "R2_phase_mlp_from_Z",
            "R2_Y0_from_Zres",
            "R2_Y0res_retention",
            "R2_cmd_from_Zres",
            "R2_mode_from_Zres",
            "R2_phase_from_Zres",
            "R2_cmd_mlp_from_Zres",
            "R2_mode_mlp_from_Zres",
            "R2_phase_mlp_from_Zres",
            "shift_ratio",
            "effective_rank",
            "effective_rank_Zres",
            "rho_G_spearman",
            "rho_G_cond_cmd_spearman",
            "rho_G_Zres_cond_cmd_spearman",
            "R2_Yphi_from_Z_phi",
            "R2_Yphi_gain_over_phi",
            "phase_shift_ratio",
            "phase_shift_same_parent_dist",
            "phase_shift_same_bucket_dist",
        ]
        heldout_mean = {k: _mean_metric(heldout_rows, k) for k in mean_keys}
        results = {
            **heldout_mean,
            "split": "all_source_heldout",
            "heldout_source_group": "mean",
            "n_samples": int(len(source_group)),
            "n_eval_samples": int(sum(int(r.get("n_eval_samples", 0)) for r in heldout_rows)),
            "mask_kind": mask_kind,
            "cond_cmd_delta": float(cond_delta),
            "d_gait": int(enc["z_a"].shape[1]),
            "probe_backend": _PROBE_BACKEND,
            "probe_device": str(_PROBE_DEVICE),
            "source_group_counts": source_counts,
            "random_split": random_results,
            "per_source_heldout": per_source,
            "OOD_drop_R2_Y0": _ood_drop(random_results, heldout_mean, "R2_Y0_from_Z"),
            "OOD_drop_rho_G": _ood_drop(random_results, heldout_mean, "rho_G_spearman"),
            "OOD_drop_rho_G_cond": _ood_drop(random_results, heldout_mean, "rho_G_cond_cmd_spearman"),
        }
    else:
        train_mask = None
        test_mask = None
        eval_mask = np.ones((len(source_group),), dtype=bool)
        heldout_group = args.heldout_source_group
        if args.split == "source_heldout":
            choices = sorted(set(source_group.tolist()))
            if not choices:
                raise ValueError("source_heldout split requested, but no source groups are available")
            if heldout_group is None:
                heldout_group = choices[-1]
            if heldout_group not in choices:
                raise ValueError(f"heldout_source_group={heldout_group!r} not in available groups={choices}")
            test_mask = source_group == heldout_group
            train_mask = ~test_mask
            eval_mask = test_mask.copy()
            if int(train_mask.sum()) < 40 or int(test_mask.sum()) < 40:
                raise ValueError(
                    f"source_heldout split too small: train={int(train_mask.sum())} test={int(test_mask.sum())}"
                )
        results = _compute_results(
            enc,
            ds,
            ckpt,
            args,
            mask_kind,
            cond_delta,
            args.split,
            heldout_group,
            train_mask,
            test_mask,
            eval_mask,
            source_counts,
        )

    safe_results = _json_sanitize(results)
    print(json.dumps(safe_results, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(safe_results, f, indent=2)
        print(f"[style-probe-v4] wrote {args.out_json}")


if __name__ == "__main__":
    main()
