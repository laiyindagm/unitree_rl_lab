"""Observation layout helpers for frnc_style_v4.

The V21/V22 policy observation is a term-major flattened history:

    [term_0_history | term_1_history | ...]

with history length 5 and a 59-dim logical single frame.  This module is the
single source of truth for offline style-encoder masks and feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HISTORY_LEN = 5
SINGLE_FRAME_DIM = 59
POLICY_OBS_DIM = HISTORY_LEN * SINGLE_FRAME_DIM


@dataclass(frozen=True)
class TermSpec:
    name: str
    dim: int
    start: int

    @property
    def stop(self) -> int:
        return self.start + self.dim * HISTORY_LEN

    def frame_slice(self, frame: int) -> slice:
        if frame < 0:
            frame = HISTORY_LEN + frame
        if frame < 0 or frame >= HISTORY_LEN:
            raise IndexError(f"frame index {frame} out of range for history_len={HISTORY_LEN}")
        a = self.start + frame * self.dim
        return slice(a, a + self.dim)

    @property
    def latest_slice(self) -> slice:
        return self.frame_slice(-1)

    @property
    def all_slice(self) -> slice:
        return slice(self.start, self.stop)


def _build_terms() -> dict[str, TermSpec]:
    dims = [
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_commands", 3),
        ("lin_speed_regime_token", 2),
        ("gait_mode_token_3", 3),
        ("joint_pos_rel", 15),
        ("joint_vel_rel", 15),
        ("last_action", 15),
    ]
    out: dict[str, TermSpec] = {}
    cursor = 0
    for name, dim in dims:
        out[name] = TermSpec(name=name, dim=dim, start=cursor)
        cursor += dim * HISTORY_LEN
    if cursor != POLICY_OBS_DIM:
        raise RuntimeError(f"layout cursor={cursor} != POLICY_OBS_DIM={POLICY_OBS_DIM}")
    return out


TERMS = _build_terms()

MASK_KEEP_GROUPS = {
    # Main V4 input: conservative style channels.  Contact is not present in the
    # policy obs, so it is used as a target only by style_gait_features.py.
    "M0_conservative": ("projected_gravity", "joint_pos_rel"),
    # Rich ablation: adds velocity-like channels that may improve decoding but
    # are expected to increase phase leakage.
    "M1_rich": ("projected_gravity", "joint_pos_rel", "joint_vel_rel", "base_ang_vel"),
    # V3-style strict mask equivalent for term-major V21/V22 obs.
    "M2_old_strict": ("projected_gravity", "joint_pos_rel"),
    "none": tuple(TERMS.keys()),
}

SHORTCUT_GROUPS = (
    "base_ang_vel",
    "velocity_commands",
    "lin_speed_regime_token",
    "gait_mode_token_3",
    "joint_vel_rel",
    "last_action",
)


def term(name: str) -> TermSpec:
    try:
        return TERMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown obs term {name!r}; known={sorted(TERMS)}") from exc


def latest_slice(name: str) -> slice:
    return term(name).latest_slice


def build_mask_indices(mask_kind: str) -> list[int]:
    """Return indices to zero for a named V4 mask."""
    if mask_kind not in MASK_KEEP_GROUPS:
        raise ValueError(f"unknown mask_kind={mask_kind!r}; choices={sorted(MASK_KEEP_GROUPS)}")
    keep = set(MASK_KEEP_GROUPS[mask_kind])
    idx: list[int] = []
    for name, spec in TERMS.items():
        if name not in keep:
            idx.extend(range(spec.start, spec.stop))
    return idx


def build_mask_spec(mask_kind: str) -> str:
    """Return a comma/colon mask spec compatible with older FRNC helpers."""
    idx = build_mask_indices(mask_kind)
    if not idx:
        return ""
    spans: list[str] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append(f"{start}:{prev + 1}" if start != prev else str(start))
        start = prev = i
    spans.append(f"{start}:{prev + 1}" if start != prev else str(start))
    return ",".join(spans)


def apply_mask(obs: np.ndarray, mask_kind: str) -> np.ndarray:
    """Return a copy of obs with masked channels zeroed.

    obs may be (..., 295).
    """
    if obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(f"expected obs last dim {POLICY_OBS_DIM}, got {obs.shape[-1]}")
    idx = build_mask_indices(mask_kind)
    out = obs.copy()
    if idx:
        out[..., idx] = 0.0
    return out


def latest_term(obs: np.ndarray, name: str) -> np.ndarray:
    """Extract the latest history frame for a term from term-major obs."""
    if obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(f"expected obs last dim {POLICY_OBS_DIM}, got {obs.shape[-1]}")
    return obs[..., latest_slice(name)]


def all_term(obs: np.ndarray, name: str) -> np.ndarray:
    """Extract a term as (..., HISTORY_LEN, term_dim)."""
    spec = term(name)
    x = obs[..., spec.all_slice]
    return x.reshape(*obs.shape[:-1], HISTORY_LEN, spec.dim)


def mode_id_from_cmd(cmd: np.ndarray, eps_xy: float = 0.1, eps_w: float = 0.1) -> np.ndarray:
    """Return 0=standing, 1=pure_wz, 2=other from velocity commands."""
    vx = np.abs(cmd[..., 0])
    vy = np.abs(cmd[..., 1])
    wz = np.abs(cmd[..., 2])
    standing = (vx < eps_xy) & (vy < eps_xy) & (wz < eps_w)
    pure_wz = (vx < eps_xy) & (vy < eps_xy) & (wz >= eps_w)
    out = np.full(cmd.shape[:-1], 2, dtype=np.int64)
    out[standing] = 0
    out[pure_wz] = 1
    return out


def randomize_shortcut_terms(obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomize shortcut channels while leaving style channels untouched."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = obs.copy()
    for name in SHORTCUT_GROUPS:
        spec = term(name)
        sl = spec.all_slice
        src = out[..., sl]
        noise = rng.standard_normal(src.shape).astype(out.dtype, copy=False)
        out[..., sl] = noise
    return out


def describe_layout() -> dict[str, dict[str, int]]:
    return {
        name: {
            "dim": spec.dim,
            "start": spec.start,
            "stop": spec.stop,
            "latest_start": spec.latest_slice.start,
            "latest_stop": spec.latest_slice.stop,
        }
        for name, spec in TERMS.items()
    }

