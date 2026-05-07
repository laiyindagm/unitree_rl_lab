# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Per-segment holistic gait feature extraction.

Computes scalar features that describe a contiguous trajectory segment as a
whole gait pattern (rather than instantaneous state). These features are the
candidate supervisory signal for `z_gait` and the regression targets for the
input-side diagnostic probe.

Joint indices (TRAIN_JOINT_NAMES order, 15 dof):
  0 left_hip_pitch  1 left_hip_roll   2 left_hip_yaw   3 left_knee
  4 left_ankle_pitch 5 left_ankle_roll
  6 right_hip_pitch 7 right_hip_roll  8 right_hip_yaw  9 right_knee
  10 right_ankle_pitch 11 right_ankle_roll
  12 waist_yaw  13 waist_roll  14 waist_pitch
"""
from __future__ import annotations
import numpy as np

# Per-frame slice into the flat 295-dim policy obs. There are 5 history frames
# concatenated; frame k starts at offset k * FRAME_DIM.
FRAME_DIM = 59
N_FRAMES = 5
PER_FRAME = {
    "base_ang_vel":            (0, 3),
    "projected_gravity":       (3, 6),
    "velocity_commands":       (6, 9),
    "lin_speed_regime_token":  (9, 11),
    "gait_mode_token_3":       (11, 14),
    "joint_pos_rel":           (14, 29),
    "joint_vel_rel":           (29, 44),
    "last_action":             (44, 59),
}


def build_mask_indices(kind: str = "cmd") -> str:
    """Return a comma/colon spec compatible with frnc_pretrain._parse_index_list.

    kind = 'cmd'        : just velocity_commands per frame (legacy).
    kind = 'cmd_tokens' : cmd + lin_speed_token + gait_mode_token + last_action.
    kind = 'strict'     : cmd_tokens + base_ang_vel + joint_vel_rel
                          (only joint_pos_rel + projected_gravity remain).
    """
    if kind == "cmd":
        groups = ["velocity_commands"]
    elif kind == "cmd_tokens":
        groups = ["velocity_commands", "lin_speed_regime_token",
                  "gait_mode_token_3", "last_action"]
    elif kind == "strict":
        groups = ["velocity_commands", "lin_speed_regime_token",
                  "gait_mode_token_3", "last_action",
                  "base_ang_vel", "joint_vel_rel"]
    elif kind == "none":
        return ""
    else:
        raise ValueError(f"unknown mask kind: {kind}")
    spans = []
    for k in range(N_FRAMES):
        base = k * FRAME_DIM
        for g in groups:
            a, b = PER_FRAME[g]
            spans.append(f"{base + a}:{base + b}")
    return ",".join(spans)


# ---------------------- per-segment gait features ---------------------- #
GAIT_FEATURE_NAMES = [
    "step_freq",        # Hz, dominant freq of foot contact toggles
    "duty_l", "duty_r", # fraction of frames each foot is in contact
    "bilat_cos", "bilat_sin",  # bilateral phase offset between L/R hip pitch
    "step_amp_lk", "step_amp_rk",  # peak-to-peak knee angle over segment
    "lat_sway",         # std(left_hip_roll) + std(right_hip_roll)
    "waist_yaw_std", "waist_pitch_std",
    "ang_act",          # std of base_ang_vel z proxy via segment cmd? -> use joint
]


def _safe_std(x):
    return float(np.std(x))


def gait_features_segment(joint_pos_seq: np.ndarray,
                          foot_contact_seq: np.ndarray,
                          fs: float = 50.0) -> dict:
    """joint_pos_seq: (T, 15) joint_pos_rel from last frame collected at each
    timestep. foot_contact_seq: (T, F) {0,1}. Returns dict of float scalars.
    Returns NaN-filled values where signal is too quiet (standing) or T < 16.
    """
    T = joint_pos_seq.shape[0]
    out = {k: float("nan") for k in GAIT_FEATURE_NAMES}
    if T < 16:
        return out
    F_ = foot_contact_seq.shape[1]
    # 1. step_freq via FFT of foot contact signal
    sig = foot_contact_seq.sum(axis=-1).astype(np.float32)
    sig = sig - sig.mean()
    if sig.std() >= 1e-3:
        sp = np.abs(np.fft.rfft(sig))
        fr = np.fft.rfftfreq(T, d=1.0 / fs)
        keep = fr >= 0.3
        if keep.any():
            out["step_freq"] = float(fr[keep][np.argmax(sp[keep])])
        else:
            out["step_freq"] = 0.0
    else:
        out["step_freq"] = 0.0
    # 2. duty
    out["duty_l"] = float(foot_contact_seq[:, 0].mean())
    out["duty_r"] = float(foot_contact_seq[:, 1].mean()) if F_ > 1 else float("nan")
    # 3. bilateral phase offset on L/R hip pitch
    try:
        from scipy.signal import hilbert
        l = joint_pos_seq[:, 0] - joint_pos_seq[:, 0].mean()
        r = joint_pos_seq[:, 6] - joint_pos_seq[:, 6].mean()
        if l.std() >= 1e-3 and r.std() >= 1e-3:
            phl = np.angle(hilbert(l)); phr = np.angle(hilbert(r))
            edge = max(4, T // 10)
            d = phr[edge:-edge] - phl[edge:-edge]
            out["bilat_cos"] = float(np.cos(d).mean())
            out["bilat_sin"] = float(np.sin(d).mean())
        else:
            out["bilat_cos"] = 1.0
            out["bilat_sin"] = 0.0
    except Exception:
        pass
    # 4. step amplitude (knee range)
    out["step_amp_lk"] = float(joint_pos_seq[:, 3].ptp())
    out["step_amp_rk"] = float(joint_pos_seq[:, 9].ptp())
    # 5. lateral sway via hip roll std
    out["lat_sway"] = _safe_std(joint_pos_seq[:, 1]) + _safe_std(joint_pos_seq[:, 7])
    # 6. waist
    out["waist_yaw_std"] = _safe_std(joint_pos_seq[:, 12])
    out["waist_pitch_std"] = _safe_std(joint_pos_seq[:, 14])
    # 7. angular activity proxy: std of waist_yaw (heading change) — already above,
    # but keep an extra column for symmetry / future use
    out["ang_act"] = _safe_std(joint_pos_seq[:, 12])
    return out


def gait_features_to_array(d: dict) -> np.ndarray:
    return np.array([d[k] for k in GAIT_FEATURE_NAMES], dtype=np.float32)
