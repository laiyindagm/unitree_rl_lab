"""Left-right symmetry augmentation for G1 15-DOF humanoid.

Observation layout (per frame, history_length=5, flatten_history_dim=True):
  base_ang_vel                 : 3 dims  × 5 = 15
  projected_gravity            : 3 dims  × 5 = 15
  velocity_commands            : 3 dims  × 5 = 15
  [lin_speed_reward_regime_token] : 2 dims  × 5 = 10  (optional)
  [gait_mode_token_3]          : 3 dims  × 5 = 15  (optional)
  [gait_mode_token]            : 5 dims  × 5 = 25  (optional)
  joint_pos_rel                : 15 dims × 5 = 75
  joint_vel_rel                : 15 dims × 5 = 75
  last_action                  : 15 dims × 5 = 75
  [gait_phase]                 : 2 dims  × 5 = 10  (optional)
  Total: 270 / 285 / 295 / 305 / 315 depending on optional tokens

Joint order (TRAIN_JOINT_NAMES resolved from URDF):
  0: left_hip_pitch     6: right_hip_pitch    12: waist_yaw
  1: left_hip_roll      7: right_hip_roll     13: waist_roll
  2: left_hip_yaw       8: right_hip_yaw      14: waist_pitch
  3: left_knee           9: right_knee
  4: left_ankle_pitch   10: right_ankle_pitch
  5: left_ankle_roll    11: right_ankle_roll

Left-right mirror (sagittal plane reflection, y → -y):
  - Swap left ↔ right limb indices
  - Negate roll / yaw joints (frontal & transverse plane)
  - Negate ang_vel x,z  (pseudovector under y-reflection)
  - Negate gravity y
  - Negate vel_cmd y, wz
  - Preserve reward-regime and gait-mode tokens (mode partition is reflection-invariant)
  - Negate gait_phase (phase shift by π)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# ── joint mirror tables (15 DOF) ──────────────────────────────────────────────
#   swap left(0-5) ↔ right(6-11), waist(12-14) stays
JOINT_MIRROR_IDX = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14]
#   negate roll / yaw components
JOINT_MIRROR_SIGN = [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1]

H = 5  # history_length (must match ObservationGroupCfg)


# ── public API ────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions via left-right mirror (2× augmentation).

    Args:
        env: The environment instance.
        obs: The original observation TensorDict. Defaults to None.
        actions: The original action tensor. Defaults to None.

    Returns:
        Augmented (obs, actions) with batch doubled: [original, mirrored].
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# ── internal helpers ──────────────────────────────────────────────────────────
def _mirror_block(block: torch.Tensor, dim: int,
                  perm: list[int] | None = None,
                  sign: torch.Tensor | None = None) -> torch.Tensor:
    """Mirror a single history-flattened observation block (N, H*D) → (N, H*D)."""
    N = block.shape[0]
    x = block.reshape(N, H, dim)
    if perm is not None:
        x = x[:, :, perm]
    if sign is not None:
        x = x * sign.to(x.device)
    return x.reshape(N, -1)


def _mirror_policy_obs(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device

    _jsign = torch.tensor(JOINT_MIRROR_SIGN, dtype=obs.dtype, device=device)
    active_terms = env.observation_manager.active_terms["policy"]
    has_lin_speed_reward_regime_token = "lin_speed_reward_regime_token" in active_terms
    has_gait_mode_token_3 = "gait_mode_token_3" in active_terms
    has_gait_mode_token = "gait_mode_token" in active_terms
    has_gait_phase = "gait_phase" in active_terms

    idx = 0
    # base_ang_vel  (3 × H)
    obs[:, idx:idx + 3 * H] = _mirror_block(
        obs[:, idx:idx + 3 * H], 3, sign=torch.tensor([-1, 1, -1], dtype=obs.dtype, device=device))
    idx += 3 * H

    # projected_gravity  (3 × H)
    obs[:, idx:idx + 3 * H] = _mirror_block(
        obs[:, idx:idx + 3 * H], 3, sign=torch.tensor([1, -1, 1], dtype=obs.dtype, device=device))
    idx += 3 * H

    # velocity_commands  (3 × H)
    obs[:, idx:idx + 3 * H] = _mirror_block(
        obs[:, idx:idx + 3 * H], 3, sign=torch.tensor([1, -1, -1], dtype=obs.dtype, device=device))
    idx += 3 * H

    # lin_speed_reward_regime_token  (2 × H) — reflection-invariant
    if has_lin_speed_reward_regime_token:
        idx += 2 * H

    # gait_mode_token_3  (3 × H) — reflection-invariant
    if has_gait_mode_token_3:
        idx += 3 * H

    # gait_mode_token  (5 × H) — reflection-invariant
    if has_gait_mode_token:
        idx += 5 * H

    # joint_pos_rel  (15 × H)
    obs[:, idx:idx + 15 * H] = _mirror_block(
        obs[:, idx:idx + 15 * H], 15, perm=JOINT_MIRROR_IDX, sign=_jsign)
    idx += 15 * H

    # joint_vel_rel  (15 × H)
    obs[:, idx:idx + 15 * H] = _mirror_block(
        obs[:, idx:idx + 15 * H], 15, perm=JOINT_MIRROR_IDX, sign=_jsign)
    idx += 15 * H

    # last_action  (15 × H)
    obs[:, idx:idx + 15 * H] = _mirror_block(
        obs[:, idx:idx + 15 * H], 15, perm=JOINT_MIRROR_IDX, sign=_jsign)
    idx += 15 * H

    # gait_phase  (2 × H) — shift by π: negate sin and cos
    if has_gait_phase:
        obs[:, idx:idx + 2 * H] = _mirror_block(
            obs[:, idx:idx + 2 * H], 2, sign=torch.tensor([-1, -1], dtype=obs.dtype, device=device))
        idx += 2 * H

    return obs


def _mirror_joints(actions: torch.Tensor) -> torch.Tensor:
    """Mirror 15-DOF action vector: swap left↔right, negate roll/yaw."""
    actions = actions.clone()
    device = actions.device
    _jsign = torch.tensor(JOINT_MIRROR_SIGN, dtype=actions.dtype, device=device)
    actions = actions[:, JOINT_MIRROR_IDX] * _jsign
    return actions
