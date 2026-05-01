from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def gait_phase_speed_adaptive(
    env: ManagerBasedRLEnv,
    walk_period: float = 1.0,
    run_period: float = 0.7,
    speed_threshold: float = 0.8,
    decay_factor: float = 0.95,
    standstill_threshold: float = 0.1,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Speed-adaptive gait phase with standstill decay (FEAP-inspired).

    - Low speed (cmd_norm < speed_threshold): uses walk_period (slower gait)
    - High speed (cmd_norm >= speed_threshold): uses run_period (faster gait)
    - Standstill (cmd_norm < standstill_threshold): phase decays toward 0

    Returns [sin(2*pi*phase), cos(2*pi*phase)] per environment.
    """
    # Initialize persistent phase buffer on first call
    if not hasattr(env, "_gait_phase_adaptive"):
        env._gait_phase_adaptive = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )

    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)  # [vx, vy, wz]

    # Select period based on speed: low speed → long period, high speed → short
    period = torch.where(
        cmd_norm >= speed_threshold,
        torch.full_like(cmd_norm, run_period),
        torch.full_like(cmd_norm, walk_period),
    )

    # Advance phase for moving envs
    dt = env.step_dt
    phase_increment = dt / period
    is_moving = cmd_norm >= standstill_threshold

    # Moving: advance phase; Standstill: decay phase toward 0
    new_phase = torch.where(
        is_moving,
        (env._gait_phase_adaptive + phase_increment) % 1.0,
        env._gait_phase_adaptive * decay_factor,
    )

    # Zero out very small phases to avoid residual oscillation
    new_phase = torch.where(new_phase < 1e-3, torch.zeros_like(new_phase), new_phase)

    env._gait_phase_adaptive = new_phase

    # Reset phase for newly reset envs (guard for init-time dimension probe)
    if hasattr(env, "reset_buf"):
        reset_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            env._gait_phase_adaptive[reset_ids] = 0.0

    # Encode as sin/cos
    output = torch.zeros(env.num_envs, 2, device=env.device)
    output[:, 0] = torch.sin(new_phase * torch.pi * 2.0)
    output[:, 1] = torch.cos(new_phase * torch.pi * 2.0)
    return output


def gait_mode_token(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    eps_x: float = 0.1,
    eps_y: float = 0.1,
    eps_w: float = 0.1,
) -> torch.Tensor:
    """5-mode one-hot gait token derived from velocity command (V20a).

    Partitions the command space into 5 disjoint regions aligned with the V19i
    sampling distribution: standing / pure_vx / pure_vy / pure_wz / joint.
    Each environment gets a one-hot vector indicating its current locomotion
    mode. By exposing the discrete mode boundary directly to the policy, the
    function approximator no longer needs to encode the high Lipschitz
    constant of pi*(.|c) across each axis threshold.

        index 0: standing  (|vx|<eps & |vy|<eps & |wz|<eps)
        index 1: pure_vx   (only vx active)
        index 2: pure_vy   (only vy active)
        index 3: pure_wz   (only wz active)
        index 4: joint     (>= 2 axes active)

    Returns: (num_envs, 5) float tensor (one-hot).
    """
    cmd = env.command_manager.get_command(command_name)  # (N, 3): vx, vy, wz
    vx_zero = cmd[:, 0].abs() < eps_x
    vy_zero = cmd[:, 1].abs() < eps_y
    wz_zero = cmd[:, 2].abs() < eps_w

    standing = vx_zero & vy_zero & wz_zero
    pure_vx = (~vx_zero) & vy_zero & wz_zero
    pure_vy = vx_zero & (~vy_zero) & wz_zero
    pure_wz = vx_zero & vy_zero & (~wz_zero)
    joint = ~(standing | pure_vx | pure_vy | pure_wz)

    token = torch.stack([standing, pure_vx, pure_vy, pure_wz, joint], dim=-1)
    return token.float()


def gait_mode_token_3(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    eps_x: float = 0.1,
    eps_y: float = 0.1,
    eps_w: float = 0.1,
) -> torch.Tensor:
    """3-mode one-hot gait token: {standing, pure_wz, other} (V20j).

    Reduces gait_mode_token's 5-way partition to 3 by collapsing
    pure_vx/pure_vy/joint into a single "other" bucket. Rationale:

    The 5-mode token enforces full per-mode subpolicy specialization. Per
    rewards.py audit (track_ang_vel_z_rotating_aware SKIP at cmd_yaw<0.05
    + wz_proportional_penalty has_cmd guard at cmd_yaw<0.08), the
    pure_vx/pure_vy modes receive ZERO yaw-control gradient because they
    are by construction the cmd_yaw=0 envs. Token-routed isolation thus
    starves their subpolicies of yaw-control learning, while the joint
    subpolicy (which sees cmd_wz!=0 envs) DOES learn yaw control.

    The 3-mode token preserves isolation where it matters (standing
    needs zero motion; pure_wz needs drift-free in-place rotation -- both
    benefit from dedicated subpolicies) but COLLAPSES pure_vx/pure_vy
    into "other" alongside joint envs. The "other" subpolicy then
    receives the joint envs' cmd_wz!=0 yaw signal as well as pure_vx/vy
    samples -- the same shared-parameter transfer that V19f relied on
    globally, scoped to the linear-motion subpolicy.

        index 0: standing  (|vx|<eps & |vy|<eps & |wz|<eps)
        index 1: pure_wz   (|vx|<eps & |vy|<eps & |wz|>=eps)
        index 2: other     (any linear motion: pure_vx, pure_vy, or joint)

    Returns: (num_envs, 3) float tensor (one-hot).
    """
    cmd = env.command_manager.get_command(command_name)  # (N, 3): vx, vy, wz
    vx_zero = cmd[:, 0].abs() < eps_x
    vy_zero = cmd[:, 1].abs() < eps_y
    wz_zero = cmd[:, 2].abs() < eps_w

    standing = vx_zero & vy_zero & wz_zero
    pure_wz = vx_zero & vy_zero & (~wz_zero)
    other = ~(standing | pure_wz)

    token = torch.stack([standing, pure_wz, other], dim=-1)
    return token.float()


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


def lin_speed_reward_regime_token(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    threshold_scale: float = 1.567,
    transition_width: float = 0.05,
    lin_cmd_min: float = 0.05,
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    cmd_lin = torch.norm(cmd[:, :2], dim=1)

    threshold = threshold_scale * std
    if transition_width <= 0.0:
        exp_weight = (cmd_lin >= threshold).float()
    else:
        lo = threshold - transition_width
        hi = threshold + transition_width
        alpha = ((cmd_lin - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
        exp_weight = _smoothstep(alpha)
    exp_weight = torch.where(cmd_lin < lin_cmd_min, torch.ones_like(exp_weight), exp_weight)
    low_speed_weight = 1.0 - exp_weight
    return torch.stack([low_speed_weight, exp_weight], dim=-1)
