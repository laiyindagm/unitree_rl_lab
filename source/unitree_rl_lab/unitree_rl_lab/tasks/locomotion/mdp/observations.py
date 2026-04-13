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
