from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg=None,
) -> torch.Tensor:
    """Terrain level curriculum based on velocity tracking."""
    return torch.mean(env.scene.terrain.terrain_levels.float())


def speed_bucketed_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    low_speed_threshold: float = 0.3,
    mid_speed_threshold: float = 0.6,
    unlock_reward_ratio: float = 0.6,
) -> torch.Tensor:
    """Speed-bucketed velocity curriculum.

    Accumulates per-env reward data from each _reset_idx call into a buffer.
    At periodic checkpoints (same gate as lin_vel_cmd_levels), evaluates the
    buffered data and decides whether to advance the bucket level.

    Compatible with init_at_random_ep_len=True where envs reset at different
    times (only ~4 envs per step). The buffer collects data across steps,
    ensuring enough samples for a reliable evaluation.

    Buckets:
      0: standstill/very low (cmd < low_speed_threshold) -- always active
      1: low speed (low_speed_threshold..mid_speed_threshold)
      2: mid speed (mid_speed_threshold..limit)
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Initialize state on first call
    if not hasattr(env, "_current_bucket_level"):
        env._current_bucket_level = 0
        env._bucket_reward_buf = []  # list of (avg_reward, cmd_lin_norm) tuples
        env._bucket_last_eval_step = 0

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    # Accumulate data from every reset batch (even between gate evaluations)
    if len(env_ids) > 0:
        episode_rewards = env.reward_manager._episode_sums[reward_term_name][env_ids]
        avg_reward = episode_rewards / env.max_episode_length_s
        cmd = env.command_manager.get_command("base_velocity")[env_ids]
        cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
        env._bucket_reward_buf.append((avg_reward.detach().clone(), cmd_lin_norm.detach().clone()))

    # Evaluate at episode boundaries (same gate as lin_vel_cmd_levels)
    if env.common_step_counter % env.max_episode_length != 0:
        return torch.tensor(env._current_bucket_level, device=env.device, dtype=torch.float32)

    # Gate passed -- evaluate buffered data
    target = reward_term.weight * unlock_reward_ratio

    if len(env._bucket_reward_buf) > 0:
        all_avg_reward = torch.cat([x[0] for x in env._bucket_reward_buf])
        all_cmd_norm = torch.cat([x[1] for x in env._bucket_reward_buf])

        if env._current_bucket_level == 0:
            bucket_0_mask = all_cmd_norm < low_speed_threshold
            if bucket_0_mask.sum() > 0:
                avg_bucket_0 = all_avg_reward[bucket_0_mask].mean()
                if avg_bucket_0 > target:
                    env._current_bucket_level = 1
                    ranges.lin_vel_x = [-low_speed_threshold, mid_speed_threshold]
                    ranges.lin_vel_y = [-low_speed_threshold, low_speed_threshold]

        elif env._current_bucket_level == 1:
            bucket_1_mask = (all_cmd_norm >= low_speed_threshold) & (all_cmd_norm < mid_speed_threshold)
            if bucket_1_mask.sum() > 0:
                avg_bucket_1 = all_avg_reward[bucket_1_mask].mean()
                if avg_bucket_1 > target:
                    env._current_bucket_level = 2
                    ranges.lin_vel_x = list(limit_ranges.lin_vel_x)
                    ranges.lin_vel_y = list(limit_ranges.lin_vel_y)

        # Also handle ang_vel progression tied to bucket level
        if env._current_bucket_level >= 1:
            current_ang = ranges.ang_vel_z
            if isinstance(current_ang, (list, tuple)):
                current_max = current_ang[1]
            else:
                current_max = float(current_ang)
            limit_max = limit_ranges.ang_vel_z[1]
            if current_max < limit_max:
                new_max = min(current_max + 0.1, limit_max)
                ranges.ang_vel_z = [-new_max, new_max]

    # Clear buffer after evaluation
    env._bucket_reward_buf = []

    return torch.tensor(env._current_bucket_level, device=env.device, dtype=torch.float32)


def iteration_based_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    expand_iterations: tuple[int, ...] = (2000, 5000, 8000),
) -> torch.Tensor:
    """Iteration-based velocity curriculum -- deterministic schedule.

    Expands velocity command ranges at fixed iteration counts, independent
    of reward performance. Most robust fallback when reward-based curricula
    are unreliable.

    Schedule:
      Before expand_iterations[0]: initial ranges (from config)
      After expand_iterations[0]:  30% of way to limit
      After expand_iterations[1]:  65% of way to limit
      After expand_iterations[2]:  full limit_ranges
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    if not hasattr(env, "_iter_curriculum_level"):
        env._iter_curriculum_level = 0
        env._iter_initial_lin_vel_x = list(ranges.lin_vel_x)
        env._iter_initial_lin_vel_y = list(ranges.lin_vel_y)
        env._iter_initial_ang_vel_z = list(ranges.ang_vel_z)

    # steps_per_iter = num_steps_per_env from PPO config (typically 24)
    # common_step_counter = total RL steps (not iterations)
    # With 24 steps per iteration: iteration ~ common_step_counter / 24
    steps_per_iter = 24  # RSL_RL default num_steps_per_env
    current_iter = env.common_step_counter // steps_per_iter

    def _lerp_range(initial, limit, alpha):
        return [
            initial[0] + alpha * (limit[0] - initial[0]),
            initial[1] + alpha * (limit[1] - initial[1]),
        ]

    new_level = 0
    for i, threshold in enumerate(expand_iterations):
        if current_iter >= threshold:
            new_level = i + 1

    if new_level > env._iter_curriculum_level:
        env._iter_curriculum_level = new_level
        alphas = [0.0, 0.3, 0.65, 1.0]
        alpha = alphas[min(new_level, len(alphas) - 1)]

        ranges.lin_vel_x = _lerp_range(
            env._iter_initial_lin_vel_x, list(limit_ranges.lin_vel_x), alpha
        )
        ranges.lin_vel_y = _lerp_range(
            env._iter_initial_lin_vel_y, list(limit_ranges.lin_vel_y), alpha
        )
        ranges.ang_vel_z = _lerp_range(
            env._iter_initial_ang_vel_z, list(limit_ranges.ang_vel_z), alpha
        )

    return torch.tensor(env._iter_curriculum_level, device=env.device, dtype=torch.float32)


def performance_weighted_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    range_expand_threshold: float = 0.5,
) -> torch.Tensor:
    """Companion curriculum for PerformanceWeightedVelocityCommand.

    Uses relative tracking accuracy (computed step-by-step inside the command
    term) instead of raw episode reward sums. This correctly evaluates
    low-speed bin performance (standing still = 0, not 0.7).

    Responsibilities:
    1. Read per-env episode accuracy from command term at reset
    2. Feed accuracy to bin performance EMA
    3. Expand command ranges when ALL bins exceed threshold
    4. Log bin performance + sampling distribution at evaluation gate

    Returns the minimum bin performance as a curriculum metric for logging.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Feed accuracy data at every reset (decoupled from reward_manager)
    if len(env_ids) > 0 and hasattr(command_term, 'get_episode_accuracy'):
        accuracy = command_term.get_episode_accuracy(env_ids)
        command_term.update_bin_performance(env_ids, accuracy)

    # Periodic range expansion + logging
    if env.common_step_counter % env.max_episode_length != 0:
        if hasattr(command_term, '_bin_perf'):
            if hasattr(command_term, '_active_mask'):
                return command_term._bin_perf[command_term._active_mask].min()
            return command_term._bin_perf.min()
        return torch.tensor(0.5, device=env.device)

    # --- Evaluation gate (every max_episode_length steps) ---
    if hasattr(command_term, '_bin_perf'):
        # Compute min over active bins only (staged discrete curriculum)
        if hasattr(command_term, '_active_mask'):
            active_perf = command_term._bin_perf[command_term._active_mask]
            min_perf = active_perf.min().item() if active_perf.numel() > 0 else 0.5
        else:
            min_perf = command_term._bin_perf.min().item()

        # Log bin stats
        if hasattr(command_term, 'log_sampling_stats'):
            command_term.log_sampling_stats()

        if min_perf > range_expand_threshold:
            if hasattr(command_term, '_active_mask'):
                # Staged discrete: expand active bin set
                command_term.expand_active_bins(n_add=2)
            elif hasattr(command_term, 'expand_outermost_bins'):
                # V2 continuous: expand only outermost bins' edges
                command_term.expand_outermost_bins(expand_rate=0.2)
            else:
                # V1 fallback: expand ranges generically
                for attr in ['lin_vel_x', 'lin_vel_y', 'ang_vel_z']:
                    current = list(getattr(ranges, attr))
                    limit = list(getattr(limit_ranges, attr))
                    new_lo = current[0] + 0.2 * (limit[0] - current[0])
                    new_hi = current[1] + 0.2 * (limit[1] - current[1])
                    setattr(ranges, attr, [new_lo, new_hi])

        return torch.tensor(min_perf, device=env.device, dtype=torch.float32)

    return torch.tensor(0.5, device=env.device, dtype=torch.float32)


def marginal_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    range_expand_threshold: float = 0.3,
    min_perf_floor: float = 0.2,
) -> torch.Tensor:
    """Companion curriculum for MarginalVelocityCommand.

    Per-direction expansion using mean(perf) instead of min(perf).
    Directions: vx_pos, vx_neg, vy, wz expand independently.
    Extra gate: no expansion if ANY active bin has perf < min_perf_floor
    (ensures no "completely non-responsive" bins exist before expanding).

    Returns the minimum of per-direction mean perfs as a curriculum metric.
    """
    command_term = env.command_manager.get_term("base_velocity")

    # Feed accuracy data at every reset
    if len(env_ids) > 0 and hasattr(command_term, 'get_episode_accuracy'):
        acc = command_term.get_episode_accuracy(env_ids)
        if isinstance(acc, tuple) and len(acc) == 3:
            command_term.update_bin_performance(env_ids, *acc)

    # Return early between evaluation gates
    if env.common_step_counter % env.max_episode_length != 0:
        if hasattr(command_term, '_vx_perf'):
            perfs = []
            for d in ('vx_pos', 'vx_neg', 'vy', 'wz'):
                idx = command_term.get_active_indices(d)
                if idx:
                    t = torch.tensor(idx, device=env.device, dtype=torch.long)
                    if d.startswith('vx'):
                        perfs.append(command_term._vx_perf[t].mean().item())
                    elif d == 'vy':
                        perfs.append(command_term._vy_perf[t].mean().item())
                    else:
                        perfs.append(command_term._wz_perf[t].mean().item())
            return torch.tensor(min(perfs) if perfs else 0.5, device=env.device, dtype=torch.float32)
        return torch.tensor(0.5, device=env.device)

    # --- Evaluation gate (every max_episode_length steps) ---
    if not hasattr(command_term, '_vx_perf'):
        return torch.tensor(0.5, device=env.device, dtype=torch.float32)

    # Log bin stats
    if hasattr(command_term, 'log_sampling_stats'):
        command_term.log_sampling_stats()

    # Per-direction expansion checks
    dir_means = {}
    for d in ('vx_pos', 'vx_neg', 'vy', 'wz'):
        idx = command_term.get_active_indices(d)
        if not idx:
            dir_means[d] = 0.5
            continue
        t = torch.tensor(idx, device=env.device, dtype=torch.long)
        if d.startswith('vx'):
            m = command_term._vx_perf[t].mean().item()
        elif d == 'vy':
            m = command_term._vy_perf[t].mean().item()
        else:
            m = command_term._wz_perf[t].mean().item()
        dir_means[d] = m

        # Min-based expansion: every active bin must exceed threshold
        if d.startswith('vx'):
            bin_perfs = command_term._vx_perf[t]
        elif d == 'vy':
            bin_perfs = command_term._vy_perf[t]
        else:
            bin_perfs = command_term._wz_perf[t]
        bin_min = bin_perfs.min().item()
        if bin_min > range_expand_threshold:
            # vx_pos/vx_neg expand by 1, vy/wz expand by 2
            n_add = 1 if d.startswith('vx') else 2
            command_term.expand_direction(d, n_add=n_add)
        elif bin_min < min_perf_floor:
            print(
                    f"[MarginalCurriculum] {d}: mean={m:.3f} > {range_expand_threshold} "
                    f"but min_bin_perf={bin_perfs.min().item():.3f} < {min_perf_floor}, "
                    f"blocking expansion (min_perf={bin_min:.3f} < floor={min_perf_floor})",
                    flush=True,
                )

    min_mean = min(dir_means.values()) if dir_means else 0.5
    return torch.tensor(min_mean, device=env.device, dtype=torch.float32)
