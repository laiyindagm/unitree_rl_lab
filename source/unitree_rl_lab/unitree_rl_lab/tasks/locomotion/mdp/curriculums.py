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
    lin_reward_term_name: str = "track_lin_vel_xy",
    ang_reward_term_name: str = "track_ang_vel_z",
    range_expand_threshold: float = 0.5,
    range_expand_delta: float = 0.1,
) -> torch.Tensor:
    """Curriculum that feeds episode-level performance data back to the
    ``PerformanceWeightedVelocityCommand`` and progressively expands the
    sampling range.

    This is the companion to ``PerformanceWeightedVelocityCommand``.  It:

    1. Accumulates per-env (reward, bin_id) data from each reset batch.
    2. At periodic checkpoints, updates the command term's per-bin EMA.
    3. Optionally expands the active command ranges toward ``limit_ranges``
       when the *worst* bin performance exceeds ``range_expand_threshold``.

    Unlike ``speed_bucketed_vel_curriculum`` which assumes monotonic
    difficulty (low → medium → high), this approach lets the data decide
    which speed/yaw regions need more training, regardless of magnitude.

    Returns:
        Scalar metric: mean sampling entropy (higher = more uniform = mastered).
    """
    from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import (
        PerformanceWeightedVelocityCommand,
    )

    command_term = env.command_manager.get_term("base_velocity")
    if not isinstance(command_term, PerformanceWeightedVelocityCommand):
        return torch.tensor(0.0, device=env.device)

    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # -- Initialise state on first call --
    if not hasattr(env, "_pw_reward_buf"):
        env._pw_reward_buf = []  # list of dicts per reset batch

    # -- Accumulate data from every reset batch --
    if len(env_ids) > 0:
        env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

        # Linear velocity tracking reward
        lin_episode_rew = env.reward_manager._episode_sums[lin_reward_term_name][env_ids_t]
        lin_avg = lin_episode_rew / env.max_episode_length_s
        lin_term_cfg = env.reward_manager.get_term_cfg(lin_reward_term_name)
        # Normalise to [0, 1] by dividing by the reward weight (max possible)
        lin_weight = abs(lin_term_cfg.weight) if lin_term_cfg.weight > 0 else 1.0
        lin_norm = (lin_avg / lin_weight).clamp(0.0, 1.0)

        # Angular velocity tracking reward
        ang_episode_rew = env.reward_manager._episode_sums[ang_reward_term_name][env_ids_t]
        ang_avg = ang_episode_rew / env.max_episode_length_s
        ang_term_cfg = env.reward_manager.get_term_cfg(ang_reward_term_name)
        ang_weight = abs(ang_term_cfg.weight) if ang_term_cfg.weight > 0 else 1.0
        ang_norm = (ang_avg / ang_weight).clamp(0.0, 1.0)

        env._pw_reward_buf.append({
            "speed_bin": command_term._env_speed_bin[env_ids_t].detach().clone(),
            "yaw_bin": command_term._env_yaw_bin[env_ids_t].detach().clone(),
            "lin_reward": lin_norm.detach().clone(),
            "ang_reward": ang_norm.detach().clone(),
        })

    # -- Gate: only evaluate at episode boundaries --
    if env.common_step_counter % env.max_episode_length != 0:
        # Return current entropy as metric
        return _sampling_entropy(command_term._speed_probs, command_term._yaw_probs)

    # -- Gate passed: feed buffered data to command term --
    if len(env._pw_reward_buf) > 0:
        all_speed_bins = torch.cat([d["speed_bin"] for d in env._pw_reward_buf])
        all_yaw_bins = torch.cat([d["yaw_bin"] for d in env._pw_reward_buf])
        all_lin_rew = torch.cat([d["lin_reward"] for d in env._pw_reward_buf])
        all_ang_rew = torch.cat([d["ang_reward"] for d in env._pw_reward_buf])

        command_term.update_bin_performance(
            speed_bin_ids=all_speed_bins,
            speed_rewards=all_lin_rew,
            yaw_bin_ids=all_yaw_bins,
            yaw_rewards=all_ang_rew,
        )

        # -- Optionally expand range when worst bin is above threshold --
        worst_speed = command_term._speed_perf.min().item()
        worst_yaw = command_term._yaw_perf.min().item()

        if worst_speed > range_expand_threshold:
            _expand_range_toward_limit(ranges, limit_ranges, "lin_vel_x", range_expand_delta, env.device)
            _expand_range_toward_limit(ranges, limit_ranges, "lin_vel_y", range_expand_delta, env.device)

        if worst_yaw > range_expand_threshold:
            _expand_range_toward_limit(ranges, limit_ranges, "ang_vel_z", range_expand_delta, env.device)

    # Clear buffer
    env._pw_reward_buf = []

    return _sampling_entropy(command_term._speed_probs, command_term._yaw_probs)


def _expand_range_toward_limit(ranges, limit_ranges, attr: str, delta: float, device) -> None:
    """Symmetrically expand a range attribute toward its limit."""
    current = getattr(ranges, attr)
    limit = getattr(limit_ranges, attr)
    new_range = torch.clamp(
        torch.tensor(current, device=device) + torch.tensor([-delta, delta], device=device),
        limit[0],
        limit[1],
    ).tolist()
    setattr(ranges, attr, new_range)


def _sampling_entropy(speed_probs: torch.Tensor, yaw_probs: torch.Tensor) -> torch.Tensor:
    """Compute mean normalised entropy of the sampling distributions.

    Returns a value in [0, 1].  1.0 means perfectly uniform (all bins
    equally sampled).  Lower values mean the sampling is concentrated
    on specific bins.
    """
    def _norm_entropy(p: torch.Tensor) -> float:
        n = len(p)
        if n <= 1:
            return 1.0  # single bin is trivially uniform
        h = -(p * (p + 1e-8).log()).sum()
        h_max = torch.tensor(n, dtype=torch.float).log()
        return (h / h_max).clamp(0.0, 1.0).item()

    s_ent = _norm_entropy(speed_probs)
    y_ent = _norm_entropy(yaw_probs)
    return speed_probs.new_tensor((s_ent + y_ent) / 2.0)
