# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Collect rollout data for offline contrastive-encoder pre-training.

Loads a trained checkpoint (typically V21e), rolls out in IsaacLab, and dumps
per-step records as ``.npz`` shards. Intended consumers: the offline encoder
pre-training scripts (Phase 1 of the Linear Evaluation Protocol).

Per record we save:
  * ``policy_obs`` (B, D_pol)   - full flat policy observation (history+aux).
  * ``critic_obs`` (B, D_cri)   - flat critic observation (privileged).
  * ``cmd``        (B, 3)       - velocity_commands (vx, vy, wz). Used as label
                                  source for bucketing / Rank-N-Contrast.
  * ``actual_lin_vel`` (B, 3)   - base linear velocity in body frame.
  * ``actual_ang_vel`` (B, 3)   - base angular velocity in body frame.
  * ``foot_contact``  (B, F)    - bool, 1 if |force_w| > threshold per foot.
  * ``env_id``        (B,)      - parallel env index.
  * ``episode_step``  (B,)      - step within current episode.
  * ``global_step``   (B,)      - global rollout step counter.

Optional cmd override: ``--cmd_sampling stratified`` overwrites the env's
``base_velocity`` command buffer every ``--cmd_resample_steps`` with a uniform
draw from a 3-D grid over (vx, vy, wz).
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # local

parser = argparse.ArgumentParser(description="Collect data for encoder pre-training.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=4000)
parser.add_argument("--shard_steps", type=int, default=500)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cmd_sampling", type=str, default="policy",
                    choices=["policy", "stratified"])
parser.add_argument("--cmd_resample_steps", type=int, default=200)
parser.add_argument("--cmd_vx_range", type=float, nargs=2, default=[-1.0, 1.5])
parser.add_argument("--cmd_vy_range", type=float, nargs=2, default=[-0.5, 0.5])
parser.add_argument("--cmd_wz_range", type=float, nargs=2, default=[-1.5, 1.5])
parser.add_argument("--cmd_grid_bins", type=int, nargs=3, default=[7, 5, 7])
parser.add_argument("--cmd_zero_prob", type=float, default=0.05)
parser.add_argument("--foot_force_threshold", type=float, default=5.0)
parser.add_argument("--foot_body_pattern", type=str, default=".*ankle_roll.*")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.assets import retrieve_file_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def stratified_cmd_sampler(num_envs, args, device):
    bins = args.cmd_grid_bins
    centers = []
    for rng, n in zip([args.cmd_vx_range, args.cmd_vy_range, args.cmd_wz_range], bins):
        centers.append(np.linspace(rng[0], rng[1], n))
    grid = np.array(np.meshgrid(*centers, indexing="ij")).reshape(3, -1).T
    K = grid.shape[0]
    idx = np.random.randint(0, K, size=num_envs)
    cmds = grid[idx].astype(np.float32)
    if args.cmd_zero_prob > 0:
        zero_mask = np.random.rand(num_envs) < args.cmd_zero_prob
        cmds[zero_mask] = 0.0
    return torch.from_numpy(cmds).to(device)


def main():
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[collect] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner.device)

    contact_sensor = inner.scene["contact_forces"]
    foot_ids, foot_names = contact_sensor.find_bodies(args_cli.foot_body_pattern)
    print(f"[collect] foot bodies ({len(foot_names)}): {foot_names}")

    cmd_term = inner.command_manager.get_term("base_velocity")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    buf = {k: [] for k in [
        "policy_obs", "critic_obs", "cmd",
        "actual_lin_vel", "actual_ang_vel", "foot_contact",
        "env_id", "episode_step", "global_step",
    ]}
    shard_idx = 0
    steps_in_shard = 0
    episode_step = torch.zeros(args_cli.num_envs, dtype=torch.int64, device=inner.device)

    def flush():
        nonlocal shard_idx, steps_in_shard, buf
        if not buf["cmd"]:
            return
        out = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        path = os.path.join(args_cli.output_dir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(path, **out)
        n = out["cmd"].shape[0]
        print(f"[collect] wrote {path}  ({n} samples)")
        shard_idx += 1
        steps_in_shard = 0
        for k in buf:
            buf[k] = []

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs

    robot = inner.scene["robot"]

    for global_step in range(args_cli.num_steps):
        if args_cli.cmd_sampling == "stratified" and global_step % args_cli.cmd_resample_steps == 0:
            new_cmd = stratified_cmd_sampler(args_cli.num_envs, args_cli, inner.device)
            if hasattr(cmd_term, "vel_command_b"):
                cmd_term.vel_command_b[:] = new_cmd
            elif hasattr(cmd_term, "command"):
                cmd_term.command[:] = new_cmd

        with torch.inference_mode():
            actions = policy(obs)
            obs_next, _, dones, _ = env.step(actions)

            # ``obs`` is a TensorDict with keys 'policy' / 'critic'.
            policy_t = obs["policy"]
            critic_t = obs["critic"]
            if isinstance(critic_t, dict):
                critic_t = torch.cat([v for v in critic_t.values()], dim=-1)
            policy_obs_np = policy_t.detach().cpu().numpy().astype(np.float32)
            critic_obs_np = critic_t.detach().cpu().numpy().astype(np.float32)

            cmd_now = cmd_term.command.detach().cpu().numpy().astype(np.float32)
            lin_vel = robot.data.root_lin_vel_b.detach().cpu().numpy().astype(np.float32)
            ang_vel = robot.data.root_ang_vel_b.detach().cpu().numpy().astype(np.float32)

            forces = contact_sensor.data.net_forces_w[:, foot_ids, :]
            foot_contact = (forces.norm(dim=-1) > args_cli.foot_force_threshold)
            foot_contact_np = foot_contact.detach().cpu().numpy().astype(np.uint8)

        env_id = np.arange(args_cli.num_envs, dtype=np.int64)
        ep_step_np = episode_step.detach().cpu().numpy().astype(np.int64)
        gstep_np = np.full(args_cli.num_envs, global_step, dtype=np.int64)

        buf["policy_obs"].append(policy_obs_np)
        buf["critic_obs"].append(critic_obs_np)
        buf["cmd"].append(cmd_now)
        buf["actual_lin_vel"].append(lin_vel)
        buf["actual_ang_vel"].append(ang_vel)
        buf["foot_contact"].append(foot_contact_np)
        buf["env_id"].append(env_id)
        buf["episode_step"].append(ep_step_np)
        buf["global_step"].append(gstep_np)

        episode_step += 1
        if isinstance(dones, torch.Tensor):
            episode_step[dones.bool()] = 0

        steps_in_shard += 1
        if steps_in_shard >= args_cli.shard_steps:
            flush()

        obs = obs_next

    flush()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
