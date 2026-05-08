# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Collect rollout shards for frnc_style_v4 offline pretraining.

This is a superset of collect_pretrain_data.py.  In addition to policy obs,
commands, velocities, and contacts, it stores raw actions and robot body state
needed to build phase-invariant gait-style targets offline.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # local


parser = argparse.ArgumentParser(description="Collect data for style encoder pretraining.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=4000)
parser.add_argument("--shard_steps", type=int, default=500)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cmd_sampling", type=str, default="policy", choices=["policy", "stratified"])
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
import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.assets import retrieve_file_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def stratified_cmd_sampler(num_envs, args, device):
    centers = []
    for rng, n_bins in zip([args.cmd_vx_range, args.cmd_vy_range, args.cmd_wz_range], args.cmd_grid_bins):
        centers.append(np.linspace(rng[0], rng[1], n_bins))
    grid = np.array(np.meshgrid(*centers, indexing="ij")).reshape(3, -1).T
    idx = np.random.randint(0, grid.shape[0], size=num_envs)
    cmds = grid[idx].astype(np.float32)
    if args.cmd_zero_prob > 0.0:
        zero_mask = np.random.rand(num_envs) < args.cmd_zero_prob
        cmds[zero_mask] = 0.0
    return torch.from_numpy(cmds).to(device)


def _to_np(x, dtype=np.float32):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(dtype, copy=False)


def _maybe_robot_body_ids(robot, pattern: str):
    try:
        body_ids, body_names = robot.find_bodies(pattern, preserve_order=True)
        return body_ids, list(body_names)
    except Exception as exc:
        print(f"[collect-style] could not resolve robot body ids for {pattern!r}: {exc}")
        return None, []


def _append(buf: dict[str, list[np.ndarray]], key: str, value):
    arr = _to_np(value)
    if arr is None:
        return
    if key not in buf:
        buf[key] = []
    buf[key].append(arr)


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
    print(f"[collect-style] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inner = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=inner.device)

    contact_sensor = inner.scene["contact_forces"]
    contact_foot_ids, contact_foot_names = contact_sensor.find_bodies(args_cli.foot_body_pattern)
    print(f"[collect-style] contact foot bodies ({len(contact_foot_names)}): {contact_foot_names}")

    robot = inner.scene["robot"]
    robot_foot_ids, robot_foot_names = _maybe_robot_body_ids(robot, args_cli.foot_body_pattern)
    if robot_foot_ids is not None:
        print(f"[collect-style] robot foot bodies ({len(robot_foot_names)}): {robot_foot_names}")

    cmd_term = inner.command_manager.get_term("base_velocity")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_envs": args_cli.num_envs,
        "num_steps": args_cli.num_steps,
        "shard_steps": args_cli.shard_steps,
        "cmd_sampling": args_cli.cmd_sampling,
        "cmd_grid_bins": args_cli.cmd_grid_bins,
        "cmd_ranges": {
            "vx": args_cli.cmd_vx_range,
            "vy": args_cli.cmd_vy_range,
            "wz": args_cli.cmd_wz_range,
        },
        "foot_force_threshold": args_cli.foot_force_threshold,
        "foot_body_pattern": args_cli.foot_body_pattern,
        "contact_foot_names": list(contact_foot_names),
        "robot_foot_names": list(robot_foot_names),
    }
    with open(os.path.join(args_cli.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    base_keys = [
        "policy_obs",
        "critic_obs",
        "cmd",
        "actual_lin_vel",
        "actual_ang_vel",
        "action",
        "foot_contact",
        "env_id",
        "episode_step",
        "global_step",
    ]
    buf: dict[str, list[np.ndarray]] = {k: [] for k in base_keys}
    shard_idx = 0
    steps_in_shard = 0
    episode_step = torch.zeros(args_cli.num_envs, dtype=torch.int64, device=inner.device)

    def flush():
        nonlocal shard_idx, steps_in_shard, buf
        if not buf["cmd"]:
            return
        out = {k: np.concatenate(v, axis=0) for k, v in buf.items() if v}
        out["foot_contact_names"] = np.asarray(contact_foot_names)
        out["robot_foot_names"] = np.asarray(robot_foot_names)
        path = os.path.join(args_cli.output_dir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(path, **out)
        n = out["cmd"].shape[0]
        keys = ", ".join(sorted(out.keys()))
        print(f"[collect-style] wrote {path} ({n} samples; keys={keys})")
        shard_idx += 1
        steps_in_shard = 0
        for k in list(buf.keys()):
            buf[k] = []

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs

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

            policy_t = obs["policy"]
            critic_t = obs["critic"]
            if isinstance(critic_t, dict):
                critic_t = torch.cat([v for v in critic_t.values()], dim=-1)

            forces = contact_sensor.data.net_forces_w[:, contact_foot_ids, :]
            foot_contact = forces.norm(dim=-1) > args_cli.foot_force_threshold

            _append(buf, "policy_obs", policy_t)
            _append(buf, "critic_obs", critic_t)
            _append(buf, "cmd", cmd_term.command)
            _append(buf, "actual_lin_vel", robot.data.root_lin_vel_b)
            _append(buf, "actual_ang_vel", robot.data.root_ang_vel_b)
            _append(buf, "action", actions)
            _append(buf, "foot_contact", foot_contact.to(torch.uint8))
            _append(buf, "root_pos_w", getattr(robot.data, "root_pos_w", None))
            _append(buf, "root_quat_w", getattr(robot.data, "root_quat_w", None))
            _append(buf, "joint_pos", getattr(robot.data, "joint_pos", None))
            _append(buf, "joint_vel", getattr(robot.data, "joint_vel", None))

            if robot_foot_ids is not None:
                if hasattr(robot.data, "body_pos_w"):
                    _append(buf, "foot_pos_w", robot.data.body_pos_w[:, robot_foot_ids, :])
                if hasattr(robot.data, "body_lin_vel_w"):
                    _append(buf, "foot_lin_vel_w", robot.data.body_lin_vel_w[:, robot_foot_ids, :])

        env_id = np.arange(args_cli.num_envs, dtype=np.int64)
        ep_step_np = episode_step.detach().cpu().numpy().astype(np.int64)
        gstep_np = np.full(args_cli.num_envs, global_step, dtype=np.int64)
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
