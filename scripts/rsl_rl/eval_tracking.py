"""Steady-state velocity-tracking accuracy evaluator.

Sweeps commands at 0.1-bin resolution across the full training ranges:
  vx in [-0.8, 1.5],  vy in [-0.5, 0.5],  wz in [-0.8, 0.8]

All bins in a sweep run IN PARALLEL: each env sub-group gets one fixed
command.  A sweep of N bins with warmup W + measure M steps costs only
(W + M) sim steps regardless of N  (vs N*(W+M) for sequential).

WHY NOT USE EXISTING PROBES:
  error_vel_xy  : absolute error, denominator = max_command_step (600),
                  includes post-resample transients  => biased & unnormalised
  curriculum lin_err: single end-of-episode snapshot  => high variance,
                  low-speed errors capped by accuracy_cmd_min

THIS SCRIPT: fixed command throughout, warmup excluded, per-env cumulative
alive mask (fallen envs excluded after first termination), trajectory-mean
velocity error per bin, multi-checkpoint comparison on the same env.

USAGE:
    ./unitree_rl_lab.sh -p scripts/rsl_rl/eval_tracking.py \\
        --task  Unitree-G1-15dof-Velocity-Rot-V21k \\
        --checkpoints \\
            /path/to/model_19999.pt \\
            /path/to/model_5000.pt \\
        --labels V21k@19k V21l@5k \\
        --envs_per_bin 4 \\
        --warmup_steps 200 \\
        --measure_steps 400 \\
        --output_csv /tmp/tracking_eval.csv \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# ---- CLI args (must come before AppLauncher) --------------------------------
parser = argparse.ArgumentParser(description="Eval steady-state velocity tracking by bin.")
parser.add_argument("--task", type=str, required=True,
                    help="Gym task name (determines env cfg and obs layout).")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                    help="Absolute path(s) to model_*.pt checkpoint file(s).")
parser.add_argument("--labels", type=str, nargs="+", default=None,
                    help="Display labels for each checkpoint (default: basename).")
parser.add_argument("--envs_per_bin", type=int, default=4,
                    help="Parallel envs per command bin (default 4). "
                         "Total envs = max_bins * envs_per_bin.")
parser.add_argument("--warmup_steps", type=int, default=200,
                    help="Steps to let robot reach steady-state (excluded from stats).")
parser.add_argument("--measure_steps", type=int, default=400,
                    help="Steps to measure per sweep (higher = less variance).")
parser.add_argument("--no_vy", action="store_true", default=False,
                    help="Skip pure-vy sweep.")
parser.add_argument("--output_csv", type=str, default=None,
                    help="Optional path to write CSV results.")
parser.add_argument("--eval_seed", type=int, default=12345,
                    help="Seed reset before every sweep. Same sweep/checkpoint gets same initial conditions.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--zero_vhat_labels", type=str, nargs="*", default=None,
                    help="Labels whose checkpoints should have v_hat zeroed during eval. "
                         "E.g. --zero_vhat_labels V21f2@19k  ")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports (after AppLauncher)."""

import csv
import importlib
import os

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# ---- Command sweep definitions ----------------------------------------------
# Matches V17c+ limit_ranges: vx[-0.8,1.5] vy[-0.5,0.5] wz[-0.8,0.8]

def _linspace_bins(lo, hi, step=0.1):
    n = round((hi - lo) / step) + 1
    return np.round(np.linspace(lo, hi, n), 3)


VX_BINS = _linspace_bins(-0.8, 1.5)   # 24 bins
VY_BINS = _linspace_bins(-0.5, 0.5)   # 11 bins
WZ_BINS = _linspace_bins(-0.8, 0.8)   # 17 bins


def _build_sweeps(include_vy):
    sweeps = [
        ("pure_vx", "vx (m/s)", VX_BINS,
         [(float(v), 0.0, 0.0) for v in VX_BINS]),
        ("pure_wz", "wz (rad/s)", WZ_BINS,
         [(0.0, 0.0, float(w)) for w in WZ_BINS]),
    ]
    if include_vy:
        sweeps.append(
            ("pure_vy", "vy (m/s)", VY_BINS,
             [(0.0, float(v), 0.0) for v in VY_BINS])
        )
    return sweeps


# ---- Runner helpers ---------------------------------------------------------

def _get_runner_cfg(task_name):
    spec = gym.spec(task_name)
    entry = spec.kwargs["rsl_rl_cfg_entry_point"]
    mod_path, cls_name = entry.rsplit(":", 1)
    return getattr(importlib.import_module(mod_path), cls_name)()


def _make_runner(env, runner_cfg, checkpoint_path):
    device = str(env.unwrapped.device)
    class_name = getattr(runner_cfg, "class_name", None) or "OnPolicyRunner"

    if class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=device)
    elif class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, runner_cfg.to_dict(), log_dir=None, device=device)
    else:
        # Custom class e.g. "unitree_rl_lab.utils.lirpg_ppo:LirpgVelocityEstimatorPPO"
        mod_path, cls = class_name.rsplit(":", 1)
        runner_cls = getattr(importlib.import_module(mod_path), cls)
        runner = runner_cls(env, runner_cfg.to_dict(), log_dir=None, device=device)

    runner.load(checkpoint_path)
    return runner



# ---- v_hat ablation helper --------------------------------------------------

def _patch_zero_vhat(policy):
    """Register a forward hook on velocity_head so its output is zeroed.

    Works for any module that has a ``velocity_head`` sub-module (e.g.
    TransformerLatentModel).  The hook is attached to the *actor* model that
    underlies the policy callable; no model weights are changed.

    Returns the hook handle (call handle.remove() to undo).
    """
    # policy is usually a bound method of the actor nn.Module, or a small
    # wrapper. We walk up to find the nn.Module.
    model = getattr(policy, "__self__", policy)  # unwrap bound method
    # VelocityEstimatorPPO stores the actor as runner.alg.actor
    # but policy = runner.get_inference_policy() returns actor.act_inference
    # which is a bound method of the actor nn.Module.
    vel_head = None
    for attr in ("velocity_head",):
        h = getattr(model, attr, None)
        if h is not None:
            vel_head = h
            break
    if vel_head is None:
        print("[eval_tracking] WARNING: velocity_head not found — zero_vhat has no effect.")
        return None

    def _zero_hook(module, input, output):  # noqa: ARG001
        return torch.zeros_like(output)

    handle = vel_head.register_forward_hook(_zero_hook)
    print("[eval_tracking]   zero_vhat hook registered on velocity_head")
    return handle


# ---- Sim helpers ------------------------------------------------------------

def _force_commands(env_unwrapped, cmds):
    """Set vel_command_b per env, prevent resampling, and align mode flags.

    Args:
        cmds: (N, 3) tensor [vx, vy, wz] per env.
    """
    cmd_term = env_unwrapped.command_manager._terms["base_velocity"]
    if cmds.shape != cmd_term.vel_command_b.shape:
        raise ValueError(
            f"Command tensor shape {tuple(cmds.shape)} does not match "
            f"base_velocity shape {tuple(cmd_term.vel_command_b.shape)}"
        )

    eps = 1e-6
    vx_zero = cmds[:, 0].abs() <= eps
    vy_zero = cmds[:, 1].abs() <= eps
    wz_zero = cmds[:, 2].abs() <= eps
    standing = vx_zero & vy_zero & wz_zero
    pure_vx = (~vx_zero) & vy_zero & wz_zero
    pure_vy = vx_zero & (~vy_zero) & wz_zero
    pure_wz = vx_zero & vy_zero & (~wz_zero)

    cmd_term.vel_command_b[:] = cmds
    # time_left <= 0 triggers resample; set large to prevent it.
    cmd_term.time_left.fill_(1e6)

    # Keep command post-processing from undoing the forced sweep command.
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = standing
    if hasattr(cmd_term, "is_rotating_env"):
        cmd_term.is_rotating_env[:] = pure_wz
    if hasattr(cmd_term, "is_pure_vx_env"):
        cmd_term.is_pure_vx_env[:] = pure_vx
    if hasattr(cmd_term, "is_pure_vy_env"):
        cmd_term.is_pure_vy_env[:] = pure_vy
    if hasattr(cmd_term, "is_linear_env"):
        cmd_term.is_linear_env[:] = False
    if hasattr(cmd_term, "is_zero_vel_x_env"):
        cmd_term.is_zero_vel_x_env[:] = vx_zero
    if hasattr(cmd_term, "is_zero_vel_y_env"):
        cmd_term.is_zero_vel_y_env[:] = vy_zero
    if hasattr(cmd_term, "is_zero_vel_yaw_env"):
        cmd_term.is_zero_vel_yaw_env[:] = wz_zero


def _actual_velocities(env_unwrapped):
    """Yaw-frame lin_vel_xy and body-z ang_vel (matching reward convention).

    Returns:
        lin_vel (N, 2): heading-aligned horizontal velocity
        ang_vel (N,):   body-z angular velocity
    """
    robot = env_unwrapped.scene["robot"]
    lin_vel = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w),
        robot.data.root_lin_vel_w[:, :3],
    )[:, :2]
    ang_vel = robot.data.root_ang_vel_b[:, 2]
    return lin_vel, ang_vel


def _compute_observations(env, update_history=False):
    """Compute observations with optional history update, matching RslRlVecEnvWrapper."""
    if hasattr(env.unwrapped, "observation_manager"):
        obs_dict = env.unwrapped.observation_manager.compute(update_history=update_history)
    else:
        obs_dict = env.unwrapped._get_observations()
    return TensorDict(obs_dict, batch_size=[env.num_envs])


def _reset_with_forced_commands(env, cmd_tensor, seed):
    """Reset env, install fixed commands, and rebuild obs history from those commands."""
    if seed is not None:
        env.seed(seed)
    env.reset()
    _force_commands(env.unwrapped, cmd_tensor)
    if hasattr(env.unwrapped, "observation_manager"):
        env.unwrapped.observation_manager.reset()
    return _compute_observations(env, update_history=True)


# ---- Per-sweep evaluation ---------------------------------------------------

def run_sweep(env, policy, sweep_cmds, envs_per_bin, warmup_steps, measure_steps, seed=None):
    """Evaluate one sweep in parallel (all bins simultaneously).

    Args:
        sweep_cmds: list of (vx, vy, wz) tuples, one per bin. Length = n_bins.
                    The env must have been created with len(sweep_cmds)*envs_per_bin envs.

    Returns dict with per-bin numpy arrays:
        abs_err_lin, std_err_lin, rel_err_lin  -- trajectory-mean linear tracking [m/s]
        abs_err_ang, std_err_ang, rel_err_ang  -- trajectory-mean angular tracking [rad/s]
        inst_abs_err_*                         -- mean step-wise error, diagnostic only
        fall_rate                              -- fraction of envs that fell
        total_steps                            -- sum of alive steps per bin
    """
    n_bins = len(sweep_cmds)
    device = env.unwrapped.device
    total_envs = n_bins * envs_per_bin

    # Build per-env command tensor (N, 3).
    cmd_tensor = torch.zeros(total_envs, 3, device=device, dtype=torch.float32)
    for i, (vx, vy, wz) in enumerate(sweep_cmds):
        lo, hi = i * envs_per_bin, (i + 1) * envs_per_bin
        cmd_tensor[lo:hi, 0] = vx
        cmd_tensor[lo:hi, 1] = vy
        cmd_tensor[lo:hi, 2] = wz

    # Reset and force command. Observation history is rebuilt after forcing,
    # otherwise policy history can contain the random command sampled at reset.
    obs = _reset_with_forced_commands(env, cmd_tensor, seed=seed)

    alive      = torch.ones(total_envs, dtype=torch.bool, device=device)
    fallen     = torch.zeros(total_envs, dtype=torch.bool, device=device)

    # Warmup: let robot reach steady state.
    for _ in range(warmup_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        done_bool = dones.bool()
        fallen |= done_bool
        alive &= ~done_bool
        _force_commands(env.unwrapped, cmd_tensor)

    # Measurement accumulators per env.
    inst_err_lin_sum  = torch.zeros(total_envs, device=device)
    inst_err_lin_sum2 = torch.zeros(total_envs, device=device)
    inst_err_ang_sum  = torch.zeros(total_envs, device=device)
    inst_err_ang_sum2 = torch.zeros(total_envs, device=device)
    step_counts       = torch.zeros(total_envs, device=device)
    vel_lin_sum       = torch.zeros(total_envs, 2, device=device)  # actual [vx, vy]
    vel_ang_sum       = torch.zeros(total_envs, device=device)     # actual wz

    for _ in range(measure_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        _force_commands(env.unwrapped, cmd_tensor)
        done_bool = dones.bool()

        lin_vel, ang_vel = _actual_velocities(env.unwrapped)
        err_lin = (lin_vel - cmd_tensor[:, :2]).norm(dim=-1)   # (N,)
        err_ang = (ang_vel - cmd_tensor[:, 2]).abs()            # (N,)

        # Exclude envs that terminated on this step. IsaacLab has already reset
        # them before we read velocities, so their current root velocity is no
        # longer part of the failed trajectory.
        step_mask = alive & ~done_bool
        mask = step_mask.float()
        inst_err_lin_sum  += err_lin * mask
        inst_err_lin_sum2 += (err_lin ** 2) * mask
        inst_err_ang_sum  += err_ang * mask
        inst_err_ang_sum2 += (err_ang ** 2) * mask
        step_counts       += mask
        vel_lin_sum       += lin_vel * mask.unsqueeze(-1)
        vel_ang_sum       += ang_vel * mask

        # Once an env terminates, exclude it from all future steps.
        fallen |= done_bool
        alive  &= ~done_bool

    # Aggregate per bin (reshape to [n_bins, envs_per_bin]).
    def rb(t):
        return t.view(n_bins, envs_per_bin)

    step_counts_r      = rb(step_counts)
    inst_err_lin_sum_r = rb(inst_err_lin_sum)
    inst_err_lin_sum2_r = rb(inst_err_lin_sum2)
    inst_err_ang_sum_r = rb(inst_err_ang_sum)
    inst_err_ang_sum2_r = rb(inst_err_ang_sum2)
    fallen_r           = rb(fallen.float())

    total_steps = step_counts_r.sum(dim=1)        # (n_bins,)
    safe_total  = total_steps.clamp(min=1.0)

    # Diagnostic only: mean of instantaneous |v_t - cmd|.
    inst_mean_lin_t = inst_err_lin_sum_r.sum(dim=1) / safe_total
    inst_mean_ang_t = inst_err_ang_sum_r.sum(dim=1) / safe_total

    # Variance via E[x^2] - (E[x])^2.
    inst_var_lin = (
        (inst_err_lin_sum2_r.sum(dim=1) / safe_total) - inst_mean_lin_t ** 2
    ).clamp(min=0)
    inst_var_ang = (
        (inst_err_ang_sum2_r.sum(dim=1) / safe_total) - inst_mean_ang_t ** 2
    ).clamp(min=0)

    fall_rate = (fallen_r.sum(dim=1) / envs_per_bin).cpu().numpy()

    # Primary metric: first average actual velocity within each trajectory,
    # then compare that trajectory-mean velocity to the command. This avoids
    # counting normal gait-cycle velocity oscillation as tracking bias.
    per_env_steps = step_counts.clamp(min=1.0)
    mean_lin_env = vel_lin_sum / per_env_steps.unsqueeze(-1)
    mean_ang_env = vel_ang_sum / per_env_steps
    traj_err_lin = (mean_lin_env - cmd_tensor[:, :2]).norm(dim=-1)
    traj_err_ang = (mean_ang_env - cmd_tensor[:, 2]).abs()
    valid_env = (~fallen) & (step_counts > 0)

    valid_r = rb(valid_env.float())
    valid_counts = valid_r.sum(dim=1)
    safe_valid = valid_counts.clamp(min=1.0)

    traj_err_lin_r = rb(traj_err_lin * valid_env.float())
    traj_err_ang_r = rb(traj_err_ang * valid_env.float())
    mean_lin_t = traj_err_lin_r.sum(dim=1) / safe_valid
    mean_ang_t = traj_err_ang_r.sum(dim=1) / safe_valid
    var_lin = (
        rb((traj_err_lin ** 2) * valid_env.float()).sum(dim=1) / safe_valid
        - mean_lin_t ** 2
    ).clamp(min=0)
    var_ang = (
        rb((traj_err_ang ** 2) * valid_env.float()).sum(dim=1) / safe_valid
        - mean_ang_t ** 2
    ).clamp(min=0)

    mean_lin = mean_lin_t.cpu().numpy()
    mean_ang = mean_ang_t.cpu().numpy()
    std_lin = var_lin.sqrt().cpu().numpy()
    std_ang = var_ang.sqrt().cpu().numpy()

    mean_lin_env_r = mean_lin_env.view(n_bins, envs_per_bin, 2)
    mean_ang_env_r = mean_ang_env.view(n_bins, envs_per_bin)
    mean_actual_vx = (
        (mean_lin_env_r[:, :, 0] * valid_r).sum(dim=1) / safe_valid
    ).cpu().numpy()
    mean_actual_vy = (
        (mean_lin_env_r[:, :, 1] * valid_r).sum(dim=1) / safe_valid
    ).cpu().numpy()
    mean_actual_wz = (
        (mean_ang_env_r * valid_r).sum(dim=1) / safe_valid
    ).cpu().numpy()

    valid_counts_np = valid_counts.cpu().numpy()
    no_valid = valid_counts_np <= 0
    for arr in (mean_lin, mean_ang, std_lin, std_ang, mean_actual_vx, mean_actual_vy, mean_actual_wz):
        arr[no_valid] = np.nan

    # Relative errors (nan where command magnitude is zero).
    cmd_lin_mag = np.array([(vx**2 + vy**2)**0.5 for vx, vy, _ in sweep_cmds])
    cmd_ang_mag = np.array([abs(wz) for _, _, wz in sweep_cmds])
    rel_lin = np.where(cmd_lin_mag > 0.01, mean_lin / cmd_lin_mag, np.nan)
    rel_ang = np.where(cmd_ang_mag > 0.01, mean_ang / cmd_ang_mag, np.nan)

    return {
        "abs_err_lin": mean_lin,
        "std_err_lin": std_lin,
        "rel_err_lin": rel_lin,
        "abs_err_ang": mean_ang,
        "std_err_ang": std_ang,
        "rel_err_ang": rel_ang,
        "fall_rate":   fall_rate,
        "total_steps": total_steps.cpu().numpy(),
        "valid_traj_count": valid_counts_np,
        "mean_actual_vx": mean_actual_vx,
        "mean_actual_vy": mean_actual_vy,
        "mean_actual_wz": mean_actual_wz,
        "inst_abs_err_lin": inst_mean_lin_t.cpu().numpy(),
        "inst_std_err_lin": inst_var_lin.sqrt().cpu().numpy(),
        "inst_abs_err_ang": inst_mean_ang_t.cpu().numpy(),
        "inst_std_err_ang": inst_var_ang.sqrt().cpu().numpy(),
    }


# ---- Printing ---------------------------------------------------------------

def _fv(v, d=4):
    if np.isnan(v): return "  ---  "
    return f"{v:.{d}f}"


def _fp(v):
    if np.isnan(v): return "  ---  "
    return f"{v*100:5.1f}%"


def print_sweep_table(sweep_name, axis_label, bins, all_results):
    labels = [lbl for lbl, _ in all_results]
    cols = ["traj_lin", "+-std", "rel_lin%", "traj_ang", "+-std", "rel_ang%", "fall%"]
    cw = 8

    print(f"\n{'='*90}")
    print(f"  Sweep: {sweep_name}   ({axis_label})")
    print(f"{'='*90}")

    # Checkpoint label headers
    lbl_row = f"{'cmd':>9}  "
    for lbl in labels:
        block_w = len(cols) * (cw + 2) - 2
        lbl_row += f"[{lbl}]".center(block_w) + "    "
    print(lbl_row)

    # Column sub-headers
    sub = f"{'':>9}  "
    for _ in labels:
        sub += "  ".join(f"{h:>{cw}}" for h in cols) + "    "
    print(sub)
    print("-" * max(90, len(sub)))

    for i, bv in enumerate(bins):
        row = f"{bv:>9.2f}  "
        for _, res in all_results:
            row += (
                f"{_fv(res['abs_err_lin'][i]):>{cw}}  "
                f"{_fv(res['std_err_lin'][i]):>{cw}}  "
                f"{_fp(res['rel_err_lin'][i]):>{cw}}  "
                f"{_fv(res['abs_err_ang'][i]):>{cw}}  "
                f"{_fv(res['std_err_ang'][i]):>{cw}}  "
                f"{_fp(res['rel_err_ang'][i]):>{cw}}  "
                f"{_fp(res['fall_rate'][i]):>{cw}}  "
                f"    "
            )
        print(row)

    print("=" * 90 + "\n")


def write_csv(sweeps_data, all_labels, path):
    suffixes = ["abs_err_lin", "std_err_lin", "rel_err_lin",
                "abs_err_ang", "std_err_ang", "rel_err_ang",
                "fall_rate", "total_steps", "valid_traj_count",
                "mean_actual_vx", "mean_actual_vy", "mean_actual_wz",
                "inst_abs_err_lin", "inst_std_err_lin",
                "inst_abs_err_ang", "inst_std_err_ang"]
    fields = ["sweep", "axis_label", "bin_value"] + [
        f"{lbl}/{s}" for lbl in all_labels for s in suffixes
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sweep_name, axis_label, bins, all_results in sweeps_data:
            for i, bv in enumerate(bins):
                row = {"sweep": sweep_name, "axis_label": axis_label,
                       "bin_value": float(bv)}
                for lbl, res in all_results:
                    for s in suffixes:
                        v = res[s][i]
                        row[f"{lbl}/{s}"] = "" if np.isnan(float(v)) else float(v)
                writer.writerow(row)
    print(f"[eval_tracking] CSV saved to {path}")


# ---- Main -------------------------------------------------------------------

def main():
    labels = args_cli.labels
    if labels is None:
        labels = [os.path.basename(p) for p in args_cli.checkpoints]
    if len(labels) != len(args_cli.checkpoints):
        raise ValueError(
            f"--labels length ({len(labels)}) != "
            f"--checkpoints length ({len(args_cli.checkpoints)})"
        )

    sweeps = _build_sweeps(include_vy=not args_cli.no_vy)
    max_n_bins = max(len(s[2]) for s in sweeps)
    total_envs = max_n_bins * args_cli.envs_per_bin

    print(f"\n[eval_tracking] Task          : {args_cli.task}")
    print(f"[eval_tracking] Sweeps        : {[s[0] for s in sweeps]}")
    print(f"[eval_tracking] Bins          : vx={len(VX_BINS)}, wz={len(WZ_BINS)}, vy={len(VY_BINS)}")
    print(f"[eval_tracking] Envs/bin      : {args_cli.envs_per_bin}  (total: {total_envs})")
    print(f"[eval_tracking] Warmup        : {args_cli.warmup_steps} steps "
          f"({args_cli.warmup_steps * 0.02:.1f} s)")
    print(f"[eval_tracking] Measure       : {args_cli.measure_steps} steps "
          f"({args_cli.measure_steps * 0.02:.1f} s)")
    print(f"[eval_tracking] Eval seed     : {args_cli.eval_seed}")
    print("[eval_tracking] Primary error : |mean_t(actual_velocity) - command| per survived trajectory")
    print(f"[eval_tracking] Checkpoints ({len(args_cli.checkpoints)}):")
    for lbl, ckpt in zip(labels, args_cli.checkpoints):
        print(f"   {lbl:<26s}  {ckpt}")
    print()

    # Build env (play cfg).
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=total_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # Episode must outlast warmup + measure so time-limit terminations don't trigger.
    env_cfg.episode_length_s = (args_cli.warmup_steps + args_cli.measure_steps + 20) * 0.02

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    runner_cfg = _get_runner_cfg(args_cli.task)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=runner_cfg.clip_actions)

    # sweeps_data: list of (sweep_name, axis_label, bins, [(label, result_dict)])
    sweeps_data = [(sn, al, bins, []) for sn, al, bins, _ in sweeps]

    for ckpt_idx, (lbl, ckpt_path) in enumerate(zip(labels, args_cli.checkpoints)):
        print(f"[eval_tracking] == Checkpoint {ckpt_idx+1}/{len(labels)}: {lbl} ==")
        runner = _make_runner(env, runner_cfg, ckpt_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        zero_vhat_set = set(args_cli.zero_vhat_labels or [])
        vhat_hook = _patch_zero_vhat(policy) if lbl in zero_vhat_set else None

        for sweep_idx, (sweep_name, axis_label, bins, sweep_cmds) in enumerate(sweeps):
            n_bins = len(bins)
            # Pad to max_n_bins with zero-commands (extra envs, results discarded).
            padded_cmds = sweep_cmds + [(0.0, 0.0, 0.0)] * (max_n_bins - n_bins)

            print(f"[eval_tracking]   {sweep_name}: {n_bins} bins, "
                  f"{n_bins * args_cli.envs_per_bin}/{total_envs} active envs ...",
                  flush=True)

            res = run_sweep(
                env=env,
                policy=policy,
                sweep_cmds=padded_cmds,
                envs_per_bin=args_cli.envs_per_bin,
                warmup_steps=args_cli.warmup_steps,
                measure_steps=args_cli.measure_steps,
                seed=args_cli.eval_seed + sweep_idx,
            )
            # Trim padding bins.
            trimmed = {k: v[:n_bins] for k, v in res.items()}
            sweeps_data[sweep_idx][3].append((lbl, trimmed))

        print(f"[eval_tracking]   done: {lbl}\n")
        if vhat_hook is not None:
            vhat_hook.remove()
            print("[eval_tracking]   zero_vhat hook removed")

    # Print per-sweep tables.
    for sweep_name, axis_label, bins, all_results in sweeps_data:
        print_sweep_table(sweep_name, axis_label, bins, all_results)

    # Summary: mean metrics over non-trivial bins.
    print("[eval_tracking] === Summary (mean over non-trivial bins) ===")
    for lbl in labels:
        for sweep_name, _, _, all_results in sweeps_data:
            res = next(r for l, r in all_results if l == lbl)
            rl = res["rel_err_lin"][~np.isnan(res["rel_err_lin"])]
            ra = res["rel_err_ang"][~np.isnan(res["rel_err_ang"])]
            fr = res["fall_rate"]
            parts = []
            if len(rl): parts.append(f"rel_lin={np.mean(rl)*100:.1f}%")
            if len(ra): parts.append(f"rel_ang={np.mean(ra)*100:.1f}%")
            parts.append(f"fall={np.mean(fr)*100:.1f}%")
            print(f"  {lbl:<26s} {sweep_name:<12s}  " + "  ".join(parts))
    print()

    # Actual velocity summary per sweep × checkpoint.
    print("[eval_tracking] === Actual mean velocity per bin (cmd → achieved) ===\n")
    for sweep_name, axis_label, bins, all_results in sweeps_data:
        print(f"  Sweep: {sweep_name}  ({axis_label})")
        hdr = f"  {'cmd':>8}"
        for lbl, _ in all_results:
            hdr += f"  {lbl:>28}"
        print(hdr)
        sub = f"  {'':>8}"
        for _ in all_results:
            sub += f"  {'act_vx':>8}  {'act_vy':>8}  {'act_wz':>8}"
        print(sub)
        print("  " + "-" * (8 + len(all_results) * 30))
        for i, bv in enumerate(bins):
            row = f"  {bv:>8.2f}"
            for _, res in all_results:
                row += (f"  {res['mean_actual_vx'][i]:>8.4f}"
                        f"  {res['mean_actual_vy'][i]:>8.4f}"
                        f"  {res['mean_actual_wz'][i]:>8.4f}")
            print(row)
        print()

    if args_cli.output_csv:
        write_csv(sweeps_data, labels, args_cli.output_csv)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
