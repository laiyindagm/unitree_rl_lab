# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Run frnc_style_v4 offline experiment matrix.

Typical usage after collecting rollout shards:

python scripts/rsl_rl/run_style_v4_experiments.py \
  --raw_data_dirs logs/style_data/v21g_final logs/style_data/v21g_ckpts \
  --feature_dir logs/frnc_style_v4/features_v1 \
  --work_dir logs/frnc_style_v4 \
  --presets E1_reg_only E2_reg_inv E3_reg_inv_rnc E4_full
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class Job:
    name: str
    preset: str
    device: str
    out_dir: str
    train_cmd: list[str]
    probe_cmd: list[str]


def _run(cmd: list[str], dry_run: bool):
    print("[style-run] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _write_cmd(log_path: str, stage: str, name: str, cmd: list[str]):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage, "name": name, "cmd": cmd}) + "\n")


def _discover_shard_dirs() -> list[str]:
    out = set()
    for root in ["logs/style_pretrain_data", "logs/pretrain_data"]:
        for path in glob.glob(os.path.join(root, "**", "shard_*.npz"), recursive=True):
            out.add(os.path.dirname(path))
    return sorted(out)


def _has_shards(path: str) -> bool:
    direct = glob.glob(os.path.join(path, "shard_*.npz"))
    nested = glob.glob(os.path.join(path, "**", "shard_*.npz"), recursive=True)
    return bool(direct or nested)


def _validate_raw_dirs(raw_dirs: list[str]):
    missing = [d for d in raw_dirs if not _has_shards(d)]
    if not missing:
        return
    candidates = _discover_shard_dirs()
    msg = [
        "raw data dirs have no shard_*.npz:",
        *[f"  - {d}" for d in missing],
    ]
    if candidates:
        msg += ["available shard dirs:", *[f"  - {d}" for d in candidates]]
        msg.append("Use one of the available dirs, or collect the requested style data first.")
    else:
        msg.append("No existing shard dirs were found. Collect data first with collect_style_pretrain_data.py.")
    msg.append("If feature_dir already exists, rerun with --skip_features.")
    raise FileNotFoundError("\n".join(msg))


def build_jobs(args) -> list[Job]:
    devices = args.devices or [args.device]
    jobs = []
    for i, preset in enumerate(args.presets):
        out_dir = os.path.join(args.work_dir, preset)
        device = devices[i % len(devices)]
        train_cmd = [
            args.python,
            "scripts/rsl_rl/style_encoder_pretrain_v4.py",
            "--data_dir",
            args.feature_dir,
            "--out_dir",
            out_dir,
            "--preset",
            preset,
            "--device",
            device,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--save_every",
            str(args.save_every),
        ]
        if args.max_train_samples is not None:
            train_cmd += ["--max_samples", str(args.max_train_samples)]
        if args.max_train_shards is not None:
            train_cmd += ["--max_shards", str(args.max_train_shards)]
        if args.lr is not None:
            train_cmd += ["--lr", str(args.lr)]

        probe_cmd = [
            args.python,
            "scripts/rsl_rl/style_encoder_probe_v4.py",
            "--encoder",
            os.path.join(out_dir, "encoder.pt"),
            "--data_dir",
            args.feature_dir,
            "--out_json",
            os.path.join(out_dir, "probe.json"),
            "--device",
            device,
            "--batch_size",
            str(args.probe_batch_size),
            "--num_workers",
            str(args.probe_num_workers),
        ]
        if args.max_probe_samples is not None:
            probe_cmd += ["--max_samples", str(args.max_probe_samples)]
        if args.max_probe_shards is not None:
            probe_cmd += ["--max_shards", str(args.max_probe_shards)]

        jobs.append(Job(name=preset, preset=preset, device=device, out_dir=out_dir, train_cmd=train_cmd, probe_cmd=probe_cmd))
    return jobs


def run_sequential(jobs: list[Job], args, cmd_log: str):
    for job in jobs:
        os.makedirs(job.out_dir, exist_ok=True)
        _write_cmd(cmd_log, "train", job.name, job.train_cmd)
        _run(job.train_cmd, dry_run=args.dry_run)
        _write_cmd(cmd_log, "probe", job.name, job.probe_cmd)
        _run(job.probe_cmd, dry_run=args.dry_run)


def run_parallel(jobs: list[Job], args, cmd_log: str):
    pending = list(jobs)
    running: list[tuple[Job, subprocess.Popen]] = []
    while pending or running:
        while pending and len(running) < args.max_jobs:
            job = pending.pop(0)
            os.makedirs(job.out_dir, exist_ok=True)
            _write_cmd(cmd_log, "train", job.name, job.train_cmd)
            print("[style-run] " + " ".join(job.train_cmd), flush=True)
            if args.dry_run:
                proc = None
            else:
                proc = subprocess.Popen(job.train_cmd)
            running.append((job, proc))
            if args.dry_run:
                break
        if args.dry_run:
            break

        time.sleep(args.poll_s)
        still_running: list[tuple[Job, subprocess.Popen]] = []
        for job, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((job, proc))
                continue
            if code != 0:
                raise subprocess.CalledProcessError(code, job.train_cmd)
            _write_cmd(cmd_log, "probe", job.name, job.probe_cmd)
            _run(job.probe_cmd, dry_run=False)
        running = still_running


def main():
    ap = argparse.ArgumentParser(description="Run frnc_style_v4 experiment matrix.")
    ap.add_argument("--raw_data_dirs", nargs="*", default=None)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--presets", nargs="+", default=["E1_reg_only", "E2_reg_inv", "E3_reg_inv_rnc", "E4_full"])
    ap.add_argument("--skip_features", action="store_true", default=False)
    ap.add_argument("--mode", choices=["sequential", "parallel"], default="sequential")
    ap.add_argument("--max_jobs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", nargs="*", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry_run", action="store_true", default=False)

    ap.add_argument("--parent_len", type=int, default=64)
    ap.add_argument("--window_len", type=int, default=32)
    ap.add_argument("--parent_stride", type=int, default=32)
    ap.add_argument("--samples_per_parent", type=int, default=1)
    ap.add_argument("--out_shard_size", type=int, default=2048)
    ap.add_argument("--feature_seed", type=int, default=0)
    ap.add_argument("--max_input_shards", type=int, default=None)
    ap.add_argument("--max_parents", type=int, default=None)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_train_shards", type=int, default=None)

    ap.add_argument("--probe_batch_size", type=int, default=512)
    ap.add_argument("--probe_num_workers", type=int, default=0)
    ap.add_argument("--max_probe_samples", type=int, default=None)
    ap.add_argument("--max_probe_shards", type=int, default=None)
    ap.add_argument("--poll_s", type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    cmd_log = os.path.join(args.work_dir, "commands.jsonl")
    with open(os.path.join(args.work_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if not args.skip_features:
        if not args.raw_data_dirs:
            raise ValueError("--raw_data_dirs is required unless --skip_features is set")
        _validate_raw_dirs(args.raw_data_dirs)
        feat_cmd = [
            args.python,
            "scripts/rsl_rl/style_gait_features.py",
            "--data_dirs",
            *args.raw_data_dirs,
            "--out_dir",
            args.feature_dir,
            "--parent_len",
            str(args.parent_len),
            "--window_len",
            str(args.window_len),
            "--parent_stride",
            str(args.parent_stride),
            "--samples_per_parent",
            str(args.samples_per_parent),
            "--out_shard_size",
            str(args.out_shard_size),
            "--seed",
            str(args.feature_seed),
        ]
        if args.max_input_shards is not None:
            feat_cmd += ["--max_input_shards", str(args.max_input_shards)]
        if args.max_parents is not None:
            feat_cmd += ["--max_parents", str(args.max_parents)]
        _write_cmd(cmd_log, "features", "features", feat_cmd)
        _run(feat_cmd, dry_run=args.dry_run)

    jobs = build_jobs(args)
    if args.mode == "sequential":
        run_sequential(jobs, args, cmd_log)
    else:
        run_parallel(jobs, args, cmd_log)

    summary = {
        "feature_dir": args.feature_dir,
        "work_dir": args.work_dir,
        "jobs": [asdict(j) for j in jobs],
    }
    with open(os.path.join(args.work_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[style-run] done; summary={os.path.join(args.work_dir, 'summary.json')}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"[style-run] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
