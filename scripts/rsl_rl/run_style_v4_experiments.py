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
import csv
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
    seed: int
    device: str
    out_dir: str
    train_cmd: list[str]
    probe_cmd: list[str]
    probe_json: str


MATRIX_SPECS = {
    "v4_1_screen": [
        ("B0_E4_base", "E4_full", {}),
        ("B1_E3_rnc", "E3_reg_inv_rnc", {}),
        ("B2_rank_var010_cov001", "E4_full", {"l_var": 0.10, "l_cov": 0.01}),
        ("B3_rank_var020_cov002", "E4_full", {"l_var": 0.20, "l_cov": 0.02}),
        ("B4_inv050_rank", "E4_full", {"l_inv": 0.5, "l_var": 0.10, "l_cov": 0.01}),
        ("B5_phase_adv005", "E4_full", {"l_var": 0.10, "l_cov": 0.01, "l_phase_adv": 0.05}),
        ("B6_phase_adv010", "E4_full", {"l_var": 0.10, "l_cov": 0.01, "l_phase_adv": 0.10}),
        (
            "C1_phase_adv005_phiK3_dec256",
            "E4_full",
            {"l_var": 0.10, "l_cov": 0.01, "l_phase_adv": 0.05, "phi_fourier_k": 3, "yphi_hidden_dim": 256},
        ),
    ],
    "v4_2_loss": [
        ("L1_inv050_adv010", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("L2_inv050_adv020", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.20, "l_var": 0.10, "l_cov": 0.01}),
        ("L3_inv075_adv010", "E4_full", {"l_inv": 0.75, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("L4_inv075_adv020", "E4_full", {"l_inv": 0.75, "l_phase_adv": 0.20, "l_var": 0.10, "l_cov": 0.01}),
        ("L5_inv050_adv010_yphi_stop", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "yphi_encoder_grad": "stopgrad"}),
        ("L6_inv050_adv010_yphi_off", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "yphi_encoder_grad": "off"}),
        ("L7_inv075_adv010_yphi_stop", "E4_full", {"l_inv": 0.75, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "yphi_encoder_grad": "stopgrad"}),
        ("L8_inv050_adv010_alt3", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "phase_adv_mode": "alternating", "phase_adv_steps": 3, "l_var": 0.10, "l_cov": 0.01}),
        ("L9_inv050_adv010_corr005", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_phase_corr": 0.05, "l_var": 0.10, "l_cov": 0.01}),
        ("L10_inv075_adv010_corr005", "E4_full", {"l_inv": 0.75, "l_phase_adv": 0.10, "l_phase_corr": 0.05, "l_var": 0.10, "l_cov": 0.01}),
    ],
    "v4_2_target": [
        ("T1_full_metric", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "metric_y0_groups": "full"}),
        ("T2_no_action_metric", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "metric_y0_groups": "no_action"}),
        ("T3_kinematic_metric", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "metric_y0_groups": "kinematic_only"}),
        ("T4_yphi025_full", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "l_yphi": 0.25}),
        ("T5_yphi_stop", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "yphi_encoder_grad": "stopgrad"}),
    ],
    "v4_2_arch": [
        ("A1_tcn", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "tcn"}),
        ("A2_stats_tcn", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "stats_tcn"}),
        ("A3_stats_only", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "stats_only"}),
        ("A4_dual_latent", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "dual_latent"}),
        ("A5_stats_tcn_dual", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "stats_tcn_dual"}),
    ],
    "v4_3_confirm": [
        ("C1_stats_tcn_50", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "stats_tcn"}),
        ("C2_stats_tcn_dual_50", "E4_full", {"l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "encoder_kind": "stats_tcn_dual"}),
    ],
    "v4_3_grid": [
        ("G1_stats_rank015", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.15, "l_cov": 0.015}),
        ("G2_stats_adv015", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.15, "l_var": 0.10, "l_cov": 0.01}),
        ("G3_stats_alt3", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.10, "phase_adv_mode": "alternating", "phase_adv_steps": 3, "l_var": 0.10, "l_cov": 0.01}),
        ("G4_stats_kin_metric", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "metric_y0_groups": "kinematic_only"}),
        ("G5_stats_no_action_metric", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "metric_y0_groups": "no_action"}),
        ("G6_stats_phiK3_dec256", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "phi_fourier_k": 3, "yphi_hidden_dim": 256}),
        ("G7_stats_yphi025", "E4_full", {"encoder_kind": "stats_tcn", "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01, "l_yphi": 0.25}),
        ("G8_stats_dual_aux16", "E4_full", {"encoder_kind": "stats_tcn_dual", "d_aux": 16, "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
    ],
    "v4_5_screen": [
        ("S0_c2_v2_uniform", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "uniform", "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("S1_c2_v2_balanced", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "sample_weight_cap": 20.0, "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("S2_c2_v3_dense_balanced", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "sample_weight_cap": 20.0, "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("N1_cmd005_mode005", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "l_cmd_adv": 0.05, "l_mode_adv": 0.05, "l_var": 0.10, "l_cov": 0.01}),
        ("N2_cmd010_mode005", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "l_cmd_adv": 0.10, "l_mode_adv": 0.05, "l_var": 0.10, "l_cov": 0.01}),
        ("N3_cmd005_mode010", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "l_cmd_adv": 0.05, "l_mode_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("N4_cmdmode_alt3", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "l_cmd_adv": 0.05, "l_mode_adv": 0.05, "cmd_mode_adv_mode": "alternating", "cmd_mode_adv_steps": 3, "l_var": 0.10, "l_cov": 0.01}),
        ("G1_phase_adv015", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.15, "l_var": 0.10, "l_cov": 0.01}),
        ("G2_phase_alt3", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "phase_adv_mode": "alternating", "phase_adv_steps": 3, "l_var": 0.10, "l_cov": 0.01}),
        ("G3_rnc_res100_delta020", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "l_rnc_res": 1.0, "cond_cmd_delta": 0.20, "l_var": 0.10, "l_cov": 0.01}),
        ("G4_adv015_rnc100", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.15, "l_rnc_res": 1.0, "l_var": 0.10, "l_cov": 0.01}),
        ("G5_yphi025", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "l_inv": 0.5, "l_phase_adv": 0.10, "l_yphi": 0.25, "l_var": 0.10, "l_cov": 0.01}),
        ("M1_back256", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "d_back": 256, "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("M2_aux16", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "d_aux": 16, "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
        ("M3_phiK3_dec256_aux16", "E4_full", {"encoder_kind": "stats_tcn_dual", "sample_strategy": "source_bucket_balanced", "d_aux": 16, "phi_fourier_k": 3, "yphi_hidden_dim": 256, "l_inv": 0.5, "l_phase_adv": 0.10, "l_var": 0.10, "l_cov": 0.01}),
    ],
}


TRAIN_OVERRIDE_FIELDS = [
    "d_back",
    "d_gait",
    "d_aux",
    "encoder_kind",
    "phi_fourier_k",
    "yphi_hidden_dim",
    "yphi_encoder_grad",
    "phase_adv_mode",
    "phase_adv_steps",
    "phase_adv_hidden",
    "cmd_mode_adv_mode",
    "cmd_mode_adv_steps",
    "cmd_mode_adv_hidden",
    "metric_y0_groups",
    "sample_strategy",
    "sample_weight_cap",
    "dataset_device",
    "weight_decay",
    "tau",
    "cond_cmd_delta",
    "rnc_max_pos",
    "max_grad_norm",
    "l_y0",
    "l_yphi",
    "l_inv",
    "l_rnc_g",
    "l_rnc_res",
    "l_var",
    "l_cov",
    "l_phase_adv",
    "l_phase_corr",
    "l_cmd_adv",
    "l_mode_adv",
]


SUMMARY_KEYS = [
    "split",
    "heldout_source_group",
    "n_samples",
    "n_eval_samples",
    "mask_kind",
    "probe_backend",
    "probe_device",
    "R2_Y0_from_Z",
    "R2_Y0_from_cmd_mode",
    "R2_Y0_gain_over_cmd_mode",
    "R2_cmd_from_Z",
    "R2_cmd_mlp_from_Z",
    "R2_mode_from_Z",
    "R2_mode_mlp_from_Z",
    "R2_phase_from_Z",
    "R2_phase_mlp_from_Z",
    "R2_Y0_from_Zres",
    "R2_Y0res_retention",
    "R2_Z_from_cmd_mode_linear",
    "R2_cmd_from_Zres",
    "R2_cmd_mlp_from_Zres",
    "R2_mode_from_Zres",
    "R2_mode_mlp_from_Zres",
    "R2_phase_from_Zres",
    "R2_phase_mlp_from_Zres",
    "shift_ratio",
    "effective_rank",
    "effective_rank_Zres",
    "shortcut_delta_ratio",
    "rho_G_spearman",
    "rho_G_cond_cmd_spearman",
    "rho_G_Zres_cond_cmd_spearman",
    "R2_Yphi_from_Z_phi",
    "R2_Yphi_gain_over_phi",
    "phase_shift_ratio",
    "phase_shift_same_parent_dist",
    "phase_shift_same_bucket_dist",
    "phase_shift_n_parent_groups",
    "OOD_drop_R2_Y0",
    "OOD_drop_rho_G",
    "OOD_drop_rho_G_cond",
]


def _run(cmd: list[str], dry_run: bool):
    print("[style-run] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _write_cmd(log_path: str, stage: str, name: str, cmd: list[str]):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage, "name": name, "cmd": cmd}) + "\n")


def _add_kv_args(cmd: list[str], values: dict[str, object]):
    for key, value in values.items():
        if value is None:
            continue
        cmd += [f"--{key}", str(value)]


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
    if args.matrix == "none":
        specs = [(preset, preset, {}) for preset in args.presets]
    else:
        specs = MATRIX_SPECS[args.matrix]
    if args.only_names:
        keep = set(args.only_names)
        specs = [spec for spec in specs if spec[0] in keep]
        missing = sorted(keep - {spec[0] for spec in specs})
        if missing:
            raise ValueError(f"--only_names not found in matrix {args.matrix}: {missing}")
    seeds = args.seeds if args.seeds is not None and len(args.seeds) > 0 else [args.seed]
    append_seed = len(seeds) > 1
    for name, preset, matrix_args in specs:
        for seed in seeds:
            job_name = f"{name}_s{seed}" if append_seed else name
            out_dir = os.path.join(args.work_dir, job_name)
            device = devices[len(jobs) % len(devices)]
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
                "--seed",
                str(seed),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--save_every",
                str(args.save_every),
            ]
            _add_kv_args(train_cmd, matrix_args)
            _add_kv_args(train_cmd, {k: getattr(args, k) for k in TRAIN_OVERRIDE_FIELDS})
            if args.max_train_samples is not None:
                train_cmd += ["--max_samples", str(args.max_train_samples)]
            if args.max_train_shards is not None:
                train_cmd += ["--max_shards", str(args.max_train_shards)]
            if args.lr is not None:
                train_cmd += ["--lr", str(args.lr)]

            probe_json = os.path.join(out_dir, "probe.json")
            probe_cmd = [
                args.python,
                "scripts/rsl_rl/style_encoder_probe_v4.py",
                "--encoder",
                os.path.join(out_dir, "encoder.pt"),
                "--data_dir",
                args.feature_dir,
                "--out_json",
                probe_json,
                "--device",
                device,
                "--batch_size",
                str(args.probe_batch_size),
                "--num_workers",
                str(args.probe_num_workers),
                "--split",
                args.probe_split,
                "--seed",
                str(seed),
            ]
            if args.heldout_source_group is not None:
                probe_cmd += ["--heldout_source_group", args.heldout_source_group]
            if args.max_probe_samples is not None:
                probe_cmd += ["--max_samples", str(args.max_probe_samples)]
            if args.max_probe_shards is not None:
                probe_cmd += ["--max_shards", str(args.max_probe_shards)]
            if args.max_frame_samples is not None:
                probe_cmd += ["--max_frame_samples", str(args.max_frame_samples)]
            if args.mlp_probe_max_samples is not None:
                probe_cmd += ["--mlp_probe_max_samples", str(args.mlp_probe_max_samples)]
            if args.mlp_probe_epochs is not None:
                probe_cmd += ["--mlp_probe_epochs", str(args.mlp_probe_epochs)]
            if args.mlp_probe_lr is not None:
                probe_cmd += ["--mlp_probe_lr", str(args.mlp_probe_lr)]
            if args.mlp_probe_batch_size is not None:
                probe_cmd += ["--mlp_probe_batch_size", str(args.mlp_probe_batch_size)]
            if args.probe_backend is not None:
                probe_cmd += ["--probe_backend", args.probe_backend]
            if args.phase_shift_max_pairs is not None:
                probe_cmd += ["--phase_shift_max_pairs", str(args.phase_shift_max_pairs)]
            if args.skip_mlp_probe:
                probe_cmd += ["--skip_mlp_probe"]

            jobs.append(Job(name=job_name, preset=preset, seed=seed, device=device, out_dir=out_dir, train_cmd=train_cmd, probe_cmd=probe_cmd, probe_json=probe_json))
    return jobs


def write_probe_summary(work_dir: str, jobs: list[Job]):
    path = os.path.join(work_dir, "probe_summary.csv")
    rows = []
    for job in jobs:
        row = {"name": job.name, "preset": job.preset, "seed": job.seed, "probe_json": job.probe_json}
        if os.path.exists(job.probe_json):
            with open(job.probe_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in SUMMARY_KEYS:
                row[key] = data.get(key)
        rows.append(row)
    fields = ["name", "preset", "seed", "probe_json", *SUMMARY_KEYS]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[style-run] wrote {path}")
    write_diagnostic_summaries(work_dir, rows)


def _float_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if out == out else float("nan")


def write_diagnostic_summaries(work_dir: str, rows: list[dict[str, object]]):
    residual_keys = [
        "R2_Y0_from_Zres",
        "R2_Y0res_retention",
        "R2_Z_from_cmd_mode_linear",
        "R2_cmd_from_Zres",
        "R2_cmd_mlp_from_Zres",
        "R2_mode_from_Zres",
        "R2_mode_mlp_from_Zres",
        "R2_phase_from_Zres",
        "R2_phase_mlp_from_Zres",
        "effective_rank_Zres",
        "rho_G_Zres_cond_cmd_spearman",
    ]
    phase_keys = [
        "R2_phase_from_Z",
        "R2_phase_mlp_from_Z",
        "phase_shift_ratio",
        "phase_shift_same_parent_dist",
        "phase_shift_same_bucket_dist",
        "phase_shift_n_parent_groups",
    ]
    for filename, keys in [("residual_summary.csv", residual_keys), ("phase_shift_summary.csv", phase_keys)]:
        path = os.path.join(work_dir, filename)
        fields = ["name", "preset", "seed", "probe_json", *keys]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})
        print(f"[style-run] wrote {path}")

    ranked_path = os.path.join(work_dir, "ranked_candidates.csv")
    ranked = []
    for row in rows:
        r2 = _float_or_nan(row.get("R2_Y0_from_Z"))
        phase = _float_or_nan(row.get("R2_phase_from_Z"))
        rank = _float_or_nan(row.get("effective_rank"))
        rho = _float_or_nan(row.get("rho_G_Zres_cond_cmd_spearman"))
        if rho != rho:
            rho = _float_or_nan(row.get("rho_G_cond_cmd_spearman"))
        ood = _float_or_nan(row.get("OOD_drop_R2_Y0"))
        cmd_res = _float_or_nan(row.get("R2_cmd_from_Zres"))
        phase_res = _float_or_nan(row.get("R2_phase_from_Zres"))
        y0_res_ret = _float_or_nan(row.get("R2_Y0res_retention"))
        if y0_res_ret == y0_res_ret:
            y0_res_ret = max(0.0, min(y0_res_ret, 1.5))
        cmd_mlp_res = _float_or_nan(row.get("R2_cmd_mlp_from_Zres"))
        mode_mlp_res = _float_or_nan(row.get("R2_mode_mlp_from_Zres"))
        score = (
            (r2 if r2 == r2 else -1.0)
            + 0.30 * (rho if rho == rho else 0.0)
            + 0.02 * (rank if rank == rank else 0.0)
            + 0.20 * (y0_res_ret if y0_res_ret == y0_res_ret else 0.0)
            - 0.5 * (phase if phase == phase else 1.0)
            - 0.3 * (ood if ood == ood else 0.2)
            - 0.15 * max(cmd_res if cmd_res == cmd_res else 0.0, 0.0)
            - 0.50 * max(phase_res if phase_res == phase_res else 0.0, 0.0)
            - 0.40 * max(cmd_mlp_res if cmd_mlp_res == cmd_mlp_res else 0.0, 0.0)
            - 0.25 * max(mode_mlp_res if mode_mlp_res == mode_mlp_res else 0.0, 0.0)
        )
        ranked.append({**row, "selection_score": score})
    ranked.sort(key=lambda r: _float_or_nan(r.get("selection_score")), reverse=True)
    fields = [
        "name",
        "preset",
        "seed",
        "selection_score",
        "R2_Y0_from_Z",
        "R2_phase_from_Z",
        "effective_rank",
        "rho_G_cond_cmd_spearman",
        "OOD_drop_R2_Y0",
        "R2_Y0res_retention",
        "R2_cmd_from_Zres",
        "R2_cmd_mlp_from_Zres",
        "R2_mode_mlp_from_Zres",
        "R2_phase_from_Zres",
        "rho_G_Zres_cond_cmd_spearman",
        "probe_json",
    ]
    with open(ranked_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k) for k in fields})
    print(f"[style-run] wrote {ranked_path}")


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
    ap.add_argument("--matrix", choices=["none", *sorted(MATRIX_SPECS)], default="none")
    ap.add_argument("--only_names", nargs="*", default=None)
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
    ap.add_argument("--save_compression", choices=["compressed", "stored"], default="compressed")
    ap.add_argument("--feature_seed", type=int, default=0)
    ap.add_argument("--max_input_shards", type=int, default=None)
    ap.add_argument("--max_parents", type=int, default=None)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d_back", type=int, default=None)
    ap.add_argument("--d_gait", type=int, default=None)
    ap.add_argument("--d_aux", type=int, default=None)
    ap.add_argument("--encoder_kind", choices=["tcn", "stats_tcn", "stats_only", "dual_latent", "stats_tcn_dual"], default=None)
    ap.add_argument("--phi_fourier_k", type=int, default=None)
    ap.add_argument("--yphi_hidden_dim", type=int, default=None)
    ap.add_argument("--yphi_encoder_grad", choices=["full", "stopgrad", "off"], default=None)
    ap.add_argument("--phase_adv_mode", choices=["grl", "alternating"], default=None)
    ap.add_argument("--phase_adv_steps", type=int, default=None)
    ap.add_argument("--phase_adv_hidden", type=int, default=None)
    ap.add_argument("--cmd_mode_adv_mode", choices=["grl", "alternating"], default=None)
    ap.add_argument("--cmd_mode_adv_steps", type=int, default=None)
    ap.add_argument("--cmd_mode_adv_hidden", type=int, default=None)
    ap.add_argument("--metric_y0_groups", choices=["full", "no_action", "kinematic_only"], default=None)
    ap.add_argument("--sample_strategy", choices=["uniform", "bucket_balanced", "source_bucket_balanced"], default=None)
    ap.add_argument("--sample_weight_cap", type=float, default=None)
    ap.add_argument("--dataset_device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--cond_cmd_delta", type=float, default=None)
    ap.add_argument("--rnc_max_pos", type=int, default=None)
    ap.add_argument("--max_grad_norm", type=float, default=None)
    for key in [
        "l_y0",
        "l_yphi",
        "l_inv",
        "l_rnc_g",
        "l_rnc_res",
        "l_var",
        "l_cov",
        "l_phase_adv",
        "l_phase_corr",
        "l_cmd_adv",
        "l_mode_adv",
    ]:
        ap.add_argument(f"--{key}", type=float, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_train_shards", type=int, default=None)

    ap.add_argument("--probe_batch_size", type=int, default=512)
    ap.add_argument("--probe_num_workers", type=int, default=0)
    ap.add_argument("--probe_split", choices=["random", "source_heldout", "all_source_heldout"], default="random")
    ap.add_argument("--heldout_source_group", default=None)
    ap.add_argument("--max_probe_samples", type=int, default=None)
    ap.add_argument("--max_probe_shards", type=int, default=None)
    ap.add_argument("--max_frame_samples", type=int, default=None)
    ap.add_argument("--mlp_probe_max_samples", type=int, default=None)
    ap.add_argument("--mlp_probe_epochs", type=int, default=None)
    ap.add_argument("--mlp_probe_lr", type=float, default=None)
    ap.add_argument("--mlp_probe_batch_size", type=int, default=None)
    ap.add_argument("--probe_backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--phase_shift_max_pairs", type=int, default=None)
    ap.add_argument("--skip_mlp_probe", action="store_true", default=False)
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
            "--save_compression",
            args.save_compression,
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
        "matrix": args.matrix,
        "jobs": [asdict(j) for j in jobs],
    }
    with open(os.path.join(args.work_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not args.dry_run:
        write_probe_summary(args.work_dir, jobs)
    print(f"[style-run] done; summary={os.path.join(args.work_dir, 'summary.json')}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"[style-run] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
