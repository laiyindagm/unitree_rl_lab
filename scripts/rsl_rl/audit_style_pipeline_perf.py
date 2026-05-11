# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Audit frnc_style_v4 pipeline performance from existing logs.

The script is intentionally read-only with respect to experiment artifacts.  It
summarizes feature stats, per-job train/probe completion, and current GPU state
into compact JSON/Markdown files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _nvidia_smi() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "util_gpu_pct": parts[2],
                "memory_used_mib": parts[3],
                "memory_total_mib": parts[4],
            }
        )
    return rows


def _job_rows(work_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for child in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        train_log = child / "train.log"
        probe_json = child / "probe.json"
        train_last = _last_jsonl(train_log)
        probe = _load_json(probe_json)
        row: dict[str, Any] = {
            "name": child.name,
            "has_encoder": (child / "encoder.pt").exists(),
            "has_train_log": train_log.exists(),
            "has_probe_json": probe_json.exists(),
            "train_elapsed_s": train_last.get("elapsed_s") if train_last else None,
            "train_epoch": train_last.get("epoch") if train_last else None,
            "probe_backend": probe.get("probe_backend") if probe else None,
            "probe_device": probe.get("probe_device") if probe else None,
            "probe_r2_y0": probe.get("R2_Y0_from_Z") if probe else None,
            "probe_r2_y0_zres": probe.get("R2_Y0_from_Zres") if probe else None,
            "probe_cmd_mlp_zres": probe.get("R2_cmd_mlp_from_Zres") if probe else None,
            "probe_mode_mlp_zres": probe.get("R2_mode_mlp_from_Zres") if probe else None,
        }
        if train_log.exists() and probe_json.exists():
            row["probe_completed_after_train_s"] = max(0.0, probe_json.stat().st_mtime - train_log.stat().st_mtime)
        else:
            row["probe_completed_after_train_s"] = None
        jobs.append(row)
    return jobs


def _summarize(feature_dir: Path | None, work_dir: Path) -> dict[str, Any]:
    feature_stats = _load_json(feature_dir / "feature_stats.json") if feature_dir else None
    jobs = _job_rows(work_dir)
    completed_probe = [j for j in jobs if j["has_probe_json"]]
    completed_train = [j for j in jobs if j["has_train_log"]]
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "feature_dir": str(feature_dir) if feature_dir else None,
        "work_dir": str(work_dir),
        "feature": {
            "n_samples": feature_stats.get("n_samples") if feature_stats else None,
            "n_parents": feature_stats.get("n_parents") if feature_stats else None,
            "n_input_paths": len(feature_stats.get("input_paths", [])) if feature_stats else None,
            "parent_len": feature_stats.get("parent_len") if feature_stats else None,
            "window_len": feature_stats.get("window_len") if feature_stats else None,
        },
        "jobs_total": len(jobs),
        "jobs_train_done": len(completed_train),
        "jobs_probe_done": len(completed_probe),
        "train_elapsed_s_mean": _mean([j.get("train_elapsed_s") for j in completed_train]),
        "probe_after_train_s_mean": _mean([j.get("probe_completed_after_train_s") for j in completed_probe]),
        "jobs": jobs,
        "probe_summary_rows": _read_csv(work_dir / "probe_summary.csv"),
        "ranked_rows": _read_csv(work_dir / "ranked_candidates.csv"),
        "gpu": _nvidia_smi(),
    }
    return summary


def _mean(vals: list[Any]) -> float | None:
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else None


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Style Pipeline Performance Audit",
        "",
        f"- Created: {summary['created_at']}",
        f"- Feature dir: `{summary['feature_dir']}`",
        f"- Work dir: `{summary['work_dir']}`",
        f"- Feature samples: {summary['feature']['n_samples']}",
        f"- Jobs: {summary['jobs_probe_done']} probe done / {summary['jobs_train_done']} train done / {summary['jobs_total']} total",
        f"- Mean train elapsed: {summary['train_elapsed_s_mean']}",
        f"- Mean probe-after-train elapsed: {summary['probe_after_train_s_mean']}",
        "",
        "## Jobs",
        "",
        "| name | train_s | probe_done | backend | device | R2_Y0 | R2_Y0_Zres | cmd_mlp_Zres | mode_mlp_Zres |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["jobs"]:
        lines.append(
            "| {name} | {train_elapsed_s} | {has_probe_json} | {probe_backend} | {probe_device} | "
            "{probe_r2_y0} | {probe_r2_y0_zres} | {probe_cmd_mlp_zres} | {probe_mode_mlp_zres} |".format(**row)
        )
    if summary["gpu"]:
        lines += ["", "## GPU Snapshot", ""]
        for gpu in summary["gpu"]:
            lines.append(
                f"- GPU {gpu['index']}: util={gpu['util_gpu_pct']}%, "
                f"mem={gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Audit frnc_style_v4 pipeline performance logs.")
    ap.add_argument("--feature_dir", default=None)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    feature_dir = Path(args.feature_dir) if args.feature_dir else None
    work_dir = Path(args.work_dir)
    summary = _summarize(feature_dir, work_dir)
    out_dir = Path(args.out_dir) if args.out_dir else work_dir
    os.makedirs(out_dir, exist_ok=True)
    json_path = out_dir / "perf_audit.json"
    md_path = out_dir / "perf_audit.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"[style-audit] wrote {json_path}")
    print(f"[style-audit] wrote {md_path}")


if __name__ == "__main__":
    main()
