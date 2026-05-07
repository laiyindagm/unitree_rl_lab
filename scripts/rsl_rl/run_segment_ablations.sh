#!/usr/bin/env bash
# Sequential launcher for segment-encoder ablation suite + result aggregation.
#
# Trains 5 encoder variants on the existing v21e_strat shards, runs the
# segment probe on each, and prints a summary table. Total wall time is
# dominated by training (5 x ~30 epochs).
#
# Usage:
#   bash scripts/rsl_rl/run_segment_ablations.sh             # full run
#   SMOKE=1 bash scripts/rsl_rl/run_segment_ablations.sh     # 1-epoch dry run
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-$REPO/logs/pretrain_data/v21e_strat}"
OUT_ROOT="${OUT_ROOT:-$REPO/logs/frnc_seg}"
MASK="6:9,64:67,122:125,180:183,238:241"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"
SEG_MIN="${SEG_MIN:-32}"
SEG_MAX="${SEG_MAX:-64}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS=1
  OUT_ROOT="$REPO/logs/frnc_seg_smoke"
  echo "[smoke] EPOCHS=1 OUT_ROOT=$OUT_ROOT"
fi

mkdir -p "$OUT_ROOT"

# ablation table: name, extra-flags-passed-to-pretrain
declare -A RUNS=(
  [seg_full]="--l_seg_rnc 1.0 --l_recon_phase 1.0 --l_recon_foot 0.5 --l_var 1.0 --l_cov 0.1 --l_slow 0.0 --l_vel 0.5 --hard_seg_mean 1"
  [seg_no_rnc]="--l_seg_rnc 0.0 --l_recon_phase 1.0 --l_recon_foot 0.5 --l_var 1.0 --l_cov 0.1 --l_slow 0.0 --l_vel 0.5 --hard_seg_mean 1"
  [seg_no_recon]="--l_seg_rnc 1.0 --l_recon_phase 0.0 --l_recon_foot 0.0 --l_var 1.0 --l_cov 0.1 --l_slow 0.0 --l_vel 0.5 --hard_seg_mean 1"
  [seg_no_div]="--l_seg_rnc 1.0 --l_recon_phase 1.0 --l_recon_foot 0.5 --l_var 0.0 --l_cov 0.0 --l_slow 0.0 --l_vel 0.5 --hard_seg_mean 1"
  [seg_soft]="--l_seg_rnc 1.0 --l_recon_phase 1.0 --l_recon_foot 0.5 --l_var 1.0 --l_cov 0.1 --l_slow 0.1 --l_vel 0.5 --hard_seg_mean 0"
)

# preserve order
ORDER=(seg_full seg_no_rnc seg_no_recon seg_no_div seg_soft)

PY="python"
PRETRAIN="$REPO/scripts/rsl_rl/frnc_segment_pretrain.py"
PROBE="$REPO/scripts/rsl_rl/frnc_segment_probe.py"

run_one () {
  local name="$1"; shift
  local extra="$*"
  local out="$OUT_ROOT/$name"
  if [[ -f "$out/probe.json" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[skip] $name already has probe.json"
    return
  fi
  mkdir -p "$out"
  echo "[train] $name -> $out"
  $PY "$PRETRAIN" \
    --data_dir "$DATA_DIR" --out_dir "$out" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --device "$DEVICE" \
    --segment_min_len "$SEG_MIN" --segment_max_len "$SEG_MAX" \
    --mask_obs_indices "$MASK" \
    $extra \
    2>&1 | tee "$out/train.stdout.log"
  echo "[probe] $name"
  $PY "$PROBE" \
    --encoder "$out/encoder.pt" --data_dir "$DATA_DIR" \
    --device "$DEVICE" --mask_obs_indices "$MASK" \
    --segment_min_len "$SEG_MIN" --segment_max_len "$SEG_MAX" \
    --out_json "$out/probe.json" \
    2>&1 | tee "$out/probe.stdout.log"
}

for name in "${ORDER[@]}"; do
  run_one "$name" ${RUNS[$name]}
done

echo
echo "================  AGGREGATE  ================"
$PY - <<'PYAGG'
import json, os, glob
root = os.environ.get("OUT_ROOT") or "logs/frnc_seg"
if os.environ.get("SMOKE", "0") == "1":
    root = "logs/frnc_seg_smoke"
keys = [
    "bucket_AUC_pairwise_mean",
    "intra_bucket_var_ratio",
    "intra_seg_z_var_ratio",
    "R2_zg_step_freq",
    "R2_zg_seg_vx",
    "R2_zg_seg_vy",
    "R2_zg_seg_wz",
    "cond_foot0_AUC",
    "cond_foot1_AUC",
    "uncond_foot0_AUC",
    "uncond_foot1_AUC",
]
runs = sorted(d for d in os.listdir(root)
              if os.path.isfile(os.path.join(root, d, "probe.json")))
if not runs:
    print("no runs with probe.json under", root); raise SystemExit
hdr = ["run"] + keys
widths = [max(len(h), 12) for h in hdr]
rows = []
for r in runs:
    js = json.load(open(os.path.join(root, r, "probe.json")))
    row = [r]
    for k in keys:
        v = js.get(k, float("nan"))
        try:
            row.append(f"{float(v):.3f}")
        except Exception:
            row.append("nan")
    rows.append(row)
    for i, c in enumerate(row):
        widths[i] = max(widths[i], len(c))
def fmt(row):
    return "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))
print(fmt(hdr))
print("  ".join("-" * w for w in widths))
for r in rows:
    print(fmt(r))
PYAGG
