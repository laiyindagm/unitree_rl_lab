#!/usr/bin/env bash
# V2 ablation suite v2 (revised after input-side diagnostic):
# 5 configs targeting the actual gradient path that matters for downstream CIC.
# Goal: produce a z that satisfies (i) low R^2(z->cmd), (ii) high R^2(z->{duty,
# bilat_cos, step_amp, lat_sway}), (iii) bucket_AUC>0.85 for downstream
# intrinsic-reward use.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-$REPO/logs/pretrain_data/v21e_strat}"
OUT_ROOT="${OUT_ROOT:-$REPO/logs/frnc_seg_v2}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"
SEG_MIN="${SEG_MIN:-32}"
SEG_MAX="${SEG_MAX:-64}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS=1
  OUT_ROOT="$REPO/logs/frnc_seg_v2_smoke"
  echo "[smoke] EPOCHS=1 OUT_ROOT=$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

PY="python"
PRE="$REPO/scripts/rsl_rl/frnc_segment_pretrain_v2.py"
PROBE="$REPO/scripts/rsl_rl/frnc_segment_probe_v2.py"
DIAG="$REPO/scripts/rsl_rl/frnc_input_diagnostic.py"

# ----- A. input-side diagnostic (cached if present) ----- #
DIAG_OUT="$OUT_ROOT/_input_diagnostic.json"
if [[ ! -f "$DIAG_OUT" ]] || [[ "${FORCE:-0}" == "1" ]]; then
  echo "[diag] running input-side recoverability diagnostic"
  $PY "$DIAG" --data_dir "$DATA_DIR" --out_json "$DIAG_OUT" \
    --segment_min_len "$SEG_MIN" --segment_max_len "$SEG_MAX" \
    2>&1 | tee "$OUT_ROOT/_input_diagnostic.log"
else
  echo "[diag] skip (cached)"
fi

# ----- B. ablation grid (5 configs) ----- #
COMMON="--l_seg_rnc 1.0 --l_recon_phase 1.0 --l_recon_foot 0.5 --l_var 1.0 --l_cov 0.1 --l_slow 0.0 --l_vel 0.5 --hard_seg_mean 1"
declare -A RUNS=(
  # baseline = V1 plumbing through V2 (expected to cmd-shortcut)
  [v2_baseline]="--mask_kind cmd --rnc_label_kind cmd --l_gait_prop 0.0 --l_adv_cmd 0.0 $COMMON"
  # +prop_head only: inject gait-prop signal but no defense against cmd shortcut
  [v2_prop]="--mask_kind cmd --rnc_label_kind cmd --l_gait_prop 1.0 --l_adv_cmd 0.0 $COMMON"
  # +adv only: kill the cmd shortcut, no positive gait signal
  [v2_adv]="--mask_kind cmd --rnc_label_kind cmd --l_gait_prop 0.0 --l_adv_cmd 1.0 $COMMON"
  # +both: main candidate ckpt for downstream CIC
  [v2_prop_adv]="--mask_kind cmd --rnc_label_kind cmd --l_gait_prop 1.0 --l_adv_cmd 1.0 $COMMON"
  # full stack: strict mask + both labels + prop + adv (test marginal of mask)
  [v2_full]="--mask_kind strict --rnc_label_kind both --l_gait_prop 1.0 --l_adv_cmd 1.0 --l_seg_rnc 1.0 --l_recon_phase 1.0 --l_recon_foot 0.5 --l_var 1.0 --l_cov 0.1 --l_slow 0.3 --l_vel 0.5 --hard_seg_mean 0"
)
ORDER=(v2_baseline v2_prop v2_adv v2_prop_adv v2_full)

run_one () {
  local name="$1"; shift
  local extra="$*"
  local out="$OUT_ROOT/$name"
  if [[ -f "$out/probe.json" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[skip] $name"; return
  fi
  mkdir -p "$out"
  echo "[train] $name -> $out"
  $PY "$PRE" \
    --data_dir "$DATA_DIR" --out_dir "$out" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --device "$DEVICE" \
    --segment_min_len "$SEG_MIN" --segment_max_len "$SEG_MAX" \
    $extra 2>&1 | tee "$out/train.stdout.log"
  echo "[probe] $name"
  $PY "$PROBE" \
    --encoder "$out/encoder.pt" --data_dir "$DATA_DIR" \
    --device "$DEVICE" \
    --segment_min_len "$SEG_MIN" --segment_max_len "$SEG_MAX" \
    --out_json "$out/probe.json" 2>&1 | tee "$out/probe.stdout.log"
}

for name in "${ORDER[@]}"; do run_one "$name" ${RUNS[$name]}; done

# ----- C. aggregate ----- #
echo
echo "================  V2 AGGREGATE  ================"
OUT_ROOT="$OUT_ROOT" $PY - <<'PYAGG'
import json, os
root = os.environ["OUT_ROOT"]
keys = [
    "bucket_AUC_pairwise_mean",
    "intra_bucket_var_ratio",
    "intra_seg_z_var_ratio",
    "R2_zg_seg_vx", "R2_zg_seg_vy", "R2_zg_seg_wz",
    "R2_zg_step_freq",
    "R2_zg_duty_l", "R2_zg_duty_r",
    "R2_zg_bilat_cos", "R2_zg_bilat_sin",
    "R2_zg_step_amp_lk", "R2_zg_step_amp_rk",
    "R2_zg_lat_sway",
    "cond_foot1_AUC", "uncond_foot1_AUC",
]
runs = sorted(d for d in os.listdir(root)
              if os.path.isfile(os.path.join(root, d, "probe.json")))
hdr = ["metric"] + runs
data = {r: json.load(open(os.path.join(root, r, "probe.json"))) for r in runs}
rows = [hdr]
for k in keys:
    row = [k]
    for r in runs:
        v = data[r].get(k, float("nan"))
        try: row.append(f"{float(v):.3f}")
        except: row.append("nan")
    rows.append(row)
widths = [max(len(c) for c in col) for col in zip(*rows)]
for row in rows:
    print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

# composite score reflecting V2 success criteria for CIC handoff
print()
print("============ V2 SUCCESS SCORE (for CIC handoff) ============")
gait_keys = ["R2_zg_duty_l","R2_zg_duty_r","R2_zg_bilat_cos","R2_zg_step_amp_lk","R2_zg_step_amp_rk","R2_zg_lat_sway"]
cmd_keys  = ["R2_zg_seg_vx","R2_zg_seg_vy","R2_zg_seg_wz"]
print(f"{'run':<18}  {'mean R2(z->gait)':>18}  {'mean R2(z->cmd)':>18}  {'bucket_AUC':>12}  {'intra_seg_var':>14}  {'pass?':>6}")
for r in runs:
    d = data[r]
    g = sum(d.get(k, 0.0) for k in gait_keys) / len(gait_keys)
    c = sum(d.get(k, 0.0) for k in cmd_keys) / len(cmd_keys)
    auc = d.get("bucket_AUC_pairwise_mean", 0.0)
    isv = d.get("intra_seg_z_var_ratio", 1.0)
    ok = (g > 0.7) and (c < 0.4) and (auc > 0.85)
    print(f"{r:<18}  {g:>18.3f}  {c:>18.3f}  {auc:>12.3f}  {isv:>14.3f}  {'YES' if ok else 'no':>6}")
PYAGG
