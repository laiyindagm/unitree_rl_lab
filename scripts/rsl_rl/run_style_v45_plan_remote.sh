#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data1/huangyifan/unitree_style_v4/unitree_rl_lab}"
PY="${PY:-/data1/huangyifan/miniconda3/envs/env_isaaclab/bin/python}"
FEATURE_DIR_V2="${FEATURE_DIR_V2:-logs/frnc_style_v4/features_v2_phase_aug}"
FEATURE_DIR_V3="${FEATURE_DIR_V3:-logs/frnc_style_v4/features_v3_phase_dense}"
RUN_ROOT="${RUN_ROOT:-logs/frnc_style_v45_plan_$(date +%Y%m%d_%H%M%S)}"

RAW_DATA_DIRS="${RAW_DATA_DIRS:-logs/style_pretrain_data/v21g_final logs/style_pretrain_data/v21g_ckpts}"
RUN_FEATURES_V3="${RUN_FEATURES_V3:-1}"
FORCE_FEATURES_V3="${FORCE_FEATURES_V3:-0}"
RUN_SCREEN="${RUN_SCREEN:-1}"
RUN_CONFIRM="${RUN_CONFIRM:-1}"
TOP_K="${TOP_K:-4}"

SCREEN_EPOCHS="${SCREEN_EPOCHS:-20}"
CONFIRM_EPOCHS="${CONFIRM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-1024}"
MAX_FRAME_SAMPLES="${MAX_FRAME_SAMPLES:-50000}"
MLP_PROBE_MAX_SAMPLES="${MLP_PROBE_MAX_SAMPLES:-20000}"
PHASE_SHIFT_MAX_PAIRS="${PHASE_SHIFT_MAX_PAIRS:-20000}"
SEEDS="${SEEDS:-0 1 2}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2}"
MAX_JOBS="${MAX_JOBS:-3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export HF_HOME="${HF_HOME:-/data1/huangyifan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data1/huangyifan/hf_cache/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/huangyifan/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data1/huangyifan/pip_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data1/huangyifan/conda_pkgs}"

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

echo "[v45-plan] repo=$REPO_DIR"
echo "[v45-plan] run_root=$RUN_ROOT"
echo "[v45-plan] python=$PY"
echo "[v45-plan] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[v45-plan] devices=$DEVICES max_jobs=$MAX_JOBS seeds=$SEEDS"
echo "[v45-plan] started=$(date -Iseconds)"

run_matrix() {
    local matrix="$1"
    local out_dir="$2"
    local feature_dir="$3"
    local epochs="$4"
    local seeds="$5"
    local skip_mlp="$6"
    shift 6
    local names=("$@")
    echo "[v45-plan] matrix=$matrix feature_dir=$feature_dir out=$out_dir epochs=$epochs seeds=$seeds names=${names[*]:-ALL}"
    local cmd=(
        "$PY" scripts/rsl_rl/run_style_v4_experiments.py
        --skip_features
        --feature_dir "$feature_dir"
        --work_dir "$out_dir"
        --matrix "$matrix"
        --mode parallel
        --max_jobs "$MAX_JOBS"
        --epochs "$epochs"
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
        --save_every "$SAVE_EVERY"
        --devices $DEVICES
        --seeds $seeds
        --python "$PY"
        --probe_batch_size "$PROBE_BATCH_SIZE"
        --probe_num_workers 0
        --probe_split all_source_heldout
        --max_frame_samples "$MAX_FRAME_SAMPLES"
        --mlp_probe_max_samples "$MLP_PROBE_MAX_SAMPLES"
        --phase_shift_max_pairs "$PHASE_SHIFT_MAX_PAIRS"
    )
    if [[ "${#names[@]}" -gt 0 ]]; then
        cmd+=(--only_names "${names[@]}")
    fi
    if [[ "$skip_mlp" == "1" ]]; then
        cmd+=(--skip_mlp_probe)
    fi
    "${cmd[@]}"
}

if [[ "$RUN_FEATURES_V3" == "1" ]]; then
    if [[ "$FORCE_FEATURES_V3" == "1" || ! -f "$FEATURE_DIR_V3/feature_stats.json" ]]; then
        echo "[v45-plan] building dense phase features -> $FEATURE_DIR_V3"
        "$PY" scripts/rsl_rl/style_gait_features.py \
            --data_dirs $RAW_DATA_DIRS \
            --out_dir "$FEATURE_DIR_V3" \
            --parent_len 64 \
            --window_len 32 \
            --parent_stride 16 \
            --samples_per_parent 8 \
            --out_shard_size 4096 \
            --seed 0
    else
        echo "[v45-plan] reusing existing dense phase features: $FEATURE_DIR_V3"
    fi
fi

if [[ "$RUN_SCREEN" == "1" ]]; then
    run_matrix "v4_5_screen" "$RUN_ROOT/screen20_v2" "$FEATURE_DIR_V2" "$SCREEN_EPOCHS" "0" "0" \
        S0_c2_v2_uniform S1_c2_v2_balanced
    run_matrix "v4_5_screen" "$RUN_ROOT/screen20_v3" "$FEATURE_DIR_V3" "$SCREEN_EPOCHS" "0" "0" \
        S2_c2_v3_dense_balanced \
        N1_cmd005_mode005 N2_cmd010_mode005 N3_cmd005_mode010 N4_cmdmode_alt3 \
        G1_phase_adv015 G2_phase_alt3 G3_rnc_res100_delta020 G4_adv015_rnc100 G5_yphi025 \
        M1_back256 M2_aux16 M3_phiK3_dec256_aux16
fi

CONFIRM_NAMES_FILE="$RUN_ROOT/confirm_names.txt"
if [[ "$RUN_CONFIRM" == "1" ]]; then
    "$PY" - "$TOP_K" "$RUN_ROOT/screen20_v2/ranked_candidates.csv" "$RUN_ROOT/screen20_v3/ranked_candidates.csv" > "$CONFIRM_NAMES_FILE" <<'PY'
import csv
import pathlib
import sys

top_k = int(sys.argv[1])
rows = []
for path_s in sys.argv[2:]:
    path = pathlib.Path(path_s)
    if not path.exists():
        continue
    for row in csv.DictReader(path.open()):
        rows.append(row)

def f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except Exception:
        return default

rows.sort(key=lambda r: f(r, "selection_score", -1e9), reverse=True)
names = []
for row in rows:
    name = row["name"]
    if name not in names:
        names.append(name)
    if len(names) >= top_k:
        break

passes_info_floor = any(
    f(row, "R2_Y0_from_Z", -1.0) >= 0.69 and f(row, "effective_rank", -1.0) >= 12.0
    for row in rows[:top_k]
)
if not passes_info_floor and "S2_c2_v3_dense_balanced" not in names:
    names.append("S2_c2_v3_dense_balanced")

for name in names:
    print(name)
PY
    echo "[v45-plan] confirm candidates:"
    cat "$CONFIRM_NAMES_FILE"
    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        feature_dir="$FEATURE_DIR_V3"
        if [[ "$name" == S0_* || "$name" == S1_* ]]; then
            feature_dir="$FEATURE_DIR_V2"
        fi
        run_matrix "v4_5_screen" "$RUN_ROOT/confirm50/$name" "$feature_dir" "$CONFIRM_EPOCHS" "$SEEDS" "0" "$name"
    done < "$CONFIRM_NAMES_FILE"
fi

echo "[v45-plan] finished=$(date -Iseconds)"
