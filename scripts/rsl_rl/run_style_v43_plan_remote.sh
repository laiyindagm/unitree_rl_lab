#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data1/huangyifan/unitree_style_v4/unitree_rl_lab}"
PY="${PY:-/data1/huangyifan/miniconda3/envs/env_isaaclab/bin/python}"
FEATURE_DIR_V1="${FEATURE_DIR_V1:-logs/frnc_style_v4/features_v1}"
FEATURE_DIR_V2="${FEATURE_DIR_V2:-logs/frnc_style_v4/features_v2_phase_aug}"
RUN_ROOT="${RUN_ROOT:-logs/frnc_style_v43_plan_$(date +%Y%m%d_%H%M%S)}"

CONFIRM_EPOCHS="${CONFIRM_EPOCHS:-50}"
GRID_EPOCHS="${GRID_EPOCHS:-20}"
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

RUN_CONFIRM="${RUN_CONFIRM:-1}"
RUN_GRID="${RUN_GRID:-1}"
RUN_FEATURES_V2="${RUN_FEATURES_V2:-0}"
RUN_V2_CONFIRM="${RUN_V2_CONFIRM:-0}"
RAW_DATA_DIRS="${RAW_DATA_DIRS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export HF_HOME="${HF_HOME:-/data1/huangyifan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data1/huangyifan/hf_cache/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/huangyifan/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data1/huangyifan/pip_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data1/huangyifan/conda_pkgs}"

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

echo "[v43-plan] repo=$REPO_DIR"
echo "[v43-plan] run_root=$RUN_ROOT"
echo "[v43-plan] python=$PY"
echo "[v43-plan] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[v43-plan] devices=$DEVICES max_jobs=$MAX_JOBS seeds=$SEEDS"
echo "[v43-plan] started=$(date -Iseconds)"

run_matrix() {
    local matrix="$1"
    local out_dir="$2"
    local feature_dir="$3"
    local epochs="$4"
    local seeds="$5"
    local skip_mlp="$6"
    echo "[v43-plan] matrix=$matrix feature_dir=$feature_dir out=$out_dir epochs=$epochs seeds=$seeds skip_mlp=$skip_mlp"
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
    if [[ "$skip_mlp" == "1" ]]; then
        cmd+=(--skip_mlp_probe)
    fi
    "${cmd[@]}"
}

if [[ "$RUN_FEATURES_V2" == "1" ]]; then
    if [[ -z "$RAW_DATA_DIRS" ]]; then
        echo "[v43-plan] ERROR: RUN_FEATURES_V2=1 requires RAW_DATA_DIRS" >&2
        exit 2
    fi
    echo "[v43-plan] building phase-aug features -> $FEATURE_DIR_V2"
    "$PY" scripts/rsl_rl/style_gait_features.py \
        --data_dirs $RAW_DATA_DIRS \
        --out_dir "$FEATURE_DIR_V2" \
        --parent_len 64 \
        --window_len 32 \
        --parent_stride 16 \
        --samples_per_parent 4 \
        --out_shard_size 2048 \
        --seed 0
fi

if [[ "$RUN_CONFIRM" == "1" ]]; then
    run_matrix "v4_3_confirm" "$RUN_ROOT/confirm50_v1" "$FEATURE_DIR_V1" "$CONFIRM_EPOCHS" "$SEEDS" "0"
fi

if [[ "$RUN_GRID" == "1" ]]; then
    run_matrix "v4_3_grid" "$RUN_ROOT/grid20_v1" "$FEATURE_DIR_V1" "$GRID_EPOCHS" "0" "1"
fi

if [[ "$RUN_V2_CONFIRM" == "1" ]]; then
    if [[ ! -d "$FEATURE_DIR_V2" ]]; then
        echo "[v43-plan] ERROR: FEATURE_DIR_V2 not found: $FEATURE_DIR_V2" >&2
        exit 2
    fi
    run_matrix "v4_3_confirm" "$RUN_ROOT/confirm50_v2_phase_aug" "$FEATURE_DIR_V2" "$CONFIRM_EPOCHS" "$SEEDS" "0"
fi

echo "[v43-plan] finished=$(date -Iseconds)"
