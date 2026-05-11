#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data1/huangyifan/unitree_style_v4/unitree_rl_lab}"
PY="${PY:-/data1/huangyifan/miniconda3/envs/env_isaaclab/bin/python}"
FEATURE_DIR="${FEATURE_DIR:-logs/frnc_style_v4/features_v1}"
RUN_ROOT="${RUN_ROOT:-logs/frnc_style_v4_v42_plan_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-1024}"
MAX_FRAME_SAMPLES="${MAX_FRAME_SAMPLES:-50000}"
MLP_PROBE_MAX_SAMPLES="${MLP_PROBE_MAX_SAMPLES:-20000}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_LOSS="${RUN_LOSS:-1}"
RUN_TARGET="${RUN_TARGET:-1}"
RUN_ARCH="${RUN_ARCH:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export HF_HOME="${HF_HOME:-/data1/huangyifan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data1/huangyifan/hf_cache/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/huangyifan/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data1/huangyifan/pip_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data1/huangyifan/conda_pkgs}"

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

echo "[v42-plan] repo=$REPO_DIR"
echo "[v42-plan] run_root=$RUN_ROOT"
echo "[v42-plan] python=$PY"
echo "[v42-plan] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[v42-plan] started=$(date -Iseconds)"

probe_existing() {
    local name="$1"
    local encoder="$2"
    local out_json="$RUN_ROOT/stage0_${name}_all_source_mlp.json"
    echo "[v42-plan] stage0 probe $name -> $out_json"
    "$PY" scripts/rsl_rl/style_encoder_probe_v4.py \
        --encoder "$encoder" \
        --data_dir "$FEATURE_DIR" \
        --out_json "$out_json" \
        --device cuda:0 \
        --batch_size "$PROBE_BATCH_SIZE" \
        --num_workers 0 \
        --split all_source_heldout \
        --max_frame_samples "$MAX_FRAME_SAMPLES" \
        --mlp_probe_max_samples "$MLP_PROBE_MAX_SAMPLES"
}

run_matrix() {
    local matrix="$1"
    local out_dir="$RUN_ROOT/$2"
    echo "[v42-plan] matrix $matrix -> $out_dir"
    "$PY" scripts/rsl_rl/run_style_v4_experiments.py \
        --skip_features \
        --feature_dir "$FEATURE_DIR" \
        --work_dir "$out_dir" \
        --matrix "$matrix" \
        --mode sequential \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY" \
        --devices cuda:0 \
        --python "$PY" \
        --probe_batch_size "$PROBE_BATCH_SIZE" \
        --probe_num_workers 0 \
        --probe_split all_source_heldout \
        --max_frame_samples "$MAX_FRAME_SAMPLES" \
        --skip_mlp_probe
}

if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
    probe_existing \
        "B6_phase_adv010_50ep" \
        "logs/frnc_style_v4_v41_confirm50_B6_phase_adv010_20260510_163530/E4_full/encoder.pt"
    probe_existing \
        "B4_inv050_rank_50ep" \
        "logs/frnc_style_v4_v41_confirm50_B4_inv050_rank_20260510_163639/E4_full/encoder.pt"
fi

if [[ "$RUN_LOSS" == "1" ]]; then
    run_matrix "v4_2_loss" "loss20"
fi
if [[ "$RUN_TARGET" == "1" ]]; then
    run_matrix "v4_2_target" "target20"
fi
if [[ "$RUN_ARCH" == "1" ]]; then
    run_matrix "v4_2_arch" "arch20"
fi

echo "[v42-plan] finished=$(date -Iseconds)"
