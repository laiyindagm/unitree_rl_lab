#!/usr/bin/env bash
# V3 ablations:
#   v3_baseline    : metric_rnc only (no axial, no lip, no prop)
#   v3_axial       : metric_rnc + axial decomposition (A1)
#   v3_axial_lip   : + magnitude Lipschitz (A2)
#   v3_full        : + prop_head with PROP_WEIGHTS (input-ceiling-aware direct supervision)
#   v3_full_loose  : v3_full but with mask_kind=cmd (less aggressive masking)
set -e
DATA_DIR=${DATA_DIR:-/root/workspace/unitree_rl_lab/logs/pretrain_data/v21e_strat}
OUT_ROOT=${OUT_ROOT:-/root/workspace/unitree_rl_lab/logs/frnc_seg_v3}
EPOCHS=${EPOCHS:-30}
BS=${BS:-128}
DEVICE=${DEVICE:-cuda:0}

cd "$(dirname "$0")"
mkdir -p "${OUT_ROOT}"

run_one() {
    local tag=$1; shift
    local out="${OUT_ROOT}/${tag}"
    mkdir -p "${out}"
    echo "=== [V3] ${tag} -> ${out} ==="
    python frnc_segment_pretrain_v3.py \
        --data_dir "${DATA_DIR}" --out_dir "${out}" \
        --device "${DEVICE}" --epochs "${EPOCHS}" --batch_size "${BS}" \
        --num_workers 2 \
        "$@" 2>&1 | tee "${out}/run.log"
    python frnc_segment_probe_v3.py \
        --encoder "${out}/encoder.pt" --data_dir "${DATA_DIR}" \
        --device "${DEVICE}" --max_shards 4 \
        --out_json "${out}/probe.json" 2>&1 | tee "${out}/probe.log"
}

run_one v3_baseline   --mask_kind strict --l_rnc 1.0 --l_axial 0.0 --l_lip 0.0 --l_prop 0.0
run_one v3_axial      --mask_kind strict --l_rnc 1.0 --l_axial 1.0 --l_lip 0.0 --l_prop 0.0
run_one v3_axial_lip  --mask_kind strict --l_rnc 1.0 --l_axial 1.0 --l_lip 1.0 --l_prop 0.0
run_one v3_full       --mask_kind strict --l_rnc 1.0 --l_axial 1.0 --l_lip 1.0 --l_prop 1.0
run_one v3_full_loose --mask_kind cmd    --l_rnc 1.0 --l_axial 1.0 --l_lip 1.0 --l_prop 1.0

echo "=== [V3] all done ==="
python - <<'PY'
import json, os
root = "/root/workspace/unitree_rl_lab/logs/frnc_seg_v3"
keys = ["bucket_AUC_pairwise_mean", "intra_bucket_var_ratio",
        "intra_seg_z_var_ratio", "cmd_distance_spearman",
        "axial_R2", "magnitude_lipschitz_median",
        "R2_zg_seg_vx", "R2_zg_seg_vy", "R2_zg_seg_wz",
        "R2_zg_duty_l", "R2_zg_lat_sway", "R2_zg_waist_yaw_std", "R2_zg_ang_act"]
print(f"{'tag':<18}" + "".join(f"{k[:14]:>16}" for k in keys))
for tag in sorted(os.listdir(root)):
    p = os.path.join(root, tag, "probe.json")
    if not os.path.isfile(p): continue
    d = json.load(open(p))
    print(f"{tag:<18}" + "".join(f"{d.get(k, float('nan')):>16.3f}" for k in keys))
PY
