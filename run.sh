#!/bin/bash
# 1. 采集 V21g final 数据
  python scripts/rsl_rl/collect_style_pretrain_data.py \
    --task Unitree-G1-15dof-Velocity-Rot-V21g \
    --checkpoint logs/rsl_rl/unitree_g1_15dof_velocity_rot_v21g/2026-05-03_10-56-35/model_19000.pt \
    --num_envs 256 \
    --num_steps 4000 \
    --shard_steps 500 \
    --output_dir logs/style_pretrain_data/v21g_final \
    --cmd_sampling stratified

  # 2. 采集 V21g 中间 checkpoint 数据
  python scripts/rsl_rl/collect_style_pretrain_data.py \
    --task Unitree-G1-15dof-Velocity-Rot-V21g \
    --checkpoint logs/rsl_rl/unitree_g1_15dof_velocity_rot_v21g/2026-05-03_10-56-35/model_5000.pt \
    --num_envs 256 \
    --num_steps 2000 \
    --shard_steps 500 \
    --output_dir logs/style_pretrain_data/v21g_ckpts/model_05000 \
    --cmd_sampling stratified

  python scripts/rsl_rl/collect_style_pretrain_data.py \
    --task Unitree-G1-15dof-Velocity-Rot-V21g \
    --checkpoint logs/rsl_rl/unitree_g1_15dof_velocity_rot_v21g/2026-05-03_10-56-35/model_10000.pt \
    --num_envs 256 \
    --num_steps 2000 \
    --shard_steps 500 \
    --output_dir logs/style_pretrain_data/v21g_ckpts/model_10000 \
    --cmd_sampling stratified

  python scripts/rsl_rl/collect_style_pretrain_data.py \
    --task Unitree-G1-15dof-Velocity-Rot-V21g \
    --checkpoint logs/rsl_rl/unitree_g1_15dof_velocity_rot_v21g/2026-05-03_10-56-35/model_15000.pt \
    --num_envs 256 \
    --num_steps 2000 \
    --shard_steps 500 \
    --output_dir logs/style_pretrain_data/v21g_ckpts/model_15000 \
    --cmd_sampling stratified

  # 3. 顺序执行 feature 构建、E1-E6 训练、probe
  python scripts/rsl_rl/run_style_v4_experiments.py \
    --raw_data_dirs logs/style_pretrain_data/v21g_final logs/style_pretrain_data/v21g_ckpts \
    --feature_dir logs/frnc_style_v4/features_v1 \
    --work_dir logs/frnc_style_v4 \
    --presets E1_reg_only E2_reg_inv E3_reg_inv_rnc E4_full E5_mask_m1_full E6_mask_m2_full \
    --mode sequential \
    --epochs 50 \
    --batch_size 256 \
    --device cuda:0