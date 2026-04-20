# CLP (Contrastive Latent Policy) 分阶段实验指导

## 概览

CLP 算法基于 V19d 环境，通过乘积球面对比学习提取速度指令相关的结构化表征，
结合 FiLM 条件序列预测和缓存式两阶段 PPO 训练。

**Gym IDs:**
- `Unitree-G1-15dof-Velocity-Rot-V19d-CLP` — TCN 编码器（默认，~77K 参数）
- `Unitree-G1-15dof-Velocity-Rot-V19d-CLP-Transformer` — Transformer 编码器（~260K 参数）

**Baseline:**
- `Unitree-G1-15dof-Velocity-Rot-V19d` — 标准 MLP PPO（BasePPORunnerV3Cfg）

---

## E-1: 代码正确性验证（离线，无需模拟器）

> **目的**: 在无 Isaac Sim 环境下，验证所有组件的数值正确性。

```bash
cd /root/workspace/unitree_rl_lab
/usr/local/miniconda3/envs/env_isaaclab/bin/python scripts/test_clp_smoke.py
```

该脚本验证：
1. **模型构建** — TCN 和 Transformer 编码器正确初始化，latent dim = 224
2. **前向传播** — `get_latent()` 输出 shape 正确，值有限
3. **乘积球面** — 各球面输出在单位球上（L2 范数 ≈ 1.0）
4. **对比投影** — 投影头输出在单位球上
5. **FiLM 生成器** — 输出 shape = [B, K×num_actions]
6. **缓存机制** — `get_cached_repr()` 一致性
7. **evaluate_from_latent** — 与 `forward()` 路径产生相同分布输出
8. **分解 InfoNCE** — 梯度存在且不为 NaN
9. **序列预测损失** — mask 为全 False 时返回 0
10. **obs 切分** — cmd、history 维度拆分正确

**通过标准**: 全部 PASS，零 FAIL。

---

## E0: 表征学习冒烟测试（~2000 iter, ~2h）

> **目的**: 验证 TCN + 乘积球面 + InfoNCE 在真实训练中独立收敛。

### 启动命令

```bash
cd /root/workspace/unitree_rl_lab

# 关闭序列预测（gen_coef=0），仅训 InfoNCE
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py  --task Unitree-G1-15dof-Velocity-Rot-V19d-CLP --headless --num_envs 2500 --max_iterations 2000 --run_name E0_nce_only
```

> **注意**: 需要手动将 `gen_coef` 临时设为 0，方法有两种：
> 1. 在 `rsl_rl_ppo_cfg.py` 中将 `G115DofContrastiveTCNPPORunnerCfg.algorithm` 
>    的 `gen_coef=0.0` 和 `gen_coef_end=0.0`
> 2. 或在训练后通过 hydra override（如果 runner 支持 CLI override）

### 监控指标

打开 TensorBoard：
```bash
tensorboard --logdir logs/rsl_rl/ --bind_all
```

| 指标 | 期望趋势 | 异常警报 |
|------|----------|---------|
| `nce_loss` | 从 ~log(B) ≈ 8-10 下降至 < 2.0 | 不下降 → 标签量化问题 |
| `uniformity_x/y/w` | 稳定在 < -1.0 | 趋近 0 → mode collapse |
| `alignment_x/y/w` | 逐渐下降 | 不变 → 编码器无区分能力 |
| `tau` | 自动调节（通常 0.1-1.0） | 爆炸到 >5 → clamp 检查 |
| `reward/mean` | 正常上升（与 V19d baseline 可比） | 崩溃 → detach 路径有问题 |
| `ep_len` | 逐渐增长 | 快速归零 → NaN 或 obs 拆分错误 |

### 通过标准

- [ ] `nce_loss` < 2.0
- [ ] 各 `uniformity` < -1.0
- [ ] `reward` 曲线与 V19d baseline 同期可比（不必优于，但不应崩溃）
- [ ] 无 NaN/Inf 异常

### 故障排除

| 症状 | 可能原因 | 修复 |
|------|---------|------|
| nce_loss 不下降 | 所有样本同一标签（量化等级太少） | 检查 vx/vy/wz_levels 覆盖范围 |
| uniformity → 0 | mode collapse（所有表征聚到一点） | 降低 tau_init，增大 nce_coef |
| reward 崩溃 | latent detach 失效 | 检查 `get_latent()` 是否 `.detach()` |
| NaN | 梯度爆炸 | 检查 `max_grad_norm`，降低 `repr_lr` |

---

## E1: 完整表征测试（+序列预测）（~5000 iter, ~5h）

> **目的**: 验证 FiLM 条件生成器 + 序列预测 + 衰减 schedule 正常工作。
> **前提**: E0 通过。

### 启动命令

```bash
# 完整配置（默认 gen_coef=0.5→0.1, nce_coef=0.1）
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V19d-CLP \
    --headless --num_envs 4096 \
    --max_iterations 5000 \
    --run_name E1_full_repr
```

### 监控指标

| 指标 | 期望趋势 |
|------|---------|
| `gen_loss` | 快速下降至 < 0.1 |
| `gen_coef` | 从 0.5 线性衰减至 0.1（10000 iter 尺度） |
| `repr_loss` | 整体下降趋势 |
| `nce_loss` | 继续保持 < 2.0 |
| `reward/mean` | 不低于 E0 |

### 通过标准

- [ ] `gen_loss` < 0.1（@5000 iter）
- [ ] `gen_coef` 按 schedule 衰减
- [ ] `reward` ≥ E0 水平
- [ ] 对比指标（uniformity/alignment）未因 gen_loss 引入干扰而退化

---

## E2: 全流程对比训练（~20000 iter, ~20h）

> **目的**: CLP (TCN) vs CLP (Transformer) vs V19d Baseline 完整对比。
> **前提**: E1 通过。

### 启动命令（三路并行）

```bash
# Terminal 1: Baseline
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V19d \
    --headless --num_envs 4096 \
    --max_iterations 20000 \
    --run_name E2_baseline

# Terminal 2: CLP-TCN
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V19d-CLP \
    --headless --num_envs 4096 \
    --max_iterations 20000 \
    --run_name E2_clp_tcn

# Terminal 3: CLP-Transformer
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V19d-CLP-Transformer \
    --headless --num_envs 4096 \
    --max_iterations 20000 \
    --run_name E2_clp_transformer
```

### 对比指标

| 指标 | 含义 | 期望 |
|------|------|------|
| `reward/mean` | 总奖励 | CLP ≥ 90% baseline |
| `track_ang_vel_z` | wz 跟踪精度 | **CLP > baseline**（核心目标） |
| `track_lin_vel_x/y` | vx/vy 跟踪精度 | CLP ≈ baseline |
| 活跃 bins 数量 | 课程扩展速度 | CLP 扩展不慢于 baseline |
| 低速精度 (vx=0.1) | 小命令区分度 | CLP ≥ baseline |
| GPU 内存 | 资源开销 | TCN < Transformer < 1.5× baseline |
| 每 iter 时间 | 训练效率 | TCN < 1.3× baseline |

### 通过标准

- [ ] CLP 在 wz 精度上显著优于 baseline
- [ ] CLP 在低速区域精度不低于 baseline
- [ ] 总 reward 不低于 baseline 的 90%
- [ ] TCN 训练速度优于 Transformer

---

## E3: 消融实验（可选）

在 E2 结果确认后，选择性进行：

### A3a: 单球面 vs 乘积球面
- 将 `num_spheres=1, sphere_dim=96` 与默认 `num_spheres=3, sphere_dim=32` 对比
- 需修改配置类并注册新 Gym ID

### A3b: 无对比损失
- 设置 `nce_coef=0`，仅保留序列预测 + 缓存两阶段 PPO
- 验证 InfoNCE 的独立贡献

### A3c: 无缓存
- 在 Phase B 中重新调用 `get_latent()`（有梯度），取消两阶段分离
- 验证缓存机制的必要性（预期：importance ratio 偏差导致不稳定）

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `utils/contrastive_latent_model.py` | CLP 模型（编码器 + 球面 + FiLM） |
| `utils/contrastive_ppo.py` | 两阶段 PPO 算法 |
| `tasks/.../agents/rsl_rl_ppo_cfg.py` | Runner 配置（TCN/Transformer） |
| `tasks/.../robots/g1/15dof_rot/__init__.py` | Gym 注册 |
| `tasks/.../robots/g1/15dof_rot/velocity_env_cfg_rot_v19d.py` | V19d 环境（不修改） |
| `scripts/test_clp_smoke.py` | 离线正确性验证脚本 |

---

## 训练参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nce_coef` | 0.1 | InfoNCE 系数 α |
| `gen_coef` | 0.5 → 0.1 | 序列预测系数 β（线性衰减） |
| `gen_decay_iters` | 10000 | β 衰减步数 |
| `tau_init` | 0.5 | 温度初始值（可学习） |
| `repr_lr` | 1e-4 | 表征优化器学习率 |
| `learning_rate` | 1e-3 (adaptive) | PPO 策略学习率 |
| `pred_gamma` | 0.9 | 序列预测时间衰减 |
| `pred_horizon` | 3 | 预测步数 K |
| `num_learning_epochs` | 5 | PPO 更新轮数 |
| `num_mini_batches` | 4 | mini-batch 数量 |
| `num_steps_per_env` | 24 | 每 env rollout 步数 |
