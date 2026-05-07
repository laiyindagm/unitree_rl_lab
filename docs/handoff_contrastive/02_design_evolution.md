# 设计与实验演进（02）

> 目标读者：所有人，尤其是想避免重蹈覆辙、做下一版实验的人。
> 本篇是表征学习线的"完整谱系"+ 每个失败的根因 + 下一版采纳了什么。

---

## 1. 演进总览表

| 版本 | 时间 | 输入 | 编码器架构 | 损失组合 | 关键超参 | 数据来源 | 验证指标 | 结果 | 失败模式 / 教训 | 下版本采纳 |
|---|---|---|---|---|---|---|---|---|---|---|
| **V19d-CLP-TCN** | 2026-04 上 | 在线 obs history (10×54) | TCN dilated [1,2,4]，3 球面 projection | InfoNCE(SupCon)×3 + obs 预测 | τ 学习, gen_coef ramp | RL rollout（在线） | nce_loss / alignment / uniformity | nce_loss 7.84 > random ln(B)=6.24 | 在线 contrastive 与 policy 漂移耦合；3 球面坍缩；τ→0.07 collapse | 改为离线、固定 τ、单球面 |
| **V19d-CLP-Transformer** | 2026-04 上 | 同上 | Transformer d=192, h=4, L=3 | 同上 | 同上 + time-shifted positives | 同上 | 同上 + label distribution probe | 仍 collapse；P0 bug：cmd 用了 normalize 后值做量化 | encoder 输入应是 raw cmd，不是 normalized |
| **frnc_segment v1** | 2026-04 中 | 离线 segment (32 帧) | MLP backbone (d_back=128) | RnC (cmd-distance 标签) | mask_kind=cmd loose | 离线 dump V21e_strat rollout | spearman, R²(z→cmd) | spearman 漂亮 but R²(z→vx)=0.997 | mask 不严格→cmd 泄露→z = cmd 的非线性变换 | mask_kind=strict（屏蔽 vel_cmd 等） |
| **frnc_segment v2** | 2026-04 下 | 同 v1 | MLP + 3 球面 projection | RnC + L_axial + adv_cmd | adv 解 cmd 泄露 | 同 v1 (strict mask) | spearman, axial_R², adv_loss | adv_cmd 实测无效；3 球面冗余 | 对抗解码不能修 cmd 泄露，转向 strict mask + L_axial 直接监督 | drop adv_cmd, drop 3 球面 |
| **frnc_segment v3** ⭐ | 2026-04 末 | segment (32×295) | MLP + axial bases (3×32) + prop_head | RnC + L_axial + L_lip + L_prop | d_gait=32, τ=0.1, σ_cmd 显式存盘 | V21e_strat 离线 dump (5 cfg × 30 ep) | spearman ≥ 0.85, lip_med ≥ 0.5, ibvr ≥ 0.5, R²(z→cmd) ≥ 0.9 | **winner = v3_full**: spearman=0.877, lip_med=0.624, ibvr=0.723, R²(z→cmd)=0.974 | axial_R²<0.15 是设计内行为（axial 只张 cmd 子空间） | 直接作为 V22a 的 frozen encoder |
| **V22a** | 2026-05 上 | 同 V21g obs + rolling buf 进 frozen v3 encoder | actor: TransformerLatentGaitModel (gait_dim=32 拼到 latent) | — (无表征损失) | encoder 完全冻结 | 在线 IsaacLab | per-bucket TB metrics | reward 与 V21g 打平；low-speed/pure_wz 略优；transition 略差 | z 是被旁路的，actor 没有梯度压力去"用" z | 加 r_int 给 actor 压力 → V22b |
| **V22b** | 2026-05 上 | 同 V22a | 同 V22a | r_int = ‖z−g*(v)‖²/d, mask ‖v‖_W<ε; SMERL gate; α ramp | α∈[0,0.02] iter 200→2000; β=0.045; κ=200; d=0.99; ε=0.1 | 同 V22a | + intrinsic/{r_int_mean, gate_mean, alpha} | smoke 5 iter PASSED；20000 iter 待训 | — | 待 20k 训完后定 V22c 路线 |

---

## 2. 逐版本展开

### 2.1 V19d-CLP（在线 contrastive，双架构 A/B）

**目标**：让 RL rollout 在训练 PPO 的同时学一个 encoder，端到端把 obs history 压成 z。

**设计**：encoder 输出经 ProductSphereProjection 拆 3 个 32d 球面（vx / vy / wz），分别做 SupCon；额外加 generator head 预测未来 obs 的子集（pred_obs_dim=36，去 last_action 防 feedback loop）。

**失败时间线**：
1. E1_full_repr 5k iter：nce_loss 7.84 > random baseline ln(B)=6.24，alignment≈0.61 ≈ random sphere，τ collapse 到 clamp 下界 0.07 → "温度套利死亡螺旋"。
2. cfg 修：tau_init=0.1 + learnable_tau=False，nce_coef 0.1→0.05，gen_coef 0.1→0.3。仍 collapse。
3. 加 time-shifted positive pairs（同 env 不同 t）+ label distribution probe → 发现 lbl_top1_x > 0.7 → 标签退化。
4. P0 root cause：`encode()` 先 obs_normalizer 再取 cmd 做 InfoNCE 标签，当 curriculum 卡 cmd_raw∈[-0.14,0.14] 时归一化后落到 [-0.74,-0.32]，几乎所有样本量化到同 1-2 个 bin → SupCon 拉近物理不相关样本。
5. P0 fix：encode() 内先 raw_cmd = self.get_cmd_from_obs(flat_obs)（pre-norm），再 normalize 给 encoder。A fix：3 球面 projection 改 ModuleList of 3 个独立 Linear。
6. 即使修了仍达不到目标 → 决定**整线下马**，转离线。

**留下的关键洞察**：
- 在线 contrastive 与 policy 漂移强耦合，"anchor 在变"。
- 学习温度（learnable τ）几乎必然 collapse，应固定 τ。
- 3 球面方案中，sphere 之间隐式参数共享会导致坍缩，需要显式独立 head。
- cmd 量化必须用 raw cmd，不能用 normalized。

### 2.2 frnc_segment v1（离线首版）

**目标**：把表征学习从 RL loop 中剥离。离线 dump rollout segment，用 RnC（cmd-distance 标签）训 encoder。

**失败**：mask_kind=cmd 是 loose mask（只屏蔽 vel_cmd 三个槽，不屏蔽其他 cmd 派生字段如 lin_speed_token、gait_mode_token）。结果 R²(z→vx)=0.997，z 几乎等于 cmd 的非线性变换。

**教训**：mask 必须 strict——所有"能从中读出 cmd"的字段都要屏蔽：vel_cmd, lin_speed_token, gait_mode_token, last_action（含 cmd 信息因为 policy 看 cmd 输出 action），base_ang_vel（pure_wz 模式下与 cmd 强相关），joint_vel_rel（动作派生）。剩下的 backbone 主要是 base orientation, joint_pos_rel 与 contact，足以学步态。

### 2.3 frnc_segment v2（加 axial + adv_cmd）

**目标**：v1 的 strict mask + L_axial（A1 性质）+ adv_cmd（对抗解 cmd 泄露的备份）。

**结果**：strict mask + L_axial 解决了 cmd 泄露；adv_cmd 实测无效——因为 strict mask 已经把 cmd 信道堵死，adv 没有信号可以对抗。3 球面 projection 也被发现冗余（继承自 V19d-CLP，单 sphere 完全够用）。

**教训**：
- 对抗解码（GAN-like 解 leakage）不如直接堵 leakage 信道（mask）。
- 不要把上一版的过度设计（3 球面）继承下来，每版重新评估。

### 2.4 frnc_segment v3（winner: v3_full）

**目标**：定型损失组合 RnC + L_axial + L_lip + L_prop，sigma_cmd 显式存到 ckpt（V22b 需要它构造 metric）。

**5 cfg × 30 epoch 全跑**（cfg 区分：是否启用 prop_head, hard_seg_mean, mask_kind, L_lip 是否启用）。

**winner = v3_full** 指标：
- spearman(z, cmd) = **0.877** ≥ 0.85 (Iso)
- lip_med = **0.624** ≥ 0.5 (A2)
- ibvr (intra-bucket var ratio) = **0.723** ≥ 0.5 (Var)
- R²(z→cmd) = **0.974**（高，但 strict mask 下 cmd 泄露 ≤ 通过 cmd-相关步态）
- R²(z→duty_l) = 0.95, R²(z→yaw) = 0.92, R²(z→act) = 0.91, R²(z→lat) = 0.88（L_prop head）
- axial_R² < 0.15

**关键认知更新**（已在 01 章 §3.2 解释）：axial_R² 低 ≠ 失败。axial bases 只张 z 中 cmd 决定子空间（≤3 维），剩下 ≥29 维方差是步态自由度，**不**该被 axial 解释。

**对照**：baseline（仅 RnC，无 L_axial）spearman=0.18 → L_axial 是 metric isometry 真正落地的关键。

**ckpt**：`/root/workspace/unitree_rl_lab/logs/frnc_seg_v3/v3_full/encoder.pt`，包含 state_dict + config（含 sigma_cmd）。

### 2.5 V22a（frozen encoder 接入 actor）

**目标**：最小可验证集成。frozen v3_full encoder 接入 RL，z 进 actor latent，**不**发 r_int，验证不退化。

**实现要点**：
- `FrozenSegmentEncoder` 包装 V3 encoder + sigma_cmd buffer。
- 算法 `SegmentEncoderVelocityEstimatorPPO`：维护 per-env rolling buffer (32, 295)，每个 act() 调 encoder 把 z 拼到 obs["z_gait"]。
- Actor `TransformerLatentGaitModel`：override `get_latent_outputs` 把 obs["z_gait"] 拼到 policy_latent。
- **关键工程坑**：RolloutStorage 用 TensorDict，shape 在第一次 act 前由 storage init 锁定。必须在 `construct_algorithm` 中**预先注入**零 z_gait 给 obs，让 storage 把 z_gait 字段建好，否则后续 storage.add 会 shape mismatch。

**结果**（20000 iter 训完）：reward 与 V21g 整体打平；per-bucket: low-speed 和 pure_wz 略优，transition 略差。

**结论**：z 进 actor 但 actor 没有"用 z"的梯度压力——因为 reward 只关 v_actual 与 v_cmd 的差，actor 完全可以把 z 当噪声忽略（latent MLP 把 z 维度的权重学小即可）。需要在 reward 侧加压力 → V22b。

### 2.6 V22b（axial-residual r_int + SMERL gate）

**目标**：给 actor 一个"用 z 才能拿到的 reward"，逼 policy 在 z 空间分化。

**设计**：见 [01 章 §3.6-3.8](01_problem_and_math.md#36-v22b-内在奖励axial-残差)。要点重申：
- r_int = ‖z − g*(v)‖²/d_gait，mask cmd 在原点附近的 env。
- α 200→2000 iter 线性 ramp 到 0.02。
- gate σ(200·(EMA−0.045))。
- z STILL frozen；encoder 不在线更新。
- 3-way mode isolation 完全保留（env = V21g）。

**实现要点**（继承 V22a）：
- 新算法 `SegmentEncoderCICPPO(SegmentEncoderVelocityEstimatorPPO)`。
- `act()` 缓存 `_last_z, _last_cmd`（cmd 取 obs["policy"][:, 42:45] 即 history 最新帧的 vel_cmd）。
- `process_env_step()` 计算 r_int 与 bonus，augment rewards 后调 super。
- per-env EMA + 死区 mask + 详细 logging（每 200 step 打印一行）。

**smoke test 5 iter PASSED**（256 envs，mean reward -1 到 -5，无 NaN/Traceback）。

---

## 3. 已弃用方案与原因（一表）

| 方案 | 哪一版试过 | 弃用原因 |
|---|---|---|
| 在线 contrastive (RL loop 内训 encoder) | V19d-CLP | encoder 跟 policy 漂移；学温度 collapse；3 球面坍缩 |
| Learnable temperature τ | V19d-CLP | 套利到 clamp 下界，alignment loss 数值上"赢"但物理无意义 |
| 3 球面 projection 共享线性层 | V19d-CLP, v2 | 球面间隐式参数共享导致坍缩；改 ModuleList 后不再有此问题，但单 sphere 也够 |
| Loose mask (`mask_kind=cmd`) | v1 | R²(z→vx)=0.997 cmd 泄露 |
| 对抗 cmd 解码 (adv_cmd) | v2 | strict mask 已堵 leakage，adv 无信号可对抗 |
| InfoNCE bucket 作为 r_int (CIC 原版) | V22b 设计期 | 桶定义=cmd → policy 不能跨桶 → 0 梯度（详见 [01 §3.6](01_problem_and_math.md#36-v22b-内在奖励axial-残差)） |
| z 进 critic | V22a/V22b 设计期 | TensorDict shape lock；advantage 偏差未带来收益 |
| 在线 EMA slow encoder（3-phase 计划中的 phase 3） | 暂未试 | 等 V22b 充分验证再考虑 |
| Episodic-return SMERL（vs per-step EMA） | V22b 设计期 | gate 突变；step-wise advantage 不喜欢 |
| `init_at_random_ep_len=True` 下做 minimum-sample curriculum 检查 | 早期 | 每步只 ~4 envs reset，min-sample 永远不满足；改 buffer 累积 |

---

## 4. 3-way mode isolation（用户硬约束）

**约束**：环境必须保留 `{standing, pure_wz, other}` 三类 mode token，不允许"简化为 2-way"或"完全去 token"。

**起源**：V19f 实验显示 standing 与 pure_wz 是两类与"普通走路"完全不同的策略（standing 需要主动抑制 vel_cmd 噪声，pure_wz 需要在原地小幅转向不能漂走）；token 是给 policy 一个"我现在该用哪种策略"的 prior。

**实现位置**：`mdp/observations.py:gait_mode_token` 三维 one-hot；env cfg 在 PolicyCfg 与 CriticCfg 都加。

**对 V22 系的影响**：z_gait 不替代 mode token——token 是策略层 prior，z 是步态层表征，正交关系。V22b 的 r_int mask cmd 在原点附近，正好把 standing mode 排除在 intrinsic 奖励之外（standing 时 ‖v‖_W < ε），不会与 mode token 冲突。

---

## 5. 下一步候选实验（设想）

| 候选 | 假设 | 风险 |
|---|---|---|
| V22c：α_max 提高到 0.05 + warmup 缩到 [200, 1000] | 当前 r_int 量级太弱（5%）不足以驱动分化 | 可能压垮 task reward |
| V22d：cmd_norm_eps 放宽到 0.05 | 现 ε=0.1 排除了一部分 low-speed pure_xy | low-speed 模式下 r_int 不稳 |
| V22e：encoder 在 V21g rollout 上重新预训练 | V3 encoder 训练数据来自 V21e_strat，policy 已演化到 V21g，分布漂移 | 需要重新 dump + pretrain |
| V22f：3-phase 在线 EMA encoder（slow-update） | 完全 frozen 限制 z 适应新 policy | 引回"漂移 anchor"问题 |
| V22g：r_int 改为 negative cosine residual `1 - cos(z, g*)` | L2 残差受 z norm 影响大 | cos 对 norm 不敏感 → 鼓励 norm 爆炸 |

V22c 是最低风险的下一步，建议在 V22b 20k 训完且 r_int 稳定后做。

---

## 6. 阅读历史脉络的最短路径

1. 读 [01 章 §3](01_problem_and_math.md#3-损失项逐个推导) 理解 V3 编码器到底学什么；
2. 读本章 §2.5–2.6 理解 V22a→V22b 的工程交接；
3. 读 [03 章](03_codebase_and_handoff.md) 理解现在代码长什么样、怎么跑；
4. 回头读本章 §2.1–2.4 理解为什么我们走到现在这一步（避免重做失败）。
