# V4.2 Style Encoder 实现、调整与有效性报告

生成时间：2026-05-10

## 1. 目标回顾

当前阶段的直接目标不是训练 locomotion policy 本身，而是先得到一个可用于后续 RL/内在奖励/残差 shaping 的 gait style encoder。根据 `new_talk.md` 的重新规约，理想的 `z` 应满足：

1. 不是 command shortcut、time code、phase code 或离散 style label。
2. 是连续的、相位不变的步态风格坐标。
3. `z -> y0` 能解释相位不变 gait 属性。
4. `(z, phi) -> yphi` 能解释相位相关 gait 属性。
5. `z` 的距离/几何应尽量对应真实 gait style 差异，而不是优先对应 command 差异。

工程上，本轮工作围绕 `frnc_style_v4` 展开：从已有 V21g 风格数据中提取窗口特征，训练 masked history encoder，再用 offline probe 验证 command/mode/phase leakage、style 信息量、rank、OOD source-heldout 泛化和 gait metric geometry。

## 2. 已实现的主要组件

### 2.1 特征与数据管线

已建立 `style_gait_features.py` 生成的 feature 数据格式，核心内容包括：

- 输入窗口：masked history observation。
- `y0`：相位不变/窗口级 gait 属性。
- `yphi`：逐帧、相位依赖 gait 属性。
- `cmd`、`mode_id`、`bucket_id`。
- `source_id`、`parent_id`：用于 source-heldout / all-source-heldout 评估。

有效性评价：

- 有效。它把问题从“直接学 latent”转成了可验证的监督/自监督表征学习问题。
- 但数据来源仍主要是已有 V21g/V21g checkpoint rollout，覆盖的 gait style 可能偏窄；当前结果只能证明 encoder 能解释这批数据中的 gait 变化，不能证明对未来 on-policy distribution 完全可靠。

### 2.2 V4 encoder 训练脚本

核心文件：

- `scripts/rsl_rl/style_encoder_pretrain_v4.py`

已实现能力：

- 基础 TCN encoder。
- `y0` decoder：监督相位不变 gait 属性。
- `yphi` decoder：用 `(z, phi)` 监督相位依赖属性。
- invariance loss：同 parent/window 的不同 crop 表征一致。
- RnC/metric loss：鼓励 latent 几何对应 gait descriptor 差异。
- rank regularization：`var/cov` 防 collapse。
- phase adversary：降低 `z` 中的 phase leakage。
- 新增 V4.2 选项：
  - `encoder_kind={tcn, stats_tcn, stats_only, dual_latent, stats_tcn_dual}`
  - `yphi_encoder_grad={full, stopgrad, off}`
  - `phase_adv_mode={grl, alternating}`
  - `l_phase_corr`
  - `metric_y0_groups={full, no_action, kinematic_only}`
  - `d_aux` dual latent 分支

有效性评价：

- 训练脚本已经能覆盖当前主要假设空间。
- 最有效的新实现是 `stats_tcn`，说明窗口级统计信息是原 TCN 的关键缺口。
- dual latent 单独收益不明显，说明“把 phase-dependent 信息塞进 aux”不是当前主要瓶颈。

### 2.3 Probe 与诊断脚本

核心文件：

- `scripts/rsl_rl/style_encoder_probe_v4.py`

已实现指标：

- `R2_Y0_from_Z`
- `R2_Y0_gain_over_cmd_mode`
- `R2_cmd_from_Z`
- `R2_mode_from_Z`
- `R2_phase_from_Z`
- `R2_phase_mlp_from_Z`
- `effective_rank`
- `shift_ratio`
- `rho_G_spearman`
- `rho_G_cond_cmd_spearman`
- `R2_Yphi_from_Z_phi`
- feature group metrics
- bucket metrics
- `source_heldout` 和 `all_source_heldout`
- OOD drop

有效性评价：

- 非常有效。V4.2 的主要结论不是从 loss 曲线得出的，而是从 probe 中暴露出来的。
- 尤其是 `all_source_heldout` 和 MLP nuisance probe，能区分“线性 phase 低”和“非线性仍可恢复 phase”的情况。
- 当前不足是 probe 仍是 offline 统计指标，还没有证明 downstream actor 会使用 `z`。

### 2.4 实验矩阵与远端执行

核心文件：

- `scripts/rsl_rl/run_style_v4_experiments.py`
- `scripts/rsl_rl/run_style_v42_plan_remote.sh`

已实现矩阵：

- `v4_1_screen`：早期 baseline/rank/phase screen。
- `v4_2_loss`：loss 权重、phase adversary、yphi 梯度、phase corr。
- `v4_2_target`：不同 `metric_y0_groups` 和 yphi 监督设置。
- `v4_2_arch`：TCN、stats、dual latent 结构消融。

远端已在 `/data1/huangyifan/unitree_style_v4/unitree_rl_lab` 完成 V4.2 计划实验：

- 结果目录：`logs/frnc_style_v4_v42_plan_20260510_185932`
- 完成时间：`2026-05-10 21:09:30 +08:00`

有效性评价：

- 编排脚本有效，避免了重复手动命令和错误配置。
- 当前只用单 GPU 顺序跑完，时间成本可接受。
- 后续 50 epoch confirmation 可以并行化到多 GPU。

## 3. 各调整的效果评价

### 3.1 Strict mask / M0 conservative

目的：

- 阻断 command、mode、last-action 等直接 shortcut，让 `z` 尽量来自真实 gait history。

观察：

- 早期 loose mask 曾出现近乎直接 command 泄漏；当前 M0 下 command probe 仍较高，但不等价于直接泄漏，因为 command 会真实诱导 gait。
- V4.1/V4.2 的 `R2_cmd_from_Z` 仍在 `0.62-0.76` 区间。

评价：

- 必要且有效，但还不充分。
- 它解决的是 direct shortcut，不解决“真实 gait 与 command 强相关导致的 indirect predictability”。

### 3.2 `y0/yphi` 分解

目的：

- 让 `z` 解释相位不变属性 `y0`。
- 让 `(z, phi)` 解释相位相关属性 `yphi`。
- 避免 `z` 单独变成 phase code。

观察：

- 最好已完成 checkpoint B6 50ep：`R2_Y0=0.6655` all-source mean，`R2_Yphi=0.3283`。
- A2 stats_tcn 20ep：`R2_Y0=0.6472`，`R2_Yphi=0.3284`。

评价：

- 有效。该分解是目前整个方法最稳的基础。
- 但 `R2_Yphi` 仍只有约 `0.33`，说明 phase-dependent 细节建模还弱，尤其 contact/foot 等组别不够强。

### 3.3 Invariance / RnC

目的：

- 同一 parent/window 的不同 crop 应得到相近 style 表征。
- latent 距离应与 gait descriptor 差异有序相关。

V4.1 观察：

| 设置 | R2_Y0 | phase | rank | rho_G | rho_G\|cmd |
|---|---:|---:|---:|---:|---:|
| B0 base | 0.6605 | 0.1305 | 5.04 | 0.6926 | 0.6827 |
| B1 RnC | 0.6697 | 0.1232 | 5.29 | 0.6954 | 0.6916 |

评价：

- 小幅有效。RnC 对 `R2_Y0`、phase、conditional rho 都有改善。
- 但它没有解决 rank 偏低，也没有明显降低 command predictability。

### 3.4 Rank regularization：`var/cov`

目的：

- 防止 latent collapse，提高 32 维空间利用率。

V4.1 观察：

| 设置 | R2_Y0 | phase | rank | rho_G |
|---|---:|---:|---:|---:|
| B0 base | 0.6605 | 0.1305 | 5.04 | 0.6926 |
| B2 var0.10/cov0.01 | 0.6510 | 0.1304 | 6.05 | 0.5944 |
| B3 var0.20/cov0.02 | 0.6483 | 0.1063 | 7.41 | 0.5977 |
| B4 50ep | 0.6604 all-source | 0.1044 | 10.12 | 0.6282 |

评价：

- 对 rank 有效。
- 但代价是 Y0 或 gait geometry 可能下降。
- B4 50ep 说明 rank 可以拉高到约 10，但 phase leakage 明显高于 B6。

### 3.5 Phase adversary

目的：

- 降低 `z` 中的 phase 信息，逼近相位不变 style code。

V4.1/V4.2 观察：

| 设置 | R2_Y0 | phase | rank | rho_G |
|---|---:|---:|---:|---:|
| B0 base 20ep | 0.6605 | 0.1305 | 5.04 | 0.6926 |
| B6 adv0.10 20ep | 0.6597 | 0.0908 | 6.25 | 0.6049 |
| B6 adv0.10 50ep | 0.6655 all-source | 0.0738 | 7.92 | 0.5950 |
| L4 adv0.20 20ep | 0.6101 | 0.0699 | 7.14 | 0.6088 |

评价：

- 对 phase leakage 明显有效。
- 但 adv 太强会牺牲 style 信息和 Y0。
- 当前较合理区间是 `l_phase_adv=0.10`，继续上调到 `0.20` 不划算。

### 3.6 yphi 梯度控制：`full/stopgrad/off`

目的：

- 检查 yphi decoder 是否把 phase-dependent 信息反向压入 `z`，导致 phase leakage。

V4.2 观察：

| 设置 | R2_Y0 | phase | rho_G | R2_Yphi |
|---|---:|---:|---:|---:|
| L1 full | 0.6214 | 0.0842 | 0.6298 | 0.3144 |
| L5 stopgrad | 0.5987 | 0.0795 | 0.6564 | 0.2888 |
| L6 off | 0.6209 | 0.0753 | 0.6539 | 0.3018 |

评价：

- `stopgrad/off` 可略降 phase 或提高 rho，但明显损害 Y0/Yphi。
- 不建议作为默认方案；更像诊断工具。

### 3.7 Phase correlation penalty

目的：

- 显式惩罚 `z` 与 phase target 的相关性。

V4.2 观察：

| 设置 | R2_Y0 | cmd | mode | phase | rank | rho_G |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 0.6214 | 0.6379 | 0.5075 | 0.0842 | 8.09 | 0.6298 |
| L9 corr0.05 | 0.6168 | 0.6847 | 0.5247 | 0.0715 | 8.18 | 0.6200 |
| L10 inv0.75+corr | 0.6300 | 0.6389 | 0.5718 | 0.0807 | 7.04 | 0.6003 |

评价：

- 可降低 phase，但不稳定，并可能提高 command/mode predictability。
- 不建议作为主线。

### 3.8 Metric target 分组

目的：

- 检查 RnC/metric loss 是否被 action 维度或非 gait 维度主导。

V4.2 观察：

| 设置 | R2_Y0 | phase | rank | rho_G |
|---|---:|---:|---:|---:|
| T1 full | 0.6271 | 0.0787 | 8.07 | 0.6161 |
| T2 no_action | 0.6185 | 0.0766 | 8.34 | 0.6284 |
| T3 kinematic | 0.6270 | 0.0861 | 8.09 | 0.6297 |

评价：

- 有一定诊断价值。
- `kinematic_only` 和 `no_action` 能略改善 rho，但不是决定性因素。
- 默认仍可用 full；后续若追求 gait geometry，可优先试 T3。

### 3.9 结构调整：`stats_tcn`

目的：

- 弥补 TCN 对窗口级统计量、全局 gait 描述不足的问题。

V4.2 观察：

| 设置 | R2_Y0 | phase | rank | rho_G | rho_G\|cmd | OOD drop |
|---|---:|---:|---:|---:|---:|---:|
| A1 tcn | 0.6203 | 0.0826 | 8.10 | 0.6337 | 0.7048 | 0.0695 |
| A2 stats_tcn | 0.6472 | 0.0722 | 8.85 | 0.6396 | 0.7207 | 0.0601 |
| A3 stats_only | 0.6032 | 0.0717 | 6.78 | 0.6001 | 0.6603 | 0.0746 |

评价：

- 这是 V4.2 最有效的调整。
- `stats_only` 不够，说明单纯统计量不能替代时序编码。
- `stats_tcn` 最符合当前目标：Y0 高、phase 低、rank/rho 也更均衡。

### 3.10 Dual latent

目的：

- 把 phase-dependent / reconstruction 辅助信息放入 aux latent，减少主 `z` 污染。

V4.2 观察：

| 设置 | R2_Y0 | phase | rank | rho_G | mode |
|---|---:|---:|---:|---:|---:|
| A4 dual_latent | 0.6250 | 0.0785 | 8.12 | 0.6341 | 0.5574 |
| A5 stats_tcn_dual | 0.6492 | 0.0742 | 8.57 | 0.6329 | 0.5753 |

评价：

- dual latent 单独不是主要收益来源。
- 与 stats_tcn 结合后 A5 的 Y0 和 OOD drop 最好，但 mode leakage 更高，conditional rho 不如 A2。
- 当前 A5 是备选，不是第一候选。

## 4. 当前最优方案评价

需要区分“当前已训练完成的最好 checkpoint”和“下一步最值得作为主线确认的方案”。

### 4.1 当前已完成 checkpoint：B6 50ep

Stage0 all-source + MLP 诊断：

| 指标 | B6 50ep |
|---|---:|
| R2_Y0_from_Z | 0.6655 |
| R2_Y0_gain_over_cmd_mode | 0.4758 |
| R2_cmd_from_Z | 0.7145 |
| R2_mode_from_Z | 0.5548 |
| R2_phase_from_Z | 0.0738 |
| R2_phase_mlp_from_Z | 0.0499 |
| effective_rank | 7.92 |
| rho_G_spearman | 0.5950 |
| rho_G_cond_cmd_spearman | 0.7035 |
| R2_Yphi_from_Z_phi | 0.3283 |
| OOD_drop_R2_Y0 | 0.0570 |

评价：

- 如果现在需要一个可用 encoder checkpoint，B6 50ep 是最稳妥的候选。
- 它的 phase leakage 最低，Y0/Yphi 都比较强。
- 缺点是 rank 和 unconditional gait geometry 不强，说明 latent 空间仍偏压缩，style 结构不够丰富。

### 4.2 当前最值得继续的方案：A2 stats_tcn

20ep 结果：

| 指标 | A2 stats_tcn 20ep |
|---|---:|
| R2_Y0_from_Z | 0.6472 |
| R2_Y0_gain_over_cmd_mode | 0.4589 |
| R2_cmd_from_Z | 0.6398 |
| R2_mode_from_Z | 0.5474 |
| R2_phase_from_Z | 0.0722 |
| effective_rank | 8.85 |
| rho_G_spearman | 0.6396 |
| rho_G_cond_cmd_spearman | 0.7207 |
| R2_Yphi_from_Z_phi | 0.3284 |
| OOD_drop_R2_Y0 | 0.0601 |

评价：

- A2 只训练 20ep，Y0 已接近 B6 50ep，phase 也相当低。
- 它的 rank、rho、conditional rho 都优于 B6 50ep。
- 因此当前最优研究方向不是继续扩大 loss grid，而是把 `stats_tcn` 拉到 50ep 做确认。

### 4.3 A5 stats_tcn_dual 的位置

20ep 结果：

| 指标 | A5 stats_tcn_dual 20ep |
|---|---:|
| R2_Y0_from_Z | 0.6492 |
| R2_cmd_from_Z | 0.6166 |
| R2_mode_from_Z | 0.5753 |
| R2_phase_from_Z | 0.0742 |
| effective_rank | 8.57 |
| rho_G_spearman | 0.6329 |
| rho_G_cond_cmd_spearman | 0.7051 |
| OOD_drop_R2_Y0 | 0.0570 |

评价：

- A5 的 Y0 和 OOD drop 很好。
- 但 mode leakage 高于 A2，conditional rho 低于 A2。
- 当前作为第二候选确认，而不是主线第一候选。

## 5. 当前卡点

### 5.1 Command/mode predictability 仍高

即使 strict mask 生效，`R2_cmd_from_Z` 仍普遍在 `0.62-0.76`。这不一定是 direct leakage，因为 gait 本身由 command 诱导；但它说明 `z` 还不是纯 residual/style 坐标。

风险：

- 后续如果直接用 `z-g*(cmd)` 做 intrinsic reward，残差可能仍混有 command/mode 结构。

需要补的验证：

- `R2(cmd <- residual)`。
- mode-conditioned predictor。
- command swap / same gait different command 的 counterfactual 数据。

### 5.2 Phase invariance 还没有完全解决

当前 best phase 约 `0.07`，MLP phase 对 B6 可降到 `0.0499`，但并非严格为零。

风险：

- `z` 中仍可能混有 segment 起点或相位别名。

需要补的验证：

- phase-shift 同步窗口测试。
- 同一周期步态不同 anchor 的 `z` 方差。
- phase-aligned data augmentation。

### 5.3 Latent rank 中等，32 维空间未充分利用

目前 rank 大多在 `7-10`，远低于 32。

解读：

- 不一定要求用满 32 维，但如果希望 sub-style 多样性，当前 capacity 可能偏紧或被 loss 压缩。

下一步：

- 对 A2/A5 50ep 观察 rank 是否继续上升。
- 试更小 `d_gait=16` 与当前 `32` 对比，确认是否过参数化。

### 5.4 Yphi reconstruction 仍弱

`R2_Yphi_from_Z_phi` 约 `0.32-0.33`。

风险：

- `(z, phi)` 对 phase-dependent gait 状态解释不充分，说明 style-conditioned phase dynamics 还弱。

可能原因：

- yphi target 中 contact/foot 受噪声和离散切换影响大。
- decoder 结构不够强。
- phase 输入可能未对齐真实 gait phase。

### 5.5 还缺 downstream 因果证据

当前所有结论都是 offline encoder/probe 结论，还没有证明：

- actor 会使用 `z`。
- residual intrinsic 会导向稳定、多样、低能耗 gait。
- `z` 的变化会可控地改变步态而不是制造噪声。

这是进入 RL 前最大的证据缺口。

## 6. 总体判断

### 已经证明的部分

1. 当前数据和监督定义可以训练出包含 gait style 信息的 `z`。
2. `y0/yphi` 分解是合理的。
3. phase adversary 能明显降低 phase leakage。
4. rank regularization 能提升 latent rank，但会有信息损失。
5. 单纯调 loss 已经收益有限。
6. `stats_tcn` 是目前最有效的结构改动。

### 尚未证明的部分

1. `z` 是否已经足够接近“纯 gait style coordinate”。
2. `z` 的 residual 是否能从 command/mode 中干净分离。
3. actor 是否会使用 `z`。
4. online intrinsic reward 是否会产生稳定 gait diversity，而不是高能噪声。

## 7. 建议的下一步

### 7.1 立即确认实验

优先跑：

1. `A2_stats_tcn` 50ep，全量 all-source + MLP probe。
2. `A5_stats_tcn_dual` 50ep，全量 all-source + MLP probe。
3. B6 50ep、B4 50ep、A2 50ep、A5 50ep 统一比较。

成功标准建议：

- `R2_Y0 >= 0.67`
- `R2_phase_from_Z <= 0.075`
- `R2_phase_mlp_from_Z <= 0.06`
- `effective_rank >= 9`
- `rho_G_cond_cmd_spearman >= 0.72`
- `OOD_drop_R2_Y0 <= 0.06`

### 7.2 必须补的诊断

1. residual probe：`cmd/mode/phase <- z - g(cmd, mode)`。
2. phase-shift invariance test。
3. source-group/bucket-group 的 failure breakdown。
4. yphi group 的 target 清理，尤其 contact/foot。

### 7.3 进入 RL 前的硬门槛

只有当下列条件满足，才建议把 encoder 接入 RL 主实验：

1. A2/A5 50ep 至少一个稳定优于 B6 50ep。
2. residual 中 command/mode/phase predictability 明显下降。
3. phase-shift test 证明 `z` 对相位起点不敏感。
4. offline gait descriptor 证明确实存在可控 style 维度。

## 8. 一句话结论

目前最优已完成 checkpoint 是 B6 50ep；目前最优研究方向是 A2 `stats_tcn` 50ep confirmation。V4.2 的核心结论是：问题主要不是训练轮数或 loss 权重，而是原 TCN 缺少窗口级 gait 统计结构；`stats_tcn` 正好补上了这个缺口。但项目离“可用于下游 RL 的干净 gait style code”还差 residual 解耦、phase-shift invariance 和 actor causality 三组证据。
