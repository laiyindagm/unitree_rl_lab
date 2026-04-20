# Contrastive Latent Policy：算法设计与理论总结

> 本文档总结了 Contrastive Latent Policy 的完整设计过程：  
> 从问题定义出发，经由三种方案的探索、隐空间几何的理论推导、  
> 编码器/生成器的选型论证、收敛性分析，最终形成基于乘积球面的缓存式对比强化学习架构。

---

## 目录

1. [问题定义](#1-问题定义)
2. [方案探索：从 CVAE 到乘积球面](#2-方案探索从-cvae-到乘积球面)
3. [核心理论：乘积球面隐空间](#3-核心理论乘积球面隐空间)
4. [架构设计](#4-架构设计)
5. [训练算法：缓存式两阶段 PPO](#5-训练算法缓存式两阶段-ppo)
6. [收敛性分析](#6-收敛性分析)
7. [参数规模论证](#7-参数规模论证)
8. [设计决策总结](#8-设计决策总结)

---

## 1. 问题定义

### 1.1 背景

面向 Unitree G1 人形机器人（15 自由度）的 locomotion 强化学习任务。机器人在 Isaac Lab 仿真环境中，以 50Hz 控制频率执行速度跟踪任务，接收连续的三维速度指令 $c = [v_x, v_y, \omega]$。

标准的端到端 RL 策略（观测 → MLP → 动作）虽可工作，但存在以下不足：

- **隐表征无结构**：隐层特征不具备可解释的几何意义
- **指令区分能力弱**：不同速度指令对应的内部表征缺乏显式分离
- **动态信息利用不充分**：历史观测中蕴含的惯性、接触模式等信息未被显式建模

### 1.2 设计目标

设计一个**观测编码器 + 条件生成器 + 策略网络**的组合架构，使隐空间满足三条几何性质：

| 性质 | 符号 | 含义 |
|------|------|------|
| **P1：同系列内分离** | Intra-series separation | 同一速度分量（如 $v_x$）的不同取值在隐空间中相互远离 |
| **P2：跨系列分离+一致性** | Inter-series separation | 不同速度分量各自在独立子空间中分离，彼此不干扰 |
| **P3：组合接近性** | Joint-component proximity | 联合指令 $(v_x, v_y)$ 的隐表征应靠近其单独分量 $v_x$、$v_y$ 的表征 |

同时，编码器的产出应具备**序列预测能力**——给定隐变量和指令，能预测未来若干步的关节动作，为策略网络提供"计划"信号。

### 1.3 指令空间

速度指令由三个连续分量组成，每个分量在训练时从离散等级中采样：

$$v_x \in \{-0.3, 0, 0.3, 0.5, 0.8, 1.0\}, \quad v_y \in \{-0.3, \ldots, 0.3\}, \quad \omega \in \{-0.5, \ldots, 0.5\}$$

组合空间为三者的笛卡尔积（数百种组合），远超简单离散分类的规模。

---

## 2. 方案探索：从 CVAE 到乘积球面

### 2.1 方案一：CVAE + InfoNCE + 序列预测器

**思路**：

- 编码器：Transformer 编码历史观测 → 条件 VAE 产出高斯隐变量 $z \sim \mathcal{N}(\mu, \sigma^2 I)$
- 对比学习：在投影空间上施加 InfoNCE，同指令为正、异指令为负
- 序列预测：MLP/GRU 解码器从 $z$ + 指令 预测未来关节角度
- 策略输出：$\pi(o_t, z_{\text{detach}}, \hat{q}_{\text{detach}}) \to a_t$

**问题**：

1. **KL 正则与表征质量的冲突**——高斯 VAE 存在 posterior collapse 风险，需要 KL warmup、free bits 等技巧
2. **隐空间无内禀度量**——$\mathbb{R}^d$ 中的欧氏距离不直接反映语义相似度，需要额外投影头
3. **指令组合空间的爆炸**——将 $v_x \times v_y \times \omega$ 的组合视为离散标签，正样本极度稀缺

### 2.2 方案二：VQ-VAE + 指令对齐

**思路**：用向量量化（codebook）替代连续 KL，每个 code 对应一种运动模式。

**问题**：
- 离散 codebook 难以表达连续速度空间的序数结构
- Straight-through estimator 带来训练不平滑
- codebook 利用率不均匀

### 2.3 方案三：CPC（对比预测编码）

**思路**：将对比学习和序列预测统一——用对比损失训练序列预测。

**问题**：
- 关节角度预测精度不如直接 MSE 监督
- 仍缺乏对指令组合结构的显式建模

### 2.4 关键洞察：需要几何化的隐空间

上述三种方案的共同局限在于：**隐空间的几何结构是后天学出来的，而非先天内建的**。

这引出了两个关键问题的探索：

1. **能否选择一种参数化空间，使距离度量本身就编码语义相似度？**
    - 探索了超球面（vMF 分布）、双曲空间（Poincaré ball）、乘积流形
    - **结论**：超球面最适合——天然余弦相似度、无 posterior collapse、与 InfoNCE 兼容

2. **能否利用指令的组合结构，先学单指令分离再组合？**
    - 探索了两阶段对比（先分离后桥接）vs 联合训练
    - **结论**：乘积球面天然分离不同分量的空间，无需显式两阶段

---

## 3. 核心理论：乘积球面隐空间

### 3.1 单球面不可行性证明

**定理**：在单一超球面 $\mathbb{S}^{d-1}$ 上，不存在嵌入方案同时满足 P1、P2、P3。

**证明思路**（Cauchy-Schwarz 反证）：

| 性质 | 要求 |
|------|------|
| P1 | 同分量不同值远离：$\langle z_{v_x=0.3}, z_{v_x=0.8} \rangle \leq -\epsilon$ |
| P2 | 跨分量独立分离：$v_x$ 方向的分离不影响 $v_y$ 方向 |
| P3 | 组合接近分量：$z_{(v_x, v_y)} \approx$ 某种 "$z_{v_x}$ 和 $z_{v_y}$ 的折中" |

在单球面中，$z_{(v_x, v_y)}$ 必须同时靠近 $z_{v_x}$（P3）和远离其他 $v_x$ 取值的 $z$（P1），这两个约束作用在同一组维度上。当组合数量增加时，球面容量不足以同时满足所有约束。

**具体矛盾**：P2 要求 $v_x$ 和 $v_y$ 的分离在正交子空间中独立进行，但单球面的所有维度是耦合的——$z_{(v_x, v_y)}$ 的 $v_x$ 相关维度的取值会干扰它在 $v_y$ 维度上的位置。

### 3.2 乘积球面方案

**定义**：

$$\mathcal{Z} = \mathbb{S}^{d_x - 1} \times \mathbb{S}^{d_y - 1} \times \mathbb{S}^{d_w - 1}$$

三个独立的单位超球面，分别对应三个速度分量。隐变量为三元组：

$$z = (z^x, z^y, z^w), \quad z^s \in \mathbb{S}^{d_s - 1}, \quad s \in \{x, y, w\}$$

**距离度量**：

$$d_{\mathcal{Z}}(z_i, z_j) = \alpha_x \cdot d_{\mathbb{S}}(z_i^x, z_j^x) + \alpha_y \cdot d_{\mathbb{S}}(z_i^y, z_j^y) + \alpha_w \cdot d_{\mathbb{S}}(z_i^w, z_j^w)$$

其中 $d_{\mathbb{S}}(u, v) = \arccos(\langle u, v \rangle)$ 为测地线距离。

### 3.3 三条性质的满足

**P1 自动满足**：在每个子球面 $\mathbb{S}^{d_s-1}$ 上，分解式 InfoNCE 迫使同分量不同值远离。32 维球面可容纳约 32 个近似正交的方向，远超单分量的等级数（5-6 种）。

**P2 自动满足**：三个子球面物理隔离，$v_x$ 的分离发生在 $\mathbb{S}^{d_x-1}$，完全不影响 $\mathbb{S}^{d_y-1}$ 中 $v_y$ 的分离。

**P3 自动满足**（关键定理）：

$$\text{若 } z^x_{(v_x, v_y)} = z^x_{v_x} \text{（对比学习仅按 } v_x \text{ 标签分类）}$$

则组合指令在 $\mathbb{S}^{d_x-1}$ 子空间中自动位于 $v_x$ 对应的聚类中心，在 $\mathbb{S}^{d_y-1}$ 中位于 $v_y$ 对应的聚类中心。因此：

$$d_{\mathcal{Z}}(z_{(v_x, v_y)}, z_{v_x}) = \alpha_y \cdot d_{\mathbb{S}}(z^y_{(v_x,v_y)}, z^y_{v_x}) \quad \text{(仅 } v_y \text{ 分量有距离)}$$

这正是"组合隐变量接近其分量隐变量"的几何实现——**不需要额外损失函数或训练阶段**。

### 3.4 分解式对比学习

对三个子球面**独立**施加 SupCon-style InfoNCE：

$$\mathcal{L}_{\text{NCE}} = \sum_{s \in \{x, y, w\}} \mathcal{L}_{\text{InfoNCE}}^s(p^s, \text{label}^s, \tau)$$

其中 $\text{label}^s$ 是将连续速度分量 $v_s$ 离散化到最近等级后的索引。

**优势**：
- 避免组合标签的指数爆炸（$N_x \times N_y \times N_w$ → $N_x + N_y + N_w$）
- 每个子空间的聚类问题更简单
- P3 作为数学推论自动成立，无需显式约束

---

## 4. 架构设计

### 4.1 编码器选型

分析了 5 种编码器候选方案后，保留两种：

| 编码器 | 参数量 | 感受野 | 优势 | 代表工作 |
|--------|--------|--------|------|---------|
| **1D Causal TCN**（推荐） | ~62K | 精确 = $T=5$ | 轻量、部署友好、无序列依赖 | Walk These Ways, RMA, DreamWaQ |
| **Transformer** | ~260K | 全局 | 灵活的注意力模式 | 现有项目已使用 |

**TCN 推荐理由**：

- 两层因果 Conv1d（kernel=3），感受野 $1 + 2(k-1) = 5$ 精确覆盖历史窗口
- 参数量仅为 Transformer 的 1/4，推理延迟更低
- 在 locomotion 领域已被 Walk These Ways、RMA 等工作验证

### 4.2 生成器选型

分析了 4 种生成器候选后，选择 MLP + FiLM：

| 方案 | 核心问题 |
|------|---------|
| **MLP + FiLM**（选用） | 确定性映射，与 on-policy RL 兼容 |
| Flow Matching | 分布拟合需要非平稳数据的分布假设，与 on-policy RL 不兼容 |
| CVAE | 增加 KL 正则复杂度，on-policy 下先验假设不成立 |
| GRU 自回归 | 串行推理慢，部署困难 |

**FiLM（Feature-wise Linear Modulation）的物理直觉**：

$$h_{\text{mod}} = (1 + \gamma_c) \odot h_z + \beta_c$$

- **乘法项** $\gamma_c$：速度指令**缩放**不同隐特征的幅度（如速度影响步幅的振幅）
- **加法项** $\beta_c$：转向指令**偏移**特征（如转向使步态产生左右不对称的偏移）

相比简单拼接 $[z; e_c]$（只有加法交互），FiLM 引入了乘法交互，表达力严格更强。

### 4.3 指令嵌入：连续值 MLP

直接将连续速度 $[v_x, v_y, \omega]$ 通过 2 层 MLP 映射到嵌入空间：

$$e_c = \text{MLP}([v_x, v_y, \omega]) \in \mathbb{R}^{32}$$

**选择连续嵌入而非离散嵌入的理由**：

1. 速度指令本身是连续物理量，保留序数结构
2. 训练中用离散等级采样，但部署时可接收任意连续速度
3. 避免 one-hot 的维度爆炸或 embedding table 的稀疏更新

### 4.4 完整数据流

```
flat_obs [B, T×54]
     │ split
     ├── history_no_cmd [B, 5, 51] ──→ Encoder(TCN/Transformer) ──→ h_enc [B, 96]
     │                                                                    │
     │                                          ProductSphereProjection ──┤
     │                                             │       │       │
     │                                            z^x     z^y     z^w   (各 [B,32], L2归一化)
     │                                             │       │       │
     │                                    ┌── [训练] ContrastiveProjector → L_NCE
     │                                    │
     │                                    └── z_cat = [z^x; z^y; z^w]  [B, 96]
     │                                              │
     ├── cmd [B, 3] ──→ CommandEmbedding ──→ e_c [B, 32]
     │                                              │
     │                                    FiLM Generator ──→ â [B, 45]
     │                                              │
     ├── o_current [B, 51]                          │
     │          │                                   │
     │          └── concat([o_cur, e_c, z_cat, â]) ──→ Policy MLP [512,256,128] ──→ a_t [B, 15]
     │                      (all detached)
```

### 4.5 梯度截断设计

**核心原则：表征学习和策略优化使用独立的梯度通路。**

策略 MLP 的输入 $[o_t, e_c, z, \hat{a}]$ 中，$e_c$、$z$、$\hat{a}$ 均经过 `.detach()`。

**数学推理**：

不截断时，编码器参数 $\phi$ 的梯度包含：

$$\nabla_\phi \mathcal{L} = \underbrace{\alpha \nabla_\phi \mathcal{L}_{\text{NCE}} + \beta \nabla_\phi \mathcal{L}_{\text{gen}}}_{\text{结构化信号（方差小）}} + \underbrace{\nabla_\phi \mathcal{L}_{\text{PPO}}}_{\text{策略信号（方差极大）}}$$

PPO 的 surrogate loss 经过采样、裁剪、归一化，梯度方差比表征损失大 1-2 个数量级，会干扰编码器学习稳定的隐空间结构。截断后两组梯度独立，各自收敛。

---

## 5. 训练算法：缓存式两阶段 PPO

### 5.1 核心设计决策：缓存方案

训练分为两个阶段（Phase A/B），在每次 `update()` 的同一个 mini-batch 迭代中交替执行：

| 阶段 | 输入来源 | 损失 | 更新参数 |
|------|---------|------|---------|
| Phase A（表征） | 从 batch 观测**重新编码** | $\alpha \mathcal{L}_{\text{NCE}} + \beta \mathcal{L}_{\text{gen}}$ | encoder, sphere_proj, contrast_proj, cmd_embed, generator |
| Phase B（策略） | 使用 rollout 时**缓存**的 z/e_c/â | $\mathcal{L}_{\text{PPO}}$ | policy MLP, distribution |

**为什么 Phase B 使用缓存而非重新编码？**

这是收敛性分析（第 6 节）的核心发现。如果 Phase B 重新编码：

1. Phase A 已更新编码器 → Phase B 编码得到 $z_{\text{new}} \neq z_{\text{old}}$
2. 但动作 $a_t$ 是在 $z_{\text{old}}$ 下采样的
3. PPO 计算 $\log \pi(a_t | z_{\text{new}})$ 的 importance ratio，此 ratio 的偏差无法被 clip 修正
4. 这构成了一个**非平稳优化目标**，可能导致策略不收敛

缓存方案确保 Phase B 的 $\log \pi(a_t | z_{\text{old}})$ 与 rollout 时一致，消除此风险。

### 5.2 Rollout 阶段

```python
# 每个 rollout step：
action = actor.act(obs)              # 内部调用 get_latent()
z_cat, e_c, a_pred, o_cur = actor.get_cached_repr()
storage.cached_z_cat[step] = z_cat   # 缓存到 Storage
storage.cached_e_c[step] = e_c
storage.cached_a_pred[step] = a_pred
storage.cached_o_current[step] = o_cur
# ... 正常存储 obs, actions, rewards, dones
```

### 5.3 Update 阶段

```python
for batch in mini_batch_generator:
    # ── Phase A：表征学习 ──
    z_x, z_y, z_w = actor.encode(batch.flat_obs)     # 重新编码（用最新编码器）
    p_x, p_y, p_w = actor.project_contrastive(z_x, z_y, z_w)
    L_NCE = factored_infonce(p_x, p_y, p_w, cmd, levels, τ)
    L_gen = sequence_prediction_loss(actor.generate(z_cat, e_c), future_actions)
    (α * L_NCE + β * L_gen).backward()
    repr_optimizer.step()

    # ── Phase B：策略优化（使用缓存） ──
    latent = actor.get_latent_from_cache(
        batch.cached_o_current, batch.cached_e_c,     # ★ 不重新编码
        batch.cached_z_cat, batch.cached_a_pred
    )
    action_mean = actor.mlp(latent)
    log_prob = distribution.log_prob(batch.actions)
    # ... 标准 PPO surrogate + value loss + entropy
    policy_optimizer.step()
```

### 5.4 损失函数

**分解式 InfoNCE**（三个子球面独立）：

$$\mathcal{L}_{\text{NCE}} = \sum_{s \in \{x,y,w\}} \left[ -\frac{1}{|\mathcal{P}_s|} \sum_{(i,j) \in \mathcal{P}_s} \log \frac{\exp(\langle p_i^s, p_j^s \rangle / \tau)}{\sum_{k \neq i} \exp(\langle p_i^s, p_k^s \rangle / \tau)} \right]$$

其中 $\mathcal{P}_s = \{(i,j) : \text{label}_s(i) = \text{label}_s(j), i \neq j\}$ 为同标签正对。

**序列预测损失**（时间衰减 MSE）：

$$\mathcal{L}_{\text{gen}} = \frac{1}{K} \sum_{k=1}^{K} \gamma^{k-1} \| \hat{a}_{t+k} - a_{t+k}^{\text{gt}} \|^2$$

其中 $\gamma = 0.9$ 使近期预测权重更大，$K=3$。

### 5.5 损失系数 Schedule

| 系数 | 初始值 | 变化 | 说明 |
|------|--------|------|------|
| $\alpha$（对比） | 0.1 | 常量 | 对比损失权重 |
| $\beta$（生成） | 0.5 | 线性衰减至 0.1（10000 iter） | 早期重视预测能力，后期重视策略 |
| $\tau$（温度） | 0.5 | 可学习 $\tau = e^{\log \tau}$ | 自适应调整对比锐度 |

---

## 6. 收敛性分析

### 6.1 已识别的 7 类风险

| 风险 | 描述 | 严重程度 | 解决方案 |
|------|------|---------|---------|
| **R1** | Phase A 更新编码器后 Phase B 的 z 不一致 | **不可忽略** | ✅ 缓存方案 |
| **R2** | $\nabla \mathcal{L}_{\text{NCE}}$ 和 $\nabla \mathcal{L}_{\text{gen}}$ 方向冲突 | 可控 | 投影头隔离 + 监控 $\cos(\nabla_{\text{NCE}}, \nabla_{\text{gen}})$ |
| **R3** | L2 归一化在零向量附近梯度爆炸 | 可忽略 | $\epsilon$ 保护：$z / (\|z\| + \epsilon)$ |
| **R4** | Batch 中某标签无正对 | 可忽略 | $B=24576 \gg$ 标签数，每标签超 100 个样本 |
| **R5** | 策略更新导致下一 rollout 的 obs 分布偏移 | 可忽略 | PPO clip 约束策略每步变化 |
| **R6** | 所有 $z^s$ 坍缩到同一方向（mode collapse） | **需监控** | InfoNCE 的鞍点性质保证不会停在坍缩点 |
| **R7** | Phase B 使用 $z_{\text{new}}$ 计算 $\log\pi(a|z_{\text{new}})$ 但 $a$ 在 $z_{\text{old}}$ 下采样 | **不可忽略** | ✅ 缓存方案（与 R1 同解） |

### 6.2 R1/R7 的详细分析

**问题形式化**：

设 rollout 时编码器为 $f_\phi$，Phase A 更新后为 $f_{\phi'}$。若 Phase B 重新编码：

$$z_{\text{new}} = f_{\phi'}(o), \quad z_{\text{old}} = f_\phi(o)$$

PPO 的 importance sampling ratio：

$$r = \frac{\pi_\theta(a | \text{latent}(z_{\text{new}}))}{\pi_{\theta_{\text{old}}}(a | \text{latent}(z_{\text{old}}))}$$

分子和分母的条件不同（$z_{\text{new}}$ vs $z_{\text{old}}$），introducing off-policy bias that PPO's clip cannot correct，因为 clip 只约束 $\theta$ 的变化，不约束 input 的变化。

**缓存方案的解决**：

$$r = \frac{\pi_\theta(a | \text{latent}(z_{\text{old}}))}{\pi_{\theta_{\text{old}}}(a | \text{latent}(z_{\text{old}}))}$$

条件相同，ratio 仅反映 $\theta$ 的变化，标准 PPO 理论保证成立。

### 6.3 R6 的分析：Mode Collapse

**定理**：在 InfoNCE 损失下，全部 $z^s$ 坍缩到同一方向是一个**鞍点**而非局部最小值。

**证明思路**：设所有 $p_i^s = e_1$（坍缩到同一方向），则：

$$\frac{\partial \mathcal{L}_{\text{NCE}}^s}{\partial p_i^s} = 0 \quad \text{（梯度为零，确实是驻点）}$$

但 Hessian 在垂直于 $e_1$ 的方向上有负特征值——任何微小扰动都会降低损失。因此，只要优化器有随机性（SGD 的 mini-batch 噪声），就不会停在坍缩点。

**实践保障**：监控 uniformity 指标 $\mathcal{U} = \log \mathbb{E}[e^{-t\|z_i - z_j\|^2}]$，若趋近 0 则报警。

---

## 7. 参数规模论证

### 7.1 合理性分析

从三个视角论证参数量是否合适：

**视角 1：信息论下界**

模型需编码的信息量：3 个速度分量各 5-6 级 → 约 $3 \times \log_2(6) \approx 8$ bits。球面上 32 维向量可编码 $\log_2(32) \approx 5$ bits/球面 → 总 15 bits，充裕。

但编码器还需从 $5 \times 51 = 255$ 维输入中提取动态信息（惯性、接触），这要求一定的网络容量。

**视角 2：过拟合比率**

$$\text{overfit ratio} = \frac{\text{train samples per update}}{\text{trainable params}} = \frac{24576 \times 5}{421\text{K}} \approx 292 \gg 10$$

远超过拟合阈值，参数量安全。

**视角 3：经验对比**

| 模型 | 参数量 | 任务 |
|------|--------|------|
| Walk These Ways (TCN policy) | ~300K | Quadruped locomotion |
| 本方案（TCN） | ~421K | Humanoid 15DOF locomotion |
| 本方案（Transformer） | ~619K | 同上 |
| 现有 TransformerHistoryModel | ~1.5M | 同上（基线） |

新方案参数量低于现有基线，与同类工作处于同一量级。

### 7.2 显存开销

| 项目 | 大小 |
|------|------|
| 标准 Rollout Storage | ~200 MB |
| 额外缓存 buffer（z, e_c, â, o_cur, future, mask） | ~106 MB |
| 总计 | ~306 MB |

在 RTX 4090 (24GB) 上可接受。

---

## 8. 设计决策总结

### 8.1 决策链路

```
问题：观测编码器 + 对比学习 + 序列预测 + 策略输出
  │
  ├── 隐空间选型：高斯(R^d) vs 超球面(S^d) vs 双曲空间
  │     └── 选择：超球面 → 天然度量、无 posterior collapse
  │
  ├── 球面结构：单球面 vs 乘积球面
  │     └── 证明：单球面无法同时满足 P1+P2+P3 → 乘积球面
  │
  ├── 编码器：Transformer vs TCN vs MLP vs GRU vs SSM
  │     └── 选择：TCN（推荐）+ Transformer（备选）
  │
  ├── 生成器：MLP vs Flow Matching vs CVAE vs GRU
  │     └── 选择：MLP + FiLM（确定性、on-policy 兼容）
  │
  ├── 指令嵌入：one-hot vs 分解嵌入 vs 连续 MLP
  │     └── 选择：连续 MLP（保留物理结构、部署灵活）
  │
  ├── 梯度截断：不截断 vs 部分截断 vs 完全截断
  │     └── 选择：完全截断（PPO 梯度方差过大会破坏表征）
  │
  └── 训练方案：重新编码 vs 缓存
        └── 选择：缓存方案（消除 R1/R7 收敛风险）
```

### 8.2 创新点

1. **乘积球面隐空间**：将速度指令的三个分量映射到三个独立超球面，使 P1/P2/P3 性质通过几何结构自动满足，无需额外损失项
2. **分解式 InfoNCE**：在三个子球面上独立施加对比损失，避免组合标签的指数爆炸
3. **缓存式两阶段训练**：rollout 时缓存编码器产物，Phase B 使用缓存值更新策略，消除编码器更新引起的 on-policy 不一致性

### 8.3 部署注意事项

部署时**不使用缓存方案**。缓存仅在训练的 Phase B 中使用。部署时直接运行完整流水线：

$$\text{obs} \to \text{Encoder} \to \text{SphereProj} \to \text{FiLM Generator} \to \text{Policy MLP} \to \text{action}$$

对比投影头（ContrastiveProjector）在部署时丢弃。

---

## 附录：文件结构

```
source/unitree_rl_lab/unitree_rl_lab/utils/
├── contrastive_latent_model.py    # 新增
│   ├── CausalConv1d, TCNEncoder, TransformerEncoder
│   ├── ProductSphereProjection, ContrastiveProjector
│   ├── CommandEmbedding, FiLMGenerator
│   └── ContrastiveLatentModel(MLPModel)
├── contrastive_ppo.py             # 新增
│   ├── factored_infonce(), sequence_prediction_loss()
│   ├── _uniformity(), _alignment()
│   └── ContrastivePPO(PPO)
└── rsl_rl_ppo_cfg.py              # 扩展
    ├── RslRlContrastiveModelCfg
    ├── RslRlContrastivePpoAlgorithmCfg
    └── G115DofContrastiveTCNPPORunnerCfg
```

---

## 参考文献

| 概念/方法 | 来源 |
|----------|------|
| InfoNCE / 对比学习互信息下界 | Oord et al., 2018 |
| 投影头保护表征 | Chen et al. (SimCLR), 2020 |
| Uniformity & Alignment on hypersphere | Wang & Isola, ICML 2020 |
| 超球面 VAE (S-VAE) | Davidson et al., AISTATS 2018 |
| 乘积流形 VAE | Skopek et al., ICLR 2020 |
| FiLM 条件化 | Perez et al., AAAI 2018 |
| Walk These Ways (TCN for locomotion) | Margolis et al., 2023 |
| RMA (Rapid Motor Adaptation) | Kumar et al., 2021 |
| DreamWaQ | Nahrendra et al., 2023 |
