# 观测编码器设计方案：对比学习 + 序列预测 + 策略生成

## 补充讨论：离散指令集合、组合指令与隐空间对比学习

> 本文档是对"观测编码器设计方案"讨论的补充，重点总结关于离散指令集合规模、两系列指令组合场景、以及隐空间参数化与组合指令对比学习方案的分析，最终给出完整的模型设计和训练方案。
> 仓库现有模块实现（TransformerHistoryModel、TransformerPPO 等）仅在"关联实现"章节简要提及。

---

## 目录

1. [问题回顾：从连续指令到离散/组合指令](#1-问题回顾从连续指令到离散组合指令)
2. [离散指令集合规模问题](#2-离散指令集合规模问题)
3. [两系列指令组合场景](#3-两系列指令组合场景)
4. [隐空间参数化方案分析](#4-隐空间参数化方案分析)
5. [组合指令对比学习方案](#5-组合指令对比学习方案)
6. [最终模型设计](#6-最终模型设计)
7. [训练方案](#7-训练方案)
8. [理论分析与收敛性讨论](#8-理论分析与收敛性讨论)
9. [与仓库现有实现的关联](#9-与仓库现有实现的关联)
10. [参考文献](#10-参考文献)

---

## 1. 问题回顾：从连续指令到离散/组合指令

### 1.1 原始场景

在前期讨论中，我们的观测编码器面对的指令空间是**低维连续的**——3D 速度指令 $(v_x, v_y, \omega_z)$，作为观测向量的一部分直接送入编码器。此时指令对编码器而言只是 3 个浮点维度，不存在结构性编码问题。

但我们在讨论中逐步意识到：当任务复杂度升级时，指令空间会发生**质变**。

### 1.2 指令空间的演化路径

| 阶段 | 指令空间 | 维度 | 性质 |
|------|---------|------|------|
| **当前** | 速度指令 $(v_x, v_y, \omega_z)$ | 3D 连续 | 低维、平坦、语义均匀 |
| **近期** | 速度 + 步态模式 (walk/run/crouch) | 3D 连续 + 离散枚举 | 混合类型 |
| **中期** | 速度 + 步态 + 上半身动作 (carry/wave/idle) | 3D + 离散 + 离散 | 两系列组合 |
| **远期** | 多模态：语音/视觉/高层规划指令 | 高维、异构 | 需要统一编码框架 |

**核心问题**：当指令从"3 个浮点数"变为"来自两个甚至多个独立维度的离散指令组合"时：

1. **编码效率**：one-hot 编码随组合数指数增长，不可扩展
2. **语义结构**：编码器需理解指令间的相似性（"慢走" ≈ "正常走" ≠ "跳跃"）
3. **组合泛化**：训练中未见过的指令组合，策略能否合理外推？
4. **与观测编码的协同**：指令编码如何与历史观测编码有效融合？

这些问题构成了本文档的核心讨论。

---

## 2. 离散指令集合规模问题

### 2.1 问题提出

> **讨论要点**："离散指令的集合比较大"

当指令空间包含大量离散指令时（例如 $|\mathcal{C}| = 50$ 种不同的行为模式），传统编码方式面临困境：

#### 2.1.1 One-hot 编码的不可扩展性

设指令集 $\mathcal{C} = \{c_1, c_2, \ldots, c_N\}$，one-hot 编码 $\text{enc}(c_i) = e_i \in \mathbb{R}^N$：

- **维度随集合线性增长**：$N=50$ 种指令 → 50D 稀疏向量，浪费观测带宽
- **语义平坦**：$\|e_i - e_j\|_2 = \sqrt{2} \;\; \forall i \neq j$，所有指令等距，丢失结构
- **无泛化能力**：未见过的指令 $c_{N+1}$ 无法从已有指令推断

#### 2.1.2 手工 embedding 的局限性

将语义相似的指令手工映射到邻近的低维向量（如 "walk" → [1,0], "run" → [1.5, 0], "crouch" → [0.5, -1]）：

- **主观性强**：什么是"语义相似"依赖设计者的先验
- **维度选择困难**：嵌入维度过低丢失信息，过高需要更多标注
- **无法适应策略需求**：固定嵌入不会随训练演化，可能与策略实际需要的区分度不匹配

### 2.2 讨论结论：学习化指令嵌入

我们讨论后的共识是：**指令编码应该被学习，而非手工设计**。

具体方案：引入可学习的指令嵌入表 (learnable command embedding table)：

$$\mathbf{E} \in \mathbb{R}^{N \times d_c}$$

其中 $N$ 为离散指令数，$d_c$ 为嵌入维度。指令 $c_i$ 的嵌入为：

$$z_c = \mathbf{E}[i] \in \mathbb{R}^{d_c}$$

**关键设计决策**：

| 决策 | 选择 | 理由 |
|------|------|------|
| 嵌入维度 $d_c$ | 16–64 | 远小于 one-hot 的 $N$，但保留足够表达力 |
| 初始化 | 正交初始化或从预训练语义向量初始化 | 避免随机初始化的对称性 |
| 是否冻结 | **不冻结**，端到端学习 | 让嵌入适应策略需求 |
| 正则化 | 使用对比损失约束结构 | 防止嵌入退化为 one-hot |

### 2.3 规模化分析

| 指令数 $N$ | One-hot 维度 | 学习嵌入维度 $d_c$ | 压缩率 |
|-----------|-------------|-------------------|--------|
| 10 | 10 | 16 | 0.6× (无优势) |
| 50 | 50 | 32 | 1.6× |
| 200 | 200 | 32 | 6.3× |
| 1000 | 1000 | 64 | 15.6× |

**结论**：当 $N > 30$ 时，学习嵌入在维度效率上显著优于 one-hot。当 $N > 100$ 时，one-hot 在实践中不可行（观测维度爆炸）。

---

## 3. 两系列指令组合场景

### 3.1 问题提出

> **讨论要点**："存在两系列的指令组合给予的情况"

现实机器人任务中，指令往往来自**多个独立维度**：

- **系列 A（运动模式）**：$\mathcal{A} = \{\text{walk}, \text{run}, \text{crouch}, \text{turn-in-place}, \text{stand}\}$，$|\mathcal{A}| = 5$
- **系列 B（上肢动作）**：$\mathcal{B} = \{\text{carry}, \text{wave}, \text{point}, \text{idle}\}$，$|\mathcal{B}| = 4$

每一步，agent 收到的指令是一个**组合** $(a_i, b_j) \in \mathcal{A} \times \mathcal{B}$。

#### 3.1.1 组合爆炸

总指令数 $|\mathcal{A} \times \mathcal{B}| = |\mathcal{A}| \cdot |\mathcal{B}| = 20$。

如果两个系列各自扩展：

| $|\mathcal{A}|$ | $|\mathcal{B}|$ | 组合数 $|\mathcal{A} \times \mathcal{B}|$ |
|--:|--:|--:|
| 5 | 4 | 20 |
| 10 | 8 | 80 |
| 20 | 15 | 300 |
| 50 | 30 | 1,500 |

如果使用 one-hot 编码每个组合，维度随乘积增长——这显然不可行。

#### 3.1.2 更深层的问题：训练覆盖率

即使编码维度可控，训练数据可能**无法覆盖所有组合**：

- 如果组合数为 300，但每次 rollout 只有 4096 个并行环境、每个 episode 只采样 1 次指令
- 那么每次 rollout 最多覆盖 ~14 个组合（假设均匀采样）
- 需要 ~22 次 rollout 才能至少见过每个组合一次（概率意义下需更多）

**核心挑战**：策略需要对未训练过的组合具有**组合泛化**（compositional generalization）能力。

### 3.2 讨论结论：因式分解嵌入

我们讨论得出的方案是**因式分解嵌入**（factored embedding）：

**不编码组合 $(a_i, b_j)$ 作为整体，而是分别编码两个系列，然后在隐空间中组合**：

$$z_A = \mathbf{E}_A[i] \in \mathbb{R}^{d_A}, \quad z_B = \mathbf{E}_B[j] \in \mathbb{R}^{d_B}$$

组合方式的选择（我们讨论了三种）：

#### 方案 1：拼接（Concatenation）

$$z_c = [z_A; z_B] \in \mathbb{R}^{d_A + d_B}$$

- ✅ 最简单，保留两个系列的完整信息
- ✅ 参数量随系列数线性增长（而非乘积）
- ❌ 无法显式建模两系列之间的交互
- ❌ 下游网络需隐式学习交互

#### 方案 2：加法（Additive Composition）

$$z_c = z_A + z_B \in \mathbb{R}^{d_c}, \quad d_A = d_B = d_c$$

- ✅ 维度不增长
- ✅ 隐式强制"组合 = 叠加"的语义结构
- ✅ 类比 Word2Vec 的加法组合性：king - man + woman ≈ queen
- ❌ 信息可能混淆——如果 $z_A$ 和 $z_B$ 占据同一子空间

#### 方案 3：双线性交互（Bilinear / Tensor Product）

$$z_c = \text{MLP}(z_A \otimes z_B) \in \mathbb{R}^{d_c}$$

或使用低秩近似：

$$z_c = W_3 \cdot \sigma(W_1 z_A \odot W_2 z_B)$$

其中 $\odot$ 为逐元素乘法（Hadamard product）。

- ✅ 显式建模两系列的交互
- ✅ 可学习"哪些维度应交互"
- ❌ 参数量略大
- ❌ 需要 $d_A = d_B$

### 3.3 讨论最终选择

经过讨论，我们确定了**混合方案**：

$$z_c = \underbrace{z_A + z_B}_{\text{加法组合}} + \underbrace{\text{MLP}_{inter}(z_A \odot z_B)}_{\text{交互修正项}}$$

**理由**：
- 加法项提供默认的组合语义（主效应）
- Hadamard product + MLP 捕获非线性交互（交互效应）
- 当交互 MLP 初始化为零时，退化为纯加法组合，训练初期稳定
- 随着训练推进，交互项逐渐学到必要的非线性修正

**维度选择**：$d_A = d_B = d_c = 32$，嵌入表大小：

$$|\mathcal{A}| \times d_c + |\mathcal{B}| \times d_c + \text{MLP}_{inter} \approx 5 \times 32 + 4 \times 32 + 32 \times 32 = 1,312 \text{ params}$$

对比 one-hot 的 20D（看似更少，但不具泛化性且不可扩展）。

---

## 4. 隐空间参数化方案分析

### 4.1 问题提出

> **讨论要点**："隐空间参数化与组合指令对比学习方案分析"

指令嵌入只是第一步。核心问题是：**如何确保学到的嵌入空间具有良好的结构？**

不加约束地端到端训练，嵌入可能退化为：
- **坍缩**：所有指令映射到同一点（$z_{c_i} \approx z_{c_j} \;\forall i,j$）
- **任意散布**：嵌入在空间中没有语义结构，语义相似的指令可能相距很远
- **维度浪费**：某些维度不被使用，有效维度远低于 $d_c$

### 4.2 隐空间参数化的三种范式

我们讨论了三种参数化范式：

#### 范式 1：确定性嵌入 + 对比正则

$$z_c = \mathbf{E}[c] \in \mathbb{R}^{d_c}$$

直接学习确定性向量，通过对比损失约束结构。

- **优点**：简单、推理无随机性
- **缺点**：无法表达指令内的不确定性（同一指令可能对应多种执行方式）

#### 范式 2：变分嵌入（VAE-style）

$$z_c \sim q_\theta(z | c) = \mathcal{N}(\mu_\theta(c), \sigma^2_\theta(c))$$

每个指令映射到一个**分布**而非点。

- **优点**：自然建模指令的多义性（"walk" 可以是快走或慢走）
- **优点**：KL 正则项防止嵌入坍缩
- **缺点**：采样引入方差，策略训练不稳定
- **缺点**：与 PPO 的 on-policy 特性冲突（每步采样不同的 $z_c$）

#### 范式 3：超球面嵌入（Hyperspherical）

$$z_c = \frac{\mathbf{E}[c]}{\|\mathbf{E}[c]\|_2} \in \mathbb{S}^{d_c - 1}$$

将嵌入归一化到单位超球面。

- **优点**：余弦相似度成为自然的距离度量
- **优点**：防止嵌入范数爆炸或坍缩
- **优点**：与 InfoNCE 对比损失的温度参数 $\tau$ 配合良好
- **缺点**：容量受限——所有嵌入被约束在球面上

### 4.3 讨论最终选择：确定性超球面嵌入

我们最终选择了**范式 3（超球面嵌入）**，理由如下：

1. **PPO 兼容性**：确定性嵌入避免了额外的采样方差，与 on-policy 训练兼容
2. **对比学习天然配合**：InfoNCE 损失中的相似度函数自然选择余弦相似度
3. **结构性保证**：球面上的均匀分布是最大熵分布——对比损失的负样本均匀性假设得到满足
4. **数值稳定性**：归一化防止梯度爆炸/消失

**对"指令内不确定性"的处理**：

我们认为，在 locomotion 场景中，指令的含义是**确定的**（"walk" 就是走），不确定性来自**环境状态**而非指令本身。环境状态的不确定性由 Transformer 的历史编码器处理（POMDP 信念推断），不需要在指令嵌入层引入。

### 4.4 超球面嵌入的数学性质

设 $z_i, z_j \in \mathbb{S}^{d_c - 1}$（单位球面），则：

- **余弦相似度** $s(z_i, z_j) = z_i^\top z_j \in [-1, 1]$
- **距离度量**：弧长 $d(z_i, z_j) = \arccos(z_i^\top z_j)$
- **均匀分布**：$d_c$ 维球面可容纳 $O(d_c)$ 个近正交向量

对于 $d_c = 32$，可容纳约 32 个近正交方向。当 $N > d_c$ 时，部分指令必然共享角度邻域——这提供了一种**自然的聚类效应**：语义相似的指令会被对比损失推到同一角度邻域。

---

## 5. 组合指令对比学习方案

### 5.1 核心思想

**将指令嵌入空间和观测编码空间对齐**：在同一 rollout 中，相同指令下的观测序列应产生与该指令嵌入一致的 latent 表征；不同指令下的观测应产生不同的 latent。

这不同于前期讨论中的"时间对比学习"（相邻时间步的 latent 应相似）——这里是**指令条件对比学习**（condition-contrastive learning）。

### 5.2 正/负样本构造

设当前时间步 $t$，指令为 $c_t$，观测历史编码为 $z_t^{obs} = f_\theta(o_{1:t})$，指令嵌入为 $z^{cmd} = g_\phi(c_t)$。

| 样本类型 | 定义 | 来源 |
|---------|------|------|
| **锚点** | $(z_t^{obs}, z^{cmd}(c_t))$ | 当前 env 的观测编码 + 当前指令嵌入 |
| **正样本** | 同 rollout 中、同指令下的其他时间步的观测编码 | 同 env、同 episode |
| **负样本** | 不同指令下的观测编码 | 其他 env（不同指令），或同 env 的不同 episode |

### 5.3 指令条件 InfoNCE 损失

$$\mathcal{L}_{\text{cmd-CL}} = -\mathbb{E}_{(z^{obs}, z^{cmd+})} \left[ \log \frac{\exp(s(z^{obs}, z^{cmd+}) / \tau)}{\exp(s(z^{obs}, z^{cmd+}) / \tau) + \sum_{k=1}^{K} \exp(s(z^{obs}, z_k^{cmd-}) / \tau)} \right]$$

其中：
- $z^{cmd+}$ 是当前观测对应的正确指令嵌入
- $z_k^{cmd-}$ 是其他指令的嵌入（负样本）
- $s(\cdot, \cdot)$ 是余弦相似度
- $\tau$ 是温度参数

**负样本来源**：在 mini-batch 内，不同 env 的指令通常不同（随机采样），自然提供负样本。如果 batch 中有 $B$ 个 env，则有 $B-1$ 个负样本——这在 $B=4096$ 时非常充足。

### 5.4 组合指令的对比学习：分层结构

当指令是两系列的组合 $(a_i, b_j)$ 时，对比学习需要更精细的结构：

#### 5.4.1 全组合对比

将 $(a_i, b_j)$ 视为一个整体指令，直接应用 §5.3 的 InfoNCE。

- ❌ **问题**：如果 env_1 执行 (walk, carry)，env_2 执行 (walk, wave)，它们被视为完全不同的负样本。但实际上它们共享"walk"，应有一定的相似性。

#### 5.4.2 分层对比（我们讨论的最终方案）

定义**两级对比损失**：

**第一级：系列内对比**（Series-level contrastive）

分别对齐观测编码与各系列的指令嵌入：

$$\mathcal{L}_{A} = -\mathbb{E} \left[ \log \frac{\exp(s(z^{obs}_A, z^{cmd}_A(a_i)) / \tau_A)}{\sum_{a \in \mathcal{A}} \exp(s(z^{obs}_A, z^{cmd}_A(a)) / \tau_A)} \right]$$

$$\mathcal{L}_{B} = -\mathbb{E} \left[ \log \frac{\exp(s(z^{obs}_B, z^{cmd}_B(b_j)) / \tau_B)}{\sum_{b \in \mathcal{B}} \exp(s(z^{obs}_B, z^{cmd}_B(b)) / \tau_B)} \right]$$

其中 $z^{obs}_A, z^{obs}_B$ 是从观测编码中分离出的两个子空间（通过投影头）。

**第二级：组合对比**（Composition-level contrastive）

在组合嵌入空间中做全局对比：

$$\mathcal{L}_{AB} = -\mathbb{E} \left[ \log \frac{\exp(s(z^{obs}, z^{cmd}(a_i, b_j)) / \tau)}{\sum_{(a,b)} \exp(s(z^{obs}, z^{cmd}(a, b)) / \tau)} \right]$$

**总对比损失**：

$$\mathcal{L}_{\text{CL}} = \lambda_A \mathcal{L}_A + \lambda_B \mathcal{L}_B + \lambda_{AB} \mathcal{L}_{AB}$$

#### 5.4.3 分层对比的理论优势

**命题**：分层对比损失诱导嵌入空间具有**子空间分解结构**——存在正交子空间 $V_A, V_B$ 使得：

$$z^{cmd}(a_i, b_j) \approx \text{proj}_{V_A}(z^{cmd}_A(a_i)) + \text{proj}_{V_B}(z^{cmd}_B(b_j))$$

**非形式化论证**：
- $\mathcal{L}_A$ 驱动观测编码在某个子空间上与系列 A 的指令对齐 → 该子空间编码系列 A 的信息
- $\mathcal{L}_B$ 驱动另一个子空间与系列 B 对齐
- 为同时最小化两个损失，两个子空间趋向正交（否则相互干扰）
- $\mathcal{L}_{AB}$ 约束组合的全局一致性

**实际意义**：这种分解结构使得**组合泛化**成为可能——即使 (run, point) 在训练中从未出现，策略也能通过组合 $z_A(\text{run})$ 和 $z_B(\text{point})$ 产生合理的行为，因为"run"的运动特征和"point"的上肢特征在独立的子空间中已经分别学好。

### 5.5 温度参数的选择

| 参数 | 推荐值 | 理由 |
|------|-------|------|
| $\tau_A$ | 0.1 | 系列 A（运动模式）区分度高，需较低温度 |
| $\tau_B$ | 0.1 | 系列 B（上肢动作）同理 |
| $\tau_{AB}$ | 0.07 | 组合级对比需更强的区分信号 |

**温度 schedule**：从 $\tau_{\text{init}} = 0.5$（宽松）逐步退火到目标值（严格），避免训练初期梯度过大。

### 5.6 与时间对比学习的关系

前期讨论中的**时间对比学习**（temporal contrastive）关注的是"相邻时间步的 latent 应相似"：

$$\mathcal{L}_{\text{temporal}} = -\log \frac{\exp(s(z_t, z_{t+1})/\tau_t)}{\sum_j \exp(s(z_t, z_j)/\tau_t)}$$

而本节的**指令条件对比**关注的是"同指令下的 latent 应相似"：

$$\mathcal{L}_{\text{cmd-CL}} = -\log \frac{\exp(s(z^{obs}, z^{cmd+})/\tau_c)}{\sum_k \exp(s(z^{obs}, z_k^{cmd})/\tau_c)}$$

两者可以**并行使用**，分别训练编码器的不同方面：

- 时间对比 → 学习观测的时序动态
- 指令对比 → 学习观测与指令的对齐

在梯度空间中，两者操作在不同的 latent 维度上，干扰有限（见前期讨论的梯度正交性分析）。

---

## 6. 最终模型设计

### 6.1 整体架构

基于所有讨论，我们给出最终的模型架构：

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         指令编码模块                                 │
    │  ┌───────────┐     ┌───────────┐     ┌──────────────────┐          │
    │  │ E_A[a_i]  │     │ E_B[b_j]  │     │  交互 MLP         │          │
    │  │ (系列A嵌入)│     │ (系列B嵌入)│     │  z_A ⊙ z_B → δ  │          │
    │  └─────┬─────┘     └─────┬─────┘     └────────┬─────────┘          │
    │        │                 │                     │                    │
    │        └────────┬────────┘                     │                    │
    │                 │ (+)                           │ (+)                │
    │                 └────────────┬──────────────────┘                   │
    │                              ↓                                      │
    │                     z_cmd ∈ ℝ^{d_c}                                │
    │                     ────────────────                                │
    │                     L2 归一化 → 超球面                               │
    └──────────────────────┬──────────────────────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────────────────────────────┐
    │                      │  观测编码模块                                 │
    │  o_{1:t}             │                                              │
    │  ┌─────────┐         │                                              │
    │  │ 历史拆分  │         │                                              │
    │  │ (T帧,dₒ) │         │                                              │
    │  └────┬────┘         │                                              │
    │       ↓              │                                              │
    │  ┌──────────────┐    │                                              │
    │  │ hist_proj     │    │                                              │
    │  │ + pos_emb     │    │                                              │
    │  └────┬─────────┘    │                                              │
    │       ↓              │                                              │
    │  ┌──────────────┐    │                                              │
    │  │ Transformer   │    │                                              │
    │  │ Encoder       │    │                                              │
    │  │ (因果掩码)     │    │                                              │
    │  └────┬─────────┘    │                                              │
    │       ↓ (last token) │                                              │
    │  ┌──────────────┐    │                                              │
    │  │ FiLM 条件调制  │←──┘ z_cmd 作为条件                                │
    │  │ z = (1+γ)⊙h+β│                                                   │
    │  └────┬─────────┘                                                   │
    │       ↓                                                             │
    │  z_obs ∈ ℝ^{d_model}                                               │
    └───────┬─────────────────────────────────────────────────────────────┘
            │
    ┌───────┼─────────────────────────────────────────────────────────────┐
    │       │         辅助损失头                                           │
    │       ├──→ [序列预测头]   →  ô_{t+1}        →  L_pred (MSE)         │
    │       ├──→ [对比投影头]   →  z_proj          →  L_CL (InfoNCE)      │
    │       │     ├──→ proj_A  →  z^obs_A         →  L_A (系列A对比)      │
    │       │     └──→ proj_B  →  z^obs_B         →  L_B (系列B对比)      │
    │       └──→ [VIB 瓶颈]    →  μ, σ² → z̃     →  L_VIB (KL)          │
    │                              (可选, Phase 3)                        │
    └───────┬─────────────────────────────────────────────────────────────┘
            │
            ↓
    ┌───────────────────┐
    │  MLP Policy Head   │
    │  [512, 256, 128]   │
    │  → 15D actions     │
    └───────┬───────────┘
            ↓
         π(a_t | z_obs)
            ↓
         L_RL (PPO)
```

### 6.2 模块详细参数

#### 6.2.1 指令编码器

```python
class CommandEncoder(nn.Module):
    def __init__(self, n_series_A, n_series_B, d_cmd=32):
        self.embed_A = nn.Embedding(n_series_A, d_cmd)
        self.embed_B = nn.Embedding(n_series_B, d_cmd)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(d_cmd, d_cmd),
            nn.ELU(),
            nn.Linear(d_cmd, d_cmd),
        )
        # 零初始化交互项，训练初期退化为纯加法
        nn.init.zeros_(self.interaction_mlp[-1].weight)
        nn.init.zeros_(self.interaction_mlp[-1].bias)

    def forward(self, cmd_A_idx, cmd_B_idx):
        z_A = self.embed_A(cmd_A_idx)          # (B, d_cmd)
        z_B = self.embed_B(cmd_B_idx)          # (B, d_cmd)
        z_inter = self.interaction_mlp(z_A * z_B)  # Hadamard + MLP
        z_cmd = z_A + z_B + z_inter            # 加法 + 交互
        z_cmd = F.normalize(z_cmd, dim=-1)     # L2 归一化到超球面
        return z_cmd
```

#### 6.2.2 观测编码器（Transformer + FiLM，指令条件化）

```python
class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim, d_model=256, d_cmd=32, ...):
        # Transformer 编码器 (同 TransformerHistoryModel)
        self.hist_proj = nn.Linear(obs_dim, d_model)
        self.pos_emb = nn.Parameter(...)
        self.hist_encoder = nn.TransformerEncoder(...)  # 因果掩码

        # FiLM：用 z_cmd 调制 Transformer 输出
        self.film = FiLMLayer(cond_dim=d_cmd, feat_dim=d_model)
        self.ln_fusion = nn.LayerNorm(d_model)

        # 辅助头
        self.next_obs_head = nn.Sequential(...)     # 序列预测
        self.proj_A = nn.Linear(d_model, d_cmd)     # 系列A对比投影
        self.proj_B = nn.Linear(d_model, d_cmd)     # 系列B对比投影
        self.proj_full = nn.Linear(d_model, d_cmd)  # 全组合对比投影

    def forward(self, history, z_cmd):
        h_emb = self.hist_proj(history) + self.pos_emb
        h_encoded = self.hist_encoder(h_emb, mask=self.causal_mask)
        last_token = h_encoded[:, -1, :]
        z_obs = self.ln_fusion(self.film(last_token, z_cmd))
        return z_obs

    def get_contrastive_projections(self, z_obs):
        return (
            F.normalize(self.proj_A(z_obs), dim=-1),
            F.normalize(self.proj_B(z_obs), dim=-1),
            F.normalize(self.proj_full(z_obs), dim=-1),
        )
```

#### 6.2.3 维度汇总

| 量 | 维度 | 说明 |
|----|------|------|
| 单帧观测 $o_t$ | $d_o$ | 如 54D (ang_vel + gravity + cmd + joints) |
| 历史长度 $T$ | 5 | 5 帧历史窗口 |
| 指令嵌入 $z_{cmd}$ | $d_c = 32$ | 超球面嵌入 |
| Transformer 隐层 $d_{model}$ | 256 | 注意力维度 |
| 观测编码 $z_{obs}$ | 256 | FiLM 输出，送入策略头 |
| 对比投影 $z_{proj}$ | 32 | 与指令嵌入同维度 |
| MLP 策略头 | [512, 256, 128] | 3 层全连接 |
| 动作维度 | 15 | 15 DOF 关节 |

---

## 7. 训练方案

### 7.1 总损失函数

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{PPO}}}_{\text{策略优化}} + \underbrace{\alpha(k) \cdot \mathcal{L}_{\text{pred}}}_{\text{序列预测}} + \underbrace{\gamma(k) \cdot \mathcal{L}_{\text{CL}}}_{\text{指令对比}} + \underbrace{\beta(k) \cdot \mathcal{L}_{\text{VIB}}}_{\text{信息瓶颈 (可选)}}$$

其中对比损失展开为：

$$\mathcal{L}_{\text{CL}} = \lambda_A \mathcal{L}_A + \lambda_B \mathcal{L}_B + \lambda_{AB} \mathcal{L}_{AB}$$

### 7.2 训练阶段调度

| 阶段 | 迭代范围 | 主导目标 | $\alpha$ (预测) | $\gamma$ (对比) | $\beta$ (VIB) | 说明 |
|------|---------|---------|----------------|----------------|---------------|------|
| **Phase 0** | 0 – 1000 | 指令嵌入预热 | 0.3 | **0.5** | 0 | 高对比系数，驱动指令嵌入空间形成结构 |
| **Phase 1** | 1000 – 5000 | 编码器协同 | **0.5** | 0.3 | 0 | 序列预测接管主导，对比维持 |
| **Phase 2** | 5000 – 10000 | 策略精调 | 0.1 | 0.1 | 0 | 辅助信号衰减，PPO 主导 |
| **Phase 3** | 10000+ | 压缩 (可选) | 0.05 | 0.05 | 0.01–0.1 | 如需 VIB，在此阶段引入 |

**权重衰减函数**（线性衰减）：

$$\alpha(k) = \max\left(\alpha_{\text{end}}, \alpha_{\text{start}} - \frac{(\alpha_{\text{start}} - \alpha_{\text{end}}) \cdot k}{k_{\text{decay}}}\right)$$

### 7.3 训练流程伪代码

```python
for iteration in range(max_iterations):
    # ======== Rollout ========
    for step in range(num_steps_per_env):
        # 采样指令组合 (可能在 episode 开头)
        cmd_A, cmd_B = env.sample_commands()
        z_cmd = command_encoder(cmd_A, cmd_B)

        # 编码观测
        z_obs = observation_encoder(history, z_cmd)

        # 策略输出
        actions = policy_head(z_obs)
        obs, rewards, dones = env.step(actions)
        storage.add(obs, actions, rewards, dones, z_cmd, cmd_A, cmd_B)

    # ======== Phase A: 辅助损失 (独立 pass) ========
    # A1: 序列预测
    pred_loss = compute_next_obs_prediction_loss(storage)

    # A2: 指令对比
    cl_loss_A, cl_loss_B, cl_loss_AB = compute_command_contrastive_loss(storage)
    cl_loss = λ_A * cl_loss_A + λ_B * cl_loss_B + λ_AB * cl_loss_AB

    # A3: 合并辅助损失
    aux_loss = α(iter) * pred_loss + γ(iter) * cl_loss
    optimizer.zero_grad()
    aux_loss.backward()
    clip_grad_norm_(encoder_params, max_grad_norm)
    optimizer.step()

    # ======== Phase B: 标准 PPO (mini-batch 循环) ========
    for batch in storage.mini_batch_generator():
        surrogate_loss, value_loss, entropy = ppo_forward(batch)
        ppo_loss = surrogate_loss + c_v * value_loss - c_e * entropy
        optimizer.zero_grad()
        ppo_loss.backward()
        clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
```

### 7.4 指令对比损失的实现细节

```python
def compute_command_contrastive_loss(storage, tau_A=0.1, tau_B=0.1, tau_AB=0.07):
    """在 rollout buffer 内跨 env 构造对比样本。"""
    # 取所有 env 在某一时间步的观测编码和指令
    t = random.randint(0, T-1)  # 随机选一个时间步
    z_obs = encoder(storage.observations[t])  # (B, d_model)
    z_proj_A, z_proj_B, z_proj_full = encoder.get_contrastive_projections(z_obs)

    cmd_A = storage.cmd_A[t]  # (B,) 系列A指令索引
    cmd_B = storage.cmd_B[t]  # (B,) 系列B指令索引

    # 系列A对比: z_proj_A vs all z_cmd_A
    z_cmd_A_all = command_encoder.embed_A.weight  # (N_A, d_cmd)
    z_cmd_A_all = F.normalize(z_cmd_A_all, dim=-1)
    logits_A = z_proj_A @ z_cmd_A_all.T / tau_A   # (B, N_A)
    loss_A = F.cross_entropy(logits_A, cmd_A)

    # 系列B对比: 同理
    z_cmd_B_all = F.normalize(command_encoder.embed_B.weight, dim=-1)
    logits_B = z_proj_B @ z_cmd_B_all.T / tau_B
    loss_B = F.cross_entropy(logits_B, cmd_B)

    # 组合对比: batch 内互为负样本
    z_cmd_combined = command_encoder(cmd_A, cmd_B)  # (B, d_cmd)
    sim_matrix = z_proj_full @ z_cmd_combined.T / tau_AB  # (B, B)
    labels = torch.arange(B, device=device)  # 对角线为正样本
    loss_AB = F.cross_entropy(sim_matrix, labels)

    return loss_A, loss_B, loss_AB
```

### 7.5 指令采样策略

为确保对比学习的负样本多样性，指令采样需满足：

1. **均匀覆盖**：每次 rollout 中，$B$ 个 env 的指令应尽可能覆盖不同组合
2. **部分组合留出**：将 10% 的组合作为验证集，用于评估组合泛化能力
3. **难例挖掘**：后期训练中，增加对比损失高的组合的采样概率

```python
def sample_diverse_commands(B, n_A, n_B, holdout_ratio=0.1):
    """确保 batch 内指令多样性。"""
    # 生成所有可能组合
    all_combos = [(a, b) for a in range(n_A) for b in range(n_B)]

    # 留出部分组合用于泛化验证
    n_holdout = max(1, int(len(all_combos) * holdout_ratio))
    train_combos = all_combos[:-n_holdout]
    test_combos = all_combos[-n_holdout:]

    # 从训练组合中均匀采样
    indices = torch.randint(0, len(train_combos), (B,))
    cmd_A = torch.tensor([train_combos[i][0] for i in indices])
    cmd_B = torch.tensor([train_combos[i][1] for i in indices])
    return cmd_A, cmd_B
```

---

## 8. 理论分析与收敛性讨论

### 8.1 多目标优化的梯度分析

总梯度对编码器参数 $\theta$ 的分解：

$$\nabla_\theta \mathcal{L}_{\text{total}} = \underbrace{\nabla_\theta \mathcal{L}_{\text{PPO}}}_{\sim O(10^{-4})} + \alpha \underbrace{\nabla_\theta \mathcal{L}_{\text{pred}}}_{\sim O(10^{-2})} + \gamma \left(\lambda_A \underbrace{\nabla_\theta \mathcal{L}_A}_{\text{系列A结构}} + \lambda_B \underbrace{\nabla_\theta \mathcal{L}_B}_{\text{系列B结构}} + \lambda_{AB} \underbrace{\nabla_\theta \mathcal{L}_{AB}}_{\text{组合一致性}}\right)$$

**关键观察**：

1. **序列预测梯度**：驱动编码器学习"什么历史信息对预测未来有用"
2. **系列A/B对比梯度**：驱动编码器在特定子空间上与各系列指令对齐
3. **组合对比梯度**：驱动编码器在全空间上与组合指令一致
4. **PPO 梯度**：最弱，但提供最终的"什么对决策有用"的信号

四类梯度操作在 latent space 的不同方面，冲突有限。衰减 schedule 确保早期由辅助目标主导（学结构），后期由 PPO 主导（学策略）。

### 8.2 组合泛化的理论保证

**命题**：若因式分解嵌入学到了正交子空间分解 $z_{cmd}(a, b) \approx \pi_A(z_A(a)) + \pi_B(z_B(b))$，且策略网络 $\pi_\psi$ 在各子空间上独立可分（即存在 $\pi_\psi(z) = f(\pi_A(z), \pi_B(z))$），则对未见过的组合 $(a_{\text{new}}, b_{\text{new}})$：

$$\mathbb{E}[\|\pi_\psi(z_{cmd}(a_{\text{new}}, b_{\text{new}})) - \pi^*(a_{\text{new}}, b_{\text{new}})\|] \leq \epsilon_A + \epsilon_B + \epsilon_{\text{inter}}$$

其中 $\epsilon_A, \epsilon_B$ 分别是各系列的学习误差，$\epsilon_{\text{inter}}$ 是交互项的近似误差。

**直觉**：如果"run"和"point"分别在各自的子空间中学好了，那么它们的组合只需要交互项提供修正——而交互项的零初始化确保了修正量有界。

### 8.3 对比学习的坍缩风险分析

在指令条件对比中，坍缩的风险低于通用对比学习，因为：

1. **指令标签是离散的**：正/负样本的区分是精确的（基于指令索引），不存在歧义
2. **负样本充足**：batch 中有 $B$ 个 env，通常 $B \gg |\mathcal{C}|$，每个指令都有大量负样本
3. **超球面归一化**：防止范数坍缩（所有向量长度为 1）

唯一需要关注的是**维度坍缩**（dimensional collapse）——所有嵌入虽然范数为 1，但集中在球面的一个小区域。这可以通过 VICReg 风格的方差正则来防止：

$$\mathcal{L}_{\text{var}} = \frac{1}{d_c} \sum_{j=1}^{d_c} \max(0, 1 - \sqrt{\text{Var}(z_{j}) + \epsilon})$$

### 8.4 计算开销分析

| 组件 | 每次 update 的额外开销 | 相对于基线 PPO |
|------|---------------------|--------------|
| 指令嵌入查表 | $O(B \cdot d_c)$ | 可忽略 |
| FiLM 调制 | $O(B \cdot d_{model})$ | ≈ 基线的 1.01× |
| 序列预测 loss (独立 pass) | $O(T \cdot B \cdot d_{model}^2)$ | ≈ 基线的 1.3× |
| 指令对比 loss | $O(B \cdot (|\mathcal{A}| + |\mathcal{B}|) \cdot d_c)$ | ≈ 基线的 1.05× |
| **总计** | — | **≈ 基线的 1.4×** |

在 $B=4096, T=24, d_{model}=256, d_c=32$ 的典型设置下，总计算开销增加约 40%——完全可接受。

---

## 9. 与仓库现有实现的关联

本节简要说明上述理论方案与 `unitree_rl_lab` 仓库现有实现的映射关系。

### 9.1 已实现模块

| 理论模块 | 仓库文件 | 状态 | 说明 |
|---------|---------|------|------|
| 因果 Transformer 编码器 | `rsl_rl_transformer_model.py` → `TransformerHistoryModel` | ✅ 已实现 | 因果掩码、FiLM 调制均已到位 |
| 序列预测损失 | `transformer_ppo.py` → `TransformerPPO._compute_aux_prediction_loss()` | ✅ 已实现 | 独立 pass + 线性衰减 |
| 速度指令处理 | `velocity_command.py` → `UniformLevelVelocityCommand` | ✅ 已实现 | 当前为 3D 连续指令 |
| FiLM 条件调制 | `rsl_rl_transformer_model.py` → `FiLMLayer` | ✅ 已实现 | 当前以 aux obs 为条件 |

### 9.2 待实现模块

| 理论模块 | 说明 | 改动范围 |
|---------|------|---------|
| 指令编码器 (`CommandEncoder`) | 因式分解嵌入 + 超球面归一化 | 新增文件 |
| 指令对比损失 | 分层 InfoNCE | `transformer_ppo.py` 扩展 |
| 指令采样器 | 离散/组合指令采样 + 覆盖率保证 | `velocity_command.py` 扩展 |
| VIB 信息瓶颈 | 变分瓶颈层 | `rsl_rl_transformer_model.py` 扩展 |
| FiLM 条件源替换 | 从 aux obs → z_cmd | `rsl_rl_transformer_model.py` 修改 |

### 9.3 从当前实现到目标方案的迁移路径

```
Phase 0: 当前状态
  └── TransformerHistoryModel + next-obs prediction
      └── FiLM 以 aux_obs 为条件

Phase 1: 最小改动
  └── 新增 CommandEncoder
  └── FiLM 条件源改为 z_cmd (保留 aux_obs 作为备选)
  └── 观测中的 cmd 维度保持不变（向后兼容）

Phase 2: 引入对比学习
  └── 新增 contrastive projection heads
  └── TransformerPPO.update() 中加入 L_CL 计算
  └── 指令采样器增加多样性机制

Phase 3: 完整方案
  └── 两系列指令支持
  └── 分层对比损失
  └── 组合泛化评估
  └── (可选) VIB 瓶颈
```

---

## 10. 参考文献

1. van den Oord, A. et al. (2018). *Representation learning with contrastive predictive coding.* (CPC/InfoNCE)
2. Grigsby, J. et al. (2023). *AMAGO: Scalable in-context reinforcement learning for adaptive agents.* ICLR 2024.
3. Perez, E. et al. (2018). *FiLM: Visual reasoning with a general conditioning layer.* AAAI.
4. Bardes, A. et al. (2022). *VICReg: Variance-invariance-covariance regularization.* ICLR.
5. Alemi, A. A. et al. (2017). *Deep variational information bottleneck.* ICLR.
6. Igl, M. et al. (2019). *Generalization in RL with selective noise injection and information bottleneck.* NeurIPS (IBAC).
7. Mikolov, T. et al. (2013). *Distributed representations of words and phrases and their compositionality.* NeurIPS. (Word2Vec 加法组合性)
8. Andreas, J. (2019). *Measuring compositionality in representation learning.* ICLR. (组合泛化评估)
9. Lake, B. & Baroni, M. (2018). *Generalization without systematicity.* ICML. (组合泛化基准)
10. Wang, T. & Isola, P. (2020). *Understanding contrastive representation learning through alignment and uniformity.* ICML. (超球面对比分析)
11. Chen, T. et al. (2020). *A simple framework for contrastive learning of visual representations.* ICML. (SimCLR, 温度参数分析)
12. Radford, A. et al. (2021). *Learning transferable visual models from natural language supervision.* ICML. (CLIP, 多模态对比对齐)
