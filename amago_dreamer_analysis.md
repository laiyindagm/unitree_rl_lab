# AMAGO 与 Dreamer：Transformer 作为 RL 历史编码器的两种范式

本文档分析两种将 Transformer/序列模型用于连续控制 RL 的代表性方法，以及它们与本项目 `TransformerHistoryModel` + 辅助预测损失方案的关系。

---

## 1. AMAGO：Shared Transformer Actor-Critic

**论文**：*AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents* (Grigsby et al., ICLR 2024 Spotlight)

### 1.1 核心架构

```
输入序列 (完整 episode):
  τ_t = (o_t, a_{t-1}, r_{t-1})  ← 观测 + 上一步动作 + 上一步奖励

  τ₁, τ₂, ..., τₜ
    ↓ (各自 embed + 相加 + pos_encoding)
  ┌────────────────────────────┐
  │  Shared Transformer        │  ← 因果掩码, 处理完整 episode 长序列
  │  (多层 self-attention)      │
  └──────┬──────────┬──────────┘
         ↓          ↓
    Actor Head   Critic Head     ← 轻量 MLP
    π(a|h_t)     V(h_t)
```

### 1.2 关键设计

#### (a) Token 构造

每个时间步打包为一个 token：

```python
token_t = obs_encoder(o_t) + action_encoder(a_{t-1}) + reward_encoder(r_{t-1})
token_t = token_t + positional_encoding(t)
```

三者**相加**（非拼接），保持 token 维度统一。观测/动作/奖励各自有独立的线性投影层。

#### (b) 共享骨干 + 梯度隔离

Actor 和 Critic 共享同一个 Transformer 编码器，用两个轻量 MLP Head 分别输出策略和值函数。通过 stop-gradient 防止 actor loss 干扰 critic（反之亦然），避免共享参数时的梯度冲突。

**为什么共享**：
- 减少 ~50% 参数和计算
- 两个任务都需要从历史中提取有用信息，避免重复学习

#### (c) Off-Policy 训练

AMAGO 使用 off-policy 算法（SAC 风格），从大规模 Replay Buffer 中采样 **完整 episode**（非单步）：

- Transformer 处理完整 episode 序列（数百~数千步）
- On-policy 方法（如 PPO）的 mini-batch shuffle 会破坏序列结构，因此 AMAGO 选择 off-policy
- 支持 Hindsight Goal Relabeling（稀疏奖励场景关键）

#### (d) In-Context Learning

AMAGO 的 meta-learning 完全隐式：

- 训练时在多个不同任务/环境上训练
- Transformer 学会 "如何从历史中推断当前任务"
- 测试时面对新任务，前几步探索 → attention 自动聚焦有信息量的历史 → 后续步骤自动调整策略

本质与 LLM 的 in-context learning 相同。

#### (e) 辅助序列建模损失

AMAGO 不仅靠 RL loss 训练 Transformer，还使用辅助的序列预测损失：

```python
h_t = transformer_output[t]
pred_next_obs = obs_prediction_head(h_t)
aux_loss = MSE(pred_next_obs, obs_encoder(o_{t+1}))

total_loss = actor_loss + critic_loss + λ * aux_loss
```

这与本项目 `transformer_aux_design.md` Phase 2 设计的辅助预测损失**完全一致**。

### 1.3 与本项目方案对比

| 维度 | 本项目 TransformerHistoryModel | AMAGO |
|------|-------------------------------|-------|
| RL 算法 | PPO (on-policy) | Off-policy (SAC 风格) |
| 序列长度 | 5 步窗口 | 完整 episode (数百~数千步) |
| Transformer 输入 | 仅观测 | 观测 + 动作 + 奖励 |
| Actor/Critic | 分离 | 共享 Transformer 骨干 |
| 辅助损失 | Phase 2 next-obs prediction | 内置 next-obs prediction |
| Meta-learning | 无 | 通过 in-context learning 自动适应 |

---

## 2. Dreamer v3：World Model + Latent Imagination

**论文**：*Mastering Diverse Domains through World Models* (Hafner et al., 2023)

### 2.1 核心思想

> 先学一个世界模型（World Model），然后在想象中（latent space）练习策略，而非直接在真实环境中试错。

### 2.2 RSSM（Recurrent State-Space Model）

RSSM 是 Dreamer 的核心，将 latent state 分为确定性和随机性两部分：

```
完整 latent state = (h_t, z_t)
                      ↑       ↑
                 确定性部分   随机性部分
                 (GRU 隐状态)  (采样的离散变量, DreamerV3 用 32×32)
```

#### 四个核心网络

| 网络 | 公式 | 作用 |
|------|------|------|
| Sequence Model (GRU) | `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` | 确定性记忆，捕获长期时序依赖 |
| Prior (先验) | `z_t ~ p(z_t \| h_t)` | 不看观测就预测 latent state（想象阶段使用） |
| Posterior (后验) | `z_t ~ q(z_t \| h_t, x_t)` | 看了观测后修正 latent state（训练阶段使用） |
| Encoder | `x_t = enc(o_t)` | 将原始观测压缩为 embedding |

#### 一步状态转移

```python
def rssm_step(h_prev, z_prev, a_prev, o_t):
    # 1. 确定性状态更新
    h_t = GRU(h_prev, concat(z_prev, a_prev))

    # 2. 先验分布（不看观测）
    prior = p(z_t | h_t)

    # 3. 编码观测 → 后验分布
    x_t = encoder(o_t)
    posterior = q(z_t | h_t, x_t)

    # 4. 训练时从后验采样，想象时从先验采样
    z_t = posterior.sample()  # 训练
    # z_t = prior.sample()   # 想象

    return h_t, z_t
```

### 2.3 训练过程（三阶段循环）

#### Phase 1: 收集真实经验

```python
for t in range(episode_length):
    action = actor(h_t, z_t)
    o_{t+1}, r_t, done = env.step(action)
    replay_buffer.add(o_t, a_t, r_t)
```

#### Phase 2: 训练 World Model

从 replay buffer 采样序列，运行 RSSM forward pass，计算三个损失：

```python
L_reconstruction = -log p(o_t | h_t, z_t)   # 重建观测
L_reward         = -log p(r_t | h_t, z_t)   # 预测奖励
L_KL             = KL(posterior || prior)     # 先验/后验对齐

L_world_model = L_reconstruction + L_reward + β * L_KL
```

**KL 损失的意义**：逼迫 Prior（不看观测就预测）和 Posterior（看了观测再预测）尽可能一致。训练充分后，Prior 即可准确预测 latent state → 想象阶段可以只用 Prior。

#### Phase 3: 在想象中训练 Actor-Critic

**Actor 和 Critic 完全不接触真实环境，只在 "梦境" 中训练**：

```python
# 从真实 latent state 出发，想象 H 步
h, z = real_latent_state
for τ in range(H):
    a = actor(h, z)                    # Actor 选动作
    h = GRU(h, concat(z, a))           # World Model 预测下一状态
    z = prior(z | h).sample()          # 仅用 Prior（无需真实观测）
    r = reward_head(h, z)              # 预测奖励
    imagined_trajectory.append((h, z, a, r))

# 用想象轨迹的 λ-return 训练 Critic
V_target[τ] = r[τ] + γ * ((1-λ) * V(h[τ+1], z[τ+1]) + λ * V_target[τ+1])
critic_loss = Σ (V(h_τ, z_τ) - V_target[τ])²

# Actor 最大化想象回报
actor_loss = -Σ V_target[τ]
```

### 2.4 核心优势

| 优势 | 解释 |
|------|------|
| 极高样本效率 | 每一步真实交互可生成大量想象轨迹训练策略，比无模型方法高 10-100x |
| 长期信用分配 | 想象 rollout 跨越很多步，Actor-Critic 自然处理长期依赖 |
| 解耦优化 | World Model 训练和 Policy 训练各自有清晰的优化目标 |
| 通用性 | DreamerV3 在 150+ 个不同任务上无需调参即可工作 |

---

## 3. 与本项目方案的关系

本项目 `transformer_aux_design.md` 中设计的辅助预测损失，本质是 Dreamer World Model 训练的简化形式：

| Dreamer 完整版 | 本项目简化版 |
|---------------|------------|
| Prior + Posterior + KL 对齐 | 单一 Transformer + next-obs MSE |
| 在 latent space 想象 rollout 训练 Actor | 在真实环境 rollout 用 PPO 训练 Actor |
| 独立的 World Model 训练阶段 | 辅助损失独立 pass（不混入 PPO mini-batch） |
| GRU 确定性 + 离散随机性 latent | Transformer 输出 + 可选 VIB 瓶颈 |

核心共识：**编码器（Transformer/RSSM）不能仅靠 RL loss 训练，需要显式的序列预测目标。**

### 可借鉴的改进方向

1. **将动作和奖励也作为 token 输入 Transformer**（AMAGO 风格）——让 Transformer 学习因果关系
2. **VIB 信息瓶颈**（Phase 3 已设计）——对应 Dreamer 的 KL 正则
3. **GTrXL-style memory**——突破 5 步窗口限制，对应 Dreamer 的 GRU 跨步记忆

---

## 参考文献

- Grigsby et al., *AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents*, ICLR 2024. [arXiv:2310.09971](https://arxiv.org/abs/2310.09971)
- Hafner et al., *Mastering Diverse Domains through World Models*, 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (DreamerV1), 2020. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Parisotto et al., *Stabilizing Transformers for Reinforcement Learning* (GTrXL), 2020. [arXiv:1910.06764](https://arxiv.org/abs/1910.06764)
