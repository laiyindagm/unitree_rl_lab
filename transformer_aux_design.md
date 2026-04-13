# Transformer + Auxiliary Prediction Loss 设计文档

## 1. 问题背景

### 1.1 初始方案

 `TransformerHistoryModel` 中，我们尝试将 Transformer 引入 PPO 的训练流程，
#
 MLP 直接处理扁平化的历史观测。初始架构：

```
flat_obs (270D = 54D × 5 frames)
  → 切分为 history (5, 54) + aux (54)
  → hist_proj(54 → 256) + learnable pos_emb
  → TransformerEncoder (2层, 4头)
  → 取最后 token h_encoded[:, -1, :]
  → cross-attention(query=last_token, key/value=aux_proj(aux))
  → LayerNorm → fused latent (256D)
  → MLP [512, 256, 128] → 15D actions
```

### 1.2 实验结果

**效果与纯 MLP 持平，甚至更差。** Transformer 没有学到有意义的时间表征。

### 1.3 根因分析

| # | 问题 | 分析 |
|---|------|------|
| 1 | **无直接训练信号** | PPO 的 surrogate loss 经过 MLP → distribution → log_prob 多层传播后，回传到 Transformer 注意力权重的梯度极弱且间接。与 LLM 中每个 token 都有明确的 next-token prediction loss 直接训练注意力权重形成鲜明对比 |
| 2 | **Cross-attention 退化** | `aux` 只是单个 token（54D → 256D），cross-attention 的 key/value 序列长度为 1，softmax 输出恒为 1.0，退化为 `context = V`，本质是一个带 normalization 的线性 gating，没有"注意力"可言 |
| 3 | **缺少因果掩码** | `nn.TransformerEncoder` 默认全连接注意力，时间步 t 可以看到 t+1 的信息，破坏了时间因果性，使得模型可以"作弊"而不学习真正的时序动态 |
| 4 | **history 与 aux 高度重叠** | `aux_start_idx=216, aux_obs_dim=54` 实际上就是 history 的最后一帧（索引 216 = 4×54），cross-attention 本质是一种冗余的 self-attention |

### 1.4 核心洞察

> Transformer 在自回归场景（LLM, Decision Transformer）表现最强，因为有**显式的序列预测目标**。
> 在 RL 的 POMDP 信念状态估计中，agent 需要从历史观测 $o_{1:t}$ 推断 belief state $b_t = P(s_t | o_{1:t})$。
> 关键不是知道这个分布是什么，而是给 Transformer **一个代理目标**（proxy objective）来逼迫它学习有信息量的压缩。

mkdir -p ~/.ssh
- AMAGO (Grigsby et al., 2023) — Transformer 作为 RL 历史编码器，靠 world-model 风格辅助损失训练
- IBAC (Igl et al., 2019) — 信息瓶颈在 RL 中的应用
- Dreamer v3 — latent dynamics model

---

## 2. 改进方案

mkdir - ~/.ssh 

### Phase 1: 架构修复

#### 1a. 因果掩码

 `TransformerEncoder` 中加入上三角因果掩码：

```python
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# 结果: step t 只能 attend to steps 0..t
# Step 0 sees: [0]
# Step 4 sees: [0, 1, 2, 3, 4]
```

**为什么重要**：如果没有因果掩码，step 0 的表征可以看到 step 4 的未来观测，
Transformer 可以简单地在各步之间传递信息而不需要学习时序压缩。加上因果掩码后，
.dockerignore .flake8 .git .gitattributes .gitignore .pre-commit-config.yaml .venv .vscode 29dof_rotation_transfer_guide.md LICENCE README.md deploy doc docker ideas.md logs outputs package-lock.json package.json pyproject.toml scripts source unitree_rl_lab.code-workspace unitree_rl_lab.sh unitree_ros  step 的表征只能基于过去和当前的信息，迫使模型真正学习时间依赖。

#### 1b. FiLM 替换 cross-attention

**Feature-wise Linear Modulation (FiLM)**:

$$z = (1 + \gamma) \odot h_T + \beta$$

-------- $\gamma, \beta = \text{Linear}(\text{aux\_proj}(\text{aux}))$。

**为什么 FiLM 优于 cross-attention**：
- Cross-attention 在 KV 序列长度为 1 时退化为 gating，浪费参数
- FiLM 语义明确：aux 信息**调制**（而非融合）history 表征
- FiLM 保持 d_model 维度不变，不影响下游 MLP
- 参数更少，更容易训练

### Phase 2: 辅助预测损失（核心改进）

 PPO 更新之前，执行一次**独立的辅助前向-反向传播**：

```
 rollout 中的每一步 t ∈ [0, T-2]:
  latent_t = TransformerEncoder(obs_t)     # 编码当前历史
  pred_t   = next_obs_head(latent_t)       # 预测下一帧
  target_t = obs_{t+1} 的最后一帧          # 实际下一帧
  
aux_loss = mean(MSE(pred_t, target_t))     # 辅助损失
```

**为什么选 next-obs prediction 而非其他**：
- **vs reward prediction (1D)**：信号维度太低，梯度信息不足
- **vs inverse dynamics (15D)**：需要额外的动作对齐，复杂度高
- **next-obs (54D)**：维度高、信息密度大，直接驱动 Transformer 学习"什么历史信息对预测未来有用"

**为什么独立 pass 而非混入 PPO mini-batch**：
- PPO 的 mini-batch generator 会 shuffle indices，无法获取时间相邻的 obs 对
- 独立 pass 遍历 storage 的时间维度，精确获取 (obs_t, obs_{t+1}) 对
- 两个目标的梯度分离，互不干扰

**线性衰减 schedule**：
```python
aux_loss_schedule = [start=0.5, end=0.05, decay_start=0, decay_steps=5000]
```
- 前期（iter 0-5000）：辅助信号强，驱动 Transformer 学习时间编码
- 后期（iter 5000+）：辅助信号弱，让策略优化主导训练

### Phase 3: 可选 VIB 瓶颈（未实施）

 Phase 2 效果不足，可在 Transformer 输出后添加 Variational Information Bottleneck：

$$z_t \rightarrow \mu, \log\sigma^2 \rightarrow z \sim \mathcal{N}(\mu, \sigma^2)$$

KL 正则 $\beta \cdot D_{KL}(q(z|o_{1:t}) \| \mathcal{N}(0, I))$，压缩 256D → 64D，
mkdir -p ~/.ssh/authorized_keys 

### 设计决策：Critic 保持 MLP

- Critic 只需估算 V(s)，不需要做信念推断
- Critic 有额外的特权信息（base_lin_vel，57D/frame vs 54D/frame）
- 减少 ~50% 模型参数和计算开销

---

## 3. 实现细节

### 3.1 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `source/.../utils/rsl_rl_transformer_model.py` | **重写** | 因果掩码、FiLM、predict_next_obs()、导出类更新 |
| `source/.../utils/transformer_ppo.py` | **新建** | TransformerPPO 类，辅助损失 + 线性衰减 |
| `source/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | **追加** | 新增 3 个 config class |
| `source/.../tasks/locomotion/robots/g1/15dof_rot/__init__.py` | **追加** | 注册 gym env |

### 3.2 TransformerHistoryModel 改动

```python
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: z = (1+gamma) * h + beta."""
    def __init__(self, cond_dim, feat_dim):
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, h, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class TransformerHistoryModel(MLPModel):
    def __init__(self, ..., enable_aux_loss=False):
        # --- 因果掩码 ---
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        # --- FiLM 替换 cross-attention ---
        self.film = FiLMLayer(cond_dim=d_model, feat_dim=d_model)
        # 移除: self.cross_attn

        # --- 辅助预测头 ---
        if enable_aux_loss:
            self.next_obs_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ELU(),
                nn.Linear(d_model, history_obs_dim),
            )

    def _encode_history_and_aux(self, history, aux):
        h_emb = self.hist_proj(history) + self.pos_emb
        h_encoded = self.hist_encoder(h_emb, mask=self.causal_mask)  # 因果!
        last_token = h_encoded[:, -1, :]
        aux_emb = self.aux_proj(aux)
        fused = self.ln_fusion(self.film(last_token, aux_emb))       # FiLM!
        return fused

    def predict_next_obs(self, obs):
        """训练时调用，返回 (latent, predicted_next_frame)"""
        flat_obs = self._flatten_and_normalize(obs)
        history, aux = self._split_from_flat(flat_obs)
        latent = self._encode_history_and_aux(history, aux)
        return latent, self.next_obs_head(latent)
```

### 3.3 TransformerPPO.update() 流程

```
update():
  ┌─────────────────────────────────────────────┐
  │  Phase A: 辅助预测 pass  (if enable_aux_loss) │
  │  for t in range(T-1):                       │
  │    _, pred = actor.predict_next_obs(obs[t])  │
  │    target = normalize(obs[t+1])[:, -54:]     │
  │    aux_loss += MSE(pred, target)             │
  │  aux_loss /= (T-1)                          │
  │  optimizer.zero_grad()                       │
  │  (aux_coef * aux_loss).backward()            │
  │  clip_grad_norm + optimizer.step()           │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │  Phase B: 标准 PPO mini-batch 循环           │
  │  for batch in generator:                     │
  │    surrogate_loss + value_loss - entropy      │
  │    (unchanged from base PPO)                  │
  └─────────────────────────────────────────────┘
```

### 3.4 配置类

```python
@configclass
class G115DofTransformerAuxPPORunnerCfg(BasePPORunnerCfg):
    clip_actions = 100.0

    actor = RslRlTransformerAuxModelCfg(
        hidden_dims=[512, 256, 128],
        history_len=5, history_obs_dim=54,
        aux_start_idx=216, aux_obs_dim=54,
        d_model=256, n_heads=4,
        enable_aux_loss=True,
    )
    critic = RslRlMLPModelCfg(          # MLP, 不是 Transformer
        hidden_dims=[512, 256, 128],
    )
    algorithm = RslRlTransformerAuxPpoAlgorithmCfg(
        class_name="...transformer_ppo:TransformerPPO",
        aux_loss_coef=0.5,
        aux_loss_schedule=[0.5, 0.05, 0, 5000],
    )
```

### 3.5 环境注册

```python
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V5c-TransformerAux",
    kwargs={
        "env_cfg_entry_point": "...velocity_env_cfg_rot_v5c:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:G115DofTransformerAuxPPORunnerCfg",
    },
)
```

---

## 4. 训练与验证

### 4.1 训练命令

```bash
# TransformerAux (本方案)
python scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V5c-TransformerAux \
    --max_iterations 15000

# Baseline MLP (对比)
python scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V5c \
    --max_iterations 15000
```

### 4.2 TensorBoard 监控指标

| 指标 | 含义 | 期望行为 |
|------|------|----------|
| `aux_pred` | 辅助预测 MSE loss | 应持续下降，表明 Transformer 在学习时序编码 |
| `aux_coef` | 当前辅助系数 | 从 0.5 线性衰减到 0.05 |
| `surrogate` | PPO 策略 loss | 应在辅助损失稳定后开始下降 |
| `value` | 值函数 loss | 正常收敛 |
| `mean_reward` | 平均回报 | 应优于纯 MLP baseline |

### 4.3 消融实验建议

| 实验 | 配置 | 验证什么 |
|------|------|----------|
| A | MLP only (BasePPORunnerV3Cfg) | Baseline |
| B | Transformer + 因果掩码 + FiLM (无辅助损失) | Phase 1 单独效果 |
| C | Transformer + 因果掩码 + FiLM + 辅助损失 | 完整方案 |
| D | 不同 aux_loss_schedule | 衰减速度的影响 |

"Done: transformer_ppo.py rewritten" B，可以设置 `enable_aux_loss=False`（或 `aux_loss_coef=0`）。

---

## 5. 后续优化方向

### 5.1 VIB 信息瓶颈 (Phase 3)

 Phase 2 效果不足，可在 Transformer 输出和 MLP 之间加 VAE-style 瓶颈：

```python
# 在 TransformerHistoryModel 中:
self.mu_head = nn.Linear(d_model, bottleneck_dim)      # 256 → 64
self.logvar_head = nn.Linear(d_model, bottleneck_dim)

# 训练时 reparameterize:
mu = self.mu_head(latent)
logvar = self.logvar_head(latent)
z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

# 推理时直接用 mu
```

**注意事项**：
- beta 从 0.001 开始，逐步增大，避免 posterior collapse
- MLP 输入维度需从 256 改为 64

### 5.2 辅助目标多样化

.dockerignore .flake8 .git .gitattributes .gitignore .pre-commit-config.yaml .venv .vscode 29dof_rotation_transfer_guide.md LICENCE README.md deploy doc docker ideas.md logs outputs package-lock.json package.json pyproject.toml scripts source unitree_rl_lab.code-workspace unitree_ros  unitree_rl_lab.sh next-obs prediction，可以叠加：
- **Reward prediction**: `reward_head(latent) → predicted_reward`
- **Inverse dynamics**: `inv_head(latent_t, obs_{t+1}) → predicted_action_t`
- **Contrastive loss**: 相邻步的 latent 应相似，远距步的应不同

### 5.3 Transformer 结构改进

- **GTrXL-style memory segment**: 让 Transformer 维护跨 rollout 的隐藏记忆，突破 5 步窗口限制
- **相对位置编码 (RoPE/ALiBi)**: 替换当前的绝对位置嵌入，更好地泛化到不同序列长度
- **Perceiver-style latent tokens**: 引入少量可学习 latent tokens 做 cross-attention，压缩序列信息

### 5.4 aux 观测选择

 aux 就是 history 的最后一帧（完全重叠）。如果有额外传感器信息
mkdir -p  height_scan、contact forces），将其作为 aux 传入 FiLM 会带来更大的信息增益。/root/.

---

## 6. 维度速查表 (15-DOF Rotation Task)

| 量 | 值 | 说明 |
|----|----|------|
| 单帧 policy obs | 54D | 3(ang_vel) + 3(gravity) + 3(cmd) + 15(joint_pos) + 15(joint_vel) + 15(action) |
| 单帧 critic obs | 57D | 54D + 3(base_lin_vel) |
| history_len | 5 | 5 帧历史 |
| policy flat obs | 270D | 54 × 5 |
| critic flat obs | 285D | 57 × 5 |
| history_start_idx | 0 | history 从 obs 开头开始 |
| aux_start_idx | 216 | = 4 × 54 (最后一帧 = 当前帧) |
| d_model | 256 | Transformer 隐藏维度 |
| n_heads | 4 | 多头注意力头数 |
| encoder_num_layers | 2 | Transformer 层数 |
| MLP hidden_dims | [512, 256, 128] | 策略/值网络 |
| action_dim | 15 | 15 个训练关节 |
