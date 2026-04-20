# Contrastive Latent Policy：实现设计文档

> 基于 ideas.md 中的理论推导，本文档给出可直接映射到 PyTorch 代码的实现规格。
> 编码器保留 **1D Causal TCN** 和 **Transformer** 两种选项。

---

## 1. 架构总览

> **核心设计决策：缓存方案**
>
> Rollout 阶段编码器产出的 `z_cat`、`e_c`、`â` 被缓存到 RolloutStorage 中。
> Phase B (PPO) 更新时直接使用缓存值构建 Policy MLP 输入，**不重新运行编码器**。
> 这避免了 Phase A 更新编码器后 z 与 rollout 期间动作的不一致问题（收敛风险 R1/R7）。

```
 ╔══════════════════════ Rollout 阶段 ════════════════════════╗
 ║                                                            ║
 ║    ┌────────────────────────────┐                          ║
 ║    │  Flat Actor Obs  (T×d_step)│                          ║
 ║    └──────────┬─────────────────┘                          ║
 ║               │ split                                      ║
 ║   ┌───────────┼───────────────────┐                        ║
 ║   ▼           ▼                   ▼                        ║
 ║ history    cmd_current        o_current                    ║
 ║ [B,T,d_enc] [B,d_cmd=3]      [B,d_enc]                    ║
 ║   │           │                   │                        ║
 ║   ▼           │                   │                        ║
 ║ Encoder       │                   │                        ║
 ║ (TCN/Trans)   │                   │                        ║
 ║   │ h_enc     │                   │                        ║
 ║   ▼           │                   │                        ║
 ║ Sphere(×3)    │                   │                        ║
 ║   │           │                   │                        ║
 ║  z^x,z^y,z^w  │                   │                        ║
 ║   │           │                   │                        ║
 ║   └──┬────────┘                   │                        ║
 ║      │ z_cat   cmd                │                        ║
 ║      ▼         ▼                  │                        ║
 ║   FiLM Generator                  │                        ║
 ║      │ â, e_c                     │                        ║
 ║      │                            │                        ║
 ║  ┌───┴───────────┐                │                        ║
 ║  │ 缓存到 Storage │                │                        ║
 ║  │ z_cat, e_c, â │                │                        ║
 ║  └───┬───────────┘                │                        ║
 ║      │ .detach()                  │                        ║
 ║      ▼                            ▼                        ║
 ║   Policy MLP π_θ                                           ║
 ║   input=[o_cur; e_c; z_cat; â]  (all detached)            ║
 ║      │                                                     ║
 ║      ▼  a_t → env.step()                                  ║
 ╚════════════════════════════════════════════════════════════╝

 ╔══════════════════════ Update 阶段 ═════════════════════════╗
 ║                                                            ║
 ║  Phase A (表征学习):                                        ║
 ║    从 batch 取 flat_obs → 重新编码 → L_NCE + L_gen         ║
 ║    更新: encoder, sphere_proj, contrast_proj,              ║
 ║          cmd_embed, generator                               ║
 ║                                                            ║
 ║  Phase B (策略优化):                                        ║
 ║    从 batch 取缓存的 z_cat, e_c, â + o_current            ║
 ║    拼接 → Policy MLP → log_prob → PPO loss                 ║
 ║    更新: Policy MLP + distribution                          ║
 ║    ⚠️ 不重新编码，使用 rollout 时的缓存值                   ║
 ║                                                            ║
 ╚════════════════════════════════════════════════════════════╝
```

## 2. 观测预处理

### 2.1 Flat 观测布局

以 G1 15DOF 为例（可通过配置泛化）：

```
每帧 d_step 维，包含 6 个 term：

 idx:   0     3     6     9          24         39       54
        ├─────┼─────┼─────┼──────────┼──────────┼────────┤
 term:  ang   grav  cmd   joint_pos  joint_vel  last_act
 dim:    3     3     3      15         15         15
                     ▲
                     └─ velocity_commands = [vx, vy, vw]

history_length = T = 5
flat actor obs = T × d_step
```

### 2.2 拆分策略

设 flat obs 维度为 `T * d_step`。

| 名称 | 下标范围 | 形状 | 用途 |
|------|---------|------|------|
| `history_no_cmd` | 每帧删去 `[cmd_offset : cmd_offset+d_cmd]` | `[B, T, d_enc]` | 编码器输入 |
| `cmd_current` | 最后一帧的 `[cmd_offset : cmd_offset+d_cmd]` | `[B, d_cmd]` | 指令嵌入 / Generator / Policy |
| `o_current_no_cmd` | 最后一帧删去 cmd | `[B, d_enc]` | Policy MLP 当前观测 |

其中 `d_enc = d_step - d_cmd`。

**拆分伪代码**：

```python
def split_obs(flat_obs, T, d_step, cmd_offset, d_cmd):
    """拆分 flat actor observation。

    Args:
        flat_obs: [B, T * d_step]
        T: history length
        d_step: per-step obs dim (including cmd)
        cmd_offset: index of velocity_commands within each step
        d_cmd: dim of velocity_commands (3)
    """
    frames = flat_obs.view(-1, T, d_step)                        # [B, T, d_step]

    # 保留的列索引（删去 cmd）
    keep = [i for i in range(d_step) if i not in range(cmd_offset, cmd_offset + d_cmd)]
    history_no_cmd = frames[:, :, keep]                           # [B, T, d_enc]

    cmd_current = frames[:, -1, cmd_offset : cmd_offset + d_cmd]  # [B, d_cmd]
    o_current_no_cmd = frames[:, -1, keep]                        # [B, d_enc]

    return history_no_cmd, cmd_current, o_current_no_cmd
```

---

## 3. 模块规格

### 3.1 编码器 — TCN 变体

**设计原则**：两层因果 Conv1d，感受野 = $1 + 2(k-1) = 5 = T$，精确覆盖全部历史。

```python
class CausalConv1d(nn.Module):
    """因果卷积：左填充 k-1，保证输出只依赖当前和过去。"""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x):                        # x: [B, C_in, T]
        x = F.pad(x, (self.pad, 0))              # 左填充
        return self.conv(x)
```

```python
class TCNEncoder(nn.Module):
    def __init__(self, d_enc: int, d_hidden: int = 128, d_out: int = 96, kernel_size: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(d_enc, d_hidden, kernel_size)
        self.conv2 = CausalConv1d(d_hidden, d_hidden, kernel_size)
        self.act = nn.ELU()
        self.proj = nn.Linear(d_hidden, d_out)

    def forward(self, history_no_cmd):
        """
        Args:
            history_no_cmd: [B, T, d_enc]
        Returns:
            h_enc: [B, d_out]
        """
        x = history_no_cmd.permute(0, 2, 1)       # [B, d_enc, T]
        x = self.act(self.conv1(x))                # [B, d_hidden, T]
        x = self.act(self.conv2(x))                # [B, d_hidden, T]
        x = x[:, :, -1]                            # [B, d_hidden]  取末时间步
        return self.proj(x)                        # [B, d_out]
```

| 参数 | 值 | 说明 |
|------|---|------|
| `d_enc` | `d_step - d_cmd` (如 51) | 编码器输入维度 |
| `d_hidden` | 128 | 卷积通道数 |
| `d_out` | 96 | 编码器输出 = 三个球面头输入 |
| `kernel_size` | 3 | 因果卷积核大小 |
| 参数量 | ~62K | 远小于 Transformer (~1.3M) |

### 3.2 编码器 — Transformer 变体

沿用现有 `TransformerHistoryModel` 的编码主体，去掉 cross-attention（不再需要单独的 aux 分支）。

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_enc: int, d_model: int = 128, d_out: int = 96,
                 n_heads: int = 4, num_layers: int = 2, dim_ff: int = 256,
                 history_len: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(d_enc, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, history_len, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, history_no_cmd):
        """
        Args:
            history_no_cmd: [B, T, d_enc]
        Returns:
            h_enc: [B, d_out]
        """
        h = self.input_proj(history_no_cmd) + self.pos_emb    # [B, T, d_model]
        h = self.encoder(h)                                    # [B, T, d_model]
        h = h[:, -1, :]                                        # [B, d_model]  末 token
        return self.proj(h)                                    # [B, d_out]
```

| 参数 | 值 | 说明 |
|------|---|------|
| `d_model` | 128 | 比现有(256)缩小，匹配下游规模 |
| `n_heads` | 4 | |
| `num_layers` | 2 | |
| `dim_ff` | 256 | d_model 的 2 倍 |
| `d_out` | 96 | 与 TCN 一致 |
| 参数量 | ~260K | |

### 3.3 乘积球面投影

将 `h_enc ∈ ℝ^{d_out}` 投影到三个独立单位球面。

```python
class ProductSphereProjection(nn.Module):
    def __init__(self, d_in: int = 96, d_sphere: int = 32):
        super().__init__()
        self.head_x = nn.Linear(d_in, d_sphere)
        self.head_y = nn.Linear(d_in, d_sphere)
        self.head_w = nn.Linear(d_in, d_sphere)

    def forward(self, h_enc):
        """
        Returns:
            z_x, z_y, z_w: each [B, d_sphere], L2-normalised on unit sphere
        """
        z_x = F.normalize(self.head_x(h_enc), dim=-1)
        z_y = F.normalize(self.head_y(h_enc), dim=-1)
        z_w = F.normalize(self.head_w(h_enc), dim=-1)
        return z_x, z_y, z_w
```

| 参数 | 值 | 说明 |
|------|---|------|
| `d_sphere` | 32 | 每个子球面维度；32 维球面可使 ~32 个向量近似正交 |
| `d_z` | 96 | $3 \times 32$，拼接后的总隐变量维度 |

### 3.4 对比投影头

对比损失在投影空间计算（SimCLR 的发现：投影头保护下游表征）。

```python
class ContrastiveProjector(nn.Module):
    def __init__(self, d_sphere: int = 32, d_proj: int = 64):
        super().__init__()
        self.proj_x = nn.Sequential(
            nn.Linear(d_sphere, d_proj), nn.ELU(), nn.Linear(d_proj, d_proj))
        self.proj_y = nn.Sequential(
            nn.Linear(d_sphere, d_proj), nn.ELU(), nn.Linear(d_proj, d_proj))
        self.proj_w = nn.Sequential(
            nn.Linear(d_sphere, d_proj), nn.ELU(), nn.Linear(d_proj, d_proj))

    def forward(self, z_x, z_y, z_w):
        """
        Returns:
            p_x, p_y, p_w: each [B, d_proj], L2-normalised
        """
        p_x = F.normalize(self.proj_x(z_x), dim=-1)
        p_y = F.normalize(self.proj_y(z_y), dim=-1)
        p_w = F.normalize(self.proj_w(z_w), dim=-1)
        return p_x, p_y, p_w
```

**仅在训练 Phase A 使用，推理和 Phase B 不调用。**

### 3.5 指令嵌入

连续值 `[vx, vy, vw]` → 嵌入向量。

```python
class CommandEmbedding(nn.Module):
    def __init__(self, d_cmd: int = 3, d_embed: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_cmd, d_embed),
            nn.ELU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, cmd):
        """
        Args:
            cmd: [B, d_cmd]  continuous velocity command
        Returns:
            e_c: [B, d_embed]
        """
        return self.mlp(cmd)
```

| 参数 | 值 |
|------|---|
| `d_cmd` | 3 |
| `d_embed` | 32 |

### 3.6 FiLM Generator

给定隐变量 $z$ 和指令嵌入 $e_c$，预测未来 $K$ 步动作。

```python
class FiLMGenerator(nn.Module):
    def __init__(self, d_z: int = 96, d_embed: int = 32,
                 d_hidden: int = 256, K: int = 3, d_action: int = 15):
        super().__init__()
        self.K = K
        self.d_action = d_action

        self.mlp_z = nn.Sequential(nn.Linear(d_z, d_hidden), nn.ELU())
        self.film = nn.Linear(d_embed, d_hidden * 2)        # → (γ, β)
        self.mlp_out = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, K * d_action),
        )

    def forward(self, z_cat, e_c):
        """
        Args:
            z_cat: [B, d_z]      concatenated sphere embeddings
            e_c:   [B, d_embed]  command embedding
        Returns:
            a_pred: [B, K * d_action]
        """
        h = self.mlp_z(z_cat)                                 # [B, d_hidden]
        film_params = self.film(e_c)                           # [B, 2*d_hidden]
        gamma, beta = film_params.chunk(2, dim=-1)             # each [B, d_hidden]
        h = (1 + gamma) * h + beta                             # FiLM modulation
        return self.mlp_out(h)                                 # [B, K*d_action]
```

| 参数 | 值 | 说明 |
|------|---|------|
| `d_hidden` | 256 | |
| `K` | 3 | 预测步长 |
| `d_action` | 15 | 关节动作维度 |
| 参数量 | ~155K | |

### 3.7 完整 Actor 模型

```python
class ContrastiveLatentModel(MLPModel):
    """Actor model with product-sphere latent space,
    factored contrastive learning, and FiLM generator.

    Compatible with rsl_rl MLPModel interface.
    Pairs with ContrastivePPO algorithm for training.
    """

    is_recurrent: bool = False

    # ── 构造 ──────────────────────────────────────────────

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # ── 新增参数 ──
        encoder_type: str = "tcn",           # "tcn" | "transformer"
        history_len: int = 5,
        d_step: int = 54,                    # 每帧 obs 维度 (含 cmd)
        cmd_offset: int = 6,                 # cmd 在每帧中的起始偏移
        d_cmd: int = 3,
        d_sphere: int = 32,
        d_proj: int = 64,
        d_cmd_embed: int = 32,
        gen_hidden: int = 256,
        pred_horizon: int = 3,               # K
        # TCN 参数
        tcn_hidden: int = 128,
        tcn_kernel: int = 3,
        # Transformer 参数
        tf_d_model: int = 128,
        tf_n_heads: int = 4,
        tf_num_layers: int = 2,
        tf_dim_ff: int = 256,
    ) -> None:
        # 先计算关键维度
        self._history_len = history_len
        self._d_step = d_step
        self._cmd_offset = cmd_offset
        self._d_cmd = d_cmd
        self._d_enc = d_step - d_cmd              # 每帧去 cmd 后维度
        self._d_z = 3 * d_sphere
        self._K = pred_horizon
        self._d_action = output_dim

        # 计算 policy MLP 输入维度
        # [o_current_no_cmd; e_c; z_cat; â_{1:K}]
        self._latent_dim = self._d_enc + d_cmd_embed + self._d_z + pred_horizon * output_dim

        # 调用父类
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        # ── 子模块 ──

        # 1. Encoder
        d_out = 3 * d_sphere                      # encoder → sphere head 的中间维度
        if encoder_type == "tcn":
            self.encoder = TCNEncoder(self._d_enc, tcn_hidden, d_out, tcn_kernel)
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                self._d_enc, tf_d_model, d_out,
                tf_n_heads, tf_num_layers, tf_dim_ff, history_len,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # 2. Product sphere projection
        self.sphere_proj = ProductSphereProjection(d_out, d_sphere)

        # 3. Contrastive projection heads (Phase A only)
        self.contrast_proj = ContrastiveProjector(d_sphere, d_proj)

        # 4. Command embedding
        self.cmd_embed = CommandEmbedding(d_cmd, d_cmd_embed)

        # 5. FiLM Generator
        self.generator = FiLMGenerator(
            self._d_z, d_cmd_embed, gen_hidden, pred_horizon, output_dim,
        )

        # 预计算 keep_indices（每帧中非 cmd 的列序号）
        self._keep_indices = [i for i in range(d_step)
                              if i not in range(cmd_offset, cmd_offset + d_cmd)]

    # ── 观测拆分 ────────────────────────────────────────

    def _split_obs(self, flat_obs: torch.Tensor):
        """将 flat actor obs 拆分为 encoder、cmd、policy 三部分。

        Args:
            flat_obs: [B, T * d_step]
        Returns:
            history_no_cmd: [B, T, d_enc]
            cmd_current:    [B, d_cmd]
            o_current:      [B, d_enc]    当前帧去 cmd
        """
        B = flat_obs.shape[0]
        frames = flat_obs.view(B, self._history_len, self._d_step)

        history_no_cmd = frames[:, :, self._keep_indices]                          # [B, T, d_enc]
        cmd_current = frames[:, -1, self._cmd_offset : self._cmd_offset + self._d_cmd]  # [B, d_cmd]
        o_current = frames[:, -1, self._keep_indices]                              # [B, d_enc]
        return history_no_cmd, cmd_current, o_current

    # ── 编码 + 投影（Phase A 使用）──────────────────────

    def encode(self, flat_obs: torch.Tensor):
        """编码历史 → 球面隐变量。对比学习和生成模块的入口。

        Returns:
            z_x, z_y, z_w:  each [B, d_sphere]
            h_enc:           [B, d_out]   (encoder raw output, 可选调试用)
        """
        history_no_cmd, _, _ = self._split_obs(flat_obs)
        h_enc = self.encoder(history_no_cmd)
        z_x, z_y, z_w = self.sphere_proj(h_enc)
        return z_x, z_y, z_w, h_enc

    def project_contrastive(self, z_x, z_y, z_w):
        """投影到对比学习空间。"""
        return self.contrast_proj(z_x, z_y, z_w)

    def generate(self, z_cat, e_c):
        """FiLM 生成器前向。"""
        return self.generator(z_cat, e_c)

    def embed_cmd(self, cmd):
        """指令嵌入。"""
        return self.cmd_embed(cmd)

    # ── Policy 推理接口（兼容 MLPModel）────────────────

    def get_latent(self, obs, masks=None, hidden_state=None):
        """标准 MLPModel 接口。Rollout 阶段使用。

        编码 + 生成 + 拼接（detach z 和 â），同时缓存中间产物
        供 ContrastivePPO 存入 RolloutStorage。
        """
        obs_list = [obs[g] for g in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)

        history_no_cmd, cmd_current, o_current = self._split_obs(flat_obs)

        # 编码
        h_enc = self.encoder(history_no_cmd)
        z_x, z_y, z_w = self.sphere_proj(h_enc)
        z_cat = torch.cat([z_x, z_y, z_w], dim=-1)         # [B, d_z]

        # 指令嵌入
        e_c = self.cmd_embed(cmd_current)                    # [B, d_cmd_embed]

        # 生成预测动作序列
        a_pred = self.generator(z_cat, e_c)                  # [B, K*d_a]

        # ★ 缓存中间产物（detach 后存储，供 Phase B 使用）
        self._cached_z_cat = z_cat.detach()
        self._cached_e_c = e_c.detach()
        self._cached_a_pred = a_pred.detach()
        self._cached_o_current = o_current.detach()

        # 拼接（detach encoder/generator 产物，policy 梯度不回传）
        latent = torch.cat([
            o_current,
            e_c.detach(),
            z_cat.detach(),
            a_pred.detach(),
        ], dim=-1)                                            # [B, latent_dim]

        return latent

    def get_cached_repr(self):
        """返回最近一次 get_latent() 缓存的中间产物。

        供 ContrastivePPO.step() 存入 RolloutStorage。

        Returns:
            z_cat:     [B, d_z]         球面隐变量拼接
            e_c:       [B, d_cmd_embed] 指令嵌入
            a_pred:    [B, K*d_a]       生成器预测动作
            o_current: [B, d_enc]       当前帧去 cmd 后的观测
        """
        return self._cached_z_cat, self._cached_e_c, self._cached_a_pred, self._cached_o_current

    def get_latent_from_cache(self, o_current, e_c, z_cat, a_pred):
        """Phase B 使用的接口：从缓存值构建 Policy MLP 输入。

        ⚠️ 不运行编码器和生成器，直接使用 rollout 时缓存的值。
        所有输入均已 detach，Policy MLP 梯度不回传到编码器。

        Args:
            o_current: [B, d_enc]       缓存的当前帧观测
            e_c:       [B, d_cmd_embed] 缓存的指令嵌入
            z_cat:     [B, d_z]         缓存的球面隐变量
            a_pred:    [B, K*d_a]       缓存的生成器预测
        Returns:
            latent:    [B, latent_dim]  Policy MLP 输入
        """
        return torch.cat([o_current, e_c, z_cat, a_pred], dim=-1)

    def _get_latent_dim(self) -> int:
        return self._latent_dim

    # ── 导出接口 ────────────────────────────────────────

    def get_repr_parameters(self):
        """返回表征学习优化的参数（Phase A）。"""
        params = (
            list(self.encoder.parameters())
            + list(self.sphere_proj.parameters())
            + list(self.contrast_proj.parameters())
            + list(self.cmd_embed.parameters())
            + list(self.generator.parameters())
        )
        return params

    def get_policy_parameters(self):
        """返回策略优化的参数（Phase B）。
        
        即基类 MLP 层和分布参数（不含 encoder/generator）。
        """
        repr_param_ids = {id(p) for p in self.get_repr_parameters()}
        return [p for p in self.parameters() if id(p) not in repr_param_ids]
```

策略 MLP 输入维度计算（G1 15DOF 为例）：

```
d_enc = 54 - 3 = 51        (当前帧去 cmd)
d_cmd_embed = 32
d_z = 96                   (3 × 32)
K × d_a = 3 × 15 = 45

latent_dim = 51 + 32 + 96 + 45 = 224
```

### 3.8 Critic 模型

**不变**。Critic 继续使用标准 MLPModel 或 TransformerHistoryModel，独立于 actor 的表征学习。Critic 拥有自己的观测集（含 `base_lin_vel` 等 privileged 信息），不受 Phase A 影响。

---

## 4. 损失函数

### 4.1 分解式 InfoNCE

对三个子球面独立施加 SupCon-style InfoNCE。

**指令标签提取**：使用离散化后的速度分量作为类别标签。

```python
def discretize_cmd(cmd, levels):
    """将连续 cmd 值映射到最近的离散等级索引。

    Args:
        cmd:    [B]         单分量连续速度值
        levels: [N_levels]  预定义的离散等级 (tensor)
    Returns:
        labels: [B]  LongTensor 索引
    """
    dists = (cmd.unsqueeze(-1) - levels.unsqueeze(0)).abs()   # [B, N]
    return dists.argmin(dim=-1)
```

**单子球面 InfoNCE**：

```python
def infonce_single_sphere(projections, labels, temperature):
    """单个子球面上的 supervised InfoNCE。

    Args:
        projections: [B, d_proj]  L2 归一化后的投影
        labels:      [B]         离散指令标签
        temperature: float
    Returns:
        loss: scalar
    """
    # 余弦相似度矩阵
    sim = projections @ projections.T / temperature           # [B, B]

    # 正样本 mask：same label, 排除自身
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)     # [B, B]
    self_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    pos_mask = label_eq & self_mask                           # [B, B]

    # 分母：所有非自身样本
    neg_mask = self_mask                                      # [B, B]

    # log-sum-exp trick for numerical stability
    sim_max = sim.detach().max(dim=-1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max) * neg_mask.float()
    log_denom = sim_max.squeeze(-1) + exp_sim.sum(dim=-1).log()

    # 正样本的平均 log-prob
    pos_count = pos_mask.float().sum(dim=-1).clamp(min=1)
    log_pos = ((sim - log_denom.unsqueeze(-1)) * pos_mask.float()).sum(dim=-1) / pos_count

    return -log_pos.mean()
```

**总对比损失**：

```python
def factored_infonce(p_x, p_y, p_w, cmd, levels_x, levels_y, levels_w, temperature):
    """
    Args:
        p_x, p_y, p_w: [B, d_proj]  投影向量
        cmd:            [B, 3]       [vx, vy, vw]
        levels_*:       [N_*]        各分量的离散等级
        temperature:    float
    """
    label_x = discretize_cmd(cmd[:, 0], levels_x)
    label_y = discretize_cmd(cmd[:, 1], levels_y)
    label_w = discretize_cmd(cmd[:, 2], levels_w)

    L_x = infonce_single_sphere(p_x, label_x, temperature)
    L_y = infonce_single_sphere(p_y, label_y, temperature)
    L_w = infonce_single_sphere(p_w, label_w, temperature)

    return L_x + L_y + L_w
```

**关键参数**：

| 参数 | 值 | 说明 |
|------|---|------|
| `temperature` | 0.5 | 可学习：`τ = exp(log_τ_param)` |
| `levels_x` | 环境配置中的离散速度集合 | 如 `[-0.3, 0, 0.3, 0.5, 0.8, 1.0]` |
| `levels_y` | 同上 | 如 `[-0.3, -0.1, 0, 0.1, 0.3]` |
| `levels_w` | 同上 | 如 `[-0.5, -0.2, 0, 0.2, 0.5]` |

**注意**：若 batch 中某个 label 只有 1 个样本（无正对），该样本对 L_NCE 的贡献为 0（`pos_count` clamp 保护）。Batch size 需足够大以保证正对存在——4096 并行环境 × 24 步 / 4 mini_batches ≈ 24576 样本/batch，充足。

### 4.2 序列预测损失

```python
def sequence_prediction_loss(a_pred, a_gt_future, valid_mask, gamma=0.9):
    """
    Args:
        a_pred:       [B, K * d_a]    生成模块输出
        a_gt_future:  [B, K * d_a]    ground truth 未来 K 步动作
        valid_mask:   [B]             True = 该样本有完整 K 步未来
        gamma:        float           时间衰减系数
    Returns:
        loss: scalar
    """
    K = a_gt_future.shape[-1] // d_a
    pred = a_pred.view(-1, K, d_a)
    gt = a_gt_future.view(-1, K, d_a)

    # 逐步 MSE
    step_mse = ((pred - gt) ** 2).sum(dim=-1)                # [B, K]

    # 时间衰减权重
    weights = gamma ** torch.arange(K, device=a_pred.device)  # [K]
    weighted_mse = (step_mse * weights).mean(dim=-1)          # [B]

    # 只对有效样本计算
    if valid_mask.any():
        return weighted_mse[valid_mask].mean()
    return torch.tensor(0.0, device=a_pred.device)
```

### 4.3 总训练目标

| Phase | 损失 | 输入来源 | 更新参数 | 优化器 |
|-------|------|---------|---------|--------|
| A（表征） | $\alpha \mathcal{L}_{\text{contrast}} + \beta \mathcal{L}_{\text{gen}}$ | 从 batch obs 重新编码 | encoder, sphere_proj, contrast_proj, cmd_embed, generator | `repr_optimizer` |
| B（策略） | $\mathcal{L}_{\text{PPO}}$ | **缓存的** z_cat, e_c, â, o_current | policy MLP 层 + distribution | `policy_optimizer` |
| B（价值） | $\mathcal{L}_{\text{value}}$（标准 PPO） | batch obs（critic独立） | critic 全部 | `policy_optimizer` |

> **⚠️ Phase B 使用缓存方案**：策略更新时直接使用 rollout 阶段缓存的 z/e_c/â，
> 避免 Phase A 更新编码器后重新编码导致的 z 不一致问题。

系数：

| 系数 | 初始值 | Schedule | 说明 |
|------|-------|----------|------|
| α | 0.1 | 常量 | 对比损失权重 |
| β | 0.5 | 线性衰减至 0.1（10000 iter） | 生成损失权重 |
| τ (temperature) | 0.5 | 可学习 | InfoNCE 温度 |

---

## 5. 训练算法：ContrastivePPO

### 5.1 Rollout Storage 扩展

标准 `RolloutStorage` 存储每步 transition。我们额外需要：

1. **缓存中间表征**（Phase B 使用 rollout 时的 z、e_c、â，避免重新编码）
2. **缓存当前帧观测**（Phase B 构建 Policy MLP 输入）
3. **未来 K 步动作**（生成模块 Phase A 监督信号）
4. **有效性 mask**（episode 边界检查）

#### 5.1.1 额外缓存 buffer

| Buffer 名称 | 形状 | 维度 | 用途 |
|-------------|------|------|------|
| `cached_z_cat` | `[T, N, d_z]` | 96 | 球面隐变量（Phase B 输入） |
| `cached_e_c` | `[T, N, d_cmd_embed]` | 32 | 指令嵌入（Phase B 输入） |
| `cached_a_pred` | `[T, N, K*d_a]` | 45 | 生成器预测（Phase B 输入） |
| `cached_o_current` | `[T, N, d_enc]` | 51 | 当前帧观测（Phase B 输入） |
| `future_actions` | `[T, N, K*d_a]` | 45 | 未来动作 GT（Phase A 监督） |
| `valid_mask` | `[T, N]` | 1 | 未来动作有效性 |

**额外显存开销**（G1 15DOF，4096 envs，24 steps，fp32）：

```
缓存: (96 + 32 + 45 + 51) × 4 bytes × 4096 × 24 = ~88 MB
未来动作: (45 + 1) × 4 bytes × 4096 × 24 = ~18 MB
总计: ~106 MB  (相对标准 Storage 的 ~200MB 增加约 50%)
```

#### 5.1.2 缓存存储（每 rollout step 调用）

```python
def store_cached_repr(storage, step, z_cat, e_c, a_pred, o_current):
    """在 rollout 的每一步存储编码器/生成器的缓存产物。

    由 ContrastivePPO.step() 在 actor.act() 之后调用。
    """
    storage.cached_z_cat[step] = z_cat
    storage.cached_e_c[step] = e_c
    storage.cached_a_pred[step] = a_pred
    storage.cached_o_current[step] = o_current
```

#### 5.1.3 未来动作目标构建

在 rollout 结束后、`update()` 前计算：

```python
def compute_future_actions(storage, K, d_action):
    """从 rollout buffer 构建 K 步未来动作目标。

    Args:
        storage: RolloutStorage  含 num_steps 步 transition
        K: prediction horizon
    Returns:
        future_actions: [num_steps, num_envs, K * d_action]
        valid_mask:     [num_steps, num_envs]  bool
    """
    T = storage.num_steps
    N = storage.num_envs
    actions = storage.actions                                  # [T, N, d_action]
    dones = storage.dones                                      # [T, N, 1]

    future = torch.zeros(T, N, K * d_action, device=actions.device)
    valid = torch.ones(T, N, dtype=torch.bool, device=actions.device)

    for k in range(K):
        t_src = torch.arange(T) + k + 1                       # 源时间步
        out_of_range = t_src >= T
        t_src = t_src.clamp(max=T - 1)
        future[:, :, k * d_action : (k + 1) * d_action] = actions[t_src]

        # Episode 边界检查：若 t+1 到 t+k+1 之间有 done，则无效
        if k + 1 < T:
            for j in range(1, k + 2):
                idx = torch.arange(T) + j
                idx = idx.clamp(max=T - 1)
                valid &= ~dones[idx].squeeze(-1).bool()
        valid[out_of_range] = False

    return future, valid
```

### 5.2 ContrastivePPO 类

```python
class ContrastivePPO(PPO):
    """PPO with Phase A representation learning (contrastive + sequence prediction).

    关键设计：缓存方案
    ─────────────────
    Rollout 阶段，actor.get_latent() 计算 z_cat/e_c/â 并缓存。
    每步调用 store_cached_repr() 存入 RolloutStorage。
    Phase B 直接使用缓存值构建 Policy MLP 输入，不重新编码。

    这解决两个收敛风险：
    - R1: Phase A 更新编码器后，Phase B 重新编码得到的 z 与 rollout 时不一致
    - R7: Policy 在 z_new 上计算 log π(a|z_new)，但 a 是在 z_old 下采样的
    """

    def __init__(
        self,
        actor: ContrastiveLatentModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        # Phase A 参数
        repr_lr: float = 3e-4,
        contrast_coef: float = 0.1,
        gen_coef: float = 0.5,
        gen_coef_schedule: list[float] | None = None,   # [start, end, begin_iter, warmup_iters]
        temperature: float = 0.5,
        learnable_temperature: bool = True,
        pred_horizon: int = 3,
        levels_x: list[float] | None = None,
        levels_y: list[float] | None = None,
        levels_w: list[float] | None = None,
        cmd_offset: int = 6,
        d_cmd: int = 3,
        # 标准 PPO 参数
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)

        self.contrast_coef = contrast_coef
        self.gen_coef = gen_coef
        self.gen_coef_schedule = gen_coef_schedule
        self.pred_horizon = pred_horizon
        self.cmd_offset = cmd_offset
        self.d_cmd = d_cmd

        # 离散化等级（注册为 buffer）
        self.register_buffer("levels_x", torch.tensor(levels_x or [0.0]))
        self.register_buffer("levels_y", torch.tensor(levels_y or [0.0]))
        self.register_buffer("levels_w", torch.tensor(levels_w or [0.0]))

        # 可学习温度
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.log_temperature = torch.tensor(temperature).log()

        # Phase A 优化器
        repr_params = actor.get_repr_parameters()
        if learnable_temperature and isinstance(self.log_temperature, nn.Parameter):
            repr_params.append(self.log_temperature)
        self.repr_optimizer = torch.optim.Adam(repr_params, lr=repr_lr)

        # Phase B 优化器（override 父类 optimizer 只含 policy 参数 + critic）
        policy_params = actor.get_policy_parameters()
        critic_params = list(critic.parameters())
        self.optimizer = torch.optim.Adam(
            policy_params + critic_params, lr=self.learning_rate
        )

        # 初始化额外 storage buffers
        self._init_extra_storage()

    def _init_extra_storage(self):
        """在 RolloutStorage 上创建额外的缓存 buffer。"""
        T = self.storage.num_steps
        N = self.storage.num_envs
        device = self.storage.device

        self.storage.cached_z_cat = torch.zeros(T, N, self.actor._d_z, device=device)
        self.storage.cached_e_c = torch.zeros(T, N, self.actor.cmd_embed.mlp[-1].out_features, device=device)
        self.storage.cached_a_pred = torch.zeros(T, N, self.actor._K * self.actor._d_action, device=device)
        self.storage.cached_o_current = torch.zeros(T, N, self.actor._d_enc, device=device)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _get_gen_coef(self):
        if self.gen_coef_schedule is None:
            return self.gen_coef
        start, end, begin, warmup = self.gen_coef_schedule
        progress = max(0.0, min(1.0, (self._step - begin) / max(warmup, 1)))
        return start + (end - start) * progress

    # ── Rollout 阶段：缓存 ─────────────────────────────

    def step(self, obs):
        """Override：在标准 step 后缓存编码器产物。

        调用顺序：actor.act(obs) → 缓存 → env.step()
        """
        # 标准前向（触发 get_latent 并缓存中间产物）
        actions = self.actor.act(obs)

        # 从 actor 取出缓存值，存入 storage
        z_cat, e_c, a_pred, o_current = self.actor.get_cached_repr()
        step_idx = self.storage.step                           # 当前写入位置
        self.storage.cached_z_cat[step_idx] = z_cat
        self.storage.cached_e_c[step_idx] = e_c
        self.storage.cached_a_pred[step_idx] = a_pred
        self.storage.cached_o_current[step_idx] = o_current

        return actions

    # ── Update 阶段 ────────────────────────────────────

    def update(self):
        """Two-phase update: A (representation) then B (cached PPO).

        Phase A: 从 batch 观测重新编码 → 对比损失 + 生成损失
        Phase B: 使用缓存的 z_cat/e_c/â → Policy MLP → PPO 损失
        ⚠️ Phase B 不重新运行编码器
        """

        # 0. 预计算未来动作目标
        future_actions, valid_mask = compute_future_actions(
            self.storage, self.pred_horizon, self.actor._d_action
        )

        mean_repr_loss = 0.0
        mean_contrast = 0.0
        mean_gen = 0.0
        mean_ppo_loss = 0.0
        mean_uniformity = 0.0
        mean_alignment = 0.0
        num_updates = 0

        # 标准 mini-batch 迭代
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            batch_indices = batch._indices   # mini-batch 索引（展平后的）

            # ── Phase A：表征学习 ──────────────────────

            # 取 flat_obs
            obs_list = [batch.observations[g] for g in self.actor.obs_groups]
            flat_obs = torch.cat(obs_list, dim=-1)
            flat_obs = self.actor.obs_normalizer(flat_obs)

            # 编码 + 投影
            z_x, z_y, z_w, _ = self.actor.encode(flat_obs)
            p_x, p_y, p_w = self.actor.project_contrastive(z_x, z_y, z_w)
            z_cat = torch.cat([z_x, z_y, z_w], dim=-1)

            # 提取当前帧 cmd
            frames = flat_obs.view(-1, self.actor._history_len, self.actor._d_step)
            cmd = frames[:, -1, self.cmd_offset : self.cmd_offset + self.d_cmd]
            e_c = self.actor.embed_cmd(cmd)

            # 生成预测
            a_pred = self.actor.generate(z_cat, e_c)

            # 对比损失
            L_contrast = factored_infonce(
                p_x, p_y, p_w, cmd,
                self.levels_x, self.levels_y, self.levels_w,
                self.temperature,
            )

            # 序列预测损失
            batch_future = future_actions.view(-1, future_actions.shape[-1])[batch_indices]
            batch_valid = valid_mask.view(-1)[batch_indices]
            L_gen = sequence_prediction_loss(a_pred, batch_future, batch_valid)

            gen_coef = self._get_gen_coef()
            L_repr = self.contrast_coef * L_contrast + gen_coef * L_gen

            self.repr_optimizer.zero_grad()
            L_repr.backward()
            nn.utils.clip_grad_norm_(self.actor.get_repr_parameters(), self.max_grad_norm)
            self.repr_optimizer.step()

            # ── Phase B：缓存 PPO 更新 ────────────────

            # ★ 从 storage 取缓存值（rollout 时产生，不受 Phase A 编码器更新影响）
            batch_z_cat = self.storage.cached_z_cat.view(-1, self.actor._d_z)[batch_indices]
            batch_e_c = self.storage.cached_e_c.view(-1, self.storage.cached_e_c.shape[-1])[batch_indices]
            batch_a_pred = self.storage.cached_a_pred.view(-1, self.storage.cached_a_pred.shape[-1])[batch_indices]
            batch_o_current = self.storage.cached_o_current.view(-1, self.actor._d_enc)[batch_indices]

            # 从缓存构建 Policy MLP 输入
            latent = self.actor.get_latent_from_cache(
                batch_o_current, batch_e_c, batch_z_cat, batch_a_pred
            )

            # 通过 Policy MLP（仅 MLP 层 + distribution 参与梯度）
            action_mean = self.actor.mlp(latent)
            self.actor.distribution.update(action_mean)
            actions_log_prob = self.actor.distribution.log_prob(batch.actions)
            entropy = self.actor.distribution.entropy()

            values = self.critic(batch.observations)

            # PPO surrogate loss（标准实现）
            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob)
            surr1 = -batch.advantages * ratio
            surr2 = -batch.advantages * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            surrogate_loss = torch.max(surr1, surr2).mean()

            # Clipped value loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + torch.clamp(
                    values - batch.values, -self.clip_param, self.clip_param
                )
                vl1 = (values - batch.returns) ** 2
                vl2 = (value_clipped - batch.returns) ** 2
                value_loss = torch.max(vl1, vl2).mean()
            else:
                value_loss = ((values - batch.returns) ** 2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.get_policy_parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            # ── 监控指标 ──
            mean_repr_loss += L_repr.item()
            mean_contrast += L_contrast.item()
            mean_gen += L_gen.item()
            mean_ppo_loss += loss.item()

            # 均匀性与对齐度（每 10 次更新算一次，避免额外开销）
            if num_updates % 10 == 0:
                with torch.no_grad():
                    mean_uniformity += _uniformity(p_x) + _uniformity(p_y) + _uniformity(p_w)
                    mean_alignment += _alignment(p_x, discretize_cmd(cmd[:, 0], self.levels_x))

            num_updates += 1

        self.storage.clear()
        self._step += 1

        n = max(num_updates, 1)
        n_monitor = max(num_updates // 10, 1)
        return {
            "repr_loss": mean_repr_loss / n,
            "contrast_loss": mean_contrast / n,
            "gen_loss": mean_gen / n,
            "ppo_loss": mean_ppo_loss / n,
            "temperature": self.temperature.item(),
            "gen_coef": self._get_gen_coef(),
            "uniformity": mean_uniformity / n_monitor,
            "alignment": mean_alignment / n_monitor,
        }


def _uniformity(embeddings, t=2.0):
    """Wang & Isola (2020) uniformity metric.

    值越负表示越均匀。若趋近 0 → 可能 mode collapse。
    """
    sq_pdist = torch.cdist(embeddings, embeddings, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().item()


def _alignment(embeddings, labels, alpha=2.0):
    """同标签样本之间的平均距离。值越小 → 对齐越好。"""
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask.fill_diagonal_(False)
    if not mask.any():
        return 0.0
    dists = torch.cdist(embeddings, embeddings, p=2).pow(alpha)
    return dists[mask].mean().item()
```

### 5.3 梯度流总结

```
Rollout 阶段:
  flat_obs ──→ encoder → sphere_proj → z_cat ──┬──→ generator → â
                                     cmd → cmd_embed → e_c ──┘
                                                │
                                     ┌──────────┴──────────┐
                                     │ 缓存到 Storage:      │
                                     │ z_cat, e_c, â,      │
                                     │ o_current            │
                                     └──────────┬──────────┘
                                                │ .detach()
  o_current, e_c, z_cat, â ──→ Policy MLP ──→ action ──→ env

Phase A (表征学习, update 内):
  flat_obs ──[grad]──→ encoder ──[grad]──→ sphere_proj ──[grad]──→ contrast_proj → L_NCE
                                    │──[grad]──→ generator → L_gen
                             cmd ──[grad]──→ cmd_embed ──┘

Phase B (策略优化, update 内):
  ⚠️ 不运行编码器，直接取 Storage 缓存值

  cached z_cat ─┐
  cached e_c   ─┤
  cached â     ─┤──→ get_latent_from_cache() → latent
  cached o_cur ─┘                                │
                                     Policy MLP ──[grad]──→ L_PPO
                                         ↑ 梯度只流经 MLP 层

  batch.observations ──→ Critic ──[grad]──→ L_value
```

**与旧方案的关键区别**：

| 方面 | 旧方案（重新编码） | 新方案（缓存） |
|------|-------------------|---------------|
| Phase B 的 z 来源 | 重新编码（Phase A 已更新编码器） | rollout 时缓存 |
| z 与 action 的一致性 | ❌ z_new ≠ z_old | ✅ z = z_old |
| 编码器在 Phase B 的计算量 | 有 | 无（节省约 30% 前向） |
| 额外显存 | 无 | ~106 MB（缓存 buffer） |
| 收敛风险 R1/R7 | 存在 | 消除 |

### 5.4 监控指标

训练中应持续监控以下指标，用于诊断收敛问题：

| 指标 | 公式/含义 | 健康值 | 异常信号 |
|------|----------|--------|---------|
| `repr_loss` | $\alpha L_{NCE} + \beta L_{gen}$ | 持续下降 | 震荡或上升 |
| `contrast_loss` | 分解式 InfoNCE | 从 ~ln(B) 下降 | 快速降到 0（投影头退化） |
| `gen_loss` | 序列预测 MSE | 持续下降 | 不下降（编码器没学到动态） |
| `uniformity` | $\log \mathbb{E}[e^{-t\|z_i-z_j\|^2}]$ | 负值（越负越均匀） | 趋近 0 → **mode collapse** |
| `alignment` | 同标签样本平均距离 | 小值 | 大值（聚类不佳） |
| `temperature` | 可学习 τ | 0.05 ~ 2.0 | <0.01 或 >5.0 |
| `ppo_loss` | PPO surrogate + value | 正常 PPO 范围 | 发散 |
| `gen_coef` | β schedule 当前值 | 按 schedule 衰减 | — |

**诊断决策树**：

```
uniformity → 0?
  └─ Yes → mode collapse → 增大 τ 或减小 α
contrast_loss 不下降?
  └─ Yes → batch 太小或 levels 设置不合理 → 检查标签分布
gen_loss 不下降?
  └─ Yes → 编码器未学到有用信息 → 暂时增大 β
ppo_loss 发散?
  └─ Yes → 检查 cached z 是否正常存储 → 降低 lr
```

---

## 6. 配置

### 6.1 模型配置类

```python
@configclass
class RslRlContrastiveModelCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.utils.contrastive_latent_model:ContrastiveLatentModel"

    encoder_type: str = "tcn"            # "tcn" | "transformer"
    history_len: int = 5
    d_step: int = MISSING                # 每帧 obs dim (含 cmd)
    cmd_offset: int = 6
    d_cmd: int = 3
    d_sphere: int = 32
    d_proj: int = 64
    d_cmd_embed: int = 32
    gen_hidden: int = 256
    pred_horizon: int = 3

    # TCN
    tcn_hidden: int = 128
    tcn_kernel: int = 3

    # Transformer
    tf_d_model: int = 128
    tf_n_heads: int = 4
    tf_num_layers: int = 2
    tf_dim_ff: int = 256
```

### 6.2 算法配置类

```python
@configclass
class RslRlContrastivePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "unitree_rl_lab.utils.contrastive_ppo:ContrastivePPO"

    repr_lr: float = 3e-4
    contrast_coef: float = 0.1
    gen_coef: float = 0.5
    gen_coef_schedule: list[float] | None = None    # [start, end, begin, warmup]
    temperature: float = 0.5
    learnable_temperature: bool = True
    pred_horizon: int = 3
    levels_x: list[float] | None = None
    levels_y: list[float] | None = None
    levels_w: list[float] | None = None
    cmd_offset: int = 6
    d_cmd: int = 3
```

### 6.3 Runner 配置（G1 15DOF TCN 示例）

```python
@configclass
class G115DofContrastiveTCNPPORunnerCfg(BasePPORunnerV3Cfg):
    """Contrastive latent policy with TCN encoder for G1 15-DOF."""

    actor = RslRlContrastiveModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="log"),
        encoder_type="tcn",
        history_len=5,
        d_step=54,             # 3+3+3+15+15+15 (15DOF obs)
        cmd_offset=6,          # velocity_commands offset
        d_cmd=3,
        d_sphere=32,
        d_proj=64,
        d_cmd_embed=32,
        gen_hidden=256,
        pred_horizon=3,
        tcn_hidden=128,
        tcn_kernel=3,
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlContrastivePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Phase A
        repr_lr=3e-4,
        contrast_coef=0.1,
        gen_coef=0.5,
        gen_coef_schedule=[0.5, 0.1, 0, 10000],
        temperature=0.5,
        learnable_temperature=True,
        pred_horizon=3,
        levels_x=[-0.3, 0.0, 0.3, 0.5, 0.8, 1.0],
        levels_y=[-0.3, -0.1, 0.0, 0.1, 0.3],
        levels_w=[-0.5, -0.2, 0.0, 0.2, 0.5],
        cmd_offset=6,
        d_cmd=3,
    )
```

---

## 7. 文件结构

```
source/unitree_rl_lab/unitree_rl_lab/utils/
├── rsl_rl_transformer_model.py         # 现有 (不修改)
├── rsl_rl_custom_ppo.py                # 现有 (不修改)
├── lcp_ppo.py                          # 现有 (不修改)
├── contrastive_latent_model.py         # 新增: ContrastiveLatentModel
│   ├── CausalConv1d
│   ├── TCNEncoder
│   ├── TransformerEncoder
│   ├── ProductSphereProjection
│   ├── ContrastiveProjector
│   ├── CommandEmbedding
│   ├── FiLMGenerator
│   └── ContrastiveLatentModel(MLPModel)
├── contrastive_ppo.py                  # 新增: ContrastivePPO
│   ├── compute_future_actions()
│   ├── discretize_cmd()
│   ├── infonce_single_sphere()
│   ├── factored_infonce()
│   ├── sequence_prediction_loss()
│   ├── _uniformity()                  # 监控: mode collapse 检测
│   ├── _alignment()                   # 监控: 聚类质量
│   └── ContrastivePPO(PPO)
│       ├── _init_extra_storage()      # 缓存 buffer 初始化
│       ├── step()                     # rollout 时缓存 z/e_c/â
│       └── update()                   # Phase A 重新编码 + Phase B 使用缓存
└── ...

source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/
└── rsl_rl_ppo_cfg.py                   # 扩展: 新增配置类
    ├── RslRlContrastiveModelCfg
    ├── RslRlContrastivePpoAlgorithmCfg
    └── G115DofContrastiveTCNPPORunnerCfg
```

---

## 8. 导出与部署

> **部署时不使用缓存方案**。缓存方案仅在训练的 Phase B 中使用，目的是避免 Phase A
> 更新编码器后 z 不一致的问题。部署时没有 Phase A/B 的分离，直接运行完整的
> encoder → sphere → generator → policy MLP 流水线即可。
> 对比投影头（`ContrastiveProjector`）在部署时可丢弃。

### 8.1 TorchScript 导出

推理时直接运行编码器 → 球面投影 → 生成器 → Policy MLP。

```python
class _TorchContrastiveLatentModel(nn.Module):
    """JIT-scriptable inference-only model."""

    def __init__(self, model: ContrastiveLatentModel):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.sphere_proj = copy.deepcopy(model.sphere_proj)
        self.cmd_embed = copy.deepcopy(model.cmd_embed)
        self.generator = copy.deepcopy(model.generator)
        self.mlp = copy.deepcopy(model.mlp)          # 基类 MLP 层

        # 缓存常量
        self.history_len = model._history_len
        self.d_step = model._d_step
        self.cmd_offset = model._cmd_offset
        self.d_cmd = model._d_cmd
        self.keep_indices = model._keep_indices

    @torch.jit.export
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.obs_normalizer(obs)
        # split
        frames = obs.view(-1, self.history_len, self.d_step)
        history = frames[:, :, self.keep_indices]
        cmd = frames[:, -1, self.cmd_offset : self.cmd_offset + self.d_cmd]
        o_cur = frames[:, -1, self.keep_indices]
        # encode
        h = self.encoder(history)
        z_x, z_y, z_w = self.sphere_proj(h)
        z_cat = torch.cat([z_x, z_y, z_w], dim=-1)
        # generate
        e_c = self.cmd_embed(cmd)
        a_pred = self.generator(z_cat, e_c)
        # policy
        latent = torch.cat([o_cur, e_c, z_cat, a_pred], dim=-1)
        return self.mlp(latent)                       # deterministic action
```

### 8.2 ONNX 导出

类似结构。TCN 编码器的 Conv1d 和 Transformer 的 `TransformerEncoder` 均有成熟的 ONNX 转换支持。

**注意**：

- `torch.nn.functional.normalize` 在 ONNX 中需显式实现为 `x / (x.norm(dim=-1, keepdim=True) + eps)`
- Causal Conv1d 的 `F.pad` 需要常量 pad size（已满足）
- Transformer 的 `batch_first=True` 需 opset ≥ 14

---

## 9. 参数量与显存汇总

### 9.1 模型参数量

| 模块 | TCN 方案 | Transformer 方案 |
|------|---------|-----------------|
| Encoder | 62K | 260K |
| ProductSphereProjection (3 heads) | 9.4K | 9.4K |
| ContrastiveProjector (3 heads, 仅训练) | 12.6K | 12.6K |
| CommandEmbedding | 0.2K | 0.2K |
| FiLMGenerator | 155K | 155K |
| Policy MLP [224→512→256→128→15] | 182K | 182K |
| **Actor 总计** | **~421K** | **~619K** |
| Critic (独立, [425→512→256→128→1]) | ~273K | ~273K |
| **全模型总计** | **~694K** | **~892K** |

对比现有 G115DofTransformerPPORunnerCfg（~1.5M）：新架构参数量更少（得益于 TCN 替代大 Transformer + 独立 critic 不变）。

### 9.2 缓存方案显存开销

| Buffer | 维度 | 单样本字节 | 总数 (T=24, N=4096) | 显存 |
|--------|------|-----------|---------------------|------|
| `cached_z_cat` | 96 | 384 | 98304 | 37.7 MB |
| `cached_e_c` | 32 | 128 | 98304 | 12.6 MB |
| `cached_a_pred` | 45 | 180 | 98304 | 17.7 MB |
| `cached_o_current` | 51 | 204 | 98304 | 20.0 MB |
| `future_actions` | 45 | 180 | 98304 | 17.7 MB |
| `valid_mask` | 1 | 1 | 98304 | 0.1 MB |
| **缓存总计** | | | | **~106 MB** |

标准 RolloutStorage 显存约 200 MB → 增加约 50%。在 RTX 4090 (24GB) 上可接受。

---

## 10. 超参调试路线

### 10.1 Phase 1：最小验证（推荐先做）

```yaml
# 最简配置，验证端到端训练能跑通
encoder_type: tcn
d_sphere: 16          # 减小到 16 加速迭代
pred_horizon: 1       # K=1 最小预测
contrast_coef: 0.0    # 先关闭对比
gen_coef: 0.5
```

验证项：
- [x] 训练不崩溃
- [x] 生成损失下降
- [x] PPO 奖励曲线正常

### 10.2 Phase 2：开启对比

```yaml
d_sphere: 32
contrast_coef: 0.1
gen_coef: 0.5
temperature: 0.5
levels_x: [...]
```

验证项：
- 对比损失下降
- 用 t-SNE 可视化 z^x, z^y, z^w 是否按指令聚类

### 10.3 Phase 3：完整系统

```yaml
pred_horizon: 3
gen_coef_schedule: [0.5, 0.1, 0, 10000]
learnable_temperature: true
```

验证项：
- 联合指令的 z 位于成分单指令的"中间位置"（P3 性质）
- 策略性能优于基线

### 10.4 Phase 4：编码器对比

在相同超参下切换 `encoder_type: transformer`，对比 TCN 和 Transformer 在以下指标上的表现：

- 训练收敛速度
- 最终奖励
- 隐空间聚类质量
- 推理延迟

---

## 附录 A：指令离散化的环境配置

需要配合修改 velocity command 生成器，使其从预定义的离散等级中采样：

```python
@configclass
class DiscreteLevelVelocityCommandCfg:
    """从离散速度等级中均匀采样。"""
    levels_x: list[float] = [-0.3, 0.0, 0.3, 0.5, 0.8, 1.0]
    levels_y: list[float] = [-0.3, -0.1, 0.0, 0.1, 0.3]
    levels_w: list[float] = [-0.5, -0.2, 0.0, 0.2, 0.5]
    resample_time: float = 10.0
    standing_prob: float = 0.05
```

或者保留连续采样，仅在对比损失内部做离散化（靠 `discretize_cmd` 映射到最近等级）。后者对环境无侵入，推荐先用。

## 附录 B：construct_algorithm 适配

`ContrastivePPO` 需在 `construct_algorithm` 中正确构建。参考 `UnitreePPO` 和 `LCPPPO` 的模式。

注意：额外缓存 buffer 在 `ContrastivePPO.__init__` 的 `_init_extra_storage()` 中自动创建，
`construct_algorithm` 无需手动初始化。

```python
class ContrastivePPO(PPO):

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        """构建 ContrastivePPO。与 UnitreePPO 同接口。"""
        # 解析 actor/critic 类
        actor_class = _resolve_class(cfg["actor"]["class_name"])
        critic_class = _resolve_class(cfg["critic"]["class_name"])

        # obs_groups
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        # 构建模型
        actor_cfg = _sanitize_model_cfg(actor_class, cfg["actor"])
        critic_cfg = _sanitize_model_cfg(critic_class, cfg["critic"])

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg)

        actor.to(device)
        critic.to(device)

        # Storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # Algorithm（_init_extra_storage 在 __init__ 中自动调用，
        # 创建 cached_z_cat, cached_e_c, cached_a_pred, cached_o_current buffers）
        alg_cfg = {k: v for k, v in cfg["algorithm"].items() if k != "class_name"}
        return ContrastivePPO(actor, critic, storage, device=device, **alg_cfg)
```
