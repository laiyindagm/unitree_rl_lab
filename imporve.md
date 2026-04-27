分析：CLP 框架下 Transformer 编码器的双重目标与改造方案
你的核心约束很清晰：在 Contrastive Latent Policy（CLP）框架中，编码器必须同时服务两个目标：

目标	损失函数	编码器需要学什么	隐空间几何要求
运动特征编码（对比学习）	分解式 InfoNCE	从历史观测中提取"当前在执行哪种运动模式"的静态特征	同指令聚拢、异指令远离（球面聚类）
运动趋势预测（序列预测）	时间衰减 MSE	从历史观测中提取"接下来会怎么动"的动态预测	latent 包含足够的时序信息来预测未来 K 步动作
这两个目标的内在张力是你面临的核心设计挑战。下面逐层分析。

1. 张力分析：对比学习 vs 序列预测对编码器的矛盾需求
1.1 对比学习想要什么
InfoNCE 驱动编码器输出 
h
enc
 → 球面投影 
z
x
,
z
y
,
z
w
 → 对比投影 
p
x
,
p
y
,
p
w
。

对比损失的梯度方向是：把相同指令的样本拉到一起，不同指令的推开。这意味着编码器被鼓励忽略与当前指令无关的变化（比如不同初始关节角度、不同惯性状态）——因为这些"噪声"会破坏聚类。

极端情况：完美的对比编码器只保留 cmd 相关信息，丢弃一切时序动态 → 
h
enc
 退化为 cmd 的非线性映射。

1.2 序列预测想要什么
FiLM Generator 从 
z
cat
 + 
e
c
 预测未来 K 步动作。这要求 
z
cat
（即 
h
enc
 经球面投影后的产物）包含丰富的动态信息——当前关节位置、速度、接触状态、惯性……因为仅知道"指令是 
v
x
=
0.5
"不足以预测具体的关节角度序列。

极端情况：完美的预测编码器保留全部动态信息 → 
z
 携带高度个体化的特征 → 同指令样本因初始状态不同而在球面上分散 → 对比损失上升。

1.3 冲突的本质
对比学习	序列预测
希望 
z
 保留什么	指令类别信号（低维、离散化）	动态状态信息（高维、连续）
希望 
z
 丢弃什么	个体差异（"噪声"）	什么都不想丢
信息量	低（~8 bits）	高（连续动态）
当前设计用同一个 
h
enc
（96D）→ 球面投影 → 同时服务两个目标，这迫使 96 个维度既要聚类又要保持个体差异，形成 Pareto 权衡。

2. 之前 Transformer 辅助预测设计的回顾
在 transformer_aux_design.md 中的方案（Phase 1+2）：

Code
history → Transformer(因果掩码) → last_token → FiLM(aux) → latent(256D) 
                                                               │
                                                        next_obs_head → pred Δo (54D)
这个方案的优点是简单——只有一个预测目标（next-obs MSE），不存在对比学习的拉力。

但搬到 CLP 框架后，问题变成：如何在乘积球面约束下同时高效完成两个目标。

3. 推荐的 Transformer 编码器改造方案
核心思路：解耦对比特征和预测特征的提取路径，在 Transformer 内部而非外部完成分离。

3.1 方案 A：双头编码器（Split-Head Encoder）— 推荐
Code
history_no_cmd [B, 5, 51]
     │
     ▼
  input_proj(51 → d_model=128) + pos_emb
     │
     ▼
  ╔═══════════════════════════════════════╗
  ║  Shared TransformerEncoder            ║
  ║  2 layers, 4 heads, causal mask       ║
  ║  Pre-LayerNorm                        ║
  ╚═══════════════════════════════════════╝
     │
     ▼  h_encoded [B, 5, 128]
     │
     ├──────────────────────────────────────────────┐
     │                                              │
     ▼                                              ▼
  last_token = h[:, -1, :]                    全序列 h[:, :, :]
  [B, 128]                                    [B, 5, 128]
     │                                              │
     ▼                                              ▼
  contrast_proj(128 → 96)                     predict_proj(128 → 96)
  [B, 96]                                    [B, 96]  (last token 或 pooled)
     │                                              │
     ▼                                              ▼
  ProductSphere → z^x, z^y, z^w              FiLM Generator → â [B, K*15]
  → ContrastiveProjector → L_NCE             → L_gen (时间衰减 MSE)
     │
     ▼
  z_cat [B, 96] → Policy MLP 输入
关键改动：

Transformer 共享骨干，但 两个独立的线性投影头 分别产出对比特征和预测特征
contrast_proj 输出的 96D 进入球面投影和 InfoNCE —— 只需编码"哪种运动模式"
predict_proj 输出的 96D 进入 FiLM Generator —— 需要保留动态细节
两个投影头使用不同的 nn.Linear，梯度在投影层分叉，不再在同一个 96D 上打架
为什么有效：

SimCLR 的核心发现是：投影头保护编码器的表征质量。这里 contrast_proj 和 predict_proj 各自扮演"投影头"角色，让 Transformer 骨干的 128D 表征自由分配容量给两个任务
Transformer 的 multi-head attention 天然支持不同注意力头关注不同信息——一些头可以学习关注运动模式（服务对比），另一些可以学习关注动态细节（服务预测）
投影头隔离了反向传播的梯度冲突——InfoNCE 的"聚类"梯度和 MSE 的"保留细节"梯度在投影层之后分开，不在 Transformer 权重上直接冲突
Policy MLP 的输入：仍然用 z_cat（对比路径的球面隐变量），因为策略需要的是"执行什么运动模式"的信息，而 FiLM Generator 的 â 已经补充了动态预测信息。

3.2 方案 B：全序列辅助预测（Token-wise Aux Loss）
在 方案 A 基础上，进一步利用 Transformer 的因果序列结构：

Code
h_encoded [B, 5, 128]    ← Transformer 全序列输出
     │
     ├── last_token [B, 128] → contrast_proj → sphere → L_NCE
     │
     ├── last_token [B, 128] → predict_proj → FiLM Generator → â → L_gen
     │
     └── 全序列 [B, 5, 128] → next_obs_head → pred Δo [B, 5, 51]
                                                │
                                      ┌─────────┴─────────┐
                                      │ 窗口内预测 (0→1, 1→2, 2→3, 3→4)
                                      │ + 跨步预测 (4→next_obs)
                                      └──── L_aux (token-wise MSE)
这样编码器有三重训练信号：

L_NCE：球面聚类 → "哪种运动"
L_gen：动作预测 → "未来怎么动"
L_aux：观测预测 → "下一帧会看到什么"
Token-wise 预测为每个注意力层的每个 token 提供直接梯度（类似 LLM 的 next-token loss），极大加速 Transformer 的注意力学习。

损失总量：

Python
L_repr = α * L_NCE + β * L_gen + γ * L_aux
衰减策略：

L_NCE：常量 α=0.1
L_gen：衰减 β: 0.5 → 0.1（已有设计，不变）
L_aux：衰减 γ: 0.3 → 0.05（新增，与 L_gen 类似衰减，前期帮助 Transformer 学习时序编码，后期减弱避免干扰对比聚类）
3.3 方案 C：显式信息瓶颈分离（如果方案 A/B 不够）
如果实验发现对比和预测梯度仍在 Transformer 骨干中冲突，可以在 Transformer 之后加一个显式信息瓶颈：

Code
h_encoded [B, 128]
     │
     ├── contrast_proj → VIB → z_contrast ~ N(μ, σ²) → sphere → L_NCE + KL
     │                         [B, 64]
     │
     └── predict_proj → z_predict [B, 96] → FiLM Generator → L_gen
                        [B, 96]
VIB 迫使对比路径的信息量被显式压缩（通过 KL 正则），只保留"最低信息量"的类别信号——这与 InfoNCE 的目标高度一致。预测路径不受 KL 约束，自由保留动态信息。

4. 具体架构参数建议
基于你现有代码（CLPTransformerEncoder 的配置），推荐的双头改造参数：

参数	当前值	建议值	理由
d_model	256	192	双头各 96D，需要足够容量分配
n_heads	4	6	更多头 → 更细粒度的注意力分工
num_layers	2	2	5 步窗口不需要更深
dim_feedforward	512	384	2×d_model
contrast_proj_out	—	96	等于 3 × sphere_dim
predict_proj_out	—	96	等于 enc_dim，FiLM Generator 输入
aux_head (可选)	—	51	next-obs 残差预测
实现要点：

Python
class CLPDualHeadTransformerEncoder(nn.Module):
    def __init__(self, input_dim, history_len, enc_dim=96,
                 d_model=192, n_heads=6, num_layers=2, dim_ff=384):
        super().__init__()
        self.output_dim = enc_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, history_len, d_model) * 0.1)
        
        causal_mask = torch.triu(torch.ones(history_len, history_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ★ 双头投影 — 关键改动
        self.contrast_proj = nn.Linear(d_model, enc_dim)   # → sphere → L_NCE
        self.predict_proj = nn.Linear(d_model, enc_dim)    # → FiLM → L_gen
        
        # ★ 可选：token-wise 观测预测头
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ELU(),
            nn.Linear(d_model, input_dim),                # → pred Δo
        )
    
    def forward(self, x):
        """x: [B, T, D] → (contrast_out [B, enc_dim], predict_out [B, enc_dim])"""
        h = self.input_proj(x) + self.pos_emb
        h = self.encoder(h, mask=self.causal_mask)
        last = h[:, -1, :]
        return self.contrast_proj(last), self.predict_proj(last)
    
    def forward_with_aux(self, x, history_raw):
        """训练时用，额外返回 token-wise 预测损失。"""
        h = self.input_proj(x) + self.pos_emb
        h_seq = self.encoder(h, mask=self.causal_mask)
        last = h_seq[:, -1, :]
        
        # 双头输出
        z_contrast = self.contrast_proj(last)
        z_predict = self.predict_proj(last)
        
        # Token-wise 残差预测
        pred_delta = self.aux_head(h_seq)                 # [B, 5, 51]
        delta_target = history_raw[:, 1:, :] - history_raw[:, :-1, :]  # [B, 4, 51]
        intra_loss = F.mse_loss(pred_delta[:, :-1, :], delta_target)
        
        return z_contrast, z_predict, intra_loss
5. 对 ContrastivePPO Phase A 的改造
当前 Phase A：

Python
h_enc = encoder(history_no_cmd)          # 单一 96D 输出
z_spheres = sphere_proj(h_enc)           # → InfoNCE
a_pred = generator(z_cat, cmd)           # → L_gen（也用 h_enc 的球面版本）
改造后的 Phase A：

Python
z_contrast, z_predict = encoder(history_no_cmd)   # 双路 96D 输出

# 对比路径
z_spheres = sphere_proj(z_contrast)                # → InfoNCE（只用对比特征）
p_spheres = contrast_projs(z_spheres)
L_NCE = factored_infonce(p_spheres, labels, τ)

# 预测路径
a_pred = generator(z_predict, cmd)                 # → L_gen（用预测特征，不经球面）
L_gen = sequence_prediction_loss(a_pred, future_gt, mask)

# 可选：token-wise 辅助
z_contrast, z_predict, L_aux = encoder.forward_with_aux(history_no_cmd, history_raw)

L_repr = α * L_NCE + β * L_gen + γ * L_aux
6. 对 Policy MLP 输入的调整
关键问题：Policy MLP 应该收到什么？

当前：[o_current, cmd, z_cat (对比球面), a_pred (从对比z生成)]

建议改为：[o_current, cmd, z_cat (对比球面), a_pred (从预测z生成)]

即 a_pred 来自 predict_proj 路径，而非对比路径。这样策略网络同时获得：

z_cat："当前运动模式是什么"（来自对比路径，结构化球面表征）
a_pred："具体怎么动"（来自预测路径，包含动态细节的规划）
o_current：当前原始观测
cmd：当前指令
7. 部署时的变化
部署时丢弃 ContrastiveProjector 和 aux_head，保留两个投影头：

Code
obs → Encoder → contrast_proj → sphere → z_cat
                predict_proj → FiLM Generator(z_predict, cmd) → a_pred
             
  latent = [o_current, cmd, z_cat, a_pred] → Policy MLP → action
与现有部署流程完全兼容，只是 FiLM Generator 的输入从 z_cat（球面）变为 z_predict（非球面），推理计算量不变。

8. 总结：优先级与推荐路径
优先级	改动	预期效果	实现难度
P0	双头投影（contrast_proj + predict_proj）	🔥🔥🔥 解耦对比/预测梯度冲突	低（新增 2 个 Linear）
P0	FiLM Generator 使用 predict_proj 输出而非 z_cat	🔥🔥🔥 预测质量提升	低（改一行调用）
P1	Token-wise 残差预测辅助损失	🔥🔥 加速 Transformer 注意力学习	中（新增 aux_head + loss）
P1	n_heads 从 4 增加到 6	🔥 更细粒度的注意力分工	无（改配置）
P2	VIB 信息瓶颈在对比路径	🔥 显式压缩，但增加训练复杂度	高（新增 KL 调度）
推荐实施顺序：先做 P0（双头 + FiLM 输入改路径），验证效果。如果 Transformer 注意力收敛仍慢，加 P1（token-wise aux loss）。P2 作为最后手段。