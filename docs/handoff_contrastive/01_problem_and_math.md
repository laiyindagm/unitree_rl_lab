# 问题陈述与数学定义（01）

> 目标读者：算法/研究者。本篇给出"为什么是这套损失、为什么是这套 r_int"的完整推导。
> 工程实现位置见 [03_codebase_and_handoff.md](03_codebase_and_handoff.md)；版本演进见 [02_design_evolution.md](02_design_evolution.md)。

---

## 1. 问题陈述

### 1.1 策略层瓶颈

外在 reward 主要是 Gaussian 跟踪核：

$$
r_\text{track}(s_t, v_t) = \exp\!\left(-\frac{\|v_\text{actual}(s_t) - v_t\|^2}{\sigma^2}\right)
$$

**该 reward 是关于"达到 v_t"的**，对"用何种步态达到 v_t"完全无差别——任何能让 $v_\text{actual} \to v_t$ 的策略 $\pi$ 都拿满分。当 $\pi$ 收敛到一个能在所有 $v$ 下"凑出"该速度的单一策略时，再往下推不动；典型表现是 V21g 在 `pure_wz` 与低速 `pure_xy` 之间复用同一种"小步原地刨"行为。

### 1.2 表征层目标

我们希望存在 $f_\theta: \mathcal H \to \mathbb R^{d_\text{gait}}$（$\mathcal H$ 为 obs history），把 segment 编码成 $z = f_\theta(h)$，并在 $z$ 空间满足三条性质（用 $W = \mathrm{diag}(1/\sigma_a^2)$ 作为 cmd 度量权重，$\sigma_a$ 为 cmd 三维各自训练数据的 std）：

- **(Iso) 度量等距**：$d_Z(z_i, z_j) \propto \|v_i - v_j\|_W$
- **(A1) 轴向可加**：存在 3 个 axial bases $E_x, E_y, E_\omega \in \mathbb R^{d_\text{gait}}$ 使 $g^*(v) := \sum_a \rho_a E_a$（$\rho_a = v_a / \|v\|_W$）能近似 $z$ 在 cmd 决定子空间上的投影
- **(A2) 幅值 Lipschitz**：同方向 cmd $v_1 = \lambda_1 \hat v$，$v_2 = \lambda_2 \hat v$ 满足 $\|z_1 - z_2\| \ge L|\lambda_1 - \lambda_2|$
- **(Var) 类内方差**：同 cmd 桶内 $z$ 仍保留方差，给"亚步态"留生长空间

(Iso) + (A1) + (A2) 把 $z$ 中的 cmd 决定分量"钉死"成一个低维线性子空间；(Var) 让 cmd 之外的自由度（步频、duty、躯干姿态等）继续被 RL 学到。

### 1.3 联合优化目标（V22b）

固定 $\theta$（encoder frozen），优化 policy：

$$
\max_\pi\ \mathbb E\Big[\sum_t r_\text{env}(s_t, a_t) + \alpha(t)\cdot \mathrm{gate}(s_t)\cdot r_\text{int}\!\big(z_t, v_t\big)\Big]
$$

其中 $r_\text{int}$ 度量 $z$ 偏离"cmd 决定子空间"的程度（详见 §3.6）。这是 SMERL 风格的"task + scaled intrinsic"。

---

## 2. 编码器输入与符号

| 符号 | 维度 | 含义 |
|---|---|---|
| $h \in \mathbb R^{T \times D}$ | $T = 32$, $D = 295$ | rolling buffer of policy obs（pretrain 与 runtime 同形） |
| $m \in \{0,1\}^D$ | $D$ | strict mask：屏蔽 cmd 相关字段（vel_cmd, lin_speed_token, gait_mode_token, last_action, base_ang_vel, joint_vel_rel） |
| $a_\text{sc} \in \mathbb R^{2}$ | 2 | anchor sin/cos：左 hip pitch joint_pos_rel 的相位锚点 |
| $z = f_\theta(h, m, a_\text{sc}) \in \mathbb R^{d_\text{gait}}$ | $d_\text{gait} = 32$ | 步态隐变量 |
| $v \in \mathbb R^3$ | 3 | velocity command (vx, vy, ωz) |
| $\sigma \in \mathbb R^3_+$ | 3 | sigma_cmd，预训练时 cmd 三维 std，存盘到 ckpt |
| $E \in \mathbb R^{3 \times d_\text{gait}}$ | $3 \times 32$ | 可学 axial bases |

**严格 mask 的必要性**：若 mask 不屏蔽 vel_cmd（"loose"模式），encoder 会直接用 history 中的 cmd 字段最优拟合 $\delta_{ij}$ 标签，导致 $R^2(z \to v_x) = 0.997$ 的 cmd 泄露（v1 实际观测到）；表面上 RnC loss 收敛漂亮，但 z 几乎等于 cmd 的非线性变换，毫无步态信息。

---

## 3. 损失项逐个推导

### 3.1 Rank-N-Contrast (L_metric_rnc)

参考 Zha et al. NeurIPS 2023。给一批 $\{(z_i, v_i)\}_{i=1}^B$，定义 cmd 距离标签 $\delta_{ij} = \|v_i - v_j\|_W$。RnC 把"距离更近的 pair 应该有更高 cosine"做成 ranking-style InfoNCE：

$$
\mathcal L_\text{rnc} = \frac{1}{B(B-1)} \sum_{i} \sum_{j \ne i}
   -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}
              {\sum\limits_{k:\ \delta_{ik} \ge \delta_{ij}} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
$$

直觉：对每个 anchor $i$ 和"较近"伙伴 $j$，分母只把"比 $j$ 更远"的样本作为负样本——这样保证学到的 $d_Z$ 与 $\delta$ 同序（rank-preserving）。$\tau = 0.1$（不学温度，避免 V19d-CLP 的 collapse）。

**与 SupCon 的关系**：当 $\delta_{ij}$ 退化为类标签（同类=0，异类=1）时，RnC ≡ SupCon。RnC 是 SupCon 在连续标签上的自然推广。

**与 cmd-bucket InfoNCE 的对比**：早期方案把 cmd 离散成桶做 SupCon，但桶定义本身就是 cmd 的函数 → encoder 极易把"预测 cmd"作为捷径 → 无步态信息。RnC 用连续度量回避了"桶=cmd"的退化。

### 3.2 Axial 损失 (L_axial)

A1 性质要求 $z$ 在 cmd 子空间上能被 $g^*(v)$ 线性表示：

$$
g^*(v) = \sum_a \rho_a(v) \cdot E_a, \quad \rho_a(v) = \frac{v_a / \sigma_a}{\|v\|_W}
$$

注意 $\rho$ 是**单位方向向量**（$\|\rho\| = 1$ when $\|v\|_W > 0$），$\sum_a \rho_a E_a$ 是方向加权的 axial bases 之和。损失：

$$
\mathcal L_\text{axial} = \mathbb E\big[\|z - g^*(v)\|_2^2\big]
$$

> **关键认知更新**（V3 实测）：$R^2(z \to g^*) < 0.15$ 不是失败——axial bases 只张成 $z$ 的 cmd 决定子空间（$\le 3$ 维），剩下的 $\ge 29$ 维方差被 (Var) 性质保留。$R^2$ 低恰恰说明 $z$ 不被 cmd 完全决定。判断 axial 是否有效，看 V22b 的 $r_\text{int}$ 是否能稳定 $> 0$ 而非 $R^2$。

**导数（用于直觉）**：固定 $E$ 看 $\nabla_z \mathcal L_\text{axial} = 2(z - g^*(v))$，固定 $z$ 看 $\nabla_{E_a} \mathcal L_\text{axial} = -2\rho_a (z - g^*(v))$。即 axial bases 朝着"残差在 $\rho$ 方向上的均值"移动。

### 3.3 Lipschitz hinge 损失 (L_lip)

A2 性质要求 $z$ 在 cmd 幅值上"撑开"，避免 $\lambda_1 \ne \lambda_2$ 但 $z_1 = z_2$ 的 collapse：

$$
\mathcal L_\text{lip} = \frac{1}{|\mathcal P|} \sum_{(i,j) \in \mathcal P}
  \max\big(0,\ L \cdot |\lambda_i - \lambda_j| - \|z_i - z_j\|\big)^2
$$

pair 集合 $\mathcal P$ 由 batch 内 $\cos(\hat v_i, \hat v_j) > 0.95$ 且 $|\lambda_i - \lambda_j| > 0.2$ 的 index 对组成（同方向、显著幅值差），最多 $|\mathcal P| \le 512$。$L = 1.0$。

**为什么是 hinge 而不是平方差**：平方差会强迫 $\|z_1 - z_2\| \approx L|\lambda_1 - \lambda_2|$，约束太强；hinge 只要求 $\ge$，不阻止编码器在 cmd 子空间外学到更多分化。

### 3.4 步态属性回归 (L_prop)

把 $z$ 喂给 MLP head 回归 4 个步态属性 $\{\text{duty}_l, \text{yaw}, \text{lat}, \text{act}\}$（步周期 duty cycle、躯干 yaw 速率、横向 sway、动作幅度），SmoothL1：

$$
\mathcal L_\text{prop} = \mathbb E\big[\mathrm{Huber}(\mathrm{MLP}_\phi(z), \text{prop}(\text{seg}))\big]
$$

prop 通过离线 `frnc_gait_features` 从 segment 直接计算。L_prop 不是设计目标，**只作为 z 是否携带步态信息的诊断 head**——v3_full ckpt 上 $R^2(z \to \text{duty}_l) = 0.95$，确认 z 编码了步态。

### 3.5 V3 总损失

$$
\mathcal L_\text{V3} = w_\text{rnc}\,\mathcal L_\text{rnc} + w_\text{axial}\,\mathcal L_\text{axial} + w_\text{lip}\,\mathcal L_\text{lip} + w_\text{prop}\,\mathcal L_\text{prop}
$$

`v3_full` 配置（ckpt 存盘）：$w = (1.0, 1.0, 1.0, 1.0)$，$d_\text{back} = 128$，$d_\text{gait} = 32$，$\tau = 0.1$，30 epoch。

### 3.6 V22b 内在奖励（axial 残差）

定义：

$$
r_\text{int}(z, v) = \mathbb 1\!\left[\|v\|_W > \epsilon_\text{cmd}\right] \cdot \frac{\|z - g^*(v)\|_2^2}{d_\text{gait}}
$$

**为什么除以 $d_\text{gait}$**：让 reward 量级与 dim 解耦，方便跨 ckpt 调 α。

**为什么 mask `‖v‖_W < ε_cmd`**：当 $v \to 0$ 时 $g^*(v) = 0$（因 $\rho$ 用 $\|v\|_W$ 归一化），残差 $\|z - 0\| = \|z\|$ 会随 $\|z\|$ 单调增长——若不 mask，policy 可以在 standing 模式下"把 z 推大"骗 reward，与速度跟踪解耦。$\epsilon_\text{cmd} = 0.1$（$W$-norm，即过 axial 之后的归一长度）。

**为什么是残差而不是 InfoNCE 桶分类**：CIC 原始论文用桶 InfoNCE，但本项目桶定义 = cmd → 与 cmd 同义 → policy 不能"主动跨桶" → r_int 梯度为 0。用 axial 残差 = "z 中超出 cmd 决定子空间的部分"，policy 可以通过"在同一 cmd 下做出不同 z"获得 reward。

### 3.7 SMERL gate

$$
\mathrm{gate}(s) = \sigma\big(\kappa \cdot (\mathrm{EMA}_r(s) - \beta)\big), \quad
\mathrm{EMA}_r \leftarrow d \cdot \mathrm{EMA}_r + (1-d) \cdot r_\text{env}
$$

per-env，EMA decay $d = 0.99$，threshold $\beta = 0.045$（≈ V21g 中段 per-step env reward 0.056 的 80%），slope $\kappa = 200$。$\sigma$ 是 sigmoid。

**为什么 EMA on env reward 而非 episodic return**：per-step gating 在 PPO 的 step-wise advantage 计算下更平滑；episodic return 需要等 done 信号，gate 突变。

**导数**：$\partial \mathrm{gate} / \partial \mathrm{EMA}_r = \kappa\, \mathrm{gate}\, (1-\mathrm{gate})$。最大斜率在 $\mathrm{EMA}_r = \beta$ 处 $= \kappa/4 = 50$，对应 EMA 偏离 $\beta$ 约 $\pm 0.01$（即 $\pm 18\%$ relative）就会跨过 0.5 → 1.0 的转换带。这是 SMERL 设计的关键："gating 比 task reward 灵敏度高一个量级，但仍在合理范围内"。

### 3.8 α schedule

$$
\alpha(t) = \alpha_\text{max} \cdot \mathrm{clip}\!\left(\frac{t - w_0}{w_1 - w_0},\ 0,\ 1\right)
$$

iter 计数 $t = $ `self.counter`（PPO 的迭代轮次），$w_0 = 200$, $w_1 = 2000$, $\alpha_\text{max} = 0.02$。

**为什么从 200 开始而非 0**：$t < 200$ 时 RolloutStorage 还没积累足够 dones，$\mathrm{EMA}_r$ 噪声大；α=0 让 V22b 在 $t \in [0, 200]$ 严格等价于 V22a，避免冷启不稳。

**$\alpha_\text{max} = 0.02$ 量级估算**：实测 z norm ≈ 2.78，$g^*$ norm ≈ 0.42，残差²/d ≈ 0.28；中段 gate ≈ 0.5；bonus ≈ $0.02 \times 0.5 \times 0.28 \approx 0.003$，约为 V21g 中段 per-step env reward 0.056 的 5%。安全。

---

## 4. 关键取舍解读

| 决策 | 选择 | 原因 |
|---|---|---|
| encoder 在线 vs 冻结 | **冻结** | 在线（V19d-CLP）→ z 跟 policy 漂移 → r_int 失稳 → τ collapse |
| z 进 critic | **不进** | (i) advantage 估计偏差；(ii) RolloutStorage 用 TensorDict，shape 在 storage init 时锁定，z 进 critic 会破坏共享 obs schema；(iii) critic 看 cmd 已足够估值 |
| z 是否被 normalizer | **不被** | z 已经是 encoder 输出（含隐式 BN），二次 normalize 会破坏 axial bases 的几何含义 |
| axial bases 学习 vs 固定 | **学习** | 固定（如 cmd 三轴的 one-hot embedding）会强加先验几何，约束 encoder 找其他子空间表达步态 |
| RnC vs Triplet vs SupCon | **RnC** | Triplet 需要 mining，SupCon 需要离散类；连续 cmd 度量天然适配 RnC |
| L_lip hinge vs MSE | **hinge** | MSE 强制 $d_Z = L|\Delta\lambda|$，会压缩 cmd 子空间外方差；hinge 只要求"足够撑开" |
| r_int = 残差 vs InfoNCE 桶 | **残差** | InfoNCE 桶定义 = cmd 时 policy 无梯度（见 §3.6） |
| SMERL 阈值 β | env reward EMA × 0.8 | 让 r_int 在"基础速度跟踪学好"后才介入；过早 → 偏离 task；过晚 → 难触发 |
| α schedule 形状 | **线性** ramp | 简单稳定；用 cosine ramp 没有显著优势，但调参更复杂 |

---

## 5. 已被否定的数学路线

| 路线 | 否定原因 |
|---|---|
| 在线 InfoNCE + 学习温度 | τ 套利死亡螺旋（V19d-CLP），3 球面坍缩到同一 vector |
| `mask_kind=cmd`（保留 cmd 字段） | $R^2(z \to v_x) = 0.997$ cmd 泄露 |
| 对抗 cmd 解码（adv_cmd） | v2 实测无效，无法消除 cmd 泄露 |
| $z$ 进 critic | RolloutStorage shape lock 冲突，且 advantage 偏差未带来收益 |
| RnC-only（无 L_axial） | spearman = 0.18，metric isometry 不成立 |
| InfoNCE bucket 作为 r_int | policy 无法跨桶 → 0 梯度（见 §3.6） |
| 在线 EMA slow encoder（先尝试） | V22a 还没拿出 baseline，先冻结验证再放开 |

---

## 6. 一句话回到工程

V22b 的设计把所有"对比表征是否有效"的判断都收口到一个可监控量：

$$
\bar r_\text{int}(t) = \frac{1}{N \cdot T} \sum_{e, \tau \in \text{batch}} r_\text{int}(z^e_\tau, v^e_\tau)
$$

——若该量在 SMERL gate 打开后稳定 > 0 且 policy 不退化，则表征学习线**整体闭环成功**。这是接下来 V22b 训练的唯一首要观测指标。
