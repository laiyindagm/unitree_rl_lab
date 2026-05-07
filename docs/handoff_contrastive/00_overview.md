# 对比表征 → CIC 内在奖励：研究线总览（00）

> 目标读者：所有想接手"用对比学习给 G1 humanoid 的 RL policy 注入步态多样性"这条线的人。
> 阅读建议：先读本篇 5 分钟扫一遍；按文末"按角色导航"挑下一篇。

---

## 1. 一句话定位

**让 RL policy 在不同速度命令 (vx, vy, ωz) 下产生有差异的步态（sub-style emergence）**，方法是：
1. **离线**用对比学习预训练一个 **冻结**的步态隐变量编码器 `f_θ: history → z ∈ ℝ^32`；
2. **在线** RL 训练时，把 `z` 拼到 actor latent，并在 reward 中加 **"利用 z"** 的内在奖励 `r_int(z, v_cmd)`；
3. 用 SMERL 门控保证基础速度跟踪学好之前不发放 `r_int`。

外在 reward 仍然是速度跟踪 + 体态约束（V21g 标准），表征只负责"在 reward 之外撑开一个能让步态分化的额外信号通道"。

---

## 2. 时间轴（4 条主线）

```
2026-04 上旬  V19d-CLP        在线 contrastive (TCN/Transformer) + obs 预测       —— 失败，废弃
2026-04 中旬  frnc_segment v1  离线 RnC，cmd-distance 标签                          —— mask 泄露，废弃
2026-04 下旬  frnc_segment v2  v1 + L_axial(A1) + adv_cmd                          —— adv 无效，3 球面淘汰
2026-04 末    frnc_segment v3  RnC + L_axial + L_lip + L_prop, sigma_cmd 显式存盘  —— winner: v3_full
2026-05 上旬  V22a             frozen v3_full 接入 actor，z 进 obs（不发 r_int）     —— 与 V21g 打平
2026-05 上旬  V22b             V22a + axial-residual r_int + SMERL gate             —— smoke OK，待训
```

V19d-CLP 与 frnc_segment v1/v2/v3 是**表征学习线**；V22a/V22b 是**RL 集成线**。中间的转折在于：在线 contrastive 失败后，把表征学习从 RL loop 中剥离到离线阶段，用 frozen encoder 解决"z 跟着 policy 漂移导致 anchor 失稳"的根本问题。

---

## 3. 5 版本一表概览（详见 02 章）

| 版本 | 信号源 | 损失 | 关键超参 | 结果 | 教训 |
|---|---|---|---|---|---|
| V19d-CLP | 在线 obs history | InfoNCE×3 球面 + obs 预测 | τ=0.07 (collapse), 3 球面 | nce_loss 7.84 > random 6.24 | 在线 contrastive + 学温度 → τ 套利死亡螺旋；z 跟 policy 漂移 |
| v1 | 离线 segment | RnC (cmd-distance 标签) | mask_kind=cmd loose | R²(z→vx)=0.997 cmd 泄露 | mask 必须严格屏蔽 cmd 相关字段 |
| v2 | 离线 segment | v1 + L_axial + adv_cmd | 3 球面 projection | adv_cmd 实证无效 | 对抗解码器解 cmd 泄露不可行 |
| **v3 (winner)** | 离线 segment | RnC + L_axial + L_lip + L_prop | d_gait=32, σ_cmd 存盘 | spearman=0.877, lip_med=0.624, R²(z→cmd)=0.974 | axial bases 只张成 z 中 cmd 子空间（axial_R²<0.15 是设计内行为，非 bug） |
| V22a | RL 集成 | — (frozen z 进 actor) | gait_dim=32, z_buffer_len=32 | 与 V21g 打平，low-speed 略优 | z 是被旁路的，actor 没有梯度压力去"用" z |
| V22b | RL + r_int | r_int = ‖z−g*(v)‖²/d, mask ‖v‖_W<ε | α∈[0,0.02] iter 200→2000, gate σ(200(EMA−0.045)) | smoke 5 iter PASSED | metric 残差 + SMERL 门控是给 actor 学 z 的最小可行信号 |

---

## 4. 数据流（运行时鸟瞰）

```
                                 ┌─────────────────────────┐
 obs history (5×59)──flat───────►│   actor (transformer    │──► action
            │                    │   latent + MLP)         │
            │                    └────────────▲────────────┘
            │  rolling buffer (32, 295)       │
            │  per-env (V22a/b only)          │ z_gait (32d)
            ▼                                 │
   ┌────────────────────┐            ┌────────┴────────┐
   │ FrozenSegmentEnc.  │── z ─────► │  cache (V22b):  │
   │   V3 ckpt frozen   │            │  _last_z, _last_cmd│
   └────────────────────┘            └────────┬────────┘
                                              │
   v_cmd = obs["policy"][:, 42:45] ──axial_predict(v, σ)──► g*(v)
                                              │
                                ┌─────────────┴──────────────┐
                                │ r_int = ‖z − g*‖² / d_gait │  (V22b only)
                                │  mask: ‖v‖_W > ε           │
                                └─────────────┬──────────────┘
                                              │
                                  α(t)·gate(EMA_r)·r_int
                                              │
   r_env (V21g 标准 reward) ──────────────────┴──► r_total → PPO
```

关键点：
- **encoder 完全冻结**，从 V3 ckpt 加载，eval 模式 + `requires_grad=False`。
- **z 不进 critic**（避免 advantage 偏差 + RolloutStorage TensorDict shape lock）。
- **z 不被 obs_normalizer 处理**（不在 obs_groups 里）。
- **V22a 与 V22b 用同一个 env**（即 V21g 复用）；差别仅在 PPO 算法子类。

---

## 5. 与 V20l/V21g 主线的交接点（≤1 页）

V20l 是另一条独立主线：解决 mode-token policy 的 `pure_xy` sim2sim 失败。V22 系不依赖 V20l 的完成度——

| 要点 | 说明 |
|---|---|
| Base | V22a/V22b 都基于 **V21g**（V20l 的 reward 后续改进版），不是 V20l 本身。 |
| 硬约束 | **3-way mode isolation `{standing, pure_wz, other}` 永远保留**（用户硬约束，不允许去 token）。 |
| 部署 | 当前 V22a/b 的 ONNX export 抛 `NotImplementedError`，需要后续把 encoder + rolling buffer 一起烘焙进 C++ 端 `State_RLBase`。V20l 的 sim2sim 修通后，V22 部署应共享同一套 deploy pipeline。 |
| 演进顺序 | 若 V20l 切换 reward base，V22 需要重新 retrain；当前没有这种依赖。 |
| 评估对照 | V22b 训完后做 V21g / V22a / V22b 三方对照（per-bucket TB），不是与 V20l 对照。 |

---

## 6. 按角色导航

| 你是… | 先看 | 然后看 |
|---|---|---|
| 算法/研究者，关心"为什么这么设计、公式怎么推" | [01_problem_and_math.md](01_problem_and_math.md) | [02_design_evolution.md](02_design_evolution.md) |
| 工程/接手代码，关心"怎么跑、改哪里" | [03_codebase_and_handoff.md](03_codebase_and_handoff.md) | [01_problem_and_math.md](01_problem_and_math.md)（按需查公式） |
| 想理解失败教训、避免踩坑 | [02_design_evolution.md](02_design_evolution.md) §"已弃用方案" | [03_codebase_and_handoff.md](03_codebase_and_handoff.md) §"工具/编辑陷阱" |
| 准备做下一版实验 (V22c+) | [02_design_evolution.md](02_design_evolution.md) | [03_codebase_and_handoff.md](03_codebase_and_handoff.md) §"下一步 roadmap" |

---

## 7. 关键缩写表

| 缩写 | 含义 |
|---|---|
| RnC | Rank-N-Contrast (Zha et al., 2023)，连续标签的 supervised contrastive |
| CIC | Contrastive Intrinsic Control (Laskin et al., 2022)，本项目借用其"用表征做 r_int"的范式，但**不**用其 InfoNCE 形式 |
| SMERL | Self-Motivated Episodic RL gating，task-reward 达标后才发 intrinsic |
| A1 / A2 / Iso | 表征性质：轴向可加 / 幅值 Lipschitz / cmd-距离等距（见 01 章） |
| g*(v) | axial 预测 = `Σ_a ρ_a · E_a`，cmd 在 z 空间的"理想线性投影" |
| z_gait | 编码器输出的步态隐变量，d_gait = 32 |
| sigma_cmd | 预训练时 cmd 三维的 std，作为度量空间权重 W = diag(1/σ²) 存到 ckpt |
