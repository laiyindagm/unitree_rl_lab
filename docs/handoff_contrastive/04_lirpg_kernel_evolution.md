# V21 Tracking-Reward 主线：LIRPG 与可学习核函数（04）

> 本文档与 `00_overview.md` ~ `03_codebase_and_handoff.md`（contrastive / V22 路线）**正交互补**。
> 目标读者：接手 V21k → V21l → V21m **可学习速度跟踪核**这一支线的工程师与研究者。
> 写作日期：2026-05-08。覆盖到 V21m 的代码已实现 + smoke 通过；V21m 训练正在进行（screen `g1_m`）。

---

## 0. 一句话定位

把 V21f2 起的"固定核函数 tracking reward"改造为**在线 meta-learning 的 reward shaping**：用一个小 MLP 在 PPO 每个 rollout 末做一步梯度下降，自适应调整速度跟踪 reward 的形状（slope 或 sigma），目标是改善低速跟踪精度并消除 lin/ang 通道的 reward 量级偏置。

这条主线**不依赖** contrastive encoder、不引入 latent，**仅修改 reward function 与一个轻量 PPO 子类**。它是 V20l → V21f2 的直接延续，也是当前 G1 15-DOF 旋转任务族的"reward 设计实验台"。

---

## 1. 时间轴 & 演化逻辑

```
V20l (milestone)            标准 exp 核 + 3-mode token，sim2sim 通过
   ↓
V21c                        hybrid_low_speed (低速 piecewise)，引入 TRACK_STD = sqrt(0.25)
   ↓
V21f / V21f2                exp 核 + gait/feet shaping 衰减 (iter 6k→12k)；V21f2 修复 V21f 的 step↔iter 单位 bug
                              → baseline lin: error_xy=0.36, ang: error_yaw=0.54 @ 20k
   ↓
V21k (固定可解释核)          constant-leaky-linear:  r = leaky_{0.1}(1 − e/b_abs)
                              b_abs = 1.5669 · TRACK_STD = 0.7834 (slope = 1/b_abs ≈ 1.276)
                              → ang 显著优于 V21f2，lin 退步 (0.53)
   ↓
V21l (LIRPG slope)           r_phi = leaky_{0.1}(1 − e · f_phi(v, v_cmd))
                              f_phi = softplus(MLP(v, v_cmd)).clamp(8); 初值 f≈1.276 复刻 V21k
                              meta_lr=2e-5, prior_l2=0.1, warmup=6000 iter
                              → ang 进一步改善 (error_yaw=0.26)，lin 停滞 (0.66, 比 V21k 还差 +25%)
                              → 加入: (a) e·f 单调性 ReLU 罚项 mono_coef;
                                       (b) 用户手动改权重 lin=1.0/ang=2.0
   ↓
V21m (LIRPG Gaussian sigma)  r = exp(−|v − v_cmd|² / σ(v_cmd)²)
                              σ = softplus(MLP(v_cmd)).clamp(0.05, 4.0); 初值 σ_0 = 0.5
                              C^∞ 单调（无需 mono_coef），σ 仅与 v_cmd 相关
                              权重 1.5:1.5
                              → 已实现 + smoke 通过；正在训练
```

> 与 V22 contrastive 路线的关系：**完全独立**。V22 改的是 actor obs（注入 z_gait）+ intrinsic reward (||z − g*||²)；V21k/l/m 改的是**外部 task reward 的核函数**。两条路线可以最终合并（V21m 训出来的 σ MLP 与 V22b 的 frozen encoder 可共存于同一 PPO），但当前两支独立推进。

---

## 2. 问题与目标

### 2.1 核心问题（来自 V20l roadmap）
1. **改善低速跟踪相对精度** `|v − v_cmd| / |v_cmd|`
2. **消除 lin/ang 通道间的 reward 量级失衡**（V21c 起 ang weight 是 lin 的 3×）
3. **避免 sigma curriculum / piecewise function 等手工调参**

### 2.2 V21k → V21m 的形式化目标
设
- $v_t \in \mathbb{R}^{d_v}$：当前速度（lin: $d=2$，ang: $d=1$）
- $v_t^{\text{cmd}} \in \mathbb{R}^{d_v}$：命令
- $e_t = \|v_t - v_t^{\text{cmd}}\|$：误差范数

要求 $r_\phi(v, v^{\text{cmd}}) \in [0, 1]$ 满足

| 性质 | V21k 固定核 | V21l slope MLP | V21m Gaussian σ MLP |
|---|---|---|---|
| 零误差归一 $e=0 \Rightarrow r=1$ | ✓ | ✓ | ✓ |
| 单调 $\partial r/\partial e \le 0$ | ✓ | △（需 mono_coef 软约束） | ✓（结构保证）|
| 可解释 | ✓ slope=1/b_abs | △ | ✓ σ=接受宽度 |
| 在线自适应 | ✗ | ✓ | ✓ |
| 状态依赖 | — | (v, v_cmd) | 仅 v_cmd |

V21m 比 V21l 更强的结构假设：σ 只与命令有关，与瞬时 $v$ 解耦。理由：sigma 表达"在该命令下我们愿意接受多大误差"，是命令空间属性；同时这避免了 V21l 中 $f_\phi(v,v_{\text{cmd}})$ 可能违反 $d(ef)/de \ge 0$ 的潜在漏洞。

---

## 3. 数学定义与推导

### 3.1 V21k：固定 leaky-linear 核（baseline）

$$
r_{\text{V21k}}(e) = \text{leaky}_{\alpha=0.1}\!\left(1 - \frac{e}{b_{\text{abs}}}\right)
$$

参数 $b_{\text{abs}} = c \cdot \sigma$，$\sigma = \text{TRACK\_STD} = \sqrt{0.25} = 0.5$，常数 $c = 1.5669$ 是方程 $(2v+1)e^{-v} = 1$ 的解（保证 $r_{\text{V21k}}(e) \le e^{-e^2/\sigma^2}$ 对 cmd=0 严格成立，即 leaky 核全局不超过 V21f2 的 exp 核）。
得 $b_{\text{abs}} = 0.7834$，slope $= 1/b_{\text{abs}} \approx 1.2764$。

### 3.2 V21l：LIRPG with slope MLP

学习一个标量函数 $f_\phi: \mathbb{R}^{2 d_v} \to \mathbb{R}_{>0}$：

$$
f_\phi(v, v^{\text{cmd}}) = \min\!\left(\text{softplus}\big(\text{MLP}_\phi([v; v^{\text{cmd}}])\big),\; f_{\max}\right)
$$

$$
r_\phi(v, v^{\text{cmd}}) = \text{leaky}_{0.1}\!\left(1 - e \cdot f_\phi(v, v^{\text{cmd}})\right), \quad e = \|v - v^{\text{cmd}}\|
$$

MLP 结构：`(d_in → 64 → ELU → 64 → ELU → 1)`，d_in $= 4$ for lin (vx,vy,cx,cy)，$= 2$ for ang。
初值：`_init_to_prior` 在 (−1.5, 1.5) 命令空间随机采样，500 步 MSE 拟合 $f_\phi \to 1/b_{\text{abs}}$，使开局严格复刻 V21k。

#### 3.2.1 Meta-objective

PPO rollout 结束（$T \cdot N$ 个样本），定义 task return（**绝对+相对误差**复合信号）：

$$
\tilde r_t^{\text{task}} = -\left(e_t + \frac{e_t}{\max(\|v_t^{\text{cmd}}\|, 10^{-2})}\right)
$$

MC discounted advantage（按 done mask 截断）：

$$
A_t = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \tilde r_{t+k}^{\text{task}} \cdot \prod_{j=0}^{k-1}(1 - d_{t+j}), \qquad \hat A_t = \frac{A_t - \bar A}{\sigma_A + \epsilon}
$$

Meta loss（$\hat A$ detach）：

$$
\mathcal{L}_{\text{meta}}(\phi) = -\mathbb{E}_t\!\left[\hat A_t \cdot r_\phi(v_t, v_t^{\text{cmd}})\right] + \lambda_{\text{prior}}\|\phi - \phi_0\|^2 + \lambda_{\text{mono}} \cdot \mathcal{L}_{\text{mono}}
$$

参数空间 trust region：$\lambda_{\text{prior}} = 0.1$，$\phi_0$ 是 init 后的 snapshot。一步 Adam（lr=2e-5）+ grad norm clip 1.0。

#### 3.2.2 单调性软约束（V21l 后期加入）

要求 $\partial(e f_\phi)/\partial e \ge 0$。沿误差方向方向导数：

$$
\hat e = \frac{v - v^{\text{cmd}}}{\max(e, 10^{-6})}, \qquad \frac{d(e f_\phi)}{de} = f_\phi + e \cdot (\nabla_v f_\phi \cdot \hat e)
$$

$$
\mathcal{L}_{\text{mono}} = \mathbb{E}_t\!\left[\mathbb{1}_{e_t > 10^{-3}} \cdot \text{ReLU}\!\big(-(f_\phi + e_t \nabla_{v_t} f_\phi \cdot \hat e_t)\big)^2\right]
$$

$\nabla_v f_\phi$ 通过 `torch.autograd.grad(..., create_graph=True)` 求得。`mono_coef` 在 V21l 中设为 1.0；其他变体默认 0.0。

### 3.3 V21m：Gaussian σ MLP

学习 $\sigma_\phi: \mathbb{R}^{d_v} \to \mathbb{R}_{>0}$，**输入只有 v_cmd**：

$$
\sigma_\phi(v^{\text{cmd}}) = \text{clamp}\Big(\text{softplus}\big(\text{MLP}_\phi(v^{\text{cmd}})\big),\; 0.05,\; 4.0\Big)
$$

$$
r_\phi(v, v^{\text{cmd}}) = \exp\!\left(-\frac{\|v - v^{\text{cmd}}\|^2}{\sigma_\phi(v^{\text{cmd}})^2}\right)
$$

性质：

- $\partial r/\partial e = -2e/\sigma^2 \cdot r \le 0$，**单调由结构保证**（无需 mono_coef）
- $r$ 关于 $e$ 是 $C^\infty$（无 leaky 不可导点）
- $\sigma_\phi$ 仅依赖 $v^{\text{cmd}}$ → 跨 rollout 稳定
- 初值 $\sigma_0 = 0.5$ = TRACK_STD，与 V21f2 exp 核**等价**

Meta-objective 与 V21l 同：相同 task return、相同 GAE、相同 prior L2、Adam lr=2e-5。**没有 mono term**。日志记 `sigma_mean`。

#### 3.3.1 σ clamp 上下限

| σ | $r(e=\sigma)$ | $r(e=2\sigma)$ | 物理意义 |
|---|---:|---:|---|
| 0.05 | 0.368 | 0.018 | 极严格 |
| 0.5 (init) | 0.368 | 0.018 | 与 V21f2 同形 |
| 4.0 | 0.368 | 0.018 | 极宽松 |

clamp 防退化：太小→梯度消失；太大→reward 几乎处处=1 失去区分度。

---

## 4. 关键诊断：为什么 V21l lin 通道停滞？

V21l @18.5k 实测：

| metric | V21k @9k | V21l @9k | V21l @18.5k |
|---|---:|---:|---:|
| error_vel_xy ⬇ | 0.532 | 0.665 (+25%) | 0.657 (停滞) |
| error_vel_yaw ⬇ | 0.407 | 0.380 | **0.263** |
| f_phi_lin | 1.276 | 1.41 | 1.51 (+18%) |
| f_phi_ang | 1.276 | 1.82 | **2.12 (+66%)** |
| Loss/entropy | -2.22 | -3.05 | **-8.54 (塌缩)** |

诊断（详见 `/memories/repo/v21_tb_comparison.md`）：

1. **reward weight 1:3 不对称**：`track_lin_vel_xy=1.0`、`track_ang_vel_z=3.0`，但 LIRPG 两通道用同 meta_lr。
2. **正反馈 vs 负反馈循环**：
   - **ang**：weight 大 → policy 响应明显 → adv 转正 → meta 看到"提高 f 提升了 policy" → f↑ → policy 再适应 → **正反馈**
   - **lin**：weight 小 → policy 不愿付代价 → adv 持续负 → meta 看到"提高 f 让 reward 更糟" → f 不动 → policy 也不动 → **死循环**
3. **entropy 塌缩**：leaky 核梯度比 exp 核陡 + LIRPG 自适应放大 ang，σ_policy 收敛到接近确定性。

V21l 后续两件事：(a) `mono_coef=1.0`；(b) 用户手动改权重 1.0/2.0。
V21m 同时回应：σ 仅依赖 v_cmd → 跨 rollout 稳定；指数核 → 梯度不陡；权重 1.5:1.5（更对称）。

---

## 5. 当前代码地图

> 路径相对 workspace 根 `/root/workspace/unitree_rl_lab/`。

### 5.1 核心模块

| 文件 | 作用 | 关键 symbol |
|---|---|---|
| [source/unitree_rl_lab/unitree_rl_lab/utils/intrinsic_reward.py](../../source/unitree_rl_lab/unitree_rl_lab/utils/intrinsic_reward.py) | LIRPG channel + registry | `IntrinsicRewardChannel` (V21l slope), `GaussianIntrinsicRewardChannel` (V21m σ), `_REGISTRY`, `get_or_create()`, `get_or_create_gauss()`, `reset_registry()` |
| [source/unitree_rl_lab/unitree_rl_lab/utils/lirpg_ppo.py](../../source/unitree_rl_lab/unitree_rl_lab/utils/lirpg_ppo.py) | PPO 子类，挂 record_dones + meta_update | `LirpgVelocityEstimatorPPO`, `lirpg_warmup_iters` (默认 6000) |
| [source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py](../../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py) | reward 函数 | `track_lin_vel_xy_constant_leaky` (V21k), `track_lin_vel_xy_intrinsic` (V21l), `track_lin_vel_xy_intrinsic_sigma` (V21m); ang_z 同名变体；`LINEAR_REL_B_RATIO=1.5669` |

### 5.2 实验配置

| 文件 | 任务 ID | 关键开关 |
|---|---|---|
| `tasks/.../velocity_env_cfg_rot_v21f2.py` | `Unitree-G1-15dof-Velocity-Rot-V21f2` | exp 核 + gait shaping 衰减 |
| `tasks/.../velocity_env_cfg_rot_v21k.py` | `Unitree-G1-15dof-Velocity-Rot-V21k` | constant_leaky；权重 1.0 / 3.0 |
| `tasks/.../velocity_env_cfg_rot_v21l.py` | `Unitree-G1-15dof-Velocity-Rot-V21l` | LIRPG slope；`META_LR=2e-5`, `PRIOR_PARAM_L2_COEF=0.1`, `MONO_COEF=1.0`；权重 1.0 / **2.0** |
| `tasks/.../velocity_env_cfg_rot_v21m.py` | `Unitree-G1-15dof-Velocity-Rot-V21m` | LIRPG Gaussian σ；`SIGMA_0=0.5`；权重 1.5 / 1.5 |

注册入口 [`tasks/.../15dof_rot/__init__.py`](../../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/__init__.py)。

### 5.3 PPO Runner

[`tasks/locomotion/agents/rsl_rl_ppo_cfg.py`](../../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py)：

- `G115DofV21eVelocityEstimatorPPORunnerCfg`：基线
- `G115DofV21lLirpgRunnerCfg(...)`：`__post_init__` 替换 `algorithm.class_name = "unitree_rl_lab.utils.lirpg_ppo:LirpgVelocityEstimatorPPO"`
- `G115DofV21mLirpgRunnerCfg(G115DofV21lLirpgRunnerCfg)`：直接继承；channel 通过 reward 函数中的 `get_or_create_gauss("lin_xy_gauss", ...)` 动态创建；与 V21l 的 `"lin_xy"` / `"ang_z"` 共存于同一 registry

---

## 6. 运行时数据流

```
        env.step()
            │
            ▼
     reward terms (manager 调用)
       │   ┌─ track_lin_vel_xy_intrinsic_sigma(env, cmd, ...)
       │   │     └─ ir.get_or_create_gauss("lin_xy_gauss", ...) → channel
       │   │     └─ channel.evaluate(v, v_cmd)            # @torch.no_grad
       │   │           ├─ 记录 (err, v_cmd) 到 channel buffer
       │   │           └─ 返回 r ∈ [0, 1]
       │   └─ ... ang_z_gauss 同理
       ▼
     env.step() 返回 (obs, rew, done, ...)
            │
            ▼
   LirpgVelocityEstimatorPPO.process_env_step(rew, done, infos)
            ├─ super().process_env_step(...)
            └─ for chan in ir.all_channels().values():
                   chan.record_dones(done)
   ... 重复 num_steps_per_env=24 次 (一个 rollout)
            │
            ▼
   LirpgVelocityEstimatorPPO.update()
            ├─ super().update()  (标准 PPO)
            └─ if iter ≥ lirpg_warmup_iters (6000):
                   for name, chan in ir.all_channels().items():
                       result = chan.meta_update(gamma=0.99, lam=0.95)
                       meta_logs[f"lirpg/{name}/{k}"] = v
```

要点：

1. `evaluate` 在 `@torch.no_grad` 下运行；`meta_update` 用 `with torch.inference_mode(False), torch.enable_grad()` 跳出 inference_mode。
2. `_REGISTRY` 是**全局单例**。切换实验须显式 `intrinsic_reward.reset_registry()` 或重启进程。
3. `record_dones` 必须每 env step 都调一次。
4. warmup 期间 `meta_update` 不被调用，但 buffer 仍清空防泄露。

---

## 7. TensorBoard 日志键

V21l/V21m 在标准 PPO 之外新增（前缀 `lirpg/<channel_name>/`）：

| key | 含义 |
|---|---|
| `lirpg/lin_xy/meta_loss` (V21l) / `lirpg/lin_xy_gauss/meta_loss` (V21m) | $-\mathbb{E}[\hat A \cdot r_\phi]$ |
| `lirpg/.../prior_loss` | $\lambda_{\text{prior}} \|\phi - \phi_0\|^2$ |
| `lirpg/.../mono_loss` | V21l 单调违反罚项（V21m 无）|
| `lirpg/.../adv_mean`, `adv_std` | task return advantage 统计 |
| `lirpg/.../r_phi_mean` | 当前 channel 平均 reward |
| `lirpg/.../f_phi_mean` (V21l) / `lirpg/.../sigma_mean` (V21m) | 学习参数批均值 |
| `lirpg/.../T` | rollout 长度（应 = 24）|
| `lirpg/warmup_remaining` | warmup 倒计时 |

**监控建议**：

- `f_phi_mean` / `sigma_mean` 仅记 mean → 丢失分布；建议追加 std/min/max 三个标量。
- `meta_loss` 应在 warmup 后稳定下降；持续震荡 → meta_lr 太高。
- `prior_loss / total_loss` >50% 长期保持 → trust region 太强；<1% → 太弱。
- `mono_loss` (V21l) 应在前几次 update 内归零；持续不为 0 → 增大 mono_coef 或回退至 V21m。

---

## 8. 离线评测脚本 `scripts/rsl_rl/eval_tracking.py`

> 这是**专门为 V21k/l/m 主线设计**的稳态跟踪误差评测器，区别于 train/play 中的在线 metric。

### 8.1 它解决的问题

训练日志里的 `error_vel_xy` 是有偏的：

- 分母是 `max_command_step`（默认 600），包含命令重采样后的瞬态；
- curriculum 中的 `lin_err` 只取 episode 末单点 → 高方差，且低速误差被 `accuracy_cmd_min` 截断。

`eval_tracking.py` 的方法：**固定 command 整段不变**，跑完 warmup 让机器人到稳态后再统计 measure 段的均值/方差/相对误差，并按 env 的 cumulative alive mask 排除已摔倒的 env。同时按 0.1 m/s（rad/s）粒度 sweep 完整训练命令域：

- `pure_vx`：vx ∈ [-0.8, 1.5]（24 bins），vy=0, wz=0
- `pure_wz`：wz ∈ [-0.8, 0.8]（17 bins）
- `pure_vy`：vy ∈ [-0.5, 0.5]（11 bins，可用 `--no_vy` 跳过）

### 8.2 并行结构（关键设计）

**所有 bin 同 rollout 并行**：env 总数 = `max_n_bins * envs_per_bin`，每个 bin 独占连续 `envs_per_bin` 个 env，用 `_force_commands` 把 `command_manager._terms["base_velocity"].vel_command_b` 写死并把 `time_left` 设为 1e6 阻止重采样。
单次 sweep 仅花 `(warmup + measure)` 个 sim step，与 bin 数无关；24 bin × 4 env/bin × 600 step 总成本 ≈ 等同跑 1 个 600 step 的 96-env env。

### 8.3 输出指标

每 (sweep, bin, checkpoint) 三元组记录：

- `abs_err_lin`, `abs_err_ang`：稳态绝对误差均值（剔除 fallen env）
- `std_err_lin`, `std_err_ang`：稳态误差标准差（同 env 内 step-wise variance + 跨 env 平均）
- `rel_err_lin`, `rel_err_ang`：相对误差 = abs / |cmd|（|cmd|<0.01 时为 nan）
- `fall_rate`：该 bin 内 env 终止比例
- `mean_actual_vx/vy/wz`：稳态实际速度均值（用于检查偏置/死区）
- `total_steps`：该 bin alive step 数（衡量统计有效样本量）

reward-convention 一致：`lin_vel = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)[:, :2]`（heading 系），`ang_vel = root_ang_vel_b[:, 2]`（body z）。

### 8.4 多 checkpoint 同环境对比

`--checkpoints` 接多个 `model_*.pt`，每个 checkpoint 在**同一个 env 实例**上轮流加载并复评同一组 sweeps；表格按 bin 横向排列各 checkpoint 列，便于 V21k vs V21l vs V21m 同等条件对比。`--labels` 给显示名（默认 basename）。

### 8.5 Runner 加载与自定义类支持

`_get_runner_cfg` 从 `gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]` 反射出 cfg 类。`_make_runner` 按 `runner_cfg.class_name` dispatch：

- `"OnPolicyRunner"` → 标准 `rsl_rl.runners.OnPolicyRunner`
- `"DistillationRunner"` → `rsl_rl.runners.DistillationRunner`
- `"unitree_rl_lab.utils.lirpg_ppo:LirpgVelocityEstimatorPPO"` 等 `mod:cls` 字符串 → `importlib.import_module + getattr`（**V21l/V21m 走这条**）

随后 `runner.load(checkpoint_path)` + `runner.get_inference_policy()` 拿到 callable policy。**LIRPG 的 meta MLP 不参与评测**（policy 只取 actor.act_inference），与 sim2sim 部署语义一致。

### 8.6 v_hat 消融

`--zero_vhat_labels LBL1 LBL2 ...` 列出的 checkpoint 在评测时会通过 `_patch_zero_vhat` 给 `actor.velocity_head` 注册一个 forward hook 把输出置零，等价于"velocity estimator 失效"的消融——用于回答"policy 是否真的依赖 v_hat"。
仅对带 `velocity_head` 的 actor（如 VelocityEstimatorPPO 系列）生效；其他模型会打 WARNING 并 no-op。

### 8.7 典型用法

```bash
# 三方对比 V21k vs V21l vs V21m
./unitree_rl_lab.sh -p scripts/rsl_rl/eval_tracking.py \
    --task Unitree-G1-15dof-Velocity-Rot-V21m \
    --checkpoints \
        logs/rsl_rl/unitree_g1_15dof_velocity_rot/<v21k_run>/model_19999.pt \
        logs/rsl_rl/unitree_g1_15dof_velocity_rot/<v21l_run>/model_18500.pt \
        logs/rsl_rl/unitree_g1_15dof_velocity_rot/<v21m_run>/model_19999.pt \
    --labels V21k@20k V21l@18.5k V21m@20k \
    --envs_per_bin 4 \
    --warmup_steps 200 \
    --measure_steps 400 \
    --output_csv /tmp/v21_compare.csv \
    --headless
```

> **关键约束**：`--task` 决定 env cfg 与 obs layout。所有 checkpoint 必须与 `--task` 的 obs/action 维度一致。跨 obs 维度对比（例如 V21f2 vs V21l）须**在每个 task 下分别跑一次再合并 CSV**，不能一次塞进 `--checkpoints`。

### 8.8 与训练日志的关系（解读判据）

| 指标 | 训练 TB | eval_tracking.py |
|---|---|---|
| 误差度量 | `error_vel_xy` (有偏) | `abs_err_lin` (稳态无偏) |
| 命令分布 | curriculum 实时 | 网格 sweep 全覆盖 |
| 摔倒处理 | 重置后立即继续累计 | env 终止后排除 |
| 低速精度 | 受 `accuracy_cmd_min` 截断 | `rel_err` 直接暴露 |
| 跨 ckpt 对比 | 不同 run 不同种子 | **同 env、同种子、轮流加载** |

V21k vs V21l 的"+25% lin 退步"判定（§4 表格）就是用这个脚本得出的。后续 V21m 的接受/否决也以此为准。

---

## 9. 已知陷阱与编辑约束

### 9.1 文件写入限制

`source/unitree_rl_lab/unitree_rl_lab/...` 路径下的文件，agent 工具 `replace_string_in_file` 与 `create_file` **可能报"文件不存在"** 即使 `grep_search` 能命中。绕路：用终端 `cat > file << EOF`、`sed -i`、`python3 - << PYEOF`。**本文档本身就是用 `python3` heredoc 写入的**。

### 9.2 PyTorch inference_mode

reward 函数运行在 `inference_mode`。`IntrinsicRewardChannel.__init__`、`_init_to_prior`、`meta_update`、mono_loss 计算都必须用：

```python
with torch.inference_mode(False), torch.enable_grad():
    ...
```

不止是 `enable_grad()`——必须先 `inference_mode(False)`。

### 9.3 Channel registry 单例

`get_or_create("lin_xy", ...)` 第二次调用时**所有 kwargs 被忽略**（已存在则复用）。修改超参后必须显式 `reset_registry()` 或重启进程。

### 9.4 V21k 权重历史

V21c → V21k 沿用 `track_lin_vel_xy = 1.0`、`track_ang_vel_z = 0.5 * 2 * 3 = 3.0`（历史多次乘数累积，非有意设计）。V21l 改为 1.0 / 2.0；V21m 设计 1.5 / 1.5。新派生实验请先确认权重是否对当前假设合理。

---

## 10. 当前实现状态（2026-05-08）

| 组件 | 状态 |
|---|---|
| V21k env + reward + 注册 | ✓ 训练完成（1 次 9k iter） |
| V21l env + reward + 注册 + LIRPG slope MLP | ✓ 训练完成（1 次 18.5k iter） |
| V21l 单调性约束 mono_coef | ✓ 代码已加，**未与 mono_coef=0 ablation 对比** |
| V21l 权重调整 lin=1.0/ang=2.0 | ✓ 配置已改，**未重训** |
| V21m env + reward + 注册 + Gaussian σ MLP | ✓ 代码 + smoke 通过，正在训练（screen `g1_m`） |
| `f_phi`/`sigma` 分布日志（std/max） | ✗ 仅记 mean，待加 |
| sim2sim 测试 V21l/V21m | ✗ 待做 |
| eval_tracking.py 三方对比 (V21k/l/m) | ✗ 待 V21m 训完后做 |

---

## 11. 推荐的下一步实验（按优先级）

### 优先级 1：V21m 完整训练 + eval_tracking 三方对比

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-15dof-Velocity-Rot-V21m \
  --headless --max_iterations 20000
```

判据（与 `/memories/repo/v21_tb_comparison.md` 一致）：
- `abs_err_lin`（pure_vx sweep 平均）≤ 0.45 → 优于 V21l 0.66
- `abs_err_ang`（pure_wz sweep 平均）≤ 0.40 → 持平 V21l 优势
- `rel_err_lin` 在低速 bin (|vx|≤0.3) ≤ 1.0 → 真正解决相对精度问题
- entropy ≥ -5

### 优先级 2：V21l mono_coef=0 ablation

把 `MONO_COEF` 改为 0，重训 V21l。用 eval_tracking 同 task 同 env 复评，对比 `mono_loss` 曲线和稳态误差差。

### 优先级 3：V21m σ MLP 输入扩展

若 V21m σ 各命令趋同到一个值（`sigma_mean` 各 bin 方差小），扩展：
(a) **V21m-A**：σ 输入 = (v_cmd, ‖v_cmd‖)
(b) **V21m-B**：σ 输入 = (v_cmd, gait_mode_token)，与 3-mode isolation 硬约束契合

### 优先级 4：V21n（不动 LIRPG，纯调权重）

不引入 LIRPG，仅在 V21k 基础上把 `track_lin_vel_xy weight: 1.0 → 3.0`。若 V21n 把 `abs_err_lin` 拉到 ≤ 0.40，则证明**问题就是权重**，LIRPG 可能多余 → 终结此条主线。

### 优先级 5：与 V22b 合并（V22c）

V22b 的 frozen z encoder + V21m 的 σ MLP 同时启用。理论上正交（一个改 obs/intrinsic，一个改 task reward），需测 LIRPG meta-loop 是否被 z 注入扰乱。

---

## 12. 与 V20l roadmap 的接口

- **V20l 是基础**：V21k/l/m 全部基于 V21f2 → V21e → V20g rewards 链，最终 base 是 V20l 的 reward 设计 + V20l 的 3-mode token observation。
- **3-way mode isolation 硬约束**（standing / pure_wz / other）在 LIRPG 路线中**未被破坏**：reward 函数只看 `(v, v_cmd)`，token 仍在 obs 里供 actor 使用。
- **Sim2sim 部署**：V21l / V21m 的 actor / critic / observation 与 V21f2 完全一致，部署侧无需改动；**meta MLP 不参与部署**（LIRPG 仅训练时用）。这是 V21k/l/m 相对 V22 路线的最大工程优势——zero deploy cost。

---

## 13. 阅读路径建议

1. `00_overview.md`（contrastive 路线，了解整体术语）
2. **本文 §1 时间轴 + §3 数学定义**
3. [`source/unitree_rl_lab/unitree_rl_lab/utils/intrinsic_reward.py`](../../source/unitree_rl_lab/unitree_rl_lab/utils/intrinsic_reward.py)（≈350 行内核代码）
4. [`source/unitree_rl_lab/unitree_rl_lab/utils/lirpg_ppo.py`](../../source/unitree_rl_lab/unitree_rl_lab/utils/lirpg_ppo.py)（≈70 行 PPO hook）
5. V21l + V21m 两个 env_cfg 文件（各 ≤70 行）
6. 本文 §4 诊断 + §8 eval 脚本 + §11 推荐实验
7. `/memories/repo/v21_tb_comparison.md`（最新一手 TB 对比数据）

整个 LIRPG 子系统**约 500 行 Python**，是 contrastive (V22) 路线的几分之一；适合作为入门改造点。
