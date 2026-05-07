# 代码地图与交接说明（03）

> 目标读者：负责跑训练 / 改算法 / 部署上机 的工程师。
> 本篇给出"找文件、跑命令、避免踩坑"的所有抓手。

---

## 1. 代码地图

### 1.1 PPO 继承链（rsl_rl 5.0.1）

```
PPO  (rsl_rl 内置)
 └── TransformerPPO                 source/unitree_rl_lab/.../utils/transformer_ppo.py
      └── VelocityEstimatorPPO       source/unitree_rl_lab/.../utils/velocity_estimator_ppo.py
           └── SegmentEncoderVelocityEstimatorPPO   source/unitree_rl_lab/.../utils/segment_encoder_ppo.py    (V22a)
                └── SegmentEncoderCICPPO            source/unitree_rl_lab/.../utils/segment_encoder_cic_ppo.py (V22b)
```

每层只在前一层基础上加最小 diff：
- TransformerPPO：obs sequence 适配 transformer actor。
- VelocityEstimatorPPO：加 vel estimator head + estimator loss。
- SegmentEncoderVelocityEstimatorPPO：维护 per-env rolling buffer (32, 295)，每 act() 调用 `FrozenSegmentEncoder` 把 z 拼到 `obs["z_gait"]`。
- SegmentEncoderCICPPO：override `act` 缓存 `_last_z, _last_cmd`；override `process_env_step` 算 r_int + SMERL gate + α 调度，augment rewards 后调 super。

### 1.2 Actor 链

```
MLPModel                                          rsl_rl 内置
 └── TransformerLatentModel                       source/unitree_rl_lab/.../utils/rsl_rl_transformer_model.py
      └── TransformerLatentGaitModel              same file
```

`TransformerLatentGaitModel` 在 `get_latent_outputs()` 中把 `obs["z_gait"]`（来自 algo 缓存）拼接到 transformer 输出的 latent 上喂给 actor head；critic 不看 z（policy-only）。

### 1.3 Encoder 包装

`source/unitree_rl_lab/unitree_rl_lab/utils/frozen_segment_encoder.py`：
- 加载 ckpt：`logs/frnc_seg_v3/v3_full/encoder.pt`（含 state_dict + config）；
- 注册 buffer `sigma_cmd`（来自 ckpt config，shape [3]）；
- 提供 `forward(obs_window) -> z`（eval, no_grad）；
- 提供 `axial_predict(cmd) -> g*(v)`（V22b 的 r_int 用）。

### 1.4 cfg / 注册 / env

| 文件 | 作用 |
|---|---|
| `source/unitree_rl_lab/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | 算法 cfg 定义：`RslRlSegmentEncoderPpoAlgorithmCfg` (V22a)、`RslRlSegmentEncoderCICPpoAlgorithmCfg` (V22b)；runner cfg：`G115DofV22aSegmentEncoderPPORunnerCfg`、`G115DofV22bSegmentEncoderCICPPORunnerCfg` |
| `.../robots/g1/15dof_rot/__init__.py` | gym 任务注册：`Unitree-G1-15dof-Velocity-Rot-V22a` / `-V22b` |
| `.../robots/g1/15dof_rot/velocity_env_cfg_rot_v22a.py` | re-export V21g `RobotEnvCfg / RobotPlayEnvCfg`（V22a 不动 env） |
| `.../robots/g1/15dof_rot/velocity_env_cfg_rot_v22b.py` | re-export V21g（V22b 也不动 env） |

> V22a/V22b **没有**改 env config——所有差异都在 algo 层，env 完全等价于 V21g（含 3-way mode token、rotation reward、curriculum）。

### 1.5 离线表征预训练

| 文件 | 作用 |
|---|---|
| `scripts/rsl_rl/collect_pretrain_data.py` | dump V21e_strat 的在线 rollout 到 segment 数据集 |
| `scripts/rsl_rl/frnc_gait_features.py` | 离线计算步态属性 (duty_l, yaw, lat, act) for L_prop |
| `scripts/rsl_rl/frnc_segment_pretrain_v3.py` | V3 训练入口；定义 `SegmentEncoderV3` + 5 cfg 套餐 |
| `scripts/rsl_rl/frnc_segment_probe.py` | V3 ckpt 验证：spearman / R² / lip_med / ibvr / axial_R² |
| `logs/frnc_seg_v3/v3_full/encoder.pt` | **V22 系唯一在用的 encoder ckpt** |

---

## 2. 运行时数据流

### 2.1 obs 帧布局（V21g, term-major）

每帧 59 维（policy obs flat），5 帧拼成 295 维：

```
[base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | vel_cmd(3) | lin_speed_token(1)
 | gait_mode_token(3) | joint_pos_rel(15) | joint_vel_rel(15) | last_action(15) | contact(2)] × 5
```

最新帧的 `vel_cmd` slice 在 flat obs 中是 **`[42:45]`**（V22b `act()` 取 cmd 的位置）。

### 2.2 Rolling buffer

`SegmentEncoderVelocityEstimatorPPO` 维护 `_obs_buffer: shape (num_envs, 32, 295)`：
- env reset 时该 env 的 buffer 清零；
- 每 act() 把当前 flat obs append 到 buffer 末尾，pop 最早；
- 调 `encoder(_obs_buffer) -> z_gait: (num_envs, 32)`；
- 把 z 写到 `obs["z_gait"]` + 缓存到 `_last_z`。

### 2.3 V22b r_int 计算时机

```
algo.act(obs)                    # 算 z, 缓存 _last_z _last_cmd
env.step(action)
algo.process_env_step(rewards, dones, ...)     # ← 在这里：
   g_star = encoder.axial_predict(_last_cmd)
   resid  = ((_last_z - g_star)**2).mean(-1)
   mask   = (_last_cmd / sigma_cmd).norm(-1) > eps_cmd
   r_int  = resid * mask
   ema_r  = ema_decay * ema_r + (1 - ema_decay) * rewards
   gate   = sigmoid(kappa * (ema_r - beta))
   alpha  = alpha_max * clip((iter - w0) / (w1 - w0), 0, 1)
   bonus  = alpha * gate * r_int
   super().process_env_step(rewards + bonus, dones, ...)
```

### 2.4 RolloutStorage shape lock 坑

RolloutStorage 在第一次 act 之前由 obs schema 锁定 shape。**必须**在 `construct_algorithm` / runner setup 阶段把零 z_gait 注入 obs，让 storage 把 `z_gait: (num_envs, 32)` 字段建好。否则后续 `storage.add()` 会 shape mismatch。该 fix 已在 V22a 的 `SegmentEncoderVelocityEstimatorPPO.__init__` 里完成，V22b 继承，无需重复。

---

## 3. 关键 cfg 字段

### 3.1 V22a (`G115DofV22aSegmentEncoderPPORunnerCfg`)

```python
algorithm = RslRlSegmentEncoderPpoAlgorithmCfg(
    class_type="unitree_rl_lab.utils.segment_encoder_ppo.SegmentEncoderVelocityEstimatorPPO",
    encoder_ckpt_path="logs/frnc_seg_v3/v3_full/encoder.pt",
    seg_len=32,
    flat_obs_dim=295,
    gait_dim=32,
    # ...继承 VelocityEstimatorPPO 的 estimator 参数
)
```

### 3.2 V22b (`G115DofV22bSegmentEncoderCICPPORunnerCfg`) 新增字段

```python
algorithm = RslRlSegmentEncoderCICPpoAlgorithmCfg(
    class_type="unitree_rl_lab.utils.segment_encoder_cic_ppo.SegmentEncoderCICPPO",
    # 继承 V22a 全部字段，外加：
    intrinsic_alpha_max=0.02,
    intrinsic_warmup_iter_start=200,
    intrinsic_warmup_iter_end=2000,
    cmd_norm_eps=0.1,             # ε，‖v‖_W 死区
    smerl_beta=0.045,             # gate threshold
    smerl_kappa=200.0,            # gate slope
    ema_decay=0.99,               # per-env env-reward EMA
)
```

> `class_type` 必须是 **完整 dotted path 字符串**（`@configclass` 在 IsaacLab 这边的硬约定）；不要写成类对象。

---

## 4. 复现命令

> 以下命令假设 `pwd = /root/workspace/unitree_rl_lab`，已 `conda activate env_isaaclab`，IsaacLab 在 `/root/IsaacLab`。

### 4.1 表征预训练（仅在数据集已 dump 后）

```bash
# Step 1: 从 V21e_strat ckpt dump 离线 segment 数据集（耗时 ~30 min）
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/collect_pretrain_data.py \
    --task Unitree-G1-15dof-Velocity-Rot-V21e_strat \
    --num_envs 256 --num_episodes 30 \
    --out logs/frnc_pretrain_data

# Step 2: 计算步态属性
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/frnc_gait_features.py \
    --in logs/frnc_pretrain_data --out logs/frnc_pretrain_data_gait

# Step 3: 训 V3 encoder（5 cfg × 30 epoch，~1h）
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/frnc_segment_pretrain_v3.py \
    --data logs/frnc_pretrain_data_gait --out logs/frnc_seg_v3

# Step 4: probe 评估（spearman / R² / lip_med / ibvr）
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/frnc_segment_probe.py \
    --ckpt logs/frnc_seg_v3/v3_full/encoder.pt --data logs/frnc_pretrain_data_gait
```

### 4.2 V22a smoke / 训练

```bash
# 5-iter smoke
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V22a \
    --num_envs 256 --max_iterations 5 --headless

# 20k 全训
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V22a \
    --num_envs 4096 --max_iterations 20000 --headless
```

### 4.3 V22b smoke / 训练

```bash
# smoke (已 PASS)
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V22b \
    --num_envs 256 --max_iterations 5 --headless

# 20k 全训
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Unitree-G1-15dof-Velocity-Rot-V22b \
    --num_envs 4096 --max_iterations 20000 --headless
```

### 4.4 Play / 评估

```bash
/root/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Unitree-G1-15dof-Velocity-Rot-V22b \
    --num_envs 16 --checkpoint logs/rsl_rl/.../model_*.pt
```

---

## 5. 工具与编辑陷阱（必读）

### 5.1 VS Code create_file / replace_string_in_file 的"静默失败"

**症状**：在 `source/unitree_rl_lab/...` 路径下，VS Code 这两个工具会**报告成功但实际不写入**。

**对策**：所有 `source/...` 下的代码改动用 bash heredoc + `python -c "import ast; ast.parse(open(p).read())"` 验证：

```bash
cat > <abs_path> << 'PYEOF'
... python code ...
PYEOF
python -c "import ast; ast.parse(open('<abs_path>').read())"
```

`docs/` 下的 markdown 也用 heredoc，避免 escape 字符问题（LaTeX `$` 之类）。

### 5.2 read_file / list_dir 在 workspace root 也可能失败

`/root/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/...` 路径上 read_file 偶尔报 "File does not exist"。改用 `sed -n '<a>,<b>p'` 与 `grep -n`。

### 5.3 `@configclass` 的 `class_type`

必须是 **完整 dotted path 字符串**（不是类对象，也不是相对导入）。例：

```python
class_type = "unitree_rl_lab.utils.segment_encoder_cic_ppo.SegmentEncoderCICPPO"  # OK
# class_type = SegmentEncoderCICPPO           # ❌ 反序列化失败
# class_type = "segment_encoder_cic_ppo.X"    # ❌ 路径不全
```

### 5.4 `gym.register` 重复

注册同一个 task id 两次，旧版 gymnasium 会 raise，新版只 warn。**今天已发现并修复** `__init__.py` 中 V22b 重复注册一次（曾在 1264 与 1280 两处出现）。提交前用：

```bash
grep -c 'id="Unitree-G1-15dof-Velocity-Rot-V22b"' \
    source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/__init__.py
# 期望输出：1
```

### 5.5 RolloutStorage TensorDict shape lock

见 [§2.4](#24-rolloutstorage-shape-lock-坑)。改 obs schema（如把 z_gait 加进 critic）会触发该 lock。必须在 `construct_algorithm` 阶段就注入零 placeholder。

### 5.6 init_at_random_ep_len 与 curriculum

V21 系 env 默认 `init_at_random_ep_len=True`，每步只 ~1/episode_len 比例的 envs reset。任何"基于 episode 完成数"的最小样本检查都要改成累积 buffer，否则永远不满足。

---

## 6. ONNX / 部署

V22a/V22b 的 frozen encoder 必须随 actor 一起导出到 deploy 端：

- 部署端在 `deploy/include/isaaclab/`（State_RLBase 等）维护**与 RL 端等形**的 rolling buffer (32, 295)。
- 推理时序：obs → flat 295 → push buffer → encoder ONNX → z (32) → concat 到 actor latent → action。
- 部署 ONNX 包：`policy.onnx`（actor + transformer latent + concat z 接口）、`encoder.onnx`（frozen V3 encoder）、`runtime.json`（含 sigma_cmd, axial_bases 给 V22b 端可选 r_int 监控；只读）。
- 当前 deploy/robots/g1_15dof_keyboard/ 还**没**接入 V22 的 encoder 路径，需要在 V22b 训练完后单开一个交付任务。

---

## 7. V20l / V21g 接口（速览，详细见 `V20L_MILESTONE_AND_ROADMAP.md`）

V22 系 env 完全 = **V21g**（不是 V20l）。两者关系：

- V20l：full-speed sim2sim 闭环成功；3-way mode token 已固化；deploy 已上机。
- V21g = V20l 的"训练 env 升级版"：加了 yaw 跟踪 reward shaping、ground friction curriculum 调整、cmd 分布扩展；用于 V22 系训练。
- V22 系不动 env，纯算法改 → 任何 V20l 上的 deploy/sim2sim 工具链对 V22 同样适用，只需替换 actor + encoder ONNX。

---

## 8. 下一步路线

按优先级：

1. **V22b 20k 全训**（4096 envs，~8h），观测：
   - `intrinsic/r_int_mean` 是否在 iter 200~2000 内从 0 平滑 ramp 到 0.2-0.4；
   - `intrinsic/gate_mean` 是否在 iter 1000+ 稳定 > 0.5；
   - `intrinsic/alpha` 是否准时到 0.02；
   - 总 reward 不显著低于 V22a/V21g。
2. **3-way bucket 评估**（standing / pure_wz / other 三类 reward + 跟踪误差），对比 V21g/V22a/V22b。
3. **消融**：
   - α_max ∈ {0.01, 0.02, 0.05}；
   - β ∈ {0.030, 0.045, 0.060}；
   - 关 SMERL gate（gate ≡ 1）vs 关 mask（mask ≡ 1）。
4. **deploy 接入**：encoder.onnx + runtime buffer，sim2sim 验证 V22b。
5. **V22f**（远期）：3-phase EMA 在线 encoder（先 frozen 收敛 → slow EMA 解冻 → 端到端 finetune）；只在 V22b 证明残差 reward 有效后再考虑。

---

## 9. 联系信息 / 内部记忆

仓库内：
- `AGENT_HANDOFF.md` — 项目级总入口（V20l 状态、deploy caveats、tool pitfalls）；
- `V20L_MILESTONE_AND_ROADMAP.md` — V20l sim2sim 成功记录与下阶段优化路线；
- `29dof_rotation_transfer_guide.md` — rotation 任务族跨 dof 迁移指南；
- 本目录 `docs/handoff_contrastive/` — 表征学习线（V19d-CLP → V22b）专题。

Copilot agent memory（已沉淀）：
- `/memories/repo/lcp_notes.md` — V19d-CLP 的 P0/A 修复细节；
- `/memories/repo/v22a_notes.md` — V22a 集成坑与 RolloutStorage shape lock；
- `/memories/repo/v22b_notes.md` — V22b r_int / SMERL / α 的实现要点。

新来的同学建议阅读顺序：
1. `AGENT_HANDOFF.md`（5 min）→ 项目全貌；
2. 本目录 `00_overview.md`（5 min）→ 表征学习线全貌；
3. 本目录 `01_problem_and_math.md`（30 min）→ 数学定义；
4. 本目录 `02_design_evolution.md`（20 min）→ 历史教训；
5. 本目录 `03_codebase_and_handoff.md`（本篇，20 min）→ 工程抓手；
6. 跑一次 §4.3 V22b smoke 确认环境通；
7. 翻一次 `/memories/repo/v22b_notes.md` 准备应付下次踩坑。
