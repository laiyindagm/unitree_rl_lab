# Unitree RL Lab — 项目架构文档

> 本文档为 G1 人形机器人 RL 训练项目的完整代码架构参考，
> 覆盖目录结构、MDP 管线、核心模块 API、配置继承链、训练流程等，
> 目标是让开发者快速理解项目并进行架构级改动。

---

## 目录

1. [项目总览](#1-项目总览)
2. [目录结构](#2-目录结构)
3. [核心架构：ManagerBasedRLEnv 管线](#3-核心架构managedbasedrlenv-管线)
4. [MDP 模块详解](#4-mdp-模块详解)
5. [环境配置继承链](#5-环境配置继承链)
6. [PPO Runner 配置](#6-ppo-runner-配置)
7. [训练流程](#7-训练流程)
8. [版本演进核心问题](#8-版本演进核心问题)
9. [部署结构](#9-部署结构)
10. [关键文件速查表](#10-关键文件速查表)

---

## 1. 项目总览

| 项 | 值 |
|---|---|
| **框架** | Isaac Lab (ManagerBasedRLEnv) + RSL-RL (PPO) |
| **机器人** | Unitree G1 人形, 29-DOF URDF, 训练用 15-DOF (腿+腰) |
| **任务** | 全向速度跟踪 (vx, vy, wz)，含旋转增强 |
| **仿真** | dt=0.005s, decimation=4 (env_dt=0.02s), 4096 并行环境, episode=20s |
| **网络** | MLP [512,256,128] ELU / 可选 Transformer History Model |
| **Python** | 3.11 (conda env: env_isaaclab) |
| **工作区** | /root/workspace/unitree_rl_lab/ |
| **Isaac Lab** | /root/IsaacLab/ |

---

## 2. 目录结构

```
unitree_rl_lab/
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py              # 训练入口
│   │   ├── play.py               # 推理回放
│   │   └── cli_args.py           # RSL-RL CLI 参数
│   └── list_envs.py              # gymnasium 环境注册扫描
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── assets/robots/            # URDF / ArticulationCfg
│   │   └── unitree.py            # UNITREE_G1_29DOF_CFG 等
│   ├── tasks/
│   │   ├── locomotion/           # ★ 核心任务模块
│   │   │   ├── agents/           # PPO Runner 配置
│   │   │   │   └── rsl_rl_ppo_cfg.py
│   │   │   ├── mdp/              # MDP 组件 (命令/奖励/观测/课程)
│   │   │   │   ├── __init__.py   # 统一导出 (wildcard imports)
│   │   │   │   ├── commands/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── velocity_command.py  # ★ 4个速度指令类 (1096行)
│   │   │   │   ├── rewards.py               # ★ 36个奖励函数 (817行)
│   │   │   │   ├── curriculums.py           # 7个课程函数 (382行)
│   │   │   │   ├── observations.py          # 2个观测函数 (82行)
│   │   │   │   └── symmetry_g1_15dof.py     # 左右对称增强 (154行)
│   │   │   ├── robots/g1/
│   │   │   │   ├── 15dof_rot/               # ★ G1-15DOF 旋转增强配置
│   │   │   │   │   ├── __init__.py          # Gym 注册 (59个版本)
│   │   │   │   │   ├── velocity_env_cfg_rot.py      # 基准配置
│   │   │   │   │   ├── velocity_env_cfg_rot_v15a.py # 奖励重平衡
│   │   │   │   │   ├── velocity_env_cfg_rot_v16b.py # 计划运动激励
│   │   │   │   │   ├── velocity_env_cfg_rot_v19c.py # 纯环境精度
│   │   │   │   │   ├── velocity_env_cfg_rot_v19d.py # ★ 当前版本
│   │   │   │   │   └── ... (v2a ~ v19d 共59个版本文件)
│   │   │   │   ├── 15dof_lcp/               # LCP (Lipschitz) 变体
│   │   │   │   └── 29dof/                   # 29-DOF 全身配置
│   │   │   ├── robots/go2/                  # Go2 四足
│   │   │   └── robots/h1/                   # H1 人形
│   │   └── mimic/                           # 模仿学习任务 (dance 等)
│   └── utils/
│       ├── rsl_rl_custom_ppo.py             # UnitreePPO (自定义PPO)
│       ├── lcp_ppo.py                       # LCPPPO (Lipschitz梯度惩罚)
│       ├── transformer_ppo.py               # TransformerPPO (aux loss)
│       ├── rsl_rl_transformer_model.py      # TransformerHistoryModel
│       ├── export_deploy_cfg.py             # 部署配置导出
│       └── parser_cfg.py                    # 配置解析工具
├── deploy/                        # C++ 部署代码 (ONNX Runtime)
│   ├── include/                   # 头文件
│   ├── robots/                    # 各机器人部署 (g1_29dof, go2, h1等)
│   └── thirdparty/               # onnxruntime
├── docker/                        # Docker 构建
├── logs/rsl_rl/                   # 训练日志
└── unitree_ros/                   # ROS 集成
```

---

## 3. 核心架构：ManagerBasedRLEnv 管线

Isaac Lab 的 ManagerBasedRLEnv 按以下顺序在每个 env step 中执行各 Manager：

```
┌─────────────────────────────────────────────────┐
│                   env.step()                    │
├─────────────────────────────────────────────────┤
│  1. Scene          渲染/物理步进 (decimation次)  │
│  2. Events         域随机化 (startup/reset/interval)│
│  3. Commands       采样/更新速度指令              │
│  4. Actions        目标关节位置 → PD 控制         │
│  5. Observations   构建 obs dict → 展平+历史     │
│  6. Rewards        计算各 reward term 加权求和    │
│  7. Terminations   检查终止条件 (超时/跌倒/朝向) │
│  8. Curriculum     评估/扩展命令范围              │
└─────────────────────────────────────────────────┘
```

每个 Manager 由 XxxCfg dataclass 配置，包含多个 XxxTermCfg，每个 Term 指向一个函数和其参数。

---

## 4. MDP 模块详解

> 所有自定义 MDP 组件位于 tasks/locomotion/mdp/。
> mdp/__init__.py 通过 wildcard import 统一导出：
> - isaaclab.envs.mdp (Isaac Lab 内建)
> - isaaclab_tasks.manager_based.locomotion.velocity.mdp (官方任务)
> - .commands / .curriculums / .observations / .rewards (项目自定义)

### 4.1 Commands（速度指令）

文件: mdp/commands/velocity_command.py (1096行)

4个指令类对 (Cfg + Command)：

| 类名 | 行号 | 用途 | 使用版本 |
|------|------|------|---------|
| UniformLevelVelocityCommandCfg/Command | 14-62 | 基础均匀采样+rotating envs | 基准rot配置 |
| PerformanceWeightedVelocityCommandCfg/Command | 64-497 | 8x8网格自适应采样 | V8-V14系列 |
| DiscreteVelocityCommandCfg/Command | 499-612 | 离散速度级别 | 实验用 |
| **MarginalVelocityCommandCfg/Command** | **615-1092** | **轴独立边际采样** | **V19c/V19d(当前)** |

#### MarginalVelocityCommand 核心设计

**理念**: 每个轴(vx,vy,wz)独立采样、独立跟踪精度，避免轴间污染。

**Cfg 关键字段:**

| 字段 | 默认值 | 含义 |
|------|--------|------|
| rel_standing_envs | 0.0 | 静止环境比例(所有轴=0) |
| rel_rotating_envs | 0.0 | 纯旋转环境比例(vx=vy=0) |
| rel_pure_vx_envs | 0.0 | 纯前后环境比例(vy=wz=0) |
| rel_pure_vy_envs | 0.0 | 纯侧移环境比例(vx=wz=0) |
| vx_pos_bins | [(0.1,0.1),...] | 正向vx离散速度箱 |
| vx_neg_bins | [(-0.1,-0.1),...] | 反向vx离散速度箱 |
| vy_bins | [(0,0),(0.1,0.1),...] | vy离散速度箱(含零) |
| wz_bins | [(0.1,0.1),...] | wz离散速度箱(无零) |
| num_active_vx_pos/neg | None | 活跃箱数(课程起始范围) |
| num_active_vy/wz | None | 活跃箱数 |
| ema_alpha | 0.1 | 性能EMA平滑系数 |
| temperature | 3.0 | 采样温度(越高→越聚焦差箱) |
| min_sampling_prob | 0.02 | 最小采样概率下限 |
| min_response_speed | 0.0 | **方向门控精度**(V19d:0.05) |

**Command 方法:**

| 方法 | 行号 | 功能 |
|------|------|------|
| __init__ | 672 | 构建边缘张量、EMA缓冲区、环境类型掩码 |
| _update_active_masks | 739 | 根据活跃箱数重建采样掩码 |
| _compute_axis_probs | 767 | 难度加权softmax→采样概率(差箱高概率) |
| _resample_command | 781 | 独立轴采样+环境类型分配+零轴路由 |
| _update_command | 864 | 逐步精度累积(方向门控+零速精度) |
| get_episode_accuracy | 927 | Episode结束时返回各轴平均精度 |
| update_bin_performance | 952 | **纯环境独占**EMA更新(vx←pure_vx,vy←pure_vy,wz←rotating) |
| expand_direction | 999 | 扩展单轴活跃箱数 |
| log_sampling_stats | 1056 | 打印各轴边际分布(perf/prob/count) |

**环境类型分配流程(_resample_command):**

```
u = rand(0,1)
if u < standing:          → all axes = zero bin
elif u < standing+pvx:    → pure_vx: vy=0, wz=0
elif u < +pvy:            → pure_vy: vx=0, wz=0
elif u < +rotating:       → pure_wz: vx=0, vy=0
else:                     → joint: 所有轴独立采样(no zero override)
```

**精度计算(_update_command, 每步调用):**

```
对于每个轴(以vx为例):
if |cmd_vx| >= accuracy_cmd_min:
    raw = clamp(1 - |actual - cmd| / |cmd|, 0, 1)
    if min_response_speed > 0:
        raw *= (actual * sign(cmd) > min_response_speed)  # 方向门控
    acc = raw
else:
    acc = clamp(1 - |actual| / zero_thr, 0, 1)  # 零速精度
```

**性能更新(update_bin_performance, episode结束时):**
- vx性能 ← 仅pure_vx环境的episode精度
- vy性能 ← 仅pure_vy环境的episode精度
- wz性能 ← 仅rotating(pure_wz)环境的episode精度
- 使用EMA: perf[bin] = (1-α)*perf[bin] + α*mean(acc[envs_in_bin])

**Cfg→Command绑定:** 通过__post_init__ monkey-patch class_type

---

### 4.2 Rewards（奖励函数）

文件: mdp/rewards.py (817行), 共36个函数。

#### 按类别分组:

**速度跟踪:**

| 函数 | 行号 | 说明 |
|------|------|------|
| track_lin_vel_xy_yaw_frame_exp | (isaaclab) | 基础xy速度跟踪 exp(-err²/σ²) |
| track_ang_vel_z_rotating_aware | 249 | **旋转感知**wz跟踪(直行时权重减半) |
| track_lin_vel_xy_adaptive_sigma | 444 | 自适应σ跟踪 |
| track_ang_vel_z_adaptive_sigma | 469 | 自适应σ wz跟踪 |
| track_lin_vel_xy_scheduled_sigma | 555 | 按迭代调度σ |
| track_lin_vel_xy_baselined | 641 | 基线化跟踪 |

**反死区/运动激励:**

| 函数 | 行号 | 说明 |
|------|------|------|
| cmd_nonresponse_penalty | 745 | 有命令但无运动→惩罚 |
| movement_incentive_scheduled | 781 | 按迭代调度的运动激励(start_step~end_step) |
| low_speed_tracking_bonus | 498 | 低速额外追踪奖励 |
| low_speed_rotation_bonus | 528 | 低速旋转额外奖励 |

**步态:**

| 函数 | 行号 | 说明 |
|------|------|------|
| feet_gait | 175 | 标准步态奖励(单脚交替接触) |
| feet_gait_speed_scaled | 697 | 速度缩放步态 |
| foot_clearance_reward | 121 | 抬脚高度奖励 |
| foot_clearance_speed_scaled | 719 | 速度缩放抬脚 |

**旋转专用:**

| 函数 | 行号 | 说明 |
|------|------|------|
| yaw_rate_l1 | 274 | 直行时寄生偏航惩罚 |
| rotation_single_support_reward | 296 | 纯旋转单脚支撑奖励 |
| rotation_double_support_slide_penalty | 311 | 纯旋转双脚滑动惩罚 |
| rotation_twist_joint_penalty | 332 | 扭转关节惩罚 |

**稳定性/正则化:**

| 函数 | 行号 | 说明 |
|------|------|------|
| energy | 23 | 能量消耗(τ·ω) |
| backward_lean_penalty | 206 | 后仰惩罚 |
| stand_still | 32 | 静止时关节偏差惩罚 |
| joint_position_penalty | 68 | 关节位置惩罚 |
| joint_mirror | 224 | 左右对称惩罚 |
| air_time_variance_penalty | 156 | 滞空时间方差 |
| feet_stumble | 85 | 绊脚惩罚 |

其余函数来自Isaac Lab内建: lin_vel_z_l2, ang_vel_xy_l2, joint_vel_l2, joint_acc_l2,
action_rate_l2, joint_pos_limits, flat_orientation_l2, base_height_l2,
joint_deviation_l1, feet_slide, undesired_contacts, is_alive

---

### 4.3 Observations（观测）

文件: mdp/observations.py (82行)

| 函数 | 说明 |
|------|------|
| gait_phase | 基于episode_length的固定周期步态相位→[sin,cos](2维) |
| gait_phase_speed_adaptive | 速度自适应步态相位(快走短周期/慢走长周期/静止衰减) |

**观测空间构成(15-DOF基准配置):**

| 观测项 | 维度 | 缩放 | 噪声 | 备注 |
|--------|------|------|------|------|
| base_ang_vel | 3 | x0.2 | ±0.2 | 角速度 |
| projected_gravity | 3 | 1.0 | ±0.05 | 重力投影 |
| velocity_commands | 3 | 1.0 | 无 | 目标[vx,vy,wz] |
| joint_pos_rel | 15 | 1.0 | ±0.01 | 相对默认位置 |
| joint_vel_rel | 15 | x0.05 | ±1.5 | 关节速度 |
| last_action | 15 | 1.0 | 无 | 上一步动作 |

- **history_length = 5** → 每帧54维 x 5 = 270维 (Policy)
- **Critic额外包含base_lin_vel**(3维,特权信息) → 57 x 5 = 285维
- enable_corruption = True (Policy有噪声, Critic无)

---

### 4.4 Curriculums（课程学习）

文件: mdp/curriculums.py (382行)

| 函数 | 行号 | 用途 | 配合指令类 |
|------|------|------|-----------|
| lin_vel_cmd_levels | 11 | 线速度范围逐步扩大 | UniformLevel |
| ang_vel_cmd_levels | 40 | 角速度范围逐步扩大 | UniformLevel |
| terrain_levels_vel | 64 | 地形难度课程 | 所有 |
| speed_bucketed_vel_curriculum | 73 | 速度分桶课程 | 早期实验 |
| iteration_based_vel_curriculum | 163 | 按迭代扩展范围 | 早期实验 |
| performance_weighted_vel_curriculum | 225 | 网格性能自适应 | PerformanceWeighted |
| **marginal_vel_curriculum** | **296** | **轴独立边际课程** | **Marginal(V19c/V19d)** |

#### marginal_vel_curriculum 关键逻辑:

```
每个 max_episode_length 步评估一次:
  对 vx_pos, vx_neg, vy, wz 四个方向:
    1. 获取活跃箱 indices
    2. 计算 bin_perfs (EMA值)
    3. if min(bin_perfs) > range_expand_threshold:
         扩展 (vx: +1箱, vy/wz: +2箱)
    4. elif min < min_perf_floor:
         打印警告, 阻止扩展
  返回 min(各方向 mean_perf) 作为课程指标
```

---

### 4.5 Symmetry（对称增强）

文件: mdp/symmetry_g1_15dof.py (154行)

基于矢状面反射(y→-y)的左右对称数据增强：
- 左右肢体关节索引交换: L(0-5) ↔ R(6-11), 腰(12-14)不变
- 取反: roll/yaw关节角, ang_vel x/z, gravity y, vel_cmd y/wz
- 用于 BasePPORunnerV4Cfg / BasePPORunnerV5bCfg 的对称训练

---

## 5. 环境配置继承链

```
ManagerBasedRLEnvCfg (Isaac Lab)
  └── velocity_env_cfg_rot.py::RobotEnvCfg            ← 基准15-DOF旋转配置
        ├── Scene: G1-29DOF, cobblestone terrain, height_scanner, contact_forces
        ├── Events: 摩擦DR(0.3-1.0), 质量DR(torso±[-1,3]kg), push每5s
        ├── Commands: UniformLevelVelocityCommandCfg (±0.1初始, ±0.5/1.5极限)
        ├── Actions: JointPositionAction, 15-DOF, scale=0.25
        ├── Observations: history=5, corruption=True
        ├── Rewards: 17 term (跟踪+稳定+步态+旋转)
        ├── Terminations: timeout(20s), height(<0.2), orientation(>0.8rad)
        └── Curriculum: terrain + lin/ang_vel levels
            │
            ├── velocity_env_cfg_rot_v13a.py           ← 中间版本
            │     └── velocity_env_cfg_rot_v15a.py     ← 奖励重平衡
            │           ├── joint_deviation_legs: -1.0 → -0.3  (释放hip_yaw)
            │           ├── flat_orientation: -5.0 → -3.0
            │           ├── cmd_nonresponse: 新增(-0.5)
            │           │
            │           └── velocity_env_cfg_rot_v16b.py ← 计划运动激励
            │                 ├── movement_incentive_scheduled: -1.0
            │                 │
            │                 ├── velocity_env_cfg_rot_v19c.py ← 纯环境精度
            │                 │     ├── Commands: MarginalVelocityCommandCfg
            │                 │     │   ├── 5%standing, 15%pure_vx, 15%pure_vy
            │                 │     │   ├── 15%rotating, 50%joint
            │                 │     │   └── temp=3.0, floor=0.02
            │                 │     ├── movement_incentive: -3.0
            │                 │     └── cmd_nonresponse: -2.0
            │                 │
            │                 └── velocity_env_cfg_rot_v19d.py ← ★当前版本
            │                       ├── Commands: MarginalVelocityCommandCfg
            │                       │   ├── 8%standing, 15%pure_vx, 15%pure_vy
            │                       │   ├── 20%rotating(wz增强), 42%joint
            │                       │   ├── temp=8.0, floor=0.01
            │                       │   └── min_response_speed=0.05(方向门控)
            │                       ├── track_ang_vel_z: weight=2.0(从1.0)
            │                       ├── movement_incentive: -3.0
            │                       ├── cmd_nonresponse: -2.0
            │                       └── Curriculum: min-based expansion, threshold=0.3
```

### 基准配置奖励权重表 (velocity_env_cfg_rot.py)

| Term | Weight | 函数 |
|------|--------|------|
| track_lin_vel_xy | +1.0 | track_lin_vel_xy_yaw_frame_exp(σ²=0.25) |
| track_ang_vel_z | +1.0 | track_ang_vel_z_rotating_aware(σ²=0.25) |
| alive | +0.15 | is_alive |
| lin_vel_z | -2.0 | lin_vel_z_l2 |
| ang_vel_xy | -0.05 | ang_vel_xy_l2 |
| joint_vel | -0.001 | joint_vel_l2 |
| joint_acc | -2.5e-7 | joint_acc_l2 |
| action_rate | -0.05 | action_rate_l2 |
| dof_pos_limits | -5.0 | joint_pos_limits |
| energy | -2e-5 | energy |
| joint_deviation_waists | -1.0 | joint_deviation_l1(waist.*) |
| joint_deviation_legs | -1.0 | joint_deviation_l1(hip_roll+hip_yaw) |
| flat_orientation | -5.0 | flat_orientation_l2 |
| base_height | -10.0 | base_height_l2(target=0.78) |
| gait | +0.5 | feet_gait(period=0.8) |
| feet_slide | -0.2 | feet_slide |
| feet_clearance | +1.0 | foot_clearance_reward |
| undesired_contacts | -1.0 | undesired_contacts |
| yaw_rate_l1 | -0.15 | yaw_rate_l1 |
| rotation_single_support | +1.0 | rotation_single_support_reward |
| rotation_double_support_slide | -1.0 | rotation_double_support_slide_penalty |
| rotation_twist_joint | -0.1 | rotation_twist_joint_penalty |

### V19d 相对基准的修改:

| 修改 | 基准值 | V19d值 | 来源 |
|------|--------|---------|------|
| joint_deviation_legs | -1.0 | -0.3 | V15a |
| flat_orientation | -5.0 | -3.0 | V15a |
| cmd_nonresponse | (无) | -2.0 | V15a→V19d |
| movement_incentive | (无) | -3.0 | V16b→V19d |
| track_ang_vel_z | +1.0 | +2.0 | V19d |
| Commands | UniformLevel | Marginal(方向门控) | V19d |
| Curriculum | lin/ang_levels | marginal_vel(min-based) | V19d |

---

## 6. PPO Runner 配置

文件: tasks/locomotion/agents/rsl_rl_ppo_cfg.py

### PPO 基础参数 (BasePPORunnerCfg)

| 参数 | 值 |
|------|------|
| Actor/Critic | MLP [512, 256, 128], ELU |
| Distribution | Gaussian, init_std=1.0, log-std |
| PPO class | UnitreePPO (自定义) |
| LR | 1e-3, adaptive schedule (based on KL) |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Steps/env | 24 |
| Epochs | 5 |
| Mini-batches | 4 |
| Clip | 0.2 |
| Entropy coef | 0.01 |
| Max iterations | 20000 |
| Save interval | 500 |

### Runner 变体

| Runner | 基于 | 差异 |
|--------|------|------|
| BasePPORunnerCfg | - | 基线MLP PPO |
| **BasePPORunnerV3Cfg** | Base | **+clip_actions=100(NaN安全)** ← V19d使用 |
| BasePPORunnerV4Cfg | Base | +clip_actions + 对称数据增强 |
| BasePPORunnerV5bCfg | Base | +clip_actions + 对称mirror loss |
| G1TransformerPPORunnerCfg | Base | Transformer历史模型(29-DOF) |
| G115DofTransformerPPORunnerCfg | Base | Transformer(15-DOF, obs=54) |
| G115DofLCPPPORunnerCfg | Base | LCP梯度惩罚 |
| G115DofLCPPPORunnerV3Cfg | LCP | +clip_actions |
| G115DofTransformerAuxPPORunnerCfg | Base | Transformer+aux next-obs prediction |

### Transformer History Model

```
输入: obs(history_length x obs_dim) → split → [history_obs | aux_obs]
                                                    ↓
                                              Transformer Encoder
                                              (causal mask, FiLM)
                                                    ↓
                                               Latent → MLP Head → Action
                                               (可选 aux loss: predict next obs)
```

参数: d_model=256, n_heads=4, 2 layers, ff=512

---

## 7. 训练流程

### 启动命令

```bash
cd /root/workspace/unitree_rl_lab
./unitree_rl_lab.sh -p scripts/rsl_rl/train.py \
  --task Unitree-G1-15dof-Velocity-Rot-V19d \
  --headless
```

### 执行链

```
train.py
  ├── import list_envs → 触发 unitree_rl_lab.tasks 的 __init__.py
  │     └── 各 robots/g1/15dof_rot/__init__.py 执行 gym.register()
  ├── AppLauncher → 初始化 Isaac Sim
  ├── @hydra_task_config(task, "rsl_rl_cfg_entry_point")
  │     └── 从 gym registry 解析 env_cfg_entry_point + rsl_rl_cfg_entry_point
  ├── gym.make(task, cfg=env_cfg) → ManagerBasedRLEnv
  ├── RslRlVecEnvWrapper(env, clip_actions=...)
  ├── OnPolicyRunner(env, agent_cfg, log_dir)
  │     ├── actor = create_model(actor_cfg)
  │     ├── critic = create_model(critic_cfg)
  │     └── algorithm = UnitreePPO(actor, critic, ...)
  ├── dump env.yaml + agent.yaml + deploy_cfg
  └── runner.learn(num_learning_iterations)
        每个 iteration:
          1. 采集 num_steps_per_env 步 rollout
          2. 计算 GAE advantages
          3. 分 mini-batch 训练 num_learning_epochs 轮
          4. 自适应调整 LR (基于 KL divergence)
          5. 定期保存 checkpoint
```

### Gym 注册格式

```python
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "...agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)
```

---

## 8. 版本演进核心问题

### 低速死区问题(核心挑战)

**症状**: 速度命令 < ~0.3 m/s 时机器人站立不动。

**原因分析**:
1. **奖励竞争**: joint_deviation_legs(-1.0) 惩罚 hip_yaw 偏离默认位 → 压制旋转和侧移
2. **精度膨胀**: 基座速度噪声(~0.03-0.05 m/s) 对小命令产生虚假高精度
3. **采样不聚焦**: 低温度+高概率下限→采样几乎均匀，无法集中训练弱箱
4. **轴间污染**: 联合命令环境中，多轴同时活跃使单轴精度信号被稀释

### V19c 解决: 纯环境
- 引入 pure_vx / pure_vy / rotating 环境类型
- 性能更新仅使用对应纯环境数据 → 消除轴间污染
- **残留问题**: 噪声仍膨胀低速精度 → vx即刻扩展到24/24

### V19d 解决: 方向门控
- min_response_speed=0.05: 实际速度必须在命令方向上超过0.05才计入精度
- min-based expansion: 每个箱都必须真正响应才能扩展
- temperature=8.0, floor=0.01: 更聚焦采样
- track_ang_vel_z=2.0: 增强wz信号对抗joint_deviation_legs

---

## 9. 部署结构

```
deploy/
├── include/
│   ├── param.h                    # 关节名/顺序参数
│   ├── unitree_articulation.h     # 关节控制接口
│   ├── unitree_joystick_dsl.hpp   # 遥控器输入
│   └── isaaclab/                  # Isaac Lab 推理参数
├── robots/
│   ├── g1_29dof/                  # G1 29-DOF 部署
│   ├── g1_29dof_keyboard/         # 键盘控制
│   ├── g1_15dof_keyboard/         # 15-DOF 键盘
│   ├── go2/ go2w/ h1/ h1_2/ b2/  # 其他机器人
│   └── g1_23dof/
└── thirdparty/
    └── onnxruntime-linux-x64-1.22.0/  # ONNX Runtime C++ 推理
```

部署流程: 训练完成 → ONNX导出 → C++ ONNX Runtime推理 → Unitree SDK控制

---

## 10. 关键文件速查表

| 目的 | 文件路径 (相对 source/unitree_rl_lab/unitree_rl_lab/) |
|------|------|
| **速度指令类** | tasks/locomotion/mdp/commands/velocity_command.py |
| **奖励函数** | tasks/locomotion/mdp/rewards.py |
| **课程函数** | tasks/locomotion/mdp/curriculums.py |
| **观测函数** | tasks/locomotion/mdp/observations.py |
| **对称增强** | tasks/locomotion/mdp/symmetry_g1_15dof.py |
| **MDP统一导出** | tasks/locomotion/mdp/__init__.py |
| **PPO Runner配置** | tasks/locomotion/agents/rsl_rl_ppo_cfg.py |
| **当前版本配置(V19d)** | tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v19d.py |
| **基准旋转配置** | tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot.py |
| **Gym注册** | tasks/locomotion/robots/g1/15dof_rot/__init__.py |
| **训练脚本** | (workspace root) scripts/rsl_rl/train.py |
| **自定义PPO** | utils/rsl_rl_custom_ppo.py |
| **LCP PPO** | utils/lcp_ppo.py |
| **Transformer模型** | utils/rsl_rl_transformer_model.py |
| **Transformer PPO** | utils/transformer_ppo.py |
| **部署配置导出** | utils/export_deploy_cfg.py |
| **机器人资产** | assets/robots/unitree.py |

---

## 附注：开发实用信息

- **训练启动**: ./unitree_rl_lab.sh -p scripts/rsl_rl/train.py --task Unitree-G1-15dof-Velocity-Rot-V19d --headless
- **新版本创建步骤**:
  1. 在 robots/g1/15dof_rot/ 创建 velocity_env_cfg_rot_vXX.py
  2. 在 __init__.py 中添加 gym.register(...) (指定 env_cfg + ppo_cfg)
  3. 如新增 mdp 函数，在对应 mdp/*.py 中添加并确保 __init__.py 导出
- **观测维度计算**: 单帧 obs_dim = 3+3+3+15+15+15 = 54 (Policy), 57 (Critic, +base_lin_vel)
  - 带历史: 54x5 = 270 (Policy), 57x5 = 285 (Critic)
