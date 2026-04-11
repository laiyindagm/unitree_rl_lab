# 29DOF 旋转增强可迁移实施指导

## 目标

将已经被当前项目训练结果证明为“切实有效”的改动迁移到另一个 locomotion 项目中，要求：

1. 不明显降低正常步行性能
2. 不明显降低静止稳定性
3. 能增强原地旋转和角速度命令跟踪能力

## 结论摘要

建议迁移且证据充分的改动，只有四类：

1. 给训练分布加入少量纯旋转样本，而不是大量纯旋转样本
2. 让角速度命令在训练早期就落在可学区间
3. 使用“只在真正需要转向时才激活”的 yaw tracking 逻辑
4. 用温和的 rotation-specific stepping reward 约束“如何转”，而不是只奖励“转到了”

不建议直接迁移的改动：

1. 过高的纯旋转样本比例，例如 0.3 或 0.5
2. 对所有 walking 命令都施加较强 yaw 辅助惩罚
3. 一次性叠加太多新的旋转负奖励项并替换原有步态结构
4. 任何已经导致 episode length 和 time_out 明显塌陷的组合方案

## 一、命令分布如何改

### 要做什么

在速度命令采样器中加入少量 pure rotation 样本：

- standing env 比例保持小量，例如 0.05
- rotating env 比例从 0 提到 0.08 左右
- 不要先上 0.3 或 0.5 这种高比例

### 为什么有效

如果训练集中没有“线速度接近 0、角速度非零”的命令，策略很难学到原地转圈。
但如果 pure rotation 样本过多，会稀释正常步行和 walk+turn 分布，策略容量被错误分配。

### 当前项目中的参考实现

- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_expB.py:35

### 迁移时的最小实现

将你的 command sampler 改成：

- rel_standing_envs = 0.05
- rel_rotating_envs = 0.08
- 线速度正常采样
- 对 rotating env，将 lin_vel 归零，仅保留 ang_vel_z

如果你的项目没有 rel_rotating_envs 概念，可以在 command resample 后额外做一次 mask：

- 随机抽 8% env_id
- 把这些 env 的 cmd[:2] 置零
- 保留 cmd[2]

## 二、初始角速度范围如何改

### 要做什么

把 early curriculum 或初始 command range 中的 ang_vel_z，从过窄区间扩大到可学区间。
推荐起点：

- ang_vel_z 初始范围设为 (-0.2, 0.2)

### 为什么有效

如果初始 yaw 命令过小，策略在早期几乎看不到足够明确的旋转学习信号，课程推进和旋转技能形成都会变慢。

### 当前项目中的参考实现

- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_expB.py:43

## 三、yaw tracking 逻辑应该怎么写

### 要做什么

不要对所有 walking 环境都削弱或覆盖 yaw tracking。
正确做法是：

1. 当命令明确要求转向时，正常追踪 yaw
2. 只有在“直行命令”上，才把寄生偏航当成惩罚对象

### 当前项目中的参考实现

- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:380
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:402

### 迁移逻辑

主 yaw reward：

- 如果 cmd_lin_norm > 0.1 且 |cmd_yaw| < 0.05，认为这是 straight walk
- straight walk 环境不使用这个 reward 去惩罚 yaw mismatch，直接返回 1
- 其他环境正常计算 exp(-(actual_yaw - cmd_yaw)^2 / std^2)

辅助寄生偏航惩罚：

- 只在 straight walk 环境上计算
- 处罚 abs(actual_yaw)
- 不要处罚 walk+turn 环境的 yaw tracking error

### 权重建议

- yaw tracking reward 是主任务的一部分，可以相对较强
- 寄生偏航惩罚只能是弱辅助项，建议先从 -0.1 到 -0.2 起步

## 四、如何奖励“正确地转”而不是“扭出来”

### 要做什么

对 pure rotation 场景增加三种轻量 shaping：

1. 奖励 single support 旋转
2. 惩罚 double support 滑脚旋转
3. 惩罚双脚着地时靠腰或胯扭转硬拧出 yaw

### 当前项目中的参考实现

- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:423
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:435
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:450
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:469

### 推荐初始权重

- single support reward: +1.0
- double support slide penalty: -1.0
- twist joint penalty: -0.1

这些 reward 只在 pure rotation 命令下生效：

- ||cmd_xy|| < 0.1
- |cmd_yaw| > 0.05

## 五、哪些旧结构不要轻易删

旋转增强时，以下结构不要随意移除：

1. 正常线速度 tracking reward
2. 正常 gait reward
3. 基本姿态稳定项
4. 基本站立稳定项

原地转圈本质上是 locomotion 的子技能，不应该凌驾于 walking stability 之上。

## 六、稳定性防线

### 要做什么

无论你是否迁移旋转增强，都建议把 actor 的高斯分布 std 改成 log-std 参数化。

### 当前项目中的参考实现

- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py:37
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py:42

### 为什么要做

当前项目已经实际踩过坑：如果 std 用 direct scalar 参数化，一旦训练不稳定，std 可能被更新成负数，训练会直接崩在 normal expects all elements of std >= 0.0。

## 七、推荐迁移顺序

### 阶段 1：只改命令分布和角速度范围

1. pure rotation 样本比例 = 0.08
2. 初始 ang_vel_z = (-0.2, 0.2)

### 阶段 2：加入 straight-walk parasitic yaw suppression

1. 加 track_ang_vel_z_rotating_aware
2. 加弱权重 yaw_rate_l1

### 阶段 3：加入 pure rotation shaping

1. single support reward
2. double support slide penalty
3. twist joint penalty

只要某一步明显让 episode length 掉很多、bad orientation 上升很多、gait 大幅下降，就应回退到上一步。

## 八、迁移评估指标

至少同时跟踪以下指标：

1. mean reward
2. mean episode length
3. track_lin_vel_xy
4. track_ang_vel_z
5. gait
6. error_vel_xy
7. error_vel_yaw
8. time_out termination rate
9. bad_orientation termination rate

判定一个改动是否“切实有效”，必须同时满足：

1. track_ang_vel_z 明显提升
2. track_lin_vel_xy 不显著下降
3. gait 不显著下降
4. time_out 维持高位
5. bad_orientation 不显著上升

## 九、手臂外部动作项目的最小验证

如果你的最终目标是手臂由外部动作源接管、下半身负责稳定走路，那么最合理的验证步骤不是直接上 15DOF，而是先做一个最小 no-arms locomotion 版本。

当前项目中的参考：

- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_expB_no_arms.py:38
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_expB_no_arms.py:48
- source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_expB_no_arms.py:82

做法：

1. 从 locomotion policy 动作中移除双臂
2. 从 policy/critic 观测中移除双臂关节位置和速度
3. 从手臂偏离奖励中移除双臂
4. stand_still 只统计下半身和腰

## 十、最小迁移清单

如果你只允许做最少改动，按这个清单迁移：

1. 命令采样：增加 8% pure rotation env
2. 初始 yaw range：改成 (-0.2, 0.2)
3. 主 yaw reward：仅对真正需要转向的命令计算 tracking
4. 辅助 yaw penalty：只在 straight walk 场景，弱权重
5. pure rotation shaping：single support / anti-slide / anti-twist
6. PPO actor std：改成 log-std
