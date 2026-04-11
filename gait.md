下面是整理好的 **完整推导报告（Markdown）**，已经结构化为可以直接用于论文草稿 / 技术文档的形式，并包含：

* 从动力学建模 → 映射推导 → 转弯扩展 → 极端情况分析
* 最终可实现公式
* 对 RL 奖励的意义
* 后续改进方向

---

# 📄 Velocity–Gait Mapping for Bipedal Locomotion

## （从线速度与角速度到步态相位的解析建模）

---

# 1. 问题定义

在强化学习（RL）训练人形机器人时，常使用启发式 gait reward（如 `feet_gait`）来约束：

* 双腿相位（phase）
* 支撑/摆动时序（stance/swing）
* 步态周期（period）

当前问题：

> 能否从机器人几何参数出发，建立
> **(v, ω) → gait parameters** 的连续解析映射？

其中：

* ( v )：线速度（范围 ([-1, 1.5]) m/s）
* ( \omega )：角速度（yaw rate）
* 输出目标：

  * 步频 ( f )
  * 支撑比例 ( \alpha )
  * 双腿相位差 ( \Delta \phi )

---

# 2. 基础动力学建模

## 2.1 倒立摆近似（Inverted Pendulum）

将单腿建模为长度为 ( L ) 的摆：

[
f_0 = \frac{1}{2\pi} \sqrt{\frac{g}{L}}
]

意义：

* 给出自然步频尺度
* 决定 gait 的时间基准

---

## 2.2 步速关系

基本关系：

[
v = f \cdot S
]

其中：

* ( S )：步长（stride length）

经验：

* 低速：增加步长
* 高速：步频 + 步长共同增加

---

## 2.3 无量纲化（Froude数）

[
Fr = \frac{v^2}{gL}
]

重要结论：

* gait 主要由 ( Fr ) 控制
* 不同机器人可归一化

---

# 3. 直线行走的步态映射

---

## 3.1 步频函数

[
\boxed{
f(v) = f_0 \cdot \left(1 + k_f \frac{|v|}{\sqrt{gL}}\right)
}
]

* ( k_f \approx 0.5 \sim 1.0 )

---

## 3.2 支撑相比例（stance ratio）

经验规律：

* 静止：100% 支撑
* 快走：接近 50%

建模：

[
\boxed{
\alpha(v) = 0.5 + 0.5 \exp(-k_\alpha |v|)
}
]

* ( k_\alpha \approx 1.5 \sim 3.0 )

---

## 3.3 相位差

直线行走：

[
\boxed{
\Delta\phi = 0.5
}
]

即：

* 左右腿反相（anti-phase gait）

---

# 4. 引入角速度（转弯建模）

---

## 4.1 关键思想：等效线速度

设：

* 髋宽：( w )

则左右腿：

[
\boxed{
v_L = v - \omega \frac{w}{2}, \quad
v_R = v + \omega \frac{w}{2}
}
]

---

## 4.2 步态参数分解

### 步频：

[
f_L = f(v_L), \quad f_R = f(v_R)
]

### 支撑比例：

[
\alpha_L = \alpha(v_L), \quad
\alpha_R = \alpha(v_R)
]

---

## 4.3 相位偏移修正

转弯引入非对称：

[
\boxed{
\Delta\phi =
0.5 + k_\phi \tanh\left(
\frac{\omega w}{2|v| + \epsilon}
\right)
}
]

* ( k_\phi \approx 0.05 \sim 0.15 )

---

# 5. 低速 + 高角速度极限推导（核心贡献）

---

## 5.1 极限条件

[
|v| \ll |\omega| w
]

---

## 5.2 等效速度退化

[
v_L \approx -\omega \frac{w}{2}, \quad
v_R \approx +\omega \frac{w}{2}
]

👉 一条腿 forward，一条 backward

---

## 5.3 步态性质变化

此时不再是 walking，而是：

> **pivot gait（原地转步态）**

特点：

* 内侧腿：长时间支撑
* 外侧腿：频繁摆动

---

## 5.4 对称模型失效

原模型：

[
\alpha(v_L) = \alpha(v_R)
]

❌ 不符合实际

---

## 5.5 非对称修正（关键公式）

引入：

[
\boxed{
\delta =
k_\omega \tanh\left(
\frac{\omega w}{2|v| + \epsilon}
\right)
}
]

最终：

[
\boxed{
\alpha_{L/R} =
\alpha(|v|) \pm \delta
}
]

* ( k_\omega \approx 0.2 \sim 0.3 )

---

## 5.6 相位行为

[
\Delta\phi \to 0.5 \pm k_\phi
]

👉 表现为：

* 一脚几乎固定
* 另一脚 stepping

---

# 6. 最终统一映射（工程可用）

---

## 输入

[
(v, \omega)
]

---

## 输出

### 步频

[
f = \frac{f(v_L) + f(v_R)}{2}
]

---

### 支撑比例

[
\alpha_L = \alpha(|v|) + \delta
]
[
\alpha_R = \alpha(|v|) - \delta
]

---

### 相位

[
\phi_L = 0,\quad
\phi_R = \Delta\phi
]

---

# 7. 对 RL 奖励函数的意义

---

## 7.1 原始问题

你的 `feet_gait`：

```python
is_stance = phase < threshold
```

问题：

* threshold 固定 ❌
* 无法适应速度变化 ❌
* 无转弯建模 ❌

---

## 7.2 改进

替换为：

```python
threshold_L = alpha_L
threshold_R = alpha_R
offset = dphi
period = 1 / f
```

---

## 7.3 本质提升

从：

> heuristic clock

变为：

> **physics-informed gait prior**

---

# 8. 改进方向（重点）

---

## 8.1 从解析模型 → CPG（推荐）

将 phase 建模为：

[
\dot{\phi}*i = \omega_i + \sum_j k*{ij} \sin(\phi_j - \phi_i - \Delta_{ij})
]

优点：

* 自然稳定
* 可学习 coupling

---

## 8.2 引入学习残差（Hybrid）

[
\alpha = \alpha_{\text{model}} + \Delta_\theta(s)
]

👉 RL 学 residual

---

## 8.3 从 reward → tracking loss

替换：

```python
reward += match(contact, phase)
```

为：

```python
loss = ||contact_prob - gait_target||
```

👉 更稳定（尤其 PPO）

---

## 8.4 融入动力学（下一步）

当前缺失：

* COM dynamics
* ZMP / capture point

👉 可升级为：

> **Model-based gait prior**

---

# 9. 总结

---

## 核心结论

✔ 可以建立连续映射：

[
(v, \omega) \rightarrow
(f, \alpha_L, \alpha_R, \Delta\phi)
]

✔ 关键机制：

* Froude scaling（速度）
* differential velocity（转弯）
* asymmetry correction（低速转弯）

---

## 最重要 insight

> **低速大角速度 ≠ 慢走**
>
> 而是：
>
> 👉 **pivot gait（非对称步态）**

---

## 对 RL 的意义

该模型：

* 提供结构先验
* 降低探索难度
* 提升稳定性
* 保持连续可导

---

如果你下一步想做：

👉 “把这个直接嵌进 PPO / IsaacLab 训练框架（包括代码级改写）”
👉 或 “做成类似 LCP 的正则项而不是 reward”

我可以帮你进一步设计一版“更像论文”的实现方案。
