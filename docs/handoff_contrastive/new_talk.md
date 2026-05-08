对，你这两个约束会显著改变规约。特别是：

[
z \neq \text{phase code},\qquad z \neq \text{time code},\qquad z \neq \text{style label}
]

它应该是一个**连续的、相位不变的步态风格坐标**。相位相关的可观测量应该由 ((z,\phi)) 共同解释，而不是由 (z) 单独解释。这样看，当前文档里的 Iso / A1 / A2 / Var 只能算早期 surrogate；如果严格坚持“(z) 是 gait style code”，那么规约应该从 gait feature geometry 出发，而不是从 command geometry 出发。当前项目原目标是让不同速度命令产生有差异的步态；但 V21g 的外在 reward 只关心速度跟踪，对“用哪种步态达到速度”无差别，这正是需要额外表征/内在奖励的原因。

下面我重新给出规约。

---

# 1. 基本符号

设一段观测历史为：

[
h_t=(o_{t-T+1},\ldots,o_t)\in\mathcal H
]

当前速度命令为：

[
c_t=(v_x,v_y,\omega_z)\in\mathbb R^3
]

当前相位为：

[
\phi_t\in S^1
]

实际实现里可以用：

[
\phi_t=(\cos\varphi_t,\sin\varphi_t)
]

encoder 为：

[
z_t=f_\theta(\bar h_t)\in\mathbb R^{d_z}
]

其中：

[
\bar h_t=M(h_t)
]

表示经过 strict mask 后的 history，去掉 command shortcut、mode shortcut、last action shortcut 等。文档里当前 V3 使用 32 帧 segment，runtime 通过 rolling buffer 得到 (z_\text{gait})，这是合理的数据形态；但这次我们要重新定义 (z) 的语义。

定义两类从真实轨迹/segment 中计算出的连续步态属性：

[
y^0_t=\Psi_0(h_t)
]

[
y^\phi_t=\Psi_\phi(h_t,\phi_t)
]

其中：

[
y^0_t
]

是**相位不变步态风格属性**，例如 duty factor、平均步宽、平均步长、步频、平均足端 clearance、平均横向 sway、躯干平均姿态、动作能量、左右脚周期关系等。

[
y^\phi_t
]

是**相位依赖步态属性**，例如当前相位下的足端高度、当前关节构型、当前接触状态、当前左右脚相对位置、当前摆动腿状态等。

你的设计可以概括为：

[
z \longrightarrow y^0
]

[
(z,\phi)\longrightarrow y^\phi
]

而不是：

[
z \longrightarrow \phi
]

也不是：

[
z \longrightarrow \text{discrete gait style label}
]

---

# 2. 规约一：严格非泄露性

encoder 不能直接读 command，也不能读到明显的 command 派生 shortcut。否则 (z) 会变成 command 的非线性编码，而不是步态风格编码。项目历史里 v1 的 loose mask 就出现过这种情况：只屏蔽 `vel_cmd` 不够，结果 (R^2(z\to v_x)=0.997)，被判断为 command 泄露。

形式化为：

[
f_\theta(h)=f_\theta(Mh)
]

更强一点，用干预式写法：

[
h,h' \text{ 具有相同物理步态轨迹，但 command 字段不同}
\Rightarrow
f_\theta(Mh)\approx f_\theta(Mh')
]

信息论写法：

[
I(Z;C_\text{direct}\mid Y^0,M_\text{mode})\le \epsilon_\text{leak}
]

注意，这不要求：

[
I(Z;C)=0
]

因为如果某个 command 确实诱导出某种步态风格，那么 (z) 通过真实步态反推出 command 是允许的。真正不能允许的是：(z) 不通过步态，而是通过输入字段 shortcut 直接恢复 command。

---

# 3. 规约二：相位不变性

这是你这次澄清后最核心的一条。

设 (\tau_\Delta h) 表示对同一周期步态 segment 做时间平移，也就是换一个相位起点，但不改变整体步态风格。

则应满足：

[
|f_\theta(\tau_\Delta h)-f_\theta(h)|\le \epsilon_\phi
]

也就是说，同一种步态风格，无论从周期的哪个相位开始截 segment，都应该得到近似相同的 (z)。

信息论形式是：

[
I(Z;\Phi\mid Y^0)\le \epsilon_\phi
]

或者用 probe 检验：

[
R^2(\Phi\leftarrow Z)\approx 0
]

但：

[
R^2(Y^0\leftarrow Z)
]

应该高。

这条性质会直接影响实现：如果当前 encoder 的 (z) 分支显式吃了 anchor sin/cos，那么必须保证这个相位输入只用于 canonicalization、alignment 或辅助 head，而不能让 phase information 进入最终 (z)。文档里当前 V3 符号表把 anchor sin/cos 作为 encoder 输入之一，这一点在新规约下需要重新检查。

---

# 4. 规约三：相位不变风格充分性

(z) 必须足以解释连续的相位不变步态属性。

存在 decoder：

[
D_0:\mathbb R^{d_z}\to\mathcal Y^0
]

使得：

[
\mathbb E\left[
\ell_0(D_0(z_t),y^0_t)
\right]\le \epsilon_0
]

或者：

[
H(Y^0\mid Z)\ll H(Y^0)
]

更适合做实验的形式是：

[
R^2(Y^0\leftarrow Z)\ge r_0
]

这里的 (Y^0) 不是 label，而是连续步态属性。

当前 V3 用 `prop_head` 回归 duty、yaw、lat、act，并报告 (R^2(z\to duty_l)=0.95)、(R^2(z\to yaw)=0.92)、(R^2(z\to act)=0.91)、(R^2(z\to lat)=0.88)。这说明当前 encoder 至少已经有“(z) 携带部分步态属性”的证据；但在新规约下，`L_prop` 不应该只是诊断 head，而应该变成核心训练/验证目标之一。

---

# 5. 规约四：相位互补性

(z) 不包含相位，但 ((z,\phi)) 应该能解释相位依赖的步态状态。

存在 decoder：

[
D_\phi:\mathbb R^{d_z}\times S^1\to\mathcal Y^\phi
]

使得：

[
\mathbb E\left[
\ell_\phi(D_\phi(z_t,\phi_t),y^\phi_t)
\right]\le \epsilon_\phi^\text{dep}
]

同时，(z) 单独不应该足以解释相位依赖属性：

[
\inf_D \mathbb E[\ell_\phi(D(z_t),y^\phi_t)]
--------------------------------------------

\inf_D \mathbb E[\ell_\phi(D(z_t,\phi_t),y^\phi_t)]
\ge \Delta_\phi
]

这条非常重要。它保证：

[
z = \text{style}
]

[
\phi = \text{where we are in the cycle}
]

[
(z,\phi)=\text{style-conditioned phase state}
]

这正好对应你之前的做法：用周期不变信息监督 (z)，用相位依赖信息监督 ((z,\phi))。

---

# 6. 规约五：步态风格度量性，而不是 command 度量性

因为你明确说 (z) 是步态风格编码，所以 (z) 的距离应该对应真实步态风格差异，而不是优先对应 command 差异。

定义一个连续步态风格距离：

[
d_G(i,j)=d_{\mathcal Y^0}(y^0_i,y^0_j)
]

例如：

[
d_G(i,j)
========

\left|
\Sigma_y^{-1/2}(y^0_i-y^0_j)
\right|_2
]

其中 (\Sigma_y) 是步态属性的协方差或尺度归一化矩阵。

则要求：

[
m_G d_G(i,j)-\epsilon
\le
|z_i-z_j|_2
\le
M_G d_G(i,j)+\epsilon
]

弱一点可以用 rank condition：

[
\rho_\text{Spearman}
\left(
|z_i-z_j|_2,,
d_G(i,j)
\right)
\ge \rho_G
]

这和当前 V3 的：

[
\rho_\text{Spearman}
\left(
|z_i-z_j|,,
|c_i-c_j|_W
\right)
]

不是一回事。当前 RnC 的理论动机是让 representation 保持连续标签空间的样本排序；RNC 论文确实是为连续 regression target 学有序表征而设计的。([NeurIPS 会议论文集][1]) 但如果 target 用的是 command distance，那么学到的是 command-induced geometry；如果 target 用的是 (d_G)，学到的才是 gait-style geometry。

因此，新规约下我会把原来的：

[
\text{Iso}(z,c)
]

替换成：

[
\text{Iso}(z,y^0)
]

或者更准确地说：

[
z \text{ preserves continuous gait-style metric.}
]

---

# 7. 规约六：同 command 下的风格残差可解释

我们不能引入 gait style label，但可以要求：在同一个或相近 command 下，(z) 的变化必须由连续步态属性解释，而不是由噪声、相位或不稳定性解释。

定义相近 command 条件：

[
|c_i-c_j|_W\le \delta_c
]

要求：

[
\rho_\text{Spearman}
\left(
|z_i-z_j|*2,,
d_G(i,j)
\right)
\Big|*{|c_i-c_j|*W\le \delta_c}
\ge \rho*{G|C}
]

也可以定义 conditional probe gain：

[
R^2(Y^0\leftarrow C,M,Z)
------------------------

R^2(Y^0\leftarrow C,M)
\ge \Delta_G
]

这条的含义是：即使 command 和 mode 已知，(z) 仍然解释了额外的步态风格差异。

这比当前的 ibvr 更强。当前 ibvr 只能说明同一 bucket 内 (z) 有方差；但新规约要求这些方差必须能由相位不变步态属性解释。否则 intra-bucket variance 可能只是 phase variance、噪声、接触随机性或动作抖动。当前文档把 ibvr 作为 Var 指标是有价值的，但不充分。

---

# 8. 规约七：非坍缩、有界、抗噪

为了后续在线训练，(z) 不能靠 norm 作弊，也不能退化成低秩常数。

有界性：

[
r_\text{min}
\le
\mathbb E[|Z|^2]
\le
r_\text{max}
]

非坍缩：

[
\mathrm{rank}_\epsilon(\mathrm{Cov}(Z))\ge k
]

抗噪：

[
\mathbb E_\xi
\left[
|f_\theta(h+\xi)-f_\theta(h)|^2
\right]
\le
L_\xi\mathbb E[|\xi|^2]
]

时间平滑性：

[
|z_{t+1}-z_t|
\le
L_G d_G(y^0_{t+1},y^0_t)+\epsilon
]

如果只是相位推进、风格不变，则：

[
d_G(y^0_{t+1},y^0_t)\approx 0
\Rightarrow
|z_{t+1}-z_t|\approx 0
]

这条可以防止 (z) 随周期振荡；周期振荡应该由 (\phi) 表达。

---

# 9. 规约八：可控性

这条不是纯 encoder 性质，而是“encoder 能否用于在线 RL”的必要性质。

定义在 command (c) 下，能够保持任务表现的可行风格集合：

[
\mathcal Z_\text{good}(c)
=========================

\left{
z:
\exists \pi,\
\mathbb E[r_\text{env}\mid c,\pi]\ge R_0,\
z=f_\theta(h^\pi_c)
\right}
]

要想 (z) 在在线训练中有用，至少需要：

[
\mathrm{diam}\left(\mathcal Z_\text{good}(c)\right)>0
]

也就是说，同一个 command 下，在不显著牺牲速度跟踪的情况下，确实存在不止一种可实现的步态风格。

如果目标是“不同 command 下不同 sub-style”，还需要：

[
d_\mathcal P
\left(
\mathcal Z_\text{good}(c_i),
\mathcal Z_\text{good}(c_j)
\right)
\ge
\kappa d_C(c_i,c_j)
]

其中 (d_\mathcal P) 可以是两个可行风格分布的 Wasserstein 距离、MMD、均值距离，或 nearest-set distance。

更干预式地写：

[
\exists \pi_1,\pi_2
\quad
\text{s.t.}
\quad
\mathbb E[r_\text{env}\mid c,\pi_1],
\mathbb E[r_\text{env}\mid c,\pi_2]\ge R_0
]

且：

[
\left|
\mathbb E[Z\mid c,\pi_1]
------------------------

\mathbb E[Z\mid c,\pi_2]
\right|
\ge \delta_z
]

如果没有这条，encoder 再好也只是“测量器”；它测得的风格不一定是 policy 可以主动改变的风格。

---

# 10. 修正后的最小形式化规约

把上面压缩成最小集合，我会写成：

[
z=f_\theta(Mh)
]

满足：

[
\boxed{
I(Z;C_\text{direct}\mid Y^0,M)\le\epsilon_\text{leak}
}
]

[
\boxed{
I(Z;\Phi\mid Y^0)\le\epsilon_\phi
}
]

[
\boxed{
\inf_D\mathbb E[\ell(D(Z),Y^0)]\le\epsilon_0
}
]

[
\boxed{
\inf_D\mathbb E[\ell(D(Z,\Phi),Y^\phi)]\le\epsilon_1
}
]

[
\boxed{
\inf_D\mathbb E[\ell(D(Z),Y^\phi)]
----------------------------------

\inf_D\mathbb E[\ell(D(Z,\Phi),Y^\phi)]
\ge\Delta_\phi
}
]

[
\boxed{
\rho_\text{Spearman}
\left(
|Z_i-Z_j|,
d_G(Y^0_i,Y^0_j)
\right)
\ge \rho_G
}
]

[
\boxed{
\rho_\text{Spearman}
\left(
|Z_i-Z_j|,
d_G(Y^0_i,Y^0_j)
\right)
\Big|_{|C_i-C_j|*W\le\delta_c}
\ge \rho*{G|C}
}
]

[
\boxed{
r_\text{min}
\le
\mathbb E[|Z|^2]
\le
r_\text{max},\qquad
\mathrm{rank}_\epsilon(\mathrm{Cov}(Z))\ge k
}
]

[
\boxed{
\mathrm{diam}\left(\mathcal Z_\text{good}(c)\right)>0
}
]

前八条是 encoder / representation 规约；最后一条是在线 RL 可用性规约。

---

# 11. 假设存在这样的 encoder，它能否诱导 AC 在不同速度命令下产生不同亚步态风格？

答案要分两层。

**仅有 encoder，不够。**

**encoder + 合适的 online objective + 可控风格集合，可以。**

这点和 DIAYN / CIC / SMERL 类方法的核心思想一致：多样性不是因为有一个 embedding 自然出现，而是因为 reward 或 mutual-information objective 让 policy 必须通过行为改变 embedding / outcome。DIAYN 通过最大化信息论目标来学无监督技能；CIC 明确最大化 state transition 与 latent skill vector 的 mutual information，并用行为 embedding 的 entropy 作为 intrinsic reward 来鼓励行为多样性。([Google Research][2]) ([OpenReview][3])

也就是说，(z) 本身只是一个**可测量坐标系**。真正推动 policy 的，是类似下面的目标：

[
J(\pi)
======

\mathbb E_\pi
\left[
r_\text{env}
+
\alpha\cdot r_\text{style}(z,c)
\right]
]

如果没有 (r_\text{style})，外在 reward 只关心速度跟踪，actor 完全可以忽略 (z)。项目里 V22a 已经观察到这一点：frozen encoder 接入 actor 但不发 intrinsic reward，reward 与 V21g 基本打平，结论是 actor 没有梯度压力去“用” (z)。

---

# 12. 如果可以，它是如何工作的？

假设有一个满足上述规约的 encoder，并且在线目标设计正确。闭环机制如下。

actor 接收：

[
(o_t,c_t,m_t,\phi_t,z_t)
]

其中：

[
z_t=f_\theta(h_t)
]

policy 输出动作：

[
a_t\sim \pi_\eta(a_t\mid o_t,c_t,m_t,\phi_t,z_t)
]

动作改变未来轨迹：

[
a_t\rightarrow h_{t+1:t+K}
]

轨迹改变相位不变步态属性：

[
h_{t+1:t+K}\rightarrow y^0_{t+1:t+K}
]

encoder 把它映射成：

[
z_{t+K}=f_\theta(h_{t+1:t+K})
]

style reward 评价：

[
r_\text{style}(z_{t+K},c_t)
]

于是 PPO / actor-critic 的梯度虽然不能直接穿过环境和 encoder，但可以通过 policy gradient 学到：

[
\text{哪些动作序列}
\Rightarrow
\text{哪些 } y^0
\Rightarrow
\text{哪些 } z
\Rightarrow
\text{更高 reward}
]

(\phi) 在这里负责“什么时候抬脚、什么时候落脚、当前周期到哪里”；(z) 负责“用哪种风格执行这个周期”。所以理想状态下，actor 学到的是：

[
\pi(a\mid o,c,\phi,z)
=====================

\text{phase-conditioned controller under style } z
]

这就是你原本“(z) 不含相位，((z,\phi)) 解释相位依赖 gait”的合理性所在。

---

# 13. 但是：当前 V22b 的 residual reward 在新规约下不一定足够

当前 V22b 用的是：

[
r_\text{int}(z,c)
=================

\mathbb 1[|c|_W>\epsilon]
\cdot
\frac{|z-g^*(c)|^2}{d_z}
]

其中 (g^*(c)) 是 axial command projection。文档里的设计意图是让 policy 增大 (z) 中“超出 command 子空间”的残差。

但如果我们现在严格规定：

[
z=\text{phase-invariant gait style code}
]

那么：

[
g^*(c)=\text{command subspace in } z
]

这个假设就变得可疑。

因为新规约下，(z) 不应该先被分解成：

[
z = z_\text{cmd}+z_\text{style}
]

而应该直接是：

[
z=z_\text{style}
]

command 可以影响 (z)，但只能通过实际产生的步态风格影响，而不能作为 (z) 的内在组成部分。

因此，当前 residual reward 可能会出现三个问题。

第一，它奖励“远离 (g^*(c))”，但不保证远离的方向是有意义的步态风格方向：

[
|z-g^*(c)|^2 \uparrow
\not\Rightarrow
d_G(y^0,y^0_\text{meaningful}) \uparrow
]

第二，它不保证不同 command 对应不同 style。policy 可能找到一个全局高 norm、高 residual、还能勉强 tracking 的统一风格，对所有 command 都用它。

第三，如果 encoder 未来在线微调，policy 和 encoder 可能通过改变 (z) 的尺度或坐标系来提高 residual，而不是真正改变步态。这就是早期在线 contrastive 失败里“anchor 在变”的风险，项目历史中也已经观察到在线 encoder 与 policy 漂移强耦合的问题。

所以在你的新定义下，我会说：**满足新规约的 encoder 是必要条件，但 V22b 的 (r_\text{int}=|z-g^*(c)|^2) 不是充分的诱导机制。**

---

# 14. 为了诱导“不同速度命令 → 不同亚步态风格”，还缺哪些性质？

缺三类东西。

## 14.1 缺 command-style coupling 目标

我们需要在线目标直接表达：

[
C \rightarrow Z
]

也就是不同 command 应诱导不同的风格分布。

形式化目标可以是：

[
D\left(
p_\pi(Z\mid C=c_i),
p_\pi(Z\mid C=c_j)
\right)
\ge
\kappa d_C(c_i,c_j)
]

或者最大化：

[
I_\pi(C;Z)
]

但要注意：这里的 (Z) 不能包含实际 base velocity，否则 (I(C;Z)) 会被速度本身 trivially satisfy。(z) 的监督特征里最好排除 raw velocity / command / tracking error，只保留步态形态、接触、周期统计、身体姿态、动作形态等。

一个可实现的 reward 是训练一个在线 discriminator / regressor：

[
q_\psi(c\mid z)
]

并给：

[
r_\text{style}=\log q_\psi(c_t\mid z_t)
]

因为 (f_\theta) 看不到 command shortcut，要让 (c) 从 (z) 中可恢复，policy 必须让不同 command 产生不同的真实步态风格。这和 DIAYN/CIC 里的“让 latent/skill 可以从状态转移中被识别”是同一类信息论机制，只是这里 latent 不是随机 skill id，而是外部 command。([Google Research][2]) ([OpenReview][3])

这不引入 gait style label；command 不是 style label，而是任务条件。

## 14.2 缺可行风格分离性

即使有 (I(C;Z)) 目标，也要确认环境和机器人动力学允许：

[
c_i\neq c_j
\Rightarrow
\mathcal Z_\text{good}(c_i)
\neq
\mathcal Z_\text{good}(c_j)
]

如果不同 command 的可行最优步态风格集合高度重叠，那么 policy 没有理由也没有能力学出显著差异。

因此必须验证：

[
D\left(
\mathcal Z_\text{good}(c_i),
\mathcal Z_\text{good}(c_j)
\right)>0
]

这个可以离线做：从多个已训练 policy / 多个 checkpoint / 多个随机种子 / 不同 reward weights 采样，计算每个 command bucket 的 (z) 分布。如果真实高 reward 行为本身在 (z) 空间不可分，那么 encoder 或目标都需要重做。

## 14.3 缺在线坐标稳定性

如果后续要开启 encoder fine-tuning，就必须加：

[
\mathcal L_\text{anchor}
========================

\mathbb E_{h\sim\mathcal D_\text{ref}}
\left[
|f_{\theta_k}(h)-f_{\theta_0}(h)|^2
\right]
]

或者允许旋转对齐的 Procrustes anchor：

[
\min_A
\mathbb E_{h\sim\mathcal D_\text{ref}}
\left[
|A f_{\theta_k}(h)-f_{\theta_0}(h)|^2
\right]
]

同时保留离线规约损失：

[
\mathcal L_\text{offline}
=========================

\mathcal L_{Y^0}
+
\mathcal L_{Y^\phi}
+
\mathcal L_\text{phase-inv}
+
\mathcal L_\text{metric}
+
\mathcal L_\text{norm}
]

否则 encoder 在线变化会让 actor 面对不断漂移的 (z) 坐标系，V19d-CLP 的“policy 漂移 + encoder 漂移”问题会回来。项目历史已经显示，在线 contrastive 与 policy 漂移强耦合是早期失败根因之一。

---

# 15. 结论：新规约下，应该如何判断当前方案？

我会这样判断。

当前“离线 encoder → actor 拼接 z → 加 intrinsic reward”的总体路线仍然合理，因为外在 reward 本身确实无法区分步态风格，单纯把 (z) 拼进 actor 也确实容易被旁路。

但在你现在澄清的语义下，当前 V3 / V22b 的数学核心需要回溯重写：

[
\text{旧核心：}
\quad
z \text{ 与 command metric 对齐，并保留 residual variance}
]

应改为：

[
\text{新核心：}
\quad
z \text{ 是 phase-invariant gait-style metric space}
]

然后在线阶段目标从：

[
r_\text{int}=|z-g^*(c)|^2
]

改成更直接的：

[
r_\text{style}
\approx
\text{command-conditioned style separation}
]

例如：

[
r_\text{style}=\log q_\psi(c\mid z)
]

或 batch 级目标：

[
|z_i-z_j|
\approx
\kappa d_C(c_i,c_j)
\quad
\text{for high-task-reward samples}
]

同时保留 SMERL gate：

[
r_\text{total}
==============

r_\text{env}
+
\alpha(t)\cdot\mathrm{gate}(r_\text{env})\cdot r_\text{style}
]

因为 gate 的作用仍然合理：先保证速度跟踪，再推动风格分化。

最关键的一句话是：

[
\boxed{
\text{一个好的 } z \text{ 只能让风格“可测、可分、可反馈”；}
}
]

[
\boxed{
\text{要让 AC 真的学出不同 command 下的不同亚步态，还必须有 command-conditioned style objective。}
}
]

如果只有满足规约的 encoder，而没有这样的 objective，那么它不会自动诱导分化；V22a 的现象已经证明了这一点。

[1]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html?utm_source=chatgpt.com "Rank-N-Contrast: Learning Continuous Representations for Regression"
[2]: https://research.google/pubs/diversity-is-all-you-need-learning-skills-without-a-reward-function/?utm_source=chatgpt.com "Diversity is All You Need: Learning Skills without a Reward Function"
[3]: https://openreview.net/forum?id=9HBbWAsZxFt&utm_source=chatgpt.com "Unsupervised Reinforcement Learning with Contrastive Intrinsic Control | OpenReview"


强调：依然分为两步走，分别是离线数据预训练后评估encoder设计是否满足需求，然后接入在线学习闭环迭代。现在，先给出离线数据预训练的设计和实验方案设计，我会基于此进行代码实现

下面给出一个新的**离线预训练设计与实验方案**。我建议把它作为新的 encoder 线，例如命名为：

```text
frnc_style_v4
```

它不是在 V3 上小修，而是把离线阶段的目标从：

[
z \approx \text{command-metric embedding}
]

改成：

[
z \approx \text{phase-invariant gait-style embedding}
]

这更符合你现在明确的设定：**(z) 是步态风格编码，不包含相位/时间信息；相位依赖信息由 ((z,\phi)) 解释；不引入 gait style label。**

---

# 0. 设计目标重述

当前项目的原始目标是：让 RL policy 在不同速度命令 ((v_x,v_y,\omega_z)) 下产生有差异的步态亚风格；历史方案是先离线训练 `history → z`，再在线把 `z` 接入 actor，并给 policy 一个使用 `z` 的内在奖励通道。

但旧 V3 离线目标主要围绕 command metric 展开：

[
d_Z(z_i,z_j)\propto |c_i-c_j|_W
]

并配合 axial bases、Lipschitz、prop head 等约束。这个路线在当时是合理的，但如果现在定义 (z) 为**纯步态风格坐标**，那么旧的 `RnC(cmd-distance) + axial` 不应再作为主目标，只应保留为对照组。旧版中 `loose mask` 导致 (R^2(z\to v_x)=0.997) 的 command 泄露教训仍然必须保留：mask 必须严格。

新的离线阶段应该验证四件事：

[
z \text{ 是否去相位}
]

[
z \text{ 是否保留相位不变的 gait style 信息}
]

[
|z_i-z_j| \text{ 是否对应真实 gait-style 距离}
]

[
z \text{ 是否没有通过 command / mode / action shortcut 作弊}
]

---

# 1. 数据设计

## 1.1 数据单元

每个训练样本建议不是单独一个 32 帧窗口，而是一个**父片段**加若干个 32 帧视图。

父片段：

[
H_i = (o_{t:t+L-1})
]

建议：

```text
L = 64 或 96
```

从父片段中采样多个 runtime-length window：

[
h_i^{(a)} = H_i[t_a:t_a+T-1]
]

[
h_i^{(b)} = H_i[t_b:t_b+T-1]
]

其中：

```text
T = 32
```

这样做的目的是：同一个父片段内，不同起始点往往对应不同相位，但同一段稳定步态风格。于是可以用：

[
f_\theta(h_i^{(a)}) \approx f_\theta(h_i^{(b)})
]

来训练 phase-invariant (z)。

当前 V22a/V22b runtime 里 encoder 接收的是 per-env rolling buffer：

```text
(num_envs, 32, 295) → z_gait: (num_envs, 32)
```

所以离线训练仍然应该保持 runtime 输入形状为 (32\times295)，否则后续接入在线阶段会产生 train/runtime mismatch。

---

## 1.2 数据来源

不要只用单一 final policy 的 rollout。否则同一个 command 下可能只有一种步态，encoder 学不到“亚风格空间”，只能学到当前 policy 的行为投影。

建议至少收集四类数据：

```text
D_final:
  V21g / 当前最强 baseline final checkpoint rollout

D_ckpt:
  同一 policy 的多个训练中间 checkpoint
  例如 2k, 5k, 10k, 15k, 20k

D_seed:
  多个 random seed 训练出的 policy rollout

D_perturb:
  在保持 tracking 不崩的前提下，加轻微 action noise / domain randomization / reward weight variants
```

目的不是引入 style label，而是让同一或相近 command 下自然出现不同连续 gait features。

最终数据集：

[
\mathcal D
==========

{H_i,c_i,m_i,\phi_{i,1:L},Y^0_i,Y^\phi_{i,1:L},\text{meta}_i}
]

其中：

* (c_i)：velocity command，只用于分层采样、probe、conditional metric，不直接输入 encoder；
* (m_i)：mode，standing / pure_wz / other，只用于分层与评估；
* (\phi)：AC 网络中已有的相位信号，用于 ((z,\phi)) decoder；
* (Y^0)：相位不变 gait-style features；
* (Y^\phi)：相位依赖 gait features；
* `meta`：policy id、checkpoint id、seed、episode id、domain randomization 参数等，仅用于数据划分和 probe。

---

## 1.3 command / mode 分布

采样时要分层，避免数据被 high-speed forward walking 主导。

建议 bucket：

```text
standing:
  ||c||_W <= eps_stand

pure_wz:
  |ωz| > eps_wz 且 sqrt(vx²+vy²) < eps_xy

low_xy:
  sqrt(vx²+vy²) in low range, |ωz| small

mid_xy:
  中速平移

high_xy:
  高速平移

mixed:
  vx/vy/ωz 同时非零
```

训练主目标建议：

```text
standing:
  不参与 metric/RnC 主损失，或低权重参与 neutral reconstruction

pure_wz + other:
  参与完整 style learning
```

原因是 standing 没有周期步态，强行让 standing 参与 gait-style metric 容易让 (z) 学到姿态噪声。在线阶段 3-way mode token 本来就必须保留，`z_gait` 不应该替代 mode token。旧项目中 3-way mode isolation 是硬约束，standing、pure_wz、other 是不同策略先验，不能去掉。

---

# 2. Encoder 输入 mask

## 2.1 推荐主 mask：保守 style mask

当前每个 policy obs frame 是：

```text
[base_lin_vel(3)
 | base_ang_vel(3)
 | projected_gravity(3)
 | vel_cmd(3)
 | lin_speed_token(1)
 | gait_mode_token(3)
 | joint_pos_rel(15)
 | joint_vel_rel(15)
 | last_action(15)
 | contact(2)] × 5
```

当前离线 / runtime flat obs 为 (295=5\times59)，最新帧 `vel_cmd` slice 在 flat obs 中是 `[42:45]`。

新的 (z) 是 gait style，不是 command / action / phase，所以我建议主实验使用更保守的 mask：

```text
保留:
  projected_gravity
  joint_pos_rel
  contact

屏蔽:
  base_lin_vel
  base_ang_vel
  vel_cmd
  lin_speed_token
  gait_mode_token
  joint_vel_rel
  last_action
```

原因：

* `vel_cmd`、`lin_speed_token`、`gait_mode_token` 是直接 command / mode shortcut；
* `last_action` 是 policy 看 command 后输出的结果，会间接泄露 command；
* `base_lin_vel` / `base_ang_vel` 很容易让 (z) 直接编码实际速度，从而在未来在线阶段把 command-style coupling 变成“速度泄露”；
* `joint_vel_rel` 对相位和动作派生信息非常敏感，容易让 (z) 带入 phase / timing。

旧版 strict mask 已经证明：靠 adversarial decoder 修泄露不如直接堵掉泄露通道。

---

## 2.2 mask 消融

做两个输入 mask 对照：

```text
M0_conservative:
  projected_gravity + joint_pos_rel + contact

M1_rich:
  M0 + joint_vel_rel + high-pass base motion

M2_old_strict:
  复现旧 V3 strict mask，用于对比
```

主线用 `M0_conservative`。如果 `M0` 的 (Y^0) 解码能力不足，再考虑 `M1_rich`。但如果 `M1` 显著提高 (R^2(Y^0\leftarrow Z)) 的同时也提高 phase leakage 或 command leakage，应继续使用 `M0`。

---

# 3. 步态特征设计

不能引入 gait style label，但可以使用从轨迹中计算出的连续特征。

## 3.1 相位不变 gait-style features：(Y^0)

[
Y^0_i = \Psi_0(H_i)
]

它们是一个父片段的统计量，不依赖当前相位。

建议按 feature group 组织：

### A. contact / duty features

```text
duty_l, duty_r
double_support_ratio
single_support_l_ratio
single_support_r_ratio
no_contact_ratio
contact_switch_rate_l/r
left_right_contact_correlation
left_right_phase_lag_from_contact
```

### B. cadence / timing features

```text
step_frequency
stride_period
stance_duration_mean/std
swing_duration_mean/std
contact_transition_entropy
```

这些是“时间尺度 / 频率”意义上的 style，不是当前相位。允许 (z) 表示步频，但不允许 (z) 表示“现在在周期第几度”。

### C. foot trajectory features

需要数据收集脚端状态，建议扩展 dump：

```text
foot_clearance_mean_l/r
foot_clearance_max_l/r
foot_height_amp_l/r
foot_forward_range_l/r
foot_lateral_range_l/r
step_width_mean/std
step_length_proxy_l/r
foot_slip_mean_l/r
```

如果当前 dump 没有 foot state，建议升级 `collect_pretrain_data.py` 保存 feet body positions / velocities。旧代码中 `frnc_gait_features.py` 只计算了 duty、yaw、lat、act 等少数属性，作为 V3 的 `L_prop` 诊断已经够用，但对新规约太少。

### D. body style features

```text
base_height_mean/std
roll_mean/std
pitch_mean/std
projected_gravity_amp
lateral_sway_amp
vertical_bounce_amp
torso_yaw_osc_amp
```

注意不要直接用 mean base velocity 作为 style target。否则 (z) 会学成实际速度编码。

### E. joint / action morphology features

```text
joint_rom_15d
joint_pos_mean_15d
joint_pos_std_15d
left_right_joint_symmetry
action_rms_15d
action_delta_rms_15d
energy_proxy
smoothness_proxy
```

`action_*` 可以作为 target，但不要作为 encoder input。这样 (z) 只能通过身体运动形态间接预测动作风格。

---

## 3.2 相位依赖 gait features：(Y^\phi)

[
Y^\phi_{i,k} = \Psi_\phi(H_i,k)
]

它们是某个时刻 / 相位下的状态。

建议包含：

```text
contact_l/r at frame k
foot_height_l/r at frame k
foot_pos_base_l/r at frame k
foot_vel_base_l/r at frame k
joint_pos_rel_15d at frame k
projected_gravity at frame k
optional: action at frame k
```

然后训练：

[
D_\phi(z_i,\phi_{i,k})\to Y^\phi_{i,k}
]

这正好实现你的设定：

[
z \to Y^0
]

[
(z,\phi)\to Y^\phi
]

但不让：

[
z\to \phi
]

---

# 4. 网络结构

## 4.1 Style encoder

建议第一版不要上复杂 Transformer，先用稳定结构：

```text
input: h_masked ∈ R^{32×295}

per-frame encoder:
  Linear/MLP on each frame

temporal encoder:
  TCN 或 small Transformer

phase-invariant pooling:
  mean pooling + std pooling over time
  或 attention pooling without positional encoding

projection:
  MLP → z ∈ R^32
```

关键约束：

```text
encoder 不输入 φ
encoder 不输入 command
encoder 不输入 mode token
encoder 不输入 last_action
```

如果使用 Transformer，建议**不要给最终 style branch 加绝对时间 positional encoding**，否则容易学到窗口内相位位置。可以用无位置编码的 temporal attention，或者使用 TCN + global pooling。

输出分两路：

```text
z:        给 decoder / 未来在线 actor 使用，不强制单位范数
p = q(z): 给 RnC / contrastive metric 使用，可 L2 normalize
```

这样可以避免 metric loss 对 actor 使用的 (z) 尺度施加过强约束。

---

## 4.2 Decoders

```text
D0(z) → Y0
Dphi(z, sinφ, cosφ) → Yφ
```

`D0` 用于相位不变 style decoding。

`Dphi` 用于验证 (z) 与 (\phi) 的互补性。

上线时只需要 encoder；decoder 仅用于离线训练和 probe。

---

# 5. 损失函数

总损失建议：

[
\mathcal L
==========

\lambda_0\mathcal L_{Y^0}
+
\lambda_\phi\mathcal L_{Y^\phi}
+
\lambda_\text{rnc}\mathcal L_\text{RNC-G}
+
\lambda_\text{res}\mathcal L_\text{RNC-res}
+
\lambda_\text{inv}\mathcal L_\text{shift-inv}
+
\lambda_\text{var}\mathcal L_\text{var}
+
\lambda_\text{cov}\mathcal L_\text{cov}
+
\lambda_\text{smooth}\mathcal L_\text{smooth}
]

---

## 5.1 相位不变特征回归

[
\mathcal L_{Y^0}
================

\mathbb E_i
\left[
\mathrm{Huber}
\left(
D_0(z_i),\widetilde Y^0_i
\right)
\right]
]

其中 (\widetilde Y^0) 是标准化后的 feature。

这一项保证：

[
z \text{ contains phase-invariant gait style information.}
]

---

## 5.2 相位依赖特征回归

对父片段中采样若干 frame (k)：

[
\mathcal L_{Y^\phi}
===================

\mathbb E_{i,k}
\left[
\mathrm{Huber}
\left(
D_\phi(z_i,\sin\phi_{i,k},\cos\phi_{i,k}),
\widetilde Y^\phi_{i,k}
\right)
\right]
]

对于 binary contact 可以用 BCE，对于连续 foot/joint/body state 用 Huber。

这一项验证：

[
(z,\phi) \text{ can reconstruct phase-dependent gait state.}
]

它不要求 (z) 单独解释相位依赖状态。

---

## 5.3 gait-style metric RnC

旧 V3 的 RnC target 是 command distance。新版本改成 gait-style feature distance：

[
d_G(i,j)
========

\left|
\Sigma_0^{-1/2}
\left(
Y^0_i-Y^0_j
\right)
\right|_2
]

然后用 RnC：

[
\mathcal L_\text{RNC-G}
=======================

\frac{1}{B(B-1)}
\sum_i
\sum_{j\ne i}
-\log
\frac{
\exp(\mathrm{sim}(p_i,p_j)/\tau)
}{
\sum_{k:d_G(i,k)\ge d_G(i,j)}
\exp(\mathrm{sim}(p_i,p_k)/\tau)
}
]

这里 (p_i=q(z_i))，通常 L2 normalize，(\tau=0.1) 固定。

RnC 本来就是为连续 regression target 学有序 representation 设计的，它根据 target-space 中的样本排序来约束 representation-space 排序；这和我们希望 (|z_i-z_j|) 对应连续 gait-style distance 是匹配的。([NeurIPS 会议论文集][1])

---

## 5.4 conditional residual metric

仅用 (d_G(Y^0_i,Y^0_j)) 有一个风险：如果数据中 (Y^0) 与 command 强相关，(z) 仍可能主要学到 command-induced style，而不是同 command 下的 residual sub-style。

所以建议额外拟合一个冻结的 baseline：

[
\mu(c,m)\approx \mathbb E[Y^0\mid c,m]
]

定义 residual style feature：

[
R^0_i
=====

Y^0_i-\mu(c_i,m_i)
]

然后在相近 command 内做 residual metric：

[
\mathcal N_i
============

{j:|c_i-c_j|_W\le \delta_c,\ m_i=m_j}
]

[
d_R(i,j)
========

\left|
\Sigma_R^{-1/2}
(R^0_i-R^0_j)
\right|_2
]

[
\mathcal L_\text{RNC-res}
=========================

\mathrm{RNC}
\left(
p_i,p_j; d_R(i,j)
\right)
\quad
j\in \mathcal N_i
]

这一项不引入 style label。它只是让 (z) 在相近 command 下也能保留连续步态差异。

如果这一项完全学不起来，通常不是 loss 失败，而是数据中同 command 下缺乏真实风格多样性。

---

## 5.5 phase-shift invariance

从同一个父片段采样两个不同相位窗口：

[
h_i^{(a)}, h_i^{(b)}
]

要求：

[
z_i^{(a)}=f_\theta(h_i^{(a)})
]

[
z_i^{(b)}=f_\theta(h_i^{(b)})
]

[
\mathcal L_\text{shift-inv}
===========================

\mathbb E_i
\left[
|z_i^{(a)}-z_i^{(b)}|_2^2
\right]
]

这条是“(z) 不含相位”的主约束。类似时间/视角不变表征学习中通过构造正样本视图来消除 nuisance 的思路；Time-Contrastive Networks 也是通过对比不同视图/时间结构学习不变表示，Temporal Cycle-Consistency 也强调用时间对应关系学习可对齐的时序表征。([Google Research][2])

---

## 5.6 非坍缩与去冗余

如果强行让不同 phase window 的 (z) 靠近，可能出现 collapse：

[
z_i = \text{constant}
]

所以需要显式非坍缩项。建议借鉴 VICReg 的 variance / covariance regularization：

[
\mathcal L_\text{var}
=====================

\frac{1}{d}
\sum_{k=1}^d
\max(0,\gamma-\mathrm{Std}(Z_k))^2
]

[
\mathcal L_\text{cov}
=====================

\frac{1}{d}
\sum_{i\ne j}
\mathrm{Cov}(Z)_{ij}^2
]

VICReg 的动机就是通过 variance、invariance、covariance 三类项显式避免 representation collapse 和维度冗余；这里我们只借用其非坍缩与去冗余思想。([Hugging Face][3])

---

## 5.7 时间平滑

相邻窗口如果 (Y^0) 变化很小，(z) 不应跳变：

[
\mathcal L_\text{smooth}
========================

\mathbb E_{i,t}
\left[
w_{i,t}
|z_{i,t+1}-z_{i,t}|_2^2
\right]
]

其中：

[
w_{i,t}
=======

\exp
\left(
-\frac{|Y^0_{i,t+1}-Y^0_{i,t}|^2}{\sigma_y^2}
\right)
]

这防止 (z) 随周期振荡。

---

# 6. 推荐初始超参

第一版不要调太多，先跑一个主配置：

```python
d_z = 32
batch_size = 512 或 1024
optimizer = AdamW
lr = 3e-4
weight_decay = 1e-4
epochs = 50
tau = 0.1  # fixed, not learnable
huber_delta = 1.0
```

loss 权重初值：

```python
lambda_y0      = 1.0
lambda_yphi    = 1.0
lambda_rnc     = 0.5
lambda_res     = 0.5
lambda_inv     = 1.0
lambda_var     = 0.1
lambda_cov     = 0.01
lambda_smooth  = 0.05
```

如果训练不稳，优先降低：

```text
lambda_rnc
lambda_res
```

不要学习 temperature。旧 V19d-CLP 中 learnable τ 曾经 collapse 到下界，导致 loss 数值看似变好但表征物理意义消失。

---

# 7. 离线评估指标

离线评估应该分成六组。

---

## 7.1 (Y^0) 解码能力

训练冻结 encoder 后，用 train / val / OOD policy split 分别评估：

[
R^2(Y^0\leftarrow Z)
]

按 feature group 报告：

```text
contact/duty R²
cadence/timing R²
foot trajectory R²
body style R²
joint/action morphology R²
macro R²
```

同时训练 baseline：

[
Y^0 \leftarrow (c,m)
]

然后看增益：

[
\Delta R^2_{Y^0}
================

## R^2(Y^0\leftarrow Z,c,m)

R^2(Y^0\leftarrow c,m)
]

验收建议：

```text
macro R²(Y0 ← Z) >= 0.60
至少 3 个 feature group R² >= 0.70
ΔR²(Y0 ← Z,c,m over c,m) >= 0.15
```

如果 (R^2(Y^0\leftarrow Z)) 高，但 (\Delta R^2) 很低，说明 (z) 大概率只学到了 command-induced style，而不是可用于亚风格分化的 residual style。

---

## 7.2 phase leakage

训练 probe：

[
\hat\phi = P_\phi(z)
]

评估：

```text
R²(sinφ, cosφ ← Z)
phase classification accuracy over phase bins
circular correlation(Z, φ)
```

验收建议：

```text
R²(sinφ, cosφ ← Z) <= 0.05
phase-bin accuracy 接近随机
```

同时评估 shift invariance ratio：

[
\rho_\text{shift}
=================

\frac{
\mathbb E[|z(h^{(a)})-z(h^{(b)})|]
}{
\mathbb E[|z_i-z_j|]_{i\ne j}
}
]

其中 (h^{(a)},h^{(b)}) 是同父片段不同相位窗口。

验收建议：

```text
rho_shift <= 0.20
```

如果 phase leakage 高，优先检查：

```text
encoder 是否输入了 φ / anchor sin-cos
是否用了 absolute positional encoding
joint_vel_rel / base_ang_vel 是否泄露 phase
shift-invariance 正样本是否构造正确
```

---

## 7.3 phase complementarity

训练三个 decoder / probe：

```text
D_z:       Z → Yφ
D_phi:     φ → Yφ
D_z_phi:  (Z, φ) → Yφ
```

要求：

[
R^2(Y^\phi\leftarrow Z,\phi)

>

R^2(Y^\phi\leftarrow \phi)
]

并且：

[
R^2(Y^\phi\leftarrow Z,\phi)

>

R^2(Y^\phi\leftarrow Z)
]

验收建议：

```text
R²(Yφ ← Z,φ) >= 0.60
R²(Yφ ← Z,φ) - R²(Yφ ← φ) >= 0.10
R²(Yφ ← Z,φ) - R²(Yφ ← Z) >= 0.20
```

解释：

* 如果 (Z\to Y^\phi) 单独很强，说明 (z) 可能带 phase；
* 如果 (\phi\to Y^\phi) 和 ((z,\phi)\to Y^\phi) 差不多，说明 (z) 没有提供 style 信息；
* 理想情况是 ((z,\phi)) 明显最好。

---

## 7.4 gait-style metric

评估整体 Spearman：

[
\rho_G
======

\rho_\text{Spearman}
\left(
|z_i-z_j|,
d_G(Y^0_i,Y^0_j)
\right)
]

还要评估相近 command 条件下的 Spearman：

[
\rho_{G|C}
==========

\rho_\text{Spearman}
\left(
|z_i-z_j|,
d_G(Y^0_i,Y^0_j)
\right)
\quad
\text{where }
|c_i-c_j|_W\le \delta_c
]

验收建议：

```text
rho_G >= 0.60
rho_G|C >= 0.35 ~ 0.45
```

条件指标比整体指标更重要。整体 metric 很容易被 command-induced gait difference 撑高；相近 command 内仍然有序，才说明 (z) 有亚风格价值。

再加 retrieval 评估：

```text
给定 z_i，找 kNN(z_i)
计算 kNN 的 dG(Y0_i,Y0_j)
与随机邻居对比
```

验收建议：

```text
kNN dG <= 50% random dG
kNN phase difference 分布接近随机
```

第二条很重要：最近邻应该风格近，而不是相位近。

---

## 7.5 command leakage / shortcut probe

不要简单要求：

[
R^2(c\leftarrow z)\approx 0
]

因为真实 gait style 本来可能随 command 变化。更合理的 leakage 检查是增量解释力：

[
\Delta R^2_c
============

## R^2(c\leftarrow z,Y^0,m)

R^2(c\leftarrow Y^0,m)
]

验收建议：

```text
ΔR²_c <= 0.05
```

也做干预测试：

```text
同一个物理 segment，
随机替换 obs 中 command / mode / token / last_action 字段，
重新过 mask + encoder，
要求 z 不变。
```

验收建议：

```text
mean ||z_original - z_intervened|| <= 1e-4 ~ 1e-3
```

如果这个失败，说明 mask 实现有 bug。

---

## 7.6 非坍缩 / 有界性

报告：

```text
mean ||z||
std ||z||
per-dim std
effective rank of Cov(Z)
off-diagonal covariance mean
largest eigenvalue ratio
```

验收建议：

```text
effective_rank(Cov(Z)) >= 8 或 12
per-dim std median >= 0.05
largest_eigenvalue / trace <= 0.5
mean ||z|| 在 train/val/OOD 上无明显漂移
```

如果 metric / reconstruction 好但 effective rank 很低，后续在线阶段很可能只能利用一两个维度，表达空间太窄。

---

# 8. 实验矩阵

建议不要一次性跑太多。第一轮跑 8 个配置即可。

## 8.1 旧方案对照

```text
E0_old_v3_reproduce:
  RnC(cmd-distance) + axial + lip + prop
```

目的：确认新评估下旧 V3 到底失败在哪里。

预期：

```text
cmd Spearman 高
Y0 部分可解码
phase leakage 未知
conditional gait metric 可能不够
```

---

## 8.2 新方案主线

```text
E1_reg_only:
  L_Y0 + L_Yφ

E2_reg_inv:
  L_Y0 + L_Yφ + L_shift_inv + VICReg

E3_reg_inv_rnc:
  E2 + RNC-G

E4_full:
  E3 + RNC-res + smooth
```

预期：

```text
E1:
  容易解码，但 metric 未必好，phase 可能泄漏

E2:
  phase leakage 应显著下降

E3:
  gait metric 应显著提升

E4:
  conditional same-command metric 应提升
```

---

## 8.3 输入 mask 对照

```text
E5_full_M1_rich:
  E4 + rich mask

E6_full_M2_old_strict:
  E4 + old strict mask
```

目的：判断保守 mask 是否损失太多 style 信息，以及 rich mask 是否引入 phase/command leakage。

---

## 8.4 数据多样性对照

```text
E7_full_single_policy:
  只用 final policy 数据

E8_full_multi_policy:
  final + checkpoints + seeds + perturbations
```

重点看：

```text
rho_G|C
ΔR²(Y0 over command baseline)
same-command z diversity
```

如果 `E8` 明显优于 `E7`，说明当前瓶颈不是 encoder 架构，而是旧数据缺乏同 command 下的风格多样性。

---

# 9. 验收表

建议每个 ckpt 输出一张表：

| 指标组                |                      指标 |   通过标准 | 意义                    |                   |
| ------------------ | ----------------------: | -----: | --------------------- | ----------------- |
| style decoding     |      macro (R^2(Y^0←Z)) | ≥ 0.60 | z 有风格信息               |                   |
| style decoding     | (\Delta R^2(Y^0←Z,c,m)) | ≥ 0.15 | z 超过 command baseline |                   |
| phase leakage      |           (R^2(\phi←Z)) | ≤ 0.05 | z 不含相位                |                   |
| phase invariance   |             shift ratio | ≤ 0.20 | 不同相位同风格 z 接近          |                   |
| phase complement   |    (R^2(Y^\phi←Z,\phi)) | ≥ 0.60 | z 与 φ 互补              |                   |
| phase complement   |        gain over φ-only | ≥ 0.10 | z 提供 style 条件         |                   |
| metric             |                (\rho_G) | ≥ 0.60 | z 距离对应 gait distance  |                   |
| conditional metric |                (\rho_{G |    C}) | ≥ 0.35–0.45           | 同 command 下有亚风格度量 |
| leakage            | (\Delta R^2(c←z,Y^0,m)) | ≤ 0.05 | 无额外 command shortcut  |                   |
| non-collapse       |          effective rank | ≥ 8–12 | 表达空间没坍缩               |                   |
| OOD                |    OOD macro (R^2) drop |  ≤ 20% | 泛化到新 policy / seed    |                   |

这张表比旧版 `spearman(z,cmd), lip_med, ibvr, R²(z→cmd)` 更符合新目标。旧版这些指标可以继续报告，但只能作为附录诊断，不应作为主验收。

---

# 10. 失败诊断逻辑

## 10.1 (Y^0) 解码低

可能原因：

```text
输入 mask 太保守
Y0 features 从 obs 中不可观测
segment 太短，不够覆盖一个步态周期
数据中 gait feature 噪声太大
```

处理：

```text
增加 L 到 96 父片段，仍用 32 runtime window
加入 foot state / base state 到 feature computation，而不是输入
试 M1_rich mask
降低 RNC 权重，先让 supervised decoding 成立
```

---

## 10.2 phase leakage 高

可能原因：

```text
encoder 输入了 anchor sin/cos 或绝对时间编码
joint_vel_rel / base_ang_vel 泄露相位
shift-inv 正样本不是真同风格
segment 太短导致 encoder 只能用当前相位猜风格
```

处理：

```text
encoder 不输入 φ
去掉 absolute positional encoding
使用同父片段多相位窗口做 positive
提高 λ_inv
使用 mean/std pooling
```

---

## 10.3 整体 metric 高，但 conditional metric 低

这说明：

[
z \text{ 主要学到了 command-induced gait difference}
]

但同 command 下的 residual sub-style 不明显。

处理：

```text
增加 multi-policy / multi-checkpoint 数据
增强 RNC-res 权重
平衡相近 command pair 采样
检查同 command 下 Y0 本身是否有方差
```

如果同 command 下 (Y^0) 方差本来很低，说明离线数据不足以支持“亚风格 encoder”验证，需要重新收集更丰富的数据，而不是继续调 loss。

---

## 10.4 phase complementarity 不成立

如果：

[
R^2(Y^\phi←Z,\phi) \approx R^2(Y^\phi←\phi)
]

说明 (z) 没有提供 style 条件。

如果：

[
R^2(Y^\phi←Z) \text{ 很高}
]

说明 (z) 可能带 phase。

处理：

```text
前者：加强 Y0 / RNC-G / 数据多样性
后者：加强 shift-inv，去掉 phase 泄露输入
```

---

## 10.5 non-collapse 失败

处理：

```text
提高 λ_var
降低 λ_inv
增大 batch size
检查是否所有 positive view 构造过宽，导致不同 style 被拉近
```

---

# 11. 产物格式

训练完成后，每个 ckpt 应保存：

```text
encoder.pt:
  encoder_state_dict
  encoder_config
  mask_spec
  obs_layout
  z_dim
  train_feature_spec
  feature_mean_std
  y0_feature_names
  yphi_feature_names
  phase_spec
  data_source_manifest
```

还要保存：

```text
probe_report.json
probe_report.csv
feature_r2_by_group.csv
metric_spearman_by_bucket.csv
phase_leakage_report.csv
z_stats.json
```

decoder 可以保存，但在线阶段不必加载：

```text
D0_state_dict
Dphi_state_dict
projection_head_state_dict
```

在线只需要：

```text
h_masked → encoder → z
```

---

# 12. 最小代码落地路径

基于当前工程结构，可以新增三类脚本：

```text
scripts/rsl_rl/collect_style_pretrain_data.py
scripts/rsl_rl/style_gait_features.py
scripts/rsl_rl/style_encoder_pretrain_v4.py
scripts/rsl_rl/style_encoder_probe_v4.py
```

其中旧代码已有：

```text
collect_pretrain_data.py
frnc_gait_features.py
frnc_segment_pretrain_v3.py
frnc_segment_probe.py
```

这些可以作为模板，但不要直接沿用 V3 的主目标，因为 V3 的训练入口和 probe 仍围绕 `spearman / R² / lip_med / ibvr / axial_R²` 设计。

---

# 13. 我建议第一轮实现的最小版本

先不要一步到位实现所有 feature。第一轮可以这样做：

## 数据

```text
父片段 L=64
runtime window T=32
数据源：V21g final + 3 个中间 checkpoint + 2 个 seed
过滤：非 standing，且 tracking reward 不太差
```

## 输入

```text
M0_conservative:
  projected_gravity + joint_pos_rel + contact
```

## (Y^0)

先实现 20–40 维：

```text
duty_l/r
double_support
contact_switch_rate
step_frequency
left_right_phase_lag
joint_pos_mean/std/rom
projected_gravity_mean/std
lateral_sway_proxy
vertical_bounce_proxy
action_rms/delta_action_rms  # target only
```

## (Y^\phi)

先实现：

```text
contact_l/r
joint_pos_rel
projected_gravity
optional action
```

## 模型

```text
MLP frame encoder
TCN temporal encoder
mean+std pooling
z_dim=32
D0, Dphi, projection head
```

## 损失

第一轮主配置：

```text
L_Y0
L_Yphi
L_shift_inv
VICReg var/cov
RNC-G
RNC-res
```

## 对照

```text
E1_reg_only
E2_reg_inv
E3_reg_inv_rnc
E4_full
E0_old_v3_reproduce
```

如果 `E4_full` 满足：

```text
Y0 macro R² >= 0.60
phase R² <= 0.05
R²(Yφ ← Z,φ) 比 φ-only 高 >= 0.10
rho_G >= 0.60
rho_G|C >= 0.35
effective_rank >= 8
```

就可以认为：离线 encoder 设计基本满足“phase-invariant gait-style code”的需求，可以进入第二步在线闭环设计。

---

# 14. 关键判断

这次离线预训练的主问题不再是：

[
z \text{ 是否能预测 command}
]

而是：

[
z \text{ 是否能作为不含相位的连续步态风格坐标}
]

所以旧 V3 中类似：

```text
spearman(z, cmd) = 0.877
R²(z → cmd) = 0.974
axial_R² < 0.15
```

这些结果不能再作为“encoder 成功”的主要证据。它们只能说明旧 encoder 学到了 command-conditioned behavior geometry。旧版 V3 的 `prop_head` 高 (R^2) 仍然是有用信号，但需要扩展成系统性的 (Y^0)、(Y^\phi)、phase leakage、conditional gait metric 评估。

新的离线验收一句话是：

[
\boxed{
z \text{ 对相位不敏感，对 gait style 敏感，且 } |z_i-z_j| \text{ 对应连续 gait-style 距离。}
}
]

只有这个成立，后续在线阶段再让 actor 通过 reward / mutual-information objective 去改变 (z)，才有可能真正诱导出不同速度命令下的不同亚步态风格。

[1]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html?utm_source=chatgpt.com "Rank-N-Contrast: Learning Continuous Representations for Regression"
[2]: https://research.google/pubs/pub46030?utm_source=chatgpt.com "Time-Contrastive Networks: Self-Supervised Learning from Video"
[3]: https://huggingface.co/papers/2105.04906?utm_source=chatgpt.com "Paper page - VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"

