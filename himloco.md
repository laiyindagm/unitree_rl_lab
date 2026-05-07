## 3.3 混合内部优化

对于内部嵌入 $(\hat{v}_t, \hat{l}_t)$ 来说，最关键的是如何优化它以模拟机器人的响应，这种响应自然地嵌入在机器人的后继状态 $o^a_{t+1}$ 中。根据IMC原理，我们可以直接估计给定 $o^a_{t-H:t}$ 的 $o^a_{t+1}$。然而，由于机器人状态的高维度以及仿真与现实的差异，这具有挑战性。或者，考虑到我们在仿真环境中训练框架，我们可以直接学习使用其真实值来回归显式速度 $\hat{v}_t$。对于隐式响应 $\hat{l}_t$，我们提出将其建模到一个潜在空间 $\mathcal{Z} \subset \mathbb{R}^{16}$ 中，并通过对比学习优化使其接近后继状态。

在实践中，在每次迭代中，我们在每个环境中收集轨迹作为一个批次。如果一对 $o^a_{t+1}$ 和 $o^a_{t-H:t}$ 属于同一条轨迹，它们是正样本对；否则，它们是负样本对。这些样本对通过类似于SwAV（Caron等人，2020）的交换分配任务进行优化。给定从rollout轨迹中采样的一系列本体感知观测，我们可以从一次转移中推导出后续观测 $o^a_{t+1}$ 作为目标向量，并将拼接的历史观测 $o^a_{t-H:t}$ 视为源向量。我们确保增强在不同时间步之间保持一致。源向量和目标向量分别输入到源编码器和目标编码器中，以获得潜在特征 $l^{source}_t$ 和 $l^{target}_t$，这些特征通过L2归一化映射到高维空间中的单位球面上。为了从 $l^{source}_t$ 和 $l^{target}_t$ 预测聚类分配概率 $p^{source}_t$ 和 $p^{target}_t$，我们首先对原型应用L2归一化以获得归一化矩阵 $E = \{\bar{e}_1, ..., \bar{e}_K\}$，然后对源向量或目标向量与所有原型的点积取softmax：

$$
p^{source}_t = \frac{\exp\left(\frac{1}{\tau} {l^{source}_t}^\top e_k\right)}{\sum_{k'} \exp\left(\frac{1}{\tau} {l^{source}_t}^\top e_{k'}\right)}, \quad p^{target}_t = \frac{\exp\left(\frac{1}{\tau} {l^{target}_t}^\top e_k\right)}{\sum_{k'} \exp\left(\frac{1}{\tau} {l^{target}_t}^\top e_{k'}\right)}. \tag{1}
$$

这里，$p^{source}_t$ 和 $p^{target}_t$ 是历史观测 $o^a_{t-H:t}$ 映射到索引为 $k$ 的单个聚类的预测概率，而 $\tau$ 是温度参数。

为了获得上述预测概率的目标 $(q^{source}_1, ..., q^{source}_K)$ 和 $(q^{target}_1, ..., q^{target}_K)$，同时避免平凡解，我们对两个编码器应用Sinkhorn-Knopp算法（Cuturi，2013）。现在我们有了聚类分配预测和目标，表示学习的目标就是最大化预测准确率：

$$
J_{SwAV} = -\frac{1}{2H} \sum_{t=1}^{H} \left(q^{source}_t \log p^{target}_t + q^{target}_t \log p^{source}_t\right). \tag{2}
$$

**分析**。我们的方法可以最大化历史观测与下一观测之间潜在特征的相似性，隐式地建模外部状态，而不需要回归。这也通过利用批级信息（即不同类型地形中的不同环境属性）提高了运动策略的性能。我们对我们的方法和回归方法（Nahrendra等人，2023）的潜在输出进行了t-SNE（Van der Maaten & Hinton，2008）可视化。可视化显示，我们的混合内部模型对环境具有更可分离的编码，这意味着我们的方法携带了更精确的环境信息，从而具有更强的能力来识别机器人处于哪种地形上。

以上是该项目在论文中对方法的描述
这是这个项目通过对比学习优化将一条轨迹编码为隐变量。我想借鉴这个方法，我也想将历史观测编码为隐变量，但不同的是，我希望最大化相同速度桶中的轨迹的潜在特征的相似性，并使不同桶中的轨迹的潜在特征原理，并期望这样可以隐式建模出不同速度下的步态信息。
我希望，你调研先进的对比学习方案给出编码器的设计和训练范式方案

1.我目前给的cmd，已经是离散的了，所以分桶本身是自然的，但是我在实际实验时发现：训练前期，你给出速度，但是实际机器人行走速度远不及cmd；就算中后期，实际的速度距离cmd也有很大差距。如果以指令而不是该轨迹实际对应的速度为标签。会不会产生问题？

2.为我介绍一下Causal Transformer，以及如何将其用于编码

3.对 (v_x, v_y, ω_z) 各自分桶后做笛卡尔乘积。

4.我有疑问，你说的“保留"同 trajectory 相邻帧"作为额外正样本”，单帧和多历史帧堆叠的不同形状的输入都是经过相同编码器编码么？him是怎么操作的？此外，我认为相同正样本应该具有相似的语义，这样扩充不好，或许可以考虑加一个学习目标，同时最大化历史观测与下一观测之间潜在特征的相似性并最大化相同速度桶中的轨迹的潜在特征的相似性？

Representation loss（你之前已经设计的，照旧）： $$\mathcal{L}{\text{proto}} = -\log \frac{\exp(z{\text{mode},t}^\top e_{b_t}/\tau)}{\sum_{b'} \exp(z_{\text{mode},t}^\top e_{b'}/\tau)}$$

内在奖励（关键的新东西）： $$\boxed{,r^{\text{int}}t = \underbrace{\log \frac{\exp(z{\text{mode},t}^\top e_{b_t}/\tau)}{\sum_{b'} \exp(z_{\text{mode},t}^\top e_{b'}/\tau)}}{=-\mathcal{L}{\text{proto}}\text{ 的样本值}} + \log K,}$$

加 
log
⁡
K
 是为了把内在 reward 居中到 0 附近（均匀分配下 reward=0）。注意这个 reward 是 detach 掉 encoder 梯度 的——它纯粹用作标量 reward 喂给 PPO；而 encoder 仍然由 
L
proto
 通过普通监督学习更新。

SMERL 门控： 


$$\mathrm{gate}t = \sigma!\left(\kappa \cdot (\bar r^{\text{track}}{[t-T:t]} - \beta)\right)$$

用 sigmoid 而非 hard 阈值，避免 reward 阶跃。
β
 取你跟踪 reward 在"勉强能跟上"水平的值；
α
 从小到大 schedule（例如 0 → 0.1）。

 $$
 v_{cmd} = 0, r = 1 - \frac{|v|}{b} 
 $$

 $$
 v_{cmd} > 0, r = 1 - \frac{|v - v_{cmd}|}{v_{cmd}}
 $$

 $$
 其中，b满足，对v>0，1 - \frac{|v|}{b}  \leq \exp(-\frac{v^2}{\sigma})
 $$

 在基于 PPO 的四足/双足机器人速度追踪任务中，标准做法是用高斯核作为追踪奖励：

记$e=|v-v_{cmd}|$
 $$
 r=exp(-\frac{e^2}{\sigma^2})
 $$
 实验发现（和猜想）
通过控制变量的 5 路对照实验（V21f2/g/h/i/j），发现：
1. 高斯核等价于"恒定斜率线性 + 饱和边界"：在误差∣e∣<σ 附近，

Derivation: solve for b such that exp(-x^2/std^2) >= 1 - x/b for all x>=0.
Tangency condition (2v+1)*exp(-v) = 1 with v = (x/std)^2 gives v* ~= 1.25643,
so b = std * exp(v*) / (2*sqrt(v*)) ~= 1.5670 * std. 
2. 相对线性核 r=1−∣e∣/max(∣vcmd∣,b) 在低速指令区间（∣vcmd∣<0.7,优于高斯核——猜测原因是分母更小，等效梯度更陡。
3. 高斯核在高速线性追踪优于相对线性——因相对线性分母=∣vcmd∣ 导致梯度被稀释
4. 猜测核形状改进的真实机制是消除梯度死区而非核函数的几何形状。

核心猜想
追踪奖励的有效信息只有两条：(1) 梯度幅度（斜率）；(2) 误差大时是否保持非零梯度防止早期终止。
即存在某个状态/指令依赖的最优斜率函数 $f(v,e)$
使追踪奖励形式为：
$$
r = leaky(1-f(v,e)*e)
$$
除了桶级参数化（离线 grid/Bayes 搜索）：将 (mode × speed_bucket) 的 $\sim$9 个斜率参数视为超参，短训枚举以外，是否存在将f建模为mlp，通过在线元强化学习的方案更新的思路？给出推导