下面给出一套我认为更严格、也更适合你当前问题的 **Validated Bilevel Meta-PPO Reward Learning** 方案。

核心变化是：

[
\text{不再优化 }
\mathbb E[A_{\text{true}} r_\phi]
]

而是优化：

[
\text{用 } r_\phi \text{ 做一次 PPO 更新之后，新 policy 在 future rollout 上的 true tracking error}
]

也就是：

[
\phi
\rightarrow
r_\phi
\rightarrow
\theta'
\rightarrow
\tau_{\text{val}}\sim\pi_{\theta'}
\rightarrow
E_{\text{true}}(\tau_{\text{val}})
]

---

# 1. 推荐的 learnable reward 形式

我不建议继续让 MLP 直接学 Gaussian 的 (\sigma(c))，因为 (\sigma) 本质上是在学 tolerance，容易出现 V21m 那种 reward broadening。

更推荐让 MLP 只学 **不同 command 区间的 tracking 惩罚强度 multiplier**，但不改变“误差越大越差”的基本语义。

设：

[
c_v = \text{linear velocity command}
]

[
c_\omega = \text{yaw angular velocity command}
]

[
e_v = |v_{xy}-c_v|
]

[
e_\omega = |\omega_z-c_\omega|
]

定义固定的 true-error-like normalized error：

[
\bar e_v
========

\frac{e_v}{v_{\max}}
+
\beta_v
\frac{e_v}{\max(|c_v|, b_v)}
]

[
\bar e_\omega
=============

\frac{e_\omega}{\omega_{\max}}
+
\beta_\omega
\frac{e_\omega}{\max(|c_\omega|, b_\omega)}
]

例如：

[
v_{\max}=1.5
]

[
b_v=0.5
]

[
\beta_v=1
]

角速度按你的 command 范围设置，例如若：

[
|c_\omega|\le 0.8
]

可以取：

[
\omega_{\max}=0.8,\quad b_\omega=0.3\sim0.4
]

然后让 reward MLP 输出 bounded multiplier：

[
m_{v,\phi}(c)
=============

\exp
\left(
\delta_v
\tanh h_{v,\phi}(c)
\right)
]

[
m_{\omega,\phi}(c)
==================

\exp
\left(
\delta_\omega
\tanh h_{\omega,\phi}(c)
\right)
]

其中 (h_\phi) 是 MLP。

这样：

[
m_{v,\phi}(c)\in [e^{-\delta_v},e^{\delta_v}]
]

如果：

[
\delta_v=0.5
]

则：

[
m_{v,\phi}\in[0.61,1.65]
]

这能让 reward learner 调整不同 command 区间的斜率，但不能无限放宽 tracking 标准。

定义 learned tracking penalty：

[
d_\phi(c,e)
===========

m_{v,\phi}(c)\bar e_v
+
m_{\omega,\phi}(c)\bar e_\omega
]

最终 learned reward：

[
\boxed{
r_\phi(c,e)
===========

\ell_\kappa
\left(
1-d_\phi(c,e)
\right)
}
]

其中 leaky 函数：

[
\ell_\kappa(x)
==============

\begin{cases}
x, & x\ge 0\
\kappa x, & x<0
\end{cases}
]

建议：

[
\kappa=0.05\sim0.2
]

这样大误差区仍然有非零信号，不会像 clipped reward 那样死掉。

---

# 2. 这个 reward 的语义约束

由于 (m_{v,\phi}(c)>0)，且 (\bar e_v) 对 (e_v) 线性递增，所以：

[
\frac{\partial r_\phi}{\partial e_v}
====================================

-\ell_\kappa'(1-d_\phi)
m_{v,\phi}(c)
\left(
\frac{1}{v_{\max}}
+
\frac{\beta_v}{\max(|c_v|,b_v)}
\right)
\le 0
]

同理：

[
\frac{\partial r_\phi}{\partial e_\omega}\le 0
]

所以这个形式天然保证：

[
\boxed{
e \uparrow \Rightarrow r_\phi \downarrow
}
]

而且它不会像 learnable (\sigma) 那样通过：

[
\sigma\to\infty
]

把所有大误差都说成“还不错”。

它只允许 reward learner 调整：

[
\text{哪个 command 区间更应该被 PPO 重视}
]

而不是改变：

[
\text{什么叫 tracking 好}
]

---

# 3. 固定 true objective

外层真实目标必须固定，不能依赖 (\phi)。

推荐：

[
R_{\text{true}}
===============

*

\left(
\bar e_v
+
\bar e_\omega
\right)
-------

## \lambda_{\text{fall}}\mathbf 1_{\text{fall}}

## \lambda_{\text{ori}}E_{\text{ori}}

## \lambda_{\text{energy}}E_{\text{energy}}

\lambda_{\text{smooth}}E_{\text{smooth}}
]

如果当前主要想研究 tracking kernel，外层可以先简化为：

[
\boxed{
R_{\text{true}}
===============

*

\left(
\bar e_v
+
\bar e_\omega
\right)
-------

\lambda_{\text{fall}}\mathbf 1_{\text{fall}}
}
]

注意：fall penalty 最好使用 fixed-horizon penalty。
不要让 episode 提前结束后因为没有后续 tracking error 而“少扣分”。

例如 episode 在 (t) 终止，可以加：

[
R_{\text{fall-tail}}
====================

-\lambda_{\text{fall-tail}}(H-t)
]

或者在 true return 里对剩余 horizon 补固定负奖励。

---

# 4. 严格 bilevel 目标

当前 policy 参数为：

[
\theta
]

reward 参数为：

[
\phi
]

内层使用 learned reward 做 PPO 更新：

[
\theta'
=======

U_{\text{PPO}}(\theta,\phi)
]

外层在新 policy (\pi_{\theta'}) 上评价 true objective：

[
J_{\text{true}}(\theta')
========================

\mathbb E_{\tau_{\text{val}}\sim\pi_{\theta'}}
\left[
\sum_t \gamma^t R_{\text{true}}(s_t,a_t)
\right]
]

最终目标：

[
\boxed{
\max_\phi
J_{\text{true}}
\left(
U_{\text{PPO}}(\theta,\phi)
\right)
-------

\Omega(\phi)
}
]

或者写成最小化 loss：

[
\boxed{
\min_\phi
\mathcal L_{\text{meta}}(\phi)
==============================

*

J_{\text{true}}
\left(
U_{\text{PPO}}(\theta,\phi)
\right)
+
\Omega(\phi)
}
]

这才对应你真正想要的因果链：

[
\phi
\rightarrow
r_\phi
\rightarrow
\text{PPO update}
\rightarrow
\theta'
\rightarrow
\text{future true tracking}
]

---

# 5. 内层 PPO update

采样训练 rollout：

[
\tau_{\text{train}}\sim\pi_\theta
]

在这个 rollout 上计算 learned reward：

[
R_\phi(s_t,a_t)
===============

w_{\text{track}}r_\phi(c_t,e_t)
+
R_{\text{other}}(s_t,a_t)
]

用 (R_\phi) 计算 GAE：

[
A_t^\phi
========

\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^\phi
]

[
\delta_t^\phi
=============

R_\phi(s_t,a_t)
+
\gamma V_\psi(s_{t+1})
----------------------

V_\psi(s_t)
]

这里为了 meta-gradient 稳定，建议：

[
V_\psi(s_t)
]

对 (\phi) 和 (\theta) 都 detach。
也就是说，meta path 只经过：

[
\phi
\rightarrow
r_\phi
\rightarrow
A^\phi
\rightarrow
\theta'
]

不要让 critic 参与二阶图。

PPO actor loss：

[
\mathcal L_{\text{PPO}}^{\text{in}}(\theta_i;\phi)
==================================================

*

\mathbb E_t
\left[
\min
\left(
\rho_t(\theta_i)A_t^\phi,
\mathrm{clip}(\rho_t(\theta_i),1-\epsilon,1+\epsilon)A_t^\phi
\right)
\right]
]

其中：

[
\rho_t(\theta_i)
================

\frac{
\pi_{\theta_i}(a_t|o_t)
}{
\pi_{\theta}(a_t|o_t)
}
]

注意分母是 rollout 时的 old policy logprob。

做 (K) 步 differentiable inner update：

[
\theta_{0}=\theta
]

[
\theta_{i+1}
============

## \theta_i

\alpha_{\text{in}}
\nabla_{\theta_i}
\mathcal L_{\text{PPO}}^{\text{in}}(\theta_i;\phi)
]

最终：

[
\boxed{
\theta'
=======

\theta_K
}
]

实践建议：

[
K=1\sim3
]

先从：

[
K=1
]

开始。

---

# 6. 外层 validation rollout

用 inner update 后的 policy：

[
\pi_{\theta'}
]

重新采样 validation rollout：

[
\tau_{\text{val}}\sim\pi_{\theta'}
]

在 validation rollout 上计算固定 true reward：

[
R_{\text{true}}
]

然后计算 true advantage：

[
A_t^{\text{true}}
=================

\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^{\text{true}}
]

[
\delta_t^{\text{true}}
======================

R_{\text{true}}(s_t,a_t)
+
\gamma V_\nu^{\text{true}}(s_{t+1})
-----------------------------------

V_\nu^{\text{true}}(s_t)
]

这里 (V_\nu^{\text{true}}) 是专门估计 true return 的 critic。
它只作为 baseline，用于降低方差。对 meta-gradient detach。

外层 policy-gradient surrogate：

[
\mathcal L_{\text{outer}}(\theta')
==================================

*

\mathbb E_{\tau_{\text{val}}}
\left[
\log \pi_{\theta'}(a_t|o_t)
\hat A_t^{\text{true}}
\right]
]

其中：

[
\hat A_t^{\text{true}}
]

是 detached 的 true advantage。

然后 meta loss：

[
\boxed{
\mathcal L_{\text{meta}}(\phi)
==============================

\mathcal L_{\text{outer}}(\theta'(\phi))
+
\Omega(\phi)
}
]

对 (\phi) 反向传播：

[
\nabla_\phi
\mathcal L_{\text{meta}}
========================

\nabla_{\theta'}\mathcal L_{\text{outer}}
\frac{\partial \theta'}{\partial \phi}
+
\nabla_\phi\Omega(\phi)
]

其中：

[
\frac{\partial \theta'}{\partial \phi}
]

来自 differentiable PPO update。

这就是完整的 bilevel meta-gradient。

---

# 7. Meta-gradient 展开

内层一步更新时：

[
\theta'
=======

## \theta

\alpha_{\text{in}}
\nabla_\theta
\mathcal L_{\text{PPO}}^{\text{in}}(\theta;\phi)
]

所以：

[
\frac{\partial \theta'}{\partial \phi}
======================================

*

\alpha_{\text{in}}
\nabla_{\phi\theta}^{2}
\mathcal L_{\text{PPO}}^{\text{in}}(\theta;\phi)
]

因此：

[
\boxed{
\nabla_\phi
\mathcal L_{\text{meta}}
========================

*

\alpha_{\text{in}}
\nabla_{\theta'}\mathcal L_{\text{outer}}^\top
\nabla_{\phi\theta}^{2}
\mathcal L_{\text{PPO}}^{\text{in}}
+
\nabla_\phi\Omega(\phi)
}
]

这和当前的：

[
-\mathbb E[A_{\text{true}}r_\phi]
]

完全不同。

当前目标只让 reward 和当前 rollout 上的 true advantage 相关；新目标要求：

[
r_\phi
]

诱导出的 PPO 更新方向真的能降低 future rollout 上的 true error。

---

# 8. Reward regularization

为了防止 reward semantic drift，建议：

[
\Omega(\phi)
============

\lambda_{\text{prior}}\Omega_{\text{prior}}
+
\lambda_{\text{smooth}}\Omega_{\text{smooth}}
+
\lambda_{\text{contrast}}\Omega_{\text{contrast}}
+
\lambda_{\text{step}}\Omega_{\text{step}}
]

## 8.1 Prior regularization

因为：

[
m_\phi(c)
=========

\exp(\delta\tanh h_\phi(c))
]

默认 (m=1) 表示手工 baseline。

正则：

[
\Omega_{\text{prior}}
=====================

\mathbb E_c
\left[
(\log m_{v,\phi}(c))^2
+
(\log m_{\omega,\phi}(c))^2
\right]
]

这会防止 MLP 远离 prior。

---

## 8.2 Smoothness regularization

防止相邻 command 区间 reward 剧烈跳变：

[
\Omega_{\text{smooth}}
======================

\mathbb E_c
\left[
\left|
\frac{\partial \log m_{v,\phi}(c)}
{\partial c}
\right|^2
+
\left|
\frac{\partial \log m_{\omega,\phi}(c)}
{\partial c}
\right|^2
\right]
]

---

## 8.3 Contrast regularization

这是针对 V21m 那种 reward flattening / broadening 的关键约束。

定义一个小误差和大误差：

[
e_{\text{good}} < e_{\text{bad}}
]

例如线速度：

[
e_{\text{good},v}=0.05,\quad e_{\text{bad},v}=0.4
]

角速度：

[
e_{\text{good},\omega}=0.05,\quad e_{\text{bad},\omega}=0.5
]

要求 reward contrast 不要太小：

[
C_\phi(c)
=========

## r_\phi(c,e_{\text{good}})

r_\phi(c,e_{\text{bad}})
]

惩罚：

[
\Omega_{\text{contrast}}
========================

\mathbb E_c
\left[
\mathrm{ReLU}
(
C_{\min}-C_\phi(c)
)^2
\right]
]

这个项的作用是：

[
\text{大误差不能也拿到很高 reward}
]

它直接防止 learned reward 变平。

---

## 8.4 Step regularization

防止 reward 每次 meta update 变化太大。

保存上一次 reward network：

[
\phi_{\text{old}}
]

惩罚：

[
\Omega_{\text{step}}
====================

\mathbb E_c
\left[
(\log m_{\phi}(c)-\log m_{\phi_{\text{old}}}(c))^2
\right]
]

---

# 9. 完整算法

## Algorithm: Validated Bilevel Meta-PPO

每一轮 iteration：

### Step 1：收集 train rollout

[
\tau_{\text{train}}\sim\pi_\theta
]

必须 command-balanced。
不要让某些 command bucket 占主导。

---

### Step 2：计算 learned reward

对 train rollout：

[
e_v=|v_{xy}-c_v|
]

[
e_\omega=|\omega_z-c_\omega|
]

[
r_\phi
======

\ell_\kappa
\left(
1-
m_{v,\phi}(c)\bar e_v
---------------------

m_{\omega,\phi}(c)\bar e_\omega
\right)
]

总 reward：

[
R_\phi
======

w_{\text{track}}r_\phi
+
R_{\text{other}}
]

---

### Step 3：计算 inner advantage

[
A^\phi
======

\mathrm{GAE}(R_\phi,V_\psi)
]

这里：

[
V_\psi
]

detach。

建议对 (A^\phi) 做 stop-stat normalization：

[
\hat A^\phi
===========

\frac{
A^\phi-\mathrm{stopgrad}(\mu_A)
}{
\mathrm{stopgrad}(\sigma_A)+\epsilon
}
]

不要直接：

[
A^\phi=\mathrm{detach}(A^\phi)
]

否则：

[
\phi
\rightarrow A^\phi
\rightarrow \theta'
]

这条 meta-gradient 会断掉。

---

### Step 4：做 differentiable PPO inner update

[
\theta'
=======

U_{\text{PPO}}(\theta,\phi;\tau_{\text{train}})
]

即：

[
\theta_{i+1}
============

## \theta_i

\alpha_{\text{in}}
\nabla_{\theta_i}
\mathcal L_{\text{PPO}}^{\text{in}}(\theta_i;\phi)
]

实现时必须使用 functional parameters。
不要用普通 optimizer.step() 做 shadow update，因为它不可微。

---

### Step 5：用 (\theta') 收集 validation rollout

[
\tau_{\text{val}}\sim\pi_{\theta'}
]

这是最关键的一步。
不能复用当前 (\pi_\theta) 的 rollout 代替。

否则又会退回 current-rollout correlation objective。

---

### Step 6：计算 true reward 和 true advantage

[
R_{\text{true}}
===============

*

## (\bar e_v+\bar e_\omega)

\lambda_{\text{fall}}\mathbf 1_{\text{fall}}
+\text{fixed stability terms}
]

[
A^{\text{true}}
===============

\mathrm{GAE}(R_{\text{true}},V_\nu^{\text{true}})
]

这里：

[
A^{\text{true}}
]

detach。

---

### Step 7：计算 outer loss

[
\mathcal L_{\text{outer}}
=========================

*

\mathbb E_{\tau_{\text{val}}}
\left[
\log\pi_{\theta'}(a_t|o_t)
\hat A_t^{\text{true}}
\right]
]

为了避免 command imbalance，建议按 command bucket 平均：

[
\mathcal L_{\text{outer}}
=========================

-\frac{1}{B}
\sum_{b=1}^{B}
\mathbb E_{t\in b}
\left[
\log\pi_{\theta'}(a_t|o_t)
\hat A_t^{\text{true}}
\right]
]

---

### Step 8：更新 reward network

[
\mathcal L_{\text{meta}}
========================

\mathcal L_{\text{outer}}
+
\Omega(\phi)
]

[
\phi
\leftarrow
\phi
----

\eta_{\text{meta}}
\nabla_\phi
\mathcal L_{\text{meta}}
]

---

### Step 9：更新实际 policy

这里有两个版本。

## 版本 A：严格版本

直接接受 shadow update：

[
\theta \leftarrow \mathrm{stopgrad}(\theta')
]

这个版本最贴近理论目标，因为外层验证的 policy 就是实际接收的 policy。

缺点是：(\theta') 是用 meta update 前的 (\phi) 产生的。

---

## 版本 B：工程推荐版本

把前面的 (\theta') 只当作 shadow update，用来训练 (\phi)。

更新完 (\phi) 后，用新的 (\phi) 在 train rollout 上重新计算 reward 和 advantage，然后做普通 PPO 更新：

[
\theta
\leftarrow
\mathrm{PPOUpdate}
(
\theta,
r_{\phi_{\text{new}}}
)
]

这个版本不完全等同于严格 bilevel，但更稳定。
我建议先实现版本 A 做因果验证，再切到版本 B 做大规模训练。

---

# 10. PyTorch 伪代码

下面是接近实现的结构。

```python
# actor: pi_theta
# critic_inner: V_psi, value for learned reward
# critic_true: V_nu, value for true reward
# reward_net: m_phi(c)
# actor_params: functional parameter dict for actor

for it in range(num_iters):

    # --------------------------------------------------
    # 1. collect train rollout under current theta
    # --------------------------------------------------
    train = collect_rollout(
        actor=actor,
        params=theta,
        envs=train_envs,
        command_sampler="balanced",
        no_grad=True,
    )

    # train contains:
    # obs, actions, old_logp, dones,
    # v_xy, omega_z, cmd_v, cmd_omega,
    # fixed_other_rewards

    # --------------------------------------------------
    # 2. compute learnable reward on train rollout
    # --------------------------------------------------
    e_v = norm(train.v_xy - train.cmd_v, dim=-1)
    e_w = abs(train.omega_z - train.cmd_omega)

    ebar_v = e_v / v_max + beta_v * e_v / clamp(abs_norm(train.cmd_v), min=b_v)
    ebar_w = e_w / w_max + beta_w * e_w / clamp(abs(train.cmd_omega), min=b_w)

    m_v, m_w = reward_net(train.cmd_v, train.cmd_omega, train.mode)
    # m_v = exp(delta_v * tanh(h_v))
    # m_w = exp(delta_w * tanh(h_w))

    d_phi = m_v * ebar_v + m_w * ebar_w

    r_track_phi = leaky(1.0 - d_phi, negative_slope=kappa)

    R_phi = w_track * r_track_phi + train.fixed_other_rewards

    # --------------------------------------------------
    # 3. compute A_phi
    # --------------------------------------------------
    with torch.no_grad():
        V_inner = critic_inner(train.obs)

    A_phi, returns_phi = gae(
        rewards=R_phi,
        values=V_inner.detach(),
        dones=train.dones,
        gamma=gamma,
        lam=gae_lambda,
    )

    # important: do NOT detach A_phi
    # only detach normalization statistics
    A_phi = stop_stat_normalize(A_phi, bucket=train.command_bucket)

    # --------------------------------------------------
    # 4. differentiable inner PPO update
    # --------------------------------------------------
    theta_shadow = theta

    for k in range(K_inner):

        dist = actor.functional_dist(train.obs, theta_shadow)
        logp = dist.log_prob(train.actions).sum(-1)

        ratio = torch.exp(logp - train.old_logp.detach())

        unclipped = ratio * A_phi
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A_phi

        inner_actor_loss = -torch.min(unclipped, clipped).mean()

        # optional entropy
        entropy = dist.entropy().sum(-1).mean()
        inner_loss = inner_actor_loss - ent_coef * entropy

        grads = torch.autograd.grad(
            inner_loss,
            theta_shadow.values(),
            create_graph=True,
            retain_graph=True,
        )

        theta_shadow = {
            name: p - inner_lr * g
            for (name, p), g in zip(theta_shadow.items(), grads)
        }

    theta_prime = theta_shadow

    # --------------------------------------------------
    # 5. collect validation rollout under theta_prime
    # --------------------------------------------------
    val = collect_rollout(
        actor=actor,
        params=theta_prime,
        envs=val_envs,
        command_sampler="balanced",
        no_grad=True,
    )

    # --------------------------------------------------
    # 6. compute true reward on validation rollout
    # --------------------------------------------------
    e_v_val = norm(val.v_xy - val.cmd_v, dim=-1)
    e_w_val = abs(val.omega_z - val.cmd_omega)

    ebar_v_val = (
        e_v_val / v_max
        + beta_v * e_v_val / clamp(abs_norm(val.cmd_v), min=b_v)
    )

    ebar_w_val = (
        e_w_val / w_max
        + beta_w * e_w_val / clamp(abs(val.cmd_omega), min=b_w)
    )

    R_true = -(ebar_v_val + ebar_w_val)

    R_true = R_true - fall_penalty(val)
    R_true = R_true - fixed_stability_penalties(val)

    with torch.no_grad():
        V_true = critic_true(val.obs)

    A_true, returns_true = gae(
        rewards=R_true.detach(),
        values=V_true.detach(),
        dones=val.dones,
        gamma=gamma,
        lam=gae_lambda,
    )

    A_true = bucket_normalize_detached(A_true, bucket=val.command_bucket)

    # --------------------------------------------------
    # 7. outer true policy-gradient loss
    # --------------------------------------------------
    dist_val = actor.functional_dist(val.obs, theta_prime)
    logp_val = dist_val.log_prob(val.actions).sum(-1)

    # bucket-balanced outer loss
    outer_loss = bucket_mean(
        -logp_val * A_true.detach(),
        bucket=val.command_bucket,
    )

    # --------------------------------------------------
    # 8. reward regularization
    # --------------------------------------------------
    prior_loss = ((torch.log(m_v) ** 2).mean() + (torch.log(m_w) ** 2).mean())

    smooth_loss = reward_net.smoothness_loss(command_grid)

    contrast_loss = reward_contrast_loss(
        reward_net=reward_net,
        command_grid=command_grid,
        e_good_v=0.05,
        e_bad_v=0.40,
        e_good_w=0.05,
        e_bad_w=0.50,
        c_min=contrast_min,
    )

    step_loss = reward_step_loss(
        reward_net=reward_net,
        reward_net_old=reward_net_old,
        command_grid=command_grid,
    )

    meta_loss = (
        outer_loss
        + lambda_prior * prior_loss
        + lambda_smooth * smooth_loss
        + lambda_contrast * contrast_loss
        + lambda_step * step_loss
    )

    reward_optimizer.zero_grad()
    meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_net.parameters(), meta_grad_clip)
    reward_optimizer.step()

    update_reward_net_old(reward_net_old, reward_net)

    # --------------------------------------------------
    # 9A. strict version: accept theta_prime
    # --------------------------------------------------
    theta = detach_param_dict(theta_prime)

    # --------------------------------------------------
    # 9B. alternatively: production PPO update with new phi
    # --------------------------------------------------
    # recompute R_phi_new and A_phi_new with updated reward_net
    # run ordinary PPO update on actor using detached A_phi_new

    # --------------------------------------------------
    # 10. critic updates
    # --------------------------------------------------
    update_critic_inner(critic_inner, train.obs, returns_phi.detach())
    update_critic_true(critic_true, val.obs, returns_true.detach())
```

---

# 11. 关键 detach / create_graph 规则

这部分非常重要。

## 11.1 train rollout 数据全部 detach

环境状态、动作、old logprob 都是采样数据：

[
(o_t,a_t,\log\pi_\theta(a_t|o_t))
]

它们不需要梯度。

---

## 11.2 (A^\phi) 不能 detach

内层 PPO loss 里：

[
A^\phi
]

必须保留对 (\phi) 的梯度。

否则：

[
\phi
\rightarrow r_\phi
\rightarrow A^\phi
\rightarrow \theta'
]

断掉。

错误写法：

```python
A_phi = A_phi.detach()
```

正确写法：

```python
A_phi = stop_stat_normalize(A_phi)
```

其中 normalization 的 mean/std 可以 detach，但 (A_\phi) 本身不能 detach。

---

## 11.3 critic value detach

为了避免 meta-gradient 穿过 critic，建议：

```python
V_inner = critic_inner(obs).detach()
A_phi = gae(R_phi, V_inner)
```

critic 单独用普通 supervised loss 更新。

---

## 11.4 inner update 必须 create_graph=True

```python
grads = torch.autograd.grad(
    inner_loss,
    theta_shadow.values(),
    create_graph=True,
)
```

否则：

[
\theta'
]

不会保留对 (\phi) 的依赖。

---

## 11.5 outer rollout 不需要对环境求导

validation rollout 是：

[
\tau_{\text{val}}\sim\pi_{\theta'}
]

环境不可微没关系。

外层梯度通过 policy-gradient surrogate：

[
-\log\pi_{\theta'}(a_t|o_t)A_t^{\text{true}}
]

估计：

[
\nabla_{\theta'}J_{\text{true}}
]

然后再通过：

[
\theta'(\phi)
]

传回 (\phi)。

---

# 12. 为什么这个算法能避免 V21m 的问题

V21m 的坏方向是：

[
\sigma(c)\uparrow
\Rightarrow
r_\sigma(e)\uparrow \quad \forall e>0
]

也就是 reward learner 可以通过“放宽标准”提高 learned reward。

新算法有三层防护。

第一，reward 参数化不学 tolerance，而是学 bounded slope multiplier：

[
m_\phi(c)\in[e^{-\delta},e^\delta]
]

第二，外层不是 learned reward，而是固定 true error：

[
R_{\text{true}}=-(\bar e_v+\bar e_\omega)+\cdots
]

第三，validation rollout 来自更新后的 policy：

[
\tau_{\text{val}}\sim\pi_{\theta'}
]

所以只有当：

[
r_\phi
]

诱导出的 policy update 真的降低 future true error 时，(\phi) 才会被强化。

---

# 13. 便宜近似：gradient alignment 版本

完整 bilevel 需要额外 validation rollout，成本较高。可以做一个中间版本作为 ablation。

定义 true gradient：

[
g_{\text{true}}
===============

\nabla_\theta
\mathcal L_{\text{PG}}(\theta;A^{\text{true}})
]

learned reward gradient：

[
g_\phi
======

\nabla_\theta
\mathcal L_{\text{PPO}}(\theta;A^\phi)
]

然后优化：

[
\boxed{
\mathcal L_{\text{align}}(\phi)
===============================

*

\frac{
g_\phi^\top g_{\text{true}}
}{
|g_\phi||g_{\text{true}}|+\epsilon
}
+
\Omega(\phi)
}
]

这个版本不需要 (\theta') rollout，但比：

[
-\mathbb E[A_{\text{true}}r_\phi]
]

强很多，因为它至少直接对齐了 policy update direction。

不过它仍然不是完整 future-rollout bilevel。
我建议把它作为 debug 版本，不作为最终版本。

---

# 14. 推荐超参数初值

可以从下面开始：

[
K_{\text{inner}}=1
]

[
\alpha_{\text{in}}=\text{当前 PPO actor lr}
]

[
\eta_{\text{meta}}=1e{-4}\ \text{或}\ 3e{-5}
]

[
\delta_v=\delta_\omega=0.5
]

[
\kappa=0.1
]

[
b_v=0.5
]

[
b_\omega=0.3\sim0.4
]

[
\lambda_{\text{prior}}=1e{-2}
]

[
\lambda_{\text{smooth}}=1e{-3}
]

[
\lambda_{\text{contrast}}=1e{-2}
]

[
\lambda_{\text{step}}=1e{-2}
]

meta update 频率可以低一点：

[
\text{每 } 2\sim5 \text{ 个 PPO iteration 更新一次 } \phi
]

训练前期可以 warmup：

[
\phi \text{ 前 } N_{\text{warmup}} \text{ iterations 不更新}
]

例如先让 policy 有基本站立/行走能力，再启动 meta reward learning。

---

# 15. 必须记录的诊断指标

为了确认算法真的解决了 objective mismatch，需要每轮记录：

## 15.1 Post-update true improvement

[
\Delta E_{\text{true}}
======================

## E_{\text{true}}(\pi_{\theta'})

E_{\text{true}}(\pi_\theta)
]

这是最核心指标。

---

## 15.2 Learned reward vs fixed diagnostic reward

记录：

[
r_\phi
]

以及固定 sharp tracker：

[
r_{\text{sharp}}
================

\exp(-e^2/\sigma_0^2)
]

如果：

[
r_\phi \uparrow
]

但：

[
r_{\text{sharp}}\downarrow,\quad E_{\text{true}}\uparrow
]

说明 reward 又在漂移。

---

## 15.3 Reward contrast

对 command grid 记录：

[
C_\phi(c)
=========

## r_\phi(c,e_{\text{good}})

r_\phi(c,e_{\text{bad}})
]

如果 (C_\phi) 下降到很小，说明 reward flattening。

---

## 15.4 Multiplier range

记录：

[
\mathrm{mean}(m_v),\quad \max(m_v),\quad \min(m_v)
]

[
\mathrm{mean}(m_\omega),\quad \max(m_\omega),\quad \min(m_\omega)
]

如果长期贴边，说明 meta lr 或 (\delta) 太大，或者 prior 太弱。

---

## 15.5 Gradient cosine

记录：

[
\cos(g_\phi,g_{\text{true}})
============================

\frac{
g_\phi^\top g_{\text{true}}
}{
|g_\phi||g_{\text{true}}|
}
]

完整 bilevel 不一定每次都需要它，但它是很好的 debug 信号。

---

# 16. 最小实现版本

先实现下面这个最小版本即可：

[
r_\phi
======

\ell_\kappa
\left(
1-
m_{v,\phi}(c)\bar e_v
---------------------

m_{\omega,\phi}(c)\bar e_\omega
\right)
]

[
m_\phi(c)
=========

\exp(\delta\tanh h_\phi(c))
]

每轮：

[
\tau_{\text{train}}\sim\pi_\theta
]

[
\theta'
=======

## \theta

\alpha
\nabla_\theta
\mathcal L_{\text{PPO}}(\theta;r_\phi)
]

[
\tau_{\text{val}}\sim\pi_{\theta'}
]

[
\mathcal L_{\text{meta}}
========================

*

\mathbb E_{\tau_{\text{val}}}
[
\log\pi_{\theta'}(a|o)A_{\text{true}}
]
+
\Omega(\phi)
]

[
\phi
\leftarrow
\phi-\eta\nabla_\phi\mathcal L_{\text{meta}}
]

这就是最小严格版本。

---

# 17. 最终算法的核心性质

这套算法满足：

[
\boxed{
\text{reward learner 不再被当前 rollout 上的 reward correlation 训练}
}
]

而是被：

[
\boxed{
\text{post-PPO-update future true tracking error}
}
]

训练。

同时 reward family 保证：

[
\boxed{
r_\phi(0)=1,\quad
\frac{\partial r_\phi}{\partial e}\le0,\quad
\text{大误差保持惩罚对比度}
}
]

所以它针对你当前观察到的 V21m 问题是直接修正：

[
\text{从 “让当前错误看起来不那么错”}
]

变成：

[
\text{让 policy 用这个 reward 更新后真的跟踪更准}
]
