import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim

class HIMEstimatorSupCon(nn.Module):
    def __init__(self, temporal_steps, num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 num_buckets=16, latent_dim=16,
                 temperature=0.1, lambda_proto=1.0, lambda_supcon=0.5,
                 lambda_div=0.01, learning_rate=1e-3, max_grad_norm=10.0,
                 use_momentum=True, momentum=0.99, queue_size=4096, **kwargs):
        super().__init__()
        self.H = temporal_steps
        self.D = num_one_step_obs
        self.K = num_buckets
        self.tau = temperature
        self.l1, self.l2, self.l3 = lambda_proto, lambda_supcon, lambda_div
        self.use_momentum = use_momentum
        self.m = momentum
        self.max_grad_norm = max_grad_norm

        # ---- encoder (TCN 推荐；这里用 MLP 演示，与原版接口一致) ----
        in_dim = self.H * self.D
        layers, d = [], in_dim
        for h in enc_hidden_dims[:-1]:
            layers += [nn.Linear(d, h), nn.ELU()]; d = h
        self.backbone = nn.Sequential(*layers)
        self.vel_head = nn.Linear(d, 3)                       # 显式速度
        self.proj_head = nn.Sequential(                       # 对比 head
            nn.Linear(d, d), nn.ELU(), nn.Linear(d, latent_dim))

        # 桶原型（替代 HIM 原 self.proto）
        self.bucket_proto = nn.Embedding(self.K, latent_dim)
        nn.init.normal_(self.bucket_proto.weight, std=0.02)

        # momentum encoder + queue (MoCo style)
        if use_momentum:
            self.backbone_m = nn.Sequential(*[nn.Linear(in_dim, enc_hidden_dims[0]), nn.ELU()] +
                [layer for h_prev, h in zip(enc_hidden_dims[:-2], enc_hidden_dims[1:-1])
                       for layer in (nn.Linear(h_prev, h), nn.ELU())])
            self.proj_head_m = nn.Sequential(
                nn.Linear(d, d), nn.ELU(), nn.Linear(d, latent_dim))
            for p in list(self.backbone_m.parameters()) + list(self.proj_head_m.parameters()):
                p.requires_grad = False
            self._copy_to_momentum()
            self.register_buffer("queue_z", F.normalize(torch.randn(queue_size, latent_dim), dim=-1))
            self.register_buffer("queue_y", -torch.ones(queue_size, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue_size = queue_size

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # -------- API 与原 HIMEstimator 保持一致 --------
    def encode(self, obs_history):
        f = self.backbone(obs_history.detach())
        vel = self.vel_head(f)
        z = F.normalize(self.proj_head(f), dim=-1)
        return vel, z

    def forward(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def get_latent(self, obs_history):
        return self.forward(obs_history)

    # -------- 速度桶函数（按 v_x 等距分桶；可换成多维 KMeans）--------
    @torch.no_grad()
    def velocity_to_bucket(self, v_cmd):  # v_cmd: (B,3)
        vx = v_cmd[:, 0].clamp(-1.0, 4.0)
        edges = torch.linspace(-1.0, 4.0, self.K + 1, device=vx.device)
        y = torch.bucketize(vx, edges) - 1
        return y.clamp(0, self.K - 1)

    # -------- 训练步 --------
    def update(self, obs_history, next_critic_obs, v_cmd=None, lr=None):
        if lr is not None:
            for g in self.optimizer.param_groups: g['lr'] = lr

        # 1) 速度回归 target & 桶标签
        v_true = next_critic_obs[:, self.D:self.D+3].detach()
        labels = self.velocity_to_bucket(v_cmd if v_cmd is not None else v_true)

        # 2) online encode
        f = self.backbone(obs_history)
        pred_v = self.vel_head(f)
        z = F.normalize(self.proj_head(f), dim=-1)        # (B, d)

        # 3) (可选) momentum encode for queue
        if self.use_momentum:
            with torch.no_grad():
                self._momentum_update()
                f_m = self.backbone_m(obs_history)
                z_m = F.normalize(self.proj_head_m(f_m), dim=-1)

        # 4) 原型损失：拉向本桶原型 / 推离他桶
        proto = F.normalize(self.bucket_proto.weight, dim=-1)   # (K, d)
        logits_p = z @ proto.T / self.tau                       # (B, K)
        loss_proto = F.cross_entropy(logits_p, labels)

        # 5) 样本级 SupCon（含 queue）
        if self.use_momentum:
            bank_z = torch.cat([z_m, self.queue_z], dim=0)
            bank_y = torch.cat([labels, self.queue_y], dim=0)
        else:
            bank_z, bank_y = z.detach(), labels
        loss_supcon = supcon_loss(z, labels, bank_z, bank_y, self.tau)

        # 6) 原型多样性正则
        sim = proto @ proto.T
        off = sim - torch.eye(self.K, device=sim.device)
        loss_div = F.relu(off - 0.3).pow(2).mean()

        # 7) HIM 速度回归
        loss_vel = F.mse_loss(pred_v, v_true)

        loss = loss_vel + self.l1 * loss_proto + self.l2 * loss_supcon + self.l3 * loss_div

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.use_momentum:
            self._dequeue_and_enqueue(z_m.detach(), labels.detach())

        # 复用 HIM runner 期待的两个返回值
        return loss_vel.item(), (self.l1*loss_proto + self.l2*loss_supcon).item()

    # ---- helpers ----
    @torch.no_grad()
    def _copy_to_momentum(self):
        for pq, p in zip(self.backbone_m.parameters(), self.backbone.parameters()):
            pq.data.copy_(p.data)
        for pq, p in zip(self.proj_head_m.parameters(), self.proj_head.parameters()):
            pq.data.copy_(p.data)

    @torch.no_grad()
    def _momentum_update(self):
        for pq, p in zip(self.backbone_m.parameters(), self.backbone.parameters()):
            pq.data.mul_(self.m).add_(p.data, alpha=1 - self.m)
        for pq, p in zip(self.proj_head_m.parameters(), self.proj_head.parameters()):
            pq.data.mul_(self.m).add_(p.data, alpha=1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z, y):
        B = z.size(0); ptr = int(self.queue_ptr)
        end = ptr + B
        if end <= self.queue_size:
            self.queue_z[ptr:end] = z; self.queue_y[ptr:end] = y
        else:
            n1 = self.queue_size - ptr; n2 = B - n1
            self.queue_z[ptr:] = z[:n1]; self.queue_y[ptr:] = y[:n1]
            self.queue_z[:n2] = z[n1:]; self.queue_y[:n2] = y[n1:]
        self.queue_ptr[0] = end % self.queue_size


def supcon_loss(z, y, bank_z, bank_y, tau):
    """ Khosla et al., 2020. z:(B,d) anchors; bank:(N,d). """
    logits = z @ bank_z.T / tau                          # (B, N)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    pos_mask = (y.unsqueeze(1) == bank_y.unsqueeze(0)).float()
    # 排除自身（如果 bank 含 z 自身，要在调用方处理；这里假设 bank 是 momentum/queue）
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
    loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
    # batch 内某些样本本桶完全无正样本则跳过
    valid = (pos_mask.sum(dim=1) > 0).float()
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)