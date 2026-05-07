class CICIntrinsicReward(nn.Module):
    def __init__(self, encoder, proto, tau=0.1, K=10):
        super().__init__()
        self.encoder, self.proto, self.tau, self.K = encoder, proto, tau, K
        self.register_buffer("track_ema", torch.tensor(0.0))
        self.ema_m = 0.99

    @torch.no_grad()
    def compute(self, obs_history, cmd_bucket, r_track,
                alpha, beta, kappa=10.0):
        z = F.normalize(self.encoder(obs_history)[1], dim=-1)   # (B, d)
        e = F.normalize(self.proto.weight, dim=-1)              # (K, d)
        logits = z @ e.T / self.tau
        log_p = F.log_softmax(logits, dim=-1)
        r_int = log_p.gather(1, cmd_bucket.unsqueeze(1)).squeeze(1) + math.log(self.K)

        self.track_ema.mul_(self.ema_m).add_(r_track.mean(), alpha=1 - self.ema_m)
        gate = torch.sigmoid(kappa * (self.track_ema - beta))
        return alpha * gate * r_int          # 直接累加到 step reward