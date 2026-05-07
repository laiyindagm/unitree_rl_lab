z_s = self.encoder(obs_history)        # 历史 → source latent
z_t = self.target(next_obs)            # 下一帧 → target latent
...
score_s = z_s @ self.proto.weight.T    # 32 个原型上的分配分数
q_s = sinkhorn(score_s); q_t = sinkhorn(score_t)
swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
estimation_loss = F.mse_loss(pred_vel, vel)