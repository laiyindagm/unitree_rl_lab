"""V22b: V22a + metric-residual intrinsic reward with SMERL gate.

Per env step we add to the env reward:

    r_intrinsic = (||z - z_axial(v_cmd)||^2 / d_gait)        # cmd-projected residual
    r_total     = r_env + alpha(t) * gate(s) * r_intrinsic

where:
  - z = FrozenSegmentEncoder(rolling buffer of policy obs)         (V22a path)
  - z_axial = encoder.axial_predict(v_cmd)                          (V3 axial bases)
  - alpha(t)   linearly ramps from 0 to alpha_max over a warmup window
  - gate(s)    = sigmoid(kappa * (track_ema - threshold))           (SMERL)
  - track_ema  per-env EMA of the env reward (NOT incl. intrinsic)

When ||v_cmd||_W < eps_cmd (standing / pure_wz boundary), z_axial = 0 so
||z - z_axial|| = ||z|| would grow unboundedly. We mask r_intrinsic = 0 in
that regime so the policy is not pushed toward larger z just for free reward.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv

from unitree_rl_lab.utils.segment_encoder_ppo import (
    SegmentEncoderVelocityEstimatorPPO,
)


class SegmentEncoderCICPPO(SegmentEncoderVelocityEstimatorPPO):
    """V22b: SegmentEncoder PPO + metric-residual intrinsic reward."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        # CIC / SMERL knobs
        cic_alpha_max: float = 0.02,
        cic_warmup_start_iter: int = 200,
        cic_warmup_end_iter: int = 2000,
        smerl_threshold: float = 0.045,        # per-step env reward threshold
        smerl_kappa: float = 200.0,
        track_ema_decay: float = 0.99,
        cmd_norm_eps: float = 0.1,             # min ||v||_W to enable r_int
        # cmd slice in the policy flat obs (term-major, last-frame)
        # default for V21c policy obs:
        #   base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + ...
        # term-major × history_len=5 → velocity_commands block 30..45,
        # last-frame indices [42, 43, 44].
        cmd_obs_indices: list[int] | tuple[int, int, int] = (42, 43, 44),
        intrinsic_log_every: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.cic_alpha_max = float(cic_alpha_max)
        self.cic_warmup_start_iter = int(cic_warmup_start_iter)
        self.cic_warmup_end_iter = int(cic_warmup_end_iter)
        self.smerl_threshold = float(smerl_threshold)
        self.smerl_kappa = float(smerl_kappa)
        self.track_ema_decay = float(track_ema_decay)
        self.cmd_norm_eps = float(cmd_norm_eps)
        self.cmd_obs_indices = [int(i) for i in cmd_obs_indices]
        self.intrinsic_log_every = int(intrinsic_log_every)

        # per-env EMA of env reward (initialised on first call)
        self._track_ema: torch.Tensor | None = None
        self._z_dim = self.encoder.z_dim
        # cached z from last act() to align with rewards in process_env_step
        self._last_z: torch.Tensor | None = None
        # rolling logs (for printing)
        self._intrinsic_step = 0
        self._intrinsic_buf: list[float] = []
        self._gate_buf: list[float] = []
        self._alpha_buf: list[float] = []

    def _alpha_now(self) -> float:
        c = self.counter
        w0, w1 = self.cic_warmup_start_iter, self.cic_warmup_end_iter
        if c <= w0:
            return 0.0
        if c >= w1:
            return self.cic_alpha_max
        return self.cic_alpha_max * (c - w0) / max(1, (w1 - w0))

    def act(self, obs, *args, **kwargs):
        # Parent (V22a) computes z, sets obs["z_gait"], calls actor/critic.
        action = super().act(obs, *args, **kwargs)
        # Cache z and cmd at the act-time obs for the next process_env_step.
        self._last_z = obs["z_gait"].detach()
        self._last_cmd = obs[self.actor_obs_key][:, self.cmd_obs_indices].detach()
        return action

    def _compute_intrinsic(
        self, env_rewards: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute r_total = r_env + alpha * gate * r_int. Returns augmented rewards."""
        if self._last_z is None or self._last_cmd is None:
            return env_rewards, {}

        # cmd norm in W metric
        sigma = self.encoder.sigma_cmd  # (3,)
        cmd_w = self._last_cmd / sigma.unsqueeze(0).clamp(min=1e-3)
        cmd_norm = cmd_w.norm(dim=-1)  # (N,)
        active = (cmd_norm > self.cmd_norm_eps).float()  # (N,)

        z_axial = self.encoder.axial_predict(self._last_cmd)  # (N, d)
        residual = self._last_z - z_axial
        r_int = (residual.pow(2).sum(dim=-1) / self._z_dim) * active  # (N,)

        # Update per-env EMA on the *env* reward (squeeze last dim).
        env_r = env_rewards.detach().view(-1)
        if self._track_ema is None:
            self._track_ema = env_r.clone()
        else:
            d = self.track_ema_decay
            self._track_ema = d * self._track_ema + (1.0 - d) * env_r

        gate = torch.sigmoid(self.smerl_kappa * (self._track_ema - self.smerl_threshold))
        alpha = self._alpha_now()
        bonus = alpha * gate * r_int  # (N,)

        out = env_rewards + bonus.view_as(env_rewards)

        info = {
            "alpha": alpha,
            "gate_mean": float(gate.mean().item()),
            "r_int_mean": float(r_int.mean().item()),
            "r_int_active_frac": float(active.mean().item()),
            "track_ema_mean": float(self._track_ema.mean().item()),
            "bonus_mean": float(bonus.mean().item()),
        }
        return out, info

    def process_env_step(self, obs, rewards, dones, extras):  # noqa: D401
        rewards_aug, info = self._compute_intrinsic(rewards)
        self._intrinsic_step += 1
        if info:
            self._intrinsic_buf.append(info["r_int_mean"])
            self._gate_buf.append(info["gate_mean"])
            self._alpha_buf.append(info["alpha"])
            if self._intrinsic_step % self.intrinsic_log_every == 0:
                import statistics as _stats
                ri = _stats.fmean(self._intrinsic_buf[-self.intrinsic_log_every:])
                gm = _stats.fmean(self._gate_buf[-self.intrinsic_log_every:])
                am = _stats.fmean(self._alpha_buf[-self.intrinsic_log_every:])
                print(
                    f"[V22b CIC] iter={self.counter} step={self._intrinsic_step} "
                    f"alpha={am:.4f} gate={gm:.3f} r_int={ri:.4f} "
                    f"bonus_mean={info['bonus_mean']:.4f}",
                    flush=True,
                )
        return super().process_env_step(obs, rewards_aug, dones, extras)
