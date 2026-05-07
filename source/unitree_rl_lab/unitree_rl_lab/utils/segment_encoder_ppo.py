"""V22a algorithm: VelocityEstimatorPPO + frozen V3 segment encoder.

Maintains a per-env rolling buffer of policy observations (T_seg frames of the
295-dim flat policy obs). At each `act()` call the buffer is rolled, the latest
flat obs is appended, the frozen encoder produces `z_gait` (32-dim), and that
tensor is injected into the obs TensorDict under the key ``"z_gait"``. The
custom actor (`TransformerLatentGaitModel`) reads ``obs["z_gait"]`` and
concatenates it into its policy latent. The critic does NOT consume z_gait
(no value-side leakage).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from unitree_rl_lab.utils.velocity_estimator_ppo import VelocityEstimatorPPO
from unitree_rl_lab.utils.frozen_segment_encoder import FrozenSegmentEncoder
from unitree_rl_lab.utils.rsl_rl_custom_ppo import _sanitize_model_cfg


class SegmentEncoderVelocityEstimatorPPO(VelocityEstimatorPPO):
    """V22a PPO: injects frozen V3 z_gait into obs TensorDict for the actor."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        encoder_path: str,
        z_buffer_len: int = 32,
        gait_dim: int = 32,
        num_envs: int,
        actor_obs_key: str = "policy",
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.encoder = FrozenSegmentEncoder(encoder_path, device=self.device)
        self.z_buffer_len = int(z_buffer_len)
        self.gait_dim = int(gait_dim)
        self.actor_obs_key = actor_obs_key
        if self.encoder.z_dim != self.gait_dim:
            raise ValueError(
                f"Encoder z_dim={self.encoder.z_dim} != configured gait_dim={self.gait_dim}"
            )
        self._num_envs = int(num_envs)
        self._buf = torch.zeros(
            self._num_envs, self.z_buffer_len, self.encoder.frame_dim,
            device=self.device,
        )

    def _compute_z_gait(self, obs: TensorDict) -> torch.Tensor:
        flat_obs = obs[self.actor_obs_key]
        # Roll buffer left by one along time axis and append the latest flat obs.
        self._buf = torch.roll(self._buf, shifts=-1, dims=1)
        self._buf[:, -1, :] = flat_obs.detach()
        return self.encoder.encode(self._buf)

    def act(self, obs, *args, **kwargs):  # noqa: D401
        z = self._compute_z_gait(obs)
        obs["z_gait"] = z
        return super().act(obs, *args, **kwargs)

    def process_env_step(self, obs, rewards, dones, extras):  # noqa: D401
        # Reset rolling buffer for envs that just finished an episode.
        if dones is not None:
            done_mask = dones.bool().view(-1)
            if done_mask.any():
                self._buf[done_mask] = 0.0
        # Note: post-step obs does NOT need z_gait — actor/critic update_normalization
        # only touches obs_groups (which excludes z_gait), and value bootstrap goes
        # through the critic which never reads z_gait.
        return super().process_env_step(obs, rewards, dones, extras)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        # Pull V22a-specific knobs out of cfg["algorithm"] before resolve_*.
        alg_cfg = cfg["algorithm"]
        encoder_path = alg_cfg.pop("encoder_path")
        z_buffer_len = int(alg_cfg.pop("z_buffer_len", 32))
        gait_dim = int(alg_cfg.pop("gait_dim", 32))
        actor_obs_key = alg_cfg.pop("actor_obs_key", "policy")

        alg_class: type[PPO] = resolve_callable(alg_cfg.pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Inject zero z_gait BEFORE resolve_obs_groups / storage allocation.
        z_zeros = torch.zeros(env.num_envs, gait_dim, device=device)
        obs["z_gait"] = z_zeros

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # gait_dim must be passed into the actor model so it sizes its MLP correctly.
        actor_cfg = _sanitize_model_cfg(cfg["actor"], actor_class)
        actor_cfg["gait_dim"] = gait_dim
        critic_cfg = _sanitize_model_cfg(cfg["critic"], critic_class)

        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
        ).to(device)
        print(f"Actor Model: {actor}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]

        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **critic_cfg
        ).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            encoder_path=encoder_path,
            z_buffer_len=z_buffer_len,
            gait_dim=gait_dim,
            num_envs=env.num_envs,
            actor_obs_key=actor_obs_key,
            multi_gpu_cfg=cfg["multi_gpu"],
            **cfg["algorithm"],
        )
        return alg
