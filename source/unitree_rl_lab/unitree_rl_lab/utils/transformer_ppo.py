"""PPO with transformer latent auxiliary objectives.

Supports optional next-observation prediction and explicit velocity regression
on top of a history encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from unitree_rl_lab.utils.rsl_rl_custom_ppo import _sanitize_model_cfg


class TransformerPPO(PPO):
    """PPO with transformer latent auxiliary losses."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        aux_loss_coef: float = 0.5,
        aux_loss_schedule: list[float] | None = None,
        velocity_aux_coef: float = 0.0,
        achieved_ang_vel_scale: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_schedule = list(aux_loss_schedule) if aux_loss_schedule is not None else None
        self.velocity_aux_coef = float(velocity_aux_coef)
        self.achieved_ang_vel_scale = float(achieved_ang_vel_scale)
        self.counter = 0

    def _get_aux_coef(self) -> float:
        if self.aux_loss_schedule is None:
            return self.aux_loss_coef

        if len(self.aux_loss_schedule) != 4:
            raise ValueError("aux_loss_schedule must be [start, end, decay_start, decay_steps].")

        start, end, decay_start, decay_steps = self.aux_loss_schedule
        if decay_steps <= 0:
            return float(end)

        progress = min(max((self.counter - decay_start), 0) / decay_steps, 1.0)
        return float(start + progress * (end - start))

    def _flatten_obs(self, obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[k] for k in obs_groups], dim=-1)

    def _get_achieved_velocity_targets(self, obs_t: TensorDict) -> torch.Tensor:
        critic_flat = self._flatten_obs(obs_t, self.critic.obs_groups)
        critic_single_dim = critic_flat.shape[-1] // self.actor.history_len
        critic_frames = critic_flat.view(-1, self.actor.history_len, critic_single_dim)
        last_critic = critic_frames[:, -1, :]
        inv_ang_scale = 1.0 / max(self.achieved_ang_vel_scale, 1e-6)
        return torch.stack(
            [
                last_critic[:, 0],
                last_critic[:, 1],
                last_critic[:, 5] * inv_ang_scale,
            ],
            dim=-1,
        )

    def _compute_aux_losses(self, aux_coef: float) -> dict[str, torch.Tensor]:
        T = self.storage.num_transitions_per_env
        if T < 2:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "next_obs": zero,
                "velocity": zero,
            }

        history_obs_dim: int = self.actor.history_obs_dim  # type: ignore[attr-defined]
        history_start_idx: int = self.actor.history_start_idx  # type: ignore[attr-defined]
        history_total_dim: int = self.actor.history_total_dim  # type: ignore[attr-defined]
        last_frame_start = history_start_idx + history_total_dim - history_obs_dim

        next_obs_loss_sum = 0.0
        velocity_loss_sum = 0.0
        self.optimizer.zero_grad()
        for t in range(T - 1):
            obs_t = self.storage.observations[t]
            obs_next = self.storage.observations[t + 1]

            outputs = self.actor.get_latent_outputs(obs_t)
            step_aux_loss = torch.tensor(0.0, device=self.device)

            if getattr(self.actor, "enable_aux_loss", False) and aux_coef > 0.0:
                predicted_next = outputs["predicted_next_obs"]
                with torch.no_grad():
                    flat_next = self._flatten_obs(obs_next, self.actor.obs_groups)
                    flat_next = self.actor.obs_normalizer(flat_next)
                    target_next = flat_next[:, last_frame_start : last_frame_start + history_obs_dim]
                next_obs_loss = nn.functional.mse_loss(predicted_next, target_next)
                step_aux_loss = step_aux_loss + aux_coef * next_obs_loss
                next_obs_loss_sum += next_obs_loss.item()

            if self.velocity_aux_coef > 0.0:
                achieved_t = self._get_achieved_velocity_targets(obs_t)
                velocity_loss = nn.functional.smooth_l1_loss(outputs["velocity_pred"], achieved_t)
                step_aux_loss = step_aux_loss + self.velocity_aux_coef * velocity_loss
                velocity_loss_sum += velocity_loss.item()

            if step_aux_loss.requires_grad:
                (step_aux_loss / (T - 1)).backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

        denom = T - 1
        return {
            "next_obs": torch.tensor(next_obs_loss_sum / denom, device=self.device),
            "velocity": torch.tensor(velocity_loss_sum / denom, device=self.device),
        }

    def _get_actor_critic_outputs(
        self,
        batch_obs: TensorDict,
        masks: torch.Tensor | None,
        hidden_states: tuple | list | None,
        stochastic_output: bool,
    ) -> tuple[tuple, torch.Tensor, torch.Tensor]:
        actor_hidden_state = hidden_states[0] if hidden_states is not None else None
        critic_hidden_state = hidden_states[1] if hidden_states is not None else None
        self.actor(
            batch_obs,
            masks=masks,
            hidden_state=actor_hidden_state,
            stochastic_output=stochastic_output,
        )
        values = self.critic(batch_obs, masks=masks, hidden_state=critic_hidden_state)
        distribution_params = self.actor.output_distribution_params
        entropy = self.actor.output_entropy
        return distribution_params, values, entropy

    def update(self) -> dict[str, float]:
        aux_coef = self._get_aux_coef()
        has_next_obs_aux = getattr(self.actor, "enable_aux_loss", False) and aux_coef > 0.0
        has_velocity_aux = self.velocity_aux_coef > 0.0

        mean_aux_loss = 0.0
        mean_velocity_aux_loss = 0.0

        if has_next_obs_aux or has_velocity_aux:
            losses = self._compute_aux_losses(aux_coef)
            mean_aux_loss = losses["next_obs"].item()
            mean_velocity_aux_loss = losses["velocity"].item()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            distribution_params, values, entropy = self._get_actor_critic_outputs(
                batch.observations,
                batch.masks,
                batch.hidden_states,
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            distribution_params = tuple(p[:original_batch_size] for p in distribution_params)
            entropy = entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            if self.symmetry:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                if not self.symmetry["use_data_augmentation"]:
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = torch.nn.MSELoss()(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self.counter += 1

        loss_dict: dict[str, float] = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "aux_pred": mean_aux_loss,
            "aux_coef": aux_coef,
            "velocity_aux": mean_velocity_aux_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict


    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_cfg = _sanitize_model_cfg(cfg["actor"], actor_class)
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
            actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        return alg
