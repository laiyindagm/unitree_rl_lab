"""Validated bilevel LIRPG for V21o.

This module keeps V21l's linear learnable reward form but changes the meta
objective from current-rollout reward/advantage correlation to a validation
objective after a differentiable one-step shadow actor update.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any

import torch
from tensordict import TensorDict
from torch.distributions import Normal
from torch.func import functional_call

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from unitree_rl_lab.utils import intrinsic_reward as ir
from unitree_rl_lab.utils.velocity_estimator_ppo import VelocityEstimatorPPO


class ValidatedBilevelOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner variant that lets the algorithm perform validation rollouts."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if hasattr(self.alg, "set_validation_env"):
            self.alg.set_validation_env(self.env)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            if hasattr(self.alg, "validated_update"):
                loss_dict, obs = self.alg.validated_update(obs)
            else:
                loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
            self.logger.stop_logging_writer()


class ValidatedBilevelLirpgVelocityEstimatorPPO(VelocityEstimatorPPO):
    """VelocityEstimatorPPO with validated bilevel meta-updates for linear LIRPG."""

    _CHANNEL_TO_REWARD_TERM = {
        "lin_xy": "track_lin_vel_xy",
        "ang_z": "track_ang_vel_z",
        "lin_xy_cmd": "track_lin_vel_xy",
        "ang_z_cmd": "track_ang_vel_z",
    }

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        lirpg_meta_gamma: float = 0.99,
        lirpg_meta_lam: float = 0.95,
        lirpg_warmup_iters: int = 6000,
        validation_steps: int = 24,
        inner_lr: float = 3.0e-4,
        inner_steps: int = 1,
        inner_max_samples: int = 8192,
        outer_max_samples: int = 8192,
        true_lin_cmd_floor: float = 0.1,
        true_ang_cmd_floor: float = 0.1,
        true_fall_tail_penalty: float = 1.0,
        meta_step_param_l2_coef: float = 1.0e-4,
        meta_grad_clip: float = 1.0,
        strict_actor_update: bool = False,
        update_aux_before_shadow: bool = False,
        true_reward_mode: str = "step",
        true_pure_wz_lin_abs_weight: float = 1.0,
        true_pure_linear_yaw_abs_weight: float = 1.0,
        true_standing_lin_abs_weight: float = 1.0,
        true_standing_yaw_abs_weight: float = 1.0,
        commit_meta_reward_to_ppo: bool = False,
        require_meta_channels: bool = False,
        log_meta_effectiveness: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.lirpg_meta_gamma = float(lirpg_meta_gamma)
        self.lirpg_meta_lam = float(lirpg_meta_lam)
        self.lirpg_warmup_iters = int(lirpg_warmup_iters)
        self.validation_steps = int(validation_steps)
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.inner_max_samples = int(inner_max_samples)
        self.outer_max_samples = int(outer_max_samples)
        self.true_lin_cmd_floor = float(true_lin_cmd_floor)
        self.true_ang_cmd_floor = float(true_ang_cmd_floor)
        self.true_fall_tail_penalty = float(true_fall_tail_penalty)
        self.meta_step_param_l2_coef = float(meta_step_param_l2_coef)
        self.meta_grad_clip = float(meta_grad_clip)
        self.strict_actor_update = bool(strict_actor_update)
        self.update_aux_before_shadow = bool(update_aux_before_shadow)
        if true_reward_mode not in {"step", "trajectory"}:
            raise ValueError(f"Unsupported true_reward_mode: {true_reward_mode!r}")
        self.true_reward_mode = str(true_reward_mode)
        self.true_pure_wz_lin_abs_weight = float(true_pure_wz_lin_abs_weight)
        self.true_pure_linear_yaw_abs_weight = float(true_pure_linear_yaw_abs_weight)
        self.true_standing_lin_abs_weight = float(true_standing_lin_abs_weight)
        self.true_standing_yaw_abs_weight = float(true_standing_yaw_abs_weight)
        self.commit_meta_reward_to_ppo = bool(commit_meta_reward_to_ppo)
        self.require_meta_channels = bool(require_meta_channels)
        self.log_meta_effectiveness = bool(log_meta_effectiveness)
        if self.commit_meta_reward_to_ppo and self.strict_actor_update:
            raise ValueError(
                "commit_meta_reward_to_ppo requires strict_actor_update=False so the real policy update "
                "remains the full PPO update."
            )
        self._lirpg_step = 0
        self._env = None
        self._last_values: torch.Tensor | None = None
        self._pending_lirpg_channel_state: dict[str, Any] | None = None

    def set_validation_env(self, env) -> None:
        self._env = env

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        super().process_env_step(obs, rewards, dones, extras)
        self._restore_pending_lirpg_channel_state()
        for chan in ir.all_channels().values():
            if hasattr(chan, "record_dones"):
                chan.record_dones(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        self._last_values = self.critic(obs).detach()
        super().compute_returns(obs)

    def validated_update(self, obs: TensorDict) -> tuple[dict[str, float], TensorDict]:
        if self._env is None:
            raise RuntimeError("ValidatedBilevelLirpgVelocityEstimatorPPO requires set_validation_env().")

        if self._lirpg_step < self.lirpg_warmup_iters:
            for chan in ir.all_channels().values():
                if hasattr(chan, "clear_buffer"):
                    chan.clear_buffer()
            self._lirpg_step += 1
            out = super().update()
            out["lirpg/warmup_remaining"] = float(self.lirpg_warmup_iters - self._lirpg_step)
            return out, obs

        aux_logs: dict[str, float] = {}
        if self.strict_actor_update and self.update_aux_before_shadow:
            aux_logs = self._update_actor_auxiliary_losses()

        channels = self._linear_channels()
        if self.require_meta_channels and not channels:
            raise RuntimeError(
                "Validated LIRPG was configured with require_meta_channels=True, but no active differentiable "
                "reward channel was recorded in the current rollout."
            )

        theta_prime, inner_logs, train_aux = self._build_shadow_actor_params()

        val_data, obs_after_val = self._collect_validation_rollout(obs, theta_prime)
        meta_loss, outer_logs = self._compute_outer_meta_loss(theta_prime, val_data, train_aux)

        params_before = [p.detach().clone() for chan in channels for p in chan.parameters()]
        for chan in channels:
            chan.optim.zero_grad()
        meta_loss.backward()
        meta_grad_norm_sq = 0.0
        for chan in channels:
            grad_norm = torch.nn.utils.clip_grad_norm_(chan.parameters(), self.meta_grad_clip)
            meta_grad_norm_sq += float(grad_norm.detach().item()) ** 2
            chan.optim.step()
            chan.snapshot_meta_state()
        params_after = [p.detach() for chan in channels for p in chan.parameters()]
        meta_param_delta_sq = 0.0
        for before, after in zip(params_before, params_after):
            meta_param_delta_sq += float((after - before).pow(2).sum().item())
        meta_update_logs = {
            "validated/meta_grad_norm": meta_grad_norm_sq ** 0.5,
            "validated/meta_param_delta_l2": meta_param_delta_sq ** 0.5,
            "validated/meta_channel_count": float(len(channels)),
        }

        if self.strict_actor_update:
            self._accept_actor_params(theta_prime)
            ppo_logs = self._update_critic_only()
            ppo_logs.update(aux_logs)
            ppo_logs["validated/strict_actor_update"] = 1.0
        else:
            # The differentiable theta_prime is only a meta-gradient probe for phi.
            # The real policy update remains the full baseline PPO update
            # (epochs x minibatches, adaptive KL, aux losses, actor+critic).
            if self.commit_meta_reward_to_ppo:
                ppo_logs = self._commit_meta_rewards_to_storage()
            else:
                ppo_logs = {"validated/ppo_reward_recomputed": 0.0}
            commit_logs = ppo_logs
            ppo_logs = super().update()
            ppo_logs.update(commit_logs)
            ppo_logs["validated/strict_actor_update"] = 0.0

        for chan in ir.all_channels().values():
            if hasattr(chan, "clear_buffer"):
                chan.clear_buffer()
        self._lirpg_step += 1

        loss_dict = {
            **ppo_logs,
            **inner_logs,
            **outer_logs,
            **meta_update_logs,
            "validated/meta_loss": float(meta_loss.detach().item()),
            "validated/accepted": 1.0,
        }
        return loss_dict, obs_after_val

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["lirpg_step"] = int(self._lirpg_step)
        channels: dict[str, dict[str, Any]] = {}
        for name, chan in ir.all_channels().items():
            if not hasattr(chan, "state_dict"):
                continue
            payload: dict[str, Any] = {"state_dict": chan.state_dict()}
            if hasattr(chan, "optim"):
                payload["optimizer_state_dict"] = chan.optim.state_dict()
            channels[name] = payload
        saved_dict["lirpg_channels"] = channels
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self._lirpg_step = int(loaded_dict.get("lirpg_step", loaded_dict.get("iter", self._lirpg_step)))
        elif "lirpg_step" in loaded_dict:
            self._lirpg_step = int(loaded_dict["lirpg_step"])
        self._pending_lirpg_channel_state = loaded_dict.get("lirpg_channels")
        self._restore_pending_lirpg_channel_state()
        return load_iteration

    def _restore_pending_lirpg_channel_state(self) -> None:
        if not self._pending_lirpg_channel_state:
            return
        remaining: dict[str, Any] = {}
        channels = ir.all_channels()
        for name, payload in self._pending_lirpg_channel_state.items():
            chan = channels.get(name)
            if chan is None:
                remaining[name] = payload
                continue
            chan.load_state_dict(payload["state_dict"])
            if "optimizer_state_dict" in payload and hasattr(chan, "optim"):
                chan.optim.load_state_dict(payload["optimizer_state_dict"])
            if hasattr(chan, "snapshot_meta_state"):
                chan.snapshot_meta_state()
        self._pending_lirpg_channel_state = remaining or None

    # ------------------------------------------------------------------ actor

    def _actor_param_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((name, p) for name, p in self.actor.named_parameters())

    def _actor_state_for_call(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = dict(params)
        state.update({name: b for name, b in self.actor.named_buffers()})
        return state

    def _accept_actor_params(self, params: dict[str, torch.Tensor]) -> None:
        own_params = dict(self.actor.named_parameters())
        with torch.no_grad():
            for name, value in params.items():
                if name in own_params:
                    own_params[name].copy_(value.detach())

    def _update_actor_auxiliary_losses(self) -> dict[str, float]:
        aux_coef = self._get_aux_coef()
        has_next_obs_aux = getattr(self.actor, "enable_aux_loss", False) and aux_coef > 0.0
        has_velocity_aux = self.velocity_aux_coef > 0.0

        mean_aux_loss = 0.0
        mean_velocity_aux_loss = 0.0
        if has_next_obs_aux or has_velocity_aux:
            losses = self._compute_aux_losses(aux_coef)
            mean_aux_loss = float(losses["next_obs"].item())
            mean_velocity_aux_loss = float(losses["velocity"].item())

        return {
            "aux_pred": mean_aux_loss,
            "aux_coef": float(aux_coef),
            "velocity_aux": mean_velocity_aux_loss,
        }

    def _update_critic_only(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_entropy = 0.0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
            with torch.no_grad():
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                entropy = self.actor.output_entropy

            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            self.optimizer.zero_grad()
            (self.value_loss_coef * value_loss).backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += float(value_loss.item())
            mean_entropy += float(entropy.mean().item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_entropy /= num_updates
        self.storage.clear()
        self.counter += 1

        return {
            "value": mean_value_loss,
            "surrogate": 0.0,
            "entropy": mean_entropy,
        }

    def _actor_mean(self, obs: TensorDict, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(
            self.actor,
            self._actor_state_for_call(params),
            (obs,),
            {"stochastic_output": False},
        )

    def _actor_std(self, params: dict[str, torch.Tensor], mean: torch.Tensor) -> torch.Tensor:
        if "distribution.log_std_param" in params:
            std = torch.exp(params["distribution.log_std_param"])
        elif "distribution.std_param" in params:
            std = params["distribution.std_param"]
        elif hasattr(self.actor.distribution, "log_std_param"):
            std = torch.exp(self.actor.distribution.log_std_param)
        else:
            std = self.actor.distribution.std_param
        return std.expand_as(mean)

    def _log_prob_entropy(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._actor_mean(obs, params)
        std = self._actor_std(params, mean)
        dist = Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)

    def _sample_actions(self, obs: TensorDict, params: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = self._actor_mean(obs, params)
        std = self._actor_std(params, mean)
        return Normal(mean, std).sample()

    # -------------------------------------------------------------- train step

    def _build_shadow_actor_params(
        self,
    ) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float], dict[str, Any]]:
        rewards_phi, train_logs, channel_aux = self._differentiable_train_rewards()
        advantages_phi = self._gae_from_rewards(rewards_phi, self.storage.values.detach(), self.storage.dones)
        advantages_phi = self._stop_stat_normalize(advantages_phi)

        flat_obs = self.storage.observations.flatten(0, 1)
        flat_actions = self.storage.actions.flatten(0, 1)
        flat_old_logp = self.storage.actions_log_prob.squeeze(-1).flatten(0, 1).detach()
        flat_adv = advantages_phi.flatten(0, 1)
        inner_idx = self._sample_flat_indices(flat_adv.shape[0], self.inner_max_samples, flat_adv.device)
        flat_obs = flat_obs[inner_idx]
        flat_actions = flat_actions[inner_idx]
        flat_old_logp = flat_old_logp[inner_idx]
        flat_adv = flat_adv[inner_idx]

        theta = self._actor_param_dict()
        inner_loss_value = torch.tensor(0.0, device=self.device)
        entropy_value = torch.tensor(0.0, device=self.device)
        for _ in range(max(self.inner_steps, 1)):
            logp, entropy = self._log_prob_entropy(flat_obs, flat_actions, theta)
            ratio = torch.exp(logp - flat_old_logp)
            unclipped = ratio * flat_adv
            clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * flat_adv
            actor_loss = -torch.min(unclipped, clipped).mean()
            inner_loss = actor_loss - self.entropy_coef * entropy.mean()
            grads = torch.autograd.grad(
                inner_loss,
                tuple(theta.values()),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            theta = OrderedDict(
                (name, p if g is None else p - self.inner_lr * g)
                for (name, p), g in zip(theta.items(), grads)
            )
            inner_loss_value = actor_loss.detach()
            entropy_value = entropy.mean().detach()

        train_logs.update(
            {
                "validated/inner_surrogate": float(inner_loss_value.item()),
                "validated/inner_entropy": float(entropy_value.item()),
                "validated/train_A_phi_mean": float(advantages_phi.mean().detach().item()),
                "validated/train_A_phi_std": float(advantages_phi.std(unbiased=False).detach().item()),
            }
        )
        return theta, train_logs, channel_aux

    def _differentiable_train_rewards(self) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        st_rewards = self.storage.rewards.squeeze(-1).detach()
        old_track = torch.zeros_like(st_rewards)
        diff_track = torch.zeros_like(st_rewards)
        logs: dict[str, float] = {}
        aux: dict[str, Any] = {"channel_tensors": []}

        dt = float(self._env.unwrapped.step_dt)
        for chan in self._linear_channels():
            feats, task_err, old_raw_reward, _ = chan.stacked_buffer()
            if feats.shape[0] != self.storage.num_transitions_per_env:
                raise RuntimeError(
                    f"Channel {chan.name} buffer length {feats.shape[0]} does not match rollout length "
                    f"{self.storage.num_transitions_per_env}."
                )
            term_name = self._CHANNEL_TO_REWARD_TERM.get(chan.name)
            if term_name is None:
                continue
            weight = float(self._env.unwrapped.reward_manager.get_term_cfg(term_name).weight)
            diff_raw_reward, ch_logs = chan.differentiable_reward_from_buffer(feats, task_err)
            old_track = old_track + weight * dt * old_raw_reward.detach()
            diff_track = diff_track + weight * dt * diff_raw_reward
            reg_loss, reg_logs = chan.regularization_loss(
                feats,
                task_err,
                step_param_l2_coef=self.meta_step_param_l2_coef,
            )
            aux["channel_tensors"].append((chan, feats, task_err, reg_loss, ch_logs, reg_logs))
            prefix = f"lirpg_{chan.name}/"
            logs[prefix + "old_r_mean"] = float(old_raw_reward.mean().detach().item())
            for key, value in ch_logs.items():
                logs[prefix + key] = float(value.detach().item())
            for key, value in reg_logs.items():
                logs[prefix + key] = float(value.detach().item())

        fixed_other = st_rewards - old_track
        rewards_phi = fixed_other + diff_track
        logs["validated/fixed_other_mean"] = float(fixed_other.mean().detach().item())
        logs["validated/diff_track_mean"] = float(diff_track.mean().detach().item())
        return rewards_phi, logs, aux

    def _commit_meta_rewards_to_storage(self) -> dict[str, float]:
        old_rewards = self.storage.rewards.squeeze(-1).detach().clone()
        old_advantages = self.storage.advantages.squeeze(-1).detach().clone()
        rewards_phi, _, _ = self._differentiable_train_rewards()
        rewards_phi = rewards_phi.detach()

        raw_advantages = self._gae_from_rewards(
            rewards_phi,
            self.storage.values.detach(),
            self.storage.dones,
        ).detach()
        returns = raw_advantages + self.storage.values.squeeze(-1).detach()
        advantages = raw_advantages
        if not self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.storage.rewards = rewards_phi.unsqueeze(-1).clone()
        self.storage.returns = returns.unsqueeze(-1).clone()
        self.storage.advantages = advantages.unsqueeze(-1).clone()

        logs = {"validated/ppo_reward_recomputed": 1.0}
        if self.log_meta_effectiveness:
            logs.update(
                {
                    "validated/ppo_reward_delta_abs_mean": float(
                        (rewards_phi - old_rewards).abs().mean().item()
                    ),
                    "validated/ppo_reward_old_mean": float(old_rewards.mean().item()),
                    "validated/ppo_reward_new_mean": float(rewards_phi.mean().item()),
                    "validated/ppo_adv_delta_abs_mean": float(
                        (advantages - old_advantages).abs().mean().item()
                    ),
                    "validated/ppo_adv_new_mean": float(advantages.mean().item()),
                    "validated/ppo_adv_new_std": float(advantages.std(unbiased=False).item()),
                }
            )
        return logs

    def _linear_channels(self):
        return [
            chan
            for chan in ir.all_channels().values()
            if hasattr(chan, "differentiable_reward_from_buffer")
            and chan.name in self._CHANNEL_TO_REWARD_TERM
            and chan.buffer_len() > 0
        ]

    def _gae_from_rewards(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = rewards.shape[0]
        values_2d = values.squeeze(-1)
        dones_2d = dones.squeeze(-1).float()
        if last_values is None:
            if self._last_values is None:
                last_values_2d = torch.zeros_like(values_2d[-1])
            else:
                last_values_2d = self._last_values.squeeze(-1).detach()
        else:
            last_values_2d = last_values.squeeze(-1).detach()

        adv = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[1], device=rewards.device)
        for t in reversed(range(T)):
            next_values = last_values_2d if t == T - 1 else values_2d[t + 1]
            next_not_done = 1.0 - dones_2d[t]
            delta = rewards[t] + self.gamma * next_values * next_not_done - values_2d[t]
            running = delta + self.gamma * self.lam * next_not_done * running
            adv[t] = running
        return adv

    @staticmethod
    def _stop_stat_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean().detach()
        std = x.std(unbiased=False).detach() + 1e-8
        return (x - mean) / std

    # ------------------------------------------------------------ validation

    def _collect_validation_rollout(
        self,
        obs: TensorDict,
        theta_prime: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], TensorDict]:
        obs_list: list[TensorDict] = []
        actions_list: list[torch.Tensor] = []
        true_rewards: list[torch.Tensor] = []
        dones_list: list[torch.Tensor] = []
        bucket_list: list[torch.Tensor] = []
        cmd_list: list[torch.Tensor] = []
        lin_vel_list: list[torch.Tensor] = []
        yaw_vel_list: list[torch.Tensor] = []
        err_xy_list: list[torch.Tensor] = []
        err_yaw_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        outer_mask_list: list[torch.Tensor] = []

        env_unwrapped = self._env.unwrapped
        saved_common_step_counter = getattr(env_unwrapped, "common_step_counter", None)
        reward_manager = getattr(env_unwrapped, "reward_manager", None)
        reward_sums_snapshot = None
        if reward_manager is not None and hasattr(reward_manager, "_episode_sums"):
            reward_sums_snapshot = {
                key: value.detach().clone()
                for key, value in reward_manager._episode_sums.items()
            }
        validation_reset_mask = torch.zeros(self.storage.num_envs, dtype=torch.bool, device=self.device)
        alive = torch.ones(self.storage.num_envs, dtype=torch.bool, device=self.device)

        ir.set_recording_enabled(False)
        try:
            for t in range(self.validation_steps):
                obs_list.append(obs.detach().clone())
                with torch.no_grad():
                    actions = self._sample_actions(obs, theta_prime).detach()
                    obs_next, _, dones, _ = self._env.step(actions.to(self._env.device))
                    obs_next = obs_next.to(self.device)
                    dones = dones.to(self.device).view(-1)
                    done_bool = dones.bool()
                    outer_valid = alive.clone()
                    step_valid = alive & ~done_bool
                    cmd, lin_vel, yaw_vel, err_xy, err_yaw = self._tracking_state()
                    if self.true_reward_mode == "step":
                        r_true = self._true_tracking_reward(cmd, err_xy, err_yaw)
                    else:
                        r_true = torch.zeros_like(err_xy)
                    remaining = float(self.validation_steps - t - 1)
                    if remaining > 0.0 and self.true_fall_tail_penalty > 0.0:
                        r_true = r_true - self.true_fall_tail_penalty * remaining * done_bool.float()
                    buckets = self._command_buckets(cmd)
                    validation_reset_mask |= done_bool
                    alive &= ~done_bool
                actions_list.append(actions.to(self.device))
                true_rewards.append(r_true.to(self.device))
                dones_list.append(dones)
                bucket_list.append(buckets.to(self.device))
                cmd_list.append(cmd.to(self.device))
                lin_vel_list.append(lin_vel.to(self.device))
                yaw_vel_list.append(yaw_vel.to(self.device))
                err_xy_list.append(err_xy.to(self.device))
                err_yaw_list.append(err_yaw.to(self.device))
                valid_list.append(step_valid.to(self.device))
                outer_mask_list.append(outer_valid.to(self.device))
                obs = obs_next
        finally:
            if saved_common_step_counter is not None:
                env_unwrapped.common_step_counter = saved_common_step_counter
            if reward_sums_snapshot is not None:
                reset_mask = validation_reset_mask.to(next(iter(reward_sums_snapshot.values())).device)
                for key, value in reward_sums_snapshot.items():
                    reward_manager._episode_sums[key].copy_(value)
                    if reset_mask.any():
                        reward_manager._episode_sums[key][reset_mask] = 0.0
            ir.set_recording_enabled(True)

        val_obs = TensorDict(
            {key: torch.stack([td[key] for td in obs_list], dim=0) for key in obs_list[0].keys()},
            batch_size=[self.validation_steps, self.storage.num_envs],
            device=self.device,
        )
        true_rewards_t = torch.stack(true_rewards, dim=0)
        err_xy_t = torch.stack(err_xy_list, dim=0)
        err_yaw_t = torch.stack(err_yaw_list, dim=0)
        valid_t = torch.stack(valid_list, dim=0)
        outer_mask_t = torch.stack(outer_mask_list, dim=0)
        if self.true_reward_mode == "trajectory":
            true_rewards_t, traj_logs = self._trajectory_true_rewards(
                torch.stack(cmd_list, dim=0),
                torch.stack(lin_vel_list, dim=0),
                torch.stack(yaw_vel_list, dim=0),
                torch.stack(dones_list, dim=0),
                valid_t,
            )
            err_xy_t = traj_logs["traj_err_xy"]
            err_yaw_t = traj_logs["traj_err_yaw"]

        val_data = {
            "obs": val_obs,
            "actions": torch.stack(actions_list, dim=0),
            "true_rewards": true_rewards_t,
            "dones": torch.stack(dones_list, dim=0).unsqueeze(-1),
            "buckets": torch.stack(bucket_list, dim=0),
            "err_xy": err_xy_t,
            "err_yaw": err_yaw_t,
            "outer_mask": outer_mask_t if self.true_reward_mode == "trajectory" else torch.ones_like(valid_t),
        }
        return val_data, obs

    def _tracking_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        env_unwrapped = self._env.unwrapped
        asset = env_unwrapped.scene["robot"]
        cmd = env_unwrapped.command_manager.get_command("base_velocity")
        lin_vel_yaw = quat_apply_inverse(
            yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
        )[:, :2]
        yaw_vel = asset.data.root_ang_vel_b[:, 2]
        err_xy = torch.norm(lin_vel_yaw - cmd[:, :2], dim=-1)
        err_yaw = torch.abs(yaw_vel - cmd[:, 2])
        return cmd.detach(), lin_vel_yaw.detach(), yaw_vel.detach(), err_xy.detach(), err_yaw.detach()

    def _true_tracking_reward(
        self,
        cmd: torch.Tensor,
        err_xy: torch.Tensor,
        err_yaw: torch.Tensor,
    ) -> torch.Tensor:
        cmd_xy = torch.norm(cmd[:, :2], dim=-1).clamp(min=self.true_lin_cmd_floor)
        cmd_yaw = torch.abs(cmd[:, 2]).clamp(min=self.true_ang_cmd_floor)
        return -(err_xy + err_xy / cmd_xy + err_yaw + err_yaw / cmd_yaw)

    def _trajectory_true_rewards(
        self,
        cmd: torch.Tensor,
        lin_vel: torch.Tensor,
        yaw_vel: torch.Tensor,
        dones: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = valid_mask.to(lin_vel.dtype)
        counts = valid.sum(dim=0).clamp(min=1.0)
        mean_cmd = (cmd * valid.unsqueeze(-1)).sum(dim=0) / counts.unsqueeze(-1)
        mean_lin = (lin_vel * valid.unsqueeze(-1)).sum(dim=0) / counts.unsqueeze(-1)
        mean_yaw = (yaw_vel * valid).sum(dim=0) / counts

        err_xy = torch.norm(mean_lin - mean_cmd[:, :2], dim=-1)
        err_yaw = torch.abs(mean_yaw - mean_cmd[:, 2])
        cmd_xy = torch.norm(mean_cmd[:, :2], dim=-1).clamp(min=self.true_lin_cmd_floor)
        cmd_yaw = torch.abs(mean_cmd[:, 2]).clamp(min=self.true_ang_cmd_floor)
        lin_metric = err_xy + err_xy / cmd_xy
        yaw_metric = err_yaw + err_yaw / cmd_yaw

        buckets = self._command_buckets(mean_cmd)
        score = -(lin_metric + yaw_metric)
        standing = buckets == 0
        pure_linear = (buckets == 1) | (buckets == 2)
        pure_wz = buckets == 3
        score = torch.where(
            standing,
            -(self.true_standing_lin_abs_weight * err_xy + self.true_standing_yaw_abs_weight * err_yaw),
            score,
        )
        score = torch.where(
            pure_linear,
            -(lin_metric + self.true_pure_linear_yaw_abs_weight * err_yaw),
            score,
        )
        score = torch.where(
            pure_wz,
            -(yaw_metric + self.true_pure_wz_lin_abs_weight * err_xy),
            score,
        )

        rewards = score.unsqueeze(0).expand_as(valid) * valid
        if self.true_fall_tail_penalty > 0.0:
            dones_2d = dones.squeeze(-1).float() if dones.dim() == 3 else dones.float()
            for t in range(rewards.shape[0]):
                remaining = float(rewards.shape[0] - t - 1)
                if remaining > 0.0:
                    rewards[t] = rewards[t] - self.true_fall_tail_penalty * remaining * dones_2d[t]

        return rewards, {
            "traj_err_xy": err_xy.unsqueeze(0).expand_as(valid),
            "traj_err_yaw": err_yaw.unsqueeze(0).expand_as(valid),
        }

    @staticmethod
    def _command_buckets(cmd: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        vx_zero = cmd[:, 0].abs() < eps
        vy_zero = cmd[:, 1].abs() < eps
        wz_zero = cmd[:, 2].abs() < eps
        standing = vx_zero & vy_zero & wz_zero
        pure_vx = (~vx_zero) & vy_zero & wz_zero
        pure_vy = vx_zero & (~vy_zero) & wz_zero
        pure_wz = vx_zero & vy_zero & (~wz_zero)
        bucket = torch.full((cmd.shape[0],), 4, dtype=torch.long, device=cmd.device)
        bucket[standing] = 0
        bucket[pure_vx] = 1
        bucket[pure_vy] = 2
        bucket[pure_wz] = 3
        return bucket

    def _compute_outer_meta_loss(
        self,
        theta_prime: dict[str, torch.Tensor],
        val_data: dict[str, Any],
        train_aux: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        true_adv = self._true_advantages(val_data["true_rewards"], val_data["dones"])
        outer_mask = val_data.get("outer_mask")
        if outer_mask is None:
            outer_mask = torch.ones_like(val_data["buckets"], dtype=torch.bool)
        else:
            outer_mask = outer_mask.bool()
        true_adv = self._bucket_stop_stat_normalize(true_adv, val_data["buckets"], outer_mask)

        flat_obs = val_data["obs"].flatten(0, 1)
        flat_actions = val_data["actions"].flatten(0, 1)
        flat_adv = true_adv.flatten(0, 1).detach()
        flat_buckets = val_data["buckets"].flatten(0, 1)
        flat_mask = outer_mask.flatten(0, 1)
        if flat_mask.any():
            flat_obs = flat_obs[flat_mask]
            flat_actions = flat_actions[flat_mask]
            flat_adv = flat_adv[flat_mask]
            flat_buckets = flat_buckets[flat_mask]
        outer_idx = self._sample_flat_indices(flat_adv.shape[0], self.outer_max_samples, flat_adv.device)
        flat_obs = flat_obs[outer_idx]
        flat_actions = flat_actions[outer_idx]
        flat_adv = flat_adv[outer_idx]
        flat_buckets = flat_buckets[outer_idx]

        logp, _ = self._log_prob_entropy(flat_obs, flat_actions, theta_prime)
        outer_loss = self._bucket_mean(-logp * flat_adv, flat_buckets)

        reg_loss = torch.tensor(0.0, device=self.device)
        logs: dict[str, float] = {}
        for chan, _, _, ch_reg_loss, _, _ in train_aux["channel_tensors"]:
            reg_loss = reg_loss + ch_reg_loss
            logs[f"lirpg_{chan.name}/reg_total"] = float(ch_reg_loss.detach().item())

        meta_loss = outer_loss + reg_loss
        mask_f = outer_mask.float()
        mask_den = mask_f.sum().clamp(min=1.0)
        masked_reward = (val_data["true_rewards"] * mask_f).sum() / mask_den
        masked_err_xy = (val_data["err_xy"] * mask_f).sum() / mask_den
        masked_err_yaw = (val_data["err_yaw"] * mask_f).sum() / mask_den
        masked_adv = true_adv[outer_mask]
        logs.update(
            {
                "validated/outer_loss": float(outer_loss.detach().item()),
                "validated/reg_loss": float(reg_loss.detach().item()),
                "validated/val_true_reward": float(masked_reward.detach().item()),
                "validated/val_true_error_xy": float(masked_err_xy.detach().item()),
                "validated/val_true_error_yaw": float(masked_err_yaw.detach().item()),
                "validated/val_A_true_mean": float(masked_adv.mean().detach().item()) if masked_adv.numel() else 0.0,
                "validated/val_A_true_std": float(masked_adv.std(unbiased=False).detach().item()) if masked_adv.numel() else 0.0,
                "validated/outer_valid_frac": float(mask_f.mean().detach().item()),
                "validated/true_reward_mode_trajectory": 1.0 if self.true_reward_mode == "trajectory" else 0.0,
            }
        )
        return meta_loss, logs

    def _true_advantages(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        T = rewards.shape[0]
        dones_2d = dones.squeeze(-1).float()
        adv = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[1], device=rewards.device)
        for t in reversed(range(T)):
            next_not_done = 1.0 - dones_2d[t]
            running = rewards[t] + self.lirpg_meta_gamma * self.lirpg_meta_lam * running * next_not_done
            adv[t] = running
        return adv

    @staticmethod
    def _bucket_stop_stat_normalize(
        values: torch.Tensor,
        buckets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.zeros_like(values)
        for b in range(5):
            mask = buckets == b
            if valid_mask is not None:
                mask = mask & valid_mask
            if mask.any():
                v = values[mask]
                out[mask] = (v - v.mean().detach()) / (v.std(unbiased=False).detach() + 1e-8)
        return out

    @staticmethod
    def _bucket_mean(values: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        parts = []
        for b in range(5):
            mask = buckets == b
            if mask.any():
                parts.append(values[mask].mean())
        if not parts:
            return values.mean()
        return torch.stack(parts).mean()

    @staticmethod
    def _sample_flat_indices(total: int, max_samples: int, device: torch.device) -> torch.Tensor:
        if max_samples <= 0 or total <= max_samples:
            return torch.arange(total, device=device)
        return torch.randperm(total, device=device)[:max_samples]
