"""PPO with Lipschitz-Constrained Policy (LCP) gradient penalty.

Reference:
    Chen et al. "Learning Smooth Humanoid Locomotion through
    Lipschitz-Constrained Policies." IROS 2025.
    arXiv: 2410.11825

The Lipschitz constraint is implemented as a gradient penalty on the actor:

    L_GP = lcp_coef * E[ || d log pi(a | o) / d o ||_2^2 ]

where (o, a) are state-action pairs sampled from policy rollouts. This matches
the formulation in Chen et al., where the Lipschitz constraint is applied to
the policy log-probability rather than directly to the deterministic action
mean. Penalizing the squared gradient norm encourages the policy to vary
smoothly with respect to its input observations.
"""

from __future__ import annotations

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


def _freeze_actor_std_parameters(actor: nn.Module) -> list[str]:
    """Freeze actor distribution std parameters when using LCP.

    The paper's public implementation uses fixed action standard deviations for
    LCP training. In this codebase, the actor distribution std is otherwise
    learnable, which allows the policy to reduce the gradient penalty by simply
    inflating the variance. Freezing std removes this degenerate solution.
    """
    frozen_names: list[str] = []
    candidate_suffixes = ("std", "log_std", "std_param", "log_std_param")

    for name, param in actor.named_parameters():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name in candidate_suffixes and param.requires_grad:
            param.requires_grad_(False)
            frozen_names.append(name)

    return frozen_names


class LCPPPO(PPO):
    """PPO with an additional Lipschitz gradient penalty on the actor."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        lcp_coef: float = 0.002,
        lcp_coef_schedule: list[float] | tuple[float, float, int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.lcp_coef = lcp_coef
        self.lcp_coef_schedule = list(lcp_coef_schedule) if lcp_coef_schedule is not None else None
        self.counter = 0

    def _get_lcp_coef(self) -> float:
        """Return the current LCP coefficient, optionally using a linear schedule."""
        if self.lcp_coef_schedule is None:
            return self.lcp_coef

        if len(self.lcp_coef_schedule) != 4:
            raise ValueError("lcp_coef_schedule must be [start, end, warmup_start, warmup_steps].")

        start, end, warmup_start, warmup_steps = self.lcp_coef_schedule
        if warmup_steps <= 0:
            return float(end)

        progress = min(max((self.counter - warmup_start), 0) / warmup_steps, 1.0)
        return float(start + progress * (end - start))

    def update(self) -> dict[str, float]:
        """PPO update with the LCP gradient penalty added to the loss."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_lcp_loss = 0.0
        lcp_coef = self._get_lcp_coef()

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

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

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

            lcp_loss = self._compute_gradient_penalty(
                batch.observations,
                batch.actions,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + lcp_coef * lcp_loss
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
            mean_lcp_loss += lcp_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_lcp_loss /= num_updates
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
            "lcp": mean_lcp_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def _compute_gradient_penalty(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        *,
        masks: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute E[||d log pi(a | o) / d o||_2^2] on rollout state-action pairs."""
        obs_tensors = [obs[key].reshape(obs[key].shape[0], -1) for key in sorted(obs.keys())]
        obs_flat = torch.cat(obs_tensors, dim=-1)
        obs_input = obs_flat.detach().clone().requires_grad_(True)

        rebuilt_obs = TensorDict({}, batch_size=obs.batch_size)
        offset = 0
        for key in sorted(obs.keys()):
            tensor = obs[key]
            numel = tensor.reshape(tensor.shape[0], -1).shape[-1]
            rebuilt_obs[key] = obs_input[:, offset : offset + numel].reshape(tensor.shape)
            offset += numel

        self.actor(
            rebuilt_obs,
            masks=masks,
            hidden_state=hidden_state,
            stochastic_output=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(actions)
        grad_log_prob = torch.autograd.grad(
            outputs=actions_log_prob.sum(),
            inputs=obs_input,
            create_graph=True,
        )[0]
        return grad_log_prob.square().sum(dim=-1).mean()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the LCP-PPO algorithm (mirrors UnitreePPO.construct_algorithm)."""
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))
        freeze_actor_std = cfg["algorithm"].pop("freeze_actor_std", False)

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
        if freeze_actor_std:
            frozen_names = _freeze_actor_std_parameters(actor)
            print(f"Frozen actor std params: {frozen_names or 'none found'}")
        print(f"Actor Model: {actor}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns

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
