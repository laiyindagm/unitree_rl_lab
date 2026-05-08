# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlLcpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with the LCP gradient-penalty settings."""

    lcp_coef: float = 0.002
    lcp_coef_schedule: list[float] | None = None
    freeze_actor_std: bool = True


@configclass
class RslRlTransformerModelCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.utils.rsl_rl_transformer_model:TransformerHistoryModel"
    history_len: int = MISSING
    history_start_idx: int = 0
    history_obs_dim: int = MISSING

    aux_start_idx: int = MISSING
    aux_obs_dim: int = MISSING

    d_model: int = 256
    n_heads: int = 4
    encoder_num_layers: int = 2
    encoder_dim_feedforward: int = 512


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = ""
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.rsl_rl_custom_ppo:UnitreePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1TransformerPPORunnerCfg(BasePPORunnerCfg):
    actor = RslRlTransformerModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        history_len=5,
        history_start_idx=0,
        history_obs_dim=96,
        aux_start_idx=384,
        aux_obs_dim=96,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
    )

    critic = RslRlTransformerModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
        history_len=5,
        history_start_idx=0,
        history_obs_dim=99,
        aux_start_idx=396,
        aux_obs_dim=99,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
    )


@configclass
class G115DofLCPPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner with Lipschitz-Constrained Policy gradient penalty for G1 15-DOF."""

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlLcpPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.lcp_ppo:LCPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        lcp_coef=0.002,
        lcp_coef_schedule=[0.002, 0.002, 700, 1000],
        freeze_actor_std=True,
    )


@configclass
class G115DofTransformerPPORunnerCfg(BasePPORunnerCfg):
    actor = RslRlTransformerModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        history_len=5,
        history_start_idx=0,
        history_obs_dim=54,
        aux_start_idx=216,
        aux_obs_dim=54,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
    )

    critic = RslRlTransformerModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
        history_len=5,
        history_start_idx=0,
        history_obs_dim=57,
        aux_start_idx=228,
        aux_obs_dim=57,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
    )


@configclass
class G115DofLCPPPORunnerV3Cfg(G115DofLCPPPORunnerCfg):
    """LCP runner with action clipping for NaN safety."""

    clip_actions = 100.0


@configclass
class BasePPORunnerV3Cfg(BasePPORunnerCfg):
    """Base runner with action clipping for NaN safety."""

    clip_actions = 100.0


from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
from unitree_rl_lab.tasks.locomotion.mdp import symmetry_g1_15dof



@configclass
class BasePPORunnerV4Cfg(BasePPORunnerCfg):
    """BasePPO with NaN safety (clip_actions) + left-right symmetry augmentation."""

    clip_actions = 100.0

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.rsl_rl_custom_ppo:UnitreePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetry_g1_15dof.compute_symmetric_states,
        ),
    )



@configclass
class BasePPORunnerV5bCfg(BasePPORunnerCfg):
    """BasePPO with clip_actions + symmetry MIRROR LOSS (not data augmentation)."""

    clip_actions = 100.0

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.rsl_rl_custom_ppo:UnitreePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func=symmetry_g1_15dof.compute_symmetric_states,
        ),
    )

@configclass
class G115DofLCPPPORunnerV3dCfg(G115DofLCPPPORunnerCfg):
    """LCP runner with action clipping + left-right symmetry data augmentation."""

    clip_actions = 100.0

    algorithm = RslRlLcpPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.lcp_ppo:LCPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        lcp_coef=0.002,
        lcp_coef_schedule=[0.002, 0.002, 700, 1000],
        freeze_actor_std=True,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetry_g1_15dof.compute_symmetric_states,
        ),
    )


# ---------------------------------------------------------------------------
# Transformer actor + MLP critic with auxiliary next-obs prediction loss
# ---------------------------------------------------------------------------


@configclass
class RslRlTransformerAuxPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config with auxiliary prediction loss settings."""

    aux_loss_coef: float = 0.5
    aux_loss_schedule: list[float] | None = None  # [start, end, decay_start, decay_steps]
    velocity_aux_coef: float = 0.0  # explicit MSE on actor velocity_head; 0 = disabled
    achieved_ang_vel_scale: float = 0.2  # legacy: forwarded to TransformerPPO target extractor


@configclass
class RslRlTransformerAuxModelCfg(RslRlTransformerModelCfg):
    """Transformer model config with auxiliary prediction enabled."""

    enable_aux_loss: bool = True


@configclass
class G115DofTransformerAuxPPORunnerCfg(BasePPORunnerCfg):
    """Transformer actor (with aux prediction) + MLP critic for 15-DOF.

    Improvements over G115DofTransformerPPORunnerCfg:
    - Causal mask in Transformer encoder.
    - FiLM conditioning instead of degenerate cross-attention.
    - Next-obs prediction auxiliary loss with linear decay schedule.
    - Critic stays as MLP (no Transformer overhead for value estimation).
    """

    clip_actions = 100.0

    actor = RslRlTransformerAuxModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        history_len=5,
        history_start_idx=0,
        history_obs_dim=54,
        aux_start_idx=216,
        aux_obs_dim=54,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
        enable_aux_loss=True,
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlTransformerAuxPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.transformer_ppo:TransformerPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_loss_coef=0.5,
        aux_loss_schedule=[0.5, 0.05, 0, 5000],
    )


@configclass
class RslRlTransformerLatentModelCfg(RslRlTransformerModelCfg):
    class_name: str = "unitree_rl_lab.utils.rsl_rl_transformer_model:TransformerLatentModel"
    velocity_pred_dim: int = 3
    enable_aux_loss: bool = False
    detach_velocity_pred: bool = False  # if True, policy MLP cannot back-prop into velocity head


@configclass
class G115DofV21dTransformerLatentPPORunnerCfg(BasePPORunnerV3Cfg):
    actor = RslRlTransformerLatentModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        history_len=5,
        history_start_idx=0,
        history_obs_dim=54,
        aux_start_idx=216,
        aux_obs_dim=54,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
        velocity_pred_dim=3,
        enable_aux_loss=False,
    )


# ---- V21e: corrected actor-side velocity estimator ----
#
# Differences vs the broken V21d setup:
#   1. Algorithm class is `VelocityEstimatorPPO` (subclass of TransformerPPO)
#      with corrected achieved-velocity target extraction. V21d used the
#      default `UnitreePPO`, so its velocity head was never supervised.
#   2. `velocity_aux_coef=1.0` enables the supervised regression of v_hat
#      against the most-recent (base_lin_vel.x, .y, base_ang_vel.z) sample
#      in the critic flat observation. The indices below assume the V21c
#      CriticCfg term order: base_lin_vel(3) | base_ang_vel(3, scale=0.2) | ...
#      with history_length=5 and term-major concatenation, so:
#          base_lin_vel last-frame x = idx 12, y = idx 13
#          base_ang_vel last-frame z = idx 29 (still scaled by 0.2)
#      Inverse scale 1/0.2 = 5.0 recovers raw wz.
#   3. `detach_velocity_pred=True` keeps the policy MLP from back-propagating
#      into the velocity estimator, so v_hat is shaped purely by the
#      supervised auxiliary loss and the actor uses it as a stationary
#      learned proprioception feature.
#   4. `history_obs_dim` is left to auto-compute (=58 for V21c policy obs);
#      V21d hard-coded 54 which silently dropped a slice of the history.
@configclass
class RslRlVelocityEstimatorPpoAlgorithmCfg(RslRlTransformerAuxPpoAlgorithmCfg):
    """Algorithm config for VelocityEstimatorPPO.

    Fields beyond the parent class:
        velocity_target_indices: indices into critic flat-obs for (vx, vy, wz).
        velocity_target_scales:  per-component multiplier to undo CriticCfg
                                 scaling (e.g. 1/0.2 = 5.0 for base_ang_vel.z).
    """

    velocity_target_indices: list[int] = [12, 13, 29]
    velocity_target_scales: list[float] = [1.0, 1.0, 5.0]


@configclass
class G115DofV21eVelocityEstimatorPPORunnerCfg(BasePPORunnerV3Cfg):
    """V21e: V21c env + actor-side detached velocity estimator with supervised aux loss."""

    actor = RslRlTransformerLatentModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        history_len=5,
        history_start_idx=0,
        # V21c policy obs single-frame dim = 3+3+3+1+3+15+15+15 = 58
        history_obs_dim=58,
        aux_start_idx=232,   # = history_start_idx + (history_len - 1) * history_obs_dim
        aux_obs_dim=58,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
        velocity_pred_dim=3,
        enable_aux_loss=False,
        detach_velocity_pred=True,
    )

    algorithm = RslRlVelocityEstimatorPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.velocity_estimator_ppo:VelocityEstimatorPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_loss_coef=0.0,            # disable next-obs prediction
        aux_loss_schedule=None,
        velocity_aux_coef=1.0,        # supervised SmoothL1 on v_hat
        achieved_ang_vel_scale=0.2,   # legacy parent kw, unused by VelocityEstimatorPPO targets
        velocity_target_indices=[12, 13, 29],
        velocity_target_scales=[1.0, 1.0, 5.0],
    )


@configclass
class G115DofV21nFiveModeVelocityEstimatorPPORunnerCfg(BasePPORunnerV3Cfg):
    """V21n: V21f2-derived 5-mode token env with matching policy obs slices."""

    actor = RslRlTransformerLatentModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        history_len=5,
        history_start_idx=0,
        # V21n policy obs single-frame dim =
        # 3+3+3+2+5+15+15+15 = 61, flat history dim = 305.
        history_obs_dim=61,
        aux_start_idx=244,   # = history_start_idx + (history_len - 1) * history_obs_dim
        aux_obs_dim=61,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
        velocity_pred_dim=3,
        enable_aux_loss=False,
        detach_velocity_pred=True,
    )

    algorithm = RslRlVelocityEstimatorPpoAlgorithmCfg(
        class_name="unitree_rl_lab.utils.velocity_estimator_ppo:VelocityEstimatorPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_loss_coef=0.0,
        aux_loss_schedule=None,
        velocity_aux_coef=1.0,
        achieved_ang_vel_scale=0.2,
        # Critic starts with base_lin_vel(3), base_ang_vel(3, scale=0.2).
        # Later token-size changes do not move these term-major last-frame indices.
        velocity_target_indices=[12, 13, 29],
        velocity_target_scales=[1.0, 1.0, 5.0],
    )



# ---- Contrastive Latent Policy (CLP) ----

@configclass
class RslRlContrastiveModelCfg(RslRlMLPModelCfg):
    """ContrastiveLatentModel configuration."""

    class_name: str = "unitree_rl_lab.utils.contrastive_latent_model:ContrastiveLatentModel"
    encoder_type: str = "tcn"        # "tcn" | "transformer"
    history_len: int = 10            # extended history (was 5)
    history_obs_dim: int = 51        # single frame without cmd
    cmd_dim: int = 3
    cmd_start_idx: int = 6           # cmd position in single-frame obs
    enc_dim: int = 96                # encoder output dim
    sphere_dim: int = 32             # per-sphere dim
    num_spheres: int = 3
    # TCN parameters
    tcn_channels: list[int] | None = None  # default [64, 96, 96] (3-layer)
    tcn_kernel_size: int = 3
    tcn_dilations: list[int] | None = None  # default [1, 2, 4] → RF=15 covers history_len=10
    # Transformer parameters (defaults tuned for history_len=10)
    tf_d_model: int = 192
    tf_n_heads: int = 4
    tf_num_layers: int = 3
    tf_dim_feedforward: int = 384
    # Generator parameters
    pred_horizon: int = 3
    pred_obs_dim: int = 36           # physical state only: ang_vel(3)+gravity(3)+joint_pos(15)+joint_vel(15)
    num_actions: int = 15


@configclass
class RslRlContrastivePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """ContrastivePPO algorithm configuration."""

    class_name: str = "unitree_rl_lab.utils.contrastive_ppo:ContrastivePPO"
    nce_coef: float = 0.1            # alpha — InfoNCE coefficient (cmd-bin SupCon)
    time_nce_coef: float = 0.0       # alpha_t — time-shifted positive InfoNCE (SimCLR)
    time_shift: int = 4              # positive pair offset in env steps (~0.08s @ 50Hz)
    use_achieved_labels: bool = False  # X: label SupCon by achieved velocity (critic obs) instead of cmd
    achieved_obs_key: str = "critic"   # X: obs group containing base_lin_vel + base_ang_vel
    achieved_ang_vel_scale: float = 0.2  # X: undo critic-obs static scale on base_ang_vel
    gen_coef: float = 0.5            # beta start — sequence prediction coefficient
    gen_coef_end: float = 0.1        # beta end
    gen_decay_iters: int = 10000     # beta decay iterations
    tau_init: float = 0.5            # temperature initial value
    learnable_tau: bool = True
    repr_lr: float = 1e-4            # representation learning rate
    warmup_iters: int = 0            # disabled — optimizer separation makes warmup redundant
    gate_open_iters: int = 1000      # gate schedule: gate = min(counter/gate_open_iters, 1.0)
    pred_gamma: float = 0.9          # temporal decay for sequence prediction
    # Velocity quantization levels (aligned with V19d MarginalVelocityCommand bins)
    vx_levels: list[float] | None = None
    vy_levels: list[float] | None = None
    wz_levels: list[float] | None = None


@configclass
class G115DofContrastiveTCNPPORunnerCfg(BasePPORunnerV3Cfg):
    """G1 15-DOF CLP Runner with TCN encoder."""

    actor = RslRlContrastiveModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        encoder_type="tcn",
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlContrastivePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        nce_coef=0.1,            # X: restored — supcon target now achievable (achieved-vel labels)
        time_nce_coef=0.0,       # disabled (E3 found shared-head conflict with SupCon)
        time_shift=4,            # B (unused while time_nce_coef=0)
        use_achieved_labels=True,  # X: label by achieved velocity (critic obs base_lin_vel + base_ang_vel_z)
        gen_coef=0.3,            # was 0.1 — keep self-supervised gen signal alive longer
        gen_coef_end=0.2,        # was 0.02 — hold high; the other working repr signal
        tau_init=0.1,            # was 0.5 — SupCon canonical
        learnable_tau=False,     # was True — kill temperature arbitrage feedback
    )


@configclass
class G115DofContrastiveTransformerPPORunnerCfg(BasePPORunnerV3Cfg):
    """G1 15-DOF CLP Runner with Transformer encoder."""

    actor = RslRlContrastiveModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        encoder_type="transformer",
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlContrastivePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        nce_coef=0.1,
        time_nce_coef=0.0,
        time_shift=4,
        use_achieved_labels=True,
        gen_coef=0.3,
        gen_coef_end=0.2,
        tau_init=0.1,
        learnable_tau=False,
    )


# ---- V22a: V21g + frozen V3 segment encoder z_gait into actor ----
@configclass
class RslRlTransformerLatentGaitModelCfg(RslRlTransformerLatentModelCfg):
    """Actor cfg for TransformerLatentGaitModel (extra gait_dim slot)."""

    class_name: str = (
        "unitree_rl_lab.utils.frozen_segment_encoder:TransformerLatentGaitModel"
    )
    gait_dim: int = 32


@configclass
class RslRlSegmentEncoderPpoAlgorithmCfg(RslRlVelocityEstimatorPpoAlgorithmCfg):
    """Algorithm cfg for SegmentEncoderVelocityEstimatorPPO (V22a)."""

    class_name: str = (
        "unitree_rl_lab.utils.segment_encoder_ppo:SegmentEncoderVelocityEstimatorPPO"
    )
    encoder_path: str = (
        "/root/workspace/unitree_rl_lab/logs/frnc_seg_v3/v3_full/encoder.pt"
    )
    z_buffer_len: int = 32
    gait_dim: int = 32
    actor_obs_key: str = "policy"


@configclass
class G115DofV22aSegmentEncoderPPORunnerCfg(G115DofV21eVelocityEstimatorPPORunnerCfg):
    """V22a: V21g env + V21e actor extended with frozen V3 z_gait."""

    actor = RslRlTransformerLatentGaitModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        history_len=5,
        history_start_idx=0,
        history_obs_dim=58,
        aux_start_idx=232,
        aux_obs_dim=58,
        d_model=256,
        n_heads=4,
        encoder_num_layers=2,
        encoder_dim_feedforward=512,
        velocity_pred_dim=3,
        enable_aux_loss=False,
        detach_velocity_pred=True,
        gait_dim=32,
    )

    algorithm = RslRlSegmentEncoderPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_loss_coef=0.0,
        aux_loss_schedule=None,
        velocity_aux_coef=1.0,
        achieved_ang_vel_scale=0.2,
        velocity_target_indices=[12, 13, 29],
        velocity_target_scales=[1.0, 1.0, 5.0],
        encoder_path="/root/workspace/unitree_rl_lab/logs/frnc_seg_v3/v3_full/encoder.pt",
        z_buffer_len=32,
        gait_dim=32,
        actor_obs_key="policy",
    )


# ---- V21l: LIRPG (learnable intrinsic tracking reward) ----

@configclass
class G115DofV21lLirpgRunnerCfg(G115DofV21eVelocityEstimatorPPORunnerCfg):
    """V21l: V21e runner with LIRPG PPO subclass.

    The intrinsic reward MLPs live in the global registry of
    `unitree_rl_lab.utils.intrinsic_reward` and are created lazily by the
    `track_*_intrinsic` reward terms in the V21l env cfg. The LIRPG PPO
    subclass adds a per-rollout meta-gradient step on each MLP.
    """

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self.algorithm.class_name = (
            "unitree_rl_lab.utils.lirpg_ppo:LirpgVelocityEstimatorPPO"
        )



class G115DofV21mLirpgRunnerCfg(G115DofV21lLirpgRunnerCfg):
    """V21m: same LIRPG PPO runner as V21l; Gaussian sigma-kernel channels
    are registered under 'lin_xy_gauss'/'ang_z_gauss' and have the same
    meta_update / record_dones interface."""
    pass

# ---- V22b: V22a + metric-residual CIC intrinsic + SMERL gate ----
@configclass
class RslRlSegmentEncoderCICPpoAlgorithmCfg(RslRlSegmentEncoderPpoAlgorithmCfg):
    """Algorithm cfg for SegmentEncoderCICPPO (V22b)."""

    class_name: str = (
        "unitree_rl_lab.utils.segment_encoder_cic_ppo:SegmentEncoderCICPPO"
    )
    cic_alpha_max: float = 0.02
    cic_warmup_start_iter: int = 200
    cic_warmup_end_iter: int = 2000
    smerl_threshold: float = 0.045
    smerl_kappa: float = 200.0
    track_ema_decay: float = 0.99
    cmd_norm_eps: float = 0.1
    cmd_obs_indices: list[int] = [42, 43, 44]
    intrinsic_log_every: int = 200


@configclass
class G115DofV22bSegmentEncoderCICPPORunnerCfg(G115DofV22aSegmentEncoderPPORunnerCfg):
    """V22b: V22a + metric-residual intrinsic reward."""

    algorithm = RslRlSegmentEncoderCICPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_loss_coef=0.0,
        aux_loss_schedule=None,
        velocity_aux_coef=1.0,
        achieved_ang_vel_scale=0.2,
        velocity_target_indices=[12, 13, 29],
        velocity_target_scales=[1.0, 1.0, 5.0],
        encoder_path="/root/workspace/unitree_rl_lab/logs/frnc_seg_v3/v3_full/encoder.pt",
        z_buffer_len=32,
        gait_dim=32,
        actor_obs_key="policy",
        cic_alpha_max=0.02,
        cic_warmup_start_iter=200,
        cic_warmup_end_iter=2000,
        smerl_threshold=0.045,
        smerl_kappa=200.0,
        track_ema_decay=0.99,
        cmd_norm_eps=0.1,
        cmd_obs_indices=[42, 43, 44],
        intrinsic_log_every=200,
    )
