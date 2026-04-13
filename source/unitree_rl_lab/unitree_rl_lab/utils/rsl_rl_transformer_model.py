from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: z = gamma * h + beta."""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.proj(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class TransformerHistoryModel(MLPModel):
    """Transformer model that fuses stacked history with current state.

    Improvements over the original version:
    - Causal mask in the TransformerEncoder (time-step t cannot attend to t+1).
    - FiLM conditioning replaces the degenerate single-token cross-attention.
    - Optional next-observation prediction head for auxiliary self-supervised loss.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_len: int = 5,
        history_start_idx: int = 0,
        history_obs_dim: int | None = None,
        aux_start_idx: int | None = None,
        aux_obs_dim: int | None = None,
        d_model: int = 256,
        n_heads: int = 4,
        encoder_num_layers: int = 2,
        encoder_dim_feedforward: int = 512,
        enable_aux_loss: bool = False,
    ) -> None:
        self.latent_dim = d_model
        self.history_len = history_len
        self.history_start_idx = history_start_idx

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        if history_obs_dim is None:
            if self.obs_dim % history_len != 0:
                raise ValueError(
                    f"obs_dim={self.obs_dim} cannot be evenly split by history_len={history_len}. "
                    "Please provide history_obs_dim explicitly."
                )
            history_obs_dim = self.obs_dim // history_len

        self.history_obs_dim = history_obs_dim
        self.history_total_dim = self.history_len * self.history_obs_dim

        if aux_start_idx is None:
            aux_start_idx = history_start_idx + self.history_total_dim - self.history_obs_dim
        if aux_obs_dim is None:
            aux_obs_dim = self.history_obs_dim

        self.aux_start_idx = aux_start_idx
        self.aux_obs_dim = aux_obs_dim

        self._validate_slices(self.obs_dim)

        # --- Transformer encoder ---
        self.hist_proj = nn.Linear(self.history_obs_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.history_len, d_model) * 0.1)

        # Causal mask: prevents time-step t from attending to future steps.
        causal_mask = torch.triu(
            torch.ones(self.history_len, self.history_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=encoder_dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)

        # --- FiLM conditioning (replaces cross-attention) ---
        self.aux_proj = nn.Linear(self.aux_obs_dim, d_model)
        self.film = FiLMLayer(cond_dim=d_model, feat_dim=d_model)
        self.ln_fusion = nn.LayerNorm(d_model)

        # --- Optional next-obs prediction head ---
        self.enable_aux_loss = enable_aux_loss
        if enable_aux_loss:
            self.next_obs_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.Linear(d_model, self.history_obs_dim),
            )

    def _validate_slices(self, obs_dim: int) -> None:
        history_end = self.history_start_idx + self.history_total_dim
        aux_end = self.aux_start_idx + self.aux_obs_dim
        if self.history_start_idx < 0 or self.aux_start_idx < 0:
            raise ValueError("history_start_idx and aux_start_idx must be non-negative.")
        if history_end > obs_dim or aux_end > obs_dim:
            raise ValueError(
                f"Configured slices exceed obs dim ({obs_dim}). "
                f"history_end={history_end}, aux_end={aux_end}."
            )

    def _split_from_flat(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_flat = flat_obs[:, self.history_start_idx : self.history_start_idx + self.history_total_dim]
        history = history_flat.view(-1, self.history_len, self.history_obs_dim)
        aux = flat_obs[:, self.aux_start_idx : self.aux_start_idx + self.aux_obs_dim]
        return history, aux

    def _encode_history_and_aux(self, history: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h_emb = self.hist_proj(history) + self.pos_emb
        h_encoded = self.hist_encoder(h_emb, mask=self.causal_mask)

        last_token = h_encoded[:, -1, :]  # (B, d_model)
        aux_emb = self.aux_proj(aux)  # (B, d_model)
        fused_state = self.ln_fusion(self.film(last_token, aux_emb))
        return fused_state

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)

        history, aux = self._split_from_flat(flat_obs)
        return self._encode_history_and_aux(history, aux)

    def predict_next_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (latent, predicted_next_obs_frame).

        Only meaningful when ``enable_aux_loss=True``; raises otherwise.
        """
        if not self.enable_aux_loss:
            raise RuntimeError("predict_next_obs requires enable_aux_loss=True")

        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)

        history, aux = self._split_from_flat(flat_obs)
        latent = self._encode_history_and_aux(history, aux)
        predicted = self.next_obs_head(latent)
        return latent, predicted

    def as_jit(self) -> nn.Module:
        return _TorchTransformerHistoryModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxTransformerHistoryModel(self, verbose)

    def _get_latent_dim(self) -> int:
        return self.latent_dim


class _TorchTransformerHistoryModel(nn.Module):
    """Exportable transformer model for TorchScript."""

    def __init__(self, model: TransformerHistoryModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)

        self.history_len = model.history_len
        self.history_start_idx = model.history_start_idx
        self.history_obs_dim = model.history_obs_dim
        self.history_total_dim = model.history_total_dim
        self.aux_start_idx = model.aux_start_idx
        self.aux_obs_dim = model.aux_obs_dim

        self.hist_proj = copy.deepcopy(model.hist_proj)
        self.pos_emb = copy.deepcopy(model.pos_emb)
        self.register_buffer("causal_mask", model.causal_mask.clone())
        self.hist_encoder = copy.deepcopy(model.hist_encoder)
        self.aux_proj = copy.deepcopy(model.aux_proj)
        self.film = copy.deepcopy(model.film)
        self.ln_fusion = copy.deepcopy(model.ln_fusion)
        self.mlp = copy.deepcopy(model.mlp)

        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def _split_from_flat(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_flat = flat_obs[:, self.history_start_idx : self.history_start_idx + self.history_total_dim]
        history = history_flat.view(-1, self.history_len, self.history_obs_dim)
        aux = flat_obs[:, self.aux_start_idx : self.aux_start_idx + self.aux_obs_dim]
        return history, aux

    def _encode_history_and_aux(self, history: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h_emb = self.hist_proj(history) + self.pos_emb
        h_encoded = self.hist_encoder(h_emb, mask=self.causal_mask)

        last_token = h_encoded[:, -1, :]
        aux_emb = self.aux_proj(aux)
        fused_state = self.ln_fusion(self.film(last_token, aux_emb))
        return fused_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        history, aux = self._split_from_flat(x)
        latent = self._encode_history_and_aux(history, aux)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxTransformerHistoryModel(nn.Module):
    """Exportable transformer model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: TransformerHistoryModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)

        self.history_len = model.history_len
        self.history_start_idx = model.history_start_idx
        self.history_obs_dim = model.history_obs_dim
        self.history_total_dim = model.history_total_dim
        self.aux_start_idx = model.aux_start_idx
        self.aux_obs_dim = model.aux_obs_dim

        self.hist_proj = copy.deepcopy(model.hist_proj)
        self.pos_emb = copy.deepcopy(model.pos_emb)
        self.register_buffer("causal_mask", model.causal_mask.clone())
        self.hist_encoder = copy.deepcopy(model.hist_encoder)
        self.aux_proj = copy.deepcopy(model.aux_proj)
        self.film = copy.deepcopy(model.film)
        self.ln_fusion = copy.deepcopy(model.ln_fusion)
        self.mlp = copy.deepcopy(model.mlp)

        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.input_size = model.obs_dim

    def _split_from_flat(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_flat = flat_obs[:, self.history_start_idx : self.history_start_idx + self.history_total_dim]
        history = history_flat.view(-1, self.history_len, self.history_obs_dim)
        aux = flat_obs[:, self.aux_start_idx : self.aux_start_idx + self.aux_obs_dim]
        return history, aux

    def _encode_history_and_aux(self, history: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h_emb = self.hist_proj(history) + self.pos_emb
        h_encoded = self.hist_encoder(h_emb, mask=self.causal_mask)

        last_token = h_encoded[:, -1, :]
        aux_emb = self.aux_proj(aux)
        fused_state = self.ln_fusion(self.film(last_token, aux_emb))
        return fused_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        history, aux = self._split_from_flat(x)
        latent = self._encode_history_and_aux(history, aux)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
