"""Frozen V3 segment encoder + actor model that consumes z_gait.

Two pieces:
  * `FrozenSegmentEncoder`: loads a V3 SegmentEncoderV3 ckpt, eval+frozen,
    exposes `encode(buf)` taking a per-env rolling buffer of policy obs.
  * `TransformerLatentGaitModel`: drop-in replacement for `TransformerLatentModel`
    that additionally concatenates `obs["z_gait"]` into the policy latent.

The encoder is fed via the algorithm-side rolling buffer; this model only reads
the precomputed z_gait from the obs TensorDict.
"""
from __future__ import annotations
import os
import sys
import copy
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from unitree_rl_lab.utils.rsl_rl_transformer_model import TransformerLatentModel


# ---- V3 encoder loading: import SegmentEncoderV3 from scripts/rsl_rl ----
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_SCRIPT_DIR = os.path.join(_REPO_ROOT, "scripts", "rsl_rl")


def _import_v3_encoder():
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from frnc_segment_pretrain_v3 import SegmentEncoderV3
    return SegmentEncoderV3


def _parse_index_list(spec: str) -> list[int]:
    out: list[int] = []
    if not spec:
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            a, b = token.split(":")
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(token))
    return out


class FrozenSegmentEncoder(nn.Module):
    """Wraps V3 SegmentEncoderV3 in eval+frozen mode for online z_gait extraction."""

    def __init__(self, ckpt_path: str, device: torch.device | str = "cpu") -> None:
        super().__init__()
        SegmentEncoderV3 = _import_v3_encoder()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]

        self.frame_dim: int = int(cfg["in_dim"])
        self.z_dim: int = int(cfg["d_gait"])
        self.mask_kind: str = cfg.get("mask_kind", "strict")
        mask_spec: str = cfg.get("mask_spec", "") or ""
        mask_list = _parse_index_list(mask_spec)
        if mask_list:
            self.register_buffer(
                "mask_idx",
                torch.as_tensor(mask_list, dtype=torch.long),
                persistent=False,
            )
        else:
            self.mask_idx = None  # type: ignore[assignment]

        model = SegmentEncoderV3(
            in_dim=self.frame_dim,
            d_back=cfg["d_back"], d_gait=self.z_dim,
            phase_dim=cfg["phase_dim"], foot_dim=cfg.get("foot_dim", 2),
            prop_dim=cfg.get("prop_dim", 0),
            hard_seg_mean=bool(cfg.get("hard_seg_mean", 1)),
        )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        self.model = model
        # cmd-axis std used during pretraining (for axial_predict)
        sigma_cmd = cfg.get("sigma_cmd", [1.0, 1.0, 1.0])
        self.register_buffer(
            "sigma_cmd",
            torch.as_tensor(sigma_cmd, dtype=torch.float32),
            persistent=False,
        )
        self.to(device)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Force frozen encoder to remain in eval mode regardless of parent calls.
        return super().train(False)

    @torch.no_grad()
    def axial_predict(self, cmd: torch.Tensor) -> torch.Tensor:
        """g*(v) = sum_a rho_a * E_a, rho_a = v_a / ||v||_W with W=diag(1/sigma^2).

        cmd: (B, 3). Returns (B, z_dim). When ||v||_W < eps, returns zeros.
        """
        return self.model.axial_predict(cmd, self.sigma_cmd)

    @torch.no_grad()
    def encode(self, buf: torch.Tensor) -> torch.Tensor:
        """buf: (B, T, frame_dim). Returns z_gait: (B, z_dim)."""
        if buf.dim() != 3 or buf.shape[-1] != self.frame_dim:
            raise ValueError(
                f"buf shape {tuple(buf.shape)} incompatible with frame_dim={self.frame_dim}"
            )
        x = buf
        if self.mask_idx is not None:
            x = x.clone()
            x[..., self.mask_idx] = 0.0
        B, T, D = x.shape
        f = self.model.backbone(x.reshape(B * T, D)).reshape(B, T, -1)
        e = self.model.gait_proj(f)         # (B, T, z_dim)
        return e.mean(dim=1)                 # (B, z_dim)


class TransformerLatentGaitModel(TransformerLatentModel):
    """V21e-style transformer latent model that also concatenates obs['z_gait']."""

    def __init__(self, *args, gait_dim: int = 32, **kwargs) -> None:
        # Set gait_dim BEFORE super().__init__ so latent_dim is sized correctly.
        self.gait_dim = int(gait_dim)
        super().__init__(*args, **kwargs)
        # Parent computed self.latent_dim = flat_obs_dim + velocity_pred_dim.
        # We rebuild latent_dim with the extra z_gait slot. Parent already used
        # the (smaller) latent_dim to build the policy MLP, so we must rebuild.
        self.latent_dim = self.flat_obs_dim + self.velocity_pred_dim + self.gait_dim
        # Rebuild the MLP head (parent created `self.mlp` with the wrong input dim).
        self._rebuild_mlp_after_init()

    def _rebuild_mlp_after_init(self) -> None:
        """Rebuild `self.mlp` so the first Linear matches the new (with-gait) latent_dim."""
        old_mlp = self.mlp
        linear_layers = [m for m in old_mlp if isinstance(m, nn.Linear)]
        hidden_dims = [lin.out_features for lin in linear_layers[:-1]]
        out_dim = linear_layers[-1].out_features
        non_lin = [m for m in old_mlp if not isinstance(m, nn.Linear)]
        act_module_cls = type(non_lin[0]) if non_lin else nn.ELU
        layers: list[nn.Module] = []
        prev = self.latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_module_cls())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        # Re-apply distribution-specific weight init (parent did this before rebuild).
        if getattr(self, "distribution", None) is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _get_latent_dim(self) -> int:
        return self.latent_dim

    def _build_policy_latent_with_gait(
        self, flat_obs: torch.Tensor, velocity_pred: torch.Tensor, z_gait: torch.Tensor
    ) -> torch.Tensor:
        v_for_policy = velocity_pred.detach() if self.detach_velocity_pred else velocity_pred
        return torch.cat([flat_obs, v_for_policy, z_gait], dim=-1)

    def get_latent_outputs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        flat_obs = self._flatten_obs(obs)
        history, aux = self._split_from_flat(flat_obs)
        history_latent = self._encode_history(history)
        velocity_pred = self.velocity_head(history_latent)
        if "z_gait" not in obs.keys():
            raise KeyError("obs must contain 'z_gait' key (set by SegmentEncoderVelocityEstimatorPPO).")
        z_gait = obs["z_gait"]
        outputs = {
            "flat_obs": flat_obs,
            "history": history,
            "aux": aux,
            "history_latent": history_latent,
            "velocity_pred": velocity_pred,
            "z_gait": z_gait,
            "policy_latent": self._build_policy_latent_with_gait(flat_obs, velocity_pred, z_gait),
        }
        if self.enable_aux_loss:
            outputs["predicted_next_obs"] = self.next_obs_head(history_latent)
        return outputs

    def as_jit(self):  # noqa: D401
        raise NotImplementedError(
            "TransformerLatentGaitModel JIT export not implemented for V22a "
            "(deploy path requires baking FrozenSegmentEncoder and rolling buffer)."
        )

    def as_onnx(self, verbose: bool = False):
        raise NotImplementedError(
            "TransformerLatentGaitModel ONNX export not implemented for V22a."
        )
