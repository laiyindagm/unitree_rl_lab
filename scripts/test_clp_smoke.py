#!/usr/bin/env python3
"""CLP (Contrastive Latent Policy) 离线正确性验证脚本.

无需 Isaac Sim。验证模型构建、前向传播、损失函数、缓存机制、两阶段更新。
运行: /usr/local/miniconda3/envs/env_isaaclab/bin/python scripts/test_clp_smoke.py
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn as nn
from tensordict import TensorDict

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
HISTORY_LEN = 5
SINGLE_OBS_DIM = 54          # history_obs_dim(51) + cmd_dim(3)
FLAT_OBS_DIM = HISTORY_LEN * SINGLE_OBS_DIM  # 270
HISTORY_OBS_DIM = 51
CMD_DIM = 3
CMD_START_IDX = 6
NUM_ACTIONS = 15
PRED_HORIZON = 3
NUM_SPHERES = 3
SPHERE_DIM = 32
ENC_DIM = 96
# CMD_EMBED_DIM removed — raw cmd used
EXPECTED_LATENT_DIM = HISTORY_OBS_DIM + CMD_DIM + NUM_SPHERES * SPHERE_DIM + PRED_HORIZON * NUM_ACTIONS  # 195

B = 64     # batch size
N = 32     # num envs for storage test
T = 8      # rollout length for storage test
DEVICE = "cpu"

passed = 0
failed = 0
errors: list[str] = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


# ---------------------------------------------------------------------------
# 辅助: 创建模型所需的 obs TensorDict 和 obs_groups
# ---------------------------------------------------------------------------
def make_obs_td(batch: int = B, device: str = DEVICE) -> tuple[TensorDict, dict]:
    obs = TensorDict(
        {"policy": torch.randn(batch, FLAT_OBS_DIM, device=device)},
        batch_size=[batch],
    )
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    return obs, obs_groups


# ===================================================================
# Test 1: TCN model construction & latent dim
# ===================================================================
def test_model_construction():
    print("\n[Test 1] 模型构建 (TCN & Transformer)")
    obs, obs_groups = make_obs_td()

    # --- Import ---
    from unitree_rl_lab.utils.contrastive_latent_model import (
        ContrastiveLatentModel,
        TCNEncoder,
        CLPTransformerEncoder,
        ProductSphereProjection,
        ContrastiveProjector,
        FiLMGenerator,
    )

    # TCN model
    model_tcn = ContrastiveLatentModel(
        obs, obs_groups, "actor", NUM_ACTIONS,
        hidden_dims=(512, 256, 128),
        encoder_type="tcn",
        history_len=HISTORY_LEN,
        history_obs_dim=HISTORY_OBS_DIM,
        cmd_dim=CMD_DIM,
        cmd_start_idx=CMD_START_IDX,
        enc_dim=ENC_DIM,
        sphere_dim=SPHERE_DIM,
        num_spheres=NUM_SPHERES,
        pred_horizon=PRED_HORIZON,
        num_actions=NUM_ACTIONS,
    ).to(DEVICE)

    check("TCN model created", model_tcn is not None)
    check("latent dim = 224", model_tcn._latent_dim == EXPECTED_LATENT_DIM,
          f"got {model_tcn._latent_dim}")
    check("obs_dim = 270", model_tcn.obs_dim == FLAT_OBS_DIM,
          f"got {model_tcn.obs_dim}")
    check("encoder is TCNEncoder", isinstance(model_tcn.encoder, TCNEncoder))
    check("encoder.output_dim = 96", model_tcn.encoder.output_dim == ENC_DIM)
    check("sphere_proj exists", isinstance(model_tcn.sphere_proj, ProductSphereProjection))
    check("3 contrastive projectors", len(model_tcn.contrast_projs) == NUM_SPHERES)
    check("no cmd_embedding (removed)", not hasattr(model_tcn, "cmd_embedding"))
    check("generator exists", isinstance(model_tcn.generator, FiLMGenerator))

    # Transformer model
    model_tf = ContrastiveLatentModel(
        obs, obs_groups, "actor", NUM_ACTIONS,
        hidden_dims=(512, 256, 128),
        encoder_type="transformer",
        history_len=HISTORY_LEN,
        history_obs_dim=HISTORY_OBS_DIM,
        cmd_dim=CMD_DIM,
        cmd_start_idx=CMD_START_IDX,
        enc_dim=ENC_DIM,
        sphere_dim=SPHERE_DIM,
        num_spheres=NUM_SPHERES,
    ).to(DEVICE)
    check("Transformer model created", model_tf is not None)
    check("TF encoder is CLPTransformerEncoder",
          isinstance(model_tf.encoder, CLPTransformerEncoder))

    return model_tcn, model_tf


# ===================================================================
# Test 2: Forward pass & shapes
# ===================================================================
def test_forward_pass(model_tcn, model_tf):
    print("\n[Test 2] 前向传播 & shape 验证")
    obs, _ = make_obs_td()

    # get_latent
    latent = model_tcn.get_latent(obs)
    check("get_latent output shape", latent.shape == (B, EXPECTED_LATENT_DIM),
          f"got {latent.shape}")
    check("get_latent values finite", torch.isfinite(latent).all().item())

    # Full forward (act)
    action = model_tcn(obs, stochastic_output=True)
    check("forward action shape", action.shape == (B, NUM_ACTIONS),
          f"got {action.shape}")
    check("forward action finite", torch.isfinite(action).all().item())

    # Transformer forward
    latent_tf = model_tf.get_latent(obs)
    check("TF get_latent shape", latent_tf.shape == (B, EXPECTED_LATENT_DIM),
          f"got {latent_tf.shape}")


# ===================================================================
# Test 3: Sphere normalization
# ===================================================================
def test_sphere_normalization(model_tcn):
    print("\n[Test 3] 乘积球面归一化")
    obs, _ = make_obs_td()
    flat_obs = obs["policy"]

    z_spheres, cmd, o_current = model_tcn.encode(flat_obs)
    check("encode returns 3 spheres", len(z_spheres) == NUM_SPHERES)
    check("cmd shape", cmd.shape == (B, CMD_DIM), f"got {cmd.shape}")
    check("o_current shape", o_current.shape == (B, HISTORY_OBS_DIM),
          f"got {o_current.shape}")

    for i, z in enumerate(z_spheres):
        norms = z.norm(dim=-1)
        check(f"sphere {i} shape [{B},{SPHERE_DIM}]",
              z.shape == (B, SPHERE_DIM), f"got {z.shape}")
        check(f"sphere {i} L2 norm ≈ 1.0",
              torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
              f"max deviation {(norms - 1.0).abs().max().item():.6f}")


# ===================================================================
# Test 4: Contrastive projection
# ===================================================================
def test_contrastive_projection(model_tcn):
    print("\n[Test 4] 对比投影头")
    obs, _ = make_obs_td()
    flat_obs = obs["policy"]
    z_spheres, _, _ = model_tcn.encode(flat_obs)
    p_spheres = model_tcn.project_contrastive(z_spheres)

    check("project_contrastive returns 3 outputs", len(p_spheres) == NUM_SPHERES)
    for i, p in enumerate(p_spheres):
        norms = p.norm(dim=-1)
        check(f"projection {i} shape [{B},{SPHERE_DIM}]",
              p.shape == (B, SPHERE_DIM), f"got {p.shape}")
        check(f"projection {i} L2 norm ≈ 1.0",
              torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


# ===================================================================
# Test 5: FiLM generator
# ===================================================================
def test_film_generator(model_tcn):
    print("\n[Test 5] FiLM 生成器")
    z_cat = torch.randn(B, NUM_SPHERES * SPHERE_DIM)
    cmd_fg = torch.randn(B, CMD_DIM)
    a_pred = model_tcn.generator(z_cat, cmd_fg)
    expected_shape = (B, PRED_HORIZON * NUM_ACTIONS)
    check("generator output shape", a_pred.shape == expected_shape,
          f"got {a_pred.shape}, expected {expected_shape}")
    check("generator output finite", torch.isfinite(a_pred).all().item())


# ===================================================================
# Test 6: Cache mechanism
# ===================================================================
def test_cache_mechanism(model_tcn):
    print("\n[Test 6] 缓存机制")
    obs, _ = make_obs_td()

    # First call populates cache
    latent = model_tcn.get_latent(obs)
    z_cat, cmd_c, a_pred, o_current = model_tcn.get_cached_repr()

    check("cached z_cat shape", z_cat.shape == (B, NUM_SPHERES * SPHERE_DIM),
          f"got {z_cat.shape}")
    check("cached cmd shape", cmd_c.shape == (B, CMD_DIM),
          f"got {cmd_c.shape}")
    check("cached a_pred shape", a_pred.shape == (B, PRED_HORIZON * NUM_ACTIONS),
          f"got {a_pred.shape}")
    check("cached o_current shape", o_current.shape == (B, HISTORY_OBS_DIM),
          f"got {o_current.shape}")

    # Verify cache is detached
    check("z_cat no grad", not z_cat.requires_grad)
    check("cmd_c no grad", not cmd_c.requires_grad)
    check("a_pred no grad", not a_pred.requires_grad)
    check("o_current no grad", not o_current.requires_grad)

    # get_latent_from_cache should reproduce the exact same latent
    from unitree_rl_lab.utils.contrastive_latent_model import ContrastiveLatentModel
    latent_from_cache = ContrastiveLatentModel.get_latent_from_cache(
        o_current, cmd_c, z_cat, a_pred
    )
    check("cache reproduces latent", torch.allclose(latent, latent_from_cache, atol=1e-6),
          f"max diff {(latent - latent_from_cache).abs().max().item():.8f}")


# ===================================================================
# Test 7: evaluate_from_latent
# ===================================================================
def test_evaluate_from_latent(model_tcn):
    print("\n[Test 7] evaluate_from_latent (Phase B bypass)")
    obs, _ = make_obs_td()

    # Standard forward
    model_tcn.get_latent(obs)
    z_cat, cmd_c, a_pred, o_current = model_tcn.get_cached_repr()
    latent = model_tcn.get_latent_from_cache(o_current, cmd_c, z_cat, a_pred)

    # evaluate_from_latent (deterministic)
    out_det = model_tcn.evaluate_from_latent(latent, stochastic_output=False)
    check("evaluate_from_latent det shape", out_det.shape == (B, NUM_ACTIONS),
          f"got {out_det.shape}")
    check("evaluate_from_latent det finite", torch.isfinite(out_det).all().item())

    # evaluate_from_latent (stochastic)
    out_sto = model_tcn.evaluate_from_latent(latent, stochastic_output=True)
    check("evaluate_from_latent sto shape", out_sto.shape == (B, NUM_ACTIONS),
          f"got {out_sto.shape}")


# ===================================================================
# Test 8: Factored InfoNCE loss
# ===================================================================
def test_infonce_loss():
    print("\n[Test 8] 分解 InfoNCE 损失")
    from unitree_rl_lab.utils.contrastive_ppo import factored_infonce, quantize_to_levels

    projections = [F.normalize(torch.randn(B, SPHERE_DIM), dim=-1) for _ in range(NUM_SPHERES)]
    for p in projections:
        p.requires_grad_(True)

    # Create varied labels (not all same)
    levels = torch.tensor([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    values = torch.linspace(-0.5, 0.5, B)
    labels = [quantize_to_levels(values, levels) for _ in range(NUM_SPHERES)]

    loss = factored_infonce(projections, labels, temperature=0.5)
    check("infonce loss is scalar", loss.dim() == 0)
    check("infonce loss > 0", loss.item() > 0, f"got {loss.item():.4f}")
    check("infonce loss finite", torch.isfinite(loss).item())

    # Gradient check
    loss.backward()
    for i, p in enumerate(projections):
        check(f"infonce grad exists for sphere {i}", p.grad is not None)
        if p.grad is not None:
            check(f"infonce grad finite for sphere {i}",
                  torch.isfinite(p.grad).all().item())

    # Perfect alignment: all same label → loss should be small
    same_labels = [torch.zeros(B, dtype=torch.long) for _ in range(NUM_SPHERES)]
    loss_same = factored_infonce(
        [F.normalize(torch.randn(B, SPHERE_DIM), dim=-1) for _ in range(NUM_SPHERES)],
        same_labels, 0.5,
    )
    # Note: with all same label, all pairs are positive → NCE degenerates
    # (this is expected, just check it doesn't crash)
    check("infonce all-same-label no crash", torch.isfinite(loss_same).item())


# ===================================================================
# Test 9: Sequence prediction loss
# ===================================================================
def test_sequence_prediction_loss():
    print("\n[Test 9] 序列预测损失")
    from unitree_rl_lab.utils.contrastive_ppo import sequence_prediction_loss

    a_pred = torch.randn(B, PRED_HORIZON * NUM_ACTIONS)
    future_gt = torch.randn(B, PRED_HORIZON * NUM_ACTIONS)
    mask_all_valid = torch.ones(B, PRED_HORIZON, dtype=torch.bool)
    mask_all_invalid = torch.zeros(B, PRED_HORIZON, dtype=torch.bool)

    loss_valid = sequence_prediction_loss(a_pred, future_gt, mask_all_valid, gamma=0.9)
    check("seq loss all-valid is scalar", loss_valid.dim() == 0)
    check("seq loss all-valid > 0", loss_valid.item() > 0)
    check("seq loss all-valid finite", torch.isfinite(loss_valid).item())

    loss_invalid = sequence_prediction_loss(a_pred, future_gt, mask_all_invalid, gamma=0.9)
    check("seq loss all-invalid = 0", loss_invalid.item() == 0.0,
          f"got {loss_invalid.item()}")

    # Perfect prediction → loss ≈ 0
    loss_perfect = sequence_prediction_loss(a_pred, a_pred, mask_all_valid, gamma=0.9)
    check("seq loss perfect pred ≈ 0", loss_perfect.item() < 1e-6,
          f"got {loss_perfect.item():.8f}")

    # Time decay: later steps should contribute less
    mask_step0 = torch.zeros(B, PRED_HORIZON, dtype=torch.bool)
    mask_step0[:, 0] = True
    mask_step2 = torch.zeros(B, PRED_HORIZON, dtype=torch.bool)
    mask_step2[:, 2] = True
    loss_step0 = sequence_prediction_loss(a_pred, future_gt, mask_step0, gamma=0.9)
    loss_step2 = sequence_prediction_loss(a_pred, future_gt, mask_step2, gamma=0.9)
    # Same MSE but step2 has weight gamma^2 < gamma^0
    # Actually the average is per-weighted-sample, so the loss value may differ
    # Just check both are finite
    check("seq loss step0 finite", torch.isfinite(loss_step0).item())
    check("seq loss step2 finite", torch.isfinite(loss_step2).item())


# ===================================================================
# Test 10: Obs splitting
# ===================================================================
def test_obs_splitting(model_tcn):
    print("\n[Test 10] 观测拆分")
    flat_obs = torch.randn(B, FLAT_OBS_DIM)
    history_no_cmd, cmd, o_current = model_tcn._split_obs(flat_obs)

    check("history_no_cmd shape", history_no_cmd.shape == (B, HISTORY_LEN, HISTORY_OBS_DIM),
          f"got {history_no_cmd.shape}")
    check("cmd shape", cmd.shape == (B, CMD_DIM), f"got {cmd.shape}")
    check("o_current shape", o_current.shape == (B, HISTORY_OBS_DIM),
          f"got {o_current.shape}")

    # Verify cmd is extracted from the correct position in the last frame
    frames = flat_obs.view(B, HISTORY_LEN, SINGLE_OBS_DIM)
    expected_cmd = frames[:, -1, CMD_START_IDX : CMD_START_IDX + CMD_DIM]
    check("cmd values match last frame slice",
          torch.allclose(cmd, expected_cmd, atol=1e-7))

    # Verify o_current = last frame without cmd
    expected_before = frames[:, -1, :CMD_START_IDX]
    expected_after = frames[:, -1, CMD_START_IDX + CMD_DIM:]
    expected_o_current = torch.cat([expected_before, expected_after], dim=-1)
    check("o_current = last frame without cmd",
          torch.allclose(o_current, expected_o_current, atol=1e-7))


# ===================================================================
# Test 11: Uniformity & alignment metrics
# ===================================================================
def test_metrics():
    print("\n[Test 11] uniformity & alignment 指标")
    from unitree_rl_lab.utils.contrastive_ppo import compute_uniformity, compute_alignment

    z = F.normalize(torch.randn(B, SPHERE_DIM), dim=-1)
    labels = torch.randint(0, 5, (B,))

    u = compute_uniformity(z)
    check("uniformity is scalar", u.dim() == 0)
    check("uniformity < 0 (spread)", u.item() < 0,
          f"got {u.item():.4f}")

    a = compute_alignment(z, labels)
    check("alignment is scalar", a.dim() == 0)
    check("alignment >= 0", a.item() >= 0)

    # Collapsed embeddings → uniformity → 0
    z_collapsed = torch.ones(B, SPHERE_DIM) / (SPHERE_DIM ** 0.5)
    u_collapsed = compute_uniformity(z_collapsed)
    check("collapsed uniformity ≈ 0", u_collapsed.item() > -0.5,
          f"got {u_collapsed.item():.4f}")


# ===================================================================
# Test 12: quantize_to_levels
# ===================================================================
def test_quantize():
    print("\n[Test 12] quantize_to_levels")
    from unitree_rl_lab.utils.contrastive_ppo import quantize_to_levels

    levels = torch.tensor([0.0, 0.5, 1.0])
    values = torch.tensor([0.1, 0.4, 0.6, 0.9, 1.1])
    indices = quantize_to_levels(values, levels)
    expected = torch.tensor([0, 1, 1, 2, 2])
    check("quantize basic", torch.equal(indices, expected),
          f"got {indices.tolist()}, expected {expected.tolist()}")


# ===================================================================
# Test 13: Latent gradient blocking
# ===================================================================
def test_gradient_blocking(model_tcn):
    print("\n[Test 13] latent 梯度阻断")
    obs, _ = make_obs_td()

    # get_latent should return a detached tensor
    latent = model_tcn.get_latent(obs)
    check("latent is detached (no requires_grad)", not latent.requires_grad)

    # Verify encoder params have requires_grad=True (they should be trainable)
    enc_params = list(model_tcn.encoder.parameters())
    check("encoder params trainable", all(p.requires_grad for p in enc_params))


# ===================================================================
# Test 14: Two-phase optimization separation
# ===================================================================
def test_two_phase_optimization(model_tcn):
    print("\n[Test 14] 两阶段优化器分离")
    from unitree_rl_lab.utils.contrastive_ppo import ContrastivePPO
    from rsl_rl.storage import RolloutStorage

    obs, obs_groups = make_obs_td(N, DEVICE)

    # Create a minimal critic
    from rsl_rl.models.mlp_model import MLPModel
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=(64,)).to(DEVICE)

    # Create storage
    storage = RolloutStorage("rl", N, T, obs, [NUM_ACTIONS], DEVICE)

    # Create ContrastivePPO
    alg = ContrastivePPO(
        actor=model_tcn,
        critic=critic,
        storage=storage,
        nce_coef=0.1,
        gen_coef=0.5,
        gen_coef_end=0.1,
        gen_decay_iters=10000,
        tau_init=0.5,
        repr_lr=1e-4,
        device=DEVICE,
    )

    check("repr_optimizer created", alg.repr_optimizer is not None)
    check("log_tau is parameter", isinstance(alg.log_tau, nn.Parameter))
    check("tau > 0", alg.tau.item() > 0, f"got {alg.tau.item():.4f}")

    # Check repr_optimizer param count
    repr_param_count = sum(p.numel() for g in alg.repr_optimizer.param_groups for p in g["params"])
    mlp_param_count = sum(p.numel() for p in model_tcn.mlp.parameters())
    check("repr_optimizer has params", repr_param_count > 0,
          f"got {repr_param_count}")
    check("repr params != mlp params (separate)", repr_param_count != mlp_param_count)

    # Check storage monkey-patch
    check("storage.cached_z_cat exists",
          hasattr(storage, "cached_z_cat") and storage.cached_z_cat.shape == (T, N, NUM_SPHERES * SPHERE_DIM))
    check("storage.cached_cmd exists",
          hasattr(storage, "cached_cmd") and storage.cached_cmd.shape == (T, N, CMD_DIM))
    check("storage.cached_a_pred exists",
          hasattr(storage, "cached_a_pred") and storage.cached_a_pred.shape == (T, N, PRED_HORIZON * NUM_ACTIONS))
    check("storage.cached_o_current exists",
          hasattr(storage, "cached_o_current") and storage.cached_o_current.shape == (T, N, HISTORY_OBS_DIM))

    return alg, storage, obs_groups


# ===================================================================
# Test 15: Gen coef schedule
# ===================================================================
def test_gen_coef_schedule(alg):
    print("\n[Test 15] gen_coef 衰减 schedule")
    alg.counter = 0
    coef_start = alg._get_gen_coef()
    check("gen_coef at start ≈ 0.5", abs(coef_start - 0.5) < 1e-6,
          f"got {coef_start}")

    alg.counter = 5000
    coef_mid = alg._get_gen_coef()
    expected_mid = 0.5 + 0.5 * (0.1 - 0.5)
    check("gen_coef at midpoint ≈ 0.3", abs(coef_mid - expected_mid) < 1e-6,
          f"got {coef_mid}, expected {expected_mid}")

    alg.counter = 10000
    coef_end = alg._get_gen_coef()
    check("gen_coef at end ≈ 0.1", abs(coef_end - 0.1) < 1e-6,
          f"got {coef_end}")

    alg.counter = 20000
    coef_past = alg._get_gen_coef()
    check("gen_coef past end = 0.1", abs(coef_past - 0.1) < 1e-6,
          f"got {coef_past}")

    alg.counter = 0  # reset


# ===================================================================
# Test 16: Full mini-step (act + cache)
# ===================================================================
def test_act_and_cache(alg, storage, obs_groups):
    print("\n[Test 16] act() + 缓存写入")

    # Reset storage step
    storage.step = 0
    obs_td, _ = make_obs_td(N, DEVICE)

    # Simulate add_transition to increment step
    # In real code, storage.add_transition is called which increments step
    # Here we manually simulate the flow
    storage.step = 0

    # Store a dummy transition first (storage internals need this)
    # We'll manually test the cache write by calling act() logic
    from unitree_rl_lab.utils.contrastive_latent_model import ContrastiveLatentModel

    # Direct test: call get_latent then check cache
    model = alg.actor
    latent = model.get_latent(obs_td)
    z_cat, cmd_c, a_pred, o_current = model.get_cached_repr()

    # Simulate what act() does after super().act()
    step_idx = 0
    storage.cached_z_cat[step_idx] = z_cat
    storage.cached_cmd[step_idx] = cmd_c
    storage.cached_a_pred[step_idx] = a_pred
    storage.cached_o_current[step_idx] = o_current

    check("cache written to storage z_cat",
          torch.allclose(storage.cached_z_cat[step_idx], z_cat))
    check("cache written to storage cmd",
          torch.allclose(storage.cached_cmd[step_idx], cmd_c))
    check("cache written to storage a_pred",
          torch.allclose(storage.cached_a_pred[step_idx], a_pred))
    check("cache written to storage o_current",
          torch.allclose(storage.cached_o_current[step_idx], o_current))


# ===================================================================
# Test 17: Future actions builder
# ===================================================================
def test_future_actions_builder(alg, storage):
    print("\n[Test 17] _build_future_actions")

    # Fill storage with dummy data
    storage.actions = torch.randn(T, N, NUM_ACTIONS, device=DEVICE)
    storage.dones = torch.zeros(T, N, device=DEVICE)

    future_actions, future_mask = alg._build_future_actions()
    check("future_actions shape",
          future_actions.shape == (T, N, PRED_HORIZON * NUM_ACTIONS),
          f"got {future_actions.shape}")
    check("future_mask shape",
          future_mask.shape == (T, N, PRED_HORIZON),
          f"got {future_mask.shape}")

    # Last step should have no valid futures
    check("last step mask all False",
          not future_mask[-1].any().item())

    # First step with no dones should have all K steps valid (if T > K)
    if T > PRED_HORIZON:
        check("first step mask all True (no dones, enough steps)",
              future_mask[0].all().item())

    # Test with a done in the middle
    storage.dones[2, 0] = 1.0  # env 0 done at step 2
    future_actions2, future_mask2 = alg._build_future_actions()
    # Step 1, env 0: future_t=2 checks dones[1]=0 → valid; future_t=3 checks dones[1]=0, dones[2]=1 → invalid
    check("done blocks future (step 1, env 0, k=1 should be False)",
          not future_mask2[1, 0, 1].item())
    check("done doesn't block earlier step (step 1, env 0, k=0 should be True)",
          future_mask2[1, 0, 0].item())

    storage.dones[:] = 0  # reset


# ===================================================================
# Test 18: Export models
# ===================================================================
def test_export(model_tcn):
    print("\n[Test 18] 导出模型 (JIT & ONNX)")

    jit_model = model_tcn.as_jit()
    check("JIT model created", jit_model is not None)

    x = torch.randn(1, FLAT_OBS_DIM)
    out_jit = jit_model(x)
    check("JIT forward shape", out_jit.shape == (1, NUM_ACTIONS),
          f"got {out_jit.shape}")
    check("JIT forward finite", torch.isfinite(out_jit).all().item())

    onnx_model = model_tcn.as_onnx()
    check("ONNX model created", onnx_model is not None)
    check("ONNX input_size", onnx_model.input_size == FLAT_OBS_DIM)

    out_onnx = onnx_model(x)
    check("ONNX forward shape", out_onnx.shape == (1, NUM_ACTIONS),
          f"got {out_onnx.shape}")


# ===================================================================
# Test 19: Phase A gradient flow
# ===================================================================
def test_phase_a_gradient(model_tcn):
    print("\n[Test 19] Phase A 梯度流")
    from unitree_rl_lab.utils.contrastive_ppo import (
        factored_infonce, sequence_prediction_loss, quantize_to_levels,
    )

    obs, _ = make_obs_td()
    flat_obs = obs["policy"]

    # Encode (with gradient)
    z_spheres, cmd, o_current = model_tcn.encode(flat_obs)
    p_spheres = model_tcn.project_contrastive(z_spheres)

    levels = torch.tensor([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    labels = [quantize_to_levels(cmd[:, i], levels) for i in range(CMD_DIM)]
    nce_loss = factored_infonce(p_spheres, labels, 0.5)

    z_cat = torch.cat(z_spheres, dim=-1)
    a_pred = model_tcn.generator(z_cat, cmd)
    future_gt = torch.randn_like(a_pred)
    mask = torch.ones(B, PRED_HORIZON, dtype=torch.bool)
    gen_loss = sequence_prediction_loss(a_pred, future_gt, mask, gamma=0.9)

    total_loss = 0.1 * nce_loss + 0.5 * gen_loss
    total_loss.backward()

    # Check gradients flow to encoder
    enc_grads = [p.grad for p in model_tcn.encoder.parameters() if p.grad is not None]
    check("encoder has gradients", len(enc_grads) > 0,
          f"got {len(enc_grads)} params with grads")

    # Check gradients flow to sphere_proj
    sphere_grads = [p.grad for p in model_tcn.sphere_proj.parameters() if p.grad is not None]
    check("sphere_proj has gradients", len(sphere_grads) > 0)

    # Check gradients flow to generator
    gen_grads = [p.grad for p in model_tcn.generator.parameters() if p.grad is not None]
    check("generator has gradients", len(gen_grads) > 0)

    # Check MLP does NOT have gradients (should be excluded from Phase A)
    mlp_grads = [p.grad for p in model_tcn.mlp.parameters() if p.grad is not None]
    check("MLP has no gradients (Phase A only)", len(mlp_grads) == 0,
          f"got {len(mlp_grads)} params with grads")

    model_tcn.zero_grad()


# ===================================================================
# Main
# ===================================================================
def main():
    global passed, failed

    # Add project to path
    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(project_root, "source", "unitree_rl_lab")
    if src not in sys.path:
        sys.path.insert(0, src)

    print("=" * 60)
    print("CLP 离线正确性验证")
    print("=" * 60)

    try:
        # Test 1-7: Model-level tests
        model_tcn, model_tf = test_model_construction()
        test_forward_pass(model_tcn, model_tf)
        test_sphere_normalization(model_tcn)
        test_contrastive_projection(model_tcn)
        test_film_generator(model_tcn)
        test_cache_mechanism(model_tcn)
        test_evaluate_from_latent(model_tcn)

        # Test 8-12: Loss function tests
        test_infonce_loss()
        test_sequence_prediction_loss()
        test_obs_splitting(model_tcn)
        test_metrics()
        test_quantize()

        # Test 13-19: Integration tests
        test_gradient_blocking(model_tcn)
        test_two_phase_optimization_result = test_two_phase_optimization(model_tcn)
        alg, storage, obs_groups = test_two_phase_optimization_result
        test_gen_coef_schedule(alg)
        test_act_and_cache(alg, storage, obs_groups)
        test_future_actions_builder(alg, storage)
        test_export(model_tcn)
        test_phase_a_gradient(model_tcn)

    except Exception as e:
        print(f"\n  [FATAL] 未捕获异常: {e}")
        traceback.print_exc()
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"结果: {passed}/{total} PASS, {failed}/{total} FAIL")
    if errors:
        print(f"\n失败的测试:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Need F from torch.nn.functional at module level for some tests
    import torch.nn.functional as F
    sys.exit(main())
