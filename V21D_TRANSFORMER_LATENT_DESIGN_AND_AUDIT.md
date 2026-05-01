# V21d Transformer Latent Algorithm Design and Implementation Audit

Date: 2026-04-30

## 1. Goal of this round

V21d is the current latent-transformer experiment built on top of the validated V21c reward/task line. The purpose is not to revisit sampler design or 3-mode token design, but to make the policy explicitly model its own motion state from observation history.

The intended algorithmic goals are:

1. Encode observation history into a latent representation `z_t` that carries motion / gait state.
2. Predict achieved velocity from `z_t` with an explicit regression head.
3. Feed both `z_t` and the predicted velocity `v_hat_t` back into policy/value decision-making, not only into auxiliary losses.
4. Shape `z_t` with HIMLoco-inspired contrastive learning, where positive pairs are defined by achieved-speed bucket agreement rather than command equality.
5. Decay gait-assist shaping rewards to zero by global step 2000, while leaving main tracking/stability terms untouched.

---

## 2. Intended architecture

### 2.1 Actor-side representation

For actor observation history `o_{t-H+1:t}`:

- Split stacked actor observation into:
  - history window
  - current-frame auxiliary slice
- Encode history with:
  - linear projection
  - positional embedding
  - causal Transformer encoder
- Fuse current-frame auxiliary slice into the final history token by FiLM.
- Obtain latent state `z_t`.

### 2.2 Explicit velocity prediction

Use a small MLP head:

- `v_hat_t = velocity_head(z_t)`
- current output target is three-dimensional achieved velocity:
  - `base_lin_vel_x`
  - `base_lin_vel_y`
  - `base_ang_vel_z`

### 2.3 Policy / critic input semantics

The approved design is:

- `policy_input = concat(flat_actor_obs, z_t, v_hat_t)`
- same semantic concatenation is intended for the critic side as well, i.e. the value function should also benefit from latent and predicted velocity information.

The key requirement is that both `z_t` and `v_hat_t` directly participate in control/value inference rather than being supervised-only side outputs.

### 2.4 Contrastive learning semantics

Borrow the source/target idea from HIMLoco, but change positive-pair semantics:

- Source branch: encode history into `z_t`, then project to contrastive heads.
- Target branch: encode current or successor achieved state into target projections.
- Positive definition: two samples are positives when their discretized achieved velocity labels match on a given factor.
- Labels are factorized over:
  - `vx`
  - `vy`
  - `wz`
- Use factorized InfoNCE rather than a sparse joint class over `(vx, vy, wz)`.

### 2.5 Reward shaping decay

Only gait-assist rewards should decay linearly to zero by global step 2000:

- `gait`
- `feet_clearance`
- optionally any purely gait-assist single-support shaping term if enabled in the active stack

Main tracking, balance, stability, smoothness, and energy terms should remain unchanged.

---

## 3. Current implementation mapping

### 3.1 Model implementation

File:
- `source/unitree_rl_lab/unitree_rl_lab/utils/rsl_rl_transformer_model.py`

Implemented pieces:

- `TransformerLatentModel`
- causal Transformer history encoder
- FiLM fusion with current-frame auxiliary slice
- explicit `velocity_head`
- `policy_latent = concat(flat_obs, history_latent, velocity_pred)`
- contrastive target encoder `target_proj`
- multiple contrastive projectors
- optional next-observation head retained behind `enable_aux_loss`

This means the actor side already satisfies:

- history -> latent
- latent -> predicted velocity
- latent + predicted velocity concatenated back with flat actor observation

### 3.2 PPO implementation

File:
- `source/unitree_rl_lab/unitree_rl_lab/utils/transformer_ppo.py`

Implemented pieces:

- `TransformerPPO`
- achieved-velocity extraction from critic observation
- successor-step contrastive target extraction via `obs_next`
- velocity regression auxiliary loss
- factored achieved-velocity InfoNCE loss
- streamed per-timestep auxiliary backward to reduce graph retention
- `contrastive_max_samples` cap for memory control
- fixed or learnable contrastive temperature support

Current auxiliary losses:

- next-observation auxiliary loss, gated by `enable_aux_loss` and `aux_coef`
- velocity regression loss on `outputs["velocity_pred"]`
- contrastive latent loss on projected history/target latents

### 3.3 Contrastive utilities

File:
- `source/unitree_rl_lab/unitree_rl_lab/utils/contrastive_ppo.py`

Implemented reusable pieces used by V21d:

- `quantize_to_levels()`
- `factored_infonce()`
- batch subsampling in `factored_infonce()` through `max_samples`

### 3.4 Reward decay implementation

Files:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v21d.py`

Implemented pieces:

- `_linear_step_decay(env, end_step, start_step)`
- `feet_gait_speed_scaled_decayed(...)`
- `foot_clearance_speed_scaled_decayed(...)`
- `rotation_single_support_reward_decayed(...)`
- V21d reward config overrides only gait-related shaping terms to their decayed versions with `end_step=2000`

### 3.5 Registration and runner config

Files:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/__init__.py`

Implemented pieces:

- `RslRlTransformerLatentPpoAlgorithmCfg`
- `RslRlTransformerLatentModelCfg`
- `G115DofV21dTransformerLatentPPORunnerCfg`
- gym id registration:
  - `Unitree-G1-15dof-Velocity-Rot-V21d-TransformerLatent`

The current V21d runner now uses fixed contrastive temperature:

- `learnable_contrastive_temperature: bool = False`
- `learnable_contrastive_temperature=False` in the V21d runner instance

---

## 4. Important implementation details and recent stability fixes

### 4.1 OOM mitigation

The original contrastive path could OOM because InfoNCE forms `B x B` similarity matrices.

Mitigations now in code:

1. `factored_infonce()` accepts `max_samples` and subsamples when `B > max_samples`.
2. `TransformerPPO._compute_aux_losses()` performs streamed per-timestep backward rather than retaining a single large auxiliary graph over the whole rollout.
3. V21d config sets `contrastive_max_samples=1024` at the runner level.

### 4.2 Double-backward fix

The first streamed-backward version still failed because a learnable temperature tensor was reused across timestep losses in one update.

Fix now applied:

- `tau = self.contrastive_temperature.clamp(min=0.01, max=2.0)` is computed inside the per-step contrastive block.

### 4.3 Fixed tau change

The current V21d configuration now fixes contrastive temperature rather than learning it. This was done to reduce training instability and avoid confounding attribution while the architecture itself is still being validated.

---

## 5. Detailed implementation audit against intended design

This section compares the intended algorithm design with the current code, item by item.

### 5.1 History encoder with FiLM and causal masking

Status: MATCH

Evidence:
- `TransformerLatentModel` uses history slicing, positional embeddings, causal mask, Transformer encoder, FiLM fusion.
- Relevant code:
  - `rsl_rl_transformer_model.py` around `TransformerLatentModel.__init__`, `_encode_history_and_aux()`.

Assessment:
- This part matches the intended V21d design.

### 5.2 Explicit velocity regression from latent

Status: MATCH

Evidence:
- `velocity_head` is defined on top of `history_latent`.
- `TransformerPPO._compute_aux_losses()` applies `smooth_l1_loss(outputs["velocity_pred"], achieved_t)`.

Assessment:
- This matches the intended explicit supervised velocity signal.

### 5.3 Feed `obs + z_t + v_hat_t` into the actor

Status: MATCH

Evidence:
- `TransformerLatentModel._build_policy_latent()` returns `torch.cat([flat_obs, history_latent, velocity_pred], dim=-1)`.
- `get_latent()` returns `policy_latent`.

Assessment:
- Actor-side concatenation is implemented as intended.

### 5.4 Feed latent-enhanced representation into the critic as well

Status: PARTIAL / LIKELY MISMATCH

Evidence:
- In `G115DofV21dTransformerLatentPPORunnerCfg`, the critic is still `RslRlMLPModelCfg(...)`, not a latent-aware critic model.
- In `TransformerPPO.update()`, critic forward is still:
  - `values = self.critic(batch.observations, ...)`
- There is no visible code path concatenating `z_t` and `v_hat_t` into critic input.

Assessment:
- The approved design said latent and predicted velocity should be concatenated into policy AC, i.e. actor and critic semantics.
- Current implementation only clearly satisfies the actor side.
- The critic still appears to consume raw critic observations only.

Consequence:
- Value estimation may not benefit from the intended enriched motion-state representation.
- This is the main architecture mismatch currently visible in code.

### 5.5 HIMLoco-style source/target contrastive structure

Status: PARTIAL MATCH

Evidence:
- Source branch uses `outputs["history_latent"]` from current `obs_t`.
- Target branch uses `_get_contrastive_target_obs(obs_next)` and `encode_contrastive_target(target_obs)`.
- This is closer to history -> successor/current-state pairing than the old CLP cache path.

Assessment:
- The code does implement a source/target split and uses successor-step target state via `obs_next`, which is consistent with the intended HIMLoco-style direction.
- However, the way source and target are combined is currently:
  - `projections = normalize(src + tgt)`
- This is not a canonical two-tower contrastive formulation where source and target embeddings are contrasted directly as separate views.

Consequence:
- The current implementation captures the intended information sources, but the loss geometry is simplified relative to a stricter source-vs-target InfoNCE setup.
- This is a design approximation, not a clean mismatch, but it should be recognized explicitly.

### 5.6 Positive-pair definition via achieved-speed buckets

Status: MOSTLY MATCH, WITH ONE SUBTLE CHOICE

Evidence:
- Labels are derived from `achieved_t = _get_achieved_velocity_targets(obs_t)`.
- Factorized labels are built from:
  - `vx`
  - `vy`
  - `wz`
- `factored_infonce()` is used.

Assessment:
- This matches the user’s requirement that achieved speed, not command speed, defines positives.
- The factorized implementation also matches the approved design.

Subtlety:
- Labels are taken from `obs_t`, while the target branch features come from `obs_next`.
- So the pair being contrasted is effectively:
  - source from time `t`
  - target from time `t+1`
  - labels from achieved velocity at `t`

Risk:
- If the user wanted strict successor-state semantics, labels might arguably need to be derived from the same time index as the target branch or from both sides depending on the intended invariance.
- This is not necessarily wrong, but it is an implementation choice that should be treated as deliberate rather than accidental.

### 5.7 Auxiliary next-observation loss disabled by default

Status: MATCH

Evidence:
- `RslRlTransformerLatentModelCfg.enable_aux_loss = False`
- V21d runner sets `enable_aux_loss=False`
- algorithm sets `aux_loss_coef=0.0`

Assessment:
- This matches the intention to isolate variables and avoid mixing next-obs aux with the new latent/velocity/contrastive experiment.

### 5.8 Gait shaping decays to zero by step 2000

Status: MATCH

Evidence:
- reward wrappers in `rewards.py`
- V21d overrides gait and feet_clearance terms with `end_step=2000`

Assessment:
- This part is implemented correctly and in the intended isolated way.

### 5.9 New V21d config built on V21c rather than reverting to older lines

Status: MATCH

Evidence:
- `velocity_env_cfg_rot_v21d.py` inherits from `velocity_env_cfg_rot_v21c.py`
- only reward overrides are applied for decayed gait shaping

Assessment:
- This matches the experimental scoping requirement.

### 5.10 Contrastive temperature fixed rather than learnable

Status: MATCH TO LATEST USER INSTRUCTION

Evidence:
- `rsl_rl_ppo_cfg.py` now sets `learnable_contrastive_temperature=False`

Assessment:
- This now matches the latest explicit user directive.

---

## 6. Main conclusions from the audit

### Correctly implemented core pieces

The following major parts appear correctly implemented:

1. V21d env registration and config inheritance from V21c.
2. Gait reward decay wrappers and application to V21d.
3. Transformer history encoder with causal mask and FiLM fusion.
4. Explicit velocity regression head from latent.
5. Actor-side concatenation of `obs + latent + predicted velocity`.
6. Achieved-velocity factorized labels for contrastive supervision.
7. Contrastive subsampling and streamed auxiliary backward for memory stability.
8. Fixed temperature in the current V21d config.

### Most important mismatch still present

The biggest implementation mismatch versus the approved design is:

- the critic does not appear to consume `z_t` and `v_hat_t`.

The current code still uses a plain MLP critic on raw critic observations. If the intended requirement was truly actor+critic concatenation, this is not yet fully implemented.

### Secondary approximation to keep in mind

The contrastive implementation is directionally aligned with HIMLoco, but the current formulation simplifies the source/target geometry by summing source and target projections before InfoNCE rather than contrasting them as distinct branches.

This may still work, but it is a design approximation rather than a literal implementation of a two-branch contrastive objective.

---

## 7. Recommended next actions

If the goal is strict correctness relative to the approved design, the next code changes should be:

1. Make the critic latent-aware as well.
   - Either use a latent-enabled critic model, or explicitly build `critic_input = concat(critic_obs, z_t, v_hat_t)`.
   - Then ensure rollout/value paths use that enriched critic input consistently.

2. Decide whether the contrastive geometry should stay in the current simplified `normalize(src + tgt)` form.
   - If yes, document it as an intentional simplification.
   - If no, refactor to a cleaner source-target contrastive loss where source and target embeddings are compared directly.

3. Decide whether achieved labels for the source-target pair should be indexed at `t`, `t+1`, or both.
   - Current code uses labels from `t` and target features from `t+1`.
   - This is plausible, but should be made explicit.

4. After any semantic fixes, rerun:
   - static compile
   - short user-run training smoke test
   - monitor OOM / double-backward recurrence

---

## 8. Files reviewed in this audit

- `source/unitree_rl_lab/unitree_rl_lab/utils/rsl_rl_transformer_model.py`
- `source/unitree_rl_lab/unitree_rl_lab/utils/transformer_ppo.py`
- `source/unitree_rl_lab/unitree_rl_lab/utils/contrastive_ppo.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v21d.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/__init__.py`
