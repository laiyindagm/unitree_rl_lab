# V4.3/V4.4 Style Encoder Experiment Analysis - 2026-05-11

## 1. Run Status And Extracted Data

Remote experiments have finished. The monitored PIDs are no longer alive, and all summary files were present on the remote server at:

- `logs/frnc_style_v43_plan_20260510_234247/confirm50_v1`
- `logs/frnc_style_v43_plan_20260510_234247/grid20_v1`
- `logs/frnc_style_v43_v2_confirm_20260511_001550/confirm50_v2_phase_aug`

Small CSV summaries were copied locally to:

- `docs/handoff_contrastive/results_v43_20260511/v1_confirm50`
- `docs/handoff_contrastive/results_v43_20260511/v1_grid20`
- `docs/handoff_contrastive/results_v43_20260511/v2_phase_confirm50`

No checkpoint, feature shard, or other large artifact was downloaded.

## 2. Evaluation Axes

The current goal is not only to maximize reconstruction/probe quality. The encoder must represent gait/style information while avoiding trivial command, mode, and phase shortcuts.

Main metrics:

- `R2_Y0_from_Z`: recoverable gait/style state information from the latent. Higher is better.
- `R2_Y0_gain_over_cmd_mode`: information beyond command and mode baselines. Higher is better.
- `R2_phase_from_Z` and `R2_phase_from_Zres`: phase leakage. Lower is better.
- `effective_rank`: latent diversity/capacity. Higher is better if shortcut leakage is controlled.
- `rho_G_cond_cmd_spearman`: gait-distance geometry correlation under similar command. Higher is better.
- `OOD_drop_R2_Y0`: held-out source-group robustness drop. Lower is better.
- `phase_shift_ratio`: same-parent phase perturbation distance relative to same-bucket distance. Lower is better.
- residual probes after linear `cmd+mode` removal indicate what remains in the latent. A key caveat is that nonlinear residual MLP probes still reveal shortcut information.

## 3. Aggregate Results

| group | n | R2_Y0 | gain | cmd_R2 | mode_R2 | phase_R2 | eff_rank | rho_cond | OOD_drop | R2_Y0_Zres | phase_Zres | phase_shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 C1 stats_tcn 50 | 3 | 0.6727 | 0.4819 | 0.7341 | 0.5512 | 0.0769 | 10.6626 | 0.6608 | 0.0527 | 0.3325 | 0.0402 | - |
| v1 C2 stats_tcn_dual 50 | 3 | 0.6724 | 0.4822 | 0.7147 | 0.5606 | 0.0595 | 10.2755 | 0.6613 | 0.0539 | 0.3346 | 0.0282 | - |
| v2 C1 phase-aug stats_tcn 50 | 3 | 0.7007 | 0.5094 | 0.7688 | 0.5365 | 0.0850 | 12.5721 | 0.6327 | 0.0523 | 0.3567 | 0.0462 | 0.1792 |
| v2 C2 phase-aug stats_tcn_dual 50 | 3 | 0.7068 | 0.5153 | 0.7618 | 0.5579 | 0.0527 | 12.6423 | 0.6462 | 0.0501 | 0.3608 | 0.0192 | 0.1785 |

The v1 20-epoch grid found a different tradeoff:

| candidate | R2_Y0 | phase_R2 | eff_rank | rho_cond | OOD_drop | phase_Zres |
|---|---:|---:|---:|---:|---:|---:|
| G1 rank015 | 0.6446 | 0.0761 | 10.0533 | 0.6539 | 0.0614 | 0.0419 |
| G2 adv015 | 0.6452 | 0.0655 | 8.7774 | 0.7168 | 0.0605 | 0.0344 |
| G3 alt3 | 0.6456 | 0.0710 | 9.0133 | 0.7173 | 0.0607 | 0.0468 |
| G7 yphi025 | 0.6406 | 0.0707 | 8.3036 | 0.7197 | 0.0620 | 0.0472 |

The grid candidates improve conditional gait geometry, but lose substantial state information and latent rank.

## 4. Best Current Candidate

The current selection ranking chooses:

`C2_stats_tcn_dual_50_s0`

Remote result directory:

`logs/frnc_style_v43_v2_confirm_20260511_001550/confirm50_v2_phase_aug/C2_stats_tcn_dual_50_s0`

Key metrics:

- `selection_score = 1.0788`
- `R2_Y0_from_Z = 0.7082`
- `R2_Y0_gain_over_cmd_mode = 0.5166`
- `effective_rank = 12.6893`
- `rho_G_cond_cmd_spearman = 0.6446`
- `OOD_drop_R2_Y0 = 0.0491`
- `R2_phase_from_Z = 0.0491`
- `R2_phase_from_Zres = 0.0168`
- `phase_shift_ratio = 0.1780`

If prioritizing raw `R2_Y0`, `C2_stats_tcn_dual_50_s1` is slightly higher at `0.7108`, but `s0` has better phase leakage and overall selection score. The seed variance is small, so the C2 phase-aug recipe is stable.

## 5. Effectiveness Assessment

### Effective Changes

Phase augmentation is clearly useful. It increases `R2_Y0` from about `0.672` to `0.707`, increases effective rank from about `10.3` to `12.6`, and slightly improves OOD drop. This suggests the model is using more behavior/state content rather than collapsing onto a narrow representation.

The dual TCN head is also useful. On phase-aug data, C2 improves `R2_Y0` over C1 and reduces phase leakage sharply: `phase_R2` drops from `0.0850` to `0.0527`, and `phase_Zres` drops from `0.0462` to `0.0192`.

The 50-epoch confirmation was justified for the selected candidates. The 20-epoch grid is useful for screening, but the confirmed 50-epoch v2 models reached a qualitatively better information/rank regime. The best 20-epoch grid models stayed around `R2_Y0 ~= 0.64`, while v2 C2 reached `~0.707`.

### Partially Effective Changes

The adversarial/alternating grid variants improved conditional geometry: G2/G3/G7 reached `rho_cond ~= 0.717-0.720`. However, this came with lower `R2_Y0` and lower rank. These settings are not the best final encoder, but they identify a useful constraint direction for the next round.

The current linear residual probe is too optimistic. Linear residual `R2_cmd_from_Zres` and `R2_mode_from_Zres` are approximately zero, but nonlinear MLP residual probes still decode command and mode strongly in v2 C2:

- `R2_cmd_mlp_from_Zres ~= 0.50`
- `R2_mode_mlp_from_Zres ~= 0.84`

So the representation is not truly shortcut-free. It has removed linear command/mode leakage, but still encodes command/mode nonlinearly.

### Not Yet Sufficient

The best phase-aug C2 model is the best current encoder for the combined objective of high gait information, low phase leakage, rank, and OOD stability. It is not yet a final solution for downstream control.

The main remaining gap is disentanglement: the latent still carries nonlinear command/mode information. A policy could exploit this as a shortcut instead of using a clean style coordinate. Also, `rho_cond` for v2 C2 is only `~0.646`, below the best grid candidates. This means the latent is information-rich, but its same-command gait geometry is not yet as clean as desired.

## 6. Current Conclusion

Use `C2_stats_tcn_dual_50_s0` as the current best baseline for downstream integration or policy-side probing. It has the strongest overall metric balance and stable multi-seed behavior.

Do not treat it as a solved style representation. The next experiment should combine the v2 phase-aug data recipe with the geometry-improving constraints from G2/G3/G7, and replace the current weak residual/adversarial checks with nonlinear adversaries or stronger residualization.

## 7. Recommended Next Experiment

The next round should target one specific hypothesis:

> Phase augmentation plus dual temporal encoding gives enough behavior information, but the latent still uses nonlinear command/mode shortcuts; stronger nonlinear invariance and geometry-preserving training should improve controllable style usefulness.

Recommended matrix:

- Data: keep `features_v2_phase_aug`; add bucket-balanced sampling and source-group-balanced batches.
- Model: keep C2 dual TCN as base; test larger projection head only after shortcut metrics improve.
- Training: add nonlinear command/mode adversary, stronger gradient reversal schedule, and an alternating objective variant.
- Geometry: keep RnC/metric terms, but tune toward G2/G3-style `rho_cond` without sacrificing C2-level `R2_Y0`.
- Validation: rank by a stricter score that includes nonlinear residual command/mode probes, not just linear residual probes.

Suggested confirmation targets before downstream RL:

- `R2_Y0_from_Z >= 0.69`
- `effective_rank >= 12`
- `R2_phase_from_Zres <= 0.025`
- `rho_G_cond_cmd_spearman >= 0.70`
- nonlinear residual `R2_cmd_mlp_from_Zres <= 0.20`
- nonlinear residual `R2_mode_mlp_from_Zres <= 0.30`
- `OOD_drop_R2_Y0 <= 0.055`

