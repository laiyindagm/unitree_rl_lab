# V20l Milestone And Next Roadmap

Last updated: 2026-04-28

This document records the V20l milestone and the next research plan. It is intended for future agents and for migration to a new server. Read together with `AGENT_HANDOFF.md`.

## Milestone: V20l Succeeds In Sim2Sim

User-confirmed result:

```text
V20l successfully achieved sensitive and stable response over the full velocity command range.
The policy responds across the full speed interval, including pure vx/vy and combined commands.
```

Representative training terminal snapshot at iter 11685/20000:

```text
Mean reward: 76.40
Mean episode length: 982.89
Mean action std: 0.34
track_lin_vel_xy: 0.6939
track_ang_vel_z: 2.0730
track_ang_vel_z_sharp: 0.3647
wz_proportional: -0.2386
waist_roll_vel: -0.1781
waist_pitch_vel: -0.0422
waist_yaw_vel: -0.0075
zero_cmd_body_vel: -0.0213
error_vel_xy: 0.4990
error_vel_yaw: 0.4694
bad_orientation: 0.0241
```

Important interpretation:

- The raw aggregate training metrics still look imperfect (`error_vel_xy≈0.50`, `error_vel_yaw≈0.47`, `waist_roll_vel≈-0.18`).
- However, sim2sim behavior improved dramatically and reached full-speed-range command responsiveness.
- Therefore some global training metrics are not sufficient proxies for deployed behavioral quality. Future evaluation must include mode-specific sim2sim tests and mode-conditioned metrics, not only aggregate RSL-RL logs.

Current successful config:

```text
Gym ID: Unitree-G1-15dof-Velocity-Rot-V20l
File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v20l.py
Definition: V20g rewards + V20j observations
```

V20l preserves the required 3-way isolation:

```text
{standing, pure_wz, other}
```

and fixes the pure_xy training environment by inheriting V20g's standard yaw tracking:

```text
track_ang_vel_z_exp for both main and sharp yaw trackers
```

## Why V20l Worked

The key mechanism is not token removal. V20i removed token and regressed to V19-like behavior, with unstable zero command. V20j preserved 3-mode token but failed because it still retained the straight-walk yaw skip from V20f.

The successful hypothesis is:

```text
Pure vx/vy commands are cmd_wz=0 samples.
If yaw tracking is skipped for cmd_wz=0 straight walking, pure_xy has no yaw-stability supervision.
Joint commands often include cmd_wz != 0, so they receive real yaw gradients.
```

V20l keeps mode isolation but removes the missing-gradient problem:

```text
standing branch: isolated for zero-command stability
pure_wz branch: isolated for in-place rotation
other branch: pure_vx + pure_vy + joint, but now cmd_wz=0 samples get real yaw supervision
```

This explains why V20l can recover full-speed command response while V20i and V20j did not.

## Lessons Learned

1. Token isolation was not the root cause.

   Removing token made zero-command worse and did not solve pure_xy. Token controls specialization and contamination, but the training reward must still provide the missing gradients.

2. The minimum useful isolation is real.

   At least `{standing, pure_wz, other}` should remain isolated. Standing and pure rotation have qualitatively distinct objectives and should not be forced into the same branch as walking.

3. Pure_xy failure was a reward-environment problem.

   Pure_vx/pure_vy have cmd_wz=0 by construction. If yaw tracking returns free reward under straight walk, pure_xy can wobble yaw/waist without penalty.

4. Sampling density cannot create missing gradients.

   V20h-style pure_xy density increase cannot fix a zero-gradient yaw axis. Data allocation helps only after the reward provides a usable signal.

5. Training metrics must be made mode-conditioned.

   Aggregate `error_vel_xy`, `error_vel_yaw`, and waist penalties can stay high even when sim2sim behavior is useful. Future evaluation should report per-mode metrics: standing, pure_wz, pure_vx, pure_vy, joint.

6. Deploy observation names matter.

   For 3-mode token policies the Python ObsTerm attribute, deploy.yaml key, and C++ `REGISTER_OBSERVATION` name must all be `gait_mode_token_3`. A mismatch previously caused immediate deployment crash.

## What Is Likely Truly Effective So Far

High-confidence effective modules:

1. 3-mode token isolation `{standing, pure_wz, other}`

   Evidence: V20i no-token made zero command unstable; V20l with 3-mode isolation succeeds. Isolation should be retained.

2. Standard yaw tracking (`track_ang_vel_z_exp`) replacing straight-walk yaw skip

   Evidence: V20j (3-mode + yaw skip) failed; V20l (3-mode + standard yaw tracking) succeeds.

3. Standard linear yaw-frame tracking replacing rotation-skip linear tracking

   Evidence from V20f/V20g lineage: removing linear skip fixed pure-rotation drift blind spot and enforces zero-commanded axes.

4. V19f base reward/command scaffold

   Evidence: V19f was the stable trained-to-completion base with zero-speed fixed and yaw response from small command. V20l still inherits much of this stack.

Medium-confidence effective modules:

1. `zero_cmd_body_vel` at the V19f/V20 lineage strength

   It contributes to standing stability, but heavy standing penalties caused V19h collapse. Keep modest or scheduled.

2. Waist velocity damping

   It may reduce upper-body motion, but too-strong waist damping hurt earlier experiments. Needs careful ablation and head-specific metrics.

3. Movement incentive / cmd_nonresponse

   May help prevent nonresponse, but current values are small in logs. Needs ablation.

Unproven or potentially harmful modules:

1. More pure_xy sampling alone

   V20h rationale was rejected and should not be repeated without a new gradient source.

2. Adaptive sigma / baselined tracking

   Earlier versions reduced easy early rewards and caused instability.

3. Heavy always-on standing or zero-command penalties

   V19h showed suicide-policy collapse.

4. Over-strong waist damping

   V17b made behavior worse; do not increase blindly.

## Required Ablation Program

Before optimizing further, identify which modules truly matter. Use V20l as the reference baseline.

Ablations should be short enough to compare early learning plus one sim2sim checkpoint. Recommended checkpoints: 8k, 12k, and optionally 16k.

### Tier 1: Core Causal Ablations

A0: V20l baseline

```text
3-mode token + standard linear tracking + standard yaw tracking
```

A1: token granularity ablation

```text
V20g-style 5-mode token vs V20l 3-mode token
Question: does splitting pure_vx/pure_vy away from joint still harm now that yaw supervision exists?
Expected: 3-mode likely better for sample sharing; 5-mode may be acceptable if yaw reward is fixed.
```

A2: yaw reward ablation

```text
V20j vs V20l already mostly answers this.
Question: is standard yaw tracking the decisive fix?
Expected: yes.
```

A3: linear reward ablation

```text
Restore track_lin_vel_xy_rotation_skip while keeping standard yaw tracking and 3-mode token.
Question: is strict linear tracking necessary for full-speed response, or was yaw reward enough?
Expected: strict linear tracking likely needed for pure_w drift/zero-axis correctness.
```

### Tier 2: Stabilization / Smoothness Ablations

B1: action-rate and action smoothing

```text
Slightly increase action_rate or deploy action_smoothing.
Question: can velocity smoothness improve without reducing responsiveness?
Measure: command step response, velocity jerk, action delta, sim2sim foot noise.
```

B2: waist/head damping

```text
Small increments only. Do not repeat V17b over-damping.
Question: can camera/head stability improve without killing gait?
Measure: base_ang_vel_xy, projected_gravity oscillation, waist roll/pitch/yaw velocities, head orientation if available.
```

B3: gait/feet frequency and clearance

```text
Reduce unnecessary low-speed stepping frequency and contact impact.
Question: can indoor noise be reduced without reintroducing dead zone?
Measure: foot contact impulses/forces if available, feet_slide, feet_clearance, gait reward, contact switch frequency.
```

### Tier 3: Distribution / Metrics Ablations

C1: mode-conditioned metrics

```text
Add logging split by standing, pure_wz, pure_vx, pure_vy, joint.
Question: which mode contributes to aggregate error and waist motion?
This should be prioritized before many more reward changes.
```

C2: command transition evaluation

```text
Evaluate step commands and ramps separately.
Question: does stable full-range response also have smooth transient behavior?
```

## Next Optimization Goals

After ablations identify active modules, optimize for three deployment goals.

### Goal 1: Velocity Tracking Stability And Accuracy

Target:

```text
Minimize |v - v_cmd| / |v_cmd| over active command dimensions.
Also minimize velocity jerk and overshoot under command transitions.
```

Plan:

1. Add mode-conditioned relative error metrics.

```text
For active linear commands: abs(v_xy - cmd_xy) / max(abs(cmd_xy), eps)
For active yaw commands: abs(wz - cmd_wz) / max(abs(cmd_wz), eps)
Report per mode and per speed bucket.
```

2. Add smoothness metrics before adding rewards.

```text
velocity jerk: |v_t - v_{t-1}|
action delta: |a_t - a_{t-1}|
command response overshoot/settling time in play/sim2sim
```

3. If needed, add small scheduled penalties, not large always-on terms.

```text
body velocity jerk penalty
command-conditioned tracking smoothness penalty
slightly stronger action_rate only after responsiveness is confirmed
```

Avoid:

```text
Do not narrow tracking sigma early.
Do not increase action_rate heavily.
Do not use aggregate reward only.
```

### Goal 2: Head / Camera Stability And Directional Accuracy

Target:

```text
Reduce three-axis head/base angular oscillation.
Keep camera accurately facing the commanded velocity direction.
```

Current proxy metrics:

```text
base_angular_velocity
projected_gravity oscillation
waist_roll_vel / waist_pitch_vel / waist_yaw_vel
flat_orientation_l2 / torso_flat_orient
```

Needed improvements:

1. Add explicit head/camera metrics if the model exposes a head or camera frame.

```text
head roll/pitch/yaw rate
head orientation error relative to desired heading
camera optical axis vs commanded velocity direction
```

2. If only base/waist proxies are available, start with small penalties:

```text
base_ang_vel_xy damping
waist_roll/pitch velocity damping, small increments only
heading alignment penalty for nonzero linear command
```

3. Keep yaw behavior command-aware.

```text
When cmd_wz != 0, allow commanded yaw.
When cmd_wz == 0 and linear cmd active, penalize parasitic yaw.
When standing, penalize all body motion gently/scheduled.
```

Avoid:

```text
Do not over-damp waist yaw; yaw may be needed for rotation.
Do not add large always-on orientation penalties that make falling attractive.
```

### Goal 3: Reduce Foot Contact Force And Low-Speed Motion Frequency

Target:

```text
Lower indoor deployment noise.
Reduce impact forces and excessive stepping cadence at low speed.
```

First add metrics:

```text
foot contact force / impulse per step if sensors expose it
contact switch frequency
step frequency under low-speed commands
foot vertical velocity at touchdown
feet_slide
feet_clearance
```

Potential reward directions:

1. Contact softness penalty

```text
Penalty on high foot contact force or vertical foot velocity at first contact.
Use clipped/saturating form to avoid huge gradients.
```

2. Low-speed cadence reduction

```text
Command-conditioned gait frequency or step-count penalty when |cmd_xy| is small.
Do not suppress stepping entirely, or dead zone may return.
```

3. Foot clearance tuning

```text
Lower excessive clearance at low speed but preserve obstacle/terrain robustness if needed.
Scale target clearance with command speed.
```

4. Deploy-side smoothing as a non-training knob

```text
Action smoothing can reduce audible jitter but may reduce responsiveness.
Evaluate separately from training changes.
```

Avoid:

```text
Do not remove gait reward at low speed; V14c showed that can cause falls.
Do not penalize contacts blindly; humanoid walking requires contact.
Penalize impact magnitude and unnecessary frequency, not contact existence.
```

## Proposed Future Version Plan

V21 should be diagnostic, not yet aggressive optimization.

```text
V21a: V20l + mode-conditioned metrics only. No reward changes.
V21b: V20l + head/camera stability metrics only. No reward changes.
V21c: V20l + foot contact/cadence metrics only. No reward changes.
```

Then choose rewards based on measured dominant issue:

```text
V22a: tracking smoothness reward if velocity jerk/overshoot dominates.
V22b: camera/head damping reward if head oscillation dominates.
V22c: contact softness/cadence reward if indoor noise dominates.
```

Only after metrics identify the active failure should combine them:

```text
V23: minimal combined deployment-polish reward stack.
```

## Immediate Next Step

1. Preserve V20l checkpoint and sim2sim videos/logs as milestone artifacts.
2. Add mode-conditioned evaluation metrics before changing rewards.
3. Run targeted ablations A1/A3 to confirm which V20l modules truly carry the success.
4. Start deployment-polish optimization only after ablation results are known.
