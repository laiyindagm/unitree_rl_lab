# Unitree RL Lab Agent Handoff

Last updated: 2026-04-27

This file is for a new agent after server migration. Read it before changing training configs, rewards, command sampling, or deploy code. The active project is a Unitree G1 15-DOF humanoid locomotion policy in IsaacLab/RSL-RL. The current research problem is pure `vx/vy` sim2sim failure under mode-token locomotion while preserving stable zero-command standing and pure yaw rotation.

## Quick Start For New Agent

Repository root:

```text
/root/workspace/unitree_rl_lab
```

Companion IsaacLab root in current workspace:

```text
/root/IsaacLab
```

Usual Python env:

```bash
conda activate env_isaaclab
```

Install/list/train commands:

```bash
./unitree_rl_lab.sh -i
./unitree_rl_lab.sh -l
python scripts/rsl_rl/train.py --task Unitree-G1-15dof-Velocity-Rot-V20l --headless
python scripts/rsl_rl/play.py --task Unitree-G1-15dof-Velocity-Rot-V20l
```

Current next experiment:

```text
Unitree-G1-15dof-Velocity-Rot-V20l
```

Do not propose removing tokens as the first fix. V20i already tested that and failed the current user's requirements.

## Top-Level Structure

```text
source/unitree_rl_lab/     Python package and task configs
deploy/                    C++ sim2sim / ONNX deployment code
scripts/rsl_rl/            train.py, play.py, CLI helpers
logs/rsl_rl/               RSL-RL experiment logs
outputs/                   timestamped Hydra/IsaacLab outputs
unitree_ros/               Unitree descriptions and ROS materials
docker/                    Docker files
doc/ and docs/             docs/licenses
```

Primary Python package:

```text
source/unitree_rl_lab/unitree_rl_lab/
```

Important subtrees:

```text
tasks/locomotion/mdp/                         rewards, observations, commands, curriculums
tasks/locomotion/agents/rsl_rl_ppo_cfg.py     RSL-RL runner configs
tasks/locomotion/robots/g1/15dof_rot/         primary experiment family
utils/                                        custom PPO/model/export utilities
```

Primary experiment directory:

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/
```

Task registry:

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/__init__.py
```

## Key Files

Current/recent configs:

```text
velocity_env_cfg_rot_v19f.py   stable V19 base: zero-speed fixed, wz responsive, pure xy weak
velocity_env_cfg_rot_v20c.py   V19f + 5-mode gait token
velocity_env_cfg_rot_v20f.py   V20c + standard linear tracking, removes lin rotation-skip blind spot
velocity_env_cfg_rot_v20g.py   V20f + standard yaw tracking, removes yaw straight-walk skip
velocity_env_cfg_rot_v20i.py   V20f without token; tested and failed current direction
velocity_env_cfg_rot_v20j.py   V20f + 3-mode token; failed because yaw skip remained
velocity_env_cfg_rot_v20k.py   V20i + standard yaw tracking; exists but no token, not current next
velocity_env_cfg_rot_v20l.py   CURRENT NEXT: V20g rewards + V20j 3-mode token
```

MDP files:

```text
mdp/observations.py
mdp/rewards.py
mdp/commands/velocity_command.py
mdp/curriculums.py
```

Important observation functions:

```text
observations.py:gait_mode_token      returns (N,5): standing, pure_vx, pure_vy, pure_wz, joint
observations.py:gait_mode_token_3    returns (N,3): standing, pure_wz, other
```

Important reward functions:

```text
rewards.py:track_ang_vel_z_rotating_aware   old yaw tracker; skips straight walk and gives free yaw reward
rewards.py:track_lin_vel_xy_rotation_skip   old lin tracker; skips pure rotation and gives free lin reward
rewards.py:wz_proportional_penalty          active only if abs(cmd_wz) > threshold; no penalty at cmd_wz=0
rewards.py:zero_cmd_body_vel                standing body velocity penalty
rewards.py:zero_cmd_body_vel_scheduled      scheduled version to avoid suicide policies
```

Deploy files:

```text
deploy/robots/g1_15dof_keyboard/src/State_RLBase.cpp
deploy/include/isaaclab/manager/observation_manager.h
deploy/robots/g1_15dof_keyboard/CMakeLists.txt
deploy/robots/g1_15dof_keyboard/build/
```

C++ deploy observation registry currently includes:

```text
REGISTER_OBSERVATION(velocity_commands)
REGISTER_OBSERVATION(gait_mode_token)    # 5 dims
REGISTER_OBSERVATION(gait_mode_token_3)  # 3 dims
```

## Current Status: V20l

V20l is the next intended experiment.

```text
Gym ID: Unitree-G1-15dof-Velocity-Rot-V20l
File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v20l.py
Registration: 15dof_rot/__init__.py in the V20 block
```

Verified implementation facts:

```text
V20l imports RobotEnvCfg from velocity_env_cfg_rot_v20g as V20gEnvCfg.
V20l imports ObservationsCfg from velocity_env_cfg_rot_v20j.
V20l class RobotEnvCfg(V20gEnvCfg) overrides observations only.
V20g RewardsCfg replaces both yaw trackers with mdp.track_ang_vel_z_exp.
V20j ObservationsCfg uses gait_mode_token_3 in both PolicyCfg and CriticCfg.
```

Therefore:

```text
V20l = V20g rewards + V20j observations
     = standard yaw tracking + 3-mode isolation {standing, pure_wz, other}
```

This preserves the user's hard constraint:

```text
0-command, pure_w, and other must remain isolated at minimum.
```

## Why V20l Exists

Recent results:

```text
V20j @ ~14.4k:
  Mean reward ~76-77
  error_vel_xy ~0.46-0.50
  error_vel_yaw ~0.46
  waist_roll_vel ~-0.18
  pure vx/vy not improved; joint walking degraded

V20i @ ~9.8k:
  Mean reward 74.86
  track_lin_vel_xy 0.6553
  error_vel_xy 0.6074
  error_vel_yaw 0.4736
  waist_roll_vel -0.1792
  zero-command unstable; walking similar to old V19 behavior
```

Interpretation:

1. Removing token is not the answer. V20i reproduced V19-like walking and made zero-command worse.
2. The problem is not token existence. The pure `vx/vy` training subdistribution itself is defective.
3. V20j kept 3-mode isolation but inherited V20f's yaw skip, so pure_xy still had no yaw supervision and contaminated the shared `other` branch.

Current root-cause hypothesis:

```text
pure_vx/pure_vy envs: cmd_wz == 0 by construction.
joint envs: often include cmd_wz != 0.
```

With `track_ang_vel_z_rotating_aware`, straight-walk commands (`cmd_lin_norm > 0.1`, `abs(cmd_wz) < 0.05`) get yaw tracking reward 1.0 regardless of actual yaw rate. With main + sharp yaw trackers, that is +4.0 free yaw reward with zero gradient toward `actual_wz -> 0`.

`wz_proportional_penalty` does not fix this because it has `has_cmd = cmd_wz_abs > cmd_threshold`; at `cmd_wz=0` it returns 0. Earlier reasoning that it punished pure_vx was wrong.

Therefore pure_xy samples can learn a wobble/yaw/waist strategy that helps satisfy linear tracking while remaining yaw-unsupervised. Joint samples often include nonzero yaw command, so they receive real yaw gradients and train better.

V20l keeps isolation but removes the yaw skip by inheriting V20g's standard `track_ang_vel_z_exp` yaw rewards.

## Next Training And Decision Criteria

Run:

```bash
python scripts/rsl_rl/train.py --task Unitree-G1-15dof-Velocity-Rot-V20l --headless
```

Do not judge by mean reward alone. V20i/V20j had acceptable reward numbers while behavior was bad.

Primary metrics:

```text
Episode_Reward/waist_roll_vel
Metrics/base_velocity/error_vel_yaw
Metrics/base_velocity/error_vel_xy
Episode_Termination/bad_orientation
Episode_Reward/track_lin_vel_xy
Episode_Reward/track_ang_vel_z
Episode_Reward/track_ang_vel_z_sharp
Episode_Reward/wz_proportional
```

Expected V20l trajectory:

```text
By iter 5k:
  bad_orientation < 8%; mean reward > 55.

By iter 8k:
  waist_roll_vel should improve from about -0.18 toward > -0.10.
  error_vel_yaw should drop below ~0.35.

By iter 12k:
  error_vel_xy should trend toward <0.35-0.40.
  Standing should be better than V20i because standing is isolated again.
  Joint walking should recover vs V20j.
```

Kill conditions:

```text
iter 5k: bad_orientation > 8% or Mean reward < 55.
iter 8k: waist_roll_vel < -0.15 AND error_vel_yaw > 0.45.
iter 12k: joint good but pure_xy still fails in sim2sim.
```

If V20l fails with joint good but pure_xy still bad, do not change token structure first. Add an explicit no-yaw penalty active only for linear commands with `cmd_wz≈0`, e.g. condition `(cmd_lin_norm > 0.1) & (abs(cmd_wz) < 0.05)` and penalize `abs(actual_wz)` or `actual_wz^2`. Keep 3-mode isolation.

## Command Sampling Model

Command generator of interest:

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/velocity_command.py
```

Relevant proportions:

```text
rel_standing_envs
rel_pure_vx_envs
rel_pure_vy_envs
rel_rotating_envs / pure_wz
remaining fraction = joint envs
```

Important current V19f-style distribution:

```text
standing 0.10
pure_vx 0.15
pure_vy 0.15
pure_wz/rotating 0.25
joint 0.35
```

Pure env type assignment zeroes other axes:

```text
pure_vx: vy -> 0, wz -> 0
pure_vy: vx -> 0, wz -> 0
pure_wz: vx -> 0, vy -> 0
standing: all zero
joint: normally multiple axes active
```

This is central to the pure_xy diagnosis: pure_vx/pure_vy always have `cmd_wz=0`, while joint often has `cmd_wz!=0`.

## Token Semantics

5-mode token:

```text
gait_mode_token(env) -> (N,5)
index 0 standing
index 1 pure_vx
index 2 pure_vy
index 3 pure_wz
index 4 joint
```

3-mode token:

```text
gait_mode_token_3(env) -> (N,3)
index 0 standing
index 1 pure_wz
index 2 other = pure_vx + pure_vy + joint
```

Current user requirement: keep at least 3-way isolation. Do not collapse all modes again without explicit user approval.

## Deploy / Sim2sim Notes

Observation manager macro:

```text
deploy/include/isaaclab/manager/observation_manager.h
```

Primary keyboard deploy source:

```text
deploy/robots/g1_15dof_keyboard/src/State_RLBase.cpp
```

Macro behavior: `REGISTER_OBSERVATION(name)` inserts `#name -> function` into a static map. Runtime lookup is by YAML observation key.

Critical V20j deploy bug that already happened:

```text
V20j initially had attribute name gait_mode_token = ObsTerm(func=mdp.gait_mode_token_3)
deploy.yaml exported key gait_mode_token
C++ returned 5 dims
ONNX was trained with 3 dims
deployment crashed immediately
```

Fix applied:

```text
V20j PolicyCfg and CriticCfg use attribute name gait_mode_token_3.
C++ State_RLBase.cpp registers gait_mode_token_3.
```

When adding any new observation variant, ensure:

```text
Python ObsTerm attribute name == deploy.yaml key == C++ REGISTER_OBSERVATION name
Returned dimension == ONNX input expectation
```

After editing C++ deploy code, rebuild:

```bash
cd /root/workspace/unitree_rl_lab/deploy/robots/g1_15dof_keyboard/build
cmake --build .
```

If deploying V20l, exported `deploy.yaml` should contain:

```yaml
gait_mode_token_3:
  params: {command_name: base_velocity, eps_x: 0.1, eps_y: 0.1, eps_w: 0.1}
```

If it contains `gait_mode_token` for a V20l/V20j 3-mode policy, deployment is wrong.

General sim2sim lessons:

- IsaacSim play can look OK while MuJoCo/C++ sim2sim twitches; PhysX-MuJoCo gap exposes marginal policies.
- `action_smoothing` in deploy config was previously used to reduce twitching. Lower alpha means more smoothing; alpha 1.0 means no smoothing.
- Observation history layout matters. Past verification: Python `flatten_history_dim=True` matches C++ `use_gym_history=false` term-major/history flattening.
- ONNX export is deterministic `actor(normalizer(x))`; if `empirical_normalization=False`, normalizer is effectively identity.

## Experiment Evolution Summary

### V2-V4: instability and overcomplication

- V2/V3 LCP attempts had NaN/fall failures from extreme observations, lack of clipping, and too many conflicting rewards.
- LCP was abandoned for this locomotion line.
- V4 symmetry/data augmentation degraded performance. rsl_rl data augmentation reused old log probs for mirrored observations and `use_mirror_loss=False` detached the loss.

### V5-V8: base recovery and dead zone discovery

- V5 returned closer to the working rot base and added DR. A stale memory once claimed V5c was best; later log audit found many V5c runs aborted early. Always check logs.
- V6 identified the low-speed dead zone: exp tracking with `std=0.5` gives high reward for doing nothing at small commands; action costs can dominate marginal tracking gain.
- V7 added large penalties and failed catastrophically. Too much always-on penalty teaches falling to terminate episodes.
- V8 adaptive sigma failed by making early training too hard. Waist velocity damping helped modestly.

### V9-V14: curriculum, easy reward, penalty budget

- V9 speed-bucketed curriculum initially stuck due to running-average accumulation; later fixed to snapshot/buffer style.
- `init_at_random_ep_len=True` spreads resets; curriculum code expecting many envs in one reset can fail.
- V10 bucketed curriculum worked but did not solve dead zone. Per-joint waist damping helped upper body.
- V11 adaptive sigma failed because it reduced early easy rewards; standing never stabilized.
- V13 action_rate attribution showed action_rate was not the sole root cause despite being far above official IsaacLab G1 values.
- V14 baselined tracking and speed-gated gait failed because they removed easy early signals.

### V15-V18: adaptive sampling and movement incentives

- V15 adaptive metric used exp reward and incorrectly treated standing at low command as good. This reduced sampling of hard low-speed bins.
- V16 redesigned adaptive sampling using relative accuracy and added scheduled movement incentive.
- V17 found softmax floor and zero-speed contamination bugs. `min_sampling_prob` above `1/N` can clamp distribution nearly uniform.
- V18 fixed adaptive sampling bugs and added pure-linear allocation, but pure xy remained hard.

### V19: marginal axis-independent sampling and stable base

- V19 introduced standing, pure_vx, pure_vy, pure_wz/rotating, and joint env types.
- V19f is the key stable baseline: trained to completion, zero-speed standing fixed, yaw responds from small command. Open issues: pure `vx/vy` small response and pure-rotation backward drift.
- V19g fixed some drift/regression but lost zero-speed standing.
- V19h/i piled standing penalties and created suicide-policy collapse. Heavy zero-cmd penalties must be scheduled.

### V20: mode token series and current conclusion

- V20a = V19i + 5-mode token. Failed because V19i base was already pathological.
- V20b = V19g + token. Deprecated because base lost zero-speed.
- V20c = V19f + 5-mode token. Clean A/B on best V19 base.
- V20d = V20c + pure rotation drift penalty. Too weak.
- V20e = V20c + sharper linear tracking.
- V20f = V20c + standard `track_lin_vel_xy_yaw_frame_exp`, removing lin rotation-skip blind spot. Pure xy still failed.
- V20g = V20f + standard `track_ang_vel_z_exp`, removing yaw straight-walk skip. Correct reward-side direction.
- V20h = V20f + more pure-linear sampling. Rejected: more zero-gradient samples do not help.
- V20i = V20f without token. Tested; zero command unstable and walking similar to V19. Not the answer.
- V20j = V20f + 3-mode token `{standing,pure_wz,other}`. Failed because it retained yaw skip; pure_xy contaminated joint under `other`.
- V20k = V20i + standard yaw tracking. Exists but has no token; not current next because user requires at least 3-way isolation.
- V20l = V20g + V20j 3-mode token. Current next experiment.

## Methodology Lessons

- Always verify logs, not memory labels. Inspect `logs/rsl_rl/...` and max `model_<N>.pt` before claiming a version is verified.
- Do mechanism-level diagnosis. The user rejects surface fixes such as density boosts when per-sample gradients are zero.
- Preserve easy early rewards. Adaptive sigma and baselined tracking failed because they removed easy standing/tracking reward early.
- Heavy always-on penalties can create suicide policies. Schedule them and compute per-step contribution.
- Token is not the root by itself. Token controls isolation/contamination; pure_xy failure is primarily reward/environment missing yaw supervision.
- Avoid long ladders of the same hypothesis. Each experiment should test a different mechanism.
- With IsaacLab `@configclass`, changing `class_type` after class creation does not update dataclass defaults. Override in `__post_init__`.
- `init_at_random_ep_len=True` desynchronizes resets; use robust curriculum accumulation.

## Tool / Editing Pitfalls

VS Code file tools have intermittently failed on this workspace, sometimes reporting success without creating files. Reliable workaround:

```bash
cat > path/to/file.py << 'PYEOF'
...
PYEOF
python3 -c "import ast; ast.parse(open('path/to/file.py').read()); print('OK')"
```

For existing files, exact Python string replacements via terminal have been more reliable than editor replace tools. Always syntax-check with `ast.parse` after edits.

Use `rg` / `rg --files` for search.

## CLP / LCP Side Branches

LCP:

- Early LCP variants degraded reward and stability; frozen std was too restrictive for rotation.
- Official LCP regularizes stochastic log-prob gradients, not deterministic action mean Jacobian.

CLP:

- Files: `utils/contrastive_latent_model.py`, `utils/contrastive_ppo.py`.
- Gym IDs: `Unitree-G1-15dof-Velocity-Rot-V19d-CLP`, `Unitree-G1-15dof-Velocity-Rot-V19d-CLP-Transformer`.
- CLP env uses history length 10 and latent model variants.
- Important CLP bug: `encode()` used normalized command for labels; fixed by extracting raw cmd before normalization.
- CLP is a side branch. Do not mix with V20 reward/token debugging unless explicitly requested.

## Migration Checklist

1. Copy both repos if possible:

```text
/root/workspace/unitree_rl_lab
/root/IsaacLab
```

2. Install/activate:

```bash
conda activate env_isaaclab
cd /root/workspace/unitree_rl_lab
./unitree_rl_lab.sh -i
```

3. Verify V20l registration:

```bash
./unitree_rl_lab.sh -l | grep Unitree-G1-15dof-Velocity-Rot-V20l
```

4. If registration fails, inspect:

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/__init__.py
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v20l.py
```

5. Train current next experiment:

```bash
python scripts/rsl_rl/train.py --task Unitree-G1-15dof-Velocity-Rot-V20l --headless
```

6. Preserve old logs if possible:

```text
logs/rsl_rl/
outputs/
```

## What Not To Do Next

- Do not remove token again as first response. V20i already tested that and user objected.
- Do not increase pure_xy sampling density alone. V20h was rejected because zero-gradient samples do not help.
- Do not add large always-on penalties to standing or zero-command envs.
- Do not assume a version is good from comments/docstrings; check logs and sim2sim reports.
- Do not deploy a 3-mode token policy with a deploy.yaml key named `gait_mode_token`.

## Short Mental Model

The project is trying to satisfy three competing behaviors:

```text
1. Stand still at zero command.
2. Rotate in place without linear drift.
3. Walk pure vx/vy and joint commands without yaw/waist wobble.
```

Latest insight: pure_vx/pure_vy fail because they are `cmd_wz=0` samples that previously got no real yaw supervision. Joint commands often include nonzero yaw and therefore train a better yaw controller. The correct direction is to keep meaningful mode isolation while making pure_xy reward supply real yaw-stability gradients.

V20l is designed to test exactly that.
