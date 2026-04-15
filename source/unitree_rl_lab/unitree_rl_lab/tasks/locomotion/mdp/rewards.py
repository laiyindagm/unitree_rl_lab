from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import yaw_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str = None,
) -> torch.Tensor:
    """Gait reward: reward matching expected stance/swing phase with actual foot contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1

    return reward


def backward_lean_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize backward body lean.

    projected_gravity_b[:, 0] > 0 corresponds to backward pitch (nose up).
    Returns the clamped positive component so only backward lean is penalized.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.projected_gravity_b[:, 0], min=0.0)


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Rotation-aware rewards (guide sections III & IV).
"""


def track_ang_vel_z_rotating_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    straight_walk_lin_threshold: float = 0.1,
    straight_walk_yaw_threshold: float = 0.05,
) -> torch.Tensor:
    """Yaw tracking that skips straight-walk envs.

    For straight walking (high lin cmd, near-zero yaw cmd), return 1.0 so this
    reward does not penalize natural drift.  For everything else, track normally.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw = cmd[:, 2]

    actual_yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    yaw_error = actual_yaw - cmd_yaw

    tracking = torch.exp(-yaw_error.square() / std**2)

    is_straight_walk = (cmd_lin_norm > straight_walk_lin_threshold) & (cmd_yaw.abs() < straight_walk_yaw_threshold)
    return torch.where(is_straight_walk, torch.ones_like(tracking), tracking)


def yaw_rate_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    straight_walk_lin_threshold: float = 0.1,
    straight_walk_yaw_threshold: float = 0.05,
) -> torch.Tensor:
    """Parasitic yaw penalty only active for straight-walk commands."""
    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw = cmd[:, 2]

    actual_yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    is_straight_walk = (cmd_lin_norm > straight_walk_lin_threshold) & (cmd_yaw.abs() < straight_walk_yaw_threshold)
    return actual_yaw.abs() * is_straight_walk


def _is_pure_rotation(env: ManagerBasedRLEnv, command_name: str, lin_threshold: float = 0.1, yaw_threshold: float = 0.05) -> torch.Tensor:
    """Helper: True where the command is pure-rotation (low lin, nonzero yaw)."""
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) < lin_threshold) & (cmd[:, 2].abs() > yaw_threshold)


def rotation_single_support_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward single-foot support during pure rotation — encourages stepping turn."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    # exactly one foot on the ground
    n_contacts = is_contact.sum(dim=-1)
    is_single = (n_contacts == 1).float()
    mask = _is_pure_rotation(env, command_name).float()
    return is_single * mask


def rotation_double_support_slide_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize foot sliding when both feet are on the ground during pure rotation."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    n_contacts = is_contact.sum(dim=-1)
    is_double = (n_contacts >= 2).float()

    # foot xy velocity magnitude
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = torch.norm(foot_vel, dim=-1).sum(dim=-1)

    mask = _is_pure_rotation(env, command_name).float()
    return slide * is_double * mask


def rotation_twist_joint_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize waist/hip yaw joint velocities when both feet contact ground during pure rotation.

    Discourages the robot from twisting its torso instead of stepping to rotate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    n_contacts = is_contact.sum(dim=-1)
    is_double = (n_contacts >= 2).float()

    twist_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    twist_penalty = torch.sum(twist_vel.abs(), dim=-1)

    mask = _is_pure_rotation(env, command_name).float()
    return twist_penalty * is_double * mask


# ---------------------------------------------------------------------------
# V7: Low-speed tracking, standstill stability, anti-oscillation
# ---------------------------------------------------------------------------


def velocity_mismatch_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Direct L1 penalty on velocity tracking error (lin_xy + ang_z).

    Unlike exp(-error²/σ²) which gives diminishing marginal reward at low
    speeds, L1 provides constant gradient proportional to the absolute error.
    This is critical for overcoming the low-speed dead zone where fixed
    action penalties exceed the marginal exp tracking reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.abs(asset.data.root_lin_vel_b[:, :2] - cmd[:, :2]), dim=1
    )
    ang_vel_error = torch.abs(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2])
    return lin_vel_error + ang_vel_error


def standstill_joint_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities when command is near zero.

    Directly fights standstill oscillation by damping joint motion.
    Unlike stand_still (which penalizes position deviation from default),
    this targets the oscillation dynamics -- any joint movement during
    standstill is penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    joint_vel_sum = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    return joint_vel_sum * (cmd_norm < 0.1)


def waist_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize waist joint angular velocities (always active).

    Targets the waist joints that cause head/torso lateral oscillation.
    Works in complement with joint_deviation_waists (position) by also
    damping the velocity of waist joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )


def action_rate_scaled_by_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    min_scale: float = 0.25,
) -> torch.Tensor:
    """Action rate penalty scaled by command magnitude.

    At high speed (|cmd| >= 1): full penalty for smooth motion.
    At low speed (|cmd| ~ 0.2): ~25% penalty to allow movement initiation.
    At standstill (|cmd| ~ 0):  min_scale * penalty for basic stability.

    This elegantly solves the low-speed dead zone where a fixed action_rate
    penalty exceeds the marginal tracking reward for small commands.
    """
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name), dim=1
    )
    scale = torch.clamp(cmd_norm, min=min_scale, max=1.0)
    action_diff_sq = torch.sum(
        torch.square(
            env.action_manager.action - env.action_manager.prev_action
        ),
        dim=1,
    )
    return action_diff_sq * scale


def track_lin_vel_xy_adaptive_sigma(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_min: float = 0.15,
    sigma_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track linear velocity with command-adaptive sigma.

    sigma = max(sigma_min, sigma_scale * cmd_norm).
    At low commands sigma shrinks, making the exp kernel much more
    discriminative and eliminating the dead zone below ~0.3 m/s.
    At high commands sigma equals the original value (0.5 at cmd=1.0).
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )[:, :2]
    cmd_norm = torch.norm(cmd, dim=1, keepdim=True).clamp(min=1e-6)
    sigma = torch.clamp(cmd_norm * sigma_scale, min=sigma_min)
    lin_vel_error = torch.sum(torch.square(cmd - vel), dim=1)
    return torch.exp(-lin_vel_error / sigma.squeeze(-1).square())


def track_ang_vel_z_adaptive_sigma(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_min: float = 0.15,
    sigma_scale: float = 0.5,
    straight_walk_lin_threshold: float = 0.1,
    straight_walk_yaw_threshold: float = 0.05,
) -> torch.Tensor:
    """Track angular velocity with command-adaptive sigma, rotation-aware.

    sigma = max(sigma_min, sigma_scale * |cmd_yaw|).
    Same straight-walk bypass as track_ang_vel_z_rotating_aware.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw = cmd[:, 2]

    actual_yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    yaw_error = (actual_yaw - cmd_yaw).square()

    sigma = torch.clamp(cmd_yaw.abs() * sigma_scale, min=sigma_min)
    tracking = torch.exp(-yaw_error / sigma.square())

    is_straight_walk = (cmd_lin_norm > straight_walk_lin_threshold) & (
        cmd_yaw.abs() < straight_walk_yaw_threshold
    )
    return torch.where(is_straight_walk, torch.ones_like(tracking), tracking)


def low_speed_tracking_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus reward for accurately tracking linear velocity at low command speeds.

    Returns linear accuracy: 1.0 at perfect tracking, 0.0 when error >= |cmd|.
    Only active when 0.05 < cmd_lin_norm < speed_threshold.

    Key property: standing still at cmd=0.2 gives accuracy=0 (no "free" reward),
    so this does NOT interfere with early stability training unlike adaptive sigma.
    Purely ADDITIVE: increases marginal gain of tracking at low speed.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )[:, :2]

    cmd_norm = torch.norm(cmd, dim=1)
    error = torch.norm(cmd - vel, dim=1)

    mask = (cmd_norm > 0.05) & (cmd_norm < speed_threshold)
    accuracy = torch.clamp(1.0 - error / cmd_norm.clamp(min=0.05), min=0.0)

    return accuracy * mask.float()


def low_speed_rotation_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    ang_threshold: float = 0.5,
) -> torch.Tensor:
    """Bonus reward for accurately tracking angular velocity at low command speeds.

    Same linear accuracy as low_speed_tracking_bonus but for yaw rotation.
    Only active when 0.05 < |cmd_yaw| < ang_threshold.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_yaw = cmd[:, 2]
    cmd_yaw_abs = cmd_yaw.abs()
    actual_yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error = (actual_yaw - cmd_yaw).abs()

    mask = (cmd_yaw_abs > 0.05) & (cmd_yaw_abs < ang_threshold)
    accuracy = torch.clamp(1.0 - error / cmd_yaw_abs.clamp(min=0.05), min=0.0)

    return accuracy * mask.float()


# ---------------------------------------------------------------------------
# V12: Scheduled sigma tracking -- global σ annealing over training iterations
# ---------------------------------------------------------------------------


def track_lin_vel_xy_scheduled_sigma(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_start: float = 0.5,
    sigma_end: float = 0.3,
    anneal_start_iter: int = 3000,
    anneal_end_iter: int = 8000,
    steps_per_iter: int = 24,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track linear velocity with globally scheduled sigma annealing.

    Unlike adaptive sigma (V11a) which varies σ per-sample by |cmd| and
    poisons early training, this anneals σ uniformly across ALL samples
    based on training iteration. Phase 1 (σ=σ_start) is identical to the
    proven V6c/V10d baseline, so early training is completely unaffected.

    Schedule:
      iter < anneal_start_iter:  σ = sigma_start  (standard training)
      anneal_start..anneal_end:  σ linearly decreases
      iter >= anneal_end_iter:   σ = sigma_end    (tighter tracking)
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )[:, :2]

    current_iter = env.common_step_counter // steps_per_iter
    if current_iter <= anneal_start_iter:
        sigma = sigma_start
    elif current_iter >= anneal_end_iter:
        sigma = sigma_end
    else:
        alpha = (current_iter - anneal_start_iter) / (anneal_end_iter - anneal_start_iter)
        sigma = sigma_start + alpha * (sigma_end - sigma_start)

    lin_vel_error = torch.sum(torch.square(cmd - vel), dim=1)
    return torch.exp(-lin_vel_error / (sigma ** 2))


def track_ang_vel_z_scheduled_sigma(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_start: float = 0.5,
    sigma_end: float = 0.25,
    anneal_start_iter: int = 3000,
    anneal_end_iter: int = 8000,
    steps_per_iter: int = 24,
    straight_walk_lin_threshold: float = 0.1,
    straight_walk_yaw_threshold: float = 0.05,
) -> torch.Tensor:
    """Track angular velocity with globally scheduled sigma annealing.

    Same straight-walk bypass as track_ang_vel_z_rotating_aware.
    sigma_end defaults to 0.25 (tighter than lin_vel's 0.3) because
    rotation requires more precise tracking to overcome the dead zone.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw = cmd[:, 2]
    actual_yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    yaw_error = (actual_yaw - cmd_yaw).square()

    current_iter = env.common_step_counter // steps_per_iter
    if current_iter <= anneal_start_iter:
        sigma = sigma_start
    elif current_iter >= anneal_end_iter:
        sigma = sigma_end
    else:
        alpha = (current_iter - anneal_start_iter) / (anneal_end_iter - anneal_start_iter)
        sigma = sigma_start + alpha * (sigma_end - sigma_start)

    tracking = torch.exp(-yaw_error / (sigma ** 2))

    is_straight_walk = (cmd_lin_norm > straight_walk_lin_threshold) & (
        cmd_yaw.abs() < straight_walk_yaw_threshold
    )
    return torch.where(is_straight_walk, torch.ones_like(tracking), tracking)


# ---------------------------------------------------------------------------
# V14: Dead-zone busting rewards
# ---------------------------------------------------------------------------


def cmd_nonresponse_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.15,
    vel_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when a non-trivial velocity command is active.

    Returns 1.0 when |cmd| > cmd_threshold AND |body_vel| < vel_threshold,
    i.e. the robot is ignoring the command.  Returns 0.0 otherwise.

    This creates a direct negative gradient against the "rational laziness"
    equilibrium where the Gaussian tracking kernel gives high free reward
    for standing still at low command speeds.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
    body_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) + asset.data.root_ang_vel_b[:, 2].abs()
    return ((cmd_norm > cmd_threshold) & (body_vel < vel_threshold)).float()


def rotation_double_support_slide_penalty_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_yaw_cmd: float = 0.3,
) -> torch.Tensor:
    """Same as rotation_double_support_slide_penalty but only active above min_yaw_cmd.

    At low yaw commands the robot is still learning *whether* to rotate;
    constraining *how* it rotates too early creates a dead zone.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    n_contacts = is_contact.sum(dim=-1)
    is_double = (n_contacts >= 2).float()

    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = torch.norm(foot_vel, dim=-1).sum(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = cmd[:, 2].abs()
    is_pure_rot = (cmd_lin_norm < 0.1) & (cmd_yaw_abs > min_yaw_cmd)
    return slide * is_double * is_pure_rot.float()


def rotation_twist_joint_penalty_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_yaw_cmd: float = 0.3,
) -> torch.Tensor:
    """Same as rotation_twist_joint_penalty but only active above min_yaw_cmd."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    n_contacts = is_contact.sum(dim=-1)
    is_double = (n_contacts >= 2).float()

    twist_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    twist_penalty = torch.sum(twist_vel.abs(), dim=-1)

    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = cmd[:, 2].abs()
    is_pure_rot = (cmd_lin_norm < 0.1) & (cmd_yaw_abs > min_yaw_cmd)
    return twist_penalty * is_double * is_pure_rot.float()
