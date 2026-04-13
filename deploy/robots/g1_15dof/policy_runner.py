"""G1 15-DOF policy runner for real-hardware deployment.

Mirrors the official C++ deployment pipeline (State_RLBase / ObservationManager /
ActionManager / JointPositionAction) in pure Python.

What this script handles
------------------------
* Loading ``deploy.yaml`` produced by ``export_deploy_cfg`` after training.
* Converting raw proprioceptive signals to the scaled, clipped, history-stacked
  observation vector expected by the policy.
* Converting raw policy outputs to joint position targets together with the
  corresponding SDK motor IDs, stiffness and damping values.

What the caller is responsible for
------------------------------------
* Reading sensor data from the robot and providing it as a dict.
* Running ONNX / TorchScript inference on the observation vector.
* Sending the returned joint position targets to the hardware.
* Providing velocity commands (joystick / planner / keyboard).

Usage
-----
    runner = PolicyRunner("path/to/deploy.yaml")
    runner.reset()

    # in the control loop (50 Hz):
    obs = runner.compute_observation(sensor_data, velocity_command)
    raw_action = model.run(obs)           # your inference call
    ctrl = runner.process_action(raw_action)
    # send ctrl["joint_pos_target"][i] to sdk_motor ctrl["sdk_joint_ids"][i]
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q = [w, x, y, z].

    Equivalent to C++: ``q.conjugate() * v`` (body-frame projection).
    """
    w, x, y, z = q
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return np.array([
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    ], dtype=np.float32)


class _ObsBuffer:
    """Per-term FIFO history buffer.

    Matches the C++ ``ObservationTermCfg::buff_`` (deque, oldest first).
    """

    def __init__(
        self,
        history_length: int,
        scale: Optional[list[float]],
        clip: Optional[list[float]],
    ):
        self.history_length = history_length
        self.scale = np.array(scale, dtype=np.float32) if scale else None
        # clip format: [lo, hi] single pair for the whole vector
        self.clip = clip
        self._buf: collections.deque[np.ndarray] = collections.deque(
            maxlen=history_length
        )

    # ------------------------------------------------------------------
    def _process(self, obs: np.ndarray) -> np.ndarray:
        """Apply clip then scale (isaaclab default: clip first)."""
        obs = obs.astype(np.float32)
        if self.clip is not None:
            obs = np.clip(obs, self.clip[0], self.clip[1])
        if self.scale is not None:
            obs = obs * self.scale
        return obs

    def reset(self, first_obs: np.ndarray) -> None:
        """Fill history with the first observation (as in C++ reset)."""
        processed = self._process(first_obs)
        self._buf.clear()
        for _ in range(self.history_length):
            self._buf.append(processed.copy())

    def add(self, obs: np.ndarray) -> None:
        """Push a new observation into the buffer (oldest dropped automatically)."""
        self._buf.append(self._process(obs))

    def get_flat(self) -> np.ndarray:
        """Return full history concatenated oldest→newest (matches C++ term.get())."""
        return np.concatenate(list(self._buf))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PolicyRunner:
    """Python equivalent of the C++ ManagerBasedRLEnv for the G1 15-DOF policy.

    Parameters
    ----------
    deploy_yaml_path:
        Path to the ``deploy.yaml`` file generated during training.
    """

    # Gravity vector in world frame (downward).
    _GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def __init__(self, deploy_yaml_path: str | Path) -> None:
        with open(deploy_yaml_path) as f:
            self._cfg = yaml.safe_load(f)

        # ---- joint mapping -------------------------------------------------
        # joint_ids_map[sim_idx] = sdk_motor_id
        # Used to read joint_pos / joint_vel from hardware (sdk order → sim order).
        self._joint_ids_map: list[int] = self._cfg["joint_ids_map"]
        self._n_joints = len(self._joint_ids_map)  # total DOF in sim (29)

        # ---- default joint positions (sim order) ---------------------------
        self._default_joint_pos = np.array(
            self._cfg["default_joint_pos"], dtype=np.float32
        )

        # ---- action term configuration -------------------------------------
        action_cfg = self._cfg["actions"]["JointPositionAction"]
        self._action_scale = np.array(action_cfg["scale"], dtype=np.float32)
        self._action_offset = np.array(action_cfg["offset"], dtype=np.float32)
        self._action_clip = action_cfg.get("clip")  # None or [[lo,hi], ...]
        self._action_dim = len(self._action_scale)  # 15 for 15-DOF

        # Indices into the 29-DOF sim array that the policy controls.
        # joint_ids == null → all joints; otherwise a subset.
        raw_jids = action_cfg.get("joint_ids")
        if raw_jids is None:
            self._action_joint_ids: list[int] = list(range(self._n_joints))
        else:
            self._action_joint_ids = raw_jids  # e.g. [0,1,2,...,14]

        # SDK motor IDs for the controlled joints (for the caller to send cmd).
        self._ctrl_sdk_ids: list[int] = [
            self._joint_ids_map[sim_i] for sim_i in self._action_joint_ids
        ]

        # Stiffness / damping in SDK order (full 29-DOF robot from FixStand).
        self._stiffness_sdk = self._cfg.get("stiffness", [])
        self._damping_sdk   = self._cfg.get("damping",   [])

        # Stiffness / damping for the 15 controlled joints.
        self._ctrl_kp = [self._stiffness_sdk[sdk_id] for sdk_id in self._ctrl_sdk_ids]
        self._ctrl_kd = [self._damping_sdk[sdk_id]   for sdk_id in self._ctrl_sdk_ids]

        # ---- observation term buffers --------------------------------------
        obs_cfg = self._cfg["observations"]
        self._obs_order: list[str] = list(obs_cfg.keys())  # preserve yaml order
        self._obs_bufs: dict[str, _ObsBuffer] = {}
        for name, tcfg in obs_cfg.items():
            self._obs_bufs[name] = _ObsBuffer(
                history_length=int(tcfg.get("history_length", 1)),
                scale=tcfg.get("scale"),
                clip=tcfg.get("clip"),
            )

        # ---- velocity command limits ---------------------------------------
        cmd_ranges = (
            self._cfg.get("commands", {})
            .get("base_velocity", {})
            .get("ranges", {})
        )
        self._vx_range = cmd_ranges.get("lin_vel_x", [-10.0, 10.0])
        self._vy_range = cmd_ranges.get("lin_vel_y", [-10.0, 10.0])
        self._wz_range = cmd_ranges.get("ang_vel_z", [-10.0, 10.0])

        # ---- runtime state ------------------------------------------------
        self._last_raw_action = np.zeros(self._action_dim, dtype=np.float32)
        self._step_dt: float = float(self._cfg.get("step_dt", 0.02))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Number of joints controlled by the policy (15)."""
        return self._action_dim

    @property
    def ctrl_sdk_ids(self) -> list[int]:
        """SDK motor IDs of the 15 controlled joints."""
        return self._ctrl_sdk_ids

    @property
    def ctrl_kp(self) -> list[float]:
        """Position gain (kp) for each controlled joint, sdk-ordered."""
        return self._ctrl_kp

    @property
    def ctrl_kd(self) -> list[float]:
        """Damping gain (kd) for each controlled joint, sdk-ordered."""
        return self._ctrl_kd

    def reset(self, sensor_data: dict, velocity_command: Sequence[float]) -> None:
        """Reset history buffers.

        Call once at the start of every episode / standing-to-walking transition.
        Matches C++ ``env->reset()`` which fills every obs buffer with the
        current observation.

        Parameters
        ----------
        sensor_data:
            Same format as :meth:`compute_observation`.
        velocity_command:
            [vx, vy, wz] in m/s and rad/s.
        """
        self._last_raw_action[:] = 0.0
        raw_obs = self._compute_raw_obs(sensor_data, velocity_command)
        for name, buf in self._obs_bufs.items():
            buf.reset(raw_obs[name])

    def compute_observation(
        self,
        sensor_data: dict,
        velocity_command: Sequence[float],
    ) -> np.ndarray:
        """Compute the policy observation vector from raw sensor data.

        Parameters
        ----------
        sensor_data : dict with keys:
            ``gyroscope``      — ``[wx, wy, wz]`` angular velocity in body frame (rad/s).
            ``quaternion``     — ``[w, x, y, z]`` base orientation quaternion.
            ``joint_pos_sdk``  — ``list[float]`` length 29, joint positions indexed
                                  by **SDK motor ID** (motor_state[i].q()).
            ``joint_vel_sdk``  — ``list[float]`` length 29, joint velocities indexed
                                  by **SDK motor ID** (motor_state[i].dq()).
        velocity_command : sequence of 3 floats
            ``[vx (m/s), vy (m/s), wz (rad/s)]`` desired base velocity.

        Returns
        -------
        np.ndarray
            Flat float32 observation vector ready for policy inference.
            Shape: (sum(term_dim * history_length) for all obs terms,)
            For 15-DOF: (3+3+3+29+29+15) * 5 = 410 elements.
        """
        raw_obs = self._compute_raw_obs(sensor_data, velocity_command)
        for name, buf in self._obs_bufs.items():
            buf.add(raw_obs[name])

        # Concatenate per-term histories (oldest → newest), then next term
        # This matches C++ default (use_gym_history=false):
        #   for term in terms: obs.extend(term.get_flat())
        parts = [self._obs_bufs[name].get_flat() for name in self._obs_order]
        return np.concatenate(parts)

    def process_action(self, raw_action: np.ndarray) -> dict:
        """Convert raw policy output to joint position targets.

        Parameters
        ----------
        raw_action : np.ndarray, shape (15,)
            Raw scalar outputs from the policy network.

        Returns
        -------
        dict with keys:
            ``joint_pos_target`` — np.ndarray (15,) position targets (rad).
                                   Apply: ``motor_cmd[sdk_id].q = target``.
            ``sdk_joint_ids``    — list[int] (15,) SDK motor indices matching
                                   ``joint_pos_target``.
            ``kp``               — list[float] (15,) position gains.
            ``kd``               — list[float] (15,) damping gains.
        """
        raw = np.asarray(raw_action, dtype=np.float32)

        # processed = raw * scale + offset  (matches C++ JointAction::process_actions)
        processed = raw * self._action_scale + self._action_offset

        # Optional per-joint clipping
        if self._action_clip is not None:
            clip = np.array(self._action_clip, dtype=np.float32)
            processed = np.clip(processed, clip[:, 0], clip[:, 1])

        # Store for last_action observation on the NEXT step
        self._last_raw_action = raw.copy()

        return {
            "joint_pos_target": processed,
            "sdk_joint_ids":    self._ctrl_sdk_ids,
            "kp":               self._ctrl_kp,
            "kd":               self._ctrl_kd,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sdk_to_sim_joint_pos(self, joint_pos_sdk: Sequence[float]) -> np.ndarray:
        """Re-index joint positions from SDK order to simulation order.

        joint_ids_map[sim_idx] = sdk_id  →  joint_pos_sim[i] = sdk[joint_ids_map[i]]
        """
        sdk = np.asarray(joint_pos_sdk, dtype=np.float32)
        return sdk[self._joint_ids_map]

    def _compute_raw_obs(
        self,
        sensor_data: dict,
        velocity_command: Sequence[float],
    ) -> dict[str, np.ndarray]:
        """Compute unscaled/unclipped raw observation vectors for each term."""

        # --- re-index joints to simulation order ---------------------------
        joint_pos_sim = self._sdk_to_sim_joint_pos(sensor_data["joint_pos_sdk"])
        joint_vel_sim = self._sdk_to_sim_joint_pos(sensor_data["joint_vel_sdk"])

        # --- base_ang_vel (gyroscope already in body frame) ----------------
        ang_vel_b = np.asarray(sensor_data["gyroscope"], dtype=np.float32)

        # --- projected_gravity ---------------------------------------------
        # C++: projected_gravity_b = quat.conjugate() * GRAVITY_W
        q = np.asarray(sensor_data["quaternion"], dtype=np.float32)  # [w,x,y,z]
        q_conj = _quat_conjugate(q)
        proj_grav = _quat_rotate_vector(q_conj, self._GRAVITY_W)

        # --- velocity_commands (clamped to configured ranges) --------------
        vc = velocity_command
        cmd = np.array([
            float(np.clip(vc[0], self._vx_range[0], self._vx_range[1])),
            float(np.clip(vc[1], self._vy_range[0], self._vy_range[1])),
            float(np.clip(vc[2], self._wz_range[0], self._wz_range[1])),
        ], dtype=np.float32)

        # --- joint_pos_rel -------------------------------------------------
        joint_pos_rel = joint_pos_sim - self._default_joint_pos

        # --- joint_vel_rel (raw joint velocities; scale applied in buffer) -
        joint_vel_rel = joint_vel_sim.copy()

        # --- last_action (raw policy output, NOT processed) ----------------
        last_action = self._last_raw_action.copy()

        return {
            "base_ang_vel":       ang_vel_b,
            "projected_gravity":  proj_grav,
            "velocity_commands":  cmd,
            "joint_pos_rel":      joint_pos_rel,
            "joint_vel_rel":      joint_vel_rel,
            "last_action":        last_action,
        }


# ---------------------------------------------------------------------------
# Quick sanity check (no hardware required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os

    yaml_path = sys.argv[1] if len(sys.argv) > 1 else None

    if yaml_path is None or not Path(yaml_path).exists():
        print("Usage: python policy_runner.py <path/to/deploy.yaml>")
        print()
        print("Running built-in dim check with synthetic deploy.yaml ...")

        # Build a minimal synthetic deploy.yaml that matches 15-DOF setup.
        JOINT_IDS_MAP_29 = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                             4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        N_ALL = 29
        # Legs+waist are sim indices 0-14 (see velocity_env_cfg comments):
        # sim0=L_hip_pitch, 1=R_hip_pitch, 2=waist_yaw,
        # 3=L_hip_roll,    4=R_hip_roll,   5=waist_roll,
        # 6=L_hip_yaw,     7=R_hip_yaw,    8=waist_pitch,
        # 9=L_knee,        10=R_knee,
        # 11=L_shoulder_pitch(*arm→skip), ...
        # Actually legs+waist are identified by joint pattern. Approximate below.
        CTRL_SIM_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18]  # 15
        DEFAULT_POS = [0.0] * 29
        DEFAULT_POS[0] = DEFAULT_POS[1] = -0.1
        DEFAULT_POS[9] = DEFAULT_POS[10] = 0.3
        DEFAULT_POS[13] = DEFAULT_POS[14] = -0.2

        ctrl_defaults = [DEFAULT_POS[i] for i in CTRL_SIM_IDS]
        stiffness = ([100,100,200,100,100,200,100,100,200,150,150,40,40,40,40]
                     + [40]*14)
        damping   = ([2,2,5,2,2,5,2,2,5,4,4,2,2,2,2] + [10]*14)

        cfg = {
            "joint_ids_map": JOINT_IDS_MAP_29,
            "step_dt": 0.02,
            "stiffness": stiffness,
            "damping": damping,
            "default_joint_pos": DEFAULT_POS,
            "commands": {
                "base_velocity": {
                    "ranges": {
                        "lin_vel_x": [-1.0, 2.0],
                        "lin_vel_y": [-0.5, 0.5],
                        "ang_vel_z": [-0.6, 0.6],
                    }
                }
            },
            "actions": {
                "JointPositionAction": {
                    "scale":  [0.25] * 15,
                    "offset": ctrl_defaults,
                    "clip":   None,
                    "joint_ids": CTRL_SIM_IDS,
                }
            },
            "observations": {
                "base_ang_vel":      {"scale": [0.2]*3,    "clip": None, "history_length": 5},
                "projected_gravity": {"scale": [1.0]*3,    "clip": None, "history_length": 5},
                "velocity_commands": {"scale": [1.0]*3,    "clip": None, "history_length": 5, "params": {"command_name": "base_velocity"}},
                "joint_pos_rel":     {"scale": [1.0]*N_ALL,"clip": None, "history_length": 5},
                "joint_vel_rel":     {"scale": [0.05]*N_ALL,"clip": None,"history_length": 5},
                "last_action":       {"scale": [1.0]*15,   "clip": None, "history_length": 5},
            },
        }

        tmp = "./_test_deploy.yaml"
        with open(tmp, "w") as f:
            yaml.dump(cfg, f)
        yaml_path = tmp

    runner = PolicyRunner(yaml_path)

    # Synthetic sensor data
    sensor = {
        "gyroscope":     [0.01, -0.01, 0.0],
        "quaternion":    [1.0, 0.0, 0.0, 0.0],   # [w,x,y,z] upright
        "joint_pos_sdk": [0.0] * 29,
        "joint_vel_sdk": [0.0] * 29,
    }
    cmd = [0.5, 0.0, 0.0]

    runner.reset(sensor, cmd)
    obs = runner.compute_observation(sensor, cmd)

    # 15dof: (3+3+3+29+29+15)*5 = 82*5 = 410
    print(f"Observation shape: {obs.shape}  (expected 410 for 15-DOF with history=5)")
    assert obs.shape == (410,), f"Shape mismatch: {obs.shape}"

    raw_act = np.zeros(runner.action_dim, dtype=np.float32)
    ctrl = runner.process_action(raw_act)
    print(f"Action dim:        {runner.action_dim}  (expected 15)")
    print(f"Ctrl SDK IDs:      {ctrl['sdk_joint_ids']}")
    print(f"Joint pos targets: {ctrl['joint_pos_target']}")
    print()
    print("All checks passed.")
