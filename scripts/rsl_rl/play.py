# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # Try rsl-rl 2.3 (alg.policy), rsl-rl 2.2 (alg.actor_critic), and
    # custom runners (e.g. VelocityEstimatorPPO) that expose alg.actor.
    if hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    elif hasattr(runner.alg, "actor_critic"):
        policy_nn = runner.alg.actor_critic
    elif hasattr(runner.alg, "actor"):
        policy_nn = runner.alg.actor
    else:
        raise AttributeError(f"Cannot find policy module in runner.alg ({type(runner.alg).__name__})")

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # For models with as_onnx() (e.g. TransformerLatentModel), use the model's own
    # export wrapper so that history_encoder / velocity_head are included and
    # output_names reflect any extra outputs (e.g. "velocity_pred").
    if hasattr(policy_nn, "as_onnx"):
        onnx_model = policy_nn.as_onnx()
        onnx_model.cpu().eval()
        os.makedirs(export_model_dir, exist_ok=True)
        dummy = onnx_model.get_dummy_inputs()
        torch.onnx.export(
            onnx_model,
            dummy,
            os.path.join(export_model_dir, "policy.onnx"),
            export_params=True,
            opset_version=18,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            dynamic_axes={},
        )
        print(f"[INFO] Exported TransformerLatentModel ONNX -> {export_model_dir}/policy.onnx")
        print(f"[INFO] ONNX inputs={onnx_model.input_names}, outputs={onnx_model.output_names}")
    else:
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0

    # --- velocity statistics ---
    vel_sum = None   # accumulated sum of root_lin_vel_b + root_ang_vel_b (env 0)
    vel_count = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # accumulate velocity for env 0
        try:
            robot = env.unwrapped.scene["robot"]
            lin_vel = robot.data.root_lin_vel_b[0].cpu()   # [3]
            ang_vel = robot.data.root_ang_vel_b[0].cpu()   # [3]
            vel_step = torch.cat([lin_vel, ang_vel])       # [vx,vy,vz,wx,wy,wz]
            if vel_sum is None:
                vel_sum = vel_step.clone()
            else:
                vel_sum += vel_step
            vel_count += 1
        except Exception:
            pass

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # --- print velocity statistics ---
    if vel_count > 0 and vel_sum is not None:
        vel_mean = vel_sum / vel_count
        print("\n[VelStats] Steps collected:", vel_count)
        print(f"[VelStats] Mean linear  vel (body frame): "
              f"vx={vel_mean[0]:.4f}  vy={vel_mean[1]:.4f}  vz={vel_mean[2]:.4f}  m/s")
        print(f"[VelStats] Mean angular vel (body frame): "
              f"wx={vel_mean[3]:.4f}  wy={vel_mean[4]:.4f}  wz={vel_mean[5]:.4f}  rad/s")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
