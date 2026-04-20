import gymnasium as gym

gym.register(
    id="Unitree-G1-15dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V2a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v2a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v2a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofLCPPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V2b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v2b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v2b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V3a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofLCPPPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V3b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V3c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofLCPPPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V3d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v3d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofLCPPPORunnerV3dCfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V4a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v4a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v4a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV4Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V4b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v4b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v4b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV4Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V4c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v4c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v4c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV4Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V5a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V5b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV5bCfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V5c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V6a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v6a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v6a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V6b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v6b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v6b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V6c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v6c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v6c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V7a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v7a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v7a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V7b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v7b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v7b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V7c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v7c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v7c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V8a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v8a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v8a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V8b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v8b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v8b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V8c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v8c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v8c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V5c-TransformerAux",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v5c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofTransformerAuxPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# V9 — Three orthogonal directions
# V9a: Gait Signal Fix  (observation layer)
# V9b: Action Penalty Reshaping  (reward layer)
# V9c: Speed-Bucketed Curriculum  (curriculum layer)
# ---------------------------------------------------------------------------
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V9a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v9a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v9a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V9b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v9b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v9b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V9c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v9c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v9c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V10 — Bucketed Curriculum Bug Fix + Stronger Waist + Variants
# V10a: Fixed snapshot bucketed curriculum + waist(-0.08)
# V10b: V10a + aligned gait clock (walk_period=0.7)
# V10c: Manual iteration-based schedule (deterministic failsafe)
# ---------------------------------------------------------------------------
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V10a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V10b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V10c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V10d: Fixed bucketed curriculum + per-joint waist differentiation
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V10d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v10d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V11a: Adaptive sigma tracking (fix dead zone from reward side)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V11a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V11b: Scaled action rate (fix dead zone from penalty side)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V11b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V11c: Adaptive sigma + scaled action rate (fix dead zone from both sides)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V11c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V11d: Low-speed tracking bonus (fix dead zone from reward side, safe additive)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V11d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V11e: V11b + low-speed bonus (combined penalty + reward fix)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V11e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11e:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v11e:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V12a: Scheduled sigma annealing (fix dead zone from reward side, safe global annealing)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V12a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v12a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v12a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V12b: Scheduled sigma + scaled action rate (fix dead zone from both sides)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V12b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v12b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v12b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V12c: Scheduled sigma + scaled action rate + low-speed bonus (triple attack)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V12c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v12c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v12c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V13: Parameter Attribution Experiments ----

# V13a: action_rate -0.12 → -0.02 (single factor, testing parameter miscalibration hypothesis)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V13a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v13a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v13a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V13b: V13a + feet_slide -0.3 → -0.1 (two largest penalty deviations)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V13b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v13b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v13b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V13c: V13b + track_lin/ang weight 1.0 → 1.5 (boost signal side)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V13c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v13c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v13c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V14: Orthogonal Dead-Zone Experiments ----

# V14a: "Baselined Tracking" — new kernel that gives 0 for standing still (tests reward landscape shape)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V14a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v14a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v14a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V14b: "Half Penalties" — all penalty weights × 0.5 (tests penalty budget magnitude)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V14b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v14b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v14b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V14c: "Speed-Gated Steps" — gait/clearance scaled by cmd magnitude (tests forced marching hypothesis)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V14c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v14c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v14c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V15: Informed by 001.md Analysis + V14 Lessons ----

# V15a: "Surgical Reward Rebalance" — targeted penalty reduction per 001.md Tier 1
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V15a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v15a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v15a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V15b: "Adaptive Command Sampling" — V15a rewards + performance-weighted sampling (001.md Plan B)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V15b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v15b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v15b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V15c: "Wide Start + Anti-Stagnation" — full command range from iter 0, no curriculum
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V15c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v15c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v15c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V16a: "Fixed Adaptive Sampling" — redesigned metric (relative accuracy), same V15a rewards
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V16a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v16a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v16a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V16b: "Low-Speed Incentive" — strong cmd_nonresponse(-2.0) + low-speed bonuses, bucketed curriculum
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V16b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v16b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v16b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V16c: "Adaptive + Incentive" — V16a sampling + V16b rewards (maximum intervention)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V16c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v16c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v16c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V17: Combined + Targeted Improvements ----

# V17a: "Combined Baseline + Standing Fix" — V16a sampling + V16b incentive + standing fix
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V17a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v17a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v17a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V17b: "Tighter Incentive + Waist Damping" — V17a + tighter std=0.15 + stronger waist penalties
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V17b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v17b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v17b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V17c: "Discrete Adaptive Sampling" — discrete velocity levels + staged curriculum + V16b movement incentive
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V17c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v17c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v17c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V18: Bug-Fixed Adaptive Experiments ----

# V18a: "Fixed Adaptive Continuous" — V17a + bug fixes + temp=3.0
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V18a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v18a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v18a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V18b: "Fixed Discrete Adaptive" — V17c + bug fixes + curriculum threshold=0.3 + temp=3.0
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V18b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v18b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v18b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V18c: "Pure-Linear Allocation" — V18a + rel_linear_envs=0.20 (dedicated pure-linear practice)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V18c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v18c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v18c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V19: Marginal Axis-Independent Sampling ----

# V19a: "Marginal Continuous" — axis-independent 1D marginal sampling with continuous bin ranges
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V19b: "Marginal Discrete" — axis-independent 1D marginal sampling with discrete (lo==hi) bins
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V19c: "Pure-Env Accuracy + Anti-Standing" — pure env types, restricted perf, stronger penalties
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)
# V19d: "Direction-Gated Accuracy + wz Boost" — noise-immune perf, min-based expansion, focused sampling
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)


# V19d-CLP: Contrastive Latent Policy (TCN encoder) on V19d env
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19d-CLP",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofContrastiveTCNPPORunnerCfg",
    },
)

# V19d-CLP-Transformer: Contrastive Latent Policy (Transformer encoder) on V19d env
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19d-CLP-Transformer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofContrastiveTransformerPPORunnerCfg",
    },
)

# V19e: "Rotation Gradient Fix" — rotation-skip lin tracking, wz nonresponse, upper body acc
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19e:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19e:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V19f: "Aggressive Rotation" — sharp wz kernel, proportional wz penalty, zero-speed fix
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19f",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19f:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)
