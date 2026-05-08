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


# V19d-CLP: Contrastive Latent Policy (TCN encoder) on V19d env (history_length=10)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19d-CLP",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotEnvCfgCLP",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotPlayEnvCfgCLP",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofContrastiveTCNPPORunnerCfg",
    },
)

# V19d-CLP-Transformer: Contrastive Latent Policy (Transformer encoder) on V19d env (history_length=10)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19d-CLP-Transformer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotEnvCfgCLP",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19d:RobotPlayEnvCfgCLP",
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

# V19g: "Balanced Rotation" — fix V19f backward-drift & vx/vy regression
# init 必须要有play的注册
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19g",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19g:RobotEnvCfg",
         "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19g:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V19h: "Force Standing" — amplify standing signals + zero_cmd_foot_height
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19h",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19h:RobotEnvCfg",
         "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19h:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V19i: "Scheduled Standing" — V19g base + scheduled zero_cmd penalties (24k->72k)
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V19i",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v19i:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# ---- V20: Mode-Token Locomotion ----

# V20a: "5-Mode Token" — V19i base + 5-dim one-hot gait token observation
# (standing/pure_vx/pure_vy/pure_wz/joint). Theoretical sample-complexity gain
# Omega((Delta_x+Delta_y+Delta_w)^2/eps^2) over V19i. Clean A/B test.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20b: "Token + V19g" — clean A/B test of gait_mode_token without V19i schedule
# pathology. V20a (Token + V19i) showed pre-collapse signature (bad_orient 23%,
# entropy_loss +5.7) identical to V19i baseline → token did NOT help because
# bottleneck is reward landscape, not observation. V20b reverts env to V19g
# (last 18.7k-iter stable cfg) + token only.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20c: "Token + V19f" (REVISED 2026-04-25 after log audit).
# V19 evolution facts (verified against logs/rsl_rl/):
#   - V5c: only 13 aborted runs (max model_999 ~1k iter); NOT a verified base.
#   - V19c-e: zero-speed instability ("robot still walks at cmd=0").
#   - V19f (18.5k iter completed): "Zero-speed standing: FIXED" + "wz responds
#     from 0.1: WORKING". Open issues: vx/vy 0.1-0.3 no response + pure-rotation
#     backward drift — both map naturally to mode-token theoretical gains
#     (pure_vx mode disambiguates parasitic vx; pure_wz mode allows drift-free
#     rotation policy without adding new reward terms).
#   - V19g: reverted V19f's lin-tracking skip → fixed drift but LOST zero-speed.
#   - V19h/i: piled standing penalty → suicide policy, full collapse.
# V19f is the ONLY V19 version satisfying: trained-to-completion + zero-speed-OK
# + wz-responsive. V20c = V19f + gait_mode_token only (clean A/B isolation).
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20d: "V20c + Anti-Drift" — V19f-style pure_rotation_drift (-1.0) added on
# top of V20c (V19f + token). Targets V20c sim2sim observation: pure rotation
# drifts backward due to V19f's rotation_skip giving 1.0 reward for any drift.
# Single-axis A/B vs V20c.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20e: "V20c + Sharp Linear Tracking" — track_lin_vel_xy_rotation_skip std
# 0.5 -> 0.30. Targets V20c observation: error_vel_xy=0.50 m/s but
# track_lin_vel_xy=0.81 (kernel too wide gives free reward). Sharp kernel
# forces real tracking gradient. Single-axis A/B vs V20c.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20e:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20e:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20f: "V20c + Eliminate Rotation-Skip Blind Spot" — replaces V19f's
# track_lin_vel_xy_rotation_skip with standard track_lin_vel_xy_yaw_frame_exp.
# Targets V20d sim2sim observation: pure_rotation_drift (-1.0) too weak vs
# +1.0 free reward from skip; half-rotation accumulates 1m drift. Single-axis
# A/B vs V20c. User insight: tracking success must check zero-commanded axes
# remain zero, which rotation_skip violates by construction.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20f",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20f:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20f:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20g: "V20f + Eliminate wz Skip Blind Spot" — replaces V19f's
# track_ang_vel_z_rotating_aware (skips on straight walk) with standard
# track_ang_vel_z_exp for both main and sharp wz trackers. Same weight/std.
# Targets V20f sim2sim observation: pure vx/vy staggers, but adding a small
# wz cmd makes walking smooth -- direct evidence of wz blind spot. Mirror
# fix to V20f's lin-side fix. Sequential A/B vs V20f.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20g",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20g:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20g:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20h: "V20f + Boost Pure-Linear Sampling" — increases rel_pure_vx_envs and
# rel_pure_vy_envs from 0.15 to 0.25 each (joint envs reduced 0.35 -> 0.15).
# All rewards/bins/std/weights unchanged. Parallel single-axis A/B vs V20f.
# Targets V20f sim2sim correction: pure_vx/pure_vy token branches fail
# while any 2+ axis combo works -- tests data-density hypothesis vs
# V20g's reward-landscape hypothesis.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20h",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20h:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20h:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20i: "V20f without gait_mode_token" — removes the 5-dim one-hot mode
# token from observations, reverting to V19f-style obs. Keeps V20f's
# linear-side rotation_skip removal. Tests the deepest hypothesis: token
# enforces per-mode subpolicy specialization that prevents transfer of
# yaw-control skill from joint-mode envs (which see cmd_wz!=0) to
# pure-linear-mode envs (which never see cmd_wz signal due to skip+
# proportional structure). Single-axis A/B vs V20f. User insight:
# data-density boost (V20h) cannot help when per-sample gradient is zero.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20i",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20i:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20i:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20j: "V20f + 3-mode token (standing|pure_wz|other)" — selective
# isolation. Keeps token routing for {standing, pure_wz} (qualitatively
# distinct objectives benefit from dedicated subpolicies) but collapses
# {pure_vx, pure_vy, joint} into one "other" bucket so joint envs'
# cmd_wz!=0 yaw signal can transfer (via shared subpolicy params) to
# pure_vx/pure_vy samples. Single-axis A/B vs V20f. Complements V20i:
# V20j tests "selective isolation"; V20i tests "no isolation". User
# insight: isolation must be necessary; full 5-way over-isolates.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20j",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20j:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20j:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20k: "V20i + Symmetric wz Tracking" — single-variable fix targeting
# the reward-structure asymmetry that drives the yaw-wobble cheat.
# Replaces both track_ang_vel_z_rotating_aware terms with the standard
# track_ang_vel_z_exp (kills wz SKIP at cmd_yaw<0.05). cmd_yaw=0 envs
# now experience strict tracking on BOTH axes — destroying the
# wobble-yaw-to-fake-v_xy local optimum at its source. No token (V20i
# inheritance). Predictions: waist_roll_vel > -0.05, error_vel_yaw <
# 0.30, bad_orient < 1%. If standing still bad → V20k-r2 adds 2-mode
# token. If wobble persists → wz_proportional weight escalation.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20k",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20k:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20k:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V20l: "V20g + 3-mode token" — keeps mandatory isolation
# {standing, pure_wz, other}, but fixes the pure_xy training environment by
# inheriting V20g's standard track_ang_vel_z_exp rewards (no straight-walk wz
# skip). Compared with V20j, only yaw reward changes; compared with V20g, only
# token granularity changes (5-mode -> 3-mode). Tests whether pure_xy fails
# because cmd_wz==0 samples lacked yaw supervision, not because token exists.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V20l",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20l:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v20l:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V21a: V20l + mode-conditioned diagnostics only. Keeps rewards, observations,
# and command distribution unchanged; adds per-mode and speed-bucket logging to
# support mechanism attribution before any ablations.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V21b: V21a + velocity-scaled action-rate penalty to improve low-speed
# response without reopening sampler, token, sigma, or tracking-kernel changes.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V21c: V21a + hybrid low-speed linear tracking reward with an explicit
# reward-regime token, kept separate from V21b's action-rate experiment.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21c:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21c:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerV3Cfg",
    },
)

# V21d: V21c-aligned env/reward stack with only the runner-side velocity-prediction
# actor change retained.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21d-TransformerLatent",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21d:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21d:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21dTransformerLatentPPORunnerCfg",
    },
)

# V21e: V21c env (unchanged) + actor-side detached velocity estimator MLP with
# corrected supervised aux loss. Three V21d defects fixed: algorithm class is
# now VelocityEstimatorPPO (not UnitreePPO), achieved-velocity targets are
# extracted from the correct term-major slice of the critic flat-obs, and
# history_obs_dim auto-computes (=58) instead of the erroneous 54.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21e:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21e:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21f: V21e (env + actor velocity estimator) + linear decay (iter 6000->12000)
# of the three gait-shaping rewards (gait, feet_clearance, rotation_single_support).
# Decay window chosen from V21e log: walking firmly established by iter ~6000
# (bad_orient<5%, ep_len>950, both track rewards>0.5). Reuses the V21e runner.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21f",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21f:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21f:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21f2: corrected V21f. V21f used iter-valued start_step/end_step for the
# *_decayed gait shaping rewards, but _linear_step_decay reads
# env.common_step_counter (env steps = iter * num_steps_per_env). The intended
# iter window 6000->12000 collapsed to step 6000->12000 (= iter 250..500),
# so V21f effectively trained without gait shaping. V21f2 multiplies by
# num_steps_per_env (=24): start=144000 (iter 6000), end=288000 (iter 12000).
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21f2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21f2:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21f2:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21g: V21f2 (best 19999-iter baseline) + linear relative-error tracking
# applied to the entire velocity space. Replaces the exp(-||err||^2/std^2)
# kernel of the two tracking rewards (track_lin_vel_xy, track_ang_vel_z) with
# r = clamp(1 - |err|/max(|cmd|, 1.5670*std), 0, 1). At |cmd|=0 this reduces
# to 1 - |x|/(1.5670*std), with the constant 1.5670 derived as the unique
# tangency multiplier so that the linear form is provably <= the exp form
# (no extra "free reward" introduced). Decayed gait shaping kept unchanged.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21g",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21g:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21g:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21h: V21f2 hybrid with rel_floor 0.05 -> 0.5 (single-variable dead-zone fix).
# Same kernel SHAPE as V21f2 (hybrid exp + linear-relative); only the divisor
# floor in the r_rel branch is raised so that for yaw cmd in [-0.8, 0.8] the
# previously dead region (err > |cmd| -> r=0, grad=0) is converted into a
# linearly-decreasing reward with constant negative slope. Isolates "kill
# dead zone" from V21g's "change kernel form".
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21h",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21h:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21h:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21i: STRICT piecewise tracking matching the literal user spec.
#   cmd = 0 :  r = 1 - |v|/b     (b = 1.5670*std)
#   cmd > 0 :  r = 1 - |v-cmd|/|cmd|     (NO max with b -> dead zones for cmd>0)
# Differs from V21g (`*_relative_full`) which used max(|cmd|, b); V21i uses
# a hard switch at |cmd|<1e-3.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21i",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21i:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21i:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21j: V21i strict piecewise + LEAKY negative tail (slope_neg=0.1).
# Same denominator structure as V21i (cmd=0 -> b; cmd>0 -> |cmd|, no max).
# But replace clamp(0,1) with leaky: r = raw if raw>=0 else 0.1*raw, capped <=1.
# Eliminates V21i's dead zone WITHOUT introducing the b-soft-floor of V21g.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21j",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21j:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21j:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21k: constant-denominator leaky linear (NO per-cmd scaling, NO piecewise).
# r = 1 - |v - v_cmd| / b_abs       if raw >= 0
# r = 0.1 * raw                     if raw <  0   (capped at <=1)
# b_abs = LINEAR_REL_B_RATIO * std = 0.7834. Tests user hypothesis that exp
# kernel is asymptotically equivalent to fixed-slope linear + leaky tail.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21k",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21k:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21k:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21eVelocityEstimatorPPORunnerCfg",
    },
)


# V21l: LIRPG (learnable intrinsic tracking reward).
# Per-channel MLP r_phi(v, v_cmd) initialized to V21k baseline and
# meta-updated online to maximize task return = -|v - v_cmd|.
# Uses LirpgVelocityEstimatorPPO subclass (V21e runner + meta hook).
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21l",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21l:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21l:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21lLirpgRunnerCfg",
    },
)

# V21m: Gaussian LIRPG - r = exp(-e^2 / sigma(v_cmd)^2) with learnable sigma.
# Same 1.5:1.5 lin:ang weights as V21l, same LIRPG PPO runner.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21m",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21m:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21m:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21mLirpgRunnerCfg",
    },
)

# V21n: V21f2 + 5-mode command token
# {standing, pure_vx, pure_vy, pure_wz, joint(>=2 axes)} and yaw tracking
# coefficient reduced to 2.0.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V21n",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21n:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v21n:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV21nFiveModeVelocityEstimatorPPORunnerCfg",
    },
)

# V22a: V21g env + frozen V3 segment-encoder z_gait injected into the actor's
# policy latent. The encoder is loaded from
# /root/workspace/unitree_rl_lab/logs/frnc_seg_v3/v3_full/encoder.pt, kept in
# eval/no_grad, and fed by a per-env rolling buffer of the 295-dim flat policy
# obs (T_seg=32). Critic is unchanged — z_gait is policy-side only.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V22a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v22a:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v22a:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV22aSegmentEncoderPPORunnerCfg",
    },
)


# V22b: V22a + metric-residual intrinsic reward + SMERL gate.
# r_intrinsic = ||z - z_axial(v_cmd)||^2 / d_gait, axial bases from V3 ckpt.
# Encoder still frozen; reward only takes effect after iter warmup and once
# SMERL gate (per-env env-reward EMA above threshold) opens.
gym.register(
    id="Unitree-G1-15dof-Velocity-Rot-V22b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v22b:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_rot_v22b:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G115DofV22bSegmentEncoderCICPPORunnerCfg",
    },
)
