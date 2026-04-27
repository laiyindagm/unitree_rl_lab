import os

file_path = "/root/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/15dof_rot/velocity_env_cfg_rot_v20c.py"

with open(file_path, 'r') as f:
    content = f.read()

old_block = """@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10"""

new_block = """@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Play: expose the full curriculum-extended command range.
        # V19f bin counts: 15 vx_pos, 8 vx_neg, 11 vy, 16 wz.
        self.commands.base_velocity.num_active_vx_pos = 15
        self.commands.base_velocity.num_active_vx_neg = 8
        self.commands.base_velocity.num_active_vy = 11
        self.commands.base_velocity.num_active_wz = 16"""

assert content.count(old_block) == 1, f"Expected 1 occurrence of old_block, found {content.count(old_block)}"

new_content = content.replace(old_block, new_block)

with open(file_path, 'w') as f:
    f.write(new_content)

print(f"Successfully patched {file_path}")
