# 15-DOF Velocity Policy

After training, copy the exported files here:

```
config/policy/velocity/
├── v0/                          # versioned subfolder
│   ├── exported/
│   │   └── policy.onnx          # from export_policy.py / rsl_rl
│   └── params/
│       └── deploy.yaml          # from export_deploy_cfg.py
```

Or point `policy_dir` in `config.yaml` directly to the training log directory:

```yaml
Velocity:
  policy_dir: ../../../../logs/rsl_rl/Unitree-G1-15dof-Velocity/2026-xx-xx/
```

## Build

```bash
cd deploy/robots/g1_15dof
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Run (sim2sim)

```bash
./build/g1_ctrl --network lo   # loopback for sim2sim
```

## Run (real hardware)

```bash
./build/g1_ctrl --network eth0
```
