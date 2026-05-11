# GPU-First Style Pipeline Notes

This project should treat GPU execution as the default for large-array work in
the style encoder pipeline.

## Current Policy

- Use torch tensor operations for train-time losses, probe regressions,
  residualization, distance metrics, and sampled pair metrics.
- Keep NumPy/SciPy/sklearn only for small control-plane work, file I/O, final
  scalar summaries, or as explicit fallback backends.
- Any new expensive metric must expose a backend flag or be implemented directly
  in torch.
- Any new experiment runner argument that controls GPU behavior should be
  propagated through `run_style_v4_experiments.py`, so remote jobs do not depend
  on manual command edits.
- Remote work must stay under `/data1/huangyifan`; do not use `/home`, do not
  download large artifacts, and do not schedule this user's jobs on physical GPU
  2 unless explicitly asked.

## Pipeline Expectations

- Data collection is IsaacLab/GPU simulation, but shard writing is CPU and disk
  bound. Keep shard sizes large enough to avoid frequent compression stalls.
- Feature generation is currently CPU-heavy. Use `--save_compression stored`
  when `/data1` space is available and wall-clock time matters more than npz
  size.
- Encoder pretraining should normally run on CUDA. For datasets that fit in
  memory, prefer `--dataset_device cuda` to bypass DataLoader CPU slicing and
  repeated host-to-device copies.
- Probe should normally run with `--probe_backend torch`. Use the sklearn backend
  only for compatibility checks.
- Every long experiment should produce a performance audit with
  `audit_style_pipeline_perf.py` and keep `perf_audit.json` next to the run.

## Adding New Functionality

Before adding a new metric, loss, target, or sampling rule:

1. Estimate whether it runs per sample, per pair, per source split, or per
   epoch.
2. If it touches more than roughly 10k rows or sits inside an epoch/source loop,
   implement it in torch first.
3. Avoid CPU round-trips inside inner loops. Convert to NumPy only at the final
   reporting boundary.
4. Add a small smoke test command and include the backend/device in JSON output.
5. If a CPU implementation is unavoidable, document why and cap the sample
   count with a CLI argument.
