# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4000 ns — **Best latency**: 4000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12,1,1), Waves/SM: 0.01
- **Registers/thread**: 34
- **Duration**: 4000 ns
- **SM Throughput**: 0.39 % | **Memory**: 0.35 % | **DRAM**: 0.06 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.73 inst/cycle | **No Eligible**: 81.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_and_autotune_tile_sizes | 0.967× | 1.250× | ✓ |  |
| B | fix_grid_launch_to_cover_y_dimension | 0.970× | 1.078× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3200 ns | -800 ns (-20.0%) | 3710 ns | -290 ns (-7.3%) |
| SM Throughput (%) | 0.39 % | 0.51 % | +0.12 pp | 0.49 % | +0.10 pp |
| Memory Throughput (%) | 0.35 % | 0.92 % | +0.57 pp | 0.42 % | +0.07 pp |
| DRAM Throughput (%) | 0.06 % | 0.21 % | +0.15 pp | 0.12 % | +0.06 pp |
| Achieved Occupancy (%) | 6.23 % | 6.14 % | -0.09 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 34.00 register/thread | 40.00 register/thread | +6 | 40.00 register/thread | +6 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 81.33 % | 88.30 % | +6.97 pp | 81.30 % | -0.03 pp |
| IPC Active | 0.73 inst/cycle | 0.50 inst/cycle | -0.23 | 0.80 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  4000 ns  (1.000×)

Final best: 4000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
