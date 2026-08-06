# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_hardswish_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2820 ns — **Best latency**: 2820 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_hardswish_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_069/kernel_2_triton_poi_fused_convolution_h/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 24
- **Duration**: 2820 ns
- **SM Throughput**: 0.25 % | **Memory**: 0.46 % | **DRAM**: 0.05 %
- **Achieved Occupancy**: 6.19 %
- **IPC Active**: 0.70 inst/cycle | **No Eligible**: 81.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_1d_grid_to_3d_grid_preserving_exact_store_index | 0.962× | 1.048× | ✓ |  |
| D | fix_grid_launch_with_correct_xblock_and_2d_y_tiling | 0.958× | 0.959× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_1d_grid_to_3d... | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2820 ns | 2690 ns | -130 ns (-4.6%) | 2940 ns | +120 ns (+4.3%) |
| SM Throughput (%) | 0.25 % | 0.25 % | +0.00 pp | 0.23 % | -0.02 pp |
| Memory Throughput (%) | 0.46 % | 0.49 % | +0.03 pp | 0.38 % | -0.08 pp |
| DRAM Throughput (%) | 0.05 % | 0.06 % | +0.01 pp | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.18 % | -0.01 pp | 6.17 % | -0.02 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 81.81 % | 77.87 % | -3.94 pp | 78.32 % | -3.49 pp |
| IPC Active | 0.70 inst/cycle | 0.75 inst/cycle | +0.05 | 0.81 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  2820 ns  (1.000×)

Final best: 2820 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
