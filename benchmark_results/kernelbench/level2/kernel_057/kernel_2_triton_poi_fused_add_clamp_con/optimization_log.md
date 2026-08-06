# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_clamp_convolution_div_mul_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2690 ns — **Best latency**: 2690 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_clamp_convolution_div_mul_relu_2`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_057/kernel_2_triton_poi_fused_add_clamp_con/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 24
- **Duration**: 2690 ns
- **SM Throughput**: 0.21 % | **Memory**: 0.52 % | **DRAM**: 0.05 %
- **Achieved Occupancy**: 6.18 %
- **IPC Active**: 0.66 inst/cycle | **No Eligible**: 82.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_correct_2d_grid_large_yblock | 0.982× | 1.011× | ✓ |  |
| B | fix_grid_launch_correctness_with_proper_2d_grid | 0.982× | 1.000× | ✓ |  |
| C | fix_grid_launch_with_correct_2d_indexing_and_vectorized_loads | 0.947× | 0.989× | ✓ |  |
| D | fix_grid_launch_with_correct_3d_grid_structure | 0.979× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ | fix_grid_launch_c... | Δ | fix_grid_launch_w... | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2690 ns | 2660 ns | -30 ns (-1.1%) | 2690 ns | +0 ns (+0.0%) | 2720 ns | +30 ns (+1.1%) | 2750 ns | +60 ns (+2.2%) |
| SM Throughput (%) | 0.21 % | 0.21 % | +0.00 pp | 0.21 % | +0.00 pp | 0.22 % | +0.01 pp | 0.23 % | +0.02 pp |
| Memory Throughput (%) | 0.52 % | 0.41 % | -0.11 pp | 0.46 % | -0.06 pp | 0.41 % | -0.11 pp | 0.41 % | -0.11 pp |
| DRAM Throughput (%) | 0.05 % | 0.06 % | +0.01 pp | 0.06 % | +0.01 pp | 0.05 % | +0.00 pp | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp | 6.18 % | +0.00 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 | 24.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 82.77 % | 77.51 % | -5.26 pp | 84.22 % | +1.45 pp | 78.98 % | -3.79 pp | 79.40 % | -3.37 pp |
| IPC Active | 0.66 inst/cycle | 0.69 inst/cycle | +0.03 | 0.71 inst/cycle | +0.05 | 0.73 inst/cycle | +0.07 | 0.73 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  2690 ns  (1.000×)

Final best: 2690 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
