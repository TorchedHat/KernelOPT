# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2240 ns — **Best latency**: 2240 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 30
- **Duration**: 2240 ns
- **SM Throughput**: 0.05 % | **Memory**: 0.44 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.11 %
- **IPC Active**: 0.47 inst/cycle | **No Eligible**: 84.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_launch_correctness_with_verified_index_math | 0.993× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2240 ns | 2210 ns | -30 ns (-1.3%) |
| SM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Memory Throughput (%) | 0.44 % | 0.46 % | +0.02 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.11 % | 6.08 % | -0.03 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 84.66 % | 84.47 % | -0.19 pp |
| IPC Active | 0.47 inst/cycle | 0.54 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  2240 ns  (1.000×)

Final best: 2240 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
