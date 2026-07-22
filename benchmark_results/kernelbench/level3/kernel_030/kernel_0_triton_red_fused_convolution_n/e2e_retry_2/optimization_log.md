# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 787680 ns — **Best latency**: 787680 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_0_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 787680 ns
- **SM Throughput**: 12.97 % | **Memory**: 38.78 % | **DRAM**: 38.78 %
- **Achieved Occupancy**: 11.24 %
- **IPC Active**: 0.20 inst/cycle | **No Eligible**: 94.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_fix_tile_sizes_eliminate_register_spill | 1.000× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 787680 ns | 785980 ns | -1700 ns (-0.2%) |
| SM Throughput (%) | 12.97 % | 12.98 % | +0.01 pp |
| Memory Throughput (%) | 38.78 % | 38.84 % | +0.06 pp |
| DRAM Throughput (%) | 38.78 % | 38.84 % | +0.06 pp |
| Achieved Occupancy (%) | 11.24 % | 11.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.62 | 0.62 | +0.00 |
| No Eligible (%) | 94.97 % | 94.97 % | +0.00 pp |
| IPC Active | 0.20 inst/cycle | 0.20 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  787680 ns  (1.000×)

Final best: 787680 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
