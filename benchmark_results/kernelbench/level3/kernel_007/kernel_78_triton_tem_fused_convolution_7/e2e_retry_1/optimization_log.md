# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_78` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 49790 ns — **Best latency**: 49790 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_78`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_78_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.20
- **Registers/thread**: 94
- **Duration**: 49790 ns
- **SM Throughput**: 7.92 % | **Memory**: 53.17 % | **DRAM**: 0.90 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.54 inst/cycle | **No Eligible**: 86.38 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.483× | 0.467× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 49790 ns | 106660 ns | +56870 ns (+114.2%) |
| SM Throughput (%) | 7.92 % | 11.79 % | +3.87 pp |
| Memory Throughput (%) | 53.17 % | 46.03 % | -7.14 pp |
| DRAM Throughput (%) | 0.90 % | 0.42 % | -0.48 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 |
| Waves/SM | 0.20 | 0.20 | +0.00 |
| No Eligible (%) | 86.38 % | 79.68 % | -6.70 pp |
| IPC Active | 0.54 inst/cycle | 0.81 inst/cycle | +0.27 |

---

## Cumulative Progress

```
Baseline:  49790 ns  (1.000×)

Final best: 49790 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
