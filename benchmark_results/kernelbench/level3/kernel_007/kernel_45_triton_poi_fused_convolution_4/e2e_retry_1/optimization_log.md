# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_45` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4640 ns — **Best latency**: 4640 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_45`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_45_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 28
- **Duration**: 4640 ns
- **SM Throughput**: 36.19 % | **Memory**: 31.93 % | **DRAM**: 4.00 %
- **Achieved Occupancy**: 69.01 %
- **IPC Active**: 2.69 inst/cycle | **No Eligible**: 32.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_and_add_autotune_to_eliminate_partial_wave | 0.979× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4640 ns | 4700 ns | +60 ns (+1.3%) |
| SM Throughput (%) | 36.19 % | 44.37 % | +8.18 pp |
| Memory Throughput (%) | 31.93 % | 30.50 % | -1.43 pp |
| DRAM Throughput (%) | 4.00 % | 3.94 % | -0.06 pp |
| Achieved Occupancy (%) | 69.01 % | 68.73 % | -0.28 pp |
| Registers/Thread | 28.00 register/thread | 27.00 register/thread | -1 |
| Waves/SM | 1.48 | 1.48 | +0.00 |
| No Eligible (%) | 32.06 % | 25.76 % | -6.30 pp |
| IPC Active | 2.69 inst/cycle | 2.93 inst/cycle | +0.24 |

---

## Cumulative Progress

```
Baseline:  4640 ns  (1.000×)

Final best: 4640 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
