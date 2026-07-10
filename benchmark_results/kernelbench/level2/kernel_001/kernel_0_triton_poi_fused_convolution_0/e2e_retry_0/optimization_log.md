# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6110 ns — **Best latency**: 6110 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_001/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6110 ns
- **SM Throughput**: 16.36 % | **Memory**: 41.36 % | **DRAM**: 20.21 %
- **Achieved Occupancy**: 55.35 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 0.990× | 1.037× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 6110 ns | 5890 ns | -220 ns (-3.6%) |
| SM Throughput (%) | 16.36 % | 16.53 % | +0.17 pp |
| Memory Throughput (%) | 41.36 % | 41.92 % | +0.56 pp |
| DRAM Throughput (%) | 20.21 % | 20.89 % | +0.68 pp |
| Achieved Occupancy (%) | 55.35 % | 54.49 % | -0.86 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.94 % | 71.22 % | +0.28 pp |
| IPC Active | 1.13 inst/cycle | 1.13 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  6110 ns  (1.000×)

Final best: 6110 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
