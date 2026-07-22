# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_mm_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 405860 ns — **Best latency**: 405860 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_mm_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_0_triton_tem_fused_mm_view_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3200,1,1), Waves/SM: 4.85
- **Registers/thread**: 96
- **Duration**: 405860 ns
- **SM Throughput**: 76.60 % | **Memory**: 70.60 % | **DRAM**: 22.78 %
- **Achieved Occupancy**: 29.75 %
- **IPC Active**: 3.17 inst/cycle | **No Eligible**: 20.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_reduce_register_pressure_and_improve_occupancy | 0.889× | 0.540× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 405860 ns | 751100 ns | +345240 ns (+85.1%) |
| SM Throughput (%) | 76.60 % | 41.56 % | -35.04 pp |
| Memory Throughput (%) | 70.60 % | 56.51 % | -14.09 pp |
| DRAM Throughput (%) | 22.78 % | 12.40 % | -10.38 pp |
| Achieved Occupancy (%) | 29.75 % | 6.25 % | -23.50 pp |
| Registers/Thread | 96.00 register/thread | 128.00 register/thread | +32 |
| Waves/SM | 4.85 | 24.24 | +19.39 |
| No Eligible (%) | 20.90 % | 57.35 % | +36.45 pp |
| IPC Active | 3.17 inst/cycle | 1.70 inst/cycle | -1.47 |

---

## Cumulative Progress

```
Baseline:  405860 ns  (1.000×)

Final best: 405860 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
