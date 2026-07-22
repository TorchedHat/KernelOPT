# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7330 ns — **Best latency**: 7330 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_225_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7168,1,1), Waves/SM: 3.39
- **Registers/thread**: 28
- **Duration**: 7330 ns
- **SM Throughput**: 54.16 % | **Memory**: 51.14 % | **DRAM**: 5.76 %
- **Achieved Occupancy**: 71.88 %
- **IPC Active**: 3.06 inst/cycle | **No Eligible**: 20.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_partial_wave_and_coalescing | 0.979× | 1.046× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 7330 ns | 7010 ns | -320 ns (-4.4%) |
| SM Throughput (%) | 54.16 % | 30.93 % | -23.23 pp |
| Memory Throughput (%) | 51.14 % | 28.76 % | -22.38 pp |
| DRAM Throughput (%) | 5.76 % | 6.13 % | +0.37 pp |
| Achieved Occupancy (%) | 71.88 % | 53.69 % | -18.19 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 |
| Waves/SM | 3.39 | 1.70 | -1.69 |
| No Eligible (%) | 20.03 % | 39.05 % | +19.02 pp |
| IPC Active | 3.06 inst/cycle | 2.37 inst/cycle | -0.69 |

---

## Cumulative Progress

```
Baseline:  7330 ns  (1.000×)

Final best: 7330 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
