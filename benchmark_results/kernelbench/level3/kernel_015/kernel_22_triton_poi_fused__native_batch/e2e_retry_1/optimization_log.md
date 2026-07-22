# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12770 ns — **Best latency**: 12770 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_22_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12770 ns
- **SM Throughput**: 46.94 % | **Memory**: 20.71 % | **DRAM**: 12.85 %
- **Achieved Occupancy**: 55.58 %
- **IPC Active**: 2.32 inst/cycle | **No Eligible**: 42.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.895× | 0.763× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.988× | 0.426× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12770 ns | 16740 ns | +3970 ns (+31.1%) | 29950 ns | +17180 ns (+134.5%) |
| SM Throughput (%) | 46.94 % | 76.28 % | +29.34 pp | 81.19 % | +34.25 pp |
| Memory Throughput (%) | 20.71 % | 21.91 % | +1.20 pp | 18.52 % | -2.19 pp |
| DRAM Throughput (%) | 12.85 % | 9.79 % | -3.06 pp | 5.47 % | -7.38 pp |
| Achieved Occupancy (%) | 55.58 % | 85.11 % | +29.53 pp | 85.89 % | +30.31 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 14.85 | +7.43 |
| No Eligible (%) | 42.08 % | 13.43 % | -28.65 pp | 12.24 % | -29.84 pp |
| IPC Active | 2.32 inst/cycle | 3.44 inst/cycle | +1.12 | 3.51 inst/cycle | +1.19 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.763×)

Increasing XBLOCK from 128 to 512–2048 via autotune improved occupancy (55.6%→85.1%) and scheduler eligibility (42%→13% no-eligible), but caused registers per thread to double (17→32) and total executed instructions to increase by 2.2x (4.6M→10.1M), resulting in a 31% wall-time regression (12.77→16.74 µs). The extra register pressure and instruction count overhead outweighed the latency-hiding benefit of higher occupancy. This pattern occurs when the kernel body is already compute-light and the larger tile forces the compiler to spill or unroll more aggressively, adding more work than the occupancy gain saves.

> **Applicable when**: Kernel has very low register usage (<20 registers/thread) AND the bottleneck is purely latency-hiding (high no-eligible-warp %) AND the kernel body is complex enough that larger tiles do not cause compiler unrolling/spilling.

> **Avoid when**: Kernel is already compute-light (few FP ops per element, e.g. simple pointwise fused BN+ReLU) where larger XBLOCK causes the compiler to double register usage and instruction count. Also avoid when the grid is already well-occupying the GPU (waves_per_sm > 5) and the stall is L1TEX latency that cannot be hidden by more work per CTA.

---

## Cumulative Progress

```
Baseline:  12770 ns  (1.000×)

Final best: 12770 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
