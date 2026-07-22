# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3260 ns — **Best latency**: 3260 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (980,1,1), Waves/SM: 0.46
- **Registers/thread**: 18
- **Duration**: 3260 ns
- **SM Throughput**: 12.99 % | **Memory**: 4.52 % | **DRAM**: 3.21 %
- **Achieved Occupancy**: 40.67 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 63.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_persistent_tile_efficiency | 0.952× | 0.621× | ✓ |  |
| B | autotune_xblock_and_fuse_hardtanh_clamp | 0.946× | 0.979× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3260 ns | 5250 ns | +1990 ns (+61.0%) | 3330 ns | +70 ns (+2.1%) |
| SM Throughput (%) | 12.99 % | 49.79 % | +36.80 pp | 8.49 % | -4.50 pp |
| Memory Throughput (%) | 4.52 % | 10.64 % | +6.12 pp | 4.37 % | -0.15 pp |
| DRAM Throughput (%) | 3.21 % | 1.98 % | -1.23 pp | 3.12 % | -0.09 pp |
| Achieved Occupancy (%) | 40.67 % | 48.26 % | +7.59 pp | 20.73 % | -19.94 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 24.00 register/thread | +6 |
| Waves/SM | 0.46 | 1.48 | +1.02 | 0.23 | -0.23 |
| No Eligible (%) | 63.80 % | 20.18 % | -43.62 pp | 73.48 % | +9.68 pp |
| IPC Active | 1.20 inst/cycle | 3.11 inst/cycle | +1.91 | 0.94 inst/cycle | -0.26 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_for_persistent_tile_efficiency` (0.621×)

Adding @triton.autotune with larger XBLOCK configs caused the compiler to select a configuration (XBLOCK=256, num_warps=4 or similar) that dramatically increased register usage from 18 to 48 registers/thread. This register pressure capped theoretical occupancy at 62.5% (Block Limit Registers dropped from 21 to 5 blocks/SM) and increased total executed instructions by 6.3x (333K to 2.09M), causing a 1.61x slowdown despite waves_per_sm improving from 0.46 to 1.48. When the baseline kernel uses very few registers (≤20/thread), adding autotune configs with larger XBLOCK can trigger the compiler to unroll loops or allocate more temporaries, causing register spill or bloat that more than offsets the wave-efficiency gain.

> **Applicable when**: Kernel has very low register usage (≤20 regs/thread) and is grid-size limited (waves_per_sm < 1.0, SOLBottleneck rule triggered, AchievedOccupancy < 50%)

> **Avoid when**: The kernel already uses few registers (≤20/thread) and theoretical occupancy is near 100% — adding autotune with larger XBLOCK configs can cause the compiler to increase register usage dramatically (e.g. 18→48 regs/thread), collapsing theoretical occupancy and increasing total instruction count, resulting in regression. Do not apply when the instruction count increase from loop unrolling would outweigh wave-efficiency gains.

---

## Cumulative Progress

```
Baseline:  3260 ns  (1.000×)

Final best: 3260 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
