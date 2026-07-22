# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3900 ns — **Best latency**: 3900 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_54_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3900 ns
- **SM Throughput**: 26.67 % | **Memory**: 25.77 % | **DRAM**: 3.12 %
- **Achieved Occupancy**: 64.37 %
- **IPC Active**: 2.19 inst/cycle | **No Eligible**: 42.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.016× | 0.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3900 ns | 686050 ns | +682150 ns (+17491.0%) |
| SM Throughput (%) | 26.67 % | 46.33 % | +19.66 pp |
| Memory Throughput (%) | 25.77 % | 58.95 % | +33.18 pp |
| DRAM Throughput (%) | 3.12 % | 0.02 % | -3.10 pp |
| Achieved Occupancy (%) | 64.37 % | 63.38 % | -0.99 pp |
| Registers/Thread | 28.00 register/thread | 40.00 register/thread | +12 |
| Waves/SM | 0.97 | 1.29 | +0.32 |
| No Eligible (%) | 42.94 % | 49.56 % | +6.62 pp |
| IPC Active | 2.19 inst/cycle | 2.01 inst/cycle | -0.18 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune` (0.006×)

Adding a persistent while-loop inside a Triton kernel to process multiple tiles per CTA caused a ~254x increase in executed instructions (819K → 208M) and a ~176x increase in kernel duration (3.9 µs → 686 µs). The while-loop tile scheduling logic (integer division and modulo for tile_id decomposition) is computed in scalar registers every iteration, and the loop itself prevents the compiler from statically unrolling or optimizing the tile traversal. Additionally, the increased register pressure (28 → 40 registers/thread) reduced theoretical occupancy from 100% to 75%, and eligible warps per scheduler dropped from 3.69 to 1.85, worsening the IssueSlotUtilization bottleneck rather than fixing it. For very short-duration kernels (< 10 µs) that are already near the 1-wave boundary, persistent kernel patterns introduce more overhead than they eliminate.

> **Applicable when**: Never applicable as described — persistent while-loop tile scheduling in Triton is harmful for small, short-duration kernels

> **Avoid when**: Kernel duration is under ~10 µs; total instruction count is already low (< 1M); the kernel is a simple memory-movement/permutation with no compute; the grid is near 1 wave (waves_per_sm ≈ 1.0); registers per thread are already near the occupancy-limiting threshold. The persistent loop's per-iteration overhead (integer div/mod for tile decomposition, loop branch) dominates over any scheduling benefit when the total work per tile is tiny.

---

## Cumulative Progress

```
Baseline:  3900 ns  (1.000×)

Final best: 3900 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
