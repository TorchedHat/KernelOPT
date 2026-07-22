# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12290 ns — **Best latency**: 12290 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12290 ns
- **SM Throughput**: 47.56 % | **Memory**: 21.02 % | **DRAM**: 13.33 %
- **Achieved Occupancy**: 55.69 %
- **IPC Active**: 2.35 inst/cycle | **No Eligible**: 41.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.771× | 0.730× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.998× | 0.223× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12290 ns | 16830 ns | +4540 ns (+36.9%) | 55170 ns | +42880 ns (+348.9%) |
| SM Throughput (%) | 47.56 % | 76.14 % | +28.58 pp | 85.60 % | +38.04 pp |
| Memory Throughput (%) | 21.02 % | 21.85 % | +0.83 pp | 16.82 % | -4.20 pp |
| DRAM Throughput (%) | 13.33 % | 9.74 % | -3.59 pp | 2.96 % | -10.37 pp |
| Achieved Occupancy (%) | 55.69 % | 85.42 % | +29.73 pp | 84.52 % | +28.83 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 29.70 | +22.28 |
| No Eligible (%) | 41.13 % | 14.36 % | -26.77 pp | 10.78 % | -30.35 pp |
| IPC Active | 2.35 inst/cycle | 3.35 inst/cycle | +1.00 | 3.56 inst/cycle | +1.21 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_and_num_warps_for_latency_hiding` (0.223×)

Adding @triton.autotune with larger XBLOCK configs (up to 2048) caused the compiler to select a 512-thread block configuration that doubled register usage per thread (17→32), inflating the total instruction count by ~8x (4.6M→37.3M instructions) and increasing kernel duration from 12.3µs to 55.2µs. Although compute SM throughput improved (47%→85%) and eligible warps per scheduler rose (1.27→8.02), the massive instruction count increase from register pressure and loop unrolling overhead dominated, causing a 4.5x regression in wall time. When a kernel is already short-duration and memory-latency-bound with low register usage, increasing block size via autotuning can trigger register spilling or excessive unrolling that inflates instruction count far beyond any latency-hiding benefit.

> **Applicable when**: Never apply this specific pattern as described. The strategy of increasing block size to hide memory latency is conceptually valid, but this implementation caused severe regression.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs when: (1) the kernel is already short-duration (<20µs); (2) register count is low (≤20/thread) and may double under larger tile sizes; (3) the grid is fixed and not updated to use meta['XBLOCK'] (implementation bug present here); (4) the total instruction count is already low — larger blocks cause proportional instruction inflation that overwhelms latency-hiding gains. Also avoid when xmask=tl.full([XBLOCK], True) was intentionally used to skip masking for performance on evenly-divisible workloads, as adding real masking adds overhead.

---

## Cumulative Progress

```
Baseline:  12290 ns  (1.000×)

Final best: 12290 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
