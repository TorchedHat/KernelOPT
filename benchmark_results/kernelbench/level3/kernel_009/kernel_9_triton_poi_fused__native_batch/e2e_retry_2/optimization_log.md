# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7420 ns — **Best latency**: 7420 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7420 ns
- **SM Throughput**: 38.53 % | **Memory**: 16.87 % | **DRAM**: 11.07 %
- **Achieved Occupancy**: 56.64 %
- **IPC Active**: 2.24 inst/cycle | **No Eligible**: 43.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_latency | 0.816× | 0.768× | ✓ |  |
| B | autotune_xblock_for_occupancy_and_cache_reuse | 0.996× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 9660 ns | +2240 ns (+30.2%) | 7550 ns | +130 ns (+1.8%) |
| SM Throughput (%) | 38.53 % | 66.98 % | +28.45 pp | 45.65 % | +7.12 pp |
| Memory Throughput (%) | 16.87 % | 18.17 % | +1.30 pp | 17.03 % | +0.16 pp |
| DRAM Throughput (%) | 11.07 % | 8.49 % | -2.58 pp | 10.90 % | -0.17 pp |
| Achieved Occupancy (%) | 56.64 % | 81.53 % | +24.89 pp | 59.38 % | +2.74 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 43.32 % | 16.81 % | -26.51 pp | 35.01 % | -8.31 pp |
| IPC Active | 2.24 inst/cycle | 3.32 inst/cycle | +1.08 | 2.63 inst/cycle | +0.39 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_reduce_partial_wave_and_latency` (0.768×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) doubled register usage per thread (17→32) and more than doubled total executed instructions (2.29M→5.03M), increasing kernel duration from 7.42µs to 9.66µs despite improving occupancy (56.6%→81.5%) and scheduler eligibility (43.3%→16.8% no-eligible cycles). The grid size remained unchanged at 7840 because the call site still used the hardcoded grid formula `triton.cdiv(xnumel, 128)` instead of the lambda `meta`-based grid, so the autotune's larger XBLOCK configs ran with the wrong (too-small) grid, causing each thread block to process far more elements than intended and inflating instruction count. Even when the grid is correct, for short-duration memory-latency-bound pointwise kernels, the autotune overhead and increased register pressure from larger tiles can outweigh the partial-wave savings.

> **Applicable when**: Kernel has a large partial-wave tail (>20% overhead), low achieved occupancy (<60%), and high no-eligible-warp cycles (>40%) due to L1TEX stalls, with very small per-block work (XBLOCK=128 or similar).

> **Avoid when**: The call-site grid formula is not updated to use `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` — leaving a hardcoded grid causes the wrong number of blocks to launch, inflating instruction count and causing regression. Also avoid when the kernel is already short (<10µs) and autotune overhead dominates, or when larger XBLOCK causes register spill (registers_per_thread doubles or more).

---

## Cumulative Progress

```
Baseline:  7420 ns  (1.000×)

Final best: 7420 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
