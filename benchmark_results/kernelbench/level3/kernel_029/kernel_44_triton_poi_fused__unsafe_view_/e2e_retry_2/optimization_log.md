# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7390 ns — **Best latency**: 7390 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_44_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 18
- **Duration**: 7390 ns
- **SM Throughput**: 50.84 % | **Memory**: 25.00 % | **DRAM**: 25.00 %
- **Achieved Occupancy**: 76.83 %
- **IPC Active**: 2.78 inst/cycle | **No Eligible**: 29.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_to_eliminate_partial_wave | 0.991× | 0.987× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave | 0.993× | 0.312× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7390 ns | 7490 ns | +100 ns (+1.4%) | 23650 ns | +16260 ns (+220.0%) |
| SM Throughput (%) | 50.84 % | 49.76 % | -1.08 pp | 72.59 % | +21.75 pp |
| Memory Throughput (%) | 25.00 % | 24.70 % | -0.30 pp | 16.16 % | -8.84 pp |
| DRAM Throughput (%) | 25.00 % | 24.70 % | -0.30 pp | 7.80 % | -17.20 pp |
| Achieved Occupancy (%) | 76.83 % | 76.89 % | +0.06 pp | 87.75 % | +10.92 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 5.57 | +2.79 |
| No Eligible (%) | 29.43 % | 30.02 % | +0.59 pp | 21.30 % | -8.13 pp |
| IPC Active | 2.78 inst/cycle | 2.77 inst/cycle | -0.01 | 3.14 inst/cycle | +0.36 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_to_eliminate_partial_wave` (0.312×)

Adding a @triton.autotune decorator to select a larger XBLOCK was intended to eliminate the partial-wave tail flagged by the LaunchConfiguration NCU rule (estimated 33% speedup). However, the autotune overhead and the selected configuration (XBLOCK=256, num_warps=4) caused registers per thread to jump from 18 to 32, reducing occupancy headroom and increasing total executed instructions by 4.5x, resulting in a 3.2x slowdown (7.39 us → 23.65 us). The grid was not updated to use a lambda (meta['XBLOCK']), so the grid remained fixed at cdiv(xnumel, 128) regardless of the autotuned XBLOCK, causing each thread to process more elements than intended and massively inflating instruction count.

> **Applicable when**: LaunchConfiguration NCU rule fires with estimated speedup >20%; kernel has low register count (<24 regs/thread), zero shared memory, and a partial wave >20% of total runtime.

> **Avoid when**: The grid at the call site is not updated to a lambda using meta['XBLOCK'] — keeping a fixed grid while autotuning XBLOCK causes each thread to cover more elements, exploding instruction count and causing severe regression. Also avoid when register count is already near the SM limit, as larger blocks may spill.

---

## Cumulative Progress

```
Baseline:  7390 ns  (1.000×)

Final best: 7390 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
