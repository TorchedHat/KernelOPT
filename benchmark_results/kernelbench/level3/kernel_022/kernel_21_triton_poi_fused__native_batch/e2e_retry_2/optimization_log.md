# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4190 ns — **Best latency**: 4190 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2450,1,1), Waves/SM: 1.16
- **Registers/thread**: 22
- **Duration**: 4190 ns
- **SM Throughput**: 25.17 % | **Memory**: 13.39 % | **DRAM**: 12.29 %
- **Achieved Occupancy**: 63.25 %
- **IPC Active**: 2.01 inst/cycle | **No Eligible**: 50.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_eliminate_partial_wave | 0.964× | 0.391× | ✓ |  |
| B | increase_xblock_tile_size_with_autotune | 0.919× | 0.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_e... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4190 ns | 10720 ns | +6530 ns (+155.8%) | 4670 ns | +480 ns (+11.5%) |
| SM Throughput (%) | 25.17 % | 65.90 % | +40.73 pp | 43.42 % | +18.25 pp |
| Memory Throughput (%) | 13.39 % | 16.42 % | +3.03 pp | 15.92 % | +2.53 pp |
| DRAM Throughput (%) | 12.29 % | 4.78 % | -7.51 pp | 10.99 % | -1.30 pp |
| Achieved Occupancy (%) | 63.25 % | 42.40 % | -20.85 pp | 70.03 % | +6.78 pp |
| Registers/Thread | 22.00 register/thread | 58.00 register/thread | +36 | 32.00 register/thread | +10 |
| Waves/SM | 1.16 | 4.64 | +3.48 | 1.16 | +0.00 |
| No Eligible (%) | 50.32 % | 20.29 % | -30.03 pp | 24.34 % | -25.98 pp |
| IPC Active | 2.01 inst/cycle | 3.18 inst/cycle | +1.17 | 2.97 inst/cycle | +0.96 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_eliminate_partial_wave` (0.391×)

Adding @triton.autotune with large XBLOCK configs (up to 2048) to a pointwise kernel that had a partial-wave tail caused a severe 2.56x regression. The winning autotuned config (XBLOCK=256, block_size=256) increased registers per thread from 22 to 58, cutting theoretical occupancy from 100% to 50% and achieved occupancy from 63% to 42%. Although warp stall cycles improved (19.8 → 8.3 cycles/instruction) and compute throughput rose (25% → 66%), the 6.7x increase in executed instructions (842K → 5.6M) and the register-pressure-induced occupancy collapse caused duration to increase from 4.19µs to 10.72µs. The restore_value=['in_out_ptr0'] parameter in @triton.autotune may have introduced redundant memory traffic by restoring the in-place output buffer between autotuning trials, and the larger XBLOCK configs likely caused register spilling that dominated the runtime.

> **Applicable when**: Never: this specific combination of autotune + restore_value on in-place buffers + large XBLOCK configs caused register pressure explosion and instruction count blowup on a short pointwise kernel.

> **Avoid when**: Do NOT use @triton.autotune with restore_value on in-place output pointers for short pointwise kernels (duration < 10µs). Do NOT include large XBLOCK configs (≥512) in autotune search space when the kernel already has 22+ registers per thread at baseline — larger tiles will push register count above 32 and halve theoretical occupancy. Do NOT apply autotune to kernels where the baseline already achieves >60% occupancy with low register count, as the autotuner overhead and register pressure from larger configs will dominate.

---

## Cumulative Progress

```
Baseline:  4190 ns  (1.000×)

Final best: 4190 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
