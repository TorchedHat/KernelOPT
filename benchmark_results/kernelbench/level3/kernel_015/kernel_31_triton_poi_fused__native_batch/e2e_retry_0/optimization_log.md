# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7710 ns — **Best latency**: 7710 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_31_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7710 ns
- **SM Throughput**: 38.64 % | **Memory**: 16.84 % | **DRAM**: 10.69 %
- **Achieved Occupancy**: 57.21 %
- **IPC Active**: 2.26 inst/cycle | **No Eligible**: 42.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.916× | 0.787× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.976× | 0.471× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7710 ns | 9790 ns | +2080 ns (+27.0%) | 16380 ns | +8670 ns (+112.5%) |
| SM Throughput (%) | 38.64 % | 66.70 % | +28.06 pp | 75.32 % | +36.68 pp |
| Memory Throughput (%) | 16.84 % | 18.13 % | +1.29 pp | 16.59 % | -0.25 pp |
| DRAM Throughput (%) | 10.69 % | 8.41 % | -2.28 pp | 5.02 % | -5.67 pp |
| Achieved Occupancy (%) | 57.21 % | 81.47 % | +24.26 pp | 85.20 % | +27.99 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 7.42 | +3.71 |
| No Eligible (%) | 42.56 % | 15.71 % | -26.85 pp | 13.75 % | -28.81 pp |
| IPC Active | 2.26 inst/cycle | 3.34 inst/cycle | +1.08 | 3.43 inst/cycle | +1.17 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.787×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused registers per thread to nearly double (17→32), which increased total executed instructions by 2.2x (2.29M→5.03M) and elapsed cycles by 27% (11449→14518), resulting in a 21% slowdown despite occupancy improving from 57% to 81%. The autotuner selected a config that introduced register pressure or additional mask-check instructions (from None→xmask on all loads/stores), overwhelming the latency-hiding benefit. When a kernel already has very low register usage (≤20 registers/thread) and the bottleneck is L1TEX scoreboard stalls rather than occupancy, increasing tile size via autotune can backfire if the selected config doubles register file usage and instruction count.

> **Applicable when**: Never apply this specific pattern (autotune larger XBLOCK + adding xmask) when the kernel has very low register count (≤20 regs/thread) and the grid is already fixed — the autotune overhead and mask instructions can double instruction count.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when: (1) the original kernel uses None masks (unmasked loads/stores) because xnumel is a multiple of XBLOCK — adding xmask introduces extra instructions; (2) the grid lambda is not updated to use meta['XBLOCK'], leaving the grid size unchanged while the kernel processes more elements per block than the grid covers; (3) register count is already low (≤20) but the autotune configs cause the compiler to allocate significantly more registers (doubling instruction count).

#### ❌ Plan B: `autotune_xblock_eliminate_partial_wave` (0.471×)

Doubling the block size from 128 to 256 (via autotune selecting XBLOCK=256) caused registers per thread to jump from 17 to 32, and executed instructions to balloon from ~2.3M to ~9.6M (4.2x increase). Despite occupancy improving from 57% to 85% and no-eligible-warp cycles dropping from 42.6% to 13.75%, the massive instruction count increase more than offset these gains, resulting in duration increasing from 7.71 µs to 16.38 µs (2.1x slower). When a kernel already has low register pressure at the baseline block size, increasing block size via autotune can trigger compiler register allocation changes that dramatically increase instruction count, negating any latency-hiding benefit.

> **Applicable when**: Kernel has high no-eligible-warp cycles (>40%), low achieved occupancy (<60%), and NCU flags LaunchConfiguration partial-wave overhead. Register pressure is already low at baseline block size.

> **Avoid when**: The kernel's register count nearly doubles when block size is increased (compiler behavior), causing instruction count to explode (>2x). Also avoid when xnumel is exactly divisible by the current XBLOCK (mask overhead from autotune adds unnecessary predicated instructions). Do not apply autotune with larger XBLOCK configs when the baseline already has low register pressure (<=20 regs/thread) — the compiler may reorganize register allocation in ways that dramatically increase instruction count.

---

## Cumulative Progress

```
Baseline:  7710 ns  (1.000×)

Final best: 7710 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
