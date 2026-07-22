# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4670 ns — **Best latency**: 4670 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_6_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 22
- **Duration**: 4670 ns
- **SM Throughput**: 25.73 % | **Memory**: 14.20 % | **DRAM**: 14.20 %
- **Achieved Occupancy**: 62.83 %
- **IPC Active**: 2.07 inst/cycle | **No Eligible**: 48.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.915× | 0.854× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.981× | 0.942× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4670 ns | 5470 ns | +800 ns (+17.1%) | 4960 ns | +290 ns (+6.2%) |
| SM Throughput (%) | 25.73 % | 49.06 % | +23.33 pp | 37.37 % | +11.64 pp |
| Memory Throughput (%) | 14.20 % | 14.60 % | +0.40 pp | 15.03 % | +0.83 pp |
| DRAM Throughput (%) | 14.20 % | 12.06 % | -2.14 pp | 13.33 % | -0.87 pp |
| Achieved Occupancy (%) | 62.83 % | 76.17 % | +13.34 pp | 71.92 % | +9.09 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 28.00 register/thread | +6 |
| Waves/SM | 1.48 | 1.48 | +0.00 | 1.48 | +0.00 |
| No Eligible (%) | 48.20 % | 21.42 % | -26.78 pp | 30.89 % | -17.31 pp |
| IPC Active | 2.07 inst/cycle | 3.01 inst/cycle | +0.94 | 2.97 inst/cycle | +0.90 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.854×)

Adding a @triton.autotune decorator with larger XBLOCK values (512–2048) to reduce grid size and eliminate the partial-wave tail (NCU LaunchConfiguration rule, 50% potential speedup) caused a regression of 0.85x. Despite improving scheduler eligibility (No Eligible dropped from 48.2% to 21.4%) and achieved occupancy (62.8% → 76.2%), the executed instruction count more than doubled (965K → 2129K) and registers per thread increased from 22 to 32, indicating the autotune selected a config that dramatically increased per-CTA work and register pressure. The L1/TEX hit rate also dropped from 70.2% to 58.3%, negating the expected cache reuse benefit, and the net effect was a 17% longer kernel duration.

> **Applicable when**: NCU LaunchConfiguration rule fires with high speedup_pct (≥30%), grid produces a large partial wave, and the kernel is a simple pointwise op with no shared memory or reductions.

> **Avoid when**: The kernel already has low register pressure (≤22 registers/thread) and the autotune search space includes very large XBLOCK values (≥1024) that may cause the compiler to spill registers or increase register count significantly (e.g., 22→32), as the instruction count explosion and L1 hit rate degradation can outweigh the partial-wave elimination benefit. Also avoid when the original xmask is tl.full([XBLOCK], True) (unconditional) and switching to conditional masking (xindex < xnumel) adds unnecessary predication overhead for workloads where xnumel is always exactly divisible by XBLOCK.

---

## Cumulative Progress

```
Baseline:  4670 ns  (1.000×)

Final best: 4670 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
