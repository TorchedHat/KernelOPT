# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3710 ns — **Best latency**: 3710 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_34_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1715,1,1), Waves/SM: 0.81
- **Registers/thread**: 22
- **Duration**: 3710 ns
- **SM Throughput**: 19.21 % | **Memory**: 9.74 % | **DRAM**: 9.74 %
- **Achieved Occupancy**: 61.65 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 50.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_work_per_cta | 0.963× | 1.000× | ✓ |  |
| B | autotune_xblock_for_underutilized_pointwise_kernel | 0.966× | 0.455× | ✓ |  |
| C | increase_xblock_and_autotune_for_better_wave_utilization | 0.966× | 0.921× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.971× | 0.659× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_f... | Δ | increase_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3710 ns | +0 ns (+0.0%) | 8160 ns | +4450 ns (+119.9%) | 4030 ns | +320 ns (+8.6%) | 5630 ns | +1920 ns (+51.8%) |
| SM Throughput (%) | 19.21 % | 19.21 % | +0.00 pp | 60.47 % | +41.26 pp | 35.35 % | +16.14 pp | 48.71 % | +29.50 pp |
| Memory Throughput (%) | 9.74 % | 9.75 % | +0.01 pp | 14.21 % | +4.47 pp | 11.25 % | +1.51 pp | 12.45 % | +2.71 pp |
| DRAM Throughput (%) | 9.74 % | 9.75 % | +0.01 pp | 4.43 % | -5.31 pp | 9.00 % | -0.74 pp | 6.40 % | -3.34 pp |
| Achieved Occupancy (%) | 61.65 % | 57.72 % | -3.93 pp | 41.58 % | -20.07 pp | 56.68 % | -4.97 pp | 74.89 % | +13.24 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 58.00 register/thread | +36 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 3.25 | +2.44 | 0.81 | +0.00 | 1.62 | +0.81 |
| No Eligible (%) | 50.44 % | 49.32 % | -1.12 pp | 19.13 % | -31.31 pp | 28.28 % | -22.16 pp | 23.55 % | -26.89 pp |
| IPC Active | 1.67 inst/cycle | 1.88 inst/cycle | +0.21 | 3.21 inst/cycle | +1.54 | 2.83 inst/cycle | +1.16 | 3.09 inst/cycle | +1.42 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_for_underutilized_pointwise_kernel` (0.455×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise kernel caused the autotuner to select a config that exploded register usage from 22 to 58 registers/thread. This cut theoretical occupancy from 100% to 50% (Block Limit Registers dropped from 21 to 4 blocks/SM) and increased total executed instructions by 6.7x, resulting in 2.2x longer kernel duration despite improved scheduler eligible-warp rate. The structural anti-pattern is introducing large-XBLOCK autotune configs to a kernel where the body already uses enough registers that unrolling a larger tile causes register pressure to spike, trading occupancy for ILP in a way that is net-negative.

> **Applicable when**: Never applicable as described — this is a negative example. The autotune-XBLOCK strategy is only safe when the kernel has very low register usage (≤16 registers/thread) so that larger tiles do not cause register pressure to limit occupancy below the baseline.

> **Avoid when**: Kernel already uses 20+ registers/thread at the baseline XBLOCK. Adding larger XBLOCK configs to autotune will cause the compiler to unroll more iterations, spiking register usage (e.g., 22→58 registers/thread), cutting Block Limit Registers from 21 to 4, halving theoretical occupancy, and massively increasing total instruction count — all net-negative even if scheduler idle rate improves.

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_pointwise_kernel` (0.659×)

Adding @triton.autotune to a small pointwise kernel (grid ~1715 blocks, ~3.7 µs) with the goal of increasing XBLOCK from 128 to 256+ caused a 34% regression. The autotune selected XBLOCK=256 with 8 warps, which increased registers per thread from 22 to 32, reducing Block Limit Registers from 21 to 8 blocks/SM and cutting the maximum concurrent blocks per SM in half. Although waves_per_sm doubled (0.81→1.62) and eligible warps improved (1.79→6.1), the total executed instruction count ballooned ~3.7x (590K→2.17M) and elapsed cycles grew from 5531 to 8399, indicating the larger tile size introduced significant overhead (likely predicated-off threads, extra address computation, or register spill pressure) that outweighed the latency-hiding benefit. The grid was not static (hardcoded to cdiv(xnumel, 128)) so the autotune's dynamic XBLOCK was not reflected in the grid calculation, meaning the grid remained fixed at 1715 blocks regardless of the selected XBLOCK, causing each block to process redundant or out-of-bounds work.

> **Applicable when**: Never apply this exact pattern. The grid must be a lambda using meta['XBLOCK'] when using autotune with variable XBLOCK.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK while keeping a static grid computed from a fixed block size. This causes the grid to be mismatched with the actual tile size selected by autotune, leading to either redundant blocks or missed elements. Also avoid increasing XBLOCK when register pressure will increase beyond the block limit register threshold, as this reduces concurrent blocks per SM and can negate occupancy gains.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | — | fail | ✓ |  |
| B | persistent_tile_scheduling | 0.961× | 0.934× | ✓ |  |
| C | persistent_tile_scheduling | — | fail | ✓ |  |
| D | persistent_tile_scheduling_to_eliminate_partial_wave_and_restore_occupancy | — | fail | ✓ |  |
| E | add_triton_autotune | 0.972× | 0.942× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3970 ns | +260 ns (+7.0%) | 3940 ns | +230 ns (+6.2%) |
| SM Throughput (%) | 19.21 % | 22.12 % | +2.91 pp | 20.41 % | +1.20 pp |
| Memory Throughput (%) | 9.74 % | 9.10 % | -0.64 pp | 9.17 % | -0.57 pp |
| DRAM Throughput (%) | 9.74 % | 9.10 % | -0.64 pp | 9.17 % | -0.57 pp |
| Achieved Occupancy (%) | 61.65 % | 58.73 % | -2.92 pp | 56.96 % | -4.69 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 50.44 % | 50.37 % | -0.07 pp | 56.03 % | +5.59 pp |
| IPC Active | 1.67 inst/cycle | 1.85 inst/cycle | +0.18 | 1.85 inst/cycle | +0.18 |

---

## Cumulative Progress

```
Baseline:  3710 ns  (1.000×)

Final best: 3710 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
