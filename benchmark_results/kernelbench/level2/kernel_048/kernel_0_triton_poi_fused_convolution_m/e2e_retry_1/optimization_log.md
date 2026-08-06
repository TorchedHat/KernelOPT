# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_mul_sigmoid_tanh_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 531870 ns — **Best latency**: 531870 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_mul_sigmoid_tanh_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_048/kernel_0_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (861056,1,1), Waves/SM: 407.70
- **Registers/thread**: 16
- **Duration**: 531870 ns
- **SM Throughput**: 54.10 % | **Memory**: 32.73 % | **DRAM**: 32.73 %
- **Achieved Occupancy**: 53.21 %
- **IPC Active**: 2.17 inst/cycle | **No Eligible**: 45.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.999× | 0.294× | ✓ |  |
| B | increase_xblock_tile_size_for_latency_hiding | 0.596× | 0.149× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 531870 ns | 1810000 ns | +1278130 ns (+240.3%) | 3560000 ns | +3028130 ns (+569.3%) |
| SM Throughput (%) | 54.10 % | 92.72 % | +38.62 pp | 93.46 % | +39.36 pp |
| Memory Throughput (%) | 32.73 % | 29.14 % | -3.59 pp | 28.50 % | -4.23 pp |
| DRAM Throughput (%) | 32.73 % | 9.61 % | -23.12 pp | 4.89 % | -27.84 pp |
| Achieved Occupancy (%) | 53.21 % | 87.39 % | +34.18 pp | 87.93 % | +34.72 pp |
| Registers/Thread | 16.00 register/thread | 28.00 register/thread | +12 | 28.00 register/thread | +12 |
| Waves/SM | 407.70 | 815.39 | +407.69 | 1630.79 | +1223.09 |
| No Eligible (%) | 45.66 % | 7.12 % | -38.54 pp | 6.47 % | -39.19 pp |
| IPC Active | 2.17 inst/cycle | 3.71 inst/cycle | +1.54 | 3.74 inst/cycle | +1.57 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.294×)

Adding @triton.autotune with larger XBLOCK configs (256, 512, 1024) alongside switching from unconditional loads (xmask=None / tl.full True) to conditional loads (xmask = xindex < xnumel) caused a ~5.8x explosion in executed instructions (227M → 1327M). The autotune selected XBLOCK=256 (block_size doubled), but registers per thread jumped from 16 to 28, halving the block limit from 32 to 8 blocks/SM. Despite improved scheduler eligibility (0.91→9.82 eligible warps/scheduler) and higher compute throughput (54%→93%), the total work done ballooned so severely that wall time increased 3.4x. The root cause is that the conditional xmask introduced per-element predication overhead across a much larger instruction stream, and the autotune overhead itself added latency on top.

> **Applicable when**: Never apply this exact pattern. The combination of adding conditional xmask AND autotune with larger XBLOCK is harmful.

> **Avoid when**: Do not add conditional xmask (xindex < xnumel) when the original kernel uses unconditional loads (tl.full True mask or None mask) and xnumel is already a multiple of the original XBLOCK — the masking adds predication overhead without correctness benefit. Do not add @triton.autotune when the grid lambda is not updated to use meta['XBLOCK'] — the grid stays fixed at the old XBLOCK size while the kernel processes larger tiles, causing massive redundant work. Do not increase XBLOCK when registers per thread will jump significantly (16→28 here), as this reduces blocks per SM and can negate occupancy gains.

#### ❌ Plan B: `increase_xblock_tile_size_for_latency_hiding` (0.149×)

Increasing XBLOCK via @triton.autotune (128→512) while keeping the grid fixed at the original cdiv(xnumel, 128) caused the grid to remain at 861K blocks instead of shrinking proportionally, resulting in each element being processed ~4x redundantly. This inflated executed instructions from 227M to 2.63B (11.6x), increasing kernel duration from 532µs to 3.56ms despite compute throughput rising from 54% to 93%. The implementation flaw was that the grid lambda was not updated to use meta['XBLOCK'], so the grid size did not shrink when XBLOCK grew — the autotune decorator selected a larger block size but the launch still used the old fixed grid, causing massive redundant work.

> **Applicable when**: Never apply this pattern (autotune with mismatched static grid). The correct pattern is always to use a dynamic grid lambda when autotuning XBLOCK.

> **Avoid when**: Do not use @triton.autotune to vary XBLOCK if the launch grid is hardcoded to cdiv(xnumel, fixed_block_size). The grid must always be a lambda using meta['XBLOCK'] to avoid redundant block execution.

---

## Cumulative Progress

```
Baseline:  531870 ns  (1.000×)

Final best: 531870 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
