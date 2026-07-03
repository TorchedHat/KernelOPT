# Kernel Optimization Log

## Summary

- **Kernel**: `triplet_margin_loss_row_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1450000 ns — **Best latency**: 1450000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triplet_margin_loss_row_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_099/synthesis/kernel_0_fused_fused_add.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 28
- **Duration**: 1450000 ns
- **SM Throughput**: 73.20 % | **Memory**: 82.03 % | **DRAM**: 45.19 %
- **Achieved Occupancy**: 97.89 %
- **IPC Active**: 2.04 inst/cycle | **No Eligible**: 48.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fuse_row_reduction_and_loss_reduction_into_single_kernel | 1.000× | 1.000× | ✓ |  |
| C | vectorized_loads_with_row_tiling | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_row_reductio... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1450000 ns | 1450000 ns | +0 ns (+0.0%) | 1450000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 73.20 % | 73.20 % | +0.00 pp | 73.21 % | +0.01 pp |
| Memory Throughput (%) | 82.03 % | 82.04 % | +0.01 pp | 82.05 % | +0.02 pp |
| DRAM Throughput (%) | 45.19 % | 45.19 % | +0.00 pp | 45.19 % | +0.00 pp |
| Achieved Occupancy (%) | 97.89 % | 97.87 % | -0.02 pp | 97.90 % | +0.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 48.88 % | 48.88 % | +0.00 pp | 48.89 % | +0.01 pp |
| IPC Active | 2.04 inst/cycle | 2.04 inst/cycle | +0.00 | 2.04 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | multi_row_tiling_coalesced_stores | 0.409× | 0.691× | ✓ |  |
| B | multi_row_per_cta_coalesced_stores | 0.354× | 0.597× | ✓ |  |
| C | multi_row_per_cta_coalesced_stores | 0.409× | 0.691× | ✓ |  |


### Metric Comparison

| Metric | Baseline | multi_row_tiling_... | Δ | multi_row_per_cta... | Δ | multi_row_per_cta... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1450000 ns | 2100000 ns | +650000 ns (+44.8%) | 2430000 ns | +980000 ns (+67.6%) | 2100000 ns | +650000 ns (+44.8%) |
| SM Throughput (%) | 73.20 % | 67.66 % | -5.54 pp | 63.36 % | -9.84 pp | 67.92 % | -5.28 pp |
| Memory Throughput (%) | 82.03 % | 57.89 % | -24.14 pp | 69.10 % | -12.93 pp | 58.11 % | -23.92 pp |
| DRAM Throughput (%) | 45.19 % | 31.27 % | -13.92 pp | 26.97 % | -18.22 pp | 31.23 % | -13.96 pp |
| Achieved Occupancy (%) | 97.89 % | 97.80 % | -0.09 pp | 24.83 % | -73.06 pp | 97.80 % | -0.09 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 106.00 register/thread | +78 | 32.00 register/thread | +4 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 62.06 | +46.54 | 15.52 | +0.00 |
| No Eligible (%) | 48.88 % | 32.21 % | -16.67 pp | 36.84 % | -12.04 pp | 32.36 % | -16.52 pp |
| IPC Active | 2.04 inst/cycle | 2.71 inst/cycle | +0.67 | 2.52 inst/cycle | +0.48 | 2.70 inst/cycle | +0.66 |


### Key Learnings

#### ❌ Plan A: `multi_row_tiling_coalesced_stores` (0.691×)

Tiling BLOCK_M rows per CTA to coalesce scalar stores into vector stores nearly doubled total executed instructions (584M → 1114M) because the 2D [BLOCK_M × BLOCK_N] tile dramatically increased register pressure and arithmetic work per CTA. Although memory throughput dropped (82% → 58% Mem Busy) and warp stall cycles improved (30.6 → 23.1 cycles/instruction), the 1.9× instruction count increase dominated, causing a 45% runtime regression (1.45ms → 2.10ms). The kernel was originally memory-bound at the L1/TEX level; adding BLOCK_M rows per CTA shifted it toward compute-bound while also increasing register usage (28 → 32 registers/thread), and the net effect was slower execution.

> **Applicable when**: Never — this approach caused regression. Do not apply multi-row tiling to coalesce scalar output stores when the kernel is already memory-bound at L1/TEX and the dominant bottleneck is load latency, not store coalescing. The instruction count explosion from 2D tiling outweighs the store coalescing benefit.

> **Avoid when**: When the kernel is L1/TEX memory-bound (Mem Busy > 80%) with high instruction count per row and the output is a single scalar per CTA. Adding BLOCK_M rows per CTA creates a [BLOCK_M × BLOCK_N] 2D tile that nearly doubles executed instructions and increases register pressure, causing regression even though store coalescing improves. Avoid when executed instructions are already high relative to memory bandwidth.

#### ❌ Plan B: `multi_row_per_cta_coalesced_stores` (0.691×)

Switching from one-row-per-CTA scalar stores to BLOCK_M-rows-per-CTA 2D tiling nearly doubled executed instructions (584M → 1114M) and increased duration from 1.45ms to 2.10ms despite reducing memory throughput from 82% to 58%. The 2D [BLOCK_M, BLOCK_N] tile layout inflates register pressure (28→32 regs/thread) and instruction count because each CTA now processes BLOCK_M×BLOCK_N elements with 2D offset arithmetic, masking, and column-axis reductions — the compute overhead outweighs the store-coalescing benefit when the kernel is already memory-bound with high occupancy. This anti-pattern occurs when the uncoalesced store volume is small relative to total memory traffic (only 2 scalar floats per CTA vs. large BLOCK_N loads), so fixing store coalescing does not meaningfully reduce total DRAM traffic.

> **Applicable when**: Never beneficial in this configuration: the uncoalesced store is only 2 floats per CTA while the dominant traffic is BLOCK_N-wide loads; 2D tiling doubles instruction count without proportionally reducing DRAM traffic.

> **Avoid when**: When the kernel is already memory-bound (Mem Busy > 75%) with high occupancy (>95%) and the uncoalesced store volume is negligible compared to total load traffic. When the per-CTA compute work is dominated by a long reduction loop (large N), adding 2D tiling inflates instruction count faster than it reduces store overhead. Avoid when registers per thread are already near the block-limit boundary (28 regs → 32 regs can reduce theoretical occupancy).

#### ❌ Plan C: `multi_row_per_cta_coalesced_stores` (0.691×)

Switching from one-row-per-CTA scalar stores to BLOCK_M-rows-per-CTA 2D tiling nearly doubled executed instructions (584M → 1114M) and increased duration from 1.45ms to 2.10ms despite reducing memory throughput from 82% to 58%. The 2D [BLOCK_M, BLOCK_N] tile layout inflates register pressure (28→32 regs/thread) and instruction count because each CTA now processes BLOCK_M×BLOCK_N elements with 2D offset arithmetic, masking, and column-axis reductions — the compute overhead outweighs the store-coalescing benefit when the kernel is already memory-bound with high occupancy. This anti-pattern occurs when the uncoalesced store volume is small relative to total memory traffic (only 2 scalar floats per CTA vs. large BLOCK_N loads), so fixing store coalescing does not meaningfully reduce total DRAM traffic.

> **Applicable when**: Never beneficial in this configuration: the uncoalesced store is only 2 floats per CTA while the dominant traffic is BLOCK_N-wide loads; 2D tiling doubles instruction count without proportionally reducing DRAM traffic.

> **Avoid when**: When the kernel is already memory-bound (Mem Busy > 75%) with high occupancy (>95%) and the uncoalesced store volume is negligible compared to total load traffic. When the per-CTA compute work is dominated by a long reduction loop (large N), adding 2D tiling inflates instruction count faster than it reduces store overhead. Avoid when registers per thread are already near the block-limit boundary (28 regs → 32 regs can reduce theoretical occupancy).

---

## Cumulative Progress

```
Baseline:  1450000 ns  (1.000×)

Final best: 1450000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,6 +2,7 @@

 import triton
 import triton.language as tl
 from triton.language.extra import libdevice
+
 
 
 @triton.autotune(
@@ -75,6 +76,7 @@

     key=['M'],
 )
 @triton.jit
+
 def triplet_margin_loss_reduce_kernel(
     dist_pos_ptr, dist_neg_ptr,
     output_ptr,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
