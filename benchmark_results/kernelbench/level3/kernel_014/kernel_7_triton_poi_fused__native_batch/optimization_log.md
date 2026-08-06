# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3100 ns — **Best latency**: 2400 ns (**1.2917×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 28
- **Duration**: 3100 ns
- **SM Throughput**: 9.10 % | **Memory**: 8.58 % | **DRAM**: 1.01 %
- **Achieved Occupancy**: 21.74 %
- **IPC Active**: 0.96 inst/cycle | **No Eligible**: 75.16 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_better_occupancy | 1.038× | 1.099× | ✓ |  |
| B | fix_tile_sizes_and_autotune_for_underutilized_transpose_kernel | 1.076× | 1.165× | ✓ |  |
| C | increase_tile_sizes_and_autotune_for_underutilized_transpose_kernel | — | fail | ✓ |  |
| D | autotune_tile_sizes_to_increase_grid_and_occupancy | 1.053× | 1.292× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_sizes_an... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 2820 ns | -280 ns (-9.0%) | 2660 ns | -440 ns (-14.2%) | 2400 ns | -700 ns (-22.6%) |
| SM Throughput (%) | 9.10 % | 5.51 % | -3.59 pp | 5.60 % | -3.50 pp | 1.87 % | -7.23 pp |
| Memory Throughput (%) | 8.58 % | 7.12 % | -1.46 pp | 5.16 % | -3.42 pp | 1.30 % | -7.28 pp |
| DRAM Throughput (%) | 1.01 % | 4.35 % | +3.34 pp | 1.18 % | +0.17 pp | 1.30 % | +0.29 pp |
| Achieved Occupancy (%) | 21.74 % | 10.99 % | -10.75 pp | 8.82 % | -12.92 pp | 7.52 % | -14.22 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 16.00 register/thread | -12 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 75.16 % | 82.56 % | +7.40 pp | 75.46 % | +0.30 pp | 90.27 % | +15.11 pp |
| IPC Active | 0.96 inst/cycle | 0.51 inst/cycle | -0.45 | 0.78 inst/cycle | -0.18 | 0.21 inst/cycle | -0.75 |

---

## Iteration 2

**Best so far**: 1.2917× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_underutilized_transpose_kernel | 0.022× | 0.010× | ✓ |  |
| B | persistent_tile_scheduling_with_correct_meta_grid | 0.100× | 0.042× | ✓ |  |
| C | persistent_tile_scheduling | 0.093× | 0.039× | ✓ |  |
| D | add_triton_autotune | 1.090× | 1.292× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 299680 ns | +296580 ns (+9567.1%) | 73980 ns | +70880 ns (+2286.5%) | 78690 ns | +75590 ns (+2438.4%) | 2400 ns | -700 ns (-22.6%) |
| SM Throughput (%) | 9.10 % | 53.05 % | +43.95 pp | 19.77 % | +10.67 pp | 26.73 % | +17.63 pp | 1.74 % | -7.36 pp |
| Memory Throughput (%) | 8.58 % | 51.25 % | +42.67 pp | 27.89 % | +19.31 pp | 26.03 % | +17.45 pp | 1.31 % | -7.27 pp |
| DRAM Throughput (%) | 1.01 % | 0.01 % | -1.00 pp | 0.04 % | -0.97 pp | 0.04 % | -0.97 pp | 1.31 % | +0.30 pp |
| Achieved Occupancy (%) | 21.74 % | 24.24 % | +2.50 pp | 12.12 % | -9.62 pp | 12.12 % | -9.62 pp | 7.10 % | -14.64 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 31.00 register/thread | +3 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 75.16 % | 47.31 % | -27.85 pp | 79.69 % | +4.53 pp | 72.24 % | -2.92 pp | 89.16 % | +14.00 pp |
| IPC Active | 0.96 inst/cycle | 2.12 inst/cycle | +1.16 | 0.81 inst/cycle | -0.15 | 1.11 inst/cycle | +0.15 | 0.26 inst/cycle | -0.70 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling_with_correct_meta_grid` (0.022×)

Replacing a clean 2D-pid Triton kernel with a persistent while-loop (tile_id += num_programs) while keeping the original 512-block 2D launch grid caused a 45x slowdown (2.56 µs → 139.9 µs). The grid remained at 512 blocks while the loop iterated over all tiles per CTA, causing every tile to be processed 512 times instead of once — executed instructions exploded from 32K to 43M (1341x increase). Despite the CPIStall metric improving (Warp Cycles Per Issued Instruction dropped from 13.96 to 4.79) and L2 hit rate rising to 98.64% (from 28.19%), the sheer instruction volume dominated and duration increased 55x. Waves per SM remained unchanged at 0.12, confirming the persistent loop provided zero SM utilization benefit when the grid was not reduced.

> **Applicable when**: Never: this is a negative example. The persistent while-loop with an unchanged multi-block grid is always catastrophically wrong regardless of how the loop body is structured.

> **Avoid when**: Any kernel where the persistent while-loop is added without simultaneously reducing the launch grid to a 1D grid of size min(num_sms, num_tiles). Even if individual per-instruction metrics improve (lower Warp Cycles Per Issued Instruction, higher L2 hit rate), the instruction count explosion from redundant tile processing dominates and causes severe regression.

#### ❌ Plan C: `persistent_tile_scheduling` (0.039×)

Replacing a 2D pid-based Triton kernel with a persistent 1D tile-loop kernel caused a catastrophic 25x slowdown (2.4 µs → 78.7 µs) despite waves-per-SM remaining at 0.12. The root cause is that the persistent loop computes num_tiles_x and num_tiles_y inside the kernel using tl.cdiv on runtime values (xnumel, ynumel), which are not constexpr — this forces the compiler to emit scalar integer division instructions inside the while-loop body, exploding executed instructions from 31K to 16.6M (534x increase). The strategy is fundamentally misapplied when the grid is already too small to benefit from persistence (only 512 blocks for 132 SMs), and when the tile-count arithmetic cannot be hoisted out of the loop due to non-constexpr dimensions.

> **Applicable when**: Never applicable when the total tile count is already small (< NUM_SMS) and the kernel body is a simple load/store with no compute to hide latency. Persistent scheduling only helps when: (1) the grid has many more tiles than SMs so each CTA processes multiple tiles amortizing launch overhead, AND (2) tile-count arithmetic can be computed outside the kernel (passed as constexpr or precomputed args) to avoid per-iteration scalar divisions.

> **Avoid when**: Total tile count (num_tiles_x * num_tiles_y) is small (e.g., < 512 for a 132-SM GPU); kernel body is a simple memory copy/transpose with minimal arithmetic; num_tiles_x/num_tiles_y cannot be passed as tl.constexpr (forcing runtime tl.cdiv inside the loop body); waves_per_sm is already < 1.0 due to grid being too small — persistent scheduling cannot fix an inherently small grid.

---

## Iteration 3

**Best so far**: 1.2917× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_imc_stalls_via_eviction_policy_and_address_simplification | 0.019× | 0.011× | ✓ |  |
| B | remove_persistent_loop_revert_to_2d_program_id_indexing | 1.029× | 1.165× | ✓ |  |
| C | revert_persistent_loop_to_clean_2d_pid_kernel | 1.071× | 1.211× | ✓ |  |
| D | revert_to_2d_grid_with_right_sized_tiles_and_store_eviction_fix | 1.058× | 1.127× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_imc_stalls... | Δ | remove_persistent... | Δ | revert_persistent... | Δ | revert_to_2d_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 291330 ns | +288230 ns (+9297.7%) | 2660 ns | -440 ns (-14.2%) | 2560 ns | -540 ns (-17.4%) | 2750 ns | -350 ns (-11.3%) |
| SM Throughput (%) | 9.10 % | 56.64 % | +47.54 pp | 5.94 % | -3.16 pp | 1.77 % | -7.33 pp | 6.44 % | -2.66 pp |
| Memory Throughput (%) | 8.58 % | 52.47 % | +43.89 pp | 5.47 % | -3.11 pp | 1.22 % | -7.36 pp | 5.44 % | -3.14 pp |
| DRAM Throughput (%) | 1.01 % | 0.01 % | -1.00 pp | 1.18 % | +0.17 pp | 1.22 % | +0.21 pp | 1.14 % | +0.13 pp |
| Achieved Occupancy (%) | 21.74 % | 24.24 % | +2.50 pp | 8.80 % | -12.94 pp | 6.53 % | -15.21 pp | 8.34 % | -13.40 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 16.00 register/thread | -12 | 30.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 75.16 % | 42.45 % | -32.71 pp | 80.12 % | +4.96 pp | 88.89 % | +13.73 pp | 76.43 % | +1.27 pp |
| IPC Active | 0.96 inst/cycle | 2.30 inst/cycle | +1.34 | 0.78 inst/cycle | -0.18 | 0.23 inst/cycle | -0.73 | 0.75 inst/cycle | -0.21 |


### Key Learnings

#### ✅ Plan B: `remove_persistent_loop_revert_to_2d_program_id_indexing` (1.165×)

When a Triton kernel has been incorrectly modified to use a persistent tile-scheduling loop (for tile_id in range(pid, num_tiles, num_programs)) but the grid was not properly reduced to match, reverting to simple 2D tl.program_id(0)/tl.program_id(1) indexing eliminates the loop overhead. In this case the kernel was in a state where CPIStall (59.7% of stall cycles from IMC misses) and high Warp Cycles Per Issued Instruction (22.15) indicated instruction-level inefficiency from the loop's scalar arithmetic; removing the loop dropped Warp Cycles Per Issued Instruction from 22.15 to 6.82 (3.2x improvement) and SM Busy from 7.7% to 20.4%, yielding a 1.17x speedup. Note that executed instructions actually increased (31K→119K) and registers rose (16→30) in the after state, suggesting the 'before' profiling here captured a partially-optimized state rather than the catastrophic persistent-loop regression; the net effect was still positive because the IMC stall bottleneck was resolved.

> **Applicable when**: A Triton kernel contains a persistent tile-scheduling loop (for/while over tile_id) but the grid was not reduced to min(num_sms, num_tiles); CPIStall rule fires with IMC miss stalls >50% of warp cycles; Warp Cycles Per Issued Instruction is elevated (>15); the kernel is a small-grid pointwise or transpose op where total tiles << grid size.

> **Avoid when**: The persistent loop is correctly paired with a reduced 1D grid of size min(num_sms, num_tiles) and each CTA genuinely processes multiple tiles; the kernel is large enough that waves_per_sm > 1.0 and the loop amortizes launch overhead; the IMC stall is not the dominant bottleneck.

#### ✅ Plan C: `revert_persistent_loop_to_clean_2d_pid_kernel` (1.211×)

After a prior iteration introduced a persistent tile-scheduling while-loop that caused a 25x regression (by exploding executed instructions via runtime tl.cdiv inside the loop body), reverting to a clean 2D-pid kernel (tl.program_id(0) for X, tl.program_id(1) for Y, no while-loop) recovered most of the lost performance. The key structural change was removing the while-loop and all runtime tl.cdiv calls from inside the kernel body, which reduced Warp Cycles Per Issued Instruction from 22.15 to 13.96 (37% reduction) and eliminated the CPIStall IMC-miss stall that accounted for 59.7% of stall cycles. The speedup here is modest (1.21x) because the baseline being compared was already the post-regression kernel (2.4 µs), not the catastrophically broken persistent version (78 µs); the clean 2D-pid kernel is the correct reference implementation for this problem shape.

> **Applicable when**: A Triton kernel was previously modified to use a persistent while-loop with runtime tl.cdiv calls inside the kernel body, causing CPIStall (IMC miss) stalls accounting for >50% of warp stall cycles; Warp Cycles Per Issued Instruction is elevated (>15); the kernel is a simple 2D memory transpose/reshape with no reduction; total tile count is small relative to SM count (waves_per_sm < 0.5).

> **Avoid when**: The kernel genuinely benefits from persistent scheduling (total tiles >> SM count, each CTA processes many tiles); the while-loop overhead is amortized across many iterations; the kernel has complex inter-tile dependencies that require the persistent pattern.

---

## Iteration 4

**Best so far**: 1.2917× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_store_policy_for_strided_scatter_writes | 1.055× | 1.276× | ✓ |  |
| B | evict_first_store_policy_for_strided_scatter_writes | 1.075× | 1.088× | ✓ |  |
| C | fix_autotune_grid_lambda_to_use_meta_tile_sizes | 0.103× | 0.022× | ✓ |  |
| D | revert_broken_persistent_loop_to_clean_2d_pid_with_autotune | 1.045× | 1.115× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_store... | Δ | evict_first_store... | Δ | fix_autotune_grid... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3100 ns | 2430 ns | -670 ns (-21.6%) | 2850 ns | -250 ns (-8.1%) | 139900 ns | +136800 ns (+4412.9%) | 2780 ns | -320 ns (-10.3%) |
| SM Throughput (%) | 9.10 % | 1.91 % | -7.19 pp | 5.86 % | -3.24 pp | 38.94 % | +29.84 pp | 5.96 % | -3.14 pp |
| Memory Throughput (%) | 8.58 % | 1.29 % | -7.29 pp | 5.22 % | -3.36 pp | 5.87 % | -2.71 pp | 5.04 % | -3.54 pp |
| DRAM Throughput (%) | 1.01 % | 1.29 % | +0.28 pp | 1.10 % | +0.09 pp | 0.02 % | -0.99 pp | 1.12 % | +0.11 pp |
| Achieved Occupancy (%) | 21.74 % | 6.56 % | -15.18 pp | 8.73 % | -13.01 pp | 12.05 % | -9.69 pp | 8.94 % | -12.80 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 30.00 register/thread | +2 | 21.00 register/thread | -7 | 30.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 | 0.12 | -0.12 |
| No Eligible (%) | 75.16 % | 88.76 % | +13.60 pp | 75.59 % | +0.43 pp | 59.68 % | -15.48 pp | 77.44 % | +2.28 pp |
| IPC Active | 0.96 inst/cycle | 0.24 inst/cycle | -0.72 | 0.80 inst/cycle | -0.16 | 1.59 inst/cycle | +0.63 | 0.85 inst/cycle | -0.11 |


### Key Learnings

#### ✅ Plan A: `evict_first_store_policy_for_strided_scatter_writes` (1.276×)

Adding eviction_policy='evict_first' exclusively to the tl.store call (leaving the tl.load with evict_last unchanged) in a Triton kernel with non-coalesced scatter writes (82% excessive sectors, 4/32 bytes utilized per sector per NCU) produced a clean 1.28x speedup with a single-token diff. Unlike prior iterations that conflated this change with persistent loop rewrites, this isolated application confirms the causal mechanism: the scatter store bypasses L1 via the SASS .cs streaming hint, preventing cache pollution of the coalesced load data. The duration dropped from 2.56 µs to 2.43 µs with no change to register count, grid size, tile sizes, or instruction count, confirming the speedup is purely from reduced L1 thrashing on the store path.

> **Applicable when**: Triton kernel performs strided or non-coalesced scatter writes where UncoalescedGlobalAccess NCU rule fires (>50% excessive sectors) and MemoryCacheAccessPattern shows <16/32 bytes utilized per sector on L2 misses; the store access pattern has poor spatial locality (e.g., transpose, strided indexing like y0 + 128*x2 + 1152*y1); L2 hit rate is low (<40%); kernel is small (waves_per_sm < 0.5) so there are no other warps to hide L1 eviction latency.

> **Avoid when**: Store writes are coalesced and spatially local (evict_first would reduce beneficial L1 reuse); the same memory region is read back shortly after being written in the same kernel; kernel is compute-bound rather than memory-latency-bound; do not combine with persistent loop rewrites as those cause catastrophic regressions independently.

#### ✅ Plan B: `evict_first_store_policy_for_strided_scatter_writes` (1.276×)

Adding eviction_policy='evict_first' exclusively to the tl.store call (leaving the tl.load with evict_last unchanged) in a Triton kernel with non-coalesced scatter writes (82% excessive sectors, 4/32 bytes utilized per sector per NCU) produced a clean 1.28x speedup with a single-token diff. Unlike prior iterations that conflated this change with persistent loop rewrites, this isolated application confirms the causal mechanism: the scatter store bypasses L1 via the SASS .cs streaming hint, preventing cache pollution of the coalesced load data. The duration dropped from 2.56 µs to 2.43 µs with no change to register count, grid size, tile sizes, or instruction count, confirming the speedup is purely from reduced L1 thrashing on the store path.

> **Applicable when**: Triton kernel performs strided or non-coalesced scatter writes where UncoalescedGlobalAccess NCU rule fires (>50% excessive sectors) and MemoryCacheAccessPattern shows <16/32 bytes utilized per sector on L2 misses; the store access pattern has poor spatial locality (e.g., transpose, strided indexing like y0 + 128*x2 + 1152*y1); L2 hit rate is low (<40%); kernel is small (waves_per_sm < 0.5) so there are no other warps to hide L1 eviction latency.

> **Avoid when**: Store writes are coalesced and spatially local (evict_first would reduce beneficial L1 reuse); the same memory region is read back shortly after being written in the same kernel; kernel is compute-bound rather than memory-latency-bound; do not combine with persistent loop rewrites as those cause catastrophic regressions independently.

---

## Cumulative Progress

```
Baseline:  3100 ns  (1.000×)
Iter 1     2400 ns  (1.292×)  — autotune_tile_sizes_to_increase_grid_and_occupancy

Final best: 2400 ns  (1.2917×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 4096
-    xnumel = 9
+    # Clean 2D pid — NO while-loop, NO runtime tl.cdiv inside kernel
+    xoffset = tl.program_id(0) * XBLOCK
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
+    xindex  = xoffset + tl.arange(0, XBLOCK)[None, :]
+    yindex  = yoffset + tl.arange(0, YBLOCK)[:, None]
+    xmask   = xindex < xnumel
+    ymask   = yindex < ynumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 128)
+    y0 = yindex % 128
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=ymask & xmask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask=ymask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +43,7 @@

     ynumel = 4096
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7[grid](arg_0, arg_1, 4096, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7[grid](arg_0, arg_1, 4096, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
