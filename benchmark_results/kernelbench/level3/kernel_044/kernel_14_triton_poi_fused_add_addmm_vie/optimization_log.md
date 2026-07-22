# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_addmm_view_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 245020 ns — **Best latency**: 245250 ns (**0.9991×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_addmm_view_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_14_triton_poi_fused_add_addmm_vie/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 18
- **Duration**: 245020 ns
- **SM Throughput**: 46.58 % | **Memory**: 66.01 % | **DRAM**: 66.01 %
- **Achieved Occupancy**: 57.08 %
- **IPC Active**: 1.43 inst/cycle | **No Eligible**: 63.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_memory_coalescing | 0.998× | 0.199× | ✓ |  |
| B | increase_xblock_with_autotune_software_pipelining | 0.677× | 0.682× | ✓ |  |
| C | increase_xblock_tile_size_for_memory_latency_hiding | 0.807× | 0.683× | ✓ |  |
| D | increase_xblock_with_autotune_for_better_l2_reuse_and_occupancy | 0.677× | 0.681× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 245020 ns | 1230000 ns | +984980 ns (+402.0%) | 359200 ns | +114180 ns (+46.6%) | 358530 ns | +113510 ns (+46.3%) | 360030 ns | +115010 ns (+46.9%) |
| SM Throughput (%) | 46.58 % | 69.51 % | +22.93 pp | 34.24 % | -12.34 pp | 34.44 % | -12.14 pp | 34.10 % | -12.48 pp |
| Memory Throughput (%) | 66.01 % | 34.55 % | -31.46 pp | 44.81 % | -21.20 pp | 44.88 % | -21.13 pp | 44.70 % | -21.31 pp |
| DRAM Throughput (%) | 66.01 % | 13.09 % | -52.92 pp | 44.81 % | -21.20 pp | 44.88 % | -21.13 pp | 44.70 % | -21.31 pp |
| Achieved Occupancy (%) | 57.08 % | 60.42 % | +3.34 pp | 48.86 % | -8.22 pp | 48.63 % | -8.45 pp | 48.91 % | -8.17 pp |
| Registers/Thread | 18.00 register/thread | 40.00 register/thread | +22 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 186.18 | 992.97 | +806.79 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 63.97 % | 30.16 % | -33.81 pp | 65.03 % | +1.06 pp | 65.08 % | +1.11 pp | 65.21 % | +1.24 pp |
| IPC Active | 1.43 inst/cycle | 2.79 inst/cycle | +1.36 | 1.38 inst/cycle | -0.05 | 1.39 inst/cycle | -0.04 | 1.39 inst/cycle | -0.04 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_software_pipelining` (0.682×)

Adding @triton.autotune with larger XBLOCK configs (512–4096) to a memory-latency-bound pointwise kernel caused a 47% slowdown instead of the expected speedup. The autotuned kernel selected a configuration that increased registers per thread from 18 to 32, which reduced achieved occupancy from 57% to 49% and active warps per scheduler from 9.03 to 7.73, worsening the scheduler starvation it was meant to fix. Simultaneously, total executed instructions increased by ~41% (67M → 95M) and DRAM throughput dropped from 66% to 45%, indicating the selected XBLOCK was too large for the register file budget on this kernel, causing register spilling or occupancy-limited execution that outweighed any MLP benefit.

> **Applicable when**: Never: this specific strategy (autotune XBLOCK 512–4096 with num_stages 2–4 on a simple 5-load pointwise kernel with 18 registers/thread baseline) caused regression.

> **Avoid when**: Kernel already has low register count (≤20 registers/thread) and moderate achieved occupancy (~57%); adding large XBLOCK autotune configs inflates register pressure, reduces occupancy further, increases instruction count, and drops both DRAM throughput and compute throughput. Do not apply when the kernel body is trivially simple (few arithmetic ops per load) — the overhead of larger tiles exceeds the MLP benefit.

#### ❌ Plan C: `increase_xblock_tile_size_for_memory_latency_hiding` (0.683×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (18→32), which reduced Block Limit Registers from 21 to 16 blocks/SM and dropped achieved occupancy from 57% to 49%. Despite the larger tile size, L2 hit rate was essentially unchanged (25.3%→25.6%), meaning the expected locality gain did not materialize. The net effect was 46% more executed instructions and 46% longer kernel duration, because the autotune overhead and register spill pressure outweighed any latency-hiding benefit.

> **Applicable when**: Never applicable as described: adding autotune with large XBLOCK configs to a simple pointwise kernel that is already memory-bound and has low register usage (≤20 regs/thread) will increase register pressure without improving L2 locality if the access pattern does not have strong spatial reuse within the tile.

> **Avoid when**: Do not increase XBLOCK via autotune on pointwise kernels where: (1) the kernel already uses few registers (≤20) and occupancy is register-limited at larger tile sizes; (2) the L2 hit rate is low due to streaming access patterns rather than reuse within a tile; (3) the grid is already very large (>100k CTAs) and the bottleneck is per-warp memory latency rather than insufficient parallelism. In these cases, larger tiles increase register count, reduce occupancy, and do not improve cache hit rates.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_l1_bias_reuse | 1.005× | 1.001× | ✓ |  |
| B | reduce_register_pressure_via_small_xblock_configs | 1.008× | 1.007× | ✓ |  |
| C | revert_to_small_xblock_to_restore_register_budget_and_occupancy | 0.999× | 1.000× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 1.012× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | reduce_register_p... | Δ | revert_to_small_x... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 245020 ns | 244830 ns | -190 ns (-0.1%) | 243360 ns | -1660 ns (-0.7%) | 244930 ns | -90 ns (-0.0%) | 244930 ns | -90 ns (-0.0%) |
| SM Throughput (%) | 46.58 % | 46.63 % | +0.05 pp | 23.00 % | -23.58 pp | 45.77 % | -0.81 pp | 46.51 % | -0.07 pp |
| Memory Throughput (%) | 66.01 % | 64.88 % | -1.13 pp | 66.40 % | +0.39 pp | 65.97 % | -0.04 pp | 65.99 % | -0.02 pp |
| DRAM Throughput (%) | 66.01 % | 64.88 % | -1.13 pp | 66.40 % | +0.39 pp | 65.97 % | -0.04 pp | 65.99 % | -0.02 pp |
| Achieved Occupancy (%) | 57.08 % | 57.58 % | +0.50 pp | 31.17 % | -25.91 pp | 56.38 % | -0.70 pp | 56.23 % | -0.85 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 24.00 register/thread | +6 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 93.09 | -93.09 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 63.97 % | 56.20 % | -7.77 pp | 79.44 % | +15.47 pp | 64.41 % | +0.44 pp | 64.41 % | +0.44 pp |
| IPC Active | 1.43 inst/cycle | 1.77 inst/cycle | +0.34 | 0.83 inst/cycle | -0.60 | 1.43 inst/cycle | +0.00 | 1.42 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0004× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_warps_per_block_to_hide_l1tex_latency | 1.008× | 1.005× | ✓ |  |
| C | fuse_fp32_additions_into_fma_instructions | 0.999× | 0.997× | ✓ |  |
| D | remove_redundant_mask_to_enable_vectorized_unmasked_loads | 1.013× | 0.995× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_warps_pe... | Δ | fuse_fp32_additio... | Δ | remove_redundant_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 245020 ns | 243740 ns | -1280 ns (-0.5%) | 245820 ns | +800 ns (+0.3%) | 246180 ns | +1160 ns (+0.5%) |
| SM Throughput (%) | 46.58 % | 45.70 % | -0.88 pp | 46.52 % | -0.06 pp | 46.01 % | -0.57 pp |
| Memory Throughput (%) | 66.01 % | 32.72 % | -33.29 pp | 65.74 % | -0.27 pp | 65.64 % | -0.37 pp |
| DRAM Throughput (%) | 66.01 % | 32.72 % | -33.29 pp | 65.74 % | -0.27 pp | 65.64 % | -0.37 pp |
| Achieved Occupancy (%) | 57.08 % | 48.23 % | -8.85 pp | 56.89 % | -0.19 pp | 55.95 % | -1.13 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 63.97 % | 63.80 % | -0.17 pp | 64.09 % | +0.12 pp | 69.22 % | +5.25 pp |
| IPC Active | 1.43 inst/cycle | 1.45 inst/cycle | +0.02 | 1.44 inst/cycle | +0.01 | 1.24 inst/cycle | -0.19 |

---

## Iteration 4

**Best so far**: 1.0004× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | 2d_tiling_bias_reuse_correct_implementation | 0.165× | 0.162× | ✓ |  |
| B | reorder_loads_for_memory_level_parallelism | 0.998× | 1.000× | ✓ |  |
| C | prefetch_independent_loads_for_ilp_to_hide_l1tex_latency | 1.013× | 0.999× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | 2d_tiling_bias_re... | Δ | reorder_loads_for... | Δ | prefetch_independ... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 245020 ns | 1510000 ns | +1264980 ns (+516.3%) | 244900 ns | -120 ns (-0.0%) | 245250 ns | +230 ns (+0.1%) |
| SM Throughput (%) | 46.58 % | 71.05 % | +24.47 pp | 46.50 % | -0.08 pp | 46.35 % | -0.23 pp |
| Memory Throughput (%) | 66.01 % | 45.63 % | -20.38 pp | 65.98 % | -0.03 pp | 65.37 % | -0.64 pp |
| DRAM Throughput (%) | 66.01 % | 10.81 % | -55.20 pp | 65.98 % | -0.03 pp | 65.37 % | -0.64 pp |
| Achieved Occupancy (%) | 57.08 % | 69.29 % | +12.21 pp | 56.53 % | -0.55 pp | 56.16 % | -0.92 pp |
| Registers/Thread | 18.00 register/thread | 40.00 register/thread | +22 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 186.18 | 248.24 | +62.06 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 63.97 % | 28.90 % | -35.07 pp | 63.74 % | -0.23 pp | 66.11 % | +2.14 pp |
| IPC Active | 1.43 inst/cycle | 2.85 inst/cycle | +1.42 | 1.42 inst/cycle | -0.01 | 1.38 inst/cycle | -0.05 |


### Key Learnings

#### ❌ Plan A: `2d_tiling_bias_reuse_correct_implementation` (0.162×)

Restructuring a flat 1D pointwise kernel into a 2D grid with XCOLS=1024 (power-of-2 covering HIDDEN=768 with masking) and YBLOCK=8 via tl.static_range caused registers per thread to double from 20 to 40. This reduced Block_Limit_Registers from 21 to 12 blocks/SM and cut theoretical occupancy from 100% to 75%. More critically, the YBLOCK=8 static unroll with XCOLS=1024 inflated total executed instructions by 12x (69M to 845M), indicating the kernel is doing vastly more work per element — likely because the grid was not updated to reflect the 2D tiling (still launched with cdiv(xnumel, 128) = 393216 blocks instead of cdiv(num_rows, YBLOCK) = 8192 blocks). The MemoryCacheAccessPattern rule flagged uncoalesced local loads/stores (only 1 of 32 bytes utilized per sector), confirming register spilling to local memory due to the large XCOLS=1024 tile held in registers across the static loop.

> **Applicable when**: Never apply this exact pattern. The 2D tiling approach for bias reuse is conceptually valid but requires the launch grid to be updated to match the 2D decomposition.

> **Avoid when**: Do not use XCOLS=1024 with tl.static_range(YBLOCK=8) unrolling when the launch grid is still computed from the original flat 1D formula (cdiv(xnumel, XBLOCK)). The grid mismatch causes each element to be processed ~48x more than needed (393216/8192 = 48x over-launch), inflating instruction count by 12x and causing register spilling from the large 1024-element tile. Also avoid XCOLS=1024 when the kernel baseline uses only 20 registers/thread — the 1024-element arange alone forces 40 registers/thread, halving Block_Limit_Registers.

---

## Iteration 5

**Best so far**: 1.0004× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_noop_fma_to_enable_native_instruction_fusion | 0.999× | 0.997× | ✓ |  |
| B | vectorized_128bit_loads_to_reduce_l1tex_transaction_count | 1.013× | 1.000× | ✓ |  |
| C | revert_to_flat_1d_coalesced_kernel_with_row_aligned_xblock | 0.999× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_noop_fm... | Δ | vectorized_128bit... | Δ | revert_to_flat_1d... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 245020 ns | 245860 ns | +840 ns (+0.3%) | 244930 ns | -90 ns (-0.0%) | 245500 ns | +480 ns (+0.2%) |
| SM Throughput (%) | 46.58 % | 45.56 % | -1.02 pp | 46.18 % | -0.40 pp | 45.75 % | -0.83 pp |
| Memory Throughput (%) | 66.01 % | 65.73 % | -0.28 pp | 66.00 % | -0.01 pp | 65.82 % | -0.19 pp |
| DRAM Throughput (%) | 66.01 % | 65.73 % | -0.28 pp | 66.00 % | -0.01 pp | 65.82 % | -0.19 pp |
| Achieved Occupancy (%) | 57.08 % | 56.68 % | -0.40 pp | 55.18 % | -1.90 pp | 55.29 % | -1.79 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 20.00 register/thread | +2 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 186.18 | +0.00 |
| No Eligible (%) | 63.97 % | 64.09 % | +0.12 pp | 69.56 % | +5.59 pp | 68.16 % | +4.19 pp |
| IPC Active | 1.43 inst/cycle | 1.44 inst/cycle | +0.01 | 1.21 inst/cycle | -0.22 | 1.28 inst/cycle | -0.15 |

---

## Cumulative Progress

```
Baseline:  245020 ns  (1.000×)
Iter 2     244930 ns  (1.000×)  — reduce_register_pressure_via_maxnreg_to_restore_occupancy

Final best: 245250 ns  (0.9991×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_addmm_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
@@ -13,17 +22,18 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 768)
+    # Issue ALL 5 loads first so hardware can overlap their latencies
     tmp0 = tl.load(in_ptr0 + (x2), xmask)
     tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tl.load(in_out_ptr0 + (x2), xmask)
     tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp6 = tl.load(in_ptr3 + (x2), xmask)
+    # All arithmetic AFTER all loads are issued
     tmp3 = tmp1 + tmp2
-    tmp4 = tmp0 + tmp3
-    tmp7 = tmp5 + tmp6
-    tmp8 = tmp4 + tmp7
+    tmp_low  = libdevice.fma(1.0, tmp0, tmp3)
+    tmp_high = libdevice.fma(1.0, tmp5, tmp6)
+    tmp8 = tmp_low + tmp_high
     tl.store(in_out_ptr0 + (x2), tmp8, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -33,7 +43,7 @@

     arg_4 = inputs[4]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_addmm_view_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 50331648, XBLOCK=128)
+    triton_poi_fused_add_addmm_view_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 50331648)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
