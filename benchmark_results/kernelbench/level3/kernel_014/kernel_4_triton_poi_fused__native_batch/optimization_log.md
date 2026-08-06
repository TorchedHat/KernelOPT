# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3580 ns — **Best latency**: 2980 ns (**1.2013×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`restrict_xblock_to_1_for_coalesced_loads_and_large_yblock_for_sm_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12,1,1), Waves/SM: 0.01
- **Registers/thread**: 34
- **Duration**: 3580 ns
- **SM Throughput**: 0.40 % | **Memory**: 0.36 % | **DRAM**: 0.06 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.72 inst/cycle | **No Eligible**: 80.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_2d_grid_launch_to_parallelize_y_dimension | 0.973× | 0.848× | ✓ |  |
| B | fix_grid_parallelism_with_autotune | 0.971× | 1.017× | ✓ |  |
| C | fix_grid_launch_and_add_vectorized_loads | 0.912× | 0.895× | ✓ |  |
| D | fix_grid_launch_to_cover_full_ynumel_dimension | 0.784× | 0.557× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_2d_grid_launc... | Δ | fix_grid_parallel... | Δ | fix_grid_launch_a... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 4220 ns | +640 ns (+17.9%) | 3520 ns | -60 ns (-1.7%) | 4000 ns | +420 ns (+11.7%) | 6430 ns | +2850 ns (+79.6%) |
| SM Throughput (%) | 0.40 % | 0.46 % | +0.06 pp | 0.54 % | +0.14 pp | 0.48 % | +0.08 pp | 1.06 % | +0.66 pp |
| Memory Throughput (%) | 0.36 % | 4.39 % | +4.03 pp | 0.52 % | +0.16 pp | 0.39 % | +0.03 pp | 3.81 % | +3.45 pp |
| DRAM Throughput (%) | 0.06 % | 0.28 % | +0.22 pp | 0.12 % | +0.06 pp | 0.11 % | +0.05 pp | 1.32 % | +1.26 pp |
| Achieved Occupancy (%) | 6.23 % | 6.09 % | -0.14 pp | 6.22 % | -0.01 pp | 6.22 % | -0.01 pp | 6.21 % | -0.02 pp |
| Registers/Thread | 34.00 register/thread | 44.00 register/thread | +10 | 40.00 register/thread | +6 | 40.00 register/thread | +6 | 148.00 register/thread | +114 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.03 | +0.02 |
| No Eligible (%) | 80.67 % | 90.69 % | +10.02 pp | 80.03 % | -0.64 pp | 80.04 % | -0.63 pp | 83.00 % | +2.33 pp |
| IPC Active | 0.72 inst/cycle | 0.34 inst/cycle | -0.38 | 0.81 inst/cycle | +0.09 | 0.76 inst/cycle | +0.04 | 0.70 inst/cycle | -0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_kernel_function_call_site_to_use_2d_lambda_grid | 0.994× | 0.957× | ✓ |  |
| B | fix_grid_launch_to_2d_with_register_safe_autotune_configs | 0.368× | 0.909× | ✓ |  |
| C | fix_grid_launch_call_site_to_use_2d_autotune_lambda | 1.008× | 0.829× | ✓ |  |
| D | fix_grid_to_2d_with_small_register_safe_tiles | 1.076× | 1.366× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_kernel_functi... | Δ | fix_grid_launch_t... | Δ | fix_grid_launch_c... | Δ | fix_grid_to_2d_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 3740 ns | +160 ns (+4.5%) | 3940 ns | +360 ns (+10.1%) | 4320 ns | +740 ns (+20.7%) | 2620 ns | -960 ns (-26.8%) |
| SM Throughput (%) | 0.40 % | 0.53 % | +0.13 pp | 0.48 % | +0.08 pp | 0.49 % | +0.09 pp | 0.17 % | -0.23 pp |
| Memory Throughput (%) | 0.36 % | 0.51 % | +0.15 pp | 0.42 % | +0.06 pp | 4.18 % | +3.82 pp | 0.46 % | +0.10 pp |
| DRAM Throughput (%) | 0.06 % | 0.12 % | +0.06 pp | 0.11 % | +0.05 pp | 0.27 % | +0.21 pp | 0.08 % | +0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.21 % | -0.02 pp | 6.22 % | -0.01 pp | 6.09 % | -0.14 pp | 3.12 % | -3.11 pp |
| Registers/Thread | 34.00 register/thread | 40.00 register/thread | +6 | 40.00 register/thread | +6 | 44.00 register/thread | +10 | 24.00 register/thread | -10 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 80.67 % | 79.36 % | -1.31 pp | 82.26 % | +1.59 pp | 91.03 % | +10.36 pp | 88.66 % | +7.99 pp |
| IPC Active | 0.72 inst/cycle | 0.80 inst/cycle | +0.08 | 0.74 inst/cycle | +0.02 | 0.34 inst/cycle | -0.38 | 0.18 inst/cycle | -0.54 |


### Key Learnings

#### ✅ Plan D: `fix_grid_to_2d_with_small_register_safe_tiles` (1.218×)

Fixing a Triton kernel's autotune configs to use XBLOCK=1 with large YBLOCK values (256–2048) and updating the kernel body's yoffset to use a 3D-compatible formula `(tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK` improved L2 Hit Rate from 44.95% to 78.56% and eliminated divergent branches (branch efficiency 83% → 100%), yielding a 1.22x speedup. The XBLOCK=1 constraint ensures all warp threads share the same X-index, making Y-dimension memory accesses coalesced; larger YBLOCK tiles increase the number of Y-elements processed per block, improving L2 cache reuse for per-channel weight tensors loaded with evict_last. Registers per thread rose modestly from 24 to 30, keeping theoretical occupancy at 100% (Block Limit Registers remained permissive at 16 warps).

> **Applicable when**: Triton kernel has a large-stride X-dimension (channel-last layout) causing uncoalesced loads when XBLOCK > 1; NCU reports CPIStall on L1TEX, low L2 Hit Rate (<50%), and divergent branches (branch efficiency < 100%); LaunchConfiguration rule fires with grid_size << sm_count; registers_per_thread < 30 at current tile sizes leaving headroom for larger YBLOCK without spill.

> **Avoid when**: Do not use XBLOCK=1 with large YBLOCK when the X-dimension is the fast-varying (stride-1) dimension — this reduces parallelism without coalescing benefit. Do not use YBLOCK > 1024 if registers_per_thread would exceed 32 (causing spill). Do not apply if the grid is already large enough to fill SMs (waves_per_sm > 1). Note: this iteration's profiling still shows grid_size=12 after (NCU captured a single autotuning candidate), so the full grid-coverage benefit may not be reflected in the after metrics.

---

## Iteration 3

**Best so far**: 1.3664× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_call_site_grid_to_2d_lambda_preserving_existing_autotune_configs | 1.144× | 1.346× | ✓ | **✓ best** |
| B | increase_tile_sizes_with_2d_grid_for_cache_utilization | 0.940× | 0.848× | ✓ |  |
| C | increase_tile_sizes_with_2d_grid_and_remove_debug_barrier | 0.979× | 1.085× | ✓ |  |
| D | fix_callsite_grid_to_2d_lambda_matching_existing_autotune_configs | 0.968× | 0.794× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_call_site_gri... | Δ | increase_tile_siz... | Δ | increase_tile_siz... | Δ | fix_callsite_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2660 ns | -920 ns (-25.7%) | 4220 ns | +640 ns (+17.9%) | 3300 ns | -280 ns (-7.8%) | 4510 ns | +930 ns (+26.0%) |
| SM Throughput (%) | 0.40 % | 0.16 % | -0.24 pp | 0.44 % | +0.04 pp | 0.23 % | -0.17 pp | 0.46 % | +0.06 pp |
| Memory Throughput (%) | 0.36 % | 0.43 % | +0.07 pp | 0.35 % | -0.01 pp | 0.43 % | +0.07 pp | 3.89 % | +3.53 pp |
| DRAM Throughput (%) | 0.06 % | 0.08 % | +0.02 pp | 0.10 % | +0.04 pp | 0.09 % | +0.03 pp | 0.26 % | +0.20 pp |
| Achieved Occupancy (%) | 6.23 % | 3.12 % | -3.11 pp | 6.22 % | -0.01 pp | 3.12 % | -3.11 pp | 6.08 % | -0.15 pp |
| Registers/Thread | 34.00 register/thread | 24.00 register/thread | -10 | 40.00 register/thread | +6 | 40.00 register/thread | +6 | 44.00 register/thread | +10 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 80.67 % | 90.06 % | +9.39 pp | 80.81 % | +0.14 pp | 82.65 % | +1.98 pp | 90.94 % | +10.27 pp |
| IPC Active | 0.72 inst/cycle | 0.21 inst/cycle | -0.51 | 0.75 inst/cycle | +0.03 | 0.34 inst/cycle | -0.38 | 0.34 inst/cycle | -0.38 |

---

## Iteration 4

**Best so far**: 1.3664× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_callsite_grid_to_proper_2d_autotune_lambda | 0.961× | 1.243× | ✓ |  |
| B | persistent_flat_1d_grid_with_y_loop_for_full_sm_coverage | 0.999× | 1.056× | ✓ |  |
| C | reduce_register_pressure_with_small_tiles_and_fix_2d_grid_callsite | 1.093× | 1.415× | ✓ |  |
| D | remove_debug_barrier_and_fix_uncoalesced_in_ptr0_load | 1.149× | 1.398× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_callsite_grid... | Δ | persistent_flat_1... | Δ | reduce_register_p... | Δ | remove_debug_barr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2880 ns | -700 ns (-19.6%) | 3390 ns | -190 ns (-5.3%) | 2530 ns | -1050 ns (-29.3%) | 2560 ns | -1020 ns (-28.5%) |
| SM Throughput (%) | 0.40 % | 0.18 % | -0.22 pp | 0.25 % | -0.15 pp | 0.17 % | -0.23 pp | 0.11 % | -0.29 pp |
| Memory Throughput (%) | 0.36 % | 0.40 % | +0.04 pp | 0.49 % | +0.13 pp | 0.54 % | +0.18 pp | 0.88 % | +0.52 pp |
| DRAM Throughput (%) | 0.06 % | 0.08 % | +0.02 pp | 0.12 % | +0.06 pp | 0.09 % | +0.03 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 3.12 % | -3.11 pp | 3.08 % | -3.15 pp | 3.12 % | -3.11 pp | 3.06 % | -3.17 pp |
| Registers/Thread | 34.00 register/thread | 24.00 register/thread | -10 | 40.00 register/thread | +6 | 24.00 register/thread | -10 | 26.00 register/thread | -8 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 80.67 % | 90.04 % | +9.37 pp | 87.00 % | +6.33 pp | 88.45 % | +7.78 pp | 92.45 % | +11.78 pp |
| IPC Active | 0.72 inst/cycle | 0.23 inst/cycle | -0.49 | 0.25 inst/cycle | -0.47 | 0.23 inst/cycle | -0.49 | 0.13 inst/cycle | -0.59 |


### Key Learnings

#### ✅ Plan A: `fix_callsite_grid_to_proper_2d_autotune_lambda` (1.243×)

When a Triton kernel uses tl.program_id(1) for a Y-dimension offset but the call-site grid is hardcoded as a 1D tuple (covering only the X dimension), the Y dimension is never processed. Replacing the hardcoded 1D grid with a 2D autotune lambda `lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))` allows the kernel to cover the full output tensor. This fix must be paired with small tile configs (XBLOCK/YBLOCK ≤ 32) to keep registers per thread low (≤ 24) and maintain 100% theoretical occupancy; the profiling showed L2 Compression Input Sectors dropped from 310 to 133, indicating reduced redundant memory traffic, and a 1.24x speedup was achieved even though NCU still reports only 12 grid blocks (suggesting the profiling captured a single autotuning candidate run rather than the winning config).

> **Applicable when**: Triton kernel uses tl.program_id(1) for a Y-dimension but the call-site grid is a 1D tuple; LaunchConfiguration NCU rule fires with grid_size << sm_count; waves_per_sm ≈ 0; achieved_occupancy << theoretical_occupancy; registers_per_thread ≤ 30 with theoretical_occupancy = 100% (register budget is safe for small tiles).

> **Avoid when**: Kernel does not use tl.program_id(1) (truly 1D kernel); registers_per_thread > 30 at current tile sizes (adding autotune may push into spill territory); tile configs include XBLOCK or YBLOCK > 32 (causes register explosion as seen in exp_001 and exp_003). Do not pass XBLOCK/YBLOCK as explicit kwargs when @triton.autotune is active.

---

## Iteration 5

**Best so far**: 1.3984× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restrict_xblock_to_1_for_coalesced_loads_and_large_yblock_for_sm_coverage | 1.152× | 1.201× | ✓ | **✓ best** |
| B | increase_yblock_tile_size_with_xblock1_coalesced_2d_grid | 1.065× | 1.218× | ✓ |  |
| C | restrict_xblock_to_1_with_large_yblock_for_coalesced_loads_and_sm_coverage | 1.150× | 1.256× | ✓ |  |
| D | reduce_register_pressure_and_fix_kernel_body_to_2d_pid | 1.081× | 1.432× | ✓ |  |
| E | add_triton_autotune | 0.554× | 0.909× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restrict_xblock_t... | Δ | increase_yblock_t... | Δ | restrict_xblock_t... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2980 ns | -600 ns (-16.8%) | 2940 ns | -640 ns (-17.9%) | 2850 ns | -730 ns (-20.4%) | 2500 ns | -1080 ns (-30.2%) | 3940 ns | +360 ns (+10.1%) |
| SM Throughput (%) | 0.40 % | 0.13 % | -0.27 pp | 0.31 % | -0.09 pp | 0.13 % | -0.27 pp | 0.19 % | -0.21 pp | 0.40 % | +0.00 pp |
| Memory Throughput (%) | 0.36 % | 0.81 % | +0.45 pp | 2.00 % | +1.64 pp | 0.83 % | +0.47 pp | 0.47 % | +0.11 pp | 0.42 % | +0.06 pp |
| DRAM Throughput (%) | 0.06 % | 0.06 % | +0.00 pp | 0.13 % | +0.07 pp | 0.06 % | +0.00 pp | 0.08 % | +0.02 pp | 0.06 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 3.04 % | -3.19 pp | 5.96 % | -0.27 pp | 3.04 % | -3.19 pp | 3.12 % | -3.11 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 34.00 register/thread | 26.00 register/thread | -8 | 30.00 register/thread | -4 | 26.00 register/thread | -8 | 24.00 register/thread | -10 | 32.00 register/thread | -2 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 80.67 % | 91.28 % | +10.61 pp | 90.72 % | +10.05 pp | 90.46 % | +9.79 pp | 88.06 % | +7.39 pp | 79.70 % | -0.97 pp |
| IPC Active | 0.72 inst/cycle | 0.17 inst/cycle | -0.55 | 0.33 inst/cycle | -0.39 | 0.17 inst/cycle | -0.55 | 0.24 inst/cycle | -0.48 | 0.58 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  3580 ns  (1.000×)
Iter 2     2620 ns  (1.366×)  — fix_grid_to_2d_with_small_register_safe_tiles
Iter 4     2560 ns  (1.398×)  — remove_debug_barrier_and_fix_uncoalesced_in_ptr0_load

Final best: 2980 ns  (1.2013×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 501760
     xnumel = 96
-    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
+    yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
@@ -19,25 +29,25 @@

     y0 = (yindex % 50176)
     y1 = yindex // 50176
     y3 = yindex
-    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
-    tmp28 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
-    tmp30 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
+    tmp17 = tl.load(in_ptr3 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr4 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp28 = tl.load(in_ptr5 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp30 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
     tmp0 = x2
     tmp1 = tl.full([1, 1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1, 1], 32, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (y0 + 50176*(x2) + 1605632*y1), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (y0 + 50176*(x2) + 1605632*y1), mask=tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1, 1], 64, tl.int64)
     tmp8 = tmp0 < tmp7
     tmp9 = tmp6 & tmp8
-    tmp10 = tl.load(in_ptr1 + (32*y3 + ((-32) + x2)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr1 + (32*y3 + ((-32) + x2)), mask=tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp0 >= tmp7
     tmp12 = tl.full([1, 1], 96, tl.int64)
     tmp13 = tmp0 < tmp12
-    tmp14 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp14 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), mask=tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp15 = tl.where(tmp9, tmp10, tmp14)
     tmp16 = tl.where(tmp4, tmp5, tmp15)
     tmp18 = tmp16 - tmp17
@@ -53,10 +63,7 @@

     tmp31 = tmp29 + tmp30
     tmp32 = tl.full([1, 1], 0, tl.int32)
     tmp33 = tl.maximum(tmp32, tmp31)
-    tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x2 + 96*y3), tmp33, xmask & ymask)
-
-
+    tl.store(in_out_ptr0 + (x2 + 96*y3), tmp33, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -70,7 +77,7 @@

     ynumel = 501760
     xnumel = 96
     grid = (triton.cdiv(xnumel, 8),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 501760, 96, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 501760, 96)
     return arg_7
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
