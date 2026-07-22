# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7300 ns — **Best latency**: 5790 ns (**1.2608×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_use_dynamic_meta_xblock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_46_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4900,1,1), Waves/SM: 2.32
- **Registers/thread**: 29
- **Duration**: 7300 ns
- **SM Throughput**: 51.05 % | **Memory**: 23.98 % | **DRAM**: 23.98 %
- **Achieved Occupancy**: 75.70 %
- **IPC Active**: 2.90 inst/cycle | **No Eligible**: 26.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.832× | 0.738× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.003× | 0.736× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.981× | 1.022× | ✓ |  |
| D | eliminate_partial_wave_via_larger_xblock | 1.000× | 1.052× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_e... | Δ | increase_xblock_f... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7300 ns | 9890 ns | +2590 ns (+35.5%) | 9920 ns | +2620 ns (+35.9%) | 7140 ns | -160 ns (-2.2%) | 6940 ns | -360 ns (-4.9%) |
| SM Throughput (%) | 51.05 % | 61.78 % | +10.73 pp | 61.99 % | +10.94 pp | 49.89 % | -1.16 pp | 49.96 % | -1.09 pp |
| Memory Throughput (%) | 23.98 % | 17.62 % | -6.36 pp | 17.65 % | -6.33 pp | 24.40 % | +0.42 pp | 25.07 % | +1.09 pp |
| DRAM Throughput (%) | 23.98 % | 17.62 % | -6.36 pp | 17.62 % | -6.36 pp | 24.40 % | +0.42 pp | 25.07 % | +1.09 pp |
| Achieved Occupancy (%) | 75.70 % | 61.04 % | -14.66 pp | 61.49 % | -14.21 pp | 74.10 % | -1.60 pp | 74.92 % | -0.78 pp |
| Registers/Thread | 29.00 register/thread | 38.00 register/thread | +9 | 38.00 register/thread | +9 | 29.00 register/thread | +0 | 29.00 register/thread | +0 |
| Waves/SM | 2.32 | 3.09 | +0.77 | 3.09 | +0.77 | 2.32 | +0.00 | 2.32 | +0.00 |
| No Eligible (%) | 26.71 % | 22.34 % | -4.37 pp | 22.14 % | -4.57 pp | 26.49 % | -0.22 pp | 26.69 % | -0.02 pp |
| IPC Active | 2.90 inst/cycle | 3.08 inst/cycle | +0.18 | 3.02 inst/cycle | +0.12 | 2.87 inst/cycle | -0.03 | 2.81 inst/cycle | -0.09 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_rsqrt_and_fix_eviction_policy_for_streaming_loads | 0.978× | 1.146× | ✓ |  |
| B | reduce_num_warps_to_improve_cache_coherence_and_reduce_cpi_stall | 1.052× | 1.267× | ✓ |  |
| C | restore_occupancy_by_capping_xblock_to_reduce_register_pressure | 1.054× | 1.254× | ✓ | **✓ best** |
| D | restore_occupancy_via_maxnreg_and_small_xblock_configs | 1.002× | 1.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_rsqrt_and_fi... | Δ | reduce_num_warps_... | Δ | restore_occupancy... | Δ | restore_occupancy... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7300 ns | 6370 ns | -930 ns (-12.7%) | 5760 ns | -1540 ns (-21.1%) | 5820 ns | -1480 ns (-20.3%) | 7200 ns | -100 ns (-1.4%) |
| SM Throughput (%) | 51.05 % | 45.78 % | -5.27 pp | 27.39 % | -23.66 pp | 31.32 % | -19.73 pp | 50.82 % | -0.23 pp |
| Memory Throughput (%) | 23.98 % | 27.44 % | +3.46 pp | 30.37 % | +6.39 pp | 15.07 % | -8.91 pp | 24.25 % | +0.27 pp |
| DRAM Throughput (%) | 23.98 % | 27.44 % | +3.46 pp | 30.37 % | +6.39 pp | 15.07 % | -8.91 pp | 24.25 % | +0.27 pp |
| Achieved Occupancy (%) | 75.70 % | 70.00 % | -5.70 pp | 59.82 % | -15.88 pp | 60.82 % | -14.88 pp | 74.31 % | -1.39 pp |
| Registers/Thread | 29.00 register/thread | 32.00 register/thread | +3 | 32.00 register/thread | +3 | 29.00 register/thread | +0 | 29.00 register/thread | +0 |
| Waves/SM | 2.32 | 2.32 | +0.00 | 1.16 | -1.16 | 1.16 | -1.16 | 2.32 | +0.00 |
| No Eligible (%) | 26.71 % | 31.04 % | +4.33 pp | 47.66 % | +20.95 pp | 34.06 % | +7.35 pp | 25.64 % | -1.07 pp |
| IPC Active | 2.90 inst/cycle | 2.74 inst/cycle | -0.16 | 2.02 inst/cycle | -0.88 | 2.54 inst/cycle | -0.36 | 2.87 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.2543× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_eviction_policy_streaming_tensor_to_protect_broadcast_cache_residency | 1.014× | 1.267× | ✓ |  |
| B | fuse_rsqrt_to_eliminate_nonfused_fp32_instruction_chain | 1.030× | 1.261× | ✓ |  |
| C | add_num_warps_2_configs_to_reduce_warp_contention | 1.018× | 1.221× | ✓ |  |
| D | reduce_num_warps_and_fuse_rsqrt_instruction | 1.083× | 1.207× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_eviction_poli... | Δ | fuse_rsqrt_to_eli... | Δ | add_num_warps_2_c... | Δ | reduce_num_warps_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7300 ns | 5760 ns | -1540 ns (-21.1%) | 5790 ns | -1510 ns (-20.7%) | 5980 ns | -1320 ns (-18.1%) | 6050 ns | -1250 ns (-17.1%) |
| SM Throughput (%) | 51.05 % | 26.83 % | -24.22 pp | 24.61 % | -26.44 pp | 25.22 % | -25.83 pp | 24.69 % | -26.36 pp |
| Memory Throughput (%) | 23.98 % | 30.28 % | +6.30 pp | 15.11 % | -8.87 pp | 29.12 % | +5.14 pp | 14.48 % | -9.50 pp |
| DRAM Throughput (%) | 23.98 % | 30.28 % | +6.30 pp | 15.11 % | -8.87 pp | 29.12 % | +5.14 pp | 14.48 % | -9.50 pp |
| Achieved Occupancy (%) | 75.70 % | 62.94 % | -12.76 pp | 56.32 % | -19.38 pp | 60.11 % | -15.59 pp | 56.65 % | -19.05 pp |
| Registers/Thread | 29.00 register/thread | 32.00 register/thread | +3 | 32.00 register/thread | +3 | 32.00 register/thread | +3 | 32.00 register/thread | +3 |
| Waves/SM | 2.32 | 1.16 | -1.16 | 1.16 | -1.16 | 1.16 | -1.16 | 1.16 | -1.16 |
| No Eligible (%) | 26.71 % | 47.06 % | +20.35 pp | 39.84 % | +13.13 pp | 47.38 % | +20.67 pp | 39.97 % | +13.26 pp |
| IPC Active | 2.90 inst/cycle | 1.88 inst/cycle | -1.02 | 2.30 inst/cycle | -0.60 | 2.01 inst/cycle | -0.89 | 2.31 inst/cycle | -0.59 |


### Key Learnings

#### ✅ Plan A: `fix_eviction_policy_streaming_tensor_to_protect_broadcast_cache_residency` (1.267×)

Switching large streaming tensor loads from eviction_policy='evict_last' to 'evict_first' in Triton reduces warp stall cycles per issued instruction (20.69 → 18.62 cycles, ~10% reduction) and slightly improves achieved occupancy (59.8% → 62.9%), yielding a 1.27x speedup at the system level. The mechanism is that 'evict_last' on large streaming tensors (>>L1 capacity) causes them to displace small broadcast arrays (e.g., batch-norm mean/var/weight/bias) from L1/L2 cache, forcing repeated DRAM fetches for those reused scalars; 'evict_first' marks the streaming lines as low-priority for retention, preserving L1 residency for the small broadcast arrays. Notably, L1/TEX hit rate and L2 hit rate did not change measurably in NCU (both ~44% and ~28%), suggesting the benefit manifests as reduced stall latency rather than a visible hit-rate shift at the granularity NCU reports.

> **Applicable when**: Kernel loads a large streaming tensor (size >> L1 cache, not reused across blocks) with evict_last AND simultaneously loads small broadcast arrays (e.g., per-channel BN parameters, bias vectors) that are reused across many thread blocks; CPIStall rule fires with L1TEX scoreboard stalls accounting for >35% of CPI; L1/TEX hit rate is moderate (40-55%) suggesting cache pressure from competing access patterns.

> **Avoid when**: The 'large' tensor is actually reused across blocks and benefits from L1 retention (evict_last is correct); kernel has no small broadcast arrays competing for L1 space; kernel is purely streaming with no reused data (eviction policy change has no effect); L1/TEX hit rate is already >70% (cache pressure is not the bottleneck).

---

## Iteration 4

**Best so far**: 1.2543× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_arithmetic_to_fma_in_batchnorm_epilogue | 1.093× | 1.261× | ✓ | **✓ best** |
| B | fix_hardcoded_grid_to_use_autotuned_xblock | 1.054× | 1.239× | ✓ |  |
| C | evict_first_on_large_conditional_tensors_to_protect_broadcast_cache | 0.979× | 1.254× | ✓ |  |
| D | pipeline_memory_loads_with_num_stages_to_hide_l1tex_stall_and_enable_fma | 1.089× | 1.261× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_arithme... | Δ | fix_hardcoded_gri... | Δ | evict_first_on_la... | Δ | pipeline_memory_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7300 ns | 5790 ns | -1510 ns (-20.7%) | 5890 ns | -1410 ns (-19.3%) | 5820 ns | -1480 ns (-20.3%) | 5790 ns | -1510 ns (-20.7%) |
| SM Throughput (%) | 51.05 % | 14.18 % | -36.87 pp | 24.74 % | -26.31 pp | 26.71 % | -24.34 pp | 24.95 % | -26.10 pp |
| Memory Throughput (%) | 23.98 % | 15.12 % | -8.86 pp | 14.85 % | -9.13 pp | 30.12 % | +6.14 pp | 15.12 % | -8.86 pp |
| DRAM Throughput (%) | 23.98 % | 15.12 % | -8.86 pp | 14.85 % | -9.13 pp | 30.12 % | +6.14 pp | 15.12 % | -8.86 pp |
| Achieved Occupancy (%) | 75.70 % | 27.58 % | -48.12 pp | 56.70 % | -19.00 pp | 59.72 % | -15.98 pp | 59.39 % | -16.31 pp |
| Registers/Thread | 29.00 register/thread | 36.00 register/thread | +7 | 32.00 register/thread | +3 | 32.00 register/thread | +3 | 32.00 register/thread | +3 |
| Waves/SM | 2.32 | 1.16 | -1.16 | 1.16 | -1.16 | 1.16 | -1.16 | 1.16 | -1.16 |
| No Eligible (%) | 26.71 % | 60.72 % | +34.01 pp | 39.74 % | +13.03 pp | 46.82 % | +20.11 pp | 42.50 % | +15.79 pp |
| IPC Active | 2.90 inst/cycle | 1.38 inst/cycle | -1.52 | 2.28 inst/cycle | -0.62 | 2.02 inst/cycle | -0.88 | 2.11 inst/cycle | -0.79 |

---

## Iteration 5

**Best so far**: 1.2608× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_activate_autotune_and_increase_block_size | 1.037× | 1.274× | ✓ |  |
| B | fix_hardcoded_grid_to_use_dynamic_meta_xblock | 1.089× | 1.261× | ✓ |  |
| C | add_num_stages_2_with_maxnreg32_to_pipeline_l1tex_stall | 1.031× | 1.274× | ✓ |  |
| D | fuse_bn_epilogue_arithmetic_to_fma_instructions | 1.056× | 1.281× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | add_num_stages_2_... | Δ | fuse_bn_epilogue_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7300 ns | 5730 ns | -1570 ns (-21.5%) | 5790 ns | -1510 ns (-20.7%) | 5730 ns | -1570 ns (-21.5%) | 5700 ns | -1600 ns (-21.9%) |
| SM Throughput (%) | 51.05 % | 25.00 % | -26.05 pp | 25.01 % | -26.04 pp | 24.92 % | -26.13 pp | 23.67 % | -27.38 pp |
| Memory Throughput (%) | 23.98 % | 15.27 % | -8.71 pp | 15.17 % | -8.81 pp | 15.24 % | -8.74 pp | 15.34 % | -8.64 pp |
| DRAM Throughput (%) | 23.98 % | 15.27 % | -8.71 pp | 15.17 % | -8.81 pp | 15.24 % | -8.74 pp | 15.34 % | -8.64 pp |
| Achieved Occupancy (%) | 75.70 % | 56.28 % | -19.42 pp | 57.07 % | -18.63 pp | 57.09 % | -18.61 pp | 57.48 % | -18.22 pp |
| Registers/Thread | 29.00 register/thread | 32.00 register/thread | +3 | 32.00 register/thread | +3 | 32.00 register/thread | +3 | 32.00 register/thread | +3 |
| Waves/SM | 2.32 | 1.16 | -1.16 | 1.16 | -1.16 | 1.16 | -1.16 | 1.16 | -1.16 |
| No Eligible (%) | 26.71 % | 40.01 % | +13.30 pp | 39.16 % | +12.45 pp | 39.18 % | +12.47 pp | 40.20 % | +13.49 pp |
| IPC Active | 2.90 inst/cycle | 2.29 inst/cycle | -0.61 | 2.30 inst/cycle | -0.60 | 2.35 inst/cycle | -0.55 | 2.20 inst/cycle | -0.70 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_activate_autotune_and_increase_block_size` (1.274×)

When a Triton kernel's launch site uses a hardcoded grid expression (e.g., `triton.cdiv(xnumel, 128)`) instead of a lambda referencing `meta['XBLOCK']`, the @triton.autotune decorator is completely non-functional: regardless of which config the autotuner selects, the grid is fixed and the kernel always executes with the smallest block size (1 warp/block). This causes critically low achieved occupancy (27.6%), 60.7% no-eligible-warp cycles, and very low issue slot utilization (35%). Fixing the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` and removing num_warps=1 configs allows the autotuner to select larger block sizes (e.g., block_size=64), raising theoretical occupancy from 50% to 100%, achieved occupancy from 27.6% to 56.3%, eligible warps per scheduler from 1.15 to 3.68, and issue slot utilization from 35% to 57.6%, yielding a 1.27x speedup.

> **Applicable when**: NCU shows block_size=32 (1 warp/block) with theoretical occupancy ~50% and achieved occupancy <30%; no-eligible-warp cycles exceed 55% of total; issue slot utilization is below 40%; the kernel has a @triton.autotune decorator but the launch site grid does not reference meta['XBLOCK'] (hardcoded cdiv with a fixed tile size). This is a definitive implementation flaw detectable by checking whether grid_size * XBLOCK_from_winning_config == xnumel.

> **Avoid when**: The grid is already a lambda referencing meta['XBLOCK'] (autotune is already functional); block_size is already >= 64 and achieved occupancy is above 40%; the kernel intentionally uses 1 warp/block for cache-coherence reasons with very small problem sizes.

---

## Cumulative Progress

```
Baseline:  7300 ns  (1.000×)
Iter 2     5820 ns  (1.254×)  — restore_occupancy_by_capping_xblock_to_reduce_register_pressure
Iter 4     5790 ns  (1.261×)  — fuse_fp32_arithmetic_to_fma_in_batchnorm_epilogue

Final best: 5790 ns  (1.2608×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK':  64}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK':  64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
     xnumel = 627200
@@ -16,21 +29,21 @@

     x2 = xindex // 4480
     x3 = xindex // 320
     x4 = xindex
-    tmp27 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp29 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
-    tmp38 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
-    tmp40 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp29 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp38 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp40 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 256, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -40,29 +53,20 @@

     tmp17 = tl.full([1], 288, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 320, tl.int64)
     tmp23 = tmp0 < tmp22
-    tmp24 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp24 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp21 & xmask, eviction_policy='evict_last', other=0.0)
     tmp25 = tl.where(tmp19, tmp20, tmp24)
     tmp26 = tl.where(tmp4, tmp15, tmp25)
     tmp28 = tmp26 - tmp27
-    tmp30 = tl.full([1], 1e-05, tl.float32)
-    tmp31 = tmp29 + tmp30
-    tmp32 = tl.sqrt_rn(tmp31)
-    tmp33 = tl.full([1], 1, tl.int32)
-    tmp34 = (tmp33 / tmp32)
-    tmp35 = tl.full([1], 1.0, tl.float32)
-    tmp36 = tmp34 * tmp35
-    tmp37 = tmp28 * tmp36
-    tmp39 = tmp37 * tmp38
-    tmp41 = tmp39 + tmp40
-    tmp42 = tl.full([1], 0, tl.int32)
-    tmp43 = tl.maximum(tmp42, tmp41)
-    tl.store(in_out_ptr0 + (x4), tmp43, xmask)
-
-
+    tmp31 = tmp29 + tl.full([1], 1e-05, tl.float32)
+    tmp32 = tl.math.rsqrt(tmp31)
+    tmp_scale = tmp28 * tmp32
+    tmp41 = tl.math.fma(tmp_scale, tmp38, tmp40)
+    tmp43 = tl.maximum(tl.zeros_like(tmp41), tmp41)
+    tl.store(in_out_ptr0 + (x4), tmp43, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +79,7 @@

     arg_7 = inputs[7]
     xnumel = 627200
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 627200, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_46[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 627200)
     return arg_7
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
