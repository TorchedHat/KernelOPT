# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 271580 ns — **Best latency**: 271330 ns (**1.0009×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_broken_autotune_grid_restore_small_xblock_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_003/kernel_1_triton_poi_fused_add_avg_pool3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (262144,1,1), Waves/SM: 124.12
- **Registers/thread**: 20
- **Duration**: 271580 ns
- **SM Throughput**: 51.62 % | **Memory**: 90.34 % | **DRAM**: 90.34 %
- **Achieved Occupancy**: 84.80 %
- **IPC Active**: 2.15 inst/cycle | **No Eligible**: 46.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_coalesced_access | 0.999× | 0.256× | ✓ |  |
| B | fix_uncoalesced_loads_via_autotune_xblock | 0.639× | 0.628× | ✓ |  |
| C | memory_coalescing_via_autotune_xblock | 0.639× | 0.630× | ✓ |  |
| D | autotune_xblock_increase_tile_size_for_cache_reuse | 0.368× | 0.100× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_l... | Δ | memory_coalescing... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 271580 ns | 1060000 ns | +788420 ns (+290.3%) | 432350 ns | +160770 ns (+59.2%) | 431300 ns | +159720 ns (+58.8%) | 2710000 ns | +2438420 ns (+897.9%) |
| SM Throughput (%) | 51.62 % | 78.66 % | +27.04 pp | 74.52 % | +22.90 pp | 74.25 % | +22.63 pp | 85.49 % | +33.87 pp |
| Memory Throughput (%) | 90.34 % | 47.58 % | -42.76 pp | 56.74 % | -33.60 pp | 56.87 % | -33.47 pp | 37.28 % | -53.06 pp |
| DRAM Throughput (%) | 90.34 % | 23.05 % | -67.29 pp | 56.74 % | -33.60 pp | 56.87 % | -33.47 pp | 9.03 % | -81.31 pp |
| Achieved Occupancy (%) | 84.80 % | 64.60 % | -20.20 pp | 89.27 % | +4.47 pp | 89.38 % | +4.58 pp | 56.78 % | -28.02 pp |
| Registers/Thread | 20.00 register/thread | 39.00 register/thread | +19 | 32.00 register/thread | +12 | 32.00 register/thread | +12 | 40.00 register/thread | +20 |
| Waves/SM | 124.12 | 330.99 | +206.87 | 124.12 | +0.00 | 124.12 | +0.00 | 661.98 | +537.86 |
| No Eligible (%) | 46.76 % | 21.18 % | -25.58 pp | 25.91 % | -20.85 pp | 25.29 % | -21.47 pp | 14.40 % | -32.36 pp |
| IPC Active | 2.15 inst/cycle | 3.15 inst/cycle | +1.00 | 2.96 inst/cycle | +0.81 | 2.95 inst/cycle | +0.80 | 3.42 inst/cycle | +1.27 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_increase_tile_size_for_cache_reuse` (0.100×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) while keeping the same grid formula (cdiv(xnumel, 128) hardcoded) caused the grid to remain at 262144 blocks but each block now launched 512 threads instead of 128, multiplying total threads by 4x (33M → 134M) and executed instructions by ~17x (108M → 1.83B). The autotune decorator selected a large XBLOCK but the launch grid was not updated to use meta['XBLOCK'], so each thread processed the same single element but with redundant work. Additionally, doubling registers per thread (20→40) cut theoretical occupancy from 100% to 75% and achieved occupancy from 84.8% to 56.8%, making the kernel compute-bound (SM throughput 51%→85%) rather than memory-bound, while DRAM throughput collapsed from 90% to 9% — confirming the kernel was doing massively redundant computation rather than reducing memory traffic.

> **Applicable when**: Never applicable in this broken form. The correct form (grid as lambda using meta['XBLOCK']) would be applicable when L2 hit rate is very low (<15%), DRAM throughput is near saturation (>85%), and UncoalescedGlobalAccess shows >40% excessive sectors with stride-2 access patterns.

> **Avoid when**: Do NOT add @triton.autotune with larger XBLOCK configs unless the launch grid is simultaneously updated to a lambda: `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`. Increasing block size without shrinking the grid multiplies total thread count and executed instructions proportionally, causing severe regression. Also avoid when registers per thread would double (e.g., 20→40) as this cuts theoretical occupancy and may make the kernel compute-bound.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_and_fuse_fp32_instructions | 1.001× | 0.993× | ✓ |  |
| B | revert_autotune_configs_to_restore_low_register_dram_bound_baseline | 0.999× | 0.994× | ✓ |  |
| C | remove_incorrect_vectorized_hints_restore_baseline_register_pressure | 1.000× | 0.997× | ✓ |  |
| D | fix_broken_autotune_grid_restore_small_xblock_occupancy | 1.682× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | revert_autotune_c... | Δ | remove_incorrect_... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 271580 ns | 273410 ns | +1830 ns (+0.7%) | 273280 ns | +1700 ns (+0.6%) | 272540 ns | +960 ns (+0.4%) | 271330 ns | -250 ns (-0.1%) |
| SM Throughput (%) | 51.62 % | 46.08 % | -5.54 pp | 65.04 % | +13.42 pp | 41.82 % | -9.80 pp | 64.89 % | +13.27 pp |
| Memory Throughput (%) | 90.34 % | 89.71 % | -0.63 pp | 89.78 % | -0.56 pp | 90.01 % | -0.33 pp | 90.43 % | +0.09 pp |
| DRAM Throughput (%) | 90.34 % | 89.71 % | -0.63 pp | 89.78 % | -0.56 pp | 90.01 % | -0.33 pp | 90.43 % | +0.09 pp |
| Achieved Occupancy (%) | 84.80 % | 87.36 % | +2.56 pp | 84.77 % | -0.03 pp | 84.75 % | -0.05 pp | 85.03 % | +0.23 pp |
| Registers/Thread | 20.00 register/thread | 31.00 register/thread | +11 | 22.00 register/thread | +2 | 16.00 register/thread | -4 | 22.00 register/thread | +2 |
| Waves/SM | 124.12 | 62.06 | -62.06 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 |
| No Eligible (%) | 46.76 % | 54.21 % | +7.45 pp | 34.88 % | -11.88 pp | 57.78 % | +11.02 pp | 35.74 % | -11.02 pp |
| IPC Active | 2.15 inst/cycle | 1.83 inst/cycle | -0.32 | 2.60 inst/cycle | +0.45 | 1.65 inst/cycle | -0.50 | 2.56 inst/cycle | +0.41 |

---

## Iteration 3

**Best so far**: 1.0009× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_nonfused_fp32_instructions_gelu_fma | 1.679× | 0.993× | ✓ |  |
| B | change_eviction_policy_streaming_loads_to_evict_first | 0.988× | 0.980× | ✓ |  |
| C | streaming_eviction_policy_for_non_reused_loads | 0.988× | 0.992× | ✓ |  |
| D | change_eviction_policy_to_evict_first_for_streaming_pool_loads | 0.998× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_nonfused_fp3... | Δ | change_eviction_p... | Δ | streaming_evictio... | Δ | change_eviction_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 271580 ns | 273540 ns | +1960 ns (+0.7%) | 277090 ns | +5510 ns (+2.0%) | 273860 ns | +2280 ns (+0.8%) | 273340 ns | +1760 ns (+0.6%) |
| SM Throughput (%) | 51.62 % | 63.74 % | +12.12 pp | 40.62 % | -11.00 pp | 63.93 % | +12.31 pp | 46.85 % | -4.77 pp |
| Memory Throughput (%) | 90.34 % | 89.70 % | -0.64 pp | 87.09 % | -3.25 pp | 88.15 % | -2.19 pp | 89.75 % | -0.59 pp |
| DRAM Throughput (%) | 90.34 % | 89.70 % | -0.64 pp | 87.09 % | -3.25 pp | 88.15 % | -2.19 pp | 89.75 % | -0.59 pp |
| Achieved Occupancy (%) | 84.80 % | 85.08 % | +0.28 pp | 84.15 % | -0.65 pp | 84.60 % | -0.20 pp | 87.85 % | +3.05 pp |
| Registers/Thread | 20.00 register/thread | 22.00 register/thread | +2 | 16.00 register/thread | -4 | 22.00 register/thread | +2 | 31.00 register/thread | +11 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 | 62.06 | -62.06 |
| No Eligible (%) | 46.76 % | 34.74 % | -12.02 pp | 58.15 % | +11.39 pp | 35.32 % | -11.44 pp | 51.87 % | +5.11 pp |
| IPC Active | 2.15 inst/cycle | 2.62 inst/cycle | +0.47 | 1.68 inst/cycle | -0.47 | 2.58 inst/cycle | +0.43 | 1.93 inst/cycle | -0.22 |

---

## Iteration 4

**Best so far**: 1.0009× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_latency_hiding_and_fix_fma_noop | 0.999× | 0.997× | ✓ |  |
| B | vectorized_store_via_contiguous_hints_on_output_index | 0.988× | 0.988× | ✓ |  |
| C | fused_gelu_polynomial_approximation_to_reduce_nonfused_fp32_instructions | 0.989× | 0.983× | ✓ |  |
| D | fuse_gelu_arithmetic_into_fma_instructions | 0.989× | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | vectorized_store_... | Δ | fused_gelu_polyno... | Δ | fuse_gelu_arithme... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 271580 ns | 272380 ns | +800 ns (+0.3%) | 274940 ns | +3360 ns (+1.2%) | 276190 ns | +4610 ns (+1.7%) | 275010 ns | +3430 ns (+1.3%) |
| SM Throughput (%) | 51.62 % | 54.49 % | +2.87 pp | 64.54 % | +12.92 pp | 55.10 % | +3.48 pp | 41.06 % | -10.56 pp |
| Memory Throughput (%) | 90.34 % | 89.82 % | -0.52 pp | 87.81 % | -2.53 pp | 87.37 % | -2.97 pp | 87.78 % | -2.56 pp |
| DRAM Throughput (%) | 90.34 % | 89.82 % | -0.52 pp | 87.81 % | -2.53 pp | 87.37 % | -2.97 pp | 87.78 % | -2.56 pp |
| Achieved Occupancy (%) | 84.80 % | 84.62 % | -0.18 pp | 84.47 % | -0.33 pp | 84.74 % | -0.06 pp | 84.23 % | -0.57 pp |
| Registers/Thread | 20.00 register/thread | 20.00 register/thread | +0 | 22.00 register/thread | +2 | 22.00 register/thread | +2 | 16.00 register/thread | -4 |
| Waves/SM | 124.12 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 | 124.12 | +0.00 |
| No Eligible (%) | 46.76 % | 45.44 % | -1.32 pp | 35.64 % | -11.12 pp | 45.35 % | -1.41 pp | 58.80 % | +12.04 pp |
| IPC Active | 2.15 inst/cycle | 2.24 inst/cycle | +0.09 | 2.54 inst/cycle | +0.39 | 2.19 inst/cycle | +0.04 | 1.63 inst/cycle | -0.52 |

---

## Cumulative Progress

```
Baseline:  271580 ns  (1.000×)
Iter 2     271330 ns  (1.001×)  — fix_broken_autotune_grid_restore_small_xblock_occupancy

Final best: 271330 ns  (1.0009×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,51 +5,42 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 33554432
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x0 = (xindex % 32)
     x1 = ((xindex // 32) % 32)
     x2 = xindex // 1024
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp2 = tmp0 + tmp1
-    tmp4 = tmp2 + tmp3
-    tmp6 = tmp4 + tmp5
-    tmp8 = tmp6 + tmp7
-    tmp10 = tmp8 + tmp9
-    tmp12 = tmp10 + tmp11
-    tmp14 = tmp12 + tmp13
-    tmp15 = tl.full([1], 8.0, tl.float32)
-    tmp16 = (tmp14 / tmp15)
-    tmp17 = tl.full([1], 0.5, tl.float32)
-    tmp18 = tmp16 * tmp17
-    tmp19 = tl.full([1], 0.7071067811865476, tl.float32)
-    tmp20 = tmp16 * tmp19
-    tmp21 = libdevice.erf(tmp20)
-    tmp22 = tl.full([1], 1.0, tl.float32)
-    tmp23 = tmp21 + tmp22
-    tmp24 = tmp18 * tmp23
-    tl.store(in_out_ptr0 + (x3), tmp24, None)
-
-
+    tmp0  = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_last')
+    tmp1  = tl.load(in_ptr0 + (1   + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_last')
+    tmp3  = tl.load(in_ptr0 + (64  + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_last')
+    tmp5  = tl.load(in_ptr0 + (65  + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_last')
+    tmp7  = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_first')
+    tmp9  = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_first')
+    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_first')
+    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), eviction_policy='evict_first')
+    avg = (tmp0 + tmp1 + tmp3 + tmp5 + tmp7 + tmp9 + tmp11 + tmp13) * 0.125
+    erf_arg = avg * 0.7071067811865476
+    erf_val = libdevice.erf(erf_arg)
+    result = libdevice.fma(avg * 0.5, erf_val + 1.0, 0.0)
+    tl.store(in_out_ptr0 + x3, result)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 33554432
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1[grid](arg_0, arg_1, 33554432, XBLOCK=128)
+    triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1[grid](arg_0, arg_1, 33554432)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
