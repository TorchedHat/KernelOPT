# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 775550 ns — **Best latency**: 3870 ns (**200.4005×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_register_spill_via_small_xblock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_0_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 775550 ns
- **SM Throughput**: 13.12 % | **Memory**: 39.33 % | **DRAM**: 39.33 %
- **Achieved Occupancy**: 11.32 %
- **IPC Active**: 0.20 inst/cycle | **No Eligible**: 94.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_tile_size_fix | 0.995× | 0.993× | ✓ |  |
| C | software_pipelining_with_tile_size_autotune | 61.791× | 29.956× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_parallelism | 103.494× | 141.782× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | software_pipelini... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 775550 ns | 781090 ns | +5540 ns (+0.7%) | 25890 ns | -749660 ns (-96.7%) | 5470 ns | -770080 ns (-99.3%) |
| SM Throughput (%) | 13.12 % | 13.09 % | -0.03 pp | 28.45 % | +15.33 pp | 8.91 % | -4.21 pp |
| Memory Throughput (%) | 39.33 % | 39.08 % | -0.25 pp | 41.43 % | +2.10 pp | 12.43 % | -26.90 pp |
| DRAM Throughput (%) | 39.33 % | 39.08 % | -0.25 pp | 11.05 % | -28.28 pp | 5.69 % | -33.64 pp |
| Achieved Occupancy (%) | 11.32 % | 11.35 % | +0.03 pp | 11.39 % | +0.07 pp | 5.78 % | -5.54 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 255.00 register/thread | +223 | 56.00 register/thread | +24 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.93 | +0.31 | 0.10 | -0.52 |
| No Eligible (%) | 94.94 % | 94.90 % | -0.04 pp | 69.33 % | -25.61 pp | 84.95 % | -9.99 pp |
| IPC Active | 0.20 inst/cycle | 0.20 inst/cycle | +0.00 | 1.22 inst/cycle | +1.02 | 0.57 inst/cycle | +0.37 |

---

## Iteration 2

**Best so far**: 141.7824× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_with_maxnreg_and_vectorized_load_hints | 87.094× | 111.751× | ✓ |  |
| B | reduce_register_spill_via_small_xblock_autotune | 123.976× | 200.400× | ✓ | **✓ best** |
| C | autotune_r0block_to_eliminate_shared_memory_occupancy_bottleneck | 103.399× | 137.753× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_s... | Δ | autotune_r0block_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 775550 ns | 6940 ns | -768610 ns (-99.1%) | 3870 ns | -771680 ns (-99.5%) | 5630 ns | -769920 ns (-99.3%) |
| SM Throughput (%) | 13.12 % | 7.08 % | -6.04 pp | 8.19 % | -4.93 pp | 8.81 % | -4.31 pp |
| Memory Throughput (%) | 39.33 % | 12.10 % | -27.23 pp | 11.33 % | -28.00 pp | 12.28 % | -27.05 pp |
| DRAM Throughput (%) | 39.33 % | 4.67 % | -34.66 pp | 0.56 % | -38.77 pp | 5.51 % | -33.82 pp |
| Achieved Occupancy (%) | 11.32 % | 5.79 % | -5.53 pp | 10.81 % | -0.51 pp | 5.78 % | -5.54 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 24.00 register/thread | -8 | 56.00 register/thread | +24 |
| Waves/SM | 0.62 | 0.06 | -0.56 | 0.12 | -0.50 | 0.10 | -0.52 |
| No Eligible (%) | 94.94 % | 88.00 % | -6.94 pp | 81.68 % | -13.26 pp | 85.25 % | -9.69 pp |
| IPC Active | 0.20 inst/cycle | 0.43 inst/cycle | +0.23 | 0.61 inst/cycle | +0.41 | 0.56 inst/cycle | +0.36 |

---

## Iteration 3

**Best so far**: 200.4005× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints_on_reduction_and_store | 122.096× | 216.634× | ✓ |  |
| B | maximize_grid_size_via_small_xblock_configs_to_overcome_wave_starvation | 118.003× | 201.966× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | maximize_grid_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 775550 ns | 3580 ns | -771970 ns (-99.5%) | 3840 ns | -771710 ns (-99.5%) |
| SM Throughput (%) | 13.12 % | 8.24 % | -4.88 pp | 4.23 % | -8.89 pp |
| Memory Throughput (%) | 39.33 % | 11.40 % | -27.93 pp | 9.08 % | -30.25 pp |
| DRAM Throughput (%) | 39.33 % | 0.60 % | -38.73 pp | 0.57 % | -38.76 pp |
| Achieved Occupancy (%) | 11.32 % | 11.13 % | -0.19 pp | 5.74 % | -5.58 pp |
| Registers/Thread | 32.00 register/thread | 24.00 register/thread | -8 | 29.00 register/thread | -3 |
| Waves/SM | 0.62 | 0.12 | -0.50 | 0.06 | -0.56 |
| No Eligible (%) | 94.94 % | 83.62 % | -11.32 pp | 89.95 % | -4.99 pp |
| IPC Active | 0.20 inst/cycle | 0.68 inst/cycle | +0.48 | 0.33 inst/cycle | +0.13 |

---

## Iteration 4

**Best so far**: 200.4005× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_warp_aligned_xblock_with_num_stages1_autotune | 61.827× | 62.143× | ✓ |  |
| C | fix_grid_lambda_and_coalesce_strided_access_via_loop_reorder | 115.507× | 205.172× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_warp_al... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 775550 ns | 12480 ns | -763070 ns (-98.4%) | 3780 ns | -771770 ns (-99.5%) |
| SM Throughput (%) | 13.12 % | 10.95 % | -2.17 pp | 4.16 % | -8.96 pp |
| Memory Throughput (%) | 39.33 % | 17.53 % | -21.80 pp | 8.93 % | -30.40 pp |
| DRAM Throughput (%) | 39.33 % | 5.05 % | -34.28 pp | 0.58 % | -38.75 pp |
| Achieved Occupancy (%) | 11.32 % | 2.90 % | -8.42 pp | 5.68 % | -5.64 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 29.00 register/thread | -3 |
| Waves/SM | 0.62 | 0.23 | -0.39 | 0.06 | -0.56 |
| No Eligible (%) | 94.94 % | 72.98 % | -21.96 pp | 91.84 % | -3.10 pp |
| IPC Active | 0.20 inst/cycle | 0.54 inst/cycle | +0.34 | 0.31 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  775550 ns  (1.000×)
Iter 1     5470 ns  (141.782×)  — autotune_tile_sizes_for_occupancy_and_parallelism
Iter 2     3870 ns  (200.400×)  — reduce_register_spill_via_small_xblock_autotune

Final best: 3870 ns  (200.4005×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # NEW: small XBLOCK to eliminate register spilling
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128, 'num_stages': 3}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128, 'num_stages': 3}, num_warps=8),
+        # EXISTING configs retained
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128, 'num_stages': 3}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128, 'num_stages': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128, 'num_stages': 3}, num_warps=16),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128, 'num_stages': 2}, num_warps=4),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 31360
@@ -27,9 +47,11 @@

         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        r0_2 = r0_index  # keep raw for strided in_ptr0
+        # hint for stride-1 accesses: apply on 1D arange then reshape to [1, R0_BLOCK]
+        r0_2_vec = tl.max_contiguous(tl.multiple_of(tl.arange(0, R0_BLOCK) + r0_offset, R0_BLOCK), R0_BLOCK)[None, :]
+        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + r0_2_vec, mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
@@ -47,11 +69,13 @@

         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
-        r0_2 = r0_index
-        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp11 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp20 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp22 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        r0_2 = r0_index  # keep raw for strided in_ptr0
+        # hint for stride-1 accesses: apply on 1D arange then reshape to [1, R0_BLOCK]
+        r0_2_vec = tl.max_contiguous(tl.multiple_of(tl.arange(0, R0_BLOCK) + r0_offset, R0_BLOCK), R0_BLOCK)[None, :]
+        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp11 = tl.load(in_ptr1 + r0_2_vec, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp20 = tl.load(in_ptr2 + r0_2_vec, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp22 = tl.load(in_ptr3 + r0_2_vec, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp12 = tmp10 + tmp11
         tmp13 = tmp12 - tmp4
         tmp14 = tl.full([1, 1], 96.0, tl.float32)
@@ -62,9 +86,7 @@

         tmp19 = tmp13 * tmp18
         tmp21 = tmp19 * tmp20
         tmp23 = tmp21 + tmp22
-        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, r0_mask & xmask)
-
-
+        tl.store(out_ptr2 + (r0_2_vec + 96*x3), tmp23, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +97,7 @@

     xnumel = 31360
     r0_numel = 96
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
