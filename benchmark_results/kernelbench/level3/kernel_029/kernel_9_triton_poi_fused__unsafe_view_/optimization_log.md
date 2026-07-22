# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21020 ns — **Best latency**: 21020 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_9_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 16
- **Duration**: 21020 ns
- **SM Throughput**: 69.74 % | **Memory**: 36.49 % | **DRAM**: 36.49 %
- **Achieved Occupancy**: 83.87 %
- **IPC Active**: 3.09 inst/cycle | **No Eligible**: 22.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_persistent_tile_scheduling | 0.961× | 0.965× | ✓ |  |
| B | autotune_xblock_tile_size | 0.534× | 0.503× | ✓ |  |
| C | increase_xblock_with_autotune_for_better_cache_reuse_and_occupancy | 0.967× | 0.967× | ✓ |  |
| D | increase_xblock_for_vectorized_memory_and_reduced_address_overhead | 0.703× | 0.130× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_p... | Δ | autotune_xblock_t... | Δ | increase_xblock_w... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21020 ns | 21790 ns | +770 ns (+3.7%) | 41820 ns | +20800 ns (+99.0%) | 21730 ns | +710 ns (+3.4%) | 162240 ns | +141220 ns (+671.8%) |
| SM Throughput (%) | 69.74 % | 72.21 % | +2.47 pp | 81.32 % | +11.58 pp | 71.90 % | +2.16 pp | 83.08 % | +13.34 pp |
| Memory Throughput (%) | 36.49 % | 35.12 % | -1.37 pp | 22.37 % | -14.12 pp | 35.25 % | -1.24 pp | 16.31 % | -20.18 pp |
| DRAM Throughput (%) | 36.49 % | 35.12 % | -1.37 pp | 18.35 % | -18.14 pp | 35.25 % | -1.24 pp | 4.71 % | -31.78 pp |
| Achieved Occupancy (%) | 83.87 % | 84.59 % | +0.72 pp | 90.86 % | +6.99 pp | 83.47 % | -0.40 pp | 91.88 % | +8.01 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 32.00 register/thread | +16 | 18.00 register/thread | +2 | 32.00 register/thread | +16 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 44.55 | +33.41 |
| No Eligible (%) | 22.31 % | 20.22 % | -2.09 pp | 14.52 % | -7.79 pp | 20.18 % | -2.13 pp | 15.56 % | -6.75 pp |
| IPC Active | 3.09 inst/cycle | 3.17 inst/cycle | +0.08 | 3.42 inst/cycle | +0.33 | 3.17 inst/cycle | +0.08 | 3.37 inst/cycle | +0.28 |


### Key Learnings

#### ❌ Plan B: `autotune_xblock_tile_size` (0.501×)

Despite the plan explicitly targeting the grid/XBLOCK mismatch, the applied diff did NOT fix the grid lambda — the launch site still used a hardcoded `triton.cdiv(xnumel, 128)` grid rather than `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. As a result, the autotuner selected XBLOCK=512+ while the grid remained sized for XBLOCK=128, causing executed instructions to more than double (12.1M → 26.3M), registers per thread to jump from 18 to 32, and duration to nearly double (21.73us → 41.98us). The index decomposition and autotune config changes were cosmetic and did not compensate for the fundamental grid mismatch. This is the same failure mode as exp_001: any autotune XBLOCK expansion without a corresponding grid lambda update will produce instruction count inflation proportional to new_XBLOCK/old_XBLOCK.

> **Applicable when**: Never apply autotune XBLOCK expansion without simultaneously updating the grid to use meta['XBLOCK']. This pattern is a confirmed anti-pattern across two iterations.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK configs if the launch grid is not simultaneously updated to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. Do not apply index decomposition as a standalone optimization when the kernel is already compute-bound with ALU saturation — it does not reduce instruction count meaningfully. Do not increase XBLOCK when registers per thread are already ≥18 and the kernel is not memory-bound, as larger tiles will trigger register pressure increases that reduce occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_for_persistent_tile_scheduling | 0.952× | 0.965× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_fma_and_vectorized_loads | 0.961× | 0.967× | ✓ |  |
| C | fix_autotune_grid_lambda_and_reduce_index_alu_pressure | 0.531× | 0.501× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_cta_execution | 0.957× | 0.130× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21020 ns | 21790 ns | +770 ns (+3.7%) | 21730 ns | +710 ns (+3.4%) | 41980 ns | +20960 ns (+99.7%) | 162240 ns | +141220 ns (+671.8%) |
| SM Throughput (%) | 69.74 % | 71.87 % | +2.13 pp | 70.65 % | +0.91 pp | 79.74 % | +10.00 pp | 83.07 % | +13.33 pp |
| Memory Throughput (%) | 36.49 % | 35.13 % | -1.36 pp | 35.30 % | -1.19 pp | 22.28 % | -14.21 pp | 16.30 % | -20.19 pp |
| DRAM Throughput (%) | 36.49 % | 35.13 % | -1.36 pp | 35.30 % | -1.19 pp | 18.25 % | -18.24 pp | 4.71 % | -31.78 pp |
| Achieved Occupancy (%) | 83.87 % | 84.82 % | +0.95 pp | 84.35 % | +0.48 pp | 90.57 % | +6.70 pp | 91.97 % | +8.10 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 44.55 | +33.41 |
| No Eligible (%) | 22.31 % | 20.49 % | -1.82 pp | 21.42 % | -0.89 pp | 16.07 % | -6.24 pp | 15.58 % | -6.73 pp |
| IPC Active | 3.09 inst/cycle | 3.15 inst/cycle | +0.06 | 3.17 inst/cycle | +0.08 | 3.35 inst/cycle | +0.26 | 3.38 inst/cycle | +0.29 |

---

## Cumulative Progress

```
Baseline:  21020 ns  (1.000×)

Final best: 21020 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x3 = xindex
     x0 = (xindex % 96)
     x1 = ((xindex // 96) % 3136)
     x2 = xindex // 301056
-    tmp0 = tl.load(in_out_ptr0 + (x3), None)
-    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 56))) % 7)) + 224*(((4 + (x1 // 56)) % 7)) + 1568*(x0 // 32) + 4704*((4 + ((x1 % 56))) // 7) + 42336*((4 + (x1 // 56)) // 7) + 381024*x2 + ((x0 % 32))), None)
-    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 56)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 56))) % 7))), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp6 = tl.load(in_ptr3 + (x3), None)
+    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
+    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 56))) % 7)) + 224*(((4 + (x1 // 56)) % 7)) + 1568*(x0 // 32) + 4704*((4 + ((x1 % 56))) // 7) + 42336*((4 + (x1 // 56)) // 7) + 381024*x2 + ((x0 % 32))), xmask)
+    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 56)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 56))) % 7))), xmask, eviction_policy='evict_last')
+    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp6 = tl.load(in_ptr3 + (x3), xmask)
     tmp3 = tmp1 + tmp2
     tmp4 = tmp0 + tmp3
     tmp7 = tmp5 + tmp6
     tmp8 = tmp4 + tmp7
-    tl.store(in_out_ptr0 + (x3), tmp8, None)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -36,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 3010560
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560, XBLOCK=128)
+    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
