# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 785700 ns — **Best latency**: 4130 ns (**190.2421×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`memory_coalescing_via_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_0_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 32
- **Duration**: 785700 ns
- **SM Throughput**: 12.97 % | **Memory**: 38.89 % | **DRAM**: 38.89 %
- **Achieved Occupancy**: 11.24 %
- **IPC Active**: 0.20 inst/cycle | **No Eligible**: 94.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | memory_coalescing_via_autotune_tile_sizes | 127.614× | 190.242× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 785700 ns | 4130 ns | -781570 ns (-99.5%) |
| SM Throughput (%) | 12.97 % | 4.42 % | -8.55 pp |
| Memory Throughput (%) | 38.89 % | 9.48 % | -29.41 pp |
| DRAM Throughput (%) | 38.89 % | 0.53 % | -38.36 pp |
| Achieved Occupancy (%) | 11.24 % | 5.74 % | -5.50 pp |
| Registers/Thread | 32.00 register/thread | 29.00 register/thread | -3 |
| Waves/SM | 0.62 | 0.06 | -0.56 |
| No Eligible (%) | 94.97 % | 89.81 % | -5.16 pp |
| IPC Active | 0.20 inst/cycle | 0.39 inst/cycle | +0.19 |

---

## Cumulative Progress

```
Baseline:  785700 ns  (1.000×)
Iter 1     4130 ns  (190.242×)  — memory_coalescing_via_autotune_tile_sizes

Final best: 4130 ns  (190.2421×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 31360
@@ -28,8 +44,8 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
@@ -48,10 +64,10 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp11 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp20 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp22 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp11 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp20 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp22 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
         tmp12 = tmp10 + tmp11
         tmp13 = tmp12 - tmp4
         tmp14 = tl.full([1, 1], 96.0, tl.float32)
@@ -62,9 +78,7 @@

         tmp19 = tmp13 * tmp18
         tmp21 = tmp19 * tmp20
         tmp23 = tmp21 + tmp22
-        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, r0_mask & xmask)
-
-
+        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +89,7 @@

     xnumel = 31360
     r0_numel = 96
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_convolution_native_layer_norm_transpose_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 31360, 96)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
