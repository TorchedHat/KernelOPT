# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_layer_norm_transpose_view_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 781760 ns — **Best latency**: 11200 ns (**69.8000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_via_access_pattern_fix_and_r0_block_tuning`** beats baseline latency.

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
- **Duration**: 781760 ns
- **SM Throughput**: 13.09 % | **Memory**: 38.84 % | **DRAM**: 38.84 %
- **Achieved Occupancy**: 11.36 %
- **IPC Active**: 0.20 inst/cycle | **No Eligible**: 94.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_access_pattern_fix_and_r0_block_tuning | 80.339× | 69.800× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 781760 ns | 11200 ns | -770560 ns (-98.6%) |
| SM Throughput (%) | 13.09 % | 24.79 % | +11.70 pp |
| Memory Throughput (%) | 38.84 % | 44.37 % | +5.53 pp |
| DRAM Throughput (%) | 38.84 % | 11.16 % | -27.68 pp |
| Achieved Occupancy (%) | 11.36 % | 11.64 % | +0.28 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 |
| Waves/SM | 0.62 | 0.93 | +0.31 |
| No Eligible (%) | 94.89 % | 67.35 % | -27.54 pp |
| IPC Active | 0.20 inst/cycle | 1.26 inst/cycle | +1.06 |

---

## Cumulative Progress

```
Baseline:  781760 ns  (1.000×)
Iter 1     11200 ns  (69.800×)  — vectorized_loads_via_access_pattern_fix_and_r0_block_tuning

Final best: 11200 ns  (69.8000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 31360
@@ -18,6 +30,7 @@

     rbase = r0_base
     x0 = (xindex % 3136)
     x1 = xindex // 3136
+    x0_vec = tl.max_contiguous(tl.multiple_of(x0, [XBLOCK, 1]), [XBLOCK, 1])
     tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
@@ -28,8 +41,8 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (x0_vec + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
@@ -48,10 +61,10 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp10 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp11 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp20 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp22 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp10 = tl.load(in_ptr0 + (x0_vec + 3136*r0_2 + 301056*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp11 = tl.load(in_ptr1 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp20 = tl.load(in_ptr2 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp22 = tl.load(in_ptr3 + (r0_2), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp12 = tmp10 + tmp11
         tmp13 = tmp12 - tmp4
         tmp14 = tl.full([1, 1], 96.0, tl.float32)
@@ -62,9 +75,7 @@

         tmp19 = tmp13 * tmp18
         tmp21 = tmp19 * tmp20
         tmp23 = tmp21 + tmp22
-        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, r0_mask & xmask)
-
-
+        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp23, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -75,7 +86,7 @@

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
